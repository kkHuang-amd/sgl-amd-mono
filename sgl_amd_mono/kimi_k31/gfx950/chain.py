# SPDX-License-Identifier: Apache-2.0
"""The post-attention chain of a wide-row layer launch, one launch (T4 -> T5 -> T6).

Stages (``plan.Plan``), every CTA walking them in order, joined by words only:

oproj  o_proj tiles (x the local attention output, K = 1536), K slices merged in
       scratch; each slice pushes its rows of the finished tile to the column
       owner (plain stores + a piece flag; ``gemm_coll``).
rsred  the owner sums a piece's TP sources in rank order, adds the residual,
       stores the new residual slice and marks the piece in local scratch.
seam   a row a task (remote stores need many CTAs): waits for its band's pieces,
       then the MLP AttnRes seam over the bank's nvb snapshots and the new prefix
       (``stages/attnres.py``, one scalar round with every rank), the normalized
       slice pushed to every rank's gather region, then (this rank, row) flagged.
front  MoE-front tiles (router + latent down + shared gate_up rows, K = 7168),
       a task waiting for the (row, owner) flags of its rows and K range and
       reading x' from the gather region.
"""

import dataclasses

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import Int32, Int64, T

from sgl_amd_mono.common import guard
from sgl_amd_mono.common.ops import CM_DEV, CM_SYS, bf2_f32, bf16_pair, load_ptr64, rsrc
from sgl_amd_mono.common.plan import (
    BLOCKS,
    THREADS,
    first_task,
    kernel_sources,
    key_tuple,
)
from sgl_amd_mono.common.sync import sreg
from sgl_amd_mono.kimi_k31.gfx950.stages import coll
from sgl_amd_mono.kimi_k31.gfx950.stages.attnres import attnres_row, red_words, scalars
from sgl_amd_mono.kimi_k31.gfx950.stages.gemm_coll import (
    _arrive_all,
    _push_band_flag,
    dev_emit,
)
from sgl_amd_mono.kimi_k31.gfx950.stages.tile import atomic_lin, bf16_emit, f32_emit
from sgl_amd_mono.kimi_k31.gfx950.stages.tile_mma import DEPTH, RW, mma_tile
from sgl_amd_mono.kimi_k31.gfx950.stamps import stamp, stamp_begin, stamp_flush
from sgl_amd_mono.kimi_k31.plan import Plan

SOURCES = kernel_sources(__name__)

TP = coll.TP
H = 7168
KO = 1536  # o_proj K a rank (KDA heads x head dim / TP)
# timeline marks (a CTA's first task in a stage marks its wait; the stage's end its last task)
MARKS = (
    "start",
    "oproj mma1",
    "oproj end",
    "rsred in1",
    "rsred end",
    "seam piece1",
    "seam pushed1",
    "seam end",
    "front in1",
    "front end",
)
TL_POINTS = len(MARKS)


@dataclasses.dataclass
class Gemm:
    """A tile GEMM's build config: bm, bn, wm x wn waves, ks K slices."""

    bm: int
    bn: int
    wm: int
    wn: int
    ks: int

    def key(self):
        return (self.bm, self.bn, self.wm, self.wn, self.ks)


def plan(m, go, gf, nf, nvb=0):
    """Regions and stages of a chain build (``go`` the o_proj config, ``gf`` the
    front's, ``nf`` the front's output rows a rank)."""
    sl = H // TP
    rb = go.bm // go.ks
    ct = sl // go.bn
    nb = (m + rb - 1) // rb
    assert (
        sl % go.bn == 0
        and go.bm % go.ks == 0
        and nb * ct <= BLOCKS
        and BLOCKS % go.ks == 0
    )
    p = Plan()
    p.region("rs", TP * m * sl * 2)
    p.region("fl", TP * BLOCKS * 8)
    p.region("xs", TP * m * scalars(nvb) * 8)
    p.region("ag", m * H * 2)
    p.region("agf", TP * m * 8)
    p.region("pdone", nb * ct * 8, peer=False)
    tasks_o = ((m + go.bm - 1) // go.bm) * (H // go.bn) * go.ks
    tasks_f = ((m + gf.bm - 1) // gf.bm) * ((nf + gf.bn - 1) // gf.bn) * gf.ks
    p.stage("oproj", tasks_o, waits_on=("oproj",))
    p.stage("rsred", nb * ct, waits_on=("oproj",), all_ranks=True)
    p.stage("seam", m, waits_on=("rsred", "seam"), all_ranks=True)
    p.stage("front", tasks_f, waits_on=("seam",), all_ranks=True)
    return p.check()


@flyc.jit
def _rs_piece(
    c, piece, m, sl, rb, bn, res, res_out, pdone, tl_on, tls, first, res_cm=0, ldo=0
):
    """Sum piece ``piece``'s TP sources in rank order, add the residual, store the
    new residual (device scope: the seam's CTA reads it), mark the piece. ``ldo``:
    ``res_out``'s row stride (0: sl)."""
    tid = c["tid"]
    q8 = bn // 8
    ct = sl // bn
    ldo = ldo or sl
    if tid < TP:
        c["poll"]([(c["fl_own"], tid * BLOCKS + piece, 1)])
    gpu.barrier()
    if first:
        stamp(tl_on, tls, tid, 3)
    band, cj = piece // ct, piece % ct
    # res None: a block-write layer, whose prefix restarts at the attention output
    r_res = rsrc(res) if res is not None else None
    r_out, r_rs = rsrc(res_out), rsrc(c["rs_own"].value)
    for u in range(tid, rb * q8, THREADS):
        row = band * rb + u // q8
        col = cj * bn + 8 * (u % q8)
        if row < m:
            got = [
                fx.Vector(
                    bo.buffer_load(
                        r_rs,
                        (src * m * sl + row * sl + col) // 2,
                        vec_width=4,
                        dtype=T.i32,
                        cache_modifier=CM_SYS,
                    )
                )
                for src in range(TP)
            ]
            if const_expr(res is not None):
                rv = fx.Vector(
                    bo.buffer_load(
                        r_res,
                        (row * sl + col) // 2,
                        vec_width=4,
                        dtype=T.i32,
                        cache_modifier=res_cm,
                    )
                )
            acc = None
            for src in range_constexpr(TP):
                cur = [v for e in range(4) for v in bf2_f32(got[src][e])]
                acc = cur if acc is None else [a + b for a, b in zip(acc, cur)]
            if const_expr(res is not None):
                r = [v for e in range(4) for v in bf2_f32(rv[e])]
                acc = [a + b for a, b in zip(acc, r)]
            words = [
                bf16_pair(acc[2 * j], acc[2 * j + 1]).bitcast(fx.Int32)
                for j in range(4)
            ]
            bo.buffer_store(
                fx.Vector.from_elements(words, fx.Int32),
                r_out,
                (row * ldo + col) // 2,
                cache_modifier=CM_DEV,
            )
    rocdl.s_waitcnt(vmcnt=0)
    gpu.barrier()
    if tid == 0:
        c["put"](pdone, piece, fx.Int32(1))


@flyc.jit
def _seam_row(
    c,
    row,
    m,
    sl,
    rb,
    ct,
    nvb,
    nbk,
    res_out,
    bank,
    cw,
    w_o,
    eps,
    pdone,
    red,
    write,
    tl_on,
    tls,
    first,
    bank_cm=0,
    ldo=0,
):
    """The MLP AttnRes seam of one row (``attnres.attnres_row``) once its band's
    pieces are in; the normalized slice goes to every rank's gather region, then
    (this rank, row) is flagged on every rank."""
    tid = c["tid"]
    if tid < ct:
        c["poll"]([(pdone, (row // rb) * ct + tid, 1)])
    gpu.barrier()
    if first:
        stamp(tl_on, tls, tid, 5)
    rank = c["rank"]
    ags = [c["ag_of"](p) for p in range(TP)]

    def emit(q, live, vec):
        if live:
            for p in range_constexpr(TP):
                bo.buffer_store(
                    vec,
                    rsrc(ags[p].value),
                    (row * TP * sl + rank * sl + 8 * q) // 2,
                    cache_modifier=CM_SYS,
                )

    attnres_row(
        c,
        row,
        m,
        sl,
        nvb,
        nbk,
        res_out,
        bank,
        cw,
        w_o,
        eps,
        eps,
        red,
        emit,
        write,
        bank_cm,
        None,
        ldo,
    )
    if first:
        stamp(tl_on, tls, tid, 6)
    rocdl.s_waitcnt(vmcnt=0)
    gpu.barrier()
    coll.put_each_peer(c, [("agf", rank * m + row, fx.Int32(1))])


@flyc.jit
def _front_wait(c, m0, bm, m, j0, nown, tl_on, tls, first):
    """Wait for the (row, owner) flags of rows m0 .. m0 + bm and owners j0 .. j0 + nown."""
    rows = fx.min(m0 + bm, m) - m0
    for i in range(c["tid"], rows * nown, THREADS):
        c["poll"]([(c["agf_own"], (j0 + i % nown) * m + m0 + i // nown, 1)])
    gpu.barrier()
    if first:
        stamp(tl_on, tls, c["tid"], 8)


@flyc.jit
def _prefetch(tid, w, ldw, n, n0, k0, kt, bn, sink):
    """Read the W tile rows n0 .. n0 + bn, columns k0 .. k0 + kt into L2 (cached
    loads folded into a value that is stored only if impossible), while the CTA
    would otherwise wait for its x'."""
    r = rsrc(w)
    pieces = bn * (kt // 8)
    acc = fx.Int32(0)
    batch = 8
    for i in range(tid, pieces, THREADS * batch):
        for j in range_constexpr(batch):
            e = fx.min(i + j * THREADS, pieces - 1)
            row = fx.min(n0 + e // (kt // 8), n - 1)
            v = fx.Vector(
                bo.buffer_load(
                    r,
                    (row * ldw + k0 + 8 * (e % (kt // 8))) // 2,
                    vec_width=4,
                    dtype=T.i32,
                )
            )
            acc = acc ^ v[0]
    if acc == fx.Int32(0x13572468):
        fx.ptr_store(acc, sink)


_BUILDS: dict = {}


def post_attn(
    m,
    go,
    gf,
    nf,
    mode="linbf",
    launches=1,
    eps=1e-6,
    timeline=False,
    front_top=True,
    prefetch=False,
    x_probe=False,
    nvb=0,
    nbk=8,
    write=False,
):
    """Build the chain launch; returns (launch, plan). Arguments: a [m, KO] bf16, wo
    [H, KO], res / res_out [m, H / TP] (this rank's prefix columns before / after the add),
    bank [m, nbk, H / TP] (this rank's columns of the snapshots; ``write``: row nvb
    := the new prefix), cw f32 [H] (score_norm x score_proj weights), w_ln [H] (the
    post-attention norm), wf
    [nf, H], out (the front's output: bf16 [m, nf], f32 [ks, m, nf] for "part"),
    acc / cnt (o_proj merge scratch and counters, zeroed once), scratch
    (``plan.scratch_bytes``, zeroed once), peers, rank, epoch, slot, tl (``timeline``:
    int64 [BLOCKS, TL_POINTS] clock marks, see MARKS; else unused). ``front_top``:
    the front's tasks on the highest CTAs, away from the seam's rows on the lowest;
    ``prefetch``: a front CTA reads its first tile's W into L2 before its wait.
    ``x_probe`` (timing only, wrong results): the front reads x from the cached local
    ``acc`` instead of the uncached gather region."""
    keyed = key_tuple(locals(), SOURCES)
    key = (
        m,
        go.key(),
        gf.key(),
        nf,
        mode,
        launches,
        eps,
        timeline,
        front_top,
        prefetch,
        x_probe,
        nvb,
        nbk,
        write,
    )
    if key in _BUILDS:
        return _BUILDS[key]
    assert nvb < nbk or (nvb == nbk and not write)
    pl = plan(m, go, gf, nf, nvb)
    offs, half = pl.peer_offsets
    soffs, _ = pl.scratch_offsets
    fo = pl.flags_offset()
    sl = H // TP
    rb = go.bm // go.ks
    ct = sl // go.bn
    nb = (m + rb - 1) // rb
    kto, ktf = KO // go.ks, H // gf.ks
    assert kto % 64 == 0 and ktf % 64 == 0 and (ktf % sl == 0 or sl % ktf == 0)
    nown = max(ktf // sl, 1)
    tno, tmo = H // go.bn, (m + go.bm - 1) // go.bm
    tnf, tmf = (nf + gf.bn - 1) // gf.bn, (m + gf.bm - 1) // gf.bm
    n_o, n_f = tmo * tno * go.ks, tmf * tnf * gf.ks
    bmx, bnx = max(go.bm, gf.bm), max(go.bn, gf.bn)
    cw = max(go.bm * (go.bn + 4), gf.bm * (gf.bn + 4))
    opart = mode == "part" and go.ks > 1
    fpart = mode == "part" and gf.ks > 1

    @fx.struct
    class AbLds:
        xs: fx.Array[fx.Int32, 2 * bmx * RW, 16]
        ws: fx.Array[fx.Int32, 2 * bnx * RW, 16]

    @fx.struct
    class CLds:
        c: fx.Array[fx.Float32, cw, 16]

    @fx.union
    class Smem:
        ab: AbLds
        c: CLds

    @fx.struct
    class Slot:
        v: fx.Array[fx.Int32, 4, 16]

    @fx.struct
    class Red:
        v: fx.Array[fx.Float32, red_words(nvb), 16]

    @fx.struct
    class Tls:
        v: fx.Array[fx.Int64, TL_POINTS, 16]

    name = (
        f"k31_post_attn_m{m}_o{'x'.join(map(str, go.key()))}_f{'x'.join(map(str, gf.key()))}_n{nf}_{mode}"
        f"_l{launches}_t{int(timeline)}_ft{int(front_top)}_pf{int(prefetch)}_xp{int(x_probe)}_v{nvb}of{nbk}w{int(write)}"
    )

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def post_attn_kernel(
        a: Int64,
        wo: Int64,
        res: Int64,
        res_out: Int64,
        bank: Int64,
        cw: Int64,
        w_ln: Int64,
        wf: Int64,
        out: Int64,
        acc: Int64,
        cnt: Int64,
        scratch: Int64,
        peers: Int64,
        rank: Int32,
        epoch: Int64,
        slot_i: Int32,
        tl: Int64,
    ):
        own = load_ptr64(peers, rank)
        with guard.scope(
            guard.Guard(
                epoch, own=own, peers=peers, rank=rank, npes=TP, flags_offset=fo
            )
        ):
            tid, bid = fx.thread_idx.x, fx.block_idx.x
            alloc = fx.SharedAllocator()
            st = alloc.allocate(Smem)
            slot = alloc.allocate(Slot).peek().v.ptr
            red = alloc.allocate(Red).peek().v.ptr
            tls = alloc.allocate(Tls).peek().v.ptr
            stamp_begin(timeline, tls, tid, TL_POINTS)
            lds, cl = st.ab.peek(), st.c.peek().c.ptr
            c = coll.region_context(
                peers, rank, epoch, slot_i, offs, half, tid, bid, launches
            )
            pdone = sreg(scratch, soffs["pdone"], "pdone")

            # oproj
            for task in range(bid, n_o, BLOCKS):
                sk = task % go.ks
                tile = task // go.ks
                m0 = (tile // tno) * go.bm
                n0 = (tile % tno) * go.bn
                lin, emit = None, None
                if const_expr(go.ks == 1):
                    emit = dev_emit(acc, H, n0, m, H, False)
                elif const_expr(opart):
                    emit = dev_emit(acc + fx.Int64(sk) * (m * H * 4), H, n0, m, H, True)
                else:
                    lin = atomic_lin(acc, H, n0, m, H, True)
                mma_tile(
                    tid,
                    a,
                    KO,
                    m,
                    m0,
                    wo,
                    KO,
                    H,
                    n0,
                    sk * kto,
                    kto,
                    go.bm,
                    go.bn,
                    go.wm,
                    go.wn,
                    lds.xs.ptr,
                    lds.ws.ptr,
                    emit,
                    DEPTH,
                    lin,
                    cl,
                )
                if task == bid:
                    stamp(timeline, tls, tid, 1)
                _arrive_all(tid, cnt, tile, go.ks, slot)
                piece = ((m0 + sk * rb) // rb) * ct + (n0 % sl) // go.bn
                _push_band_flag(
                    c, acc, m, H, m0 + sk * rb, rb, n0, go.bn, go.ks, mode, sl, piece
                )
                stamp(timeline, tls, tid, 2)
            # rsred
            for piece in range(bid, nb * ct, BLOCKS):
                _rs_piece(
                    c,
                    piece,
                    m,
                    sl,
                    rb,
                    go.bn,
                    res,
                    res_out,
                    pdone,
                    timeline,
                    tls,
                    piece == bid,
                )
                stamp(timeline, tls, tid, 4)
            # seam
            for row in range(bid, m, BLOCKS):
                _seam_row(
                    c,
                    row,
                    m,
                    sl,
                    rb,
                    ct,
                    nvb,
                    nbk,
                    res_out,
                    bank,
                    cw,
                    w_ln,
                    eps,
                    pdone,
                    red,
                    write,
                    timeline,
                    tls,
                    row == bid,
                )
                stamp(timeline, tls, tid, 7)
            # front
            x = acc if x_probe else c["ag_own"].value
            fbase = max(BLOCKS - n_f, 0) if front_top else 0
            f0 = first_task(bid, fbase)
            if const_expr(prefetch):
                if f0 < n_f:
                    t = f0 // gf.ks
                    _prefetch(
                        tid,
                        wf,
                        H,
                        nf,
                        (t % tnf) * gf.bn,
                        (f0 % gf.ks) * ktf,
                        ktf,
                        gf.bn,
                        slot,
                    )
            for task in range(f0, n_f, BLOCKS):
                sk = task % gf.ks
                tile = task // gf.ks
                m0 = (tile // tnf) * gf.bm
                n0 = (tile % tnf) * gf.bn
                _front_wait(
                    c, m0, gf.bm, m, (sk * ktf) // sl, nown, timeline, tls, task == f0
                )
                lin, emit = None, None
                if const_expr(gf.ks == 1):
                    emit = bf16_emit(out, nf, n0, m, nf)
                elif const_expr(fpart):
                    emit = f32_emit(out + fx.Int64(sk) * (m * nf * 4), nf, n0, m, nf)
                else:
                    lin = atomic_lin(out, nf, n0, m, nf, True)
                mma_tile(
                    tid,
                    x,
                    H,
                    m,
                    m0,
                    wf,
                    H,
                    nf,
                    n0,
                    sk * ktf,
                    ktf,
                    gf.bm,
                    gf.bn,
                    gf.wm,
                    gf.wn,
                    lds.xs.ptr,
                    lds.ws.ptr,
                    emit,
                    DEPTH,
                    lin,
                    cl,
                )
                stamp(timeline, tls, tid, 9)
            stamp_flush(timeline, tls, tl, tid, bid, TL_POINTS)

    @flyc.jit
    def launch(
        a: Int64,
        wo: Int64,
        res: Int64,
        res_out: Int64,
        bank: Int64,
        cw: Int64,
        w_ln: Int64,
        wf: Int64,
        out: Int64,
        acc: Int64,
        cnt: Int64,
        scratch: Int64,
        peers: Int64,
        rank: Int32,
        epoch: Int64,
        slot_i: Int32,
        tl: Int64,
        stream: fx.Stream = fx.Stream(None),
    ):  # noqa: B008
        _ = keyed
        post_attn_kernel(
            a,
            wo,
            res,
            res_out,
            bank,
            cw,
            w_ln,
            wf,
            out,
            acc,
            cnt,
            scratch,
            peers,
            rank,
            epoch,
            slot_i,
            tl,
        ).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)

    _BUILDS[key] = (launch, pl)
    return launch, pl
