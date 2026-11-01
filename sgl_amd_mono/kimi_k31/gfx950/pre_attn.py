# SPDX-License-Identifier: Apache-2.0
"""The pre-attention chain of a wide-row layer launch, one launch (T1 -> T2 -> in_proj).

SGLang's latent-MoE tail (``kimi_k3.KimiK3MoE._column_split_tail``) and the
attention-side AttnRes seam, on column-sharded state:

    latent = RMSNorm_n(AR(latent partials));  y = AR(shared partials + latent @ W_up^T)
    prefix' = prefix + y;  x = AttnRes(bank, prefix') -> input norm;  in_proj(x)

Stages (``plan.Plan``), joined by words only:

push    a (row, owner) a task: this rank's latent [448] and shared [896] columns of
        the owner pushed (plain stores), then (this rank, row) flagged at the owner.
lred    a row: the owner sums its latent and shared columns over the TP sources in
        rank order; the latent slice goes to every rank's gather region with the
        slice's sum of squares (tagged), the shared sum stays in local scratch.
up      latent up tiles of the owner's 896 columns over the gathered latent; the
        norm weight is folded into W_up's columns (``fold_up``) and the row's
        rsqrt(mean square) scales the epilogue. A tile's K slices merge, arrive,
        and each finishes its rows: y = up + shared, prefix' = prefix + y.
seam    a row: ``attnres_row`` with the attention weights -> x pushed to every rank.
inproj  in_proj tiles over the gathered x (as ``chain``'s front).
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import Int32, Int64, T

from sgl_amd_mono.common import guard
from sgl_amd_mono.common.ops import CM_DEV, CM_SYS, bf2_f32, bf16_pair, hw_rsq, load_ptr64, rsrc, wave_sum
from sgl_amd_mono.common.plan import BLOCKS, THREADS, first_task
from sgl_amd_mono.common.sync import sreg
from sgl_amd_mono.kimi_k31.gfx950.chain import Gemm
from sgl_amd_mono.kimi_k31.gfx950.stages import coll
from sgl_amd_mono.kimi_k31.gfx950.stages.attnres import attnres_row, red_words, scalars
from sgl_amd_mono.kimi_k31.gfx950.stages.gemm_coll import _arrive_all, dev_emit
from sgl_amd_mono.kimi_k31.gfx950.stages.tile import atomic_lin, bf16_emit, f32_emit
from sgl_amd_mono.kimi_k31.gfx950.stages.tile_mma import DEPTH, RW, mma_tile
from sgl_amd_mono.kimi_k31.plan import Plan

TP = coll.TP
H = 7168
LAT = 3584
SL, LSL = H // TP, LAT // TP


def fold_up(w_up, w_n, rank):
    """This rank's W_up rows [896, 3584] with the latent norm weight folded into
    its columns (host side, once at load)."""
    rows = w_up[rank * SL : (rank + 1) * SL]
    return (rows.float() * w_n.float()[None, :]).to(rows.dtype).contiguous()


def plan(m, gu, gi, nin, nvb):
    rb = gu.bm // gu.ks
    ct = SL // gu.bn
    nb = (m + rb - 1) // rb
    assert gu.bm % gu.ks == 0 and SL % gu.bn == 0 and BLOCKS % gu.ks == 0
    p = Plan()
    p.region("lrs", TP * m * LSL * 2)
    p.region("srs", TP * m * SL * 2)
    p.region("rsf", TP * m * 8)
    p.region("lag", m * LAT * 2)
    p.region("lss", TP * m * 8)
    p.region("lagf", TP * m * 8)
    p.region("xs", TP * m * scalars(nvb) * 8)
    p.region("ag", m * H * 2)
    p.region("agf", TP * m * 8)
    p.region("shs", m * SL * 4, peer=False)
    p.region("pdone", nb * ct * 8, peer=False)
    tasks_u = ((m + gu.bm - 1) // gu.bm) * (SL // gu.bn) * gu.ks
    tasks_i = ((m + gi.bm - 1) // gi.bm) * ((nin + gi.bn - 1) // gi.bn) * gi.ks
    p.stage("push", m * TP)
    p.stage("lred", m, waits_on=("push",), all_ranks=True)
    p.stage("up", tasks_u, waits_on=("lred", "up"), all_ranks=True)
    p.stage("seam", m, waits_on=("up", "seam"), all_ranks=True)
    p.stage("inproj", tasks_i, waits_on=("seam",), all_ranks=True)
    return p.check()


@flyc.jit
def _push(c, row, j, m, lat_p, sh_p):
    """This rank's latent and shared columns of owner j for ``row`` -> the owner."""
    tid = c["tid"]
    rank = c["rank"]
    if tid < LSL // 8:
        v = fx.Vector(bo.buffer_load(rsrc(lat_p), (row * LAT + j * LSL + 8 * tid) // 2, vec_width=4, dtype=T.i32))
        bo.buffer_store(v, rsrc(c["lrs_of"](j).value), ((rank * m + row) * LSL + 8 * tid) // 2, cache_modifier=CM_SYS)
    u = tid - 64
    if (u >= 0) & (u < SL // 8):
        v = fx.Vector(bo.buffer_load(rsrc(sh_p), (row * H + j * SL + 8 * u) // 2, vec_width=4, dtype=T.i32))
        bo.buffer_store(v, rsrc(c["srs_of"](j).value), ((rank * m + row) * SL + 8 * u) // 2, cache_modifier=CM_SYS)
    rocdl.s_waitcnt(vmcnt=0)
    gpu.barrier()
    if tid == 0:
        c["put"](c["rsf_of"](j), rank * m + row, fx.Int32(1))


def _sum8(c, region, base, stride):
    got = [fx.Vector(bo.buffer_load(rsrc(c[region].value), (src * stride + base) // 2, vec_width=4, dtype=T.i32,
                                    cache_modifier=CM_SYS)) for src in range(TP)]
    acc = None
    for src in range_constexpr(TP):
        cur = [v for e in range(4) for v in bf2_f32(got[src][e])]
        acc = cur if acc is None else [a + b for a, b in zip(acc, cur)]
    return acc


@flyc.jit
def _lred(c, row, m, shs):
    """The owner's row: latent sum -> every rank's gather region + the slice's sum
    of squares (tagged) to every rank; shared sum (f32) -> local scratch."""
    tid = c["tid"]
    rank = c["rank"]
    lane = tid % 64
    if tid < TP:
        c["poll"]([(c["rsf_own"], tid * m + row, 1)])
    gpu.barrier()
    if tid < 64:
        live = tid < LSL // 8
        q = fx.min(tid, LSL // 8 - 1)
        acc = _sum8(c, "lrs_own", row * LSL + 8 * q, m * LSL)
        words = [bf16_pair(acc[2 * k], acc[2 * k + 1]).bitcast(fx.Int32) for k in range(4)]
        vb = [x for w in words for x in bf2_f32(w)]
        part = vb[0] * vb[0]
        for x in vb[1:]:
            part = part + x * x
        ss = wave_sum(live.select(part, fx.Float32(0.0)))
        vec = fx.Vector.from_elements(words, fx.Int32)
        if live:
            for p in range_constexpr(TP):
                bo.buffer_store(vec, rsrc(c["lag_of"](p).value), (row * LAT + rank * LSL + 8 * q) // 2,
                                cache_modifier=CM_SYS)
        if lane == 0:
            for p in range_constexpr(TP):
                c["put"](c["lss_of"](p), rank * m + row, ss)
    u = tid - 64
    if (u >= 0) & (u < SL // 8):
        acc = _sum8(c, "srs_own", row * SL + 8 * u, m * SL)
        for k in range_constexpr(2):
            bo.buffer_store(fx.Vector.from_elements(acc[4 * k : 4 * k + 4], fx.Float32), rsrc(shs),
                            row * SL + 8 * u + 4 * k, cache_modifier=CM_DEV)
    rocdl.s_waitcnt(vmcnt=0)
    gpu.barrier()
    for p in range_constexpr(TP):
        if tid == p:
            c["put"](c["lagf_of"](p), rank * m + row, fx.Int32(1))


@flyc.jit
def _rows_wait(c, flags, m0, bm, m, j0, nown):
    """Wait for the (row, owner) flags in region ``flags`` of rows m0 .. m0 + bm and
    owners j0 .. j0 + nown."""
    rows = fx.min(m0 + bm, m) - m0
    for i in range(c["tid"], rows * nown, THREADS):
        c["poll"]([(c[flags], (j0 + i % nown) * m + m0 + i // nown, 1)])
    gpu.barrier()


@flyc.jit
def _row_rms(c, m0, bm, m, eps, rmsl):
    """rsqrt(mean square) of the gathered latent rows m0 .. m0 + bm -> LDS."""
    tid = c["tid"]
    if tid < fx.min(bm, m - m0):
        got = c["poll"]([(c["lss_own"], src * m + m0 + tid, 1) for src in range(TP)])
        tot = got[0][0].bitcast(fx.Float32)
        for src in range_constexpr(1, TP):
            tot = tot + got[src][0].bitcast(fx.Float32)
        fx.ptr_store(hw_rsq(tot * (1.0 / LAT) + eps), rmsl + tid)
    gpu.barrier()


@flyc.jit
def _finish(c, acc, shs, prefix, prefix_out, m, r0, rb, n0, bn, ks, pdone, piece):
    """Rows r0 .. r0 + rb of a finished up tile: y = up + shared, prefix' = prefix +
    y (bf16 roundings as SGLang's two adds), zero the up tile, mark the piece."""
    tid = c["tid"]
    q4 = bn // 4
    for it in range(tid, rb * q4, THREADS):
        row = r0 + it // q4
        col = n0 + 4 * (it % q4)
        if row < m:
            uv = fx.Vector(bo.buffer_load(rsrc(acc), (row * SL + col) // 2, vec_width=2, dtype=T.i32,
                                          cache_modifier=CM_DEV))
            sv = fx.Vector(bo.buffer_load(rsrc(shs), row * SL + col, vec_width=4, dtype=T.f32, cache_modifier=CM_DEV))
            pv = fx.Vector(bo.buffer_load(rsrc(prefix), (row * SL + col) // 2, vec_width=2, dtype=T.i32))
            up = [x for w in (uv[0], uv[1]) for x in bf2_f32(w)]
            pr = [x for w in (pv[0], pv[1]) for x in bf2_f32(w)]
            y = [fx.Float32(fx.Float32(up[e] + sv[e]).to(fx.BFloat16)) for e in range(4)]
            o = [y[e] + pr[e] for e in range(4)]
            words = [bf16_pair(o[0], o[1]).bitcast(fx.Int32), bf16_pair(o[2], o[3]).bitcast(fx.Int32)]
            bo.buffer_store(fx.Vector.from_elements(words, fx.Int32), rsrc(prefix_out), (row * SL + col) // 2,
                            cache_modifier=CM_DEV)
            if const_expr(ks > 1):
                bo.buffer_store(fx.Vector.filled(2, 0, fx.Int32), rsrc(acc), (row * SL + col) // 2,
                                cache_modifier=CM_DEV)
    rocdl.s_waitcnt(vmcnt=0)
    gpu.barrier()
    if tid == 0:
        c["put"](pdone, piece, fx.Int32(1))


@flyc.jit
def _seam(c, row, m, rb, ct, nvb, nbk, prefix_out, bank, cw, w_in, eps, pdone, red, write):
    tid = c["tid"]
    if tid < ct:
        c["poll"]([(pdone, (row // rb) * ct + tid, 1)])
    gpu.barrier()
    rank = c["rank"]
    ags = [c["ag_of"](p) for p in range(TP)]

    def emit(q, live, vec):
        if live:
            for p in range_constexpr(TP):
                bo.buffer_store(vec, rsrc(ags[p].value), (row * H + rank * SL + 8 * q) // 2, cache_modifier=CM_SYS)

    attnres_row(c, row, m, SL, nvb, nbk, prefix_out, bank, cw, w_in, eps, eps, red, emit, write)
    rocdl.s_waitcnt(vmcnt=0)
    gpu.barrier()
    for p in range_constexpr(TP):
        if tid == p:
            c["put"](c["agf_of"](p), rank * m + row, fx.Int32(1))


_BUILDS: dict = {}


def pre_attn(m, gu, gi, nin, nvb=0, nbk=8, write=False, mode="linbf", launches=1, eps_n=1e-5, eps=1e-5):
    """Build the launch; returns (launch, plan). Arguments: lat_p [m, LAT] / sh_p [m, H]
    (this rank's latent / shared partials), w_up [SL, LAT] (``fold_up``), prefix /
    prefix_out [m, SL], bank [m, nbk, SL], cw f32 [H], w_in [H] (input norm), w_ip
    [nin, H] (in_proj), out (bf16 [m, nin], f32 [ks, m, nin] for "part"), acc bf16
    [m, SL] / cnt [up tiles] (zeroed once), scratch (``plan.scratch_bytes``), peers,
    rank, epoch, slot. The up GEMM merges linbf (or writes at ks 1)."""
    key = (m, gu.key(), gi.key(), nin, nvb, nbk, write, mode, launches, eps_n, eps)
    if key in _BUILDS:
        return _BUILDS[key]
    pl = plan(m, gu, gi, nin, nvb)
    offs, half = pl.peer_offsets
    soffs, _ = pl.scratch_offsets
    fo = pl.flags_offset()
    rb = gu.bm // gu.ks
    ct = SL // gu.bn
    ktu, kti = LAT // gu.ks, H // gi.ks
    assert ktu % 64 == 0 and kti % 64 == 0 and (ktu % LSL == 0 or LSL % ktu == 0) and (kti % SL == 0 or SL % kti == 0)
    nown_u, nown_i = max(ktu // LSL, 1), max(kti // SL, 1)
    tnu, tmu = SL // gu.bn, (m + gu.bm - 1) // gu.bm
    tni, tmi = (nin + gi.bn - 1) // gi.bn, (m + gi.bm - 1) // gi.bm
    n_u, n_i = tmu * tnu * gu.ks, tmi * tni * gi.ks
    bmx, bnx = max(gu.bm, gi.bm), max(gu.bn, gi.bn)
    cw_words = max(gu.bm * (gu.bn + 4), gi.bm * (gi.bn + 4))
    ipart = mode == "part" and gi.ks > 1

    @fx.struct
    class AbLds:
        xs: fx.Array[fx.Int32, 2 * bmx * RW, 16]
        ws: fx.Array[fx.Int32, 2 * bnx * RW, 16]

    @fx.struct
    class CLds:
        c: fx.Array[fx.Float32, cw_words, 16]

    @fx.union
    class Smem:
        ab: AbLds
        c: CLds

    @fx.struct
    class Small:
        slot: fx.Array[fx.Int32, 4, 16]
        rms: fx.Array[fx.Float32, gu.bm, 16]
        red: fx.Array[fx.Float32, red_words(nvb), 16]

    name = (f"k31_pre_attn_m{m}_u{'x'.join(map(str, gu.key()))}_i{'x'.join(map(str, gi.key()))}_n{nin}_v{nvb}of{nbk}"
            f"w{int(write)}_{mode}_l{launches}")

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def pre_attn_kernel(lat_p: Int64, sh_p: Int64, w_up: Int64, prefix: Int64, prefix_out: Int64, bank: Int64,
                        cw: Int64, w_in: Int64, w_ip: Int64, out: Int64, acc: Int64, cnt: Int64, scratch: Int64,
                        peers: Int64, rank: Int32, epoch: Int64, slot_i: Int32):
        own = load_ptr64(peers, rank)
        with guard.scope(guard.Guard(epoch, own=own, peers=peers, rank=rank, npes=TP, flags_offset=fo)):
            tid, bid = fx.thread_idx.x, fx.block_idx.x
            alloc = fx.SharedAllocator()
            st = alloc.allocate(Smem)
            small = alloc.allocate(Small).peek()
            slot, rmsl, red = small.slot.ptr, small.rms.ptr, small.red.ptr
            lds, cl = st.ab.peek(), st.c.peek().c.ptr
            c = coll.region_context(peers, rank, epoch, slot_i, offs, half, tid, bid, launches)
            shs = scratch + fx.Int64(soffs["shs"])
            pdone = sreg(scratch, soffs["pdone"], "pdone")
            # push
            for t in range(bid, m * TP, BLOCKS):
                _push(c, t // TP, t % TP, m, lat_p, sh_p)
            # lred
            for row in range(bid, m, BLOCKS):
                _lred(c, row, m, shs)
            # up
            lagx = c["lag_own"].value
            for task in range(bid, n_u, BLOCKS):
                sk = task % gu.ks
                tile = task // gu.ks
                m0 = (tile // tnu) * gu.bm
                n0 = (tile % tnu) * gu.bn
                _rows_wait(c, "lagf_own", m0, gu.bm, m, (sk * ktu) // LSL, nown_u)
                _row_rms(c, m0, gu.bm, m, eps_n, rmsl)
                lin, emit = None, None
                if const_expr(gu.ks == 1):
                    base = dev_emit(acc, SL, n0, m, SL, False)

                    def emit(row, col, v, live, base=base, m0=m0):
                        s = fx.ptr_load(rmsl + fx.min(row - m0, gu.bm - 1))
                        base(row, col, fx.Vector.from_elements([v[e] * s for e in range(4)], fx.Float32), live)
                else:
                    e_, add = atomic_lin(acc, SL, n0, m, SL, True)

                    def scaled(row, col, vals, add=add, m0=m0):
                        s = fx.ptr_load(rmsl + fx.min(row - m0, gu.bm - 1))
                        add(row, col, [v * s for v in vals])

                    lin = (e_, scaled)
                mma_tile(tid, lagx, LAT, m, m0, w_up, LAT, SL, n0, sk * ktu, ktu, gu.bm, gu.bn, gu.wm, gu.wn,
                         lds.xs.ptr, lds.ws.ptr, emit, DEPTH, lin, cl)
                _arrive_all(tid, cnt, tile, gu.ks, slot)
                r0 = m0 + sk * rb
                _finish(c, acc, shs, prefix, prefix_out, m, r0, rb, n0, gu.bn, gu.ks, pdone, (r0 // rb) * ct + n0 // gu.bn)
            # seam
            for row in range(bid, m, BLOCKS):
                _seam(c, row, m, rb, ct, nvb, nbk, prefix_out, bank, cw, w_in, eps, pdone, red, write)
            # inproj
            x = c["ag_own"].value
            f0 = first_task(bid, max(BLOCKS - n_i, 0))
            for task in range(f0, n_i, BLOCKS):
                sk = task % gi.ks
                tile = task // gi.ks
                m0 = (tile // tni) * gi.bm
                n0 = (tile % tni) * gi.bn
                _rows_wait(c, "agf_own", m0, gi.bm, m, (sk * kti) // SL, nown_i)
                lin, emit = None, None
                if const_expr(gi.ks == 1):
                    emit = bf16_emit(out, nin, n0, m, nin)
                elif const_expr(ipart):
                    emit = f32_emit(out + fx.Int64(sk) * (m * nin * 4), nin, n0, m, nin)
                else:
                    lin = atomic_lin(out, nin, n0, m, nin, True)
                mma_tile(tid, x, H, m, m0, w_ip, H, nin, n0, sk * kti, kti, gi.bm, gi.bn, gi.wm, gi.wn, lds.xs.ptr,
                         lds.ws.ptr, emit, DEPTH, lin, cl)

    @flyc.jit
    def launch(lat_p: Int64, sh_p: Int64, w_up: Int64, prefix: Int64, prefix_out: Int64, bank: Int64, cw: Int64,
               w_in: Int64, w_ip: Int64, out: Int64, acc: Int64, cnt: Int64, scratch: Int64, peers: Int64,
               rank: Int32, epoch: Int64, slot_i: Int32, stream: fx.Stream = fx.Stream(None)):  # noqa: B008
        pre_attn_kernel(lat_p, sh_p, w_up, prefix, prefix_out, bank, cw, w_in, w_ip, out, acc, cnt, scratch, peers,
                        rank, epoch, slot_i).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)

    _BUILDS[key] = (launch, pl)
    return launch, pl
