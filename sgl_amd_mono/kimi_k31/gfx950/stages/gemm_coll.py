# SPDX-License-Identifier: Apache-2.0
"""Tile GEMMs fused with the column-slice collectives (one launch).

GEMM -> RS (o_proj, shared down): every rank computes its partial
out[m, n] = x @ W^T in [bm, bn] tiles split over ``ks`` K slices. A K slice
merges into a local scratch tile (device scope: the reader may sit on another
XCD), then counts itself in the tile's arrival counter and waits for the
tile's other slices; each slice then reads 1 / ks of the finished tile's rows,
pushes them to the owner of its columns (``n0 // sl``, so bn must divide sl)
and, with linbf, zeroes them (the consumer zeroes what it reads). After their
GEMM tasks the CTAs reduce this rank's slice as the pieces land, so the RS
overlaps the GEMM tail.

Remote stores are the cost: one CTA pushes only ~2-4 GB/s, so the push is
spread over every slice's CTA, and at 128+ rows the tagged format's doubled
bytes reach the links' limit, so "flag" (plain stores + a flag a piece) wins.
With ks = 1 there is no wait, no atomic and nothing to zero (deterministic).

Merge modes: "linbf" packed-bf16 atomics into a zeroed bf16 tile (K-slice
order follows timing); "part" f32 partials [ks, m, n] the last slice sums in
order (deterministic). The counters only grow: a launch adds ks a tile, so
the last arrival is the one that makes the count a multiple of ks.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import Int32, Int64, T

from sgl_amd_mono.common import guard
from sgl_amd_mono.common.ops import (
    CM_DEV,
    CM_SYS,
    bf2_f32,
    bf16_pair,
    ld_i32,
    load_ptr64,
    rsrc,
)
from sgl_amd_mono.common.plan import BLOCKS, THREADS, kernel_sources, key_tuple
from sgl_amd_mono.common.sync import _spin
from sgl_amd_mono.kimi_k31.gfx950.stages import coll
from sgl_amd_mono.kimi_k31.gfx950.stages.tile import atomic_lin
from sgl_amd_mono.kimi_k31.gfx950.stages.tile_mma import DEPTH, RW, mma_tile

SOURCES = kernel_sources(__name__)

TP = coll.TP
MODES = ("linbf", "part")


def scratch_bytes(m, n, ks, mode):
    """The local merge scratch: a bf16 [m, n] tile buffer (zeroed once) or f32 partials."""
    return ks * m * n * 4 if mode == "part" and ks > 1 else m * n * 2


def counters(m, n, bm, bn):
    return ((m + bm - 1) // bm) * (n // bn)


@flyc.jit
def _arrive(tid, cnt, tile, ks, slot):
    """Drain this CTA's merge, count it in ``tile``'s counter; True on the last slice."""
    rocdl.s_waitcnt(vmcnt=0)
    gpu.barrier()
    if tid == 0:
        ptr = fx.inttoptr(
            fx.PointerType.get(T.i32, fx.AddressSpace.Global, 4),
            cnt + fx.Int64(tile * 4),
        )
        fx.ptr_store(fx.llvm.atomic_add(ptr, fx.Int32(1), syncscope="agent"), slot)
    gpu.barrier()
    return (fx.ptr_load(slot) + 1) % ks == 0


@flyc.jit
def _arrive_all(tid, cnt, tile, ks, slot, stride=1):
    """Count this K slice in ``tile``'s counter, then wait until every slice of
    the tile has: they are consecutive tasks of one round, so the wait is short."""
    rocdl.s_waitcnt(vmcnt=0)
    gpu.barrier()
    if tid == 0:
        ptr = fx.inttoptr(
            fx.PointerType.get(T.i32, fx.AddressSpace.Global, 4),
            cnt + fx.Int64(tile * (4 * stride)),
        )
        old = fx.llvm.atomic_add(ptr, fx.Int32(1), syncscope="agent")
        target = (old // ks + 1) * ks
        _spin(lambda: ld_i32(cnt, tile * stride, CM_DEV), lambda v: (v - target) < 0)
    gpu.barrier()


def dev_emit(out, ldo, n0, m, n, f32):
    """``emit`` storing bf16 (or f32 partials) at device scope: another XCD's CTA reads them."""
    r_o = rsrc(out, m * ldo * (4 if f32 else 2))

    def emit(row, col, v, live):
        ok = (row < m) & (n0 + col < n)
        ok = ok if live is None else ok & live
        if const_expr(f32):
            bo.buffer_store(
                v, r_o, row * ldo + n0 + col, mask=ok, cache_modifier=CM_DEV
            )
        else:
            words = fx.Vector.from_elements(
                [
                    bf16_pair(v[0], v[1]).bitcast(fx.Int32),
                    bf16_pair(v[2], v[3]).bitcast(fx.Int32),
                ],
                fx.Int32,
            )
            bo.buffer_store(
                words, r_o, (row * ldo + n0 + col) // 2, mask=ok, cache_modifier=CM_DEV
            )

    return emit


@flyc.jit
def _push_tile(c, acc, m, n, m0, rb, n0, bn, ks, mode, sl):
    """Read rows m0 .. m0 + rb of the finished tile at column n0 (bn wide) and push
    them to their owner's rs region: every load is issued before the first push."""
    owner = n0 // sl
    dst = c["rs_of"](owner)
    r = rsrc(acc)
    q4 = bn // 4
    units = rb * q4
    base = c["rank"] * m * sl - owner * sl
    part = mode == "part" and ks > 1
    got = []
    for i in range_constexpr((units + THREADS - 1) // THREADS):
        it = fx.min(c["tid"] + i * THREADS, units - 1)
        row = m0 + it // q4
        rowc = fx.min(row, m - 1)
        col = n0 + 4 * (it % q4)
        if const_expr(part):
            vs = [
                fx.Vector(
                    bo.buffer_load(
                        r,
                        k * m * n + rowc * n + col,
                        vec_width=4,
                        dtype=T.f32,
                        cache_modifier=CM_DEV,
                    )
                )
                for k in range(ks)
            ]
        else:
            vs = [
                fx.Vector(
                    bo.buffer_load(
                        r,
                        (rowc * n + col) // 2,
                        vec_width=2,
                        dtype=T.i32,
                        cache_modifier=CM_DEV,
                    )
                )
            ]
        got.append((row, col, vs))
    for row, col, vs in got:
        if const_expr(part):
            s = [vs[0][e] for e in range(4)]
            for k in range_constexpr(1, ks):
                s = [s[e] + vs[k][e] for e in range(4)]
            words = [
                bf16_pair(s[0], s[1]).bitcast(fx.Int32),
                bf16_pair(s[2], s[3]).bitcast(fx.Int32),
            ]
        else:
            words = [vs[0][0], vs[0][1]]
        if row < m:
            c["put_words"](dst, (base + row * sl + col) // 2, words)
            if const_expr(not part and ks > 1):
                bo.buffer_store(
                    fx.Vector.filled(2, 0, fx.Int32),
                    r,
                    (row * n + col) // 2,
                    cache_modifier=CM_DEV,
                )


@flyc.jit
def _push_band_flag(c, acc, m, n, m0, rb, n0, bn, ks, mode, sl, piece):
    """Flag format: rows m0 .. m0 + rb of the finished tile as plain 16 B stores into
    the owner's rs region [src][m][sl], drained, then flag (this rank, piece)."""
    owner = n0 // sl
    dst = c["rs_of"](owner)
    r = rsrc(acc)
    q8 = bn // 8
    units = rb * q8
    part = mode == "part" and ks > 1
    got = []
    for i in range_constexpr((units + THREADS - 1) // THREADS):
        it = fx.min(c["tid"] + i * THREADS, units - 1)
        row = m0 + it // q8
        rowc = fx.min(row, m - 1)
        col = n0 + 8 * (it % q8)
        if const_expr(part):
            vs = [
                fx.Vector(
                    bo.buffer_load(
                        r,
                        k * m * n + rowc * n + col + 4 * h,
                        vec_width=4,
                        dtype=T.f32,
                        cache_modifier=CM_DEV,
                    )
                )
                for k in range(ks)
                for h in range(2)
            ]
        else:
            vs = [
                fx.Vector(
                    bo.buffer_load(
                        r,
                        (rowc * n + col) // 2,
                        vec_width=4,
                        dtype=T.i32,
                        cache_modifier=CM_DEV,
                    )
                )
            ]
        got.append((row, col, vs))
    for row, col, vs in got:
        if const_expr(part):
            s = [vs[h][e] for h in range(2) for e in range(4)]
            for k in range_constexpr(1, ks):
                s = [
                    s[4 * h + e] + vs[2 * k + h][e] for h in range(2) for e in range(4)
                ]
            words = [
                bf16_pair(s[2 * j], s[2 * j + 1]).bitcast(fx.Int32) for j in range(4)
            ]
        else:
            words = [vs[0][e] for e in range(4)]
        if row < m:
            bo.buffer_store(
                fx.Vector.from_elements(words, fx.Int32),
                rsrc(dst.value),
                (c["rank"] * m * sl + row * sl + col - owner * sl) // 2,
                cache_modifier=CM_SYS,
            )
            if const_expr(not part and ks > 1):
                bo.buffer_store(
                    fx.Vector.filled(4, 0, fx.Int32),
                    r,
                    (row * n + col) // 2,
                    cache_modifier=CM_DEV,
                )
    rocdl.s_waitcnt(vmcnt=0)
    gpu.barrier()
    # a partial tile's pieces past row m: no piece the owner reduces
    if (c["tid"] == 0) & (m0 < m):
        c["put"](c["fl_of"](owner), c["rank"] * BLOCKS + piece, fx.Int32(1))


@flyc.jit
def _reduce_flag(c, m, sl, rb, bn, y):
    """The owner's slice by pieces (rb rows x bn columns): wait for the TP sources'
    flags of a piece, then sum them in rank order."""
    q8 = bn // 8
    ct = sl // bn
    npieces = ((m + rb - 1) // rb) * ct
    r_y = rsrc(y)
    tid = c["tid"]
    for piece in range(c["bid"], npieces, BLOCKS):
        if tid < TP:
            c["poll"]([(c["fl_own"], tid * BLOCKS + piece, 1)])
        gpu.barrier()
        band, cj = piece // ct, piece % ct
        for u in range(tid, rb * q8, THREADS):
            row = band * rb + u // q8
            col = cj * bn + 8 * (u % q8)
            if row < m:
                got = [
                    fx.Vector(
                        bo.buffer_load(
                            rsrc(c["rs_own"].value),
                            (src * m * sl + row * sl + col) // 2,
                            vec_width=4,
                            dtype=T.i32,
                            cache_modifier=CM_SYS,
                        )
                    )
                    for src in range(TP)
                ]
                acc = None
                for src in range_constexpr(TP):
                    cur = [v for e in range(4) for v in bf2_f32(got[src][e])]
                    acc = cur if acc is None else [a + b for a, b in zip(acc, cur)]
                words = [
                    bf16_pair(acc[2 * j], acc[2 * j + 1]).bitcast(fx.Int32)
                    for j in range(4)
                ]
                bo.buffer_store(
                    fx.Vector.from_elements(words, fx.Int32), r_y, (row * sl + col) // 2
                )


_BUILDS: dict = {}


def gemm_rs(
    m,
    n,
    k,
    bm=128,
    bn=128,
    wm=2,
    wn=4,
    ks=4,
    depth=DEPTH,
    mode="linbf",
    phases=3,
    spread=True,
    fmt="tag",
    launches=1,
    stress=False,
):
    """y[m, n / TP] = this rank's slice of sum over ranks of x_r @ W_r^T, one launch.
    ``acc``: ``scratch_bytes`` zeroed once; ``cnt``: ``counters`` int32 zeroed once.
    ``phases`` != 3 (timing only): 1 the GEMM, 4 + the arrival, 2 + the pushes.
    ``spread``: every K slice of a tile waits for the others and pushes 1 / ks of its
    rows (else the last arrival pushes it all). ``fmt``: "tag" pairs, or "flag"
    plain stores + a flag a (rows, tile) piece (half the bytes; needs spread).
    ``slot_i`` / ``launches``: the launch's place in its step (``coll.context``);
    ``stress``: CTA b waits delay[b] ticks before each arrival (skews the slices)."""
    keyed = key_tuple(locals(), SOURCES)
    key = (
        m,
        n,
        k,
        bm,
        bn,
        wm,
        wn,
        ks,
        depth,
        mode,
        phases,
        spread,
        fmt,
        launches,
        stress,
    )
    if key in _BUILDS:
        return _BUILDS[key]
    sl = n // TP
    kt = k // ks
    assert mode in MODES and k % ks == 0 and kt % 64 == 0 and sl % bn == 0
    # a tile's slices in one round (no wait on a later round), and whole rows a slice
    assert BLOCKS % ks == 0 and (not spread or bm % ks == 0)
    tn, tm = n // bn, (m + bm - 1) // bm
    ntasks = tm * tn * ks
    fo = coll.flags_offset(m, n, fmt)
    rb = bm // ks if spread else bm
    assert fmt == "tag" or (spread and ((m + rb - 1) // rb) * (sl // bn) <= BLOCKS)

    @fx.struct
    class AbLds:
        xs: fx.Array[fx.Int32, 2 * bm * RW, 16]
        ws: fx.Array[fx.Int32, 2 * bn * RW, 16]

    @fx.struct
    class CLds:
        c: fx.Array[fx.Float32, bm * (bn + 4), 16]

    @fx.union
    class Smem:
        ab: AbLds
        c: CLds

    @fx.struct
    class Slot:
        v: fx.Array[fx.Int32, 4, 16]

    name = f"k31_gemm_rs_m{m}_n{n}_k{k}_t{bm}x{bn}_w{wm}x{wn}_ks{ks}_d{depth}_{mode}_p{phases}_s{int(spread)}_{fmt}_l{launches}_x{int(stress)}"

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def gemm_rs_kernel(
        x: Int64,
        w: Int64,
        y: Int64,
        acc: Int64,
        cnt: Int64,
        peers: Int64,
        rank: Int32,
        epoch: Int64,
        delay: Int64,
        slot_i: Int32,
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
            lds, cl = st.ab.peek(), st.c.peek().c.ptr
            c = coll.context(peers, rank, epoch, slot_i, m, n, fmt, tid, bid, launches)
            # a tile's K slices are consecutive tasks: they finish close together
            for task in range(bid, ntasks, BLOCKS):
                sk = task % ks
                tile = task // ks
                m0 = (tile // tn) * bm
                n0 = (tile % tn) * bn
                lin = None
                emit = None
                if const_expr(ks == 1):
                    emit = dev_emit(acc, n, n0, m, n, False)
                elif const_expr(mode == "part"):
                    emit = dev_emit(acc + fx.Int64(sk) * (m * n * 4), n, n0, m, n, True)
                else:
                    lin = atomic_lin(acc, n, n0, m, n, True)
                mma_tile(
                    tid,
                    x,
                    k,
                    m,
                    m0,
                    w,
                    k,
                    n,
                    n0,
                    sk * kt,
                    kt,
                    bm,
                    bn,
                    wm,
                    wn,
                    lds.xs.ptr,
                    lds.ws.ptr,
                    emit,
                    depth,
                    lin,
                    cl,
                )
                if const_expr(phases >= 2 and spread):
                    if const_expr(stress):
                        coll._delay(tid, ld_i32(delay, bid))
                    _arrive_all(tid, cnt, tile, ks, slot)
                    if const_expr(phases != 4 and fmt == "flag"):
                        piece = ((m0 + sk * rb) // rb) * (sl // bn) + (n0 % sl) // bn
                        _push_band_flag(
                            c, acc, m, n, m0 + sk * rb, rb, n0, bn, ks, mode, sl, piece
                        )
                    elif const_expr(phases != 4):
                        _push_tile(c, acc, m, n, m0 + sk * rb, rb, n0, bn, ks, mode, sl)
                elif const_expr(phases >= 2):
                    if _arrive(tid, cnt, tile, ks, slot):
                        if const_expr(phases != 4):
                            _push_tile(c, acc, m, n, m0, bm, n0, bn, ks, mode, sl)
            if const_expr(phases == 3 and fmt == "flag"):
                _reduce_flag(c, m, sl, rb, bn, y)
            elif const_expr(phases == 3):
                r_y = rsrc(y)

                def store(row, col, words):
                    bo.buffer_store(
                        fx.Vector.from_elements(words, fx.Int32),
                        r_y,
                        (row * sl + col) // 2,
                    )

                coll.tag_reduce(c, m, sl, bid * THREADS + tid, BLOCKS * THREADS, store)

    @flyc.jit
    def launch(
        x: Int64,
        w: Int64,
        y: Int64,
        acc: Int64,
        cnt: Int64,
        peers: Int64,
        rank: Int32,
        epoch: Int64,
        delay: Int64,
        slot_i: Int32,
        stream: fx.Stream = fx.Stream(None),
    ):  # noqa: B008
        _ = keyed
        gemm_rs_kernel(x, w, y, acc, cnt, peers, rank, epoch, delay, slot_i).launch(
            grid=(BLOCKS,), block=(THREADS,), stream=stream
        )

    _BUILDS[key] = launch
    return launch


@flyc.jit
def _ag_push(c, xs, m, sl, cnt):
    """Push this rank's x slice [m, sl] to every rank's ag region [m, TP sl] (plain
    16 B stores; CTA group j feeds rank j); the last CTA to finish flags every rank."""
    grp = BLOCKS // TP
    j = c["bid"] // grp
    dst = c["ag_of"](j)
    q8 = sl // 8
    h = TP * sl
    r_x = rsrc(xs)
    for u in range((c["bid"] % grp) * THREADS + c["tid"], m * q8, grp * THREADS):
        row, q = u // q8, u % q8
        v = fx.Vector(
            bo.buffer_load(r_x, (row * sl + 8 * q) // 2, vec_width=4, dtype=T.i32)
        )
        bo.buffer_store(
            v,
            rsrc(dst.value),
            (row * h + c["rank"] * sl + 8 * q) // 2,
            cache_modifier=CM_SYS,
        )
    rocdl.s_waitcnt(vmcnt=0)
    gpu.barrier()
    if c["tid"] == 0:
        ptr = fx.inttoptr(fx.PointerType.get(T.i32, fx.AddressSpace.Global, 4), cnt)
        old = fx.llvm.atomic_add(ptr, fx.Int32(1), syncscope="agent")
        if (old + 1) % BLOCKS == 0:
            for p in range_constexpr(TP):
                c["put"](c["fl_of"](p), c["rank"], fx.Int32(1))


@flyc.jit
def _ag_wait(c, j0, nown):
    """Wait for the flags of owners j0 .. j0 + nown (their slices have landed here)."""
    if c["tid"] < nown:
        c["poll"]([(c["fl_own"], j0 + c["tid"], 1)])
    gpu.barrier()


def ag_gemm(
    m,
    n,
    k,
    bm=128,
    bn=128,
    wm=2,
    wn=4,
    ks=4,
    depth=DEPTH,
    mode="linbf",
    phases=3,
    launches=1,
    stress=False,
):
    """out[m, n] = AG(x slices) @ W^T in one launch: every rank pushes its x slice
    [m, k / TP] (``xs``), and a GEMM task starts once the owners of its K range
    have landed, reading x from the ag region. ``out``: bf16 [m, n] (zeroed for
    linbf, ks > 1) or f32 partials [ks, m, n] ("part"). ``cnt``: one int32, zeroed
    once. ``phases`` 1 (timing only): the GEMM on the ag region, no push, no wait.
    ``slot_i`` / ``launches``: the launch's place in its step (``coll.context``);
    ``stress``: CTA b waits delay[b] ticks before the push, delay[BLOCKS + b] after."""
    keyed = key_tuple(locals(), SOURCES)
    key = (m, n, k, bm, bn, wm, wn, ks, depth, mode, phases, launches, stress)
    if key in _AG_BUILDS:
        return _AG_BUILDS[key]
    sl = k // TP
    kt = k // ks
    assert (
        mode in MODES
        and k % ks == 0
        and kt % 64 == 0
        and (kt % sl == 0 or sl % kt == 0)
    )
    nown = max(kt // sl, 1)
    tn, tm = (n + bn - 1) // bn, (m + bm - 1) // bm
    ntasks = tm * tn * ks
    fo = coll.flags_offset(m, k, "flag")

    @fx.struct
    class AbLds:
        xs: fx.Array[fx.Int32, 2 * bm * RW, 16]
        ws: fx.Array[fx.Int32, 2 * bn * RW, 16]

    @fx.struct
    class CLds:
        c: fx.Array[fx.Float32, bm * (bn + 4), 16]

    @fx.union
    class Smem:
        ab: AbLds
        c: CLds

    name = f"k31_ag_gemm_m{m}_n{n}_k{k}_t{bm}x{bn}_w{wm}x{wn}_ks{ks}_d{depth}_{mode}_p{phases}_l{launches}_x{int(stress)}"

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def ag_gemm_kernel(
        xs: Int64,
        w: Int64,
        out: Int64,
        cnt: Int64,
        peers: Int64,
        rank: Int32,
        epoch: Int64,
        delay: Int64,
        slot_i: Int32,
    ):
        own = load_ptr64(peers, rank)
        with guard.scope(
            guard.Guard(
                epoch, own=own, peers=peers, rank=rank, npes=TP, flags_offset=fo
            )
        ):
            tid, bid = fx.thread_idx.x, fx.block_idx.x
            st = fx.SharedAllocator().allocate(Smem)
            lds, cl = st.ab.peek(), st.c.peek().c.ptr
            c = coll.context(
                peers, rank, epoch, slot_i, m, k, "flag", tid, bid, launches
            )
            if const_expr(stress):
                coll._delay(tid, ld_i32(delay, bid))
            if const_expr(phases == 3):
                _ag_push(c, xs, m, sl, cnt)
            if const_expr(stress):
                coll._delay(tid, ld_i32(delay, BLOCKS + bid))
            x = c["ag_own"].value
            for task in range(bid, ntasks, BLOCKS):
                sk = task % ks
                tile = task // ks
                m0 = (tile // tn) * bm
                n0 = (tile % tn) * bn
                if const_expr(phases == 3):
                    _ag_wait(c, (sk * kt) // sl, nown)
                lin = None
                emit = None
                if const_expr(ks == 1):
                    emit = dev_emit(out, n, n0, m, n, False)
                elif const_expr(mode == "part"):
                    emit = dev_emit(out + fx.Int64(sk) * (m * n * 4), n, n0, m, n, True)
                else:
                    lin = atomic_lin(out, n, n0, m, n, True)
                mma_tile(
                    tid,
                    x,
                    k,
                    m,
                    m0,
                    w,
                    k,
                    n,
                    n0,
                    sk * kt,
                    kt,
                    bm,
                    bn,
                    wm,
                    wn,
                    lds.xs.ptr,
                    lds.ws.ptr,
                    emit,
                    depth,
                    lin,
                    cl,
                )

    @flyc.jit
    def launch(
        xs: Int64,
        w: Int64,
        out: Int64,
        cnt: Int64,
        peers: Int64,
        rank: Int32,
        epoch: Int64,
        delay: Int64,
        slot_i: Int32,
        stream: fx.Stream = fx.Stream(None),
    ):  # noqa: B008
        _ = keyed
        ag_gemm_kernel(xs, w, out, cnt, peers, rank, epoch, delay, slot_i).launch(
            grid=(BLOCKS,), block=(THREADS,), stream=stream
        )

    _AG_BUILDS[key] = launch
    return launch


_AG_BUILDS: dict = {}
