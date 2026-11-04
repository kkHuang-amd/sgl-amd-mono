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
from sgl_amd_mono.common.ops import (
    CM_DEV,
    CM_SYS,
    bf2_f32,
    bf16_pair,
    hw_rsq,
    ld_i32,
    load_ptr64,
    rsrc,
    wave_sum,
)
from sgl_amd_mono.common.plan import (
    BLOCKS,
    THREADS,
    first_task,
    kernel_sources,
    key_tuple,
)
from sgl_amd_mono.common.sync import sreg
from sgl_amd_mono.kimi_k31.gfx950 import kda_defs as k1
from sgl_amd_mono.kimi_k31.gfx950.stages import coll
from sgl_amd_mono.kimi_k31.gfx950.stages import kda_wide as kdaw
from sgl_amd_mono.kimi_k31.gfx950.stages.attnres import attnres_row, red_words, scalars
from sgl_amd_mono.kimi_k31.gfx950.stages.gemm_coll import _arrive_all, dev_emit
from sgl_amd_mono.kimi_k31.gfx950.stages.tile import atomic_lin, bf16_emit, f32_emit
from sgl_amd_mono.kimi_k31.gfx950.stages.tile_mma import DEPTH, RW, mma_tile
from sgl_amd_mono.kimi_k31.plan import Plan

SOURCES = kernel_sources(__name__)

TP = coll.TP
H = 7168
LAT = 3584
SL, LSL = H // TP, LAT // TP


def fold_up(w_up, w_n, rank):
    """This rank's W_up rows [896, 3584] with the latent norm weight folded into
    its columns (host side, once at load)."""
    rows = w_up[rank * SL : (rank + 1) * SL]
    return (rows.float() * w_n.float()[None, :]).to(rows.dtype).contiguous()


# the KDA stage's arguments, an int64 table (``kda_table``): pointers, then strides
KDA_PTRS = (
    "w_fb",
    "conv_w",
    "conv_st",
    "a_log",
    "dt_bias",
    "on_w",
    "rstate",
    "st_idx",
    "num_acc",
    "ring_k",
    "ring_v",
    "ring_g",
    "ring_b",
    "core_out",
    "conv_win",
    "win_row",
)
KDA_INTS = (
    "cs_seq",
    "cs_dim",
    "cs_tok",
    "rs_seq",
    "st_stride",
    "rk_seq",
    "rv_seq",
    "rg_seq",
    "rb_seq",
    "ring_len",
    "cw_seq",
    "cw_tok",
    "cw_q",
)


def kda_table(
    device,
    *,
    w_fb,
    conv_w,
    conv_state,
    a_log,
    dt_bias,
    on_w,
    rstate,
    st_idx,
    num_acc,
    core_out,
    conv_win,
    win_row,
    ring=None,
):
    """The KDA stage's arguments (Kimi-K3's K1 layouts and strides), packed for the
    launch's ``kda_tab``.

    ``conv_state`` [slot, channel, token] by its strides (SGLang's conv cache
    [slot, CONV_W - 1, C] is its ``.mT``), read only. ``conv_win`` [row, L,
    CONV_W - 1, C] (SGLang's ``intermediate_conv_window`` of the layer; channels
    contiguous) gets request r's verify windows at row ``win_row[r]`` (int32,
    < 0: none)."""
    import torch

    assert conv_win.dim() == 4 and conv_win.stride(3) == 1, (
        tuple(conv_win.shape),
        conv_win.stride(),
    )
    assert win_row.dtype == torch.int32 and win_row.is_contiguous()
    rk, rv, rg, rb = ring if ring is not None else (None,) * 4
    ptrs = dict(
        w_fb=w_fb,
        conv_w=conv_w,
        conv_st=conv_state,
        a_log=a_log,
        dt_bias=dt_bias,
        on_w=on_w,
        rstate=rstate,
        st_idx=st_idx,
        num_acc=num_acc,
        ring_k=rk,
        ring_v=rv,
        ring_g=rg,
        ring_b=rb,
        core_out=core_out,
        conv_win=conv_win,
        win_row=win_row,
    )
    ints = dict(
        cs_seq=conv_state.stride(0),
        cs_dim=conv_state.stride(1),
        cs_tok=conv_state.stride(2),
        rs_seq=rstate.stride(0),
        st_stride=st_idx.stride(0),
        rk_seq=rk.stride(0) if rk is not None else 0,
        rv_seq=rv.stride(0) if rv is not None else 0,
        rg_seq=rg.stride(0) if rg is not None else 0,
        rb_seq=rb.stride(0) if rb is not None else 0,
        ring_len=rk.shape[2] if rk is not None else 0,
        cw_seq=conv_win.stride(0),
        cw_tok=conv_win.stride(1),
        cw_q=conv_win.stride(2),
    )
    vals = [0 if ptrs[k] is None else ptrs[k].data_ptr() for k in KDA_PTRS] + [
        ints[k] for k in KDA_INTS
    ]
    return torch.tensor(vals, dtype=torch.int64, device=device)


class _KdaArgs:
    """The table's entries, each loaded where it is read (none held across the stage)."""

    def __init__(self, tab):
        self.tab = tab

    def __getitem__(self, k):
        if k in KDA_PTRS:
            return load_ptr64(self.tab, KDA_PTRS.index(k))
        return ld_i32(self.tab, 2 * (len(KDA_PTRS) + KDA_INTS.index(k)))


def _kda_args(tab):
    return _KdaArgs(tab)


def plan(m, gu, gi, nin, nvb, kda=None, wide=True):
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
    p.stage(
        "inproj",
        tasks_i,
        waits_on=("seam", "inproj") if kda else ("seam",),
        all_ranks=True,
    )
    if kda:
        p.region("proj", m * k1.NPROJ * 8, peer=False)
        p.region("qk", m * k1.NH * 2 * k1.HD * 8, peer=False)
        p.region("onrm", m * k1.NH * k1.NVCH * 8, peer=False)
        p.stage(
            "kda",
            (m // kda) * k1.NH * (1 if wide else k1.NVCH),
            waits_on=("inproj",) if wide else ("inproj", "kda"),
        )
    return p.check()


@flyc.jit
def _finish_proj(c, out, m, nin, r0, rb, n0, bn, ks, proj):
    """Rows r0 .. r0 + rb of a finished in_proj tile -> the KDA stage's tagged f32
    pairs (two a store); zero the tile."""
    tid = c["tid"]
    q2 = bn // 2
    for it in range(tid, rb * q2, THREADS):
        row = r0 + it // q2
        col = n0 + 2 * (it % q2)
        if (row < m) & (col < nin):
            w = ld_i32(out, (row * nin + col) // 2, CM_DEV)
            lo, hi = bf2_f32(w)
            c["put_words"](
                proj, row * k1.NPROJ + col, [lo.bitcast(fx.Int32), hi.bitcast(fx.Int32)]
            )
            if const_expr(ks > 1):
                bo.buffer_store(
                    fx.Int32(0),
                    rsrc(out),
                    (row * nin + col) // 2,
                    cache_modifier=CM_DEV,
                )


@flyc.jit
def _push(c, row, j, m, lat_p, sh_p):
    """This rank's latent and shared columns of owner j for ``row`` -> the owner."""
    tid = c["tid"]
    rank = c["rank"]
    if tid < LSL // 8:
        v = fx.Vector(
            bo.buffer_load(
                rsrc(lat_p),
                (row * LAT + j * LSL + 8 * tid) // 2,
                vec_width=4,
                dtype=T.i32,
            )
        )
        bo.buffer_store(
            v,
            rsrc(c["lrs_of"](j).value),
            ((rank * m + row) * LSL + 8 * tid) // 2,
            cache_modifier=CM_SYS,
        )
    u = tid - 64
    if (u >= 0) & (u < SL // 8):
        v = fx.Vector(
            bo.buffer_load(
                rsrc(sh_p), (row * H + j * SL + 8 * u) // 2, vec_width=4, dtype=T.i32
            )
        )
        bo.buffer_store(
            v,
            rsrc(c["srs_of"](j).value),
            ((rank * m + row) * SL + 8 * u) // 2,
            cache_modifier=CM_SYS,
        )
    rocdl.s_waitcnt(vmcnt=0)
    gpu.barrier()
    if tid == 0:
        c["put"](c["rsf_of"](j), rank * m + row, fx.Int32(1))


def _sum8(c, region, base, stride):
    got = [
        fx.Vector(
            bo.buffer_load(
                rsrc(c[region].value),
                (src * stride + base) // 2,
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
    return acc


@flyc.jit
def _lred(c, row, m, shs, wn=None):
    """The owner's row: latent sum -> every rank's gather region + the slice's sum
    of squares (tagged) to every rank; shared sum (f32) -> local scratch. ``wn``: the
    latent norm weight (f32 [LAT]) the gathered latent is scaled by (the up projection
    then takes SGLang's weight as is); the sum of squares stays the unscaled row's."""
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
        words = [
            bf16_pair(acc[2 * k], acc[2 * k + 1]).bitcast(fx.Int32) for k in range(4)
        ]
        vb = [x for w in words for x in bf2_f32(w)]
        part = vb[0] * vb[0]
        for x in vb[1:]:
            part = part + x * x
        ss = wave_sum(live.select(part, fx.Float32(0.0)))
        if const_expr(wn is not None):
            wr = rsrc(wn)
            wv = [
                fx.Vector(
                    bo.buffer_load(
                        wr, rank * LSL + 8 * q + 4 * h, vec_width=4, dtype=T.f32
                    )
                )
                for h in range(2)
            ]
            wl = [wv[e // 4][e % 4] for e in range(8)]
            words = [
                bf16_pair(vb[2 * k] * wl[2 * k], vb[2 * k + 1] * wl[2 * k + 1]).bitcast(
                    fx.Int32
                )
                for k in range(4)
            ]
        vec = fx.Vector.from_elements(words, fx.Int32)
        if live:
            for p in range_constexpr(TP):
                bo.buffer_store(
                    vec,
                    rsrc(c["lag_of"](p).value),
                    (row * LAT + rank * LSL + 8 * q) // 2,
                    cache_modifier=CM_SYS,
                )
        if lane == 0:
            for p in range_constexpr(TP):
                c["put"](c["lss_of"](p), rank * m + row, ss)
    u = tid - 64
    if (u >= 0) & (u < SL // 8):
        acc = _sum8(c, "srs_own", row * SL + 8 * u, m * SL)
        for k in range_constexpr(2):
            bo.buffer_store(
                fx.Vector.from_elements(acc[4 * k : 4 * k + 4], fx.Float32),
                rsrc(shs),
                row * SL + 8 * u + 4 * k,
                cache_modifier=CM_DEV,
            )
    rocdl.s_waitcnt(vmcnt=0)
    gpu.barrier()
    coll.put_each_peer(c, [("lagf", rank * m + row, fx.Int32(1))])


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
def _finish(
    c,
    acc,
    shs,
    prefix,
    prefix_out,
    m,
    r0,
    rb,
    n0,
    bn,
    ks,
    pdone,
    piece,
    ldp=SL,
    pbytes=None,
):
    """Rows r0 .. r0 + rb of a finished up tile: y = up + shared, prefix' = prefix +
    y (bf16 roundings as SGLang's two adds), zero the up tile, mark the piece.
    ``ldp``: the prefix's row stride (H: this rank's columns of a full-width row);
    ``pbytes``: bytes readable from ``prefix`` (rows past it load 0)."""
    tid = c["tid"]
    q4 = bn // 4
    for it in range(tid, rb * q4, THREADS):
        row = r0 + it // q4
        col = n0 + 4 * (it % q4)
        if row < m:
            uv = fx.Vector(
                bo.buffer_load(
                    rsrc(acc),
                    (row * SL + col) // 2,
                    vec_width=2,
                    dtype=T.i32,
                    cache_modifier=CM_DEV,
                )
            )
            sv = fx.Vector(
                bo.buffer_load(
                    rsrc(shs),
                    row * SL + col,
                    vec_width=4,
                    dtype=T.f32,
                    cache_modifier=CM_DEV,
                )
            )
            pv = fx.Vector(
                bo.buffer_load(
                    rsrc(prefix, pbytes),
                    (row * ldp + col) // 2,
                    vec_width=2,
                    dtype=T.i32,
                )
            )
            up = [x for w in (uv[0], uv[1]) for x in bf2_f32(w)]
            pr = [x for w in (pv[0], pv[1]) for x in bf2_f32(w)]
            y = [
                fx.Float32(fx.Float32(up[e] + sv[e]).to(fx.BFloat16)) for e in range(4)
            ]
            o = [y[e] + pr[e] for e in range(4)]
            words = [
                bf16_pair(o[0], o[1]).bitcast(fx.Int32),
                bf16_pair(o[2], o[3]).bitcast(fx.Int32),
            ]
            bo.buffer_store(
                fx.Vector.from_elements(words, fx.Int32),
                rsrc(prefix_out),
                (row * SL + col) // 2,
                cache_modifier=CM_DEV,
            )
            if const_expr(ks > 1):
                bo.buffer_store(
                    fx.Vector.filled(2, 0, fx.Int32),
                    rsrc(acc),
                    (row * SL + col) // 2,
                    cache_modifier=CM_DEV,
                )
    rocdl.s_waitcnt(vmcnt=0)
    gpu.barrier()
    # a partial tile's pieces past row m: no piece of the region (the index would land past it)
    if (tid == 0) & (r0 < m):
        c["put"](pdone, piece, fx.Int32(1))


@flyc.jit
def _seam(
    c,
    row,
    m,
    rb,
    ct,
    nvb,
    nbk,
    prefix_out,
    bank,
    cw,
    w_in,
    eps,
    pdone,
    red,
    write,
    capture=False,
    wait=True,
):
    tid = c["tid"]
    if const_expr(wait):
        if tid < ct:
            c["poll"]([(pdone, (row // rb) * ct + tid, 1)])
        gpu.barrier()
    rank = c["rank"]
    ags = [c["ag_of"](p) for p in range(TP)]

    def emit(q, live, vec):
        if live:
            for p in range_constexpr(TP):
                bo.buffer_store(
                    vec,
                    rsrc(ags[p].value),
                    (row * H + rank * SL + 8 * q) // 2,
                    cache_modifier=CM_SYS,
                )

    emit_mixed = None
    if const_expr(capture):
        caps = [c["cap_of"](p) for p in range(TP)]

        def emit_mixed(q, live, vec):
            if live:
                for p in range_constexpr(TP):
                    bo.buffer_store(
                        vec,
                        rsrc(caps[p].value),
                        (row * H + rank * SL + 8 * q) // 2,
                        cache_modifier=CM_SYS,
                    )

    attnres_row(
        c,
        row,
        m,
        SL,
        nvb,
        nbk,
        prefix_out,
        bank,
        cw,
        w_in,
        eps,
        eps,
        red,
        emit,
        write,
        0,
        emit_mixed,
    )
    rocdl.s_waitcnt(vmcnt=0)
    gpu.barrier()
    coll.put_each_peer(
        c,
        [("agf", rank * m + row, fx.Int32(1))]
        + ([("capf", rank * m + row, fx.Int32(1))] if capture else []),
    )


@flyc.jit
def capture_row(c, row, m, cap_out, cap_ld):
    """The row's gathered pre-norm mixture -> the DSpark capture slot (row stride
    ``cap_ld`` elements, e.g. a column block of the packed aux buffer)."""
    tid = c["tid"]
    if tid < TP:
        c["poll"]([(c["capf_own"], tid * m + row, 1)])
    gpu.barrier()
    for u in range(tid, H // 8, THREADS):
        v = fx.Vector(
            bo.buffer_load(
                rsrc(c["cap_own"].value),
                (row * H + 8 * u) // 2,
                vec_width=4,
                dtype=T.i32,
                cache_modifier=CM_SYS,
            )
        )
        bo.buffer_store(v, rsrc(cap_out), (row * cap_ld + 8 * u) // 2)


_BUILDS: dict = {}


def pre_attn(
    m,
    gu,
    gi,
    nin,
    nvb=0,
    nbk=8,
    write=False,
    mode="linbf",
    launches=1,
    eps_n=1e-5,
    eps=1e-5,
    kda=None,
    state_len=0,
    ring=False,
    onorm_eps=1e-5,
    lower_bound=-5.0,
    kda_wide=True,
    kpt=16,
):
    """Build the launch; returns (launch, plan). Arguments: lat_p [m, LAT] / sh_p [m, H]
    (this rank's latent / shared partials), w_up [SL, LAT] (``fold_up``), prefix /
    prefix_out [m, SL], bank [m, nbk, SL], cw f32 [H], w_in [H] (input norm), w_ip
    [nin, H] (in_proj), out (bf16 [m, nin], f32 [ks, m, nin] for "part"), acc bf16
    [m, SL] / cnt [up tiles] (zeroed once), scratch (``plan.scratch_bytes``), peers,
    rank, epoch, slot. The up GEMM merges linbf (or writes at ks 1).

    ``kda`` = L (tokens a request): in_proj's tiles (``out`` is then their bf16 merge
    buffer [m, nin], zeroed once; ``cnt`` holds the in_proj tiles' counters after
    the up tiles') are handed to the KDA stage (conv, gates, recurrence, gated norm on
    the ReplaySSM ring, ``ring``; a task a (request, head), ``stages/kda_wide.py``),
    whose core_out [m, 1536] is the launch's output; ``kda_tab`` = ``kda_table(...)``."""
    keyed = key_tuple(locals(), SOURCES)
    key = (
        m,
        gu.key(),
        gi.key(),
        nin,
        nvb,
        nbk,
        write,
        mode,
        launches,
        eps_n,
        eps,
        kda,
        state_len,
        ring,
        onorm_eps,
        lower_bound,
        kda_wide,
        kpt,
    )
    assert kda is None or (ring and kda_wide), (
        "the KDA stage is the wide task (stages/kda_wide.py), on the ReplaySSM ring"
    )
    assert kda is None or (
        nin == k1.NPROJ
        and m % kda == 0
        and (mode == "linbf" or gi.ks == 1)
        and gi.bm % gi.ks == 0
        and BLOCKS % gi.ks == 0
    )
    if key in _BUILDS:
        return _BUILDS[key]
    pl = plan(m, gu, gi, nin, nvb, kda, kda_wide)
    offs, half = pl.peer_offsets
    soffs, _ = pl.scratch_offsets
    fo = pl.flags_offset()
    rb = gu.bm // gu.ks
    ct = SL // gu.bn
    ktu, kti = LAT // gu.ks, H // gi.ks
    assert (
        ktu % 64 == 0
        and kti % 64 == 0
        and (ktu % LSL == 0 or LSL % ktu == 0)
        and (kti % SL == 0 or SL % kti == 0)
    )
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

    kl_n = kda or 1
    kda_wide_lds = kdaw.lds_floats(kl_n)

    @fx.struct
    class KdaLds:
        fa: fx.Array[fx.Float32, kl_n * k1.HD, 16]
        qkl: fx.Array[fx.Float32, kl_n * 2 * k1.HD, 16]
        decay: fx.Array[fx.Float32, kl_n * k1.HD, 16]
        vloc: fx.Array[fx.Float32, kl_n * k1.VCH, 16]
        ol: fx.Array[fx.Float32, kl_n * k1.VCH, 16]
        beta: fx.Array[fx.Float32, 16, 16]

    @fx.struct
    class KdwLds:
        v: fx.Array[fx.Float32, kda_wide_lds, 16]

    @fx.union
    class Smem:
        ab: AbLds
        c: CLds
        kd: KdaLds
        kw: KdwLds

    @fx.struct
    class Small:
        slot: fx.Array[fx.Int32, 4, 16]
        rms: fx.Array[fx.Float32, gu.bm, 16]
        red: fx.Array[fx.Float32, red_words(nvb), 16]

    name = (
        f"k31_pre_attn_m{m}_u{'x'.join(map(str, gu.key()))}_i{'x'.join(map(str, gi.key()))}_n{nin}_v{nvb}of{nbk}"
        f"w{int(write)}_{mode}_l{launches}_k{kda}s{state_len}r{int(ring)}w{int(kda_wide)}p{kpt}"
    )

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def pre_attn_kernel(
        lat_p: Int64,
        sh_p: Int64,
        w_up: Int64,
        prefix: Int64,
        prefix_out: Int64,
        bank: Int64,
        cw: Int64,
        w_in: Int64,
        w_ip: Int64,
        out: Int64,
        acc: Int64,
        cnt: Int64,
        scratch: Int64,
        peers: Int64,
        rank: Int32,
        epoch: Int64,
        slot_i: Int32,
        kda_tab: Int64,
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
            small = alloc.allocate(Small).peek()
            slot, rmsl, red = small.slot.ptr, small.rms.ptr, small.red.ptr
            lds, cl = st.ab.peek(), st.c.peek().c.ptr
            c = coll.region_context(
                peers, rank, epoch, slot_i, offs, half, tid, bid, launches
            )
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
                        base(
                            row,
                            col,
                            fx.Vector.from_elements(
                                [v[e] * s for e in range(4)], fx.Float32
                            ),
                            live,
                        )
                else:
                    e_, add = atomic_lin(acc, SL, n0, m, SL, True)

                    def scaled(row, col, vals, add=add, m0=m0):
                        s = fx.ptr_load(rmsl + fx.min(row - m0, gu.bm - 1))
                        add(row, col, [v * s for v in vals])

                    lin = (e_, scaled)
                mma_tile(
                    tid,
                    lagx,
                    LAT,
                    m,
                    m0,
                    w_up,
                    LAT,
                    SL,
                    n0,
                    sk * ktu,
                    ktu,
                    gu.bm,
                    gu.bn,
                    gu.wm,
                    gu.wn,
                    lds.xs.ptr,
                    lds.ws.ptr,
                    emit,
                    DEPTH,
                    lin,
                    cl,
                )
                _arrive_all(tid, cnt, tile, gu.ks, slot)
                r0 = m0 + sk * rb
                _finish(
                    c,
                    acc,
                    shs,
                    prefix,
                    prefix_out,
                    m,
                    r0,
                    rb,
                    n0,
                    gu.bn,
                    gu.ks,
                    pdone,
                    (r0 // rb) * ct + n0 // gu.bn,
                )
            # seam
            for row in range(bid, m, BLOCKS):
                _seam(
                    c,
                    row,
                    m,
                    rb,
                    ct,
                    nvb,
                    nbk,
                    prefix_out,
                    bank,
                    cw,
                    w_in,
                    eps,
                    pdone,
                    red,
                    write,
                )
            # inproj
            proj = sreg(scratch, soffs["proj"], "proj") if kda is not None else None
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
                if const_expr(kda is not None and gi.ks == 1):
                    emit = dev_emit(out, nin, n0, m, nin, False)
                mma_tile(
                    tid,
                    x,
                    H,
                    m,
                    m0,
                    w_ip,
                    H,
                    nin,
                    n0,
                    sk * kti,
                    kti,
                    gi.bm,
                    gi.bn,
                    gi.wm,
                    gi.wn,
                    lds.xs.ptr,
                    lds.ws.ptr,
                    emit,
                    DEPTH,
                    lin,
                    cl,
                )
                if const_expr(kda is not None):
                    _arrive_all(tid, cnt + fx.Int64(4 * tmu * tnu), tile, gi.ks, slot)
                    rbi = gi.bm // gi.ks
                    _finish_proj(
                        c, out, m, nin, m0 + sk * rbi, rbi, n0, gi.bn, gi.ks, proj
                    )
            if const_expr(kda is not None):
                kd = st.kd.peek()
                c.update(
                    {
                        "S": m,
                        "L": kda,
                        "SL": state_len,
                        "lane": tid % 64,
                        "wave": tid // 64,
                        "kda0": 0,
                        "lb": lower_bound,
                        "onorm_eps": onorm_eps,
                        "ring": ring,
                        "xp": 0,
                        "on": False,
                        "tls": None,
                        "red": red,
                        "fa": kd.fa.ptr,
                        "qkl": kd.qkl.ptr,
                        "decay": kd.decay.ptr,
                        "vloc": kd.vloc.ptr,
                        "ol": kd.ol.ptr,
                        "beta": kd.beta.ptr,
                        "proj": proj,
                        "args": _kda_args(kda_tab),
                        "qk": sreg(scratch, soffs["qk"], "qk"),
                        "onrm": sreg(scratch, soffs["onrm"], "onrm"),
                    }
                )
                kdaw.kda_heads(c, st.kw.peek().v.ptr, 0, kpt)

    @flyc.jit
    def launch(
        lat_p: Int64,
        sh_p: Int64,
        w_up: Int64,
        prefix: Int64,
        prefix_out: Int64,
        bank: Int64,
        cw: Int64,
        w_in: Int64,
        w_ip: Int64,
        out: Int64,
        acc: Int64,
        cnt: Int64,
        scratch: Int64,
        peers: Int64,
        rank: Int32,
        epoch: Int64,
        slot_i: Int32,
        kda_tab: Int64,
        stream: fx.Stream = fx.Stream(None),
    ):  # noqa: B008
        _ = keyed
        pre_attn_kernel(
            lat_p,
            sh_p,
            w_up,
            prefix,
            prefix_out,
            bank,
            cw,
            w_in,
            w_ip,
            out,
            acc,
            cnt,
            scratch,
            peers,
            rank,
            epoch,
            slot_i,
            kda_tab,
        ).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)

    _BUILDS[key] = (launch, pl)
    return launch, pl
