# SPDX-License-Identifier: Apache-2.0
"""The MoE front's tail inside the layer launch: what the routed experts and the
next layer's MoE tail need from the front GEMM.

A rank's front GEMM rows (``chain``'s front, merged in f32 here) are its 112
router experts | 448 latent-down rows | 768 shared gate + 768 shared up. Its
finished tiles go three ways:

router  f32 logits -> every rank's gather region [m, 896] (expert e of rank r at
        112 r + e)
latent  bf16 -> every rank's gather region [m, 3584]
shared  bf16 gate / up -> local scratch

then, a row a task: ``route_row`` copies the gathered latent into the experts'
input and picks the top 16 (SGLang's noaux_tc: rank by sigmoid(logit) + bias,
weights sigmoid(logit) renormalized); ``situ_row`` applies SiTU to the shared
gate / up (bf16, as the shared MLP rounds), and the shared down GEMM follows.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T

from sgl_amd_mono.common.ops import (
    CM_DEV,
    CM_SYS,
    bf2_f32,
    bf16_pair,
    hw_exp2,
    hw_rcp,
    read_lane_i32,
    rsrc,
    wave_umax,
    write_lane_i32,
)
from sgl_amd_mono.common.plan import THREADS
from sgl_amd_mono.kimi_k31.gfx950.kda_defs import ld_f32
from sgl_amd_mono.kimi_k31.gfx950.stages import coll

TP = 8
E, E_R = 896, 112  # experts, a rank's
LAT, L_R = 3584, 448
SI = 768  # shared intermediate a rank (2 shared experts x 3072 / TP)
R0, L0, S0 = 0, E_R, E_R + L_R  # column starts in a rank's front rows
NF = E_R + L_R + 2 * SI  # 2096
TOPK = 16
LOG2E = 1.4426950408889634


def gathered_tiles(bn):
    """Front tiles holding router / latent columns (0 .. S0)."""
    return (S0 + bn - 1) // bn


@flyc.jit
def finish_front(c, fbuf, sgu, m, r0, rb, n0, bn, band, t, nbf, smark, tnf):
    """Rows r0 .. r0 + rb of finished front tile t (columns n0 ..): router / latent
    columns to every rank, shared columns to local scratch; zero the tile; flag
    (this rank, band, t) on every rank and / or mark (band, t) locally."""
    tid = c["tid"]
    rank = c["rank"]
    q4 = bn // 4
    rgs = [c["rgat_of"](p) for p in range(TP)]
    lgs = [c["lgat_of"](p) for p in range(TP)]
    for it in range(tid, rb * q4, THREADS):
        row = r0 + it // q4
        col = n0 + 4 * (it % q4)
        if (row < m) & (col < NF):
            v = fx.Vector(
                bo.buffer_load(
                    rsrc(fbuf),
                    row * NF + col,
                    vec_width=4,
                    dtype=T.f32,
                    cache_modifier=CM_DEV,
                )
            )
            bo.buffer_store(
                fx.Vector.filled(4, 0.0, fx.Float32),
                rsrc(fbuf),
                row * NF + col,
                cache_modifier=CM_DEV,
            )
            words = fx.Vector.from_elements(
                [
                    bf16_pair(v[0], v[1]).bitcast(fx.Int32),
                    bf16_pair(v[2], v[3]).bitcast(fx.Int32),
                ],
                fx.Int32,
            )
            if col < L0:
                for p in range_constexpr(TP):
                    bo.buffer_store(
                        v,
                        rsrc(rgs[p].value),
                        row * E + rank * E_R + col,
                        cache_modifier=CM_SYS,
                    )
            if (col >= L0) & (col < S0):
                for p in range_constexpr(TP):
                    bo.buffer_store(
                        words,
                        rsrc(lgs[p].value),
                        (row * LAT + rank * L_R + col - L0) // 2,
                        cache_modifier=CM_SYS,
                    )
            if col >= S0:
                bo.buffer_store(
                    words,
                    rsrc(sgu),
                    (row * 2 * SI + col - S0) // 2,
                    cache_modifier=CM_DEV,
                )
    rocdl.s_waitcnt(vmcnt=0)
    gpu.barrier()
    # a partial tile's pieces past row m hold no rows: their band (>= nbf) would flag another rank's
    if (n0 < S0) & (r0 < m):
        coll.put_each_peer(
            c, [("fgf", (rank * nbf + band) * gathered_tiles(bn) + t, fx.Int32(1))]
        )
    if (n0 + bn > S0) & (tid == 0) & (r0 < m):
        c["put"](smark, band * tnf + t, fx.Int32(1))


def _ld_sys(r, i):
    """One f32 of a peer region other ranks wrote (system scope: no stale cached line)."""
    return fx.Float32(
        bo.buffer_load(r, i, vec_width=1, dtype=T.f32, cache_modifier=CM_SYS)
    )


def _sigmoid(x):
    return hw_rcp(1.0 + hw_exp2(x * (-LOG2E)))


def _batcher(n):
    """Batcher's odd-even merge sort network on n (a power of two) slots: (a, b) pairs."""
    pairs = []
    p = 1
    while p < n:
        k = p
        while k >= 1:
            for j in range(k % p, n - k, 2 * k):
                for i in range(min(k, n - j - k)):
                    if (i + j) // (2 * p) == (i + j + k) // (2 * p):
                        pairs.append((i + j, i + j + k))
            k //= 2
        p *= 2
    return pairs


SLOTS = 16  # a lane's 14 candidates (896 / 64) padded to the network's width
NET = _batcher(SLOTS)


def _okey(v):
    """An order-preserving uint32 key of an f32 (larger float, larger key)."""
    b = v.bitcast(fx.Int32)
    return fx.Uint32((b < fx.Int32(0)).select(~b, b ^ fx.Int32(-2147483648)))


@flyc.jit
def route_row(
    c, row, m, rbf, nbf, bn, bias, lat_out, topk_ids, topk_w, logits_out=None
):
    """The row's gathered latent -> the experts' input; top-16 of the gathered logits
    (and the logits themselves -> ``logits_out`` f32 [m, 896], for SGLang's own top-k)."""
    tid = c["tid"]
    lane, wave = tid % 64, tid // 64
    nt = gathered_tiles(bn)
    band = row // rbf
    if tid < TP * nt:
        c["poll"]([(c["fgf_own"], ((tid // nt) * nbf + band) * nt + tid % nt, 1)])
    gpu.barrier()
    lg = rsrc(c["lgat_own"].value)
    rg = rsrc(c["rgat_own"].value)
    for u in range(tid, LAT // 8, THREADS):
        v = fx.Vector(
            bo.buffer_load(
                lg,
                (row * LAT + 8 * u) // 2,
                vec_width=4,
                dtype=T.i32,
                cache_modifier=CM_SYS,
            )
        )
        bo.buffer_store(v, rsrc(lat_out), (row * LAT + 8 * u) // 2)
    if const_expr(logits_out is not None):
        for e in range(tid, E, THREADS):
            bo.buffer_store(_ld_sys(rg, row * E + e), rsrc(logits_out), row * E + e)
    if wave == 0:
        # max-extraction (as the K3 K2 launch, mono/kimi-k31 05dfa48): each lane sorts its 14 candidates (expert
        # lane + 64 i; key descending, the lower expert first on a tie), then 16 rounds of a wave max pop the heads;
        # pick k lands in lane k, ranked. Same picks, order and weights as one rescan a pick.
        per = E // 64
        ks, ids, sg = [], [], []
        for i in range_constexpr(per):
            e = lane + 64 * i
            s = _sigmoid(_ld_sys(rg, row * E + e))
            ks.append(_okey(s + ld_f32(bias, e)))
            ids.append(fx.Int32(e))
            sg.append(s.bitcast(fx.Int32))
        ks += [fx.Uint32(0)] * (SLOTS - per)
        ids += [fx.Int32(1023)] * (SLOTS - per)
        sg += [fx.Int32(0)] * (SLOTS - per)
        for x, y in NET:
            first = (ks[x] > ks[y]) | ((ks[x] == ks[y]) & (ids[x] < ids[y]))
            kx, ky, ix, iy, sx, sy = ks[x], ks[y], ids[x], ids[y], sg[x], sg[y]
            ks[x], ks[y] = first.select(kx, ky), first.select(ky, kx)
            ids[x], ids[y] = first.select(ix, iy), first.select(iy, ix)
            sg[x], sg[y] = first.select(sx, sy), first.select(sy, sx)
        ks = [fx.Int32(k) for k in ks] + [fx.Int32(0)]
        ids = ids + [fx.Int32(1023)]
        sg = sg + [fx.Int32(0)]
        tot, sel_i, sel_s = fx.Float32(0.0), fx.Int32(0), fx.Int32(0)
        for k in range_constexpr(TOPK):
            mx = wave_umax(ks[0])
            expert = fx.Int32(1023) - wave_umax(
                (ks[0] == mx).select(fx.Int32(1023) - ids[0], fx.Int32(0))
            )
            sig = read_lane_i32(
                sg[0], expert % 64
            )  # the winner's sigmoid, before its lane pops its head
            hit = (ks[0] == mx) & (ids[0] == expert)
            ks = [hit.select(ks[i + 1], ks[i]) for i in range(SLOTS)] + [ks[-1]]
            ids = [hit.select(ids[i + 1], ids[i]) for i in range(SLOTS)] + [ids[-1]]
            sg = [hit.select(sg[i + 1], sg[i]) for i in range(SLOTS)] + [sg[-1]]
            tot = tot + sig.bitcast(fx.Float32)
            sel_i = write_lane_i32(expert, k, sel_i)
            sel_s = write_lane_i32(sig, k, sel_s)
        inv = hw_rcp(tot)
        if lane < TOPK:
            bo.buffer_store(sel_i, rsrc(topk_ids), row * TOPK + lane)
            bo.buffer_store(
                sel_s.bitcast(fx.Float32) * inv, rsrc(topk_w), row * TOPK + lane
            )
    gpu.barrier()


def _situ(g, u):
    # SiTU(g, u) = 2 tanh(g / 4) (1 + tanh(g / 2)) x 25 tanh(u / 25)  (beta 4, linear beta 25)
    def tanh(x):
        e2 = hw_exp2(x * (2.0 * LOG2E))
        return (e2 - 1.0) * hw_rcp(e2 + 1.0)

    return (2.0 * tanh(g * 0.25)) * (1.0 + tanh(g * 0.5)) * (25.0 * tanh(u * 0.04))


@flyc.jit
def situ_row(c, row, m, rbf, tnf, bn, sgu, smark, hbuf, hmark):
    """SiTU of the row's shared gate / up (bf16 in, bf16 out) -> hbuf; mark the row."""
    tid = c["tid"]
    t0 = S0 // bn
    band = row // rbf
    if tid < tnf - t0:
        c["poll"]([(smark, band * tnf + t0 + tid, 1)])
    gpu.barrier()
    for u in range(tid, SI // 4, THREADS):
        gv = fx.Vector(
            bo.buffer_load(
                rsrc(sgu),
                (row * 2 * SI + 4 * u) // 2,
                vec_width=2,
                dtype=T.i32,
                cache_modifier=CM_DEV,
            )
        )
        uv = fx.Vector(
            bo.buffer_load(
                rsrc(sgu),
                (row * 2 * SI + SI + 4 * u) // 2,
                vec_width=2,
                dtype=T.i32,
                cache_modifier=CM_DEV,
            )
        )
        gs = [x for w in (gv[0], gv[1]) for x in bf2_f32(w)]
        us = [x for w in (uv[0], uv[1]) for x in bf2_f32(w)]
        h = [_situ(gs[e], us[e]) for e in range(4)]
        words = fx.Vector.from_elements(
            [
                bf16_pair(h[0], h[1]).bitcast(fx.Int32),
                bf16_pair(h[2], h[3]).bitcast(fx.Int32),
            ],
            fx.Int32,
        )
        bo.buffer_store(
            words, rsrc(hbuf), (row * SI + 4 * u) // 2, cache_modifier=CM_DEV
        )
    rocdl.s_waitcnt(vmcnt=0)
    gpu.barrier()
    if tid == 0:
        c["put"](hmark, row, fx.Int32(1))
