# SPDX-License-Identifier: Apache-2.0
"""S3, the routed experts of Laguna's FFN launch on gfx950 (functional prototype):
512 MXFP4 experts (EP1, inter 512 per rank, gate rows then up rows), aiter's
(16, 16)-shuffled weights and ``shuffle_scale`` scales read in place; activations
quantized with aiter's hip per-1x32 rule (bit-exact with fused_moe's input quant).

One launch, BLOCKS CTAs x THREADS threads, stages in order:

    U    : every CTA builds the step's expert union in LDS from the top-k ids /
           weights (expert-id order): expert, token mask, route weight per token
    Q    : CTA t < tokens: the token's 512 groups -> XQ / XS (hip rule)
    UG   : CTA b, channels 2b, 2b + 1, for each union expert (dynamic loop): the
           routed tokens' gate / up dot products (fp4 decoded exactly, fp32 FMA,
           lane = 32-element K group, waves summed in order), silu(gate) * up in
           fp32 -> bf16 MID[u][token]
    RQ   : a thread a (expert, token, 32-group) of MID -> MQ / MS (hip rule)
    DOWN : CTA b, rows [64 b, 64 b + 64), wave w eight; for each union expert the
           routed tokens' dot products, times the route weight, summed in fp32 over
           the experts (a CTA owns its rows: no atomics, deterministic); bf16 out
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import Int32, Int64, T

from sgl_amd_mono.common import guard
from sgl_amd_mono.common.mx import FP4, mfma_scaled, scale_index
from sgl_amd_mono.common.ops import (
    CM_DEV, ballot, bf2_f32, butterfly, hw_exp2, hw_rcp, lanes_below, popcount, rsrc, uniform, xshfl,
)
from sgl_amd_mono.common.plan import BLOCKS, THREADS, WAVES, kernel_sources
from sgl_amd_mono.common.sync import Mailbox, sreg
from sgl_amd_mono.laguna.gfx950 import mxq
from sgl_amd_mono.laguna.gfx950.mxq import decode_word, pow2, quant_group

HIDDEN = 16384
INTER = 512
EXPERTS = 512
TOPK = 16
MAX_TOKENS = 16
U_MAX = MAX_TOKENS * TOPK  # distinct experts a step at most
LAYER_SLOTS = 64
GROUPS = HIDDEN // 32
MGROUPS = INTER // 32
W13_BYTES = 2 * INTER * HIDDEN // 2  # one expert
W2_BYTES = HIDDEN * INTER // 2
LOG2E = 1.4426950408889634
assert EXPERTS == THREADS and GROUPS == THREADS and 2 * BLOCKS == INTER

# scratch layout (bytes)
XQ = 0
XS = XQ + MAX_TOKENS * HIDDEN // 2
MID = XS + MAX_TOKENS * GROUPS * 4
MQ = MID + U_MAX * MAX_TOKENS * INTER * 2
MS = MQ + U_MAX * MAX_TOKENS * INTER // 2
QF = MS + U_MAX * MAX_TOKENS * MGROUPS * 4
GF = QF + MAX_TOKENS * WAVES * 8
RF = GF + BLOCKS * 8
SCRATCH_BYTES = RF + BLOCKS * 8


def _release():
    rocdl.s_waitcnt(vmcnt=0)
    fx.memory_fence(syncscope=rocdl.SyncScope.Workgroup, ordering=fx.AtomicOrdering.Release)


def _byte(ptr, byte_index, cm=0):
    w = fx.Int32(bo.buffer_load(rsrc(ptr), byte_index // 4, vec_width=1, dtype=T.i32, cache_modifier=cm))
    return (w >> ((byte_index % 4) * 8)) & fx.Int32(0xFF)


def _silu_mul(g, u):
    return g * hw_rcp(fx.Float32(1.0) + hw_exp2(-g * fx.Float32(LOG2E))) * u


def _w13_index(r, gi):
    """i32 index of row r's K group gi (16 bytes) in one expert's (16, 16)-shuffled W13."""
    n0, ni = r // 16, r % 16
    return ((((n0 * (HIDDEN // 64) + gi // 2) * 2 + gi % 2) * 16 + ni) * 16) // 4


def _w2_index(row, lane):
    """i32 index of row ``row``'s K dword ``lane`` (4 bytes) in one expert's shuffled W2."""
    n0, ni = row // 16, row % 16
    kb = lane * 4
    return ((((n0 * (INTER // 64) + kb // 32) * 2 + (kb // 16) % 2) * 16 + ni) * 16 + kb % 16) // 4


def _w2_group_index(row, grp):
    """i32 index of row ``row``'s 32-element K group ``grp`` (16 bytes) in one expert's shuffled W2."""
    n0, ni = row // 16, row % 16
    return ((((n0 * (INTER // 64) + grp // 2) * 2 + grp % 2) * 16 + ni) * 16) // 4


STAGES = ("U", "Q", "UG", "RQ", "DOWN")


def build_routed(tokens: int, upto: str = "DOWN"):
    """``upto``: the last stage to run (a timing aid; anything short of DOWN writes no output)."""
    assert 1 <= tokens <= MAX_TOKENS
    last = STAGES.index(upto)
    keyed = (
        tokens,
        upto,
        kernel_sources(__name__),
        kernel_sources(mxq.__name__),
        kernel_sources(guard.__name__),
    )
    name = f"laguna_routed_t{tokens}" + ("" if upto == "DOWN" else f"_upto{upto}")

    @fx.struct
    class Smem:
        uexp: fx.Array[fx.Int32, U_MAX, 16]
        umask: fx.Array[fx.Int32, U_MAX, 16]
        uw: fx.Array[fx.Float32, U_MAX * MAX_TOKENS, 16]
        sids: fx.Array[fx.Int32, MAX_TOKENS * TOPK, 16]
        swts: fx.Array[fx.Float32, MAX_TOKENS * TOPK, 16]
        wc: fx.Array[fx.Int32, WAVES + 1, 16]
        part: fx.Array[fx.Float32, 2 * WAVES * 32 * MAX_TOKENS, 16]
        gu: fx.Array[fx.Float32, 4 * MAX_TOKENS, 16]

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def k(
        x: Int64, ids: Int64, wts: Int64, w13: Int64, s13: Int64, w2: Int64, s2: Int64, out: Int64,
        scratch: Int64, epoch: Int64, layer: Int32, skip: Int32,
    ):
        _ = keyed
        with guard.scope(guard.Guard(epoch)):
            tid = fx.thread_idx.x
            bid = fx.block_idx.x
            lane = tid % 64
            wave = tid // 64
            lds = fx.SharedAllocator().allocate(Smem).peek()
            uexp, umask, uw = lds.uexp.ptr, lds.umask.ptr, lds.uw.ptr
            sids, swts = lds.sids.ptr, lds.swts.ptr
            wc, part, gu = lds.wc.ptr, lds.part.ptr, lds.gu.ptr
            step = uniform(bo.buffer_load(rsrc(epoch), 0, vec_width=1, dtype=T.i32))
            box = {
                "mb": Mailbox(step * LAYER_SLOTS + layer + 1),
                "qf": sreg(scratch, QF, "qf"),
                "gf": sreg(scratch, GF, "gf"),
                "rf": sreg(scratch, RF, "rf"),
            }

            # ---- U: the union, thread e = expert e
            # the top-k table lands in LDS once (a thread an entry), then every thread scans it
            if tid < tokens * TOPK:
                fx.ptr_store(fx.Int32(bo.buffer_load(rsrc(ids), tid, vec_width=1, dtype=T.i32)), sids + tid)
                fx.ptr_store(fx.Float32(bo.buffer_load(rsrc(wts), tid, vec_width=1, dtype=T.f32)), swts + tid)
            gpu.barrier()
            e_me = fx.Int32(tid)
            m_me = fx.Int32(0)
            w_me = [fx.Float32(0.0) for _ in range(tokens)]
            for t in range_constexpr(tokens):
                for kk in range_constexpr(TOPK):
                    v = fx.ptr_load(sids + t * TOPK + kk)
                    hit = v == e_me
                    m_me = m_me | hit.select(fx.Int32(1 << t), fx.Int32(0))
                    w_me[t] = hit.select(fx.ptr_load(swts + t * TOPK + kk), w_me[t])
            used = m_me != fx.Int32(0)
            bal = ballot(used)
            if lane == 0:
                fx.ptr_store(popcount(bal), wc + wave)
            gpu.barrier()
            off = fx.Int32(0)
            for w in range_constexpr(WAVES):
                off = off + (fx.Int32(w) < fx.Int32(wave)).select(fx.ptr_load(wc + w), fx.Int32(0))
            if used:
                pos = off + lanes_below(bal, lane)
                fx.ptr_store(e_me, uexp + pos)
                fx.ptr_store(m_me, umask + pos)
                for t in range_constexpr(tokens):
                    fx.ptr_store(w_me[t], uw + pos * MAX_TOKENS + t)
            if tid == 0:
                tot = fx.Int32(0)
                for w in range_constexpr(WAVES):
                    tot = tot + fx.ptr_load(wc + w)
                fx.ptr_store(tot, wc + WAVES)
            gpu.barrier()
            nu = uniform(fx.ptr_load(wc + WAVES))

            if const_expr(last >= 1):
                # ---- Q: quantize x (hip rule), a thread a group
                if bid < tokens:
                    vals = []
                    for c in range_constexpr(4):
                        wv4 = fx.Vector(
                            bo.buffer_load(rsrc(x), (bid * HIDDEN + tid * 32) // 2 + 4 * c, vec_width=4, dtype=T.i32)
                        )
                        for i in range_constexpr(4):
                            a, b = bf2_f32(fx.Int32(wv4[i]))
                            vals += [a, b]
                    words, byte = quant_group(vals, "hip")
                    bo.buffer_store(fx.Vector.from_elements(words, fx.Int32), rsrc(scratch),
                                    (XQ + bid * (HIDDEN // 2) + tid * 16) // 4, cache_modifier=CM_DEV)
                    bo.buffer_store(byte, rsrc(scratch), (XS + (bid * GROUPS + tid) * 4) // 4, cache_modifier=CM_DEV)
                    _release()
                    if lane == 0:
                        box["mb"].put(box["qf"], bid * WAVES + wave, fx.Int32(1))

            if const_expr(last >= 2):
                # ---- UG (MFMA): tasks (union expert, 16-channel block); per wave K groups
                # [64 w, 64 w + 64) in 16 steps of 128 K. A = 16 gate (and 16 up) rows, a
                # lane's 16 B of the (16, 16) shuffle group (row lane % 16, K group lane // 16);
                # B = 16 token columns of x (token lane % 16, zero past the batch). One
                # scaled fp4 x fp4 MFMA per row group a step.
                if tid < tokens * WAVES:
                    box["mb"].poll([(box["qf"], tid, 1)])
                gpu.barrier()
                g_ni = lane % 16
                g_sub = lane // 16
                g_col_ok = g_ni < tokens
                g_col = fx.min(g_ni, fx.Int32(tokens - 1))
                g_ntask = ((skip & fx.Int32(1)) == fx.Int32(0)).select(nu * (INTER // 16), fx.Int32(0))  # timing aid
                g_zero4 = fx.Vector.filled(4, 0, fx.Int32)

                def g_task_info(g_tk):
                    g_tc = fx.min(g_tk, fx.max(g_ntask - 1, fx.Int32(0)))
                    g_uu = g_tc // (INTER // 16)
                    g_cc = g_tc % (INTER // 16)
                    g_ee = uniform(fx.ptr_load(uexp + g_uu))
                    return g_uu, g_cc, g_ee

                def g_quarter_of(g_ee, g_cc, g_q):
                    g_bb = w13 + fx.Int64(g_ee) * W13_BYTES
                    g_rr = [g_cc * 16 + g_ni, INTER + g_cc * 16 + g_ni]
                    g_l = []
                    for g_j in range_constexpr(4):
                        g_gi = wave * 64 + (g_q * 4 + g_j) * 4 + g_sub
                        g_w = [
                            fx.Vector(bo.buffer_load(rsrc(g_bb), _w13_index(g_rr[g_r], g_gi), vec_width=4, dtype=T.i32))
                            for g_r in range(2)
                        ]
                        g_ws = [_byte(s13, scale_index(g_ee * (2 * INTER) + g_rr[g_r], g_gi, GROUPS)) for g_r in range(2)]
                        g_xv = fx.Vector(bo.buffer_load(
                            rsrc(scratch), (XQ + g_col * (HIDDEN // 2) + g_gi * 16) // 4, vec_width=4,
                            dtype=T.i32, cache_modifier=CM_DEV))
                        g_xs = fx.Int32(bo.buffer_load(
                            rsrc(scratch), (XS + (g_col * GROUPS + g_gi) * 4) // 4, vec_width=1,
                            dtype=T.i32, cache_modifier=CM_DEV))
                        g_l += g_w + g_ws + [g_xv, g_xs]
                    return g_l

                def g_unpack(g_l):
                    return [(g_l[6 * g_j:6 * g_j + 2], g_l[6 * g_j + 2:6 * g_j + 4], g_l[6 * g_j + 4], g_l[6 * g_j + 5])
                            for g_j in range(4)]

                _, g_c0, g_e0 = g_task_info(bid)
                g_first = g_quarter_of(g_e0, g_c0, 0)  # the first task's first quarter, in flight
                for g_task in range(bid, g_ntask, BLOCKS):
                    g_u, g_cb, g_e = g_task_info(g_task)
                    g_m = uniform(fx.ptr_load(umask + g_u))
                    g_base = w13 + fx.Int64(g_e) * W13_BYTES
                    g_rows = [g_cb * 16 + g_ni, INTER + g_cb * 16 + g_ni]

                    def g_quarter(g_q):
                        g_l = []
                        for g_j in range_constexpr(4):
                            g_gi = wave * 64 + (g_q * 4 + g_j) * 4 + g_sub
                            g_w = [
                                fx.Vector(bo.buffer_load(rsrc(g_base), _w13_index(g_rows[g_r], g_gi),
                                                         vec_width=4, dtype=T.i32))
                                for g_r in range(2)
                            ]
                            g_ws = [_byte(s13, scale_index(g_e * (2 * INTER) + g_rows[g_r], g_gi, GROUPS))
                                    for g_r in range(2)]
                            g_xv = fx.Vector(bo.buffer_load(
                                rsrc(scratch), (XQ + g_col * (HIDDEN // 2) + g_gi * 16) // 4, vec_width=4,
                                dtype=T.i32, cache_modifier=CM_DEV))
                            g_xs = fx.Int32(bo.buffer_load(
                                rsrc(scratch), (XS + (g_col * GROUPS + g_gi) * 4) // 4, vec_width=1,
                                dtype=T.i32, cache_modifier=CM_DEV))
                            g_l.append((g_w, g_ws, g_xv, g_xs))
                        return g_l

                    g_c = [fx.Vector.filled(4, 0.0, fx.Float32) for _ in range(2)]
                    g_next = g_unpack(g_first)
                    for g_q in range_constexpr(4):
                        g_ld = g_next
                        if const_expr(g_q + 1 < 4):
                            g_next = g_quarter(g_q + 1)
                        else:
                            # the next task's first quarter goes out before this task's tail
                            _, g_cn, g_en = g_task_info(g_task + BLOCKS)
                            g_first = g_quarter_of(g_en, g_cn, 0)
                        for g_j in range_constexpr(4):
                            g_w, g_ws, g_xv, g_xs = g_ld[g_j]
                            g_b = g_col_ok.select(g_xv, g_zero4)
                            g_bs = g_col_ok.select(g_xs, fx.Int32(127))
                            for g_r in range_constexpr(2):
                                g_c[g_r] = mfma_scaled(g_w[g_r], g_b, g_c[g_r], g_ws[g_r], g_bs, FP4, FP4)
                    # C: lane holds rows 4 (lane // 16) + i of column lane % 16 (a token)
                    g_par = (g_task // BLOCKS % 2) * (WAVES * 32 * MAX_TOKENS)
                    for g_r in range_constexpr(2):
                        for g_i in range_constexpr(4):
                            fx.ptr_store(
                                fx.Float32(g_c[g_r][g_i]),
                                part + g_par + ((wave * 2 + g_r) * 16 + g_sub * 4 + g_i) * MAX_TOKENS + g_ni,
                            )
                    gpu.barrier()
                    if tid < tokens * 8:
                        g_t2 = tid // 8
                        g_c2 = (tid % 8) * 2
                        if ((g_m >> g_t2) & fx.Int32(1)) != fx.Int32(0):
                            g_h2 = []
                            for g_q in range_constexpr(2):
                                g_gg = fx.ptr_load(part + g_par + (0 * 16 + g_c2 + g_q) * MAX_TOKENS + g_t2)
                                g_uu = fx.ptr_load(part + g_par + (1 * 16 + g_c2 + g_q) * MAX_TOKENS + g_t2)
                                for g_w2 in range_constexpr(1, WAVES):
                                    g_gg = g_gg + fx.ptr_load(part + g_par + ((g_w2 * 2) * 16 + g_c2 + g_q) * MAX_TOKENS + g_t2)
                                    g_uu = g_uu + fx.ptr_load(part + g_par + ((g_w2 * 2 + 1) * 16 + g_c2 + g_q) * MAX_TOKENS + g_t2)
                                g_h2.append(_silu_mul(g_gg, g_uu))
                            bo.buffer_store(
                                fx.Vector.from_elements(g_h2, fx.Float32).to(fx.BFloat16), rsrc(scratch),
                                MID // 2 + (g_u * MAX_TOKENS + g_t2) * INTER + g_cb * 16 + g_c2, cache_modifier=CM_DEV,
                            )
                _release()
                gpu.barrier()
                if tid == 0:
                    box["mb"].put(box["gf"], bid, fx.Int32(1))

            if const_expr(last >= 3):
                # ---- RQ: q_a thread q_a (expert, token, group) of MID
                if tid < BLOCKS:
                    box["mb"].poll([(box["gf"], tid, 1)])
                gpu.barrier()
                q_n_tasks = nu * (tokens * MGROUPS)
                for q_task in range(bid * THREADS + tid, q_n_tasks, BLOCKS * THREADS):
                    q_u = q_task // (tokens * MGROUPS)
                    q_t = (q_task // MGROUPS) % tokens
                    q_grp = q_task % MGROUPS
                    q_m = fx.ptr_load(umask + q_u)
                    if ((q_m >> q_t) & fx.Int32(1)) != fx.Int32(0):
                        q_vals = []
                        for q_c in range_constexpr(4):
                            q_wv4 = fx.Vector(bo.buffer_load(
                                rsrc(scratch), (MID + ((q_u * MAX_TOKENS + q_t) * INTER + q_grp * 32) * 2) // 4 + 4 * q_c,
                                vec_width=4, dtype=T.i32, cache_modifier=CM_DEV))
                            for q_i in range_constexpr(4):
                                q_a, q_b = bf2_f32(fx.Int32(q_wv4[q_i]))
                                q_vals += [q_a, q_b]
                        q_words, q_byte = quant_group(q_vals, "hip")
                        bo.buffer_store(fx.Vector.from_elements(q_words, fx.Int32), rsrc(scratch),
                                        (MQ + (q_u * MAX_TOKENS + q_t) * (INTER // 2) + q_grp * 16) // 4, cache_modifier=CM_DEV)
                        bo.buffer_store(q_byte, rsrc(scratch), (MS + ((q_u * MAX_TOKENS + q_t) * MGROUPS + q_grp) * 4) // 4,
                                        cache_modifier=CM_DEV)
                _release()
                gpu.barrier()
                if tid == 0:
                    box["mb"].put(box["rf"], bid, fx.Int32(1))

            if const_expr(last >= 4):
                # ---- DOWN (MFMA): rows [64 b, 64 b + 64) = four shuffle groups; wave w takes
                # group w // 2 and K steps (w % 2) * 2 .. + 2 (128 K each). A = 16 W2 rows, B = 16
                # token columns of the expert's quantized mid (zero where the token is not
                # routed there); per expert C times the column's route weight, summed in fp32
                if tid < BLOCKS:
                    box["mb"].poll([(box["rf"], tid, 1)])
                gpu.barrier()
                d_ni = lane % 16
                d_sub = lane // 16
                d_row = bid * 64 + (wave // 2) * 16 + d_ni
                d_gis = [(wave % 2) * 8 + d_j * 4 + d_sub for d_j in range(2)]
                d_col = fx.min(d_ni, fx.Int32(tokens - 1))
                d_col_ok = d_ni < tokens
                d_zero4 = fx.Vector.filled(4, 0, fx.Int32)

                def d_load(d_i):
                    d_c = fx.min(d_i, fx.max(nu - 1, fx.Int32(0)))
                    de = uniform(fx.ptr_load(uexp + d_c))
                    dm = uniform(fx.ptr_load(umask + d_c))
                    dbase = w2 + fx.Int64(de) * W2_BYTES
                    dww = [
                        fx.Vector(bo.buffer_load(rsrc(dbase), _w2_group_index(d_row, dg), vec_width=4, dtype=T.i32))
                        for dg in d_gis
                    ]
                    dsc = [_byte(s2, scale_index(de * HIDDEN + d_row, dg, MGROUPS)) for dg in d_gis]
                    dmv = [fx.Vector(bo.buffer_load(
                        rsrc(scratch), (MQ + (d_c * MAX_TOKENS + d_col) * (INTER // 2) + dg * 16) // 4,
                        vec_width=4, dtype=T.i32, cache_modifier=CM_DEV)) for dg in d_gis]
                    dms = [fx.Int32(bo.buffer_load(
                        rsrc(scratch), (MS + ((d_c * MAX_TOKENS + d_col) * MGROUPS + dg) * 4) // 4,
                        vec_width=1, dtype=T.i32, cache_modifier=CM_DEV)) for dg in d_gis]
                    return [de, dm] + dww + dsc + dmv + dms

                d_acc = fx.Vector.filled(4, 0.0, fx.Float32)
                DEPTH = 2  # experts in flight ahead of the one computing (4 measured no better)
                d_fly = [d_load(fx.Int32(d_i)) for d_i in range(DEPTH)]
                d_nu = ((skip & fx.Int32(2)) == fx.Int32(0)).select(nu, fx.Int32(0))  # timing aid
                for d_u in range(d_nu):
                    d_cur = d_fly[0]
                    d_fly = d_fly[1:] + [d_load(d_u + DEPTH)]
                    d_m = d_cur[1]
                    d_on = d_col_ok & (((d_m >> d_col) & fx.Int32(1)) != fx.Int32(0))
                    d_cc = fx.Vector.filled(4, 0.0, fx.Float32)
                    for d_j in range_constexpr(2):
                        d_mv = d_cur[6 + d_j]
                        d_ms = d_cur[8 + d_j]
                        d_cc = mfma_scaled(d_cur[2 + d_j], d_on.select(d_mv, d_zero4), d_cc, d_cur[4 + d_j],
                                           d_on.select(d_ms, fx.Int32(127)), FP4, FP4)
                    d_wt = d_on.select(fx.ptr_load(uw + d_u * MAX_TOKENS + d_col), fx.Float32(0.0))
                    d_acc = fx.Vector.from_elements(
                        [fx.Float32(d_acc[d_i]) + d_wt * fx.Float32(d_cc[d_i]) for d_i in range(4)], fx.Float32
                    )
                # the wave pair (K halves) through LDS, then rows x tokens out
                for d_i in range_constexpr(4):
                    fx.ptr_store(fx.Float32(d_acc[d_i]), part + (wave * 16 + d_sub * 4 + d_i) * MAX_TOKENS + d_ni)
                gpu.barrier()
                if tid < 4 * 8 * tokens:
                    d_t2 = tid // 32
                    d_rb = (tid % 32) // 8
                    d_p = (tid % 8) * 2
                    d_vals = []
                    for d_q in range_constexpr(2):
                        d_vals.append(
                            fx.ptr_load(part + ((d_rb * 2) * 16 + d_p + d_q) * MAX_TOKENS + d_t2)
                            + fx.ptr_load(part + ((d_rb * 2 + 1) * 16 + d_p + d_q) * MAX_TOKENS + d_t2)
                        )
                    bo.buffer_store(
                        fx.Vector.from_elements(d_vals, fx.Float32).to(fx.BFloat16),
                        rsrc(out), d_t2 * HIDDEN + bid * 64 + d_rb * 16 + d_p,
                    )

    @flyc.jit
    def launch(
        x: Int64, ids: Int64, wts: Int64, w13: Int64, s13: Int64, w2: Int64, s2: Int64, out: Int64,
        scratch: Int64, epoch: Int64, layer: Int32, skip: Int32 = 0, stream: fx.Stream = fx.Stream(None),
    ):
        """``skip``: timing aid (bit 0 skips UG's tasks, bit 1 DOWN's experts; the same binary)."""
        _ = keyed
        k(x, ids, wts, w13, s13, w2, s2, out, scratch, epoch, layer, skip).launch(
            grid=(BLOCKS,), block=(THREADS,), stream=stream
        )

    return launch
