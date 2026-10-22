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
from flydsl.expr import gpu, range_constexpr, rocdl
from flydsl.expr.typing import Int32, Int64, T

from sgl_amd_mono.common import guard
from sgl_amd_mono.common.mx import scale_index
from sgl_amd_mono.common.ops import (
    CM_DEV, ballot, bf2_f32, butterfly, hw_exp2, hw_rcp, lanes_below, popcount, rsrc, uniform,
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


def build_routed(tokens: int):
    assert 1 <= tokens <= MAX_TOKENS
    keyed = (
        tokens,
        kernel_sources(__name__),
        kernel_sources(mxq.__name__),
        kernel_sources(guard.__name__),
    )
    name = f"laguna_routed_t{tokens}"

    @fx.struct
    class Smem:
        uexp: fx.Array[fx.Int32, U_MAX, 16]
        umask: fx.Array[fx.Int32, U_MAX, 16]
        uw: fx.Array[fx.Float32, U_MAX * MAX_TOKENS, 16]
        wc: fx.Array[fx.Int32, WAVES + 1, 16]
        part: fx.Array[fx.Float32, WAVES * 4 * MAX_TOKENS, 16]
        gu: fx.Array[fx.Float32, 4 * MAX_TOKENS, 16]

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def k(
        x: Int64, ids: Int64, wts: Int64, w13: Int64, s13: Int64, w2: Int64, s2: Int64, out: Int64,
        scratch: Int64, epoch: Int64, layer: Int32,
    ):
        _ = keyed
        with guard.scope(guard.Guard(epoch)):
            tid = fx.thread_idx.x
            bid = fx.block_idx.x
            lane = tid % 64
            wave = tid // 64
            lds = fx.SharedAllocator().allocate(Smem).peek()
            uexp, umask, uw = lds.uexp.ptr, lds.umask.ptr, lds.uw.ptr
            wc, part, gu = lds.wc.ptr, lds.part.ptr, lds.gu.ptr
            step = uniform(bo.buffer_load(rsrc(epoch), 0, vec_width=1, dtype=T.i32))
            box = {
                "mb": Mailbox(step * LAYER_SLOTS + layer + 1),
                "qf": sreg(scratch, QF, "qf"),
                "gf": sreg(scratch, GF, "gf"),
                "rf": sreg(scratch, RF, "rf"),
            }

            # ---- U: the union, thread e = expert e
            e_me = fx.Int32(tid)
            m_me = fx.Int32(0)
            w_me = [fx.Float32(0.0) for _ in range(tokens)]
            for t in range_constexpr(tokens):
                for kk in range_constexpr(TOPK):
                    v = fx.Int32(bo.buffer_load(rsrc(ids), t * TOPK + kk, vec_width=1, dtype=T.i32))
                    hit = v == e_me
                    m_me = m_me | hit.select(fx.Int32(1 << t), fx.Int32(0))
                    wv = fx.Float32(bo.buffer_load(rsrc(wts), t * TOPK + kk, vec_width=1, dtype=T.f32))
                    w_me[t] = hit.select(wv, w_me[t])
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

            # ---- UG: channels 2b, 2b + 1 of every union expert
            box["mb"].poll([(box["qf"], g_t * WAVES + g_w, 1) for g_t in range(tokens) for g_w in range(WAVES)])
            gi = wave * 64 + lane
            xw = [
                fx.Vector(bo.buffer_load(rsrc(scratch), (XQ + g_t * (HIDDEN // 2) + gi * 16) // 4, vec_width=4,
                                         dtype=T.i32, cache_modifier=CM_DEV))
                for g_t in range(tokens)
            ]
            xsc = [
                pow2(fx.Int32(bo.buffer_load(rsrc(scratch), (XS + (g_t * GROUPS + gi) * 4) // 4, vec_width=1,
                                             dtype=T.i32, cache_modifier=CM_DEV)))
                for g_t in range(tokens)
            ]
            rows = [bid * 2, bid * 2 + 1, INTER + bid * 2, INTER + bid * 2 + 1]
            for u in range(nu):
                g_e = uniform(fx.ptr_load(uexp + u))
                g_m = uniform(fx.ptr_load(umask + u))
                g_wbase = w13 + fx.Int64(g_e) * W13_BYTES
                g_ww = [
                    fx.Vector(bo.buffer_load(rsrc(g_wbase), _w13_index(g_r, gi), vec_width=4, dtype=T.i32))
                    for g_r in rows
                ]
                g_wsc = [pow2(_byte(s13, scale_index(g_e * (2 * INTER) + g_r, gi, GROUPS))) for g_r in rows]
                for g_t in range_constexpr(tokens):
                    if ((g_m >> g_t) & fx.Int32(1)) != fx.Int32(0):
                        g_acc = [fx.Float32(0.0) for _ in range(4)]
                        for g_c in range_constexpr(4):
                            g_xd = decode_word(fx.Int32(xw[g_t][g_c]), xsc[g_t])
                            for g_r in range_constexpr(4):
                                g_wd = decode_word(fx.Int32(g_ww[g_r][g_c]), g_wsc[g_r])
                                g_s = g_acc[g_r]
                                for g_j in range_constexpr(8):
                                    g_s = g_s + g_wd[g_j] * g_xd[g_j]
                                g_acc[g_r] = g_s
                        for g_r in range_constexpr(4):
                            g_v = butterfly(g_acc[g_r], (32, 16, 8, 4, 2, 1))
                            if lane == 0:
                                fx.ptr_store(g_v, part + (wave * 4 + g_r) * MAX_TOKENS + g_t)
                gpu.barrier()
                if tid < tokens:
                    if ((g_m >> tid) & fx.Int32(1)) != fx.Int32(0):
                        g_g4 = []
                        for g_r in range_constexpr(4):
                            g_tot = fx.ptr_load(part + g_r * MAX_TOKENS + tid)
                            for g_w in range_constexpr(1, WAVES):
                                g_tot = g_tot + fx.ptr_load(part + (g_w * 4 + g_r) * MAX_TOKENS + tid)
                            g_g4.append(g_tot)
                        g_h0 = _silu_mul(g_g4[0], g_g4[2])
                        g_h1 = _silu_mul(g_g4[1], g_g4[3])
                        bo.buffer_store(
                            fx.Vector.from_elements([g_h0, g_h1], fx.Float32).to(fx.BFloat16), rsrc(scratch),
                            MID // 2 + (u * MAX_TOKENS + tid) * INTER + bid * 2, cache_modifier=CM_DEV,
                        )
                gpu.barrier()
            _release()
            gpu.barrier()
            if tid == 0:
                box["mb"].put(box["gf"], bid, fx.Int32(1))

            # ---- RQ: q_a thread q_a (expert, token, group) of MID
            box["mb"].poll([(box["gf"], q_i, 1) for q_i in range(BLOCKS)])
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

            # ---- DOWN: rows [64 b, 64 b + 64), the route-weighted sum over the union
            box["mb"].poll([(box["rf"], i, 1) for i in range(BLOCKS)])
            d_acc = [fx.Float32(0.0) for _ in range(8 * tokens)]
            for d_u in range(nu):
                d_e = uniform(fx.ptr_load(uexp + d_u))
                d_m = uniform(fx.ptr_load(umask + d_u))
                d_wbase = w2 + fx.Int64(d_e) * W2_BYTES
                d_wd = []
                for d_rr in range_constexpr(8):
                    d_row = bid * 64 + wave * 8 + d_rr
                    d_wv = fx.Int32(bo.buffer_load(rsrc(d_wbase), _w2_index(d_row, lane), vec_width=1, dtype=T.i32))
                    d_wd.append(decode_word(d_wv, pow2(_byte(s2, scale_index(d_e * HIDDEN + d_row, lane // 4, MGROUPS)))))
                for d_t in range_constexpr(tokens):
                    d_rt = ((d_m >> d_t) & fx.Int32(1)) != fx.Int32(0)
                    d_mw = fx.Int32(bo.buffer_load(rsrc(scratch), (MQ + (d_u * MAX_TOKENS + d_t) * (INTER // 2)) // 4 + lane,
                                                 vec_width=1, dtype=T.i32, cache_modifier=CM_DEV))
                    d_ms = fx.Int32(bo.buffer_load(rsrc(scratch), (MS + ((d_u * MAX_TOKENS + d_t) * MGROUPS + lane // 4) * 4) // 4,
                                                 vec_width=1, dtype=T.i32, cache_modifier=CM_DEV))
                    d_md = decode_word(d_mw, pow2(d_ms))
                    d_wt = d_rt.select(fx.ptr_load(uw + d_u * MAX_TOKENS + d_t), fx.Float32(0.0))
                    for d_rr in range_constexpr(8):
                        d_s = fx.Float32(0.0)
                        for d_j in range_constexpr(8):
                            d_s = d_s + d_wd[d_rr][d_j] * d_md[d_j]
                        d_acc[d_rr * tokens + d_t] = d_acc[d_rr * tokens + d_t] + d_wt * d_s
            for d_pr in range_constexpr(4):
                for d_t in range_constexpr(tokens):
                    d_v0 = butterfly(d_acc[(2 * d_pr) * tokens + d_t], (32, 16, 8, 4, 2, 1))
                    d_v1 = butterfly(d_acc[(2 * d_pr + 1) * tokens + d_t], (32, 16, 8, 4, 2, 1))
                    if lane == 0:
                        bo.buffer_store(
                            fx.Vector.from_elements([d_v0, d_v1], fx.Float32).to(fx.BFloat16),
                            rsrc(out), d_t * HIDDEN + bid * 64 + wave * 8 + d_pr * 2,
                        )

    @flyc.jit
    def launch(
        x: Int64, ids: Int64, wts: Int64, w13: Int64, s13: Int64, w2: Int64, s2: Int64, out: Int64,
        scratch: Int64, epoch: Int64, layer: Int32, stream: fx.Stream = fx.Stream(None),
    ):
        _ = keyed
        k(x, ids, wts, w13, s13, w2, s2, out, scratch, epoch, layer).launch(
            grid=(BLOCKS,), block=(THREADS,), stream=stream
        )

    return launch
