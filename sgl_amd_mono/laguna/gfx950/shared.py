# SPDX-License-Identifier: Apache-2.0
"""S4, the shared expert of Laguna's FFN launch on gfx950 (W4A4, quark layout):
the TP-sharded LagunaMLP (merged gate_up rows [gate 512 | up 512] x 16384, down
16384 x 512 per rank), MXFP4 weights [N, K/2] uint8 + e8m0 scales [N, K/32]
row-major, activations quantized with aiter's triton rule (``mxq``), as
SGLang's quark W4A4 linear does (``dynamic_mxfp4_quant`` + ``gemm_afp4wfp4``).

One launch, BLOCKS CTAs x THREADS threads, stages in order (tagged scratch
mailboxes between them):

    Q    : CTA t < tokens: the token's 512 groups, a thread a group -> XQ / XS
    GU   : CTA b: output channels 2b, 2b + 1 (gate and up rows: 4 rows); wave w
           takes K groups [64 w, 64 w + 64), a lane a group (32 elements), fp4
           decoded exactly to fp32, fp32 FMA; waves summed in order; bf16 (the
           GEMM's output), silu(gate) * up in fp32 -> bf16 MID
    RQ   : CTA b < 16 tokens: token b // 16, group b % 16 of MID -> MQ / MS
    DOWN : CTA b: rows [64 b, 64 b + 64), wave w eight of them, a lane a dword
           (8 of the 512 K); bf16 partial rows of this rank -> ``out``
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import Int32, Int64, T

from sgl_amd_mono.common import guard
from sgl_amd_mono.common.mx import FP4, mfma_scaled
from sgl_amd_mono.common.ops import CM_DEV, bf2_f32, bf16_round, hw_exp2, hw_rcp, rsrc, uniform
from sgl_amd_mono.common.plan import BLOCKS, THREADS, WAVES, kernel_sources
from sgl_amd_mono.common.sync import Mailbox, sreg
from sgl_amd_mono.laguna.gfx950 import mxq
from sgl_amd_mono.laguna.gfx950.mxq import decode_word, pow2, quant_group

HIDDEN = 16384
INTER = 512  # shared expert inter per rank (TP8)
MAX_TOKENS = 16
LAYER_SLOTS = 64
GROUPS = HIDDEN // 32  # 512 K groups of x
MGROUPS = INTER // 32  # 16 groups of MID
LOG2E = 1.4426950408889634
assert 2 * BLOCKS == INTER and GROUPS == THREADS and HIDDEN == 64 * BLOCKS

# scratch layout (bytes)
XQ = 0
XS = XQ + MAX_TOKENS * HIDDEN // 2
MID = XS + MAX_TOKENS * GROUPS * 4
MQ = MID + MAX_TOKENS * INTER * 2
MS = MQ + MAX_TOKENS * INTER // 2
KSLICES = 8  # GU splits K over 8 CTAs a 16-channel block
PART = MS + MAX_TOKENS * MGROUPS * 4  # fp32 [slice][row 1024][token]
QF = PART + KSLICES * 2 * INTER * MAX_TOKENS * 4  # flags: [token][wave]
GF = QF + MAX_TOKENS * WAVES * 8  # [CTA]: its K-slice partials are out
HF = GF + BLOCKS * 8  # [channel block]: its MID channels are out
RF = HF + (INTER // 16) * 8  # [token][group]
SCRATCH_BYTES = RF + MAX_TOKENS * MGROUPS * 8


def _release():
    """Order this wave's device-scope stores before a flag other CTAs poll."""
    rocdl.s_waitcnt(vmcnt=0)
    fx.memory_fence(syncscope=rocdl.SyncScope.Workgroup, ordering=fx.AtomicOrdering.Release)


def _byte(ptr, byte_index, cm=0):
    """One byte of a uint8 buffer as Int32."""
    w = fx.Int32(bo.buffer_load(rsrc(ptr), byte_index // 4, vec_width=1, dtype=T.i32, cache_modifier=cm))
    return (w >> ((byte_index % 4) * 8)) & fx.Int32(0xFF)


def _silu_mul(g, u):
    return g * hw_rcp(fx.Float32(1.0) + hw_exp2(-g * fx.Float32(LOG2E))) * u


STAGES = ("Q", "GU", "GR", "RQ", "DOWN")


@flyc.jit
def emit_shared(ctx, args, tokens, stages=STAGES, in_cm=0, out_cm=0):
    """S4's code inside a kernel, the stages in ``stages`` (each waits on its inputs' flags, so a
    launch may spread them out): ``ctx`` = thread ids, the mailbox dict ``box`` (its ``mb``) and
    the LDS ``spart`` (WAVES x 32 x MAX_TOKENS fp32); ``args`` = the pointers (``scratch``:
    ``SCRATCH_BYTES``). ``in_cm`` / ``out_cm``: the cache policy of the x loads / out stores."""
    tid, bid, lane, wave, box, part = (ctx[n] for n in ("tid", "bid", "lane", "wave", "box", "spart"))
    x, w13, s13, w2, s2, out, scratch = (args[n] for n in ("x", "w13", "s13", "w2", "s2", "out", "scratch"))
    box["qf"] = sreg(scratch, QF, "qf")
    box["gf"] = sreg(scratch, GF, "gf")
    box["rf"] = sreg(scratch, RF, "rf")
    box["hf"] = sreg(scratch, HF, "hf")
    ni = lane % 16
    sub = lane // 16
    col_ok = ni < tokens
    col = fx.min(ni, fx.Int32(tokens - 1))
    zero4 = fx.Vector.filled(4, 0, fx.Int32)

    if const_expr("Q" in stages):
        # ---- Q: quantize x (triton rule), a thread a group
        if bid < tokens:
            vals = []
            for c in range_constexpr(4):
                w = fx.Vector(
                    bo.buffer_load(rsrc(x), (bid * HIDDEN + tid * 32) // 2 + 4 * c, vec_width=4, dtype=T.i32, cache_modifier=in_cm)
                )
                for i in range_constexpr(4):
                    a, b = bf2_f32(fx.Int32(w[i]))
                    vals += [a, b]
            words, byte = quant_group(vals, "triton")
            bo.buffer_store(
                fx.Vector.from_elements(words, fx.Int32), rsrc(scratch),
                (XQ + bid * (HIDDEN // 2) + tid * 16) // 4, cache_modifier=CM_DEV,
            )
            bo.buffer_store(byte, rsrc(scratch), (XS + (bid * GROUPS + tid) * 4) // 4, cache_modifier=CM_DEV)
            _release()
            if lane == 0:
                box["mb"].put(box["qf"], bid * WAVES + wave, fx.Int32(1))

    if const_expr("GU" in stages):
        # ---- GU (MFMA): CTA b = (16-channel block cb = b // 8, K slice ks = b % 8 of 64
        # groups); wave w: groups 64 ks + 8 w .. + 8 in two 128-K steps. A = 16 gate (and 16
        # up) rows, row-major quark layout (lane: row lane % 16, 16 B of K group lane // 16),
        # B = 16 token columns of x (zero past the batch); fp32 slice partials -> PART
        if tid < tokens * WAVES:
            box["mb"].poll([(box["qf"], tid, 1)])
        gpu.barrier()
        cb = bid // KSLICES
        ks = bid % KSLICES
        grow = [cb * 16 + ni, INTER + cb * 16 + ni]
        cacc = [fx.Vector.filled(4, 0.0, fx.Float32) for _ in range(2)]
        ld = []
        for j in range_constexpr(2):
            gk = ks * 64 + wave * 8 + j * 4 + sub
            ld.append((
                [fx.Vector(bo.buffer_load(rsrc(w13), (gr * (HIDDEN // 2) + gk * 16) // 4, vec_width=4, dtype=T.i32))
                 for gr in grow],
                [_byte(s13, gr * GROUPS + gk) for gr in grow],
                fx.Vector(bo.buffer_load(rsrc(scratch), (XQ + col * (HIDDEN // 2) + gk * 16) // 4, vec_width=4,
                                         dtype=T.i32, cache_modifier=CM_DEV)),
                fx.Int32(bo.buffer_load(rsrc(scratch), (XS + (col * GROUPS + gk) * 4) // 4, vec_width=1,
                                        dtype=T.i32, cache_modifier=CM_DEV)),
            ))
        for j in range_constexpr(2):
            wv2, ws2, xv, xs = ld[j]
            bv = col_ok.select(xv, zero4)
            bs = col_ok.select(xs, fx.Int32(127))
            for r in range_constexpr(2):
                cacc[r] = mfma_scaled(wv2[r], bv, cacc[r], ws2[r], bs, FP4, FP4)
        # waves summed in order through LDS (C: rows 4 sub + i, column = token ni)
        for r in range_constexpr(2):
            for i in range_constexpr(4):
                fx.ptr_store(fx.Float32(cacc[r][i]), part + ((wave * 2 + r) * 16 + sub * 4 + i) * MAX_TOKENS + ni)
        gpu.barrier()
        if tid < 32 * tokens:
            prow = tid // tokens  # 0..31: 16 gate rows then 16 up rows
            pt = tid % tokens
            ptot = fx.ptr_load(part + ((0 * 2 + prow // 16) * 16 + prow % 16) * MAX_TOKENS + pt)
            for w in range_constexpr(1, WAVES):
                ptot = ptot + fx.ptr_load(part + ((w * 2 + prow // 16) * 16 + prow % 16) * MAX_TOKENS + pt)
            grow_g = (prow // 16) * INTER + cb * 16 + prow % 16
            bo.buffer_store(ptot, rsrc(scratch), (PART + ((ks * 2 * INTER + grow_g) * MAX_TOKENS + pt) * 4) // 4,
                            cache_modifier=CM_DEV)
        _release()
        gpu.barrier()
        if tid == 0:
            box["mb"].put(box["gf"], bid, fx.Int32(1))

    if const_expr("GR" in stages):
        # ---- GR: CTA cb < 32: the block's 8 K-slice partials summed in slice order, bf16
        # (the GEMM's output), silu(gate) * up -> bf16 MID
        if bid < INTER // 16:
            if tid < KSLICES:
                box["mb"].poll([(box["gf"], bid * KSLICES + tid, 1)])
            gpu.barrier()
            if tid < 8 * tokens:
                ht = tid // 8
                hc = (tid % 8) * 2
                hh = []
                for q in range_constexpr(2):
                    hv = []
                    for half in range_constexpr(2):
                        hrow = half * INTER + bid * 16 + hc + q
                        htot = fx.Float32(0.0)
                        for k2 in range_constexpr(KSLICES):
                            htot = htot + fx.Float32(bo.buffer_load(
                                rsrc(scratch), (PART + ((k2 * 2 * INTER + hrow) * MAX_TOKENS + ht) * 4) // 4,
                                vec_width=1, dtype=T.f32, cache_modifier=CM_DEV))
                        hv.append(bf16_round(htot))
                    hh.append(_silu_mul(hv[0], hv[1]))
                bo.buffer_store(
                    fx.Vector.from_elements(hh, fx.Float32).to(fx.BFloat16), rsrc(scratch),
                    MID // 2 + ht * INTER + bid * 16 + hc, cache_modifier=CM_DEV,
                )
            _release()
            gpu.barrier()
            if tid == 0:
                box["mb"].put(box["hf"], bid, fx.Int32(1))

    if const_expr("RQ" in stages):
        # ---- RQ: requantize MID group (token, group) = (b // 16, b % 16)
        if bid < MGROUPS * tokens:
            t = bid // MGROUPS
            grp = bid % MGROUPS
            if wave == 0:
                box["mb"].poll([(box["hf"], grp * 2 + i, 1) for i in range(2)])
                if lane == 0:
                    vals = []
                    for c in range_constexpr(4):
                        w = fx.Vector(
                            bo.buffer_load(rsrc(scratch), (MID + (t * INTER + grp * 32) * 2) // 4 + 4 * c,
                                           vec_width=4, dtype=T.i32, cache_modifier=CM_DEV)
                        )
                        for i in range_constexpr(4):
                            a, b = bf2_f32(fx.Int32(w[i]))
                            vals += [a, b]
                    words, byte = quant_group(vals, "triton")
                    bo.buffer_store(fx.Vector.from_elements(words, fx.Int32), rsrc(scratch),
                                    (MQ + t * (INTER // 2) + grp * 16) // 4, cache_modifier=CM_DEV)
                    bo.buffer_store(byte, rsrc(scratch), (MS + (t * MGROUPS + grp) * 4) // 4,
                                    cache_modifier=CM_DEV)
                    _release()
                    box["mb"].put(box["rf"], t * MGROUPS + grp, fx.Int32(1))

    if const_expr("DOWN" in stages):
        # ---- DOWN (MFMA): rows [64 b, 64 b + 64) = four 16-row groups; wave w: group w // 2,
        # K steps (w % 2) * 2 .. + 2 (128 K each). A = 16 W2 rows (row-major), B = 16 token
        # columns of the quantized MID (zero past the batch)
        if tid < tokens * MGROUPS:
            box["mb"].poll([(box["rf"], tid, 1)])
        gpu.barrier()
        drow = bid * 64 + (wave // 2) * 16 + ni
        dacc = fx.Vector.filled(4, 0.0, fx.Float32)
        for j in range_constexpr(2):
            dg = (wave % 2) * 8 + j * 4 + sub
            dw = fx.Vector(bo.buffer_load(rsrc(w2), (drow * (INTER // 2) + dg * 16) // 4, vec_width=4, dtype=T.i32))
            dsc = _byte(s2, drow * MGROUPS + dg)
            dmv = fx.Vector(bo.buffer_load(rsrc(scratch), (MQ + col * (INTER // 2) + dg * 16) // 4, vec_width=4,
                                           dtype=T.i32, cache_modifier=CM_DEV))
            dms = fx.Int32(bo.buffer_load(rsrc(scratch), (MS + (col * MGROUPS + dg) * 4) // 4, vec_width=1,
                                          dtype=T.i32, cache_modifier=CM_DEV))
            dacc = mfma_scaled(dw, col_ok.select(dmv, zero4), dacc, dsc, col_ok.select(dms, fx.Int32(127)), FP4, FP4)
        gpu.barrier()
        for i in range_constexpr(4):
            fx.ptr_store(fx.Float32(dacc[i]), part + (wave * 16 + sub * 4 + i) * MAX_TOKENS + ni)
        gpu.barrier()
        if tid < 4 * 8 * tokens:
            ot = tid // 32
            orb = (tid % 32) // 8
            op = (tid % 8) * 2
            ov = []
            for q in range_constexpr(2):
                ov.append(
                    fx.ptr_load(part + ((orb * 2) * 16 + op + q) * MAX_TOKENS + ot)
                    + fx.ptr_load(part + ((orb * 2 + 1) * 16 + op + q) * MAX_TOKENS + ot)
                )
            bo.buffer_store(
                fx.Vector.from_elements(ov, fx.Float32).to(fx.BFloat16),
                rsrc(out), ot * HIDDEN + bid * 64 + orb * 16 + op, cache_modifier=out_cm,
            )


def build_shared(tokens: int):
    assert 1 <= tokens <= MAX_TOKENS
    keyed = (
        tokens,
        kernel_sources(__name__),
        kernel_sources(mxq.__name__),
        kernel_sources(guard.__name__),
    )
    name = f"laguna_shared_t{tokens}"

    @fx.struct
    class Smem:
        part: fx.Array[fx.Float32, WAVES * 32 * MAX_TOKENS, 16]

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def k(
        x: Int64, w13: Int64, s13: Int64, w2: Int64, s2: Int64, out: Int64,
        scratch: Int64, epoch: Int64, layer: Int32,
    ):
        _ = keyed
        with guard.scope(guard.Guard(epoch)):
            tid = fx.thread_idx.x
            bid = fx.block_idx.x
            lane = tid % 64
            wave = tid // 64
            lds = fx.SharedAllocator().allocate(Smem).peek()
            part = lds.part.ptr
            step = uniform(bo.buffer_load(rsrc(epoch), 0, vec_width=1, dtype=T.i32))
            box = {"mb": Mailbox(step * LAYER_SLOTS + layer + 1)}
            ctx = {"tid": tid, "bid": bid, "lane": lane, "wave": wave, "box": box, "spart": part}
            args = {"x": x, "w13": w13, "s13": s13, "w2": w2, "s2": s2, "out": out, "scratch": scratch}
            emit_shared(ctx, args, tokens)

    @flyc.jit
    def launch(
        x: Int64, w13: Int64, s13: Int64, w2: Int64, s2: Int64, out: Int64,
        scratch: Int64, epoch: Int64, layer: Int32, stream: fx.Stream = fx.Stream(None),
    ):
        _ = keyed
        k(x, w13, s13, w2, s2, out, scratch, epoch, layer).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)

    return launch
