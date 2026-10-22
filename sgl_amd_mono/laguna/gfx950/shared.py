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
from flydsl.expr import gpu, range_constexpr, rocdl
from flydsl.expr.typing import Int32, Int64, T

from sgl_amd_mono.common import guard
from sgl_amd_mono.common.ops import CM_DEV, bf2_f32, bf16_round, butterfly, hw_exp2, hw_rcp, rsrc, uniform
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
QF = MS + MAX_TOKENS * MGROUPS * 4  # flags: [token][wave]
GF = QF + MAX_TOKENS * WAVES * 8  # [CTA]
RF = GF + BLOCKS * 8  # [token][group]
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
        part: fx.Array[fx.Float32, WAVES * 4 * MAX_TOKENS, 16]
        gu: fx.Array[fx.Float32, 4 * MAX_TOKENS, 16]

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
            gu = lds.gu.ptr
            step = uniform(bo.buffer_load(rsrc(epoch), 0, vec_width=1, dtype=T.i32))
            box = {
                "mb": Mailbox(step * LAYER_SLOTS + layer + 1),
                "qf": sreg(scratch, QF, "qf"),
                "gf": sreg(scratch, GF, "gf"),
                "rf": sreg(scratch, RF, "rf"),
            }

            # ---- Q: quantize x (triton rule), a thread a group
            if bid < tokens:
                vals = []
                for c in range_constexpr(4):
                    w = fx.Vector(
                        bo.buffer_load(rsrc(x), (bid * HIDDEN + tid * 32) // 2 + 4 * c, vec_width=4, dtype=T.i32)
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

            # ---- GU: channels 2b, 2b + 1 (rows n0, n1, 512 + n0, 512 + n1)
            box["mb"].poll([(box["qf"], t * WAVES + w, 1) for t in range(tokens) for w in range(WAVES)])
            gi = wave * 64 + lane  # this lane's K group
            rows = [bid * 2, bid * 2 + 1, INTER + bid * 2, INTER + bid * 2 + 1]
            wsc = [pow2(_byte(s13, r * GROUPS + gi)) for r in rows]
            ww = [
                fx.Vector(bo.buffer_load(rsrc(w13), (r * (HIDDEN // 2) + gi * 16) // 4, vec_width=4, dtype=T.i32))
                for r in rows
            ]
            acc = [[fx.Float32(0.0) for _ in range(tokens)] for _ in range(4)]
            xw = [
                fx.Vector(
                    bo.buffer_load(
                        rsrc(scratch), (XQ + t * (HIDDEN // 2) + gi * 16) // 4, vec_width=4, dtype=T.i32,
                        cache_modifier=CM_DEV,
                    )
                )
                for t in range(tokens)
            ]
            xsc = [
                pow2(fx.Int32(bo.buffer_load(rsrc(scratch), (XS + (t * GROUPS + gi) * 4) // 4, vec_width=1,
                                             dtype=T.i32, cache_modifier=CM_DEV)))
                for t in range(tokens)
            ]
            for c in range_constexpr(4):
                wd = [decode_word(fx.Int32(ww[r][c]), wsc[r]) for r in range(4)]
                for t in range_constexpr(tokens):
                    xd = decode_word(fx.Int32(xw[t][c]), xsc[t])
                    for r in range_constexpr(4):
                        s = acc[r][t]
                        for j in range_constexpr(8):
                            s = s + wd[r][j] * xd[j]
                        acc[r][t] = s
            for r in range_constexpr(4):
                for t in range_constexpr(tokens):
                    v = butterfly(acc[r][t], (32, 16, 8, 4, 2, 1))
                    if lane == 0:
                        fx.ptr_store(v, part + (wave * 4 + r) * MAX_TOKENS + t)
            gpu.barrier()
            if tid < 4 * tokens:
                r = tid // tokens
                t = tid % tokens
                tot = fx.ptr_load(part + r * MAX_TOKENS + t)
                for w in range_constexpr(1, WAVES):
                    tot = tot + fx.ptr_load(part + (w * 4 + r) * MAX_TOKENS + t)
                fx.ptr_store(bf16_round(tot), gu + r * MAX_TOKENS + t)
            gpu.barrier()
            if tid < tokens:
                h0 = _silu_mul(fx.ptr_load(gu + 0 * MAX_TOKENS + tid), fx.ptr_load(gu + 2 * MAX_TOKENS + tid))
                h1 = _silu_mul(fx.ptr_load(gu + 1 * MAX_TOKENS + tid), fx.ptr_load(gu + 3 * MAX_TOKENS + tid))
                bo.buffer_store(
                    fx.Vector.from_elements([h0, h1], fx.Float32).to(fx.BFloat16), rsrc(scratch),
                    MID // 2 + tid * INTER + bid * 2, cache_modifier=CM_DEV,
                )
            _release()
            gpu.barrier()
            if tid == 0:
                box["mb"].put(box["gf"], bid, fx.Int32(1))

            # ---- RQ: requantize MID group (token, group) = (b // 16, b % 16)
            if bid < MGROUPS * tokens:
                t = bid // MGROUPS
                grp = bid % MGROUPS
                if wave == 0:
                    box["mb"].poll([(box["gf"], grp * 16 + i, 1) for i in range(16)])
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

            # ---- DOWN: rows [64 b, 64 b + 64), wave w eight, a lane a dword of K
            box["mb"].poll([(box["rf"], t * MGROUPS + g, 1) for t in range(tokens) for g in range(MGROUPS)])
            md = []
            for t in range_constexpr(tokens):
                mw = fx.Int32(bo.buffer_load(rsrc(scratch), (MQ + t * (INTER // 2)) // 4 + lane, vec_width=1,
                                             dtype=T.i32, cache_modifier=CM_DEV))
                ms = pow2(fx.Int32(bo.buffer_load(rsrc(scratch), (MS + (t * MGROUPS + lane // 4) * 4) // 4,
                                                  vec_width=1, dtype=T.i32, cache_modifier=CM_DEV)))
                md.append(decode_word(mw, ms))
            for pr in range_constexpr(4):
                sums = []
                for half in range_constexpr(2):
                    row = bid * 64 + wave * 8 + pr * 2 + half
                    wv = fx.Int32(bo.buffer_load(rsrc(w2), row * (INTER // 8) + lane, vec_width=1, dtype=T.i32))
                    wd = decode_word(wv, pow2(_byte(s2, row * MGROUPS + lane // 4)))
                    per_t = []
                    for t in range_constexpr(tokens):
                        s = fx.Float32(0.0)
                        for j in range_constexpr(8):
                            s = s + wd[j] * md[t][j]
                        per_t.append(butterfly(s, (32, 16, 8, 4, 2, 1)))
                    sums.append(per_t)
                if lane == 0:
                    for t in range_constexpr(tokens):
                        bo.buffer_store(
                            fx.Vector.from_elements([sums[0][t], sums[1][t]], fx.Float32).to(fx.BFloat16),
                            rsrc(out), t * HIDDEN + bid * 64 + wave * 8 + pr * 2,
                        )

    @flyc.jit
    def launch(
        x: Int64, w13: Int64, s13: Int64, w2: Int64, s2: Int64, out: Int64,
        scratch: Int64, epoch: Int64, layer: Int32, stream: fx.Stream = fx.Stream(None),
    ):
        _ = keyed
        k(x, w13, s13, w2, s2, out, scratch, epoch, layer).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)

    return launch
