# SPDX-License-Identifier: Apache-2.0
"""S2, the router of Laguna's FFN launch on gfx950: fp32 router GEMV over the
post-attention-norm output, sigmoid scores, top-16 by score + correction bias
(ties to the lowest expert id), weights = the chosen sigmoid scores renormalized
(sum + 1e-20), as SGLang's ``_router_triton_kernel`` (sigmoid, renormalize).

One launch, BLOCKS CTAs x THREADS threads:

    logit : CTA b owns experts 2b, 2b + 1; wave w takes expert 2b + w // 4 and the
            K quarter w % 4 (4096 hidden elements, fp32 weight rows, bf16 inputs
            widened to fp32); lanes reduce, the quarters are summed in order and
            the logits published to the ``lg`` scratch mailbox [token][expert]
    topk  : CTA t < tokens, wave 0: 8 experts a lane; 16 rounds of a wave max,
            lowest id among the maxima; the weights renormalized in pick order
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import gpu, range_constexpr
from flydsl.expr.typing import Int32, Int64, T

from sgl_amd_mono.common import guard
from sgl_amd_mono.common.ops import bf2_f32, butterfly, hw_exp2, hw_rcp, rsrc, uniform
from sgl_amd_mono.common.plan import BLOCKS, THREADS, WAVES, kernel_sources
from sgl_amd_mono.common.sync import Mailbox, sreg

HIDDEN = 16384
EXPERTS = 512
TOPK = 16
MAX_TOKENS = 16
LAYER_SLOTS = 64
QUARTERS = 4
KQ = HIDDEN // QUARTERS  # 4096 hidden elements a wave
STEPS = KQ // 256  # 16 rounds of 64 lanes x 4 elements
PER_LANE = EXPERTS // 64  # 8 experts a lane in the top-k wave
LOG2E = 1.4426950408889634
assert 2 * BLOCKS == EXPERTS and WAVES == 2 * QUARTERS

# scratch: the logits [token][expert] as tagged words
LG_BYTES = MAX_TOKENS * EXPERTS * 8


def _max(a, b):
    return (a > b).select(a, b)


def _min_i(a, b):
    return (a < b).select(a, b)


def build_router(tokens: int):
    assert 1 <= tokens <= MAX_TOKENS
    keyed = (tokens, kernel_sources(__name__), kernel_sources(guard.__name__))
    name = f"laguna_router_t{tokens}"

    @fx.struct
    class Smem:
        part: fx.Array[fx.Float32, WAVES * MAX_TOKENS, 16]

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def k(
        x: Int64,
        weight: Int64,
        bias: Int64,
        ids_out: Int64,
        wts_out: Int64,
        scratch: Int64,
        epoch: Int64,
        layer: Int32,
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
            box = {"mb": Mailbox(step * LAYER_SLOTS + layer + 1), "lg": sreg(scratch, 0, "lg")}

            # logit: expert e, K quarter q of this wave
            e = bid * 2 + wave // QUARTERS
            q = wave % QUARTERS
            acc = [fx.Float32(0.0) for _ in range(tokens)]
            for s in range_constexpr(STEPS):
                kk = q * KQ + s * 256 + lane * 4
                wv = fx.Vector(
                    bo.buffer_load(rsrc(weight), e * HIDDEN + kk, vec_width=4, dtype=T.f32)
                )
                for t in range_constexpr(tokens):
                    xv = fx.Vector(
                        bo.buffer_load(rsrc(x), (t * HIDDEN + kk) // 2, vec_width=2, dtype=T.i32)
                    )
                    a0, a1 = bf2_f32(fx.Int32(xv[0]))
                    a2, a3 = bf2_f32(fx.Int32(xv[1]))
                    acc[t] = (
                        acc[t]
                        + a0 * fx.Float32(wv[0])
                        + a1 * fx.Float32(wv[1])
                        + a2 * fx.Float32(wv[2])
                        + a3 * fx.Float32(wv[3])
                    )
            for t in range_constexpr(tokens):
                r = butterfly(acc[t], (32, 16, 8, 4, 2, 1))
                if lane == 0:
                    fx.ptr_store(r, part + wave * MAX_TOKENS + t)
            gpu.barrier()
            if tid < 2 * tokens:
                el = tid // tokens
                t = tid % tokens
                tot = fx.ptr_load(part + (el * QUARTERS) * MAX_TOKENS + t)
                for qq in range_constexpr(1, QUARTERS):
                    tot = tot + fx.ptr_load(part + (el * QUARTERS + qq) * MAX_TOKENS + t)
                box["mb"].put(box["lg"], t * EXPERTS + bid * 2 + el, tot)

            # topk: token bid, wave 0
            if (bid < tokens) & (wave == 0):
                li = fx.Int32(lane)
                got = box["mb"].poll(
                    [(box["lg"], bid * EXPERTS + lane + 64 * j, 1) for j in range(PER_LANE)]
                )
                act = []
                key = []
                for j in range_constexpr(PER_LANE):
                    lg = got[j][0].bitcast(fx.Float32)
                    a = hw_rcp(fx.Float32(1.0) + hw_exp2(-lg * fx.Float32(LOG2E)))
                    b = fx.Float32(
                        bo.buffer_load(rsrc(bias), lane + 64 * j, vec_width=1, dtype=T.f32)
                    )
                    act.append(a)
                    key.append(a + b)
                neg = fx.Float32(-3.0e38)
                tot = fx.Float32(0.0)
                pick_id = []
                pick_w = []
                for r in range_constexpr(TOPK):
                    m = key[0]
                    for j in range_constexpr(1, PER_LANE):
                        m = _max(m, key[j])
                    m = butterfly(m, (32, 16, 8, 4, 2, 1), _max)
                    cand = fx.Int32(1 << 30)
                    for j in range_constexpr(PER_LANE):
                        cand = _min_i(cand, (key[j] == m).select(li + 64 * j, fx.Int32(1 << 30)))
                    win = butterfly(cand, (32, 16, 8, 4, 2, 1), _min_i)
                    wa = neg
                    for j in range_constexpr(PER_LANE):
                        hit = (li + 64 * j) == win
                        wa = _max(wa, hit.select(act[j], neg))
                        key[j] = hit.select(neg, key[j])
                    wa = butterfly(wa, (32, 16, 8, 4, 2, 1), _max)
                    pick_id.append(win)
                    pick_w.append(wa)
                    tot = tot + wa
                inv = fx.Float32(1.0) / (tot + fx.Float32(1e-20))
                if lane == 0:
                    for r in range_constexpr(TOPK):
                        bo.buffer_store(fx.Int32(pick_id[r]), rsrc(ids_out), bid * TOPK + r)
                        bo.buffer_store(fx.Float32(pick_w[r] * inv), rsrc(wts_out), bid * TOPK + r)

    @flyc.jit
    def launch(
        x: Int64, weight: Int64, bias: Int64, ids_out: Int64, wts_out: Int64,
        scratch: Int64, epoch: Int64, layer: Int32, stream: fx.Stream = fx.Stream(None),
    ):
        _ = keyed
        k(x, weight, bias, ids_out, wts_out, scratch, epoch, layer).launch(
            grid=(BLOCKS,), block=(THREADS,), stream=stream
        )

    return launch
