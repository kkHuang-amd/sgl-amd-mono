# SPDX-License-Identifier: Apache-2.0
"""S1, the seam of Laguna's FFN launch on gfx950: the attention all-reduce of the
o_proj partials, the residual add and the post-attention RMSNorm.

One launch, BLOCKS CTAs x THREADS threads; CTA t owns hidden columns
[t * COLS, (t + 1) * COLS) of every token:

    push : wave w copies the CTA's columns of this rank's bf16 partial rows into
           peer w's ``attn`` region at [this rank][token] (to itself too)
    sum  : every rank's partials of those columns, summed in fp32 in rank order,
           rounded to bf16 (the all-reduce's output), plus the residual -> the new
           residual (bf16, stored), and the columns' sum of squares per token,
           published to the ``ss`` scratch mailbox [token][CTA]
    norm : the token's 256 partial sums of squares, summed -> rstd; the columns'
           RMSNorm output (bf16, stored)

Mailboxes are tagged ``step * LAYER_SLOTS + layer + 1`` (word 0 of the epoch
buffer is the step, moved on by the host after a step's last launch), so nothing
is cleared between steps and a captured graph replays correctly.
"""


import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.typing import Int32, Int64, T

from sgl_amd_mono.common import guard
from sgl_amd_mono.common.ops import (
    CM_DEV,
    bf2_f32,
    bf16_round,
    butterfly,
    hw_rsq,
    load_ptr64,
    rsrc,
    sum_partials,
    uniform,
)
from sgl_amd_mono.common.plan import BLOCKS, THREADS, WAVES, kernel_sources
from sgl_amd_mono.common.sync import Mailbox, preg, sreg

HIDDEN = 16384
MAX_TOKENS = 16
LAYER_SLOTS = 64
COLS = HIDDEN // BLOCKS  # 64 hidden columns a CTA
PAIRS = COLS // 2  # bf16 pairs a token a CTA (32)
assert COLS * BLOCKS == HIDDEN and PAIRS == 32

# peer buffer: two regions of partials [src rank][token][HIDDEN] as tagged bf16 pairs
# (two words a pair) -- phase 0 the attention output (S1), phase 1 the MoE output
# (S5) -- then the guard flags. The phases alternate regions, so a rank that runs
# ahead into the next launch never overwrites partials a slower rank still reads:
# it can only start a phase after every rank pushed the other one.
PHASES = 2

# aiter's opus fused add + RMSNorm (SGLang's, hidden > 8192) normalizes its register-
# cached first 8192 columns from the fp32 sum and reloads the rest as the rounded
# (stored) residual; the sum of squares is over the fp32 sum. One rounding at the end.
NORM_FP32_COLS = 8192


def region_bytes(npes: int) -> int:
    return npes * MAX_TOKENS * HIDDEN * 4


def peer_bytes(npes: int) -> int:
    return PHASES * region_bytes(npes) + guard.FLAG_BYTES


# scratch: the per-CTA sums of squares [phase][token][CTA] as tagged words
SS_BYTES = PHASES * MAX_TOKENS * BLOCKS * 8


def tag_slot(layer: int, phase: int) -> int:
    """The ``layer`` argument of a launch: one tag slot per (layer, phase)."""
    return PHASES * layer + phase


@flyc.jit
def emit_seam(ctx, args, npes, tokens, eps, phase, in_cm=0, out_cm=0):
    """The seam's code inside a kernel: ``ctx`` = thread ids, the mailbox dict ``box`` (its
    ``mb``) and the LDS ``red`` (WAVES fp32); ``args`` = the launch's pointers (``scratch``: the
    seam's ``SS_BYTES``, ``sym``: this rank's seam peer regions; optional ``partial2``: a second
    partial, added to ``partial`` before the push). ``in_cm``: the cache policy of
    the partial / residual loads (CM_DEV when this launch wrote them); ``out_cm``: of the x /
    residual stores (CM_DEV when this launch reads them on other CTAs: the XCDs' L2s are not coherent)."""
    W = npes
    REG = phase * region_bytes(W)
    SS = phase * MAX_TOKENS * BLOCKS * 8
    tid, bid, lane, wave, box, red, fsum = (ctx[n] for n in ("tid", "bid", "lane", "wave", "box", "red", "fsum"))
    partial, residual, norm_w, x_out, res_out = (args[n] for n in ("partial", "residual", "norm_w", "x_out", "res_out"))
    peers, rank = args["peers"], args["rank"]
    partial2 = args.get("partial2")
    box["own"] = preg(args["sym"], REG, "attn")
    box["ss"] = sreg(args["scratch"], SS, "ss")
    c0 = bid * COLS

    # push: wave w -> peer w, a bf16 pair a lane
    dst = load_ptr64(peers, fx.Int32(wave) % W)
    box["dst"] = preg(dst, REG, "attn")
    for ci in range_constexpr((PAIRS * tokens + 63) // 64):
        i = lane + 64 * ci
        if (wave < W) & (i < PAIRS * tokens):
            tk = i // PAIRS
            col = c0 + (i % PAIRS) * 2
            w = fx.Int32(
                bo.buffer_load(
                    rsrc(partial), (tk * HIDDEN + col) // 2, vec_width=1, dtype=T.i32,
                    cache_modifier=in_cm,
                )
            )
            v0, v1 = bf2_f32(w)
            if const_expr(partial2 is not None):
                # a second bf16 partial (S4 next to S3): added and rounded to bf16 once, as a bf16 add
                w2 = fx.Int32(
                    bo.buffer_load(
                        rsrc(partial2), (tk * HIDDEN + col) // 2, vec_width=1, dtype=T.i32,
                        cache_modifier=in_cm,
                    )
                )
                u0, u1 = bf2_f32(w2)
                v0 = v0 + u0
                v1 = v1 + u1
            box["mb"].put_bf(
                box["dst"],
                (rank * MAX_TOKENS + tk) * HIDDEN + col,
                [v0, v1],
            )

    # sum: every rank's partials in rank order, + residual; sum of squares
    sq = fx.Float32(0.0)
    if tid < PAIRS * tokens:
        tk = tid // PAIRS
        col = c0 + (tid % PAIRS) * 2
        s0, s1 = sum_partials(
            box["mb"].poll,
            box["own"],
            lambda src: ((src * MAX_TOKENS + tk) * HIDDEN + col) // 2,
            W,
        )
        rw = fx.Int32(
            bo.buffer_load(
                rsrc(residual), (tk * HIDDEN + col) // 2, vec_width=1, dtype=T.i32,
                cache_modifier=in_cm,
            )
        )
        r0, r1 = bf2_f32(rw)
        # SGLang's fused add + RMSNorm (aiter opus): the residual is the rounded sum, the
        # sum of squares is over the fp32 sum, the norm reads it (see NORM_FP32_COLS)
        f0 = bf16_round(s0) + r0
        f1 = bf16_round(s1) + r1
        n0 = bf16_round(f0)
        n1 = bf16_round(f1)
        fx.ptr_store(f0, fsum + tk * COLS + (tid % PAIRS) * 2)
        fx.ptr_store(f1, fsum + tk * COLS + (tid % PAIRS) * 2 + 1)
        bo.buffer_store(
            fx.Vector.from_elements([n0, n1], fx.Float32).to(fx.BFloat16),
            rsrc(res_out),
            tk * HIDDEN + col,
            cache_modifier=out_cm,
        )
        sq = f0 * f0 + f1 * f1
    # the token's 32 threads are one half-wave: reduce within it
    sq = butterfly(sq, (16, 8, 4, 2, 1))
    if (tid < PAIRS * tokens) & (tid % PAIRS == 0):
        box["mb"].put(box["ss"], (tid // PAIRS) * BLOCKS + bid, sq)

    # norm: each token's BLOCKS partial sums of squares -> rstd
    for tk in range_constexpr(tokens):
        part = fx.Float32(0.0)
        if tid < BLOCKS:
            got = box["mb"].poll([(box["ss"], tk * BLOCKS + tid, 1)])
            part = got[0][0].bitcast(fx.Float32)
        part = butterfly(part, (32, 16, 8, 4, 2, 1))
        gpu.barrier()
        if lane == 0:
            fx.ptr_store(part, red + wave)
        gpu.barrier()
        tot = fx.ptr_load(red + 0)
        for w in range_constexpr(1, BLOCKS // 64):
            tot = tot + fx.ptr_load(red + w)
        rstd = hw_rsq(tot / fx.Float32(float(HIDDEN)) + fx.Float32(eps))
        if tid < PAIRS:
            col = c0 + tid * 2
            nw = fx.Int32(bo.buffer_load(rsrc(norm_w), col // 2, vec_width=1, dtype=T.i32))
            w0, w1 = bf2_f32(nw)
            n0 = fx.ptr_load(fsum + tk * COLS + tid * 2)
            n1 = fx.ptr_load(fsum + tk * COLS + tid * 2 + 1)
            n0 = (c0 >= NORM_FP32_COLS).select(bf16_round(n0), n0)
            n1 = (c0 >= NORM_FP32_COLS).select(bf16_round(n1), n1)
            bo.buffer_store(
                fx.Vector.from_elements([n0 * rstd * w0, n1 * rstd * w1], fx.Float32).to(fx.BFloat16),
                rsrc(x_out),
                tk * HIDDEN + col,
                cache_modifier=out_cm,
            )



def build_seam(npes: int, tokens: int, eps: float, phase: int = 0):
    """The seam launch for ``tokens`` rows at TP ``npes``: phase 0 = S1 (attention
    all-reduce + residual + post-attention norm), phase 1 = S5 (MoE all-reduce +
    residual + the next layer's input norm, or the final norm). ``layer`` at launch
    is ``tag_slot(layer, phase)``."""
    assert 1 <= tokens <= MAX_TOKENS and 1 <= npes <= WAVES and phase in (0, 1)
    W = npes
    keyed = (W, tokens, eps, phase, kernel_sources(__name__), kernel_sources(guard.__name__))
    name = f"laguna_seam_tp{W}_t{tokens}_p{phase}"

    @fx.struct
    class Smem:
        red: fx.Array[fx.Float32, WAVES, 16]
        fsum: fx.Array[fx.Float32, MAX_TOKENS * COLS, 16]

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def k(
        partial: Int64,
        residual: Int64,
        norm_w: Int64,
        x_out: Int64,
        res_out: Int64,
        scratch: Int64,
        sym: Int64,
        peers: Int64,
        epoch: Int64,
        rank: Int32,
        layer: Int32,
    ):
        _ = keyed
        with guard.scope(
            guard.Guard(
                epoch, own=sym, peers=peers, rank=rank, npes=W,
                flags_offset=PHASES * region_bytes(W),
            )
        ):
            tid = fx.thread_idx.x
            bid = fx.block_idx.x
            lane = tid % 64
            wave = tid // 64
            lds = fx.SharedAllocator().allocate(Smem).peek()
            red = lds.red.ptr
            step = uniform(bo.buffer_load(rsrc(epoch), 0, vec_width=1, dtype=T.i32))
            # held in a dict: traced branches carry IR values only
            box = {"mb": Mailbox(step * LAYER_SLOTS + layer + 1)}
            c = {"tid": tid, "bid": bid, "lane": lane, "wave": wave, "box": box, "red": red, "fsum": lds.fsum.ptr}
            a = {
                "partial": partial, "residual": residual, "norm_w": norm_w, "x_out": x_out,
                "res_out": res_out, "scratch": scratch, "sym": sym, "peers": peers, "rank": rank,
            }
            emit_seam(c, a, W, tokens, eps, phase)

    @flyc.jit
    def launch(
        partial: Int64,
        residual: Int64,
        norm_w: Int64,
        x_out: Int64,
        res_out: Int64,
        scratch: Int64,
        sym: Int64,
        peers: Int64,
        epoch: Int64,
        rank: Int32,
        layer: Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        _ = keyed
        k(partial, residual, norm_w, x_out, res_out, scratch, sym, peers, epoch, rank, layer).launch(
            grid=(BLOCKS,), block=(THREADS,), stream=stream
        )

    return launch
