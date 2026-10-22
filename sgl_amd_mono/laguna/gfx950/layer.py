# SPDX-License-Identifier: Apache-2.0
"""Laguna's whole FFN as one launch per layer on gfx950 (TP8 EP1, decode rows M <= 16):

    S1   seam phase 0: the attention all-reduce of the o_proj partials, + residual,
         post-attention RMSNorm -> H (bf16) and the mid residual       (CTA b: columns 64 b..)
    sync every CTA's H columns stored (``xf`` flags: the router and Q read whole rows)
    S2   TP-sharded router: logits of the 64 local experts, exchanged, top-16 on every rank;
    S4   the shared expert's Q and GU run while the logits cross the GPUs, its GR / RQ
         before the top-k wait (S4 needs no routing; its stages wait on their own flags)
    sync the top-k of token t stored by CTA t (``rt`` flags)
    S3   routed experts -> MOE, then S4's DOWN -> SH (bf16; CTA b: columns 64 b.., the
         seam's own columns, so no grid sync)
    S5   seam phase 1: MOE + SH added in bf16 (as SGLang adds them), the MoE all-reduce,
         + residual, the next layer's input norm (or the final norm) -> x_out, res_out

Hand-offs read by other CTAs are stored and loaded with CM_DEV (the XCDs' L2s are not
coherent within a launch; a launch boundary is what made the standalone launches safe).
The stages' code is the standalone launches' (``emit_seam``, ``emit_router``,
``emit_shared``, ``emit_routed``), so each is checked on its own and the launch only adds
the two syncs.
One mailbox tag a launch: ``step * LAYER_SLOTS + layer + 1``.
"""

import os

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu
from flydsl.expr.typing import Int32, Int64, T

from sgl_amd_mono.common import guard
from sgl_amd_mono.common.ops import CM_DEV, clock, rsrc, uniform
from sgl_amd_mono.common.plan import BLOCKS, THREADS, WAVES, kernel_sources
from sgl_amd_mono.common.sync import Mailbox, sreg
from sgl_amd_mono.laguna.gfx950 import mxq, routed, router_tp, seam, shared

HIDDEN = seam.HIDDEN
MAX_TOKENS = seam.MAX_TOKENS
TOPK = routed.TOPK
LAYER_SLOTS = seam.LAYER_SLOTS
assert routed.LAYER_SLOTS == router_tp.LAYER_SLOTS == LAYER_SLOTS
assert seam.COLS == 64  # S3's DOWN CTA b writes columns [64 b, 64 b + 64): the seam's


def peer_layout(npes: int):
    """(LG offset, guard flags offset, bytes) of the peer buffer: seam regions, router logits, flags."""
    lg = seam.PHASES * seam.region_bytes(npes)
    flags = lg + router_tp.LG_BYTES
    return lg, flags, flags + guard.FLAG_BYTES


def peer_bytes(npes: int) -> int:
    return peer_layout(npes)[2]


# scratch: S3's (offsets inside it are routed's own), the seam's, the router's, the
# two sync flag arrays, then the stage hand-offs H / RES_MID / MOE / IDS / WTS
SS = routed.SCRATCH_BYTES
PQ = SS + seam.SS_BYTES
XF = PQ + router_tp.PQ_BYTES
RT = XF + BLOCKS * 8
H = RT + MAX_TOKENS * 8
RES_MID = H + MAX_TOKENS * HIDDEN * 2
MOE = RES_MID + MAX_TOKENS * HIDDEN * 2
IDS = MOE + MAX_TOKENS * HIDDEN * 2
WTS = IDS + MAX_TOKENS * TOPK * 4
SHS = WTS + MAX_TOKENS * TOPK * 4  # S4's scratch
SH = SHS + shared.SCRATCH_BYTES  # S4's output
PROF = SH + MAX_TOKENS * HIDDEN * 2
# S3's Q (x quantized for the routed experts) right after the router's logit push, on CTAs tokens ..
# 2 tokens - 1, instead of after S3's union: its flags are visible by UG (critical path: the U -> UG gap
# was ~6-7 us a layer at every M, ~5 us of it the qf hand-off)
Q_EARLY = os.environ.get("LAGUNA_Q_EARLY", "1") == "1"  # TP8 M1 82.4 -> 76.8 us/layer, TP4 116.7 -> 110.9
MARKS = 16  # prof build: per CTA, s_memrealtime at the stage boundaries (7-12: inside S3)
SCRATCH_BYTES = PROF + BLOCKS * MARKS * 8


def build_layer(npes: int, tokens: int, eps: float, prof: bool = False):
    """The FFN launch for ``tokens`` rows at TP ``npes``; ``eps`` of both norms. ``prof``: CTA b's
    thread 0 stores the clock at the stage boundaries to PROF[b][0..6] (a timing aid)."""
    assert WAVES % npes == 0 and 1 <= tokens <= MAX_TOKENS
    W = npes
    LG, FLAGS, _ = peer_layout(W)
    U = routed.U_MAX
    keyed = (
        W, tokens, eps, prof, routed.INTER, Q_EARLY,
        kernel_sources(__name__), kernel_sources(seam.__name__), kernel_sources(router_tp.__name__),
        kernel_sources(routed.__name__), kernel_sources(shared.__name__), kernel_sources(mxq.__name__), kernel_sources(guard.__name__),
    )
    name = f"laguna_ffn_tp{W}_t{tokens}" + ("_prof" if prof else "")

    @fx.struct
    class Smem:
        red: fx.Array[fx.Float32, WAVES, 16]
        fsum: fx.Array[fx.Float32, MAX_TOKENS * seam.COLS, 16]
        rpart: fx.Array[fx.Float32, WAVES * MAX_TOKENS, 16]
        uexp: fx.Array[fx.Int32, U, 16]
        umask: fx.Array[fx.Int32, U, 16]
        uw: fx.Array[fx.Float32, U * MAX_TOKENS, 16]
        sids: fx.Array[fx.Int32, MAX_TOKENS * TOPK, 16]
        swts: fx.Array[fx.Float32, MAX_TOKENS * TOPK, 16]
        wc: fx.Array[fx.Int32, WAVES + 1, 16]
        part: fx.Array[fx.Float32, 2 * WAVES * 32 * MAX_TOKENS, 16]
        gu: fx.Array[fx.Float32, 4 * MAX_TOKENS, 16]
        spart: fx.Array[fx.Float32, WAVES * 32 * MAX_TOKENS, 16]

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def k(
        partial: Int64, residual: Int64, norm1: Int64, rweight: Int64, rbias: Int64,
        w13: Int64, s13: Int64, w2: Int64, s2: Int64, sw13: Int64, ss13: Int64, sw2: Int64, ss2: Int64,
        norm2: Int64, x_out: Int64, res_out: Int64,
        scratch: Int64, sym: Int64, peers: Int64, epoch: Int64, rank: Int32, layer: Int32,
    ):
        _ = keyed
        with guard.scope(guard.Guard(epoch, own=sym, peers=peers, rank=rank, npes=W, flags_offset=FLAGS)):
            tid = fx.thread_idx.x
            bid = fx.block_idx.x
            lane = tid % 64
            wave = tid // 64
            lds = fx.SharedAllocator().allocate(Smem).peek()
            step = uniform(bo.buffer_load(rsrc(epoch), 0, vec_width=1, dtype=T.i32))
            # held in dicts: traced branches carry IR values only
            box = {
                "mb": Mailbox(step * LAYER_SLOTS + layer + 1),
                "xf": sreg(scratch, XF, "xf"),
                "rt": sreg(scratch, RT, "rt"),
            }
            ctx = {"tid": tid, "bid": bid, "lane": lane, "wave": wave, "box": box,
                   "red": lds.red.ptr, "fsum": lds.fsum.ptr, "rpart": lds.rpart.ptr}
            for n in ("uexp", "umask", "uw", "sids", "swts", "wc", "part", "gu", "spart"):
                ctx[n] = getattr(lds, n).ptr
            h = scratch + fx.Int64(H)
            res_mid = scratch + fx.Int64(RES_MID)
            moe = scratch + fx.Int64(MOE)
            ids = scratch + fx.Int64(IDS)
            wts = scratch + fx.Int64(WTS)
            sh = scratch + fx.Int64(SH)
            shs = scratch + fx.Int64(SHS)

            def mark(i):
                if const_expr(prof):
                    if tid == 0:
                        now = clock()
                        bo.buffer_store(fx.Vector.from_elements([fx.Int32(now), fx.Int32(now >> 32)], fx.Int32),
                                        rsrc(scratch), (PROF + (bid * MARKS + i) * 8) // 4)

            ctx["mark"] = mark
            mark(0)

            # S1
            seam.emit_seam(ctx, {
                "partial": partial, "residual": residual, "norm_w": norm1, "x_out": h, "res_out": res_mid,
                "scratch": scratch + fx.Int64(SS), "sym": sym, "peers": peers, "rank": rank,
            }, W, tokens, eps, 0, out_cm=CM_DEV)
            mark(1)
            routed._release()
            gpu.barrier()
            if tid == 0:
                box["mb"].put(box["xf"], bid, fx.Int32(1))
            if tid < BLOCKS:
                box["mb"].poll([(box["xf"], tid, 1)])
            gpu.barrier()
            mark(2)

            # S2's logits and their exchange; S4's front half (it needs no routing) while the
            # logits cross the GPUs; S2's top-k; S4's MID; S3; S4's DOWN
            sh_args = {"x": h, "w13": sw13, "s13": ss13, "w2": sw2, "s2": ss2, "out": sh, "scratch": shs}
            r_args = {
                "x": h, "weight": rweight, "bias": rbias, "ids_out": ids, "wts_out": wts,
                "scratch": scratch + fx.Int64(PQ), "sym": sym, "peers": peers, "rank": rank, "lg_off": LG,
            }
            shared.emit_shared(ctx, sh_args, tokens, ("Q",), in_cm=CM_DEV)
            router_tp.emit_router(ctx, r_args, W, tokens, ("part", "push"), in_cm=CM_DEV)
            if const_expr(Q_EARLY):
                routed.emit_routed_q(ctx, {"x": h, "scratch": scratch}, tokens, False, in_cm=CM_DEV, cta0=tokens)
            shared.emit_shared(ctx, sh_args, tokens, ("GU",))
            mark(3)
            router_tp.emit_router(ctx, r_args, W, tokens, ("topk",), out_cm=CM_DEV)
            routed._release()
            if (bid < tokens) & (tid == 0):
                box["mb"].put(box["rt"], bid, fx.Int32(1))
            shared.emit_shared(ctx, sh_args, tokens, ("GR", "RQ"))
            if tid < tokens:
                box["mb"].poll([(box["rt"], tid, 1)])
            gpu.barrier()
            mark(4)

            routed.emit_routed(ctx, {
                "x": h, "ids": ids, "wts": wts, "w13": w13, "s13": s13, "w2": w2, "s2": s2, "out": moe,
                "scratch": scratch, "skip": fx.Int32(0), "sw13": sw13, "ss13": ss13, "sw2": sw2, "ss2": ss2,
            }, tokens, len(routed.STAGES) - 1, False, in_cm=CM_DEV, q_early=Q_EARLY)
            shared.emit_shared(ctx, sh_args, tokens, ("DOWN",))
            routed._release()
            gpu.barrier()
            mark(5)

            # S5: the CTA's own columns of S3's and S4's outputs, added in bf16 as SGLang does
            seam.emit_seam(ctx, {
                "partial": moe, "partial2": sh, "residual": res_mid, "norm_w": norm2, "x_out": x_out,
                "res_out": res_out, "scratch": scratch + fx.Int64(SS), "sym": sym, "peers": peers, "rank": rank,
            }, W, tokens, eps, 1, in_cm=CM_DEV)
            mark(6)

    @flyc.jit
    def launch(
        partial: Int64, residual: Int64, norm1: Int64, rweight: Int64, rbias: Int64,
        w13: Int64, s13: Int64, w2: Int64, s2: Int64, sw13: Int64, ss13: Int64, sw2: Int64, ss2: Int64,
        norm2: Int64, x_out: Int64, res_out: Int64,
        scratch: Int64, sym: Int64, peers: Int64, epoch: Int64, rank: Int32, layer: Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        _ = keyed
        k(partial, residual, norm1, rweight, rbias, w13, s13, w2, s2, sw13, ss13, sw2, ss2, norm2, x_out, res_out,
          scratch, sym, peers, epoch, rank, layer).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)

    return launch
