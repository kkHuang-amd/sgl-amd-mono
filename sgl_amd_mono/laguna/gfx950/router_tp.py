# SPDX-License-Identifier: Apache-2.0
"""S2, TP-sharded: the router of Laguna's FFN launch on gfx950 with the fp32 router
weight split over the TP ranks (rank r: experts [64 r, 64 r + 64)), so a rank reads
4 MiB a layer instead of 32 MiB; the logits are exchanged through the peer buffer and
every rank runs the same top-16 (sigmoid + correction bias, lowest-id ties, weights
renormalized with sum + 1e-20, as SGLang's ``_router_triton_kernel``).

One launch, BLOCKS CTAs x THREADS threads:

    part  : CTA b = (local expert b // 4, K quarter b % 4); wave w takes 512 of the quarter's
            4096 elements (a lane 8: two fp32 x4 weight loads, one bf16 x8 input load a
            token); lanes and waves reduced in order -> the ``pq`` scratch mailbox
    push  : the quarter-0 CTA of each local expert sums the four quarters in order and
            pushes the logit to every rank's ``lg`` region [token][expert] (wave w -> rank w)
    topk  : CTA t < tokens, wave 0: the token's 512 logits, 16 rounds of a wave max
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.typing import Int32, Int64, T

from sgl_amd_mono.common import guard
from sgl_amd_mono.common.ops import bf2_f32, butterfly, hw_exp2, hw_rcp, load_ptr64, rsrc, uniform
from sgl_amd_mono.common.plan import BLOCKS, THREADS, WAVES, kernel_sources
from sgl_amd_mono.common.sync import Mailbox, preg, sreg

HIDDEN = 16384
EXPERTS = 512
TOPK = 16
MAX_TOKENS = 16
LAYER_SLOTS = 64
QUARTERS = 4
KQ = HIDDEN // QUARTERS  # 4096 a CTA
KW = KQ // WAVES  # 512 a wave
PER_LANE = EXPERTS // 64
LOG2E = 1.4426950408889634


STAGES = ("part", "push", "topk")


def local_experts(npes: int) -> int:
    return EXPERTS // npes


# peer buffer: the logits [token][expert] as tagged words, then the guard flags
LG_BYTES = MAX_TOKENS * EXPERTS * 8


def peer_bytes() -> int:
    return LG_BYTES + guard.FLAG_BYTES


# scratch: the quarter partials [token][local expert][quarter] as tagged words
PQ_BYTES = MAX_TOKENS * EXPERTS * QUARTERS * 8


def _max(a, b):
    return (a > b).select(a, b)


def _min_i(a, b):
    return (a < b).select(a, b)


@flyc.jit
def emit_router(ctx, args, npes, tokens, stages=STAGES, in_cm=0, out_cm=0):
    """The router's code inside a kernel: ``ctx`` = thread ids, the mailbox dict ``box`` (its
    ``mb``) and the LDS ``rpart`` (WAVES x MAX_TOKENS fp32); ``args`` = the launch's pointers
    (``scratch``: the router's ``PQ_BYTES``, ``sym`` / ``peers``: this rank's / every rank's peer
    buffer, its ``LG_BYTES`` region at byte ``lg_off``, default 0). ``stages``: a subset of STAGES (each waits on its inputs). ``in_cm`` / ``out_cm``: the cache policy of the x loads / the top-k stores."""
    W = npes
    EL = local_experts(W)
    tid, bid, lane, wave, box, part = (ctx[n] for n in ("tid", "bid", "lane", "wave", "box", "rpart"))
    x, weight, bias, ids_out, wts_out = (args[n] for n in ("x", "weight", "bias", "ids_out", "wts_out"))
    peers, rank = args["peers"], args["rank"]
    box["pq"] = sreg(args["scratch"], 0, "pq")
    lg_off = args.get("lg_off", 0)
    box["lg"] = preg(args["sym"], lg_off, "lg")

    el = bid // QUARTERS
    q = bid % QUARTERS

    if const_expr("part" in stages):
        # part: local expert el, quarter q
        kk = q * KQ + wave * KW + lane * 8
        w0 = fx.Vector(bo.buffer_load(rsrc(weight), el * HIDDEN + kk, vec_width=4, dtype=T.f32))
        w1 = fx.Vector(bo.buffer_load(rsrc(weight), el * HIDDEN + kk + 4, vec_width=4, dtype=T.f32))
        wv = [fx.Float32(w0[i]) for i in range(4)] + [fx.Float32(w1[i]) for i in range(4)]
        for t in range_constexpr(tokens):
            xv = fx.Vector(bo.buffer_load(rsrc(x), (t * HIDDEN + kk) // 2, vec_width=4, dtype=T.i32, cache_modifier=in_cm))
            a = fx.Float32(0.0)
            for i in range_constexpr(4):
                lo, hi = bf2_f32(fx.Int32(xv[i]))
                a = a + lo * wv[2 * i] + hi * wv[2 * i + 1]
            a = butterfly(a, (32, 16, 8, 4, 2, 1))
            if lane == 0:
                fx.ptr_store(a, part + wave * MAX_TOKENS + t)
        gpu.barrier()
        if tid < tokens:
            tot = fx.ptr_load(part + tid)
            for w in range_constexpr(1, WAVES):
                tot = tot + fx.ptr_load(part + w * MAX_TOKENS + tid)
            box["mb"].put(box["pq"], (tid * EL + el) * QUARTERS + q, tot)

    if const_expr("push" in stages):
        # push: quarter-0 CTAs; wave = peer (the peer table is read wave-uniform), lane = token
        assert WAVES == W
        pt = lane % tokens
        box["dst"] = preg(load_ptr64(peers, wave), lg_off, "lg")
        if q == 0:
            if lane < tokens:
                got = box["mb"].poll([(box["pq"], (pt * EL + el) * QUARTERS + qq, 1) for qq in range(QUARTERS)])
                lgt = got[0][0].bitcast(fx.Float32)
                for qq in range_constexpr(1, QUARTERS):
                    lgt = lgt + got[qq][0].bitcast(fx.Float32)
                box["mb"].put(box["dst"], pt * EXPERTS + rank * EL + el, lgt)

    if const_expr("topk" in stages):
        # topk: token bid, wave 0
        if (bid < tokens) & (wave == 0):
            li = fx.Int32(lane)
            got2 = box["mb"].poll([(box["lg"], bid * EXPERTS + lane + 64 * j, 1) for j in range(PER_LANE)])
            act = []
            key = []
            for j in range_constexpr(PER_LANE):
                lg = got2[j][0].bitcast(fx.Float32)
                av = hw_rcp(fx.Float32(1.0) + hw_exp2(-lg * fx.Float32(LOG2E)))
                bv = fx.Float32(bo.buffer_load(rsrc(bias), lane + 64 * j, vec_width=1, dtype=T.f32))
                act.append(av)
                key.append(av + bv)
            neg = fx.Float32(-3.0e38)
            tsum = fx.Float32(0.0)
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
                tsum = tsum + wa
            inv = fx.Float32(1.0) / (tsum + fx.Float32(1e-20))
            if lane == 0:
                for r in range_constexpr(TOPK):
                    bo.buffer_store(fx.Int32(pick_id[r]), rsrc(ids_out), bid * TOPK + r, cache_modifier=out_cm)
                    bo.buffer_store(fx.Float32(pick_w[r] * inv), rsrc(wts_out), bid * TOPK + r, cache_modifier=out_cm)


def build_router_tp(npes: int, tokens: int):
    EL = local_experts(npes)
    assert EL * QUARTERS == BLOCKS, "TP8: 64 local experts x 4 quarters = one CTA each"
    assert 1 <= tokens <= MAX_TOKENS
    W = npes
    keyed = (W, tokens, kernel_sources(__name__), kernel_sources(guard.__name__))
    name = f"laguna_router_tp{W}_t{tokens}"

    @fx.struct
    class Smem:
        part: fx.Array[fx.Float32, WAVES * MAX_TOKENS, 16]

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def k(
        x: Int64, weight: Int64, bias: Int64, ids_out: Int64, wts_out: Int64,
        scratch: Int64, sym: Int64, peers: Int64, epoch: Int64, rank: Int32, layer: Int32,
    ):
        _ = keyed
        with guard.scope(guard.Guard(epoch, own=sym, peers=peers, rank=rank, npes=W, flags_offset=LG_BYTES)):
            tid = fx.thread_idx.x
            bid = fx.block_idx.x
            lane = tid % 64
            wave = tid // 64
            lds = fx.SharedAllocator().allocate(Smem).peek()
            part = lds.part.ptr
            step = uniform(bo.buffer_load(rsrc(epoch), 0, vec_width=1, dtype=T.i32))
            box = {"mb": Mailbox(step * LAYER_SLOTS + layer + 1)}
            c = {"tid": tid, "bid": bid, "lane": lane, "wave": wave, "box": box, "rpart": part}
            a = {
                "x": x, "weight": weight, "bias": bias, "ids_out": ids_out, "wts_out": wts_out,
                "scratch": scratch, "sym": sym, "peers": peers, "rank": rank,
            }
            emit_router(c, a, W, tokens)

    @flyc.jit
    def launch(
        x: Int64, weight: Int64, bias: Int64, ids_out: Int64, wts_out: Int64,
        scratch: Int64, sym: Int64, peers: Int64, epoch: Int64, rank: Int32, layer: Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        _ = keyed
        k(x, weight, bias, ids_out, wts_out, scratch, sym, peers, epoch, rank, layer).launch(
            grid=(BLOCKS,), block=(THREADS,), stream=stream
        )

    return launch
