# SPDX-License-Identifier: Apache-2.0
"""An index source's indexer query in K1, bit for bit SGLang's decode chain.

It replaces the four launches of ``spec._index_query``: qr's MXFP8 quant and
aiter's group-32 GEMM with the index wq_b (``iq``, on the bytes K1 already quantized
for its own wq_b, the same as ``mxfp8_operand``'s), the head weights (``iw``:
``rocm_router_gemv_split_k``) and ``_index_q_pack_weights_kernel`` (``iquant``:
RoPE, the FP4 fake-quant, the FP4 query in the MQA-logits layout). The indexer
is replicated: every rank computes every head."""

import flydsl.compiler as flyc
import flydsl.expr as fx
import numpy as np
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu
from flydsl.expr import math as fmath
from flydsl.expr import range_constexpr
from flydsl.expr.typing import T

from sgl_amd_mono.common.ops import (
    CM_DEV,
    bf16_round,
    butterfly,
    div_rn,
    mfma_bf16,
    row_sum,
    rsrc,
)
from sgl_amd_mono.common.plan import first_task
from sgl_amd_mono.deepseek_v41.gfx950.stages.gemv import ld_f32, plus

from .device import BLOCKS, WAVES, ld_i32
from .gemv import _step_load, _step_mfma
from .plan import HIDDEN, Q_RANK, ROPE, ROWS

HEADS = 32
DIM = 128
IQ_ROWS = HEADS * DIM  # 4096
IQ_TASKS = IQ_ROWS // ROWS  # 256: 16 output columns a task
IQ_STEPS = Q_RANK // 128  # 10
IQ_BATCH = 5  # a chain's weight steps in flight together (registers)
IW_SPLIT = 512
IW_SPLITS = HIDDEN // IW_SPLIT  # 10
IW_STEPS = IW_SPLIT // 32  # 16 MFMAs a partial
IW_CHAINS = IW_SPLITS * (HEADS // 16)  # (split, 16-head tile)
FP4_AMAX_FLOOR = 6 * 2.0**-126
INV6 = float(np.float32(1.0 / 6.0))  # Triton's fp32 constant
E2M1_EDGES = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
QS_ROW = 4 * 16 * 4  # a token's q_scale bytes: [1, 4, 16, 4]


def iq_chains(s: int):
    """Each index wq_b output's K chains at s rows, as aiter's group-32 GEMM runs
    them (its gfx950 config for N=4096, K=1280): 128-K steps in chain order (None:
    a step past K, which adds zero), the chains then summed in order. None for
    s <= 4: K_PACK=4 sums its four panels across a lane quad, not reproduced."""
    if s <= 4:
        return None
    z = None
    if s <= 8:  # packed, K_PACK 2, 256-K panels
        return [[0, 1, 4, 5, 8, 9], [2, 3, 6, 7, z, z]]
    if s <= 16:  # packed, K_PACK 2, 512-K panels
        return [[0, 1, 2, 3, 8, 9, z, z], [4, 5, 6, 7, z, z, z, z]]
    return [list(range(IQ_STEPS))]  # K_PACK 1 / unpacked: one chain


def fused(s: int, tok_par: bool = False) -> bool:
    """Whether an s-row K1 computes its index query (else SGLang's chain does)."""
    return iq_chains(s) is not None


# ---------------------------------------------------------------- iw


def _iw_chains(c, x, w, t0, n, splits, ht, lds):
    """Head-weight partials, a chain each, their steps interleaved (one load
    trip for all): rows t0 .. t0 + n of x [T, HIDDEN] bf16 against heads 16 ht ..
    of w [HEADS, HIDDEN], K 512 split .. in 32-K MFMA steps -> ``lds``
    [split][16 rows][16 heads] f32."""
    lane = c["lane"]
    row = lane % 16
    k8 = 8 * (lane // 16)
    live = row < n
    t = plus(t0, fx.min(row, n - 1))
    accs = [fx.Vector.filled(4, 0.0, fx.Float32) for _ in splits]
    for st in range_constexpr(IW_STEPS):
        for j in range_constexpr(len(splits)):
            k = splits[j] * IW_SPLIT + st * 32 + k8
            xv = fx.Vector(
                bo.buffer_load(
                    rsrc(x),
                    (t * HIDDEN + k) // 2,
                    vec_width=4,
                    dtype=T.i32,
                    cache_modifier=CM_DEV,
                )
            )
            xv = live.select(xv, fx.Vector.filled(4, 0, fx.Int32))
            wv = fx.Vector(
                bo.buffer_load(
                    rsrc(w),
                    ((16 * ht + row) * HIDDEN + k) // 2,
                    vec_width=4,
                    dtype=T.i32,
                )
            )
            accs[j] = mfma_bf16(
                xv.bitcast(fx.BFloat16), wv.bitcast(fx.BFloat16), accs[j]
            )
    for j in range_constexpr(len(splits)):
        # D[row 4 (lane / 16) + q][head lane % 16]
        for q in range_constexpr(4):
            r = 4 * (lane // 16) + q
            at = (splits[j] * 16 + r) * 16 + lane % 16
            fx.ptr_store(accs[j][q].bitcast(fx.Int32), lds + at)


@flyc.jit
def stage_iw(c, x, w, out, scale, t0, n, ht, lds, push=None):
    """Head tile ht's head weights of rows t0 .. t0 + n -> ``out`` [T, HEADS]
    bf16: its IW_SPLITS partials, split s on wave s % WAVES (waves 0 .. take two,
    interleaved), summed in split order; ``lds`` holds them [split][16 rows][16
    heads] f32. ``push(row, head, w0, w1)`` (token parallel) hands a head pair on
    to the other ranks."""
    tid, wave = c["tid"], c["wave"]
    twice = IW_SPLITS - WAVES  # the waves with a second split
    if wave < twice:
        _iw_chains(c, x, w, t0, n, [wave, wave + WAVES], ht, lds)
    else:
        _iw_chains(c, x, w, t0, n, [wave], ht, lds)
    gpu.barrier()
    if tid < n * 8:
        r, h = tid // 8, 2 * (tid % 8)
        ws = []
        for d in range_constexpr(2):
            acc = fx.ptr_load(lds + (r * 16 + h + d)).bitcast(fx.Float32)
            for sp in range_constexpr(1, IW_SPLITS):
                at = (sp * 16 + r) * 16 + h + d
                acc = acc + fx.ptr_load(lds + at).bitcast(fx.Float32)
            ws.append(bf16_round(bf16_round(acc) * scale))
        word = (
            fx.Vector.from_elements(ws, fx.Float32).to(fx.BFloat16).bitcast(fx.Int32)[0]
        )
        row = plus(t0, r)
        bo.buffer_store(word, rsrc(out), (row * HEADS + 16 * ht + h) // 2)
        if const_expr(push is not None):
            push(row, 16 * ht + h, ws[0], ws[1])
    gpu.barrier()


# ---------------------------------------------------------------- iq


def iq_weights(c, w, ws, task, chains):
    """This wave's chain's weight operands, a batch of IQ_BATCH steps at a time
    (waves past the chains read through a zero-size buffer), in the layout of K1's
    other weights (``shuffled``)."""
    wave = c["wave"]
    nb = (wave < len(chains)).select(fx.Int64(0xFFFFFFFF), fx.Int64(0))
    r_w, r_ws = rsrc(w, nb), rsrc(ws, nb)

    def step(i):
        sts = [ch[i] for ch in chains]
        if const_expr(len(chains) == 1):
            return fx.Int32(sts[0] if sts[0] is not None else 0)
        return (wave == 0).select(fx.Int32(sts[0] or 0), fx.Int32(sts[1] or 0))

    def batch(b0):
        return [
            _step_load(
                c, r_w, r_ws, Q_RANK, task * ROWS, step(i), shuffled=c["shuffled"]
            )
            for i in range(b0, min(b0 + IQ_BATCH, len(chains[0])))
        ]

    return batch


@flyc.jit
def stage_iq(c, task, chains, batch, xl, xsl, iq, t0, n):
    """Index wq_b rows 16 task .. for tokens t0 .. t0 + n (MXFP8 rows in LDS) ->
    bf16 IQ [T][IQ_ROWS], the chains on waves 0 .. len(chains)."""
    tid, lane, wave, red = c["tid"], c["lane"], c["wave"], c["red"]
    nch, steps = len(chains), len(chains[0])
    acc = fx.Vector.filled(4, 0.0, fx.Float32)
    for b0 in range_constexpr(0, steps, IQ_BATCH):
        ops = batch(b0)
        for i in range_constexpr(b0, min(b0 + IQ_BATCH, steps)):
            sts = [ch[i] for ch in chains]
            if const_expr(all(st is None for st in sts)):
                continue
            # a masked step adds zero products (the GEMM's out-of-range K)
            live = None
            if const_expr(any(st is None for st in sts)):
                live = (wave == 0).select(
                    fx.Int32(int(sts[0] is not None)), fx.Int32(int(sts[1] is not None))
                ) == 1
            st = (wave == 0).select(fx.Int32(sts[0] or 0), fx.Int32(sts[-1] or 0))
            acc = _step_mfma(c, IQ_STEPS, xl, xsl, st, ops[i - b0], acc, live, n)
    fx.ptr_store(acc, red + (wave * 64 + lane) * 4)
    gpu.barrier()
    if tid < ROWS * n:
        r, tl = tid % ROWS, tid // ROWS
        v = row_sum(red, r, tl, list(range(nch)))
        bo.buffer_store(
            v.to(fx.BFloat16),
            rsrc(iq),
            plus(t0, tl) * IQ_ROWS + task * ROWS + r,
            cache_modifier=CM_DEV,
        )
    gpu.barrier()


# ---------------------------------------------------------------- iquant


def _e2m1_code(x):
    """quantize_fp4_indexer_row's RNE E2M1 code of x (already at its group scale)."""
    ax = fx.min(abs(x), fx.Float32(6.0))
    idx = fx.Int32(0)
    edge = ax < fx.Float32(-1.0)  # false
    for e in E2M1_EDGES:
        idx = idx + (ax >= fx.Float32(e)).select(fx.Int32(1), fx.Int32(0))
        edge = edge | (ax == fx.Float32(e))
    idx = (edge & ((idx & 1) == 1)).select(idx - 1, idx)
    sign = (x < fx.Float32(0.0)) & (idx != 0)
    return idx | sign.select(fx.Int32(8), fx.Int32(0))


def _pow2_bits(e):
    """2^(e - 127) for a biased exponent e in [1, 254]."""
    return (e << 23).bitcast(fx.Float32)


@flyc.jit
def stage_iquant(c, a, iq, t, h, num_pos):
    """Token t, head h on this wave: lane l takes elements 2l, 2l + 1."""
    lane = c["lane"]
    w = fx.Int32(
        bo.buffer_load(
            rsrc(iq), (t * IQ_ROWS + h * DIM) // 2 + lane, vec_width=1, dtype=T.i32,
            cache_modifier=CM_DEV,
        )
    )  # fmt: skip
    v0 = (w << 16).bitcast(fx.Float32)
    v1 = (w & fx.Int32(-65536)).bitcast(fx.Float32)
    # RoPE (GPT-J pairs) on dims 64 ..: an out-of-table position reads entry 0
    pos = ld_i32(a["pos"], 2 * t)
    pos = ((pos >= 0) & (pos < num_pos)).select(pos, fx.Int32(0))
    j = fx.max(lane - DIM // 4, 0)
    cs = ld_f32(a["cos_sin"], pos * ROPE + j)
    sn = ld_f32(a["cos_sin"], pos * ROPE + ROPE // 2 + j)
    tail = lane >= DIM // 4
    # separate products and sums, as the Triton kernel compiles
    re = bf16_round(v0 * cs - v1 * sn)
    im = bf16_round(v0 * sn + v1 * cs)
    v0 = tail.select(re, v0)
    v1 = tail.select(im, v1)
    # rope_tail_fake_quant_fp4_row: a 32-group = 16 lanes
    amax = butterfly(fx.max(abs(v0), abs(v1)), (1, 2, 4, 8), fx.max)
    a6 = fx.max(amax, fx.Float32(FP4_AMAX_FLOOR)) * fx.Float32(INV6)
    bits = a6.bitcast(fx.Int32)
    e = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).select(
        fx.Int32(1), fx.Int32(0)
    )
    scale, inv = _pow2_bits(e), _pow2_bits(254 - e)

    def fake(v):
        s = fx.min(fx.max(v * inv, fx.Float32(-6.0)), fx.Float32(6.0))
        mag = abs(s)
        step = (mag < 2.0).select(
            fx.Float32(0.5), (mag < 4.0).select(fx.Float32(1.0), fx.Float32(2.0))
        )
        istep = (mag < 2.0).select(
            fx.Float32(2.0), (mag < 4.0).select(fx.Float32(1.0), fx.Float32(0.5))
        )
        sgn = (s > 0.0).select(
            fx.Float32(1.0), (s < 0.0).select(fx.Float32(-1.0), fx.Float32(0.0))
        )
        q = fx.Float32(fmath.roundeven(mag * istep)) * step * sgn
        return bf16_round(q * scale)

    v0, v1 = fake(v0), fake(v1)
    # quantize_fp4_indexer_row (RNE)
    amax = butterfly(fx.max(abs(v0), abs(v1)), (1, 2, 4, 8), fx.max)
    sf = fx.max(div_rn(amax, fx.Float32(6.0), fx.Float32(INV6)), fx.Float32(1.0e-4))
    sbits = sf.bitcast(fx.Int32)
    se = ((sbits >> 23) & 0xFF) + ((sbits & 0x7FFFFF) != 0).select(
        fx.Int32(1), fx.Int32(0)
    )
    se = fx.min(fx.max(se, fx.Int32(1)), fx.Int32(254))
    sinv = _pow2_bits(254 - se)
    byte = _e2m1_code(v0 * sinv) | (_e2m1_code(v1 * sinv) << 4)
    bo.buffer_store(
        fx.Int8(byte), rsrc(a["iq_out"]), (t * HEADS + h) * (DIM // 2) + lane
    )
    # the scale byte of 32-group g at [t, 0, g, h % 16, h // 16]
    if lane % 16 == 0:
        g = lane // 16
        at = t * QS_ROW + g * 64 + (h % 16) * 4
        bo.buffer_store(fx.Int8(se), rsrc(a["iqs_out"]), at + h // 16)
        if h < 16:
            # the head groups this head count lacks stay zero
            for hg in range_constexpr(HEADS // 16, 4):
                bo.buffer_store(fx.Int8(0), rsrc(a["iqs_out"]), at + hg)


# ---------------------------------------------------------------- drivers


@flyc.jit
def run_iw(c, a, rows, iw0, lds, row0=0, push=None):
    """The head weights of rows row0 .. row0 + rows (the step's, or token
    parallel this rank's), a (token tile, 16-head tile) a CTA from CTA ``iw0``,
    once its rows' norms published them (``xrdy``: two flags a row, at its local
    index)."""
    tiles = -(-rows // 16)
    for task in range(first_task(c["bid"], iw0), 2 * tiles, BLOCKS):
        t0 = (task // 2) * 16
        n = fx.min(fx.Int32(16), rows - t0)
        if c["tid"] < 2 * n:
            c["mb"].poll([(c["xrdy"], 2 * t0 + fx.min(c["tid"], 2 * n - 1), 1)])
        gpu.barrier()
        stage_iw(
            c, a["normed_out"], a["iw_w"], a["iw_out"], a["iw_scale"], plus(row0, t0),
            n, task % 2, lds, push,
        )  # fmt: skip


def iw_scratch(s: int) -> dict:
    """The split head weights' partials ``run_iw_split``: [pair][split][16][16] f32,
    then a flag a task."""
    pairs = 2 * -(-s // 16)
    part = pairs * IW_SPLITS * 256 * 4
    return {"iwp": (0, part), "iwr": (part, pairs * IW_SPLITS * 8)}


@flyc.jit
def run_iw_split(c, a, rows, base, row0=0):
    """``run_iw``'s partials a (token tile, 16-head tile, split) a CTA from CTA ``base``,
    wave 0 running the split's chain as ``stage_iw`` does, to IWP / IWR."""
    tid, lds = c["tid"], c["xl"]
    tiles = -(-rows // 16)
    for task in range(first_task(c["bid"], base), 2 * tiles * IW_SPLITS, BLOCKS):
        pair, sp = task // IW_SPLITS, task % IW_SPLITS
        t0 = (pair // 2) * 16
        n = fx.min(fx.Int32(16), rows - t0)
        gpu.barrier()
        if tid < 2 * n:
            c["mb"].poll([(c["xrdy"], 2 * t0 + fx.min(tid, 2 * n - 1), 1)])
        gpu.barrier()
        if c["wave"] == 0:
            _iw_chains(
                c, a["normed_out"], a["iw_w"], plus(row0, t0), n, [sp], pair % 2, lds
            )
        gpu.barrier()
        if tid < 256:
            v = fx.ptr_load(lds + (sp * 256 + tid)).bitcast(fx.Float32)
            bo.buffer_store(v, rsrc(c["iwp"]), task * 256 + tid, cache_modifier=CM_DEV)
        from .front import publish

        publish(c, c["iwr"], task)


@flyc.jit
def run_iw_reduce(c, a, rows, base, n_ctas, row0=0, push=None):
    """``stage_iw``'s sum of ``run_iw_split``'s partials in split order, a (token tile,
    16-head tile) a CTA on the n_ctas CTAs from ``base``: bit for bit ``run_iw``."""
    tid = c["tid"]
    tiles = -(-rows // 16)
    first = first_task(c["bid"], base)
    if first < n_ctas:
        for pair in range(first, 2 * tiles, n_ctas):
            t0 = (pair // 2) * 16
            n = fx.min(fx.Int32(16), rows - t0)
            ht = pair % 2
            if tid < IW_SPLITS:
                c["mb"].poll(
                    [(c["iwr"], pair * IW_SPLITS + fx.min(tid, IW_SPLITS - 1), 1)]
                )
            gpu.barrier()
            if tid < n * 8:
                r, h = tid // 8, 2 * (tid % 8)
                ws = []
                for d in range_constexpr(2):
                    acc = None
                    for sp in range_constexpr(IW_SPLITS):
                        v = fx.Float32(
                            bo.buffer_load(
                                rsrc(c["iwp"]),
                                (pair * IW_SPLITS + sp) * 256 + r * 16 + h + d,
                                vec_width=1,
                                dtype=T.f32,
                                cache_modifier=CM_DEV,
                            )
                        )
                        acc = v if acc is None else acc + v
                    ws.append(bf16_round(bf16_round(acc) * a["iw_scale"]))
                word = (
                    fx.Vector.from_elements(ws, fx.Float32)
                    .to(fx.BFloat16)
                    .bitcast(fx.Int32)[0]
                )
                row = plus(row0, plus(t0, r))
                bo.buffer_store(
                    word, rsrc(a["iw_out"]), (row * HEADS + 16 * ht + h) // 2
                )
                if const_expr(push is not None):
                    push(row, 16 * ht + h, ws[0], ws[1])
            gpu.barrier()


@flyc.jit
def run_iquant(c, a, s):
    """Every (token, head), a wave each, once its head's 8 IQ tasks published."""
    lane, wave = c["lane"], c["wave"]
    for k in range(first_task(c["bid"], 0), -(-s * HEADS // WAVES), BLOCKS):
        u = k * WAVES + wave
        if u < s * HEADS:
            t, h = u // HEADS, u % HEADS
            if lane < DIM // ROWS:
                c["mb"].poll(
                    [(c["iqrdy"], (t // 16) * IQ_TASKS + h * (DIM // ROWS) + lane, 1)]
                )
            stage_iquant(c, a, c["iq"], t, h, a["num_pos"])


@flyc.jit
def iw_from(c, iwg, t):
    """Row t, another rank's: its head weights from IWG (bf16 pairs) -> ``iw_out``."""
    tid = c["tid"]
    if tid < HEADS // 2:
        (w,) = c["mb"].poll([(iwg, t * (HEADS // 2) + tid, 1)])
        bo.buffer_store(w[0], rsrc(c["args"]["iw_out"]), t * (HEADS // 2) + tid)
