# SPDX-License-Identifier: Apache-2.0
"""A KV source's ratio-2 compressor in K1: SGLang's ``_low_ratio_compress_fused`` on a
decode or static-verify step, which otherwise runs after K1 (``spec._select``).

- ``gemm``: ``project_fused``, K1's normed input against ``wkv_gate`` [1024, 5120] bf16
  (kv, then score) in f32: 64 column tiles x 3 K splits, a task a CTA in q's wait
  (``layer.seam_gate``), each its partials to CPART and a flag to CPRDY.
- ``finish``: a row a CTA past the qkv CTAs, once every partial landed:
  ``flash_c2_decode_kernel`` (the pair's other half from the ring, or from the block's
  previous row; the two-way softmax pool, RMSNorm, RoPE of dims 448.. at pos - 1, the
  FP4 fake quant, the V4 record at ``raw_out_loc >> 1``; an even position parks its
  pair in the ring instead) and, where the layer owns its index keys, ``wk``,
  ``k_norm``, RoPE and the FP4 index key of ``flash_index_k_split_kernel`` at
  ``c2_out_loc`` (> 0 only on a completing row).

Close to SGLang's, not bit for bit: the projection and ``wk`` sum in other orders than
hipBLASLt's GEMMs."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu
from flydsl.expr import math as fmath
from flydsl.expr import range_constexpr
from flydsl.expr.typing import T

from sgl_amd_mono.common.epoch import gstore
from sgl_amd_mono.common.mx import clamp_fp8
from sgl_amd_mono.common.ops import (
    CM_DEV,
    bf16_round,
    butterfly,
    fp8_pack4,
    fp8_roundtrip,
    hw_rsq,
    mfma_bf16,
    rsrc,
    xshfl,
)
from sgl_amd_mono.common.plan import first_task
from sgl_amd_mono.common.stamps import stamp
from sgl_amd_mono.deepseek_v41.gfx950.stages.gemv import ld_f32, plus

from .device import BLOCKS, THREADS, WAVES, ceil_exp, div_rn, fma, ld_i32, mx_mul
from .front import publish, record_scales
from .index_query import FP4_AMAX_FLOOR, INV6, _e2m1_code, _pow2_bits
from .plan import DATA, HIDDEN, NOPE

DIM = 512  # the compressed latent
COLS = 2 * DIM  # the projection: kv, then score
TILES = COLS // 16
KSPLITS = 3
GEMM_TASKS = TILES * KSPLITS  # 192
STEPS = HIDDEN // 32  # 32-K MFMA steps
WAVE_STEPS = -(
    -(-(-STEPS // KSPLITS)) // WAVES
)  # a wave's steps of a split, the last masked
ROPE_D = DIM - NOPE  # 64
IDX = 128  # an index key
FT = DIM // 2  # the finish threads, two dims each (flash_c2_decode_kernel's CTA)
NOPE_T = NOPE // 2
PLANE_PAGE = 64  # index plane rows a page
# token parallel: the peer regions' words a row (``token_par.peer_layout``)
RING_WORDS, REC_WORDS, KEY_WORDS = COLS, DATA // 4 + 2, 34
LAT = 0  # LDS words of the finish: the latent, then the index key
KEY = DIM


def comp_scratch(s: int) -> dict:
    """CPART [KSPLITS][s][COLS] f32 partials, then CPRDY: a flag a GEMM task."""
    part = KSPLITS * s * COLS * 4
    return {"cpart": (0, part), "cprdy": (part, GEMM_TASKS * 8)}


@flyc.jit
def _push(c, region, i, words):
    """Token parallel: ``words`` at pairs i.. of every other rank's ``region``."""
    if const_expr(c.get("tok_par", False)):
        for p in range_constexpr(c["d"].tp):
            if c["rank"] != p:
                c["mb"].put_words(c["comp_dsts"][region][p], i, words)


def _split(task):
    """A GEMM task's column tile, K split and the split's [lo, hi) steps."""
    ks = task // TILES
    return task % TILES, ks, ks * STEPS // KSPLITS, (ks + 1) * STEPS // KSPLITS


def _k(c, lo, j):
    return fx.min(lo + j * WAVES + c["wave"], STEPS - 1) * 32 + 8 * (c["lane"] // 16)


def _x_loads(c, a, lo, rows, t0, nb=0xFFFFFFFF):
    """Every step's normed rows (16-row tiles) of this lane, a list a step."""
    out = []
    for j in range_constexpr(WAVE_STEPS):
        k = _k(c, lo, j)
        out.append(
            [
                fx.Vector(
                    bo.buffer_load(
                        rsrc(a["normed_out"], nb),
                        (
                            plus(t0, fx.min(rt * 16 + c["lane"] % 16, rows - 1))
                            * HIDDEN
                            + k
                        )
                        // 2,
                        vec_width=4,
                        dtype=T.i32,
                        cache_modifier=CM_DEV,
                    )
                )
                for rt in range(-(-rows // 16))
            ]
        )
    return out


def _task(c, base):
    task = first_task(c["bid"], base)
    nb = (task < GEMM_TASKS).select(fx.Int64(0xFFFFFFFF), fx.Int64(0))
    return _split(fx.min(task, GEMM_TASKS - 1)), nb


def gemm_rows(c, a, base, rows, t0):
    """This CTA's GEMM task's rows (none past the tasks: a zero-size buffer), once it
    saw every row's XRDY."""
    (_, _, lo, _), nb = _task(c, base)
    return _x_loads(c, a, lo, rows, t0, nb)


def gemm_weights(c, a, base):
    """This CTA's GEMM task's weights (none past the tasks), issued early:
    ``front.run_front(prefetch=)`` puts them behind the wqkv's weights."""
    lane = c["lane"]
    (ct, _, lo, _), nb = _task(c, base)
    return [
        fx.Vector(
            bo.buffer_load(
                rsrc(a["comp_w"], nb),
                ((16 * ct + lane % 16) * HIDDEN + _k(c, lo, j)) // 2,
                vec_width=4,
                dtype=T.i32,
            )
        )
        for j in range(WAVE_STEPS)
    ]


@flyc.jit
def stage_gemm(c, a, task, rows, t0, ops):
    """Column tile ``task % TILES`` of rows t0 .. t0 + rows over K split ``task // TILES``
    (``ops``: its weights and rows, ``gemm_operands``): the 8 waves' MFMA chains to
    LDS, summed in wave order to CPART, then CPRDY."""
    lane, wave, tid, lds = c["lane"], c["wave"], c["tid"], c["xl"]
    ct, ks, lo, hi = _split(task)
    row = lane % 16
    tiles = -(-rows // 16)
    wvs, xs = ops
    if const_expr(xs is None):
        xs = _x_loads(c, a, lo, rows, t0)
    zero = fx.Vector.filled(4, 0, fx.Int32)
    accs = [fx.Vector.filled(4, 0.0, fx.Float32) for _ in range(tiles)]
    for j in range_constexpr(WAVE_STEPS):
        live = (lo + j * WAVES + wave) < hi
        wv = live.select(wvs[j], zero)
        for rt in range_constexpr(tiles):
            r = rt * 16 + row
            xv = (live & (r < rows)).select(xs[j][rt], zero)
            accs[rt] = mfma_bf16(
                xv.bitcast(fx.BFloat16), wv.bitcast(fx.BFloat16), accs[rt]
            )
    stamp(c, 42)
    gpu.barrier()
    # D[row 4 (lane / 16) + q][column lane % 16]
    for rt in range_constexpr(tiles):
        for q in range_constexpr(4):
            at = ((wave * tiles + rt) * 16 + 4 * (lane // 16) + q) * 16 + lane % 16
            fx.ptr_store(accs[rt][q].bitcast(fx.Int32), lds + at)
    gpu.barrier()
    for e in range_constexpr(-(-tiles * 256 // THREADS)):
        u = tid + THREADS * e
        rt = u // 256
        r = rt * 16 + (u % 256) // 16
        col = u % 16
        if (u < tiles * 256) & (r < rows):
            acc = fx.ptr_load(lds + (rt * 256 + u % 256)).bitcast(fx.Float32)
            for w in range_constexpr(1, WAVES):
                at = (w * tiles + rt) * 256 + u % 256
                acc = acc + fx.ptr_load(lds + at).bitcast(fx.Float32)
            bo.buffer_store(
                acc,
                rsrc(c["cpart"]),
                (ks * c["S"] + plus(t0, r)) * COLS + 16 * ct + col,
                cache_modifier=CM_DEV,
            )
    publish(c, c["cprdy"], task)
    stamp(c, 43)


def _pair(c, row, col):
    """CPART's two columns col, col + 1 of step row ``row``, its K splits summed in order."""
    out = None
    for ks in range_constexpr(KSPLITS):
        v = fx.Vector(
            bo.buffer_load(
                rsrc(c["cpart"]),
                (ks * c["S"] + row) * COLS + col,
                vec_width=2,
                dtype=T.f32,
                cache_modifier=CM_DEV,
            )
        )
        out = [v[0], v[1]] if out is None else [out[0] + v[0], out[1] + v[1]]
    return out


def _ring_pair(a, rrow, col):
    v = fx.Vector(
        bo.buffer_load(
            rsrc(a["comp_ring"]), rrow * COLS + col, vec_width=2, dtype=T.f32
        )
    )
    return [v[0], v[1]]


def _fake_kv(v, scale, inv):
    """fake_quant_compressed_kv: RNE onto the e2m1 grid at an E4M3 scale."""
    s = div_rn(v, scale, inv)
    mag = fx.min(abs(s), fx.Float32(6.0))
    step = (mag < 2.0).select(
        fx.Float32(0.5), (mag < 4.0).select(fx.Float32(1.0), fx.Float32(2.0))
    )
    istep = (mag < 2.0).select(
        fx.Float32(2.0), (mag < 4.0).select(fx.Float32(1.0), fx.Float32(0.5))
    )
    q = fx.Float32(fmath.roundeven(mag * istep)) * step
    return (s < 0.0).select(-q, q) * scale


def _fake_fp4(v, scale, inv):
    """fp4::fake_quant_x2 at a power-of-two scale."""
    s = fx.min(fx.max(v * inv, fx.Float32(-6.0)), fx.Float32(6.0))
    mag = abs(s)
    step = (mag < 2.0).select(
        fx.Float32(0.5), (mag < 4.0).select(fx.Float32(1.0), fx.Float32(2.0))
    )
    istep = (mag < 2.0).select(
        fx.Float32(2.0), (mag < 4.0).select(fx.Float32(1.0), fx.Float32(0.5))
    )
    q = fx.Float32(fmath.roundeven(mag * istep)) * step
    return (s < 0.0).select(-q, q) * scale


@flyc.jit
def stage_index_key(c, a, L, wks):
    """``wk`` (this thread's operands ``wks``) on the latent in LDS, then
    flash_index_k_split_kernel's row: k_norm, RoPE of dims 64.. at pos & ~1, the FP4
    key and its exponents into the split plane (``L``: ``finish_loads``)."""
    iloc = L["iloc"]
    tid, lane, wave, lds = c["tid"], c["lane"], c["wave"], c["xl"]
    o, p = tid // 4, tid % 4
    acc = fx.Float32(0.0)
    v4 = fx.Vector.make_type(4, fx.Int32)
    for i in range_constexpr(DIM // 4 // 8):
        k = p * (DIM // 4) + i * 8
        wv = wks[i]
        xs = [fx.ptr_load(lds + (LAT + k + 4 * h), result_type=v4) for h in range(2)]
        for e in range_constexpr(4):
            w = wv[e]
            lo = (w << 16).bitcast(fx.Float32)
            hi = (w & fx.Int32(-65536)).bitcast(fx.Float32)
            x0 = xs[e // 2][2 * (e % 2)].bitcast(fx.Float32)
            x1 = xs[e // 2][2 * (e % 2) + 1].bitcast(fx.Float32)
            acc = fma(x1, hi, fma(x0, lo, acc))
    acc = butterfly(acc, (1, 2))
    if p == 0:
        fx.ptr_store(bf16_round(acc).bitcast(fx.Int32), lds + (KEY + o))
    gpu.barrier()
    if wave == 0:
        lane32 = lane % 32
        ld = lambda i: fx.ptr_load(lds + (KEY + i)).bitcast(fx.Float32)  # noqa: E731
        h0, h1 = ld(2 * lane32), ld(2 * lane32 + 1)
        q0, q1 = ld(IDX // 2 + 2 * lane32), ld(IDX // 2 + 2 * lane32 + 1)
        ssq = butterfly(h0 * h0 + h1 * h1 + q0 * q0 + q1 * q1, (1, 2, 4, 8, 16))
        inv = hw_rsq(ssq * fx.Float32(1.0 / IDX) + a["comp_keps"])
        wn = L["kn"]
        head = [bf16_round(wn[0] * (h0 * inv)), bf16_round(wn[1] * (h1 * inv))]
        tail = [bf16_round(wn[2] * (q0 * inv)), bf16_round(wn[3] * (q1 * inv))]
        fre, fim = L["ifre"], L["ifim"]
        data = head + [
            bf16_round(fma(tail[0], fre, -(tail[1] * fim))),
            bf16_round(fma(tail[0], fim, tail[1] * fre)),
        ]
        for h in range_constexpr(2):
            amax = butterfly(
                fx.max(abs(data[2 * h]), abs(data[2 * h + 1])), (1, 2, 4, 8), fx.max
            )
            e = ceil_exp(fx.max(amax, fx.Float32(FP4_AMAX_FLOOR)) * fx.Float32(INV6))
            scale, sinv = _pow2_bits(e), _pow2_bits(254 - e)
            data[2 * h] = _fake_fp4(data[2 * h], scale, sinv)
            data[2 * h + 1] = _fake_fp4(data[2 * h + 1], scale, sinv)
        bytes_, exps = [], []
        for h in range_constexpr(2):
            amax = butterfly(
                fx.max(abs(data[2 * h]), abs(data[2 * h + 1])), (1, 2, 4, 8), fx.max
            )
            sf = fx.max(
                div_rn(amax, fx.Float32(6.0), fx.Float32(INV6)), fx.Float32(1.0e-4)
            )
            se = fx.min(fx.max(ceil_exp(sf), fx.Int32(1)), fx.Int32(254))
            sinv = _pow2_bits(254 - se)
            bytes_.append(
                _e2m1_code(data[2 * h] * sinv)
                | (_e2m1_code(data[2 * h + 1] * sinv) << 4)
            )
            exps.append(se)
        if (lane < 32) & (iloc > 0):
            at = L["tg"] * KEY_WORDS
            _push(c, "cig", at + lane32, [bytes_[0] | (bytes_[1] << 8)])
            if lane32 % 16 == 0:
                _push(c, "cig", at + 32 + lane32 // 16, [exps[0] | (exps[1] << 8)])
            page, slot = iloc // PLANE_PAGE, iloc % PLANE_PAGE
            pay = page * (4 * PLANE_PAGE * 16) + slot * 16
            for h in range_constexpr(2):
                bo.buffer_store(
                    fx.Int8(bytes_[h]),
                    rsrc(a["comp_ipay"]),
                    pay + (2 * h + lane32 // 16) * (PLANE_PAGE * 16) + lane32 % 16,
                )
            if lane32 % 16 == 0:
                at = page * (4 * PLANE_PAGE) + (slot % 16) * 4 + slot // 16
                first = lane32 // 16
                for h in range_constexpr(2):
                    bo.buffer_store(
                        fx.Int8(exps[h]),
                        rsrc(a["comp_iscale"]),
                        at + (2 * h + first) * PLANE_PAGE,
                    )


def finish_loads(c, a, t, t0, per_req, owns_k):
    """Every load of row t's finish that waits on nothing (``stage_finish``), issued
    before its wait on the GEMM."""
    tid, lane = c["tid"], c["lane"]
    tg = plus(t0, t)
    tx = fx.min(tid, FT - 1)
    d0 = 2 * tx
    pos = ld_i32(a["pos"], 2 * tg)
    req = ld_i32(a["comp_req"], 2 * tg)
    L = dict(
        tg=tg, tx=tx, d0=d0, pos=pos, req=req,
        loc=ld_i32(a["comp_loc"], a["comp_loc_words"] * tg),
        nw=[c["ldbf"](a["comp_nw"], d0 + i) for i in range(2)],
    )  # fmt: skip
    ring = a["comp_ring_size"]
    wo = pos % ring
    rrow = req * ring + (wo == 0).select(ring - 1, wo - 1)
    L |= dict(wo=wo, ko=_ring_pair(a, rrow, d0), so=_ring_pair(a, rrow, DIM + d0))
    fb = (pos - 1) * ROPE_D + 2 * fx.max(tx - NOPE_T, 0)
    L |= dict(fre=ld_f32(a["comp_freqs"], fb), fim=ld_f32(a["comp_freqs"], fb + 1))
    if const_expr(owns_k):
        lane32 = lane % 32
        L["iloc"] = ld_i32(a["comp_iloc"], 2 * tg)
        L["kn"] = [
            c["ldbf"](a["comp_kn"], i)
            for i in (2 * lane32, 2 * lane32 + 1, IDX // 2 + 2 * lane32, IDX // 2 + 2 * lane32 + 1)
        ]  # fmt: skip
        ifb = (pos & fx.Int32(-2)) * ROPE_D + 2 * lane32
        L |= dict(
            ifre=ld_f32(a["comp_freqs"], ifb), ifim=ld_f32(a["comp_freqs"], ifb + 1)
        )
    return L


@flyc.jit
def stage_finish(c, a, t, t0, per_req, wks):
    """Row t (step row t0 + t) once every GEMM task published: flash_c2_decode_kernel's
    row (``flash_c2_decode_kernel<Verify>`` for per_req > 1), then its index key where
    ``wks`` (``wk_weights``) is given."""
    tid, lane, wave, red, lds = c["tid"], c["lane"], c["wave"], c["red"], c["xl"]
    L = finish_loads(c, a, t, t0, per_req, wks is not None)
    if tid < GEMM_TASKS:
        c["mb"].poll([(c["cprdy"], fx.min(tid, GEMM_TASKS - 1), 1)])
    gpu.barrier()
    stamp(c, 44)
    tg, tx, d0, pos, req, loc = (L[k] for k in ("tg", "tx", "d0", "pos", "req", "loc"))
    mine = tid < FT
    ring = a["comp_ring_size"]
    wo = L["wo"]
    kn, sn = _pair(c, tg, d0), _pair(c, tg, DIM + d0)
    ko, so = L["ko"], L["so"]
    if const_expr(per_req > 1):
        # a verify block: every row but the first pairs with the row before it
        inb = (fx.Int32(tg) % per_req) != 0
        prev = fx.max(fx.Int32(tg) - 1, 0)
        kp, sp = _pair(c, prev, d0), _pair(c, prev, DIM + d0)
        ko = [inb.select(kp[i], ko[i]) for i in range(2)]
        so = [inb.select(sp[i], so[i]) for i in range(2)]
    live = loc != 0
    odd = (pos & 1) == 1
    if live & ((pos & 1) == 0) & mine:
        wrow = (req * ring + wo) * COLS
        for col, v in ((d0, kn), (DIM + d0, sn)):
            bo.buffer_store(
                fx.Vector.from_elements(v, fx.Float32), rsrc(a["comp_ring"]), wrow + col
            )
            _push(c, "crg", tg * RING_WORDS + col, [x.bitcast(fx.Int32) for x in v])
    if live & odd:
        # the pool: with two scores exp(-|s0 - s1|) is the whole softmax
        stg = []
        for i in range_constexpr(2):
            delta = so[i] - sn[i]
            e = fx.Float32(fmath.exp(-abs(delta)))
            up = delta > 0.0
            s0 = up.select(fx.Float32(1.0), e)
            s1 = up.select(e, fx.Float32(1.0))
            stg.append(bf16_round((ko[i] * s0 + kn[i] * s1) / (fx.Float32(1.0) + e)))
        ssq = butterfly(stg[0] * stg[0] + stg[1] * stg[1], (32, 16, 8, 4, 2, 1))
        if (lane == 0) & mine:
            fx.ptr_store(ssq, red + wave)
        gpu.barrier()
        tot = ((fx.ptr_load(red + 0) + fx.ptr_load(red + 1)) + fx.ptr_load(red + 2)) + (
            fx.ptr_load(red + 3)
        )
        nf = hw_rsq(tot * fx.Float32(1.0 / DIM) + a["comp_eps"])
        out = [bf16_round((stg[i] * nf) * L["nw"][i]) for i in range(2)]
        if mine:
            for i in range_constexpr(2):
                fx.ptr_store(out[i].bitcast(fx.Int32), lds + (LAT + d0 + i))
        # RoPE of the tail at the position the latent stands for
        is_rope = tx >= NOPE_T
        fre, fim = L["fre"], L["fim"]
        re = bf16_round(out[0] * fre - out[1] * fim)
        im = bf16_round(out[0] * fim + out[1] * fre)
        v = [is_rope.select(re, out[0]), is_rope.select(im, out[1])]
        # FP4 fake quant over 16 elements (8 threads), its scale E4M3
        amax = butterfly(fx.max(abs(v[0]), abs(v[1])), (1, 2, 4), fx.max)
        raw = fx.min(
            fx.max(amax * fx.Float32(1.0 / 6.0), fx.Float32(2.0**-9)), fx.Float32(448.0)
        )
        scale = fp8_roundtrip(raw, raw)[0]
        sinv = fx.Float32(1.0) / scale
        v = [_fake_kv(v[i], scale, sinv) for i in range(2)]
        # the V4 record: e4m3 nope with a UE8M0 scale a 64-block (32 threads), bf16 rope
        amax = butterfly(fx.max(abs(v[0]), abs(v[1])), (1, 2, 4, 8, 16), fx.max)
        rcode = ceil_exp(
            div_rn(
                fx.max(amax, fx.Float32(1e-4)),
                fx.Float32(448.0),
                fx.Float32(1.0 / 448.0),
            )
        )
        inv = mx_mul(rcode)
        q0, q1 = clamp_fp8(v[0] * inv), clamp_fp8(v[1] * inv)
        word = fp8_pack4(q0, q1, xshfl(q0, 1), xshfl(q1, 1))
        rope_word = (
            fx.Vector.from_elements(v, fx.Float32).to(fx.BFloat16).bitcast(fx.Int32)[0]
        )
        out_loc = loc >> 1
        blk = a["comp_block"]
        base = a["comp_pool"] + fx.Int64(out_loc // blk) * fx.Int64(
            a["comp_pool_stride"]
        )
        off = out_loc % blk
        rec = base + fx.Int64(off * DATA)
        if mine & (tx % 32 == 0) & (tx < NOPE_T):
            fx.ptr_store(rcode.bitcast(fx.Float32), red + 8 + tx // 32)
        gpu.barrier()
        stamp(c, 45)
        if mine:
            at = tg * REC_WORDS
            if tx < NOPE_T:
                if tx % 2 == 0:
                    gstore(rec + fx.Int64(2 * tx), word, words=1)
                    _push(c, "cpg", at + tx // 2, [word])
            else:
                gstore(rec + fx.Int64(NOPE + 4 * (tx - NOPE_T)), rope_word, words=1)
                _push(c, "cpg", at + NOPE // 4 + (tx - NOPE_T), [rope_word])
            if tx == 0:
                sc = base + fx.Int64(blk * DATA + off * 8)
                words = record_scales(red)
                gstore(sc, fx.Vector.from_elements(words, fx.Int32), words=2)
                _push(c, "cpg", at + DATA // 4, words)
        if const_expr(wks is not None):
            gpu.barrier()
            stamp(c, 46)
            stage_index_key(c, a, L, wks)
    gpu.barrier()
    stamp(c, 47)


@flyc.jit
def run_gemm(c, a, own, t0, base, ops):
    """The GEMM task from CTA ``base`` (one a CTA at most), once this rank's rows'
    norms published (XRDY, unless ``ops`` holds the rows already)."""
    assert GEMM_TASKS <= BLOCKS
    task = first_task(c["bid"], base)
    if task < GEMM_TASKS:
        gpu.barrier()
        if const_expr(ops[1] is None):
            if c["tid"] < 2 * own:
                c["mb"].poll([(c["xrdy"], fx.min(c["tid"], 2 * own - 1), 1)])
            gpu.barrier()
        stamp(c, 41)
        stage_gemm(c, a, task, own, t0, ops)


def wk_weights(c, a, base, n):
    """``wk``'s operands of this thread (output tid // 4, K quarter tid % 4) on the
    finish CTAs (n from ``base``; none elsewhere), loaded ahead of the finish."""
    o, p = c["tid"] // 4, c["tid"] % 4
    nb = (first_task(c["bid"], base) < n).select(fx.Int64(0xFFFFFFFF), fx.Int64(0))
    return [
        fx.Vector(
            bo.buffer_load(
                rsrc(a["comp_wk"], nb),
                (o * DIM + p * (DIM // 4) + i * 8) // 2,
                vec_width=4,
                dtype=T.i32,
            )
        )
        for i in range(DIM // 4 // 8)
    ]


@flyc.jit
def run_finish(c, a, own, t0, base, n, per_req, wks):
    """This rank's rows' finish tasks on the n CTAs from ``base`` (``wks``: their
    ``wk`` operands, where the layer owns its index keys)."""
    ft = first_task(c["bid"], base)
    if ft < n:
        for t in range(ft, own, n):
            stage_finish(c, a, t, t0, per_req, wks)


@flyc.jit
def comp_from(c, a, t, owns_k):
    """Row t, another rank's (token parallel): its ring pair, V4 record and index key
    from CRG / CPG / CIG -> this rank's ring, pool and plane, as ``stage_finish``
    stores them (the row's metadata says which it pushed)."""
    tid, src = c["tid"], c["comp_srcs"]
    pos = ld_i32(a["pos"], 2 * t)
    loc = ld_i32(a["comp_loc"], a["comp_loc_words"] * t)
    live = loc != 0
    if live & ((pos & 1) == 0):
        ring = a["comp_ring_size"]
        rrow = ld_i32(a["comp_req"], 2 * t) * ring + pos % ring
        (w,) = c["mb"].poll([(src["crg"], t * RING_WORDS + 2 * tid, 2)])
        bo.buffer_store(
            fx.Vector.from_elements(w, fx.Int32),
            rsrc(a["comp_ring"]),
            rrow * COLS + 2 * tid,
        )
    if live & ((pos & 1) == 1):
        out_loc = loc >> 1
        blk = a["comp_block"]
        base = a["comp_pool"] + fx.Int64(out_loc // blk) * fx.Int64(
            a["comp_pool_stride"]
        )
        off = out_loc % blk
        if tid < REC_WORDS:
            (w,) = c["mb"].poll([(src["cpg"], t * REC_WORDS + tid, 1)])
            data = tid < DATA // 4
            at = data.select(
                off * DATA + 4 * tid, blk * DATA + off * 8 + 4 * (tid - DATA // 4)
            )
            gstore(base + fx.Int64(at), w[0], words=1)
        if const_expr(owns_k):
            iloc = ld_i32(a["comp_iloc"], 2 * t)
            if (iloc > 0) & (tid < KEY_WORDS):
                (w,) = c["mb"].poll([(src["cig"], t * KEY_WORDS + tid, 1)])
                page, slot = iloc // PLANE_PAGE, iloc % PLANE_PAGE
                lo, hi = w[0] & 0xFF, (w[0] >> 8) & 0xFF
                if tid < 32:
                    pay = page * (4 * PLANE_PAGE * 16) + slot * 16 + tid % 16
                    for h, b in ((0, lo), (1, hi)):
                        bo.buffer_store(
                            fx.Int8(b),
                            rsrc(a["comp_ipay"]),
                            pay + (2 * h + tid // 16) * (PLANE_PAGE * 16),
                        )
                else:
                    at = page * (4 * PLANE_PAGE) + (slot % 16) * 4 + slot // 16
                    for h, b in ((0, lo), (1, hi)):
                        bo.buffer_store(
                            fx.Int8(b),
                            rsrc(a["comp_iscale"]),
                            at + (2 * h + tid - 32) * PLANE_PAGE,
                        )
