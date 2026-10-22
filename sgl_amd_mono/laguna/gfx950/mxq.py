# SPDX-License-Identifier: Apache-2.0
"""MXFP4 activation quantizers that reproduce SGLang's bit for bit, and FP4 decode.

``triton``: aiter's ``dynamic_mxfp4_quant`` (``_mxfp4_scale_from_amax``, "even"):
    E = biased exponent of (bits(amax) + 0x200000) with the mantissa dropped;
    scale byte = clamp(E - 2, 0, 254); x * 2^(127 - byte), packed with
    ``v_cvt_scalef32_pk_fp4_f32`` at scale 1.0 (aiter: same byte as its bit path).
``hip``: aiter's ``dynamic_per_group_scaled_quant`` fp4 (``fp_f32_to_e8m0_scale``
    RoundUp, amax clamped to >= 1e-10): u = bits(amax * (1/6)); byte = exponent(u) + (mantissa(u) != 0)
    (below 255); x * 2^(127 - byte), the same packing.
A group is 32 consecutive elements; a lane quantizes one group here.
"""

import flydsl.expr as fx
from flydsl.expr import range_constexpr, rocdl
from flydsl.expr.typing import T


def _imax(a, b):
    return (a > b).select(a, b)


def _imin(a, b):
    return (a < b).select(a, b)


def scale_byte(amax, rule):
    """The group's E8M0 byte (Int32) from its fp32 absolute maximum."""
    if rule == "triton":
        bits = amax.bitcast(fx.Int32)
        e = ((bits + fx.Int32(0x200000)) >> 23) & fx.Int32(0xFF)
        return _imax(_imin(e - 2, fx.Int32(254)), fx.Int32(0))
    assert rule == "hip"
    # aiter clamps the group maximum to 1e-10 first: an all-zero group gets byte 92
    amax = (amax > fx.Float32(1e-10)).select(amax, fx.Float32(1e-10))
    u = (amax * fx.Float32(1.0 / 6.0)).bitcast(fx.Int32)
    e = (u >> 23) & fx.Int32(0xFF)
    bump = (e < fx.Int32(0xFF)) & ((u & fx.Int32(0x7FFFFF)) != fx.Int32(0))
    return e + bump.select(fx.Int32(1), fx.Int32(0))


def pow2(byte):
    """2^(byte - 127) as fp32 (byte in [1, 254]; 0 -> 2^-127 is not representable,
    the multiplier side uses ``inv_pow2``)."""
    return (byte << 23).bitcast(fx.Float32)


def inv_pow2(byte):
    """2^(127 - byte) exactly for byte in [0, 254] (2^127 at byte 0)."""
    return ((fx.Int32(254) - byte) << 23).bitcast(fx.Float32)


def quant_group(vals, rule):
    """32 fp32 values -> (four packed fp4 dwords, scale byte Int32)."""
    assert len(vals) == 32
    amax = abs_f(vals[0])
    for v in vals[1:]:
        m = abs_f(v)
        amax = (m > amax).select(m, amax)
    byte = scale_byte(amax, rule)
    q = inv_pow2(byte)
    words = []
    for w in range_constexpr(4):
        packed = fx.Int32(0)
        for h in range_constexpr(4):
            i = w * 8 + h * 2
            packed = fx.Int32(
                rocdl.cvt_scalef32_pk_fp4_f32(
                    T.i32, packed, vals[i] * q, vals[i + 1] * q, fx.Float32(1.0), h
                )
            )
        words.append(packed)
    return words, byte


def abs_f(v):
    return (v.bitcast(fx.Int32) & fx.Int32(0x7FFFFFFF)).bitcast(fx.Float32)


def decode_word(word, scale_f32):
    """One packed dword of 8 fp4 -> 8 fp32, each times ``scale_f32`` (exact)."""
    out = []
    for sel in range_constexpr(4):
        pair = fx.Vector(rocdl.cvt_scalef32_pk_f32_fp4(T.vec(2, T.f32), word, scale_f32, sel))
        out += [fx.Float32(pair[0]), fx.Float32(pair[1])]
    return out
