# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from ROCm/ATOM (https://github.com/ROCm/ATOM) at 97359d6, MIT License,
# Copyright (C) 2026, Advanced Micro Devices, Inc.:
# atom/mono/device/mx.py
"""FP8 / MX arithmetic shared by the mono kernels."""

import os

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import rocdl
from flydsl.expr.typing import T, as_ir_value

from sgl_amd_mono.common.ops import CM_NT

FP8_MAX = 448.0
# the f8f6f4 MFMA's operand formats (cbsz / blgp)
FP8, FP4 = 0, 4
UNIT_SCALE = 127  # the E8M0 code of 1.0


def clamp_fp8(v):
    """``v`` clamped to the E4M3 range."""
    return fx.max(fx.min(v, fx.Float32(FP8_MAX)), fx.Float32(-FP8_MAX))


def code_ceil(v):
    """Return the biased exponent of the smallest power of two >= ``v``, clamped to [1, 254]."""
    bits = v.bitcast(fx.Int32)
    e = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).select(
        fx.Int32(1), fx.Int32(0)
    )
    return fx.max(fx.min(e, fx.Int32(254)), fx.Int32(1))


def rcp_pow2(code):
    """Return 1 / ``pow2(code)`` exactly (code in [1, 253])."""
    return ((254 - code) << 23).bitcast(fx.Float32)


def e8m0_roundup(amax):
    """AITER's MXFP4 activation scale (``_e8m0_roundup(amax, 6)``): the E8M0 code of
    the smallest power of two >= amax * f32(1 / 6), at most 254."""
    wi = (amax * fx.Float32(1.0 / 6.0)).bitcast(fx.Int32)
    b = fx.Int32(fx.Uint32(wi + fx.Int32(0x7FFFFF)) >> fx.Uint32(23)) & 0xFF
    return fx.min(b, fx.Int32(254))


def e8m0_even(amax):
    """AITER's Triton MXFP4 scale (``_mxfp4_scale_from_amax``): the E8M0 code of
    amax's exponent once 2^21 is added to its bits, over 4, in [0, 254]."""
    bits = (amax.bitcast(fx.Int32) + fx.Int32(0x200000)) & fx.Int32(0x7F800000)
    return fx.max(fx.min((bits >> 23) - fx.Int32(2), fx.Int32(254)), fx.Int32(0))


def fp4x2(a, b, code):
    """The E2M1 byte of f32 ``a`` (low nibble) and ``b`` over the E8M0 scale ``code``,
    rounded to nearest even and saturating, as AITER's MXFP4 quants round."""
    pair = rocdl.cvt_scalef32_pk_fp4_f32(
        T.i32,
        as_ir_value(fx.Int32(0)),
        as_ir_value(fx.Float32(a)),
        as_ir_value(fx.Float32(b)),
        as_ir_value((fx.Int32(code) << 23).bitcast(fx.Float32)),
        0,
    )
    return fx.Int32(pair) & 0xFF


def scale_index(row, col, cols):
    """Return the byte index of E8M0 scale (row, col) in aiter's ``shuffle_scale`` layout."""
    r32, a, b = row // 32, (row // 16) % 2, row % 16
    c8, d, e = col // 8, (col // 4) % 2, col % 4
    return ((((r32 * (cols // 8) + c8) * 4 + e) * 16 + b) * 2 + d) * 2 + a


# ABLATION ONLY (SGLANG_AMD_MONO_ABL=wcm0): weight loads with the default cache policy
_W_CM = 0 if "wcm0" in os.environ.get("SGLANG_AMD_MONO_ABL", "").split(",") else CM_NT


def fp4_tile_load(r_w, base, rg, k, st, at):
    """Load a lane's 16 B of an FP4 weight's (16, 16)-shuffled row group at step ``st``."""
    return fx.Vector(
        bo.buffer_load(
            r_w,
            base + rg * (2 * k) + st * 256 + at,
            vec_width=4,
            dtype=T.i32,
            cache_modifier=_W_CM,
        )
    )


def mfma_scaled(a, b, c, sa, sb, a_fmt=FP8, b_fmt=FP8, sa_byte=0, sb_byte=0):
    """One 16x16x128 scaled MFMA (gfx950 f8f6f4)."""
    return fx.Vector(
        rocdl.mfma_scale_f32_16x16x128_f8f6f4(
            T.vec(4, T.f32), [a, b, c, a_fmt, b_fmt, sa_byte, sa, sb_byte, sb]
        )
    )
