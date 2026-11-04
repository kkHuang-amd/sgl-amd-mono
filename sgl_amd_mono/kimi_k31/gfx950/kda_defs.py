# SPDX-License-Identifier: Apache-2.0
"""Kimi-K3.1's KDA layer shapes a rank holds at TP8, and the scalar device helpers
the wide launch's stages read with.

The in_proj rows a rank holds, in order: q, k, v (``PROJ`` each), g2
(``PROJ``), f_a (``HD``) and beta (``NH``), padded to whole 16-row groups.
"""

import math

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr.typing import T

from sgl_amd_mono.common.ops import hw_exp2, rsrc

HIDDEN = 7168
HD = 128
NH = 12  # local heads at TP8
PROJ = NH * HD  # 1536
ROWS = 16
LOG2E = 1.0 / math.log(2.0)
QKV = 3 * PROJ
G2_0 = QKV
FA_0 = G2_0 + PROJ  # 6144
BETA_0 = FA_0 + HD  # 6272
NDEF = BETA_0 + NH  # 6284: the in_proj rows the kernel defines; beta ends them
NPROJ = 6288  # in_proj rows a rank holds (NDEF, padded to whole 16-row groups)
VCH = 16  # state rows a value chunk
NVCH = HD // VCH  # 8
CONV_W = 4


def ld_bf(ptr, i, cm=0):
    return fx.Float32(
        fx.BFloat16(
            bo.buffer_load(rsrc(ptr), i, vec_width=1, dtype=T.bf16, cache_modifier=cm)
        )
    )


def ld_f32(ptr, i):
    return fx.Float32(bo.buffer_load(rsrc(ptr), i, vec_width=1, dtype=T.f32))


def sigmoid(z):
    return fx.Float32(1.0) / (fx.Float32(1.0) + hw_exp2(z * fx.Float32(-LOG2E)))


def bf16r(v):
    return fx.Float32(fx.Float32(v).to(fx.BFloat16))


def f32_of(w):
    return w.bitcast(fx.Float32)
