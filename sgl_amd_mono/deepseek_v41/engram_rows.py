# SPDX-License-Identifier: Apache-2.0
"""Every Engram layer's shard rows of a step in one gather launch (``SGLANG_AMD_MONO_DSV41_ENGRAM_ONE``):
SGLang's ``engram_gather`` arithmetic (``_dequant_rows``) for up to two layers, reading each layer's ids
straight from ``hash_ids`` [T, hash layers, heads] (no transposed copy) into ``out`` [layers, T, heads, dim]."""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from sglang.kernels.ops.embeddings.engram_gather import _E8M0_ZERO, _dequant_rows

# by file under sglang/ (test_contract.py pins them)
SGLANG_READS = {
    "kernels/ops/embeddings/engram_gather.py": ("_dequant_rows", "_E8M0_ZERO"),
}

MAX_LAYERS = 2


@triton.jit
def _layer_rows(H, hs0, hs1, hs2, t, c, j, W, S, LO, HI, OUT, row, DIM: tl.constexpr, BLK: tl.constexpr,
                E8M0_ZERO: tl.constexpr, MX: tl.constexpr):  # fmt: skip
    idx = tl.load(H + t * hs0 + c * hs1 + j * hs2).to(tl.int64)
    owned = (idx >= LO) & (idx < HI)
    local = tl.where(owned, idx - LO, 0)
    offs = tl.arange(0, DIM)
    vals = _dequant_rows(
        W.to(tl.int64), S.to(tl.int64), local, offs, owned, DIM, BLK, E8M0_ZERO, MX
    )
    tl.store(OUT + row * DIM + offs, tl.where(owned, vals, 0.0).to(tl.bfloat16))


@triton.jit
def _engram_rows_kernel(
    H, hs0, hs1, hs2, T, HEADS,
    W0, S0, LO0, HI0, C0, W1, S1, LO1, HI1, C1, OUT,
    DIM: tl.constexpr, BLK: tl.constexpr, E8M0_ZERO: tl.constexpr, MX0: tl.constexpr, MX1: tl.constexpr,
):  # fmt: skip
    pid = tl.program_id(0).to(tl.int64)
    per = T * HEADS
    k = pid // per
    r = pid % per
    t, j = r // HEADS, r % HEADS
    if k == 0:
        _layer_rows(
            H,
            hs0,
            hs1,
            hs2,
            t,
            C0,
            j,
            W0,
            S0,
            LO0,
            HI0,
            OUT,
            pid,
            DIM,
            BLK,
            E8M0_ZERO,
            MX0,
        )
    else:
        _layer_rows(
            H,
            hs0,
            hs1,
            hs2,
            t,
            C1,
            j,
            W1,
            S1,
            LO1,
            HI1,
            OUT,
            pid,
            DIM,
            BLK,
            E8M0_ZERO,
            MX1,
        )


def engram_rows(
    hash_ids: torch.Tensor, layers: list, out: torch.Tensor, block_size: int
) -> None:
    """``layers``: (hash index, embed) per Engram layer (at most MAX_LAYERS); ``out`` [layers, T, heads, dim]
    bf16 contiguous gets each layer's rows of this rank's shard (zeros elsewhere)."""
    assert (
        1 <= len(layers) <= MAX_LAYERS
        and out.is_contiguous()
        and out.dtype == torch.bfloat16
    )
    t, _, heads = hash_ids.shape
    dim = out.shape[-1]
    args = []
    mx = []
    for c, embed in layers + layers[-1:] * (MAX_LAYERS - len(layers)):
        w, s = embed._table_ptrs()
        args += [w, s, embed.row_start, embed.row_start + embed.rows, c]
        mx.append(bool(getattr(embed, "mxfp4", False)))
    n = len(layers) * t * heads
    if n:
        _engram_rows_kernel[(n,)](
            hash_ids, hash_ids.stride(0), hash_ids.stride(1), hash_ids.stride(2), t, heads, *args, out,
            DIM=dim, BLK=block_size, E8M0_ZERO=_E8M0_ZERO, MX0=mx[0], MX1=mx[1],
        )  # fmt: skip
