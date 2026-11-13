# SPDX-License-Identifier: Apache-2.0
"""The DSpark verify head after the target's LM head (``head.hip``): the greedy and the
sampled accept, the simulated bonus and the accept outputs in two launches, the ranks'
partials swapped through peer buffers instead of the logits all-gather."""

from __future__ import annotations

import functools
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
MAX_BS = 64
MAX_SLOTS = 8
CHUNK = 2048
PART = 10
# head.hip's exchange: [round 2][request MAX_BS][parity 2][source 8][32 packets of 16 B]
EXCHANGE_BYTES = 2 * MAX_BS * 2 * 8 * 32 * 16


@functools.cache
def ops():
    """``torch.ops`` of ``head.hip``, built at first use."""
    from sgl_amd_mono.common.hip_ext import load

    sources = {p.name: p.read_text() for p in sorted(_HERE.glob("*.hip"))}
    return load("sgl_amd_mono_dsv41_head", sources, "gfx950")


STATE = 8


def nchunk(width: int) -> int:
    return -(-width // CHUNK)


def part_words(max_bs: int, slots: int, width: int) -> int:
    """head.hip's f32 scratch: the scan partials, the verdict state, the race partials."""
    n = nchunk(width)
    return max_bs * slots * n * PART + MAX_BS * STATE + max_bs * n * 2


class Exchange:
    """One TP group's exchange buffers and per-request call counters (collective:
    every rank builds it together, ``group`` the gloo group)."""

    def __init__(self, group, rank: int, size: int, device: torch.device):
        from sgl_amd_mono.common.peer import PeerBuffer

        self.rank, self.size = rank, size
        self._buffer = PeerBuffer(EXCHANGE_BYTES, group, rank, size, device)
        self._buffer.bytes.zero_()
        self.peers = self._buffer.addresses.cpu()
        self.calls = torch.zeros(MAX_BS, dtype=torch.int32, device=device)

    def close(self) -> None:
        self._buffer.close()
