# SPDX-License-Identifier: Apache-2.0
"""The DSpark draft's sampled Markov steps on gfx950 (``markov.hip``): a step's bias
GEMV, scores, Gumbel-max pick and the ranks' bests swapped in two launches, every
rank's token alike without a collective."""

from __future__ import annotations

import functools
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
DIM = 256  # the Markov rank the kernels hold in registers
MAX_SEQUENCES = 8
BLOCK_ROWS = 64
# a sequence's 16 packet slots of 16 B (markov.hip's SLOTS)
EXCHANGE_BYTES = MAX_SEQUENCES * 16 * 16
# markov_sample_block (SGLANG_AMD_MONO_DSV41_MARKOV_PERSIST): two step parities of every sequence's rank slots
PBLOCKS, MAX_TP = 256, 8
BLOCK_EXCHANGE_BYTES = 2 * MAX_SEQUENCES * MAX_TP * 16


@functools.cache
def ops():
    """``torch.ops`` of ``markov.hip``, built at first use."""
    from sgl_amd_mono.common.hip_ext import load

    sources = {p.name: p.read_text() for p in sorted(_HERE.glob("*.hip"))}
    return load("sgl_amd_mono_dsv41_markov", sources, "gfx950")


class Exchange:
    """One TP group's exchange: every rank's buffer address (int64 on the host), and
    the per-sequence call counters and last tokens (collective: every rank builds it
    together, ``group`` the gloo group)."""

    def __init__(self, group, rank: int, size: int, device: torch.device):
        from sgl_amd_mono.common.peer import PeerBuffer

        self.rank, self.size = rank, size
        self._buffer = PeerBuffer(EXCHANGE_BYTES, group, rank, size, device)
        self._buffer.bytes.zero_()
        self.peers = self._buffer.addresses.cpu()
        self.calls = torch.zeros(MAX_SEQUENCES, dtype=torch.int32, device=device)
        self.tok = torch.zeros(MAX_SEQUENCES, dtype=torch.int64, device=device)

    def close(self) -> None:
        self._buffer.close()


class BlockExchange:
    """``markov_sample_block``'s exchange (collective): every rank's slot buffer, this GPU's block
    partials and the launch counter; separate from ``Exchange`` (other slots and tags)."""

    def __init__(self, group, rank: int, size: int, device: torch.device):
        from sgl_amd_mono.common.peer import PeerBuffer

        self.rank, self.size = rank, size
        self._buffer = PeerBuffer(BLOCK_EXCHANGE_BYTES, group, rank, size, device)
        self._buffer.bytes.zero_()
        self.peers = self._buffer.addresses.cpu()
        self.partials = torch.zeros(
            2 * MAX_SEQUENCES * PBLOCKS * 4, dtype=torch.int32, device=device
        )
        self.calls = torch.zeros(1, dtype=torch.int32, device=device)

    def close(self) -> None:
        self._buffer.close()


def blocks(width: int) -> int:
    return -(-width // BLOCK_ROWS)
