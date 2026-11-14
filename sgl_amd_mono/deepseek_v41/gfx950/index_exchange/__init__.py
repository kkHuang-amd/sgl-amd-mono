# SPDX-License-Identifier: Apache-2.0
"""The index sources' top-512 rows swapped across the TP ranks (``exchange.hip``) when
each rank selects only its own requests' rows (``SGLANG_AMD_MONO_DSV41_INDEX_SPLIT=1``)."""

from __future__ import annotations

import functools
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
MAX_TP = 8
MAX_ROWS = 48
TOPK = 512
BLOCKS = 16
# exchange.hip's buffer: [parity 2][source 8][row 48][array 2][512] int32, then the tags
EXCHANGE_BYTES = 2 * MAX_TP * MAX_ROWS * 2 * TOPK * 4 + 2 * MAX_TP * BLOCKS * 16


@functools.cache
def ops():
    """``torch.ops`` of ``exchange.hip``, built at first use."""
    from sgl_amd_mono.common.hip_ext import load

    sources = {p.name: p.read_text() for p in sorted(_HERE.glob("*.hip"))}
    return load("sgl_amd_mono_dsv41_index_exchange", sources, "gfx950")


class IndexExchange:
    """One TP group's exchange buffers and per-block call counters (collective: every
    rank builds it together, ``group`` the gloo group)."""

    def __init__(
        self,
        group,
        rank: int,
        size: int,
        device: torch.device,
        deadline_s: float,
        check: bool = False,
    ):
        from sgl_amd_mono.common.peer import PeerBuffer

        self.rank, self.size, self.deadline_s = rank, size, deadline_s
        self.ops = ops()
        self._buffer = PeerBuffer(EXCHANGE_BYTES, group, rank, size, device)
        self._buffer.bytes.zero_()
        self.peers = self._buffer.addresses.cpu()
        self.calls = torch.zeros(BLOCKS, dtype=torch.int32, device=device)
        # every rank built and zeroed before any peer's first push
        torch.cuda.synchronize(device)
        torch.distributed.barrier(group=group)
        # the check's host counters: [BLOCKS] mismatched own, [BLOCKS] mismatched peers',
        # [BLOCKS] compared 16 B vectors
        self.check = None
        if check:
            self.check = torch.zeros(3 * BLOCKS, dtype=torch.int64, pin_memory=True)

    def __call__(self, raw: torch.Tensor, page: torch.Tensor, ranges) -> None:
        """Every rank's rows ``ranges[q] = (lo, hi)`` of ``raw`` / ``page`` from their
        owner; this rank's own rows are already in place."""
        lo = [int(a) for a, _ in ranges]
        hi = [int(b) for _, b in ranges]
        self.ops.index_exchange(
            raw, page, lo, hi, self.calls, self.peers, self.rank, self.deadline_s
        )

    def compare(self, got, want, rows: int, own) -> None:
        """The check: rows [0, rows) of ``got`` / ``want`` (raw, page) compared row by
        row as sets (sorted first: a REINDEX layer's top-k is unordered), into the
        host counters; ``own`` (lo, hi) this rank's rows."""
        for g, w in zip(got, want):
            a = torch.sort(g[:rows], dim=1).values
            b = torch.sort(w[:rows], dim=1).values
            self.ops.index_compare(a, b, int(own[0]), int(own[1]), self.check)

    def checked(self) -> tuple[int, int, int]:
        """The check's (mismatched own, mismatched peers', compared) vectors so far: host memory only, so a
        thread may read it while the GPU runs or a graph is captured."""
        c = self.check.tolist()
        return sum(c[:BLOCKS]), sum(c[BLOCKS : 2 * BLOCKS]), sum(c[2 * BLOCKS :])

    def close(self) -> None:
        self._buffer.close()
