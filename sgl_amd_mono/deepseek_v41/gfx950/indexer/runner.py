# SPDX-License-Identifier: Apache-2.0
# Adapted from sgl-project/sglang#43984 at 134919476c (kernels/ops/mono/gfx950/
# deepseek_v41/indexer/runner.py)
"""K2i's host side on one rank (``DSV41MonoRunner.enable_k2i``): K2i's logits, then SGLang's top-k.

Every tensor a replayed launch reads or writes lives here, so captured graphs stay valid. A
``full`` source keeps SGLang's top 512 of every visible column; a ``producer`` also publishes
SGLang's candidate blocks of its logits; a ``reindex`` source scores only those blocks' columns
and maps its top 512 back to positions and slots (``candidate_table``)."""

from __future__ import annotations

import os

import torch
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled
from sglang.kernels.ops.attention.dsv4.candidate_blocks_hip import (
    select_candidate_blocks_hip,
    topk_transform_paged_hip,
)

from . import index_plan as ip
from .candidate_table import PLANE_ROWS, map_compact_selection, reindex_table
from .k2i import MAX_ROWS, build_k2i
from .mono_topk import MAX_SEGS, SEG
from .mono_topk import K as TOPK_K
from .mono_topk import build_mono_topk
from .walk_plan import fill_walk_plan_device, new_plan

# the SGLang names this module reads, pinned by test/unit/test_contract.py
SGLANG_READS = {
    "kernels/ops/attention/dsv4/candidate_blocks_hip.py": (
        "select_candidate_blocks_hip",
        "topk_transform_paged_hip",
        "select_candidate_blocks_hip(logits)",
        "select_candidate_blocks_hip(seq_lens)",
        "select_candidate_blocks_hip(topk_blocks)",
        "select_candidate_blocks_hip(block_size)",
        "topk_transform_paged_hip(scores)",
        "topk_transform_paged_hip(seq_lens)",
        "topk_transform_paged_hip(page_table)",
        "topk_transform_paged_hip(page_indices)",
        "topk_transform_paged_hip(page_size)",
        "topk_transform_paged_hip(raw_indices)",
        # ``CandidateBlocks.ids``
        "ids",
    ),
}

COMPACT = ip.TOPK_BLOCKS * ip.BLOCK_ROWS  # a REINDEX row's candidate columns
KINDS = ("full", "producer", "reindex")


def _roundup(x: int, m: int) -> int:
    return -(-x // m) * m


def check_plane(*, payload: torch.Tensor, scale: torch.Tensor) -> None:
    """Raise unless the index plane is SGLang's split FP4 layout of 64-row pages."""
    u8 = payload.view(torch.uint8)
    pages = scale.shape[0]
    if not (
        u8.is_contiguous()
        and scale.is_contiguous()
        and scale.dtype == torch.uint8
        and u8.numel() == pages * 4 * PLANE_ROWS * 16
        and scale.numel() == pages * 4 * PLANE_ROWS
    ):
        raise RuntimeError(
            f"K2i index plane: payload {tuple(payload.shape)} {payload.dtype}, scales "
            f"{tuple(scale.shape)} {scale.dtype}; want [pages, 1, 4, {PLANE_ROWS}, 16] "
            f"and [pages, 1, 4, {PLANE_ROWS}] bytes"
        )


class K2iRunner:
    """K2i on one device, for rows up to MAX_ROWS and column bounds up to
    ``bound_max``."""

    def __init__(self, device: torch.device, bound_max: int):
        f32 = dict(dtype=torch.float32, device=device)
        i32 = dict(dtype=torch.int32, device=device)
        self.bound_max = _roundup(max(bound_max, COMPACT), ip.CHUNK)
        self.ilog = torch.empty(MAX_ROWS * self.bound_max, **f32)
        # ``reindex_table``'s outputs: blocks newest last, their key table, each row's columns
        self.cand_ids = torch.empty(MAX_ROWS, ip.TOPK_BLOCKS, **i32)
        self.cand_table = torch.empty(MAX_ROWS, ip.TOPK_BLOCKS, **i32)
        self.cand_bound = torch.empty(MAX_ROWS, **i32)
        # the compact top-k as one page at slot 0, so a column is its own slot
        self.compact_pos = torch.empty(MAX_ROWS, ip.TOPK, **i32)
        self.compact_table = torch.zeros(MAX_ROWS, 1, **i32)
        self.plans = {ratio: new_plan(device) for ratio in (1, 2)}
        # SGLANG_AMD_MONO_DSV41_FUSED_TOPK=1: the top-512 (and a REINDEX layer's position map) in one
        # launch (``mono_topk``); rows wider than MAX_SEGS * SEG columns keep SGLang's top-k
        self.fused = os.environ.get("SGLANG_AMD_MONO_DSV41_FUSED_TOPK", "0") == "1"
        self.fused_check = (
            self.fused
            and os.environ.get("SGLANG_AMD_MONO_DSV41_FUSED_TOPK_CHECK", "0") == "1"
        )
        if self.fused:
            self.cand = torch.empty(MAX_ROWS * MAX_SEGS * TOPK_K * 2, **i32)
            self.count = torch.zeros(MAX_ROWS, **i32)
            self._topk_kernels: dict = {}
        if self.fused_check:
            # SGLang's selection beside the fused one: [rows seen, rows differing], on the host too
            self.check_raw = torch.empty(MAX_ROWS, TOPK_K, **i32)
            self.check_page = torch.empty(MAX_ROWS, TOPK_K, **i32)
            self.check = torch.zeros(2, dtype=torch.int64, device=device)
            self.check_host = torch.zeros(2, dtype=torch.int64, pin_memory=True)
            _report_topk_check(self.check_host)
        # owners[p - 1][t]: row t's request's first row, for p rows a request
        t = torch.arange(MAX_ROWS)
        self.owners = torch.stack([t // p * p for p in range(1, MAX_ROWS + 1)]).to(
            **i32
        )
        self._kernels: dict = {}

    def kernel(self, rows: int):
        from ..runner import no_capture

        if rows not in self._kernels:
            no_capture(f"K2i width {rows}")
            self._kernels[rows] = build_k2i(rows)
        return self._kernels[rows]

    def walk_plan(
        self, *, ratio: int, bounds: torch.Tensor, rows: int, per_req: int
    ) -> None:
        """Lay out the step's FULL walk plan for compress ratio ``ratio`` from device bounds.

        Once a step and ratio, before its first FULL or producer layer, which both read it."""
        fill_walk_plan_device(
            bounds=bounds,
            owners=self.owners[per_req - 1],
            plan=self.plans[ratio],
            rows=rows,
        )

    def select(
        self,
        *,
        kind: str,
        rows: int,
        per_req: int,
        ratio: int,
        query: tuple,  # [rows, 32, 64] E2M1 pairs, [rows, 1, 4, 16, 4] E8M0, [rows, 32] bf16
        plane: tuple,  # the layer's index plane (``check_plane``)
        page_table: torch.Tensor,  # int32 [rows, pages] of 64-row pages
        bounds: torch.Tensor,  # int32 [rows]: the columns each row sees
        raw_out: torch.Tensor,  # int32 [rows, 512]: sparse_raw_indices
        page_out: torch.Tensor,  # int32 [rows, 512]: sparse_page_indices
        candidates=None,  # a REINDEX layer's: the producer's CandidateBlocks
        reindex_table_built=False,  # a later REINDEX layer of the step: table kept
    ):
        """An index source's top-512 of each row (``kind`` in KINDS).

        FULL and producer layers read the step's walk plan of ``ratio``; a producer returns its
        candidate blocks for the step's REINDEX layers."""
        assert kind in KINDS, kind
        _check(
            rows=rows, per_req=per_req, query=query, page_table=page_table,
            bounds=bounds, raw_out=raw_out, page_out=page_out,
        )  # fmt: skip
        inputs = dict(
            rows=rows, per_req=per_req, query=query, plane=plane,
            page_table=page_table, bounds=bounds,
        )  # fmt: skip
        if kind == "reindex":
            compact, lens, ids = self.score_candidates(
                candidates=candidates, table_built=reindex_table_built, **inputs
            )
            if self._mono_topk(
                rows=rows, logits=compact, bounds=lens, page_table=page_table,
                raw_out=raw_out, page_out=page_out, ids=ids,
            ):  # fmt: skip
                return None
            pos = self.compact_pos[:rows]
            topk_transform_paged_hip(
                scores=compact,
                seq_lens=lens,
                page_table=self.compact_table[:rows],
                page_indices=pos,
                page_size=COMPACT,
                raw_indices=None,
            )
            map_compact_selection(
                compact_pos=pos,
                ids=ids,
                page_table=page_table,
                raw_out=raw_out,
                page_out=page_out,
            )
            return None
        logits = self.score(ratio=ratio, **inputs)
        published = None
        if kind == "producer":
            published = select_candidate_blocks_hip(
                logits=logits,
                seq_lens=bounds,
                topk_blocks=ip.TOPK_BLOCKS,
                block_size=ip.BLOCK_ROWS,
            )
        if not self._mono_topk(
            rows=rows, logits=logits, bounds=bounds, page_table=page_table,
            raw_out=raw_out, page_out=page_out, ids=None,
        ):  # fmt: skip
            topk_transform_paged_hip(
                scores=logits,
                seq_lens=bounds,
                page_table=page_table,
                page_indices=page_out,
                page_size=PLANE_ROWS,
                raw_indices=raw_out,
            )
        return published

    def _mono_topk(
        self, *, rows, logits, bounds, page_table, raw_out, page_out, ids
    ) -> bool:
        """The fused top-512 of ``logits`` [rows, width] into ``raw_out`` / ``page_out``
        (``ids``: a REINDEX layer's blocks), or False: SGLang's top-k runs instead."""
        if not self.fused:
            return False
        width = logits.shape[1]
        segs = -(-width // SEG)
        if segs > MAX_SEGS:
            return False
        reindex = ids is not None
        key = (rows, reindex)
        if key not in self._topk_kernels:
            from ..runner import no_capture

            no_capture(f"mono top-k width {rows}")
            self._topk_kernels[key] = build_mono_topk(rows, reindex)
        ids_t = ids if reindex else page_table
        _run_compiled(
            self._topk_kernels[key],
            logits.data_ptr(), logits.stride(0), bounds.data_ptr(), page_table.data_ptr(),
            page_table.stride(0), ids_t.data_ptr(), ids_t.stride(0), raw_out.data_ptr(),
            page_out.data_ptr(), self.cand.data_ptr(), self.count.data_ptr(), segs,
            torch.cuda.current_stream(),
        )  # fmt: skip
        if self.fused_check:
            ref_raw, ref_page = self.check_raw[:rows], self.check_page[:rows]
            if reindex:
                pos = self.compact_pos[:rows]
                topk_transform_paged_hip(
                    scores=logits, seq_lens=bounds, page_table=self.compact_table[:rows],
                    page_indices=pos, page_size=COMPACT, raw_indices=None,
                )  # fmt: skip
                map_compact_selection(
                    compact_pos=pos,
                    ids=ids,
                    page_table=page_table,
                    raw_out=ref_raw,
                    page_out=ref_page,
                )
            else:
                topk_transform_paged_hip(
                    scores=logits, seq_lens=bounds, page_table=page_table,
                    page_indices=ref_page, page_size=PLANE_ROWS, raw_indices=ref_raw,
                )  # fmt: skip
            big = torch.iinfo(torch.int32).max
            a = torch.where(raw_out < 0, big, raw_out).sort(dim=1).values
            b = torch.where(ref_raw < 0, big, ref_raw).sort(dim=1).values
            self.check[0] += rows
            self.check[1] += (a != b).any(dim=1).sum()
            self.check_host.copy_(self.check, non_blocking=True)
        return True

    def score(
        self, *, rows, per_req, ratio, query, plane, page_table, bounds
    ) -> torch.Tensor:
        """A FULL or producer layer's logits [rows, width] f32, valid below each row's bound.

        ``width`` spans the page table's columns, as wide as SGLang's logits for its top-k."""
        width = min(page_table.shape[1] * PLANE_ROWS, self.bound_max)
        # a page-table entry is 8 virtual 8-row pages (ishift 3)
        self._score(
            rows=rows, query=query, plane=plane, plan=self.plans[ratio], itab=page_table,
            itab_stride=page_table.stride(0), itab_len=page_table.shape[1],
            ibatch=self.owners[per_req - 1], ishift=3, ibound=bounds, lstride=width,
            reindex=0,
        )  # fmt: skip
        return self.ilog[: rows * width].view(rows, width)

    def score_candidates(
        self, *, candidates, rows, per_req, query, plane, page_table, bounds,
        table_built=False,
    ):  # fmt: skip
        """A REINDEX layer's logits of its candidate blocks' columns -> (compact, lens, ids).

        ``ids`` is ``candidates.ids`` with the newest block last; compact column j * 8 + i is row
        i of block ids[j], valid below ``lens``."""
        if tuple(candidates.ids.shape) != (rows, ip.TOPK_BLOCKS):
            raise RuntimeError(
                f"K2i: candidate blocks {tuple(candidates.ids.shape)} for {rows} rows"
            )
        owners = self.owners[per_req - 1]
        ids, lens = self.cand_ids[:rows], self.cand_bound[:rows]
        if not table_built:
            reindex_table(
                ids=candidates.ids,
                page_table=page_table,
                owners=owners,
                bounds=bounds,
                ids_out=ids,
                table=self.cand_table,
                context=lens,
            )
        # a REINDEX launch ignores the plan: its task comes from its CTA index
        self._score(
            rows=rows, query=query, plane=plane, plan=self.plans[1], itab=self.cand_table,
            itab_stride=ip.TOPK_BLOCKS, itab_len=ip.TOPK_BLOCKS, ibatch=owners, ishift=0,
            ibound=lens, lstride=COMPACT, reindex=1,
        )  # fmt: skip
        return self.ilog[: rows * COMPACT].view(rows, COMPACT), lens, ids

    def _score(
        self, *, rows, query, plane, plan, itab, itab_stride, itab_len, ibatch, ishift,
        ibound, lstride, reindex,
    ) -> None:  # fmt: skip
        q_fp4, q_scale, weights = query
        payload, scale = plane
        _run_compiled(
            self.kernel(rows),
            q_fp4.data_ptr(), q_scale.data_ptr(), weights.data_ptr(), payload.data_ptr(), scale.data_ptr(),
            plan.data_ptr(), itab.data_ptr(), itab_stride, itab_len, ibatch.data_ptr(), ishift,
            ibound.data_ptr(), self.ilog.data_ptr(), lstride, reindex,
            torch.cuda.current_stream(),
        )  # fmt: skip


def _report_topk_check(host: torch.Tensor, every_s: float = 60.0) -> None:
    """Log the fused top-k check's counts every ``every_s`` from a daemon thread (host memory only)."""
    import logging
    import threading
    import time

    log = logging.getLogger(__name__)

    def loop():
        while True:
            time.sleep(every_s)
            seen, differ = int(host[0]), int(host[1])
            log.info(
                "DSV4.1 fused top-k check: %d / %d rows differ from SGLang's",
                differ,
                seen,
            )

    threading.Thread(target=loop, name="fused-topk-check", daemon=True).start()


def _check(*, rows, per_req, query, page_table, bounds, raw_out, page_out) -> None:
    """Raise unless a selection's tensors are the shapes K2i reads and writes."""
    if not (0 < rows <= MAX_ROWS and rows % per_req == 0):
        raise RuntimeError(f"K2i: {rows} rows, {per_req} a request")
    q_fp4, q_scale, weights = query
    if not (
        q_fp4.is_contiguous()
        and q_fp4.numel() == rows * ip.HEADS * ip.DIM // 2
        and q_fp4.element_size() == 1
        and q_scale.is_contiguous()
        and q_scale.numel() == rows * 4 * 16 * 4
        and tuple(weights.shape) == (rows, ip.HEADS)
        and weights.dtype == torch.bfloat16
        and weights.is_contiguous()
    ):
        raise RuntimeError(
            f"K2i query: {tuple(q_fp4.shape)} {q_fp4.dtype}, scales "
            f"{tuple(q_scale.shape)}, weights {tuple(weights.shape)} {weights.dtype}"
        )
    if not (
        page_table.dtype == torch.int32
        and page_table.dim() == 2
        and page_table.stride(1) == 1
        and page_table.shape[0] == rows
        and bounds.dtype == torch.int32
        and bounds.is_contiguous()
        and bounds.shape[0] == rows
    ):
        raise RuntimeError(
            f"K2i page table {tuple(page_table.shape)} {page_table.dtype}, "
            f"bounds {tuple(bounds.shape)} {bounds.dtype}"
        )
    for name, t in (("raw", raw_out), ("page", page_out)):
        if not (
            t.dtype == torch.int32
            and t.stride(1) == 1
            and t.stride(0) == ip.TOPK
            and t.shape[0] == rows
        ):
            raise RuntimeError(
                f"K2i {name} indices: {tuple(t.shape)} {t.stride()} {t.dtype}"
            )
