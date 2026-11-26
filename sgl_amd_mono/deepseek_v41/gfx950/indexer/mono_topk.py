# SPDX-License-Identifier: Apache-2.0
"""An index source's top 512 a row in one launch, after K2i's logits (``SGLANG_AMD_MONO_DSV41_FUSED_TOPK``).

Two levels on ATOM's CTA radix select (``common.topk``), as ATOM's indexer does: a segment task takes
the top 512 of SEG columns of a row (columns at or past the row's bound do not count) and leaves them
as candidates; once a row's segments are all in (an agent-scope counter), its merge task takes the
top 512 of the candidates and writes SGLang's ``sparse_raw_indices`` / ``sparse_page_indices`` (-1
padded, any order). Ties go to the lower column, as SGLang's top-k. A REINDEX layer's columns are its
compact candidate columns: the merge maps them to positions through ``ids`` (``map_compact_selection``'s
arithmetic). A row of one segment writes its selection from the segment task. Tasks run on BLOCKS
persistent CTAs, every CTA's segment tasks before its merge tasks, so a merge task waits only on tasks
that never wait."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import gpu, range_constexpr, rocdl
from flydsl.expr.typing import Int32, Int64, T

from sgl_amd_mono.common import topk
from sgl_amd_mono.common.epoch import kernel_symbol
from sgl_amd_mono.common.ops import CM_DEV, rsrc, uniform
from sgl_amd_mono.common.plan import BLOCKS, THREADS, kernel_sources
from sgl_amd_mono.common.sync import _spin

from . import index_plan as ip

SOURCES = kernel_sources(__name__)

_STREAM = fx.Stream(None)
K = ip.TOPK  # 512
SEG = 8192  # a segment task's columns
E_SEG = SEG // THREADS
MAX_SEGS = 16  # a row's segments at most: wider rows keep SGLang's top-k
E_MERGE = MAX_SEGS  # candidates a thread holds in the merge
INDEX_BITS = 24  # columns below 2^24 - 1
PAGE_ROWS = 64  # the index plane's page rows (the page table's entries)
assert K == THREADS


def lds_words() -> int:
    return topk.lds_words(K, index_bits=INDEX_BITS, ranked=False, in_order=True)


def build_mono_topk(rows: int, reindex: bool):
    """The top-512 launch for a step of ``rows`` rows; ``reindex``: a REINDEX layer's compact columns."""
    name = kernel_symbol("dsv41_mono_topk", s=rows, r=int(reindex))
    keyed = (("rows", rows), ("reindex", reindex), ("sources", SOURCES))
    words = lds_words()

    @fx.struct
    class Lds:
        w: fx.Array[fx.Int32, words, 16]

    def _store_out(row, col, a):
        """Thread ``tid``'s selected column -> raw / page index (or -1) at slot ``tid``."""
        tid = fx.thread_idx.x
        valid = col >= 0
        c = valid.select(col, fx.Int32(0))
        if reindex:
            block = fx.Int32(
                bo.buffer_load(
                    rsrc(a["ids"]),
                    row * a["ids_stride"] + c // ip.BLOCK_ROWS,
                    vec_width=1,
                    dtype=T.i32,
                )
            )
            pos = block * ip.BLOCK_ROWS + c % ip.BLOCK_ROWS
        else:
            pos = c
        page = fx.Int32(
            bo.buffer_load(
                rsrc(a["page_table"]),
                row * a["pt_stride"] + pos // PAGE_ROWS,
                vec_width=1,
                dtype=T.i32,
            )
        )
        raw = valid.select(pos, fx.Int32(-1))
        slot = valid.select(page * PAGE_ROWS + pos % PAGE_ROWS, fx.Int32(-1))
        bo.buffer_store(raw, rsrc(a["raw_out"]), row * K + tid)
        bo.buffer_store(slot, rsrc(a["page_out"]), row * K + tid)

    @flyc.jit
    def _segment(lds, row, seg, segs, a):
        tid = fx.thread_idx.x
        bound = fx.Int32(
            bo.buffer_load(rsrc(a["bounds"]), row, vec_width=1, dtype=T.i32)
        )
        r_log = rsrc(a["ilog"])
        us, idxs, valid = [], [], []
        for e in range_constexpr(E_SEG):
            col = seg * SEG + tid * E_SEG + e
            bits = fx.Int32(
                bo.buffer_load(
                    r_log,
                    row * a["lstride"] + col,
                    vec_width=1,
                    dtype=T.i32,
                    cache_modifier=CM_DEV,
                )
            )
            us.append(topk.order_key(bits))
            idxs.append(col)
            valid.append(col < bound)
        n_out = topk.select(
            lds,
            tid,
            tid % 64,
            uniform(tid // 64),
            us,
            idxs,
            K,
            index_bits=INDEX_BITS,
            valid=valid,
            in_order=True,
        )
        u, col = topk.slot(lds, tid, n_out, K, index_bits=INDEX_BITS, in_order=True)
        if segs == 1:
            _store_out(row, col, a)
        else:
            at = ((row * MAX_SEGS + seg) * K + tid) * 2
            bo.buffer_store(
                fx.Vector.from_elements([u, col], fx.Int32),
                rsrc(a["cand"]),
                at,
                cache_modifier=CM_DEV,
            )
            rocdl.s_waitcnt(vmcnt=0)
            gpu.barrier()
            if tid == 0:
                ptr = fx.inttoptr(
                    fx.PointerType.get(T.i32, fx.AddressSpace.Global, 4),
                    a["count"] + fx.Int64(row * 4),
                )
                fx.llvm.atomic_add(ptr, fx.Int32(1), syncscope="agent")

    @flyc.jit
    def _merge(lds, row, segs, a):
        tid = fx.thread_idx.x
        r_cnt = rsrc(a["count"])

        def load_count():
            return fx.Int32(
                bo.buffer_load(
                    r_cnt, row, vec_width=1, dtype=T.i32, cache_modifier=CM_DEV
                )
            )

        _ = _spin(load_count, lambda v: v < segs, guarded=False)
        r_c = rsrc(a["cand"])
        us, idxs, valid = [], [], []
        for m in range_constexpr(E_MERGE):
            f = tid * E_MERGE + m  # (segment, slot) in column order
            seg, j = f // K, f % K
            w = fx.Vector(
                bo.buffer_load(
                    r_c,
                    ((row * MAX_SEGS + seg) * K + j) * 2,
                    vec_width=2,
                    dtype=T.i32,
                    cache_modifier=CM_DEV,
                )
            )
            ok = (seg < segs) & (w[1] >= 0)
            us.append(w[0])
            idxs.append(ok.select(w[1], fx.Int32(0)))
            valid.append(ok)
        n_out = topk.select(
            lds,
            tid,
            tid % 64,
            uniform(tid // 64),
            us,
            idxs,
            K,
            index_bits=INDEX_BITS,
            valid=valid,
            in_order=True,
        )
        _, col = topk.slot(lds, tid, n_out, K, index_bits=INDEX_BITS, in_order=True)
        _store_out(row, col, a)
        if tid == 0:
            bo.buffer_store(fx.Int32(0), r_cnt, row, cache_modifier=CM_DEV)

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def kern(
        ilog: Int64, lstride: Int32, bounds: Int64, page_table: Int64, pt_stride: Int32, ids: Int64,
        ids_stride: Int32, raw_out: Int64, page_out: Int64, cand: Int64, count: Int64, segs: Int32,
    ):  # fmt: skip
        _ = keyed
        lds = fx.SharedAllocator().allocate(Lds).peek().w.ptr
        a = {
            "ilog": ilog, "lstride": lstride, "bounds": bounds, "page_table": page_table,
            "pt_stride": pt_stride, "ids": ids, "ids_stride": ids_stride, "raw_out": raw_out,
            "page_out": page_out, "cand": cand, "count": count,
        }  # fmt: skip
        bid = fx.block_idx.x
        n_seg = rows * segs
        n_all = n_seg + (segs > 1).select(fx.Int32(rows), fx.Int32(0))
        for t in range(bid, n_all, BLOCKS):
            if t < n_seg:
                _segment(lds, t // segs, t % segs, segs, a)
            else:
                _merge(lds, t - n_seg, segs, a)
            gpu.barrier()

    @flyc.jit
    def launch(
        ilog: Int64, lstride: Int32, bounds: Int64, page_table: Int64, pt_stride: Int32, ids: Int64,
        ids_stride: Int32, raw_out: Int64, page_out: Int64, cand: Int64, count: Int64, segs: Int32,
        stream: fx.Stream = _STREAM,
    ):  # fmt: skip
        _ = keyed
        kern(
            ilog, lstride, bounds, page_table, pt_stride, ids, ids_stride, raw_out, page_out, cand, count, segs,
        ).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)  # fmt: skip

    return launch
