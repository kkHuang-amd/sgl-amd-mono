"""Check gfx950/indexer/mono_topk.py against SGLang's top-k (topk_transform_paged_hip; REINDEX: then
map_compact_selection) on random logits: each row's raw / page index sets equal, eager and under graph
replay; then the time of one launch against SGLang's launches.

HIP_VISIBLE_DEVICES=4 python check_mono_topk.py [--rows 6 48] [--widths 16384 65536] [--trials 4]"""

import argparse

import torch
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled
from sglang.kernels.ops.attention.dsv4.candidate_blocks_hip import (
    topk_transform_paged_hip,
)

from sgl_amd_mono.deepseek_v41.gfx950.indexer import index_plan as ip
from sgl_amd_mono.deepseek_v41.gfx950.indexer.candidate_table import (
    map_compact_selection,
)
from sgl_amd_mono.deepseek_v41.gfx950.indexer.mono_topk import (
    MAX_SEGS,
    SEG,
    K,
    build_mono_topk,
)

COMPACT = ip.TOPK_BLOCKS * ip.BLOCK_ROWS


class Case:
    def __init__(self, rows, width, reindex, gen, dev, ties):
        self.rows, self.width, self.reindex = rows, width, reindex
        pages = -(-(width if not reindex else 1 << 18) // 64)
        self.ilog = torch.randn(rows, width, generator=gen, device=dev)
        if ties:
            self.ilog = (
                self.ilog * 4
            ).round() / 4 + 0.0  # no -0.0 (ordered below +0.0 by the key)
        self.bounds = torch.randint(
            1, width + 1, (rows,), generator=gen, device=dev, dtype=torch.int32
        )
        self.bounds[0] = width
        self.bounds[-1] = min(300, width)
        self.page_table = torch.randint(
            0, 1 << 20, (rows, pages), generator=gen, device=dev, dtype=torch.int32
        )
        self.ids = torch.randint(
            0,
            (1 << 18) // 8,
            (rows, ip.TOPK_BLOCKS),
            generator=gen,
            device=dev,
            dtype=torch.int32,
        )
        self.zero_table = torch.zeros(rows, 1, dtype=torch.int32, device=dev)

    def sglang(self, raw, page):
        if self.reindex:
            pos = torch.empty(self.rows, K, dtype=torch.int32, device=self.ilog.device)
            topk_transform_paged_hip(
                scores=self.ilog, seq_lens=self.bounds, page_table=self.zero_table, page_indices=pos,
                page_size=COMPACT, raw_indices=None,
            )  # fmt: skip
            map_compact_selection(
                compact_pos=pos,
                ids=self.ids,
                page_table=self.page_table,
                raw_out=raw,
                page_out=page,
            )
        else:
            topk_transform_paged_hip(
                scores=self.ilog, seq_lens=self.bounds, page_table=self.page_table, page_indices=page,
                page_size=64, raw_indices=raw,
            )  # fmt: skip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, nargs="+", default=[6, 48])
    ap.add_argument("--widths", type=int, nargs="+", default=[16384, 32768, 65536])
    ap.add_argument("--trials", type=int, default=4)
    a = ap.parse_args()
    dev = torch.device("cuda")
    cand = torch.empty(48 * MAX_SEGS * K * 2, dtype=torch.int32, device=dev)
    count = torch.zeros(48, dtype=torch.int32, device=dev)
    cases = [(r, w, False) for r in a.rows for w in a.widths] + [
        (r, COMPACT, True) for r in a.rows
    ]
    for rows, width, reindex in cases:
        launch = build_mono_topk(rows, reindex)
        segs = -(-width // SEG)
        assert segs <= MAX_SEGS
        bad = 0
        for trial in range(a.trials):
            g = torch.Generator(device=dev).manual_seed(1000 * trial + width + rows)
            c = Case(rows, width, reindex, g, dev, ties=trial % 2 == 1)
            want_raw = torch.full((rows, K), -2, dtype=torch.int32, device=dev)
            want_page = torch.full_like(want_raw, -2)
            c.sglang(want_raw, want_page)
            got_raw = torch.full_like(want_raw, -3)
            got_page = torch.full_like(want_raw, -3)
            args = (
                launch, c.ilog.data_ptr(), width, c.bounds.data_ptr(), c.page_table.data_ptr(),
                c.page_table.stride(0), c.ids.data_ptr(), c.ids.stride(0), got_raw.data_ptr(),
                got_page.data_ptr(), cand.data_ptr(), count.data_ptr(), segs,
            )  # fmt: skip
            _run_compiled(*args, torch.cuda.current_stream())
            torch.cuda.synchronize()
            for r in range(rows):
                for name, w, gt in (
                    ("raw", want_raw, got_raw),
                    ("page", want_page, got_page),
                ):
                    ws = set(x for x in w[r].tolist() if x >= 0)
                    gs = set(x for x in gt[r].tolist() if x >= 0)
                    if ws != gs or (gt[r] < -1).any():
                        bad += 1
                        if bad <= 4:
                            print(
                                f"rows {rows} width {width} reindex {reindex} trial {trial} row {r} {name}: "
                                f"{len(ws ^ gs)} differ, bound {int(c.bounds[r])}"
                            )
            if int(count.abs().sum()) != 0:
                print("count not reset")
                bad += 1
        # graph replay + timing
        g = torch.Generator(device=dev).manual_seed(5)
        c = Case(rows, width, reindex, g, dev, ties=False)
        raw = torch.empty(rows, K, dtype=torch.int32, device=dev)
        page = torch.empty_like(raw)
        args = (
            launch, c.ilog.data_ptr(), width, c.bounds.data_ptr(), c.page_table.data_ptr(), c.page_table.stride(0),
            c.ids.data_ptr(), c.ids.stride(0), raw.data_ptr(), page.data_ptr(), cand.data_ptr(), count.data_ptr(), segs,
        )  # fmt: skip
        s = torch.cuda.Stream()
        with torch.cuda.stream(s):
            for _ in range(3):
                _run_compiled(*args, s)
                c.sglang(torch.empty_like(raw), torch.empty_like(raw))
        torch.cuda.synchronize()
        gm, gs_ = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
        with torch.cuda.graph(gm, stream=s):
            for _ in range(20):
                _run_compiled(*args, s)
        ref_raw, ref_page = torch.empty_like(raw), torch.empty_like(raw)
        with torch.cuda.graph(gs_, stream=s):
            for _ in range(20):
                c.sglang(ref_raw, ref_page)
        times = []
        for gr in (gm, gs_):
            gr.replay()
            torch.cuda.synchronize()
            e0, e1 = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            e0.record()
            for _ in range(5):
                gr.replay()
            e1.record()
            torch.cuda.synchronize()
            times.append(e0.elapsed_time(e1) / 100 * 1e3)
        rep_ok = all(
            set(x for x in raw[r].tolist() if x >= 0)
            == set(x for x in ref_raw[r].tolist() if x >= 0)
            for r in range(rows)
        )
        print(
            f"rows {rows:2d} width {width:6d} reindex {int(reindex)}: bad {bad}, replay {'ok' if rep_ok else 'BAD'}, "
            f"mono {times[0]:.1f} us vs SGLang {times[1]:.1f} us a call (graph)"
        )


if __name__ == "__main__":
    main()
