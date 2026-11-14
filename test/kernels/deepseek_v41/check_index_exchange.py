# SPDX-License-Identifier: Apache-2.0
"""The index exchange (``gfx950/index_exchange``) on every TP rank: each rank fills its own
requests' rows of the sparse raw / page indices, the exchange brings every other rank's rows;
all rows must equal their owner's, eager and under a graph replay; then the time a call.

torchrun --nproc-per-node 2 --master-port 29621 check_index_exchange.py [--trials 20]
"""

from __future__ import annotations

import argparse
import os

import torch
import torch.distributed as dist


def ranges_of(bs: int, per_req: int, tp: int):
    return [(q * bs // tp * per_req, (q + 1) * bs // tp * per_req) for q in range(tp)]


def expected(rows: int, seed: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator(device=device).manual_seed(seed)
    raw = torch.randint(0, 2**30, (rows, 512), generator=g, device=device, dtype=torch.int32)
    page = torch.randint(0, 2**30, (rows, 512), generator=g, device=device, dtype=torch.int32)
    return raw, page


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=20)
    args = ap.parse_args()
    dist.init_process_group("nccl")
    rank, tp = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)))
    device = torch.device("cuda")
    gloo = dist.new_group(backend="gloo")
    from sgl_amd_mono.deepseek_v41.gfx950.index_exchange import IndexExchange

    ex = IndexExchange(gloo, rank, tp, device, deadline_s=5.0, check=True)
    per_req, cap = 6, 48
    # SGLang's buffers: wider than the step, a row stride past 512
    raw_buf = torch.full((cap + 4, 516), -1, dtype=torch.int32, device=device)
    page_buf = torch.full((cap + 4, 516), -1, dtype=torch.int32, device=device)
    raw, page = raw_buf[:, :512], page_buf[:, :512]
    fails = []
    for t in range(args.trials):
        for bs in range(tp, 9):
            rows = bs * per_req
            want_raw, want_page = expected(rows, 1000 * t + bs, device)
            rg = ranges_of(bs, per_req, tp)
            lo, hi = rg[rank]
            raw.fill_(-7), page.fill_(-7)
            raw[lo:hi].copy_(want_raw[lo:hi]), page[lo:hi].copy_(want_page[lo:hi])
            ex(raw, page, rg)
            torch.cuda.synchronize()
            if not (torch.equal(raw[:rows], want_raw) and torch.equal(page[:rows], want_page)):
                fails.append((t, bs))
            if not (bool((raw_buf[:, 512:] == -1).all()) and bool((raw[rows:] == -7).all())):
                fails.append((t, bs, "wrote outside"))
    # a graph replay: the call tags advance on the device
    bs = 8
    rows, rg = bs * per_req, ranges_of(bs, per_req, tp)
    lo, hi = rg[rank]
    want_raw, want_page = expected(rows, 7, device)
    graph = torch.cuda.CUDAGraph()
    dist.barrier()
    with torch.cuda.graph(graph):
        ex(raw, page, rg)
    replay_bad = 0
    for n in range(6):
        raw.fill_(-3), page.fill_(-3)
        raw[lo:hi].copy_(want_raw[lo:hi] + n), page[lo:hi].copy_(want_page[lo:hi] + n)
        torch.cuda.synchronize()
        dist.barrier()
        graph.replay()
        torch.cuda.synchronize()
        replay_bad += not (
            torch.equal(raw[:rows], want_raw + n) and torch.equal(page[:rows], want_page + n)
        )
    # the check: a matching replicated selection counts no mismatch, one wrong row its vectors
    bs = 8
    rows, rg = bs * per_req, ranges_of(bs, per_req, tp)
    lo, hi = rg[rank]
    want_raw, want_page = expected(rows, 11, device)
    ref_raw, ref_page = want_raw.clone(), want_page.clone()
    check_bad = []
    for wrong in (False, True):
        ex.check.zero_()
        if wrong:
            ref_page[rows - 1, :4] += 1  # the last request's last row: one vector
        raw.fill_(-5), page.fill_(-5)
        raw[lo:hi].copy_(want_raw[lo:hi]), page[lo:hi].copy_(want_page[lo:hi])
        ex(raw, page, rg)
        ex.compare((raw, page), (ref_raw, ref_page), rows, rg[rank])
        torch.cuda.synchronize()
        own, peers, seen = ex.checked()
        # the wrong row is the last rank's own row, a peer's row everywhere else
        want = (wrong and rank == tp - 1, wrong and rank != tp - 1)
        if seen != rows * 2 * 128 or (own > 0, peers > 0) != want:
            check_bad.append((wrong, own, peers, seen))
    # time: 8 calls a step in one graph, as the index sources of a step
    timing = {}
    for bs in (tp, 4 if tp <= 4 else tp, 8):
        rg = ranges_of(bs, per_req, tp)
        g8 = torch.cuda.CUDAGraph()
        dist.barrier()
        with torch.cuda.graph(g8):
            for _ in range(8):
                ex(raw, page, rg)
        g8.replay()
        torch.cuda.synchronize()
        dist.barrier()
        ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        ev[0].record()
        for _ in range(20):
            g8.replay()
        ev[1].record()
        torch.cuda.synchronize()
        timing[bs] = ev[0].elapsed_time(ev[1]) * 1e3 / 160
    print(
        f"rank {rank} tp {tp}: fails {len(fails)} {fails[:3]} | replay bad {replay_bad} | "
        f"check {check_bad or 'ok'} | "
        + " ".join(f"bs{b} {us:.1f} us a call" for b, us in timing.items()),
        flush=True,
    )
    torch.cuda.synchronize()
    dist.barrier()  # no rank frees its buffer while a peer's kernel still writes it
    ex.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
