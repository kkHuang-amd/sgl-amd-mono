# SPDX-License-Identifier: Apache-2.0
"""Column-slice collectives (``kimi_k31/gfx950/stages/coll.py``) at TP8: check and time.

Every launch of a graph has its own inputs, and three input sets rotate between
replays (the parity halves alternate by 2), so data left over from an earlier
launch never passes for this one's. Results must match bit for bit: the
reference sums the ranks' bf16 partials in f32 in rank order, as the kernel does.

    torchrun --nproc-per-node 8 test/kernels/kimi_k31/coll_bench.py \
        --shapes 8x7168,128x7168 --kinds ar,ag,rs --fmts tag,flag [--time] [--stress 2000 --delay 20000]

``--step S`` runs the graph as steps of S launches (the epoch bumped once a step,
slot = launch index in the step), as the layer launches of a served step do;
``--parity epoch`` shares one buffer half within a step (shows the hazard).
"""

import argparse
import os

import torch
import torch.distributed as dist

TP = 8
SETS = 3
BF = torch.bfloat16


def inputs(kind, m, h, s, launches, dev):
    """All ranks' inputs of set ``s`` [launches, TP, ...] (every rank builds them all)."""
    g = torch.Generator(device=dev).manual_seed(1000 + s)
    cols = h // TP if kind == "ag" else h
    return (torch.randn(launches, TP, m, cols, generator=g, device=dev) * 0.5).to(BF)


def reference(kind, xs, rank):
    """[launches, ...] this rank's expected output from the all-rank inputs."""
    if kind == "ag":
        return torch.cat(list(xs.unbind(1)), dim=-1)
    acc = xs[:, 0].float()
    for r in range(1, TP):
        acc = acc + xs[:, r].float()
    full = acc.to(BF)
    if kind == "ar":
        return full
    sl = full.shape[-1] // TP
    return full[..., rank * sl : (rank + 1) * sl].contiguous()


def run_case(args, kind, fmt, m, h, rank, dev, cpu):
    from sgl_amd_mono.common import guard
    from sgl_amd_mono.common.peer import PeerBuffer
    from sgl_amd_mono.common.plan import BLOCKS
    from sgl_amd_mono.kimi_k31.gfx950.stages import coll

    stress = args.stress > 0
    nl = args.launches
    launch = coll.standalone(m, h, kind, fmt, stress, args.step, args.parity)
    peers = PeerBuffer(coll.peer_bytes(m, h, fmt), cpu, rank, TP, dev)
    peers.bytes.zero_()
    epoch = torch.zeros(guard.GUARD_WORDS, dtype=torch.int32, device=dev)
    ticks = guard.deadline_ticks(args.deadline)
    lo = ticks & 0xFFFFFFFF
    epoch[guard.GUARD_DEADLINE] = lo - (1 << 32) if lo >= 1 << 31 else lo
    epoch[guard.GUARD_DEADLINE + 1] = ticks >> 32

    sets = [inputs(kind, m, h, s, nl, dev) for s in range(SETS)]
    xsets = [s[:, rank].contiguous() for s in sets]
    refs = [reference(kind, s, rank) for s in sets]
    del sets
    x = torch.empty_like(xsets[0])
    y = torch.empty_like(refs[0])
    delay = torch.zeros(nl, 2 * BLOCKS, dtype=torch.int32, device=dev)
    gen = torch.Generator(device=dev).manual_seed(77 + rank)

    assert nl % args.step == 0

    def one(i):
        if i % args.step == 0:
            epoch[0].add_(1)
        launch(
            x[i].data_ptr(),
            y[i].data_ptr(),
            peers.addresses.data_ptr(),
            rank,
            epoch.data_ptr(),
            delay[i].data_ptr(),
            i % args.step,
            stream=torch.cuda.current_stream(),
        )

    x.copy_(xsets[0])
    for i in range(nl):  # compile + eager check of the first set
        one(i)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for i in range(nl):
            one(i)

    bad = torch.zeros((), dtype=torch.int64, device=dev)
    bad += (y != refs[0]).sum()
    reps = max(args.stress, SETS * 2)
    for r in range(reps):
        k = r % SETS
        x.copy_(xsets[k])
        if stress:
            delay.random_(0, args.delay, generator=gen)
        graph.replay()
        bad += (y != refs[k]).sum()
    torch.cuda.synchronize()

    us = float("nan")
    if args.time:
        x.copy_(xsets[0])
        delay.zero_()
        for _ in range(3):
            graph.replay()
        torch.cuda.synchronize()
        dist.barrier()
        e0, e1 = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        e0.record()
        for _ in range(args.reps):
            graph.replay()
        e1.record()
        torch.cuda.synchronize()
        us = e0.elapsed_time(e1) * 1000 / (args.reps * nl)

    stats = torch.tensor([float(bad.item()), us], device=dev)
    every = [torch.empty_like(stats) for _ in range(TP)]
    dist.all_gather(every, stats)
    if rank == 0:
        nbad = int(sum(e[0].item() for e in every))
        worst = max(e[1].item() for e in every)
        mb = m * h * 2 / 1e6
        launches = nl * (reps + 1)
        print(
            f"{kind:2s} {fmt:4s} {m:4d} x {h:5d} {mb:5.2f} MB  launches {launches:7d}  "
            f"mismatches {nbad:8d}  {worst:7.1f} us  {'PASS' if nbad == 0 else 'FAIL'}",
            flush=True,
        )
    peers.close()
    return bad.item() == 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--shapes", default="8x7168,32x7168,128x3584,128x7168,128x10752,384x7168"
    )
    p.add_argument("--kinds", default="ar,ag,rs")
    p.add_argument("--fmts", default="tag,flag")
    p.add_argument("--launches", type=int, default=20)
    p.add_argument("--time", action="store_true")
    p.add_argument("--reps", type=int, default=10)
    p.add_argument(
        "--stress", type=int, default=0, help="replays with random CTA delays"
    )
    p.add_argument("--delay", type=int, default=20000, help="max delay, 100 MHz ticks")
    p.add_argument("--deadline", type=float, default=10.0)
    p.add_argument(
        "--step", type=int, default=1, help="launches a step (one epoch bump)"
    )
    p.add_argument("--parity", default="launch", choices=("launch", "epoch"))
    args = p.parse_args()

    rank = int(os.environ["RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl")
    cpu = dist.new_group(backend="gloo")
    dev = torch.device("cuda", rank)
    ok = True
    for shape in args.shapes.split(","):
        m, h = (int(v) for v in shape.split("x"))
        for kind in args.kinds.split(","):
            for fmt in args.fmts.split(","):
                ok &= run_case(args, kind, fmt, m, h, rank, dev, cpu)
    bad = torch.tensor([int(not ok)], device=dev)
    dist.all_reduce(bad, op=dist.ReduceOp.MAX)
    dist.destroy_process_group()
    raise SystemExit(int(bad.item()))


if __name__ == "__main__":
    main()
