# SPDX-License-Identifier: Apache-2.0
"""Fused tile GEMM -> reduce-scatter (``stages/gemm_coll.py``) at TP8: check and time.

Checks y against the f32 reference (rel L2) on rotating inputs, and with
``part`` that two replays of the same inputs agree bit for bit. Times, a
launch, graph-replayed: fused | tile GEMM + memset + coll RS | F.linear + RCCL.

    torchrun --nproc-per-node 8 test/kernels/kimi_k31/gemm_coll_bench.py \
        --cfgs 16:16:128:1:8:4,32:32:128:2:4:2,128:64:64:2:4:1,384:192:64:4:2:1 --modes linbf,part

GC_FMT=tag|flag (default flag) picks the push format; GC_PHASES=1,4,2 also times
the GEMM alone, + the arrival, + the pushes (no reduce).
"""

import argparse
import os

import torch
import torch.distributed as dist
import torch.nn.functional as F

BLOCKS = 256

TP = 8
SETS = 3
BF = torch.bfloat16


def graph_us(fn, nl, reps=10):
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for i in range(nl):
            fn(i)
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    dist.barrier()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(reps):
        g.replay()
    e1.record()
    torch.cuda.synchronize()
    return g, e0.elapsed_time(e1) * 1000 / (reps * nl)


def stress_loop(args, graph, x, xsets, y, refs, delay, rank, dev, label, result=None):
    """Replays with random CTA delays and rotating inputs; every replay's outputs
    are checked (rel L2 a launch). Returns nothing; prints the worst over ranks."""
    gen = torch.Generator(device=dev).manual_seed(91 + rank)
    worst = torch.zeros((), device=dev)
    for r in range(args.stress):
        s = r % SETS
        x.copy_(xsets[s])
        delay.random_(0, args.delay, generator=gen)
        graph.replay()
        out = result() if result else y.float()
        err = (out - refs[s]).flatten(1).norm(dim=1) / refs[s].flatten(1).norm(dim=1)
        worst = torch.maximum(worst, err.max())
    torch.cuda.synchronize()
    every = [torch.empty_like(worst) for _ in range(TP)]
    dist.all_gather(every, worst)
    if rank == 0:
        e = max(v.item() for v in every)
        print(
            f"stress {label}  launches {args.stress * x.shape[0]}  step {args.step}  worst rel_l2 {e:.2e}  "
            f"{'PASS' if e < 1e-2 else 'FAIL'}",
            flush=True,
        )


def run(args, cfg, mode, rank, dev, cpu):
    from sgl_amd_mono.common import guard
    from sgl_amd_mono.common.peer import PeerBuffer
    from sgl_amd_mono.kimi_k31.gfx950.stages import coll, gemm_coll, tile_mma

    m, bm, bn, wm, wn, ks = cfg
    n, k, nl = args.n, args.k, args.launches
    sl = n // TP
    fmt = os.environ.get("GC_FMT", "flag")
    stress = args.stress > 0
    fused = gemm_coll.gemm_rs(
        m,
        n,
        k,
        bm,
        bn,
        wm,
        wn,
        ks,
        mode=mode,
        fmt=fmt,
        launches=args.step,
        stress=stress,
    )
    peers = PeerBuffer(coll.peer_bytes(m, n, fmt), cpu, rank, TP, dev)
    peers.bytes.zero_()
    epoch = torch.zeros(guard.GUARD_WORDS, dtype=torch.int32, device=dev)
    ticks = guard.deadline_ticks(10.0)
    lo = ticks & 0xFFFFFFFF
    epoch[guard.GUARD_DEADLINE] = lo - (1 << 32) if lo >= 1 << 31 else lo
    epoch[guard.GUARD_DEADLINE + 1] = ticks >> 32

    g = torch.Generator(device=dev).manual_seed(5 + rank)
    w = (torch.randn(n, k, generator=g, device=dev) / k**0.5).to(BF)
    xsets = [torch.randn(nl, m, k, generator=g, device=dev).to(BF) for _ in range(SETS)]
    refs = []
    for xs in xsets:
        p = torch.einsum("lmk,nk->lmn", xs.float(), w.float())
        dist.all_reduce(p)
        refs.append(p[..., rank * sl : (rank + 1) * sl].contiguous())
    x = xsets[0].clone()
    y = torch.zeros(nl, m, sl, dtype=BF, device=dev)
    acc = torch.zeros(
        gemm_coll.scratch_bytes(m, n, ks, mode), dtype=torch.uint8, device=dev
    )
    cnt = torch.zeros(gemm_coll.counters(m, n, bm, bn), dtype=torch.int32, device=dev)
    pa = peers.addresses.data_ptr()
    delay = torch.zeros(nl, 2 * BLOCKS, dtype=torch.int32, device=dev)
    assert nl % args.step == 0

    def one(i):
        if i % args.step == 0:
            epoch[0].add_(1)
        fused(
            x[i].data_ptr(),
            w.data_ptr(),
            y[i].data_ptr(),
            acc.data_ptr(),
            cnt.data_ptr(),
            pa,
            rank,
            epoch.data_ptr(),
            delay[i].data_ptr(),
            i % args.step,
            stream=torch.cuda.current_stream(),
        )

    for i in range(nl):
        one(i)
    torch.cuda.synchronize()
    graph, us_fused = graph_us(one, nl)
    if stress:
        return stress_loop(
            args,
            graph,
            x,
            xsets,
            y,
            refs,
            delay,
            rank,
            dev,
            f"m {m:3d} t{bm}x{bn} w{wm}x{wn} ks{ks} {mode:5s} {fmt}",
        )
    worst, det, prev = 0.0, True, {}
    for r in range(SETS * 2):
        s = r % SETS
        x.copy_(xsets[s])
        graph.replay()
        e = ((y.float() - refs[s]).norm() / refs[s].norm()).item()
        worst = max(worst, e)
        if s in prev:
            det &= bool(torch.equal(y, prev[s]))
        prev[s] = y.clone()
    torch.cuda.synchronize()
    us_ph = []
    for ph in [int(v) for v in os.environ.get("GC_PHASES", "1,2").split(",")]:
        part_launch = gemm_coll.gemm_rs(
            m, n, k, bm, bn, wm, wn, ks, mode=mode, phases=ph, fmt=fmt
        )

        def one_ph(i, part_launch=part_launch):
            epoch[0].add_(1)
            part_launch(
                x[i].data_ptr(),
                w.data_ptr(),
                y[i].data_ptr(),
                acc.data_ptr(),
                cnt.data_ptr(),
                pa,
                rank,
                epoch.data_ptr(),
                delay[i].data_ptr(),
                0,
                stream=torch.cuda.current_stream(),
            )

        one_ph(0)
        torch.cuda.synchronize()
        us_ph.append(graph_us(one_ph, nl)[1])
    # p1 / p2 leave the counters and (linbf) tiles dirty: restore them
    acc.zero_()
    cnt.zero_()

    # separate launches: tile GEMM (same merge) -> zero -> coll RS
    tile = tile_mma.standalone(
        m, n, k, bm, bn, wm, wn, ks, mode=mode if ks > 1 else "part"
    )
    rs = coll.standalone(m, n, "rs", "tag")
    part = mode == "part" and ks > 1
    out = torch.zeros(
        (ks if part else 1) * m * n * (2 if part else 1), dtype=BF, device=dev
    )
    dummy = torch.zeros(1, dtype=torch.int32, device=dev)
    peers2 = PeerBuffer(coll.peer_bytes(m, n, "tag"), cpu, rank, TP, dev)
    peers2.bytes.zero_()
    psum = torch.zeros(m, n, dtype=BF, device=dev)

    def sep(i):
        if mode == "linbf" and ks > 1:
            out.zero_()
        tile(
            x[i].data_ptr(),
            w.data_ptr(),
            out.data_ptr(),
            stream=torch.cuda.current_stream(),
        )
        src = out
        if part:
            src = psum
            psum.copy_(out.view(torch.float32).view(ks, m, n).sum(0))
        epoch[0].add_(1)
        rs(
            src.data_ptr(),
            y[i].data_ptr(),
            peers2.addresses.data_ptr(),
            rank,
            epoch.data_ptr(),
            dummy.data_ptr(),
            0,
            stream=torch.cuda.current_stream(),
        )

    for i in range(nl):
        sep(i)
    torch.cuda.synchronize()
    _, us_sep = graph_us(sep, nl)

    full = torch.empty(m * n, dtype=BF, device=dev)
    shard = torch.empty(m * sl, dtype=BF, device=dev)

    def ref(i):
        full.copy_(F.linear(x[i], w).view(-1))
        dist.reduce_scatter_tensor(shard, full)

    for i in range(nl):
        ref(i)
    torch.cuda.synchronize()
    _, us_ref = graph_us(ref, nl)

    stats = torch.tensor(
        [worst, float(det), us_fused, us_sep, us_ref] + us_ph, device=dev
    )
    every = [torch.empty_like(stats) for _ in range(TP)]
    dist.all_gather(every, stats)
    if rank == 0:
        e = torch.stack(every)
        ok = e[:, 0].max().item() < 1e-2 and (
            mode != "part" or e[:, 1].min().item() == 1.0
        )
        print(
            f"{fmt:4s} m {m:3d} t{bm}x{bn} w{wm}x{wn} ks{ks} {mode:5s}  rel_l2 {e[:, 0].max().item():.2e}"
            f"{'  det ' + str(bool(e[:, 1].min().item())) if mode == 'part' else ''}  us: fused "
            f"{e[:, 2].max().item():6.1f} (phases "
            + " ".join(f"{v:5.1f}" for v in e[:, 5:].max(0).values.tolist())
            + ")"
            f" | tile+zero+rs {e[:, 3].max().item():6.1f} | linear+rccl "
            f"{e[:, 4].max().item():6.1f}  {'PASS' if ok else 'FAIL'}",
            flush=True,
        )
    peers.close()
    peers2.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--cfgs",
        default="16:16:128:1:8:4,32:32:128:2:4:2,128:64:64:2:4:1,384:192:64:4:2:1",
        help="m:bm:bn:wm:wn:ks,...",
    )
    p.add_argument("--modes", default="linbf,part")
    p.add_argument("--n", type=int, default=7168)
    p.add_argument("--k", type=int, default=1536)
    p.add_argument("--launches", type=int, default=8)
    p.add_argument(
        "--stress",
        type=int,
        default=0,
        help="replays with random CTA delays (no timing)",
    )
    p.add_argument("--delay", type=int, default=5000, help="max delay, 100 MHz ticks")
    p.add_argument(
        "--step", type=int, default=1, help="launches a step (one epoch bump)"
    )
    args = p.parse_args()
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl")
    cpu = dist.new_group(backend="gloo")
    dev = torch.device("cuda", rank)
    for spec in args.cfgs.split(","):
        cfg = tuple(int(v) for v in spec.split(":"))
        for mode in args.modes.split(","):
            run(args, cfg, mode, rank, dev, cpu)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
