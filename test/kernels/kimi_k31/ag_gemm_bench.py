# SPDX-License-Identifier: Apache-2.0
"""Fused all-gather -> tile GEMM (``stages/gemm_coll.ag_gemm``) at TP8: check and time.

in_proj shape by default (n 6288, k 7168): each rank owns x[:, 896 r ..] and its
own W shard. Checks out against the f32 reference (rel L2) on rotating inputs;
times, a launch, graph-replayed: fused | coll AG + tile GEMM | RCCL all_gather +
F.linear, plus the fused GEMM on the ag region without push or wait (p1).
Every column includes the output memset a linbf split-K GEMM needs.

    torchrun --nproc-per-node 8 test/kernels/kimi_k31/ag_gemm_bench.py \
        --cfgs 16:16:128:1:8:4,128:128:128:2:4:4 --modes linbf,part
"""

import argparse
import os

import torch
import torch.distributed as dist
import torch.nn.functional as F
from gemm_coll_bench import BLOCKS, graph_us, stress_loop

TP = 8
SETS = 3
BF = torch.bfloat16


def run(args, cfg, mode, rank, dev, cpu):
    from sgl_amd_mono.common import guard
    from sgl_amd_mono.common.peer import PeerBuffer
    from sgl_amd_mono.kimi_k31.gfx950.stages import coll, gemm_coll, tile_mma

    m, bm, bn, wm, wn, ks = cfg
    n, k, nl = args.n, args.k, args.launches
    sl = k // TP
    part = mode == "part" and ks > 1
    peers = PeerBuffer(coll.peer_bytes(m, k, "flag"), cpu, rank, TP, dev)
    peers.bytes.zero_()
    epoch = torch.zeros(guard.GUARD_WORDS, dtype=torch.int32, device=dev)
    ticks = guard.deadline_ticks(10.0)
    lo = ticks & 0xFFFFFFFF
    epoch[guard.GUARD_DEADLINE] = lo - (1 << 32) if lo >= 1 << 31 else lo
    epoch[guard.GUARD_DEADLINE + 1] = ticks >> 32

    g = torch.Generator(device=dev).manual_seed(5 + rank)
    w = (torch.randn(n, k, generator=g, device=dev) / k**0.5).to(BF)
    xsets, refs = [], []
    for s in range(SETS):
        gs = torch.Generator(device=dev).manual_seed(100 + s)
        full = torch.randn(nl, m, k, generator=gs, device=dev).to(BF)
        xsets.append(full[..., rank * sl : (rank + 1) * sl].contiguous())
        refs.append(torch.einsum("lmk,nk->lmn", full.float(), w.float()))
    x = xsets[0].clone()
    out = torch.zeros(
        nl, (ks if part else 1) * m * n * (2 if part else 1), dtype=BF, device=dev
    )
    cnt = torch.zeros(1, dtype=torch.int32, device=dev)
    pa = peers.addresses.data_ptr()

    def result():
        if part:
            return out.view(torch.float32).view(nl, ks, m, n).sum(1)
        return out.view(nl, m, n).float()

    delay = torch.zeros(nl, 2 * BLOCKS, dtype=torch.int32, device=dev)
    step = args.step
    assert nl % step == 0

    def fused_fn(launch, step=1):
        def fn(i):
            if mode == "linbf" and ks > 1:
                out[i].zero_()
            if i % step == 0:
                epoch[0].add_(1)
            launch(
                x[i].data_ptr(),
                w.data_ptr(),
                out[i].data_ptr(),
                cnt.data_ptr(),
                pa,
                rank,
                epoch.data_ptr(),
                delay[i].data_ptr(),
                i % step,
                stream=torch.cuda.current_stream(),
            )

        return fn

    stress = args.stress > 0
    one = fused_fn(
        gemm_coll.ag_gemm(
            m, n, k, bm, bn, wm, wn, ks, mode=mode, launches=step, stress=stress
        ),
        step,
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
            None,
            refs,
            delay,
            rank,
            dev,
            f"m {m:3d} t{bm}x{bn} w{wm}x{wn} ks{ks} {mode:5s}",
            result=result,
        )
    worst = 0.0
    for r in range(SETS * 2):
        s = r % SETS
        x.copy_(xsets[s])
        graph.replay()
        y = result()
        worst = max(worst, ((y - refs[s]).norm() / refs[s].norm()).item())
    torch.cuda.synchronize()

    p1 = fused_fn(gemm_coll.ag_gemm(m, n, k, bm, bn, wm, wn, ks, mode=mode, phases=1))
    p1(0)
    torch.cuda.synchronize()
    us_p1 = graph_us(p1, nl)[1]

    ag = coll.standalone(m, k, "ag", "tag")
    tile = tile_mma.standalone(
        m, n, k, bm, bn, wm, wn, ks, mode=mode if ks > 1 else "part"
    )
    peers2 = PeerBuffer(coll.peer_bytes(m, k, "tag"), cpu, rank, TP, dev)
    peers2.bytes.zero_()
    xfull = torch.zeros(m, k, dtype=BF, device=dev)
    dummy = torch.zeros(1, dtype=torch.int32, device=dev)

    def sep(i):
        if mode == "linbf" and ks > 1:
            out[i].zero_()
        epoch[0].add_(1)
        ag(
            x[i].data_ptr(),
            xfull.data_ptr(),
            peers2.addresses.data_ptr(),
            rank,
            epoch.data_ptr(),
            dummy.data_ptr(),
            0,
            stream=torch.cuda.current_stream(),
        )
        tile(
            xfull.data_ptr(),
            w.data_ptr(),
            out[i].data_ptr(),
            stream=torch.cuda.current_stream(),
        )

    for i in range(nl):
        sep(i)
    torch.cuda.synchronize()
    _, us_sep = graph_us(sep, nl)

    gath = torch.empty(TP, m, sl, dtype=BF, device=dev)
    yref = torch.empty(m, n, dtype=BF, device=dev)

    def ref(i):
        dist.all_gather_into_tensor(gath, x[i])
        yref.copy_(F.linear(gath.permute(1, 0, 2).reshape(m, k), w))

    for i in range(nl):
        ref(i)
    torch.cuda.synchronize()
    _, us_ref = graph_us(ref, nl)

    stats = torch.tensor([worst, us_fused, us_p1, us_sep, us_ref], device=dev)
    every = [torch.empty_like(stats) for _ in range(TP)]
    dist.all_gather(every, stats)
    if rank == 0:
        e = torch.stack(every).max(0).values.tolist()
        print(
            f"m {m:3d} t{bm}x{bn} w{wm}x{wn} ks{ks} {mode:5s}  rel_l2 {e[0]:.2e}  us: fused {e[1]:6.1f} "
            f"(gemm on ag region {e[2]:5.1f}) | coll ag + tile {e[3]:6.1f} | rccl ag + linear {e[4]:6.1f}  "
            f"{'PASS' if e[0] < 1e-2 else 'FAIL'}",
            flush=True,
        )
    peers.close()
    peers2.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--cfgs",
        default="16:16:128:1:8:4,32:32:128:2:4:4,128:128:128:2:4:4,384:192:128:4:2:2",
        help="m:bm:bn:wm:wn:ks,...",
    )
    p.add_argument("--modes", default="linbf")
    p.add_argument("--n", type=int, default=6288)
    p.add_argument("--k", type=int, default=7168)
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
