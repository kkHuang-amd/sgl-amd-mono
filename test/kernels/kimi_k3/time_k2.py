# SPDX-License-Identifier: Apache-2.0
"""Kernel time and stage timeline of the Kimi-K3 layer launch at TP8, on the
synthetic weights of check_k2.py.

Prints the graph-replayed time a launch (20 launches a graph, the epoch bumped
between them, as a served step does) and, from a ``timeline=True`` build, when
each stage mark is reached across the CTAs that reach it (microseconds from the
earliest CTA start; marks are listed in ``layer.py``).

    torchrun --nproc-per-node 8 test/kernels/kimi_k3/time_k2.py
"""

import torch
from check_common import MAX_BLOCKS, NVB, ROWS, TP, moe_weights, setup
from check_k2 import DEADLINE_S, layer_weights

LAUNCHES = 20


def _launch(k2, w, scratch, peers, epoch, rank, key, out, prefix, tl=None):
    k2.k2_launch(
        key, core=w["core"], w_o=w["w_o"], prefix=prefix, blocks=w["blocks"],
        ares_nw=w["ares_nw"], ares_qk=w["ares_qk"], in_nw=w["in_nw"],
        w_gate=w["w_gate"], bias=w["bias"], w_ld=w["w_ld"], w_sgu=w["w_sgu"],
        w_sd=w["w_sd"], w13=w["w13"], w13s=w["w13s"], w2=w["w2"], w2s=w["w2s"],
        ln_w=w["ln_w"], w_up=w["w_up_shard"], out=out, scratch=scratch,
        peers=peers.addresses, rank=rank, epoch=epoch, layer=1, tl=tl,
    )  # fmt: skip


def main():
    rank, dev, cpu = setup()
    from sgl_amd_mono.common import guard
    from sgl_amd_mono.common.peer import PeerBuffer
    from sgl_amd_mono.common.plan import BLOCKS
    from sgl_amd_mono.kimi_k3.gfx950 import layer as k2

    peers = PeerBuffer(k2.flags_offset() + guard.FLAG_BYTES, cpu, rank, TP, dev)
    peers.bytes.zero_()
    epoch = torch.zeros(guard.GUARD_WORDS, dtype=torch.int32, device=dev)
    ticks = guard.deadline_ticks(DEADLINE_S)
    lo = ticks & 0xFFFFFFFF
    epoch[guard.GUARD_DEADLINE] = lo - (1 << 32) if lo >= 1 << 31 else lo
    epoch[guard.GUARD_DEADLINE + 1] = ticks >> 32
    scratch = torch.zeros(
        k2.scratch_bytes(k2.K2Build(tokens=ROWS, nblocks=MAX_BLOCKS)), dtype=torch.uint8, device=dev
    )
    w = moe_weights(rank, dev, guint=False)
    w.update(layer_weights(rank, dev))
    prefix = w["prefix"].clone()
    out = torch.empty_like(prefix)

    key = k2.K2Build(tokens=ROWS, nblocks=NVB, eps=w["eps"], out_eps=w["out_eps"], reset=False, guint=False)
    for _ in range(3):
        epoch[0].add_(1)
        _launch(k2, w, scratch, peers, epoch, rank, key, out, prefix)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(LAUNCHES):
            epoch[0].add_(1)
            _launch(k2, w, scratch, peers, epoch, rank, key, out, prefix)
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(10):
        graph.replay()
    e1.record()
    torch.cuda.synchronize()
    us = e0.elapsed_time(e1) * 1000 / (10 * LAUNCHES)

    tkey = k2.K2Build(tokens=ROWS, nblocks=NVB, eps=w["eps"], out_eps=w["out_eps"], reset=False,
                      guint=False, timeline=True)  # fmt: skip
    tl = torch.zeros(BLOCKS, k2.TL_POINTS, dtype=torch.int64, device=dev)
    epoch[0].add_(1)
    _launch(k2, w, scratch, peers, epoch, rank, tkey, out, prefix, tl=tl)  # compile
    torch.cuda.synchronize()
    # behind plain launches, so the previous launch's all-reduce has aligned the
    # ranks: an eager first launch would charge the host's rank skew to the AR
    tgraph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(tgraph):
        for _ in range(3):
            epoch[0].add_(1)
            _launch(k2, w, scratch, peers, epoch, rank, key, out, prefix)
        epoch[0].add_(1)
        tl.zero_()
        _launch(k2, w, scratch, peers, epoch, rank, tkey, out, prefix, tl=tl)
    for _ in range(3):
        tgraph.replay()
    torch.cuda.synchronize()
    if rank == 0:
        print(f"layer launch, rows {ROWS}: {us:.1f} us a launch (graph-replayed)", flush=True)
        t = tl.cpu()
        start = t[:, 0][t[:, 0] > 0].min()
        for p in range(k2.TL_POINTS):
            col = t[:, p]
            hit = col[col > 0]
            if hit.numel() == 0:
                continue
            rel = (hit - start).double() / 100.0  # 100 MHz -> us
            print(f"  mark {p:2d}: {hit.numel():3d} CTAs  first {rel.min():7.1f}  median {rel.median():7.1f}"
                  f"  last {rel.max():7.1f} us", flush=True)
    peers.close()


if __name__ == "__main__":
    main()
