# SPDX-License-Identifier: Apache-2.0
"""The post-attention chain launch (``kimi_k31/gfx950/chain.py``) at TP8: check and time.

o_proj -> RS + residual -> RMSNorm (sum of squares exchanged) -> AG -> MoE front,
one launch. Checks the new residual slice and the front output against an f32
reference on rotating inputs; times it against the standard path (F.linear ->
RCCL all-reduce -> add -> F.rms_norm -> F.linear), graph-replayed.

    torchrun --nproc-per-node 8 test/kernels/kimi_k31/chain_bench.py \
        --cfgs 16/16:128:1:8:4/16:128:1:8:8,128/64:64:2:4:1/128:128:2:4:8 [--plan]
"""

import argparse
import os

import torch
import torch.distributed as dist
import torch.nn.functional as F
from gemm_coll_bench import BLOCKS, graph_us

TP = 8
SETS = 3
BF = torch.bfloat16
H, KO, EPS = 7168, 1536, 1e-6


def attn_res_ref(prefix, bank, nvb, w_s, w_p, w_o):
    """SGLang's aggregation (``aggregate_stream_torch``) + out_norm, on [..., H] rows."""
    if nvb == 0:
        mixed = prefix
    else:
        rows = torch.cat([bank[..., :nvb, :], prefix.unsqueeze(-2)], dim=-2)
        normed = F.rms_norm(rows.float(), (H,), w_s.float(), EPS).to(BF)
        scores = (normed.float() @ w_p.float()).to(BF)
        probs = torch.softmax(scores.float(), dim=-1)
        mixed = (probs.unsqueeze(-1) * rows.float()).sum(dim=-2).to(BF)
    return F.rms_norm(mixed.float(), (H,), w_o.float(), EPS).to(BF)


def timeline(args, chain, m, go, gf, nf, mode, step, one, tl, nl, rank, knobs):
    """One marked launch (the last of a step) behind warm launches, in a graph; per
    mark over every rank's CTAs that reach it: first / median / last, us from that
    rank's earliest CTA start."""
    tlaunch, _ = chain.post_attn(
        m, go, gf, nf, mode=mode, launches=step, eps=EPS, timeline=True, **knobs
    )
    one(step - 1, tlaunch)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for i in range(step - 1):
            one(i)
        tl.zero_()
        one(step - 1, tlaunch)
    for _ in range(5):
        g.replay()
    torch.cuda.synchronize()
    every = [torch.empty_like(tl) for _ in range(TP)]
    dist.all_gather(every, tl)
    if rank != 0:
        return
    t = torch.stack(every).cpu()  # [rank, cta, mark]
    # the GPUs' clocks are not aligned: each rank's marks from its own earliest start
    t0 = torch.stack([t[r, :, 0][t[r, :, 0] > 0].min() for r in range(TP)]).view(
        TP, 1, 1
    )
    t = torch.where(t > 0, t - t0 + 1, torch.zeros_like(t))
    t0 = 1
    print(f"timeline m {m}: mark  CTAs  first  median  last (us)", flush=True)
    for k, name in enumerate(chain.MARKS):
        v = t[..., k]
        v = v[v > 0]
        if v.numel():
            r = (v - t0).double() / 100.0
            print(
                f"  {k:2d} {name:12s} {v.numel():5d} {r.min():7.1f} {r.median():7.1f} {r.max():7.1f}",
                flush=True,
            )
    ends = [(t[r, :, 9][t[r, :, 9] > 0].max() - t0).item() / 100.0 for r in range(TP)]
    print("  front end by rank: " + " ".join(f"{e:5.1f}" for e in ends), flush=True)


def run(args, spec, mode, rank, dev, cpu):
    from sgl_amd_mono.common import guard
    from sgl_amd_mono.common.peer import PeerBuffer
    from sgl_amd_mono.kimi_k31.gfx950 import chain

    ms, so, sf = spec.split("/")
    m = int(ms)
    go = chain.Gemm(*(int(v) for v in so.split(":")))
    gf = chain.Gemm(*(int(v) for v in sf.split(":")))
    nf, nl, step = args.nf, args.launches, args.step
    sl = H // TP
    knobs = dict(
        front_top=bool(int(os.environ.get("CH_TOP", "1"))),
        prefetch=bool(int(os.environ.get("CH_PF", "0"))),
        x_probe=bool(int(os.environ.get("CH_XPROBE", "0"))),
    )
    nvb, nbk, write = args.nvb, args.nbk, args.write
    knobs.update(nvb=nvb, nbk=nbk, write=write)
    launch, pl = chain.post_attn(
        m, go, gf, nf, mode=mode, launches=step, eps=EPS, **knobs
    )
    if rank == 0 and args.plan:
        print(pl.describe(), flush=True)
    peers = PeerBuffer(pl.peer_bytes(), cpu, rank, TP, dev)
    peers.bytes.zero_()
    epoch = torch.zeros(guard.GUARD_WORDS, dtype=torch.int32, device=dev)
    ticks = guard.deadline_ticks(10.0)
    lo = ticks & 0xFFFFFFFF
    epoch[guard.GUARD_DEADLINE] = lo - (1 << 32) if lo >= 1 << 31 else lo
    epoch[guard.GUARD_DEADLINE + 1] = ticks >> 32

    g = torch.Generator(device=dev).manual_seed(5 + rank)
    wo = (torch.randn(H, KO, generator=g, device=dev) / KO**0.5).to(BF)
    wf = (torch.randn(nf, H, generator=g, device=dev) / H**0.5).to(BF)
    gs = torch.Generator(device=dev).manual_seed(11)
    w_ln = (1.0 + 0.1 * torch.randn(H, generator=gs, device=dev)).to(BF)
    w_s = (1.0 + 0.1 * torch.randn(H, generator=gs, device=dev)).to(BF)
    w_p = (torch.randn(H, generator=gs, device=dev) / H**0.5 * 4).to(BF)
    cw = (w_s.float() * w_p.float()).contiguous()
    asets, rsets, bsets, refs = [], [], [], []
    for s in range(SETS):
        a = torch.randn(nl, m, KO, generator=g, device=dev).to(BF)
        gb = torch.Generator(device=dev).manual_seed(200 + s)
        res = torch.randn(nl, m, H, generator=gb, device=dev).to(BF)
        bk = torch.randn(nl, m, nbk, H, generator=gb, device=dev).to(BF)
        y = torch.einsum("lmk,nk->lmn", a.float(), wo.float())
        dist.all_reduce(y)
        h = (y + res.float()).to(BF)
        xn = attn_res_ref(h, bk, nvb, w_s, w_p, w_ln)
        refs.append(
            (
                h[..., rank * sl : (rank + 1) * sl].float(),
                torch.einsum("lmk,nk->lmn", xn.float(), wf.float()),
            )
        )
        asets.append(a)
        rsets.append(res)
        bsets.append(bk.view(nl, m, nbk, TP, sl)[:, :, :, rank].contiguous())
    bank = bsets[0].clone()
    a = asets[0].clone()
    res_full = rsets[0].clone()
    res = res_full[..., rank * sl : (rank + 1) * sl].contiguous()
    res_out = torch.zeros(nl, m, sl, dtype=BF, device=dev)
    fpart = mode == "part" and gf.ks > 1
    out = torch.zeros(
        nl, (gf.ks if fpart else 1) * m * nf * (2 if fpart else 1), dtype=BF, device=dev
    )
    opart = mode == "part" and go.ks > 1
    acc = torch.zeros(
        (go.ks * m * H * 4) if opart else m * H * 2, dtype=torch.uint8, device=dev
    )
    cnt = torch.zeros(
        ((m + go.bm - 1) // go.bm) * (H // go.bn), dtype=torch.int32, device=dev
    )
    scratch = torch.zeros(pl.scratch_bytes(), dtype=torch.uint8, device=dev)
    pa = peers.addresses.data_ptr()
    assert nl % step == 0

    def front():
        if fpart:
            return out.view(torch.float32).view(nl, gf.ks, m, nf).sum(1)
        return out.view(nl, m, nf).float()

    tl = torch.zeros(BLOCKS, chain.TL_POINTS, dtype=torch.int64, device=dev)

    def one(i, launch=launch):
        if mode == "linbf" and gf.ks > 1:
            out[i].zero_()
        if i % step == 0:
            epoch[0].add_(1)
        launch(
            a[i].data_ptr(),
            wo.data_ptr(),
            res[i].data_ptr(),
            res_out[i].data_ptr(),
            bank[i].data_ptr(),
            cw.data_ptr(),
            w_ln.data_ptr(),
            wf.data_ptr(),
            out[i].data_ptr(),
            acc.data_ptr(),
            cnt.data_ptr(),
            scratch.data_ptr(),
            pa,
            rank,
            epoch.data_ptr(),
            i % step,
            tl.data_ptr(),
            stream=torch.cuda.current_stream(),
        )

    for i in range(nl):
        one(i)
    torch.cuda.synchronize()
    graph, us_fused = graph_us(one, nl)
    worst_h, worst_f, bank_ok = 0.0, 0.0, True
    for r in range(max(args.reps_check, SETS)):
        s = r % SETS
        a.copy_(asets[s])
        res.copy_(rsets[s][..., rank * sl : (rank + 1) * sl])
        bank.copy_(bsets[s])
        graph.replay()
        eh = ((res_out.float() - refs[s][0]).norm() / refs[s][0].norm()).item()
        ef = ((front() - refs[s][1]).norm() / refs[s][1].norm()).item()
        # a NaN would vanish in max()
        worst_h = max(worst_h, eh if eh == eh else float("inf"))
        worst_f = max(worst_f, ef if ef == ef else float("inf"))
        if write:
            bank_ok &= bool(torch.equal(bank[:, :, nvb], res_out))
    torch.cuda.synchronize()

    o = torch.empty(m, H, dtype=BF, device=dev)
    yo = torch.empty(m, nf, dtype=BF, device=dev)

    bank_full = torch.zeros(nl, m, nbk, H, dtype=BF, device=dev)

    def std(i):
        o.copy_(F.linear(a[i], wo))
        dist.all_reduce(o)
        hh = o + res_full[i]
        yo.copy_(F.linear(attn_res_ref(hh, bank_full[i], nvb, w_s, w_p, w_ln), wf))

    for i in range(nl):
        std(i)
    torch.cuda.synchronize()
    _, us_std = graph_us(std, nl)

    if args.timeline:
        timeline(args, chain, m, go, gf, nf, mode, step, one, tl, nl, rank, knobs)

    stats = torch.tensor([worst_h, worst_f, us_fused, us_std], device=dev)
    every = [torch.empty_like(stats) for _ in range(TP)]
    dist.all_gather(every, stats)
    if rank == 0:
        e = torch.stack(every).max(0).values.tolist()
        ok = e[0] < 1e-2 and e[1] < 1e-2 and bank_ok
        print(
            f"nvb {nvb}{'w' if write else ''} m {m:3d} o t{go.bm}x{go.bn} ks{go.ks} f t{gf.bm}x{gf.bn} ks{gf.ks} {mode:5s}  rel_l2 residual "
            f"{e[0]:.2e} front {e[1]:.2e}  us: chain {e[2]:6.1f} | standard {e[3]:6.1f}  "
            f"{'PASS' if ok else 'FAIL'}",
            flush=True,
        )
    peers.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--cfgs",
        default="16/16:128:1:8:4/16:128:1:8:8,32/32:128:2:4:2/32:128:2:4:8,"
        "128/64:64:2:4:1/128:128:2:4:8,384/192:64:4:2:1/128:128:2:4:4",
        help="m/o_proj bm:bn:wm:wn:ks/front bm:bn:wm:wn:ks,...",
    )
    p.add_argument("--modes", default="linbf")
    p.add_argument(
        "--nf",
        type=int,
        default=2096,
        help="front rows a rank: 112 router + 448 latent + 1536 sgu",
    )
    p.add_argument("--launches", type=int, default=8)
    p.add_argument("--step", type=int, default=4)
    p.add_argument("--reps-check", type=int, default=6)
    p.add_argument("--plan", action="store_true")
    p.add_argument("--timeline", action="store_true")
    p.add_argument("--nvb", type=int, default=0, help="bank snapshots the seam mixes")
    p.add_argument("--nbk", type=int, default=8, help="bank rows (capacity)")
    p.add_argument(
        "--write", action="store_true", help="the new prefix becomes bank row nvb"
    )
    args = p.parse_args()
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl")
    cpu = dist.new_group(backend="gloo")
    dev = torch.device("cuda", rank)
    for spec in args.cfgs.split(","):
        for mode in args.modes.split(","):
            run(args, spec, mode, rank, dev, cpu)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
