# SPDX-License-Identifier: Apache-2.0
"""The pre-attention chain launch (``kimi_k31/gfx950/pre_attn.py``) at TP8: check and time.

MoE tail (latent AR -> latent norm -> latent up on owner columns + shared AR ->
+ prefix) -> attention AttnRes seam -> all-gather -> in_proj, one launch. Checks
the new prefix slice and the in_proj output against SGLang's op order in f32 on
rotating inputs; times the standard path in torch ops, graph-replayed.

    torchrun --nproc-per-node 8 test/kernels/kimi_k31/pre_attn_bench.py \
        --cfgs 16/16:128:1:8:8/16:128:1:8:4,128/128:32:4:2:8/128:128:2:4:4 [--nvb 4 --write]
"""

import argparse
import os

import torch
import torch.distributed as dist
import torch.nn.functional as F
from chain_bench import EPS, attn_res_ref
from gemm_coll_bench import graph_us

TP = 8
SETS = 3
BF = torch.bfloat16
H, LAT, EPS_N = 7168, 3584, 1e-5


def run(args, spec, mode, rank, dev, cpu):
    from sgl_amd_mono.common import guard
    from sgl_amd_mono.common.peer import PeerBuffer
    from sgl_amd_mono.kimi_k31.gfx950 import pre_attn as pa
    from sgl_amd_mono.kimi_k31.gfx950.chain import Gemm

    ms, su, si = spec.split("/")
    m = int(ms)
    gu = Gemm(*(int(v) for v in su.split(":")))
    gi = Gemm(*(int(v) for v in si.split(":")))
    nin, nl, step, nvb, nbk, write = args.nin, args.launches, args.step, args.nvb, args.nbk, args.write
    sl = H // TP
    launch, pl = pa.pre_attn(m, gu, gi, nin, nvb=nvb, nbk=nbk, write=write, mode=mode, launches=step, eps_n=EPS_N,
                             eps=EPS)
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
    gs = torch.Generator(device=dev).manual_seed(11)
    w_up = (torch.randn(H, LAT, generator=gs, device=dev) / LAT**0.5).to(BF)
    w_n = (1.0 + 0.1 * torch.randn(LAT, generator=gs, device=dev)).to(BF)
    w_in = (1.0 + 0.1 * torch.randn(H, generator=gs, device=dev)).to(BF)
    w_s = (1.0 + 0.1 * torch.randn(H, generator=gs, device=dev)).to(BF)
    w_p = (torch.randn(H, generator=gs, device=dev) / H**0.5 * 4).to(BF)
    cw = (w_s.float() * w_p.float()).contiguous()
    w_ip = (torch.randn(nin, H, generator=g, device=dev) / H**0.5).to(BF)
    w_up_r = pa.fold_up(w_up, w_n, rank)
    cols = slice(rank * sl, (rank + 1) * sl)

    sets = []
    for s in range(SETS):
        lat_p = (torch.randn(nl, m, LAT, generator=g, device=dev) * 0.3).to(BF)
        sh_p = (torch.randn(nl, m, H, generator=g, device=dev) * 0.3).to(BF)
        gb = torch.Generator(device=dev).manual_seed(300 + s)
        prefix = torch.randn(nl, m, H, generator=gb, device=dev).to(BF)
        bk = torch.randn(nl, m, nbk, H, generator=gb, device=dev).to(BF)
        lat = lat_p.float()
        dist.all_reduce(lat)
        lat_n = F.rms_norm(lat.to(BF).float(), (LAT,), w_n.float(), EPS_N).to(BF)
        up = lat_n.float() @ w_up.float().T
        sh = sh_p.float()
        dist.all_reduce(sh)
        y = (sh + up).to(BF)
        pre = (y.float() + prefix.float()).to(BF)
        xa = attn_res_ref(pre, bk, nvb, w_s, w_p, w_in)
        ref_out = xa.float() @ w_ip.float().T
        sets.append(dict(lat_p=lat_p, sh_p=sh_p, prefix=prefix[..., cols].contiguous(), prefix_full=prefix,
                         bank=bk.view(nl, m, nbk, TP, sl)[:, :, :, rank].contiguous(), bank_full=bk,
                         ref_pre=pre[..., cols].float(), ref_out=ref_out))
    cur = {k: sets[0][k].clone() for k in ("lat_p", "sh_p", "prefix", "bank")}
    prefix_out = torch.zeros(nl, m, sl, dtype=BF, device=dev)
    ipart = mode == "part" and gi.ks > 1
    out = torch.zeros(nl, (gi.ks if ipart else 1) * m * nin * (2 if ipart else 1), dtype=BF, device=dev)
    acc = torch.zeros(m * sl, dtype=BF, device=dev)
    cnt = torch.zeros(((m + gu.bm - 1) // gu.bm) * (sl // gu.bn), dtype=torch.int32, device=dev)
    scratch = torch.zeros(pl.scratch_bytes(), dtype=torch.uint8, device=dev)
    pa_ = peers.addresses.data_ptr()
    assert nl % step == 0

    def result():
        if ipart:
            return out.view(torch.float32).view(nl, gi.ks, m, nin).sum(1)
        return out.view(nl, m, nin).float()

    def one(i):
        if mode == "linbf" and gi.ks > 1:
            out[i].zero_()
        if i % step == 0:
            epoch[0].add_(1)
        launch(cur["lat_p"][i].data_ptr(), cur["sh_p"][i].data_ptr(), w_up_r.data_ptr(), cur["prefix"][i].data_ptr(),
               prefix_out[i].data_ptr(), cur["bank"][i].data_ptr(), cw.data_ptr(), w_in.data_ptr(), w_ip.data_ptr(),
               out[i].data_ptr(), acc.data_ptr(), cnt.data_ptr(), scratch.data_ptr(), pa_, rank, epoch.data_ptr(),
               i % step, stream=torch.cuda.current_stream())

    for i in range(nl):
        one(i)
    torch.cuda.synchronize()
    graph, us_fused = graph_us(one, nl)
    worst_p, worst_o, bank_ok = 0.0, 0.0, True
    for r in range(max(args.reps_check, SETS)):
        s = sets[r % SETS]
        for k in cur:
            cur[k].copy_(s[k])
        graph.replay()
        ep = ((prefix_out.float() - s["ref_pre"]).norm() / s["ref_pre"].norm()).item()
        eo = ((result() - s["ref_out"]).norm() / s["ref_out"].norm()).item()
        worst_p = max(worst_p, ep if ep == ep else float("inf"))
        worst_o = max(worst_o, eo if eo == eo else float("inf"))
        if write:
            bank_ok &= bool(torch.equal(cur["bank"][:, :, nvb], prefix_out))
    torch.cuda.synchronize()

    lat_b = torch.empty(m, LAT, dtype=BF, device=dev)
    sh_b = torch.empty(m, H, dtype=BF, device=dev)
    yo = torch.empty(m, nin, dtype=BF, device=dev)
    pfull = sets[0]["prefix_full"]
    bfull = sets[0]["bank_full"]

    def std(i):
        lat_b.copy_(cur["lat_p"][i])
        dist.all_reduce(lat_b)
        ln = F.rms_norm(lat_b, (LAT,), w_n, EPS_N)
        sh_b.copy_(cur["sh_p"][i])
        sh_b[:, cols] += F.linear(ln, w_up[cols])
        dist.all_reduce(sh_b)
        pre = sh_b + pfull[i]
        yo.copy_(F.linear(attn_res_ref(pre, bfull[i], nvb, w_s, w_p, w_in), w_ip))

    for i in range(nl):
        std(i)
    torch.cuda.synchronize()
    _, us_std = graph_us(std, nl)

    stats = torch.tensor([worst_p, worst_o, float(bank_ok), us_fused, us_std], device=dev)
    every = [torch.empty_like(stats) for _ in range(TP)]
    dist.all_gather(every, stats)
    if rank == 0:
        e = torch.stack(every)
        mx = e.max(0).values.tolist()
        ok = mx[0] < 1e-2 and mx[1] < 1e-2 and e[:, 2].min().item() == 1.0
        print(f"nvb {nvb}{'w' if write else ''} m {m:3d} up t{gu.bm}x{gu.bn} ks{gu.ks} in t{gi.bm}x{gi.bn} ks{gi.ks} "
              f"{mode:5s}  rel_l2 prefix {mx[0]:.2e} in_proj {mx[1]:.2e}  us: pre_attn {mx[3]:6.1f} | standard "
              f"{mx[4]:6.1f}  {'PASS' if ok else 'FAIL'}", flush=True)
    # every rank unmaps the peers' buffers before any frees its own (a freed
    # buffer's address can come back for the next case's, still imported elsewhere)
    torch.cuda.synchronize()
    dist.barrier()
    peers.close()
    dist.barrier()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cfgs", default="16/16:128:1:8:8/16:128:1:8:4,128/128:32:4:2:8/128:128:2:4:4,"
                   "384/128:32:4:2:4/192:128:4:2:2", help="m/up bm:bn:wm:wn:ks/in_proj bm:bn:wm:wn:ks,...")
    p.add_argument("--modes", default="linbf")
    p.add_argument("--nin", type=int, default=6288)
    p.add_argument("--launches", type=int, default=8)
    p.add_argument("--step", type=int, default=4)
    p.add_argument("--reps-check", type=int, default=6)
    p.add_argument("--nvb", type=int, default=0)
    p.add_argument("--nbk", type=int, default=8)
    p.add_argument("--write", action="store_true")
    p.add_argument("--plan", action="store_true")
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
