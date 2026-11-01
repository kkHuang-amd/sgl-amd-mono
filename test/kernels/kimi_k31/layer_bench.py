# SPDX-License-Identifier: Apache-2.0
"""One wide-row KDA layer launch (``kimi_k31/gfx950/layer.py``) at TP8: check and time.

MoE tail -> attention AttnRes -> in_proj -> KDA (ring) -> o_proj -> RS + residual
-> MLP AttnRes -> MoE front, one launch. Checks the prefix after the MoE tail and
after the attention add, core_out, and the MoE front output against the f32
reference in SGLang's op order (``pre_attn_bench`` / ``chain_bench``); times it
against the sum of the two half launches it replaces (``pre_attn`` with KDA +
``chain.post_attn``).

    torchrun --nproc-per-node 8 test/kernels/kimi_k31/layer_bench.py --rows 16,128 [--nvb-a 2 --write-a --nvb-m 3]
"""

import argparse
import os

import torch
import torch.distributed as dist
import torch.nn.functional as F
from chain_bench import EPS, attn_res_ref
from gemm_coll_bench import graph_us
from pre_attn_bench import kda_ref

TP = 8
SETS = 3
BF = torch.bfloat16
H, LAT, KO = 7168, 3584, 1536
CFGS = {  # m -> (up, in_proj, o_proj, front, shared down): bm:bn:wm:wn:ks
    16: ("16:128:1:8:8", "16:128:1:8:4", "16:128:1:8:4", "16:128:1:8:8", "16:128:1:8:1"),
    32: ("32:128:2:4:8", "32:128:2:4:4", "32:128:2:4:2", "32:128:2:4:8", "32:128:2:4:1"),
    128: ("128:32:4:2:8", "128:128:2:4:4", "64:64:2:4:1", "128:128:2:4:8", "64:64:2:4:1"),
    384: ("128:32:4:2:4", "192:128:4:2:2", "192:64:4:2:1", "128:128:2:4:4", "128:64:2:4:1"),
}
E, TOPK, SI = 896, 16, 768


def situ(g, u):
    g, u = g.float(), u.float()
    return ((2.0 * torch.tanh(g * 0.25)) * (1.0 + torch.tanh(g * 0.5)) * (25.0 * torch.tanh(u * 0.04))).to(BF)


def rel(a, b):
    e = ((a.float() - b.float()).norm() / b.float().norm()).item()
    return e if e == e else float("inf")


def run(args, m, rank, dev, cpu):
    from sgl_amd_mono.common import guard
    from sgl_amd_mono.common.peer import PeerBuffer
    from sgl_amd_mono.kimi_k3.gfx950.attention import kda as k1
    from sgl_amd_mono.kimi_k31.gfx950 import layer as lay
    from sgl_amd_mono.kimi_k31.gfx950 import pre_attn as pa
    from sgl_amd_mono.kimi_k31.gfx950.chain import Gemm

    L, nl, step, nf, nbk = args.L, args.launches, args.step, args.nf, args.nbk
    nva, nvm, wa = args.nvb_a, args.nvb_m, args.write_a
    sl = H // TP
    cols = slice(rank * sl, (rank + 1) * sl)
    cfg = lay.LayerCfg(*(Gemm(*(int(v) for v in c.split(":"))) for c in CFGS[m]))
    launch, pl = lay.kda_layer(m, cfg, L, nf, nvb_a=nva, nvb_m=nvm, write_a=wa, nbk=nbk, launches=step, eps=EPS,
                               eps_n=1e-5)
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
    ln = {k: (1.0 + 0.1 * torch.randn(H, generator=gs, device=dev)).to(BF) for k in ("in", "post", "sa", "sm")}
    wp = {k: (torch.randn(H, generator=gs, device=dev) / H**0.5 * 4).to(BF) for k in ("a", "m")}
    cw_a = (ln["sa"].float() * wp["a"].float()).contiguous()
    cw_m = (ln["sm"].float() * wp["m"].float()).contiguous()
    w_ip = (torch.randn(k1.NPROJ, H, generator=g, device=dev) / H**0.5).to(BF)
    wo = (torch.randn(H, KO, generator=g, device=dev) / KO**0.5).to(BF)
    wf = (torch.randn(nf, H, generator=g, device=dev) / H**0.5).to(BF)
    w_sd = (torch.randn(H, SI, generator=g, device=dev) / SI**0.5).to(BF)
    bias = (torch.randn(E, generator=gs, device=dev) * 0.05).contiguous()
    w_up_r = pa.fold_up(w_up, w_n, rank)
    # KDA state (ring): conv state per launch (written), committed state shared (read only)
    r_n = m // L
    gk = torch.Generator(device=dev).manual_seed(70 + rank)
    dim = 3 * k1.PROJ
    state_len = L + k1.CONV_W - 1
    w_fb = (torch.randn(k1.NH * k1.HD, k1.HD, generator=gk, device=dev) * 0.02).to(BF)
    conv_w = torch.randn(dim, k1.CONV_W, generator=gk, device=dev) * 0.3
    a_log = torch.randn(k1.NH, generator=gk, device=dev) * 0.1
    dt_bias = torch.randn(k1.NH * k1.HD, generator=gk, device=dev) * 0.1
    on_w = (1 + 0.1 * torch.randn(k1.HD, generator=gk, device=dev)).to(BF)
    st_idx = torch.arange(m, dtype=torch.int32, device=dev).view(r_n, L).contiguous()
    num_acc = (torch.arange(r_n, dtype=torch.int32, device=dev) % L) + 1
    conv0 = (torch.randn(m, dim, state_len, generator=gk, device=dev) * 0.5).to(BF)
    conv = conv0.unsqueeze(0).repeat(nl, 1, 1, 1)
    rstate = torch.randn(m, k1.NH, k1.HD, k1.HD, generator=gk, device=dev) * 0.05
    ring = (torch.zeros(m, k1.NH, 16, k1.HD, dtype=BF, device=dev), torch.zeros(m, k1.NH, 16, k1.HD, dtype=BF, device=dev),
            torch.zeros(m, k1.NH, 16, k1.HD, device=dev), torch.zeros(m, k1.NH, 16, device=dev))
    core = torch.zeros(nl, m, k1.PROJ, dtype=BF, device=dev)
    tabs = torch.stack([pa.kda_table(dev, w_fb=w_fb, conv_w=conv_w, conv_state=conv[i], a_log=a_log, dt_bias=dt_bias,
                                     on_w=on_w, rstate=rstate, st_idx=st_idx, num_acc=num_acc, core_out=core[i],
                                     ring=ring) for i in range(nl)])
    sc, s0 = st_idx[:, 0].tolist(), [int(st_idx[r, int(num_acc[r]) - 1]) for r in range(r_n)]
    ai = (num_acc - 1).tolist()

    sets = []
    for s in range(SETS):
        gb = torch.Generator(device=dev).manual_seed(300 + s)
        lat_p = (torch.randn(nl, m, LAT, generator=g, device=dev) * 0.3).to(BF)
        sh_p = (torch.randn(nl, m, H, generator=g, device=dev) * 0.3).to(BF)
        prefix = torch.randn(nl, m, H, generator=gb, device=dev).to(BF)
        bk = torch.randn(nl, m, nbk, H, generator=gb, device=dev).to(BF)
        lat = lat_p.float()
        dist.all_reduce(lat)
        lat_n = F.rms_norm(lat.to(BF).float(), (LAT,), w_n.float(), 1e-5).to(BF)
        sh = sh_p.float()
        dist.all_reduce(sh)
        mid = ((sh + lat_n.float() @ w_up.float().T).to(BF).float() + prefix.float()).to(BF)
        xa = attn_res_ref(mid, bk, nva, ln["sa"], wp["a"], ln["in"])
        ip = (xa.float() @ w_ip.float().T).to(BF)
        cr = torch.stack([kda_ref(ip[i], conv_w, conv0, sc, rstate, s0, ai, w_fb, dt_bias, a_log, on_w, k1, L, 1e-5)
                          for i in range(nl)])
        y = torch.einsum("lmk,nk->lmn", cr.float(), wo.float())
        dist.all_reduce(y)
        h = (y + mid.float()).to(BF)
        bk2 = bk.clone()
        if wa:
            bk2[:, :, nva] = mid
        xm = attn_res_ref(h, bk2, nvm, ln["sm"], wp["m"], ln["post"])
        front = xm.float() @ wf.float().T
        logits = [torch.empty_like(front[..., :112]) for _ in range(TP)]
        dist.all_gather(logits, front[..., :112].contiguous())
        logits = torch.cat(logits, dim=-1)
        sc_ = torch.sigmoid(logits)
        ids = torch.topk(sc_ + bias, TOPK, dim=-1).indices
        wts = sc_.gather(-1, ids)
        wts = wts / wts.sum(-1, keepdim=True)
        lats = [torch.empty_like(front[..., 112:560]) for _ in range(TP)]
        dist.all_gather(lats, front[..., 112:560].contiguous())
        latent = torch.cat(lats, dim=-1).to(BF)
        gu_ = front[..., 560:].to(BF)
        shp = situ(gu_[..., :SI], gu_[..., SI:]).float() @ w_sd.float().T
        sets.append(dict(lat_p=lat_p, sh_p=sh_p, prefix=prefix[..., cols].contiguous(),
                         bank=bk.view(nl, m, nbk, TP, sl)[:, :, :, rank].contiguous(),
                         ref=dict(mid=mid[..., cols], h=h[..., cols], core=cr, latent=latent, sh=shp),
                         ids=ids.sort(-1).values, wts=wts.gather(-1, ids.sort(-1).indices)))
    cur = {k: sets[0][k].clone() for k in ("lat_p", "sh_p", "prefix", "bank")}
    mid = torch.zeros(nl, m, sl, dtype=BF, device=dev)
    pout = torch.zeros(nl, m, sl, dtype=BF, device=dev)
    ip_buf = torch.zeros(nl, m, k1.NPROJ, dtype=BF, device=dev)
    acc_u = torch.zeros(nl, m, sl, dtype=BF, device=dev)
    acc_o = torch.zeros(nl, m * H * 2, dtype=torch.uint8, device=dev)
    lat_out = torch.zeros(nl, m, 3584, dtype=BF, device=dev)
    tk_ids = torch.zeros(nl, m, TOPK, dtype=torch.int32, device=dev)
    tk_w = torch.zeros(nl, m, TOPK, device=dev)
    sh_out = torch.zeros(nl, m, H, dtype=BF, device=dev)
    cnt = torch.zeros(lay.counters(m, cfg, nf), dtype=torch.int32, device=dev)
    scratch = torch.zeros(pl.scratch_bytes(), dtype=torch.uint8, device=dev)
    pa_ = peers.addresses.data_ptr()
    assert nl % step == 0

    def one(i):
        if i % step == 0:
            epoch[0].add_(1)
        launch(cur["lat_p"][i].data_ptr(), cur["sh_p"][i].data_ptr(), w_up_r.data_ptr(), cur["prefix"][i].data_ptr(),
               mid[i].data_ptr(), pout[i].data_ptr(), cur["bank"][i].data_ptr(), cw_a.data_ptr(), ln["in"].data_ptr(),
               w_ip.data_ptr(), ip_buf[i].data_ptr(), acc_u[i].data_ptr(), tabs[i].data_ptr(), core[i].data_ptr(),
               wo.data_ptr(), cw_m.data_ptr(), ln["post"].data_ptr(), wf.data_ptr(), acc_o[i].data_ptr(),
               bias.data_ptr(), lat_out[i].data_ptr(), tk_ids[i].data_ptr(), tk_w[i].data_ptr(), w_sd.data_ptr(),
               sh_out[i].data_ptr(), cnt.data_ptr(), scratch.data_ptr(), pa_, rank, epoch.data_ptr(), i % step,
               stream=torch.cuda.current_stream())

    for i in range(nl):
        one(i)
    torch.cuda.synchronize()
    graph, us = graph_us(one, nl)
    worst = dict(mid=0.0, h=0.0, core=0.0, latent=0.0, sh=0.0, w=0.0, w_own=0.0)
    id_bad, id_rows, own_bad = 0, 0, 0
    for r in range(max(args.reps_check, SETS)):
        s = sets[r % SETS]
        for k in cur:
            cur[k].copy_(s[k])
        conv.copy_(conv0.unsqueeze(0).expand_as(conv))
        graph.replay()
        for k, got in (("mid", mid), ("h", pout), ("core", core), ("latent", lat_out), ("sh", sh_out)):
            worst[k] = max(worst[k], rel(got, s["ref"][k]))
        got_ids, order = tk_ids.long().sort(-1)
        same = (got_ids == s["ids"]).all(-1)
        id_bad += int((~same).sum())
        id_rows += same.numel()
        if same.any():
            worst["w"] = max(worst["w"], rel(tk_w.gather(-1, order)[same], s["wts"][same]))
        # the routing alone: the last launch's own gathered logits (its half of the peer buffer)
        offs, half = pl.peer_offsets
        par = (int(epoch[0]) * step + (nl - 1) % step) % 2
        o = par * half + offs["rgat"]
        lg = peers.bytes[o : o + m * E * 4].view(torch.float32).view(m, E)
        sc_ = torch.sigmoid(lg)
        own_ids = torch.topk(sc_ + bias, TOPK, dim=-1).indices
        own_w = sc_.gather(-1, own_ids)
        own_w = own_w / own_w.sum(-1, keepdim=True)
        oi, oo = own_ids.sort(-1)
        gi_, go_ = tk_ids[-1].long().sort(-1)
        own_bad += int((~(gi_ == oi).all(-1)).sum())
        worst["w_own"] = max(worst["w_own"], rel(tk_w[-1].gather(-1, go_), own_w.gather(-1, oo)))
    torch.cuda.synchronize()

    stats = torch.tensor([worst["mid"], worst["h"], worst["core"], worst["latent"], worst["sh"], worst["w"],
                          id_bad / max(id_rows, 1), us, float(own_bad), worst["w_own"]], device=dev)
    every = [torch.empty_like(stats) for _ in range(TP)]
    dist.all_gather(every, stats)
    if rank == 0:
        e = torch.stack(every).max(0).values.tolist()
        # end to end a top-16 flips where the 16th / 17th keys are closer than the
        # upstream rounding; the routing itself must match on its own logits exactly
        ok = (e[0] < 1e-2 and e[1] < 1e-2 and e[2] < 3e-2 and e[3] < 3e-2 and e[4] < 3e-2 and e[5] < 1e-3
              and e[8] == 0 and e[9] < 1e-5)
        print(f"layer m {m:3d} L {L} nvb a{nva}{'w' if wa else ''} m{nvm}  rel_l2 mid {e[0]:.2e} h {e[1]:.2e} core "
              f"{e[2]:.2e} latent {e[3]:.2e} shared {e[4]:.2e} topk_w {e[5]:.1e} topk rows differing {e[6]:.2%} "
              f"(own logits: {int(e[8])} rows, w {e[9]:.1e})  "
              f"us: layer {e[7]:6.1f}  {'PASS' if ok else 'FAIL'}", flush=True)
    torch.cuda.synchronize()
    dist.barrier()
    peers.close()
    dist.barrier()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rows", default="16,128")
    p.add_argument("--L", type=int, default=8)
    p.add_argument("--nf", type=int, default=2096)
    p.add_argument("--nbk", type=int, default=8)
    p.add_argument("--nvb-a", type=int, default=0)
    p.add_argument("--nvb-m", type=int, default=0)
    p.add_argument("--write-a", action="store_true")
    p.add_argument("--launches", type=int, default=8)
    p.add_argument("--step", type=int, default=4)
    p.add_argument("--reps-check", type=int, default=6)
    p.add_argument("--plan", action="store_true")
    args = p.parse_args()
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl")
    cpu = dist.new_group(backend="gloo")
    dev = torch.device("cuda", rank)
    for m in (int(v) for v in args.rows.split(",")):
        run(args, m, rank, dev, cpu)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
