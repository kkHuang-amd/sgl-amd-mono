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


def kda_ref(
    proj,
    conv_w,
    conv_in,
    slot_c,
    rstate,
    slot0,
    acc_idx,
    w_fb,
    dt_bias,
    a_log,
    on_w,
    k1,
    L,
    eps,
):
    """core_out [S, 1536] in torch from the bf16 in_proj rows ``proj`` [S, NPROJ], R = S / L
    requests at once (Kimi-K3's K1 reference math, sgl-project/sgl-amd-mono#13, per request)."""
    hd, nh, pw = k1.HD, k1.NH, k1.PROJ
    r_n = proj.shape[0] // L
    p = proj.float().view(r_n, L, -1)
    win = torch.stack(
        [
            conv_in[slot_c[r], :, acc_idx[r] : acc_idx[r] + k1.CONV_W - 1]
            for r in range(r_n)
        ]
    ).float()
    seq = torch.cat(
        [win, p[..., : 3 * pw].transpose(1, 2)], dim=2
    )  # [R, 3 PROJ, CONV_W - 1 + L]
    cw = conv_w.float()
    conv = torch.stack(
        [sum(cw[:, q] * seq[..., j + q] for q in range(k1.CONV_W)) for j in range(L)],
        dim=1,
    )
    conv = conv * torch.sigmoid(conv)  # [R, L, 3 PROJ]
    q, k, v = (
        conv[..., i * pw : (i + 1) * pw].reshape(r_n, L, nh, hd) for i in range(3)
    )
    q = q * torch.rsqrt(q.pow(2).sum(-1, keepdim=True) + 1e-6) * hd**-0.5
    k = k * torch.rsqrt(k.pow(2).sum(-1, keepdim=True) + 1e-6)
    fa = p[..., k1.FA_0 : k1.FA_0 + hd].to(BF).float()
    beta = torch.sigmoid(p[..., k1.BETA_0 : k1.BETA_0 + nh])  # [R, L, NH]
    g1 = torch.einsum("rlk,hdk->rlhd", fa, w_fb.float().view(nh, hd, hd))
    g = g1 + dt_bias.float().view(nh, hd)
    decay = torch.exp(
        -5.0 * torch.sigmoid(torch.exp(a_log.float()).view(nh, 1) * g)
    )  # [R, L, NH, K]
    st = torch.stack([rstate[slot0[r]] for r in range(r_n)]).float()  # [R, NH, V, K]
    outs = []
    for j in range(L):
        st = st * decay[:, j].unsqueeze(2)
        dl = (v[:, j] - torch.einsum("rhvk,rhk->rhv", st, k[:, j])) * beta[
            :, j
        ].unsqueeze(-1)
        st = st + dl.unsqueeze(-1) * k[:, j].unsqueeze(2)
        outs.append(torch.einsum("rhvk,rhk->rhv", st, q[:, j]))
    o = torch.stack(outs, dim=1)  # [R, L, NH, V]
    rr = torch.rsqrt(o.pow(2).mean(-1, keepdim=True) + eps)
    g2 = p[..., k1.G2_0 : k1.G2_0 + pw].view(r_n, L, nh, hd)
    out = o * rr * on_w.float() * torch.sigmoid(g2)
    return out.reshape(r_n * L, pw).to(BF)


def win_ref(proj, conv_in, slot_c, acc_idx, k1, L):
    """The verify conv windows [R, L, CONV_W - 1, 3 PROJ] the KDA stage writes: token j's
    window is the CONV_W - 1 conv inputs up to and including token j."""
    r_n = proj.shape[0] // L
    win = torch.stack(
        [
            conv_in[slot_c[r], :, acc_idx[r] : acc_idx[r] + k1.CONV_W - 1]
            for r in range(r_n)
        ]
    )
    seq = torch.cat(
        [win.transpose(1, 2), proj.view(r_n, L, -1)[..., : 3 * k1.PROJ]], dim=1
    )
    return torch.stack([seq[:, j + 1 : j + k1.CONV_W] for j in range(L)], dim=1)


def win_bufs(nl, r_n, L, k1, dev):
    """Window rows of nl launches: request r at row r_n - 1 - r, row r_n never written."""
    conv_win = torch.zeros(
        nl, r_n + 1, L, k1.CONV_W - 1, 3 * k1.PROJ, dtype=BF, device=dev
    )
    win_row = torch.arange(r_n - 1, -1, -1, dtype=torch.int32, device=dev)
    return conv_win, win_row


def run(args, spec, mode, rank, dev, cpu):
    from sgl_amd_mono.common import guard
    from sgl_amd_mono.common.peer import PeerBuffer
    from sgl_amd_mono.kimi_k31.gfx950 import pre_attn as pa
    from sgl_amd_mono.kimi_k31.gfx950.chain import Gemm

    ms, su, si = spec.split("/")
    m = int(ms)
    gu = Gemm(*(int(v) for v in su.split(":")))
    gi = Gemm(*(int(v) for v in si.split(":")))
    nin, nl, step, nvb, nbk, write = (
        args.nin,
        args.launches,
        args.step,
        args.nvb,
        args.nbk,
        args.write,
    )
    sl = H // TP
    from sgl_amd_mono.kimi_k31.gfx950 import kda_defs as k1

    L = args.kda
    state_len = L + k1.CONV_W - 1 if L else 0
    launch, pl = pa.pre_attn(
        m,
        gu,
        gi,
        nin,
        nvb=nvb,
        nbk=nbk,
        write=write,
        mode=mode,
        launches=step,
        eps_n=EPS_N,
        eps=EPS,
        kda=L or None,
        state_len=state_len,
        ring=bool(L),
        onorm_eps=EPS_N,
        kpt=int(os.environ.get("PA_KPT", "16")),
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
        sets.append(
            dict(
                lat_p=lat_p,
                sh_p=sh_p,
                prefix=prefix[..., cols].contiguous(),
                prefix_full=prefix,
                bank=bk.view(nl, m, nbk, TP, sl)[:, :, :, rank].contiguous(),
                bank_full=bk,
                ref_pre=pre[..., cols].float(),
                ref_out=ref_out,
            )
        )
    cur = {k: sets[0][k].clone() for k in ("lat_p", "sh_p", "prefix", "bank")}
    prefix_out = torch.zeros(nl, m, sl, dtype=BF, device=dev)
    ipart = mode == "part" and gi.ks > 1
    out = torch.zeros(
        nl,
        (gi.ks if ipart else 1) * m * nin * (2 if ipart else 1),
        dtype=BF,
        device=dev,
    )
    acc = torch.zeros(m * sl, dtype=BF, device=dev)
    cnt = torch.zeros(
        ((m + gu.bm - 1) // gu.bm) * (sl // gu.bn)
        + ((m + gi.bm - 1) // gi.bm) * ((nin + gi.bn - 1) // gi.bn),
        dtype=torch.int32,
        device=dev,
    )
    tabs = torch.zeros(nl, 1, dtype=torch.int64, device=dev)
    if L:
        r_n = m // L
        gk = torch.Generator(device=dev).manual_seed(70 + rank)
        dim = 3 * k1.PROJ
        w_fb = (torch.randn(k1.NH * k1.HD, k1.HD, generator=gk, device=dev) * 0.02).to(
            BF
        )
        conv_w = torch.randn(dim, k1.CONV_W, generator=gk, device=dev) * 0.3
        a_log = torch.randn(k1.NH, generator=gk, device=dev) * 0.1
        dt_bias = torch.randn(k1.NH * k1.HD, generator=gk, device=dev) * 0.1
        on_w = (1 + 0.1 * torch.randn(k1.HD, generator=gk, device=dev)).to(BF)
        st_idx = (
            torch.arange(m, dtype=torch.int32, device=dev).view(r_n, L).contiguous()
        )
        num_acc = (torch.arange(r_n, dtype=torch.int32, device=dev) % L) + 1
        conv0 = (torch.randn(m, dim, state_len, generator=gk, device=dev) * 0.5).to(BF)
        conv = conv0.unsqueeze(0).repeat(nl, 1, 1, 1)
        rstate = torch.randn(m, k1.NH, k1.HD, k1.HD, generator=gk, device=dev) * 0.05
        ring_len = 16
        ring = (
            torch.zeros(m, k1.NH, ring_len, k1.HD, dtype=BF, device=dev),
            torch.zeros(m, k1.NH, ring_len, k1.HD, dtype=BF, device=dev),
            torch.zeros(m, k1.NH, ring_len, k1.HD, device=dev),
            torch.zeros(m, k1.NH, ring_len, device=dev),
        )
        core = torch.zeros(nl, m, k1.PROJ, dtype=BF, device=dev)
        conv_win, win_row = win_bufs(nl, r_n, L, k1, dev)
        tabs = torch.stack(
            [
                pa.kda_table(
                    dev,
                    w_fb=w_fb,
                    conv_w=conv_w,
                    conv_state=conv[i],
                    a_log=a_log,
                    dt_bias=dt_bias,
                    on_w=on_w,
                    rstate=rstate,
                    st_idx=st_idx,
                    num_acc=num_acc,
                    core_out=core[i],
                    conv_win=conv_win[i],
                    win_row=win_row,
                    ring=ring,
                )
                for i in range(nl)
            ]
        )
        sc, s0 = (
            st_idx[:, 0].tolist(),
            [int(st_idx[r, int(num_acc[r]) - 1]) for r in range(r_n)],
        )
        ai = (num_acc - 1).tolist()
        for s in sets:
            s["ref_core"] = torch.stack(
                [
                    kda_ref(
                        s["ref_out"][i].to(BF),
                        conv_w,
                        conv0,
                        sc,
                        rstate,
                        s0,
                        ai,
                        w_fb,
                        dt_bias,
                        a_log,
                        on_w,
                        k1,
                        L,
                        EPS_N,
                    )
                    for i in range(nl)
                ]
            ).float()
    scratch = torch.zeros(pl.scratch_bytes(), dtype=torch.uint8, device=dev)
    pa_ = peers.addresses.data_ptr()
    assert nl % step == 0

    def result():
        if ipart:
            return out.view(torch.float32).view(nl, gi.ks, m, nin).sum(1)
        return out.view(nl, m, nin).float()

    def one(i):
        if (
            mode == "linbf" and gi.ks > 1 and not L
        ):  # the KDA build zeroes its tiles itself
            out[i].zero_()
        if i % step == 0:
            epoch[0].add_(1)
        launch(
            cur["lat_p"][i].data_ptr(),
            cur["sh_p"][i].data_ptr(),
            w_up_r.data_ptr(),
            cur["prefix"][i].data_ptr(),
            prefix_out[i].data_ptr(),
            cur["bank"][i].data_ptr(),
            cw.data_ptr(),
            w_in.data_ptr(),
            w_ip.data_ptr(),
            out[i].data_ptr(),
            acc.data_ptr(),
            cnt.data_ptr(),
            scratch.data_ptr(),
            pa_,
            rank,
            epoch.data_ptr(),
            i % step,
            tabs[i].data_ptr(),
            stream=torch.cuda.current_stream(),
        )

    for i in range(nl):
        one(i)
    torch.cuda.synchronize()
    graph, us_fused = graph_us(one, nl)
    worst_p, worst_o, worst_k, bank_ok, win_bad = 0.0, 0.0, 0.0, True, 0
    for r in range(max(args.reps_check, SETS)):
        s = sets[r % SETS]
        for k in cur:
            cur[k].copy_(s[k])
        if L:
            conv.copy_(conv0.unsqueeze(0).expand_as(conv))
        graph.replay()
        ep = ((prefix_out.float() - s["ref_pre"]).norm() / s["ref_pre"].norm()).item()
        if L:
            eo = ((core.float() - s["ref_core"]).norm() / s["ref_core"].norm()).item()
            # the KDA stage alone: the last launch's own in_proj rows, as it read them
            off = pl.scratch_offsets[0]["proj"]
            pairs = (
                scratch[off : off + m * k1.NPROJ * 8]
                .view(torch.int32)
                .view(m, k1.NPROJ, 2)
            )
            own = pairs[..., 0].view(torch.float32).to(BF)
            want = kda_ref(
                own,
                conv_w,
                conv0,
                sc,
                rstate,
                s0,
                ai,
                w_fb,
                dt_bias,
                a_log,
                on_w,
                k1,
                L,
                EPS_N,
            ).float()
            ek = ((core[-1].float() - want).norm() / want.norm()).item()
            worst_k = max(worst_k, ek if ek == ek else float("inf"))
            # the windows: copies of the conv inputs, exact; the unused row untouched
            got = conv_win[-1][win_row.long()]
            win_bad += int((got != win_ref(own, conv0, sc, ai, k1, L)).sum()) + int(
                conv_win[:, -1].count_nonzero()
            )
        else:
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

    us_std = float("nan")
    if not L:
        for i in range(nl):
            std(i)
        torch.cuda.synchronize()
        _, us_std = graph_us(std, nl)

    stats = torch.tensor(
        [worst_p, worst_o, float(bank_ok), us_fused, us_std, worst_k, float(win_bad)],
        device=dev,
    )
    every = [torch.empty_like(stats) for _ in range(TP)]
    dist.all_gather(every, stats)
    if rank == 0:
        e = torch.stack(every)
        mx = e.max(0).values.tolist()
        # end to end, KDA (l2norm, gates, recurrence) amplifies the in_proj rounding:
        # its own check is against the rows it read (mx[5])
        ok = (
            mx[0] < 1e-2
            and mx[1] < (3e-2 if L else 1e-2)
            and e[:, 2].min().item() == 1.0
            and mx[5] < 5e-3
            and mx[6] == 0
        )
        print(
            f"nvb {nvb}{'w' if write else ''} m {m:3d} up t{gu.bm}x{gu.bn} ks{gu.ks} in t{gi.bm}x{gi.bn} ks{gi.ks} "
            f"{mode:5s}{' kda' + str(L) if L else ''}  rel_l2 prefix {mx[0]:.2e} {'core_out' if L else 'in_proj'} {mx[1]:.2e}{f' (kda alone {mx[5]:.2e}, windows off {int(mx[6])})' if L else ''}  us: pre_attn {mx[3]:6.1f} | standard "
            f"{mx[4]:6.1f}  {'PASS' if ok else 'FAIL'}",
            flush=True,
        )
    # every rank unmaps the peers' buffers before any frees its own (a freed
    # buffer's address can come back for the next case's, still imported elsewhere)
    torch.cuda.synchronize()
    dist.barrier()
    peers.close()
    dist.barrier()


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--cfgs",
        default="16/16:128:1:8:8/16:128:1:8:4,128/128:32:4:2:8/128:128:2:4:4,"
        "384/128:32:4:2:4/192:128:4:2:2",
        help="m/up bm:bn:wm:wn:ks/in_proj bm:bn:wm:wn:ks,...",
    )
    p.add_argument("--modes", default="linbf")
    p.add_argument("--nin", type=int, default=6288)
    p.add_argument("--launches", type=int, default=8)
    p.add_argument("--step", type=int, default=4)
    p.add_argument("--reps-check", type=int, default=6)
    p.add_argument("--nvb", type=int, default=0)
    p.add_argument("--nbk", type=int, default=8)
    p.add_argument("--write", action="store_true")
    p.add_argument("--plan", action="store_true")
    p.add_argument(
        "--kda",
        type=int,
        default=0,
        help="tokens a request: run the KDA stage (ReplaySSM ring)",
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
