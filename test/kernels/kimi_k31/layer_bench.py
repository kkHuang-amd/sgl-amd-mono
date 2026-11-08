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
from pre_attn_bench import kda_ref, win_bufs, win_ref

TP = 8
SETS = 3
BF = torch.bfloat16
H, LAT, KO = 7168, 3584, 1536
E, TOPK, SI = 896, 16, 768


def situ(g, u):
    g, u = g.float(), u.float()
    return (
        (2.0 * torch.tanh(g * 0.25))
        * (1.0 + torch.tanh(g * 0.5))
        * (25.0 * torch.tanh(u * 0.04))
    ).to(BF)


def mixed_ref(prefix, bank, nvb, w_s, w_p):
    """SGLang's ``aggregate_stream`` (the pre-norm mixture DSpark captures)."""
    if nvb == 0:
        return prefix
    rows = torch.cat([bank[..., :nvb, :], prefix.unsqueeze(-2)], dim=-2)
    normed = F.rms_norm(rows.float(), (H,), w_s.float(), EPS).to(BF)
    probs = torch.softmax((normed.float() @ w_p.float()).to(BF).float(), dim=-1)
    return (probs.unsqueeze(-1) * rows.float()).sum(dim=-2).to(BF)


NCAP, CAP_SLOT = (
    5,
    2,
)  # DSpark's 5 captured layers; this launch's block of the packed buffer


def rel(a, b):
    e = ((a.float() - b.float()).norm() / b.float().norm()).item()
    return e if e == e else float("inf")


def timeline(pl, scratch, m, rank):
    """The last replayed launch's marks over every rank's CTAs: first / median / last, us from that rank's
    earliest CTA start (the GPUs' clocks are not aligned)."""
    from sgl_amd_mono.common.plan import BLOCKS
    from sgl_amd_mono.kimi_k31.gfx950 import layer as lay

    o = pl.scratch_offsets[0]["tl"]
    tl = (
        scratch[o : o + BLOCKS * lay.TL_POINTS * 8]
        .view(torch.int64)
        .view(BLOCKS, lay.TL_POINTS)
        .clone()
    )
    every = [torch.empty_like(tl) for _ in range(TP)]
    dist.all_gather(every, tl)
    if rank != 0:
        return
    t = torch.stack(every).cpu()
    t0 = torch.stack([t[r, :, 0][t[r, :, 0] > 0].min() for r in range(TP)]).view(
        TP, 1, 1
    )
    t = torch.where(t > 0, t - t0, torch.full_like(t, -1))
    print(f"timeline m {m}: mark  CTAs  first  median  last (us)", flush=True)
    for k, name in enumerate(lay.MARKS):
        v = t[..., k]
        v = v[v >= 0].double() / 100.0
        if v.numel():
            print(
                f"  {k:2d} {name:11s} {v.numel():5d} {v.min():7.1f} {v.median():7.1f} {v.max():7.1f}",
                flush=True,
            )
    ends = [t[r, :, -1].max().item() / 100.0 for r in range(TP)]
    print("  end by rank: " + " ".join(f"{e:5.1f}" for e in ends), flush=True)


def sort_check(m, nl, tk_ids, tk_w, lat_out, moe_out, sortb, rank):
    """The launch's sort + quant (``--sort``) vs SGLang's small-sort Triton kernel on the launch's own ids /
    weights / latent, bit for bit: num_valid, sorted ids / weights / expert ids up to num_valid, every fp8 byte,
    the e8m0 codes of the real sorted rows (padding rows are never read), and the MoE output zeroed."""
    from sglang.kernels.ops.moe.moe_sorting_small import _run_small_sort

    from sgl_amd_mono.kimi_k31.wide import _sort_views

    bad = 0
    pad = (TOPK << 24) | m
    for i in range(nl):
        v = _sort_views(sortb[i], m)
        ids, w = tk_ids[i].contiguous(), tk_w[i].contiguous()
        maxp = v["ids"].numel()
        r_ids = torch.empty(maxp, dtype=torch.int32, device=ids.device)
        r_w = torch.empty(maxp, dtype=torch.float32, device=ids.device)
        r_e = torch.empty(v["eids"].numel(), dtype=torch.int32, device=ids.device)
        r_nv = torch.empty(2, dtype=torch.int32, device=ids.device)
        r_buf = torch.ones(m, LAT, dtype=BF, device=ids.device)
        r_q, r_qs = _run_small_sort(
            ids, w, r_ids, r_w, r_e, r_nv, r_buf, 32, lat_out[i], E
        )
        nv = int(r_nv[0])
        e = dict(
            nv=int((v["nv"] != r_nv).sum()),
            ids=int((v["ids"][:nv] != r_ids[:nv]).sum()),
            w=int((v["w"][:nv].view(torch.int32) != r_w[:nv].view(torch.int32)).sum()),
            eids=int((v["eids"][: nv // 32] != r_e[: nv // 32]).sum()),
            q=int((v["q"].view(torch.uint8) != r_q.view(torch.uint8)).sum()),
            out=int((moe_out[i] != 0).sum()),
        )
        d = torch.nonzero(r_ids[:nv] != pad).flatten()
        y = torch.arange(112, device=ids.device)
        base = (d // 32) * (112 * 32) + (d % 16) * 4 + (d % 32) // 16
        sw = (
            base[:, None]
            + ((y // 8) * 256 + (y % 4) * 64 + ((y % 8) // 4) * 2)[None, :]
        )
        e["qs"] = int(
            (v["qs"].flatten()[sw] != r_qs.view(torch.uint8).flatten()[sw]).sum()
        )
        n_bad = sum(e.values())
        bad += n_bad
        if rank == 0 and (i == 0 or n_bad):
            print(
                f"sort check m {m} launch {i}: num_valid {nv} rows {d.numel()} mismatches {e}"
                + (" PASS" if n_bad == 0 else " FAIL"),
                flush=True,
            )
    return bad


def sort_check_aux16(m, nl, tk_ids, tk_w, moe_out, sortb, rank):
    """The launch's aux16 sort (``--sort aux16``) vs aiter's own BM16 aux sort on the launch's ids / weights,
    bit for bit below num_valid (``presort.check_aux16``), and the MoE output zeroed."""
    from sgl_amd_mono.kimi_k31.presort import check_aux16
    from sgl_amd_mono.kimi_k31.wide import _sort_views

    bad = 0
    for i in range(nl):
        v = _sort_views(sortb[i], m, "aux16")
        e = check_aux16(v, tk_ids[i], tk_w[i])
        nv = e.pop("nv")
        e["out"] = int((moe_out[i] != 0).sum())
        n_bad = sum(e.values())
        bad += n_bad
        if rank == 0 and (i == 0 or n_bad):
            print(
                f"sort check aux16 m {m} launch {i}: num_valid {nv} mismatches {e}"
                + (" PASS" if n_bad == 0 else " FAIL"),
                flush=True,
            )
    return bad


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
    from sgl_amd_mono.kimi_k31.wide import cfg_rows

    cfg = lay.LayerCfg(*(Gemm(*(int(v) for v in c.split(":"))) for c in cfg_rows(m, L)))
    for st in ("up", "inp", "oproj", "front", "sdown"):
        ovr = (
            dict(kv.split("=") for kv in getattr(args, st).split(","))
            if getattr(args, st)
            else {}
        )
        if str(m) in ovr:
            setattr(cfg, st, Gemm(*(int(v) for v in ovr[str(m)].split(":"))))
    cap = args.capture
    from sgl_amd_mono.kimi_k31.gfx950.stages import moe_front as mf

    # --front-map: wf rows from a buffer laid out as SGLang's merged front weight
    # [shared gate_up | router (all experts) | latent down (all)], as serving reads it
    n_sgu = nf - mf.E_R - mf.L_R
    front_rows = (
        (n_sgu + rank * mf.E_R, n_sgu + mf.E + rank * mf.L_R, 0)
        if args.front_map
        else None
    )
    launch, pl = lay.kda_layer(
        m,
        cfg,
        L,
        nf,
        nvb_a=nva,
        nvb_m=nvm,
        write_a=wa,
        nbk=nbk,
        launches=step,
        eps=EPS,
        eps_n=1e-5,
        capture=cap,
        t1=not args.no_t1,
        restart=args.restart,
        timeline=args.timeline,
        front_rows=front_rows,
        up_norm=args.up_norm,
        sort=args.sort,
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
    ln = {
        k: (1.0 + 0.1 * torch.randn(H, generator=gs, device=dev)).to(BF)
        for k in ("in", "post", "sa", "sm")
    }
    wp = {
        k: (torch.randn(H, generator=gs, device=dev) / H**0.5 * 4).to(BF)
        for k in ("a", "m")
    }
    cw_a = (ln["sa"].float() * wp["a"].float()).contiguous()
    cw_m = (ln["sm"].float() * wp["m"].float()).contiguous()
    w_ip = (torch.randn(k1.NPROJ, H, generator=g, device=dev) / H**0.5).to(BF)
    wo = (torch.randn(H, KO, generator=g, device=dev) / KO**0.5).to(BF)
    wf = (torch.randn(nf, H, generator=g, device=dev) / H**0.5).to(BF)
    wf_dev = wf
    if front_rows is not None:
        wf_dev = torch.randn(n_sgu + mf.E + mf.LAT, H, generator=g, device=dev).to(BF)
        wf_dev[front_rows[0] : front_rows[0] + mf.E_R] = wf[: mf.E_R]
        wf_dev[front_rows[1] : front_rows[1] + mf.L_R] = wf[mf.E_R : mf.E_R + mf.L_R]
        wf_dev[:n_sgu] = wf[mf.E_R + mf.L_R :]
    w_sd = (torch.randn(H, SI, generator=g, device=dev) / SI**0.5).to(BF)
    bias = (torch.randn(E, generator=gs, device=dev) * 0.05).contiguous()
    w_up_r = pa.fold_up(w_up, w_n, rank)
    bias_dev = bias
    if args.up_norm:
        # serving's layout: SGLang's up rows as is, the norm weight after the bias
        w_up_r = w_up[rank * (H // TP) : (rank + 1) * (H // TP)]
        bias_dev = torch.cat([bias, w_n.float()])
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
    ring = (
        torch.zeros(m, k1.NH, 16, k1.HD, dtype=BF, device=dev),
        torch.zeros(m, k1.NH, 16, k1.HD, dtype=BF, device=dev),
        torch.zeros(m, k1.NH, 16, k1.HD, device=dev),
        torch.zeros(m, k1.NH, 16, device=dev),
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

    sets = []
    for s in range(SETS):
        gb = torch.Generator(device=dev).manual_seed(300 + s)
        lat_p = (torch.randn(nl, m, LAT, generator=g, device=dev) * 0.3).to(BF)
        sh_p = (torch.randn(nl, m, H, generator=g, device=dev) * 0.3).to(BF)
        prefix = torch.randn(nl, m, H, generator=gb, device=dev).to(BF)
        prefix_l = prefix
        if args.pfx_short:  # rows past the prefix tensor's: NaN in memory, 0 to the launch (the descriptor bound)
            prefix[:, m - args.pfx_short :] = 0
            prefix_l = prefix.clone()
            prefix_l[:, m - args.pfx_short :] = float("nan")
        bk = torch.randn(nl, m, nbk, H, generator=gb, device=dev).to(BF)
        lat = lat_p.float()
        dist.all_reduce(lat)
        lat_n = F.rms_norm(lat.to(BF).float(), (LAT,), w_n.float(), 1e-5).to(BF)
        sh = sh_p.float()
        dist.all_reduce(sh)
        mid = (
            (sh + lat_n.float() @ w_up.float().T).to(BF).float() + prefix.float()
        ).to(BF)
        xa = attn_res_ref(mid, bk, nva, ln["sa"], wp["a"], ln["in"])
        ip = (xa.float() @ w_ip.float().T).to(BF)
        cr = torch.stack(
            [
                kda_ref(
                    ip[i],
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
                    1e-5,
                )
                for i in range(nl)
            ]
        )
        y = torch.einsum("lmk,nk->lmn", cr.float(), wo.float())
        dist.all_reduce(y)
        h = (y if args.restart else y + mid.float()).to(BF)
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
        sets.append(
            dict(
                lat_p=lat_p,
                sh_p=sh_p,
                prefix=prefix_l,  # full rows: the launch reads its columns by stride H
                bank=bk.view(nl, m, nbk, TP, sl)[:, :, :, rank].contiguous(),
                ref=dict(
                    mid=mid[..., cols],
                    h=h[..., cols],
                    core=cr,
                    latent=latent,
                    sh=shp,
                    win=torch.stack(
                        [win_ref(ip[i], conv0, sc, ai, k1, L) for i in range(nl)]
                    ),
                    cap=mixed_ref(mid, bk, nva, ln["sa"], wp["a"]),
                ),
                ids=ids.sort(-1).values,
                wts=wts.gather(-1, ids.sort(-1).indices),
            )
        )
    cur = {k: sets[0][k].clone() for k in ("lat_p", "sh_p", "prefix", "bank")}
    mid = torch.zeros(nl, m, sl, dtype=BF, device=dev)
    pout = torch.zeros(
        nl, m, H, dtype=BF, device=dev
    )  # the launch writes its columns of full rows
    col0 = 2 * rank * sl
    pfx_rows = (
        m - args.pfx_short
    )  # rows of the prefix tensor (SGLang's step rows <= the bucket's)
    ip_buf = torch.zeros(nl, m, k1.NPROJ, dtype=BF, device=dev)
    unc = set(
        filter(None, os.environ.get("K31_UNCACHED", "scratch").split(","))
    )  # as wide.py: scratch uncached
    keep = []

    def zeros(name, shape, dtype):
        if name not in unc:
            return torch.zeros(*shape, dtype=dtype, device=dev)
        n = 1
        for d in shape:
            n *= d
        pb = PeerBuffer(
            n * torch.tensor([], dtype=dtype).element_size(), cpu, rank, 1, dev
        )
        keep.append(pb)
        t = pb.bytes.view(dtype).view(*shape)
        t.zero_()
        return t

    acc_u = zeros("acc_u", (nl, m, sl), BF)
    acc_o = zeros("acc_o", (nl, m * H * 2), torch.uint8)
    lat_out = torch.zeros(nl, m, 3584, dtype=BF, device=dev)
    tk_ids = torch.zeros(nl, m, TOPK, dtype=torch.int32, device=dev)
    tk_w = torch.zeros(nl, m, TOPK, device=dev)
    if args.sort:
        # wide's flat ping: the MoE output [m, 3584] right before sh_out (the launch zeroes it: filled with 1 here)
        moe_sh = torch.zeros(nl, m * (LAT + H), dtype=BF, device=dev)
        moe_out = moe_sh[:, : m * LAT].view(nl, m, LAT)
        sh_out = moe_sh[:, m * LAT :].view(nl, m, H)
        moe_out.fill_(1.0)
        sort_lay = mf.sort_layout(m, args.sort)
        sortb = torch.zeros(nl, sort_lay["nbytes"], dtype=torch.uint8, device=dev)
        for i in range(nl):
            mf.sort_init(sortb[i], m, args.sort)
    else:
        sh_out = torch.zeros(nl, m, H, dtype=BF, device=dev)
        sortb = torch.zeros(nl, 256, dtype=torch.uint8, device=dev)
    aux = torch.zeros(nl, m, NCAP * H, dtype=BF, device=dev)
    logits_out = torch.zeros(nl, m, E, device=dev)
    cap_view = aux[..., CAP_SLOT * H : (CAP_SLOT + 1) * H]
    cnt = zeros("cnt", (lay.counters(m, cfg, nf),), torch.int32)
    scratch = zeros("scratch", (pl.scratch_bytes(),), torch.uint8)
    pa_ = peers.addresses.data_ptr()
    assert nl % step == 0

    def one(i, sb=None):
        if i % step == 0:
            epoch[0].add_(1)
        launch(
            cur["lat_p"][i].data_ptr(),
            cur["sh_p"][i].data_ptr(),
            w_up_r.data_ptr(),
            cur["prefix"][i].data_ptr() + col0,
            mid[i].data_ptr(),
            pout[i].data_ptr() + col0,
            cur["bank"][i].data_ptr(),
            cw_a.data_ptr(),
            ln["in"].data_ptr(),
            w_ip.data_ptr(),
            ip_buf[i].data_ptr(),
            acc_u[i].data_ptr(),
            tabs[i].data_ptr(),
            core[i].data_ptr(),
            wo.data_ptr(),
            cw_m.data_ptr(),
            ln["post"].data_ptr(),
            wf_dev.data_ptr(),
            acc_o[i].data_ptr(),
            bias_dev.data_ptr(),
            lat_out[i].data_ptr(),
            tk_ids[i].data_ptr(),
            tk_w[i].data_ptr(),
            w_sd.data_ptr(),
            sh_out[i].data_ptr(),
            cap_view[i].data_ptr(),
            NCAP * H,
            logits_out[i].data_ptr(),
            cnt.data_ptr(),
            scratch.data_ptr(),
            pa_,
            rank,
            epoch.data_ptr(),
            i % step,
            pfx_rows * H * 2 - col0,
            sortb[i if sb is None else sb].data_ptr(),
            stream=torch.cuda.current_stream(),
        )

    for i in range(nl):
        one(i)
    torch.cuda.synchronize()

    def check(sb):
        if args.sort == "aux16":
            return sort_check_aux16(m, nl, tk_ids, tk_w, moe_out, sb, rank)
        return sort_check(m, nl, tk_ids, tk_w, lat_out, moe_out, sb, rank)

    sort_bad = check(sortb) if args.sort else 0
    if args.sort and nl > 1:
        # again with each sort buffer holding another launch's rows (the reset-to-pad path serving runs: one
        # buffer a bucket, a different routing every layer)
        moe_out.fill_(1.0)
        for i in range(nl):
            one(i, (i + 1) % nl)
        torch.cuda.synchronize()
        rot = torch.stack([sortb[(i + 1) % nl] for i in range(nl)])
        sort_bad += check(rot)
        # back to each launch's own buffer for the timing loop
        for i in range(nl):
            one(i)
        torch.cuda.synchronize()
    graph, us = graph_us(one, nl)
    worst = dict(
        mid=0.0, h=0.0, core=0.0, latent=0.0, sh=0.0, w=0.0, w_own=0.0, cap=0.0, win=0.0
    )
    id_bad, id_rows, own_bad = 0, 0, 0
    for r in range(max(args.reps_check, SETS)):
        s = sets[r % SETS]
        for k in cur:
            cur[k].copy_(s[k])
        conv.copy_(conv0.unsqueeze(0).expand_as(conv))
        graph.replay()
        for k, got in (
            ("mid", mid),
            ("h", pout[..., cols]),
            ("core", core),
            ("latent", lat_out),
            ("sh", sh_out),
        ):
            worst[k] = max(worst[k], rel(got, s["ref"][k]))
        others = torch.cat(
            [pout[..., : rank * sl], pout[..., (rank + 1) * sl :]], dim=-1
        )
        worst["h"] = max(
            worst["h"], float(others.abs().max().item() > 0) * 9.0
        )  # wrote outside its columns
        worst["win"] = max(
            worst["win"],
            rel(conv_win[:, win_row.long()], s["ref"]["win"]),
            float(conv_win[:, -1].count_nonzero().item() > 0) * 9.0,
        )  # the unused row
        if cap:
            worst["cap"] = max(worst["cap"], rel(cap_view, s["ref"]["cap"]))
            others = torch.cat(
                [aux[..., : CAP_SLOT * H], aux[..., (CAP_SLOT + 1) * H :]], dim=-1
            )
            worst["cap"] = max(
                worst["cap"], float(others.abs().max().item() > 0) * 9.0
            )  # wrote outside its block
        got_ids, order = tk_ids.long().sort(-1)
        same = (got_ids == s["ids"]).all(-1)
        id_bad += int((~same).sum())
        id_rows += same.numel()
        if same.any():
            worst["w"] = max(
                worst["w"], rel(tk_w.gather(-1, order)[same], s["wts"][same])
            )
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
        # a row differs only by a swap of keys closer than the kernel sigmoid's error (v_exp + v_rcp, ~1-3 ulp
        # off torch.sigmoid) is a near-tie, not a routing error
        key_ = sc_ + bias
        for r_ in (~(gi_ == oi).all(-1)).nonzero().flatten().tolist():
            k_got = key_[r_, gi_[r_]].sort().values
            k_ref = key_[r_, oi[r_]].sort().values
            own_bad += int(float((k_got - k_ref).abs().max()) >= 1e-6)
        same_own = (gi_ == oi).all(-1)
        if same_own.any():
            worst["w_own"] = max(
                worst["w_own"],
                rel(tk_w[-1].gather(-1, go_)[same_own], own_w.gather(-1, oo)[same_own]),
            )
        # a peer must not start the next replay (whose pushes land in this rank's buffer)
        # before this rank has read the logits it checks
        torch.cuda.synchronize()
        dist.barrier(group=cpu)
    torch.cuda.synchronize()

    stats = torch.tensor(
        [
            worst["mid"],
            worst["h"],
            worst["core"],
            worst["latent"],
            worst["sh"],
            worst["w"],
            id_bad / max(id_rows, 1),
            us,
            float(own_bad),
            worst["w_own"],
            worst["cap"],
            worst["win"],
        ],
        device=dev,
    )
    every = [torch.empty_like(stats) for _ in range(TP)]
    dist.all_gather(every, stats)
    if rank == 0:
        e = torch.stack(every).max(0).values.tolist()
        # end to end a top-16 flips where the 16th / 17th keys are closer than the
        # upstream rounding; the routing itself must match on its own logits exactly
        # restart: h is the attention output alone (no prefix), so it carries core's error undiluted
        rs = args.restart
        ok = (
            e[0] < 1e-2
            and e[1] < (3e-2 if rs else 1e-2)
            and e[2] < 3e-2
            and e[3] < 3e-2
            and e[4] < 3e-2
            and e[5] < (2e-3 if rs else 1e-3)
            and e[8] == 0
            and e[9] < 1e-5
            and e[10] < 1e-2
            and e[11] < 1e-2
        )
        print(
            f"layer m {m:3d} L {L} nvb a{nva}{'w' if wa else ''} m{nvm}  rel_l2 mid {e[0]:.2e} h {e[1]:.2e} core "
            f"{e[2]:.2e} windows {e[11]:.2e} latent {e[3]:.2e} shared {e[4]:.2e} topk_w {e[5]:.1e} topk rows differing {e[6]:.2%} "
            f"(own logits: {int(e[8])} rows, w {e[9]:.1e}){f' capture {e[10]:.2e}' if cap else ''}  "
            f"us: layer {e[7]:6.1f}  {'PASS' if ok else 'FAIL'}",
            flush=True,
        )
    if args.timeline:
        timeline(pl, scratch, m, rank)
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
    p.add_argument(
        "--capture", action="store_true", help="DSpark capture of the previous layer"
    )
    p.add_argument(
        "--no-t1", action="store_true", help="no MoE tail (a run's first layer)"
    )
    p.add_argument(
        "--pfx-short",
        type=int,
        default=0,
        help="the prefix holds this many rows fewer than m (they read 0)",
    )
    p.add_argument(
        "--restart",
        action="store_true",
        help="block-write layer: the prefix restarts at the attention output",
    )
    p.add_argument(
        "--timeline", action="store_true", help="per-stage CTA marks of the last launch"
    )
    p.add_argument(
        "--sort",
        nargs="?",
        const="sgl",
        default=None,
        choices=("sgl", "aux16"),
        help="the launch makes the routed experts' sort + output zeroing; sgl (default): + mxfp8 quant, checked "
        "bit for bit against SGLang's small-sort kernel; aux16: aiter's BM16 aux sort, checked against aiter's",
    )
    p.add_argument(
        "--inp",
        default="",
        help="in_proj config by rows, e.g. 192=96:128:2:4:2,288=144:128:1:8:2",
    )
    p.add_argument("--up", default="", help="latent up config by rows (as --inp)")
    p.add_argument("--oproj", default="", help="o_proj config by rows (as --inp)")
    p.add_argument("--front", default="", help="MoE front config by rows (as --inp)")
    p.add_argument("--sdown", default="", help="shared down config by rows (as --inp)")
    p.add_argument(
        "--front-map",
        action="store_true",
        help="front weight rows read in place from a merged buffer",
    )
    p.add_argument(
        "--up-norm",
        action="store_true",
        help="up projection rows as is, the latent scaled by the norm weight",
    )
    args = p.parse_args()
    if os.environ.get("SGLANG_AMD_MONO_ROCM_LLVM"):
        from sgl_amd_mono.common import rocm_codegen

        rocm_codegen.install(os.environ["SGLANG_AMD_MONO_ROCM_LLVM"])
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
