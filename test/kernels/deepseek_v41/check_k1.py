"""Check K1 bitwise against SGLang's decode chain for the seam and SWA KV write on gfx950.

Variants: the owed post folded in, a settled residual (after an Engram write), the first layer
(copy 0 of the broadcast embedding) and an index source (normed input, qr, index query); each
also token parallel where the width allows (the seam's outputs on this rank's rows, the rest
whole)."""

import argparse
import math
import os
import struct

import torch
import torch.distributed as dist
from check_ffn import (
    HC,
    HC_EPS,
    HIDDEN,
    RMS_EPS,
    finish,
    make_weights,
    mono_weights,
    seam_inputs,
)

from sgl_amd_mono.deepseek_v41.spec import NUMERICS, SPEC

RECORD, BLOCK, PAGES, POSITIONS = 584, 64, 64, 131072
Q_RANK, NOPE, HEAD = 1280, 448, 512
# the spec's bound; measured mixes <= 5.4e-7 (TP2 / TP4 synthetic, TP4 server)
MIX_TOL = NUMERICS.bound("k1.mix.max_abs")
# K1 rounds q as SGLang does but sums wq_b in another order: bf16 flips only
Q_COS, Q_REL = NUMERICS.bound("k1.q.cosine"), NUMERICS.bound("k1.q.rel")
WIDTHS = ",".join(str(m) for m in SPEC.devices["gfx950"].widths)
# (name, fold, index, first), each against its own SGLang chain
VARIANTS = (
    ("fold", True, False, False),
    ("settled", False, False, False),
    ("first", False, False, True),
    ("index", True, True, False),
)
# also run token parallel (``token_par``) where the width allows
TOKEN_PAR_VARIANTS = ("fold", "settled", "index")
# fold runs at every width, the other variants only at these
VARIANT_WIDTHS = (1, 6, 12, 24, 48)
QR_REL = NUMERICS.bound("k1.qr.rel")
# SGLang's wqkv_a configs K1 matches: the GEMV per M bucket up to 32 rows, the split-K tile above
GEMV = {1: "w16s1r16t16k", 2: "w16s1r16t16k", 4: "w16s1r16t16k", 8: "w8s2r16t16k", 16: "w8s2r16t16k",
        32: "w8s1r16t32k"}  # fmt: skip
LARGE_M = (64, 64, 256, 4, 5)


def attn_weights(heads, dev, g):
    """K1's weights of one layer, wide-range: e4m3 [N, K] with varied E8M0 32 x 32 block codes."""

    def fp8(n, k, lo, hi):
        w = (torch.randn(n, k, generator=g, device=dev) * 0.5).to(torch.float8_e4m3fn)
        return w, torch.randint(lo, hi, (n // 32, k // 32), generator=g, device=dev).to(
            torch.uint8
        )

    wqkv, wqkv_s = fp8(1792, HIDDEN, 116, 126)
    wqb, wqb_s = fp8(heads * 512, 1280, 121, 122)
    q_norm = torch.ones(1280, dtype=torch.bfloat16, device=dev)
    kv_norm = (torch.randn(512, generator=g, device=dev).abs() * 2 + 0.05).to(
        torch.bfloat16
    )
    return dict(
        wqkv=wqkv, wqkv_s=wqkv_s, q_norm=q_norm, kv_norm=kv_norm, wqb=wqb, wqb_s=wqb_s
    )


def rope_table(dev):
    j = torch.arange(32, device=dev, dtype=torch.float64)
    ang = torch.arange(POSITIONS, device=dev, dtype=torch.float64)[
        :, None
    ] * 10000.0 ** (-j / 32)
    return torch.cat([ang.cos(), ang.sin()], -1).float().contiguous()


def check_wqkv_route(M):
    """Return whether SGLang's wqkv_a config at M is the one K1 implements."""
    from sglang.kernels.ops.gemm import mxfp8_native_amd_gfx95 as mx

    from sgl_amd_mono.deepseek_v41.gfx950.attention.plan import (
        wqkv_order,
    )

    if M <= 32:
        b = mx._m_bucket(M)
        got, want = mx._select_config(b, 1792, HIDDEN), mx._GemvConfig.parse(GEMV[b])
        return got == want and wqkv_order(M)[0] == want.waves, f"GEMV {got}"
    got = tuple(mx._large_m_tile(M, 1792, HIDDEN))
    return got == LARGE_M and wqkv_order(M)[0] == LARGE_M[4], f"tile {got}"


def query_ref(qkv, a, cos_sin, pos):
    """fp64 query from SGLang's bf16 wqkv_a output: q_norm, MXFP8, wq_b, bf16, GPT-J RoPE of dims 448.."""
    x = qkv[:, :Q_RANK].float()
    y = (
        (
            x
            * torch.rsqrt(x.pow(2).mean(1, keepdim=True) + RMS_EPS)
            * a["q_norm"].float()
        )
        .bfloat16()
        .float()
    )
    g = y.view(-1, Q_RANK // 32, 32)
    mant, e = torch.frexp(
        g.abs().amax(-1, keepdim=True).clamp_min(1.1754943508222875e-38) / 448
    )
    code = (e - (mant == 0.5).int()).clamp(-127, 127).float()
    x8 = (g * torch.exp2(-code)).clamp(-448, 448).to(
        torch.float8_e4m3fn
    ).double() * torch.exp2(code).double()
    w = a["wqb"].double() * torch.exp2(a["wqb_s"].double() - 127).repeat_interleave(
        32, 0
    ).repeat_interleave(32, 1)
    o = (x8.view(-1, Q_RANK) @ w.T).bfloat16().double().view(len(pos), -1, HEAD)
    cs = cos_sin[pos].double()
    c, s = cs[:, None, :32], cs[:, None, 32:]
    v0, v1 = o[..., NOPE::2], o[..., NOPE + 1 :: 2]
    o[..., NOPE::2], o[..., NOPE + 1 :: 2] = v0 * c - v1 * s, v0 * s + v1 * c
    return o


K1_MARKS = {
    32: "start", 33: "slices", 34: "norm", 35: "kt", 36: "wqkv", 37: "qkv",
    38: "qrdy", 39: "wq_b", 40: "gate",
}  # fmt: skip


def k1_timeline(M, tl) -> str:
    """Each K1 stage's µs across CTAs (median, max) and on the CTA that ends last."""
    marks = list(K1_MARKS)
    last = int(tl[:, marks[-1]].argmax())
    span = (tl[:, marks[-1]].max() - tl[:, marks[0]].min()).item() / 100.0
    lines = [f"M={M} K1 {span:.2f} us; stage us (median / max / CTA {last}, last):"]
    for a, b in zip(marks, marks[1:]):
        d = (tl[:, b] - tl[:, a]).double() / 100.0  # 100 MHz ticks
        lines.append(
            f"  {K1_MARKS[b]:6s} {d.median().item():7.2f} {d.max().item():7.2f}"
            f" {d[last].item():7.2f}"
        )
    return "\n".join(lines)


E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
C_BLOCK, C_PAGES, RING, REQS, PLANE = 64, 16, 8, 64, 64


class CompCase:
    """A KV source's compressor in K1 (``attention.compressor``) against SGLang's chain on
    the normed rows K1 wrote: ``project_fused`` (``linear_bf16_fp32``), c2's decode or
    static-verify kernel (V4 records), ``wk`` and the split index-K store, on random rings."""

    def __init__(self, M, per_req, dev, g):
        from sglang.kernels.ops.attention.dsv4.kv_layout import KVLayout

        self.M, self.per_req, self.dev, self.g = M, per_req, dev, g
        bf = torch.bfloat16
        self.w = (torch.randn(1024, HIDDEN, generator=g, device=dev) * 0.02).to(bf)
        self.nw = (torch.rand(512, generator=g, device=dev) + 0.5).to(bf)
        self.wk = (torch.randn(128, 512, generator=g, device=dev) * 0.05).to(bf)
        self.kn = (torch.rand(128, generator=g, device=dev) + 0.5).to(bf)
        self.eps, self.keps = 1e-6, 1e-6
        self.stride = KVLayout.V4.page_bytes(C_BLOCK)
        self.pool = torch.full(
            (C_PAGES, self.stride), 255, dtype=torch.uint8, device=dev
        )
        self.ring = torch.empty(REQS * RING, 1024, dtype=torch.float32, device=dev)
        self.pay = torch.zeros(C_PAGES, 1, 4, PLANE, 16, dtype=torch.uint8, device=dev)
        self.isc = torch.zeros(C_PAGES, 1, 4, PLANE, dtype=torch.uint8, device=dev)
        self.req = torch.zeros(M, dtype=torch.int64, device=dev)
        self.loc = torch.zeros(M, dtype=torch.int64, device=dev)
        self.iloc = torch.zeros(M, dtype=torch.int64, device=dev)

    def args(self, freqs):
        f32 = lambda v: struct.unpack("<i", struct.pack("<f", v))[0]  # noqa: E731
        return (
            self.w.data_ptr(), self.nw.data_ptr(), freqs.data_ptr(), self.req.data_ptr(),
            self.loc.data_ptr(), self.iloc.data_ptr(), self.ring.data_ptr(),
            self.pool.data_ptr(), self.stride, self.wk.data_ptr(), self.kn.data_ptr(),
            self.pay.data_ptr(), self.isc.data_ptr(), 2, RING, C_BLOCK, f32(self.eps),
            f32(self.keps),
        )  # fmt: skip

    def inputs(self, pos, trial):
        """Requests of per_req consecutive positions (one padded row on odd trials)."""
        M, pr, g, dev = self.M, self.per_req, self.g, self.dev
        nreq = M // pr
        reqs = torch.randperm(REQS, generator=g, device=dev)[:nreq]
        base = torch.randint(1, POSITIONS - pr - 1, (nreq,), generator=g, device=dev)
        self.req.copy_(reqs.repeat_interleave(pr))
        pos.copy_((base[:, None] + torch.arange(pr, device=dev)).flatten())
        slots = torch.randperm(C_PAGES * C_BLOCK - 1, generator=g, device=dev)[:M] + 1
        self.loc.copy_(2 * slots + (pos % 2 == 0))  # raw slots: >> 1 is the record
        if trial % 2:
            self.loc[-1] = 0
        self.iloc.copy_(torch.where((pos + 1) % 2 == 0, self.loc >> 1, -1))
        self.iloc[self.loc == 0] = 0
        self.ring.copy_(torch.randn(self.ring.shape, generator=g, device=dev))
        self.pool.fill_(255)
        self.pay.zero_()
        self.isc.zero_()

    def outputs(self):
        return [t.clone() for t in (self.pool, self.ring, self.pay, self.isc)]

    def restore(self, ring0):
        self.ring.copy_(ring0)
        self.pool.fill_(255)
        self.pay.zero_()
        self.isc.zero_()

    def reference(self, normed, pos, freqs, ring0):
        """SGLang's chain on copies: (pool, ring, payload, scale)."""
        from sglang.kernels.ops.attention.dsv4.fp4_indexer_rope_hip import (
            index_k_norm_rope_pack_store_split,
        )
        from sglang.kernels.ops.attention.dsv4.low_ratio_compress import (
            c2_decode_norm_rope_store,
        )
        from sglang.kernels.ops.gemm.bf16_fp32 import linear_bf16_fp32

        pool = torch.full_like(self.pool, 255)
        ring = ring0.clone()
        self.ring0 = ring0
        pay, isc = torch.zeros_like(self.pay), torch.zeros_like(self.isc)
        latent = c2_decode_norm_rope_store(
            linear_bf16_fp32(normed, self.w), ring, self.nw, pos, self.req, self.loc,
            self.eps, freqs, pool, page_size=C_BLOCK, ring_size=RING,
            draft_len=self.per_req, layout="v4",
        )  # fmt: skip
        index_k_norm_rope_pack_store_split(
            torch.nn.functional.linear(latent, self.wk), self.kn, self.keps, freqs, pos,
            self.iloc, pay, isc, ratio=2,
        )  # fmt: skip
        return [pool, ring, pay, isc]

    def records(self, pool):
        """The dequantized V4 records [M, 512] at each completing row's slot."""
        live = (self.iloc > 0).nonzero().flatten()
        s = self.loc[live] >> 1
        page, off = s // C_BLOCK, s % C_BLOCK
        cols = off[:, None] * 576 + torch.arange(576, device=self.dev)
        data = pool[page[:, None].expand_as(cols), cols]
        scol = C_BLOCK * 576 + off[:, None] * 8 + torch.arange(7, device=self.dev)
        sc = pool[page[:, None].expand_as(scol), scol].float() - 127
        nope = data[:, :448].view(torch.float8_e4m3fn).float() * torch.exp2(
            sc.repeat_interleave(64, 1)
        )
        rope = data[:, 448:].contiguous().view(torch.bfloat16).float()
        return torch.cat([nope, rope], 1)

    def keys(self, pay, isc):
        """The dequantized index keys [rows, 128] at each completing row's slot."""
        s = self.iloc[self.iloc > 0]
        page, slot = s // PLANE, s % PLANE
        b = pay[page, 0, :, slot, :].reshape(-1, 64).long()  # byte j: chunk j // 16
        tab = torch.tensor(E2M1, device=self.dev)
        val = lambda c: tab[c & 7] * (1 - 2 * ((c >> 3) & 1))  # noqa: E731
        v = torch.stack([val(b & 15), val(b >> 4)], -1).reshape(-1, 128)
        e = isc[page, 0, :, (slot % 16) * 4 + slot // 16].float() - 127
        return v * torch.exp2(e).repeat_interleave(32, 1)

    def compare(self, got, want) -> dict:
        """cosine and differing fraction of records and keys, max rel of the ring."""
        out = {}
        for name, a, b in (
            ("rec", self.records(got[0]), self.records(want[0])),
            ("key", self.keys(got[2], got[3]), self.keys(want[2], want[3])),
        ):
            if b.numel() == 0:
                out[name] = (1.0, 0.0)
                continue
            cos = torch.nn.functional.cosine_similarity(
                a.flatten().double(), b.flatten().double(), dim=0
            ).item()
            out[name] = (cos, (a != b).float().mean().item())
        d = (got[1] - want[1]).abs().max() / want[1].abs().max()
        out["ring"] = d.item()
        out["rows"] = int((self.iloc > 0).sum())
        # the even rows' pairs parked, not left as they were
        parked = (want[1] != self.ring0).any(1).sum()
        out["parked"] = int(parked)
        return out


def time_launches(launch, reps: int, trials: int) -> float:
    """Median us a launch over ``trials`` replays of a graph of ``reps`` launches."""
    graph = torch.cuda.CUDAGraph()
    launch()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        for _ in range(reps):
            launch()
    graph.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(max(trials, 5)):
        e0, e1 = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        e0.record()
        graph.replay()
        e1.record()
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1) * 1e3 / reps)
    del graph
    return sorted(ts)[len(ts) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--widths", default=WIDTHS)
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument(
        "--variants-at",
        default=",".join(map(str, VARIANT_WIDTHS)),
        help="the widths the variants other than fold run at",
    )
    ap.add_argument(
        "--timeline", action="store_true", help="print each width's stage timeline"
    )
    # K1 reads wqkv and wq_b in aiter's (16, 16) preshuffle; the references keep row-major
    ap.add_argument("--shuffled", action="store_true")
    ap.add_argument(
        "--delays",
        default="",
        help="timing only: K1's us a launch under each delay slot:us:base:n "
        "(``layer.parse_delay``; 0: none), comma separated, instead of the checks",
    )
    ap.add_argument("--sweep-variants", default="index,index:tp")
    ap.add_argument("--reps", type=int, default=20, help="launches a timed graph")
    ap.add_argument(
        "--comp-per-req",
        default="1,6",
        help="the compressor variants' rows a request (1: decode, else static verify)",
    )
    ap.add_argument(
        "--comp-k", type=int, default=1, help="the compressor's index keys too"
    )
    ap.add_argument(
        "--iw-split",
        type=int,
        default=1,
        help="the head weights split: 0 off, 1 on the compressor variants, 2 on every index one",
    )
    args = ap.parse_args()
    rank, tp = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    dist.init_process_group("nccl")
    cpu = dist.new_group(backend="gloo")
    from sglang.kernels.ops.attention.dsv4.fp4_indexer_hip import (
        index_q_rope_pack_weights_flydsl,
    )
    from sglang.kernels.ops.attention.dsv4.norm_rope import fused_k_norm_rope_flashmla
    from sglang.kernels.ops.gemm.mxfp8_aiter_gfx95 import aiter_mxfp8_blockscaled_linear
    from sglang.kernels.ops.gemm.mxfp8_native_amd_gfx95 import (
        mxfp8_native_blockscaled_linear,
        prepare_mxfp8_native_weight,
    )
    from sglang.kernels.ops.gemm.router_gemv_hip import rocm_router_gemv_split_k
    from sglang.kernels.ops.layernorm.mhc_boundary_hip import (
        hc_boundary_fused_deferred,
        rmsnorm_with_sinkhorn,
    )

    from sgl_amd_mono.deepseek_v41.gfx950.attention.index_query import fused as iq_fused
    from sgl_amd_mono.deepseek_v41.gfx950.layer import (
        NO_COMP,
        MonoBuild,
        build_mono_k1,
        comp_fits,
        iw_split_fits,
        parse_delay,
        scratch_layout,
    )
    from sgl_amd_mono.deepseek_v41.gfx950.runner import (
        DSV41MonoRunner,
    )
    from sgl_amd_mono.deepseek_v41.gfx950.token_par import token_par

    w, _, _ = make_weights(tp, rank, dev, seed=0)
    m = mono_weights(w)
    del w
    if args.timeline:  # the runner builds its launches with stage marks
        os.environ["SGLANG_AMD_MONO_TIMELINE"] = "1"
    r = DSV41MonoRunner(tp, rank, cpu, dev)
    heads = r.q.shape[1]
    g = torch.Generator(device=dev).manual_seed(11)
    a = attn_weights(heads, dev, g)
    m.ffn_norm.copy_(
        (torch.randn(HIDDEN, generator=g, device=dev) * 1.5).exp().to(m.ffn_norm.dtype)
    )
    if args.shuffled:
        from aiter.ops.shuffle import shuffle_weight

        kw = {n: shuffle_weight(a[n], (16, 16)) for n in ("wqkv", "wqb")}
    else:
        kw = {n: a[n] for n in ("wqkv", "wqb")}
    cos_sin = rope_table(dev)
    # an indexer, replicated on every rank: wq_b, weights_proj and the head-weight scale 2^-6
    gi = torch.Generator(device=dev).manual_seed(29)
    iq_w = (torch.randn(4096, Q_RANK, generator=gi, device=dev) * 64).clamp(-448, 448)
    iq_w = iq_w.to(torch.float8_e4m3fn)
    iq_ws = torch.randint(
        116,
        124,
        (4096 // 32, Q_RANK // 32),
        generator=gi,
        device=dev,
        dtype=torch.uint8,
    )
    iw_w = (torch.randn(32, HIDDEN, generator=gi, device=dev) * 0.02).bfloat16()
    kw["iq_w"] = shuffle_weight(iq_w, (16, 16)) if args.shuffled else iq_w
    iw_scale = 128**-0.5 * 32**-0.5
    iw_bits = struct.unpack("<i", struct.pack("<f", iw_scale))[0]
    freqs = torch.complex(cos_sin[:, :32], cos_sin[:, 32:])  # SGLang's freqs_cis
    w_sh, w_su = prepare_mxfp8_native_weight(
        a["wqkv"], torch.exp2(a["wqkv_s"].float() - 127), (32, 32)
    )
    stride = math.ceil(BLOCK * RECORD / 576) * 576
    # page edges and the last slot
    edges = torch.tensor([BLOCK - 1, BLOCK, PAGES * BLOCK - 1], device=dev)

    def records(buf, s):
        """Return the 584 bytes of slots ``s``: 448 NOPE e4m3, 64 bf16 rope, then 8 scale bytes (7, pad)."""
        page, off = s // BLOCK, s % BLOCK
        col = torch.cat([off[:, None] * 576 + torch.arange(576, device=dev),
                         BLOCK * 576 + off[:, None] * 8 + torch.arange(8, device=dev)], 1)  # fmt: skip
        return buf[page[:, None].expand_as(col), col]

    def inputs(M):
        _, res, post, comb, pre = seam_inputs(M, tp, rank, dev)
        res = (res * (torch.randn(HIDDEN, generator=g, device=dev) * 1.5).exp()).to(
            res.dtype
        )
        x = (torch.randn(M, HIDDEN, generator=g, device=dev) * 0.1).to(torch.bfloat16)
        pos = torch.randint(0, POSITIONS, (M,), generator=g, device=dev)
        pos[: min(M, 2)] = torch.tensor([POSITIONS - 1, 65536], device=dev)[: min(M, 2)]
        perm = torch.randperm(PAGES * BLOCK, generator=g, device=dev)
        # int32, as SGLang's HIP backend hands the SWA slots out
        slot = torch.cat([edges, perm[~torch.isin(perm, edges)]])[:M].int()
        return res, x, post, comb, pre, pos, slot

    fails = []
    widths = [int(v) for v in args.widths.split(",")]
    variants_at = {int(v) for v in args.variants_at.split(",")}
    runs = [
        (M, v, False)
        for M in widths
        for v in VARIANTS
        if v[0] == "fold" or M in variants_at
    ]
    runs += [
        (M, v, True)
        for M, v, _ in list(runs)
        if v[0] in TOKEN_PAR_VARIANTS and token_par(M, tp)
    ]
    # a KV source's compressor in K1, at the widths whose grid has its CTAs
    runs += [
        (M, (f"comp{pr}", True, True, False), tp_)
        for M in widths
        if M in variants_at
        for pr in (int(v) for v in args.comp_per_req.split(","))
        for tp_ in (False, True)
        if (not tp_ or token_par(M, tp))
        and (M // tp if tp_ else M) % pr == 0
        and comp_fits(M, tp, tp_)
    ]
    freqs_il = torch.view_as_real(
        torch.complex(cos_sin[:, :32], cos_sin[:, 32:])
    ).flatten(-2).contiguous()  # fmt: skip
    for M, (vname, fold, index, first), tok_par in runs:
        ok, route = check_wqkv_route(M)
        route = r.route_at(M)
        tl = r.timeline_on
        iq = index and iq_fused(M, tok_par)
        pr = int(vname[4:]) if vname.startswith("comp") else 0
        # the split head weights on the compressor's layers (and "index:is" alone)
        iws = bool(
            args.iw_split and iq and (pr or args.iw_split == 2)
            and iw_split_fits(M, tp, tok_par)
        )  # fmt: skip
        cc = CompCase(M, pr, dev, g) if pr else None
        comp_args = cc.args(freqs_il) if pr else NO_COMP
        k1 = build_mono_k1(
            MonoBuild(
                M, tp, fold=fold, index=index, timeline=tl, tok_par=tok_par, iq=iq,
                shuffled=args.shuffled, comp=pr, comp_k=bool(pr and args.comp_k),
                iw_split=iws,
                **route,
            )
        )  # fmt: skip
        # the seam's rows this rank computes: its own, token parallel
        n = M // tp if tok_par else M
        own = slice(rank * n, (rank + 1) * n) if tok_par else slice(0, M)
        vname += ":tp" if tok_par else ""
        lay = scratch_layout(M, tp, tl, **route)
        sc = r.scratch(M)
        static = [t.clone() for t in inputs(M)]
        swa = torch.full((PAGES, stride), 255, dtype=torch.uint8, device=dev)
        res, x, post, comb, pre, pos, slot = static
        if first:
            pre = r.first_pre[:M]
        # a settled residual: no post owed, the seam reads only the residual
        xk, postk, combk = (x, post, comb) if fold else (res, res, res)
        # DSpark's capture of the layer, which K1 writes beside its residual
        # a column block of the packed captures, as the forward hands it out
        packed = torch.full(
            (M, 3 * HIDDEN), float("nan"), dtype=torch.bfloat16, device=dev
        )
        aux = packed[:, HIDDEN : 2 * HIDDEN]

        def launch():
            # the FFN launch moves the epoch; K1 alone must not see the last launch's tags
            r.epoch[0] += 1
            k1(
                res.data_ptr(), xk.data_ptr(), postk.data_ptr(), combk.data_ptr(), pre.data_ptr(),
                m.hc_ffn_fn.data_ptr(), m.hc_ffn_scale.data_ptr(), m.hc_ffn_base.data_ptr(),
                m.ffn_norm.data_ptr(),
                r.res_mid.data_ptr(), aux.data_ptr(), aux.stride(0), r.post_a.data_ptr(), r.comb_a.data_ptr(),
                r.pre_a.data_ptr(), kw["wqkv"].data_ptr(), a["wqkv_s"].data_ptr(), a["q_norm"].data_ptr(), a["kv_norm"].data_ptr(),
                kw["wqb"].data_ptr(), a["wqb_s"].data_ptr(), cos_sin.data_ptr(), pos.data_ptr(),
                slot.data_ptr(), 1, swa.data_ptr(), swa.stride(0), BLOCK, 0, 0, 0, 0, 0, 0, r.q.data_ptr(),
                r.normed.data_ptr(), r.qr.data_ptr(), kw["iq_w"].data_ptr(), iq_ws.data_ptr(),
                iw_w.data_ptr(), iw_bits, cos_sin.shape[0], r.iq.data_ptr(),
                r.iqs.data_ptr(), r.iw.data_ptr(), *comp_args, sc.data_ptr(),
                r.peer.local, r.peer.addresses.data_ptr(), r.rank, r.epoch.data_ptr(),
                stream=torch.cuda.current_stream(),
            )  # fmt: skip

        if args.delays:
            if vname not in args.sweep_variants.split(","):
                continue
            for spec in args.delays.split(","):
                k1 = build_mono_k1(
                    MonoBuild(
                        M, tp, fold=fold, index=index, timeline=tl, tok_par=tok_par,
                        iq=iq, shuffled=args.shuffled, comp=pr, comp_k=bool(pr and args.comp_k),
                iw_split=iws,
                        **parse_delay(spec), **route,
                    )
                )  # fmt: skip
                if pr:
                    cc.inputs(static[5], 0)
                us = time_launches(launch, args.reps, args.trials)
                span = ""
                if tl:
                    launch()
                    torch.cuda.synchronize()
                    span = " | " + k1_timeline(M, r.timeline(M).cpu()).split(";")[0]
                if rank == 0:
                    print(f"M={M:2d} {vname:8s} delay {spec:12s} K1 {us:7.2f} us a launch{span}", flush=True)  # fmt: skip
            continue
        if pr:
            cc.inputs(pos, 0)
        launch()
        torch.cuda.synchronize()
        if args.timeline and (vname in ("fold", "fold:tp", "index", "index:tp") or pr):
            for _ in range(5):
                launch()
            torch.cuda.synchronize()
            if rank == 0:
                print(k1_timeline(M, r.timeline(M).cpu()), flush=True)
                if os.environ.get("TL_SAVE_DIR"):
                    torch.save(
                        r.timeline(M).cpu(),
                        f"{os.environ['TL_SAVE_DIR']}/k1_M{M}_{vname.replace(':', '_')}.pt",
                    )
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            launch()
        diff = dict(
            res=0, aux=0, x8=0, x8s=0, nope=0, scale=0, rope=0, replay=0, normed=0
        )
        if iq:
            diff |= dict(iq=0, iqs=0, iw=0)
        if pr:
            diff |= dict(creplay=0)
        cst = dict(rec=(1.0, 0.0), key=(1.0, 0.0), ring=0.0)
        crows = [0, 0]
        mix, q_cos, q_rel, qr_rel = 0.0, 1.0, 0.0, 0.0
        for trial in range(args.trials):
            for t, v in zip(static, inputs(M)):
                t.copy_(v)
            if pr:
                cc.inputs(pos, trial)
                ring0 = cc.ring.clone()
            swa.fill_(255)
            launch()
            torch.cuda.synchronize()
            if pr:
                comp_eager = cc.outputs()
                want = cc.reference(r.normed[:M], pos, freqs_il, ring0)
                got = cc.compare(comp_eager, want)
                crows = [crows[0] + got.pop("rows"), crows[1] + got.pop("parked")]
                cst = {
                    k: (min(cst[k][0], got[k][0]), max(cst[k][1], got[k][1]))
                    if k != "ring" else max(cst[k], got[k])
                    for k in cst
                }  # fmt: skip
                cc.restore(ring0)
            o = lay["x8"][0]
            x8 = sc[o : o + n * HIDDEN].clone()
            o = lay["x8s"][0]
            x8s = (
                sc[o : o + n * HIDDEN // 32 * 4]
                .view(torch.int32)
                .view(n, HIDDEN // 32)
                .to(torch.uint8)
            )
            got = records(swa, slot.long())
            # the mean SGLang's DSpark takes of the layer's input residual
            want_aux = r.res_mid[:M][own].mean(dim=1)
            # the blocks beside this layer's stay unwritten
            diff["aux"] += int(packed[:, :HIDDEN].isnan().logical_not().sum())
            diff["aux"] += int(packed[:, 2 * HIDDEN :].isnan().logical_not().sum())
            diff["aux"] += int(
                (aux[own].view(torch.int16) != want_aux.view(torch.int16)).sum()
            )
            eager = [
                t[:M].clone() for t in (r.q, r.res_mid, r.post_a, r.comb_a, r.pre_a)
            ] + [got.clone()]
            # the same step replayed: every output and record bitwise equal to eager
            for t in (r.q, r.res_mid, r.post_a, r.comb_a, r.pre_a):
                t.fill_(float("nan"))
            swa.fill_(255)
            graph.replay()
            torch.cuda.synchronize()
            if pr:
                diff["creplay"] += sum(
                    int((u != v).sum()) for u, v in zip(comp_eager, cc.outputs())
                )
            replayed = [
                t[:M] for t in (r.q, r.res_mid, r.post_a, r.comb_a, r.pre_a)
            ] + [records(swa, slot.long())]
            diff["replay"] += sum(
                int((u[rows].view(torch.uint8) != v[rows].view(torch.uint8)).sum())
                for u, v, rows in zip(
                    eager, replayed, [slice(0, M)] + [own] * 4 + [slice(0, M)]
                )
            )

            # SGLang's boundary: no x when nothing is owed, no pre_prev on the first layer (copy 0)
            ro, y, co = hc_boundary_fused_deferred(
                x if fold else None,
                res,
                post if fold else None,
                comb if fold else None,
                None if first else pre,
                m.hc_ffn_fn,
                m.hc_ffn_scale,
                m.hc_ffn_base,
                HC,
                20,
                RMS_EPS,
                HC_EPS,
            )
            ro = res if ro is None else ro
            y = res[:, 0, :].contiguous() if y is None else y
            xq, normed = rmsnorm_with_sinkhorn(
                y, m.ffn_norm, RMS_EPS, co, emit_fp8=True
            )
            pre_r, post_r, comb_r = co.tensors()
            qkv = mxfp8_native_blockscaled_linear(
                xq.q[:M], w_sh, w_su, input_scale=xq.scale[:M]
            )
            ref = torch.full((PAGES, stride), 255, dtype=torch.uint8, device=dev)
            fused_k_norm_rope_flashmla(
                qkv[:, Q_RANK:].contiguous(),
                a["kv_norm"],
                RMS_EPS,
                freqs,
                pos,
                slot.int(),
                ref,
                BLOCK,
            )
            want = records(ref, slot.long())

            diff["res"] += int(
                (eager[1][own].view(torch.int16) != ro[own].view(torch.int16)).sum()
            )
            diff["x8"] += int(
                (x8 != xq.q[:M][own].contiguous().view(torch.uint8).flatten()).sum()
            )
            diff["x8s"] += int(
                (
                    x8s
                    != xq.scale[:M][own].contiguous().view(torch.uint8).reshape(n, -1)
                ).sum()
            )
            # byte 583 pads the 7 scales: no reader, and SGLang's store leaves it as it was
            for name, cols in (
                ("nope", slice(0, 448)),
                ("rope", slice(448, 576)),
                ("scale", slice(576, 583)),
            ):
                diff[name] += int((got[:, cols] != want[:, cols]).sum())
            for k1_t, ref_t in (
                (eager[4], pre_r),
                (eager[2], post_r),
                (eager[3], comb_r),
            ):
                d = (k1_t.reshape(ref_t.shape) - ref_t)[own]
                mix = max(mix, d.abs().max().item())
            if index:
                # an index source's outputs, its compressor and indexer's inputs
                diff["normed"] += int(
                    (r.normed[:M].view(torch.int16) != normed.view(torch.int16)).sum()
                )
                xr = qkv[:, :Q_RANK].float()
                qr_ref = (
                    xr * torch.rsqrt(xr.pow(2).mean(1, keepdim=True) + RMS_EPS)
                    * a["q_norm"].float()
                ).bfloat16().float()  # fmt: skip
                qr_rel = max(
                    qr_rel,
                    ((r.qr[:M].float() - qr_ref).norm() / qr_ref.norm()).item(),
                )
            if iq:
                # bit for bit vs SGLang's chain (``spec._index_query``) on K1's qr and normed input
                ref = index_q_rope_pack_weights_flydsl(
                    q=aiter_mxfp8_blockscaled_linear(r.qr[:M], iq_w, iq_ws),
                    freqs_cis=freqs,
                    positions=pos,
                    rope_dim=64,
                    head_weight_partials=rocm_router_gemv_split_k(
                        x=r.normed[:M], w=iw_w
                    ),
                    weight_scale=iw_scale,
                    num_heads=32,
                )
                for name, got_t, want_t in zip(
                    ("iq", "iqs", "iw"), (r.iq, r.iqs, r.iw), ref
                ):
                    diff[name] += int(
                        (got_t[:M].view(torch.uint8) != want_t.view(torch.uint8)).sum()
                    )
            qr, qk = (
                query_ref(qkv, a, cos_sin, pos.long()).flatten(),
                eager[0].double().flatten(),
            )
            q_cos = min(
                q_cos, torch.nn.functional.cosine_similarity(qk, qr, dim=0).item()
            )
            q_rel = max(q_rel, ((qk - qr).norm() / qr.norm()).item())
        del graph
        bad = ([] if ok else ["wqkv route"]) + [k for k, v in diff.items() if v]
        bad += (["mix"] if mix > MIX_TOL else []) + (
            ["q"] if q_cos < Q_COS or q_rel > Q_REL else []
        )
        bad += ["qr"] if qr_rel > QR_REL else []
        if pr:
            # one fp4 / fp8 step flips where the GEMM's sum order moves a value across a
            # rounding edge: a few elements off, the rows otherwise SGLang's
            bad += ["comp rec"] if cst["rec"][0] < 0.999 or cst["rec"][1] > 0.02 else []
            bad += ["comp key"] if cst["key"][0] < 0.99 or cst["key"][1] > 0.05 else []
            bad += ["comp ring"] if cst["ring"] > 1e-4 else []
            bad += ["comp rows"] if not (crows[0] and crows[1]) else []
            if rank == 0:
                print(
                    f"M={M:2d} {vname:8s} compressor: record cos {cst['rec'][0]:.6f} "
                    f"differing {cst['rec'][1]:.4f} | index key cos {cst['key'][0]:.6f} "
                    f"differing {cst['key'][1]:.4f} | ring max rel {cst['ring']:.1e} | "
                    f"{crows[0]} records, {crows[1]} ring rows parked",
                    flush=True,
                )
        if rank == 0:
            print(f"M={M:2d} {vname:7s} {'PASS' if not bad else 'FAIL ' + ','.join(bad)} | qr rel {qr_rel:.1e} | wqkv_a {route} | differing: "
                  f"{diff} | pre/post/comb max|d| {mix:.2e} (bound {MIX_TOL:.0e}) | q cos {q_cos:.7f} "
                  f"rel {q_rel:.2e}", flush=True)  # fmt: skip
        fails += [f"M={M} {vname} {b}" for b in bad]
    finish(fails, dev)


if __name__ == "__main__":
    main()
