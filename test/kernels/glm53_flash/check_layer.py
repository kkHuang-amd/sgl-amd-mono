"""Run the GLM-5.3-Flash layer launch on random weights in SGLang's layouts under
torchrun (TP 2, 4 or 8): ``torchrun --nproc-per-node 4
test/kernels/glm53_flash/check_layer.py --kind dsa --rows 4``. Kinds: ``ffn`` (the
MoE alone), ``kda``, ``dsa``, ``dense`` (KDA and the dense FP8 MLP), ``embed`` (the
dense layer 0 from the step's tokens) and ``head`` (the tail). Every rank must
finish with no error word set and the same output (the TP sums run in rank order);
``--writes``: the launch changes exactly its valid rows' state or cache rows;
``--compare-shuffled``: a sparse-attention or dense launch on AITER-shuffled FP8
weights equals the row-major one bitwise; ``--check-routing``: the top-8 equals
SGLang's on keys that tie in bf16; ``--check-keys``: rows past 2048 tokens attend to
the selected pools (the 512 best of a torch scoring) and then their tail, as SGLang's
kpool top-k selects;
``--check-dense``: the dense MLP's output against an f32 MLP on the launch's own FFN
input; ``--check-embed``: the launch from the tokens equals the dense launch on the
gathered embedding rows, and writes the spec's step tables (``--step-rows``: the
step's rows, fewer than the width); ``--check-head``: the tail's logits against a
torch tail and an f32 tail; ``--replays N``: a graph of ``--layers`` launches
replays the eager step bitwise; ``--verify L`` (``kda`` or ``dsa``): an MTP verify
launch of L rows a request equals its tokens' L decode launches bitwise."""

import argparse
import os
import sys

import torch
import torch.distributed as dist

from sgl_amd_mono.glm53_flash.gfx950.config import (
    DENSE_INTER,
    GLOBAL_INTER,
    HC_MULT,
    HIDDEN,
    N_EXPERTS,
    VOCAB,
)
from sgl_amd_mono.glm53_flash.spec import GROUPED_FROM

N_MIX = HC_MULT * (2 + HC_MULT)


def _gen(seed: int, device) -> torch.Generator:
    return torch.Generator(device=device).manual_seed(seed)


def ffn_weights(rank: int, tp: int, device):
    """One MoE layer's random weights (every rank builds the same) as this rank's
    shard in SGLang's post-load AITER A4W4 layout; the shared expert is slot 288."""
    from aiter.ops.shuffle import shuffle_weight
    from aiter.utility.fp4_utils import e8m0_shuffle

    from sgl_amd_mono.glm53_flash.gfx950.packing import (
        pack_bf16,
    )
    from sgl_amd_mono.glm53_flash.gfx950.runner import (
        FlashFfnWeights,
    )

    g = _gen(0, device)
    E, full, inter = N_EXPERTS + 1, GLOBAL_INTER, GLOBAL_INTER // tp
    lo, hi = rank * inter, (rank + 1) * inter

    def mxfp4(rows, k):
        q = torch.randint(0, 256, (E, rows, k // 2), generator=g, device=device)
        s = torch.randint(119, 124, (E, rows, k // 32), generator=g, device=device)
        return q.to(torch.uint8), s.to(torch.uint8)

    def scales(s):
        e, r, c = s.shape
        return e8m0_shuffle(s.contiguous().view(e * r, c)).view(e, r, c)

    q13, s13 = mxfp4(2 * full, HIDDEN)
    q2, s2 = mxfp4(HIDDEN, full)
    fp4 = torch.float4_e2m1fn_x2
    rows13 = (
        lambda t: torch.cat(  # noqa: E731
            [t[:, lo:hi], t[:, full + lo : full + hi]], 1
        ).contiguous()
    )
    w_ug = shuffle_weight(rows13(q13).view(fp4), layout=(16, 16)).view(torch.uint8)
    s_ug = scales(rows13(s13))
    w_dn = shuffle_weight(
        q2[:, :, lo // 2 : hi // 2].contiguous().view(fp4), layout=(16, 16)
    ).view(torch.uint8)
    s_dn = scales(s2[:, :, lo // 32 : hi // 32])
    w_r = (torch.randn(N_EXPERTS, HIDDEN, generator=g, device=device) * 0.02).bfloat16()
    bias = (torch.rand(N_EXPERTS, generator=g, device=device) - 0.5) * 0.2
    return FlashFfnWeights(
        w_r=pack_bf16(w_r),
        bias=bias,
        w_ug=w_ug,
        s_ug=s_ug,
        w_dn=w_dn,
        s_dn=s_dn,
        w_sh_ug=w_ug[N_EXPERTS],
        s_sh_ug=s_ug[N_EXPERTS],
        w_sh_dn=w_dn[N_EXPERTS],
        s_sh_dn=s_dn[N_EXPERTS],
        hc_fn=torch.randn(N_MIX, HC_MULT * HIDDEN, generator=g, device=device) * 0.01,
        hc_scale=torch.rand(3, generator=g, device=device) + 0.5,
        hc_base=torch.randn(N_MIX, generator=g, device=device) * 0.1,
        g_norm=(1 + 0.1 * torch.randn(HIDDEN, generator=g, device=device)).bfloat16(),
    )


def kda_weights(rank: int, tp: int, device):
    """One KDA layer's random attention weights (every rank builds the same) as this
    rank's shard in SGLang's post-load layout."""
    from sgl_amd_mono.glm53_flash.gfx950.config import (
        KDA_DIM,
        KDA_HEADS,
        KDA_PROJ,
    )
    from sgl_amd_mono.glm53_flash.gfx950.runner import (
        FlashKdaWeights,
    )

    g = _gen(1, device)
    D, PF, HF = KDA_DIM, KDA_PROJ, KDA_HEADS
    P, H = PF // tp, HF // tp

    def rnd(*shape, scale=1.0):
        return torch.randn(*shape, generator=g, device=device) * scale

    def sh(t, n):
        return t[rank * n : (rank + 1) * n]

    wq, wk, wv = (rnd(PF, HIDDEN, scale=HIDDEN**-0.5) for _ in range(3))
    wb = rnd(HF, HIDDEN, scale=HIDDEN**-0.5)
    wfa, wga = (rnd(D, HIDDEN, scale=HIDDEN**-0.5) for _ in range(2))
    w_fb, w_gb = (rnd(PF, D, scale=D**-0.5).bfloat16() for _ in range(2))
    w_o = rnd(HIDDEN, PF, scale=PF**-0.5).bfloat16()
    conv = rnd(3, PF, 4, scale=0.5)
    a_log = torch.log(torch.rand(HF, generator=g, device=device) * 8 + 1)
    return FlashKdaWeights(
        hc_fn=rnd(N_MIX, HC_MULT * HIDDEN, scale=0.01),
        hc_scale=torch.rand(3, generator=g, device=device) + 0.5,
        hc_base=rnd(N_MIX, scale=0.1),
        g_norm=(1 + 0.1 * rnd(HIDDEN)).bfloat16(),
        w_in=torch.cat(
            [sh(wq, P), sh(wk, P), sh(wv, P), sh(wb, H), wfa, wga]
        ).bfloat16(),
        w_fb=sh(w_fb, P).contiguous(),
        w_gb=sh(w_gb, P).contiguous(),
        conv_w=torch.cat([sh(conv[i], P) for i in range(3)]).contiguous(),
        a_log=sh(a_log, H).contiguous(),
        dt_bias=sh(rnd(PF, scale=0.5), P).contiguous(),
        w_on=(1 + 0.1 * rnd(D)).float(),
        w_o=w_o[:, rank * P : (rank + 1) * P].contiguous(),
    )


def kda_state(rows: int, tp: int, device):
    """SGLang's KDA state pool: bf16 conv [slots, 3, 3P], f32 recurrent
    [slots, H, 128, 128]; slot 0 is the padding slot."""
    from sgl_amd_mono.glm53_flash.gfx950.config import (
        KDA_DIM,
        KDA_HEADS,
        KDA_PROJ,
    )
    from sgl_amd_mono.glm53_flash.gfx950.runner import (
        FlashKdaState,
    )

    g = _gen(3, device)
    P, H, slots = KDA_PROJ // tp, KDA_HEADS // tp, rows + 2
    conv = torch.randn(slots, 3, 3 * P, generator=g, device=device).bfloat16()
    rec = torch.randn(slots, H, KDA_DIM, KDA_DIM, generator=g, device=device) * 0.05
    idx = torch.arange(1, rows + 1, dtype=torch.int32, device=device)
    return FlashKdaState(conv_state=conv, rec_state=rec), idx


def _fp8_block(w):
    """FP8 E4M3 with f32 128 x 128 block scales: (q, scales)."""
    R, K = w.shape
    wb = w.float().view(R // 128, 128, K // 128, 128)
    s = wb.abs().amax((1, 3)) / 448.0 + 1e-12
    q = (wb / s[:, None, :, None]).to(torch.float8_e4m3fn)
    return q.view(R, K), s.contiguous()


def dsa_weights(rank: int, tp: int, device):
    """One sparse-attention layer's random weights as this rank's shard: q_a | kv_a,
    q_b and o_proj FP8 with block scales, as the amd/ checkpoint loads."""
    from sgl_amd_mono.glm53_flash.gfx950.config import (
        INDEX_DIM,
        INDEX_HEADS,
        KV_LORA,
        MLA_HEADS,
        NOPE_DIM,
        Q_LORA,
        V_DIM,
    )
    from sgl_amd_mono.glm53_flash.gfx950.runner import (
        FlashDsaWeights,
    )

    g = _gen(2, device)
    H, per = MLA_HEADS // tp, NOPE_DIM + V_DIM

    def rnd(*shape, scale=1.0):
        return torch.randn(*shape, generator=g, device=device) * scale

    w_qkv, s_qkv = _fp8_block(rnd(Q_LORA + KV_LORA, HIDDEN, scale=HIDDEN**-0.5))
    q_b, s_qb = _fp8_block(rnd(MLA_HEADS * NOPE_DIM, Q_LORA, scale=Q_LORA**-0.5))
    kv_b = rnd(MLA_HEADS * per, KV_LORA, scale=KV_LORA**-0.5)
    w_o, s_o = _fp8_block(
        rnd(HIDDEN, MLA_HEADS * V_DIM, scale=(MLA_HEADS * V_DIM) ** -0.5)
    )
    w_kvb = kv_b[rank * H * per : (rank + 1) * H * per].bfloat16().contiguous()
    return FlashDsaWeights(
        hc_fn=rnd(N_MIX, HC_MULT * HIDDEN, scale=0.01),
        hc_scale=torch.rand(3, generator=g, device=device) + 0.5,
        hc_base=rnd(N_MIX, scale=0.1),
        g_norm=(1 + 0.1 * rnd(HIDDEN)).bfloat16(),
        w_qkv=w_qkv,
        s_qkv=s_qkv,
        w_idx=rnd(INDEX_DIM + INDEX_HEADS, HIDDEN, scale=HIDDEN**-0.5).bfloat16(),
        w_gate=rnd(INDEX_DIM, HIDDEN, scale=HIDDEN**-0.5).bfloat16(),
        g_qa=(1 + 0.1 * rnd(Q_LORA)).bfloat16(),
        g_kv=(1 + 0.1 * rnd(KV_LORA)).bfloat16(),
        ln_w=1 + 0.1 * rnd(INDEX_DIM),
        ln_b=0.1 * rnd(INDEX_DIM),
        ape=0.5 * rnd(4, INDEX_DIM),
        w_qb=q_b[rank * H * NOPE_DIM : (rank + 1) * H * NOPE_DIM].contiguous(),
        s_qb=s_qb[rank * H * 2 : (rank + 1) * H * 2].contiguous(),
        w_iq=rnd(INDEX_HEADS * INDEX_DIM, Q_LORA, scale=Q_LORA**-0.5).bfloat16(),
        w_uk=w_kvb.view(H, per, KV_LORA)[:, :NOPE_DIM].transpose(1, 2).contiguous(),
        w_kvb=w_kvb,
        w_o=w_o[:, rank * H * V_DIM : (rank + 1) * H * V_DIM].contiguous(),
        s_o=s_o[:, rank * H * 2 : (rank + 1) * H * 2].contiguous(),
    )


MAX_LEN = 8192
# tokens of each row including the new one (0: a pad row); 4100, 1500, 7000, 64
# complete a pool, 2050 / 2049 / 3333 keep a tail
LENS = (4100, 1500, 2050, 7000, 2049, 64, 3333)


def dsa_caches(rows: int, device, lens=None):
    """SGLang's DSA caches for ``rows`` requests of up to MAX_LEN tokens (256-token
    pages; page 0 is the padding page), its 64-token page table and the step's row
    table."""
    from sgl_amd_mono.glm53_flash.gfx950.config import (
        INDEX_DIM,
        KPOOL,
        KV_LORA,
    )
    from sgl_amd_mono.glm53_flash.gfx950.layout import (
        ROW_BT,
        ROW_POOL_SLOT,
        ROW_POS,
        ROW_SLOT,
        ROW_TAIL_SLOT,
        ROW_WORDS,
    )
    from sgl_amd_mono.glm53_flash.gfx950.runner import (
        FlashDsaCaches,
    )

    g = _gen(5, device)
    per = MAX_LEN // 256
    pages = 1 + rows * per
    kv = torch.randn(pages * 256, KV_LORA, generator=g, device=device).bfloat16()
    idx = torch.zeros(pages, 64 * (INDEX_DIM + 4), dtype=torch.uint8, device=device)
    off = torch.arange(64, device=device)[:, None]
    d = torch.arange(INDEX_DIM, device=device)[None, :]
    perm = ((off // 16) * 2048 + (d // 16) * 256 + (off % 16) * 16 + d % 16).reshape(-1)
    vals = (torch.randn(pages, 64 * INDEX_DIM, generator=g, device=device) * 64).clamp(
        -448, 448
    )
    idx[:, perm] = vals.to(torch.float8_e4m3fn).view(torch.uint8)
    scales = torch.exp2(
        -torch.randint(6, 10, (pages, 64), generator=g, device=device).float()
    )
    idx[:, 64 * INDEX_DIM :] = scales.view(torch.uint8)
    tail_k = torch.randn(rows + 1, KPOOL, INDEX_DIM, generator=g, device=device)
    tail_g = torch.randn(rows + 1, KPOOL, INDEX_DIM, generator=g, device=device)
    order = torch.randperm(pages - 1, generator=g, device=device) + 1
    req_pages = order[: rows * per].view(rows, per)
    # 64-token page table: page p of a request is 64-token pages 4p .. 4p + 3
    pt = (req_pages[:, :, None] * 4 + torch.arange(4, device=device)).view(rows, -1)
    table = torch.full((rows, ROW_WORDS), -1, dtype=torch.int32, device=device)
    lens = lens or list(LENS[: rows - 1]) + [0]
    for r, n in enumerate(lens):
        table[r, ROW_BT] = r
        if n == 0:
            continue
        pos = n - 1
        slot = int(req_pages[r, pos // 256]) * 256 + pos % 256
        table[r, ROW_POS], table[r, ROW_SLOT] = pos, slot
        table[r, ROW_POOL_SLOT], table[r, ROW_TAIL_SLOT] = (
            slot // KPOOL,
            (r + 1) * KPOOL,
        )
    caches = FlashDsaCaches(
        kv_cache=kv,
        idx_cache=idx,
        tail_k=tail_k.bfloat16().contiguous(),
        tail_g=tail_g.bfloat16().contiguous(),
    )
    return caches, table, pt.int().contiguous()


def layer_inputs(rows: int, device):
    """The previous layer's (x, residual, post, comb), its FFN post owed."""
    g = _gen(7, device)
    x = torch.randn(rows, HIDDEN, generator=g, device=device).bfloat16()
    res = torch.randn(rows, HC_MULT, HIDDEN, generator=g, device=device).bfloat16()
    post = torch.rand(rows, HC_MULT, 1, generator=g, device=device) * 2
    comb = torch.softmax(
        torch.randn(rows, HC_MULT, HC_MULT, generator=g, device=device), -1
    )
    return x, res, post, comb


def dense_weights(rank: int, tp: int, device, shuffled: bool = False):
    """One dense layer's random FFN weights (every rank builds the same) as this
    rank's shard: gate | up and down FP8 with 128 x 128 block scales, row-major or in
    AITER's (16, 16) layout."""
    from aiter.ops.shuffle import shuffle_weight

    from sgl_amd_mono.glm53_flash.gfx950.runner import FlashDenseWeights

    g = _gen(4, device)
    I = DENSE_INTER // tp
    lo, hi = rank * I, (rank + 1) * I

    def rnd(*shape, scale=1.0):
        return torch.randn(*shape, generator=g, device=device) * scale

    gate, gs = _fp8_block(rnd(DENSE_INTER, HIDDEN, scale=HIDDEN**-0.5))
    up, us = _fp8_block(rnd(DENSE_INTER, HIDDEN, scale=HIDDEN**-0.5))
    down, ds = _fp8_block(rnd(HIDDEN, DENSE_INTER, scale=DENSE_INTER**-0.5))
    w_ug = torch.cat([gate[lo:hi], up[lo:hi]]).contiguous()
    s_ug = torch.cat(
        [gs[lo // 128 : hi // 128], us[lo // 128 : hi // 128]]
    ).contiguous()
    w_dn = down[:, lo:hi].contiguous()
    s_dn = ds[:, lo // 128 : hi // 128].contiguous()
    if shuffled:
        w_ug, w_dn = shuffle_weight(w_ug, (16, 16)), shuffle_weight(w_dn, (16, 16))
    return FlashDenseWeights(
        w_ug=w_ug,
        s_ug=s_ug,
        w_dn=w_dn,
        s_dn=s_dn,
        hc_fn=rnd(N_MIX, HC_MULT * HIDDEN, scale=0.01),
        hc_scale=torch.rand(3, generator=g, device=device) + 0.5,
        hc_base=rnd(N_MIX, scale=0.1),
        g_norm=(1 + 0.1 * rnd(HIDDEN)).bfloat16(),
        shuffled=shuffled,
    )


# vocabulary rows a rank holds in the test's embedding (the launch takes any count)
EMBED_ROWS = 1024


def embedding(rank: int, tp: int, device):
    """A random vocab-parallel embedding of ``EMBED_ROWS`` rows a rank: this rank's
    shard, and the whole table."""
    from sgl_amd_mono.glm53_flash.gfx950.runner import FlashEmbedding

    g = _gen(6, device)
    table = torch.randn(tp * EMBED_ROWS, HIDDEN, generator=g, device=device).bfloat16()
    lo = rank * EMBED_ROWS
    shard = table[lo : lo + EMBED_ROWS].contiguous()
    return FlashEmbedding(weight=shard, start=lo, end=lo + EMBED_ROWS), table


def step_inputs(n: int, width: int, tp: int, device) -> dict:
    """A step of ``n`` rows (int64, as SGLang's decode buffers hold them): tokens of
    every rank's shard, positions, MLA slots (row 1 SGLang's padding slot 0) and
    request rows; the KDA slot and row tables to write, of the width."""
    from sgl_amd_mono.glm53_flash.gfx950.layout import ROW_WORDS

    g = _gen(8, device)
    ids = torch.randint(0, tp * EMBED_ROWS, (n,), generator=g, device=device)
    pos = torch.randint(0, 4000, (n,), generator=g, device=device)
    loc = torch.randint(256, 1 << 20, (n,), generator=g, device=device)
    if n > 1:
        loc[1] = 0
    rpi = torch.randint(1, 60, (n,), generator=g, device=device)
    return {
        "ids": ids,
        "positions": pos,
        "out_cache_loc": loc,
        "req_pool_indices": rpi,
        "slots": torch.full((width,), 77, dtype=torch.int32, device=device),
        "rows": torch.full((width, ROW_WORDS), 77, dtype=torch.int32, device=device),
    }


def spec_tables(step: dict, slots: torch.Tensor, width: int):
    """The KDA slot and row tables as the spec's own (torch) tables hold them."""
    from sgl_amd_mono.glm53_flash.gfx950.config import KPOOL
    from sgl_amd_mono.glm53_flash.gfx950.layout import (
        ROW_BT,
        ROW_POOL_SLOT,
        ROW_POS,
        ROW_SLOT,
        ROW_TAIL_SLOT,
    )

    n = step["ids"].shape[0]
    kda = torch.zeros(width, dtype=torch.int32, device=slots.device)
    kda[:n] = slots[:n]
    rows = step["rows"].clone()
    rows[:, : ROW_TAIL_SLOT + 1] = -1
    rows[:, ROW_BT] = 0
    loc = step["out_cache_loc"]
    valid = loc != 0
    none = torch.full_like(loc, -1)
    rows[:n, ROW_POS] = torch.where(valid, step["positions"], none).int()
    rows[:n, ROW_SLOT] = torch.where(valid, loc, none).int()
    rows[:n, ROW_POOL_SLOT] = torch.where(valid, loc // KPOOL, none).int()
    rows[:n, ROW_TAIL_SLOT] = torch.where(
        valid, step["req_pool_indices"] * KPOOL, none
    ).int()
    rows[:n, ROW_BT] = torch.arange(n, dtype=torch.int32, device=slots.device)
    return kda, rows


def head_weights(rank: int, tp: int, device):
    """The tail's random final-norm weight and this rank's LM-head rows (every rank
    builds the same whole head)."""
    from sgl_amd_mono.glm53_flash.gfx950.runner import FlashHeadWeights

    g = _gen(9, device)
    rows = VOCAB // tp
    w = torch.empty(rows, HIDDEN, dtype=torch.bfloat16, device=device)
    for r in range(tp):  # this rank's rows of the same random head, a shard at a time
        part = (torch.randn(rows, HIDDEN, generator=g, device=device) * 0.02).bfloat16()
        if r == rank:
            w.copy_(part)
    norm = (1 + 0.1 * torch.randn(HIDDEN, generator=g, device=device)).bfloat16()
    return FlashHeadWeights(g_norm=norm, w_head=w)


def shuffled_copy(w):
    """``w`` with its FP8 projections in AITER's (16, 16) layout, as SGLang's gfx950
    block-FP8 linear method shuffles them in place."""
    import msgspec
    from aiter.ops.shuffle import shuffle_weight

    return msgspec.structs.replace(
        w,
        w_qkv=shuffle_weight(w.w_qkv, (16, 16)),
        w_qb=shuffle_weight(w.w_qb, (16, 16)),
        w_o=shuffle_weight(w.w_o, (16, 16)),
        shuffled=(True, True, True),
    )


def compare_shuffled(args, rank, tp, group, device) -> bool:
    """One sparse-attention or dense launch on row-major and on AITER-shuffled FP8
    weights: the outputs and every cache or state write must be bitwise equal."""
    if args.kind == "dense":
        results = []
        for shuffled in (False, True):
            rt, launches, held, ins = build(
                args, "dense", rank, tp, group, device, shuffled=shuffled
            )
            op, kw = launches[0]
            out = op.launch(*ins, **kw)
            torch.cuda.synchronize()
            results.append([t.clone() for t in (*out, *held)])
            rt.close()
        same = all(torch.equal(a, b) for a, b in zip(*results))
        print(f"rank {rank}: dense shuffled == row-major: {same}", flush=True)
        return same
    from msgspec.structs import astuple, replace

    from sgl_amd_mono.glm53_flash.gfx950.kernel import (
        FlashBuild,
    )
    from sgl_amd_mono.glm53_flash.gfx950.runner import (
        FlashLayerOp,
        FlashRuntime,
    )

    key = FlashBuild(
        rows=args.rows,
        tp=tp,
        launches_per_step=1,
        inter=GLOBAL_INTER // tp,
        dsa=True,
        max_pools=args.max_pools,
        stop=False,
    )
    ffn, w = ffn_weights(rank, tp, device), dsa_weights(rank, tp, device)
    x, res, post, comb = layer_inputs(args.rows, device)
    lens = [int(v) for v in args.lens.split(",")] if args.lens else None
    caches, table, pt = dsa_caches(args.rows, device, lens)
    results = []
    for k, weights in (
        (key, w),
        (
            replace(key, qkv_shuffled=True, qb_shuffled=True, o_shuffled=True),
            shuffled_copy(w),
        ),
    ):
        rt = FlashRuntime(k, rank, group, device)
        rt.set_deadline(args.deadline)
        op = FlashLayerOp(rt, ffn, 0, dsa=weights)
        c = type(caches)(*(t.clone() for t in astuple(caches)))
        out = op.launch(x, res, post, comb, caches=c, rows=table, page_table=pt)
        torch.cuda.synchronize()
        results.append((tuple(t.clone() for t in out), c))
        rt.close()
    (out_a, c_a), (out_b, c_b) = results
    same_out = all(torch.equal(a, b) for a, b in zip(out_a, out_b))
    same_cache = all(torch.equal(a, b) for a, b in zip(astuple(c_a), astuple(c_b)))
    wrote = not torch.equal(c_a.kv_cache, caches.kv_cache)
    print(
        f"rank {rank}: shuffled == row-major: outputs {same_out}, caches {same_cache} "
        f"(kv written {wrote})",
        flush=True,
    )
    return same_out and same_cache and wrote


# router weights and biases whose keys tie after bf16 rounding (see tie_router)
TIE_GROUPS, TIE_MEMBERS = 24, 12
TIE_DELTA = 2.0**-12


def tie_router(ffn, device):
    """``ffn`` with router weights and biases whose keys tie after bf16 rounding: 24
    groups of 12 experts at random ids. Group g reads column j_g with weight 1 and has
    bias b_g (bf16); its odd members add +-TIE_DELTA x column j2_g, and every member's
    f32 bias is b_g plus less than a quarter bf16 ulp. The logits are exact in f32 (two
    nonzero products), so the keys tie exactly in bf16 and differ in f32. Returns the
    weights and the plain router [288, HIDDEN] bf16."""
    from msgspec.structs import replace

    from sgl_amd_mono.glm53_flash.gfx950.packing import pack_bf16

    g = _gen(11, device)
    assert TIE_GROUPS * TIE_MEMBERS == N_EXPERTS
    ids = torch.randperm(N_EXPERTS, generator=g, device=device).view(
        TIE_GROUPS, TIE_MEMBERS
    )
    cols = torch.randperm(HIDDEN, generator=g, device=device)[: 2 * TIE_GROUPS]
    cols = cols.view(2, TIE_GROUPS)
    b = ((torch.rand(TIE_GROUPS, generator=g, device=device) - 0.5) * 0.4).bfloat16()
    b = b.float()
    ulp = torch.exp2(torch.floor(torch.log2(b.abs())) - 7)
    w = torch.zeros(N_EXPERTS, HIDDEN, device=device)
    bias = torch.zeros(N_EXPERTS, device=device)
    for grp in range(TIE_GROUPS):
        for m in range(TIE_MEMBERS):
            e = int(ids[grp, m])
            w[e, cols[0, grp]] = 1.0
            if m % 2:
                w[e, cols[1, grp]] = TIE_DELTA if m % 4 == 1 else -TIE_DELTA
            eps = (torch.rand((), generator=g, device=device) - 0.5) * 0.5 * ulp[grp]
            bias[e] = b[grp] + eps
    w = w.bfloat16()
    return replace(ffn, w_r=pack_bf16(w), bias=bias), w


def ffn_input(rt, rows: int) -> torch.Tensor:
    """The launch's own RMSNorm'd FFN input rows (bf16), from its scratch."""
    a = rt.scratch[rt.sc["a"] :].view(torch.int32)[: rows * HIDDEN : 2]
    return a.contiguous().view(torch.bfloat16).view(rows, HIDDEN)


def routed(rt, rows: int) -> torch.Tensor:
    """The launch's 8 routed experts a row, best first (slot 0 is the shared expert)."""
    from sgl_amd_mono.glm53_flash.gfx950.config import MOE_SLOTS

    sel = rt.scratch[rt.sc["sel"] :].view(torch.int32)[: 2 * rows * MOE_SLOTS : 2]
    return sel.view(rows, MOE_SLOTS)[:, 1:].clone()


def stock_topk(x, w_r, bias) -> torch.Tensor:
    """SGLang's top-8 for GLM-5.3-Flash with AITER: bf16 logits and bias into AITER's
    ``biased_grouped_topk`` (one group, renormalized, scale 2.5), best first."""
    from aiter import biased_grouped_topk

    from sgl_amd_mono.glm53_flash.gfx950.config import ROUTE_SCALE, TOP_K

    logits = (x.float() @ w_r.float().T).bfloat16()
    weights = torch.empty(x.shape[0], TOP_K, dtype=torch.float32, device=x.device)
    ids = torch.empty(x.shape[0], TOP_K, dtype=torch.int32, device=x.device)
    biased_grouped_topk(logits, bias.bfloat16(), weights, ids, 1, 1, True, ROUTE_SCALE)
    return ids


def key_order(x, w_r, bias, rounded: bool, tie: str) -> torch.Tensor:
    """Top-8 ids of sigmoid(logits) + bias, best first: logits and bias in bf16
    (``rounded``) or f32; equal keys in AITER's lane order (``tie="aiter"``) or by
    lower id (``tie="id"``)."""
    from sgl_amd_mono.glm53_flash.gfx950.config import TOP_K
    from sgl_amd_mono.glm53_flash.gfx950.kernel import tie_rank

    logits, b = x.float() @ w_r.float().T, bias.float()
    if rounded:
        logits, b = logits.bfloat16().float(), b.bfloat16().float()
    keys = torch.sigmoid(logits) + b
    ids = torch.arange(N_EXPERTS, device=x.device)
    if tie == "aiter":
        rank = torch.tensor([tie_rank(e) for e in range(N_EXPERTS)], device=x.device)
    else:
        rank = ids
    by_rank = torch.argsort(rank)
    order = torch.argsort(-keys[:, by_rank], dim=-1, stable=True)
    return by_rank[order][:, :TOP_K].int()


def check_routing(args, rank, tp, group, device) -> bool:
    """FFN launches on router keys that tie in bf16: the launch's top-8 (order
    included) equals SGLang's (AITER's ``biased_grouped_topk`` on bf16 logits and bias)
    on the launch's own FFN input, every draw; and the inputs tell that apart from
    lower-id ties and from f32 keys (negative controls). The host-side key order is
    printed as a diagnostic only: it restates the kernel's ``tie_rank``."""
    from sgl_amd_mono.glm53_flash.gfx950.kernel import FlashBuild
    from sgl_amd_mono.glm53_flash.gfx950.runner import FlashLayerOp, FlashRuntime

    key = FlashBuild(
        rows=args.rows, tp=tp, launches_per_step=1, inter=GLOBAL_INTER // tp, stop=False
    )
    ffn, w_r = tie_router(ffn_weights(rank, tp, device), device)
    rt = FlashRuntime(key, rank, group, device)
    rt.set_deadline(args.deadline)
    op = FlashLayerOp(rt, ffn, 0)
    ok, by_id, by_f32, rows = True, 0, 0, 0
    for draw in range(3):
        g = _gen(20 + draw, device)
        x = torch.randn(args.rows, HIDDEN, generator=g, device=device).bfloat16()
        _, res, post, comb = layer_inputs(args.rows, device)
        op.launch(x / tp, res, post, comb)
        rt.advance_step()
        torch.cuda.synchronize()
        a, got = ffn_input(rt, args.rows), routed(rt, args.rows)
        same = torch.equal(got, stock_topk(a, w_r, ffn.bias))
        own = torch.equal(got, key_order(a, w_r, ffn.bias, True, "aiter"))
        print(
            f"rank {rank} draw {draw}: top-8 == AITER biased_grouped_topk {same} "
            f"(diagnostic: == bf16 keys in tie_rank order {own})",
            flush=True,
        )
        ok = ok and same
        by_id += int((got != key_order(a, w_r, ffn.bias, True, "id")).any(-1).sum())
        by_f32 += int(
            (got != key_order(a, w_r, ffn.bias, False, "aiter")).any(-1).sum()
        )
        rows += args.rows
    print(
        f"rank {rank}: lower-id ties would differ in {by_id}/{rows} rows, f32 keys "
        f"in {by_f32}/{rows}",
        flush=True,
    )
    rt.close()
    return ok and by_id > 0 and by_f32 > 0


# row lengths for the key check: scored rows with tails of 2, 3 and 1 tokens, and a
# row of 2051 tokens (2048 + a tail of 3), which attends to every token
KEY_LENS = (4102, 5003, 7001, 2051)


def kv_slot(pt: torch.Tensor, row: int, token: int) -> int:
    """MLA cache row of ``token`` of request ``row`` through the 64-token page table."""
    return int(pt[row, token // 64]) * 64 + token % 64


def check_keys(args, rank, tp, group, device) -> bool:
    """Sparse-attention launches on rows past 2048 tokens. A scored row (more than 512
    complete pools) attends to the selected pools' 2048 tokens and then its tail (up to
    2051 keys), as SGLang's kpool top-k appends it: its selection holds 512 whole
    distinct complete pools, the 512 best of a torch scoring of the launch's index
    query, then the tail tokens in order. Perturbing the MLA rows of
    the tokens a 2048-key selection gives up for the tail (its last selected pool's last
    positions) changes the row's output (negative control: a 2048-key selection never
    reads them); perturbing an unselected pool changes nothing. A row of 2051 tokens
    attends to every token: perturbing its first token changes its output."""
    from msgspec.structs import replace

    from sgl_amd_mono.glm53_flash.gfx950.config import KPOOL, TOPK_POOLS, TOPK_TOKENS
    from sgl_amd_mono.glm53_flash.gfx950.kernel import FlashBuild
    from sgl_amd_mono.glm53_flash.gfx950.layout import SEL_STRIDE
    from sgl_amd_mono.glm53_flash.gfx950.runner import FlashLayerOp, FlashRuntime

    key_lens = [int(v) for v in args.lens.split(",")] if args.lens else list(KEY_LENS)
    rows = len(key_lens)
    key = FlashBuild(
        rows=rows,
        tp=tp,
        launches_per_step=1,
        inter=GLOBAL_INTER // tp,
        dsa=True,
        max_pools=args.max_pools,
        stop=False,
    )
    rt = FlashRuntime(key, rank, group, device)
    rt.set_deadline(args.deadline)
    op = FlashLayerOp(
        rt, ffn_weights(rank, tp, device), 0, dsa=dsa_weights(rank, tp, device)
    )
    x, res, post, comb = layer_inputs(rows, device)
    caches, table, pt = dsa_caches(rows, device, key_lens)

    def run(kv):
        c = replace(
            caches,
            kv_cache=kv,
            idx_cache=caches.idx_cache.clone(),
            tail_k=caches.tail_k.clone(),
            tail_g=caches.tail_g.clone(),
        )
        out = op.launch(x, res, post, comb, caches=c, rows=table, page_table=pt)
        rt.advance_step()
        torch.cuda.synchronize()
        sel = rt.scratch[rt.sc["tok_sel"] :].view(torch.int32)[: 2 * rows * SEL_STRIDE]
        return out[1].clone(), sel.view(rows, SEL_STRIDE, 2).clone(), c.idx_cache

    base, sel, idx = run(caches.kv_cache.clone())
    pool_scores = index_scores(rt, rows, idx, pt)
    # (token, launch tag) pairs; kv_ready holds this launch's tag
    tags = rt.scratch[rt.sc["kv_ready"] :].view(torch.int32)[1 : 2 * rows : 2].clone()
    sel, sel_tags = sel[..., 0], sel[..., 1]
    g = _gen(30, device)
    ok = True
    for r, n in enumerate(key_lens):
        bound, tail = n // KPOOL, n % KPOOL
        if bound <= TOPK_POOLS:
            victims, spare = [0], None
        else:
            picked = sel[r, :TOPK_TOKENS].long()
            pools = picked.view(TOPK_POOLS, KPOOL)
            whole = bool(
                (pools == pools[:, :1] + torch.arange(KPOOL, device=device)).all()
            )
            whole = whole and bool((pools[:, 0] % KPOOL == 0).all())
            ids = pools[:, 0] // KPOOL
            distinct = ids.unique().numel() == TOPK_POOLS and int(ids.max()) < bound
            tail_ok = sel[r, TOPK_TOKENS : TOPK_TOKENS + tail].tolist() == [
                bound * KPOOL + i for i in range(tail)
            ]
            tagged = bool((sel_tags[r, : TOPK_TOKENS + tail] == tags[r]).all())
            missed, worse = off_top_pools(pool_scores(r, bound), ids)
            top = missed == worse == 0
            print(
                f"rank {rank} row {r} ({n} tokens): 512 whole distinct pools "
                f"{whole and distinct}, tail {tail_ok}, tagged {tagged}, the torch "
                f"top-512 {top} ({missed} better pools missed, {worse} worse picked)",
                flush=True,
            )
            ok = ok and whole and distinct and tail_ok and tagged and top
            victims = picked[TOPK_TOKENS - max(tail, 1) : TOPK_TOKENS].tolist()
            unpicked = sorted(set(range(bound)) - set(ids.tolist()))
            spare = unpicked[0] * KPOOL
        kv = caches.kv_cache.clone()
        for t in victims:
            kv[kv_slot(pt, r, t)] = (
                30 * torch.randn(kv.shape[1], generator=g, device=device).bfloat16()
            )
        moved = not torch.equal(run(kv)[0][r], base[r])
        print(
            f"rank {rank} row {r}: perturbing tokens {victims} changes the row {moved}",
            flush=True,
        )
        ok = ok and moved
        if spare is not None:
            kv = caches.kv_cache.clone()
            for t in range(spare, spare + KPOOL):
                kv[kv_slot(pt, r, t)] = (
                    30 * torch.randn(kv.shape[1], generator=g, device=device).bfloat16()
                )
            still = torch.equal(run(kv)[0][r], base[r])
            print(
                f"rank {rank} row {r}: perturbing unselected pool {spare // KPOOL} "
                f"leaves the row unchanged {still}",
                flush=True,
            )
            ok = ok and still
    rt.close()
    return ok


def index_scores(rt, rows, idx, pt):
    """``scores(r, bound)``: f64 index scores of row r's complete pools from the
    launch's FP8 query and head weights and the index cache it read, as the kernel
    defines them: scale(pool) * sum_h w_h * relu(q_h . k_pool)."""
    from sgl_amd_mono.glm53_flash.gfx950.config import INDEX_DIM, INDEX_HEADS, KPOOL

    def words(name, n):
        return rt.scratch[rt.sc[name] :].view(torch.int32)[: 2 * n].view(n, 2)[:, 0]

    f8 = torch.float8_e4m3fn
    QW = INDEX_HEADS * INDEX_DIM // 4
    q = words("iq8", rows * QW).contiguous().view(torch.uint8).view(f8)
    q = q.double().view(rows, INDEX_HEADS, INDEX_DIM)
    w = words("iqw", rows * INDEX_HEADS).contiguous().view(torch.float32).double()
    w = w.view(rows, INDEX_HEADS)
    device = idx.device
    off = torch.arange(64, device=device)[:, None]
    d = torch.arange(INDEX_DIM, device=device)[None, :]
    perm = ((off // 16) * 2048 + (d // 16) * 256 + (off % 16) * 16 + d % 16).reshape(-1)
    keys = idx[:, perm].view(f8).view(-1, 64, INDEX_DIM)
    scale = idx[:, 64 * INDEX_DIM :].contiguous().view(torch.float32)

    def scores(r, bound):
        p = torch.arange(bound, device=device)
        page = pt[r, p * KPOOL // 64].long() // 4
        k = keys[page, p % 64].double()
        s = (k @ q[r].T).clamp_min(0) @ w[r]
        return s * scale[page, p % 64].double()

    return scores


def off_top_pools(ref, ids):
    """(pools above the 512th best of ``ref`` that ``ids`` misses, picked pools below
    it): a pool within the f32 rounding of the 512th best counts either way."""
    from sgl_amd_mono.glm53_flash.gfx950.config import TOPK_POOLS

    kth = ref.topk(TOPK_POOLS).values[-1]
    tol = 1e-5 * ref.abs().max()
    picked = torch.zeros_like(ref, dtype=torch.bool)
    picked[ids] = True
    missed = int((~picked & (ref > kth + tol)).sum())
    worse = int((picked & (ref < kth - tol)).sum())
    return missed, worse


KINDS = ("ffn", "kda", "dsa", "dense", "embed", "head")


def build(args, kind, rank, tp, group, device, layers=1, shuffled=False):
    """A runtime of ``kind`` at ``args.rows`` rows, ``layers`` launches a step: each
    layer's op and its launch arguments (its own state or caches), the tensors those
    launches write, and the step's input. An ``embed`` step has ``args.step_rows``
    rows (default: the width)."""
    from sgl_amd_mono.glm53_flash.gfx950.kernel import FlashBuild
    from sgl_amd_mono.glm53_flash.gfx950.runner import (
        FlashHeadOp,
        FlashLayerOp,
        FlashRuntime,
    )

    kda, dsa = kind in ("kda", "dense", "embed"), kind == "dsa"
    dense = kind in ("dense", "embed")
    key = FlashBuild(
        rows=args.rows,
        tp=tp,
        launches_per_step=layers,
        inter=GLOBAL_INTER // tp,
        kda=kda,
        dsa=dsa,
        max_pools=args.max_pools if dsa else 0,
        qkv_shuffled=shuffled and dsa,
        qb_shuffled=shuffled and dsa,
        o_shuffled=shuffled and dsa,
        dense=dense,
        dense_shuffled=shuffled and dense,
        tokens=kind == "embed",
        head=kind == "head",
        grouped=args.rows >= GROUPED_FROM and not dense and kind != "head",
        stop=False,
    )
    rt = FlashRuntime(key, rank, group, device)
    rt.set_deadline(args.deadline)
    x, res, post, comb = layer_inputs(args.rows, device)
    if kind == "head":
        w = head_weights(rank, tp, device)
        ops = [FlashHeadOp(rt, w, n) for n in range(layers)]
        return rt, [(op, {}) for op in ops], [], (x, res, post, comb)
    ffn = dense_weights(rank, tp, device, shuffled) if dense else None
    ffn = ffn or ffn_weights(rank, tp, device)
    attn = kda_weights(rank, tp, device) if kda else None
    attn = dsa_weights(rank, tp, device) if dsa else attn
    emb = embedding(rank, tp, device)[0] if kind == "embed" else None
    if kind == "ffn":
        x = x / tp  # the FFN launch takes this rank's unreduced attention share
    if kind == "embed":
        x, res = torch.zeros_like(x), torch.zeros_like(res)
        post, comb = torch.ones_like(post), torch.zeros_like(comb)
    launches, held = [], []
    valid = args.rows - args.pad_rows
    for n in range(layers):
        op = FlashLayerOp(
            rt,
            ffn,
            n,
            kda=attn if kda else None,
            dsa=attn if dsa else None,
            embedding=emb,
        )
        kw = {}
        if kda:
            state, idx = kda_state(args.rows, tp, device)
            idx[valid:] = 0  # not positive: a pad row
            kw = {"state": state, "state_idx": idx}
            held += [state.conv_state, state.rec_state]
        if kind == "embed":
            step = step_inputs(args.step_rows or args.rows, args.rows, tp, device)
            kw["step"] = step
            kw["state_idx"] = idx[: step["ids"].shape[0]].clone()
            held += [step["slots"], step["rows"]]
        if dsa:
            from sgl_amd_mono.glm53_flash.gfx950.layout import ROW_BT

            lens = [int(v) for v in args.lens.split(",")] if args.lens else list(LENS)
            # the valid rows past the given lengths: SGLang's padding (slot 0)
            lens = lens[:valid] + [0] * (args.rows - min(valid, len(lens)))
            caches, table, pt = dsa_caches(args.rows, device, lens)
            # SGLang's page table has the step's rows; the width's pad rows read row 0
            pt = pt[:valid].clone()
            table[valid:, ROW_BT] = 0
            kw = {"caches": caches, "rows": table, "page_table": pt}
            held += [caches.kv_cache, caches.idx_cache, caches.tail_k, caches.tail_g]
        launches.append((op, kw))
    return rt, launches, held, (x, res, post, comb)


def run_step(kind, launches, ins):
    """One step: the launches in order, each on the previous one's outputs (the FFN
    and the tail launches each on the step's input)."""
    h = ins
    for op, kw in launches:
        if kind in ("ffn", "head"):
            h = op.launch(*ins, **kw)
        else:
            h = op.launch(*h, **kw)
    return h if isinstance(h, tuple) else (h,)


def check_replays(args, rank, tp, group, device) -> bool:
    """A step of ``--layers`` launches of one kind on one runtime (each layer its own
    state or caches, inputs chained), captured in a graph after an eager step: from the
    same state, each of ``--replays`` replays gives the eager step's outputs and writes
    bitwise, with no error word set, and every rank the same output. A step of tails
    runs them back to back (each tail's logits in the other slot of the last)."""
    from sgl_amd_mono.common import guard

    rt, launches, held, ins = build(
        args, args.kind, rank, tp, group, device, args.layers
    )
    saved = [t.clone() for t in held]

    def restore():
        for t, v in zip(held, saved):
            t.copy_(v)

    def step():
        out = run_step(args.kind, launches, ins)
        rt.advance_step()
        return out

    # eager first: the launch loads its code outside the capture
    want = [t.clone() for t in step()]
    torch.cuda.synchronize()
    want_held = [t.clone() for t in held]
    restore()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        got = step()
    ok = True
    for n in range(args.replays):
        restore()
        dist.barrier(group=group)
        graph.replay()
        torch.cuda.synchronize()
        same = all(torch.equal(a, b) for a, b in zip(got, want)) and all(
            torch.equal(a, b) for a, b in zip(held, want_held)
        )
        if not same:
            print(f"rank {rank} replay {n}: differs from the eager step", flush=True)
            ok = False
            break
    words = rt.epoch[guard.ERR_DEADLINE : guard.ERR_FAULT + 1].tolist()
    xs = [None] * tp
    dist.all_gather_object(xs, got[0].cpu())
    agree = all(torch.equal(xs[0], t) for t in xs)
    print(
        f"rank {rank}: {args.replays} replays of {args.layers} {args.kind} launches "
        f"equal the eager step {ok}, error words {words}, ranks agree {agree}",
        flush=True,
    )
    rt.close()
    return ok and not any(words) and agree


def _dequant(w, s, shuffled: bool) -> torch.Tensor:
    """A block-FP8 weight (row-major or AITER (16, 16)-shuffled) in f32."""
    n, k = w.shape
    if shuffled:
        w = w.view(torch.uint8).reshape(n // 16, k // 32, 2, 16, 16)
        w = w.permute(0, 3, 1, 2, 4).reshape(n, k).view(torch.float8_e4m3fn)
    return w.float() * s.repeat_interleave(128, 0).repeat_interleave(128, 1)


def _rel(a, b) -> float:
    a, b = a.float(), b.float()
    return ((a - b).norm() / b.norm()).item()


# the spec's dense.x_vs_fp32.rel
DENSE_X_VS_FP32 = 1.2e-2


def check_dense(args, rank, tp, group, device) -> bool:
    """A dense launch's MLP output against an f32 MLP (dequantized weights, the
    clamped SwiGLU) on the launch's own FFN input, summed over the ranks: within the
    spec's ``dense.x_vs_fp32.rel``. With one rank's down scales doubled the launch
    must fail it (negative control)."""
    import msgspec

    from sgl_amd_mono.glm53_flash.gfx950.config import SWIGLU_LIMIT

    torch.backends.cuda.matmul.allow_tf32 = False
    results = []
    for control in (False, True):
        rt, launches, _, ins = build(args, "dense", rank, tp, group, device)
        op, kw = launches[0]
        w = op.w
        if control and rank == 0:
            op.w = msgspec.structs.replace(w, s_dn=w.s_dn * 2)
        out = op.launch(*ins, **kw)
        torch.cuda.synchronize()
        a = ffn_input(rt, args.rows).float()
        gate, up = (a @ _dequant(w.w_ug, w.s_ug, w.shuffled).T).chunk(2, -1)
        mid = torch.nn.functional.silu(gate.clamp(max=SWIGLU_LIMIT))
        mid = mid * up.clamp(-SWIGLU_LIMIT, SWIGLU_LIMIT)
        x32 = mid @ _dequant(w.w_dn, w.s_dn, w.shuffled).T
        dist.all_reduce(x32)
        valid = args.rows - args.pad_rows
        results.append(_rel(out[0][:valid], x32[:valid]))
        rt.close()
    ok = results[0] <= DENSE_X_VS_FP32 < results[1]
    print(
        f"rank {rank}: dense x vs f32 MLP rel {results[0]:.3e} (bound "
        f"{DENSE_X_VS_FP32}); one rank's down scales x2: {results[1]:.3e} (caught "
        f"{results[1] > DENSE_X_VS_FP32})",
        flush=True,
    )
    return ok


def check_embed(args, rank, tp, group, device) -> bool:
    """The launch from the step's tokens against the dense launch on the gathered
    embedding rows (x = the row on every rank; post 1, comb 0, residual 0) on the same
    weights and state: outputs and state writes bitwise equal, the width's rows past
    the step's pads; and the step tables it writes equal the spec's torch tables."""

    def decode_state(s):
        return (s.conv_state, s.rec_state)

    rt, launches, _, ins = build(args, "embed", rank, tp, group, device)
    op, kw = launches[0]
    step = kw["step"]
    n = step["ids"].shape[0]
    _, table = embedding(rank, tp, device)
    state0 = [t.clone() for t in decode_state(kw["state"])]
    out = [t.clone() for t in op.launch(*ins, **kw)]
    torch.cuda.synchronize()
    kda, rows = spec_tables(step, kw["state_idx"], args.rows)
    tables = torch.equal(step["slots"], kda) and torch.equal(step["rows"], rows)
    ref_rt, ref_launches, _, _ = build(args, "dense", rank, tp, group, device)
    ref_op, ref_kw = ref_launches[0]
    for t, v in zip(decode_state(ref_kw["state"]), state0):
        t.copy_(v)
    x = torch.zeros(args.rows, HIDDEN, dtype=torch.bfloat16, device=device)
    x[:n] = table[step["ids"]]
    ref = ref_op.launch(x, *ins[1:], state=ref_kw["state"], state_idx=kda)
    torch.cuda.synchronize()
    same = all(torch.equal(a, b) for a, b in zip(out, ref))
    writes = all(
        torch.equal(a, b)
        for a, b in zip(decode_state(kw["state"]), decode_state(ref_kw["state"]))
    )
    print(
        f"rank {rank}: {n} of {args.rows} rows from the tokens == the dense launch on "
        f"their rows: outputs {same}, state writes {writes}; step tables == the "
        f"spec's {tables}",
        flush=True,
    )
    rt.close()
    ref_rt.close()
    return same and writes and tables


def check_head(args, rank, tp, group, device) -> bool:
    """The tail's logits against a torch tail on the same input (the four streams
    rounded to bf16 as hc_post rounds them, their mean in bf16, the RMSNorm'd rows in
    bf16, the LM head, bf16 logits) and an f32 tail; ``--launches`` back-to-back tails give the same
    logits; with the final norm x 1.01 the logits must move well past the launch's
    own distance (negative control)."""
    import msgspec

    from sgl_amd_mono.glm53_flash.gfx950.config import EPS

    torch.backends.cuda.matmul.allow_tf32 = False
    rt, launches, _, ins = build(args, "head", rank, tp, group, device)
    op, _ = launches[0]
    x, res, post, comb = ins
    outs = []
    for _ in range(args.launches):
        outs.append(op.launch(*ins).clone())
        rt.advance_step()
    torch.cuda.synchronize()
    repeat = all(torch.equal(outs[0], o) for o in outs)
    S = x.shape[0]
    streams = post.reshape(S, -1, 1) * x.float()[:, None] + torch.einsum(
        "sij,sih->sjh", comb, res.float()
    )

    def tail(h, bf16):
        n = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + EPS) * op.w.g_norm.float()
        shard = (n.bfloat16().float() if bf16 else n) @ op.w.w_head.float().T
        parts = [torch.empty_like(shard) for _ in range(tp)]
        dist.all_gather(parts, shard.contiguous())
        out = torch.cat(parts, -1)
        return out.bfloat16().float() if bf16 else out

    ref = tail(streams.bfloat16().float().mean(1).bfloat16().float(), True)
    ref32 = tail(streams.mean(1), False)
    rel, rel32 = _rel(outs[0], ref), _rel(outs[0], ref32)
    op.w = msgspec.structs.replace(op.w, g_norm=(op.w.g_norm.float() * 1.01).bfloat16())
    bad = _rel(op.launch(*ins), ref)
    torch.cuda.synchronize()
    caught = bad > 4 * rel
    print(
        f"rank {rank}: logits vs torch tail rel {rel:.3e}, vs f32 tail {rel32:.3e}; "
        f"{args.launches} back-to-back tails equal {repeat}; final norm x1.01: rel "
        f"{bad:.3e} (caught {caught})",
        flush=True,
    )
    rt.close()
    return rel < 2e-3 and rel32 < 1e-2 and repeat and caught


def _max_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().max()) if a.numel() else 0.0


def check_verify(args, rank, tp, group, device) -> bool:
    """An MTP verify launch (``--verify`` rows a request, ``--rows`` wide: whole
    requests, then pad rows) against the requests' draft tokens run one decode
    launch a token, on the same weights, inputs, state and caches: each row's output
    equals its token's decode output; with KDA the verify scratch after token j
    equals the state after j + 1 decode launches and the committed state is
    unchanged; with sparse attention the MLA cache, the index cache and the kpool
    rings equal the decode launches' (a chain's pool reads the ring entries its
    earlier rows wrote)."""
    from msgspec.structs import replace

    from sgl_amd_mono.common import guard
    from sgl_amd_mono.glm53_flash.gfx950.config import (
        INDEX_DIM,
        KDA_DIM,
        KDA_HEADS,
        KDA_PROJ,
        KPOOL,
    )
    from sgl_amd_mono.glm53_flash.gfx950.kernel import FlashBuild
    from sgl_amd_mono.glm53_flash.gfx950.layout import (
        ROW_BT,
        ROW_POOL_SLOT,
        ROW_POS,
        ROW_SLOT,
        ROW_TAIL_SLOT,
        ROW_WORDS,
    )
    from sgl_amd_mono.glm53_flash.gfx950.runner import FlashLayerOp, FlashRuntime

    kind, L, W = args.kind, args.verify, args.rows
    assert kind in ("kda", "dsa") and 2 <= L <= W
    R = W // L
    # the decode launches at the same width (a width's tiles set its sums' order):
    # the requests' rows, then pads
    DW = W
    kda = kind == "kda"

    def runtime(rows, verify):
        key = FlashBuild(
            rows=rows,
            tp=tp,
            launches_per_step=1,
            inter=GLOBAL_INTER // tp,
            kda=kda,
            dsa=not kda,
            max_pools=0 if kda else args.max_pools,
            verify=verify,
            grouped=rows >= GROUPED_FROM,
            stop=False,
        )
        rt = FlashRuntime(key, rank, group, device)
        rt.set_deadline(args.deadline)
        return rt

    vrt, drt = runtime(W, L), runtime(DW, 1)
    ffn = ffn_weights(rank, tp, device)
    attn = kda_weights(rank, tp, device) if kda else dsa_weights(rank, tp, device)
    a = {"kda": attn} if kda else {"dsa": attn}
    vop, dop = FlashLayerOp(vrt, ffn, 0, **a), FlashLayerOp(drt, ffn, 0, **a)
    x, res, post, comb = layer_inputs(W, device)
    ins = (x, res, post, comb)

    def rows_of(t, j):
        """Decode step j's input rows: request r's row r * L + j, then pads."""
        out = torch.zeros((DW,) + tuple(t.shape[1:]), dtype=t.dtype, device=device)
        out[:R] = t[[r * L + j for r in range(R)]]
        return out

    def run(op, rt, ins, **kw):
        out = op.launch(*ins, **kw)
        rt.advance_step()
        torch.cuda.synchronize()
        words = rt.epoch[guard.ERR_DEADLINE : guard.ERR_FAULT + 1].tolist()
        if any(words):
            print(f"rank {rank}: error words {words}", flush=True)
            return None
        return tuple(t.clone() for t in out)

    ok = True
    if kda:
        P, H, D = KDA_PROJ // tp, KDA_HEADS // tp, KDA_DIM
        state, _ = kda_state(R, tp, device)
        inter_rec = torch.full(
            (R + 1, L, H, D, D), float("nan"), dtype=torch.float32, device=device
        )
        inter_conv = torch.zeros(
            R + 1, L, 3, 3 * P, dtype=torch.bfloat16, device=device
        )
        vstate = replace(
            state,
            conv_state=state.conv_state.clone(),
            rec_state=state.rec_state.clone(),
            inter_rec=inter_rec,
            inter_conv=inter_conv,
        )
        vidx = torch.zeros(W, dtype=torch.int32, device=device)
        vidx[: R * L] = torch.arange(1, R + 1, device=device).repeat_interleave(L)
        vout = run(vop, vrt, ins, state=vstate, state_idx=vidx)
        dstate = replace(
            state,
            conv_state=state.conv_state.clone(),
            rec_state=state.rec_state.clone(),
        )
        didx = torch.zeros(DW, dtype=torch.int32, device=device)
        didx[:R] = torch.arange(1, R + 1, device=device)
        if vout is None:
            return False
        for name in ("conv_state", "rec_state"):
            same = torch.equal(getattr(vstate, name), getattr(state, name))
            ok &= same
            print(f"rank {rank}: verify keeps the committed {name}: {same}")
        for j in range(L):
            dins = tuple(rows_of(t, j) for t in ins)
            dout = run(dop, drt, dins, state=dstate, state_idx=didx)
            if dout is None:
                return False
            for n, (v, d) in enumerate(zip(vout, dout)):
                diff = _max_diff(v[[r * L + j for r in range(R)]], d[:R])
                ok &= diff == 0.0
                if diff:
                    print(f"rank {rank}: token {j} output {n} differs by {diff}")
            # the request's scratch row is its batch position r (slot r + 1)
            dr = _max_diff(inter_rec[:R, j], dstate.rec_state[1 : R + 1])
            dc = _max_diff(inter_conv[:R, j], dstate.conv_state[1 : R + 1])
            ok &= dr == 0.0 and dc == 0.0
            print(f"rank {rank}: token {j} scratch state diff {dr}, conv {dc}")
        print(f"rank {rank}: verify vs {L} decode launches ok {ok}", flush=True)
        return ok

    lens = [int(v) for v in args.lens.split(",")] if args.lens else list(LENS)
    lens = (lens * R)[:R]
    caches, _, pt = dsa_caches(R, device, lens)
    ring = 4 + L
    g = _gen(11, device)
    tails = [
        torch.randn(R + 1, ring, INDEX_DIM, generator=g, device=device).bfloat16()
        for _ in range(2)
    ]
    caches = replace(caches, tail_k=tails[0], tail_g=tails[1])

    def table_row(r, pos):
        slot = kv_slot(pt, r, pos)
        return [pos, slot, slot // KPOOL, (r + 1) * KPOOL]

    def clone(c):
        return replace(
            c,
            kv_cache=c.kv_cache.clone(),
            idx_cache=c.idx_cache.clone(),
            tail_k=c.tail_k.clone(),
            tail_g=c.tail_g.clone(),
        )

    vc, dc = clone(caches), clone(caches)
    vt = torch.full((W, ROW_WORDS), -1, dtype=torch.int32, device=device)
    vt[:, ROW_BT] = 0
    for r, n in enumerate(lens):
        for j in range(L):
            s = r * L + j
            vt[s, [ROW_POS, ROW_SLOT, ROW_POOL_SLOT, ROW_TAIL_SLOT]] = torch.tensor(
                table_row(r, n - 1 + j), dtype=torch.int32, device=device
            )
            vt[s, ROW_BT] = s
    # SGLang's verify page table: a request's row for each of its rows
    vpt = pt.repeat_interleave(L, 0).contiguous()
    vout = run(vop, vrt, ins, caches=vc, rows=vt, page_table=vpt)
    if vout is None:
        return False
    for j in range(L):
        dt = torch.full((DW, ROW_WORDS), -1, dtype=torch.int32, device=device)
        dt[:, ROW_BT] = 0
        for r, n in enumerate(lens):
            dt[r, [ROW_POS, ROW_SLOT, ROW_POOL_SLOT, ROW_TAIL_SLOT]] = torch.tensor(
                table_row(r, n - 1 + j), dtype=torch.int32, device=device
            )
            dt[r, ROW_BT] = r
        dout = run(
            dop,
            drt,
            tuple(rows_of(t, j) for t in ins),
            caches=dc,
            rows=dt,
            page_table=pt,
        )
        if dout is None:
            return False
        for n, (v, d) in enumerate(zip(vout, dout)):
            diff = _max_diff(v[[r * L + j for r in range(R)]], d[:R])
            ok &= diff == 0.0
            if diff:
                print(f"rank {rank}: token {j} output {n} differs by {diff}")
    for name in ("kv_cache", "idx_cache", "tail_k", "tail_g"):
        same = torch.equal(getattr(vc, name), getattr(dc, name))
        ok &= same
        print(f"rank {rank}: {name} equals the decode launches' {same}")
    print(f"rank {rank}: verify vs {L} decode launches ok {ok}", flush=True)
    return ok


def _changed_rows(before: torch.Tensor, after: torch.Tensor) -> set:
    d = (before != after).reshape(before.shape[0], -1).any(1)
    return set(d.nonzero().flatten().tolist())


def writes_only_its_rows(extra: dict, before: dict) -> bool:
    """After one launch: the state or cache rows that changed are exactly the valid
    rows' (pad rows write nothing)."""
    from msgspec.structs import asdict

    from sgl_amd_mono.glm53_flash.gfx950.layout import (
        ROW_POOL_SLOT,
        ROW_POS,
        ROW_SLOT,
        ROW_TAIL_SLOT,
    )

    if "state" in extra:
        idx = extra["state_idx"].tolist()
        want = {
            "conv_state": {i for i in idx if i > 0},
            "rec_state": {i for i in idx if i > 0},
        }
        now = asdict(extra["state"])
    else:
        rows = [r for r in extra["rows"].tolist() if r[ROW_POS] >= 0]
        want = {
            "kv_cache": {r[ROW_SLOT] for r in rows},
            "idx_cache": {r[ROW_POOL_SLOT] // 64 for r in rows if r[ROW_POS] % 4 == 3},
            "tail_k": {r[ROW_TAIL_SLOT] // 4 for r in rows},
            "tail_g": {r[ROW_TAIL_SLOT] // 4 for r in rows},
        }
        now = asdict(extra["caches"])
    ok = True
    for name, rows_want in want.items():
        got = _changed_rows(before[name], now[name])
        if got != rows_want:
            print(
                f"{name}: changed rows {sorted(got)}, the valid rows' {sorted(rows_want)}"
            )
            ok = False
    return ok


def main() -> int:
    global MAX_LEN
    from sgl_amd_mono.common import guard

    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=KINDS, default="ffn")
    ap.add_argument("--rows", type=int, default=4)
    ap.add_argument("--deadline", type=float, default=10.0)
    ap.add_argument("--launches", type=int, default=3)
    ap.add_argument(
        "--max-len", type=int, default=MAX_LEN, help="a row's tokens at most"
    )
    ap.add_argument("--max-pools", type=int, default=0, help="default: --max-len / 4")
    ap.add_argument("--lens", type=str, default="", help="comma-separated row lengths")
    ap.add_argument("--compare-shuffled", action="store_true")
    ap.add_argument("--check-routing", action="store_true")
    ap.add_argument("--check-keys", action="store_true")
    ap.add_argument("--check-dense", action="store_true")
    ap.add_argument("--check-embed", action="store_true")
    ap.add_argument("--check-head", action="store_true")
    ap.add_argument("--replays", type=int, default=0, help="graph replays of a step")
    ap.add_argument("--layers", type=int, default=1, help="launches a step (--replays)")
    ap.add_argument("--pad-rows", type=int, default=0, help="trailing pad rows")
    ap.add_argument("--step-rows", type=int, default=0, help="an embed step's rows")
    ap.add_argument(
        "--verify", type=int, default=0, help="an MTP verify launch's rows a request"
    )
    ap.add_argument(
        "--writes", action="store_true", help="check the rows a launch writes"
    )
    args = ap.parse_args()
    MAX_LEN = args.max_len
    args.max_pools = args.max_pools or MAX_LEN // 4
    rank, tp = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    device = torch.device("cuda", torch.cuda.current_device())
    dist.init_process_group("gloo")
    group = dist.new_group(backend="gloo")
    for flag, check in (
        (args.compare_shuffled, compare_shuffled),
        (args.check_routing, check_routing),
        (args.check_keys, check_keys),
        (args.check_dense, check_dense),
        (args.check_embed, check_embed),
        (args.check_head, check_head),
        (args.replays, check_replays),
        (args.verify, check_verify),
    ):
        if flag:
            return 0 if check(args, rank, tp, group, device) else 1
    rt, launches, _, ins = build(args, args.kind, rank, tp, group, device)
    op, extra = launches[0]
    outs, written = [], True
    for n in range(args.launches):
        if args.writes and n == 0 and args.kind in ("kda", "dsa", "dense"):
            from msgspec.structs import asdict

            held = asdict(extra.get("state") or extra["caches"])
            before = {k: t.clone() for k, t in held.items() if t is not None}
        out = run_step(args.kind, launches, ins)
        rt.advance_step()
        torch.cuda.synchronize()
        words = rt.epoch[guard.ERR_DEADLINE : guard.ERR_FAULT + 1].tolist()
        outs.append(tuple(t.clone() for t in out))
        print(f"rank {rank} launch {n}: error words {words}", flush=True)
        if any(words):
            return 1
        if args.writes and n == 0 and args.kind in ("kda", "dsa", "dense"):
            written = writes_only_its_rows(extra, before)
            print(f"rank {rank}: writes only its rows {written}", flush=True)
    xs = [torch.zeros_like(outs[-1][0]) for _ in range(tp)]
    dist.all_gather_object(xs, outs[-1][0].cpu())
    same = all(torch.equal(xs[0], t) for t in xs)
    finite = bool(torch.isfinite(outs[-1][0].float()).all())
    print(f"rank {rank}: ranks agree {same}, finite {finite}", flush=True)
    rt.close()
    return 0 if same and finite and written else 1


if __name__ == "__main__":
    sys.exit(main())
