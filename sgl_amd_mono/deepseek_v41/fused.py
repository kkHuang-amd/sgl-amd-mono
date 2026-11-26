# SPDX-License-Identifier: Apache-2.0
"""A layer's owed mHC post fused into the launch that reads its streams next: the head's
``mhc_combine``, or the next layer's Engram gate; aiter's ``mhc_post`` bits, checked at
startup (``prewarm``). ``SGLANG_AMD_MONO_DSV41_FUSED_POST=0`` keeps the separate launches."""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

ENV = "SGLANG_AMD_MONO_DSV41_FUSED_POST"

# by file under sglang/ (test_contract.py pins them)
SGLANG_READS = {
    "kernels/ops/layernorm/mhc.py": ("mhc_combine",),
    # the gate _post_gate_kernel reproduces
    "kernels/ops/embeddings/engram_gate.py": ("_engram_gate_kernel",),
    "srt/layers/engram.py": (
        "engram_gate",
        "engram_gate(input_ids)",
        "image_token_id",
        "q_weight",
        "k_weight",
        "eps",
        "clamp_value",
    ),
}


def enabled() -> bool:
    from sgl_amd_mono.deepseek_v41.glue import flag

    return flag(ENV)


@triton.jit
def _post_stream(X, R, POST, COMB, m, i, h, live, H, HC: tl.constexpr):
    """Stream ``i`` of ``mhc_post(x, residual, post, comb)`` at row ``m``, columns ``h``."""
    x = tl.load(X + m * H + h, mask=live, other=0.0).to(tl.float32)
    r0 = tl.load(R + m * HC * H + h, mask=live, other=0.0).to(tl.float32)
    v = tl.fma(x, tl.load(POST + m * HC + i), r0 * tl.load(COMB + m * HC * HC + i))
    for j in tl.static_range(1, HC):
        r = tl.load(R + m * HC * H + j * H + h, mask=live, other=0.0).to(tl.float32)
        v = tl.fma(r, tl.load(COMB + m * HC * HC + j * HC + i), v)
    return v.to(tl.bfloat16)


@triton.jit
def _post_combine_kernel(
    X, R, POST, COMB, PRE, OUT, Y, H, HC: tl.constexpr, BLOCK: tl.constexpr
):
    m = tl.program_id(0)
    h = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    live = h < H
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for i in tl.static_range(HC):
        vb = _post_stream(X, R, POST, COMB, m, i, h, live, H, HC)
        tl.store(OUT + m * HC * H + i * H + h, vb, mask=live)
        # mhc_combine of the rounded stream, as its kernel sums it
        acc += tl.load(PRE + m * HC + i).to(tl.float32) * vb.to(tl.float32)
    tl.store(Y + m * H + h, acc.to(tl.bfloat16), mask=live)


@triton.jit
def _post_combine_norm_kernel(
    X, R, POST, COMB, PRE, OUT, Y, NW, N, H, EPS, HC: tl.constexpr, BLOCK: tl.constexpr
):
    # _post_combine_kernel a row a program, then the RMSNorm of its combine (the draft's last norm)
    m = tl.program_id(0)
    h = tl.arange(0, BLOCK)
    live = h < H
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for i in tl.static_range(HC):
        vb = _post_stream(X, R, POST, COMB, m, i, h, live, H, HC)
        tl.store(OUT + m * HC * H + i * H + h, vb, mask=live)
        acc += tl.load(PRE + m * HC + i).to(tl.float32) * vb.to(tl.float32)
    yb = acc.to(tl.bfloat16)
    tl.store(Y + m * H + h, yb, mask=live)
    y = tl.where(live, yb.to(tl.float32), 0.0)
    rstd = tl.rsqrt(tl.sum(y * y, 0) / H + EPS)
    w = tl.load(NW + h, mask=live, other=0.0).to(tl.float32)
    tl.store(N + m * H + h, (y * rstd * w).to(tl.bfloat16), mask=live)


@triton.jit
def _post_gate_kernel(
    XO,
    R,
    POST,
    COMB,
    KV,
    QW,
    KW,
    IDS,
    O,
    image_token_id,
    D: tl.constexpr,
    HC: tl.constexpr,
    EPS: tl.constexpr,
    CLAMP: tl.constexpr,
    B: tl.constexpr,
    HAS_IDS: tl.constexpr,
):
    # SGLang's _engram_gate_kernel on the stream the owed post makes
    row = tl.program_id(0)
    token = row // HC
    hc = row % HC
    col = tl.arange(0, B)
    mask = col < D
    x = _post_stream(XO, R, POST, COMB, token, hc, col, mask, D, HC).to(tl.float32)
    key = tl.load(KV + token * (HC + 1) * D + hc * D + col, mask, 0).to(tl.float32)
    q_weight = tl.load(QW + hc * D + col, mask, 0).to(tl.float32)
    k_weight = tl.load(KW + hc * D + col, mask, 0).to(tl.float32)
    weight = q_weight * k_weight
    rstd = tl.rsqrt(tl.sum(x * x, 0) / D + EPS) * tl.rsqrt(
        tl.sum(key * key, 0) / D + EPS
    )
    dot = tl.sum((x * weight) * key, 0) * rstd * (D**-0.5)
    gate = tl.sigmoid(libdevice.copysign(tl.sqrt(tl.maximum(tl.abs(dot), CLAMP)), dot))
    value = tl.load(KV + token * (HC + 1) * D + HC * D + col, mask, 0).to(tl.float32)
    out = x + gate * value
    if HAS_IDS:
        out = tl.where(tl.load(IDS + token) == image_token_id, x, out)
    tl.store(O + row * D + col, out, mask)


def _fits(x, residual, post, comb) -> bool:
    """x [M, H] and residual [M, HC, H] bf16, post [M, HC] and comb [M, HC, HC] fp32,
    all contiguous."""
    return (
        residual.dim() == 3
        and x.dtype == residual.dtype == torch.bfloat16
        and post.dtype == comb.dtype == torch.float32
        and tuple(x.shape) == (residual.shape[0], residual.shape[2])
        and tuple(post.shape) == residual.shape[:2]
        and tuple(comb.shape) == (*residual.shape[:2], residual.shape[1])
        and all(t.is_contiguous() for t in (x, residual, post, comb))
    )


def _post_combine(x, residual, post, comb, pre):
    m, hc, h = residual.shape
    out = torch.empty_like(residual)
    y = torch.empty((m, h), dtype=residual.dtype, device=residual.device)
    if m:
        block = 1024
        _post_combine_kernel[(m, triton.cdiv(h, block))](
            x, residual, post, comb, pre, out, y, h, HC=hc, BLOCK=block
        )
    return out, y


def post_combine_norm(layer, pending, pre, hc: int, norm):
    """``post_combine`` and ``norm`` (an RMSNorm with ``weight`` / ``variance_epsilon``) of its
    combine in one launch -> (streams, combine, normed), or None (the caller keeps the separate
    launches): the normed rows must stay within bf16 rounding of ``norm(combine)`` on every rank."""
    from sgl_amd_mono.deepseek_v41.glue import agreed

    x, residual, post, comb = pending
    w = getattr(norm, "weight", None)
    eps = getattr(norm, "variance_epsilon", getattr(norm, "eps", None))
    if not (
        enabled()
        and _fits(x, residual, post, comb)
        and pre.dtype == torch.float32
        and pre.is_contiguous()
        and w is not None
        and eps is not None
        and w.dtype == torch.bfloat16
        and w.numel() == residual.shape[2]
    ):
        return None

    def why():
        out, y, n = _post_combine_norm(x, residual, post, comb, pre, w, eps)
        want_out, want_y = _post_combine(x, residual, post, comb, pre)
        if not (_same(out, want_out) and _same(y, want_y)):
            return "combine bits differ"
        want_n = norm(want_y).float()
        if not ((n.float() - want_n).abs() <= want_n.abs() * 2**-7 + 1e-3).all():
            return "normed rows past a bf16 rounding"
        return None

    if not agreed("post_combine_norm", why):
        return None
    return _post_combine_norm(x, residual, post, comb, pre, w, eps)


def _post_combine_norm(x, residual, post, comb, pre, w, eps):
    m, hc, h = residual.shape
    out = torch.empty_like(residual)
    y = torch.empty((m, h), dtype=residual.dtype, device=residual.device)
    n = torch.empty_like(y)
    if m:
        _post_combine_norm_kernel[(m,)](
            x, residual, post, comb, pre, out, y, w, n, h, float(eps), HC=hc,
            BLOCK=triton.next_power_of_2(h),
        )  # fmt: skip
    return out, y, n


def _post_gate(x, residual, post, comb, kv, engram, input_ids):
    t, hc, d = residual.shape
    out = torch.empty_like(residual)
    image = engram.image_token_id
    if t:
        _post_gate_kernel[(t * hc,)](
            x,
            residual,
            post,
            comb,
            kv,
            engram.q_weight,
            engram.k_weight,
            input_ids,
            out,
            image if image is not None else 0,
            d,
            hc,
            engram.eps,
            engram.clamp_value,
            triton.next_power_of_2(d),
            image is not None,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return out


def _same(a: torch.Tensor, b: torch.Tensor) -> bool:
    return torch.equal(a.view(torch.int16), b.view(torch.int16))


def post_combine(layer, pending, pre, hc: int):
    """(the streams the owed post (x, residual, post, comb) of ``layer`` settles, their
    ``mhc_combine`` by ``pre``), in one launch where it agrees."""
    from sglang.kernels.ops.layernorm.mhc import mhc_combine

    from sgl_amd_mono.deepseek_v41.glue import agreed

    x, residual, post, comb = pending

    def separate():
        out = layer.hc_post(x, residual, post, comb)
        return out, mhc_combine(out.flatten(1), pre, hc, out.dtype)

    def why():
        got = _post_combine(x, residual, post, comb, pre)
        return None if all(map(_same, separate(), got)) else "bits differ"

    if (
        enabled()
        and _fits(x, residual, post, comb)
        and pre.dtype == torch.float32
        and pre.is_contiguous()
        and agreed("post_combine", why)
    ):
        return _post_combine(x, residual, post, comb, pre)
    return separate()


def post_gate(engram, layer, pending, kv, input_ids):
    """``engram_gate`` on the streams the owed post (x, residual, post, comb) of
    ``layer`` settles (image-token rows kept), in one launch where it agrees."""
    from sglang.srt.layers.engram import engram_gate

    from sgl_amd_mono.deepseek_v41.glue import agreed

    x, residual, post, comb = pending
    ids = input_ids if engram.image_token_id is not None else None

    def separate(ids=ids):
        settled = layer.hc_post(x, residual, post, comb)
        return engram_gate(
            settled,
            kv,
            engram.q_weight,
            engram.k_weight,
            engram.eps,
            engram.clamp_value,
            ids,
            engram.image_token_id,
        )

    def why():
        if not _same(
            separate(), _post_gate(x, residual, post, comb, kv, engram, input_ids)
        ):
            return "bits differ"
        if ids is not None:  # both kinds of rows, whatever this batch holds
            image = input_ids.clone()
            image[::2] = engram.image_token_id
            got = _post_gate(x, residual, post, comb, kv, engram, image)
            if not _same(separate(image), got):
                return "image rows differ"
        return None

    if (
        enabled()
        and _fits(x, residual, post, comb)
        and kv.dtype == torch.bfloat16
        and kv.is_contiguous()
        and engram.q_weight.is_contiguous()
        and engram.k_weight.is_contiguous()
        and (ids is None or (ids.is_contiguous() and ids.shape == (x.shape[0],)))
        and agreed("post_gate", why)
    ):
        return _post_gate(x, residual, post, comb, kv, engram, input_ids)
    return separate()


def prewarm(model, rows: int = 6) -> None:
    """Both verdicts on random rows at startup: the first step calls are a capture's warmup,
    whose garbage streams (tiny values Triton flushes) say nothing of a real step's."""
    if not enabled():
        return
    m = model.model
    dev = m.layers[m.end_layer - 1].input_layernorm.weight.device
    g = torch.Generator(device=dev).manual_seed(0)
    hc, h = m.hc_mult, m.layers[m.end_layer - 1].input_layernorm.weight.shape[0]

    def rand(*shape, dtype=torch.float32, normal=False):
        f = torch.randn if normal else torch.rand
        return f(shape, generator=g, device=dev).to(dtype)

    bf = torch.bfloat16
    pending = (
        rand(rows, h, dtype=bf, normal=True),
        rand(rows, hc, h, dtype=bf, normal=True),
        rand(rows, hc) * 2,
        rand(rows, hc, hc),
    )
    post_combine(m.layers[m.end_layer - 1], pending, rand(rows, hc), hc)
    for i in range(m.start_layer + 1, m.end_layer):
        engram = m.layers[i].engram
        if engram is not None:
            kv = rand(rows, (hc + 1) * h, dtype=bf, normal=True)
            ids = torch.zeros(rows, dtype=torch.int64, device=dev)
            post_gate(engram, m.layers[i - 1], pending, kv, ids)
            break
