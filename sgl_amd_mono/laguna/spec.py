# SPDX-License-Identifier: Apache-2.0
"""Laguna on gfx950: the monokernel spec and its forward (RFC #43949).

A routed decode step runs ``_Built.forward``, which takes and returns what
``LagunaForCausalLM.forward`` does: the embedding, then per layer SGLang's input
RMSNorm and attention (o_proj leaves a TP partial) followed by the FFN launch,
then the logits processor. The FFN launch (``gfx950/layer.py``, one a layer) runs
from the o_proj partial to the next layer's input norm (or the final norm):
attention all-reduce, residual, post-attention norm, router, routed + shared
experts, MoE all-reduce, residual, norm. ``SGLANG_AMD_MONO_LAGUNA_REFERENCE=1``
runs ``_ffn_reference`` (SGLang's ops in the normal path's order) instead.

``SGLANG_AMD_MONO_CHECK=1`` (eager steps) runs ``_ffn_reference`` and SGLang's
norm on a copy of every launch's inputs and logs the largest difference of the
next layer's input and residual over the layers. Rerunning
the attention instead is not a usable check: SGLang's own path differs from itself
by ~3e-2 at the attention output when a decode step is run twice.
"""

from __future__ import annotations

import logging
import os

import torch
from sglang.srt.monokernels.spec import (
    DeviceLimits,
    MonoSpec,
    Numerics,
    NumericsKind,
    Requirements,
)

from sgl_amd_mono.common.plan import BLOCKS
from sgl_amd_mono.laguna import SERVED_ROWS, TP

logger = logging.getLogger(__name__)

# The SGLang names this glue reads, by file under the sglang package
SGLANG_READS = {
    "srt/monokernels/spec.py": (
        "MONOKERNEL_API_VERSION",
        "DeviceLimits",
        "MonoSpec",
        "Numerics",
        "NumericsKind",
        "Requirements",
    ),
    "srt/models/laguna.py": (
        "embed_tokens",
        "layers",
        "input_layernorm",
        "post_attention_layernorm",
        "self_attn",
        "mlp",
        "norm",
        "layers_to_capture",
        "routed_scaling_factor",
        "shared_expert",
        "gate",
        "e_score_correction_bias",
        "is_layer_sparse",
        "logits_processor",
        "lm_head",
    ),
}

# P0 reference: SGLang's ops with an unfused AR + RMSNorm in place of the normal
# path's fused one, so the bound is loose until the kernels' check runs set it
NUMERICS = Numerics(
    NumericsKind.TOLERANCE,
    {"layer.hidden.rel": 0.05, "layer.hidden.cosine": 0.99},
)


def _config_disable_reasons(cfg, hf_config) -> list[str]:
    reasons = []
    if getattr(hf_config, "num_hidden_layers", None) is None:
        reasons.append("no num_hidden_layers in the HF config")
    # the FFN launch's fixed shape (gfx950/layer.py) and routing
    want = {
        "hidden_size": 16384,
        "num_experts": 512,
        "num_experts_per_tok": 16,
        "moe_intermediate_size": 4096,
        "shared_expert_intermediate_size": 4096,
        "hidden_act": "silu",
    }
    # the proxy checkpoint's config.json has null here: SGLang's Laguna router is sigmoid then
    if getattr(hf_config, "moe_router_score_func", None) not in (None, "sigmoid"):
        reasons.append(f"router score {hf_config.moe_router_score_func} (the FFN launch serves sigmoid)")
    for k, v in want.items():
        got = getattr(hf_config, k, None)
        if got != v:
            reasons.append(f"{k} {got} (the FFN launch serves {v})")
    if getattr(hf_config, "num_hidden_layers", 0) > 64:
        reasons.append("more than 64 layers (the launch's mailbox tag slots)")
    return reasons


def _model_disable_reasons(model, hf_config, copies: dict) -> list[str]:
    from sglang.srt.layers.dp_attention import is_dp_attention_enabled
    from sglang.srt.runtime_context import get_parallel

    m, p = model.model, get_parallel()
    reasons = []
    if (p.tp_size, p.attn_tp_size, p.moe_tp_size, p.moe_ep_size) != (TP, TP, TP, 1):
        reasons.append(
            f"TP {p.tp_size} / attention TP {p.attn_tp_size} / MoE TP {p.moe_tp_size} / "
            f"EP {p.moe_ep_size}: the launch shards experts over TP {TP}, EP 1"
        )
    if is_dp_attention_enabled():
        reasons.append("DP attention")
    if m.start_layer != 0 or m.end_layer != len(m.layers):
        reasons.append(f"layers {m.start_layer}..{m.end_layer} of {len(m.layers)}")
    if m.layers_to_capture:
        reasons.append("auxiliary hidden-state capture (EAGLE3) is not served")
    if getattr(model, "capture_aux_hidden_states", False):
        reasons.append("capture_aux_hidden_states")
    dense = [i for i, layer in enumerate(m.layers) if not layer.is_layer_sparse]
    if dense:
        reasons.append(f"dense MLP layers {dense}; the launch serves MoE layers only")
    for i, layer in enumerate(m.layers):
        moe = layer.mlp
        if getattr(moe, "_shared_expert_tp1", False):
            reasons.append(f"layer {i}: replicated (TP1) shared expert")
            break
        if moe.router_logit_softcapping > 0.0:
            reasons.append(f"layer {i}: router logit softcapping")
            break
        if moe.routed_scaling_factor != 1.0:
            reasons.append(f"layer {i}: routed scaling {moe.routed_scaling_factor}")
            break
        if getattr(moe.experts, "apply_router_weight_on_input", False):
            reasons.append(f"layer {i}: router weight applied on the input")
            break
        if getattr(moe.shared_expert.gate_up_proj, "dequantized_bf16", False):
            reasons.append(f"layer {i}: shared expert dequantized to bf16")
            break
        eps = {layer.input_layernorm.variance_epsilon, layer.post_attention_layernorm.variance_epsilon}
        if eps != {m.layers[0].post_attention_layernorm.variance_epsilon}:
            reasons.append(f"layer {i}: RMSNorm eps {eps} differ")
            break
    return reasons


class _Built:
    """One rank's monokernels and the forward that runs them."""

    def __init__(self, lm, rt=None):
        self.lm, self.model, self.rt = lm, lm.model, rt
        mode = os.environ.get("SGLANG_AMD_MONO_CHECK", "0")
        self.check = mode in ("1", "2")
        self.check_parts = mode == "2"  # also the launch's hand-offs vs SGLang's ops on them
        self._checked = 0

    def _check_parts(self, i, layer, partial, residual, forward_batch, x=None, res=None, norm=None):
        _check_parts_impl(self.rt, i, layer, partial, residual, x, res, norm)

    def forward(self, plan, input_ids, positions, forward_batch, *args, **kwargs):
        lm = self.lm
        hidden = self._backbone(input_ids, positions, forward_batch)
        return lm.logits_processor(input_ids, hidden, lm.lm_head, forward_batch, None)

    def _backbone(self, input_ids, positions, forward_batch):
        m = self.model
        layers = m.layers
        check = self.check and not torch.cuda.is_current_stream_capturing()
        hidden = m.embed_tokens(input_ids)
        residual = hidden
        x = layers[0].input_layernorm(hidden)
        rels = []
        for i, layer in enumerate(layers):
            partial = layer.self_attn(
                positions=positions, hidden_states=x, forward_batch=forward_batch
            )
            norm = layers[i + 1].input_layernorm if i + 1 < len(layers) else m.norm
            if check:  # the reference sees the launch's exact inputs
                ref_out, ref_res = _ffn_reference(
                    layer, partial.clone(), residual.clone(), forward_batch
                )
                ref_x, ref_res = norm(ref_out, ref_res)
            if self.rt is None:
                out, residual = _ffn_reference(layer, partial, residual, forward_batch)
                x, residual = norm(out, residual)
            else:
                if self.check_parts and check:
                    pin, rin = partial.clone(), residual.clone()
                x, residual = self.rt.launch(i, partial.contiguous(), residual)
                if self.check_parts and check and self._checked < 2:
                    self._check_parts(i, layer, pin, rin, forward_batch, x, residual, norm)
            if check:
                rels.append((_rel(x, ref_x), _rel(residual, ref_res)))
        if self.rt is not None:
            self.rt.advance_step()
        if check and self._checked < 3:
            logger.info("SGLANG_AMD_MONO_CHECK laguna per layer next input rel: %s",
                        " ".join(f"{r[0]:.1e}" for r in rels))
        if check and self._checked < 50:
            self._checked += 1
            logger.info(
                "SGLANG_AMD_MONO_CHECK laguna rows=%d next input / residual rel_l2 max over layers %.3e / %.3e%s",
                x.shape[0], max(r[0] for r in rels), max(r[1] for r in rels),
                "" if self.rt is None else f" guard errors {self.rt.errors()}",
            )
        return x

def _check_parts_impl(rt, i, layer, partial, residual, x_out=None, res_out=None, norm=None):
    """Layer i's hand-offs in the launch's scratch vs SGLang's ops on the same inputs."""
    from sglang.srt.distributed import tensor_model_parallel_all_reduce

    from sgl_amd_mono.laguna.gfx950 import layer as L

    torch.cuda.synchronize()
    w, H = partial.shape[0], L.HIDDEN
    sb = rt.scratch

    def view(off, n, dt):
        return sb[off:off + n * torch.empty(0, dtype=dt).element_size()].view(dt)

    h = view(L.H, w * H, torch.bfloat16).view(w, H).clone()
    ids = view(L.IDS, w * 16, torch.int32).view(w, 16).clone()
    wts = view(L.WTS, w * 16, torch.float32).view(w, 16).clone()
    moe = view(L.MOE, w * H, torch.bfloat16).view(w, H).clone()
    sh = view(L.SH, w * H, torch.bfloat16).view(w, H).clone()
    res_mid = view(L.RES_MID, w * H, torch.bfloat16).view(w, H).clone()
    if x_out is not None:
        # S5 alone: SGLang's MoE all-reduce + add + norm on our own partials and residual
        part = (moe.float() + sh.float()).to(torch.bfloat16)
        s5_ar = tensor_model_parallel_all_reduce(part.clone())
        s5_x, s5_res = norm(s5_ar.clone(), res_mid.clone())
        logger.info(
            "SGLANG_AMD_MONO_CHECK parts layer %d S5 alone: x rel %.2e res rel %.2e | ar norm %.3e res_mid norm %.3e "
            "| S1 alone: res_mid vs SGLang add rel %.2e",
            i, _rel(x_out, s5_x), _rel(res_out, s5_res), s5_ar.float().norm().item(), res_mid.float().norm().item(),
            _rel(res_mid, layer.post_attention_layernorm(tensor_model_parallel_all_reduce(partial.clone()), residual.clone())[1]),
        )
    attn = tensor_model_parallel_all_reduce(partial)
    h_ref, _ = layer.post_attention_layernorm(attn, residual)
    mlp = layer.mlp
    logits = mlp.gate(h)
    tk = mlp.topk(h, logits)
    rid, rwt = tk.topk_ids.to(torch.int32), tk.topk_weights.float()
    same_set = (ids.sort(1).values == rid.sort(1).values).all(1)
    shared_ref = mlp.shared_expert(h)
    shared_ref2 = mlp.shared_expert(h)
    from aiter.ops.triton.gemm_afp4wfp4 import gemm_afp4wfp4
    from aiter.ops.triton.quant import dynamic_mxfp4_quant

    gu_l, dn_l = mlp.shared_expert.gate_up_proj, mlp.shared_expert.down_proj
    xq, xs = dynamic_mxfp4_quant(h)
    gu = gemm_afp4wfp4(xq, gu_l.weight.view(torch.uint8), xs, gu_l.weight_scale.view(torch.uint8), torch.bfloat16)
    nI = gu.shape[1] // 2
    mid = (torch.nn.functional.silu(gu[:, :nI].float()) * gu[:, nI:].float()).to(torch.bfloat16)
    mq, ms = dynamic_mxfp4_quant(mid)
    quark = gemm_afp4wfp4(mq, dn_l.weight.view(torch.uint8), ms, dn_l.weight_scale.view(torch.uint8), torch.bfloat16)
    gu_sg = gu_l(h)[0]
    mid_sg = mlp.shared_expert.act_fn(gu)
    logger.info(
        "SGLANG_AMD_MONO_CHECK parts layer %d shared: gate_up SGLang vs ops %.2e | act SGLang vs fp32 silu*up->bf16 %.2e "
        "(%d of %d elements differ; SGLang vs bf16(silu(g)) * u %.2e, vs bf16(bf16(silu(g)) * u) %.2e)",
        i, _rel(gu_sg, gu), _rel(mid_sg, mid), int((mid_sg != mid).sum()), mid.numel(),
        _rel(mid_sg, (torch.nn.functional.silu(gu[:, :nI].float()).to(torch.bfloat16).float() * gu[:, nI:].float()).to(torch.bfloat16)),
        _rel(mid_sg, (torch.nn.functional.silu(gu[:, :nI].float()).to(torch.bfloat16) * gu[:, nI:])),
    )
    routed_ref = mlp.experts(h, tk)
    # our routed output with SGLang's routing only when the sets agree for every row
    logger.info(
        "SGLANG_AMD_MONO_CHECK parts layer %d rows %d: h rel %.2e | top-k sets equal %d/%d, order equal %d, "
        "weight sums ours %.4f ref %.4f | shared rel %.2e (SGLang vs itself %.2e, ours vs quark ops %.2e, "
        "SGLang vs quark ops %.2e) | routed rel %.2e | ids0 %s ref %s",
        i, w, _rel(h, h_ref), int(same_set.sum()), w, int((ids == rid).all(1).sum()),
        wts.sum(1).mean().item(), rwt.sum(1).mean().item(), _rel(sh, shared_ref), _rel(shared_ref2, shared_ref),
        _rel(sh, quark), _rel(shared_ref, quark), _rel(moe, routed_ref),
        ids[0, :6].tolist(), rid[0, :6].tolist(),
    )


def _rel(a, b) -> float:
    return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-30)).item()


def _ffn_reference(layer, partial, residual, forward_batch):
    """P0 FFN launch: attention AR, residual add + post-attention RMSNorm, the
    MoE block (shared + routed, TP partial) and its AR. Returns the MoE output
    (its residual add still pending) and the residual stream after attention."""
    from sglang.srt.distributed import tensor_model_parallel_all_reduce

    attn = tensor_model_parallel_all_reduce(partial)
    h, residual = layer.post_attention_layernorm(attn, residual)
    mlp_partial = layer.mlp(h, forward_batch=forward_batch)
    return tensor_model_parallel_all_reduce(mlp_partial), residual


def _build(model, hf_config, ctx) -> _Built:
    if os.environ.get("SGLANG_AMD_MONO_LAGUNA_REFERENCE", "0") == "1":
        return _Built(model)
    from sglang.srt.runtime_context import get_parallel

    from sgl_amd_mono.laguna.gfx950.runner import LagunaFfnRuntime

    p = get_parallel()
    m = model.model
    device = m.layers[0].input_layernorm.weight.device
    eps = m.layers[0].post_attention_layernorm.variance_epsilon
    widths = [w for w in ctx.widths if w <= SERVED_ROWS]
    rt = LagunaFfnRuntime(model, widths, p.tp_size, p.tp_rank, p.tp_group.cpu_group, device, eps, ctx.deadline_s)
    rt.precompile()
    logger.info("laguna FFN launch: widths %s, %d layers, deadline %.0f s", widths, len(m.layers), ctx.deadline_s)
    return _Built(model, rt)


SPEC = MonoSpec(
    name="laguna_ffn_hip",
    devices={
        "gfx950": DeviceLimits(
            widths=tuple(range(1, SERVED_ROWS + 1)),
            tp_sizes=(TP,),
            resident_blocks=BLOCKS,
        )
    },
    features=frozenset(),
    requires=Requirements(
        dtypes=frozenset({"bfloat16"}),
        attention_backends=frozenset({"aiter"}),
        server_args={
            "moe_runner_backend": frozenset({"auto", "aiter"}),
            # the launches record no expert distribution
            "expert_distribution_recorder_mode": frozenset({None}),
        },
        env={"SGLANG_USE_AITER": True},
    ),
    numerics=NUMERICS,
    model_disable_reasons=_model_disable_reasons,
    build=_build,
    api_version=1,  # sglang.srt.monokernels.spec.MONOKERNEL_API_VERSION
    config_disable_reasons=_config_disable_reasons,
)
