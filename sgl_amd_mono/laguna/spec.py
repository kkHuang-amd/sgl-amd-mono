# SPDX-License-Identifier: Apache-2.0
"""Laguna on gfx950: the monokernel spec and its forward (RFC #43949).

A routed decode step runs ``_Built.forward``, which takes and returns what
``LagunaForCausalLM.forward`` does: the embedding, then per layer SGLang's input
RMSNorm and attention (o_proj leaves a TP partial) followed by the FFN launch,
then SGLang's final norm and logits processor. P0: the FFN launch is
``_ffn_reference``, SGLang's ops in the normal path's order.

``SGLANG_AMD_MONO_CHECK=1`` (eager steps) runs ``_ffn_reference`` on a copy of
every launch's inputs and logs the largest difference over the layers. Rerunning
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
from sgl_amd_mono.laguna import MAX_ROWS, TP

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
    return reasons


class _Built:
    """One rank's monokernels and the forward that runs them."""

    def __init__(self, lm):
        self.lm, self.model = lm, lm.model
        self.check = os.environ.get("SGLANG_AMD_MONO_CHECK", "0") == "1"
        self._checked = 0

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
            if check:  # the reference sees the launch's exact inputs
                ref_out, ref_res = _ffn_reference(
                    layer, partial.clone(), residual.clone(), forward_batch
                )
            out, residual = self._ffn(layer, partial, residual, forward_batch)
            if check:
                rels.append((_rel(out, ref_out), _rel(residual, ref_res)))
            norm = layers[i + 1].input_layernorm if i + 1 < len(layers) else m.norm
            x, residual = norm(out, residual)
        if check and self._checked < 50:
            self._checked += 1
            logger.info(
                "SGLANG_AMD_MONO_CHECK laguna rows=%d ffn out / residual rel_l2 max over layers %.3e / %.3e",
                x.shape[0], max(r[0] for r in rels), max(r[1] for r in rels),
            )
        return x

    def _ffn(self, layer, partial, residual, forward_batch):
        """The FFN launch (P0: the reference)."""
        return _ffn_reference(layer, partial, residual, forward_batch)


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
    return _Built(model)


SPEC = MonoSpec(
    name="laguna_ffn_hip",
    devices={
        "gfx950": DeviceLimits(
            widths=tuple(range(1, MAX_ROWS + 1)),
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
