# SPDX-License-Identifier: Apache-2.0
"""Kimi-K3.1 on gfx950: the monokernel spec and its forward (RFC #43949).

A routed step is a speculative verify step (DSpark or DFLASH) of R >= 2 requests x
L draft tokens (L = 6 or 8) above 8 rows. It runs ``wide.Wide.backbone``: one
persistent launch a KDA layer (the previous layer's MoE tail, the attention seam,
in_proj, the KDA recurrence on the ReplaySSM ring, o_proj, the MLP seam and the MoE
front), SGLang's routed experts between launches, and SGLang's own path for layer 0,
the DSA full-attention layers and each run's exit. Every other step runs SGLang's
standard path.

The DSA layers keep SGLang's attention, so the indexer, the sparse MLA and the
decode context parallel (DCP) combine stay on SGLang's path; a DSA layer's MoE tail
runs in the next run's first launch.
"""

from __future__ import annotations

import importlib.util

import torch

try:
    from sglang.kernels.plugin_kernels.amd.spec import (
        DeviceLimits,
        Feature,
        MonoSpec,
        Numerics,
        NumericsKind,
        Requirements,
    )
except ModuleNotFoundError:  # SGLang before sgl-project/sglang#45389
    from sglang.srt.monokernels.spec import (
        DeviceLimits,
        Feature,
        MonoSpec,
        Numerics,
        NumericsKind,
        Requirements,
    )

from sgl_amd_mono.common.plan import BLOCKS
from sgl_amd_mono.kimi_k31 import CORE, HF_CONFIG, LINEAR_ATTN_CONFIG, MAX_BLOCKS, wide

TP_SIZE = 8
# The decode context parallel sizes the launch was checked at.
DCP_SIZES = (1, 8)

SGLANG_READS = {
    "kernels/plugin_kernels/amd/spec.py|srt/monokernels/spec.py": (
        "MONOKERNEL_API_VERSION",
        "DeviceLimits",
        "Feature",
        "MonoSpec",
        "Numerics",
        "NumericsKind",
        "Requirements",
    ),
    "srt/utils/common.py": ("require_mlp_sync",),
    # the model's own forward, which _Built reproduces for a routed step
    "srt/managers/mm_utils.py": ("general_mm_embed_routine",),
    "srt/managers/schedule_batch.py": ("Modality",),
    "srt/runtime_context.py": ("get_parallel",),
    "srt/models/kimi_k3.py": (
        "KimiK31ForConditionalGeneration",
        "use_latent_moe",
        "routed_expert_norm",
        "routed_expert_up_proj",
        "use_attn_residuals",
        "_sp_moe",
        "all_reduce_fusion",
        "o_proj",
        "layers",
        "language_model",
    ),
}

NUMERICS = Numerics(
    NumericsKind.TOLERANCE,
    {
        # SGLANG_AMD_MONO_K31_WIDE_ENTRY_CHECK on the real model (DFLASH, DCP8, rows
        # 12 .. 288): a run's first launch's prefix after the MoE tail vs SGLang's
        # whole MoE on the same input, median 3.6e-3; and a run's exit stream vs the
        # gathered-prefix tail, median 3.2e-3. bf16 rounding in another order.
        "layer.mid.rel": 0.006,
        "run.stream.rel": 0.006,
    },
)


def _text_config(hf_config):
    """Kimi-K3.1's shapes live on the text config."""
    return getattr(hf_config, "text_config", None) or hf_config


def _backbone(model):
    return getattr(model, "language_model", model).model


def _config_disable_reasons(cfg, hf_config) -> list[str]:
    """What ``SPEC.requires`` cannot state: the shapes on the text config, the
    draft tokens a request, DCP and FlyDSL."""
    t = _text_config(hf_config)
    linear = getattr(t, "linear_attn_config", None) or {}
    reasons = [
        f"text config {k} {getattr(t, k, None)!r} is not {v!r}"
        for k, v in HF_CONFIG.items()
        if getattr(t, k, None) != v
    ]
    reasons += [
        f"linear_attn_config {k} {linear.get(k)!r} is not {v!r}"
        for k, v in LINEAR_ATTN_CONFIG.items()
        if linear.get(k) != v
    ]
    if not getattr(t, "dsa_config", None):
        reasons.append("no dsa_config: not a Kimi-K3.1 checkpoint")
    block = getattr(t, "attn_res_block_size", None)
    if not block:
        reasons.append("needs the attention-residual stream (attn_res_block_size)")
    else:
        last = int(t.num_hidden_layers) - 1
        blocks = -(-last // block) + int(last % block == 0)
        if blocks > MAX_BLOCKS:
            reasons.append(
                f"attn_res_block_size {block} gives layer {last} {blocks} attention-residual blocks, "
                f"the launch is built for {MAX_BLOCKS}"
            )
    if not wide.enabled():
        reasons.append("SGLANG_AMD_MONO_K31_WIDE=0")
    tokens = getattr(cfg, "speculative_num_draft_tokens", None)
    if tokens not in wide.SUPPORTED_L:
        reasons.append(
            f"--speculative-num-draft-tokens {tokens} is not one of {list(wide.SUPPORTED_L)}"
        )
    dcp = getattr(cfg, "dcp_size", 1) or 1
    if dcp not in DCP_SIZES:
        reasons.append(f"--dcp-size {dcp} is not one of {list(DCP_SIZES)}")
    if (getattr(cfg, "attn_cp_size", 1) or 1) > 1 or getattr(
        cfg, "enable_prefill_cp", False
    ):
        # CONTEXT_PARALLEL is accepted for DCP; attention CP reshapes the decode rows
        reasons.append("attention context parallel is on; only --dcp-size is accepted")
    if importlib.util.find_spec("flydsl") is None:
        reasons.append("needs FlyDSL")
    return reasons


def _model_disable_reasons(model, hf_config, copies: dict) -> list[str]:
    """Every per-layer check the config cannot answer: the kernels read each weight
    as a raw address, so a wrong dtype, shape or layout reads as the expected one."""
    from sglang.srt.utils.common import require_mlp_sync

    hidden = HF_CONFIG["hidden_size"]
    layers = _backbone(model).layers
    covered = [i for i, layer in enumerate(layers) if wide._is_covered(layer)]
    reasons = ["mlp-sync padding is on"] if require_mlp_sync() else []
    if not covered:
        reasons.append("no KDA layer with a latent MoE")
    for i in covered:
        layer = layers[i]
        attn, moe = layer.self_attn, layer.mlp
        up = getattr(moe, "routed_expert_up_proj", None)
        o_proj = getattr(attn, "o_proj", None)
        weight = getattr(o_proj, "weight", None)
        checks = (
            (moe.use_latent_moe, "not a latent MoE layer"),
            (moe.routed_expert_norm is not None, "no latent RMSNorm"),
            (
                up is not None
                and up.weight.shape[0] == hidden
                and up.weight.is_contiguous(),
                f"the latent up projection is not a contiguous [{hidden}, latent]",
            ),
            (layer.use_attn_residuals, "no attention-residual stream"),
            # SP-MoE reduce-scatters o_proj's output and the fused all-reduce completes
            # it out of a symmetric buffer; the launch does neither
            (not layer._sp_moe, "SP-MoE is on"),
            (not layer.all_reduce_fusion, "the fused attention all-reduce is on"),
            (
                getattr(attn, "_qkvgbfa_layer", None) is not None,
                "the KDA in_proj is not merged",
            ),
            (
                weight is not None
                and weight.dtype == torch.bfloat16
                and tuple(weight.shape) == (hidden, CORE)
                and weight.is_contiguous(),
                f"o_proj is not a contiguous bf16 [{hidden}, {CORE}]",
            ),
            (
                o_proj is not None and getattr(o_proj, "bias", None) is None,
                "o_proj has a bias",
            ),
        )
        reasons += [f"layer {i}: {why}" for ok, why in checks if not ok]
        if reasons and i > covered[0]:
            break  # one layer's reasons say it for all
    return reasons


class _LanguageModelShim:
    """``general_mm_embed_routine``'s language model for a routed step: the reads go
    to the real model, the call to ``_Built.language_forward``."""

    def __init__(self, built: _Built) -> None:
        self._built, self._lm = built, built.lm

    def __getattr__(self, name):
        return getattr(self._lm, name)

    def __call__(self, input_ids=None, forward_batch=None, input_embeds=None, **kw):
        return self._built.language_forward(
            input_ids, forward_batch, input_embeds, **kw
        )


class _Built:
    """One rank's wide launches and the forward of a routed step."""

    def __init__(self, model, wide_runner) -> None:
        self.model = model
        self.lm = getattr(model, "language_model", model)
        self._wide = wide_runner
        self._shim = _LanguageModelShim(self)

    def forward(self, plan, input_ids, positions, forward_batch, *args, **kwargs):
        """``KimiK3ForConditionalGeneration.forward`` for a routed step: SGLang's
        multimodal routine around the backbone, which it calls through the shim."""
        from sglang.srt.managers.mm_utils import general_mm_embed_routine
        from sglang.srt.managers.schedule_batch import Modality

        self._plan, self._positions = plan, positions
        return general_mm_embed_routine(
            input_ids=input_ids,
            forward_batch=forward_batch,
            language_model=self._shim,
            data_embedding_funcs={Modality.IMAGE: self.model.get_image_feature},
            positions=positions,
            pp_proxy_tensors=kwargs.get("pp_proxy_tensors"),
        )

    def language_forward(self, input_ids, forward_batch, input_embeds, **kw):
        """``KimiK3LinearForCausalLM.forward``: the backbone, then the head."""
        lm = self.lm
        try:
            hidden = self._wide.backbone(self, input_ids, forward_batch, input_embeds)
        except (
            BaseException
        ) as e:  # surface it: the caller's handling must not hide a wide-path failure
            import traceback

            wide._say(
                f"wide backbone raised {type(e).__name__}: {e}\n{traceback.format_exc()[-2000:]}"
            )
            raise
        aux = None
        if lm.capture_aux_hidden_states:
            hidden, aux = hidden
        return lm.logits_processor(input_ids, hidden, lm.lm_head, forward_batch, aux)


def _build(model, hf_config, ctx):
    """The wide launches of every bucket, compiled by running each once (``warmup``)
    before graph capture."""
    from sglang.srt.runtime_context import get_parallel

    from sgl_amd_mono.kimi_k31 import wide_check

    def mem():
        return (
            torch.cuda.mem_get_info()[0],
            torch.cuda.memory_allocated(),
            torch.cuda.memory_reserved(),
        )

    def used(a, b):
        return "device {:.2f} / torch allocated {:.2f} / reserved {:.2f} GiB".format(
            *((x - y) / 2**30 for x, y in ((a[0], b[0]), (b[1], a[1]), (b[2], a[2])))
        )

    # an eager check of the layer launch on this model (off unless asked)
    wide_check.install(model)
    p = get_parallel()
    device = _backbone(model).layers[0].input_layernorm.weight.device
    m0 = mem()
    w = wide.Wide(model, p.tp_rank, p.tp_group.cpu_group, device, L=wide.tokens())
    m1 = mem()
    w.warmup()
    m2 = mem()
    torch.cuda.empty_cache()
    wide._say(
        f"memory: wide layers {used(m0, m1)}; warmup {used(m1, m2)}; after empty_cache {used(m0, mem())}"
    )
    return _Built(model, w)


SPEC = MonoSpec(
    name="kimi_k31_hip",
    devices={
        "gfx950": DeviceLimits(
            widths=wide.wide_widths(), tp_sizes=(TP_SIZE,), resident_blocks=BLOCKS
        )
    },
    # DCP (CONTEXT_PARALLEL) shards the DSA layers' KV, which run on SGLang's path
    features=frozenset({Feature.SPECULATIVE_DECODING, Feature.CONTEXT_PARALLEL}),
    requires=Requirements(
        dtypes=frozenset({"bfloat16"}),
        attention_backends=frozenset({"dsa"}),
        server_args={
            # the routed experts run on SGLang between launches, TP sharded
            "moe_a2a_backend": frozenset({None, "none"}),
            # DSpark's and DFLASH's target steps are uniform verify steps of L rows a request
            "speculative_algorithm": frozenset({"DSPARK", "DFLASH"}),
            # the KDA stage reads and writes the ReplaySSM ring
            "enable_linear_replayssm_spec": frozenset({True}),
        },
        hf_config={"model_type": "kimi_k31"},
        env={"SGLANG_USE_AITER": True},
    ),
    numerics=NUMERICS,
    model_disable_reasons=_model_disable_reasons,
    build=_build,
    api_version=1,  # sglang.srt.monokernels.spec.MONOKERNEL_API_VERSION
    config_disable_reasons=_config_disable_reasons,
)
