# SPDX-License-Identifier: Apache-2.0
"""Kimi-K3.1 on gfx950: the monokernel spec (RFC #43949).

Every step this spec routes runs Kimi-K3's launches: K1 on the KDA layers and
the layer launch (o_proj to the MoE output) on every MoE layer. The DSA
full-attention layers keep SGLang's attention up to ``forward_pre_o_proj``, as
K3's MLA layers do, so the indexer, the sparse MLA and the decode context
parallel (DCP) combine all stay on SGLang's path. Under DCP the attention
hands o_proj the same TP-local 12-head core as without it, which is what the
layer launch starts from.

What differs from K3's spec: the model type, DCP accepted, DFLASH drafts
accepted, the decode attention backend K3.1 serves with (dsa), and K1's state
check. K1 reads the
per-token fp32 ``intermediate_ssm`` snapshots, which
``--enable-linear-replayssm-spec`` does not allocate; under it every KDA layer
declines K1 and runs SGLang's attention before the layer launch.
"""

from __future__ import annotations


import msgspec

try:
    from sglang.kernels.plugin_kernels.amd.spec import Feature
except ModuleNotFoundError:  # SGLang before sgl-project/sglang#45389
    from sglang.srt.monokernels.spec import Feature

from sgl_amd_mono.kimi_k3 import spec as k3
from sgl_amd_mono.kimi_k31 import wide

SGLANG_READS = {
    "kernels/plugin_kernels/amd/spec.py|srt/monokernels/spec.py": ("Feature",),
    "srt/models/kimi_k3.py": ("KimiK31ForConditionalGeneration",),
}

# The decode context parallel size the launch was checked at.
DCP_SIZES = (1, 8)


def _config_disable_reasons(cfg, hf_config) -> list[str]:
    reasons = k3._config_disable_reasons(cfg, hf_config)
    t = k3._text_config(hf_config)
    if not getattr(t, "dsa_config", None):
        reasons.append("no dsa_config: a Kimi-K3 checkpoint, served by kimi_k3")
    dcp = getattr(cfg, "dcp_size", 1) or 1
    if dcp not in DCP_SIZES:
        reasons.append(f"--dcp-size {dcp} is not one of {list(DCP_SIZES)}")
    if (getattr(cfg, "attn_cp_size", 1) or 1) > 1 or getattr(cfg, "enable_prefill_cp", False):
        # CONTEXT_PARALLEL is accepted for DCP; attention CP reshapes the decode rows
        reasons.append("attention context parallel is on; only --dcp-size is accepted")
    return reasons


class _K31Built(k3._Built):
    """Kimi-K3's built launches for the narrow widths, and the wide-row layer
    launch (``wide.py``) for the R x 8 verify steps above them."""

    def __init__(self, base, wide_runner):
        self.__dict__.update(base.__dict__)
        self._wide = wide_runner
        self._shim = k3._LanguageModelShim(self)

    def _backbone(self, input_ids, forward_batch, input_embeds):
        if self._plan.key.rows > wide.NARROW:
            try:
                return self._wide.backbone(self, input_ids, forward_batch, input_embeds)
            except BaseException as e:  # surface it: the caller's handling must not hide a wide-path failure
                import traceback

                wide._say(f"wide backbone raised {type(e).__name__}: {e}\n{traceback.format_exc()[-2000:]}")
                raise
        return super()._backbone(input_ids, forward_batch, input_embeds)


def _build(model, hf_config, ctx):
    narrow = tuple(w for w in ctx.widths if w <= wide.NARROW)
    built = k3.SPEC.build(model, hf_config, msgspec.structs.replace(ctx, widths=narrow))
    # an eager check of the wide-row layer launch on this model (off unless asked)
    from sgl_amd_mono.kimi_k31 import wide_check

    wide_check.install(model)
    if wide.enabled() and any(w > wide.NARROW for w in ctx.widths):
        from sglang.srt.runtime_context import get_parallel

        p = get_parallel()
        device = k3._backbone(model).layers[0].input_layernorm.weight.device
        w = wide.Wide(model, p.tp_rank, p.tp_group.cpu_group, device)
        w.warmup(buckets=wide.buckets())
        built = _K31Built(built, w)
    return built


def _devices():
    devs = dict(k3.SPEC.devices)
    if wide.enabled():
        lim = devs["gfx950"]
        devs["gfx950"] = msgspec.structs.replace(lim, widths=tuple(lim.widths) + wide.wide_widths())
    return devs


SPEC = msgspec.structs.replace(
    k3.SPEC,
    build=_build,
    devices=_devices(),
    name="kimi_k31_hip",
    features=k3.SPEC.features | {Feature.CONTEXT_PARALLEL},
    requires=msgspec.structs.replace(
        k3.SPEC.requires,
        attention_backends=frozenset({"dsa"}),
        # DFLASH's target steps are DSpark's: uniform verify steps of block_size
        # rows a request, its taps the bank-path captures (set_dflash_layers_to_capture)
        server_args={**k3.SPEC.requires.server_args, "speculative_algorithm": frozenset({None, "DSPARK", "DFLASH"})},
        hf_config={"model_type": "kimi_k31"},
    ),
    config_disable_reasons=_config_disable_reasons,
)
