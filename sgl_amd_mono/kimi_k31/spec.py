# SPDX-License-Identifier: Apache-2.0
"""Kimi-K3.1 on gfx950: the monokernel spec (RFC #43949).

Every step this spec routes runs Kimi-K3's launches: K1 on the KDA layers and
the layer launch (o_proj to the MoE output) on every MoE layer. The DSA
full-attention layers keep SGLang's attention up to ``forward_pre_o_proj``, as
K3's MLA layers do, so the indexer, the sparse MLA and the decode context
parallel (DCP) combine all stay on SGLang's path. Under DCP the attention
hands o_proj the same TP-local 12-head core as without it, which is what the
layer launch starts from.

What differs from K3's spec: the model type, DCP accepted, the decode
attention backends K3.1 serves with, and K1's state check. K1 reads the
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


SPEC = msgspec.structs.replace(
    k3.SPEC,
    name="kimi_k31_hip",
    features=k3.SPEC.features | {Feature.CONTEXT_PARALLEL},
    requires=msgspec.structs.replace(
        k3.SPEC.requires,
        attention_backends=frozenset({"aiter", "triton"}),
        hf_config={"model_type": "kimi_k31"},
    ),
    config_disable_reasons=_config_disable_reasons,
)
