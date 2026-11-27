# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1-Flash's DSpark draft on gfx950: its spec and forward (RFC #43949).

A routed draft step (a block of rows a request, run as a TARGET_VERIFY batch) runs
``_Draft.forward``, which takes and returns what ``DeepseekV4ForCausalLMDSpark.forward``
does. Each stage is the target's K1 and K2 on the draft's route (128 experts, top-3, a
FP4 group an ug task) and window (the 128-token window plus the block's keys); the
last stage settles its post with SGLang's ``hc_post`` and the head's combine reads its
pre, as SGLang's forward does. The stages' seams follow the fused mHC boundary's
arithmetic where SGLang's draft runs the unfused one; the two agree within the
kernels' bounds, so DSpark's acceptance with and without the switch is the gate.
"""

from __future__ import annotations

import re

import torch

try:
    from sglang.kernels.plugin_kernels.amd.spec import (
        DeviceLimits,
        Feature,
        MonoSpec,
        Requirements,
    )
except ModuleNotFoundError:  # SGLang before sgl-project/sglang#45389
    from sglang.srt.monokernels.spec import (
        DeviceLimits,
        Feature,
        MonoSpec,
        Requirements,
    )

from sgl_amd_mono.common.plan import BLOCKS
from sgl_amd_mono.deepseek_v41 import draft_sampler, fused
from sgl_amd_mono.deepseek_v41 import spec as target

EXPERTS, TOPK, UG_GROUPS = 128, 3, 1  # the draft's route; a FP4 group an ug task
WINDOW, SWA_WIDTH = 128, 192  # a row's window index slots: the window + the block
# the checkpoint's draft stages are mtp.{stage}
_CKPT_SHARED = re.compile(
    r"mtp\.(?P<i>\d+)\.ffn\.shared_experts\.(?P<w>w[123])\.(?P<kind>weight|scale)"
)

# by file under sglang/ (``f(p)``: a parameter, ``"x"``: a literal); test_contract.py pins them
SGLANG_READS = {
    "kernels/plugin_kernels/amd/spec.py|srt/monokernels/spec.py": (
        "DeviceLimits",
        "Feature",
        "MonoSpec",
        "Requirements",
    ),
    "kernels/ops/layernorm/mhc.py": ("mhc_combine",),
    "srt/environ.py": ("envs", "SGLANG_DSPARK_EMBED_IN_GRAPH"),
    "srt/layers/logits_processor.py": (
        "LogitsProcessorOutput",
        "next_token_logits",
        "hidden_states",
    ),
    # DeepseekV4ForCausalLMDSpark.forward, which the routed forward stands in for
    "srt/models/deepseek_v4_dspark.py": (
        "forward(input_ids)",
        "forward(positions)",
        "forward(forward_batch)",
        "forward(input_embeds)",
        "stages",
        "forward_embed",
        "hc_mult",
        "hc_pre_from_prev_sublayer",
        "sample_from_anchor",
        '"dspark_n_routed_experts"',
        '"dspark_num_experts_per_tok"',
    ),
    "srt/models/deepseek_v4.py": (
        "wq_a",
        "wkv",
        "fuse_wqa_wkv",
        "q_head_norm",
        "n_local_heads",
        "compress_ratio",
    ),
    "srt/runtime_context.py": ("get_parallel", "get_spec", '"tp_group"'),
    "srt/speculative/dspark_components/dspark_config.py": (
        "dspark_gamma_from_num_draft_tokens",
    ),
}


def _tokens_per_request(model) -> int:
    """A draft block's rows a request: gamma, or gamma + 1 without the anchor."""
    from sglang.srt.runtime_context import get_spec
    from sglang.srt.speculative.dspark_components.dspark_config import (
        dspark_gamma_from_num_draft_tokens,
    )

    gamma = dspark_gamma_from_num_draft_tokens(
        int(get_spec().speculative_num_draft_tokens)
    )
    return gamma if model.sample_from_anchor else gamma + 1


def _model_disable_reasons(model, hf_config, copies: dict) -> list[str]:
    from sglang.srt.environ import envs
    from sglang.srt.runtime_context import get_parallel

    from sgl_amd_mono.deepseek_v41.gfx950.attention.plan import Dims

    reasons, unseen = [], []
    route = (
        getattr(hf_config, "dspark_n_routed_experts", None),
        getattr(hf_config, "dspark_num_experts_per_tok", None),
    )
    if route != (EXPERTS, TOPK):
        reasons.append(
            f"draft route {route}: the kernels take {EXPERTS} experts, top-{TOPK}"
        )
    rows = _tokens_per_request(model)
    if WINDOW + rows > SWA_WIDTH:
        reasons.append(f"a draft block of {rows} rows past {SWA_WIDTH} window slots")
    if not getattr(model, "hc_pre_from_prev_sublayer", False):
        reasons.append("needs V4.1's mHC hand-off between stages")
    if not envs.SGLANG_DSPARK_EMBED_IN_GRAPH.get():
        reasons.append(
            "needs SGLANG_DSPARK_EMBED_IN_GRAPH: the forward embeds the block"
        )
    heads = Dims(get_parallel().tp_size).heads
    for i, stage in enumerate(model.stages):
        a, moe = stage.self_attn, stage.mlp
        if a.compress_ratio != 0 or a.q_head_norm or a.n_local_heads != heads:
            reasons.append(
                f"stage {i}: needs a window-only attention with {heads} local heads "
                "and no q head norm"
            )
        if moe.gate.weight.shape[0] != EXPERTS or not getattr(
            moe.experts.w13_weight, "is_shuffled", False
        ):
            reasons.append(f"stage {i}: needs {EXPERTS} experts in aiter's shuffle")
        reasons += target.attn_layout_reasons(f"stage {i}", a)
        if any(
            ("shared expert fp8", i, p, k) not in copies
            for p in target._SHARED
            for k in ("weight", "scale")
        ):
            unseen.append(i)
    if target.shuffled_layout(s.self_attn for s in model.stages) is None:
        reasons.append(
            "wqkv / wq_b / wo_b are aiter-preshuffled on some stages and row-major "
            "on others"
        )
    if unseen:
        reasons.append(
            "the shared expert's FP8 checkpoint tensors were not seen at load "
            f"(stages {unseen})"
        )
    return reasons


def _load_tensor(copies: dict, model, name: str, tensor: torch.Tensor) -> None:
    """Keep this rank's shard of each stage's FP8 shared expert, by its checkpoint
    name (``mtp.{stage}.ffn.shared_experts.{w1,w2,w3}.{weight,scale}``)."""
    from sglang.srt.runtime_context import get_parallel

    m = _CKPT_SHARED.fullmatch(name)
    if m is None or int(m["i"]) >= len(model.stages):
        return
    i, proj = int(m["i"]), target._CKPT_PROJ[m["w"]]
    p = get_parallel()
    dim = 1 if proj == "down_proj" else 0  # the TP-sharded intermediate dimension
    t = tensor.view(torch.uint8)
    n = t.shape[dim] // p.tp_size
    shard = t.narrow(dim, p.tp_rank * n, n).to(torch.cuda.current_device())
    copies["shared expert fp8", i, proj, m["kind"]] = shard.contiguous()


# SGLANG_AMD_MONO_DSV41_DRAFT_TAIL=1: the last stage's post + combine and the draft's last norm in one launch
TAIL_KEY = "_sgl_amd_mono_draft_tail"


def _draft_tail() -> bool:
    from sgl_amd_mono.deepseek_v41.glue import flag

    return flag("SGLANG_AMD_MONO_DSV41_DRAFT_TAIL", "0")


DRAFT_TAIL = _draft_tail()


class _Draft:
    """One rank's draft monokernels: its runner, each stage's weights, the forward."""

    def __init__(self, lm, runner, stages: list):
        self.lm, self.runner, self.stages = lm, runner, stages

    def forward(
        self, plan, input_ids, positions, forward_batch, input_embeds=None, *_, **__
    ):
        """``DeepseekV4ForCausalLMDSpark.forward`` for a routed draft step."""
        from sglang.kernels.ops.layernorm.mhc import mhc_combine
        from sglang.srt.layers.logits_processor import LogitsProcessorOutput

        lm = self.lm
        x = input_embeds
        if x is None:  # ``forward_embed``, from the target's whole table
            if lm.embed_tokens is None:
                raise ValueError(
                    "DeepseekV4ForCausalLMDSpark requires the target embed_tokens "
                    "(call attach_shared_modules first)."
                )
            x = target.embed(lm.embed_tokens, input_ids).unsqueeze(1)
            x = x.repeat(1, lm.hc_mult, 1)
        # a stage's post folds into the next stage's K1, the last stage's into the head's
        # combine (``fused.post_combine``)
        pending, pre = (None, x, None, None), None
        fuse = fused.enabled()
        for j, L in enumerate(self.stages):
            out = target._launch(
                self.runner, L, plan, positions, forward_batch, pending, pre
            )
            defer = j + 1 < len(self.stages) or fuse
            x, pre, post = target._boundary_out(L.module, defer, *out)
            pending = post if defer else (None, x, None, None)
        if pending[0] is None:
            # mhc_combine widens x on load: no f32 copy first
            x = mhc_combine(x.flatten(1), pre, lm.hc_mult, x.dtype)
        else:
            got = None
            if DRAFT_TAIL:
                got = fused.post_combine_norm(
                    self.stages[-1].module, pending, pre, lm.hc_mult, lm.stages[-1].norm
                )
            if got is None:
                _, x = fused.post_combine(
                    self.stages[-1].module, pending, pre, lm.hc_mult
                )
            else:
                # the draft sampler's base logits read ``normed`` (``draft_sampler._Sampler.step``)
                _, x, normed = got
                lm.__dict__[TAIL_KEY] = (x, normed)
        return LogitsProcessorOutput(next_token_logits=None, hidden_states=x)


def _build(model, hf_config, ctx) -> _Draft:
    from sglang.srt.runtime_context import get_parallel

    from sgl_amd_mono.deepseek_v41.gfx950.runner import DSV41MonoRunner

    first = len(target.ROPE_TABLES)
    stages = [
        target._weights(ctx.copies, target.ROPE_TABLES, stage, i)
        for i, stage in enumerate(model.stages)
    ]
    for n, (_, t) in enumerate(target.ROPE_TABLES[first:], first):
        ctx.copies["rope table", n] = t
    p = get_parallel()
    runner = DSV41MonoRunner(
        p.tp_size,
        p.tp_rank,
        p.tp_group.cpu_group,
        model.stages[0].input_layernorm.weight.device,
        experts=EXPERTS,
        topk=TOPK,
        ug_groups=UG_GROUPS,
        swa_width=SWA_WIDTH,
        shuffled=bool(target.shuffled_layout(s.self_attn for s in model.stages)),
    )
    runner.set_deadline(ctx.deadline_s)
    widths = list(ctx.widths)
    start = len(widths) * p.tp_rank // p.tp_size
    for m in widths[start:] + widths[:start]:
        # K1 with and without an owed post; K2 on the window: no index source, no top-k
        runner.precompile(m, k1={(False, False), (True, False)}, k2={(False, False)})
    draft_sampler.install(model, ctx.deadline_s)
    return _Draft(model, runner, stages)


SPEC = MonoSpec(
    name="deepseek_v41_dspark_draft_hip",
    devices={
        "gfx950": DeviceLimits(
            widths=tuple(range(1, target.MAX_ROWS + 1)),
            tp_sizes=(2, 4),
            resident_blocks=BLOCKS,
        )
    },
    # its target's PD gate (decode on MoRI) holds for the draft as well
    features=frozenset({Feature.SPECULATIVE_DECODING, Feature.PD_DISAGGREGATION}),
    # the launcher checks the target's; SGLang checks the draft runner's at its first routed step
    requires=Requirements(attention_backends=target.SPEC.requires.attention_backends),
    numerics=target.NUMERICS,
    model_disable_reasons=_model_disable_reasons,
    build=_build,
    api_version=1,  # SGLang's MONOKERNEL_API_VERSION
    load_tensor=_load_tensor,
    tokens_per_request=_tokens_per_request,
)
