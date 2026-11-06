# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3-Flash on gfx950: the monokernel spec and its forward (RFC
sgl-project/sglang#43949).

Every layer is one launch for the whole decoder layer (``gfx950/kernel.py``); the
dense layers 0 to 2 run the FP8 SwiGLU MLP in place of the MoE. Layer 0 takes the
step's tokens (SGLang's embedding is not run) and writes the step's tables. Between
layers the forward hands on ``_Owed`` (x, residual, post, comb): the kernel owes
each layer's FFN post to the next layer's seam. After the last layer the tail launch
runs the owed post, the stream mean, the final norm, the LM head and the logits'
all-gather; SGLang's final norm is not run, and its logits processor returns the
tail's logits (``_get_logits``) and stores the final norm's rows only when the step
captures hidden states.

A KDA pad row (state index not positive) keeps every state; a sparse-attention pad
row (SGLang's padding slot 0) writes no cache. After each KDA launch the forward runs
SGLang's decode track-state copy for the radix cache.

The kernels read SGLang's weights in place, except the copies logged at startup:
the router packed for the MFMA tiles and its bias in f32, W_UK^T from kv_b_proj, the indexer's
wk | weights_proj in bf16, the o_norm weight in f32, and the shared expert packed
as one AITER expert where SGLang keeps it apart (TP2).
"""

from __future__ import annotations

import importlib.util
import os

import msgspec
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
from sgl_amd_mono.glm53_flash import (
    HF_CONFIG,
    LINEAR_ATTN_CONFIG,
)

# kernel widths; a step of fewer rows runs at the next width, the rest pad rows
# (12: two MTP5 or three MTP3 requests)
WIDTHS = (2, 4, 8, 12, 16)
# above this width the MoE runs expert-major (``FlashBuild.grouped``)
GROUPED_FROM = 9
ROWS = tuple(range(1, max(WIDTHS) + 1))
TP_SIZES = (2, 4, 8)
# the rows a step runs on the kernels at most at this TP size: at TP2 a sparse-
# attention launch of 8 rows does not fit the LDS, so wider steps run the model's
# forward
TP_MAX_ROWS = {2: 4}
# SGLang's DSA cache: 256-token KV pages (an index page holds 64 kpool pools)
PAGE_SIZE = 256


def _width(rows: int) -> int:
    return next(w for w in WIDTHS if w >= rows)


# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "srt/arg_groups/fields/*.py": (
        "page_size",
        "enable_linear_replayssm",
        "enable_linear_replayssm_spec",
        "speculative_algorithm",
        "speculative_eagle_topk",
        "speculative_num_steps",
        "speculative_num_draft_tokens",
        "linear_attn_backend",
        "linear_attn_decode_backend",
        "linear_attn_verify_backend",
        "enable_hisparse",
        "expert_distribution_recorder_mode",
        "watchdog_timeout",
    ),
    "srt/configs/glm5_next.py": (
        "num_hidden_layers",
        "first_k_dense_replace",
        "moe_layer_freq",
        "linear_attn_config",
    ),
    "srt/configs/mamba_utils.py": ("mamba2_state_dtype",),
    "srt/environ.py": (
        "SGLANG_SIMULATE_UNIFORM_EXPERTS",
        "SGLANG_SIMULATE_ROUND_ROBIN_EXPERTS",
        "SGLANG_OPT_USE_JIT_KERNEL_GROUPED_TOPK",
        "SGLANG_ENABLE_PP_SPEC",
    ),
    "srt/configs/model_config.py": ("context_len",),
    "srt/distributed/parallel_state.py": ("cpu_group",),
    "srt/layers/attention/dsa/utils.py": ("aiter_can_use_preshuffle_paged_mqa",),
    "srt/layers/attention/dsa_backend.py": ("forward_metadata", "real_page_table"),
    "srt/layers/attention/hybrid_linear_attn_backend.py": (
        "full_attn_backend",
        "linear_attn_backend",
        "_track_mamba_state_decode",
        "_track_mamba_state_decode(forward_batch)",
        "_track_mamba_state_decode(conv_states)",
        "_track_mamba_state_decode(ssm_states)",
        "_track_mamba_state_decode(cache_indices)",
        "_track_mamba_state_decode(layer_id)",
        "state_layer_cache",
    ),
    "srt/layers/attention/mamba/mamba2_metadata.py": ("mamba_cache_indices",),
    # fused accept: the verify kernel seeds from the accepted step, not the state
    "srt/layers/attention/linear/kda_backend.py": ("accept_lens_pool",),
    "srt/layers/layer_boundary/__init__.py": ("get_attn_tp_context",),
    "srt/layers/layer_boundary/adapters/attention.py": ("maybe_input_scattered",),
    "srt/layers/layer_boundary/residual/batch.py": ("start", "clear"),
    # the logits, replaced on a routed step by the tail's
    "srt/layers/logits_processor.py": (
        "_get_logits",
        "_get_logits(hidden_states)",
        "_get_logits(lm_head)",
        "_get_logits(logits_metadata)",
        "do_tensor_parallel_all_gather",
        "do_tensor_parallel_all_gather_dp_attn",
        "use_attn_tp_group",
        "use_fp32_lm_head",
        "logit_scale",
        "final_logit_softcapping",
    ),
    "srt/mem_cache/memory_pool.py": (
        "conv",
        "temporal",
        "intermediate_ssm",
        "intermediate_conv_window",
        "get_key_buffer",
        "get_index_k_with_scale_buffer",
        "get_compress_tail_buffers",
    ),
    "srt/model_executor/forward_batch_info.py": (
        "out_cache_loc",
        "req_pool_indices",
        "capture_hidden_mode",
    ),
    # ForwardMode and CaptureHiddenMode, in forward_batch_info.py before forward_mode.py
    "srt/model_executor/forward_mode.py|srt/model_executor/forward_batch_info.py": (
        "need_capture",
    ),
    "srt/model_executor/forward_context.py": (
        "get_attn_backend",
        "get_token_to_kv_pool",
    ),
    "srt/models/deepseek_v2.py": ("indexer",),
    "srt/models/glm5_next.py": (
        "model",
        "layers",
        "input_layernorm",
        "do_fuse_qkvbfg",
        "self_attn",
        "mlp",
        "hc_post",
        "norm",
        "is_linear_attn",
        "is_mrope_enabled",
        "logits_processor",
        "lm_head",
    ),
    "kernels/plugin_kernels/amd/spec.py|srt/monokernels/spec.py": (
        "DeviceLimits",
        "MonoSpec",
        "Numerics",
        "NumericsKind",
        "Requirements",
        "Feature",
        "widths",
        "deadline_s",
        "copies",
        "tokens_per_request",
    ),
    # NEXTN resolves to EAGLE
    "srt/arg_groups/speculative_hook.py": ('"EAGLE"', '"NEXTN"'),
    "srt/runtime_context.py": (
        "get_device",
        "get_parallel",
        "get_spec",
        "process_model_config",
        '"tp_rank"',
        '"tp_size"',
        '"tp_group"',
    ),
}

# Each launch against SGLang's normal path on the same input and caches
# (SGLANG_AMD_MONO_CHECK, the serving check test): the streams it writes and the
# state or cache rows; the last layer's are its contracted output. The kernel
# quantizes the expert inputs to MXFP8 and SGLang's AITER MoE its own way, so the
# streams differ by about the MoE's error. The dense MLP runs on bf16 activations
# where SGLang quantizes them to FP8 per 128, so the bound that catches errors in it
# is the one against an f32 MLP on the launch's own FFN input. The dense bounds are
# 3x the largest difference this check measured in serving (TP2, TP4, TP8): on layers
# 0 to 2 the streams are the embedding's, small, so SGLang's FP8-activation MLP sets
# the streams difference. The tail's are 3x the largest over 270 real inputs.
NUMERICS = Numerics(
    NumericsKind.TOLERANCE,
    {
        "kda.streams.cosine": 0.95,
        "kda.conv.cosine": 0.9999,
        "kda.state.cosine": 0.9999,
        "dsa.streams.cosine": 0.95,
        "dsa.kv.cosine": 0.995,
        # the completed index pools' keys, dequantized (FP8 value x pool scale)
        "dsa.index.cosine": 0.999,
        "dsa.tail_k.cosine": 0.9999,
        "dsa.tail_g.cosine": 0.9999,
        "dense.streams.rel": 0.11,
        "dense.conv.rel": 3e-3,
        "dense.state.rel": 3.5e-3,
        "dense.x_vs_fp32.rel": 1.2e-2,
        # the launch's FFN input against the input of SGLang's MLP (serving max
        # 6.2e-3 at TP2 and TP4)
        "dense.mlp_in.rel": 2e-2,
        # the tail's logits (f32 of bf16, as SGLang's LM head and all-gather leave
        # them): in one row of a few the two bf16 roundings part (median 3e-5, max
        # 5.8e-4 over 270 inputs; launch and SGLang equally far from an f32 tail)
        "head.logits.rel": 1.8e-3,
        # the bound that catches errors in the tail: against an f32 tail on the same
        # input (hc_post, the stream mean, RMSNorm and the LM head in f32), each row's
        # distance in excess of SGLang's, relative to SGLang's (median 1.7e-5, max
        # 1.95e-2 over 270 inputs)
        "head.logits_vs_fp32.rel": 0.06,
    },
)


def _text(config):
    return getattr(config, "text_config", config)


def _covered_layers(config) -> list[int]:
    """Every layer: the dense layers below ``first_k_dense_replace``, then the MoE
    layers (one in ``moe_layer_freq``, which HF_CONFIG fixes at 1)."""
    return list(range(_text(config).num_hidden_layers))


def _dense(config, i: int) -> bool:
    return i < _text(config).first_k_dense_replace


def _config_disable_reasons(cfg, hf_config) -> list[str]:
    from sglang.srt.configs.mamba_utils import mamba2_state_dtype
    from sglang.srt.layers.attention.dsa.utils import (
        aiter_can_use_preshuffle_paged_mqa,
    )

    text = _text(hf_config)
    reasons = [
        f"needs GLM-5.3-Flash's {key} = {want!r}, got {getattr(text, key, None)!r}"
        for key, want in HF_CONFIG.items()
        if getattr(text, key, None) != want
    ]
    linear = getattr(text, "linear_attn_config", None) or {}
    reasons += [
        f"needs GLM-5.3-Flash's linear_attn_config {key} = {want!r}"
        for key, want in LINEAR_ATTN_CONFIG.items()
        if linear.get(key) != want
    ]
    dtypes = mamba2_state_dtype(hf_config)
    if dtypes.temporal != torch.float32 or dtypes.conv != torch.bfloat16:
        reasons.append(
            f"needs the KDA state in f32 and its conv state in bf16, got "
            f"{dtypes.temporal} and {dtypes.conv}"
        )
    if cfg.page_size not in (None, PAGE_SIZE):
        reasons.append(
            f"--page-size {cfg.page_size}: the kpool caches need {PAGE_SIZE}"
        )
    # a verify request's rows are one draft chain, each row the next position
    if cfg.speculative_algorithm is not None and (
        cfg.speculative_eagle_topk != 1
        or cfg.speculative_num_draft_tokens != cfg.speculative_num_steps + 1
    ):
        reasons.append("MTP verifies one draft chain: topk 1, draft tokens = steps + 1")
    # SGLang's FlashInfer KDA verify fuses the accept (KDAAttnBackend's
    # _can_fuse_accept_state): it skips the KDA state commit, which a verify
    # launch reads. Unset, the verify backend follows a FlashInfer decode backend
    decode = cfg.linear_attn_decode_backend or cfg.linear_attn_backend
    verify = cfg.linear_attn_verify_backend or (
        "flashinfer" if decode == "flashinfer" else None
    )
    if cfg.speculative_algorithm is not None and verify == "flashinfer":
        reasons.append(
            "MTP with the FlashInfer KDA verify backend fuses the accept, leaving "
            "the KDA state a verify launch reads uncommitted"
        )
    if not aiter_can_use_preshuffle_paged_mqa():
        reasons.append(
            "needs the index cache in AITER's preshuffled layout, which the kernels "
            "write; without AITER's preshuffle paged MQA, SGLang keeps it row-major"
        )
    quant = (getattr(hf_config, "quantization_config", None) or {}).get("quant_method")
    if quant != "quark":
        reasons.append(
            f"needs a Quark checkpoint (MXFP4 experts, FP8 block-128 attention), got "
            f"{quant!r}"
        )
    if importlib.util.find_spec("flydsl") is None:
        reasons.append("needs FlyDSL")
    return reasons


def _model_disable_reasons(model, hf_config, copies: dict) -> list[str]:
    from sglang.srt.runtime_context import get_parallel

    from sgl_amd_mono.glm53_flash.gfx950.weights import (
        dense_weights,
        embed_weights,
        experts_reason,
        head_weights,
    )

    tp = get_parallel().tp_size
    reasons = []
    if model.is_mrope_enabled:
        reasons.append("multimodal rotary positions: the kernels take one per row")
    lp = model.logits_processor
    if (
        not lp.do_tensor_parallel_all_gather
        or lp.do_tensor_parallel_all_gather_dp_attn
        or lp.use_attn_tp_group
    ):
        reasons.append(
            "the tail all-gathers the logits over the TP group, as the logits "
            "processor does without DP attention or a DP LM head"
        )
    if lp.use_fp32_lm_head:
        reasons.append("the tail runs the LM head in bf16, not f32")
    for name, value in (
        ("logit_scale", lp.logit_scale),
        ("final_logit_softcapping", lp.final_logit_softcapping),
    ):
        if value:
            reasons.append(f"the tail does not apply the logits processor's {name}")
    for name, weights in (("LM head", head_weights), ("embedding", embed_weights)):
        try:
            weights(model, tp)
        except ValueError as e:
            reasons.append(f"{name}: {e}")
    for i in _covered_layers(hf_config):
        layer = model.model.layers[i]
        if _dense(hf_config, i):
            try:
                dense_weights(layer, i, tp, {})
            except ValueError as e:
                reasons.append(f"layer {i}: {e}")
        else:
            why = experts_reason(layer)
            if why is not None:
                reasons.append(f"layer {i}: {why}")
        attn = layer.self_attn
        if layer.is_linear_attn and not attn.do_fuse_qkvbfg:
            reasons.append(
                f"layer {i}: needs the fused q | k | v | beta | f_a | g_a projection"
            )
        if not layer.is_linear_attn and attn.indexer is None:
            reasons.append(f"layer {i}: sparse attention without its indexer")
    return reasons


class _Owed(msgspec.Struct, frozen=True):
    """A covered layer's output with its FFN post still owed: x [S, HIDDEN], the
    streams after its attention [S, HC, HIDDEN], post [S, HC, 1], comb [S, HC, HC]."""

    x: torch.Tensor
    residual: torch.Tensor
    post: torch.Tensor
    comb: torch.Tensor


def normed_rows(lm, owed: _Owed, S: int) -> torch.Tensor:
    """The final norm's rows of the last layer's owed output, as SGLang's forward
    hands them to the logits processor."""
    from sgl_amd_mono.glm53_flash import check

    owed_rows = (owed.x, owed.residual, owed.post, owed.comb)
    written = _written(lm.model.layers[-1], *(t[:S] for t in owed_rows))
    return lm.model.norm(check.contract(written))


def _written(module, x, residual, post, comb) -> torch.Tensor:
    """The streams with the owed FFN post applied, as SGLang's FFN exit writes them."""
    s = x.shape[0]
    return module.hc_post(x, residual.reshape(s, -1), comb, post.reshape(s, -1))


class _Layer:
    """One layer: its module, kind ("kda", "dsa" or "dense": KDA attention and the
    dense MLP) and per-width ops; its state or caches bind at its first launch and
    rebind if they move."""

    def __init__(self, i: int, module, kind: str, attn):
        self.i, self.module, self.kind, self.attn = i, module, kind, attn
        self.kda = kind != "dsa"  # the launch updates a KDA state
        self.ops: dict = {}  # width -> FlashLayerOp
        self.state = None  # FlashKdaState
        self.caches = None  # FlashDsaCaches


class _Built:
    """One rank's monokernels: the runtimes per (width, kind), each layer's ops, the
    tail's ops, and the forward."""

    def __init__(
        self,
        lm,
        layers: dict,
        runtimes: dict,
        heads: dict,
        tp: int,
        device,
        verify: int = 1,
        max_position: int = 0,
    ):
        from sgl_amd_mono.glm53_flash.gfx950.config import (
            HC_MULT,
            HIDDEN,
        )
        from sgl_amd_mono.glm53_flash.gfx950.layout import (
            ROW_BT,
            ROW_WORDS,
        )

        self.lm, self.model = lm, lm.model
        self._layers, self.runtimes, self.heads = layers, runtimes, heads
        self.last = max(layers)
        self.max_rows = TP_MAX_ROWS.get(tp, max(WIDTHS))
        # a step's rows a request: the verify step's draft tokens with MTP, else 1
        self.verify, self.max_position = verify, max_position
        # the launch checks run SGLang's decode layer: decode steps only
        self.check_layers = (
            os.environ.get("SGLANG_AMD_MONO_CHECK") == "1" and verify == 1
        )
        # layer 0's input besides its tokens: every stream a copy of the embedding row
        # (x = the row, post 1, comb 0, residual 0)
        n = max(WIDTHS)
        self.zero_x = torch.zeros(n, HIDDEN, dtype=torch.bfloat16, device=device)
        self.zero_res = torch.zeros(
            n, HC_MULT, HIDDEN, dtype=torch.bfloat16, device=device
        )
        self.one_post = torch.ones(n, HC_MULT, 1, dtype=torch.float32, device=device)
        self.zero_comb = torch.zeros(
            n, HC_MULT, HC_MULT, dtype=torch.float32, device=device
        )
        # per width: the step's sparse-attention row table and KDA state indices,
        # which layer 0's launch writes
        self.rows, self.kda_idx = {}, {}
        for w in {w for w, _ in runtimes}:
            rows = torch.full((w, ROW_WORDS), -1, dtype=torch.int32, device=device)
            rows[:, ROW_BT] = 0
            self.rows[w] = rows
            self.kda_idx[w] = torch.zeros(w, dtype=torch.int32, device=device)
        # a verify step's positions and request rows, made per row: a graph's step
        # table holds their addresses, so they are an eager step's
        self.verify_in = {
            w: torch.zeros(2, w, dtype=torch.int64, device=device) for w, _ in runtimes
        }
        # the logits processor returns the tail's logits on a routed step
        lp = lm.logits_processor
        self._own_logits, self._logits = lp._get_logits, None
        lp._get_logits = self._get_logits

    def _get_logits(self, hidden_states, lm_head, logits_metadata, *args, **kwargs):
        """The logits processor's logits: the tail's on a routed step (f32 [rows,
        vocab], as SGLang's LM head and all-gather leave them), else its own."""
        if self._logits is None:
            return self._own_logits(
                hidden_states, lm_head, logits_metadata, *args, **kwargs
            )
        return self._logits[: hidden_states.shape[0]]

    def forward(
        self,
        plan,
        input_ids,
        positions,
        forward_batch,
        input_embeds=None,
        pp_proxy_tensors=None,
    ):
        """``Glm5NextForConditionalGeneration.forward`` for a routed decode or MTP
        verify step (a verify step's rows are its requests' draft chains)."""
        lm = self.lm
        if (
            plan.key.rows > self.max_rows
            or plan.key.tokens_per_request != self.verify
            or input_embeds is not None
        ):
            return type(lm).forward(
                lm, input_ids, positions, forward_batch, input_embeds, pp_proxy_tensors
            )
        from sglang.srt.layers.layer_boundary import get_attn_tp_context

        with get_attn_tp_context().maybe_input_scattered(forward_batch):
            owed, self._logits = self._backbone(
                plan, input_ids, positions, forward_batch
            )
        S = plan.key.rows
        # the logits processor stores its hidden states only when the step captures
        # them (fixed per decode graph): then the final norm's rows, else any rows
        mode = forward_batch.capture_hidden_mode
        if mode is not None and mode.need_capture():
            hidden = normed_rows(lm, owed, S)
        else:
            hidden = owed.x[:S]
        try:
            return lm.logits_processor(
                input_ids, hidden, lm.lm_head, forward_batch, None
            )
        finally:
            self._logits = None

    def _backbone(self, plan, input_ids, positions, forward_batch):
        """``Glm5NextModel.forward`` with every layer a launch, layer 0 from the
        tokens, then the tail: the last layer's ``_Owed`` and the gathered logits."""
        from sglang.srt.layers.layer_boundary.residual.batch import clear, start

        from sgl_amd_mono.glm53_flash import check

        start(forward_batch)
        S = plan.key.rows
        W = _width(S)
        owed = None
        for i in sorted(self._layers):
            owed = self._layer(
                self._layers[i], plan, input_ids, positions, owed, forward_batch
            )
        logits = self.heads[W].launch(owed.x, owed.residual, owed.post, owed.comb)
        if self.check_layers and not torch.cuda.is_current_stream_capturing():
            check.tail(self.lm, self._own_logits, owed, logits, forward_batch, S)
        for (w, _), rt in self.runtimes.items():
            if w == W:
                rt.advance_step()
        clear(forward_batch)
        return owed, logits

    def _layer(self, L: _Layer, plan, input_ids, positions, owed, forward_batch):
        """Layer 0 from the step's tokens, or the previous launch's ``_Owed``; returns
        the layer's ``_Owed``."""
        from sglang.srt.model_executor.forward_context import get_attn_backend

        from sgl_amd_mono.glm53_flash import check

        S = plan.key.rows
        W = _width(S)
        checking = self.check_layers and not torch.cuda.is_current_stream_capturing()
        backend = get_attn_backend()
        V = self.verify
        # the KDA state indices and request rows are a request's (a verify step:
        # bs of them), the launches' a row's
        slots = backend.linear_attn_backend.forward_metadata.mamba_cache_indices[
            : S // V
        ]
        op = L.ops[W]
        if L.i == 0:
            ins = (
                self.zero_x[:W],
                self.zero_res[:W],
                self.one_post[:W],
                self.zero_comb[:W],
            )
            rpi = forward_batch.req_pool_indices[: S // V]
            pos = positions[:S]
            if V > 1:
                if backend.linear_attn_backend.accept_lens_pool is not None:
                    raise RuntimeError(
                        "GLM-5.3-Flash monokernel: a verify launch reads the committed "
                        "KDA state, not the fused accept's"
                    )
                slots = slots.repeat_interleave(V)
                buf = self.verify_in[W]
                # verify rows past the context are lookahead the scheduler
                # discards; a row's pools end at the context's
                buf[0, :S].copy_(pos.clamp(max=self.max_position))
                buf[1, :S].copy_(rpi.repeat_interleave(V))
                pos, rpi = buf[0, :S], buf[1, :S]
            idx = slots if slots.dtype == torch.int32 else slots.int()
            page_table = backend.full_attn_backend.forward_metadata.real_page_table
            # the DSA backend repeats a verify request's page-table row for each row
            plan.state["dsa_rows"] = (self.rows[W], page_table[:S])
            step = {
                "ids": input_ids[:S],
                "positions": pos,
                "out_cache_loc": forward_batch.out_cache_loc[:S],
                "req_pool_indices": rpi,
                "slots": self.kda_idx[W],
                "rows": self.rows[W],
            }
            streams = None
        else:
            ins = (owed.x, owed.residual, owed.post, owed.comb)
            idx = self.kda_idx[W]
            step = None
            streams = _written(L.module, *(t[:S] for t in ins)) if checking else None
        if checking:
            hidden = self.model.embed_tokens(input_ids[:S]) if L.i == 0 else streams
            ref = check.reference(L, positions, hidden, forward_batch, S, idx[:S])
        if L.kda:
            self._bind_kda(L, backend.linear_attn_backend, V)
            out = op.launch(*ins, state=L.state, state_idx=idx, step=step)
            # a verify step's states are committed after it, with their radix copy
            if V == 1:
                backend.linear_attn_backend._track_mamba_state_decode(
                    forward_batch, L.state.conv_state, L.state.rec_state, slots, L.i
                )
        else:
            self._bind_dsa(L, V)
            rows, page_table = plan.state["dsa_rows"]
            out = op.launch(*ins, caches=L.caches, rows=rows, page_table=page_table)
        if checking:
            written = _written(L.module, *(t[:S] for t in out))
            if L.i == self.last:
                written = check.contract(written)
            check.compare(L, ref, written, out, positions, forward_batch, S, idx[:S])
        return _Owed(*out)

    @staticmethod
    def _bind_kda(L: _Layer, kda, verify: int) -> None:
        """The layer's KDA state, and with MTP (``verify`` rows a request) its
        verify scratch."""
        from sgl_amd_mono.glm53_flash.gfx950.runner import (
            FlashKdaState,
        )

        cache = kda.state_layer_cache(L.i)
        conv, rec = cache.conv[0], cache.temporal
        ir = ic = None
        if verify > 1:
            ir, ic = cache.intermediate_ssm, cache.intermediate_conv_window[0]
        bound = (conv, rec, ir, ic)
        if L.state is not None and all(
            (a is None and b is None)
            or (a is not None and b is not None and a.data_ptr() == b.data_ptr())
            for a, b in zip(
                (
                    L.state.conv_state,
                    L.state.rec_state,
                    L.state.inter_rec,
                    L.state.inter_conv,
                ),
                bound,
            )
        ):
            return
        P3, H, D = L.attn.conv_w.shape[0], L.attn.a_log.shape[0], 128
        if not (
            conv.dtype == torch.bfloat16
            and tuple(conv.shape[1:]) == (3, P3)
            and conv.stride()[1:] == (P3, 1)
            and rec.dtype == torch.float32
            and tuple(rec.shape[1:]) == (H, D, D)
            and rec.stride()[1:] == (D * D, D, 1)
            and conv.shape[0] == rec.shape[0]
        ):
            raise RuntimeError(
                f"GLM-5.3-Flash monokernel: layer {L.i}: KDA state conv "
                f"{conv.dtype} {tuple(conv.shape)} {conv.stride()}, recurrent "
                f"{rec.dtype} {tuple(rec.shape)} {rec.stride()}; needs bf16 "
                f"[slots, 3, {P3}] and f32 [slots, {H}, {D}, {D}]"
            )
        if verify > 1 and not (
            ir is not None
            and ir.dtype == torch.float32
            and tuple(ir.shape[1:]) == (verify, H, D, D)
            and ir.stride()[1:] == (H * D * D, D * D, D, 1)
            and ic.dtype == torch.bfloat16
            and tuple(ic.shape[1:]) == (verify, 3, P3)
            and ic.stride()[1:] == (3 * P3, P3, 1)
        ):
            raise RuntimeError(
                f"GLM-5.3-Flash monokernel: layer {L.i}: KDA verify scratch "
                f"{None if ir is None else (ir.dtype, tuple(ir.shape), ir.stride())}, "
                f"conv window {ic.dtype} {tuple(ic.shape)} {ic.stride()}; needs f32 "
                f"[rows, {verify}, {H}, {D}, {D}] and dense bf16 "
                f"[rows, {verify}, 3, {P3}]"
            )
        L.state = FlashKdaState(
            conv_state=conv, rec_state=rec, inter_rec=ir, inter_conv=ic
        )

    @staticmethod
    def _bind_dsa(L: _Layer, verify: int) -> None:
        from sglang.srt.model_executor.forward_context import get_token_to_kv_pool

        from sgl_amd_mono.glm53_flash.gfx950.runner import (
            INDEX_BLOCK_POOLS,
            FlashDsaCaches,
        )

        pool = get_token_to_kv_pool()
        kv = pool.get_key_buffer(L.i)
        kv = kv.view(kv.shape[0], -1)
        idx = pool.get_index_k_with_scale_buffer(L.i)
        tail_k, tail_g = pool.get_compress_tail_buffers(L.i)
        if L.caches is not None and all(
            a.data_ptr() == b.data_ptr()
            for a, b in zip(
                (
                    L.caches.kv_cache,
                    L.caches.idx_cache,
                    L.caches.tail_k,
                    L.caches.tail_g,
                ),
                (kv, idx, tail_k, tail_g),
            )
        ):
            return
        reasons = []
        if kv.dtype != torch.bfloat16 or kv.shape[1] < 512 or kv.stride(1) != 1:
            reasons.append(
                f"MLA cache {kv.dtype} {tuple(kv.shape)} {kv.stride()}, needs bf16 "
                "[slots, >= 512] with unit stride"
            )
        if (
            idx.dtype != torch.uint8
            or idx.dim() != 2
            or idx.shape[1] != INDEX_BLOCK_POOLS * 132
            or not idx.is_contiguous()
        ):
            reasons.append(
                f"index cache {idx.dtype} {tuple(idx.shape)}, needs contiguous uint8 "
                f"[pages, {INDEX_BLOCK_POOLS * 132}]"
            )
        # SGLang's ring: 4 slots a request, plus a verify step's draft tokens; a
        # verify chain's pool reads the 3 slots before its row, after the chain's
        # earlier rows wrote theirs
        ring = 4 if verify == 1 else verify + 3
        for name, t in (("k", tail_k), ("gate", tail_g)):
            if (
                t.dtype != torch.bfloat16
                or t.dim() != 3
                or t.shape[1] < ring
                or t.shape[2] != 128
                or not t.is_contiguous()
            ):
                reasons.append(
                    f"kpool {name} ring {t.dtype} {tuple(t.shape)}, needs contiguous "
                    f"bf16 [slots, >= {ring}, 128]"
                )
        if reasons:
            raise RuntimeError(
                f"GLM-5.3-Flash monokernel: layer {L.i}: " + "; ".join(reasons)
            )
        L.caches = FlashDsaCaches(
            kv_cache=kv, idx_cache=idx, tail_k=tail_k, tail_g=tail_g
        )


def _deadline(key, deadline_s: float, longest_s: int) -> float:
    """A step's first launch waits up to half the watchdog, at most ``longest_s``."""
    from sglang.srt.runtime_context import get_device

    if not key.tokens:
        return deadline_s
    return min(max(deadline_s, 0.5 * get_device().watchdog_timeout), longest_s)


def _build(model, hf_config, ctx) -> _Built:
    from sglang.srt.runtime_context import get_parallel, get_spec, process_model_config

    from sgl_amd_mono.common import guard
    from sgl_amd_mono.glm53_flash.gfx950.config import (
        GLOBAL_INTER,
        KPOOL,
    )
    from sgl_amd_mono.glm53_flash.gfx950.kernel import (
        FlashBuild,
    )
    from sgl_amd_mono.glm53_flash.gfx950.layout import (
        POOLS_PER_TASK,
    )
    from sgl_amd_mono.glm53_flash.gfx950.runner import (
        FlashHeadOp,
        FlashLayerOp,
        FlashRuntime,
    )
    from sgl_amd_mono.glm53_flash.gfx950.weights import (
        dense_weights,
        dsa_weights,
        embed_weights,
        head_weights,
        kda_weights,
        layer_weights,
    )

    p = get_parallel()
    tp, rank = p.tp_size, p.tp_rank
    inter = GLOBAL_INTER // tp
    device = model.model.layers[0].input_layernorm.weight.device
    copies = ctx.copies
    layers, weights = {}, {}
    for i in _covered_layers(hf_config):
        module = model.model.layers[i]
        if _dense(hf_config, i):
            kind, ffn = "dense", dense_weights(module, i, tp, copies)
        else:
            kind = "kda" if module.is_linear_attn else "dsa"
            ffn = layer_weights(module, i, inter, copies)
        if module.is_linear_attn:
            attn = kda_weights(module, i, tp, copies)
        else:
            attn = dsa_weights(module, i, tp, copies)
        layers[i] = _Layer(i, module, kind, attn)
        weights[i] = (ffn, attn)
    embedding, head = embed_weights(model, tp), head_weights(model, tp)
    dsa_formats = {
        (weights[i][1].fp8, weights[i][1].shuffled)
        for i in layers
        if layers[i].kind == "dsa"
    }
    if len(dsa_formats) > 1:
        raise ValueError(
            f"GLM-5.3-Flash monokernel: sparse-attention projections differ between "
            f"layers (fp8, shuffled): {sorted(dsa_formats)}"
        )
    (qkv8, qb8, o8), (qkv_sh, qb_sh, o_sh) = (
        dsa_formats.pop() if dsa_formats else ((True, True, True), (False,) * 3)
    )
    dense_formats = {
        weights[i][0].shuffled for i in layers if layers[i].kind == "dense"
    }
    if len(dense_formats) > 1:
        raise ValueError(
            "GLM-5.3-Flash monokernel: the dense MLPs' FP8 layouts differ between layers"
        )
    dense_sh = dense_formats.pop() if dense_formats else False
    # layer 0 (from the tokens) has its own runtime; the other dense layers share one
    kinds = {"embed": [0]}
    for kind in ("dense", "kda", "dsa"):
        kinds[kind] = sorted(i for i in layers if layers[i].kind == kind and i != 0)
    if layers[0].kind != "dense":
        raise ValueError("GLM-5.3-Flash monokernel: layer 0 must be a dense layer")
    fault_site, fault_after = guard.fault_injection(rank) or (0, 0)
    # the pools a sparse-attention row scores at most: the served context's, whole
    # score tasks of 256 pools
    context_len = process_model_config().context_len
    pools = -(-context_len // KPOOL)
    max_pools = -(-pools // POOLS_PER_TASK) * POOLS_PER_TASK
    # MTP: the target's verify steps, a draft chain of draft-tokens rows a request
    sp = get_spec()
    verify = (
        int(sp.speculative_num_draft_tokens)
        if sp.speculative_algorithm is not None
        else 1
    )
    runtimes, heads = {}, {}
    cap = TP_MAX_ROWS.get(tp, max(WIDTHS))
    widths = sorted({_width(r) for r in ctx.widths if r <= cap})
    if not widths:
        raise ValueError(
            f"GLM-5.3-Flash monokernel: TP{tp} runs at most {cap} rows on the kernels, "
            f"and this server's decode graphs route {sorted(ctx.widths)}"
        )

    def runtime(key):
        rt = FlashRuntime(key, rank, p.tp_group.cpu_group, device)
        # at most the guard's longest deadline, in whole seconds
        rt.set_deadline(
            _deadline(key, ctx.deadline_s, guard.MAX_DEADLINE_TICKS // guard.CLOCK_HZ)
        )
        return rt

    # each runtime's peer handshake is collective: every rank builds them in one order
    for w in widths:
        for kind, ids in kinds.items():
            if not ids:
                continue
            key = FlashBuild(
                rows=w,
                tp=tp,
                launches_per_step=len(ids),
                inter=inter,
                kda=kind != "dsa",
                dsa=kind == "dsa",
                max_pools=max_pools if kind == "dsa" else 0,
                qkv_fp8=qkv8,
                qb_fp8=qb8,
                o_fp8=o8,
                qkv_shuffled=qkv_sh,
                qb_shuffled=qb_sh,
                o_shuffled=o_sh,
                dense=kind in ("dense", "embed"),
                dense_shuffled=dense_sh if kind in ("dense", "embed") else False,
                tokens=kind == "embed",
                verify=verify,
                grouped=w >= GROUPED_FROM and kind not in ("dense", "embed"),
                fault_site=fault_site,
                fault_after=fault_after,
            )
            runtimes[w, kind] = rt = runtime(key)
            for n, i in enumerate(ids):
                ffn, attn = weights[i]
                layers[i].ops[w] = FlashLayerOp(
                    rt,
                    ffn,
                    n,
                    kda=attn if kind != "dsa" else None,
                    dsa=attn if kind == "dsa" else None,
                    embedding=embedding if kind == "embed" else None,
                )
        key = FlashBuild(
            rows=w,
            tp=tp,
            launches_per_step=1,
            inter=inter,
            head=True,
            fault_site=fault_site,
            fault_after=fault_after,
        )
        runtimes[w, "head"] = rt = runtime(key)
        heads[w] = FlashHeadOp(rt, head)
    # Every rank builds every width. FlyDSL's disk cache hands a rank the kernels
    # another rank built (a lock per kernel), so each rank starts at its own share.
    start = len(widths) * rank // tp
    for w in widths[start:] + widths[:start]:
        for (rw, _), rt in runtimes.items():
            if rw == w:
                rt.precompile()
    return _Built(model, layers, runtimes, heads, tp, device, verify, context_len - 1)


SPEC = MonoSpec(
    name="glm53_flash_hip",
    devices={
        "gfx950": DeviceLimits(widths=ROWS, tp_sizes=TP_SIZES, resident_blocks=BLOCKS)
    },
    # decode steps, or the target's verify steps of MTP (the NextN draft runs
    # SGLang's forward)
    features=frozenset({Feature.SPECULATIVE_DECODING}),
    requires=Requirements(
        dtypes=frozenset({"bfloat16"}),
        attention_backends=frozenset({"dsa"}),
        server_args={
            "kv_cache_dtype": frozenset({"auto", "bfloat16"}),
            "moe_runner_backend": frozenset({"auto", "aiter"}),
            # the launch updates the KDA state in place; ReplaySSM defers it to a ring
            "enable_linear_replayssm": frozenset({False}),
            # MTP: the NextN draft's verify steps (NEXTN resolves to EAGLE); a verify
            # launch writes every draft token's KDA state to the verify scratch,
            # which ReplaySSM's spec mode does not keep
            "speculative_algorithm": frozenset({None, "EAGLE"}),
            "enable_linear_replayssm_spec": frozenset({False}),
            # HiSparse maps the logical KV slots to device rows; the kernels write raw slots
            "enable_hisparse": frozenset({False}),
            # the launches record no expert distribution, and layer 44 would record
            # outside its layer (SGLang's forward sets it per layer)
            "expert_distribution_recorder_mode": frozenset({None}),
        },
        env={
            # benchmark routing overrides: the kernel routes with the model's own router
            "SGLANG_SIMULATE_UNIFORM_EXPERTS": False,
            "SGLANG_SIMULATE_ROUND_ROBIN_EXPERTS": False,
            # the kernel breaks router ties in AITER's grouped top-k order, the path
            # SGLang takes with this off; on, its Triton gate keeps the lower expert id
            "SGLANG_OPT_USE_JIT_KERNEL_GROUPED_TOPK": False,
            # PP spec remaps the verify scratch rows; a verify launch writes request
            # r's to row r
            "SGLANG_ENABLE_PP_SPEC": False,
        },
    ),
    numerics=NUMERICS,
    model_disable_reasons=_model_disable_reasons,
    build=_build,
    api_version=1,  # SGLang's MONOKERNEL_API_VERSION
    config_disable_reasons=_config_disable_reasons,
)
