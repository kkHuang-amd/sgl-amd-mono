# SPDX-License-Identifier: Apache-2.0
"""Kimi-K3 on gfx950: the monokernel spec and its forward (RFC #43949).

SGLang's runtime switches the model's forward at attach: a routed decode or
DSpark verify step runs ``_Built.forward`` here, which takes and returns what
``KimiK3ForConditionalGeneration.forward`` does.

One launch a layer. It starts at the attention's TP-local core output, which
is o_proj's input, and ends at the layer's MoE output. It runs, in order:

1. o_proj and the attention-TP all-reduce;
2. the MLP attention-residual seam;
3. the router and its top-16;
4. the latent down and the shared gate_up projections;
5. the MXFP4 experts;
6. the shared down and the latent up projections;
7. the MoE all-reduce.

Both reduces go over a peer buffer. Neither leaves the kernel.

The attention itself stays on SGLang's path. That is where this contract
differs from DeepSeek-V4.1's: the model runs the attention up to
``forward_pre_o_proj``, and the launch owns everything after it. The KDA and
the full-attention layers reach that point the same way. At TP8 both stop at a
``CORE``-wide output. So the launch covers every MoE layer, not one attention
kind.

The kernels read SGLang's weights in place. The dense weights are read as bf16
in their checkpoint layout. The experts are read in whichever MXFP4 layout
``Mxfp4MoEMethod`` left behind. There is no prepared-weight copy, so the KV
cache budget pays nothing.

Two further differences from DeepSeek-V4.1 are deliberate. Each is argued in
the module that holds it. The mailbox epoch is bumped by the host, not advanced
on the device (``gfx950/tag.py``). And the build compiles by running each
launch once on real weights, not through FlyDSL's compile-only mode
(``_warmup``).
"""

from __future__ import annotations

import importlib.util
import logging
import os
from typing import Any

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
from sgl_amd_mono.kimi_k3 import (
    CORE,
    HF_CONFIG,
    LINEAR_ATTN_CONFIG,
    MAX_BLOCKS,
    MAX_ROWS,
)

logger = logging.getLogger(__name__)

TP_SIZE = 8
# K2's OPT_* bits (gfx950/layer.py: sentinel all-reduces, preloads, tagged x rows,
# router partials, batched polls), each bit for bit the launch without it; unset
# is layer.K2_OPT, all of them.
K2_OPT = os.environ.get("SGLANG_AMD_MONO_K3_K2_OPT")
# The MLA (DSA) layers' projection front as one launch (gfx950/attention/mla_pre.py):
# AttnRes, fused_qkv_a, the q / kv norms, q_b, g_proj and the indexer's wk,
# weights_proj and wq_b, handed to SGLang's MLA forward and indexer.
MLA_PRE = os.environ.get("SGLANG_AMD_MONO_K3_MLA_PRE", "1") == "1"
# SGLANG_AMD_MONO_K3_K1=0 builds no K1: every KDA layer keeps SGLang's attention.
K1 = os.environ.get("SGLANG_AMD_MONO_K3_K1", "1") == "1"

# The SGLang names this glue reads, by file under the sglang package (``f(p)``:
# a parameter, ``"x"``: a string literal); test/unit/test_contract.py pins them.
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
    # the MXFP4 expert layout the launch reads, as Mxfp4MoEMethod chose it
    "srt/layers/quantization/mxfp4.py": ("_aiter_situ_uses_gu_interleaved_weights",),
    # mlp-sync padding, which the launch has no handling for
    "srt/utils/common.py": (
        "BumpAllocator",
        "get_bool_env_var",
        "require_mlp_sync",
    ),
    # the model's own forward, which _Built reproduces for a routed step
    "srt/managers/mm_utils.py": ("general_mm_embed_routine",),
    "srt/managers/schedule_batch.py": ("Modality",),
    "srt/layers/layer_boundary/__init__.py": (
        "AttentionInputs",
        "get_attn_tp_context",
    ),
    "srt/layers/layer_boundary/residual/batch.py": (
        "final_norm",
        "set_written",
        "start",
        "written_residual",
    ),
    "srt/runtime_context.py": ("get_parallel",),
    # the attention stops here and the launch runs o_proj; both attention
    # classes expose it. The rest are what the forward and the checks read.
    "srt/models/kimi_k3.py": (
        "AuxHiddenStateList",
        "_cdiv",
        "forward_pre_o_proj",
        "use_latent_moe",
        "routed_expert_norm",
        "routed_expert_up_proj",
        "use_attn_residuals",
        "_sp_moe",
        "all_reduce_fusion",
        "experts",
        "o_proj",
        "layers",
        "language_model",
    ),
}


def _text_config(hf_config):
    """Kimi-K3's shapes live on the text config; a text-only checkpoint has
    none and is its own text config."""
    return getattr(hf_config, "text_config", None) or hf_config


def _linear_attn_config(hf_config) -> dict:
    return getattr(_text_config(hf_config), "linear_attn_config", None) or {}


def _backbone(model):
    """The K3 language backbone, under the VLM wrapper or on its own."""
    return getattr(model, "language_model", model).model


def _covered_layers(hf_config) -> list[int]:
    """The layers the launch can run: every MoE layer. Which attention a layer
    runs does not matter -- the launch starts after it."""
    t = _text_config(hf_config)
    dense = int(getattr(t, "first_k_dense_replace", 0) or 0)
    freq = int(getattr(t, "moe_layer_freq", 1) or 1)
    return [i for i in range(dense, int(t.num_hidden_layers)) if i % freq == 0]


def _config_disable_reasons(cfg, hf_config) -> list[str]:
    """What ``SPEC.requires`` cannot state: the shapes, which live on the text
    config and so are out of ``requires.hf_config``'s reach; aiter's own flags;
    and FlyDSL."""
    from sglang.srt.layers.quantization.mxfp4 import (
        _aiter_situ_uses_gu_interleaved_weights,
    )
    from sglang.srt.utils.common import get_bool_env_var

    t = _text_config(hf_config)
    linear = _linear_attn_config(hf_config)
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
    # The bank rows a layer's MLP seam reads grow with the layer index; the
    # launch is built for MAX_BLOCKS of them.
    block = getattr(t, "attn_res_block_size", None)
    if not block:
        reasons.append("needs the attention-residual stream (attn_res_block_size)")
    else:
        last = max(_covered_layers(hf_config), default=0)
        blocks = -(-last // block) + int(last % block == 0)
        if blocks > MAX_BLOCKS:
            reasons.append(
                f"attn_res_block_size {block} gives layer {last} {blocks} "
                f"attention-residual blocks, the launch is built for {MAX_BLOCKS}"
            )
    if not _aiter_situ_uses_gu_interleaved_weights():
        reasons.append(
            "needs the MXFP4 experts in aiter's gate / up interleaved layout; "
            "set AITER_SITUV2_A8W4=1 or leave AITER_SITUV2_A4W4 unset"
        )
    if not get_bool_env_var("SGLANG_AITER_K3_OPT"):
        reasons.append("needs SGLANG_AITER_K3_OPT=1")
    if importlib.util.find_spec("flydsl") is None:
        reasons.append("needs FlyDSL")
    return reasons


def _model_disable_reasons(model, hf_config, copies: dict) -> list[str]:
    """Every per-layer check the config cannot answer. The kernels read each
    weight as a raw address, so a wrong dtype, shape or layout is read as if it
    were the expected one."""
    from sglang.srt.utils.common import require_mlp_sync

    hidden = HF_CONFIG["hidden_size"]
    covered = _covered_layers(hf_config)
    layers = _backbone(model).layers
    reasons = []
    if require_mlp_sync():
        # mlp-sync pads extend batches to a multiple of attn_tp_size; the
        # launch has no padded-row handling.
        reasons.append("mlp-sync padding is on")
    for i in covered:
        layer = layers[i]
        attn, moe = layer.self_attn, layer.mlp
        experts = getattr(moe, "experts", None)
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
            (
                experts is not None
                and getattr(experts.w13_weight, "is_shuffled", False)
                and getattr(experts.w2_weight, "is_shuffled", False),
                "the experts are not in aiter's preshuffled MXFP4 layout",
            ),
            (layer.use_attn_residuals, "no attention-residual stream"),
            # SP-MoE reduce-scatters o_proj's output and the fused all-reduce
            # completes it out of a symmetric buffer; the launch does neither.
            (not layer._sp_moe, "SP-MoE is on"),
            (not layer.all_reduce_fusion, "the fused attention all-reduce is on"),
            (
                hasattr(attn, "forward_pre_o_proj"),
                "the attention cannot stop before o_proj",
            ),
            # The MLA cores reach o_proj from several places, so stopping
            # before it rides on the gated o_proj wrapper, which only exists
            # with the output gate. The KDA layers have no such attribute and
            # stop before o_proj on their own.
            (
                getattr(attn, "use_output_gate", True),
                "the MLA output gate is off, and the split before o_proj "
                "rides on its wrapper (mla_use_output_gate)",
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


def _build(model, hf_config, ctx):
    """Pack every covered layer and compile the widths, before graph capture."""
    from sglang.srt.runtime_context import get_parallel

    from sgl_amd_mono.kimi_k3.gfx950 import k1 as k1_mod
    from sgl_amd_mono.kimi_k3.gfx950.runner import K3MonoRunner

    backbone = _backbone(model)
    p = get_parallel()
    device = backbone.layers[0].input_layernorm.weight.device
    # the peer handshake is collective: every rank reaches it here
    runner = K3MonoRunner(p.tp_size, p.tp_rank, p.tp_group.cpu_group, device)
    runner.set_deadline(ctx.deadline_s)
    layers, batches = {}, {}
    for rows in ctx.widths:
        for i in _covered_layers(hf_config):
            packed = _weights(backbone.layers[i], i, rows, runner)
            layers.setdefault(rows, {})[i] = packed
        # FlyDSL compiles a launcher on its first call, and the first real call
        # is inside a CUDA graph capture, where compiling aborts the process.
        # A compile-only call cannot stand in: with zero arguments the buffer
        # loads fold to a pattern LLVM cannot select. So run each launch once,
        # on this layer's real weights and throwaway activations. Every rank
        # runs the same launches in the same order, so the peer waits pair up
        # exactly as they do on a served step.
        #
        # K1's conv rows come from its batch, so that is built first.
        batches[rows] = k1_mod.build_batch(layers[rows], device)
        _warmup(runner, layers[rows], rows)
    return _Built(model, runner, layers, batches)


def _warmup(runner, packed_layers: dict, rows: int) -> None:
    """Launch every layer once at this width, before any graph capture."""
    from sgl_amd_mono.kimi_k3.gfx950 import k1 as k1_mod
    from sgl_amd_mono.kimi_k3.gfx950.stages.dims import HIDDEN

    device = runner.device
    core = torch.zeros(rows, CORE, dtype=torch.bfloat16, device=device)
    out = torch.empty(rows, HIDDEN, dtype=torch.bfloat16, device=device)
    banks: dict[int, torch.Tensor] = {}

    def bank(nb: int):
        # One row past the nb blocks the launch reads: a block-write layer's
        # launch (K1 or mla_pre, key.write_idx == nb) stores its updated prefix
        # in block nb, as a served step does into the bank's spare capacity.
        # An nb-row bank put the last token's write one HIDDEN row past the end
        # and zeroed whatever the allocator had placed there (seen: a later
        # layer's indexer k_norm at 8 draft tokens, which silently broke that
        # layer's DSA top-k).
        if nb not in banks:
            banks[nb] = torch.zeros(
                rows, nb + 1, HIDDEN, dtype=torch.bfloat16, device=device
            )
        return banks[nb]

    for packed in packed_layers.values():
        # K1 first: on a served step it runs before this layer's K2, so the
        # waits pair up in the same order here as there.
        if packed.k1 is not None:
            k1_mod.warmup(
                runner, packed.k1, runner.prefix[:rows], bank(packed.k1.key.nblocks)
            )
        if packed.mla is not None:
            from sgl_amd_mono.kimi_k3.gfx950.attention import mla_pre as mp

            key, w = packed.mla
            mp.mla_pre_launch(
                key, prefix=runner.prefix[:rows], blocks=bank(key.nblocks), w=w,
                out=_mla_outputs(rows, device), scratch=runner.scratch,
                layer=packed.i, epoch=runner.epoch,
            )  # fmt: skip
        _k2_launch(
            runner, packed, core, runner.prefix[:rows], bank(packed.key.nblocks), out
        )
    torch.cuda.synchronize()


# Bounds against SGLang's standard path on the same GPU. They come from what
# test/kernels/kimi_k3 measures at TP8, with a little room. Each number below
# is the worse of the two prefix modes, and for the MoE the worst of widths
# 1, 4 and 8:
#
#   k2.prefix  cos 1.000000            k2.x  cos 1.000000
#   k2.out     cos 0.999994 rel 0.0036
#   moe.out    cos 0.999847 rel 0.0175
NUMERICS = Numerics(
    NumericsKind.TOLERANCE,
    {
        # The updated attention-residual prefix: o_proj, the TP reduce and
        # the add. The reduce sums the same bf16 partials in the same rank
        # order as the standard path, so the result is exact. A bound below
        # 1.0 would let a reordered reduce through.
        "k2.prefix.cosine": 1.0 - 1e-7,
        # The seam: slice, gate, mix, then the FFN input norm. Also exact.
        "k2.x.cosine": 1.0 - 1e-7,
        # The layer output: the MoE on top of the seam. The expert sums land in
        # another order, so this is bf16 rounding, not exact.
        "k2.out.cosine": 0.99999,
        "k2.out.rel": 0.005,
        # The MoE launch, which runs in no forward: the kernel checks use it as
        # the reference for the MoE half of the layer launch. Looser than
        # k2.out because the check feeds it a fresh input at every width, and
        # one row has the least averaging.
        "moe.out.cosine": 0.9998,
        "moe.out.rel": 0.02,
        # SGLANG_AMD_MONO_CHECK, which scores K1 against SGLang's own
        # attention on the real weights at TP8, over 28 steps of one request:
        #
        #   k1.core    cos 0.99999678  rel 0.0026
        #   k1.bank    rel 0 (bit for bit); cos 0.99999982
        #   k1.ssm     cos 0.99999976  rel 0.00033
        #   k1.window  cos 0.99999988  rel 0.00032
        #
        # The gated core output: the in-projection, f_b, the conv and the
        # recurrence. Its reference sums in another order, so this is bf16
        # rounding.
        "k1.core.cosine": 0.99999,
        "k1.core.rel": 0.004,
        # The bank row the launch writes in place of AttnResidual.write. The
        # same values in the same order, so it is exact: rel is what holds it
        # to that, and DSpark's capture reads the bank, so a row that drifts
        # changes the draft's input. The cosine bound is the looser of the two
        # on purpose. Cosine of two 4608-wide bf16 rows carries its own
        # rounding, measured down to 0.99999982 on rows that are bit for bit.
        "k1.bank.cosine": 1.0 - 1e-6,
        "k1.bank.rel": 0.0,
        # The recurrent state staged for the step's verify rows.
        "k1.ssm.cosine": 1.0 - 1e-6,
        "k1.ssm.rel": 0.001,
        # The conv window handed back to SGLang's speculative state, which the
        # accept-time commit reads. Nothing in the serving path reports a wrong
        # one: it shows up as a lower acceptance length and a GSM8K score a few
        # points down. Measured exact, so the bound is tight on purpose.
        "k1.window.cosine": 1.0 - 1e-6,
        "k1.window.rel": 0.001,
    },
)

SPEC = MonoSpec(
    name="kimi_k3_hip",
    devices={
        "gfx950": DeviceLimits(
            widths=tuple(range(1, MAX_ROWS + 1)),
            tp_sizes=(TP_SIZE,),
            resident_blocks=BLOCKS,
        )
    },
    # The measured win is on DSpark target-verify steps; a plain decode step of
    # at most MAX_ROWS rows takes the same launch.
    features=frozenset({Feature.SPECULATIVE_DECODING}),
    requires=Requirements(
        dtypes=frozenset({"bfloat16"}),
        attention_backends=frozenset({"aiter"}),
        server_args={
            # the launch holds the experts TP sharded, not expert-parallel
            "moe_a2a_backend": frozenset({None, "none"}),
            # DSpark's target steps are uniform verify steps; other drafts differ
            "speculative_algorithm": frozenset({None, "DSPARK"}),
        },
        # The shapes are on the text config, which this cannot reach; they are
        # checked in _config_disable_reasons.
        hf_config={"model_type": "kimi_k3"},
        env={"SGLANG_USE_AITER": True},
    ),
    numerics=NUMERICS,
    model_disable_reasons=_model_disable_reasons,
    build=_build,
    api_version=1,  # sglang.srt.monokernels.spec.MONOKERNEL_API_VERSION
    config_disable_reasons=_config_disable_reasons,
)


class _Layer(msgspec.Struct):
    """One layer's launches: their build keys and the weights they read in
    place. ``k1`` is None on the full-attention layers, which keep SGLang's
    attention and enter at ``forward_pre_o_proj``."""

    i: int
    key: Any
    module: Any  # the KimiK3DecoderLayer, for the tensors read per step
    k1: Any = None  # the KDA front launch (gfx950/k1.py), where it applies
    mla: Any = None  # (MlaPreBuild, weights): an MLA layer's projection front


def _k2_blocks(layer) -> int:
    """Bank rows readable at this layer's MLP seam. A block-write layer
    snapshots its own pre-attention prefix on the attention side, so by the
    seam it has one row more than it inherited."""
    return layer.prev_valid_blocks + int(layer.is_block_write_layer)


def _weights(layer, i: int, rows: int, runner) -> _Layer:
    """This layer's build key at one launch width. The weights themselves are
    read in place at launch, so nothing is packed or copied here."""
    from sgl_amd_mono.kimi_k3.gfx950 import k1 as k1_mod
    from sgl_amd_mono.kimi_k3.gfx950 import layer as k2
    from sgl_amd_mono.kimi_k3.gfx950.layer import K2Build

    moe = layer.mlp
    return _Layer(
        k1=k1_mod.build(layer, i, rows) if K1 else None,
        mla=_mla_build(layer, rows) if MLA_PRE else None,
        i=i,
        key=K2Build(
            tokens=rows,
            nblocks=_k2_blocks(layer),
            eps=layer.mlp_res_norm.variance_epsilon,
            out_eps=layer.post_attention_layernorm.variance_epsilon,
            ln_eps=moe.routed_expert_norm.variance_epsilon,
            reset=layer.is_block_write_layer,
            opt=k2.K2_OPT if K2_OPT is None else int(K2_OPT),
            guint=_gu_interleaved(moe),
            fault_site=runner.fault_site,
            fault_after=runner.fault_after,
        ),
        module=layer,
    )


def _mla_build(layer, rows: int):
    """An MLA layer's projection-front key and the weights it reads in place, or
    None (a KDA layer, or weights the launch does not take: SGLang's path)."""
    from sglang.srt.runtime_context import get_parallel

    from sgl_amd_mono.kimi_k3.gfx950.attention import mla_pre as mp
    from sgl_amd_mono.kimi_k3.gfx950.stages.dims import HIDDEN

    attn = layer.self_attn
    idx = getattr(attn, "indexer", None)
    fused = getattr(attn, "fused_qkv_a_proj_with_mqa", None)
    if hasattr(attn, "qkv_conv1d") or idx is None or fused is None:
        return None
    if get_parallel().attn_dcp_size > 1:
        # under DCP SGLang's MLA forward computes every head's q itself (the
        # replicated q_b); the front hands over this rank's 12
        return None
    w = {
        "ares_nw": layer.self_attention_res_norm.weight,
        "ares_qk": layer.self_attention_res_proj.weight.view(HIDDEN),
        "in_nw": layer.input_layernorm.weight,
        "w_qkv": fused.weight,
        "w_g": getattr(getattr(attn, "g_proj", None), "weight", None),
        "w_k": getattr(getattr(idx, "wk", None), "weight", None),
        "w_p": getattr(getattr(idx, "weights_proj", None), "weight", None),
        "w_qb": attn.q_b_proj.weight,
        "w_iq": getattr(getattr(idx, "wq_b", None), "weight", None),
        "q_nw": attn.q_a_layernorm.weight,
        "kv_nw": attn.kv_a_layernorm.weight,
    }
    want = {
        "w_qkv": (mp.LATW, HIDDEN), "w_g": (mp.GATE, HIDDEN), "w_k": (mp.IK, HIDDEN),
        "w_p": (mp.IW, HIDDEN), "w_qb": (mp.QB, mp.QL), "w_iq": (mp.IQ, mp.QL),
    }  # fmt: skip
    for n, shp in want.items():
        t = w[n]
        if (
            not isinstance(t, torch.Tensor)
            or t.dtype != torch.bfloat16
            or tuple(t.shape) != shp
            or not t.is_contiguous()
        ):
            logger.info("monokernel: no MLA front for layer %d: %s is %s", layer.layer_id if hasattr(layer, "layer_id") else -1, n, None if t is None else (t.dtype, tuple(t.shape)))  # fmt: skip
            return None
    # the absorb's weight as SGLang's bf16 bmm multiplies it: w_kc itself, unscaled
    wkc, wsc = getattr(attn, "w_kc", None), getattr(attn, "w_scale", None)
    qn = (
        isinstance(wkc, torch.Tensor)
        and wkc.dtype == torch.bfloat16
        and tuple(wkc.shape) == (mp.NHQ, mp.QN, mp.KV)
        and isinstance(wsc, (int, float))
        and wsc == 1.0
    )
    if qn:
        w["w_kc"] = wkc.transpose(1, 2).contiguous().view(mp.NHQ * mp.KV, mp.QN)
    key = mp.MlaPreBuild(
        tokens=rows,
        qn=qn,
        nblocks=layer.prev_valid_blocks,
        eps=layer.self_attention_res_norm.variance_epsilon,
        out_eps=layer.input_layernorm.variance_epsilon,
        q_eps=attn.q_a_layernorm.variance_epsilon,
        kv_eps=attn.kv_a_layernorm.variance_epsilon,
        write_idx=(layer.prev_valid_blocks if layer.is_block_write_layer else -1),
    )
    return (key, w)


def _mla_outputs(rows: int, device) -> dict:
    from sgl_amd_mono.kimi_k3.gfx950.attention import mla_pre as mp

    def z(n):
        return torch.zeros(rows, n, dtype=torch.bfloat16, device=device)

    return {
        "q": z(mp.QB), "latent": z(mp.KV + mp.ROPE), "k_nope": z(mp.KV),
        "q_lora": z(mp.QL), "gate": z(mp.GATE), "idx_q": z(mp.IQ),
        "idx_k": z(mp.IK), "idx_w": z(mp.IW),
        "qn": torch.zeros(rows, mp.NHQ, mp.KV, dtype=torch.bfloat16, device=device),
    }  # fmt: skip


def _gu_interleaved(moe) -> bool:
    """Whether the experts carry the gate / up interleaved MXFP4 layout.

    ``is_shuffled`` alone cannot answer this: Mxfp4MoEMethod sets it on both of
    its branches and only the SiTU A8W4 branch interleaves. Reading those bytes
    with the wrong layout does not fail, it returns wrong numbers, so ask the
    same question Mxfp4MoEMethod asked.
    """
    from sglang.srt.layers.quantization.mxfp4 import (
        _aiter_situ_uses_gu_interleaved_weights,
    )

    if getattr(moe.experts.moe_runner_config, "activation", None) != "situ":
        return False
    return _aiter_situ_uses_gu_interleaved_weights()


def _up_shard(moe, rank: int):
    """This rank's rows of the replicated latent up projection. Slicing the
    first dimension of a contiguous weight keeps it contiguous: a view."""
    from sgl_amd_mono.kimi_k3.gfx950.stages.moe import UP_N

    return moe.routed_expert_up_proj.weight.narrow(0, rank * UP_N, UP_N)


def _k2_launch(
    runner, packed: _Layer, core, prefix, blocks, out, prefix_in=None
) -> None:
    """The launch itself: this layer's weights read in place, the activations
    the caller owns."""
    from sgl_amd_mono.kimi_k3.gfx950 import layer as k2
    from sgl_amd_mono.kimi_k3.gfx950.stages.dims import HIDDEN

    module = packed.module
    moe, experts = module.mlp, module.mlp.experts
    k2.k2_launch(
        packed.key,
        core=core,
        w_o=module.self_attn.o_proj.weight,
        prefix=prefix,
        blocks=blocks,
        ares_nw=module.mlp_res_norm.weight,
        ares_qk=module.mlp_res_proj.weight.view(HIDDEN),
        in_nw=module.post_attention_layernorm.weight,
        w_gate=moe.gate.weight,
        bias=moe.gate.e_score_correction_bias,
        w_ld=moe.routed_expert_down_proj.weight,
        w_sgu=moe.shared_experts.gate_up_proj.weight,
        w_sd=moe.shared_experts.down_proj.weight,
        w13=experts.w13_weight,
        w13s=experts.w13_weight_scale,
        w2=experts.w2_weight,
        w2s=experts.w2_weight_scale,
        ln_w=moe.routed_expert_norm.weight,
        w_up=_up_shard(moe, runner.rank),
        out=out,
        prefix_in=prefix_in,
        **runner.launch_args(),
        # One tag a layer: a layer takes at most one launch in a step.
        layer=128 + packed.i,
    )


def _launch(runner, packed: _Layer, attn_res, core, forward_batch):
    """One layer, from the attention's TP-local core to the layer's output."""
    from sglang.srt.layers.layer_boundary.residual.batch import written_residual

    module = packed.module
    rows = core.shape[0]
    core = core.contiguous()
    # A block-write layer restarts the prefix, so it takes none.
    prefix_sum = (
        None if module.is_block_write_layer else written_residual(forward_batch)
    )
    _check(packed, core, prefix_sum, attn_res, rows)
    prefix = runner.prefix[:rows]
    out = torch.empty_like(prefix)
    # The kernel reads the stream from prefix_sum (lent out read-only), updates its
    # own prefix and writes out + prefix itself.
    _k2_launch(runner, packed, core, prefix, attn_res.block_residual, out, prefix_sum)
    return out


def _check(packed: _Layer, core, prefix_sum, attn_res, rows: int) -> None:
    """What the routing decision could not know, fail-loud.

    Every tensor reaches the kernel as a raw address, so a wrong dtype or
    stride is read as if it were the expected one. The step was routed for
    every rank at once, so falling back here is not an option.
    """
    from sgl_amd_mono.kimi_k3.gfx950.stages.dims import HIDDEN

    module = packed.module
    assert core.shape == (rows, CORE) and core.dtype == torch.bfloat16, (
        tuple(core.shape),
        core.dtype,
    )
    assert attn_res is not None, "the attention-residual stream is not open"
    # The kernel's two prefix modes are chosen at build time, so the caller's
    # mode has to be the one this layer was built for.
    assert (prefix_sum is None) == module.is_block_write_layer
    if prefix_sum is not None:
        assert (
            prefix_sum.shape == (rows, HIDDEN)
            and prefix_sum.dtype == torch.bfloat16
            and prefix_sum.is_contiguous()
        ), (tuple(prefix_sum.shape), prefix_sum.dtype)
    assert attn_res.num_valid_blocks == packed.key.nblocks, (
        attn_res.num_valid_blocks,
        packed.key.nblocks,
    )
    bank = attn_res.block_residual
    assert (
        bank.dtype == torch.bfloat16
        and bank.size(0) >= rows
        and bank.size(2) == HIDDEN
        and bank.stride(2) == 1
    ), (bank.dtype, tuple(bank.shape), bank.stride())


def _mla_covered(packed, hidden, bank) -> bool:
    """The MLA front takes a layer whose stream is a written tensor (not an owed
    contribution), with its bank open at the row count it was built for."""
    from sgl_amd_mono.kimi_k3.gfx950.stages.dims import HIDDEN

    key, _ = packed.mla
    return (
        bank is not None
        and isinstance(hidden, torch.Tensor)
        and hidden.dtype == torch.bfloat16
        and hidden.is_contiguous()
        and hidden.dim() == 2
        and hidden.shape[1] == HIDDEN
        and bank.num_valid_blocks == key.nblocks
        and bank.block_residual.dtype == torch.bfloat16
    )


class _LanguageModelShim:
    """``general_mm_embed_routine``'s language model for a routed step.

    The routine reads the real model (``get_input_embeddings``, ``pp_group``)
    and then calls it; this hands the reads through and takes the call.
    """

    def __init__(self, built: _Built) -> None:
        self._built, self._lm = built, built.lm

    def __getattr__(self, name):
        return getattr(self._lm, name)

    def __call__(self, input_ids=None, forward_batch=None, input_embeds=None, **kw):
        return self._built.language_forward(
            input_ids, forward_batch, input_embeds, **kw
        )


class _Built:
    """One rank's monokernels: the runner, each layer's weights, and the forward."""

    def __init__(self, model, runner, by_width: dict, k1_by_width: dict) -> None:
        self.model = model
        self.lm = getattr(model, "language_model", model)
        self.backbone = self.lm.model
        # {launch width: {layer id: its packed launch}}; a layer's build
        # key carries the width, so each width has its own compiled kernel
        self.runner, self._by_width = runner, by_width
        # {launch width: that width's K1 layers as one batch}, None where K1
        # took no layer
        self._k1_by_width = k1_by_width
        # This step's shared K1 state and the layers that took it, set by
        # _backbone before the first layer and read by _core and finish().
        self._k1_state, self._k1_engaged = None, []
        # Read once: _core asks this for every layer of every step, and a
        # served step must not pay for a check that is off.
        self._check = os.environ.get("SGLANG_AMD_MONO_CHECK") == "1"
        # The conv windows waiting for finish(), under that flag only.
        self._check_handoff = []
        self._shim = _LanguageModelShim(self)
        # The MLA front's outputs, one set a width (the layers run in sequence)
        dev = runner.device
        self._mla_out = {
            rows: _mla_outputs(rows, dev)
            for rows, packed in by_width.items()
            if any(p.mla is not None for p in packed.values())
        }

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
        hidden = self._backbone(input_ids, forward_batch, input_embeds)
        aux = None
        if lm.capture_aux_hidden_states:
            hidden, aux = hidden
        return lm.logits_processor(input_ids, hidden, lm.lm_head, forward_batch, aux)

    def _backbone(self, input_ids, forward_batch, input_embeds):
        """``KimiK3LinearModel.forward``: the embedding, every layer, the final
        norm. Pipeline and DP attention are refused, so this rank is the first
        and the last, and the bank is always open."""
        from sglang.srt.layers.layer_boundary.residual.batch import final_norm, start
        from sglang.srt.models.kimi_k3 import AuxHiddenStateList, _cdiv
        from sglang.srt.utils.common import BumpAllocator

        from sgl_amd_mono.kimi_k3.gfx950 import k1 as k1_mod

        m = self.backbone
        self.runner.step_begin()
        # Once a step, before the first layer: every K1 layer's conv window
        # staged in one write. _k1_engaged collects the layers that take it,
        # which is what finish() hands back for.
        k1_batch = self._k1_by_width[self._plan.key.rows]
        self._k1_state = k1_mod.prepare(k1_batch, forward_batch)
        self._k1_engaged = []
        self._check_handoff = []
        hidden = m.embed_tokens(input_ids) if input_embeds is None else input_embeds
        start(forward_batch)
        zero_allocator = BumpAllocator(
            buffer_size=2 * (m.end_layer - m.start_layer),
            dtype=torch.float32,
            device=hidden.device,
        )
        m.attn_bank.open(hidden, _cdiv(m.end_layer, m.config.attn_res_block_size), None)
        # The bank's stream is captured after a layer, not at the next layer's
        # input (``captures_at_input`` is False whenever the bank is open).
        aux = AuxHiddenStateList()
        captures = m.dspark_layers_to_capture or ()
        for i in range(m.start_layer, m.end_layer):
            hidden = self._layer(i, hidden, forward_batch, zero_allocator)
            if i in captures:
                aux.append(m._dspark_capture_stream(i, hidden, forward_batch))
        # Once a step, after the last layer: the conv inputs handed back to
        # SGLang's speculative window, which only reads them at accept.
        k1_mod.finish(k1_batch, self._k1_state, self._k1_engaged)
        if self._check:
            from sgl_amd_mono.kimi_k3 import check as check_mod

            # The conv window is handed back once a step, so it is scored here
            # and not in the layer that produced it.
            check_mod.compare_handoff(self)
        hidden = final_norm(
            hidden, forward_batch, m._final_read, capture=None, skip_empty=True
        )
        m.attn_bank.close()
        if m.dspark_layers_to_capture is not None:
            return hidden, aux
        return hidden

    def _layer(self, i, hidden, forward_batch, zero_allocator):
        """One layer: the launch where it is covered, else the model's own."""
        from sglang.srt.layers.layer_boundary.residual.batch import set_written

        layer = self.backbone.layers[i]
        packed = self._by_width[self._plan.key.rows].get(i)
        if packed is None:  # the dense prefix keeps the model's forward
            return layer(
                positions=self._positions,
                hidden_states=hidden,
                forward_batch=forward_batch,
                zero_allocator=zero_allocator,
                capture_gathered=None,
            )
        bank = self.backbone.attn_bank.current
        core = self._core(layer, packed, hidden, bank, forward_batch, zero_allocator)
        out = _launch(self.runner, packed, bank, core, forward_batch)
        # The launch folded the residual in, so the stream is written rather
        # than finished through ffn_boundary.
        return set_written(out, forward_batch)

    def _core(self, layer, packed, hidden, bank, forward_batch, zero_allocator):
        """This layer's TP-local attention output, which the layer launch takes.

        A KDA layer takes the K1 launch where it is covered: K1 runs the
        attention-side seam, ``input_layernorm`` and the whole KDA attention in
        one launch, so it replaces both the boundary's prepare and the model's
        attention. K1 reads the residual stream and does not add to it -- the
        attention's contribution is added at the MLP seam, which is K2's -- so
        the stream is handed on unchanged, which is what the boundary's prepare
        would have left for K2's ``written_residual``.

        Everything else -- the full-attention layers, and a KDA layer K1
        declines -- keeps SGLang's path. K1 writes nothing before it declines,
        so that fallback is clean.
        """
        from sglang.srt.layers.layer_boundary.residual.batch import set_written

        from sgl_amd_mono.kimi_k3.gfx950 import k1 as k1_mod

        state = self._k1_state
        if packed.mla is not None and _mla_covered(packed, hidden, bank):
            return self._mla_core(
                layer, packed, hidden, bank, forward_batch, zero_allocator
            )
        if packed.k1 is not None and k1_mod.covered(packed.k1, hidden, bank, state):
            ref = None
            if self._check:
                from sgl_amd_mono.kimi_k3 import check as check_mod

                # Before K1, on the same input and the caches as they were.
                ref = check_mod.reference(
                    self, layer, packed, hidden, bank, forward_batch, zero_allocator
                )
            set_written(hidden, forward_batch)
            self._k1_engaged.append(packed.i)
            core = k1_mod.launch(self.runner, packed.k1, hidden, bank, state)
            if self._check:
                from sgl_amd_mono.kimi_k3 import check as check_mod

                check_mod.compare(self, packed.i, core, ref, bank)
            return core
        x = layer.attn_boundary.prepare(hidden, forward_batch, capture_gathered=None)
        return self._pre_o_proj(layer, x, forward_batch, zero_allocator)

    def _mla_core(self, layer, packed, hidden, bank, forward_batch, zero_allocator):
        """An MLA layer: the projection front as one launch, then SGLang's MLA
        forward (RoPE, the KV write, the indexer's norms / quantization / logits /
        top-k, the sparse attention) on its outputs, up to o_proj's input."""
        from sglang.srt.layers.layer_boundary.residual.batch import set_written

        from sgl_amd_mono.kimi_k3.gfx950.attention import mla_pre as mp
        from sgl_amd_mono.kimi_k3.gfx950.stages.dims import HIDDEN

        key, w = packed.mla
        rows = hidden.shape[0]
        out = self._mla_out[rows]
        # The launch reads the stream and does not add to it, as K1 does.
        set_written(hidden, forward_batch)
        mp.mla_pre_launch(
            key, prefix=hidden, blocks=bank.block_residual, w=w, out=out,
            scratch=self.runner.scratch, layer=packed.i, epoch=self.runner.epoch,
        )  # fmt: skip
        if key.write_idx >= 0:
            bank._count_write(None)  # the launch wrote the bank row
        off, nbytes = mp.scratch_layout(key)["x"]
        x = (
            self.runner.scratch[off : off + nbytes]
            .view(torch.bfloat16)
            .view(rows, HIDDEN)
        )
        attn = layer.self_attn
        attn.__dict__["_mono_pre"] = {
            "q": out["q"], "latent": out["latent"], "k_nope": out["k_nope"],
            "q_lora": out["q_lora"], "gate": out["gate"],
            "q_nope_out": out["qn"] if key.qn else None,
        }  # fmt: skip
        attn.indexer.__dict__["_mono_pre"] = {
            "q": out["idx_q"], "k": out["idx_k"], "w": out["idx_w"],
        }  # fmt: skip
        try:
            return self._pre_o_proj(layer, x, forward_batch, zero_allocator)
        finally:
            attn.__dict__.pop("_mono_pre", None)
            attn.indexer.__dict__.pop("_mono_pre", None)

    def _pre_o_proj(self, layer, x, forward_batch, zero_allocator):
        """``KimiK3DecoderLayer._run_self_attn`` stopped before o_proj: the
        attention's TP-local output, which is where the launch starts."""
        from sglang.srt.layers.layer_boundary import (
            AttentionInputs,
            get_attn_tp_context,
        )

        attn = layer.self_attn
        latent = getattr(attn, "prepare_qkv_latent", None)
        if latent is not None:
            # MLA layers with q_lora_rank read their latent inputs from here.
            get_attn_tp_context().set_attn_inputs(
                AttentionInputs(x, forward_batch, latent)
            )
        try:
            return attn.forward_pre_o_proj(
                hidden_states=x,
                positions=self._positions,
                forward_batch=forward_batch,
                zero_allocator=zero_allocator,
            )
        finally:
            if latent is not None:
                get_attn_tp_context().clear_attn_inputs()
