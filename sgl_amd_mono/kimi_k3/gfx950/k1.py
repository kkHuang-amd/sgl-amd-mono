# SPDX-License-Identifier: Apache-2.0
"""The K1 launch's host side: one Kimi-K3 KDA layer from the attention-residual
seam to the gated core output.

K1 covers the attention-side seam, ``input_layernorm``, the in-projection,
``f_b``, the conv, the recurrence and the gated output norm. Its output is what
``forward_pre_o_proj`` returns, so K1 and the layer launch (K2) chain with no
glue: K1 produces the core, K2 consumes it.

Only the KDA layers. The full-attention layers keep SGLang's path up to
``forward_pre_o_proj`` and K2 takes them from there, so the layer launch still
covers every MoE layer.

Two things here are worth knowing before changing any of it.

**K1 reads its own conv buffer, not SGLang's.** SGLang's speculative conv
window (``intermediate_conv_window``) is one entry short of what K1 reads: K1
reads up to ``acc_idx + CONV_W - 1``, which at a full acceptance is
``qlen + CONV_W - 2`` and so needs ``state_len >= qlen + CONV_W - 1`` = 11,
while the pool allocates ``num_draft + CONV_W - 2`` = 10. Rather than widen a
pool every linear-attention model shares, ``K1Batch`` holds one buffer for all
the layers and ``prepare`` stages it from the committed conv state once a step.
That pins ``num_acc`` at 1, which drops the deepest read to ``CONV_W - 1``, and
leaves K1 depending on no buffer SGLang lets go stale.
``test/kernels/test_kimi_k3_k1.py`` measures both the footprint and the
one-short case, which does not fault -- FlyDSL's ``buffer_load`` is OOB-clamped,
so a short window loses precision silently.

**One request a launch.** K1 indexes the conv buffer with the same number it
indexes the recurrent state with, and that number is a row of SGLang's scratch,
not of this buffer. The buffer is expanded to a zero slot stride so every such
number folds onto its single row. That is what makes one request work and what
makes two impossible, so ``covered()`` refuses a batch above one.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

import msgspec
import torch
import triton
import triton.language as tl

from sgl_amd_mono.kimi_k3 import MAX_BLOCKS, MAX_ROWS
from sgl_amd_mono.kimi_k3.gfx950.stages.dims import HIDDEN

logger = logging.getLogger(__name__)

# The conv window's staging (prepare) and hand-back (finish) run as one Triton
# launch each, instead of ~13 small torch kernels a step. Same values: copies
# only; check mode keeps torch's.
_CONV_BLOCK = 512


@triton.jit
def _stage_window(
    committed, c_l, c_s, c_t, c_c, lidx, slot, conv, d_n, d_c, d_t, C,
    WIN: tl.constexpr, BLOCK: tl.constexpr,
):  # fmt: skip
    """conv[n, c, t] = committed[lidx[n], slot, t, c] for the window's t < WIN."""
    n = tl.program_id(0)
    c = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    m = c < C
    src = committed + tl.load(lidx + n) * c_l + tl.load(slot) * c_s + c * c_c
    dst = conv + n * d_n + c * d_c
    for t in tl.static_range(WIN):
        tl.store(dst + t * d_t, tl.load(src + t * c_t, mask=m), mask=m)


@triton.jit
def _hand_back(
    committed, c_l, c_s, c_t, c_c, conv, d_n, d_c, d_t, view, v_l, v_r, v_j, v_t, v_c,
    lidx, bidx, slot, row, C, WIN: tl.constexpr, QLEN: tl.constexpr, BLOCK: tl.constexpr,
):  # fmt: skip
    """view[lidx[n], row, j, t, c] = seq[c, j + 1 + t], seq the committed window
    followed by this step's conv inputs (conv[bidx[n], c, WIN ..])."""
    n = tl.program_id(0)
    c = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    m = c < C
    li = tl.load(lidx + n)
    src = committed + li * c_l + tl.load(slot) * c_s + c * c_c
    new = conv + tl.load(bidx + n) * d_n + c * d_c
    dst = view + li * v_l + tl.load(row) * v_r + c * v_c
    for j in tl.static_range(QLEN):
        for t in tl.static_range(WIN):
            if j + 1 + t < WIN:
                v = tl.load(src + (j + 1 + t) * c_t, mask=m)
            else:
                v = tl.load(new + (j + 1 + t) * d_t, mask=m)
            tl.store(dst + j * v_j + t * v_t, v, mask=m)


def _conv_triton() -> bool:
    return os.environ.get("SGLANG_AMD_MONO_CHECK") != "1"


# Said once a process. This catches a launch that is built but never reached.
# Such a launch costs nothing at run time. It looks exactly like a launch that
# does not help. Without a line in the log it reads as a measurement.
_said: set[str] = set()


def _say(message: str) -> None:
    if message not in _said:
        _said.add(message)
        logger.info("Kimi-K3 K1: %s", message)


# The SGLang names this launch reads, by file under the sglang package;
# test/unit/test_contract.py pins them.
SGLANG_READS = {
    # the KDA layer's speculative state. Taken from the backend here rather
    # than through an accessor on it, the way glm53_flash reads its own.
    "srt/model_executor/forward_context.py": ("get_attn_backend",),
    "srt/layers/attention/hybrid_linear_attn_backend.py": (
        "linear_attn_backend",
        "req_to_token_pool",
        "forward_metadata",
    ),
    "srt/layers/attention/linear/kda_backend.py": (
        "verify_intermediate_state_indices",
    ),
    "srt/layers/attention/linear/utils.py": (
        "select_verify_intermediate_state_indices",
    ),
    "srt/mem_cache/memory_pool.py": (
        "mamba2_layer_cache",
        "intermediate_ssm",
        "intermediate_conv_window",
        "replayssm_rawv",
        "replayssm_rawk",
        "replayssm_g",
        "replayssm_beta",
    ),
    "srt/runtime_context.py": ("get_exec",),
    "srt/layers/attention/mamba/mamba2_metadata.py": ("mamba_cache_indices",),
    # the kernel writes the bank row; this is the count that goes with it
    "srt/layers/attn_residual.py": ("_count_write", "num_valid_blocks"),
    # this layer's weights, read in place at launch
    "srt/models/kimi_k3.py": (
        "use_full_rank_gate",
        "_qkvgbfa_layer",
        "qkv_conv1d",
        "f_b_proj",
        "A_log",
        "dt_bias",
        "o_norm",
        "input_layernorm",
        "self_attention_res_norm",
        "self_attention_res_proj",
    ),
}


class K1Layer(msgspec.Struct):
    """One KDA layer's K1 launch: its build key, the module whose weights the
    launch reads in place, and this layer's row of the step's conv buffer."""

    i: int
    key: Any
    module: Any  # the KimiK3DecoderLayer
    on_w: Any  # o_norm's weight as the kernel reads it, cast once
    conv: Any  # a row of K1Batch.conv, not an allocation of its own
    mamba_idx: int  # this layer's index into the KDA state pools


class K1Batch(msgspec.Struct):
    """Every K1 layer of one width, and the state they share.

    The conv staging and the hand-off are the same few tensor ops on every
    layer, and their cost is the dispatch, not the bytes: 912 KiB a layer moves
    in 0.2 us at this GPU's bandwidth and took 23 us as six per-layer ops. So
    they run once a step over all the layers at once, which is what the layer
    index below is for, and the conv buffer is one allocation they slice.
    """

    layers: Any  # the K1Layer list, in layer order
    conv: Any  # [n, 3 * PROJ, state_len], one row a layer
    idx: Any = None  # int64 [n]: those layers' indices into the state pools
    where: Any = None  # int64: the rows that engage, None when every one does
    where_key: Any = None  # the engaged layer ids `where` was built for
    ramp: Any = None  # int64 [1, qlen]: 0 .. qlen - 1, the step's token offsets
    ones: Any = None  # int32 [1]: num_acc, which this form pins at 1
    bidx: Any = None  # int64: the engaged layers' rows of conv (the Triton hand-back)
    lidx: Any = None  # int64: their indices into the state pools


def _is_kda_layer(layer) -> bool:
    """Whether this layer's attention is the KDA one. K1 covers only those."""
    return hasattr(layer.self_attn, "qkv_conv1d")


def _unsupported_reasons(layer) -> list[str]:
    """Why K1 cannot take this layer, fail-loud at build rather than per step."""
    attn = layer.self_attn
    reasons = []
    # w_in is the merged [q,k,v,g | f_a | b | pad] buffer. Without the merge
    # there is no single tensor in K1's in-projection layout, and the pieces
    # carry their own quantization scales that a row-cat would drop.
    #
    # A quantized checkpoint is not a reason on its own. SGLang takes the merge
    # whenever a monokernel backend is selected, and Kimi-K3's checkpoint
    # quantizes only its experts. What is left is the gate, the environment
    # variable, and attention weights the merge may not concatenate.
    if not getattr(attn, "use_full_rank_gate", False):
        reasons.append(
            "the KDA gate is not full-rank, so there is no wide projection to "
            "extend (linear_attn_config.use_full_rank_gate)"
        )
    elif getattr(attn, "_qkvgbfa_layer", None) is None:
        reasons.append(
            "the KDA in-projection is not merged "
            "(SGLANG_ROCM_K3_FUSE_KDA_INPROJ=0, or attention weights this "
            "merge may not concatenate)"
        )
    conv = getattr(attn, "qkv_conv1d", None)
    if conv is None or conv.weight.dim() != 3 or conv.weight.shape[1] != 1:
        shape = None if conv is None else tuple(conv.weight.shape)
        reasons.append(f"the KDA conv weight is {shape}, not [channels, 1, width]")
    return reasons


def _ring_mode() -> bool:
    """Whether the KDA spec state is the ReplaySSM ring rather than per-token
    snapshots (``--enable-linear-replayssm-spec``); fixed for the server's life."""
    from sglang.srt.runtime_context import get_exec

    return bool(get_exec().mamba.enable_linear_replayssm_spec)


def build(layer, i: int, rows: int) -> Optional[K1Layer]:
    """This layer's K1 build key at one launch width, or None when K1 does not
    take it. The weights are read in place; the conv row is filled in by
    ``build_batch``, which allocates every layer's in one buffer."""
    from sgl_amd_mono.kimi_k3.gfx950.attention import kda

    if not _is_kda_layer(layer):
        return None  # a full-attention layer, not a refusal
    reasons = _unsupported_reasons(layer)
    if reasons:
        # Said once: a KDA layer K1 could not take is a configuration fact the
        # log has to carry, or K1 reads as built and unhelpful.
        _say(f"not built for the KDA layers: {'; '.join(reasons)}")
        return None
    attn = layer.self_attn
    return K1Layer(
        i=i,
        key=kda.KdaPreBuild(
            tokens=rows,
            qlen=rows,
            nblocks=layer.prev_valid_blocks,
            delta=False,
            state_len=rows + kda.CONV_W - 1,
            eps=layer.self_attention_res_norm.variance_epsilon,
            out_eps=layer.input_layernorm.variance_epsilon,
            onorm_eps=float(attn.o_norm.eps),
            lower_bound=float(attn.attn.lower_bound),
            write_idx=(layer.prev_valid_blocks if layer.is_block_write_layer else -1),
            ring=_ring_mode(),
            # K1 reads the initial state from the committed pool, so the step
            # stages no copy of it (the ring reads the committed state itself)
            init_committed=not _ring_mode(),
        ),
        module=layer,
        # The same cast SGLang's own fused KDA decode makes once at load
        # (kimi_k3.py _prepare_fused_decode); a step must not repeat it.
        on_w=attn.o_norm.weight.data.to(torch.bfloat16).contiguous(),
        conv=None,  # build_batch
        mamba_idx=-1,  # build_batch
    )


def build_batch(packed_layers, device) -> Optional[K1Batch]:
    """Every K1 layer of one width in one block, after the per-layer builds.

    The conv buffer is ``n x 3 PROJ x state_len`` bf16 -- 7 MiB for Kimi-K3's 69
    KDA layers -- and each layer takes a row of it, so the step's staging is one
    write rather than one a layer.

    The pool indices are not taken here: reaching the state pools needs a
    forward context and this runs at load. ``prepare`` fills them in on its
    first call, which is an eager warmup forward, before any capture.
    """
    from sgl_amd_mono.kimi_k3.gfx950.attention import kda

    ks = [p.k1 for p in packed_layers.values() if p.k1 is not None]
    if not ks:
        return None
    ks.sort(key=lambda k: k.i)
    sl = ks[0].key.state_len
    assert all(k.key.state_len == sl for k in ks), "the K1 layers disagree on state_len"
    buf = torch.empty(len(ks), 3 * kda.PROJ, sl, dtype=torch.bfloat16, device=device)
    for n, k in enumerate(ks):
        k.conv = buf[n : n + 1]
    return K1Batch(layers=ks, conv=buf)


def scratch_bytes(rows: int) -> int:
    """What one K1 launch needs, for the runner's shared scratch.

    The layout grows with the layer's bank depth, so it is sized for the
    deepest layer a launch is built for: a deeper layer's tail would land in
    whatever the allocator placed after the scratch.
    """
    from sgl_amd_mono.kimi_k3.gfx950.attention import kda

    return kda.scratch_bytes(
        kda.KdaPreBuild(
            tokens=rows,
            qlen=rows,
            nblocks=MAX_BLOCKS,
            delta=False,
            state_len=rows + kda.CONV_W - 1,
        )
    )


def _weight_args(packed: K1Layer) -> dict:
    """The tensors the launch reads in place. One list, because the warmup and
    a served step hand the kernel the same weights and only the activations
    and the state differ."""
    attn = packed.module.self_attn
    return {
        "ares_nw": packed.module.self_attention_res_norm.weight,
        "ares_qk": packed.module.self_attention_res_proj.weight.view(HIDDEN),
        "in_nw": packed.module.input_layernorm.weight,
        "w_in": attn._qkvgbfa_layer.weight,
        "w_fb": attn.f_b_proj.weight,
        "conv_w": attn.qkv_conv1d.weight.squeeze(1),
        "a_log": attn.A_log.detach().reshape(-1),
        "dt_bias": attn.dt_bias,
        "on_w": packed.on_w,
        # Expanded so that its slot stride is zero: K1 indexes this buffer with
        # a row number of SGLang's scratch, and a zero stride folds every such
        # number onto the single row here. See the module docstring.
        "conv_state": packed.conv.expand(2, -1, -1),
    }


def warmup(runner, packed: K1Layer, prefix, blocks) -> None:
    """Launch K1 once on this layer's real weights and throwaway activations.

    FlyDSL compiles a launcher on its first call, and the first real call is
    inside a CUDA graph capture, where compiling aborts the process. The
    state tensors are not available yet at build, so they are stand-ins of the
    right shape; only the build key reaches the compiler.
    """
    from sgl_amd_mono.kimi_k3.gfx950.attention import kda

    need = kda.scratch_bytes(packed.key)
    assert need <= runner.scratch.numel(), (
        f"K1 layer {packed.i} needs {need} B of scratch, the runner has "
        f"{runner.scratch.numel()}"
    )
    # A block-write layer stores its bank row at block write_idx: the bank must
    # hold that block, or the last token's row lands past the allocation.
    assert packed.key.write_idx < blocks.shape[1], (
        f"K1 layer {packed.i} writes bank block {packed.key.write_idx}, the "
        f"bank has {blocks.shape[1]}"
    )
    rows, device = packed.key.qlen, runner.device
    ring, slots = None, rows
    if packed.key.ring:
        slots = 1
        f32, bf16 = torch.float32, torch.bfloat16
        ring = tuple(
            torch.zeros(1, kda.NH, rows, *tail, dtype=dt, device=device)
            for tail, dt in (
                ((kda.HD,), bf16),
                ((kda.HD,), bf16),
                ((kda.HD,), f32),
                ((), f32),
            )
        )
    kda.kda_pre(
        packed.key,
        prefix=prefix,
        delta=None,
        blocks=blocks,
        **_weight_args(packed),
        rstate=torch.zeros(
            slots, kda.NH, kda.HD, kda.HD, dtype=torch.float32, device=device
        ),
        st_idx=(
            torch.zeros(1, rows, dtype=torch.int32, device=device)
            if ring is not None
            else torch.arange(rows, dtype=torch.int32, device=device).view(1, rows)
        ),
        ring=ring,
        num_acc=torch.ones(1, dtype=torch.int32, device=device),
        # an init_committed build reads its initial state from these: a stand-in
        # pool slot 0 (a null pointer here faults the warmup at address 0)
        rstate0=(
            torch.zeros(1, kda.NH, kda.HD, kda.HD, dtype=torch.float32, device=device)
            if packed.key.init_committed
            else None
        ),
        init_slot=(
            torch.zeros(1, dtype=torch.int32, device=device)
            if packed.key.init_committed
            else None
        ),
        core_out=torch.empty(rows, kda.PROJ, dtype=torch.bfloat16, device=device),
        scratch=runner.scratch,
        epoch=runner.epoch,
        layer=packed.i,
    )


def prepare(batch: Optional[K1Batch], forward_batch) -> Optional[dict]:
    """The step's KDA state, and the conv buffer staged for every K1 layer.

    Called once before the layer loop. The pools are laid out layer-major
    (``State.at_layer_idx`` is ``v[layer]``), so every layer's committed window
    is one indexed read and the whole buffer is one write -- the per-layer form
    cost six dispatches a layer for 912 KiB.

    Staging here is safe for a layer that then declines: the buffer is K1's own,
    and nothing else reads it.

    None when this step has no speculative state to hand over.
    """
    from sglang.srt.layers.attention.linear.utils import (
        select_verify_intermediate_state_indices,
    )
    from sglang.srt.model_executor.forward_context import get_attn_backend

    if batch is None:
        return None
    backend = getattr(get_attn_backend(), "linear_attn_backend", None)
    pool = getattr(backend, "req_to_token_pool", None)
    metadata = getattr(backend, "forward_metadata", None)
    spec_info = getattr(forward_batch, "spec_info", None)
    if pool is None or metadata is None or spec_info is None:
        return None
    cache = pool.mamba_pool.mamba_cache  # every field [num_layers, ...]
    inter_ssm = getattr(cache, "intermediate_ssm", None)
    windows = getattr(cache, "intermediate_conv_window", None)
    ring = getattr(cache, "replayssm_rawv", None) is not None and inter_ssm is None
    if (inter_ssm is None and not ring) or not windows:
        return None
    if batch.idx is None:
        # First call, which is an eager warmup forward: the pool indices and
        # the tensor that gathers them, once. A capture would otherwise take
        # the host-to-device copy of this constant with it.
        for k in batch.layers:
            k.mamba_idx = pool.mamba2_layer_index(k.module.self_attn.attn.layer_id)
        batch.idx = torch.tensor(
            [k.mamba_idx for k in batch.layers],
            dtype=torch.int64,
            device=cache.temporal.device,
        )
    num_reqs = forward_batch.req_pool_indices.shape[0]
    slots = metadata.mamba_cache_indices[:num_reqs]
    rows = select_verify_intermediate_state_indices(
        backend.verify_intermediate_state_indices,
        forward_batch.req_pool_indices,
        slots >= 0,
        pool.size,
    )
    st = {
        "committed_conv": cache.conv[0],
        "committed_ssm": cache.temporal,
        "inter_ssm": inter_ssm,
        "ring": (
            (
                cache.replayssm_rawk,
                cache.replayssm_rawv,
                cache.replayssm_g,
                cache.replayssm_beta,
            )
            if ring
            else None
        ),
        "inter_window": windows[0],
        "slots": slots,
        "rows": rows,
        "num_draft": spec_info.draft_token_num,
        "idx": batch.idx,
    }
    if not _step_shapes_ok(batch, st):
        return None
    # One request a launch, so one slot and one state row for the whole step.
    slot = slots[:1].long()
    row = rows[:1].long()
    st["slot"], st["row"] = slot, row
    st["slot_i32"] = slot.to(torch.int32)
    qlen = st["num_draft"]
    if batch.ramp is None:
        # Constants of the width, not of the step. Build them here, not at
        # every step: a capture would take the build with it and replay it.
        # acc_idx = 0 is what pins num_acc at 1. See the module docstring.
        batch.ramp = torch.arange(qlen, device=slots.device, dtype=torch.long).view(
            1, qlen
        )
        batch.ones = torch.ones(1, dtype=torch.int32, device=slots.device)
    if ring:
        # The committed state is read at the request's slot and never written,
        # so every token's index is that slot (the kernel stores no state).
        st["st_idx"] = slot.view(1, 1).expand(1, qlen)
    else:
        st["st_idx"] = (row * qlen).view(1, 1) + batch.ramp
    st["st_idx_i32"] = st["st_idx"].to(torch.int32).contiguous()
    st["num_acc"] = batch.ones
    win = batch.conv.shape[-1] - qlen
    # [n, slots, win, C] -> these layers -> this slot -> [n, win, C] -> [n, C, win].
    # Every index stays a tensor: reading one out would be a host sync, which a
    # capture rejects, and this runs inside a captured decode graph.
    if _conv_triton():
        cm, cv = st["committed_conv"], batch.conv
        n, chans = cv.shape[0], cv.shape[1]
        _stage_window[(n, triton.cdiv(chans, _CONV_BLOCK))](
            cm, cm.stride(0), cm.stride(1), cm.stride(2), cm.stride(3),
            batch.idx, slot, cv, cv.stride(0), cv.stride(1), cv.stride(2), chans,
            WIN=win, BLOCK=_CONV_BLOCK,
        )  # fmt: skip
        st["window"] = None
    else:
        window = st["committed_conv"].index_select(0, batch.idx).index_select(1, slot)
        window = window.squeeze(1)
        batch.conv[:, :, :win] = window.mT
        st["window"] = window
    # The SSM staging copies the committed pool's state into the speculative
    # intermediate_ssm scratch. SGLang's own fused-accept staging makes the
    # same copy. It writes a production buffer, so only the layers that launch
    # may make it. The step that built the hand-off mask names those layers.
    # Until then every layer stages its own in launch().
    st["staged"] = ring or batch.where_key is not None
    if st["staged"] and not ring and not batch.layers[0].key.init_committed:
        taken = batch.idx if batch.where is None else batch.idx[batch.where]
        st["inter_ssm"][taken, row, 0] = st["committed_ssm"][taken, slot]
    return st


def _step_shapes_ok(batch: K1Batch, st: dict) -> bool:
    """What prepare needs before it writes anything."""
    from sgl_amd_mono.kimi_k3.gfx950.attention import kda

    qlen = st["num_draft"]
    key = batch.layers[0].key
    ring = st["ring"]
    if ring is None:
        state_ok = (
            st["inter_ssm"].dtype == torch.float32 and st["inter_ssm"].shape[2] == qlen,
            f"intermediate_ssm is {st['inter_ssm'].dtype} {tuple(st['inter_ssm'].shape)}",
        )
    else:
        rk, rv, rg, rb = ring
        state_ok = (
            rk.dtype == rv.dtype == torch.bfloat16
            and rg.dtype == rb.dtype == torch.float32
            and rk.shape[3] >= qlen
            and st["committed_ssm"].dtype == torch.float32,
            f"the ReplaySSM ring is k {rk.dtype} {tuple(rk.shape)}, g {rg.dtype}, "
            f"state {st['committed_ssm'].dtype}",
        )
    checks = [
        (st["rows"].numel() >= 1 and st["slots"].numel() >= 1, "the step has no state"),
        (0 < qlen <= MAX_ROWS, f"{qlen} drafted tokens is not 1 .. {MAX_ROWS}"),
        (qlen == key.qlen, f"{qlen} drafted tokens, built for {key.qlen}"),
        (
            key.ring == (ring is not None),
            f"built for the {'ring' if key.ring else 'snapshots'}, the pool has the other",
        ),
        state_ok,
        (
            st["committed_conv"].shape[-1] == 3 * kda.PROJ
            and st["committed_conv"].shape[-2] == kda.CONV_W - 1,
            f"the committed conv state is {tuple(st['committed_conv'].shape)}",
        ),
    ]
    for ok, reason in checks:
        if not ok:
            _say(f"declined: {reason}")
            return False
    return True


def finish(batch: Optional[K1Batch], st: Optional[dict], engaged: list) -> None:
    """Hand the step's conv inputs to SGLang's speculative window, for every
    layer that engaged, in one write.

    SGLang's own accept-time commit reads this to advance the committed state.

    This window is dense on a KDA model, so every one of its
    ``qlen x (CONV_W - 1)`` entries has to be written. SGLang's
    ``conv_window_dedup_enabled`` returns False whenever the model is KDA: the
    deduplicated layout is one overlapping ``as_strided`` view, and KDA
    transposes the window before the conv, which would corrupt its stores. Each
    draft token therefore owns its window, and no entry is shared with another
    token.

    ``unfold`` gives token ``j`` the window ``conv[..., j : j + CONV_W - 1]``.
    That is the same sliding read the overlapping layout expressed with its
    strides, written out once per token because nothing here is shared.

    Masked to the engaged layers: a layer that declined ran SGLang's attention,
    which wrote this itself.
    """
    if batch is None or st is None or not engaged:
        return
    qlen = st["num_draft"]
    win = batch.conv.shape[-1] - qlen
    row, idx = st["row"], st["idx"]
    taken = tuple(engaged)
    if batch.where_key is None:
        # First call, an eager warmup forward: which layers engage follows from
        # the layer structure, so it is the same on every step of this width.
        # Built here because a capture cannot take the host-to-device copy.
        batch.where_key = taken
        batch.where = (
            None
            if len(taken) == len(batch.layers)
            else torch.tensor(
                [n for n, k in enumerate(batch.layers) if k.i in engaged],
                dtype=torch.int64,
                device=idx.device,
            )
        )
    assert batch.where_key == taken, (
        f"K1 engaged on {len(taken)} layers, {len(batch.where_key)} when this "
        "width was captured; the hand-off mask is fixed at capture"
    )
    if st["window"] is None:
        _hand_back_triton(batch, st, qlen, win, row, idx)
        return
    if batch.where is None:
        conv, window = batch.conv, st["window"]
    else:
        idx, conv, window = (
            idx[batch.where],
            batch.conv[batch.where],
            st["window"][batch.where],
        )
    view = st["inter_window"]
    # The sequence the conv actually read, which is not what the buffer holds
    # now. The launch rolls its state left by one and appends this step's
    # projections, so conv[.., :win] is already the next step's window. The
    # window prepare() staged is the one this step convolved against.
    seq = torch.cat([window.mT, conv[:, :, win:]], dim=2)
    # What SGLang commits for token j is the state after j, which is the last
    # win entries up to and including j's own input. The conv reads
    # seq[j .. j + win] for token j, so j's input is seq[j + win] and the
    # window to hand back is seq[j + 1 .. j + win]. The last token's input is
    # the final entry of seq, so every entry is used.
    windows = seq.unfold(2, win, 1)[:, :, 1 : qlen + 1]
    # row stays a tensor and broadcasts against idx: reading it out would be a
    # host sync, which the capture this runs in rejects.
    view[idx, row] = windows.permute(0, 2, 3, 1)


def _hand_back_triton(batch, st, qlen, win, row, idx) -> None:
    """finish()'s hand-back in one launch (prepare staged no window snapshot:
    the committed window is unchanged until the step's accept, so it is read
    from the pool again)."""
    if batch.bidx is None:
        # the engaged layers' rows of the K1 buffer and their pool indices: fixed
        # from the first (eager) call on, so a capture takes no index op with it
        n = len(batch.layers)
        batch.bidx = (
            torch.arange(n, dtype=torch.int64, device=idx.device)
            if batch.where is None
            else batch.where
        )
        batch.lidx = idx if batch.where is None else idx[batch.where]
    lidx = batch.lidx
    cm, cv, view = st["committed_conv"], batch.conv, st["inter_window"]
    chans = cv.shape[1]
    _hand_back[(lidx.shape[0], triton.cdiv(chans, _CONV_BLOCK))](
        cm, cm.stride(0), cm.stride(1), cm.stride(2), cm.stride(3),
        cv, cv.stride(0), cv.stride(1), cv.stride(2),
        view, view.stride(0), view.stride(1), view.stride(2), view.stride(3), view.stride(4),
        lidx, batch.bidx, st["slot"], row, chans,
        WIN=win, QLEN=qlen, BLOCK=_CONV_BLOCK,
    )  # fmt: skip


def covered(packed: K1Layer, prefix, attn_res, state) -> bool:
    """Fail closed, like every launch here: the kernel reads raw addresses, so
    a wrong dtype or stride is read, not rejected.

    The first decision either way is logged with its reason. A launch that is
    built and then never reached costs nothing and reads exactly like a launch
    that is not helping, so the log is what tells a measurement from a bug.
    """
    if state is None:
        _say("declined: this step has no speculative state")
        return False
    if attn_res is None:
        _say("declined: the attention-residual stream is not open")
        return False
    if not isinstance(prefix, torch.Tensor):
        # An owed-output handle: the previous stage's contribution is still in
        # flight and the boundary's prepare is what completes it. K1 replaces
        # that prepare, so it cannot take the layer that has one to complete.
        # Only the first covered layer does -- every covered layer hands the
        # next one a written tensor -- so this costs one layer a step.
        _say("declined: the residual stream owes a contribution")
        return False
    qlen = state["num_draft"]
    bank = attn_res.block_residual
    # The step-wide shapes were taken in prepare(); these are this layer's.
    refusals = [
        (prefix.size(0) == qlen, f"{prefix.size(0)} rows for {qlen} tokens"),
        (prefix.size(1) == HIDDEN, f"the prefix is {prefix.size(1)} wide"),
        (prefix.dtype == torch.bfloat16, f"the prefix is {prefix.dtype}"),
        (prefix.is_contiguous(), "the prefix is not contiguous"),
        (
            attn_res.num_valid_blocks == packed.key.nblocks,
            f"{attn_res.num_valid_blocks} bank rows, built for {packed.key.nblocks}",
        ),
        (bank.dtype == torch.bfloat16, f"the bank is {bank.dtype}"),
        (
            bank.size(2) == HIDDEN and bank.stride(2) == 1,
            "the bank rows are not contiguous hidden vectors",
        ),
    ]
    for ok, reason in refusals:
        if not ok:
            _say(f"declined: {reason}")
            return False
    _say("engaged on a decode step")
    return True


def launch(runner, packed: K1Layer, prefix, attn_res, state) -> torch.Tensor:
    """Run K1 and return this layer's gated core output.

    Every index below stays on the device. Reading one out with ``int()`` is a
    host sync, which a CUDA graph capture rejects, and this runs inside a
    captured decode graph.
    """
    from sgl_amd_mono.kimi_k3.gfx950.attention import kda

    qlen = packed.key.qlen
    device = prefix.device
    # prepare() filled the conv window for every layer, and the SSM scratch
    # for the layers that launch once it knew which those are. Until then this
    # layer stages its own, which is the same copy over one layer.
    mi, ring = packed.mamba_idx, None
    if state["ring"] is not None:
        rstate = state["committed_ssm"][mi]
        ring = tuple(t[mi] for t in state["ring"])
    else:
        inter = state["inter_ssm"]
        rstate = inter[mi].reshape(-1, *inter.shape[3:])
    if not state["staged"] and not packed.key.init_committed:
        rstate.index_copy_(
            0,
            state["st_idx"][:, 0],
            state["committed_ssm"][packed.mamba_idx].index_select(0, state["slot"]),
        )

    core = torch.empty(qlen, kda.PROJ, dtype=torch.bfloat16, device=device)
    kda.kda_pre(
        packed.key,
        prefix=prefix,
        delta=None,
        blocks=attn_res.block_residual,
        **_weight_args(packed),
        rstate=rstate,
        st_idx=state["st_idx_i32"],
        num_acc=state["num_acc"],
        ring=ring,
        rstate0=(
            state["committed_ssm"][packed.mamba_idx]
            if packed.key.init_committed
            else None
        ),
        init_slot=state["slot_i32"] if packed.key.init_committed else None,
        core_out=core,
        scratch=runner.scratch,
        epoch=runner.epoch,
        # K2 takes slot 128 + i. The slots below it are free for exactly one
        # more launch a layer. A tag only has to tell apart the pairs live at
        # one address. These two differ by slot, so the host-bumped epoch
        # still carries the step. See gfx950/tag.py.
        layer=packed.i,
    )
    # The hand-off to the conv window is finish()'s, once for every layer that engaged.
    if packed.key.write_idx >= 0:
        # The kernel wrote the bank row that AttnResidual.write would have.
        # write_idx is the count it was at. The count itself is the boundary's
        # host-side bookkeeping, and K1 replaced the boundary. K2 reads the
        # count to size its own seam. It refuses a bank it did not expect.
        attn_res._count_write(None)
    return core
