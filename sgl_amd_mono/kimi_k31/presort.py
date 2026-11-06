# SPDX-License-Identifier: Apache-2.0
"""Hand aiter's 2-stage MoE the routed experts' sort + stage-1 mxfp8 quant that the wide launch made
(``SGLANG_AMD_MONO_K31_WIDE_LAUNCH_SORT=1``, ``gfx950/stages/moe_front.sort_rows``), instead of SGLang's
small-sort Triton kernel (sglang/kernels/ops/moe/moe_sorting_small.py), which does the same three jobs: the
BM = 32 sort, the mxfp8 quant of the latent, and zeroing the output (the launch zeroes it in route_row).

aiter.fused_moe._moe_sorting_impl is wrapped (on top of SGLang's own wrapper); ``injected(...)`` sets the
buffers for one ``moe.experts`` call. Any mismatch (block size, shapes, an unexpected aiter path) falls back to
the wrapped sort, which is always correct."""

import contextlib
import contextvars
import functools
import inspect

import torch

# The SGLang names this module reads (test/unit/test_contract.py): the small-sort patch whose
# mx_quant_wrapper hands the stage-1 quant on through _emitted_quant.
SGLANG_READS = {
    "kernels/ops/moe/moe_sorting_small.py": (
        "_emitted_quant",
        "apply_aiter_small_moe_sort_patch",
        "mx_quant_wrapper",
    ),
}

# aiter.fused_moe._moe_sorting_impl's parameters as impl() below takes them; aiter is not covered by
# SGLANG_READS, so ensure_installed() checks them and leaves the sort to SGLang when they change.
_AITER_SORT_PARAMS = (
    "topk_ids",
    "topk_weights",
    "num_experts",
    "model_dim",
    "moebuf_dtype",
    "block_size",
    "expert_mask",
    "num_local_tokens",
    "dispatch_policy",
    "use_opus",
    "return_local_topk_ids",
    "accumulate",
    "output_aux",
    "output",
)

_INJ = contextvars.ContextVar("k31_launch_sort", default=None)
_SAID: set = set()


def _say_once(msg):
    if msg in _SAID:
        return
    _SAID.add(msg)
    import torch.distributed as dist

    if not dist.is_initialized() or dist.get_rank() == 0:
        print(f"[k31 wide] {msg}", flush=True)


def _mismatch(
    inj,
    topk_ids,
    num_experts,
    model_dim,
    block_size,
    expert_mask,
    num_local_tokens,
    flags,
    output,
):
    if int(block_size) != 32:
        return f"block_size {int(block_size)}"
    if int(num_experts) != 896 or int(model_dim) != 3584:
        return f"experts {int(num_experts)} dim {int(model_dim)}"
    if tuple(topk_ids.shape) != (inj["m"], 16) or topk_ids.data_ptr() != inj["ids_ptr"]:
        return f"topk_ids {tuple(topk_ids.shape)} not the launch's"
    if expert_mask is not None or num_local_tokens is not None:
        return "expert_mask / num_local_tokens"
    return_local, accumulate, output_aux = flags
    if return_local or not accumulate or output_aux:
        return f"return_local {return_local} accumulate {accumulate} output_aux {output_aux}"
    if (
        output is None
        or output.data_ptr() != inj["out_ptr"]
        or output.shape[0] != inj["m"]
    ):
        return "output is not the launch-zeroed MoE output"
    return None


def ensure_installed():
    """Wrap aiter's sorting entry once more if something (SGLang re-applying its patch) replaced ours.
    An aiter whose entry takes other parameters is left unwrapped: the experts then sort as SGLang does."""
    import aiter.fused_moe as fm
    from sglang.kernels.ops.moe.moe_sorting_small import _emitted_quant

    cur = fm._moe_sorting_impl
    if getattr(cur, "_k31_presort", False):
        return
    params = tuple(inspect.signature(cur).parameters)
    if params != _AITER_SORT_PARAMS:
        _say_once(f"launch sort off: aiter _moe_sorting_impl takes {params}")
        return

    @functools.wraps(cur)
    def impl(
        topk_ids,
        topk_weights,
        num_experts,
        model_dim,
        moebuf_dtype,
        block_size,
        expert_mask,
        num_local_tokens,
        dispatch_policy,
        use_opus,
        return_local_topk_ids=False,
        accumulate=True,
        output_aux=False,
        **kw,
    ):
        inj = _INJ.get()
        if inj is not None:
            out = kw.get("output")
            why = _mismatch(
                inj,
                topk_ids,
                num_experts,
                model_dim,
                block_size,
                expert_mask,
                num_local_tokens,
                (return_local_topk_ids, accumulate, output_aux),
                out,
            )
            if why is None:
                # mx_quant_wrapper (SGLang) returns this pair for the stage-1 quant (shape-checked there)
                _emitted_quant.set((inj["q"], inj["qs"].view(torch.float8_e8m0fnu)))
                inj["used"] = True
                return inj["ids"], inj["w"], inj["eids"], inj["nv"], out
            _say_once(f"launch sort skipped: {why}")
        return cur(
            topk_ids,
            topk_weights,
            num_experts,
            model_dim,
            moebuf_dtype,
            block_size,
            expert_mask,
            num_local_tokens,
            dispatch_policy,
            use_opus,
            return_local_topk_ids=return_local_topk_ids,
            accumulate=accumulate,
            output_aux=output_aux,
            **kw,
        )

    impl._k31_presort = True
    fm._moe_sorting_impl = impl
    _say_once("launch sort: aiter sorting entry wrapped")


@contextlib.contextmanager
def injected(views, m, ids, out):
    """Within: aiter's sort for this (ids, out) is the launch's ``views`` (wide's ``sortv``)."""
    ensure_installed()
    inj = dict(views, m=m, ids_ptr=ids.data_ptr(), out_ptr=out.data_ptr(), used=False)
    tok = _INJ.set(inj)
    try:
        yield inj
    finally:
        _INJ.reset(tok)
