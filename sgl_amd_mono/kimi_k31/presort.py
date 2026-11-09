# SPDX-License-Identifier: Apache-2.0
"""Hand aiter's 2-stage MoE the routed experts' sort that the wide launch made
(``SGLANG_AMD_MONO_K31_WIDE_LAUNCH_SORT=1``, ``gfx950/stages/moe_front.sort_rows``), two kinds:

"sgl"   instead of SGLang's small-sort Triton kernel (sglang/kernels/ops/moe/moe_sorting_small.py), which does
        the same three jobs: the BM = 32 sort, the mxfp8 quant of the latent, and zeroing the output (the launch
        zeroes it in route_row). aiter.fused_moe._moe_sorting_impl is wrapped (on top of SGLang's own wrapper);
        ``injected(...)`` sets the buffers for one ``moe.experts`` call. Any mismatch (block size, shapes, an
        unexpected aiter path) falls back to the wrapped sort, which is always correct.
"aux16" aiter >= d4defd456 (upstream mxmoe rows, BM = 16, stage 1 quantizes inline): through aiter's own API
        ``aiter.presorted_moe.register_presorted_moe`` (``registered(...)``), with the launch-zeroed output
        (moe_buf_zeroed). aiter checks the registration against the call and re-sorts on any mismatch."""

import contextlib
import contextvars
import functools
import importlib.util
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


# ---- "aux16": aiter's BM = 16 presort API

AUX16_KEYS = (
    "sorted_token_ids",
    "sorted_weights",
    "sorted_expert_ids",
    "num_valid_ids",
    "m_indices",
    "reverse_sorted",
)
_E, _TOPK, _LAT = 896, 16, 3584


def sort_kind():
    """The launch sort kind for this aiter: "aux16" iff it has ``aiter.presorted_moe``, else "sgl"."""
    return "aux16" if importlib.util.find_spec("aiter.presorted_moe") else "sgl"


_REG = contextvars.ContextVar("k31_launch_presort", default=None)


def _ensure_match_hook():
    """Record whether fused_moe took the registration (aiter.fused_moe calls match_presorted_moe once a call;
    a None result means it re-sorted, and aiter logs why once)."""
    import aiter.fused_moe as fm

    cur = getattr(fm, "match_presorted_moe", None)
    if cur is None or getattr(cur, "_k31_presort", False):
        return

    @functools.wraps(cur)
    def match(aux, *a, **kw):
        got = cur(aux, *a, **kw)
        reg = _REG.get()
        if reg is not None and aux is reg["aux"]:
            reg["used"] = got is not None
        return got

    match._k31_presort = True
    fm.match_presorted_moe = match


@contextlib.contextmanager
def registered(views, ids, w, out):
    """Within: the next fused_moe on this thread takes the launch's BM16 sort ``views`` (wide's ``sortv``,
    "aux16") for routing (ids, w) [m, 16] and the launch-zeroed output ``out``."""
    from aiter.presorted_moe import register_presorted_moe

    _ensure_match_hook()
    aux = {k: views[k] for k in AUX16_KEYS}
    aux.update(
        topk_ids_ptr=ids.data_ptr(),
        topk_weights_ptr=w.data_ptr(),
        M=ids.shape[0],
        block_m=16,
        num_experts=_E,
        topk=_TOPK,
        # route_row zeroed out in the same launch (captured with it under CUDA graphs)
        moe_buf_zeroed=True,
        moe_buf_ptr=out.data_ptr(),
    )
    reg = dict(aux=aux, used=False)
    tok = _REG.set(reg)
    register_presorted_moe(aux)
    try:
        yield reg
    finally:
        register_presorted_moe(None)  # fused_moe pops it; clear it if it never ran
        _REG.reset(tok)


def ref_aux16(ids, w):
    """aiter's own BM16 aux sort of (ids, w), as fused_moe would run it: dict under AUX16_KEYS."""
    import aiter.fused_moe as fm

    r = fm._adaptive_moe_sort(
        ids.contiguous(),
        w.contiguous(),
        _E,
        _TOPK,
        16,
        _LAT,
        atomic=True,
        emit_aux=True,
    )
    st, sw, se, nv, _buf, mi, rs = r
    return dict(
        sorted_token_ids=st,
        sorted_weights=sw,
        sorted_expert_ids=se,
        num_valid_ids=nv,
        m_indices=mi,
        reverse_sorted=rs,
    )


def check_aux16(views, ids, w):
    """Mismatch counts of the launch's aux16 sort ``views`` vs ``ref_aux16`` (rows below num_valid; aiter
    leaves the rows above it unwritten), plus nv (the padded row count). aiter's sort places an expert's pairs
    in atomic (run to run) order and the stages do not care, so rows are compared as a set a block-run of one
    expert (num_valid, expert ids bit for bit; ids / weights / m_indices sorted within each expert), and
    reverse_sorted by what it points at: pair (t, s)'s row holds (s << 24) | t, weight w[t, s], m_index t."""
    ref = ref_aux16(ids, w)
    nv = int(ref["num_valid_ids"][0])
    nb = nv // 16
    m, k = ids.shape
    out = dict(
        nv=nv,
        num_valid=int((views["num_valid_ids"][:2] != ref["num_valid_ids"][:2]).sum()),
        eids=int(
            (views["sorted_expert_ids"][:nb] != ref["sorted_expert_ids"][:nb]).sum()
        ),
    )
    # key a row by (its expert, its content): sorting the keys compares the sets per expert
    blk_e = ref["sorted_expert_ids"][:nb].long().repeat_interleave(16)

    def rows(v):
        st = v["sorted_token_ids"][:nv].long() & 0xFFFFFFFF
        wb = v["sorted_weights"][:nv].view(torch.int32).long() & 0xFFFFFFFF
        mi = v["m_indices"][:nv].long()
        return torch.stack([blk_e, st, wb, mi], 1)

    def canon(r):
        key = r[:, 0] * (1 << 32) + r[:, 1]  # expert major, then the id word
        return r[torch.argsort(key, stable=True)]

    a, b = canon(rows(views)), canon(rows(ref))
    out["rows"] = int((a != b).any(1).sum())
    rs = views["reverse_sorted"].long()
    p = torch.arange(m * k, device=ids.device)
    t, sl = p // k, p % k
    ok_rs = (rs >= 0) & (rs < nv)
    rsc = rs.clamp(0, max(nv - 1, 0))
    out["reverse"] = int(
        (
            ~ok_rs
            | (views["sorted_token_ids"][rsc].long() != ((sl << 24) | t))
            | (views["m_indices"][rsc].long() != t)
            | (
                views["sorted_weights"][rsc].view(torch.int32)
                != w.contiguous().view(-1).view(torch.int32)
            )
        ).sum()
    )
    return out
