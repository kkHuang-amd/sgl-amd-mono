# SPDX-License-Identifier: Apache-2.0
"""SGLang's DSpark step and prefill glue around the DeepSeek-V4.1 monokernels, in fewer
launches. Each piece wraps an SGLang function and keeps SGLang's where its first eager call
disagrees or its kernel does not build. ``SGLANG_AMD_MONO_DSV41_GLUE=0`` keeps all of SGLang's."""

from __future__ import annotations

import functools
import logging
import os

import torch

_LOGGER = logging.getLogger(__name__)

ENV = "SGLANG_AMD_MONO_DSV41_GLUE"
_INSTALLED = []
# piece -> whether it agreed with SGLang at its first eager call, on every rank
_AGREED: dict[str, bool] = {}

# by file under sglang/ (test_contract.py pins them)
SGLANG_READS = {
    "srt/layers/dp_attention.py": ("get_dp_tp_group",),
    "srt/distributed/parallel_state.py": ("world_size", "cpu_group"),
}


def flag(name: str, default: str = "1") -> bool:
    return os.environ.get(ENV, "1") != "0" and os.environ.get(name, default) != "0"


def _on_every_rank(ok: bool) -> bool:
    import torch.distributed as dist
    from sglang.srt.layers.dp_attention import get_dp_tp_group

    group = get_dp_tp_group()
    if group.world_size == 1:
        return ok
    t = torch.tensor([int(ok)], dtype=torch.int32)
    dist.all_reduce(t, op=dist.ReduceOp.MIN, group=group.cpu_group)
    return bool(t.item())


def route(owner, name: str, wrapper) -> None:
    """``owner.name`` becomes ``wrapper(orig, *args, **kwargs)``."""
    orig = getattr(owner, name)

    @functools.wraps(orig)
    def routed(*args, **kwargs):
        return wrapper(orig, *args, **kwargs)

    setattr(owner, name, routed)


def agreed(piece: str, why) -> bool | None:
    """The piece's verdict, made at its first eager call (``why()`` None: it agrees);
    None under a capture before any verdict."""
    if piece not in _AGREED:
        if torch.cuda.is_current_stream_capturing():
            return None
        try:
            reason = why()
        except Exception as e:  # noqa: BLE001 (a kernel that does not build)
            reason = f"{type(e).__name__}: {e}"
        _AGREED[piece] = _on_every_rank(reason is None)
        if reason is None and not _AGREED[piece]:
            reason = "another rank disagrees"
        _LOGGER.info("DSV4.1 glue %s: %s", piece, reason or "one launch")
    return _AGREED[piece]


def install() -> None:
    """Once a process, before the DSpark worker and the graphs are built."""
    if _INSTALLED or os.environ.get(ENV, "1") == "0":
        return
    _INSTALLED.append(True)
    from sgl_amd_mono.deepseek_v41.glue import (
        accept,
        draft_head,
        inject,
        metadata,
        prefill,
        verify_head,
    )

    for piece in (metadata, accept, inject, draft_head, prefill, verify_head):
        piece.install()


def prewarm(device) -> None:
    from sgl_amd_mono.deepseek_v41.glue import prefill

    prefill.prewarm(device)
