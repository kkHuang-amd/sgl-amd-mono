"""The Kimi-K3 forward that kimi_k31 serves through kimi_k3's ``_Built``, pinned
by source digest on the SGLang Kimi-K3.1 is served with (dev_amd, 3d2aa95c9d).

Same contract as test_kimi_k3_forward_drift.py. Read against the digests that
file pins: ``_run_self_attn`` unwraps a tuple result, which ``_pre_o_proj``
does not need (``forward_pre_o_proj`` returns the core tensor); the layer
forward adds the MegaMoE and CP-shard MLP branches and the causal-LM forward a
CP branch, none of which a routed decode or verify step takes (no MoE a2a
backend, CP only on context-parallel extends).
"""

import hashlib
import inspect

import pytest
from sglang.srt.models import kimi_k3

PINNED = {
    ("KimiK3ForConditionalGeneration", "forward"): ("e35d07d1e2292cff", "_Built.forward"),
    ("KimiK3LinearForCausalLM", "forward"): ("e30a51aa0f01f5a0", "_Built.language_forward"),
    ("KimiK3LinearModel", "forward"): ("9a6d94921969be24", "_Built._backbone"),
    ("KimiK3DecoderLayer", "forward"): ("04f140797269349c", "_Built._layer"),
    ("KimiK3DecoderLayer", "_run_self_attn"): ("d0d26643498322e1", "_Built._pre_o_proj"),
}


def _digest(cls: str, method: str) -> str:
    src = inspect.getsource(getattr(getattr(kimi_k3, cls), method))
    return hashlib.sha256(src.encode()).hexdigest()[:16]


@pytest.mark.parametrize(
    ("cls", "method"), sorted(PINNED), ids=lambda p: p if isinstance(p, str) else str(p)
)
def test_the_served_forward_has_not_drifted(cls, method):
    want, mirror = PINNED[cls, method]
    got = _digest(cls, method)
    assert got == want, (
        f"{cls}.{method} changed ({want} -> {got}). sgl_amd_mono/kimi_k3/spec.py's "
        f"{mirror} reproduces it for kimi_k31: read the diff, update {mirror} if a "
        f"routed decode or verify step is affected, then put {got} here."
    )
