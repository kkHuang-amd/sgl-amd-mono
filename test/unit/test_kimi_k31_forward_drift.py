"""The Kimi-K3.1 forward that kimi_k31 reproduces for a routed step, pinned by
source digest on the SGLang Kimi-K3.1 is served with (dev_amd, e67b2062e2).

``spec._Built`` reproduces the two outer forwards; ``wide.Wide.backbone`` the
backbone, and ``Wide._tail_front`` a DSA layer's forward up to its MoE's TP
partials. The layer forward's MegaMoE and CP-shard MLP branches and the causal-LM
forward's CP branch are not taken by a routed verify step (no MoE a2a backend, CP
only on context-parallel extends).
"""

import hashlib
import inspect

import pytest
from sglang.srt.models import kimi_k3

PINNED = {
    ("KimiK3ForConditionalGeneration", "forward"): (
        "e35d07d1e2292cff",
        "spec._Built.forward",
    ),
    ("KimiK3LinearForCausalLM", "forward"): (
        "e30a51aa0f01f5a0",
        "spec._Built.language_forward",
    ),
    ("KimiK3LinearModel", "forward"): ("9a6d94921969be24", "wide.Wide.backbone"),
    ("KimiK3DecoderLayer", "forward"): ("04f140797269349c", "wide.Wide._tail_front"),
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
        f"{cls}.{method} changed ({want} -> {got}). sgl_amd_mono/kimi_k31's "
        f"{mirror} reproduces it: read the diff, update {mirror} if a routed "
        f"verify step is affected, then put {got} here."
    )
