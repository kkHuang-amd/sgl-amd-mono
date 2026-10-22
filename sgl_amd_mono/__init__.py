# SPDX-License-Identifier: Apache-2.0
"""Monokernels for SGLang on AMD Instinct GPUs (RFC sgl-project/sglang#43949):
persistent kernels that run most of a decoder layer, including its TP
all-reduce, in one or two launches.

SGLang holds the interface (``sglang.srt.monokernels``). This package
holds the HIP device runtime (``common``) and, per model, its specs and their
forwards (``<model>/spec.py``, ``<model>/draft.py``) and its kernels per ISA
(``<model>/<isa>/``). SGLang loads the package as a general plugin
(``pyproject.toml``: the ``sglang.srt.plugins`` entry point), which registers
each spec with SGLang's monokernel framework.
"""

__version__ = "0.1.0.dev0"

# The --monokernel-backend name this package registers
BACKEND = "amd"
# Each spec by the model's HF architecture; a spec module imports only when a
# server runs this backend
SPECS = {
    "DeepseekV4ForCausalLM": "sgl_amd_mono.deepseek_v41.spec:SPEC",
    "DeepseekV4ForCausalLMDSpark": "sgl_amd_mono.deepseek_v41.draft:SPEC",
    "Glm5NextForConditionalGeneration": "sgl_amd_mono.glm53_flash.spec:SPEC",
    "GlmMoeDsaForCausalLM": "sgl_amd_mono.glm53.spec:SPEC",
    "MiniMaxM3SparseForCausalLM": "sgl_amd_mono.minimax_m3.spec:SPEC",
    "MiniMaxM3SparseForConditionalGeneration": "sgl_amd_mono.minimax_m3.spec:SPEC",
    "LagunaForCausalLM": "sgl_amd_mono.laguna.spec:SPEC",
}

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {"srt/monokernels/spec.py": ("register",)}


def register_specs() -> None:
    """The plugin entry point, which SGLang runs in every process."""
    from sglang.srt.monokernels.spec import register

    for architecture, path in SPECS.items():
        register(backend=BACKEND, architecture=architecture, path=path)
