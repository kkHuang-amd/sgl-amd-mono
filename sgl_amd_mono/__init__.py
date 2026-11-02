# SPDX-License-Identifier: Apache-2.0
"""Monokernels for SGLang on AMD Instinct GPUs (RFC sgl-project/sglang#43949).

Persistent kernels run most of a decoder layer, its TP collectives included, in one
or two launches. SGLang holds the interface (``sglang.kernels.plugin_kernels.amd``);
this package holds the HIP device runtime (``common``) and, per model, its specs and
their forwards (``<model>/spec.py``, ``<model>/draft.py``) and its kernels per ISA
(``<model>/<isa>/``). SGLang loads it as a general plugin (the ``sglang.srt.plugins``
entry point in ``pyproject.toml``), which registers each spec.
"""

import os

__version__ = "0.1.0.dev0"

# the backend name every spec registers under
BACKEND = "amd"
# each spec by (HF architecture, model type), None for any model type; its module
# imports only when a server runs this backend. DeepSeek-V4-Pro and V4.1 share an
# architecture in SGLang, so their specs name their model types
SPECS = {
    ("DeepseekV4ForCausalLM", "deepseek_v4"): "sgl_amd_mono.deepseek_v4.spec:SPEC",
    ("DeepseekV4ForCausalLM", "deepseek_v41"): "sgl_amd_mono.deepseek_v41.spec:SPEC",
    (
        "DeepseekV4ForCausalLMDSpark",
        "deepseek_v41",
    ): "sgl_amd_mono.deepseek_v41.draft:SPEC",
    ("Glm5NextForConditionalGeneration", None): "sgl_amd_mono.glm53_flash.spec:SPEC",
    ("GlmMoeDsaForCausalLM", None): "sgl_amd_mono.glm53.spec:SPEC",
    ("MiniMaxM3SparseForCausalLM", None): "sgl_amd_mono.minimax.spec:SPEC",
    (
        "MiniMaxM3SparseForConditionalGeneration",
        None,
    ): "sgl_amd_mono.minimax.spec:SPEC",
    ("KimiK3ForConditionalGeneration", None): "sgl_amd_mono.kimi_k3.spec:SPEC",
    ("KimiK31ForConditionalGeneration", None): "sgl_amd_mono.kimi_k31.spec:SPEC",
}

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "kernels/plugin_kernels/amd/spec.py|srt/monokernels/spec.py": (
        "register",
        "register(model_type)",
    )
}


def register_specs() -> None:
    """The plugin entry point, which SGLang runs in every process."""
    try:
        from sglang.kernels.plugin_kernels.amd.spec import register
    except ModuleNotFoundError:  # SGLang before sgl-project/sglang#45389
        from sglang.srt.monokernels.spec import register

    for (architecture, model_type), path in SPECS.items():
        register(
            backend=BACKEND, architecture=architecture, path=path, model_type=model_type
        )

    # SGLANG_AMD_MONO_ROCM_LLVM=<llvm prefix, e.g. /opt/rocm/llvm>: compile this
    # package's kernels with that LLVM instead of FlyDSL's bundled one (common/rocm_codegen.py)
    llvm_dir = os.environ.get("SGLANG_AMD_MONO_ROCM_LLVM")
    if llvm_dir:
        from sgl_amd_mono.common import rocm_codegen

        rocm_codegen.install(llvm_dir)
