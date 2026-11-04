# SPDX-License-Identifier: Apache-2.0
"""Device codegen through an external ROCm LLVM instead of FlyDSL's bundled one.

An alternative AMDGPU backend for this package's kernels (both FlyDSL 0.4.1's bundled
LLVM and ROCm's have miscompiled heavily spilling variants, each a different one).
FlyDSL's own external path
(``FLYDSL_COMPILE_LLVM_DIR``) hands MLIR to the external ``mlir-opt``, which does
not parse the bundled MLIR's ROCDL syntax; this hands it LLVM IR instead:
gpu.module -> LLVM IR (bundled translation) -> ``clang`` (external) -> hsaco ->
``gpu.binary``.

``install(llvm_dir)`` sets ``FLYDSL_COMPILE_LLVM_DIR`` (so FlyDSL takes its external
path and keys its cache on it) and replaces that path's codegen step. The setting is
process-wide, so only kernels compiled from this repository's code go through
``llvm_dir`` (``SGLANG_AMD_MONO_ROCM_LLVM_SCOPE=all``: every kernel); the others
(aiter's FlyDSL kernels, say) run FlyDSL's bundled codegen as before.
"""

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

_OPT = int(os.environ.get("SGLANG_AMD_MONO_ROCM_LLVM_OPT", "3"))
_SCOPE_ALL = os.environ.get("SGLANG_AMD_MONO_ROCM_LLVM_SCOPE", "mono") == "all"
_REPO = str(Path(__file__).resolve().parents[2]) + os.sep
_SELF = str(Path(__file__).resolve())
_FLYDSL = str(Path(__import__("flydsl").__file__).resolve().parent) + os.sep
_installed = None


def _from_repo():
    """Whether the kernel being compiled was called from this repository's code.

    Decided by the innermost frame outside FlyDSL (and this module): a FlyDSL kernel of SGLang / aiter that runs
    inside a mono forward (wide.backbone -> SGLang's bf16 GEMM, say) keeps the bundled codegen, whose LLVM options
    (``slp-vectorize-once-used``) ROCm's clang does not know.
    """
    f = sys._getframe(1)
    while f is not None:
        fn = f.f_code.co_filename
        if fn != _SELF and not fn.startswith(_FLYDSL):
            return fn.startswith(_REPO)
        f = f.f_back
    return False


def _codegen(
    llvm_dir,
    module,
    binary_fragment,
    *,
    llvm_options=None,
    work_dir=None,
    stage_prefix="external_binary",
):
    from flydsl._mlir import ir
    from flydsl._mlir._mlir_libs._mlirDialectsLLVM import translate_module_to_llvmir
    from flydsl.compiler import external_llvm as ext

    gm = ext._single_top_level_op(module, "gpu.module")
    name = ext._symbol_name(gm)
    chip = re.search(
        r'chip\s*=\s*"([^"]+)"', str(gm.operation.attributes["targets"])
    ).group(1)
    llir = translate_module_to_llvmir(gm.operation)
    mllvm = []
    for k, v in (llvm_options or {}).items():
        mllvm += ["-mllvm", f"-{k}={str(v).lower() if isinstance(v, bool) else v}"]
    tmp = None
    if work_dir is None:
        tmp = tempfile.TemporaryDirectory(prefix="mono_rocm_llvm_")
        work_dir = Path(tmp.name)
    try:
        work_dir.mkdir(parents=True, exist_ok=True)
        src, out = work_dir / f"{stage_prefix}.ll", work_dir / f"{stage_prefix}.hsaco"
        src.write_text(llir, encoding="utf-8")
        cmd = [
            str(Path(llvm_dir) / "bin" / "clang"),
            "-target",
            "amdgcn-amd-amdhsa",
            f"-mcpu={chip}",
            f"-O{_OPT}",
            "-nogpulib",
            "-mcode-object-version=6",
            "-Wno-override-module",
            *mllvm,
            str(src),
            "-o",
            str(out),
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        if r.returncode != 0:
            raise ext.ExternalLLVMError(
                f"ROCm LLVM codegen failed:\n{' '.join(cmd)}\n{r.stderr[-4000:]}"
            )
        blob = "".join(f"\\{b:02X}" for b in out.read_bytes())
    finally:
        if tmp is not None:
            tmp.cleanup()
    text = (
        f"module attributes {{gpu.container_module}} {{\n  gpu.binary @{name} "
        f'[#gpu.object<#rocdl.target<chip = "{chip}">, bin = "{blob}">]\n}}'
    )
    ext._replace_gpu_module_with_binary_op(
        module, ir.Module.parse(text, context=module.context)
    )


def install(llvm_dir="/opt/rocm/llvm"):
    """Route FlyDSL device codegen through ``llvm_dir``'s clang (idempotent)."""
    global _installed
    if _installed == llvm_dir:
        return
    assert (Path(llvm_dir) / "bin" / "clang").is_file(), llvm_dir
    from flydsl.compiler import external_llvm as ext

    os.environ["FLYDSL_COMPILE_LLVM_DIR"] = llvm_dir

    def run_external_binary_codegen(module, binary_fragment, **kw):
        if _SCOPE_ALL or _from_repo():
            return _codegen(llvm_dir, module, binary_fragment, **kw)
        from flydsl.compiler.jit_function import _run_pipeline

        _run_pipeline(module, [binary_fragment], verifier=False, print_after_all=False)

    ext.run_external_binary_codegen = run_external_binary_codegen
    _installed = llvm_dir
