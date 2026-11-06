"""The GLM-5.3-Flash servers that the monokernel serving test launches: MI35x, TP4,
decode steps, and with ``MTP`` the NextN draft's verify steps."""

import os

MODEL = os.environ.get("GLM_53_FLASH_MODEL_PATH", "amd/GLM-5.3-Flash-Quark-MXFP4")
ENV = {"SGLANG_USE_AITER": "1", "AITER_ONLINE_TUNE": "0"}
# The cookbook's MI355X high-throughput MXFP4 cell, except: a checkpoint whose shared
# expert is MXFP4 (SGLang fuses it into the routed experts), EP 1 (the kernels shard
# every expert by TP) and a bf16 KV cache. Without --monokernel-backend, so a test can
# launch the normal path too.
ARGS = [
    "--trust-remote-code",
    "--quantization",
    "quark",
    "--tp-size",
    "4",
    "--ep-size",
    "1",
    "--mem-fraction-static",
    "0.82",
    "--max-running-requests",
    "64",
    "--cuda-graph-max-bs-decode",
    "64",
    "--chunked-prefill-size",
    "8192",
    "--max-prefill-tokens",
    "8192",
    "--dsa-prefill-backend",
    "tilelang",
    "--dsa-decode-backend",
    "tilelang",
    "--kv-cache-dtype",
    "bfloat16",
    "--moe-runner-backend",
    "aiter",
    "--reasoning-parser",
    "glm45",
    "--mm-feature-transport",
    "cpu",
]
# MTP with one draft chain of 3 tokens: 4 verify rows a request, so verify steps of 1
# to 4 requests fit the 16-row width
MTP = [
    "--speculative-algorithm",
    "EAGLE",
    "--speculative-num-steps",
    "3",
    "--speculative-eagle-topk",
    "1",
    "--speculative-num-draft-tokens",
    "4",
]
