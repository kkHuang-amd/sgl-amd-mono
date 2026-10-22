# SPDX-License-Identifier: Apache-2.0
"""Laguna (proxy MoE) on gfx950: FFN-only monokernels for decode (prototype).

Each layer's attention runs SGLang's modules up to the o_proj partial; the FFN
launch takes that partial and the residual and does the attention all-reduce,
the residual add and post-attention RMSNorm, the router, the shared and routed
MXFP4 experts and the MoE all-reduce, in one launch (``gfx950/layer.py``).
"""

import os

# rows one launch serves at most (decode batch size; no speculative decoding)
MAX_ROWS = 16
# the widths routed to the plugin: 1..SERVED_ROWS (SGLANG_AMD_MONO_LAGUNA_MAX_ROWS);
# wider decode steps run SGLang's normal path, where it is faster than the launch
# (TP8, 1024/256 serving 2026-10-22, two-shot seams + S3 NT / sched_barrier / folded RQ: TPOT
# -26 / -12 / -15 / -6 / -3 % at CONC 1 / 4 / 6 / 8 / 16, so every width by default)
SERVED_ROWS = max(1, min(MAX_ROWS, int(os.environ.get("SGLANG_AMD_MONO_LAGUNA_MAX_ROWS", MAX_ROWS))))
# TP of the served model (the launch shards the experts' inter over TP, EP 1): 8 or 4
TP = int(os.environ.get("SGLANG_AMD_MONO_LAGUNA_TP", "8"))
assert TP in (4, 8), f"SGLANG_AMD_MONO_LAGUNA_TP={TP}: 8 or 4"
INTER_TOTAL = 4096  # routed and shared expert inter (all ranks)
