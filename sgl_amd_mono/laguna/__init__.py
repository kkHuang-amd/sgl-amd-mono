# SPDX-License-Identifier: Apache-2.0
"""Laguna (proxy MoE) on gfx950: FFN-only monokernels for decode (prototype).

Each layer's attention runs SGLang's modules up to the o_proj partial; the FFN
launch takes that partial and the residual and does the attention all-reduce,
the residual add and post-attention RMSNorm, the router, the shared and routed
MXFP4 experts and the MoE all-reduce. Until the kernels exist, the FFN is a
reference made of SGLang's own ops (``spec._ffn_reference``).
"""

# rows one launch serves at most (decode batch size; no speculative decoding)
MAX_ROWS = 16
TP = 8
