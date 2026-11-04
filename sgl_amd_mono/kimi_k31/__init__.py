# SPDX-License-Identifier: Apache-2.0
"""Kimi-K3.1 monokernels (RFC #43949): the wide verify-step layer launch.

A speculative verify step of R >= 2 requests x L draft tokens runs one persistent
launch a KDA layer (``wide``, ``gfx950/layer.py``); SGLang runs every other step
and the parts of a wide step the launch does not cover (``spec``).
"""

# Kimi-K3.1's shapes and routing, which the kernels build in. These live on the
# checkpoint's text config: the top-level config of the VLM wrapper carries none.
HF_CONFIG = dict(
    hidden_size=7168,
    routed_expert_hidden_size=3584,
    num_experts=896,
    num_experts_per_token=16,
    num_shared_experts=2,
    moe_router_activation_func="sigmoid",
    routed_scaling_factor=1.0,
    first_k_dense_replace=1,
    moe_layer_freq=1,
)
# The KDA layers' shapes (the launch runs their attention).
LINEAR_ATTN_CONFIG = dict(head_dim=128, num_heads=96, short_conv_kernel_size=4)
# The attention's TP-local output per rank at TP8: 12 heads of 128, o_proj's input.
CORE = 1536
# Attention-residual bank rows a layer's seams read at most (the launch's bank).
MAX_BLOCKS = 8
