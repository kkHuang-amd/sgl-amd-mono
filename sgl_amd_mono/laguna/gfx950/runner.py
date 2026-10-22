# SPDX-License-Identifier: Apache-2.0
"""One rank's state for Laguna's fused FFN launch (``layer.py``): the peer buffer, the
scratch, the epoch / guard words, the launches by width and every layer's weights.

``launch(i, partial, residual)`` runs layer i's FFN and returns (x, residual) for the
next layer's attention (or the final norm's output after the last layer), as SGLang's
``norm(all_reduce(mlp(post_attention_layernorm(all_reduce(partial), residual))), residual)``.
Outputs alternate between two buffers by layer parity, so a launch never writes what it
reads. ``advance_step`` moves the mailbox tags on after a step's last launch.
"""

import inspect
import os
from pathlib import Path

import torch

from sgl_amd_mono.common import guard
from sgl_amd_mono.common.peer import PeerBuffer
from sgl_amd_mono.laguna.gfx950 import layer as L
from sgl_amd_mono.laguna.gfx950 import router_tp

HIDDEN = L.HIDDEN


def compile_only(launch) -> None:
    """Compile a launch for this process without running it (FlyDSL's compile-only
    mode, for the one call); the first real call then only loads the code."""
    from flydsl.utils.env import CompileEnvManager

    names = [n for n in inspect.signature(launch.func).parameters if n != "stream"]
    var = CompileEnvManager.options["compile_only"].env_var
    old = os.environ.get(var)
    os.environ[var] = "1"
    try:
        launch(*([0] * len(names)), stream=torch.cuda.current_stream())
    finally:
        if old is None:
            os.environ.pop(var, None)
        else:
            os.environ[var] = old


def ensure_writable_flydsl_cache() -> None:
    """Point FlyDSL at a writable cache (aiter sets a read-only one)."""
    cur = os.environ.get("FLYDSL_RUNTIME_CACHE_DIR")
    if cur and os.access(cur, os.W_OK):
        return
    path = Path.home() / ".flydsl" / "cache"
    path.mkdir(parents=True, exist_ok=True)
    os.environ["FLYDSL_RUNTIME_CACHE_DIR"] = str(path)


def _need(name, t, shape, dtypes):
    if tuple(t.shape) != tuple(shape) or t.dtype not in dtypes or not t.is_contiguous():
        raise ValueError(f"Laguna FFN launch needs {name} as contiguous {dtypes} {shape}, got {t.dtype} {tuple(t.shape)}")


class LagunaFfnRuntime:
    def __init__(self, model, widths, tp: int, rank: int, group, device, eps: float, deadline_s: float):
        ensure_writable_flydsl_cache()
        m = model.model
        self.tp, self.rank, self.widths = tp, rank, sorted(widths)
        self.peer = PeerBuffer(L.peer_bytes(tp), group, rank, tp, device)
        self.peer.bytes.zero_()
        self.scratch = torch.zeros(L.SCRATCH_BYTES, dtype=torch.uint8, device=device)
        self.epoch = torch.zeros(guard.GUARD_WORDS, dtype=torch.int32, device=device)
        ticks = guard.deadline_ticks(deadline_s)
        lo = ticks & 0xFFFFFFFF
        self.epoch[guard.GUARD_DEADLINE] = lo - (1 << 32) if lo >= 1 << 31 else lo
        self.epoch[guard.GUARD_DEADLINE + 1] = ticks >> 32
        self.launches = {w: L.build_layer(tp, w, eps) for w in self.widths}
        self.x = {w: [torch.empty(w, HIDDEN, dtype=torch.bfloat16, device=device) for _ in range(2)] for w in self.widths}
        self.res = {w: [torch.empty(w, HIDDEN, dtype=torch.bfloat16, device=device) for _ in range(2)] for w in self.widths}

        el = router_tp.local_experts(tp)
        n = len(m.layers)
        self._keep, self.weights = [], []
        u8 = (torch.uint8, torch.float4_e2m1fn_x2)
        for i, lay in enumerate(m.layers):
            moe, sh = lay.mlp, lay.mlp.shared_expert
            ex = moe.experts
            rw = moe.gate.weight.data[rank * el:(rank + 1) * el].contiguous()
            rb = moe.gate.e_score_correction_bias.data
            nxt = m.layers[i + 1].input_layernorm.weight if i + 1 < n else m.norm.weight
            ts = [
                lay.post_attention_layernorm.weight.data, rw, rb,
                ex.w13_weight.data, ex.w13_weight_scale.data, ex.w2_weight.data, ex.w2_weight_scale.data,
                sh.gate_up_proj.weight.data, sh.gate_up_proj.weight_scale.data,
                sh.down_proj.weight.data, sh.down_proj.weight_scale.data, nxt.data,
            ]
            I = L.routed.INTER
            _need("post_attention_layernorm", ts[0], (HIDDEN,), (torch.bfloat16,))
            _need("router weight shard", ts[1], (el, HIDDEN), (torch.float32,))
            _need("correction bias", ts[2], (router_tp.EXPERTS,), (torch.float32,))
            _need("w13", ts[3], (router_tp.EXPERTS, 2 * I, HIDDEN // 2), u8)
            _need("w13 scale", ts[4], (router_tp.EXPERTS, 2 * I, HIDDEN // 32), u8)
            _need("w2", ts[5], (router_tp.EXPERTS, HIDDEN, I // 2), u8)
            _need("w2 scale", ts[6], (router_tp.EXPERTS, HIDDEN, I // 32), u8)
            _need("shared gate_up", ts[7], (2 * I, HIDDEN // 2), u8)
            _need("shared gate_up scale", ts[8], (2 * I, HIDDEN // 32), u8)
            _need("shared down", ts[9], (HIDDEN, I // 2), u8)
            _need("shared down scale", ts[10], (HIDDEN, I // 32), u8)
            _need("next norm", ts[11], (HIDDEN,), (torch.bfloat16,))
            self._keep.append(ts)
            self.weights.append([t.data_ptr() for t in ts])

    def precompile(self) -> None:
        # FlyDSL's disk cache hands a rank the kernels another rank built (a lock per
        # kernel), so each rank starts at its own share of the widths
        ws = self.widths
        start = len(ws) * self.rank // self.tp
        for w in ws[start:] + ws[:start]:
            compile_only(self.launches[w])

    def launch(self, i: int, partial: torch.Tensor, residual: torch.Tensor):
        w = partial.shape[0]
        _need("partial", partial, (w, HIDDEN), (torch.bfloat16,))
        _need("residual", residual, (w, HIDDEN), (torch.bfloat16,))
        x, res = self.x[w][i % 2], self.res[w][i % 2]
        nm1, rw, rb, w13, s13, w2, s2, sw13, ss13, sw2, ss2, nm2 = self.weights[i]
        self.launches[w](
            partial.data_ptr(), residual.data_ptr(), nm1, rw, rb, w13, s13, w2, s2, sw13, ss13, sw2, ss2, nm2,
            x.data_ptr(), res.data_ptr(), self.scratch.data_ptr(), self.peer.local, self.peer.addresses.data_ptr(),
            self.epoch.data_ptr(), self.rank, i, stream=torch.cuda.current_stream(),
        )
        return x, res

    def advance_step(self) -> None:
        self.epoch[:1].add_(1)

    def errors(self) -> list:
        return [int(self.epoch[i]) for i in guard.ERROR_WORDS]

    def close(self) -> None:
        self.peer.close()
