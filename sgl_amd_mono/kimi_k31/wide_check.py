# SPDX-License-Identifier: Apache-2.0
"""Eager check of the wide-row KDA layer launch on a served model's real weights,
activations and KDA state (no routing: SGLang's own path serves every step).

``SGLANG_AMD_MONO_K31_WIDE_CHECK=<layer>`` (0-based, a KDA layer that is not a
block-write layer) with ``--disable-cuda-graph``. On every verify step whose rows
are R x num_draft with a configured width, at that layer:

* attention seam (``AttnResidual.forward`` with the layer's attention weights):
  its bank and the prefix it returns are this launch's inputs (``t1=False``:
  the MoE tail of the previous layer is not part of the check); the launch runs
  right there, before SGLang's attention touches the KDA state, on copies of the
  ring rows, SGLang's conv state (read only) and its own window rows, so SGLang's
  state is never written;
* the references, as SGLang computes them: the seam's normed x, o_proj's input
  (core_out), the MLP seam's normed x' and prefix, top-k, latent down and the
  shared experts' output (the hooks that fire on this path), and after the layer
  the verify conv windows SGLang's attention wrote.

After the layer, rank 0 prints the worst rel L2 over the ranks per hand-off.
"""

import os

import torch
import torch.distributed as dist

# The SGLang names this module reads (test/unit/test_contract.py).
SGLANG_READS = {
    "srt/runtime_context.py": ("get_parallel",),
    "srt/model_executor/forward_context.py": ("get_attn_backend",),
    "srt/layers/attention/linear/utils.py": (
        "select_verify_intermediate_state_indices",
    ),
    "srt/layers/attn_residual.py": ("AttnResidual",),
}

TP, H, NF, E = 8, 7168, 2096, 896


def _say(msg):
    if dist.get_rank() == 0:
        print(f"[k31 wide check] {msg}", flush=True)


def _rel(a, b):
    a, b = a.float(), b.float()
    e = ((a - b).norm() / b.norm().clamp_min(1e-30)).item()
    return e if e == e else float("inf")


class WideCheck:
    def __init__(self, model, layer_i: int, rank: int, group, device):
        from sgl_amd_mono.kimi_k31.gfx950 import pre_attn as pa

        lm = getattr(model, "language_model", model)
        self.backbone = lm.model
        self.layer = layer = self.backbone.layers[layer_i]
        self.i, self.rank, self.group, self.device = layer_i, rank, group, device
        attn, moe = layer.self_attn, layer.mlp
        assert hasattr(attn, "qkv_conv1d"), f"layer {layer_i} is not a KDA layer"
        assert not layer.is_block_write_layer, f"layer {layer_i} is a block-write layer"
        sl = H // TP
        r = rank
        bf = torch.bfloat16
        self.w = dict(
            w_ip=attn._qkvgbfa_layer.weight,
            cw_a=(
                layer.self_attention_res_norm.weight.float()
                * layer.self_attention_res_proj.weight.view(H).float()
            ).contiguous(),
            w_in=layer.input_layernorm.weight,
            wo=attn.o_proj.weight,
            cw_m=(
                layer.mlp_res_norm.weight.float()
                * layer.mlp_res_proj.weight.view(H).float()
            ).contiguous(),
            w_ln=layer.post_attention_layernorm.weight,
            wf=torch.cat(
                [
                    moe.gate.weight[r * 112 : (r + 1) * 112].to(bf),
                    moe.routed_expert_down_proj.weight[r * 448 : (r + 1) * 448].to(bf),
                    moe.shared_experts.gate_up_proj.weight.to(bf),
                ]
            ).contiguous(),
            bias=moe.gate.e_score_correction_bias.float().contiguous(),
            w_sd=moe.shared_experts.down_proj.weight,
        )
        prev = self.backbone.layers[layer_i - 1].mlp
        self.prev = prev
        self.w_up = pa.fold_up(
            prev.routed_expert_up_proj.weight, prev.routed_expert_norm.weight, rank
        )
        self.eps_n = prev.routed_expert_norm.variance_epsilon
        self.kda_w = dict(
            w_fb=attn.f_b_proj.weight,
            conv_w=attn.qkv_conv1d.weight.squeeze(1),
            a_log=attn.A_log.detach().reshape(-1).contiguous(),
            dt_bias=attn.dt_bias,
            on_w=attn.o_norm.weight.data.to(bf).contiguous(),
        )
        eps = {
            layer.self_attention_res_norm.variance_epsilon,
            layer.input_layernorm.variance_epsilon,
            layer.mlp_res_norm.variance_epsilon,
            layer.post_attention_layernorm.variance_epsilon,
        }
        assert len(eps) == 1, f"seam eps differ: {eps}"
        self.eps = eps.pop()
        self.onorm_eps = float(attn.o_norm.eps)
        self.nvb_a = layer.prev_valid_blocks
        _say(
            f"layer {layer_i}: nvb {self.nvb_a}, eps {self.eps}, w_ip {tuple(self.w['w_ip'].shape)}, "
            f"front {tuple(self.w['wf'].shape)} gate dtype {moe.gate.weight.dtype}, conv_w "
            f"{self.kda_w['conv_w'].dtype}"
        )
        self.pa = pa
        self.peers = {}
        self.epoch = None
        self.ref = {}
        self.ours = None
        self.stats = []
        self.fb = None
        self._hooks()

    # ---------------------------------------------------------------- hooks
    def _hooks(self):
        import sglang.srt.layers.attn_residual as attn_residual

        chk = self
        orig = attn_residual.AttnResidual.forward

        def forward(
            res,
            hidden_states,
            prefix_sum,
            score_proj,
            score_norm,
            out_norm,
            rows=None,
            write=False,
        ):
            seam = (
                "attn"
                if score_proj is chk.layer.self_attention_res_proj
                else "mlp"
                if score_proj is chk.layer.mlp_res_proj
                else None
            )
            bank, nvb = None, 0
            if seam == "attn":
                nvb = res.num_valid_blocks
                bank = res.block_residual[:, :nvb].clone() if nvb else None
            normed, prefix = orig(
                res,
                hidden_states,
                prefix_sum,
                score_proj,
                score_norm,
                out_norm,
                rows,
                write,
            )
            if seam == "attn":
                chk.ref.update(
                    x=normed.clone(), mid=prefix.clone(), nvb=nvb, write=write
                )
                chk._run(prefix, bank)
            elif seam == "mlp":
                chk.ref.update(xm=normed.clone(), h=prefix.clone())
            return normed, prefix

        attn_residual.AttnResidual.forward = forward
        moe = self.layer.mlp

        def grab(name):
            def hook(_m, _inp, out):
                if self.ours is not None:
                    self.ref[name] = out[0] if isinstance(out, tuple) else out

            return hook

        def grab_in(name):
            def hook(_m, inp):
                if self.ours is not None:
                    self.ref[name] = inp[0].clone()

            return hook

        self.layer.self_attn.o_proj.register_forward_pre_hook(grab_in("core"))

        # the merged-front path calls neither routed_expert_down_proj nor the shared
        # MLP module: the experts' own inputs are the latent and the routing
        def experts_in(_m, inp):
            if self.ours is not None and len(inp) >= 2:
                self.ref["latent"], self.ref["topk"] = inp[0].clone(), inp[1]

        moe.experts.register_forward_pre_hook(experts_in)

        def wrap_shared(mod, store):
            # the new dev_amd's fp32 front casts the routed input in the shared launch
            # (_forward_shared_casting_routed); otherwise _forward_shared
            for name in ("_forward_shared", "_forward_shared_casting_routed"):
                orig = getattr(mod, name)

                def fs(gate_up, shared_output, *a, _orig=orig, **k):
                    out = _orig(gate_up, shared_output, *a, **k)
                    store(shared_output.clone())
                    return out

                setattr(mod, name, fs)

        def wrap_routed(mod, store):
            orig_fr = mod._forward_routed

            def fr(hidden_states, router_logits, routed_input, latent, *a, **k):
                out = orig_fr(
                    hidden_states, router_logits, routed_input, latent, *a, **k
                )
                store(latent.clone())
                return out

            mod._forward_routed = fr

        def wrap_fused(mod):
            orig_ff = mod._forward_fused

            def ff(hidden_states, *a, prefix_sum=None, **k):
                self.t1_in = {
                    "prefix": None if prefix_sum is None else prefix_sum.clone()
                }
                return orig_ff(hidden_states, *a, prefix_sum=prefix_sum, **k)

            mod._forward_fused = ff

        def keep_t1(name):
            def store(t):
                if getattr(self, "t1_in", None) is not None:
                    self.t1_in[name] = t

            return store

        def keep_ref(t):
            if self.ours is not None:
                self.ref["shared"] = t

        wrap_shared(moe, keep_ref)
        wrap_fused(self.prev)
        wrap_shared(self.prev, keep_t1("sh_p"))
        wrap_routed(self.prev, keep_t1("lat_p"))
        self.layer.register_forward_hook(lambda *_: self._compare())

        def keep_batch(_m, args, kwargs):
            self.fb = kwargs.get("forward_batch", args[2] if len(args) > 2 else None)

        self.layer.register_forward_pre_hook(keep_batch, with_kwargs=True)

    # ---------------------------------------------------------------- launch
    @staticmethod
    def _backend():
        from sglang.srt.model_executor.forward_context import get_attn_backend

        return get_attn_backend().linear_attn_backend

    def _state(self, rows):
        """The step's KDA state, as K1's prepare reads it, for every request."""
        from sgl_amd_mono.kimi_k31.gfx950 import kda_defs as k1

        backend = self._backend()
        pool = backend.req_to_token_pool
        meta = backend.forward_metadata
        cache = pool.mamba_pool.mamba_cache
        idx = pool.mamba2_layer_index(self.layer.self_attn.attn.layer_id)
        return dict(cache=cache, idx=idx, meta=meta, k1=k1)

    def _run(self, mid, bank):
        from sgl_amd_mono.common import guard
        from sgl_amd_mono.common.peer import PeerBuffer
        from sgl_amd_mono.kimi_k31.gfx950 import layer as lay
        from sgl_amd_mono.kimi_k31.gfx950.chain import Gemm

        spec = getattr(self.fb, "spec_info", None)
        rows = mid.shape[0]
        L = getattr(spec, "draft_token_num", 0) if spec is not None else 0
        # serving pads to a bucket: check only steps that fill one, with that bucket's config
        from sgl_amd_mono.kimi_k31 import wide

        if L not in wide.SUPPORTED_L or rows % L or rows // L not in wide.REQ_BUCKETS:
            self.ours = None
            return
        st = self._state(rows)
        k1, cache, idx = st["k1"], st["cache"], st["idx"]
        r_n = rows // L
        slots = st["meta"].mamba_cache_indices[:r_n].long()
        dev, sl, bf = self.device, H // TP, torch.bfloat16
        cfg = lay.LayerCfg(
            *(Gemm(*(int(v) for v in c.split(":"))) for c in wide.cfg_rows(rows, L))
        )
        nvb, write = self.ref["nvb"], self.ref["write"]
        t1_in, self.t1_in = getattr(self, "t1_in", None), None
        t1 = (
            t1_in is not None
            and {"lat_p", "sh_p", "prefix"} <= set(t1_in)
            and t1_in["prefix"] is not None
            and t1_in["lat_p"].shape[0] == rows
        )
        assert self.eps_n == self.onorm_eps, (self.eps_n, self.onorm_eps)
        launch, pl = lay.kda_layer(
            rows,
            cfg,
            L,
            NF,
            nvb_a=nvb,
            nvb_m=nvb + int(write),
            write_a=write,
            nbk=8,
            eps=self.eps,
            eps_n=self.onorm_eps,
            t1=t1,
        )
        if rows not in self.peers:
            self.peers[rows] = PeerBuffer(
                pl.peer_bytes(), self.group, self.rank, TP, dev
            )
            self.peers[rows].bytes.zero_()
        if self.epoch is None:
            self.epoch = torch.zeros(guard.GUARD_WORDS, dtype=torch.int32, device=dev)
            ticks = guard.deadline_ticks(20.0)
            lo = ticks & 0xFFFFFFFF
            self.epoch[guard.GUARD_DEADLINE] = lo - (1 << 32) if lo >= 1 << 31 else lo
            self.epoch[guard.GUARD_DEADLINE + 1] = ticks >> 32
        self.epoch[0] += 1
        cols = slice(self.rank * sl, (self.rank + 1) * sl)
        # KDA state: the conv state and the committed state read only, ring copies, own window rows
        conv_all = cache.conv[0][idx]  # [slots, CONV_W - 1, C]
        win = torch.zeros(
            r_n, L, k1.CONV_W - 1, conv_all.shape[-1], dtype=bf, device=dev
        )
        win_row = torch.arange(r_n, dtype=torch.int32, device=dev)
        ring = tuple(
            t[idx].clone()
            for t in (
                cache.replayssm_rawk,
                cache.replayssm_rawv,
                cache.replayssm_g,
                cache.replayssm_beta,
            )
        )
        st_idx = slots.to(torch.int32).view(r_n, 1).expand(r_n, L).contiguous()
        num_acc = torch.ones(r_n, dtype=torch.int32, device=dev)
        core = torch.zeros(rows, k1.PROJ, dtype=bf, device=dev)
        tab = self.pa.kda_table(
            dev,
            **self.kda_w,
            conv_state=conv_all.mT,
            rstate=cache.temporal[idx],
            st_idx=st_idx,
            num_acc=num_acc,
            core_out=core,
            conv_win=win,
            win_row=win_row,
            ring=ring,
        )
        nbk = 8
        bank_s = torch.zeros(rows, nbk, sl, dtype=bf, device=dev)
        if bank is not None:
            bank_s[:, : bank.shape[1]] = bank[..., cols]
        mid_s = (
            torch.zeros(rows, sl, dtype=bf, device=dev)
            if t1
            else mid[:, cols].contiguous()
        )
        if t1:
            lat_p = t1_in["lat_p"].contiguous()
            sh_p = t1_in["sh_p"].contiguous()
            pre_s = t1_in[
                "prefix"
            ].contiguous()  # full rows: the launch reads its columns by stride H
        else:
            lat_p = sh_p = pre_s = None
        o = {
            k: torch.zeros(*shape, dtype=dt, device=dev)
            for k, shape, dt in (
                ("pout", (rows, H), bf),
                ("ip_buf", (rows, k1.NPROJ), bf),
                ("acc_u", (rows, sl), bf),
                ("acc_o", (rows * H,), bf),
                ("lat", (rows, 3584), bf),
                ("ids", (rows, 16), torch.int32),
                ("w", (rows, 16), torch.float32),
                ("sh", (rows, H), bf),
                ("lg", (rows, E), torch.float32),
            )
        }
        cnt = torch.zeros(lay.counters(rows, cfg, NF), dtype=torch.int32, device=dev)
        scratch = torch.zeros(pl.scratch_bytes(), dtype=torch.uint8, device=dev)
        z = torch.zeros(1, dtype=bf, device=dev)
        w = self.w
        ptr = lambda t: z.data_ptr() if t is None else t.data_ptr()  # noqa: E731
        col0 = 2 * cols.start
        launch(
            ptr(lat_p),
            ptr(sh_p),
            self.w_up.data_ptr(),
            ptr(pre_s) + (col0 if t1 else 0),
            mid_s.data_ptr(),
            o["pout"].data_ptr() + col0,
            bank_s.data_ptr(),
            w["cw_a"].data_ptr(),
            w["w_in"].data_ptr(),
            w["w_ip"].data_ptr(),
            o["ip_buf"].data_ptr(),
            o["acc_u"].data_ptr(),
            tab.data_ptr(),
            core.data_ptr(),
            w["wo"].data_ptr(),
            w["cw_m"].data_ptr(),
            w["w_ln"].data_ptr(),
            w["wf"].data_ptr(),
            o["acc_o"].data_ptr(),
            w["bias"].data_ptr(),
            o["lat"].data_ptr(),
            o["ids"].data_ptr(),
            o["w"].data_ptr(),
            w["w_sd"].data_ptr(),
            o["sh"].data_ptr(),
            z.data_ptr(),
            0,
            o["lg"].data_ptr(),
            cnt.data_ptr(),
            scratch.data_ptr(),
            self.peers[rows].addresses.data_ptr(),
            self.rank,
            self.epoch.data_ptr(),
            0,
            rows * H * 2 - col0 if t1 else 0,
            stream=torch.cuda.current_stream(),
        )
        torch.cuda.synchronize()
        # the gathered seam outputs, from this launch's half of the peer buffer
        offs, half = pl.peer_offsets
        par = int(self.epoch[0]) % 2
        b = self.peers[rows].bytes

        def region(name, n):
            at = par * half + offs[name]
            return b[at : at + rows * n * 2].view(bf).view(rows, n).clone()

        at = par * half + offs["rgat"]
        logits = b[at : at + rows * E * 4].view(torch.float32).view(rows, E).clone()
        from sglang.srt.layers.attention.linear.utils import (
            select_verify_intermediate_state_indices,
        )

        backend = self._backend()
        vrows = select_verify_intermediate_state_indices(
            backend.verify_intermediate_state_indices,
            self.fb.req_pool_indices,
            slots >= 0,
            backend.req_to_token_pool.size,
        )[:r_n].long()
        self.ours = dict(
            x=region("ag", H),
            xm=region("o_ag", H),
            core=core,
            h=o["pout"][:, cols],
            cols=cols,
            logits=logits,
            win=win,
            win_at=(cache.intermediate_conv_window[0][idx], vrows, slots >= 0),
            t1=t1,
            mid=mid_s,
            lat=o["lat"],
            ids=o["ids"],
            w=o["w"],
            sh=o["sh"],
            rows=rows,
            R=r_n,
            L=L,
        )

    # ---------------------------------------------------------------- compare
    def _compare(self):
        ours, ref = self.ours, self.ref
        self.ours = None
        if ours is None:
            return
        cols = ours["cols"]
        e = {
            "mid": _rel(ours["mid"], ref["mid"][:, cols]) if ours["t1"] else -1.0,
            "x": _rel(ours["x"], ref["x"]),
            "core": _rel(ours["core"], ref.get("core", ours["core"] * 0)),
            "xm": _rel(ours["xm"], ref["xm"]) if "xm" in ref else -1.0,
            "h": _rel(ours["h"], ref["h"][:, cols]) if "h" in ref else -1.0,
            "lat": _rel(ours["lat"], ref["latent"]) if "latent" in ref else -1.0,
            "sh": _rel(ours["sh"], ref["shared"]) if "shared" in ref else -1.0,
        }
        sg_win, vrows, live = ours["win_at"]
        e["win"] = (
            _rel(ours["win"][live], sg_win[vrows[live]]) if bool(live.any()) else -1.0
        )
        if "topk" in ref:
            tk = ref["topk"]
            ids = getattr(tk, "topk_ids", None)
            if ids is None and isinstance(tk, (tuple, list)) and len(tk) >= 2:
                ids = tk[1]
            if not getattr(self, "_said_topk", False):
                self._said_topk = True
                _say(
                    f"topk output: {type(tk).__name__} "
                    f"{[(f, tuple(getattr(tk, f).shape)) for f in getattr(tk, '_fields', ()) if hasattr(getattr(tk, f), 'shape')]}"
                )
            lg = getattr(tk, "router_logits", None)
            if lg is not None:
                n = ours["rows"]
                e["logits"] = _rel(ours["logits"], lg[:n])
                bias = self.w["bias"]

                def pick(x):
                    sc = torch.sigmoid(x.float())
                    return torch.topk(sc + bias, 16, dim=-1).indices.sort(-1).values

                # the same rule on SGLang's logits must give SGLang's set, and on ours ours
                e["rule_vs_sglang"] = float(
                    (pick(lg[:n]) != ids[:n].long().sort(-1).values)
                    .any(-1)
                    .float()
                    .mean()
                )
                e["ours_vs_rule"] = float(
                    (pick(ours["logits"]) != ours["ids"].long().sort(-1).values)
                    .any(-1)
                    .float()
                    .mean()
                )
            if ids is not None and ids.shape[-1] == 16:
                a = ours["ids"].long().sort(-1).values
                b2 = ids[: ours["rows"]].long().sort(-1).values
                e["topk_rows_diff"] = float((a != b2).any(-1).float().mean())
                wts = getattr(tk, "topk_weights", None)
                if wts is not None:
                    same = (a == b2).all(-1)
                    oa = ours["ids"].long().sort(-1).indices
                    ob = ids[: ours["rows"]].long().sort(-1).indices
                    if same.any():
                        e["topk_w"] = _rel(
                            ours["w"].gather(-1, oa)[same],
                            wts[: ours["rows"]].float().gather(-1, ob)[same],
                        )
        t = torch.tensor(
            [
                e.get(k, -1.0)
                for k in (
                    "x",
                    "core",
                    "xm",
                    "h",
                    "lat",
                    "sh",
                    "topk_rows_diff",
                    "topk_w",
                    "logits",
                    "rule_vs_sglang",
                    "ours_vs_rule",
                    "mid",
                    "win",
                )
            ],
            device=self.device,
        )
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        v = t.tolist()
        _say(
            f"layer {self.i} rows {ours['rows']} (R {ours['R']} x {ours['L']}) nvb {ref['nvb']} t1 {int(ours['t1'])}: "
            f"rel L2 mid (T1) {v[11]:.2e} x {v[0]:.2e} "
            f"core {v[1]:.2e} windows {v[12]:.2e} x' {v[2]:.2e} h {v[3]:.2e} latent {v[4]:.2e} shared {v[5]:.2e} topk rows "
            f"differing {v[6]:.1%} (w {v[7]:.1e}; logits {v[8]:.1e}, rule on SGLang's logits vs SGLang {v[9]:.1%}, "
            f"ours vs rule on our logits {v[10]:.1%})   (-1 = hook did not fire); fired: {sorted(k for k in ref)}"
        )
        self.ref = {}


_CHECK = None


def install(model):
    """Called from the spec's build when the env asks for it."""
    global _CHECK
    spec = os.environ.get("SGLANG_AMD_MONO_K31_WIDE_CHECK")
    if not spec or _CHECK is not None:
        return
    from sglang.srt.runtime_context import get_parallel

    p = get_parallel()
    lm = getattr(model, "language_model", model)
    device = lm.model.layers[0].input_layernorm.weight.device
    _CHECK = WideCheck(model, int(spec), p.tp_rank, p.tp_group.cpu_group, device)
