# SPDX-License-Identifier: Apache-2.0
"""Wide-row verify steps (R requests x L rows, R >= 2, L = the server's draft tokens,
6 or 8) on the layer launch (``gfx950/layer.py``), eager v0.

A step walks the layers as SGLang's backbone does, except for every run of covered
layers (KDA attention + latent MoE: all but layer 0, which has a dense MLP, and the
DSA layers), which go launch by launch:

    enter    after a DSA layer (MoE on the merged front): SGLang runs that layer up
             to its MoE's TP partials and the first launch runs its tail (t1 on);
             else the stream after the previous layer, folded to the full prefix.
             The bank -> this rank's column slices
    layer i  launch i (MoE tail of i - 1 unless it starts the run; restart on a
             block-write layer; DSpark capture of i - 1 when asked; reads the conv
             state from SGLang's cache by slot and writes the verify windows for
             SGLang's accept) -> a written bank row gathered into SGLang's bank
             -> SGLang's routed experts on the launch's latent and router logits
    leave    the last layer's MoE tail on SGLang (``_column_split_tail``), this rank's
             prefix columns added into its shared columns before the tail's all-reduce
             -> the stream the next standard layer reads

The column-sharded state lives only inside a run, so the standard layers see exactly
what they see today. Rows are padded to a bucket (padded requests have no KDA
state: slot -1).
"""

import os

import torch

TP, H, SL, E, NF, LAT = 8, 7168, 896, 896, 2096, 3584
REQ_BUCKETS = (2, 4, 8, 16, 32, 48)  # requests a bucket holds; its rows are L x requests
SUPPORTED_L = (6, 8)
BUCKETS = tuple(8 * r for r in REQ_BUCKETS)
CFGS = {  # 8-token bucket -> (up, in_proj, o_proj, front, shared down): bm:bn:wm:wn:ks; an L-token
    # bucket of the same requests takes the same configs
    16: ("16:128:1:8:8", "16:128:1:8:4", "16:128:1:8:4", "16:128:1:8:8", "16:128:1:8:1"),
    32: ("32:128:2:4:8", "32:128:2:4:4", "32:128:2:4:2", "32:128:2:4:8", "32:128:2:4:1"),
    64: ("64:128:2:4:8", "64:128:2:4:4", "64:64:2:4:2", "64:128:2:4:8", "64:64:2:4:1"),
    128: ("128:32:4:2:8", "128:128:2:4:4", "64:64:2:4:1", "128:128:2:4:8", "64:64:2:4:1"),
    256: ("128:32:4:2:8", "128:128:2:4:4", "128:64:2:4:1", "128:128:2:4:8", "128:64:2:4:1"),
    384: ("128:32:4:2:4", "192:128:4:2:2", "192:64:4:2:1", "128:128:2:4:4", "128:64:2:4:1"),
}
# bucket rows -> MoE front config, where the CFGS one is a poor fit: 17 column tiles x the row tiles x ks must stay
# within BLOCKS (one round), and a row tile past m is wasted work
FRONT_ROWS = {192: "64:128:2:4:4", 256: "96:128:2:4:4", 288: "96:128:2:4:4"}
NARROW = 8  # rows up to this stay on Kimi-K3's launches (S1)


def enabled() -> bool:
    return os.environ.get("SGLANG_AMD_MONO_K31_WIDE") == "1"


def tokens() -> int:
    """Rows a request takes in this server's verify steps (DSpark / DFLASH draft tokens)."""
    from sglang.srt.runtime_context import get_spec

    spec = get_spec()
    if spec.speculative_algorithm is None:
        return 1
    return int(spec.speculative_num_draft_tokens or 1)


def req_buckets() -> tuple:
    # SGLANG_AMD_MONO_K31_WIDE_BUCKETS lists rows at 8 tokens a request
    env = os.environ.get("SGLANG_AMD_MONO_K31_WIDE_BUCKETS")
    return tuple(sorted(int(x) // 8 for x in env.split(","))) if env else REQ_BUCKETS


def buckets(L: int = 8) -> tuple:
    return tuple(L * r for r in req_buckets())


def cfg_rows(b: int, L: int) -> str:
    """The CFGS entry of an L-token bucket of b rows."""
    assert b % L == 0 and b // L in REQ_BUCKETS, (b, L)
    c = CFGS[8 * (b // L)]
    f = FRONT_ROWS.get(b)
    return c if f is None else c[:3] + (f,) + c[4:]


def wide_widths(max_reqs: int = 48) -> tuple:
    # only widths a warmed bucket holds: a graph capture must not meet a cold one; rows
    # up to a bucket left out below the smallest one stay on SGLang's own path. The
    # server's draft tokens are not known at import: every supported L's widths (the
    # binding keeps those a multiple of the server's)
    rs = req_buckets()
    lo = max((x for x in REQ_BUCKETS if x < rs[0]), default=1)
    top = min(max_reqs, rs[-1])
    return tuple(sorted({L * r for L in SUPPORTED_L for r in range(max(2, lo + 1), top + 1) if L * r > NARROW}))


def _say(msg):
    import torch.distributed as dist

    if not dist.is_initialized() or dist.get_rank() == 0:
        print(f"[k31 wide] {msg}", flush=True)


class _Layer:
    """A covered layer's weights as the launch reads them (views where SGLang's
    layout already fits; the front rows and the folded up projection are copies)."""

    def __init__(self, backbone, i, rank, tails):
        from sgl_amd_mono.kimi_k31.gfx950 import pre_attn as pa

        bf = torch.bfloat16
        layer = backbone.layers[i]
        attn, moe = layer.self_attn, layer.mlp
        self.i, self.layer, self.moe = i, layer, moe
        self.t1 = (i - 1) in tails  # the previous layer's MoE tail runs in this launch
        self.write = bool(layer.is_block_write_layer)
        self.nvb = layer.prev_valid_blocks
        r = rank
        self.w = dict(
            w_ip=attn._qkvgbfa_layer.weight,
            cw_a=(layer.self_attention_res_norm.weight.float() * layer.self_attention_res_proj.weight.view(H).float())
            .contiguous(),
            w_in=layer.input_layernorm.weight,
            wo=attn.o_proj.weight,
            cw_m=(layer.mlp_res_norm.weight.float() * layer.mlp_res_proj.weight.view(H).float()).contiguous(),
            w_ln=layer.post_attention_layernorm.weight,
            wf=torch.cat([moe.gate.weight[r * 112 : (r + 1) * 112].to(bf),
                          moe.routed_expert_down_proj.weight[r * 448 : (r + 1) * 448].to(bf),
                          moe.shared_experts.gate_up_proj.weight.to(bf)]).contiguous(),
            bias=moe.gate.e_score_correction_bias.float().contiguous(),
            w_sd=moe.shared_experts.down_proj.weight,
        )
        if self.t1:
            prev = backbone.layers[i - 1].mlp
            self.w["w_up"] = pa.fold_up(prev.routed_expert_up_proj.weight, prev.routed_expert_norm.weight, r)
        self.kda_w = dict(w_fb=attn.f_b_proj.weight, conv_w=attn.qkv_conv1d.weight.squeeze(1),
                          a_log=attn.A_log.detach().reshape(-1).contiguous(), dt_bias=attn.dt_bias,
                          on_w=attn.o_norm.weight.data.to(bf).contiguous())
        self.eps = layer.input_layernorm.variance_epsilon
        self.onorm_eps = float(attn.o_norm.eps)
        self.mamba_idx = None


def _is_covered(layer) -> bool:
    from sglang.srt.models.kimi_k3 import KimiK3MoE

    return hasattr(layer.self_attn, "qkv_conv1d") and isinstance(layer.mlp, KimiK3MoE)


def _tail_fusable(layer) -> bool:
    """A standard layer (DSA) whose MoE tail the next run's first launch can take:
    a stream-writing latent MoE on the merged front."""
    from sglang.srt.models.kimi_k3 import KimiK3MoE

    mlp = layer.mlp
    return (isinstance(mlp, KimiK3MoE) and layer._ffn_writes_stream and mlp._eligible_for_fused_front
            and not getattr(layer, "_sp_moe", False) and not getattr(mlp, "_use_mega_moe", False))


def _check_steps() -> int:
    return int(os.environ.get("SGLANG_AMD_MONO_K31_WIDE_ENTRY_CHECK", "0"))


class Wide:
    def __init__(self, model, rank, group, device, L=8):
        from sgl_amd_mono.common import guard

        assert L in SUPPORTED_L, L
        self.L = L
        lm = getattr(model, "language_model", model)
        self.bb = bb = lm.model
        self.rank, self.group, self.device = rank, group, device
        idx = [i for i in range(bb.start_layer, bb.end_layer) if _is_covered(bb.layers[i])]
        cov = set(idx)
        # standard layers whose MoE tail runs in the next run's first launch (t1 on):
        # SGLang runs them up to the routed / shared partials only
        self.entry = set()
        if os.environ.get("SGLANG_AMD_MONO_K31_WIDE_ENTRY_T1", "1") == "1":
            self.entry = {i - 1 for i in idx if (i - 1) not in cov and i - 1 >= bb.start_layer
                          and _tail_fusable(bb.layers[i - 1])}
        self.layers = {i: _Layer(bb, i, rank, cov | self.entry) for i in idx}
        self.check_left = _check_steps()
        self.column_exit = os.environ.get("SGLANG_AMD_MONO_K31_WIDE_COLUMN_EXIT", "1") == "1"
        self.slots = {i: n for n, i in enumerate(idx)}  # a launch's slot in the step
        caps = set(bb.dspark_layers_to_capture or ())
        self.peers, self.bufs, self.half = {}, {}, {}
        self.buckets = buckets(L)
        self.tabs = {}  # (bucket, layer) -> KDA table: every pointer in it is fixed once built
        self.epoch = torch.zeros(guard.GUARD_WORDS, dtype=torch.int32, device=device)
        # long enough for a capture variant's first compile on one rank while the
        # others already wait (every other variant is compiled by warmup())
        ticks = guard.deadline_ticks(float(os.environ.get("SGLANG_AMD_MONO_K31_WIDE_DEADLINE_S", "900")))
        lo = ticks & 0xFFFFFFFF
        self.epoch[guard.GUARD_DEADLINE] = lo - (1 << 32) if lo >= 1 << 31 else lo
        self.epoch[guard.GUARD_DEADLINE + 1] = ticks >> 32
        self.cols = slice(rank * SL, (rank + 1) * SL)
        self.kda_ready = False
        _say(f"{len(idx)} covered layers ({idx[0]} .. {idx[-1]}), {L} tokens a request, buckets {self.buckets}, "
             f"captures {sorted(caps)}, "
             f"block-write {[i for i in idx if self.layers[i].write]}, tails into a run's first launch "
             f"{sorted(self.entry)}")

    # ------------------------------------------------------------ per bucket
    def _cfg(self, b):
        from sgl_amd_mono.kimi_k31.gfx950 import layer as lay
        from sgl_amd_mono.kimi_k31.gfx950.chain import Gemm

        return lay.LayerCfg(*(Gemm(*(int(v) for v in c.split(":"))) for c in cfg_rows(b, self.L)))

    def _variant(self, pk, b, L, captures):
        cap = (pk.i - 1) in captures
        return dict(nvb_a=pk.nvb, nvb_m=pk.nvb + int(pk.write), write_a=pk.write, capture=cap, t1=pk.t1,
                    restart=pk.write)

    def _ensure(self, b, L, captures):
        """Peer buffer and activation buffers of bucket b (collective on first use:
        every rank reaches it on the same step)."""
        if b in self.bufs:
            return self.bufs[b]
        from sgl_amd_mono.common.peer import PeerBuffer
        from sgl_amd_mono.kimi_k31.gfx950 import layer as lay

        cfg = self._cfg(b)
        half, scratch = 0, 0
        for pk in self.layers.values():
            v = self._variant(pk, b, L, captures)
            for cap in (False, True):  # sized for a capture launch too: the captured layers come later
                pl = lay.plan(b, cfg, L, v["nvb_a"], v["nvb_m"], NF, cap)[0]
                half, scratch = max(half, pl.peer_offsets[1]), max(scratch, pl.scratch_bytes())
        from sgl_amd_mono.common import guard

        self.half[b] = half  # one layout for every variant: same halves, the guard's flags after both
        need = 2 * half + guard.FLAG_BYTES
        self.peers[b] = PeerBuffer(need, self.group, self.rank, TP, self.device)
        self.peers[b].bytes.zero_()
        dev, bf = self.device, torch.bfloat16

        def z(*shape, dt=bf):
            return torch.zeros(*shape, dtype=dt, device=dev)

        from sgl_amd_mono.kimi_k3.gfx950.attention import kda as k1

        # the routed latent and shared partials of a ping share one buffer: the column exit
        # reduces the pair in one collective, as SGLang's fused front does
        flat = [z(b * (LAT + H)), z(b * (LAT + H))]
        bufs = dict(
            cfg=cfg, mid=z(b, SL), bank=z(b, 8, SL), pout=[z(b, SL), z(b, SL)], lat=z(b, LAT), flat=flat,
            lat_p=[f[: b * LAT].view(b, LAT) for f in flat], sh=[f[b * LAT :].view(b, H) for f in flat],
            pfull=z(b, H), ids=z(b, 16, dt=torch.int32), w=z(b, 16, dt=torch.float32),
            logits=z(b, E, dt=torch.float32), cap=z(b, H), core=z(b, k1.PROJ), ip_buf=z(b, k1.NPROJ),
            acc_u=z(b, SL), acc_o=z(b * H), cnt=torch.zeros(lay.counters(b, cfg, NF), dtype=torch.int32, device=dev),
            scratch=torch.zeros(scratch, dtype=torch.uint8, device=dev),
            st_idx=torch.full((b // L, L), -1, dtype=torch.int32, device=dev),
            num_acc=torch.ones(b // L, dtype=torch.int32, device=dev),
            win_row=torch.full((b // L,), -1, dtype=torch.int32, device=dev))
        self.bufs[b] = bufs
        _say(f"bucket {b}: peer {need / 1e6:.1f} MB, scratch {scratch / 1e6:.1f} MB")
        return bufs

    def _launch_args(self, pk, b, bufs, tab, ping, slot):
        w, zt = pk.w, bufs["mid"]
        prev, cur = ping, 1 - ping
        return (bufs["lat_p"][prev].data_ptr(), bufs["sh"][prev].data_ptr(), w.get("w_up", zt).data_ptr(),
                bufs["pout"][prev].data_ptr(), bufs["mid"].data_ptr(), bufs["pout"][cur].data_ptr(),
                bufs["bank"].data_ptr(), w["cw_a"].data_ptr(), w["w_in"].data_ptr(), w["w_ip"].data_ptr(),
                bufs["ip_buf"].data_ptr(), bufs["acc_u"].data_ptr(), tab.data_ptr(), bufs["core"].data_ptr(),
                w["wo"].data_ptr(), w["cw_m"].data_ptr(), w["w_ln"].data_ptr(), w["wf"].data_ptr(),
                bufs["acc_o"].data_ptr(), w["bias"].data_ptr(), bufs["lat"].data_ptr(), bufs["ids"].data_ptr(),
                bufs["w"].data_ptr(), w["w_sd"].data_ptr(), bufs["sh"][cur].data_ptr(), bufs["cap"].data_ptr(), H,
                bufs["logits"].data_ptr(), bufs["cnt"].data_ptr(), bufs["scratch"].data_ptr(),
                self.peers[b].addresses.data_ptr(), self.rank, self.epoch.data_ptr(), slot)

    def _build_launch(self, pk, b, L, v, cfg):
        from sgl_amd_mono.kimi_k31.gfx950 import layer as lay

        return lay.kda_layer(b, cfg, L, NF, nvb_a=v["nvb_a"], nvb_m=v["nvb_m"], write_a=v["write_a"], nbk=8,
                             launches=len(self.layers), eps=pk.eps, eps_n=pk.onorm_eps, capture=v["capture"],
                             t1=v["t1"], restart=v["restart"], half_bytes=self.half[b])[0]

    def warmup(self, buckets=None):
        """Compile every non-capture variant of every bucket by launching it once on
        the real weights and dummy activations, with no KDA state (every slot -1).
        Every rank runs the same launches in the same order, before any step."""
        import time

        L = self.L
        from sgl_amd_mono.kimi_k3.gfx950.attention import kda as k1
        from sgl_amd_mono.kimi_k31.gfx950 import pre_attn as pa

        t0 = time.time()
        dev, bf, f32 = self.device, torch.bfloat16, torch.float32
        dummy = dict(conv_state=torch.zeros(1, k1.CONV_W - 1, 3 * k1.PROJ, dtype=bf, device=dev).mT,
                     conv_win=torch.zeros(1, L, k1.CONV_W - 1, 3 * k1.PROJ, dtype=bf, device=dev),
                     rstate=torch.zeros(1, k1.NH, k1.HD, k1.HD, dtype=f32, device=dev),
                     ring=(torch.zeros(1, k1.NH, 16, k1.HD, dtype=bf, device=dev),
                           torch.zeros(1, k1.NH, 16, k1.HD, dtype=bf, device=dev),
                           torch.zeros(1, k1.NH, 16, k1.HD, dtype=f32, device=dev),
                           torch.zeros(1, k1.NH, 16, dtype=f32, device=dev)))
        n = 0
        for b in buckets or self.buckets:
            bufs = self._ensure(b, L, set())
            st_idx = torch.full((b // L, L), -1, dtype=torch.int32, device=dev)
            num_acc = torch.ones(b // L, dtype=torch.int32, device=dev)
            win_row = torch.full((b // L,), -1, dtype=torch.int32, device=dev)
            seen = set()
            for pk in self.layers.values():
                v = self._variant(pk, b, L, set())
                key = tuple(sorted(v.items()))
                if key in seen:
                    continue
                seen.add(key)
                tab = pa.kda_table(dev, **pk.kda_w, **dummy, st_idx=st_idx, num_acc=num_acc, win_row=win_row,
                                   core_out=bufs["core"])
                self.epoch[0] += 1
                self._build_launch(pk, b, L, v, bufs["cfg"])(*self._launch_args(pk, b, bufs, tab, 0,
                                                                                 self.slots[pk.i]),
                                                             stream=torch.cuda.current_stream())
                n += 1
            torch.cuda.synchronize()
        _say(f"warmup: {n} launches compiled and run in {time.time() - t0:.0f} s")

    # ------------------------------------------------------------ KDA state
    def _kda_step(self, fb, R, L, bufs):
        """The step's KDA state (K1's prepare, for R requests). Device work only:
        a graph replays it."""
        from sglang.srt.layers.attention.linear.utils import select_verify_intermediate_state_indices
        from sglang.srt.model_executor.forward_context import get_attn_backend

        backend = get_attn_backend().linear_attn_backend
        pool, meta = backend.req_to_token_pool, backend.forward_metadata
        cache = pool.mamba_pool.mamba_cache
        assert getattr(cache, "replayssm_rawv", None) is not None, "the wide launch runs on the ReplaySSM ring"
        if not self.kda_ready:
            self.kda_ready = True
            for pk in self.layers.values():
                pk.mamba_idx = pool.mamba2_layer_index(pk.layer.self_attn.attn.layer_id)
        slots = meta.mamba_cache_indices[:R]
        vrows = select_verify_intermediate_state_indices(backend.verify_intermediate_state_indices,
                                                         fb.req_pool_indices, slots >= 0, pool.size)[:R]
        bufs["st_idx"].fill_(-1)
        bufs["st_idx"][:R] = slots.to(torch.int32).view(R, 1)
        bufs["win_row"].fill_(-1)
        bufs["win_row"][:R] = vrows.to(torch.int32)
        return dict(cache=cache)

    def _kda_table(self, pk, ks, b, bufs):
        """The launch reads the conv state from SGLang's cache by slot (``st_idx``)
        and writes the verify windows at ``win_row`` (K1's ``finish`` for R requests)."""
        from sgl_amd_mono.kimi_k31.gfx950 import pre_attn as pa

        tab = self.tabs.get((b, pk.i))
        if tab is None:
            # a host-to-device copy: the eager warmup forwards before a capture build it
            assert not torch.cuda.is_current_stream_capturing(), f"KDA table ({b}, {pk.i}) first built in a capture"
            c, i = ks["cache"], pk.mamba_idx
            conv, win = c.conv[0][i], c.intermediate_conv_window[0][i]
            assert conv.shape[1:] == win.shape[2:] and conv.shape[1] == 3, (tuple(conv.shape), tuple(win.shape))
            ring = (c.replayssm_rawk[i], c.replayssm_rawv[i], c.replayssm_g[i], c.replayssm_beta[i])
            tab = self.tabs[(b, pk.i)] = pa.kda_table(self.device, **pk.kda_w, conv_state=conv.mT,
                                                      rstate=c.temporal[i], st_idx=bufs["st_idx"],
                                                      num_acc=bufs["num_acc"], core_out=bufs["core"],
                                                      conv_win=win,
                                                      win_row=bufs["win_row"], ring=ring)
        return tab

    # ------------------------------------------------------------ the step
    def _tail_front(self, layer, hidden, positions, fb, zero_allocator, bufs, rows, ping):
        """A standard layer whose MoE tail the next launch takes: SGLang's layer up to
        the MoE's TP partials (routed latent -> lat_p, shared -> sh) and this rank's
        columns of the prefix the tail adds (-> pout), the next launch's T1 inputs."""
        from sglang.srt.layers.layer_boundary.residual.batch import written_residual

        h = layer.attn_boundary.prepare(hidden, fb, capture_gathered=None)
        h = layer._run_self_attn(h, positions, fb, zero_allocator)
        h = layer.attn_boundary.finish(h, fb)
        x = layer.ffn_boundary.prepare(h, fb)
        prefix = written_residual(fb)
        assert x.shape == (rows, H) and prefix is not None and prefix.shape == (rows, H), (x.shape, prefix)
        from sglang.srt.models.kimi_k3 import _vector_aligned

        moe = layer.mlp
        # as SGLang's fused front: the routed input's bf16 cast rides the shared SiTU launch
        defer = moe._defer_routed_cast_eligible
        gate_up, logits, routed = moe._fused_front(x, cast_routed=not defer)
        if defer and not _vector_aligned(routed):
            defer, routed = False, routed.to(x.dtype)
        cast = (routed, torch.empty(routed.shape, dtype=x.dtype, device=x.device)) if defer else None
        moe._forward_shared(gate_up, bufs["sh"][ping][:rows], routed_cast=cast)
        moe._forward_routed(x, logits, cast[1] if defer else routed, bufs["lat_p"][ping][:rows])
        bufs["pout"][ping][:rows] = prefix[:, self.cols]
        return x, prefix

    def _column_exit(self, moe, b, rows, bufs, ping):
        """A run's exit as SGLang's column tail (<= 64 rows): one all-reduce of the
        [latent | shared] pair, this rank's stream columns (its prefix columns added
        there), and the stream's all-gather that also stages the next layer's attention
        read. None, before anything runs, when SGLang's kernels do not take the batch."""
        from sglang.kernels.ops.attention import k3_attn_res_read_hip as hip
        from sglang.srt.distributed import tensor_model_parallel_all_reduce
        from sglang.srt.models import kimi_k3 as k3m

        read = moe.next_attn_read
        if (not self.column_exit or read is None or rows > k3m._COLUMN_TAIL_MAX_TOKENS
                or not moe._fused_latent_tail_eligible):
            return None
        prefix = bufs["pfull"][:rows]
        if not (hip.tail_covered(bufs["sh"][ping][:rows], prefix)
                and read.bank.require().gather_covers(rows, SL, read.writes_block)):
            return None
        prefix[:, self.cols] = bufs["pout"][ping][:rows]
        red = tensor_model_parallel_all_reduce(bufs["flat"][ping])
        stream = moe._column_tail_and_next_read(red[: b * LAT].view(b, LAT)[:rows], red[b * LAT :].view(b, H)[:rows],
                                                prefix)
        assert stream is not None
        return stream

    def _rel(self, got, ref):
        """rel L2 over the TP group of tensors each rank holds a part of."""
        import torch.distributed as dist
        from sglang.srt.distributed import get_tp_group

        t = torch.stack([(got.float() - ref.float()).square().sum(), ref.float().square().sum()])
        dist.all_reduce(t, group=get_tp_group().device_group)
        return (t[0] / t[1]).sqrt().item()

    def backbone(self, built, input_ids, fb, input_embeds):
        from sglang.srt.distributed import tensor_model_parallel_all_gather
        from sglang.srt.layers.layer_boundary.residual.batch import final_norm, fold, set_written, start
        from sglang.srt.models.kimi_k3 import AuxHiddenStateList, _cdiv
        from sglang.srt.utils.common import BumpAllocator

        m = self.bb
        hidden = m.embed_tokens(input_ids) if input_embeds is None else input_embeds
        rows = hidden.shape[0]
        L = int(fb.spec_info.draft_token_num)
        assert L == self.L and rows % L == 0, (rows, L, self.L)
        if not getattr(self, "_said", False):
            self._said = True
            _say(f"first wide step: rows {rows}, L {L}")
        R = rows // L
        b = next(x for x in self.buckets if x >= rows)
        captures = set(m.dspark_layers_to_capture or ())
        bufs = self._ensure(b, L, captures)
        cfg = bufs["cfg"]
        start(fb)
        zero_allocator = BumpAllocator(buffer_size=2 * (m.end_layer - m.start_layer), dtype=torch.float32,
                                       device=hidden.device)
        m.attn_bank.open(hidden, _cdiv(m.end_layer, m.config.attn_res_block_size), None)
        aux = AuxHiddenStateList()
        ks = self._kda_step(fb, R, L, bufs)
        self.epoch[0].add_(1)
        cols = self.cols
        check = self.check_left > 0 and not torch.cuda.is_current_stream_capturing()
        self.check_left -= int(check)
        i = m.start_layer
        ping = 0
        ref_mid = None
        while i < m.end_layer:
            if i in self.entry:
                x, prefix = self._tail_front(m.layers[i], hidden, built._positions, fb, zero_allocator, bufs, rows,
                                             ping)
                if check:  # SGLang's whole MoE on the same input, this rank's columns
                    ref_mid = m.layers[i].mlp(x, prefix_sum=prefix)[:, cols].clone()
                i += 1
                continue
            if i not in self.layers:
                hidden = m.layers[i](positions=built._positions, hidden_states=hidden, forward_batch=fb,
                                     zero_allocator=zero_allocator, capture_gathered=None)
                if i in captures and (i + 1) not in self.layers:
                    aux.append(m._dspark_capture_stream(i, hidden, fb))
                i += 1
                continue
            # a run of covered layers
            res = m.attn_bank.require()
            if not self.layers[i].t1:  # else its first launch runs the previous layer's MoE tail
                full = fold(hidden, fb)
                bufs["mid"][:rows] = full[:, cols]
            bufs["bank"][:rows] = res.block_residual[:rows, :, cols]
            run = []
            while i in self.layers:
                run.append(i)
                i += 1
            for n, li in enumerate(run):
                pk = self.layers[li]
                v = self._variant(pk, b, L, captures)
                assert v["t1"] == (n > 0 or (li - 1) in self.entry)
                assert res.num_valid_blocks == pk.nvb, (li, res.num_valid_blocks, pk.nvb)
                tab = self._kda_table(pk, ks, b, bufs)
                cur = 1 - ping
                self._build_launch(pk, b, L, v, cfg)(*self._launch_args(pk, b, bufs, tab, ping, self.slots[li]),
                                                     stream=torch.cuda.current_stream())
                if n == 0 and ref_mid is not None:
                    e = self._rel(bufs["mid"][:rows], ref_mid)
                    _say(f"entry check layer {li - 1} -> {li} rows {rows}: mid rel {e:.2e}")
                    ref_mid = None
                if v["capture"]:
                    aux.append(bufs["cap"][:rows].clone())
                if pk.write:
                    res.write(tensor_model_parallel_all_gather(bufs["bank"][:rows, pk.nvb].contiguous(), dim=-1))
                lat_p = bufs["lat_p"][cur]
                pk.moe._forward_routed(bufs["lat"][:rows], bufs["logits"][:rows], bufs["lat"][:rows], lat_p[:rows])
                ping = cur
            # leave: the last layer's MoE tail on SGLang
            last = self.layers[run[-1]]
            sh = bufs["sh"][ping][:rows]
            if check:  # the gathered-prefix tail
                full = tensor_model_parallel_all_gather(bufs["pout"][ping][:rows].contiguous(), dim=-1)
                ref = last.moe._column_split_tail(bufs["lat_p"][ping][:rows].clone(), sh.clone(), full, cols=cols)
            stream = self._column_exit(last.moe, b, rows, bufs, ping)
            if stream is None:
                # this rank's prefix columns ride in its columns of the shared partial, so
                # the tail's all-reduce carries them (in place: no launch reads sh after)
                sh[:, cols] += bufs["pout"][ping][:rows]
                stream = last.moe._column_split_tail(bufs["lat_p"][ping][:rows], sh, None, cols=cols)
            if check:
                _say(f"exit check layer {run[-1]} rows {rows}: stream rel "
                     f"{((stream.float() - ref.float()).norm() / ref.float().norm()).item():.2e}")
            hidden = set_written(stream, fb)
            if run[-1] in captures:
                aux.append(m._dspark_capture_stream(run[-1], hidden, fb))
        hidden = final_norm(hidden, fb, m._final_read, capture=None, skip_empty=True)
        m.attn_bank.close()
        if m.dspark_layers_to_capture is not None:
            return hidden, aux
        return hidden
