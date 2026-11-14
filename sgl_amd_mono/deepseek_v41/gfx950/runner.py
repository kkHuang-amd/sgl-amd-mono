# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host runner and buffers of the mono decode launches."""

from __future__ import annotations

import os
import struct
from pathlib import Path

import msgspec
import torch
from aiter.ops.flydsl.kernels.tensor_shim import _preload_compiled, _run_compiled

from sgl_amd_mono.common import guard
from sgl_amd_mono.common.plan import BLOCKS

from .attention.index_query import DIM as IQ_DIM
from .attention.index_query import HEADS as IQ_HEADS
from .attention.index_query import fused as iq_fused
from .attention.plan import DATA, HEAD_DIM, KEYS, Q_RANK, TOPK, WINDOW, Dims
from .layer import (
    EPOCH_WORDS,
    HIDDEN,
    MAX_TOKENS,
    MonoBuild,
    build_mono_ffn,
    build_mono_k1,
    build_mono_k2,
    peer_half_bytes,
    scratch_bytes,
    scratch_layout,
)
from .stages.dims import Dims as MoeDims
from .stages.moe_shape import EXPERTS
from .stages.moe_shape import TOPK as ROUTE_TOPK
from .token_par import TOKEN_PAR_FROM, token_par

# the index split pays from TP4: at TP2 a bs8 step's indexer sheds ~90 us and the exchange
# costs ~65
INDEX_SPLIT_MIN_TP = 4


def _report_index_check(exchange, rank: int, every_s: float = 60.0) -> None:
    """Log the index split check's counts every ``every_s`` (rank 0, and any rank with a
    mismatch) from a daemon thread that reads host memory only."""
    import logging
    import threading
    import time

    log = logging.getLogger(__name__)

    def loop():
        while True:
            time.sleep(every_s)
            own, peers, seen = exchange.checked()
            if rank == 0 or own or peers:
                log.info(
                    "DSV4.1 index split check (rank %d): %d own + %d peers' / %d 16 B "
                    "vectors differ",
                    rank, own, peers, seen,
                )  # fmt: skip

    threading.Thread(target=loop, name="index-split-check", daemon=True).start()

__all__ = [
    "BLOCKS",
    "MAX_TOKENS",
    "DSV41MonoRunner",
    "K1Weights",
    "K2Weights",
    "FFNWeights",
]

HC = 4
# below this many rows an ug task takes one FP4 group (``route_at``)
FINE_UG_ROWS = 7
RECORD = DATA + 8  # a token's KV record: 576 B of data, then 8 B of scales
_LOG2E = 1.4426950408889634
# a wait's deadline until the framework sets one from its watchdog timeout
DEFAULT_DEADLINE_S = 60.0


def compile_only(launch) -> None:
    """Build the launcher without dispatching its kernels."""
    import inspect

    names = [n for n in inspect.signature(launch.func).parameters if n != "stream"]
    _preload_compiled(launch, *([0] * len(names)), torch.cuda.current_stream())


def _ensure_writable_flydsl_cache() -> None:
    """Point FlyDSL at a writable cache (aiter sets a read-only one)."""
    cur = os.environ.get("FLYDSL_RUNTIME_CACHE_DIR")
    if cur and os.access(cur, os.W_OK):
        return
    path = Path.home() / ".flydsl" / "cache"
    path.mkdir(parents=True, exist_ok=True)
    os.environ["FLYDSL_RUNTIME_CACHE_DIR"] = str(path)


def _check_tensors(owner, tag: str, want: dict) -> None:
    """Raise unless each named tensor has the expected shape and dtype and is contiguous."""
    for name, (shape, dtype) in want.items():
        t = getattr(owner, name)
        if not (
            tuple(t.shape) == shape
            and t.dtype == dtype
            and t.is_contiguous()
            and t.is_cuda
        ):
            raise RuntimeError(
                f"{tag} {name}: {tuple(t.shape)} {t.dtype}, want {shape} {dtype}"
            )


def check_rows(tag: str, rows: int, device, want: dict) -> None:
    """Raise unless each (tensor, row size, dtype) is ``rows`` contiguous rows on ``device``."""
    for name, (t, size, dtype) in want.items():
        if not (
            t.dtype == dtype
            and t.numel() == rows * size
            and t.is_contiguous()
            and t.device == device
        ):
            raise RuntimeError(
                f"{tag} {name}: {tuple(t.shape)} {t.dtype} {t.device}, want {rows} x {size} {dtype}"
            )


# SWA cache slots as SGLang's backends hand them out; the kernels read the low word
SLOT_DTYPES = (torch.int32, torch.int64)


def slot_words(slot: torch.Tensor) -> int:
    """A slot's stride in int32 words."""
    return slot.element_size() // 4


def check_cache(tag: str, cache: torch.Tensor, block: int) -> None:
    """Raise unless ``cache`` is uint8 [pages, page bytes] with ``block`` records a page."""
    if not (
        cache.dtype == torch.uint8
        and cache.dim() == 2
        and cache.stride(1) == 1
        and cache.stride(0) >= block * RECORD
    ):
        raise RuntimeError(
            f"{tag}: {cache.dtype} {tuple(cache.shape)} {cache.stride()}, "
            f"want uint8 pages of {block} x {RECORD} B"
        )


def no_capture(what: str) -> None:
    """Raise inside a CUDA graph capture: builds and allocations run eagerly."""
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            f"DSv4.1 mono decode: {what} first reached inside a CUDA graph capture"
        )


class FFNWeights(msgspec.Struct):
    """One layer's FFN-launch tensors on this rank, in SGLang's loaded layout."""

    hc_ffn_fn: torch.Tensor  # [24, 4 * 5120] f32
    hc_ffn_scale: torch.Tensor  # [3] f32
    hc_ffn_base: torch.Tensor  # [24] f32
    ffn_norm: torch.Tensor  # [5120] bf16
    gate_w: torch.Tensor  # [384, 5120] bf16
    bias: torch.Tensor  # [384] f32 (e_score_correction_bias)
    w13: torch.Tensor  # [384, 2 inter, 2560] fp4x2, aiter (16, 16) shuffle
    w13_s: torch.Tensor  # its scales, aiter shuffle_scale order
    w2: torch.Tensor  # [384, 5120, inter / 2] fp4x2
    w2_s: torch.Tensor
    sgu: torch.Tensor  # shared gate_up [2 inter, 5120] e4m3, row-major
    sgu_s: torch.Tensor  # [2 inter / 32, 160] E8M0
    sw2: torch.Tensor  # shared down [5120, inter] e4m3
    sw2_s: torch.Tensor

    def check(self, tp: int, experts: int = EXPERTS) -> None:
        """Raise unless the MoE tensors have SGLang's loaded layout."""
        d, fp4, e4m3, u8 = (
            MoeDims(tp),
            torch.float4_e2m1fn_x2,
            torch.float8_e4m3fn,
            torch.uint8,
        )
        want = {
            "gate_w": ((experts, HIDDEN), torch.bfloat16),
            "bias": ((experts,), torch.float32),
            "w13": ((experts, 2 * d.inter, HIDDEN // 2), fp4),
            "w13_s": ((experts * 2 * d.inter, HIDDEN // 32), u8),
            "w2": ((experts, HIDDEN, d.inter // 2), fp4),
            "w2_s": ((experts * HIDDEN, d.down_scale_cols), u8),
            "sgu": ((2 * d.sh_inter, HIDDEN), e4m3),
            "sgu_s": ((2 * d.sh_inter // 32, HIDDEN // 32), u8),
            "sw2": ((HIDDEN, d.sh_inter), e4m3),
            "sw2_s": ((HIDDEN // 32, d.sh_inter // 32), u8),
        }
        _check_tensors(self, "MoE", want)


class K1Weights(msgspec.Struct):
    """One layer's attention-seam and attention-front tensors on this rank, as K1 reads them."""

    hc_fn: torch.Tensor  # [24, 4 * 5120] f32
    hc_scale: torch.Tensor  # [3] f32
    hc_base: torch.Tensor  # [24] f32
    norm: torch.Tensor  # [5120] input_layernorm
    wqkv: torch.Tensor  # [1792, 5120] e4m3, row-major: wq_a then wkv_a
    wqkv_s: torch.Tensor  # [56, 160] E8M0 (32 x 32 blocks)
    q_norm: torch.Tensor  # [1280] bf16
    kv_norm: torch.Tensor  # [512] bf16
    wqb: torch.Tensor  # [heads * 512, 1280] e4m3, row-major
    wqb_s: torch.Tensor  # [heads * 16, 40] E8M0
    cos_sin: torch.Tensor  # [positions, 64] f32: 32 cos, then 32 sin
    # an index source's indexer (replicated), for K1's index query; None without one
    iq_w: torch.Tensor | None = None  # wq_b [4096, 1280] e4m3
    iq_ws: torch.Tensor | None = None  # its E8M0 blocks [128, 40]
    iw_w: torch.Tensor | None = None  # weights_proj [32, 5120] bf16
    iw_scale: float = 0.0

    def check(self, d: Dims) -> None:
        """Raise unless the attention-front tensors have K1's layout."""
        e4m3, u8, f32, bf = (
            torch.float8_e4m3fn,
            torch.uint8,
            torch.float32,
            torch.bfloat16,
        )
        want = {
            "hc_fn": ((24, HC * HIDDEN), f32),
            "hc_scale": ((3,), f32),
            "hc_base": ((24,), f32),
            "wqkv": ((1792, HIDDEN), e4m3),
            "wqkv_s": ((1792 // 32, HIDDEN // 32), u8),
            "q_norm": ((1280,), bf),
            "kv_norm": ((512,), bf),
            "wqb": ((d.heads * HEAD_DIM, 1280), e4m3),
            "wqb_s": ((d.heads * HEAD_DIM // 32, 1280 // 32), u8),
        }
        if self.iq_w is not None:
            want |= {
                "iq_w": ((IQ_HEADS * IQ_DIM, 1280), e4m3),
                "iq_ws": ((IQ_HEADS * IQ_DIM // 32, 1280 // 32), u8),
                "iw_w": ((IQ_HEADS, HIDDEN), bf),
            }
        _check_tensors(self, "K1", want)
        if not (
            self.norm.shape == (HIDDEN,)
            and self.norm.is_contiguous()
            and self.norm.is_cuda
        ):
            raise RuntimeError(f"K1 norm: {tuple(self.norm.shape)}")
        cs = self.cos_sin
        if not (
            cs.dtype == f32
            and cs.dim() == 2
            and cs.shape[1] == 64
            and cs.is_contiguous()
        ):
            raise RuntimeError(
                f"K1 cos_sin: {tuple(cs.shape)} {cs.dtype}, want [positions, 64] f32"
            )


class K2Weights(msgspec.Struct):
    """One layer's attention-back tensors on this rank, as K2 reads them."""

    woa: torch.Tensor  # [G * 1024, 4096] e4m3, row-major
    woa_s: torch.Tensor  # [G * 32, 128] E8M0 (32 x 32 blocks)
    wob: torch.Tensor  # [5120, G * 1024] e4m3, row-major
    wob_s: torch.Tensor  # [160, G * 32] E8M0
    sink: torch.Tensor  # [heads] f32, this rank's
    cos_sin: torch.Tensor  # [positions, 64] f32: 32 cos, then 32 sin

    def check(self, d: Dims) -> None:
        """Raise unless the attention-back tensors have K2's layout."""
        g, e4m3, u8 = d.groups, torch.float8_e4m3fn, torch.uint8
        want = {
            "woa": ((g * 1024, d.group_k), e4m3),
            "woa_s": ((g * 32, d.group_k // 32), u8),
            "wob": ((HIDDEN, g * 1024), e4m3),
            "wob_s": ((HIDDEN // 32, g * 32), u8),
            "sink": ((d.heads,), torch.float32),
        }
        _check_tensors(self, "K2", want)
        cs = self.cos_sin
        if not (cs.dtype == torch.float32 and cs.dim() == 2 and cs.shape[1] == 64):
            raise RuntimeError(f"K2 cos_sin: {tuple(cs.shape)} {cs.dtype}")


class DSV41MonoRunner:
    """The mono decode runner of one TP rank (``group``: gloo, for the peer handle exchange).

    The route and window are the target's by default; the DSpark draft's runner has its
    own (128 experts, top-3, a FP4 group an ug task, and a window of 128 plus the block's
    keys)."""

    def __init__(
        self,
        tp: int,
        rank: int,
        group,
        device: torch.device | str = "cuda",
        experts: int = EXPERTS,
        topk: int = ROUTE_TOPK,
        ug_groups: int | None = None,
        swa_width: int = WINDOW,
        shuffled: bool = False,
    ):
        _ensure_writable_flydsl_cache()
        from sgl_amd_mono.common.peer import PeerBuffer

        self.tp, self.rank = tp, rank
        self.route = dict(experts=experts, topk=topk, swa_width=swa_width)
        if ug_groups not in (None, 1, 2):
            raise ValueError(
                f"ug_groups is the FP4 groups an ug task takes, 1 or 2 (None: by "
                f"width), got {ug_groups}"
            )
        self.ug_groups = ug_groups
        self.timeline_on = os.environ.get("SGLANG_AMD_MONO_TIMELINE") == "1"
        self.woa_spread2_on = os.environ.get("SGLANG_AMD_MONO_DSV41_WOA_SPREAD2") == "1"
        self.token_par_on = os.environ.get("SGLANG_AMD_MONO_TOKEN_PAR", "1") == "1"
        # the index sources' rows split by request across ranks (``index_rows``): "1", or
        # "emulate" (timing only: a rank scores its own rows and exchanges nothing)
        self.index_split = os.environ.get("SGLANG_AMD_MONO_DSV41_INDEX_SPLIT", "0")
        if self.index_split not in ("0", "1", "emulate"):
            raise ValueError(
                f"SGLANG_AMD_MONO_DSV41_INDEX_SPLIT is 0, 1 or emulate, got {self.index_split}"
            )
        # K1 / K2 read wqkv, wq_b and wo_b in aiter's (16, 16) preshuffle
        self.shuffled = shuffled
        self.d = Dims(tp)
        dev = torch.device(device)
        if dev.index is None:
            dev = torch.device(dev.type, torch.cuda.current_device())
        self.device = dev
        self._scratch: dict[int, torch.Tensor] = {}
        # [epoch, the guard block (deadline, error words), a mark per CTA, the MoE's counters]
        self.epoch = torch.zeros(EPOCH_WORDS, dtype=torch.int32, device=dev)
        self.fault_site, self.fault_after = guard.fault_injection(rank) or (0, 0)
        self.set_deadline(DEFAULT_DEADLINE_S)
        self.q = torch.zeros(
            MAX_TOKENS, self.d.heads, HEAD_DIM, dtype=torch.bfloat16, device=dev
        )
        # K1's outputs: the attention seam's residual and mixes
        self.res_mid = torch.zeros(
            MAX_TOKENS, HC, HIDDEN, dtype=torch.bfloat16, device=dev
        )
        self.post_a = torch.zeros(MAX_TOKENS, HC, dtype=torch.float32, device=dev)
        self.comb_a = torch.zeros(MAX_TOKENS, HC, HC, dtype=torch.float32, device=dev)
        self.pre_a = torch.zeros(MAX_TOKENS, HC, dtype=torch.float32, device=dev)
        # K2's key table, by K1 (by K2 on an index source): a slot a key, (kv_len, ntopk) a token
        self.kt = torch.zeros(MAX_TOKENS * KEYS, dtype=torch.int32, device=dev)
        self.klen = torch.zeros(2 * MAX_TOKENS, dtype=torch.int32, device=dev)
        # a window-only layer's top-k: none (K1 reads the table, ntopk is 0)
        self.no_topk = torch.zeros(MAX_TOKENS, TOPK, dtype=torch.int32, device=dev)
        self.no_topk_lens = torch.zeros(MAX_TOKENS, dtype=torch.int32, device=dev)
        # the first layer's incoming pre mix: copy 0 of the broadcast embedding
        self.first_pre = torch.zeros(MAX_TOKENS, HC, dtype=torch.float32, device=dev)
        self.first_pre[:, 0] = 1.0
        # an index source's K1 outputs, its compressor and indexer's inputs
        self.normed = torch.zeros(MAX_TOKENS, HIDDEN, dtype=torch.bfloat16, device=dev)
        self.qr = torch.zeros(MAX_TOKENS, Q_RANK, dtype=torch.bfloat16, device=dev)
        # K1's index query (``index_query``) in SGLang's index_q_rope_pack_weights_flydsl layout
        self.iq = torch.zeros(
            MAX_TOKENS, IQ_HEADS, IQ_DIM // 2, dtype=torch.int8, device=dev
        )
        self.iqs = torch.zeros(MAX_TOKENS, 1, 4, 16, 4, dtype=torch.uint8, device=dev)
        self.iw = torch.zeros(MAX_TOKENS, IQ_HEADS, dtype=torch.bfloat16, device=dev)
        # the record an absent key reads: zero data and scales, masked anyway
        self.zero_rec = torch.zeros(1024, dtype=torch.uint8, device=dev)
        self.qk_scale = (
            torch.tensor([HEAD_DIM**-0.5 * _LOG2E], dtype=torch.float32)
            .view(torch.int32)
            .item()
        )
        # two parity halves, then the guard's flags: rank r's failed wait sets word r on every rank
        half = peer_half_bytes(tp)
        self.peer = PeerBuffer(2 * half + guard.FLAG_BYTES, group, rank, tp, dev)
        self.peer.bytes.zero_()
        # the index sources' rows from their owner ranks (``index_rows``), collective
        self.index_exchange = None
        # SGLANG_AMD_MONO_DSV41_INDEX_SPLIT_CHECK=1: a replicated selection beside the split
        # one (``k2i_check``), compared row by row in the exchange
        # ("full": the split path scores every row too, isolating the reference and the exchange)
        self.index_check_mode = os.environ.get("SGLANG_AMD_MONO_DSV41_INDEX_SPLIT_CHECK", "0")
        self.index_check = self.index_check_mode in ("1", "full")
        self.k2i_check = None
        if self.index_split == "1" and tp >= INDEX_SPLIT_MIN_TP:
            from .index_exchange import IndexExchange

            self.index_exchange = IndexExchange(
                group, rank, tp, dev, DEFAULT_DEADLINE_S, check=self.index_check
            )
        self._kernels: dict = {}
        # the index sources' indexer (``enable_k2i``): the target's runner only
        self.k2i = None

    def enable_k2i(self, bound_max: int):
        """K2i's buffers and builds, for column bounds up to ``bound_max`` (the
        server's longest context: a ratio-1 source's compressed length)."""
        from .indexer.runner import K2iRunner

        no_capture("the K2i buffers")
        self.k2i = K2iRunner(device=self.device, bound_max=bound_max)
        if self.index_exchange is not None and self.index_check:
            self.k2i_check = K2iRunner(device=self.device, bound_max=bound_max)
            i32 = dict(dtype=torch.int32, device=self.device)
            self.index_ref = (
                torch.empty(MAX_TOKENS, 512, **i32),
                torch.empty(MAX_TOKENS, 512, **i32),
            )
            _report_index_check(self.index_exchange, self.rank)
        return self.k2i

    def set_deadline(self, seconds: float) -> None:
        """Every wait's deadline (below the server's watchdog timeout)."""
        ticks = guard.deadline_ticks(seconds)
        exchange = getattr(self, "index_exchange", None)  # None before __init__ builds it
        if exchange is not None:
            exchange.deadline_s = seconds
        lo = ticks & 0xFFFFFFFF
        self.epoch[guard.GUARD_DEADLINE] = lo - (1 << 32) if lo >= 1 << 31 else lo
        self.epoch[guard.GUARD_DEADLINE + 1] = ticks >> 32

    def scratch(self, tokens: int) -> torch.Tensor:
        """Return this width's scratch, allocated before graph capture."""
        buf = self._scratch.get(tokens)
        if buf is None:
            no_capture(f"step width {tokens}")
            buf = torch.zeros(
                scratch_bytes(
                    tokens, self.tp, self.timeline_on, **self.route_at(tokens)
                ),
                dtype=torch.uint8,
                device=self.device,
            )
            self._scratch[tokens] = buf
        return buf

    def route_at(self, tokens: int) -> dict:
        """The route every launch and the scratch of this width share.

        Below ``FINE_UG_ROWS`` rows an ug task takes one FP4 group, so twice the tasks
        end closer together (ROCm/ATOM's fine ug); the draft fixes its own."""
        ug = self.ug_groups
        if ug is None:
            ug = 1 if tokens < FINE_UG_ROWS else 2
        return {**self.route, "ug_groups": ug}

    def timeline(self, tokens: int) -> torch.Tensor:
        """The last launch's stage marks at this width: int64 [BLOCKS, TL_POINTS],
        100 MHz ticks (SGLANG_AMD_MONO_TIMELINE=1 builds only)."""
        from sgl_amd_mono.common.stamps import TL_POINTS

        lay = scratch_layout(tokens, self.tp, True, **self.route_at(tokens))
        off, n = lay["timeline"]
        return (
            self.scratch(tokens)[off : off + n]
            .view(torch.int64)
            .view(BLOCKS, TL_POINTS)
        )

    def tok_par(self, tokens: int) -> bool:
        """Whether a target step of this width runs token parallel (``token_par``)."""
        return self.token_par_on and token_par(tokens, self.tp)

    @staticmethod
    def supports(tokens: int) -> bool:
        return 1 <= tokens <= MAX_TOKENS

    def index_ranges(self, tokens: int, per_req: int) -> list[tuple[int, int]] | None:
        """Every rank's rows [lo, hi) of an index source's top-k when they split by
        request (whole requests a rank, so a request's rows share one key load), or None."""
        if (
            self.index_split == "0"
            or self.tp < INDEX_SPLIT_MIN_TP
            or tokens < TOKEN_PAR_FROM
        ):
            return None
        nreq = tokens // per_req
        if nreq * per_req != tokens or nreq < self.tp:
            return None
        return [
            (q * nreq // self.tp * per_req, (q + 1) * nreq // self.tp * per_req)
            for q in range(self.tp)
        ]

    def index_rows(self, tokens: int, per_req: int) -> tuple[int, int] | None:
        """This rank's rows of ``index_ranges``, or None."""
        ranges = self.index_ranges(tokens, per_req)
        return None if ranges is None else ranges[self.rank]

    def iq_fused(self, tokens: int, tok_par: bool = False) -> bool:
        """Whether an index source's K1 at this width computes its index query
        (``index_query``); else SGLang's chain does, after K1."""
        return iq_fused(tokens, tok_par)

    def k1_kernel(
        self,
        tokens: int,
        fold: bool = True,
        index: bool = False,
        iq: bool = False,
        tok_par: bool = False,
    ):
        """K1 at this width; it writes K2's key table unless the layer is an index
        source (whose top-k comes after K1). ``iq``: the index query too, where
        ``iq_fused``."""
        iq = iq and self.iq_fused(tokens, tok_par)
        key = (tokens, "k1", fold, index, iq, tok_par)
        if key not in self._kernels:
            no_capture(f"K1 width {tokens}")
            self._kernels[key] = build_mono_k1(
                MonoBuild(
                    tokens,
                    self.tp,
                    kt=not index,
                    fold=fold,
                    index=index,
                    shuffled=self.shuffled,
                    fault_site=self.fault_site,
                    fault_after=self.fault_after,
                    timeline=self.timeline_on,
                    tok_par=tok_par,
                    iq=iq,
                    **self.route_at(tokens),
                )
            )
        return self._kernels[key]

    def precompile(
        self,
        tokens: int,
        k1=(),
        k2=(),
        ffn: bool = False,
        k2i: bool = False,
        tok_par: bool = False,
    ) -> None:
        """Allocate this width's scratch and compile its launches without running them.

        K1 for each (fold, index, iq) in ``k1``, K2 for each (compressed, index) in
        ``k2``, the FFN launch, and K2i (``enable_k2i`` first)."""
        self.scratch(tokens)
        kernels = [self.k1_kernel(tokens, *v, tok_par=tok_par) for v in sorted(k1)]
        kernels += [self.k2_kernel(tokens, *v, tok_par=tok_par) for v in sorted(k2)]
        if ffn:
            kernels.append(self.ffn_kernel(tokens))
        if k2i:
            kernels.append(self.k2i.kernel(tokens))
        for launch in kernels:
            compile_only(launch)

    def ffn_kernel(self, tokens: int):
        key = (tokens, "ffn")
        if key not in self._kernels:
            no_capture(f"FFN launch width {tokens}")
            self._kernels[key] = build_mono_ffn(
                MonoBuild(
                    tokens,
                    self.tp,
                    fault_site=self.fault_site,
                    fault_after=self.fault_after,
                    timeline=self.timeline_on,
                    **self.route_at(tokens),
                )
            )
        return self._kernels[key]

    def k2_kernel(
        self, tokens: int, compressed: bool = True, index: bool = False, tok_par=False
    ):
        key = (tokens, "k2", compressed, index, tok_par)
        if key not in self._kernels:
            no_capture(f"K2 width {tokens}")
            self._kernels[key] = build_mono_k2(
                MonoBuild(
                    tokens,
                    self.tp,
                    kt=True,
                    index=index,
                    compressed=compressed,
                    shuffled=self.shuffled,
                    fault_site=self.fault_site,
                    fault_after=self.fault_after,
                    timeline=self.timeline_on,
                    tok_par=tok_par,
                    woa_spread2=self.woa_spread2_on,
                    **self.route_at(tokens),
                )
            )
        return self._kernels[key]

    @staticmethod
    def _outs(M: int, residual: torch.Tensor) -> tuple:
        dev = residual.device
        return (
            torch.empty(M, HIDDEN, dtype=torch.bfloat16, device=dev),
            torch.empty_like(residual),
            torch.empty(M, HC, 1, dtype=torch.float32, device=dev),
            torch.empty(M, HC, HC, dtype=torch.float32, device=dev),
            torch.empty(M, HC, dtype=torch.float32, device=dev),
        )

    def ffn(
        self,
        w: FFNWeights,
        part: torch.Tensor,  # [M, 5120] bf16: this rank's unreduced wo_b output
        residual: torch.Tensor,  # [M, 4, 5120] bf16, after the attention seam
        post_mix: torch.Tensor,  # [M, 4, 1] f32, the attention seam's
        res_mix: torch.Tensor,  # [M, 4, 4] f32
        pre_mix: torch.Tensor,  # [M, 4] f32
        outs: tuple | None = None,
    ):
        """Run the FFN launch: attention TP reduce, FFN seam, MoE and all-reduce."""
        M = part.shape[0]
        if not self.supports(M):
            raise RuntimeError(f"DSv4.1 mono decode: {M} rows")
        bf, f32 = torch.bfloat16, torch.float32
        check_rows(
            "FFN",
            M,
            self.device,
            {
                "part": (part, HIDDEN, bf),
                "residual": (residual, HC * HIDDEN, bf),
                "post": (post_mix, HC, f32),
                "comb": (res_mix, HC * HC, f32),
                "pre": (pre_mix, HC, f32),
            },
        )
        if outs is None:
            outs = self._outs(M, residual)
        out, res_out, post_out, comb_out, pre_out = outs
        check_rows(
            "FFN",
            M,
            self.device,
            {
                "out": (out, HIDDEN, bf),
                "res_out": (res_out, HC * HIDDEN, bf),
                "post_out": (post_out, HC, f32),
                "comb_out": (comb_out, HC * HC, f32),
                "pre_out": (pre_out, HC, f32),
            },
        )
        _run_compiled(
            self.ffn_kernel(M),
            part.data_ptr(),
            residual.data_ptr(),
            post_mix.data_ptr(),
            res_mix.data_ptr(),
            pre_mix.data_ptr(),
            w.hc_ffn_fn.data_ptr(),
            w.hc_ffn_scale.data_ptr(),
            w.hc_ffn_base.data_ptr(),
            w.ffn_norm.data_ptr(),
            res_out.data_ptr(),
            post_out.data_ptr(),
            comb_out.data_ptr(),
            pre_out.data_ptr(),
            w.gate_w.data_ptr(),
            w.bias.data_ptr(),
            w.w13.data_ptr(),
            w.w13_s.data_ptr(),
            w.w2.data_ptr(),
            w.w2_s.data_ptr(),
            w.sgu.data_ptr(),
            w.sgu_s.data_ptr(),
            w.sw2.data_ptr(),
            w.sw2_s.data_ptr(),
            out.data_ptr(),
            self.scratch(M).data_ptr(),
            self.peer.local,
            self.peer.addresses.data_ptr(),
            self.rank,
            self.epoch.data_ptr(),
            torch.cuda.current_stream(),
        )
        return outs

    def k1(
        self,
        w: K1Weights,
        x: torch.Tensor | None,  # [M, 5120] bf16: the prior layer's pending MoE output
        residual: torch.Tensor,  # [M, 4, 5120] bf16: its pending residual
        post: torch.Tensor | None,  # [M, 4, 1] f32: its pending FFN post mix
        comb: torch.Tensor | None,  # [M, 4, 4] f32
        pre: torch.Tensor,  # [M, 4] f32: the incoming pre mix (its FFN pre mix)
        pos: torch.Tensor,  # [>= M] int64 positions
        slot: torch.Tensor,  # [>= M] int32 or int64 SWA cache slots
        swa: torch.Tensor,  # the layer's SWA cache: uint8 [pages, page bytes]
        swa_block: int,
        keys: tuple,  # int32 [>= M, ...]: window slots, window lengths, top-k slots, top-k lengths
        index: bool = False,
        tok_par: bool = False,
        aux: torch.Tensor | None = None,  # [>= M, 5120] bf16, any row stride: a capture
    ):
        """Run K1: the pending post, this layer's attention seam and the attention front.

        It writes the step's SWA KV records and K2's key table, and returns (q, residual,
        post, comb, pre) as views of the runner's buffers. ``x`` None: the residual
        arrives settled, with no post to fold in (the first layer, after an Engram
        write). ``index``: an index source, whose top-k comes after K1, so K1 writes
        ``normed`` and ``qr`` for the indexer and leaves the key table to K2.
        ``tok_par`` (``token_par``): the residual, post, comb and pre rows in and out
        are this rank's alone. ``aux`` receives the residual streams' mean for DSpark."""
        M = residual.shape[0]
        if not self.supports(M):
            raise RuntimeError(f"DSv4.1 mono decode: {M} rows")
        bf, f32 = torch.bfloat16, torch.float32
        fold = x is not None
        want = {
            "residual": (residual, HC * HIDDEN, bf),
            "pre": (pre, HC, f32),
        }
        if fold:
            want |= {
                "x": (x, HIDDEN, bf),
                "post": (post, HC, f32),
                "comb": (comb, HC * HC, f32),
            }
        check_rows("K1", M, self.device, want)
        if aux is not None and not (
            aux.dtype == bf
            and aux.shape[0] >= M
            and aux.shape[-1] == HIDDEN
            and aux.stride(-1) == 1
            and aux.stride(0) % 8 == 0
            and aux.device == self.device
        ):
            raise RuntimeError(f"K1 aux: {tuple(aux.shape)} {aux.stride()} {aux.dtype}")
        if not fold:  # unread: the seam takes the residual as it is
            x = post = comb = residual
        for name, t, dtypes in (
            ("positions", pos, (torch.int64,)),
            ("slots", slot, SLOT_DTYPES),
        ):
            if not (t.dtype in dtypes and t.numel() >= M and t.is_contiguous()):
                raise RuntimeError(f"K1 {name}: {tuple(t.shape)} {t.dtype}")
        check_cache("K1 SWA cache", swa, swa_block)
        for name, t, width in zip(
            ("window slots", "window lengths", "top-k slots", "top-k lengths"),
            keys,
            ((self.route["swa_width"],), None, (TOPK,), None),
        ):
            if not (
                t is not None
                and t.dtype == torch.int32
                and t.is_contiguous()
                and t.shape[0] >= M
                and (width is None or tuple(t.shape[1:]) == width)
            ):
                raise RuntimeError(
                    f"K1 {name}: "
                    + ("None" if t is None else f"{tuple(t.shape)} {t.dtype}")
                )
        swa_idx, swa_lens, topk, topk_lens = (t.data_ptr() for t in keys)
        iq = w.iq_w is not None and self.iq_fused(M, tok_par)
        iqa = (
            (w.iq_w.data_ptr(), w.iq_ws.data_ptr(), w.iw_w.data_ptr())
            if iq
            else (0, 0, 0)
        )
        iw_bits = struct.unpack("<i", struct.pack("<f", w.iw_scale))[0]
        _run_compiled(
            self.k1_kernel(M, fold, index, iq, tok_par),
            residual.data_ptr(),
            x.data_ptr(),
            post.data_ptr(),
            comb.data_ptr(),
            pre.data_ptr(),
            w.hc_fn.data_ptr(),
            w.hc_scale.data_ptr(),
            w.hc_base.data_ptr(),
            w.norm.data_ptr(),
            self.res_mid.data_ptr(),
            0 if aux is None else aux.data_ptr(),
            HIDDEN if aux is None else aux.stride(0),
            self.post_a.data_ptr(),
            self.comb_a.data_ptr(),
            self.pre_a.data_ptr(),
            w.wqkv.data_ptr(),
            w.wqkv_s.data_ptr(),
            w.q_norm.data_ptr(),
            w.kv_norm.data_ptr(),
            w.wqb.data_ptr(),
            w.wqb_s.data_ptr(),
            w.cos_sin.data_ptr(),
            pos.data_ptr(),
            slot.data_ptr(),
            slot_words(slot),
            swa.data_ptr(),
            swa.stride(0),
            swa_block,
            swa_idx,
            swa_lens,
            topk,
            topk_lens,
            self.kt.data_ptr(),
            self.klen.data_ptr(),
            self.q.data_ptr(),
            self.normed.data_ptr(),
            self.qr.data_ptr(),
            *iqa,
            iw_bits,
            w.cos_sin.shape[0],
            self.iq.data_ptr(),
            self.iqs.data_ptr(),
            self.iw.data_ptr(),
            self.scratch(M).data_ptr(),
            self.peer.local,
            self.peer.addresses.data_ptr(),
            self.rank,
            self.epoch.data_ptr(),
            torch.cuda.current_stream(),
        )
        return (
            self.q[:M],
            self.res_mid[:M],
            self.post_a[:M],
            self.comb_a[:M],
            self.pre_a[:M],
        )

    def k2(
        self,
        aw: K2Weights,
        w: FFNWeights,
        rows: int,
        pos: torch.Tensor,  # [>= rows] int64, K1's positions
        swa: torch.Tensor,  # the layer's SWA cache: uint8 [pages, page bytes]
        swa_block: int,
        comp: torch.Tensor | None,  # its compressed cache, the same layout
        comp_block: int,
        outs: tuple | None = None,
        slot: torch.Tensor | None = None,  # an index source's: K1's slots and keys
        keys: tuple | None = None,
        index: bool = False,
        tok_par: bool = False,
    ):
        """Run K2 on K1's outputs: attention back, TP reduce, FFN seam, MoE and all-reduce.

        ``comp`` None: a window-only layer. ``index``: an index source; K2 first builds
        the key table from ``keys``, whose top-k the indexer selected after K1."""
        M = rows
        if not self.supports(M):
            raise RuntimeError(f"DSv4.1 mono decode: {M} rows")
        residual = self.res_mid[:M]
        post, comb, pre = self.post_a[:M], self.comb_a[:M], self.pre_a[:M]
        if outs is None:
            outs = self._outs(M, residual)
        out, res_out, post_out, comb_out, pre_out = outs
        bf, f32 = torch.bfloat16, torch.float32
        check_rows(
            "K2",
            M,
            self.device,
            {
                "out": (out, HIDDEN, bf),
                "res_out": (res_out, HC * HIDDEN, bf),
                "post_out": (post_out, HC, f32),
                "comb_out": (comb_out, HC * HC, f32),
                "pre_out": (pre_out, HC, f32),
            },
        )
        check_cache("K2 SWA cache", swa, swa_block)
        compressed = comp is not None
        if compressed:
            check_cache("K2 compressed cache", comp, comp_block)
        else:  # unread: the window's keys only
            comp, comp_block = swa, swa_block
        if not (pos.dtype == torch.int64 and pos.numel() >= M and pos.is_contiguous()):
            raise RuntimeError(f"K2 positions: {tuple(pos.shape)} {pos.dtype}")
        if index:
            if slot is None or keys is None:
                raise RuntimeError("K2 of an index source needs K1's slots and keys")
            table = (slot, *keys)
        else:  # unread
            table = (self.klen, self.klen, self.klen, self.no_topk, self.no_topk_lens)
        slot_ptr, *key_ptrs = (t.data_ptr() for t in table)
        _run_compiled(
            self.k2_kernel(M, compressed, index, tok_par),
            self.q.data_ptr(),
            swa.data_ptr(),
            swa.stride(0),
            swa_block,
            comp.data_ptr(),
            comp.stride(0),
            comp_block,
            self.kt.data_ptr(),
            self.klen.data_ptr(),
            slot_ptr,
            slot_words(table[0]),
            *key_ptrs,
            pos.data_ptr(),
            aw.sink.data_ptr(),
            self.qk_scale,
            aw.cos_sin.data_ptr(),
            aw.woa.data_ptr(),
            aw.woa_s.data_ptr(),
            aw.wob.data_ptr(),
            aw.wob_s.data_ptr(),
            self.zero_rec.data_ptr(),
            residual.data_ptr(),
            post.data_ptr(),
            comb.data_ptr(),
            pre.data_ptr(),
            w.hc_ffn_fn.data_ptr(),
            w.hc_ffn_scale.data_ptr(),
            w.hc_ffn_base.data_ptr(),
            w.ffn_norm.data_ptr(),
            res_out.data_ptr(),
            post_out.data_ptr(),
            comb_out.data_ptr(),
            pre_out.data_ptr(),
            w.gate_w.data_ptr(),
            w.bias.data_ptr(),
            w.w13.data_ptr(),
            w.w13_s.data_ptr(),
            w.w2.data_ptr(),
            w.w2_s.data_ptr(),
            w.sgu.data_ptr(),
            w.sgu_s.data_ptr(),
            w.sw2.data_ptr(),
            w.sw2_s.data_ptr(),
            out.data_ptr(),
            self.scratch(M).data_ptr(),
            self.peer.local,
            self.peer.addresses.data_ptr(),
            self.rank,
            self.epoch.data_ptr(),
            torch.cuda.current_stream(),
        )
        return outs
