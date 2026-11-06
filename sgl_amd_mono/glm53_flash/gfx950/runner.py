# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from vllm-project/vllm at ca328e297b (Apache-2.0):
# vllm/models/glm5next/amd/ops/mono/ffn/op.py
"""Host side of the GLM-5.3-Flash layer launch: one runtime per (width, kind),
shared by every covered layer of that kind in a step, and one op per layer
holding its weights; the tail launch's op."""

from __future__ import annotations

import inspect
import os
from functools import partial
from pathlib import Path
from typing import Optional, Union

import msgspec
import torch

from .config import HC_MULT, HIDDEN, KPOOL, VOCAB
from .layout import DSA_TAB, ROW_WORDS, STEP_TAB


class FlashFfnWeights(msgspec.Struct, frozen=True):
    """One layer's MoE-side tensors as the kernel reads them; ``*_sh_*`` is the
    shared expert, a slice of the routed tensors or a packed copy."""

    w_r: torch.Tensor  # router [288, HIDDEN] bf16, packed in 16 x 64 MFMA tiles
    bias: torch.Tensor  # [288] f32
    w_ug: torch.Tensor  # routed gate | up, MXFP4 bytes in AITER's (16, 16) shuffle
    s_ug: torch.Tensor  # their E8M0 scales, AITER's e8m0_shuffle
    w_dn: torch.Tensor
    s_dn: torch.Tensor
    w_sh_ug: torch.Tensor
    s_sh_ug: torch.Tensor
    w_sh_dn: torch.Tensor
    s_sh_dn: torch.Tensor
    hc_fn: torch.Tensor  # [24, 4 * HIDDEN] f32: the FFN mHC
    hc_scale: torch.Tensor  # [3] f32
    hc_base: torch.Tensor  # [24] f32
    g_norm: torch.Tensor  # [HIDDEN] bf16: post_attention_layernorm


class FlashDenseWeights(msgspec.Struct, frozen=True):
    """One dense layer's FFN tensors: the merged gate | up [2 I, HIDDEN] and down
    [HIDDEN, I], FP8 E4M3 with f32 128 x 128 block scales, row-major or in AITER's
    (16, 16) preshuffled layout (``shuffled``); the FFN mHC and its norm weight."""

    w_ug: torch.Tensor
    s_ug: torch.Tensor
    w_dn: torch.Tensor
    s_dn: torch.Tensor
    hc_fn: torch.Tensor
    hc_scale: torch.Tensor
    hc_base: torch.Tensor
    g_norm: torch.Tensor
    shuffled: bool = False


class FlashHeadWeights(msgspec.Struct, frozen=True):
    """The model's tail: the final RMSNorm weight [HIDDEN] bf16 and this rank's rows
    of the vocab-parallel LM head [VOCAB / TP, HIDDEN] bf16, row-major."""

    g_norm: torch.Tensor
    w_head: torch.Tensor


class FlashEmbedding(msgspec.Struct, frozen=True):
    """This rank's rows of the vocab-parallel input embedding [rows, HIDDEN] bf16,
    vocabulary ids ``start`` .. ``end``."""

    weight: torch.Tensor
    start: int
    end: int


class FlashKdaWeights(msgspec.Struct, frozen=True):
    """One KDA layer's attention-side tensors, every projection row-major bf16:
    ``w_in`` [3P + H + 2 * 128, HIDDEN] (q | k | v | beta | f_a | g_a), ``w_fb`` and
    ``w_gb`` [P, 128], ``w_o`` [HIDDEN, P]; ``conv_w`` [3P, 4], ``a_log`` [H],
    ``dt_bias`` [P] and ``w_on`` [128] f32."""

    hc_fn: torch.Tensor
    hc_scale: torch.Tensor
    hc_base: torch.Tensor
    g_norm: torch.Tensor
    w_in: torch.Tensor
    w_fb: torch.Tensor
    w_gb: torch.Tensor
    conv_w: torch.Tensor
    a_log: torch.Tensor
    dt_bias: torch.Tensor
    w_on: torch.Tensor
    w_o: torch.Tensor


class FlashDsaWeights(msgspec.Struct, frozen=True):
    """One sparse-attention layer's attention-side tensors. ``w_qkv`` [1536 + 512,
    HIDDEN] (q_a | kv_a), ``w_qb`` [H * 256, 1536] and ``w_o`` [HIDDEN, H * 256] are
    each FP8 with f32 128 x 128 block scales, or bf16 with no scale; ``w_idx`` bf16
    [128 + 32, HIDDEN] (wk | weights_proj), ``w_gate`` bf16 [128, HIDDEN]; q_a / kv_a
    RMSNorm weights; k LayerNorm weight and bias and kpool ``ape`` [4, 128] f32;
    ``w_iq`` bf16 [32 * 128, 1536]; ``w_uk`` bf16 [H, 512, 256] (W_UK^T, packed
    once); ``w_kvb`` bf16 [H * 512, 512] (kv_b_proj, W_UV rows at 256 ..).
    ``shuffled``: which FP8 weights are in AITER's (16, 16) preshuffled layout."""

    hc_fn: torch.Tensor
    hc_scale: torch.Tensor
    hc_base: torch.Tensor
    g_norm: torch.Tensor
    w_qkv: torch.Tensor
    s_qkv: Optional[torch.Tensor]
    w_idx: torch.Tensor
    w_gate: torch.Tensor
    g_qa: torch.Tensor
    g_kv: torch.Tensor
    ln_w: torch.Tensor
    ln_b: torch.Tensor
    ape: torch.Tensor
    w_qb: torch.Tensor
    s_qb: Optional[torch.Tensor]
    w_iq: torch.Tensor
    w_uk: torch.Tensor
    w_kvb: torch.Tensor
    w_o: torch.Tensor
    s_o: Optional[torch.Tensor]
    shuffled: tuple[bool, bool, bool] = (False, False, False)

    @property
    def fp8(self) -> tuple[bool, bool, bool]:
        """Which of q_a | kv_a, q_b, o_proj are FP8 (scales present)."""
        return tuple(sc is not None for sc in (self.s_qkv, self.s_qb, self.s_o))


class FlashDsaCaches(msgspec.Struct, frozen=True):
    """One sparse-attention layer's SGLang caches: the bf16 MLA latent cache
    [slots, row] (``row`` >= 512 elements), the uint8 index cache [index pages,
    64 pools x 132 B] (the pools' FP8 values in 16-pool x 16-byte tiles, then their
    f32 scales), and the bf16 kpool k and gate rings [request slots, 4, 128]."""

    kv_cache: torch.Tensor
    idx_cache: torch.Tensor
    tail_k: torch.Tensor
    tail_g: torch.Tensor


class FlashKdaState(msgspec.Struct, frozen=True):
    """One KDA layer's decode state: bf16 conv state [slots, 3, 3P] and f32
    recurrent state [slots, H, 128 (v), 128 (k)], slot strides in elements. A verify
    launch also writes SGLang's verify scratch: the f32 recurrent state
    [rows, steps, H, 128, 128] and the bf16 conv window [rows, steps, 3, 3P] after
    each draft token (a scratch row a request, its batch position)."""

    conv_state: torch.Tensor
    rec_state: torch.Tensor
    inter_rec: Optional[torch.Tensor] = None
    inter_conv: Optional[torch.Tensor] = None


# Rows of SGLang's DSA page table an index block spans: an index page holds 64
# pools (256 tokens), and the table holds one entry per 64 tokens.
PAGE_TOKENS = 64
INDEX_BLOCK_POOLS = 64
IDX_BT_GROUP = INDEX_BLOCK_POOLS * KPOOL // PAGE_TOKENS
# the KDA launch arguments: the attention-side mHC and KDA weights, the state
# and its indices and strides, the verify scratch and its strides
KDA_ARGS = 21


def compile_only(launch) -> None:
    """Build the launcher without dispatching its kernels."""
    # AITER probes the GPU when imported; test/unit imports this module on CPU
    from aiter.ops.flydsl.kernels.tensor_shim import _preload_compiled

    names = [n for n in inspect.signature(launch.func).parameters if n != "stream"]
    _preload_compiled(launch, *([0] * len(names)), torch.cuda.current_stream())


def ensure_writable_flydsl_cache() -> None:
    """Point FlyDSL at a writable cache (aiter sets a read-only one)."""
    cur = os.environ.get("FLYDSL_RUNTIME_CACHE_DIR")
    if cur and os.access(cur, os.W_OK):
        return
    path = Path.home() / ".flydsl" / "cache"
    path.mkdir(parents=True, exist_ok=True)
    os.environ["FLYDSL_RUNTIME_CACHE_DIR"] = str(path)


class FlashRuntime:
    """The compiled launch for one (width, kind) plus the state its layers share:
    scratch mailboxes, the symmetric peer buffer, and the guard block, whose word 0
    is the step epoch (``advance_step``)."""

    def __init__(self, key, rank: int, group, device: torch.device):
        from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled

        from sgl_amd_mono.common import guard
        from sgl_amd_mono.common.peer import PeerBuffer

        from .kernel import build_flash_kernel

        ensure_writable_flydsl_cache()
        self.key, self.width, self.rank = key, key.rows, rank
        self.launch, self.sc, self.sy = build_flash_kernel(key)
        self.dispatch = partial(_run_compiled, self.launch)
        self.peer = PeerBuffer(self.sy["_bytes"], group, rank, key.tp, device)
        self.peer.bytes.zero_()
        # the hand-off mailboxes, uncached as the peer buffer is: a line is read at its
        # home memory, not through the producer's L2
        self._scratch = PeerBuffer(self.sc["_bytes"], group, rank, 1, device)
        self.scratch = self._scratch.bytes
        self.epoch = torch.zeros(guard.GUARD_WORDS, dtype=torch.int32, device=device)
        # the tail's gathered logits; a layer's two output sets, by layer parity: a
        # layer never writes what it reads
        w = key.rows
        self.logits = (
            torch.empty(w, VOCAB, dtype=torch.float32, device=device)
            if key.head
            else None
        )
        self.outs = []
        if not key.head:
            self.outs = [
                (
                    torch.empty(w, HIDDEN, dtype=torch.bfloat16, device=device),
                    torch.empty(
                        w, HC_MULT, HIDDEN, dtype=torch.bfloat16, device=device
                    ),
                    torch.empty(w, HC_MULT, 1, dtype=torch.float32, device=device),
                    torch.empty(
                        w, HC_MULT, HC_MULT, dtype=torch.float32, device=device
                    ),
                )
                for _ in range(2)
            ]

    def set_deadline(self, seconds: float) -> None:
        """Every wait's deadline (below the server's watchdog timeout)."""
        from sgl_amd_mono.common import guard

        ticks = guard.deadline_ticks(seconds)
        lo = ticks & 0xFFFFFFFF
        self.epoch[guard.GUARD_DEADLINE] = lo - (1 << 32) if lo >= 1 << 31 else lo
        self.epoch[guard.GUARD_DEADLINE + 1] = ticks >> 32

    def precompile(self) -> None:
        compile_only(self.launch)

    def advance_step(self) -> None:
        self.epoch[:1].add_(1)

    def close(self) -> None:
        self.scratch = None
        self._scratch.close()
        self.peer.close()


def _check(name: str, t: torch.Tensor, shape: tuple, dtype: torch.dtype) -> None:
    if tuple(t.shape) != shape or t.dtype != dtype or not t.is_contiguous():
        raise ValueError(
            f"GLM-5.3-Flash monokernel needs {name} as contiguous {dtype} {shape}, "
            f"got {t.dtype} {tuple(t.shape)}"
        )


class FlashLayerOp:
    """One covered layer: its weights and its ordinal among its kind's layers. A
    dense layer takes ``FlashDenseWeights``; the step's first launch (a ``tokens``
    runtime) also this rank's embedding rows."""

    def __init__(
        self,
        runtime: FlashRuntime,
        weights: Union[FlashFfnWeights, FlashDenseWeights],
        layer: int,
        kda: Optional[FlashKdaWeights] = None,
        dsa: Optional[FlashDsaWeights] = None,
        embedding: Optional[FlashEmbedding] = None,
    ):
        if (kda is None) == runtime.key.kda:
            raise ValueError("a KDA runtime needs KDA weights, and only it")
        if (dsa is None) == runtime.key.dsa:
            raise ValueError(
                "a sparse-attention runtime needs DSA weights, and only it"
            )
        key = runtime.key
        if isinstance(weights, FlashDenseWeights) != key.dense:
            raise ValueError("a dense runtime needs dense FFN weights, and only it")
        if key.dense and weights.shuffled != key.dense_shuffled:
            raise ValueError(
                f"dense weights (shuffled {weights.shuffled}) on another runtime"
            )
        if (embedding is None) == key.tokens:
            raise ValueError(
                "a launch from the tokens needs the embedding, and only it"
            )
        if dsa is not None and (dsa.fp8, dsa.shuffled) != (
            (key.qkv_fp8, key.qb_fp8, key.o_fp8),
            (key.qkv_shuffled, key.qb_shuffled, key.o_shuffled),
        ):
            raise ValueError(
                f"DSA weights (fp8 {dsa.fp8}, shuffled {dsa.shuffled}) on another runtime"
            )
        self.rt, self.w, self.layer, self.kda, self.dsa = (
            runtime,
            weights,
            layer,
            kda,
            dsa,
        )
        # the layer's argument table (``DSA_TAB`` order), rewritten outside capture
        # when a cache moves; the page table is a launch argument
        self.tab: Optional[torch.Tensor] = None
        self._tab_host: list[int] = []
        if dsa is not None:
            self.tab = torch.zeros(
                len(DSA_TAB), dtype=torch.int64, device=dsa.w_qkv.device
            )
        # a launch from the tokens: its step-input table (``STEP_TAB`` order) per row
        # count, each graph's own; SGLang keeps the step's inputs in buffers of
        # fixed address per batch size
        self.embedding = embedding
        # step-input tables by content: graphs read the ones made by the eager step
        # before their capture, and an eager step never rewrites those
        self.step_tabs: dict[tuple, torch.Tensor] = {}
        self.captured: set[tuple] = set()

    def _dsa_table(self, caches: FlashDsaCaches) -> torch.Tensor:
        d, c = self.dsa, caches
        kv, idx = c.kv_cache, c.idx_cache
        values = {
            **{
                name: 0 if getattr(d, name) is None else getattr(d, name).data_ptr()
                for name in DSA_TAB[:16]
            },
            "kv_cache": kv.data_ptr(),
            "idx_cache": idx.data_ptr(),
            "tail_k": c.tail_k.data_ptr(),
            "tail_g": c.tail_g.data_ptr(),
            "idx_bt_group": IDX_BT_GROUP,
            "kv_block_elems": PAGE_TOKENS * kv.stride(0),
            "kv_block_tokens": PAGE_TOKENS,
            "kv_row_elems": kv.stride(0),
            "idx_block_bytes": idx.stride(0),
            "idx_block_pools": INDEX_BLOCK_POOLS,
            "tail_block_elems": c.tail_k.stride(0),
        }
        host = [values[name] for name in DSA_TAB]
        if host != self._tab_host:
            # every graph of this width reads the one table: set it outside capture
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "the sparse-attention table changed during a graph capture"
                )
            self.tab.copy_(torch.tensor(host, dtype=torch.int64))
            self._tab_host = host
        return self.tab

    def _step_table(self, step: dict) -> torch.Tensor:
        """The step-input table of ``step``: its token ids, positions, MLA slots and
        request rows (int32 or int64 [S], S at most the width) and the KDA slot table
        (int32 [width]) and row table (int32 [width, ROW_WORDS]) to write."""
        e, w = self.embedding, self.rt.width
        n = step["ids"].shape[0]
        if not 1 <= n <= w:
            raise ValueError(f"a launch of width {w} takes 1 to {w} rows, got {n}")
        values = {
            "emb": e.weight.data_ptr(),
            "vocab_lo": e.start,
            "vocab_hi": e.end,
            "n_rows": n,
        }
        for name, words in (
            ("ids", "ids_words"),
            ("positions", "positions_words"),
            ("out_cache_loc", "loc_words"),
            ("req_pool_indices", "rpi_words"),
        ):
            t = step[name]
            if (
                t.dtype not in (torch.int32, torch.int64)
                or tuple(t.shape) != (n,)
                or t.stride(0) != 1
            ):
                raise ValueError(
                    f"GLM-5.3-Flash monokernel needs {name} as int32 or int64 "
                    f"[{n}], got {t.dtype} {tuple(t.shape)}"
                )
            values[name], values[words] = t.data_ptr(), t.element_size() // 4
        _check("slots", step["slots"], (w,), torch.int32)
        _check("rows", step["rows"], (w, ROW_WORDS), torch.int32)
        values["slots"] = step["slots"].data_ptr()
        values["rows"] = step["rows"].data_ptr()
        host = tuple(values[name] for name in STEP_TAB)
        tab = self.step_tabs.get(host)
        if torch.cuda.is_current_stream_capturing():
            if tab is None:
                raise RuntimeError(
                    "a graph captured a step whose inputs no eager step used first"
                )
            self.captured.add(host)
            return tab
        if tab is None:
            rows = STEP_TAB.index("n_rows")
            for old in [h for h in self.step_tabs if h[rows] == n]:
                if old not in self.captured:
                    del self.step_tabs[old]
            tab = torch.tensor(host, dtype=torch.int64, device=e.weight.device)
            self.step_tabs[host] = tab
        return tab

    def launch(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
        state: Optional[FlashKdaState] = None,
        state_idx: Optional[torch.Tensor] = None,
        caches: Optional[FlashDsaCaches] = None,
        rows: Optional[torch.Tensor] = None,
        page_table: Optional[torch.Tensor] = None,
        step: Optional[dict] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """The previous layer's (x [S, HIDDEN], residual [S, HC, HIDDEN], post
        [S, HC, 1], comb [S, HC, HC]), with its FFN post still owed. A KDA op takes
        its layer's state and ``state_idx`` int32 [S] (not positive: padding); a
        sparse-attention op its caches, the step's ``rows`` int32 [S, ROW_WORDS]
        and SGLang's int32 64-token page table. The step's first launch (from the
        tokens) reads ``step`` (``_step_table``) instead of x, and ``state_idx`` is
        SGLang's KDA slot indices of the step's rows (int32, at most the width).
        Returns the layer's (x, residual, post, comb), its own FFN post owed."""
        rt, w, S = self.rt, self.w, self.rt.width
        _check("x", x, (S, HIDDEN), torch.bfloat16)
        _check("residual", residual, (S, HC_MULT, HIDDEN), torch.bfloat16)
        _check("post", post, (S, HC_MULT, 1), torch.float32)
        _check("comb", comb, (S, HC_MULT, HC_MULT), torch.float32)
        x_out, res_out, post_out, comb_out = rt.outs[self.layer % 2]
        p = lambda t: t.data_ptr()  # noqa: E731
        kda_args = [0] * KDA_ARGS
        dsa_args = [0, 0, 0, 0]
        if self.dsa is not None:
            if caches is None or rows is None or page_table is None:
                raise ValueError(
                    "a sparse-attention launch needs caches, rows and a page table"
                )
            _check("rows", rows, (S, ROW_WORDS), torch.int32)
            if page_table.dtype != torch.int32 or page_table.stride(1) != 1:
                raise ValueError(
                    "GLM-5.3-Flash monokernel needs an int32 page table with contiguous "
                    f"rows, got {page_table.dtype} {page_table.stride()}"
                )
            d = self.dsa
            kda_args[:4] = [p(d.hc_fn), p(d.hc_scale), p(d.hc_base), p(d.g_norm)]
            dsa_args = [
                p(self._dsa_table(caches)),
                p(rows),
                p(page_table),
                page_table.stride(0),
            ]
        if self.kda is not None:
            if state is None or state_idx is None:
                raise ValueError("a KDA launch needs its state and state indices")
            n = step["ids"].shape[0] if self.embedding is not None else S
            _check("state_idx", state_idx, (n,), torch.int32)
            k = self.kda
            inter = [0, 0, 0, 0]
            if self.rt.key.verify > 1:
                ir, ic = state.inter_rec, state.inter_conv
                if ir is None or ic is None:
                    raise ValueError("a verify launch needs the verify scratch")
                inter = [p(ir), p(ic), ir.stride(0), ic.stride(0)]
            kda_args = [
                *map(p, (k.hc_fn, k.hc_scale, k.hc_base, k.g_norm, k.w_in, k.w_fb)),
                *map(p, (k.w_gb, k.conv_w, k.a_log, k.dt_bias, k.w_on, k.w_o)),
                p(state.conv_state),
                p(state.rec_state),
                p(state_idx),
                state.conv_state.stride(0),
                state.rec_state.stride(0),
                *inter,
            ]
        if self.embedding is not None:
            dsa_args = [p(self._step_table(step)), 0, 0, 0]
        if self.rt.key.dense:
            ffn_args = [0, 0, *map(p, (w.w_ug, w.s_ug, w.w_dn, w.s_dn)), 0, 0, 0, 0]
        else:
            ffn_args = [
                *map(p, (w.w_r, w.bias, w.w_ug, w.s_ug, w.w_dn, w.s_dn)),
                *map(p, (w.w_sh_ug, w.s_sh_ug, w.w_sh_dn, w.s_sh_dn)),
            ]
        rt.dispatch(
            p(x),
            p(residual),
            p(post),
            p(comb),
            p(w.hc_fn),
            p(w.hc_scale),
            p(w.hc_base),
            p(w.g_norm),
            p(res_out),
            p(post_out),
            p(comb_out),
            p(x_out),
            *ffn_args,
            *kda_args,
            *dsa_args,
            p(rt.scratch),
            rt.peer.local,
            p(rt.peer.addresses),
            p(rt.epoch),
            rt.rank,
            self.layer,
            torch.cuda.current_stream(),
        )
        return x_out, res_out, post_out, comb_out


class FlashHeadOp:
    """The tail launch: the terminal layer's pending mHC post, the mean of the four
    streams, the final norm, the LM head and the logits' all-gather; ``layer`` is its
    ordinal among the runtime's launches a step."""

    def __init__(
        self, runtime: FlashRuntime, weights: FlashHeadWeights, layer: int = 0
    ):
        if not runtime.key.head:
            raise ValueError("the tail needs a head runtime")
        self.rt, self.w, self.layer = runtime, weights, layer

    def launch(self, x, residual, post, comb) -> torch.Tensor:
        """The terminal layer's (x, residual, post, comb); returns the gathered f32
        logits [S, VOCAB] (the runtime's buffer)."""
        rt, w, S = self.rt, self.w, self.rt.width
        _check("x", x, (S, HIDDEN), torch.bfloat16)
        _check("residual", residual, (S, HC_MULT, HIDDEN), torch.bfloat16)
        _check("post", post, (S, HC_MULT, 1), torch.float32)
        _check("comb", comb, (S, HC_MULT, HC_MULT), torch.float32)
        p = lambda t: t.data_ptr()  # noqa: E731
        rt.dispatch(
            p(x),
            p(residual),
            p(post),
            p(comb),
            0,
            0,
            0,
            p(w.g_norm),
            0,
            0,
            0,
            p(rt.logits),
            p(w.w_head),
            *[0] * 9,
            *[0] * KDA_ARGS,
            *[0] * 4,
            p(rt.scratch),
            rt.peer.local,
            p(rt.peer.addresses),
            p(rt.epoch),
            rt.rank,
            self.layer,
            torch.cuda.current_stream(),
        )
        return rt.logits
