# SPDX-License-Identifier: Apache-2.0
"""The static verify graph's accept as the verify head (``gfx950/head``): the greedy
accept SGLang's graph folds, and the all-sampled accept (accept_sampling_sharded, its
simulated bonus) SGLang runs eagerly after the graph, in four launches inside the graph
with the ranks' partials swapped through peer buffers. Checked against SGLang's greedy
tail at its first eager call on full logits on every rank. ``SGLANG_AMD_MONO_DSV41_HEAD=1``
turns it on; ``SGLANG_AMD_MONO_DSV41_HEAD_SHARDS=1`` also has the steps it serves verify
on this rank's vocab shard (no logits all-gather; SGLang gives shards only without a
softcap, so the head reads them as they are)."""

from __future__ import annotations

import logging

import torch

from sgl_amd_mono.deepseek_v41.glue import agreed, flag, route

_LOGGER = logging.getLogger(__name__)

ENV = "SGLANG_AMD_MONO_DSV41_HEAD"
SHARDS_ENV = ENV + "_SHARDS"
VERIFY = "sglang.srt.speculative.dspark_components.dspark_verify"
DEADLINE_S = 10.0

# by file under sglang/ (test_contract.py pins them)
SGLANG_READS = {
    "srt/speculative/dspark_components/dspark_verify.py": (
        "DsparkVerifyEpilogue",
        "_accept_static_greedy",
        "serves_sampled_fold",
        "begin_fold",
        "wants_vocab_shards",
        "serves_shards",
        "shard_step",
        "max_bs",
        "stride",
        "gamma",
        "commit_ctx",
        "simulated_correct_len",
        "sim_seed_buf",
        "correct_len_buf",
        "bonus_buf",
        "cap_trim_lens_buf",
        "commit_lens_buf",
        "new_seq_lens_buf",
        "out_tokens_buf",
        "gated_commit_lens_buf",
        "block_accept_lens_buf",
        "TargetVerifyExecutor",
        "accept_and_finalize",
        "verify_epilogue",
        "folded_accept",
    ),
    "srt/speculative/dspark_components/dspark_draft.py": (
        "DraftBlockResult",
        "corrected_logits",
        "temperatures",
        "greedy_draft",
        "vocab_shard",
        "noise",
    ),
    "srt/models/dspark.py": ("VocabShard", "start", "width", "markov_head"),
    "srt/distributed/parallel_state.py": (
        "get_tp_group",
        "cpu_group",
        "rank_in_group",
        "world_size",
    ),
    "srt/sampling/sampling_batch_info.py": (
        "is_any_greedy",
        "need_top_k_sampling",
        "need_top_p_sampling",
        "need_min_p_sampling",
        "has_watermark_candidates",
        "sampling_mask_batch_indices",
    ),
}

_KEY = "_sgl_amd_mono_verify_head"


class _Head:
    """One epilogue's head: its buffers, its exchange and the step's staged inputs."""

    def __init__(self, epi, io):
        from sglang.srt.distributed.parallel_state import get_tp_group

        from sgl_amd_mono.deepseek_v41.gfx950 import head

        group = get_tp_group()
        device = io.logits.device
        self.ops = head.ops()
        self.rank, self.tp = int(group.rank_in_group), int(group.world_size)
        shard = None
        ctx = epi.commit_ctx
        lm_head = getattr(ctx.draft_model, "lm_head", None) if ctx is not None else None
        # the first eager call may be on vocab shards: the vocabulary from the LM head
        self.vocab = int(getattr(lm_head, "org_vocab_size", io.logits.shape[1]))
        if ctx is not None:
            markov = getattr(ctx.draft_model, "markov_head", None)
            shard = getattr(markov, "vocab_shard", None)
        if shard is not None:
            self.start, self.width = int(shard.start), int(shard.width)
        else:
            assert self.vocab % (4 * self.tp) == 0, self.vocab
            self.width = self.vocab // self.tp
            self.start = self.rank * self.width
        assert self.start % 4 == 0 and self.start + self.width <= self.vocab
        self.max_bs, self.slots = int(epi.max_bs), int(epi.stride)
        assert self.max_bs <= head.MAX_BS and self.slots <= head.MAX_SLOTS
        self.padded = int(getattr(lm_head, "num_embeddings_per_partition", self.width))
        f32 = dict(dtype=torch.float32, device=device)
        self.part = torch.empty(head.part_words(self.max_bs, self.slots, self.width), **f32)
        self.temps = torch.zeros(self.max_bs, **f32)
        self.sim_temps = torch.zeros(self.max_bs, **f32)
        self.draft = torch.zeros(self.max_bs, self.slots - 1, self.padded, **f32)
        # the head reads the draft's logits where they are: their address, kept alive in _draft_ref
        self._draft_ptr = self.draft.data_ptr()
        self._draft_ref = self.draft
        self.draft_addr = torch.full((1,), self._draft_ptr, dtype=torch.int64, device=device)
        self.key = torch.zeros(1, dtype=torch.int64, device=device)
        self.offset = torch.zeros(1, dtype=torch.int64, device=device)
        self._key = None
        self._greedy_bs: int | None = None
        self.exchange = head.Exchange(group.cpu_group, self.rank, self.tp, device)

    def on_shard(self, io) -> bool:
        """Whether the target logits are this rank's vocab shard (else the full row)."""
        cols = int(io.logits.shape[1])
        if cols == self.vocab:
            return False
        assert cols == self.width, (cols, self.width, self.vocab)
        return True

    def run(self, epi, io, verify_lens, gate, gated: bool):
        bs = io.candidates.shape[0]
        e = epi
        col0 = 0 if self.on_shard(io) else self.start
        self.ops.verify_head(
            io.logits, col0, self.start, self.width, self.vocab, 0.0, self.draft,
            self.draft_addr, io.candidates, self.temps, self.sim_temps, epi.simulated_correct_len,
            epi.sim_seed_buf, self.key, self.offset, io.seq_lens, verify_lens, gate,
            self.part, e.correct_len_buf, e.bonus_buf, e.cap_trim_lens_buf,
            e.commit_lens_buf, e.new_seq_lens_buf, e.out_tokens_buf,
            e.gated_commit_lens_buf if gated else None, e.block_accept_lens_buf,
            self.exchange.calls, self.exchange.peers, self.rank, DEADLINE_S,
        )  # fmt: skip
        return e.gated_commit_lens_buf[:bs] if gated else None

    def serves(self, sampling_info, draft_block, block_verification: bool) -> str | None:
        """Why the head cannot take this all-sampled step, or None."""
        if block_verification:
            return "block verification"
        if draft_block.greedy_draft or draft_block.noise is None:
            return "a greedy draft"
        shard = draft_block.vocab_shard
        if shard is None or (int(shard.start), int(shard.width)) != (self.start, self.width):
            return "the draft's vocab is not this head's shard"
        logits = draft_block.corrected_logits
        if (
            logits is None
            or logits.dtype != torch.float32
            or logits.dim() != 3
            or logits.shape[1] != self.slots - 1
            or logits.shape[2] != self.padded
            or logits.shape[0] > self.max_bs
            or not logits.is_contiguous()
        ):
            return "draft logits not [bs, gamma, padded] f32"
        info = sampling_info
        if info.is_any_greedy or info.need_top_k_sampling or info.need_top_p_sampling:
            return "greedy rows or top-k / top-p"
        if info.need_min_p_sampling or info.has_watermark_candidates:
            return "min-p or watermark"
        if info.sampling_mask_batch_indices is not None:
            return "sampling masks"
        return None

    def stage(self, bs: int, sampled: bool, draft_block, sim_info) -> None:
        if not sampled:
            if self._greedy_bs != bs:
                self.temps.zero_()
                self.sim_temps.zero_()
                self._greedy_bs = bs
            return
        self._greedy_bs = None
        self.temps[:bs].copy_(draft_block.temperatures.view(-1)[:bs])
        if sim_info is not None:
            self.sim_temps[:bs].copy_(sim_info.temperatures.view(-1)[:bs])
        logits = draft_block.corrected_logits
        if logits.data_ptr() != self._draft_ptr:
            self._draft_ptr = logits.data_ptr()
            self.draft_addr.fill_(self._draft_ptr)
        self._draft_ref = logits
        key = int(draft_block.noise.key)
        if key != self._key:
            self.key.fill_(key)
            self._key = key
        self.offset.copy_(draft_block.noise.offset.view(-1)[:1])


def _head(epi):
    h = epi.__dict__.get(_KEY)
    return h if isinstance(h, _Head) else None


def _ensure(epi, io) -> _Head | None:
    """The epilogue's head, built and checked at its first eager static accept."""
    if _KEY in epi.__dict__:
        return _head(epi)
    if torch.cuda.is_current_stream_capturing():
        return None
    state = {}

    def why():
        state["head"] = _Head(epi, io)
        return None

    ok = agreed("verify_head_build", why)
    epi.__dict__[_KEY] = state.get("head") if ok else False
    return _head(epi)


def _accept_static_greedy(orig, self, io, *, verify_lens, gate, gated):
    h = _ensure(self, io)
    if h is None:
        return orig(self, io, verify_lens=verify_lens, gate=gate, gated=gated)
    # SGLang's tail reads full logits: the check waits for a full-logits call
    if (
        not torch.cuda.is_current_stream_capturing()
        and not self.__dict__.get(_KEY + "_checked")
        and not h.on_shard(io)
    ):
        self.__dict__[_KEY + "_checked"] = True
        bs = io.candidates.shape[0]
        out = orig(self, io, verify_lens=verify_lens, gate=gate, gated=gated)
        want = [
            b[:bs].clone()
            for b in (
                self.correct_len_buf, self.bonus_buf, self.commit_lens_buf,
                self.new_seq_lens_buf, self.out_tokens_buf,
            )
        ]  # fmt: skip
        want_gated = None if out is None else out.clone()
        h.stage(bs, False, None, None)
        got_gated = h.run(self, io, verify_lens, gate, gated)
        got = [
            b[:bs]
            for b in (
                self.correct_len_buf, self.bonus_buf, self.commit_lens_buf,
                self.new_seq_lens_buf, self.out_tokens_buf,
            )
        ]  # fmt: skip

        def why():
            if not all(torch.equal(a.long(), b.long()) for a, b in zip(want, got)):
                return "the greedy accept disagrees with SGLang's"
            if (want_gated is None) != (got_gated is None) or (
                want_gated is not None and not torch.equal(want_gated, got_gated)
            ):
                return "the gated commit lengths disagree"
            return None

        if not agreed("verify_head", why):
            self.__dict__[_KEY].exchange.close()
            self.__dict__[_KEY] = False
            return out
        return got_gated
    return h.run(self, io, verify_lens, gate, gated)


def _serves_sampled_fold(orig, self, *, sampling_info, draft_block, block_verification):
    h = _head(self)
    if h is None or self.__dict__.get(_KEY + "_checked") is not True:
        return orig(
            self,
            sampling_info=sampling_info,
            draft_block=draft_block,
            block_verification=block_verification,
        )
    # every rank sees the same batch: no agreement needed
    why = h.serves(sampling_info, draft_block, block_verification)
    if why is None and not self.__dict__.get(_KEY + "_served"):
        self.__dict__[_KEY + "_served"] = True
        if h.rank == 0:
            _LOGGER.info("DSpark verify head: an all-sampled step accepts in the graph")
    elif why is not None and self.__dict__.get(_KEY + "_why") != why:
        self.__dict__[_KEY + "_why"] = why
        if h.rank == 0:
            _LOGGER.info("DSpark verify head: SGLang's eager accept (%s)", why)
    return why is None


def _begin_fold(orig, self, bs, *, sampled, draft_block, simulate_bonus_sampling_info):
    orig(
        self,
        bs,
        sampled=sampled,
        draft_block=draft_block,
        simulate_bonus_sampling_info=simulate_bonus_sampling_info,
    )
    h = _head(self)
    if h is not None:
        h.stage(bs, sampled, draft_block, simulate_bonus_sampling_info)
        if self.__dict__.get(_KEY + "_shards"):
            k = "shards" if self.shard_step else "full"
            _VARIANTS[k] += 1
            total = _VARIANTS["shards"] + _VARIANTS["full"]
            if h.rank == 0 and (_VARIANTS[k] == 1 or total % 1000 == 0):
                _LOGGER.info(
                    "DSpark verify head: verify logits variant shards %d / full %d folded steps",
                    _VARIANTS["shards"], _VARIANTS["full"],
                )  # fmt: skip


_VARIANTS = {"shards": 0, "full": 0}


def _wants_vocab_shards(orig, self) -> bool:
    self.__dict__[_KEY + "_shards"] = True
    return True


def _serves_shards(orig, self) -> bool:
    return _head(self) is not None and self.__dict__.get(_KEY + "_checked") is True


_CHECK = {"left": 0, "steps": 0, "same": 0, "sampled": 0}


def _accept_and_finalize(orig, self, **kw):
    """SGLANG_AMD_MONO_DSV41_HEAD_CHECK=N: the first N folded steps also take SGLang's
    eager accept on the same logits, compared with the head's."""
    got = orig(self, **kw)
    epi = self.verify_epilogue
    if not kw["folded_accept"] or _CHECK["left"] <= 0 or epi is None or _head(epi) is None:
        return got
    if getattr(epi, "shard_step", False):
        return got  # SGLang's eager accept reads full logits
    _CHECK["left"] -= 1
    want = orig(self, **{**kw, "folded_accept": False})
    names = ("correct_len", "bonus", "commit_lens", "new_seq_lens", "out_tokens")
    bad = [n for n in names if not torch.equal(getattr(want, n).long(), getattr(got, n).long())]
    h = _head(epi)
    _CHECK["steps"] += 1
    _CHECK["same"] += not bad
    _CHECK["sampled"] += h._greedy_bs is None
    if h.rank == 0 and (bad or _CHECK["left"] == 0 or _CHECK["steps"] % 100 == 0):
        _LOGGER.info(
            "DSpark verify head check: %d / %d folded steps identical (%d sampled)%s",
            _CHECK["same"], _CHECK["steps"], _CHECK["sampled"],
            f"; bs {kw['bs']} differs in {bad}: head {got.correct_len.tolist()} / "
            f"{got.bonus.tolist()} vs {want.correct_len.tolist()} / {want.bonus.tolist()}"
            if bad else "",
        )  # fmt: skip
    return got


def install() -> None:
    if not flag(ENV, "0"):
        return
    import importlib
    import os

    verify = importlib.import_module(VERIFY)
    epi = verify.DsparkVerifyEpilogue
    if not hasattr(epi, "serves_sampled_fold"):
        _LOGGER.info("DSV4.1 glue verify_head: SGLang has no sampled fold hook")
        return
    route(epi, "_accept_static_greedy", _accept_static_greedy)
    route(epi, "serves_sampled_fold", _serves_sampled_fold)
    route(epi, "begin_fold", _begin_fold)
    if flag(SHARDS_ENV, "0"):
        if hasattr(epi, "wants_vocab_shards"):
            route(epi, "wants_vocab_shards", _wants_vocab_shards)
            route(epi, "serves_shards", _serves_shards)
        else:
            _LOGGER.info("DSV4.1 glue verify_head: SGLang has no vocab-shard hooks")
    _CHECK["left"] = int(os.environ.get(ENV + "_CHECK", "0"))
    if _CHECK["left"] > 0:
        route(verify.TargetVerifyExecutor, "accept_and_finalize", _accept_and_finalize)
