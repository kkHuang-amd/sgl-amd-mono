# SPDX-License-Identifier: Apache-2.0
"""A sampled DSpark draft's Markov block in two launches a step (``gfx950/markov``), offered
through SGLang's draft sampler entry; checked against SGLang's ``draft_markov_block`` before
capture on every rank. ``SGLANG_AMD_MONO_DSV41_MARKOV=0`` keeps SGLang's steps."""

from __future__ import annotations

import functools
import logging

import torch

_LOGGER = logging.getLogger(__name__)

ENV = "SGLANG_AMD_MONO_DSV41_MARKOV"

# by file under sglang/ (test_contract.py pins them)
SGLANG_READS = {
    "kernels/plugin_kernels/nvidia/spec.py": ("draft_sampler", "graph_variants"),
    "srt/kernel_plugins/nvidia/binding.py": (
        "built_megakernel",
        '"_megakernel"',
        # _InstalledMegakernel's fields, which _Entries stands in for
        "tensor_entry",
        "binding",
    ),
    "srt/models/deepseek_v4_dspark.py": (
        "DSparkV4MarkovHead",
        "compute_base_logits",
        "collapse_hc_head",
        "_use_fp32_lm_head",
        "_opt_markov_w2_tp_shard",
        "markov_head",
        "markov_w1",
        "markov_w2",
        "_tp_shard",
        "_shard_group",
        "org_vocab_start",
        "org_vocab_end",
        "_opt_markov_w2_bf16",
        "step_logits_dtype",
        "lm_head",
        "num_embeddings_per_partition",
    ),
    "srt/models/dspark.py": (
        "vocab_shard",
        "gather_and_crop_vocab",
        "project_through_lm_head",
    ),
    "srt/speculative/dspark_components/dspark_draft.py": (
        "draft_markov_block",
        "draft_markov_block(markov_head)",
        "draft_markov_block(base_logits)",
        "draft_markov_block(anchor_tokens)",
        "draft_markov_block(draft_hidden)",
        "draft_markov_block(step_temperatures)",
        "draft_markov_block(noise)",
        "draft_markov_block(tp_sync)",
        "draft_markov_block(logits_out)",
        "draft_markov_block(greedy_draft)",
        "draft_markov_block(sharded_accept)",
        "DraftNoise",
        "key",
        "offset",
    ),
}


class _Entries:
    """What SGLang reads of a built draft through ``built_megakernel``: the draft
    sampler only (no covered step, no graph variants, no EAGLE entries)."""

    widths: tuple = ()
    draft_loop = draft_extend = begin_draft_round = graph_variants = None
    # current SGLang reads .built and .prepare of the entry, older SGLang the entry itself
    prepare = None
    # SGLang's decode graph capture reads .tensor_entry (None: no tensor entry) and
    # prepare_megakernel .binding, as of _InstalledMegakernel's tensor_entry field
    tensor_entry = binding = None

    def __init__(self, draft_sampler):
        self.draft_sampler = draft_sampler

    @property
    def built(self):
        return self

    def summary(self) -> str:
        return "the sampled DSpark draft's Markov block"


def install(model, deadline_s: float) -> None:
    """Offer the kernels' Markov block to the DSpark draft ``model``'s sampler."""
    from sgl_amd_mono.deepseek_v41.glue import flag

    if not flag(ENV) or "_megakernel" in model.__dict__:
        return
    model.__dict__["_megakernel"] = _Entries(
        functools.partial(_factory, model, deadline_s)
    )


def _why_not(model, step_logits) -> str | None:
    from sgl_amd_mono.deepseek_v41.gfx950 import markov

    head = getattr(model, "markov_head", None)
    if type(head).__name__ != "DSparkV4MarkovHead":
        return f"a {type(head).__name__}"
    shard, group = getattr(head, "_tp_shard", None), getattr(head, "_shard_group", None)
    if shard is None or group is None or head.vocab_shard is None:
        return "markov_w2 not TP-sharded"
    if not 1 < group.world_size <= 8:
        return f"TP {group.world_size}"
    w1, w2 = head.markov_w1.weight, head.markov_w2.weight
    if w1.dtype != torch.bfloat16 or w2.dtype != torch.bfloat16:
        return f"markov weights {w1.dtype} / {w2.dtype}, not bf16"
    if w1.shape[1] != markov.DIM or w2.shape[1] != markov.DIM:
        return f"markov rank {w1.shape[1]}, the kernels' is {markov.DIM}"
    if head.step_logits_dtype(torch.bfloat16) != torch.float32:
        return "step logits not fp32"
    part = int(model.lm_head.num_embeddings_per_partition)
    if step_logits is None or step_logits.dtype != torch.float32:
        return "no fp32 step logits"
    if step_logits.shape[-1] != part:
        return "step logits gathered whole, not on the shards"
    return None


class _Sampler:
    def __init__(
        self, model, deadline_s, gamma, noise, draft_tokens, step_logits, temps
    ):
        from sgl_amd_mono.deepseek_v41.gfx950 import markov

        head = model.markov_head
        shard, group = head._tp_shard, head._shard_group
        self.model, self.gamma, self.noise = model, gamma, noise
        self.out, self.step_logits, self.temps = draft_tokens, step_logits, temps
        self.deadline_s = float(deadline_s)
        self.vocab_start = int(shard.org_vocab_start)
        self.embed = head.markov_w1.weight.contiguous()
        self.head = head.markov_w2.weight[shard.org_vocab_start : shard.org_vocab_end]
        assert self.head.is_contiguous()
        device = self.head.device
        part = int(model.lm_head.num_embeddings_per_partition)
        self.part = torch.empty(
            markov.MAX_SEQUENCES,
            markov.blocks(part),
            2,
            dtype=torch.float32,
            device=device,
        )
        self.exchange = markov.Exchange(
            group.cpu_group, group.rank_in_group, group.world_size, device
        )
        from sgl_amd_mono.deepseek_v41.glue import flag

        # SGLANG_AMD_MONO_DSV41_MARKOV_PERSIST=1: every step of a block in one launch
        self.persist = None
        if flag("SGLANG_AMD_MONO_DSV41_MARKOV_PERSIST", "0"):
            self.persist = markov.BlockExchange(
                group.cpu_group, group.rank_in_group, group.world_size, device
            )
        self.ops = markov.ops()

    def block(self, base_logits, anchor, temps, noise, out, step_logits):
        """Every step of a [bs, gamma, padded] base logits block: tokens to ``out``
        ([bs * gamma]), the step logits to ``step_logits`` ([bs, gamma, padded])."""
        if self.persist is not None:
            b = self.persist
            self.ops.markov_sample_block(
                base_logits, self.embed, self.head, anchor, step_logits, temps, noise.offset,
                int(noise.key), self.vocab_start, b.partials, out, self.gamma, b.calls, b.peers,
                b.rank, self.deadline_s,
            )  # fmt: skip
            return
        x = self.exchange
        for s in range(self.gamma):
            self.ops.markov_sample_step(
                base_logits[:, s], self.embed, self.head, anchor,
                None if step_logits is None else step_logits[:, s], temps, noise.offset,
                int(noise.key), self.vocab_start, s, self.part, out, self.gamma, x.tok,
                x.calls, x.peers, x.rank, s == self.gamma - 1, self.deadline_s,
            )  # fmt: skip

    def step(self, hidden_states, anchor_tokens):
        """SGLang's base logits and Markov block of a bs-row draft block."""
        base_logits, confidence_features = self._base_logits(hidden_states)
        bs = anchor_tokens.shape[0]
        base = base_logits.view(bs, self.gamma, -1)
        out = self.out[: bs * self.gamma]
        self.block(
            base, anchor_tokens.long(), self.temps[:bs], self.noise, out,
            self.step_logits[:bs],
        )  # fmt: skip
        return out.view(bs, self.gamma), confidence_features

    def _base_logits(self, hidden_states):
        """``compute_base_logits``; from the draft forward's normed rows when it left them
        (``draft.DRAFT_TAIL``) for these very rows."""
        from sgl_amd_mono.deepseek_v41.draft import TAIL_KEY

        model = self.model
        tail = model.__dict__.get(TAIL_KEY)
        if (
            tail is None
            or tail[0].data_ptr() != hidden_states.data_ptr()
            or tail[0].numel() != hidden_states.numel()
            or getattr(model, "_use_fp32_lm_head", False)
        ):
            return model.compute_base_logits(hidden_states)
        from sglang.srt.models.dspark import (
            gather_and_crop_vocab,
            project_through_lm_head,
        )

        x_post_hc = model.collapse_hc_head(hidden_states)
        logits = project_through_lm_head(tail[1].view_as(x_post_hc), model.lm_head)
        if not model._opt_markov_w2_tp_shard:
            logits = gather_and_crop_vocab(logits, model.lm_head)
        return logits, x_post_hc

    def disagrees(self, max_bs: int) -> str | None:
        """Why the kernels' block is not SGLang's on random inputs, or None."""
        from sglang.srt.speculative.dspark_components.dspark_draft import (
            DraftNoise,
            draft_markov_block,
        )

        class _NoSync:
            def sync(self, site, value):
                return value

        head, gamma = self.model.markov_head, self.gamma
        device = self.head.device
        width = int(self.model.lm_head.num_embeddings_per_partition)
        g = torch.Generator(device=device).manual_seed(1234)
        for bs in range(1, max_bs + 1):
            base = (
                torch.randn(bs, gamma, width, generator=g, device=device) * 3
            ).bfloat16()
            anchor = torch.randint(
                0, head.vocab_size, (bs,), generator=g, device=device
            ).long()
            temps = torch.tensor(
                [1.0, 0.0, 0.7, 1.3, 0.5, 1.0, 0.9, 1.1][:bs], device=device
            )
            offset = torch.randint(0, 1 << 30, (1,), generator=g, device=device)
            want_logits = torch.empty(
                bs, gamma, width, dtype=torch.float32, device=device
            )
            want, _ = draft_markov_block(
                markov_head=head, base_logits=base, anchor_tokens=anchor,
                draft_hidden=None, step_temperatures=temps,
                noise=DraftNoise(key=int(self.noise.key), offset=offset.clone()),
                tp_sync=_NoSync(), logits_out=want_logits, greedy_draft=False,
                sharded_accept=True,
            )  # fmt: skip
            got_logits = torch.empty_like(want_logits)
            got = torch.empty(bs * gamma, dtype=torch.int64, device=device)
            noise = DraftNoise(key=int(self.noise.key), offset=offset.clone())
            self.block(base, anchor, temps, noise, got, got_logits)
            if not torch.equal(got.view(bs, gamma), want.long()):
                return f"{bs} rows: tokens {got.tolist()} vs SGLang's {want.tolist()}"
            # the bias GEMV sums in another order: a bf16 rounding flip of the bias now and then
            bias = (want_logits - base.float()).abs()
            apart = (got_logits - want_logits).abs()
            if not (apart <= bias * 2**-7 + 1e-6).all():
                return f"{bs} rows: step logits apart by {apart.max().item():.3g}"
            if int(noise.offset.item()) != int(offset.item()) + 1:
                return "the noise counter did not advance once"
        return None


def _factory(
    model, deadline_s, *, gamma, max_bs, greedy_draft, noise, draft_tokens,
    step_logits, step_temperatures,
):  # fmt: skip
    """SGLang's draft sampler factory: bs -> the kernels' step, or None (SGLang's)."""
    import torch.distributed as dist

    from sgl_amd_mono.deepseek_v41.gfx950 import markov

    def keep(why):
        _LOGGER.info("DSpark draft: SGLang's Markov block runs (%s)", why)
        return lambda bs: None

    if greedy_draft:
        return keep("a greedy draft")
    why = _why_not(model, step_logits)
    group = getattr(model.markov_head, "_shard_group", None)

    def agree(ok: bool) -> bool:
        if group is None:
            return ok
        flag = torch.tensor([int(ok)], dtype=torch.int32)
        dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=group.cpu_group)
        return bool(flag.item())

    if not agree(why is None):
        return keep(why or "another rank's Markov head")
    sampler = None
    try:
        sampler = _Sampler(
            model,
            deadline_s,
            gamma,
            noise,
            draft_tokens,
            step_logits,
            step_temperatures,
        )
    except Exception as e:  # noqa: BLE001 (every rank must still reach the agreement)
        why = f"the kernels do not build: {type(e).__name__}: {e}"
    if not agree(sampler is not None):
        return keep(why or "the kernels did not build on another rank")
    # SGLANG_AMD_MONO_DSV41_MARKOV_MAX_BS: the kernels' batch sizes (up to MAX_SEQUENCES; the step kernel's
    # time grows with the batch, so at 8 it only matches SGLang's block at TP2: default 4)
    import os

    top = min(
        max_bs,
        markov.MAX_SEQUENCES,
        int(os.environ.get("SGLANG_AMD_MONO_DSV41_MARKOV_MAX_BS", "4")),
    )
    why = sampler.disagrees(top)
    if not agree(why is None):
        sampler.exchange.close()
        if sampler.persist is not None:
            sampler.persist.close()
        return keep(why or "the kernels disagree with SGLang on another rank")
    if group.rank_in_group == 0:
        _LOGGER.info(
            "DSpark draft: the sampled Markov block in the gfx950 kernels at batch "
            "sizes 1 to %d",
            top,
        )
    return lambda bs: sampler.step if 1 <= bs <= top else None
