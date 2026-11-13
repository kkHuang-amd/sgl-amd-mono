# SPDX-License-Identifier: Apache-2.0
"""The verify head (``gfx950/head``) against SGLang's accept ops on every TP rank:
greedy (AcceptGreedy + static_accept_tail) and the simulated override bit for bit, the
sampled accept (accept_sampling_sharded) by agreement count, a graph replay, timings.

torchrun --nproc-per-node 2 --master-port 29621 check_head.py [--vocab 129280] [--trials 40]
"""

from __future__ import annotations

import argparse
import os

import torch
import torch.distributed as dist


def _gather_stacked(t: torch.Tensor) -> torch.Tensor:
    out = [torch.empty_like(t) for _ in range(dist.get_world_size())]
    dist.all_gather(out, t.contiguous())
    return torch.cat(out, 0)


class Case:
    """One step's inputs, alike on every rank (the full logits) and this rank's shards."""

    def __init__(self, g, bs, slots, vocab, rank, tp, device, *, greedy, sim_len, hit):
        gamma = slots - 1
        per = vocab // tp
        self.start, self.width = rank * per, per
        # bf16 values: the shard case reads the same logits as the bf16 LM head output
        target = (torch.randn(bs * slots, vocab, generator=g, device=device) * 4).bfloat16().float()
        draft = target[: bs * slots].view(bs, slots, vocab)[:, :gamma] + torch.randn(
            bs, gamma, vocab, generator=g, device=device
        )
        cand = torch.empty(bs, slots, dtype=torch.int64, device=device)
        cand[:, 0] = torch.randint(0, vocab, (bs,), generator=g, device=device)
        best = target.view(bs, slots, vocab).argmax(-1)
        pick = draft.argmax(-1)
        # some drafts are the target's argmax: chains of several accepts
        keep = torch.rand(bs, gamma, generator=g, device=device) < hit
        cand[:, 1:] = torch.where(keep, best[:, :gamma], pick)
        self.target, self.cand = target, cand
        self.shard = target[:, self.start : self.start + per].bfloat16().contiguous()
        self.draft = draft[:, :, self.start : self.start + per].contiguous()
        temps = torch.tensor([1.0, 0.7, 1.3, 1.0, 0.9, 1.1, 1.0, 0.8] * 8, device=device)[:bs]
        self.temps = torch.zeros(bs, device=device) if greedy else temps
        self.sim_temps = self.temps.clone()
        self.sim_len = None
        if sim_len is not None:
            self.sim_len = torch.full((1,), sim_len, dtype=torch.int64, device=device)
        self.sim_seed = torch.randint(0, 2**31, (1,), generator=g, device=device).int()
        self.key = int(torch.randint(0, 2**62, (1,), generator=g, device=device).item())
        self.offset = torch.randint(0, 2**40, (1,), generator=g, device=device)
        self.seq_lens = torch.randint(100, 10000, (bs,), generator=g, device=device)
        self.verify_lens = torch.full((bs,), slots, dtype=torch.int64, device=device)
        self.gate = torch.ones(1, dtype=torch.int32, device=device)
        self.bs, self.slots, self.vocab, self.greedy = bs, slots, vocab, greedy


def reference(c: Case) -> dict:
    from sglang.kernels.ops.speculative.dspark.sharded_accept_hip import (
        accept_sampling_sharded,
    )
    from sglang.kernels.ops.speculative.dspark.simulated_bonus import (
        simulated_bonus_sample_graph,
    )

    bs, S = c.bs, c.slots
    if c.greedy:
        pred = c.target.argmax(-1).view(bs, S)
        match = (c.cand[:, 1:] == pred[:, : S - 1]).int().cumprod(1).sum(1)
        correct = match.long()
        bonus = pred[torch.arange(bs, device=pred.device), correct]
        if c.sim_len is not None:
            correct = c.sim_len.expand(bs).long()
            bonus = pred[torch.arange(bs, device=pred.device), correct]
    else:
        correct, bonus, _ = accept_sampling_sharded(
            candidates=c.cand, target_logits=c.target, draft_logits=c.draft,
            temperatures=c.temps, vocab_start=c.start, width=c.width,
            gather_stacked=_gather_stacked, key=c.key, offset=c.offset,
        )  # fmt: skip
        correct = correct.long()
        if c.sim_len is not None:
            correct = c.sim_len.expand(bs).long()
            bonus = simulated_bonus_sample_graph(
                target_logits=c.target, correct_len=correct.contiguous(),
                temperatures=c.sim_temps, seed=c.sim_seed, bs=bs, rows_per_request=S,
            )  # fmt: skip
    commit = correct.int() + 1
    tokens = torch.zeros(bs, S, dtype=torch.int64, device=c.target.device)
    keep = torch.arange(S - 1, device=tokens.device)[None, :] < correct[:, None]
    tokens[:, : S - 1] = torch.where(keep, c.cand[:, 1:], 0)
    tokens.scatter_(1, correct[:, None], bonus.long()[:, None])
    return dict(
        correct=correct, bonus=bonus.long(), commit=commit,
        new_seq=c.seq_lens + commit.long(), tokens=tokens,
    )  # fmt: skip


class Head:
    def __init__(self, ex, device, max_bs=8, slots=6, width=64640):
        from sgl_amd_mono.deepseek_v41.gfx950 import head

        self.ex, self.ops = ex, head.ops()
        n = head.part_words(max_bs, slots, width)
        self.part = torch.empty(n, dtype=torch.float32, device=device)
        # the case's draft is read through draft_addr; this buffer only carries the layout
        self.draft = torch.zeros(max_bs, slots - 1, width, dtype=torch.float32, device=device)
        self.draft_addr = torch.zeros(1, dtype=torch.int64, device=device)
        self.out = dict(
            correct=torch.zeros(max_bs, dtype=torch.int64, device=device),
            bonus=torch.zeros(max_bs, dtype=torch.int64, device=device),
            trim=torch.zeros(max_bs, dtype=torch.int32, device=device),
            commit=torch.zeros(max_bs, dtype=torch.int32, device=device),
            new_seq=torch.zeros(max_bs, dtype=torch.int64, device=device),
            tokens=torch.zeros(max_bs, slots, dtype=torch.int64, device=device),
            gated=torch.zeros(max_bs, dtype=torch.int32, device=device),
        )
        self.key = torch.zeros(1, dtype=torch.int64, device=device)

    def run(self, c: Case, stage: bool = True, shard: bool = False):
        """shard: the target logits as this rank's bf16 vocab shard (column 0 = c.start)."""
        o = self.out
        if stage:
            self.key.fill_(c.key)
            self.draft_addr.fill_(c.draft.data_ptr())
        target, col0 = (c.shard, 0) if shard else (c.target, c.start)
        self.ops.verify_head(
            target, col0, c.start, c.width, c.vocab, 0.0, self.draft, self.draft_addr, c.cand,
            c.temps, c.sim_temps, c.sim_len, c.sim_seed, self.key, c.offset,
            c.seq_lens, c.verify_lens, c.gate, self.part, o["correct"], o["bonus"],
            o["trim"], o["commit"], o["new_seq"], o["tokens"], o["gated"], None,
            self.ex.calls, self.ex.peers, self.ex.rank, 5.0,
        )  # fmt: skip
        bs = c.bs
        return {k: o[k][:bs] for k in ("correct", "bonus", "commit", "new_seq", "tokens")}


def same(a: dict, b: dict) -> list[str]:
    return [k for k in a if not torch.equal(a[k].long(), b[k].long())]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=int, default=129280)
    ap.add_argument("--trials", type=int, default=40)
    args = ap.parse_args()
    dist.init_process_group("nccl")
    rank, tp = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)))
    device = torch.device("cuda")
    gloo = dist.new_group(backend="gloo")
    from sgl_amd_mono.deepseek_v41.gfx950 import head

    ex = head.Exchange(gloo, rank, tp, device)
    S, V = 6, args.vocab
    h = Head(ex, device, max_bs=8, slots=S, width=V // tp)
    g = torch.Generator(device=device).manual_seed(7)
    fails, shard_fails, sampled_agree, sampled_total = [], [], 0, 0
    for t in range(args.trials):
        for bs in (1, 3, 8):
            for greedy in (True, False):
                for sim in (None, 0, 2, S - 1):
                    c = Case(g, bs, S, V, rank, tp, device, greedy=greedy, sim_len=sim,
                             hit=0.7)  # fmt: skip
                    want, got = reference(c), h.run(c)
                    got = {k: v.clone() for k, v in got.items()}
                    on_shard = h.run(c, shard=True)
                    torch.cuda.synchronize()
                    bad = same(want, got)
                    # the shard read gives the full read's outputs, sampled included
                    if same(got, on_shard):
                        shard_fails.append((t, bs, greedy, sim, same(got, on_shard)))
                    if greedy or sim is not None:
                        if bad:
                            fails.append((t, bs, greedy, sim, bad))
                    else:
                        sampled_total += bs
                        sampled_agree += int(
                            (want["correct"] == got["correct"]).logical_and(
                                want["bonus"] == got["bonus"]
                            ).sum()
                        )
    # a graph replay: the call tags advance on the device
    c = Case(g, 8, S, V, rank, tp, device, greedy=False, sim_len=2, hit=0.7)
    want = reference(c)
    graph = torch.cuda.CUDAGraph()
    h.run(c)
    torch.cuda.synchronize()
    dist.barrier()
    with torch.cuda.graph(graph):
        got = h.run(c)
    replay_bad = []
    for _ in range(5):
        graph.replay()
        torch.cuda.synchronize()
        replay_bad += same(want, got)
    # timing: the head vs SGLang's eager accept chain, bs 8
    c = Case(g, 8, S, V, rank, tp, device, greedy=False, sim_len=2, hit=0.7)
    ev = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
    for _ in range(5):
        h.run(c), reference(c)
    dist.barrier()
    torch.cuda.synchronize()
    ev[0].record()
    for _ in range(20):
        h.run(c)
    ev[1].record()
    for _ in range(20):
        reference(c)
    ev[2].record()
    torch.cuda.synchronize()
    head_us = ev[0].elapsed_time(ev[1]) * 1e3 / 20
    ref_us = ev[1].elapsed_time(ev[2]) * 1e3 / 20
    # the head's launches alone in a graph (as in the verify graph): 20 calls a replay
    graph = torch.cuda.CUDAGraph()
    dist.barrier()
    with torch.cuda.graph(graph):
        for _ in range(20):
            h.run(c, stage=False)
    graph.replay()
    torch.cuda.synchronize()
    dist.barrier()
    ev[0].record()
    for _ in range(10):
        graph.replay()
    ev[1].record()
    torch.cuda.synchronize()
    graph_us = ev[0].elapsed_time(ev[1]) * 1e3 / 200
    print(
        f"rank {rank}: exact fails {len(fails)} {fails[:3]} | shard fails "
        f"{len(shard_fails)} {shard_fails[:3]} | sampled agree "
        f"{sampled_agree}/{sampled_total} | replay {replay_bad or 'ok'} | "
        f"bs8 head {head_us:.1f} us (graph {graph_us:.1f} us), reference {ref_us:.1f} us",
        flush=True,
    )
    ex.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
