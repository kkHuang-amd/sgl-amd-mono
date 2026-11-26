"""Check the sampled DSpark Markov block (gfx950/markov) against SGLang's arithmetic (torchrun,
TP2 or TP4): the tokens on every rank alike, the step logits and the noise counter, eager
and under graph replay."""

import argparse
import os

import torch
import torch.distributed as dist
import torch.nn.functional as F

VOCAB, DIM = 129280, 256


def reference(base, w1, w2s, anchor, temps, key, offset, vocab_start, group):
    """(tokens [bs, gamma], step logits [bs, gamma, padded]) as SGLang computes them."""
    from sglang.kernels.ops.speculative.dspark.markov_sample import _philox

    bs, gamma, padded = base.shape
    width = w2s.shape[0]
    prev = anchor.clone()
    tokens, logits = [], torch.empty(bs, gamma, padded, device=base.device)
    g = torch.arange(width, device=base.device) + vocab_start
    for s in range(gamma):
        bias = F.linear(w1[prev], w2s).float()  # [bs, width] (bf16 out)
        value = base[:, s].float().clone()
        value[:, :width] += bias
        logits[:, s] = value
        keys = value[:, :width].clone()
        for r in range(bs):
            t = float(temps[r])
            if t != 0.0:
                c0 = (g // 4).to(torch.int64)
                c1 = torch.full_like(c0, r)
                c2 = torch.full_like(c0, s)
                c3 = torch.full_like(c0, int(offset) & 0xFFFFFFFF)
                words = _philox(key, c0, c1, c2, c3)
                w = torch.stack(words, 1).gather(1, (g % 4)[:, None]).squeeze(1)
                u = (w.to(torch.float32) + 0.5) * 2.3283064365386963e-10
                keys[r] = value[r, :width] / t - torch.log(-torch.log1p(-u))
        best, idx = keys.max(1)  # the lowest index on ties
        pair = torch.stack([best, (idx + vocab_start).float()], 1)
        every = [torch.empty_like(pair) for _ in range(dist.get_world_size(group))]
        dist.all_gather(every, pair, group=group)
        every = torch.stack(every)  # [tp, bs, 2]
        top = every[:, :, 0].argmax(0)  # the lowest rank on ties
        prev = every[top, torch.arange(bs), 1].long()
        tokens.append(prev)
    return torch.stack(tokens, 1), logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gamma", type=int, default=5)
    ap.add_argument("--replays", type=int, default=20)
    ap.add_argument(
        "--persist",
        action="store_true",
        help="markov_sample_block: the block in one launch",
    )
    args = ap.parse_args()
    rank, tp = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    dist.init_process_group("nccl")
    cpu = dist.new_group(backend="gloo")
    from sgl_amd_mono.deepseek_v41.gfx950 import markov

    g = torch.Generator(device=dev).manual_seed(7)
    w1 = (torch.randn(VOCAB, DIM, generator=g, device=dev) * 0.05).bfloat16()
    w2 = (torch.randn(VOCAB, DIM, generator=g, device=dev) * 0.5).bfloat16()
    part = VOCAB // tp
    padded = part + 64  # the partition's padding past the vocab
    vocab_start = rank * part
    w2s = w2[vocab_start : vocab_start + part]
    ops, x = markov.ops(), markov.Exchange(cpu, rank, tp, dev)
    bx = markov.BlockExchange(cpu, rank, tp, dev) if args.persist else None
    scratch = torch.empty(
        markov.MAX_SEQUENCES, markov.blocks(padded), 2, dtype=torch.float32, device=dev
    )
    key, gamma, fails = 0x1234_5678_9ABC, args.gamma, []
    for bs in range(1, markov.MAX_SEQUENCES + 1):
        gr = torch.Generator(device=dev).manual_seed(100 + 13 * rank + bs)
        base = (torch.randn(bs, gamma, padded, generator=gr, device=dev) * 3).bfloat16()
        anchor = torch.randint(
            0,
            VOCAB,
            (bs,),
            device=dev,
            generator=torch.Generator(device=dev).manual_seed(bs),
        )
        temps = torch.tensor([1.0, 0.0, 0.7, 1.3, 0.5, 1.0, 0.9, 1.1][:bs], device=dev)
        offset0 = 17 * bs
        want, want_logits = reference(
            base, w1, w2s, anchor, temps, key, offset0, vocab_start, None
        )
        offset = torch.tensor([offset0], dtype=torch.int64, device=dev)
        out = torch.zeros(bs * gamma, dtype=torch.int64, device=dev)
        logits = torch.zeros(bs, gamma, padded, dtype=torch.float32, device=dev)

        def block():
            if bx is not None:
                ops.markov_sample_block(
                    base, w1, w2s, anchor, logits, temps, offset, key, vocab_start, bx.partials,
                    out, gamma, bx.calls, bx.peers, rank, 20.0,
                )  # fmt: skip
                return
            for s in range(gamma):
                ops.markov_sample_step(
                    base[:, s], w1, w2s, anchor, logits[:, s], temps, offset, key,
                    vocab_start, s, scratch, out, gamma, x.tok, x.calls, x.peers, rank,
                    s == gamma - 1, 20.0,
                )  # fmt: skip

        block()
        torch.cuda.synchronize()
        bad_tok = int((out.view(bs, gamma) != want).sum())
        bias = (want_logits - base.float()).abs()
        apart = (logits - want_logits).abs()
        bad_logit = int((apart > bias * 2**-7 + 1e-6).sum())
        counter = int(offset.item()) == offset0 + 1
        # graph replay: the same block, the counter put back each time
        graph = torch.cuda.CUDAGraph()
        offset.fill_(offset0)
        with torch.cuda.graph(graph):
            block()
        bad_replay = 0
        for _ in range(args.replays):
            offset.fill_(offset0)
            out.zero_()
            graph.replay()
            torch.cuda.synchronize()
            bad_replay += int((out.view(bs, gamma) != want).sum())
        t0, t1 = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        t0.record()
        for _ in range(args.replays):
            graph.replay()
        t1.record()
        torch.cuda.synchronize()
        us = t0.elapsed_time(t1) / args.replays * 1e3
        every = [torch.empty_like(out) for _ in range(tp)]
        dist.all_gather(every, out)
        alike = all(torch.equal(every[0], e) for e in every)
        ok = bad_tok == 0 and bad_logit == 0 and counter and bad_replay == 0 and alike
        if rank == 0:
            print(
                f"bs={bs} {'PASS' if ok else 'FAIL'} | tokens differing {bad_tok} of "
                f"{bs * gamma} | step logits past a bias rounding {bad_logit} | max "
                f"{apart.max().item():.3g} | counter {counter} | replays differing "
                f"{bad_replay} | ranks alike {alike} | {us:.1f} us a block (graph)",
                flush=True,
            )
        if not ok:
            fails.append(bs)
    x.close()
    if bx is not None:
        bx.close()
    dist.barrier()
    if rank == 0:
        print("ALL PASS" if not fails else f"FAIL at {fails}", flush=True)
    dist.destroy_process_group()
    raise SystemExit(1 if fails else 0)


if __name__ == "__main__":
    main()
