"""Kernel-only time of the GLM-5.3-Flash layer launch, under torchrun (TP 2, 4 or 8):

    torchrun --nproc-per-node 4 test/kernels/glm53_flash/time_layers.py \\
        --kind model --rows 2,4,8 [--pos 2047] [--layers 12] [--row-major] [--per-row-moe]

One HIP graph holds a step of launches, inputs chained layer to layer, every row
live. Each layer has its own random weights (``check_layer.py``'s, seeded by layer,
so they stream from HBM as in a served step), its own state or caches, and every
sparse-attention row sits at position ``--pos``. ``--kind model`` is the served
step: layers 3 to 43 in model order (30 KDA, 11 sparse attention) on one runtime a
kind; ``--kind full`` the whole served step as the spec builds it: layer 0 from the
step's tokens, the dense layers 1 and 2, layers 3 to 44, then the tail. ``kda``,
``dsa``, ``dense``, ``embed`` (layer 0 from the tokens) and ``head`` (the tail) run
``--layers`` launches of one kind; each ``embed`` and ``head`` launch takes the step's
input, the others the previous launch's output. FP8 projections are read in place (AITER's (16, 16) layout), or row-major with
``--row-major``. After an eager step and 20 warm-up replays, each of ``--trials``
trials replays the graph ``--reps`` times; rank 0 prints per-launch microseconds
(trial time / (reps x launches)): the median, minimum and maximum over trials, the
slowest rank's median, and whether every error word stayed clean."""

import argparse
import gc
import os
import statistics
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import check_layer as cl  # noqa: E402

from sgl_amd_mono.glm53_flash.spec import GROUPED_FROM  # noqa: E402

_GEN = cl._gen


def _seed_layer(layer: int) -> None:
    """``check_layer``'s generators seeded for one layer: distinct weights a layer."""
    cl._gen = lambda seed, device: _GEN(seed + 1000 * layer, device)


def _kinds(args) -> list:
    if args.kind == "model":
        return ["dsa" if i % 4 == 3 else "kda" for i in range(3, 44)]
    if args.kind == "full":
        layers = ["dsa" if i % 4 == 3 else "kda" for i in range(3, 45)]
        return ["embed", "dense", "dense", *layers, "head"]
    return [args.kind] * args.layers


def _build(args, kinds: list, rows: int, rank: int, tp: int, group, device):
    """The runtimes (one a kind), each layer's op and launch arguments."""
    from sgl_amd_mono.glm53_flash.gfx950.config import GLOBAL_INTER
    from sgl_amd_mono.glm53_flash.gfx950.kernel import FlashBuild
    from sgl_amd_mono.glm53_flash.gfx950.runner import (
        FlashHeadOp,
        FlashLayerOp,
        FlashRuntime,
    )

    shuffled = not args.row_major
    runtimes = {}
    for kind in sorted(set(kinds)):
        dsa, head = kind == "dsa", kind == "head"
        dense = kind in ("dense", "embed")
        key = FlashBuild(
            rows=rows,
            tp=tp,
            launches_per_step=kinds.count(kind),
            inter=GLOBAL_INTER // tp,
            kda=not (dsa or head),
            dsa=dsa,
            max_pools=args.max_pools if dsa else 0,
            qkv_shuffled=shuffled and dsa,
            qb_shuffled=shuffled and dsa,
            o_shuffled=shuffled and dsa,
            dense=dense,
            dense_shuffled=shuffled and dense,
            tokens=kind == "embed",
            head=head,
            grouped=not args.per_row_moe
            and rows >= GROUPED_FROM
            and not (dense or head),
        )
        runtimes[kind] = FlashRuntime(key, rank, group, device)
        runtimes[kind].set_deadline(60.0)
    ops, extra, ordinal = [], [], {kind: 0 for kind in runtimes}
    for layer, kind in enumerate(kinds):
        _seed_layer(layer)
        n = ordinal[kind]
        ordinal[kind] += 1
        if kind == "head":
            ops.append(
                FlashHeadOp(runtimes[kind], cl.head_weights(rank, tp, device), n)
            )
            extra.append({})
            continue
        if kind in ("dense", "embed"):
            ffn = cl.dense_weights(rank, tp, device, shuffled)
            attn = cl.kda_weights(rank, tp, device)
            state, idx = cl.kda_state(rows, tp, device)
            embedding = cl.embedding(rank, tp, device)[0] if kind == "embed" else None
            op = FlashLayerOp(runtimes[kind], ffn, n, kda=attn, embedding=embedding)
            ops.append(op)
            kw = dict(state=state, state_idx=idx)
            if embedding is not None:
                kw["step"] = cl.step_inputs(rows, rows, tp, device)
            extra.append(kw)
            continue
        ffn = cl.ffn_weights(rank, tp, device)
        if kind == "kda":
            attn = cl.kda_weights(rank, tp, device)
            state, idx = cl.kda_state(rows, tp, device)
            ops.append(FlashLayerOp(runtimes[kind], ffn, n, kda=attn))
            extra.append(dict(state=state, state_idx=idx))
        else:
            attn = cl.dsa_weights(rank, tp, device)
            if shuffled:
                attn = cl.shuffled_copy(attn)
            caches, table, pt = cl.dsa_caches(rows, device, [args.pos + 1] * rows)
            ops.append(FlashLayerOp(runtimes[kind], ffn, n, dsa=attn))
            extra.append(dict(caches=caches, rows=table, page_table=pt))
    cl._gen = _GEN
    return runtimes, ops, extra


def _time(args, rows: int, rank: int, tp: int, group, device) -> None:
    from sgl_amd_mono.common import guard

    kinds = _kinds(args)
    runtimes, ops, extra = _build(args, kinds, rows, rank, tp, group, device)
    x, res, post, comb = cl.layer_inputs(rows, device)
    # layer 0's input besides its tokens: every stream a copy of the embedding row
    entry = (
        torch.zeros_like(x),
        torch.zeros_like(res),
        torch.ones_like(post),
        torch.zeros_like(comb),
    )

    def step():
        h = (x, res, post, comb)
        for op, kw, kind in zip(ops, extra, kinds):
            if kind == "embed":
                h = op.launch(*entry, **kw)
            elif kind == "head":
                op.launch(*(h if args.kind == "full" else (x, res, post, comb)))
            else:
                h = op.launch(*h, **kw)
        for rt in runtimes.values():
            rt.advance_step()

    step()  # eager first: each launch loads its code outside the capture
    torch.cuda.synchronize()
    dist.barrier(group=group)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()
    for _ in range(20):
        graph.replay()
    torch.cuda.synchronize()
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    per_launch = []
    for _ in range(args.trials):
        dist.barrier(group=group)
        begin.record()
        for _ in range(args.reps):
            graph.replay()
        end.record()
        torch.cuda.synchronize()
        per_launch.append(begin.elapsed_time(end) * 1e3 / (args.reps * len(kinds)))
    clean = not any(
        int(rt.epoch[i]) for rt in runtimes.values() for i in guard.ERROR_WORDS
    )
    median = statistics.median(per_launch)
    medians = [None] * tp
    dist.all_gather_object(medians, (median, clean), group=group)
    if rank == 0:
        print(
            f"TIME kind={args.kind} rows={rows} tp={tp} launches={len(kinds)} "
            f"pos={args.pos} fp8={'row-major' if args.row_major else 'in-place'} "
            f"us_per_launch median={median:.2f} min={min(per_launch):.2f} "
            f"max={max(per_launch):.2f} "
            f"slowest_rank_median={max(m for m, _ in medians):.2f} "
            f"clean={all(c for _, c in medians)}",
            flush=True,
        )
    del graph
    dist.barrier(group=group)
    for rt in runtimes.values():
        rt.close()
    ops.clear()
    extra.clear()
    runtimes.clear()
    # free this width's code objects now: an unload during the next capture breaks it
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--kind",
        choices=("kda", "dsa", "dense", "embed", "head", "model", "full"),
        default="model",
    )
    ap.add_argument("--rows", default="2,4,8", help="comma-separated widths")
    ap.add_argument("--layers", type=int, default=12, help="launches of one kind")
    ap.add_argument("--reps", type=int, default=200, help="graph replays a trial")
    ap.add_argument("--trials", type=int, default=7)
    ap.add_argument("--pos", type=int, default=2047, help="sparse-attention position")
    ap.add_argument("--max-pools", type=int, default=262144, help="1M-token context")
    ap.add_argument("--row-major", action="store_true", help="row-major FP8 weights")
    ap.add_argument(
        "--per-row-moe",
        action="store_true",
        help="the per-row MoE at every width, the baseline of the grouped one",
    )
    args = ap.parse_args()
    cl.MAX_LEN = max(cl.MAX_LEN, (args.pos + 256) // 256 * 256)
    rank, tp = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    device = torch.device("cuda", torch.cuda.current_device())
    dist.init_process_group("gloo")
    group = dist.new_group(backend="gloo")
    for rows in map(int, args.rows.split(",")):
        if tp == 2 and rows > 4:
            continue  # TP2 builds no launch wider than 4 rows
        _time(args, rows, rank, tp, group, device)
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
