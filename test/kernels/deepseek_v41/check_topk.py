"""Check common/topk.py (ATOM's CTA radix select) on gfx950: a CTA a row selects the top 512 of up to
``E * 512`` f32 scores (valid below each row's bound), against torch's (value desc, index asc) set.

HIP_VISIBLE_DEVICES=4 python check_topk.py [--widths 4096 16384] [--rows 48] [--trials 5]"""

import argparse

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from aiter.ops.flydsl.kernels import buffer_ops as bo
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled
from flydsl.expr import range_constexpr
from flydsl.expr.typing import Int32, Int64, T

from sgl_amd_mono.common import topk
from sgl_amd_mono.common.ops import rsrc, uniform
from sgl_amd_mono.common.plan import THREADS

K = 512
_STREAM = fx.Stream(None)


def build(width: int, in_order: bool = True):
    E = width // THREADS
    assert E * THREADS == width and width < 1 << 16
    words = topk.lds_words(K, index_bits=16, ranked=False, in_order=in_order)

    @fx.struct
    class Lds:
        w: fx.Array[fx.Int32, words, 16]

    @flyc.kernel(
        name=f"check_topk_w{width}_o{int(in_order)}", known_block_size=[THREADS, 1, 1]
    )
    def kern(scores: Int64, bounds: Int64, out: Int64, stride: Int32):
        tid = fx.thread_idx.x
        row = fx.block_idx.x
        lds = fx.SharedAllocator().allocate(Lds).peek().w.ptr
        r_s = rsrc(scores)
        r_b = rsrc(bounds)
        r_o = rsrc(out)
        bound = fx.Int32(bo.buffer_load(r_b, row, vec_width=1, dtype=T.i32))
        us, idxs, valid = [], [], []
        for e in range_constexpr(E):
            i = tid * E + e
            bits = fx.Int32(
                bo.buffer_load(r_s, row * stride + i, vec_width=1, dtype=T.i32)
            )
            us.append(topk.order_key(bits))
            idxs.append(i)
            valid.append(i < bound)
        n_out = topk.select(
            lds,
            tid,
            tid % 64,
            uniform(tid // 64),
            us,
            idxs,
            K,
            index_bits=16,
            valid=valid,
            in_order=in_order,
        )
        _, idx = topk.slot(lds, tid, n_out, K, index_bits=16, in_order=in_order)
        bo.buffer_store(idx, r_o, row * K + tid)

    @flyc.jit
    def launch(
        scores: Int64,
        bounds: Int64,
        out: Int64,
        stride: Int32,
        rows: Int32,
        stream: fx.Stream = _STREAM,
    ):
        kern(scores, bounds, out, stride).launch(
            grid=(rows,), block=(THREADS,), stream=stream
        )

    return launch


def reference(scores, bounds):
    rows, width = scores.shape
    col = torch.arange(width, device=scores.device)
    masked = scores.masked_fill(col[None, :] >= bounds[:, None], float("-inf"))
    out = []
    for r in range(rows):
        n = int(bounds[r])
        if n <= K:
            out.append(set(range(n)))
            continue
        v = masked[r, :n]
        # value desc, index asc: sort by (-v, idx) via a stable sort on descending values
        order = torch.sort(v, descending=True, stable=True).indices[:K]
        out.append(set(order.tolist()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--widths", type=int, nargs="+", default=[4096, 8192, 16384])
    ap.add_argument("--rows", type=int, default=48)
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--no-in-order", action="store_true")
    a = ap.parse_args()
    dev = torch.device("cuda")
    for width in a.widths:
        launch = build(width, not a.no_in_order)
        bad = 0
        for trial in range(a.trials):
            g = torch.Generator(device=dev).manual_seed(7 * trial + width)
            scores = torch.randn(a.rows, width, generator=g, device=dev)
            if trial % 2:  # ties: coarse values
                scores = (
                    scores * 4
                ).round() / 4 + 0.0  # no -0.0: the order key puts it below +0.0
            bounds = torch.randint(
                1, width + 1, (a.rows,), generator=g, device=dev, dtype=torch.int32
            )
            bounds[0] = width
            out = torch.full((a.rows, K), -2, dtype=torch.int32, device=dev)
            _run_compiled(
                launch, scores.data_ptr(), bounds.data_ptr(), out.data_ptr(), width, a.rows,
                torch.cuda.current_stream(),
            )  # fmt: skip
            torch.cuda.synchronize()
            want = reference(scores, bounds)
            for r in range(a.rows):
                got = set(x for x in out[r].tolist() if x >= 0)
                if got != want[r]:
                    bad += 1
                    if bad <= 3:
                        print(
                            f"width {width} trial {trial} row {r}: {len(got ^ want[r])} differ (bound {int(bounds[r])})"
                        )
        # timing: graph-free eager, many launches
        scores = torch.randn(a.rows, width, device=dev)
        bounds = torch.full((a.rows,), width, dtype=torch.int32, device=dev)
        out = torch.empty(a.rows, K, dtype=torch.int32, device=dev)
        args = (
            launch,
            scores.data_ptr(),
            bounds.data_ptr(),
            out.data_ptr(),
            width,
            a.rows,
        )
        for _ in range(5):
            _run_compiled(*args, torch.cuda.current_stream())
        s, e = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        s.record()
        for _ in range(50):
            _run_compiled(*args, torch.cuda.current_stream())
        e.record()
        torch.cuda.synchronize()
        print(
            f"width {width}: rows {a.rows}, bad rows {bad} / {a.rows * a.trials}, {s.elapsed_time(e) / 50 * 1e3:.1f} us a launch"
        )


if __name__ == "__main__":
    main()
