# SPDX-License-Identifier: Apache-2.0
"""Row-tile GEMM stage (``kimi_k31.gfx950.stages.tile``) on 1 gfx950 GPU: rel L2
against an fp32 reference and, with ``--time``, graph-replayed us against
aiter's tgemm (weights rotate past the 256 MB MALL, so HBM is what is timed).

    python tile_bench.py [--time] [--m 16,128,384] [--shapes 6288:7168,...] [--nb 1,2,4] [--ks 1,4]

ks > 1 times the stage alone (f32 partials [ks, m, n]); in the layer launch the
consumer sums them, which this does not count.
"""

import argparse
import os

import torch

os.environ.setdefault("SGLANG_USE_AITER", "1")

from sgl_amd_mono.kimi_k31.gfx950.stages import tile  # noqa: E402

# per-rank (N, K) of the KDA layer at TP8: in_proj, MoE front, latent up, its owner slice, o_proj
SHAPES = "6288:7168,6016:7168,7168:3584,896:3584,7168:1536"


def bench(fn, ws, iters=48):
    for w in ws[:3]:
        fn(w)
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        for w in ws:
            fn(w)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        for i in range(iters):
            fn(ws[i % len(ws)])
    g.replay()
    torch.cuda.synchronize()
    st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    st.record()
    for _ in range(5):
        g.replay()
    en.record()
    torch.cuda.synchronize()
    return st.elapsed_time(en) / (5 * iters) * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--time", action="store_true")
    ap.add_argument("--m", default="16,128,384")
    ap.add_argument("--shapes", default=SHAPES)
    ap.add_argument("--nb", default="2")
    ap.add_argument("--ks", default="1")
    a = ap.parse_args()
    if a.time:
        from aiter.tuned_gemm import tgemm
    worst = 0.0
    for m in map(int, a.m.split(",")):
        for nk in a.shapes.split(","):
            n, k = map(int, nk.split(":"))
            ncopy = max(4, -(-640 * 2**20 // (n * k * 2))) if a.time else 1
            ws = [
                torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.02
                for _ in range(ncopy)
            ]
            x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
            ref = x.float() @ ws[0].float().T
            tg = ""
            if a.time:
                ut = bench(lambda w: tgemm.mm(x, w, None, otype=torch.bfloat16), ws)
                tg = f"  tgemm {ut:6.1f} us"
            for nb, ks in (
                (nb, ks)
                for nb in map(int, a.nb.split(","))
                for ks in map(int, a.ks.split(","))
            ):
                if (k // ks) % tile.KCH:
                    continue
                if ks == 1:
                    out = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
                else:
                    out = torch.empty(ks, m, n, device="cuda", dtype=torch.float32)
                launch = tile.standalone(m, n, k, nb, ks)
                fn = lambda w, launch=launch, out=out: launch(  # noqa: E731
                    x.data_ptr(),
                    w.data_ptr(),
                    out.data_ptr(),
                    stream=torch.cuda.current_stream(),
                )
                fn(ws[0])
                torch.cuda.synchronize()
                got = out.float() if ks == 1 else out.sum(0)
                err = ((got - ref).norm() / ref.norm()).item()
                worst = max(worst, err)
                line = f"m={m:3d} n={n:5d} k={k:5d} nb={nb} ks={ks} relL2 {err:.1e}"
                if a.time:
                    us = bench(fn, ws)
                    tb = (n * k + m * k + m * n) * 2 / us / 1e6
                    line += f"  tile {us:6.1f} us {tb:4.2f} TB/s"
                print(line + tg, flush=True)
            del ws
            torch.cuda.empty_cache()
    print("PASS" if worst < 1e-2 else "FAIL", f"worst relL2 {worst:.1e}")


if __name__ == "__main__":
    main()
