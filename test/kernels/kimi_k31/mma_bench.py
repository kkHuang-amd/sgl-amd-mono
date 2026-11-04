# SPDX-License-Identifier: Apache-2.0
"""LDS tile GEMM stage (``kimi_k31.gfx950.stages.tile_mma``) on 1 gfx950 GPU:
rel L2 against fp32 and, with ``--time``, graph-replayed us against tgemm.
ks > 1 times the stage alone (``--mode`` part / atom / atombf); ``--sum`` adds
the partials' torch sum, or the atomics' zeroing, each one launch.

    python mma_bench.py [--time] [--sum] [--m 128] [--shapes N:K,...] [--cfg bm:bn:wm:wn:ks[:depth],...]
"""

import argparse
import os
import sys

import torch

os.environ.setdefault("SGLANG_USE_AITER", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tile_bench import SHAPES, bench  # noqa: E402

from sgl_amd_mono.kimi_k31.gfx950.stages import tile_mma  # noqa: E402

CFG = "128:128:2:4:1,128:128:2:4:2,128:128:2:4:4,128:64:4:2:4,128:64:2:4:4,128:32:8:1:4"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--time", action="store_true")
    ap.add_argument("--sum", action="store_true")
    ap.add_argument("--m", default="128")
    ap.add_argument("--shapes", default=SHAPES)
    ap.add_argument("--cfg", default=CFG)
    ap.add_argument(
        "--mode", default="part", help="part,atom,atombf,lin,linbf (ks > 1)"
    )
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
                tg = (
                    f"  tgemm {bench(lambda w: tgemm.mm(
                                x, w, None, otype=torch.bfloat16
                            ), ws):6.1f} us"
                )
            for cfg, mode in (
                (c, md) for c in a.cfg.split(",") for md in a.mode.split(",")
            ):
                bm, bn, wm, wn, ks, *d = map(int, cfg.split(":"))
                if ks == 1 and mode != a.mode.split(",")[0]:
                    continue
                depth = d[0] if d else tile_mma.DEPTH
                if (k // ks) % tile_mma.BK or k % ks:
                    continue
                part = ks > 1 and mode == "part"
                shape = (ks, m, n) if part else (m, n)
                dt = (
                    torch.bfloat16
                    if ks == 1 or mode in ("atombf", "linbf")
                    else torch.float32
                )
                out = torch.zeros(shape, device="cuda", dtype=dt)
                launch = tile_mma.standalone(m, n, k, bm, bn, wm, wn, ks, depth, mode)

                def fn(w, launch=launch, out=out, part=part, ks=ks):
                    if a.sum and ks > 1 and not part:
                        out.zero_()
                    launch(
                        x.data_ptr(),
                        w.data_ptr(),
                        out.data_ptr(),
                        stream=torch.cuda.current_stream(),
                    )
                    if a.sum and part:
                        out.sum(0)

                fn(ws[0])
                torch.cuda.synchronize()
                got = out.sum(0) if part else out.float()
                err = ((got - ref).norm() / ref.norm()).item()
                worst = max(worst, err)
                line = f"m={m:3d} n={n:5d} k={k:5d} t{bm}x{bn} w{wm}x{wn} ks{ks} d{depth} {mode if ks > 1 else 'bf16':6s} relL2 {err:.1e}"
                if a.time:
                    line += f"  mma {bench(fn, ws):6.1f} us"
                print(line + tg, flush=True)
            del ws
            torch.cuda.empty_cache()
    print("PASS" if worst < 1e-2 else "FAIL", f"worst relL2 {worst:.1e}")


if __name__ == "__main__":
    main()
