"""Check deepseek_v41/engram_rows.py against SGLang's engram_gather per layer (the transposed-ids path in
spec._engram_rows): bitwise rows, then the time of one launch against the copy + per-layer gathers.

HIP_VISIBLE_DEVICES=4 python check_engram_rows.py"""

import torch
from sglang.kernels.ops.embeddings.engram_gather import engram_gather

from sgl_amd_mono.deepseek_v41.engram_rows import engram_rows

BLK = 128


class Embed:
    def __init__(self, rows, dim, row_start, g, dev):
        self.rows, self.dim, self.row_start, self.mxfp4 = rows, dim, row_start, False
        self.weight = (torch.randn(rows, dim, generator=g, device=dev) * 2).to(
            torch.float8_e4m3fn
        )
        self.scale = torch.randint(
            110, 135, (rows, dim // BLK), generator=g, device=dev, dtype=torch.uint8
        )

    def _table_ptrs(self):
        return self.weight.data_ptr(), self.scale.data_ptr()


def old(hash_ids, layers, out):
    by_layer = hash_ids.transpose(0, 1).contiguous()
    for k, (c, e) in enumerate(layers):
        engram_gather(*e._table_ptrs(), by_layer[c].view(-1), out[k].view(-1, e.dim), e.dim, BLK,
                      row_lo=e.row_start, row_hi=e.row_start + e.rows)  # fmt: skip


def main():
    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(3)
    dim, vocab = 1024, 1 << 16
    for t in (6, 48):
        for heads in (4, 8):
            embeds = [
                Embed(vocab // 4, dim, vocab // 4, g, dev),
                Embed(vocab // 4, dim, 0, g, dev),
            ]
            hash_ids = torch.randint(
                0, vocab // 2, (t, 3, heads), generator=g, device=dev, dtype=torch.int64
            )
            layers = [(2, embeds[0]), (0, embeds[1])]
            want = torch.full((2, t, heads, dim), 7.0, dtype=torch.bfloat16, device=dev)
            got = torch.full_like(want, 9.0)
            old(hash_ids, layers, want)
            engram_rows(hash_ids, layers, got, BLK)
            torch.cuda.synchronize()
            same = torch.equal(want.view(torch.int16), got.view(torch.int16))
            times = []
            for fn, o in (
                (lambda: old(hash_ids, layers, want), want),
                (lambda: engram_rows(hash_ids, layers, got, BLK), got),
            ):
                gr = torch.cuda.CUDAGraph()
                s = torch.cuda.Stream()
                with torch.cuda.stream(s):
                    fn()
                torch.cuda.synchronize()
                with torch.cuda.graph(gr, stream=s):
                    for _ in range(20):
                        fn()
                gr.replay()
                torch.cuda.synchronize()
                e0, e1 = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                e0.record()
                for _ in range(5):
                    gr.replay()
                e1.record()
                torch.cuda.synchronize()
                times.append(e0.elapsed_time(e1) / 100 * 1e3)
            print(
                f"t {t} heads {heads}: {'PASS' if same else 'FAIL'} | old {times[0]:.1f} us vs one launch {times[1]:.1f} us"
            )


if __name__ == "__main__":
    main()
