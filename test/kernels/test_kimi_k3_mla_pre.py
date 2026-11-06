# SPDX-License-Identifier: Apache-2.0
"""K1m (mla_pre): one Kimi-K3.1 MLA layer's projection front against torch at
its shapes (TP8 rank slices), one GPU (the launch has no peer traffic).

x is the attention-side seam (as test_kimi_k3_k1's reference); every output is
the projection SGLang computes from it: fused_qkv_a's latent, q / kv RMSNorm
(aiter's (x rstd) w), q_b and the indexer's wq_b from the normed q latent, and
g_proj / wk / weights_proj from x. Each against its reference by cosine; stage
2 also against its own normed input, so a stage-1 norm error cannot hide it."""

import unittest

import torch
import torch.nn.functional as F
from sglang.test.test_utils import CustomTestCase

BF = torch.bfloat16
NVB, MAX_BLOCKS, EPS = 3, 8, 1e-5
COS_MIN = 0.9999


def _randn(g, *shape, scale=1.0):
    return (torch.randn(*shape, generator=g, device=g.device) * scale).to(BF)


def _cos(a, b):
    return float(F.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0))


def _rms(x, w, eps):
    x = x.float()
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w.float()


@unittest.skipIf(not torch.cuda.is_available(), "needs a GPU")
class TestMlaPre(CustomTestCase):
    def _run(self, rows, qn=False):
        from sgl_amd_mono.kimi_k3.gfx950.attention import mla_pre as mp

        dev = torch.device("cuda", 0)
        g = torch.Generator(device=dev).manual_seed(11)
        H = 7168
        prefix = _randn(g, rows, H, scale=0.3)
        blocks = _randn(g, rows, MAX_BLOCKS, H, scale=0.3)
        w = {
            "ares_nw": (1 + _randn(g, H, scale=0.1)), "ares_qk": _randn(g, H, scale=0.02),
            "in_nw": (1 + _randn(g, H, scale=0.1)),
            "w_qkv": _randn(g, mp.LATW, H, scale=0.02), "w_g": _randn(g, mp.GATE, H, scale=0.02),
            "w_k": _randn(g, mp.IK, H, scale=0.02), "w_p": _randn(g, mp.IW, H, scale=0.02),
            "w_qb": _randn(g, mp.QB, mp.QL, scale=0.03), "w_iq": _randn(g, mp.IQ, mp.QL, scale=0.03),
            "q_nw": (1 + _randn(g, mp.QL, scale=0.1)), "kv_nw": (1 + _randn(g, mp.KV, scale=0.1)),
        }  # fmt: skip
        w_kc = _randn(
            g, mp.NHQ, mp.QN, mp.KV, scale=0.05
        )  # SGLang's [heads, nope, kv_lora]
        if qn:
            w["w_kc"] = w_kc.transpose(1, 2).contiguous().view(mp.NHQ * mp.KV, mp.QN)

        def z(n):
            return torch.zeros(rows, n, dtype=BF, device=dev)

        out = {
            "q": z(mp.QB), "latent": z(mp.KV + mp.ROPE), "k_nope": z(mp.KV), "q_lora": z(mp.QL),
            "gate": z(mp.GATE), "idx_q": z(mp.IQ), "idx_k": z(mp.IK), "idx_w": z(mp.IW),
        }  # fmt: skip
        if qn:
            out["qn"] = torch.zeros(rows, mp.NHQ, mp.KV, dtype=BF, device=dev)
        key = mp.MlaPreBuild(
            tokens=rows,
            nblocks=NVB,
            eps=EPS,
            out_eps=EPS,
            q_eps=1e-6,
            kv_eps=1e-6,
            qn=qn,
        )
        scratch = torch.zeros(mp.scratch_bytes(key), dtype=torch.uint8, device=dev)
        epoch = torch.ones(1, dtype=torch.int32, device=dev)
        before = prefix.clone()
        mp.mla_pre_launch(
            key,
            prefix=prefix,
            blocks=blocks,
            w=w,
            out=out,
            scratch=scratch,
            layer=3,
            epoch=epoch,
        )
        torch.cuda.synchronize()

        # reference: the seam, then every projection SGLang runs
        src = torch.cat([blocks[:, :NVB, :], prefix.unsqueeze(1)], dim=1)
        scores = (_rms(src, w["ares_nw"], EPS) * w["ares_qk"].float()).sum(-1)
        mixed = (torch.softmax(scores, -1).unsqueeze(-1) * src.float()).sum(1)
        x = _rms(mixed, w["in_nw"], EPS).to(BF)
        off, n = mp.scratch_layout(key)["x"]
        xk = scratch[off : off + n].view(BF).view(rows, H)
        lat = (x.float() @ w["w_qkv"].float().t()).to(BF)
        q_lora = _rms(lat[:, : mp.QL], w["q_nw"], 1e-6).to(BF)
        k_nope = _rms(lat[:, mp.QL : mp.QL + mp.KV], w["kv_nw"], 1e-6).to(BF)

        def lin(a, wt):
            return (a.float() @ wt.float().t()).to(BF)

        checks = {
            "x": (xk, x),
            "latent": (out["latent"], lat[:, mp.QL :]),
            "k_nope": (out["k_nope"], k_nope),
            "gate": (out["gate"], lin(x, w["w_g"])),
            "idx_k": (out["idx_k"], lin(x, w["w_k"])),
            "idx_w": (out["idx_w"], lin(x, w["w_p"])),
            "q": (out["q"], lin(q_lora, w["w_qb"])),
            "idx_q": (out["idx_q"], lin(q_lora, w["w_iq"])),
        }
        if qn:
            # the absorb against the kernel's own q (SGLang's bmm: [heads, S, nope] @ w_kc)
            q_nope = out["q"].view(rows, mp.NHQ, mp.QHD)[:, :, : mp.QN]
            ref_qn = (
                torch.bmm(q_nope.transpose(0, 1).float(), w_kc.float())
                .transpose(0, 1)
                .to(BF)
            )
            checks["qn"] = (out["qn"], ref_qn)
        for name, (got, ref) in checks.items():
            c = _cos(got, ref)
            print(f"rows {rows} {name}: cos {c:.6f}")
            self.assertGreater(c, COS_MIN, name)
        self.assertTrue(torch.equal(prefix, before), "the prefix was written")

    def test_six_rows(self):
        self._run(6)

    def test_eight_rows(self):
        self._run(8)

    def test_one_row(self):
        self._run(1)

    def test_six_rows_absorb(self):
        self._run(6, qn=True)

    def test_eight_rows_absorb(self):
        self._run(8, qn=True)

    def test_bank_without_the_write_block_is_refused(self):
        # A block-write layer stores its row at block write_idx; a bank of exactly
        # write_idx blocks would put the last token's row past the allocation.
        from sgl_amd_mono.kimi_k3.gfx950.attention import mla_pre as mp

        dev = torch.device("cuda", 0)
        key = mp.MlaPreBuild(
            tokens=8, nblocks=NVB, eps=EPS, out_eps=EPS, q_eps=1e-6, kv_eps=1e-6,
            write_idx=NVB,
        )  # fmt: skip
        blocks = torch.zeros(8, NVB, 7168, dtype=BF, device=dev)
        with self.assertRaisesRegex(AssertionError, "writes bank block"):
            mp.mla_pre_launch(
                key, prefix=None, blocks=blocks, w=None, out=None, scratch=None,
                layer=3, epoch=None,
            )  # fmt: skip


if __name__ == "__main__":
    unittest.main()
