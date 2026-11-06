"""The GLM-5.3-Flash gfx950 layer launch on random weights in SGLang's layouts
(RFC #43949), under torchrun at TP2, TP4 and TP8 (``glm53_flash/check_layer.py``):
each launch kind finishes on every rank with no error word and the same output,
and writes exactly its valid rows' state or cache rows (none of a pad row's); a
sparse-attention launch on AITER-shuffled FP8 projections equals the row-major one
bitwise; the router's top-8 equals SGLang's on keys that tie in bf16; and rows past
2048 tokens attend to their selected pools and then their tail, as SGLang selects; a
graph of several launches of each kind replays the eager step bitwise at every width
the TP size builds. The dense launch's MLP output is within the spec's bound of an f32
MLP on its own FFN input, on shuffled and row-major FP8 alike; the launch from the
step's tokens equals the dense launch on the gathered embedding rows and writes the
spec's step tables, for steps of fewer rows than the width too; an MTP verify launch
equals its draft tokens' decode launches bitwise (outputs, KDA verify scratch,
caches and kpool rings) and keeps the committed KDA state; sparse-attention
launches and key selection hold for rows past 65536 pools of a 1M-token context, up
to 650000 tokens; the tail's logits match a torch tail and an f32 tail, back-to-back
tails included."""

import subprocess
import sys
import unittest
from pathlib import Path

import torch
from sglang.test.test_utils import CustomTestCase


def _gfx950_gpus() -> int:
    if not (torch.cuda.is_available() and torch.version.hip):
        return 0
    if "gfx950" not in torch.cuda.get_device_properties(0).gcnArchName:
        return 0
    return torch.cuda.device_count()


CHECK = Path(__file__).with_name("glm53_flash") / "check_layer.py"


@unittest.skipIf(_gfx950_gpus() < 8, "needs 8 gfx950 GPUs")
class TestMonoGlm53FlashKernels(CustomTestCase):
    def _check(self, *args: str, widths=None) -> None:
        for tp in (2, 4, 8):
            if widths is not None:
                # TP2 builds no launch wider than 4 rows
                for w in (w for w in widths if tp > 2 or w <= 4):
                    self._run(tp, *args, "--rows", str(w))
                continue
            self._run(tp, *args)

    def _run(self, tp: int, *args: str) -> None:
        cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            f"--nproc-per-node={tp}",
            f"--master-port={29610 + tp}",
            str(CHECK),
            *args,
        ]
        result = subprocess.run(cmd, timeout=1800)
        self.assertEqual(result.returncode, 0, f"{' '.join(args)} at TP{tp}")

    def test_each_launch_kind(self):
        # 4 rows (TP2 builds no wider launch), the last a pad row
        for kind in ("ffn", "kda", "dsa", "dense", "embed", "head"):
            self._check("--kind", kind, "--rows", "4", "--pad-rows", "1", "--writes")

    def test_shuffled_fp8_projections_match_row_major(self):
        for kind in ("dsa", "dense"):
            self._check("--kind", kind, "--rows", "4", "--compare-shuffled")

    def test_routing_matches_sglang_on_ties(self):
        self._check("--kind", "ffn", "--rows", "4", "--check-routing")

    def test_long_rows_attend_their_tail(self):
        self._check("--kind", "dsa", "--check-keys")

    def test_graph_replays_equal_the_eager_step(self):
        for kind, layers in (("kda", 4), ("dsa", 4), ("dense", 2), ("embed", 1)):
            self._check(
                "--kind",
                kind,
                "--layers",
                str(layers),
                "--replays",
                "20",
                widths=(2, 4, 8),
            )
        # back-to-back tails: each writes its logits in the slot the last did not
        self._check(
            "--kind", "head", "--layers", "4", "--replays", "20", widths=(2, 4, 8)
        )

    def test_dense_mlp_matches_an_fp32_mlp(self):
        self._check("--kind", "dense", "--check-dense", widths=(2, 4, 8))

    def test_the_launch_from_the_tokens_matches_the_dense_launch(self):
        # 3 of 4 rows: the width's last row a pad row
        for rows, step in ((2, 2), (4, 3), (8, 5)):
            for tp in (2, 4, 8):
                if tp > 2 or rows <= 4:
                    self._run(
                        tp,
                        "--kind",
                        "embed",
                        "--check-embed",
                        "--rows",
                        str(rows),
                        "--step-rows",
                        str(step),
                    )

    def test_a_verify_launch_matches_its_draft_tokens_decode_launches(self):
        # MTP 2, 3 and 5 (3, 4 and 6 rows a request), whole requests then pad rows;
        # the lengths complete a pool mid-chain and cross 2048 tokens mid-chain
        for kind, rows, verify, lens in (
            ("kda", 4, 4, ""),
            ("kda", 8, 3, ""),
            ("kda", 8, 6, ""),
            ("dsa", 8, 4, "4101,2050"),
            ("dsa", 8, 3, "4102,2051"),
            ("dsa", 8, 6, "2047"),
            ("dsa", 4, 3, "64"),
        ):
            args = ["--kind", kind, "--rows", str(rows), "--verify", str(verify)]
            for tp in (2, 4, 8):
                if tp > 2 or rows <= 4:
                    self._run(tp, *args, *(["--lens", lens] if lens else []))

    def test_rows_past_65536_pools_of_a_1m_context(self):
        # SGLang's pools for a 1M-token context; 262148 tokens complete pool 65536,
        # whose id takes 17 bits; up to 650000 tokens, as agentic traces reach. Past
        # 65536 pools a wrong score may not fault: --check-keys compares the
        # selection with a torch top-512
        long = ["--kind", "dsa", "--max-len", "655360", "--max-pools", "262144"]
        mixed = "262150,331001,650000,409000,65537,3000,1,0"
        for rows, args in (
            (4, ["--launches", "2", "--lens", "262148,650000,65537,0"]),
            (8, ["--launches", "2", "--lens", mixed]),
            (4, ["--check-keys", "--lens", "331001,262149,650000,100003"]),
        ):
            for tp in (2, 4, 8):
                if tp > 2 or rows <= 4:
                    self._run(tp, *long, "--rows", str(rows), *args)

    def test_the_tail_matches_a_torch_tail(self):
        self._check("--kind", "head", "--check-head", widths=(2, 4, 8))


if __name__ == "__main__":
    unittest.main()
