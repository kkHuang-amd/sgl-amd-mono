"""GLM-5.3-Flash with --monokernel-backend amd on MI35x, TP4 (RFC #43949), across the
switch points: decode steps of 1 to 16 rows run the kernels, wider ones the normal
path, and requests join and leave a running batch that steps past 16 rows and back
(a short GSM8K run, for gross failures); with MTP, verify steps of 1 to 4 requests
(4 to 16 rows) on the kernels (GSM8K and the draft's accept length); the tokens of covered steps scored again by
the normal prefill path (KL consistency); each layer's launch against SGLang's own
layer, and the tail's logits against SGLang's, under SGLANG_AMD_MONO_CHECK; the
hidden states a covered step returns; and a fault on one rank that stops the server
before any token of the failed step is sent."""

import os
import re
import tempfile
import time
import unittest
from types import SimpleNamespace

import requests
import torch
from glm53_flash_server import ARGS, ENV, MODEL, MTP
from sglang.test.kl_test_utils import (
    test_input_output_logprobs_match_helper as assert_logprobs_match,
)
from sglang.test.sgl_eval_utils import run_sgl_eval
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

from sgl_amd_mono import BACKEND

LAUNCH_TIMEOUT = 3600
# The normal path scores about 0.97 on GSM8K (thinking on); the bound is for gross
# failures
GSM8K_BOUND = 0.90
# Decode against prefill, one batch of prompts at a time (every decode step covered);
# the normal path itself averages about 0.04 on 8 prompts
KL_BOUND = 0.06
# MTP (3 draft tokens) accepts about 3.3 tokens a step on this GSM8K run at TP4;
# the bound is for gross failures
MTP_ACCEPT_BOUND = 2.5


def _launch(extra_env=None, extra_args=(), log=None):
    env = {**os.environ, **ENV, **(extra_env or {})}
    return popen_launch_server(
        MODEL,
        DEFAULT_URL_FOR_TEST,
        timeout=LAUNCH_TIMEOUT,
        other_args=ARGS + ["--monokernel-backend", BACKEND, *extra_args],
        env=env,
        return_stdout_stderr=log,
    )


def _gsm8k(threads=12):
    """GSM8K, 200 questions ``threads`` at a time, thinking on."""
    return run_sgl_eval(
        SimpleNamespace(
            base_url=DEFAULT_URL_FOR_TEST,
            model=MODEL,
            eval_name="gsm8k",
            num_examples=200,
            num_threads=threads,
            max_tokens=32768,
            temperature=1.0,
            top_p=0.95,
            seed=42,
            sgl_eval_thinking=True,
        )
    )


class TestMonokernelDecode(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.process = _launch(extra_args=["--watchdog-timeout", "600"])

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "process", None) is not None:
            terminate_and_kill_process_tree(cls.process)

    def test_gsm8k_past_the_largest_width(self):
        # 24 requests at a time: the batch steps past 16 rows as it fills, and back
        # as it drains
        metrics = _gsm8k(threads=24)
        print(f"{metrics=}")
        self.assertGreater(metrics["score"], GSM8K_BOUND)

    def test_covered_steps_match_prefill(self):
        # one graph per batch (ignore_eos keeps its rows): 2, 3, 7, 8, 11 and 16 rows
        # (widths 2, 4, 8, 12 and 16, pad rows at 3, 7 and 11). Each graph reads its
        # own metadata buffers; with the cache flushed, a graph replayed first on
        # another's reads none of this batch's pages
        for rows in (2, 3, 7, 8, 11, 16):
            requests.post(DEFAULT_URL_FOR_TEST + "/flush_cache").raise_for_status()
            assert_logprobs_match(
                DEFAULT_URL_FOR_TEST,
                {MODEL: {"kl_div": KL_BOUND}},
                MODEL,
                max_samples=rows,
                max_new_tokens=512,
                trust_remote_code=True,
            )


class TestMonokernelMtp(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.process = _launch(extra_args=[*MTP, "--watchdog-timeout", "1200"])

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "process", None) is not None:
            terminate_and_kill_process_tree(cls.process)

    def test_gsm8k_and_accept_length_on_the_kernels(self):
        # 4 requests at a time, so every verify step fits the 16-row width and runs
        # the kernels. A verify row that reads the wrong KDA state or keys of its
        # draft chain lowers the accept length before the score
        metrics = _gsm8k(threads=4)
        info = requests.get(DEFAULT_URL_FOR_TEST + "/get_server_info").json()
        accept = info["internal_states"][0]["avg_spec_accept_length"]
        print(f"{metrics=} {accept=}")
        self.assertGreater(metrics["score"], GSM8K_BOUND)
        self.assertGreater(accept, MTP_ACCEPT_BOUND)


class TestMonokernelCheck(CustomTestCase):
    def test_each_covered_layer_within_its_numerics(self):
        # SGLANG_AMD_MONO_CHECK (eager) runs the normal layer next to each layer's
        # launch, and SGLang's final norm and logits next to the tail; 1, 3, 8, 11
        # and 16 rows run at widths 2, 4, 8, 12 and 16 (pad rows at 1, 3 and 11),
        # and the long prompt's positions pass 2048, where the sparse attention
        # selects keys
        para = "The river valley archive lists the flood height of every year. "
        with (
            tempfile.NamedTemporaryFile("w+", suffix=".out") as out,
            tempfile.NamedTemporaryFile("w+", suffix=".err") as err,
        ):
            process = _launch(
                extra_env={"SGLANG_AMD_MONO_CHECK": "1"},
                extra_args=["--disable-cuda-graph"],
                log=(out, err),
            )
            try:
                for n, repeats in ((1, 2), (3, 2), (8, 2), (11, 2), (16, 2), (1, 200)):
                    r = requests.post(
                        DEFAULT_URL_FOR_TEST + "/generate",
                        json={
                            "text": [
                                f"Record {i}. " + para * repeats for i in range(n)
                            ],
                            "sampling_params": {"max_new_tokens": 4, "temperature": 0},
                        },
                        timeout=600,
                    )
                    r.raise_for_status()
            finally:
                terminate_and_kill_process_tree(process)
            err.seek(0)
            out.seek(0)
            log = err.read() + out.read()
        checks = re.findall(
            r"monokernel check layer \d+ \((kda|dsa|dense), (\d+) rows, positions "
            r"\d+\.\.(\d+)\): (.*)",
            log,
        )
        values = {}
        for _, _, _, fields in checks:
            for name, v in re.findall(r"(\S+)=(\S+)", fields):
                values.setdefault(name, []).append(float(v))
        for name, vs in sorted(values.items()):
            print(f"{name}: min {min(vs):.6g} max {max(vs):.6g} ({len(vs)} checks)")
        seen = {(kind, int(rows)) for kind, rows, _, _ in checks}
        covered = (1, 3, 8, 11, 16)
        want = {(kind, rows) for kind in ("kda", "dsa", "dense") for rows in covered}
        self.assertTrue(want <= seen, seen)
        tails = re.findall(r"monokernel check tail \((\d+) rows\): (.*)", log)
        self.assertEqual({int(rows) for rows, _ in tails} & set(covered), set(covered))
        self.assertTrue(any(k == "dsa" and int(hi) >= 2048 for k, _, hi, _ in checks))
        violations = re.findall(r"monokernel check violation: .*", log)
        self.assertEqual(violations, [], "\n".join(violations[:20]))


def _lm_head() -> torch.Tensor:
    """The checkpoint's LM head [vocab, hidden]."""
    import json

    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    def path(name):
        if os.path.isdir(MODEL):
            return os.path.join(MODEL, name)
        return hf_hub_download(MODEL, name)

    index = json.load(open(path("model.safetensors.index.json")))
    with safe_open(path(index["weight_map"]["lm_head.weight"]), "pt") as f:
        return f.get_tensor("lm_head.weight")


class TestMonokernelHiddenStates(CustomTestCase):
    def test_returned_hidden_states_give_the_sampled_tokens(self):
        # each covered decode step's returned row is the final norm's: times the LM
        # head, its argmax is the token the server sampled (greedy)
        process = _launch(extra_args=["--enable-return-hidden-states"])
        try:
            r = requests.post(
                DEFAULT_URL_FOR_TEST + "/generate",
                json={
                    "text": [
                        "The capital of France is",
                        "Count from one to ten: one, two,",
                        "Write a haiku about the sea.",
                    ],
                    "sampling_params": {"max_new_tokens": 8, "temperature": 0},
                    "return_hidden_states": True,
                },
                timeout=600,
            )
            r.raise_for_status()
        finally:
            terminate_and_kill_process_tree(process)
        head = _lm_head().float()
        hits = rows = 0
        for out in r.json():
            # hidden_states[0] is the prefill's block; row i the step sampling token i
            decode = out["meta_info"]["hidden_states"][1:]
            got = (torch.tensor(decode) @ head.T).argmax(-1)
            want = torch.tensor(out["output_ids"][1 : 1 + len(decode)])
            hits += int((got == want).sum())
            rows += len(decode)
        print(f"decode rows giving the sampled token: {hits} / {rows}")
        self.assertGreaterEqual(rows, 15)
        # bf16 near ties may part one row's argmax
        self.assertGreaterEqual(hits, rows - 1)


class TestMonokernelFault(CustomTestCase):
    def test_a_stalled_rank_stops_the_server_without_tokens(self):
        # past the warmup (20 steps of a width), rank 1 stalls at its first wait;
        # at the deadline (half the watchdog) it stops, and its peers stop on its flag
        with (
            tempfile.NamedTemporaryFile("w+", suffix=".out") as out,
            tempfile.NamedTemporaryFile("w+", suffix=".err") as err,
        ):
            process = _launch(
                extra_env={
                    "SGLANG_AMD_MONO_FAULT_INJECT": "1:1:20",
                    # the crash dumps (each rank's VRAM, py-spy of each scheduler with
                    # no timeout, a 60 s wait) can outlast the wait for the stop below
                    "HSA_DISABLE_COREDUMP_ON_EXCEPTION": "1",
                    "SGLANG_PYSPY_DUMP_BEFORE_CRASH": "0",
                    "SGLANG_CUDA_COREDUMP_BEFORE_CRASH": "0",
                },
                extra_args=["--watchdog-timeout", "20"],
                log=(out, err),
            )
            try:
                texts = []

                def generate():
                    r = requests.post(
                        DEFAULT_URL_FOR_TEST + "/generate",
                        json={
                            "text": ["Write a long essay about lighthouses."] * 2,
                            "sampling_params": {
                                "max_new_tokens": 1024,
                                "temperature": 0,
                                "ignore_eos": True,
                            },
                        },
                        timeout=300,
                    )
                    if r.status_code == 200:
                        texts.extend(item.get("text") for item in r.json())

                try:
                    generate()
                except requests.RequestException:
                    pass
                self.assertFalse(
                    any(texts), "a failed step's tokens reached the client"
                )
                deadline = time.time() + 300
                while process.poll() is None and time.time() < deadline:
                    time.sleep(2)
                self.assertIsNotNone(process.poll(), "the server kept running")
                err.seek(0)
                out.seek(0)
                log = err.read() + out.read()
                # a launch never runs on after a failed wait: it stops with a store
                # to the null page, so the only faults are at address 0x0
                faults = re.findall(r"faulting addr: (0x[0-9a-fA-F]+)", log)
                self.assertTrue(faults, log[-2000:])
                self.assertEqual(set(faults) - {"0x0"}, set(), faults)
            finally:
                terminate_and_kill_process_tree(process)


if __name__ == "__main__":
    unittest.main()
