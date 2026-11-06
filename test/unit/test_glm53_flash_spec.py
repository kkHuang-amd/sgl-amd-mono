"""The GLM-5.3-Flash gfx950 monokernel spec on CPU (RFC #43949): the layers its
forward covers on the GLM-5.3-Flash layer structure, its row routing and TP2 row
cap, the tail's logits in place of the logits processor's, its registry entry, its
config and model refusal reasons, and the dense, embedding and LM-head views. SGLang
checks its declared requirements, the same way for every spec."""

import math
from types import SimpleNamespace

import pytest
import torch

try:
    from sglang.kernels.plugin_kernels.amd import spec as registry
except ModuleNotFoundError:  # SGLang before sgl-project/sglang#45389
    from sglang.srt.monokernels import spec as registry

import sgl_amd_mono
from sgl_amd_mono.glm53_flash import HF_CONFIG, LINEAR_ATTN_CONFIG
from sgl_amd_mono.glm53_flash import spec as flash

# GLM-5.3-Flash's text-model layer structure (its config.json)
TEXT = SimpleNamespace(
    **HF_CONFIG,
    num_hidden_layers=45,
    first_k_dense_replace=3,
    linear_attn_config={
        **LINEAR_ATTN_CONFIG,
        "kda_layers": [i for i in range(45) if i % 4 != 3],
        "full_attn_layers": [i for i in range(45) if i % 4 == 3],
    },
)
FLASH = SimpleNamespace(
    text_config=TEXT,
    architectures=["Glm5NextForConditionalGeneration"],
    quantization_config={"quant_method": "quark"},
)


def test_the_forward_covers_every_layer():
    # layers 0 to 2 are dense (KDA and the dense MLP), the rest MoE layers
    assert flash._covered_layers(FLASH) == list(range(45))
    assert [i for i in range(45) if flash._dense(FLASH, i)] == [0, 1, 2]


def test_the_plugin_registers_the_spec():
    # what SGLang's load_plugins() runs, from the sglang.srt.plugins entry point
    sgl_amd_mono.register_specs()
    arch = ["Glm5NextForConditionalGeneration"]
    found, why = registry.find_spec(
        backend=sgl_amd_mono.BACKEND, architectures=arch, isa="gfx950"
    )
    assert found is flash.SPEC, why
    found, why = registry.find_spec(
        backend=sgl_amd_mono.BACKEND, architectures=arch, isa="gfx942"
    )
    assert found is None and "gfx942" in why


class _Logits:
    """A logits processor whose own logits are the rows' sums."""

    def _get_logits(self, hidden_states, lm_head, logits_metadata, **kw):
        return hidden_states.sum(-1, keepdim=True)


class _Lm:
    """A model whose own forward records that it ran."""

    def __init__(self):
        self.model, self.ran = SimpleNamespace(), []
        self.logits_processor = _Logits()

    def forward(self, *args):
        self.ran.append(args)
        return "the model's forward"


def _built(tp=4, widths=(2, 4, 8), verify=1):
    lm = _Lm()
    kinds = ("embed", "dense", "kda", "dsa", "head")
    runtimes = {(w, k): None for w in widths for k in kinds}
    layers = {i: SimpleNamespace(i=i) for i in range(45)}
    built = flash._Built(lm, layers, runtimes, {}, tp=tp, device="cpu", verify=verify)
    return lm, built


def _plan(rows, tokens=1):
    key = registry.ShapeKey(
        forward_mode="decode" if tokens == 1 else "target_verify",
        batch_size=rows // tokens,
        tokens_per_request=tokens,
    )
    return registry.Plan(key=key)


def test_a_step_runs_at_the_next_width_and_tp2_takes_four_rows():
    # a step of fewer rows than a width runs at the next width, the rest pad rows
    assert [flash._width(r) for r in flash.ROWS] == [2, 2, 4, 4] + [8] * 4 + [
        12
    ] * 4 + [16] * 4
    # TP2 builds no 8-row launch: a wider step runs the model's own forward
    lm, built = _built(tp=2, widths=(2, 4))
    out = built.forward(_plan(8), "ids", "positions", "batch")
    assert out == "the model's forward"
    assert lm.ran == [("ids", "positions", "batch", None, None)]
    # layer 0's input besides its tokens: every stream a copy of the embedding row
    assert not built.zero_x.any() and not built.zero_res.any()
    assert (built.one_post == 1).all() and not built.zero_comb.any()
    # the width's rows past the step's read page-table row 0 until layer 0 writes them
    assert set(built.rows) == {2, 4} and (built.rows[4][:, 4] == 0).all()


def test_a_verify_step_of_other_draft_chains_runs_the_model_s_forward():
    # the launches are built for one draft-chain length (the server's draft tokens)
    lm, built = _built(verify=4)
    out = built.forward(_plan(8, tokens=2), "ids", "positions", "batch")
    assert out == "the model's forward" and len(lm.ran) == 1
    # TP2: a verify step past 4 rows runs the model's forward too
    lm, built = _built(tp=2, widths=(2, 4), verify=4)
    out = built.forward(_plan(8, tokens=4), "ids", "positions", "batch")
    assert out == "the model's forward" and len(lm.ran) == 1


def test_a_step_from_embeddings_runs_the_model_s_forward():
    # layer 0's launch reads the step's tokens; given embeddings, SGLang's path runs
    lm, built = _built()
    out = built.forward(_plan(2), "ids", "positions", "batch", input_embeds="emb")
    assert out == "the model's forward" and lm.ran[0][3] == "emb"


def test_the_logits_processor_returns_the_tail_s_logits_on_a_routed_step():
    lm, built = _built()
    lp, h = lm.logits_processor, torch.ones(3, 5)
    # not routed: the processor's own logits
    assert torch.equal(lp._get_logits(h, None, None), torch.full((3, 1), 5.0))
    # routed: the tail's, the step's rows of the width's
    built._logits = torch.arange(8.0).view(4, 2)
    assert torch.equal(lp._get_logits(h, None, None), torch.arange(6.0).view(3, 2))


def _cfg(**kw):
    return SimpleNamespace(
        **{
            "page_size": 256,
            "speculative_algorithm": None,
            "linear_attn_backend": "triton",
            "linear_attn_decode_backend": None,
            "linear_attn_verify_backend": None,
            **kw,
        }
    )


def _dtypes(temporal=torch.float32, conv=torch.bfloat16):
    return lambda config: SimpleNamespace(temporal=temporal, conv=conv)


def test_config_reasons_name_each_mismatch(monkeypatch):
    import sglang.srt.configs.mamba_utils as mamba_utils
    import sglang.srt.layers.attention.dsa.utils as dsa_utils

    monkeypatch.setattr(mamba_utils, "mamba2_state_dtype", _dtypes())
    monkeypatch.setattr(dsa_utils, "aiter_can_use_preshuffle_paged_mqa", lambda: True)
    assert flash.SPEC.config_disable_reasons(_cfg(), FLASH) == []
    kda = TEXT.linear_attn_config["kda_layers"]
    text = SimpleNamespace(
        **{
            **vars(TEXT),
            "hidden_size": 6144,
            "moe_layer_freq": 2,
            "linear_attn_config": {
                **TEXT.linear_attn_config,
                "num_heads": 32,
                "kda_layers": kda,
            },
        }
    )
    monkeypatch.setattr(
        mamba_utils, "mamba2_state_dtype", _dtypes(temporal=torch.bfloat16)
    )
    # SGLang's k-pool writer keeps the index cache row-major; the kernels write tiles
    monkeypatch.setattr(dsa_utils, "aiter_can_use_preshuffle_paged_mqa", lambda: False)
    other = SimpleNamespace(
        text_config=text,
        architectures=FLASH.architectures,
        quantization_config={"quant_method": "fp8"},
    )
    reasons = flash.SPEC.config_disable_reasons(_cfg(page_size=64), other)
    joined = " | ".join(reasons)
    for want in (
        "hidden_size = 4096",
        "moe_layer_freq = 1",
        "num_heads = 64",
        "KDA state in f32",
        "--page-size 64",
        "preshuffled layout",
        "Quark checkpoint",
    ):
        assert want in joined, joined
    assert len(reasons) == 7, reasons


@pytest.mark.parametrize(
    "topk, steps, draft, ok",
    [(1, 3, 4, True), (1, 5, 6, True), (4, 3, 8, False), (1, 3, 3, False)],
)
def test_mtp_verifies_one_draft_chain(monkeypatch, topk, steps, draft, ok):
    # a verify request's rows are one chain of draft tokens, each the next position
    import sglang.srt.configs.mamba_utils as mamba_utils
    import sglang.srt.layers.attention.dsa.utils as dsa_utils

    monkeypatch.setattr(mamba_utils, "mamba2_state_dtype", _dtypes())
    monkeypatch.setattr(dsa_utils, "aiter_can_use_preshuffle_paged_mqa", lambda: True)
    cfg = _cfg(
        speculative_algorithm="EAGLE",
        speculative_eagle_topk=topk,
        speculative_num_steps=steps,
        speculative_num_draft_tokens=draft,
    )
    reasons = flash.SPEC.config_disable_reasons(cfg, FLASH)
    assert (reasons == []) == ok, reasons


@pytest.mark.parametrize(
    "spec, backends, ok",
    [
        ("EAGLE", {}, True),
        ("EAGLE", {"linear_attn_verify_backend": "triton"}, True),
        ("EAGLE", {"linear_attn_verify_backend": "flashinfer"}, False),
        ("EAGLE", {"linear_attn_decode_backend": "flashinfer"}, False),
        ("EAGLE", {"linear_attn_backend": "flashinfer"}, False),
        (
            "EAGLE",
            {
                "linear_attn_decode_backend": "flashinfer",
                "linear_attn_verify_backend": "triton",
            },
            True,
        ),
        (None, {"linear_attn_backend": "flashinfer"}, True),
    ],
)
def test_mtp_refuses_the_fused_accept(monkeypatch, spec, backends, ok):
    # SGLang's FlashInfer KDA verify skips the KDA state commit a verify launch
    # reads; unset, the verify backend follows a FlashInfer decode backend
    import sglang.srt.configs.mamba_utils as mamba_utils
    import sglang.srt.layers.attention.dsa.utils as dsa_utils

    monkeypatch.setattr(mamba_utils, "mamba2_state_dtype", _dtypes())
    monkeypatch.setattr(dsa_utils, "aiter_can_use_preshuffle_paged_mqa", lambda: True)
    cfg = _cfg(
        speculative_algorithm=spec,
        speculative_eagle_topk=1,
        speculative_num_steps=3,
        speculative_num_draft_tokens=4,
        **backends,
    )
    reasons = flash.SPEC.config_disable_reasons(cfg, FLASH)
    assert (reasons == []) == ok, reasons
    assert ok or "fuses the accept" in reasons[0], reasons


def test_the_spec_requires_sglang_jit_grouped_topk_off():
    # the kernel's router ties follow AITER's grouped top-k; SGLang's JIT grouped
    # top-k (a Triton gate) keeps the lower expert id, so the spec requires it off
    # (SGLang refuses a setup against a spec's declared env)
    assert flash.SPEC.requires.env["SGLANG_OPT_USE_JIT_KERNEL_GROUPED_TOPK"] is False


def _layer(kda=True, shuffled=True, scale_dims=3, fused=True, indexer=True):
    w13 = SimpleNamespace(is_shuffled=shuffled)
    experts = SimpleNamespace(
        w13_weight=w13, w13_weight_scale=torch.zeros((1,) * scale_dims)
    )
    attn = SimpleNamespace(
        do_fuse_qkvbfg=fused, indexer=SimpleNamespace() if indexer else None
    )
    return SimpleNamespace(
        is_linear_attn=kda, self_attn=attn, mlp=SimpleNamespace(experts=experts)
    )


def _lp(**kw):
    """A logits processor as SGLang's TP4 one (bf16 head, TP all-gather)."""
    return SimpleNamespace(
        **{
            "do_tensor_parallel_all_gather": True,
            "do_tensor_parallel_all_gather_dp_attn": False,
            "use_attn_tp_group": False,
            "use_fp32_lm_head": False,
            "logit_scale": None,
            "final_logit_softcapping": None,
            **kw,
        }
    )


def test_model_reasons(monkeypatch):
    import sglang.srt.runtime_context as runtime_context

    import sgl_amd_mono.glm53_flash.gfx950.weights as weights

    monkeypatch.setattr(
        runtime_context, "get_parallel", lambda: SimpleNamespace(tp_size=4)
    )
    refused = {}

    def view(name):
        def f(*args):
            if name in refused:
                raise ValueError(refused[name])

        return f

    monkeypatch.setattr(weights, "head_weights", view("head"))
    monkeypatch.setattr(weights, "embed_weights", view("embed"))
    monkeypatch.setattr(weights, "dense_weights", view("dense"))
    kda = TEXT.linear_attn_config["kda_layers"]
    layers = [_layer(kda=i in kda) for i in range(45)]
    model = SimpleNamespace(
        model=SimpleNamespace(layers=layers),
        is_mrope_enabled=False,
        logits_processor=_lp(),
    )
    assert flash.SPEC.model_disable_reasons(model, FLASH, {}) == []
    # layer 4's experts are not shuffled, layer 5's are GU-interleaved (2-D
    # scales), layer 6 (KDA) runs unfused projections, layer 7 has no indexer
    layers[4] = _layer(shuffled=False)
    layers[5] = _layer(scale_dims=2)
    layers[6] = _layer(fused=False)
    layers[7] = _layer(kda=False, indexer=False)
    model.is_mrope_enabled = True
    reasons = flash.SPEC.model_disable_reasons(model, FLASH, {})
    assert [r.split(":")[0] for r in reasons] == [
        "multimodal rotary positions",
        "layer 4",
        "layer 5",
        "layer 6",
        "layer 7",
    ], reasons
    assert "shuffled" in reasons[1] and "GU-interleaved" in reasons[2]


def test_model_reasons_of_the_tail_the_embedding_and_the_dense_layers(monkeypatch):
    import sglang.srt.runtime_context as runtime_context

    import sgl_amd_mono.glm53_flash.gfx950.weights as weights

    monkeypatch.setattr(
        runtime_context, "get_parallel", lambda: SimpleNamespace(tp_size=4)
    )

    def refuse(why):
        def f(*args):
            raise ValueError(why)

        return f

    monkeypatch.setattr(weights, "head_weights", refuse("an fp8 LM head"))
    monkeypatch.setattr(weights, "embed_weights", refuse("padded rows"))
    monkeypatch.setattr(weights, "dense_weights", refuse("needs block FP8"))
    kda = TEXT.linear_attn_config["kda_layers"]
    model = SimpleNamespace(
        model=SimpleNamespace(layers=[_layer(kda=i in kda) for i in range(45)]),
        is_mrope_enabled=False,
        logits_processor=_lp(
            use_attn_tp_group=True,
            use_fp32_lm_head=True,
            logit_scale=0.5,
            final_logit_softcapping=30.0,
        ),
    )
    reasons = flash.SPEC.model_disable_reasons(model, FLASH, {})
    joined = " | ".join(reasons)
    for want in (
        "all-gathers the logits",
        "LM head in bf16",
        "logit_scale",
        "final_logit_softcapping",
        "LM head: an fp8 LM head",
        "embedding: padded rows",
        "layer 0: needs block FP8",
        "layer 1: needs block FP8",
        "layer 2: needs block FP8",
    ):
        assert want in joined, joined
    assert len(reasons) == 9, reasons


def _fp8_linear(rows, cols, shuffled=True, dtype=torch.float8_e4m3fn):
    w = torch.zeros(rows, cols, dtype=dtype)
    linear = SimpleNamespace(weight=w, aiter_bpreshuffled=shuffled)
    if dtype == torch.float8_e4m3fn:
        linear.weight_scale_inv = torch.ones(rows // 128, cols // 128)
    return linear


def _dense_layer(tp, ug=None, dn=None, limit=10.0):
    from sgl_amd_mono.glm53_flash.gfx950.config import DENSE_INTER, HIDDEN

    inter = DENSE_INTER // tp
    mlp = SimpleNamespace(
        swiglu_limit=limit,
        gate_up_proj=ug or _fp8_linear(2 * inter, HIDDEN),
        down_proj=dn or _fp8_linear(HIDDEN, inter),
    )
    return SimpleNamespace(
        mlp=mlp,
        hc_ffn_fn=torch.zeros(24, 4 * HIDDEN),
        hc_ffn_scale=torch.zeros(3),
        hc_ffn_base=torch.zeros(24),
        post_attention_layernorm=SimpleNamespace(
            weight=torch.zeros(HIDDEN, dtype=torch.bfloat16)
        ),
    )


def test_the_dense_mlp_is_read_in_place_and_must_be_block_fp8():
    import pytest

    from sgl_amd_mono.glm53_flash.gfx950.config import DENSE_INTER, HIDDEN
    from sgl_amd_mono.glm53_flash.gfx950.weights import dense_weights

    tp, copies = 8, {}
    layer = _dense_layer(tp)
    w = dense_weights(layer, 0, tp, copies)
    assert w.shuffled and not copies
    assert w.w_ug.data_ptr() == layer.mlp.gate_up_proj.weight.data_ptr()
    inter = DENSE_INTER // tp
    # a bf16 dense MLP (OneNexus-style), mixed layouts and another SwiGLU limit refuse
    bf16 = _dense_layer(tp, ug=_fp8_linear(2 * inter, HIDDEN, dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="block FP8"):
        dense_weights(bf16, 0, tp, copies)
    mixed = _dense_layer(tp, dn=_fp8_linear(HIDDEN, inter, shuffled=False))
    with pytest.raises(ValueError, match="one FP8 layout"):
        dense_weights(mixed, 0, tp, copies)
    with pytest.raises(ValueError, match="SwiGLU limit"):
        dense_weights(_dense_layer(tp, limit=7.0), 0, tp, copies)


def _vocab_module(tp, rank=1, padding=0, added=0, method="UnquantizedEmbeddingMethod"):
    from sgl_amd_mono.glm53_flash.gfx950.config import HIDDEN, VOCAB

    rows = VOCAB // tp
    si = SimpleNamespace(
        org_vocab_start_index=rank * rows,
        org_vocab_end_index=(rank + 1) * rows,
        added_vocab_start_index=VOCAB,
        added_vocab_end_index=VOCAB + added,
        num_org_vocab_padding=padding,
    )
    return SimpleNamespace(
        quant_method=type(method, (), {})(),
        enable_tp=True,
        use_attn_tp_group=False,
        tp_size=tp,
        shard_indices=si,
        output_dtype=None,
        weight=torch.zeros(rows, HIDDEN, dtype=torch.bfloat16),
    )


def test_the_embedding_and_the_lm_head_are_plain_vocab_shards():
    import pytest

    from sgl_amd_mono.glm53_flash.gfx950.config import HIDDEN, VOCAB
    from sgl_amd_mono.glm53_flash.gfx950.weights import embed_weights, head_weights

    tp = 4

    def model(emb, head):
        norm = SimpleNamespace(weight=torch.zeros(HIDDEN, dtype=torch.bfloat16))
        return SimpleNamespace(
            model=SimpleNamespace(embed_tokens=emb, norm=norm), lm_head=head
        )

    m = model(_vocab_module(tp), _vocab_module(tp))
    e = embed_weights(m, tp)
    assert (e.start, e.end) == (VOCAB // tp, 2 * VOCAB // tp)
    assert head_weights(m, tp).w_head.data_ptr() == m.lm_head.weight.data_ptr()
    for bad, match in (
        (_vocab_module(tp, padding=64), "no added or padded rows"),
        (_vocab_module(tp, added=8), "no added or padded rows"),
        (_vocab_module(tp, method="Fp8LinearMethod"), "unquantized"),
        (_vocab_module(2), "sharded over the TP group"),
    ):
        with pytest.raises(ValueError, match=match):
            embed_weights(model(bad, _vocab_module(tp)), tp)
        with pytest.raises(ValueError, match=match):
            head_weights(model(_vocab_module(tp), bad), tp)


def test_every_bound_names_an_output_the_check_reports():
    # a bound the check never reports would be skipped silently
    from sgl_amd_mono.glm53_flash import check

    for name in flash.NUMERICS.bounds:
        kind, out, metric = name.split(".")
        assert out in check.REPORTED[kind] and metric in check.METRICS, name


def test_the_check_rejects_an_output_it_does_not_list():
    import pytest

    from sgl_amd_mono.glm53_flash import check

    with pytest.raises(ValueError, match="dense.mlp_out.rel"):
        check._report("layer 1", {"dense.mlp_out.rel": 0.0})


def test_a_step_s_first_launch_waits_up_to_half_the_watchdog(monkeypatch):
    import sglang.srt.runtime_context as runtime_context

    try:
        from sglang.srt.kernel_plugins.amd.binding import deadline_s
    except ModuleNotFoundError:  # SGLang before sgl-project/sglang#45389
        from sglang.srt.monokernels.guards import deadline_s

    first, other = SimpleNamespace(tokens=True), SimpleNamespace(tokens=False)
    longest = 5497  # the guard's longest deadline, in whole seconds
    # SGLang's default watchdog, the serving test's and the fault test's; a watchdog
    # past twice the longest deadline, and none
    cases = (
        (300, 150.0),
        (600, 300.0),
        (20, 10.0),
        (11100, longest),
        (math.inf, longest),
    )
    for watchdog, entry in cases:
        monkeypatch.setattr(
            runtime_context,
            "get_device",
            lambda w=watchdog: SimpleNamespace(watchdog_timeout=w),
        )
        deadline = deadline_s(watchdog)
        assert flash._deadline(first, deadline, longest) == entry
        assert flash._deadline(other, deadline, longest) == deadline


class _Op:
    """A layer's launch: records its call, returns its four outputs (x is the input's
    plus one)."""

    def __init__(self, i, calls):
        self.i, self.calls = i, calls

    def launch(self, x, residual, post, comb, **kw):
        self.calls.append((self.i, kw))
        return (x + 1, residual, post, comb)


class _Head:
    def __init__(self, calls):
        self.calls = calls

    def launch(self, x, residual, post, comb):
        self.calls.append(("head", {}))
        return torch.cat([x[:, :1], x[:, :1] * 2], 1).float()


class _Rt:
    def __init__(self):
        self.steps = 0

    def advance_step(self):
        self.steps += 1


class _Processor:
    """SGLang's logits processor's call: records the hidden rows it is handed and
    returns its logits of them."""

    def __init__(self):
        self.hidden = None

    def _get_logits(self, hidden_states, lm_head, logits_metadata, **kw):
        raise AssertionError("a routed step returns the tail's logits")

    def __call__(self, input_ids, hidden, lm_head, forward_batch, aux):
        self.hidden = hidden
        return self._get_logits(hidden, lm_head, forward_batch)


# SGLang's layer_boundary import reaches torch.cuda.get_device_properties
_needs_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="SGLang's layer_boundary needs a visible GPU"
)


def _routed_step(monkeypatch, capture=None):
    """One routed decode step of 1 row on fake launches: every layer adds 1 to x; the
    last layer's hc_post adds 100 to x, the stream mean keeps it, the final norm
    doubles it. ``capture``: the step's ``capture_hidden_mode.need_capture()``."""
    import contextlib

    import sglang.srt.layers.layer_boundary as boundary
    import sglang.srt.model_executor.forward_context as forward_context

    from sgl_amd_mono.glm53_flash import check

    calls, tracked = [], []
    kinds = {
        i: "dense" if i < 3 else ("dsa" if i % 4 == 3 else "kda") for i in range(45)
    }
    layers = {}
    for i, kind in kinds.items():
        layers[i] = flash._Layer(i, None, kind, None)
        layers[i].ops[2] = _Op(i, calls)
        layers[i].state = SimpleNamespace(conv_state=f"conv{i}", rec_state=f"rec{i}")
    runtimes = {(w, k): _Rt() for w in (2, 4, 8) for k in ("embed", "kda", "head")}
    lm = _Lm()
    lm.logits_processor = _Processor()
    lm.lm_head = "lm_head"
    last = SimpleNamespace(hc_post=lambda x, residual, comb, post: x + 100)
    lm.model = SimpleNamespace(layers=[last], norm=lambda h: h * 2)
    monkeypatch.setattr(check, "contract", lambda streams: streams)
    built = flash._Built(lm, layers, runtimes, {2: _Head(calls)}, tp=4, device="cpu")
    monkeypatch.setattr(
        flash._Built, "_bind_kda", staticmethod(lambda L, kda, verify: None)
    )
    monkeypatch.setattr(flash._Built, "_bind_dsa", staticmethod(lambda L, verify: None))
    slots = torch.tensor([5], dtype=torch.int32)
    linear = SimpleNamespace(
        forward_metadata=SimpleNamespace(mamba_cache_indices=slots),
        _track_mamba_state_decode=lambda fb, conv, rec, idx, i: tracked.append(
            (conv, i, idx.tolist())
        ),
    )
    full = SimpleNamespace(
        forward_metadata=SimpleNamespace(real_page_table=torch.zeros(1, 4))
    )
    backend = SimpleNamespace(linear_attn_backend=linear, full_attn_backend=full)
    monkeypatch.setattr(forward_context, "get_attn_backend", lambda: backend)
    ctx = SimpleNamespace(maybe_input_scattered=lambda fb: contextlib.nullcontext())
    monkeypatch.setattr(boundary, "get_attn_tp_context", lambda: ctx)
    mode = None if capture is None else SimpleNamespace(need_capture=lambda: capture)
    fb = SimpleNamespace(
        out_cache_loc=torch.tensor([300]),
        req_pool_indices=torch.tensor([7]),
        residual_stream=None,
        capture_hidden_mode=mode,
    )
    ids, pos = torch.tensor([11]), torch.tensor([42])
    out = built.forward(_plan(1), ids, pos, fb)
    return SimpleNamespace(
        out=out,
        calls=calls,
        tracked=tracked,
        kinds=kinds,
        built=built,
        runtimes=runtimes,
        fb=fb,
        hidden=lm.logits_processor.hidden,
    )


@_needs_gpu
def test_a_routed_step_runs_every_layer_and_the_tail(monkeypatch):
    r = _routed_step(monkeypatch)
    calls, built = r.calls, r.built
    # every layer in order, then the tail; layer 0 from the step's tokens, writing the
    # width's tables, which the later KDA and sparse-attention launches read
    assert [c[0] for c in calls] == list(range(45)) + ["head"]
    step = calls[0][1]["step"]
    assert step["ids"].tolist() == [11] and step["slots"] is built.kda_idx[2]
    assert calls[0][1]["state_idx"].tolist() == [5]
    assert all(c[1]["state_idx"] is built.kda_idx[2] for c in calls[1:3])
    assert calls[3][1]["rows"] is built.rows[2]
    # SGLang's track-state copy after each KDA launch (dense layers included)
    assert [t[1] for t in r.tracked] == [i for i, k in r.kinds.items() if k != "dsa"]
    # the logits processor returns the tail's logits: x (0, one added per layer) of
    # the step's row, and twice it
    assert torch.equal(r.out, torch.tensor([[45.0, 90.0]]))
    assert built._logits is None and r.fb.residual_stream is None
    # this width's runtimes moved to the next step, once
    assert [rt.steps for (w, _), rt in r.runtimes.items() if w == 2] == [1, 1, 1]
    assert all(rt.steps == 0 for (w, _), rt in r.runtimes.items() if w != 2)


@_needs_gpu
def test_a_step_that_captures_hidden_states_hands_on_the_normed_rows(monkeypatch):
    # capture on: the final norm's rows of the last layer's output, (45 + 100) x 2,
    # which the logits processor stores as the step's hidden states
    from sgl_amd_mono.glm53_flash.gfx950.config import HIDDEN

    on = _routed_step(monkeypatch, capture=True)
    assert on.hidden.shape == (1, HIDDEN)
    assert torch.equal(on.hidden.float()[:, 0], torch.tensor([290.0]))
    # capture off: the logits processor stores nothing; the last layer's x stands in
    for capture in (False, None):
        off = _routed_step(monkeypatch, capture=capture)
        assert torch.equal(off.hidden.float()[:, 0], torch.tensor([45.0]))
    # the logits are the tail's either way
    assert torch.equal(on.out, off.out)
