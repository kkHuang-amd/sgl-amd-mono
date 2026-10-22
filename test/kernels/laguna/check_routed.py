"""Check Laguna's S3 routed-expert launch on one gfx950 GPU against aiter fused_moe (the op
SGLang runs; its stage 2 accumulates with atomics, so it differs from itself run to run), at
the proxy EP1/TP8 shape with real-layout (shuffled) random weights; graph stability and time."""
import argparse, sys, torch

ap = argparse.ArgumentParser()
ap.add_argument("--widths", default="1,2,4,8,16")
ap.add_argument("--replays", type=int, default=50)
a = ap.parse_args()
sys.path.insert(0, "/workspace/claude-skills/proxy-moe/scripts")
import ep_moe_check as E  # noqa: E402
E.I, E.E_LOCAL = 512, 512
from aiter import ActivationType, QuantType  # noqa: E402
from aiter.fused_moe import fused_moe  # noqa: E402
from aiter.ops.flydsl.moe_common import GateMode  # noqa: E402
from sgl_amd_mono.common import guard  # noqa: E402
from sgl_amd_mono.laguna.gfx950 import routed  # noqa: E402

dev = torch.device("cuda", 0)
H, K = routed.HIDDEN, routed.TOPK
widths = [int(v) for v in a.widths.split(",")]
w1, w2, w1s, w2s, _, cases = E.build(widths)
scratch = torch.zeros(routed.SCRATCH_BYTES // 4, dtype=torch.int32, device=dev)
epoch = torch.zeros(64, dtype=torch.int32, device=dev)
ticks = guard.deadline_ticks(10.0); lo = ticks & 0xFFFFFFFF
epoch[guard.GUARD_DEADLINE] = lo - (1 << 32) if lo >= 1 << 31 else lo
epoch[guard.GUARD_DEADLINE + 1] = ticks >> 32


def ref(x, wt, ids):
    return fused_moe(x, w1, w2, wt, ids, quant_type=QuantType.per_1x32, activation=ActivationType.Silu,
                     w1_scale=w1s, w2_scale=w2s, gate_mode=GateMode.SEPARATED.value)


def rel(p, q):
    return ((p.float() - q.float()).norm() / q.float().norm()).item()


ok = True
for m in widths:
    x, wt, ids = cases[m]
    launch = routed.build_routed(m)
    out = torch.empty(m, H, dtype=torch.bfloat16, device=dev)
    args = (x.data_ptr(), ids.data_ptr(), wt.data_ptr(), w1.data_ptr(), w1s.data_ptr(), w2.data_ptr(),
            w2s.data_ptr(), out.data_ptr(), scratch.data_ptr(), epoch.data_ptr(), 0)
    launch(*args, stream=torch.cuda.current_stream()); epoch[0] += 1; torch.cuda.synchronize()
    r1, r2 = ref(x, wt, ids), ref(x, wt, ids)
    graph = torch.cuda.CUDAGraph(); first = out.clone()
    with torch.cuda.graph(graph):
        launch(*args, stream=torch.cuda.current_stream()); epoch[0] += 1
    graph.replay(); torch.cuda.synchronize()
    stable = torch.equal(first, out)
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(a.replays):
        graph.replay()
    e.record(); torch.cuda.synchronize()
    us = s.elapsed_time(e) * 1e3 / a.replays
    err = [int(epoch[i]) for i in guard.ERROR_WORDS]
    rr = rel(out, r1)
    print(f"M={m:2d} union={int(torch.unique(ids).numel())} | out vs fused_moe rel_l2 {rr:.2e} "
          f"(fused_moe vs itself {rel(r2, r1):.2e}) | stable={stable} errors={err} | {us:.1f} us/launch", flush=True)
    ok &= stable and err == [0, 0, 0] and rr < 1e-2
print("PASS" if ok else "FAIL")
