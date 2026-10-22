"""Check Laguna's S4 shared-expert launch on one gfx950 GPU against SGLang's quark W4A4
path (triton dynamic_mxfp4_quant + gemm_afp4wfp4, silu * up, requant, gemm_afp4wfp4);
intermediate MID / requant and the output; graph replay stability and time."""
import argparse, torch, torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("--widths", default="1,2,4,8,16")
ap.add_argument("--replays", type=int, default=200)
a = ap.parse_args()
dev = torch.device("cuda", 0)
from sgl_amd_mono.common import guard  # noqa: E402
from sgl_amd_mono.laguna.gfx950 import shared  # noqa: E402
from aiter.ops.triton.quant import dynamic_mxfp4_quant  # noqa: E402
from aiter.ops.triton.gemm_afp4wfp4 import gemm_afp4wfp4  # noqa: E402

H, I = shared.HIDDEN, shared.INTER
g = torch.Generator(device=dev).manual_seed(11)
w13 = torch.randint(0, 256, (2 * I, H // 2), dtype=torch.uint8, device=dev, generator=g)
s13 = torch.randint(118, 124, (2 * I, H // 32), dtype=torch.uint8, device=dev, generator=g)
w2 = torch.randint(0, 256, (H, I // 2), dtype=torch.uint8, device=dev, generator=g)
s2 = torch.randint(118, 124, (H, I // 32), dtype=torch.uint8, device=dev, generator=g)
scratch = torch.zeros(shared.SCRATCH_BYTES // 4, dtype=torch.int32, device=dev)
epoch = torch.zeros(64, dtype=torch.int32, device=dev)
ticks = guard.deadline_ticks(10.0); lo = ticks & 0xFFFFFFFF
epoch[guard.GUARD_DEADLINE] = lo - (1 << 32) if lo >= 1 << 31 else lo
epoch[guard.GUARD_DEADLINE + 1] = ticks >> 32


def reference(x):
    xq, xs = dynamic_mxfp4_quant(x)
    gu = gemm_afp4wfp4(xq, w13, xs, s13, torch.bfloat16)
    gte, up = gu[:, :I].float(), gu[:, I:].float()
    mid = (F.silu(gte) * up).to(torch.bfloat16)
    mq, ms = dynamic_mxfp4_quant(mid)
    return gemm_afp4wfp4(mq, w2, ms, s2, torch.bfloat16), mid, mq


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


ok = True
for m in [int(v) for v in a.widths.split(",")]:
    launch = shared.build_shared(m)
    x = torch.randn(m, H, generator=g, device=dev).to(torch.bfloat16)
    out = torch.empty(m, H, dtype=torch.bfloat16, device=dev)
    args = (x.data_ptr(), w13.data_ptr(), s13.data_ptr(), w2.data_ptr(), s2.data_ptr(), out.data_ptr(),
            scratch.data_ptr(), epoch.data_ptr(), 0)
    launch(*args, stream=torch.cuda.current_stream()); epoch[0] += 1; torch.cuda.synchronize()
    ref, rmid, rmq = reference(x)
    mid = scratch.view(torch.uint8)[shared.MID:shared.MID + shared.MAX_TOKENS * I * 2].view(torch.bfloat16).view(-1, I)[:m]
    mq = scratch.view(torch.uint8)[shared.MQ:shared.MQ + shared.MAX_TOKENS * I // 2].view(-1, I // 2)[:m]
    mid_eq = (mid == rmid).float().mean().item()
    mq_eq = (mq == rmq.view(torch.uint8)[:m]).float().mean().item()
    r_mid, r_out = rel(mid, rmid), rel(out, ref)
    graph = torch.cuda.CUDAGraph()
    first = out.clone()
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
    print(f"M={m:2d} MID bf16 equal {mid_eq*100:.2f}% rel {r_mid:.1e} | MID fp4 bytes equal {mq_eq*100:.2f}% | "
          f"out rel_l2 {r_out:.2e} | stable={stable} errors={err} | {us:.1f} us/launch", flush=True)
    ok &= stable and err == [0, 0, 0] and r_out < 2e-2
print("PASS" if ok else "FAIL")
