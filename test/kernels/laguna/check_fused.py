"""Check Laguna's S3 launch with the shared expert fused in (build shared=True) on one gfx950 GPU
against SGLang's path: aiter fused_moe (routed) + the quark W4A4 shared expert, added in bf16.
Also times it against the separate S3 (routed only) and S4 (shared) launches."""
import argparse, sys, torch, torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("--widths", default="1,2,4,8,16")
ap.add_argument("--replays", type=int, default=30)
a = ap.parse_args()
sys.path.insert(0, "/workspace/claude-skills/proxy-moe/scripts")
import ep_moe_check as E  # noqa: E402
from sgl_amd_mono.laguna import INTER_TOTAL, TP  # noqa: E402
E.I, E.E_LOCAL = INTER_TOTAL // TP, 512
from aiter import ActivationType, QuantType  # noqa: E402
from aiter.fused_moe import fused_moe  # noqa: E402
from aiter.ops.flydsl.moe_common import GateMode  # noqa: E402
from aiter.ops.triton.quant import dynamic_mxfp4_quant  # noqa: E402
from aiter.ops.triton.gemm_afp4wfp4 import gemm_afp4wfp4  # noqa: E402
from sgl_amd_mono.common import guard  # noqa: E402
from sgl_amd_mono.laguna.gfx950 import routed, shared  # noqa: E402

dev = torch.device("cuda", 0)
H, I = routed.HIDDEN, routed.INTER
widths = [int(v) for v in a.widths.split(",")]
w1, w2, w1s, w2s, _, cases = E.build(widths)
g = torch.Generator(device=dev).manual_seed(11)
sw13 = torch.randint(0, 256, (2 * I, H // 2), dtype=torch.uint8, device=dev, generator=g)
ss13 = torch.randint(118, 124, (2 * I, H // 32), dtype=torch.uint8, device=dev, generator=g)
sw2 = torch.randint(0, 256, (H, I // 2), dtype=torch.uint8, device=dev, generator=g)
ss2 = torch.randint(118, 124, (H, I // 32), dtype=torch.uint8, device=dev, generator=g)
scr = torch.zeros(routed.SCRATCH_BYTES // 4, dtype=torch.int32, device=dev)
scr4 = torch.zeros(shared.SCRATCH_BYTES // 4, dtype=torch.int32, device=dev)
epoch = torch.zeros(64, dtype=torch.int32, device=dev)
ticks = guard.deadline_ticks(10.0); lo = ticks & 0xFFFFFFFF
epoch[guard.GUARD_DEADLINE] = lo - (1 << 32) if lo >= 1 << 31 else lo
epoch[guard.GUARD_DEADLINE + 1] = ticks >> 32


def shared_ref(x):
    xq, xs = dynamic_mxfp4_quant(x)
    gu = gemm_afp4wfp4(xq, sw13, xs, ss13, torch.bfloat16)
    mid = F.silu(gu[:, :I]) * gu[:, I:]  # SGLang's SiluAndMul: bf16(silu(gate)) * up
    mq, ms = dynamic_mxfp4_quant(mid)
    return gemm_afp4wfp4(mq, sw2, ms, ss2, torch.bfloat16)


def routed_ref(x, wt, ids):
    return fused_moe(x, w1, w2, wt, ids, quant_type=QuantType.per_1x32, activation=ActivationType.Silu,
                     w1_scale=w1s, w2_scale=w2s, gate_mode=GateMode.SEPARATED.value)


def rel(p, q):
    return ((p.float() - q.float()).norm() / q.float().norm()).item()


def timed(fn):
    fn(); epoch[:1].add_(1)
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        fn(); epoch[:1].add_(1)
    gr.replay(); torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(a.replays):
        gr.replay()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) * 1e3 / a.replays


ok = True
for m in widths:
    x, wt, ids = cases[m]
    out = torch.empty(m, H, dtype=torch.bfloat16, device=dev)
    o3 = torch.empty_like(out); o4 = torch.empty_like(out)
    fz = routed.build_routed(m, shared=True)
    s3 = routed.build_routed(m)
    s4 = shared.build_shared(m)
    run_f = lambda: fz(x.data_ptr(), ids.data_ptr(), wt.data_ptr(), w1.data_ptr(), w1s.data_ptr(), w2.data_ptr(),
                       w2s.data_ptr(), out.data_ptr(), scr.data_ptr(), epoch.data_ptr(), 0, 0,
                       sw13.data_ptr(), ss13.data_ptr(), sw2.data_ptr(), ss2.data_ptr(), stream=torch.cuda.current_stream())
    run_3 = lambda: s3(x.data_ptr(), ids.data_ptr(), wt.data_ptr(), w1.data_ptr(), w1s.data_ptr(), w2.data_ptr(),
                       w2s.data_ptr(), o3.data_ptr(), scr.data_ptr(), epoch.data_ptr(), 0, 0, stream=torch.cuda.current_stream())
    run_4 = lambda: s4(x.data_ptr(), sw13.data_ptr(), ss13.data_ptr(), sw2.data_ptr(), ss2.data_ptr(), o4.data_ptr(),
                       scr4.data_ptr(), epoch.data_ptr(), 1, stream=torch.cuda.current_stream())
    run_f(); epoch[:1].add_(1); run_3(); epoch[:1].add_(1); run_4(); epoch[:1].add_(1); torch.cuda.synchronize()
    ref = (routed_ref(x, wt, ids) + shared_ref(x)).to(torch.bfloat16)
    ref2 = (routed_ref(x, wt, ids) + shared_ref(x)).to(torch.bfloat16)
    mine = (o3.float() + o4.float()).to(torch.bfloat16)  # the separate S3 + S4 launches, added as SGLang does
    r = rel(out, ref)
    r_own = rel(out, mine)
    tf, t3, t4 = timed(run_f), timed(run_3), timed(run_4)
    err = [int(epoch[i]) for i in guard.ERROR_WORDS]
    print(f"M={m:2d} fused vs S3+S4 {r_own:.2e} | vs SGLang (routed + shared) {r:.2e} "
          f"(S3+S4 vs SGLang {rel(mine, ref):.2e}, SGLang vs itself {rel(ref2, ref):.2e}) "
          f"errors={err} | fused {tf:.1f} us vs S3 {t3:.1f} + S4 {t4:.1f} = {t3 + t4:.1f} us", flush=True)
    ok &= err == [0, 0, 0] and r_own < 5e-3
print("PASS" if ok else "FAIL")
