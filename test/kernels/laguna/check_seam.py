"""Check Laguna's S1 seam launch (attention AR + residual + RMSNorm) on gfx950, torchrun TP8:
numerics vs an exact torch reference (rank-ordered fp32 sum, bf16 round, fused add-RMSNorm
semantics), and bit-stable CUDA-graph replay of every width, plus the replay time."""
import argparse, os, torch, torch.distributed as dist

ap = argparse.ArgumentParser()
ap.add_argument("--widths", default="1,2,4,8,16")
ap.add_argument("--replays", type=int, default=200)
a = ap.parse_args()
rank, ws = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
torch.cuda.set_device(rank)
dev = torch.device("cuda", rank)
dist.init_process_group("nccl", device_id=dev)
gloo = dist.new_group(backend="gloo")

from sgl_amd_mono.common import guard  # noqa: E402
from sgl_amd_mono.common.peer import PeerBuffer  # noqa: E402
from sgl_amd_mono.laguna.gfx950 import seam  # noqa: E402

H, EPS = seam.HIDDEN, 1e-6
peer = PeerBuffer(seam.peer_bytes(ws), gloo, rank, ws, dev)
scratch = torch.zeros(seam.SS_BYTES // 4, dtype=torch.int32, device=dev)
epoch = torch.zeros(64, dtype=torch.int32, device=dev)
ticks = guard.deadline_ticks(30.0)
lo = ticks & 0xFFFFFFFF
epoch[guard.GUARD_DEADLINE] = lo - (1 << 32) if lo >= 1 << 31 else lo
epoch[guard.GUARD_DEADLINE + 1] = ticks >> 32
g = torch.Generator(device=dev).manual_seed(1)
norm_w = (torch.randn(H, generator=g, device=dev) * 0.1 + 1).to(torch.bfloat16)


def reference(parts, residual):
    s = torch.zeros_like(parts[0], dtype=torch.float32)
    for p in parts:  # rank order
        s = s + p.float()
    # SGLang's fused add + RMSNorm (aiter opus, hidden 16384): squares of the fp32 sum; the
    # first seam.NORM_FP32_COLS columns normalized from it, the rest from the rounded residual
    f = s.to(torch.bfloat16).float() + residual.float()
    res = f.to(torch.bfloat16)
    rstd = torch.rsqrt(f.pow(2).mean(-1, keepdim=True) + EPS)
    v = torch.cat([f[:, :seam.NORM_FP32_COLS], res.float()[:, seam.NORM_FP32_COLS:]], 1)
    x = v * rstd * norm_w.float()
    return x.to(torch.bfloat16), res


def ulps(a, b):
    a, b = a.float(), b.float()
    _, e = torch.frexp(torch.maximum(a.abs(), b.abs()))
    u = (a - b).abs() / torch.ldexp(torch.ones_like(a), e - 8)
    return u.max().item(), (a != b).float().mean().item()


LAYERS = 3
ok = True
for m in [int(x) for x in a.widths.split(",")]:
    # S1 (phase 0) and S5 (phase 1) alternate over LAYERS layers, as the plugin runs them
    s1 = seam.build_seam(ws, m, EPS, phase=0)
    s5 = seam.build_seam(ws, m, EPS, phase=1)
    gp = torch.Generator(device=dev).manual_seed(100 + rank)
    parts_in = [torch.randn(m, H, generator=gp, device=dev).to(torch.bfloat16) for _ in range(2 * LAYERS)]
    gr = torch.Generator(device=dev).manual_seed(7)
    residual0 = (torch.randn(m, H, generator=gr, device=dev) * 4).to(torch.bfloat16)
    outs = [torch.empty(m, H, dtype=torch.bfloat16, device=dev) for _ in range(2 * LAYERS)]
    ress = [torch.empty(m, H, dtype=torch.bfloat16, device=dev) for _ in range(2 * LAYERS)]

    def chain():
        res = residual0
        for j in range(2 * LAYERS):
            launch = s1 if j % 2 == 0 else s5
            launch(parts_in[j].data_ptr(), res.data_ptr(), norm_w.data_ptr(), outs[j].data_ptr(),
                   ress[j].data_ptr(), scratch.data_ptr(), peer.local, peer.addresses.data_ptr(),
                   epoch.data_ptr(), rank, seam.tag_slot(j // 2, j % 2), stream=torch.cuda.current_stream())
            res = ress[j]
        epoch[0] += 1

    chain(); torch.cuda.synchronize()
    ux = ur = fx_ = 0.0
    res = residual0
    for j in range(2 * LAYERS):
        parts = [torch.empty_like(parts_in[j]) for _ in range(ws)]
        dist.all_gather(parts, parts_in[j])
        xr, rr = reference(parts, res)
        u = ulps(outs[j], xr); ux = max(ux, u[0]); fx_ = max(fx_, u[1]); ur = max(ur, ulps(ress[j], rr)[0])
        res = rr
    first = [o.clone() for o in outs]
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        chain()
    dist.barrier()
    graph.replay(); torch.cuda.synchronize(); dist.barrier()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(a.replays):
        graph.replay()
    e.record(); torch.cuda.synchronize()
    us = s.elapsed_time(e) * 1e3 / a.replays / (2 * LAYERS)
    stable = all(torch.equal(f, o) for f, o in zip(first, outs))
    err = [int(epoch[i]) for i in guard.ERROR_WORDS]
    t = torch.tensor([us], device=dev); dist.all_reduce(t, op=dist.ReduceOp.MAX)
    if rank == 0:
        print(f"M={m:2d} {LAYERS} layers S1/S5 chained: x max {ux:.0f} ulp ({fx_ * 100:.3f} % differ: hw rsqrt / reduction order), residual max {ur:.0f} ulp | "
              f"graph stable={stable} errors={err} | {t.item():.1f} us/launch (slowest rank)", flush=True)
    ok &= stable and err == [0, 0, 0] and ur == 0 and ux <= 1 and fx_ < 1e-3
dist.barrier()
peer.close()
if rank == 0:
    print("PASS" if ok else "FAIL", flush=True)
