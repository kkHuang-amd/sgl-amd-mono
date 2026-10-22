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
    res = (s.to(torch.bfloat16).float() + residual.float()).to(torch.bfloat16)
    var = res.float().pow(2).mean(-1, keepdim=True)
    x = (res.float() * torch.rsqrt(var + EPS)).to(torch.bfloat16).float() * norm_w.float()
    return x.to(torch.bfloat16), res


def ulps(a, b):
    a, b = a.float(), b.float()
    _, e = torch.frexp(torch.maximum(a.abs(), b.abs()))
    u = (a - b).abs() / torch.ldexp(torch.ones_like(a), e - 8)
    return u.max().item(), (a != b).float().mean().item()


ok = True
for m in [int(x) for x in a.widths.split(",")]:
    launch = seam.build_seam(ws, m, EPS)
    gp = torch.Generator(device=dev).manual_seed(100 + rank)
    partial = torch.randn(m, H, generator=gp, device=dev).to(torch.bfloat16)
    gr = torch.Generator(device=dev).manual_seed(7)
    residual = (torch.randn(m, H, generator=gr, device=dev) * 4).to(torch.bfloat16)
    x_out = torch.empty(m, H, dtype=torch.bfloat16, device=dev)
    res_out = torch.empty_like(x_out)
    args = lambda: (partial.data_ptr(), residual.data_ptr(), norm_w.data_ptr(), x_out.data_ptr(),
                    res_out.data_ptr(), scratch.data_ptr(), peer.local, peer.addresses.data_ptr(),
                    epoch.data_ptr(), rank, 0)
    launch(*args(), stream=torch.cuda.current_stream())
    epoch[0] += 1
    torch.cuda.synchronize()
    parts = [torch.empty_like(partial) for _ in range(ws)]
    dist.all_gather(parts, partial)
    xr, rr = reference(parts, residual)
    ux, fx_ = ulps(x_out, xr)
    ur, fr = ulps(res_out, rr)
    # graph: launch + step bump, replayed; output must stay bit-identical
    first_x = x_out.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch(*args(), stream=torch.cuda.current_stream())
        epoch[0] += 1
    dist.barrier()
    graph.replay(); torch.cuda.synchronize(); dist.barrier()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(a.replays):
        graph.replay()
    e.record(); torch.cuda.synchronize()
    us = s.elapsed_time(e) * 1e3 / a.replays
    stable = torch.equal(first_x, x_out)
    err = [int(epoch[i]) for i in guard.ERROR_WORDS]
    t = torch.tensor([us], device=dev); dist.all_reduce(t, op=dist.ReduceOp.MAX)
    if rank == 0:
        print(f"M={m:2d} x: max {ux:.0f} ulp, {fx_*100:.2f}% differ | residual: max {ur:.0f} ulp, "
              f"{fr*100:.2f}% differ | graph stable={stable} errors={err} | {t.item():.1f} us/launch "
              f"(slowest rank)", flush=True)
    ok &= stable and err == [0, 0, 0] and ur == 0
dist.barrier()
peer.close()
if rank == 0:
    print("PASS" if ok else "FAIL", flush=True)
