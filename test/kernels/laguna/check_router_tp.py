"""Check Laguna's TP-sharded router (S2) on gfx950, torchrun TP8: every rank's top-16 vs SGLang's
routing on the full fp32 router, ranks agreeing, graph replay stability and time."""
import argparse, os, torch, torch.distributed as dist, torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("--widths", default="1,2,4,8,16")
ap.add_argument("--replays", type=int, default=200)
ap.add_argument("--trials", type=int, default=4)
a = ap.parse_args()
rank, ws = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
torch.cuda.set_device(rank); dev = torch.device("cuda", rank)
dist.init_process_group("nccl", device_id=dev); gloo = dist.new_group(backend="gloo")
from sgl_amd_mono.common import guard  # noqa: E402
from sgl_amd_mono.common.peer import PeerBuffer  # noqa: E402
from sgl_amd_mono.laguna.gfx950 import router_tp as R  # noqa: E402

H, E, K = R.HIDDEN, R.EXPERTS, R.TOPK
EL = R.local_experts(ws)
g = torch.Generator(device=dev).manual_seed(3)
W = torch.randn(E, H, generator=g, device=dev) * 0.02  # same on every rank (seeded)
bias = torch.randn(E, generator=g, device=dev) * 0.01
Wl = W[rank * EL:(rank + 1) * EL].contiguous()
peer = PeerBuffer(R.peer_bytes(), gloo, rank, ws, dev)
scratch = torch.zeros(R.PQ_BYTES // 4, dtype=torch.int32, device=dev)
epoch = torch.zeros(64, dtype=torch.int32, device=dev)
t = guard.deadline_ticks(30.0); lo = t & 0xFFFFFFFF
epoch[guard.GUARD_DEADLINE] = lo - (1 << 32) if lo >= 1 << 31 else lo; epoch[guard.GUARD_DEADLINE + 1] = t >> 32


def torch_route(logits):
    act = torch.sigmoid(logits); key = act + bias
    ids = torch.empty(logits.shape[0], K, dtype=torch.int32, device=dev); w = torch.empty(logits.shape[0], K, device=dev)
    for tt in range(logits.shape[0]):
        kk = key[tt].clone()
        for r in range(K):
            m = kk.max(); e = int((kk == m).nonzero()[0]); ids[tt, r] = e; w[tt, r] = act[tt, e]; kk[e] = -float("inf")
    return ids, w / (w.sum(1, keepdim=True) + 1e-20)


ok = True
for m in [int(v) for v in a.widths.split(",")]:
    launch = R.build_router_tp(ws, m)
    ids = torch.empty(m, K, dtype=torch.int32, device=dev); wts = torch.empty(m, K, device=dev)
    agree = 0; wrel = 0.0
    for trial in range(a.trials):
        gx = torch.Generator(device=dev).manual_seed(100 + trial * 17 + m)
        x = torch.randn(m, H, generator=gx, device=dev).to(torch.bfloat16)
        args = lambda: (x.data_ptr(), Wl.data_ptr(), bias.data_ptr(), ids.data_ptr(), wts.data_ptr(), scratch.data_ptr(),
                        peer.local, peer.addresses.data_ptr(), epoch.data_ptr(), rank, 0)
        launch(*args(), stream=torch.cuda.current_stream()); epoch[0] += 1; torch.cuda.synchronize()
        rid, rw = torch_route(F.linear(x.float(), W))
        same = (ids == rid).all(1); agree += int(same.sum())
        if bool(same.all()):
            wrel = max(wrel, ((wts - rw).norm() / rw.norm()).item())
        allids = [torch.empty_like(ids) for _ in range(ws)]; dist.all_gather(allids, ids)
        ok &= all(torch.equal(allids[0], i) for i in allids)
    first = (ids.clone(), wts.clone())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch(*args(), stream=torch.cuda.current_stream()); epoch[0] += 1
    dist.barrier(); graph.replay(); torch.cuda.synchronize(); dist.barrier()
    stable = torch.equal(first[0], ids) and torch.equal(first[1], wts)
    s, e = torch.cuda.Event(True), torch.cuda.Event(True); s.record()
    for _ in range(a.replays):
        graph.replay()
    e.record(); torch.cuda.synchronize()
    us = torch.tensor([s.elapsed_time(e) * 1e3 / a.replays], device=dev); dist.all_reduce(us, op=dist.ReduceOp.MAX)
    err = [int(epoch[i]) for i in guard.ERROR_WORDS]
    if rank == 0:
        print(f"M={m:2d} expert ids identical (order too) {agree}/{a.trials * m} | weights rel_l2 max {wrel:.1e} | "
              f"ranks agree | stable={stable} errors={err} | {us.item():.1f} us/launch (slowest rank)", flush=True)
    ok &= stable and err == [0, 0, 0] and agree == a.trials * m
dist.barrier(); peer.close()
if rank == 0:
    print("PASS" if ok else "FAIL")
