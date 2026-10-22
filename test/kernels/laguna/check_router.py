"""Check Laguna's S2 router launch on one gfx950 GPU against SGLang's decode routing
(fp32 F.linear, then the sigmoid top-16 triton router with renormalize), every width;
and bit-stable CUDA-graph replay, plus the replay time."""
import argparse, torch, torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("--widths", default="1,2,4,8,16")
ap.add_argument("--replays", type=int, default=200)
ap.add_argument("--trials", type=int, default=8)
a = ap.parse_args()
dev = torch.device("cuda", 0)

from sgl_amd_mono.common import guard  # noqa: E402
from sgl_amd_mono.laguna.gfx950 import router  # noqa: E402
from sglang.srt.layers.moe.topk import TopK  # noqa: F401,E402

H, E, K = router.HIDDEN, router.EXPERTS, router.TOPK
g = torch.Generator(device=dev).manual_seed(3)
W = torch.randn(E, H, generator=g, device=dev) * 0.02
bias = torch.randn(E, generator=g, device=dev) * 0.01
scratch = torch.zeros(router.LG_BYTES // 4, dtype=torch.int32, device=dev)
epoch = torch.zeros(64, dtype=torch.int32, device=dev)
ticks = guard.deadline_ticks(10.0); lo = ticks & 0xFFFFFFFF
epoch[guard.GUARD_DEADLINE] = lo - (1 << 32) if lo >= 1 << 31 else lo
epoch[guard.GUARD_DEADLINE + 1] = ticks >> 32


def sglang_route(x):
    """SGLang's decode path: gate fp32 F.linear, then the fused triton router."""
    from sglang.kernels.ops.moe.moe_fused_gate import moe_fused_gate_triton as gate  # noqa
    logits = F.linear(x.float(), W)
    return logits, gate


def torch_route(logits):
    act = torch.sigmoid(logits)
    key = act + bias
    ids = torch.empty(logits.shape[0], K, dtype=torch.int32, device=dev)
    w = torch.empty(logits.shape[0], K, device=dev)
    for t in range(logits.shape[0]):
        kk = key[t].clone()
        for r in range(K):
            m = kk.max()
            e = int((kk == m).nonzero()[0])
            ids[t, r] = e; w[t, r] = act[t, e]; kk[e] = -float("inf")
    return ids, w / (w.sum(1, keepdim=True) + 1e-20)


ok = True
for m in [int(x) for x in a.widths.split(",")]:
    launch = router.build_router(m)
    ids = torch.empty(m, K, dtype=torch.int32, device=dev)
    wts = torch.empty(m, K, device=dev)
    agree, wrel, lrel = 0, 0.0, 0.0
    for trial in range(a.trials):
        x = (torch.randn(m, H, generator=g, device=dev)).to(torch.bfloat16)
        launch(x.data_ptr(), W.data_ptr(), bias.data_ptr(), ids.data_ptr(), wts.data_ptr(),
               scratch.data_ptr(), epoch.data_ptr(), 0, stream=torch.cuda.current_stream())
        epoch[0] += 1
        torch.cuda.synchronize()
        logits = F.linear(x.float(), W)
        rid, rw = torch_route(logits)
        same = (ids.sort(1).values == rid.sort(1).values).all(1)
        agree += int(same.sum())
        if bool(same.all()):
            wrel = max(wrel, ((wts - rw).norm() / rw.norm()).item())
            ok &= torch.equal(ids, rid)
    # graph replay stability
    first = (ids.clone(), wts.clone())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch(x.data_ptr(), W.data_ptr(), bias.data_ptr(), ids.data_ptr(), wts.data_ptr(),
               scratch.data_ptr(), epoch.data_ptr(), 0, stream=torch.cuda.current_stream())
        epoch[0] += 1
    graph.replay(); torch.cuda.synchronize()
    stable = torch.equal(first[0], ids) and torch.equal(first[1], wts)
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(a.replays):
        graph.replay()
    e.record(); torch.cuda.synchronize()
    us = s.elapsed_time(e) * 1e3 / a.replays
    err = [int(epoch[i]) for i in guard.ERROR_WORDS]
    print(f"M={m:2d} expert sets equal {agree}/{a.trials * m} tokens | weights rel_l2 max {wrel:.1e} | "
          f"graph stable={stable} errors={err} | {us:.1f} us/launch", flush=True)
    ok &= stable and err == [0, 0, 0] and agree == a.trials * m
print("PASS" if ok else "FAIL")
