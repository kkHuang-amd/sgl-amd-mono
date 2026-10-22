"""Critical path inside Laguna's fused FFN launch (prof build), torchrun TP8 / TP4: LAYERS chained in
one CUDA graph as in serving (no CPU barrier between layers), each layer with its own scratch so its
per-CTA marks survive. Per mark (in program order): median and max over the 256 CTAs of
(mark - layer start = min mark 0), in us; the middle layers of the last replay, averaged."""
import argparse, os, sys, torch, torch.distributed as dist

ap = argparse.ArgumentParser()
ap.add_argument("--widths", default="1,8,16")
ap.add_argument("--layers", type=int, default=6)
ap.add_argument("--replays", type=int, default=20)
a = ap.parse_args()
rank, ws = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
torch.cuda.set_device(rank); dev = torch.device("cuda", rank)
dist.init_process_group("nccl", device_id=dev); gloo = dist.new_group(backend="gloo")
sys.path.insert(0, "/workspace/claude-skills/proxy-moe/scripts")
import ep_moe_check as E  # noqa: E402
from sgl_amd_mono.laguna import INTER_TOTAL, TP  # noqa: E402
E.I, E.E_LOCAL = INTER_TOTAL // TP, 512
from sgl_amd_mono.common import guard  # noqa: E402
from sgl_amd_mono.common.peer import PeerBuffer  # noqa: E402
from sgl_amd_mono.laguna.gfx950 import layer, routed, router_tp  # noqa: E402

H, I, EPS, L = layer.HIDDEN, routed.INTER, 1e-6, a.layers
widths = [int(v) for v in a.widths.split(",")]
w1, w2, w1s, w2s, _, _ = E.build(widths)
g = torch.Generator(device=dev).manual_seed(11 + rank)
sw13 = torch.randint(0, 256, (2 * I, H // 2), dtype=torch.uint8, device=dev, generator=g)
ss13 = torch.randint(118, 124, (2 * I, H // 32), dtype=torch.uint8, device=dev, generator=g)
sw2 = torch.randint(0, 256, (H, I // 2), dtype=torch.uint8, device=dev, generator=g)
ss2 = torch.randint(118, 124, (H, I // 32), dtype=torch.uint8, device=dev, generator=g)
gs = torch.Generator(device=dev).manual_seed(3)
EL = router_tp.local_experts(ws)
rw = (torch.randn(router_tp.EXPERTS, H, generator=gs, device=dev) * 0.02)[rank * EL:(rank + 1) * EL].contiguous()
rb = torch.randn(router_tp.EXPERTS, generator=gs, device=dev) * 0.01
norm1 = (torch.randn(H, generator=gs, device=dev) * 0.1 + 1).to(torch.bfloat16)
norm2 = (torch.randn(H, generator=gs, device=dev) * 0.1 + 1).to(torch.bfloat16)
ep = torch.zeros(64, dtype=torch.int32, device=dev)
t = guard.deadline_ticks(30.0); lo = t & 0xFFFFFFFF
ep[guard.GUARD_DEADLINE] = lo - (1 << 32) if lo >= 1 << 31 else lo; ep[guard.GUARD_DEADLINE + 1] = t >> 32
pf = PeerBuffer(layer.peer_bytes(ws), gloo, rank, ws, dev)
scr = [torch.zeros(layer.SCRATCH_BYTES // 4, dtype=torch.int32, device=dev) for _ in range(L)]
P = lambda t: t.data_ptr()
cs = torch.cuda.current_stream
# program order of the marks (layer.py 0-6, routed.py 7-12)
ORDER = [(0, "start"), (1, "S1 seam"), (2, "xf sync"), (3, "S2 logits+push, S4 Q/GU"), (4, "topk, S4 GR/RQ, rt sync"),
         (7, "S3 U"), (8, "S3 Q"), (9, "S3 qf wait"), (10, "S3 UG"), (11, "S3 rf wait"), (12, "S3 DOWN"),
         (5, "S3 out, S4 DOWN, barrier"), (6, "S5 seam")]

for m in widths:
    fused = layer.build_layer(ws, m, EPS, prof=True)
    gp = torch.Generator(device=dev).manual_seed(100 + rank)
    parts = [torch.randn(m, H, generator=gp, device=dev).to(torch.bfloat16) for _ in range(L)]
    res0 = (torch.randn(m, H, generator=torch.Generator(device=dev).manual_seed(7), device=dev) * 4).to(torch.bfloat16)
    xo = [torch.empty(m, H, dtype=torch.bfloat16, device=dev) for _ in range(L)]
    ro = [torch.empty(m, H, dtype=torch.bfloat16, device=dev) for _ in range(L)]

    def run():
        res = res0
        for l in range(L):
            fused(P(parts[l]), P(res), P(norm1), P(rw), P(rb), P(w1), P(w1s), P(w2), P(w2s), P(sw13), P(ss13), P(sw2),
                  P(ss2), P(norm2), P(xo[l]), P(ro[l]), P(scr[l]), pf.local, P(pf.addresses), P(ep), rank, l, stream=cs())
            res = ro[l]
        ep[0] += 1

    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        run()
    dist.barrier(); gr.replay(); torch.cuda.synchronize(); dist.barrier()
    for _ in range(a.replays):
        gr.replay()
    torch.cuda.synchronize()
    marks = [s.view(torch.uint8)[layer.PROF:layer.PROF + 256 * layer.MARKS * 8].view(torch.int64).view(256, layer.MARKS).cpu()
             for s in scr]
    mid = range(1, L - 1)
    med = torch.zeros(len(ORDER), dtype=torch.float64); mx = torch.zeros(len(ORDER), dtype=torch.float64)
    gap = 0.0
    for l in mid:
        t0 = marks[l][:, 0].min()
        rel = (marks[l] - t0).double() / 100.0  # 100 MHz -> us
        for j, (i, _) in enumerate(ORDER):
            med[j] += rel[:, i].median() / len(mid); mx[j] += rel[:, i].max() / len(mid)
        gap += float(t0 - marks[l - 1][:, 6].max()) / 100.0 / len(mid)
    tot = torch.tensor([mx[-1].item()], device=dev); allt = [torch.zeros_like(tot) for _ in range(ws)]
    dist.all_gather(allt, tot)
    slow = max(range(ws), key=lambda r: allt[r].item())
    rows = torch.stack([med, mx]).to(dev); allr = [torch.zeros_like(rows) for _ in range(ws)]
    dist.all_gather(allr, rows)
    if rank == 0:
        print(f"M={m:2d} TP{ws}: layer (rank 0 last CTA) {mx[-1]:.1f} us, previous layer end -> start {gap:.1f} us; "
              f"slowest rank {slow} {allt[slow].item():.1f} us", flush=True)
        for r in sorted({0, slow}):
            md, mm = allr[r][0].tolist(), allr[r][1].tolist()
            line, pm, pM = [], 0.0, 0.0
            for j, (_, name) in enumerate(ORDER):
                if j:
                    line.append(f"{name} +{md[j] - pm:.1f}/+{mm[j] - pM:.1f}")
                pm, pM = md[j], mm[j]
            print(f"  rank {r} (median CTA / last CTA, us): " + " | ".join(line), flush=True)
dist.barrier(); pf.close()
