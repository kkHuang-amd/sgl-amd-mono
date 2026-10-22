"""Stage times inside Laguna's fused FFN launch (prof build), torchrun TP8: per stage boundary the
median over CTAs of (mark - launch start = min mark 0), rank 0 and the slowest rank, in us."""
import os, sys, torch, torch.distributed as dist
sys.argv += [] if "--widths" in sys.argv else ["--widths", "1,2,4,8,16"]
w = [int(v) for v in sys.argv[sys.argv.index("--widths") + 1].split(",")]
rank, ws = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
torch.cuda.set_device(rank); dev = torch.device("cuda", rank)
dist.init_process_group("nccl", device_id=dev); gloo = dist.new_group(backend="gloo")
sys.path.insert(0, "/workspace/claude-skills/proxy-moe/scripts")
import ep_moe_check as E  # noqa: E402
E.I, E.E_LOCAL = 512, 512
from sgl_amd_mono.common import guard  # noqa: E402
from sgl_amd_mono.common.peer import PeerBuffer  # noqa: E402
from sgl_amd_mono.laguna.gfx950 import layer, routed, router_tp  # noqa: E402
H, I = layer.HIDDEN, routed.INTER
w1, w2, w1s, w2s, _, _ = E.build(w)
g = torch.Generator(device=dev).manual_seed(11 + rank)
sw13 = torch.randint(0, 256, (2 * I, H // 2), dtype=torch.uint8, device=dev, generator=g)
ss13 = torch.randint(118, 124, (2 * I, H // 32), dtype=torch.uint8, device=dev, generator=g)
sw2 = torch.randint(0, 256, (H, I // 2), dtype=torch.uint8, device=dev, generator=g)
ss2 = torch.randint(118, 124, (H, I // 32), dtype=torch.uint8, device=dev, generator=g)
gs = torch.Generator(device=dev).manual_seed(3); EL = router_tp.local_experts(ws)
rw = (torch.randn(512, H, generator=gs, device=dev) * 0.02)[rank * EL:(rank + 1) * EL].contiguous()
rb = torch.randn(512, generator=gs, device=dev) * 0.01
n1 = torch.ones(H, dtype=torch.bfloat16, device=dev)
ep = torch.zeros(64, dtype=torch.int32, device=dev)
t = guard.deadline_ticks(30.0); lo = t & 0xFFFFFFFF
ep[guard.GUARD_DEADLINE] = lo - (1 << 32) if lo >= 1 << 31 else lo; ep[guard.GUARD_DEADLINE + 1] = t >> 32
pf = PeerBuffer(layer.peer_bytes(ws), gloo, rank, ws, dev)
scr = torch.zeros(layer.SCRATCH_BYTES // 4, dtype=torch.int32, device=dev)
P = lambda t: t.data_ptr()
names = ["S1", "xf", "S2part+S4QGU", "topk+S4GRRQ+rt", "S3+S4down", "S5"]
for m in w:
    k = layer.build_layer(ws, m, 1e-6, prof=True)
    part = torch.randn(m, H, device=dev).to(torch.bfloat16); res = torch.randn(m, H, device=dev).to(torch.bfloat16)
    xo, ro = torch.empty_like(part), torch.empty_like(part)
    rows = []
    for it in range(12):
        dist.barrier()
        k(P(part), P(res), P(n1), P(rw), P(rb), P(w1), P(w1s), P(w2), P(w2s), P(sw13), P(ss13), P(sw2), P(ss2), P(n1),
          P(xo), P(ro), P(scr), pf.local, P(pf.addresses), P(ep), rank, 0, stream=torch.cuda.current_stream())
        ep[0] += 1; torch.cuda.synchronize()
        pr = scr.view(torch.uint8)[layer.PROF:layer.PROF + 256 * layer.MARKS * 8].view(torch.int64).view(256, layer.MARKS)
        t0 = pr[:, 0].min()
        rel = (pr[:, :7] - t0).double() / 100.0  # 100 MHz -> us
        if it >= 2:
            rows.append(rel.median(0).values)
    med = torch.stack(rows).median(0).values
    seg = torch.diff(med).tolist()
    tot = torch.tensor([med[6].item()], device=dev); dist.all_reduce(tot, op=dist.ReduceOp.MAX)
    if rank == 0:
        print(f"M={m:2d} total {med[6]:.1f} us (slowest rank {tot.item():.1f}) | " +
              " ".join(f"{n} {v:.1f}" for n, v in zip(names, seg)), flush=True)
dist.barrier(); pf.close()
