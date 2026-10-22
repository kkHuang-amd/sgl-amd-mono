"""Check Laguna's fused FFN launch (S1 + S2 + S3/S4 + S5, ``layer.py``), torchrun TP8: LAYERS chained
layers vs the same layers run as the standalone launches (seam p0, TP router, routed, shared, bf16 add,
seam p1),
bit for bit; graph replay stability; time per layer, fused vs the four launches."""
import argparse, os, sys, torch, torch.distributed as dist

ap = argparse.ArgumentParser()
ap.add_argument("--widths", default="1,2,4,8,16")
ap.add_argument("--replays", type=int, default=30)
ap.add_argument("--layers", type=int, default=3)
ap.add_argument("--debug", action="store_true")
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
from sgl_amd_mono.laguna.gfx950 import layer, routed, router_tp, seam, shared  # noqa: E402

H, I, EPS, L = layer.HIDDEN, routed.INTER, 1e-6, a.layers
widths = [int(v) for v in a.widths.split(",")]
w1, w2, w1s, w2s, _, _ = E.build(widths)
g = torch.Generator(device=dev).manual_seed(11 + rank)  # the shared expert differs per rank: partials differ
sw13 = torch.randint(0, 256, (2 * I, H // 2), dtype=torch.uint8, device=dev, generator=g)
ss13 = torch.randint(118, 124, (2 * I, H // 32), dtype=torch.uint8, device=dev, generator=g)
sw2 = torch.randint(0, 256, (H, I // 2), dtype=torch.uint8, device=dev, generator=g)
ss2 = torch.randint(118, 124, (H, I // 32), dtype=torch.uint8, device=dev, generator=g)
gs = torch.Generator(device=dev).manual_seed(3)  # the same on every rank
EL = router_tp.local_experts(ws)
rw = (torch.randn(router_tp.EXPERTS, H, generator=gs, device=dev) * 0.02)[rank * EL:(rank + 1) * EL].contiguous()
rb = torch.randn(router_tp.EXPERTS, generator=gs, device=dev) * 0.01
norm1 = (torch.randn(H, generator=gs, device=dev) * 0.1 + 1).to(torch.bfloat16)
norm2 = (torch.randn(H, generator=gs, device=dev) * 0.1 + 1).to(torch.bfloat16)


def new_epoch():
    ep = torch.zeros(64, dtype=torch.int32, device=dev)
    t = guard.deadline_ticks(30.0); lo = t & 0xFFFFFFFF
    ep[guard.GUARD_DEADLINE] = lo - (1 << 32) if lo >= 1 << 31 else lo; ep[guard.GUARD_DEADLINE + 1] = t >> 32
    return ep


ep_f, ep_r = new_epoch(), new_epoch()
pf = PeerBuffer(layer.peer_bytes(ws), gloo, rank, ws, dev)
ps = PeerBuffer(seam.peer_bytes(ws), gloo, rank, ws, dev)
pr = PeerBuffer(router_tp.peer_bytes(), gloo, rank, ws, dev)
scr_f = torch.zeros(layer.SCRATCH_BYTES // 4, dtype=torch.int32, device=dev)
scr_s = torch.zeros(seam.SS_BYTES // 4, dtype=torch.int32, device=dev)
scr_r = torch.zeros(router_tp.PQ_BYTES // 4, dtype=torch.int32, device=dev)
scr_3 = torch.zeros(routed.SCRATCH_BYTES // 4, dtype=torch.int32, device=dev)
scr_4 = torch.zeros(shared.SCRATCH_BYTES // 4, dtype=torch.int32, device=dev)
P = lambda t: t.data_ptr()
cs = torch.cuda.current_stream

ok = True
for m in widths:
    fused = layer.build_layer(ws, m, EPS)
    s1, s5 = seam.build_seam(ws, m, EPS, 0), seam.build_seam(ws, m, EPS, 1)
    s2, s3, s4 = router_tp.build_router_tp(ws, m), routed.build_routed(m), shared.build_shared(m)
    gp = torch.Generator(device=dev).manual_seed(100 + rank)
    parts = [torch.randn(m, H, generator=gp, device=dev).to(torch.bfloat16) for _ in range(L)]
    res0 = (torch.randn(m, H, generator=torch.Generator(device=dev).manual_seed(7), device=dev) * 4).to(torch.bfloat16)
    bf = lambda: torch.empty(m, H, dtype=torch.bfloat16, device=dev)
    fx_, fr_ = [bf() for _ in range(L)], [bf() for _ in range(L)]
    rx_, rr_, rh, rm, rmoe = [bf() for _ in range(L)], [bf() for _ in range(L)], bf(), bf(), bf()
    o3, o4 = bf(), bf()
    ids = torch.empty(m, 16, dtype=torch.int32, device=dev); wts = torch.empty(m, 16, device=dev)

    def run_fused():
        res = res0
        for l in range(L):
            fused(P(parts[l]), P(res), P(norm1), P(rw), P(rb), P(w1), P(w1s), P(w2), P(w2s), P(sw13), P(ss13), P(sw2),
                  P(ss2), P(norm2), P(fx_[l]), P(fr_[l]), P(scr_f), pf.local, P(pf.addresses), P(ep_f), rank, l,
                  stream=cs())
            res = fr_[l]
        ep_f[0] += 1

    def run_ref():
        res = res0
        for l in range(L):
            s1(P(parts[l]), P(res), P(norm1), P(rh), P(rm), P(scr_s), ps.local, P(ps.addresses), P(ep_r), rank,
               seam.tag_slot(l, 0), stream=cs())
            s2(P(rh), P(rw), P(rb), P(ids), P(wts), P(scr_r), pr.local, P(pr.addresses), P(ep_r), rank, l, stream=cs())
            s3(P(rh), P(ids), P(wts), P(w1), P(w1s), P(w2), P(w2s), P(o3), P(scr_3), P(ep_r), l, 0, stream=cs())
            s4(P(rh), P(sw13), P(ss13), P(sw2), P(ss2), P(o4), P(scr_4), P(ep_r), l, stream=cs())
            torch.add(o3, o4, out=rmoe)  # bf16 + bf16 -> bf16, as SGLang adds routed and shared
            s5(P(rmoe), P(rm), P(norm2), P(rx_[l]), P(rr_[l]), P(scr_s), ps.local, P(ps.addresses), P(ep_r), rank,
               seam.tag_slot(l, 1), stream=cs())
            res = rr_[l]
        ep_r[0] += 1

    run_ref(); run_fused(); torch.cuda.synchronize()
    if a.debug:  # the last layer's hand-offs, fused scratch vs the standalone chain's buffers
        sb = scr_f.view(torch.uint8)
        view = lambda off, n, dt: sb[off:off + n * torch.tensor([], dtype=dt).element_size()].view(dt)
        fh = view(layer.H, m * H, torch.bfloat16).view(m, H); fm = view(layer.RES_MID, m * H, torch.bfloat16).view(m, H)
        fmoe = (view(layer.MOE, m * H, torch.bfloat16).view(m, H).float() + view(layer.SH, m * H, torch.bfloat16).view(m, H).float()).to(torch.bfloat16)
        fid = view(layer.IDS, m * 16, torch.int32).view(m, 16); fw = view(layer.WTS, m * 16, torch.float32).view(m, 16)
        d = lambda p, q: f"{torch.equal(p, q)}/{(p.float() - q.float()).abs().max().item():.3g}"
        print(f"rank{rank} M={m} h {d(fh, rh)} res_mid {d(fm, rm)} ids {d(fid, ids)} wts {d(fw, wts)} moe {d(fmoe, rmoe)} "
              f"x {d(fx_[-1], rx_[-1])} res {d(fr_[-1], rr_[-1])}", flush=True)
    same = all(torch.equal(fx_[l], rx_[l]) and torch.equal(fr_[l], rr_[l]) for l in range(L))
    nz = all(bool(fx_[l].float().abs().sum() > 0) for l in range(L))
    first = [t.clone() for t in fx_ + fr_]
    us = {}
    for name, fn in (("fused", run_fused), ("ref", run_ref)):
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            fn()
        dist.barrier(); gr.replay(); torch.cuda.synchronize(); dist.barrier()
        s, e = torch.cuda.Event(True), torch.cuda.Event(True); s.record()
        for _ in range(a.replays):
            gr.replay()
        e.record(); torch.cuda.synchronize()
        t = torch.tensor([s.elapsed_time(e) * 1e3 / a.replays / L], device=dev)
        dist.all_reduce(t, op=dist.ReduceOp.MAX); us[name] = t.item()
    stable = all(torch.equal(p, q) for p, q in zip(first, fx_ + fr_))
    err = [int(ep_f[i]) for i in guard.ERROR_WORDS] + [int(ep_r[i]) for i in guard.ERROR_WORDS]
    if rank == 0:
        print(f"M={m:2d} {L} layers: fused == standalone chain (x, residual bit-exact) {same} nonzero={nz} | "
              f"stable={stable} errors={err} | {us['fused']:.1f} us/layer fused vs {us['ref']:.1f} us/layer "
              f"standalone (slowest rank)", flush=True)
    ok &= same and nz and stable and err == [0] * 6
dist.barrier(); pf.close(); ps.close(); pr.close()
if rank == 0:
    print("PASS" if ok else "FAIL")
