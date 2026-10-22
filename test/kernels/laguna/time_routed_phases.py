"""Time Laguna's S3 launch with its UG / DOWN loops skipped at run time (one binary): the phases' cost."""
import sys, torch
sys.path.insert(0, "/workspace/claude-skills/proxy-moe/scripts")
import ep_moe_check as E  # noqa: E402
E.I, E.E_LOCAL = 512, 512
from sgl_amd_mono.common import guard  # noqa: E402
from sgl_amd_mono.laguna.gfx950 import routed  # noqa: E402

widths = [int(v) for v in (sys.argv[1] if len(sys.argv) > 1 else "1,4,16").split(",")]
dev = torch.device("cuda", 0)
w1, w2, w1s, w2s, _, cases = E.build(widths)
scratch = torch.zeros(routed.SCRATCH_BYTES // 4, dtype=torch.int32, device=dev)
epoch = torch.zeros(64, dtype=torch.int32, device=dev)
ticks = guard.deadline_ticks(10.0); lo = ticks & 0xFFFFFFFF
epoch[guard.GUARD_DEADLINE] = lo - (1 << 32) if lo >= 1 << 31 else lo
epoch[guard.GUARD_DEADLINE + 1] = ticks >> 32
for m in widths:
    x, wt, ids = cases[m]
    out = torch.empty(m, routed.HIDDEN, dtype=torch.bfloat16, device=dev)
    launch = routed.build_routed(m)
    t = {}
    for skip in (0, 1, 2, 3):
        args = (x.data_ptr(), ids.data_ptr(), wt.data_ptr(), w1.data_ptr(), w1s.data_ptr(), w2.data_ptr(),
                w2s.data_ptr(), out.data_ptr(), scratch.data_ptr(), epoch.data_ptr(), 0, skip)
        launch(*args, stream=torch.cuda.current_stream()); epoch[0] += 1
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            launch(*args, stream=torch.cuda.current_stream()); epoch[0] += 1
        g.replay(); torch.cuda.synchronize()
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        for _ in range(30):
            g.replay()
        e.record(); torch.cuda.synchronize()
        t[skip] = s.elapsed_time(e) * 1e3 / 30
    nu = int(torch.unique(ids).numel())
    ug, dn = t[0] - t[1], t[0] - t[2]
    print(f"M={m:2d} union={nu}: full {t[0]:.1f} | UG {ug:.1f} us ({nu * 8.65e6 / ug / 1e6:.2f} TB/s) | "
          f"DOWN {dn:.1f} us ({nu * 4.33e6 / dn / 1e6:.2f} TB/s) | rest {t[3]:.1f} us", flush=True)
