"""Bit-check the in-kernel MXFP4 quantizers (laguna/gfx950/mxq.py) against aiter's
triton dynamic_mxfp4_quant and hip per_1x32 quant, on one gfx950 GPU; and the decode."""
import torch
import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import range_constexpr
from flydsl.expr.typing import Int64, T

from sgl_amd_mono.common.ops import bf2_f32, rsrc
from sgl_amd_mono.common.plan import kernel_sources
from sgl_amd_mono.laguna.gfx950 import mxq

M, H = 16, 16384
G = H // 32


def build(rule):
    keyed = (rule, kernel_sources(mxq.__name__))

    @flyc.kernel(name=f"laguna_mxq_test_{rule}", known_block_size=[512, 1, 1])
    def k(x: Int64, q: Int64, s: Int64, dq: Int64):
        _ = keyed
        t = fx.block_idx.x
        g = fx.thread_idx.x
        vals = []
        for c in range_constexpr(4):
            w = fx.Vector(bo.buffer_load(rsrc(x), (t * H + g * 32) // 2 + 4 * c, vec_width=4, dtype=T.i32))
            for i in range_constexpr(4):
                a, b = bf2_f32(fx.Int32(w[i]))
                vals += [a, b]
        words, byte = mxq.quant_group(vals, rule)
        bo.buffer_store(fx.Vector.from_elements(words, fx.Int32), rsrc(q), (t * H + g * 32) // 8)
        bo.buffer_store(byte, rsrc(s), t * G + g)
        # decode back: x_hat = fp4 * 2^(byte - 127)
        out = []
        for i in range_constexpr(4):
            out += mxq.decode_word(words[i], mxq.pow2(byte))
        for i in range_constexpr(8):
            bo.buffer_store(fx.Vector.from_elements(out[4 * i:4 * i + 4], fx.Float32), rsrc(dq),
                            t * H + g * 32 + 4 * i)

    @flyc.jit
    def launch(x: Int64, q: Int64, s: Int64, dq: Int64, stream: fx.Stream = fx.Stream(None)):
        _ = keyed
        k(x, q, s, dq).launch(grid=(M,), block=(512,), stream=stream)
    return launch


def deq(qb, sb):
    from aiter.utility import fp4_utils
    v = fp4_utils.mxfp4_to_f32(qb).view(qb.shape[0], -1)
    return v * fp4_utils.e8m0_to_f32(sb.to(torch.uint8)).repeat_interleave(32, dim=1)


g = torch.Generator(device="cuda").manual_seed(5)
x = (torch.randn(M, H, generator=g, device="cuda") * torch.logspace(-3, 3, M, device="cuda")[:, None]).to(torch.bfloat16)
x[0, :64] = 0  # an all-zero group
ok = True
for rule in ("triton", "hip"):
    q = torch.zeros(M, H // 2, dtype=torch.uint8, device="cuda")
    s = torch.zeros(M, G, dtype=torch.int32, device="cuda")
    dq = torch.zeros(M, H, dtype=torch.float32, device="cuda")
    build(rule)(x.data_ptr(), q.data_ptr(), s.data_ptr(), dq.data_ptr(), stream=torch.cuda.current_stream())
    torch.cuda.synchronize()
    if rule == "triton":
        from aiter.ops.triton.quant import dynamic_mxfp4_quant
        rq, rs = dynamic_mxfp4_quant(x)
    else:
        from aiter import QuantType, dtypes
        from aiter.ops.quant import get_hip_quant
        rq, rs = get_hip_quant(QuantType.per_1x32)(x, quant_dtype=dtypes.fp4x2, shuffle=False)
    rq = rq.view(torch.uint8)[:M]; rs = rs.view(torch.uint8)[:M, :G].to(torch.int32)
    qeq = (q == rq).float().mean().item(); seq = (s == rs).float().mean().item()
    dqerr = (dq - deq(q, s)).abs().max().item()
    print(f"{rule}: fp4 bytes equal {qeq*100:.4f}%  scales equal {seq*100:.4f}%  decode max|err| {dqerr:.3g}")
    ok &= qeq == 1.0 and seq == 1.0 and dqerr == 0.0
print("PASS" if ok else "FAIL")
# where do the hip scales differ?
from aiter import QuantType, dtypes
from aiter.ops.quant import get_hip_quant
q = torch.zeros(M, H // 2, dtype=torch.uint8, device="cuda"); s = torch.zeros(M, G, dtype=torch.int32, device="cuda")
dq = torch.zeros(M, H, dtype=torch.float32, device="cuda")
build("hip")(x.data_ptr(), q.data_ptr(), s.data_ptr(), dq.data_ptr(), stream=torch.cuda.current_stream()); torch.cuda.synchronize()
rq, rs = get_hip_quant(QuantType.per_1x32)(x, quant_dtype=dtypes.fp4x2, shuffle=False)
rs = rs.view(torch.uint8)[:M, :G].to(torch.int32)
bad = (s != rs).nonzero()
for t, gi in bad.tolist()[:6]:
    grp = x[t, gi * 32:(gi + 1) * 32].float()
    print(f"diff at token {t} group {gi}: mine {int(s[t, gi])} aiter {int(rs[t, gi])} amax {grp.abs().max().item():.6g}")
