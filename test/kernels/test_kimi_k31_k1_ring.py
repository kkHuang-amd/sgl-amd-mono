"""Single-GPU checks of K1's ReplaySSM ring build (``KdaPreBuild.ring``), which
Kimi-K3.1 serves with (``--enable-linear-replayssm-spec``).

* core_out of the ring build equals the snapshot build's, bit for bit, from the
  same initial state: the ring changes what is stored, not the recurrence.
* the committed state is read and left untouched.
* the ring rows are what SGLang's commit replays (as the HIP verify kernel
  writes them): per token, the conv + SiLU k (before its l2norm) and v in bf16,
  the lower-bound gate in log space (fp32, per K) and sigmoid(b) (fp32).

Run directly with ``python test/kernels/test_kimi_k31_k1_ring.py``.
"""

import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(__file__))
from test_kimi_k3_k1 import BF, EPS, NVB, QLEN, K1Case, _cos, _kda_reference, _skip_reasons  # noqa: E402

RING_LEN = 16  # SGLang's --linear-replayssm-cache-len default


def _run(case, ring):
    kda = case.kda
    key = kda.KdaPreBuild(
        tokens=case.S,
        qlen=case.L,
        nblocks=NVB,
        delta=False,
        state_len=case.state_len,
        eps=EPS,
        out_eps=EPS,
        onorm_eps=EPS,
        debug=True,
        ring=ring is not None,
    )
    scratch = torch.zeros(kda.scratch_bytes(key), dtype=torch.uint8, device=case.device)
    epoch = torch.ones(1, dtype=torch.int32, device=case.device)
    kda.kda_pre(
        key,
        prefix=case.prefix,
        delta=None,
        blocks=case.blocks,
        ares_nw=case.ares_nw,
        ares_qk=case.ares_qk,
        in_nw=case.in_nw,
        w_in=case.w_in,
        w_fb=case.w_fb,
        conv_w=case.conv_w,
        conv_state=case.conv_state,
        a_log=case.a_log,
        dt_bias=case.dt_bias,
        on_w=case.on_w,
        rstate=case.rstate,
        st_idx=case.st_idx,
        num_acc=case.num_acc_t,
        core_out=case.core_out,
        scratch=scratch,
        layer=1,
        epoch=epoch,
        x_dbg=case.x_dbg,
        proj_dbg=case.proj_dbg,
        ring=ring,
    )
    torch.cuda.synchronize()


def _ring_reference(case, slot):
    """The ring rows from the kernel's own in-projection (see _kda_reference)."""
    kda = case.kda
    L, hd, nh, pw = case.L, kda.HD, kda.NH, kda.PROJ
    win = kda.CONV_W - 1
    proj = case.proj_dbg.float()
    xv = proj[:, : 3 * pw].t()
    window = case.conv_state_in[slot][:, :win].float()
    seq = torch.cat([window, xv], dim=1)
    cw = case.conv_w.float()
    conv = torch.stack(
        [sum(cw[:, q] * seq[:, j + q] for q in range(kda.CONV_W)) for j in range(L)], dim=1
    )
    conv = (conv * torch.sigmoid(conv)).t().to(BF)  # [L, 3 PROJ]
    rk = conv[:, pw : 2 * pw].reshape(L, nh, hd).transpose(0, 1)
    rv = conv[:, 2 * pw :].reshape(L, nh, hd).transpose(0, 1)
    fa = proj[:, kda.FA_0 : kda.FA_0 + hd]
    rg = torch.empty(nh, L, hd, device=proj.device)
    rb = torch.empty(nh, L, device=proj.device)
    for h in range(nh):
        g1 = (fa.to(BF).float() @ case.w_fb[h * hd : (h + 1) * hd].float().t()).to(BF).float()
        g = g1 + case.dt_bias[h * hd : (h + 1) * hd].float()
        rg[h] = -5.0 * torch.sigmoid(torch.exp(case.a_log[h].float()) * g)
        rb[h] = torch.sigmoid(proj[:, kda.BETA_0 + h])
    return rk, rv, rg, rb


class TestKimiK31K1Ring(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        reason = _skip_reasons()
        if reason:
            raise unittest.SkipTest(reason)
        cls.device = torch.device("cuda", 0)
        torch.cuda.set_device(cls.device)

    def _cases(self, slot):
        kda = K1Case(self.device, num_acc=1).kda
        # snapshot rows 0 .. L - 1, row 0 holding the initial state and conv window
        snap = K1Case(self.device, num_acc=1)
        _run(snap, None)

        # the same initial state and window, at a slot other than 0
        ring = K1Case(self.device, num_acc=1)
        ring.rstate[slot].copy_(ring.rstate[0])
        ring.conv_state[slot].copy_(ring.conv_state[0])
        ring.rstate_in, ring.conv_state_in = ring.rstate.clone(), ring.conv_state.clone()
        ring.st_idx = torch.full((1, QLEN), slot, dtype=torch.int32, device=self.device)
        nslots = ring.rstate.shape[0]
        mk = lambda *tail, dt: torch.full((nslots, kda.NH, RING_LEN, *tail), float("nan"), dtype=dt, device=self.device)  # noqa: E731
        rows = (mk(kda.HD, dt=BF), mk(kda.HD, dt=BF), mk(kda.HD, dt=torch.float32), mk(dt=torch.float32))
        _run(ring, rows)
        return snap, ring, rows

    def test_ring_build(self):
        slot = 3
        snap, ring, (rk, rv, rg, rb) = self._cases(slot)
        self.assertTrue(torch.equal(ring.core_out, snap.core_out), "core_out differs from the snapshot build")
        self.assertGreater(_cos(ring.core_out, _kda_reference(ring)), 0.999)
        self.assertTrue(torch.equal(ring.rstate, ring.rstate_in), "the ring build wrote the committed state")
        wk, wv, wg, wb = _ring_reference(ring, slot)
        L = QLEN
        for name, got, want in (
            ("k", rk[slot, :, :L], wk),
            ("v", rv[slot, :, :L], wv),
            ("g", rg[slot, :, :L], wg),
            ("beta", rb[slot, :, :L], wb),
        ):
            cos = _cos(got, want)
            err = (got.float() - want.float()).abs().max().item()
            print(f"ring {name}: cos {cos:.7f}, max abs err {err:.3g}")
            self.assertGreater(cos, 0.9999, f"ring {name} differs")
        for name, t in (("k", rk), ("v", rv), ("g", rg), ("beta", rb)):
            others = torch.cat([t[:slot], t[slot + 1 :]]).float()
            self.assertTrue(bool(torch.isnan(others).all()), f"ring {name}: another slot was written")
            self.assertTrue(bool(torch.isnan(t[slot, :, L:].float()).all()), f"ring {name}: rows past L written")


if __name__ == "__main__":
    unittest.main(verbosity=2)
