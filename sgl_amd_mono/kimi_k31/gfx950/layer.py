# SPDX-License-Identifier: Apache-2.0
"""One wide-row KDA layer launch: T1 .. T6 (``pre_attn`` -> KDA -> ``chain``).

    MoE tail (L-1) -> attention AttnRes seam -> in_proj -> KDA (a task a (request,
    head)) -> o_proj -> RS + residual -> MLP AttnRes seam -> MoE front (L)

The stages are ``pre_attn``'s and ``chain``'s, joined in one plan: the two halves
keep their region names in two contexts on one peer buffer (the post half's
regions after the pre half's), one tag. New hand-offs inside the launch, all at
device scope because the reader may sit on another XCD: KDA's core_out (each
(request, head) marked done in scratch; an o_proj tile waits for the heads of
its rows and K range), the prefix after the MoE tail (the RS stage's residual
input), and the bank row the attention seam writes (the MLP seam reads it).
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu
from flydsl.expr.typing import Int32, Int64

from sgl_amd_mono.common import guard
from sgl_amd_mono.common.ops import CM_DEV, load_ptr64
from sgl_amd_mono.common.plan import BLOCKS, THREADS, WAVES, first_task
from sgl_amd_mono.common.sync import sreg
from sgl_amd_mono.kimi_k3.gfx950.attention import kda as k1
from sgl_amd_mono.kimi_k31.gfx950 import chain as post
from sgl_amd_mono.kimi_k31.gfx950 import pre_attn as pre
from sgl_amd_mono.kimi_k31.gfx950.stages import coll
from sgl_amd_mono.kimi_k31.gfx950.stages import kda_wide as kdaw
from sgl_amd_mono.kimi_k31.gfx950.stages import moe_front as mf
from sgl_amd_mono.kimi_k31.gfx950.stages.attnres import red_words
from sgl_amd_mono.kimi_k31.gfx950.stages.gemm_coll import _arrive_all, _push_band_flag, dev_emit
from sgl_amd_mono.kimi_k31.gfx950.stages.tile import atomic_lin, bf16_emit, f32_emit
from sgl_amd_mono.kimi_k31.gfx950.stages.tile_mma import DEPTH, RW, mma_tile
from sgl_amd_mono.kimi_k31.plan import ALIGN, Plan

TP, H, SL, LAT, LSL = coll.TP, pre.H, pre.SL, pre.LAT, pre.LSL
KO = post.KO


class LayerCfg:
    """GEMM configs of the five tile GEMMs: latent up, in_proj, o_proj, MoE front, shared down."""

    def __init__(self, up, inp, oproj, front, sdown):
        self.up, self.inp, self.oproj, self.front, self.sdown = up, inp, oproj, front, sdown

    def key(self):
        return tuple(g.key() for g in (self.up, self.inp, self.oproj, self.front, self.sdown))


def plan(m, cfg, L, nvb_a, nvb_m, nf, capture=False):
    """(combined Plan, pre plan, post plan, post region offset shift, post scratch shift)."""
    pp = pre.plan(m, cfg.up, cfg.inp, k1.NPROJ, nvb_a, L, True)
    po = post.plan(m, cfg.oproj, cfg.front, nf, nvb_m)
    p = Plan()
    for r in pp.regions:
        p.region(r.name, r.nbytes, r.peer)
    p.region("kdone", (m // L) * k1.NH * 8, peer=False)
    for r in po.regions:
        p.region("o_" + r.name, r.nbytes, r.peer)
    for s in pp.stages:
        p.stage(s.name, s.tasks, s.waits_on, s.all_ranks)
    for s in po.stages:
        waits = tuple(w for w in s.waits_on) + (("kda",) if s.name == "oproj" else ())
        p.stage(s.name if s.name != "seam" else "mseam", s.tasks, tuple(w if w != "seam" else "mseam" for w in waits),
                s.all_ranks)
    gf, gs = cfg.front, cfg.sdown
    nbf = (m + gf.bm // gf.ks - 1) // (gf.bm // gf.ks)
    tnf = (nf + gf.bn - 1) // gf.bn
    p.region("rgat", m * mf.E * 4)
    p.region("lgat", m * mf.LAT * 2)
    p.region("fgf", TP * nbf * mf.gathered_tiles(gf.bn) * 8)
    p.region("fbuf", m * nf * 4, peer=False)
    p.region("sgu", m * 2 * mf.SI * 2, peer=False)
    p.region("smark", nbf * tnf * 8, peer=False)
    p.region("hbuf", m * mf.SI * 2, peer=False)
    p.region("hmark", m * 8, peer=False)
    p.stage("route", m, waits_on=("front",), all_ranks=True)
    p.stage("situ", m, waits_on=("front",))
    p.stage("sdown", ((m + gs.bm - 1) // gs.bm) * (H // gs.bn) * gs.ks, waits_on=("situ",))
    if capture:
        p.region("cap", m * H * 2)
        p.region("capf", TP * m * 8)
        p.stage("capture", m, waits_on=("seam",), all_ranks=True)
    return p.check(), pp, po


@flyc.jit
def _heads_wait(c, m0, bm, m, L, h0, nh):
    """Wait for the KDA (request, head) marks of rows m0 .. m0 + bm, heads h0 .. h0 + nh."""
    r0 = m0 // L
    nr = (fx.min(m0 + bm, m) - 1) // L - r0 + 1
    for i in range(c["tid"], nr * nh, THREADS):
        c["poll"]([(c["kdone"], (r0 + i // nh) * k1.NH + h0 + i % nh, 1)])
    gpu.barrier()


_BUILDS: dict = {}


def kda_layer(m, cfg, L, nf, nvb_a=0, nvb_m=0, write_a=False, write_m=False, nbk=8, launches=1, eps=1e-5,
              eps_n=1e-5, kpt=16, capture=False):
    """Build the layer launch; returns (launch, plan). Arguments: the pre half's
    (lat_p, sh_p, w_up (``pre.fold_up``), prefix, cw_a, w_in, w_ip, ip_buf (in_proj
    merge, bf16 [m, 6288] zeroed once), acc_u (bf16 [m, 896] zeroed once)), mid
    (the prefix after the MoE tail, [m, 896]), kda_tab (``pre.kda_table``; its
    core_out is the o_proj input ``core``), the post half's (wo, cw_m, w_ln, wf,
    out (MoE front output), acc_o (o_proj merge, zeroed once)), prefix_out (after
    the attention add), bank, the T6 tail's (bias f32 [896] (e_score_correction_bias),
    lat_out bf16 [m, 3584] + topk_ids int32 / topk_w f32 [m, 16] (the routed
    experts' input), w_sd [H, 768] (shared down rows of this rank), sh_out bf16
    [m, H] (the shared partial the next layer's MoE tail reads)), cnt (every tile
    counter, zeroed once), scratch, peers, rank, epoch, slot. linbf merges, the
    front in f32 (router logits). ``capture``: DSpark's capture of the previous
    layer (the attention seam's pre-norm mixture, gathered) -> cap_out [m, H] with
    row stride cap_ld; else cap_out is unused."""
    key = (m, cfg.key(), L, nf, nvb_a, nvb_m, write_a, write_m, nbk, launches, eps, eps_n, kpt, capture)
    if key in _BUILDS:
        return _BUILDS[key]
    gu, gi, go, gf, gs = cfg.up, cfg.inp, cfg.oproj, cfg.front, cfg.sdown
    assert nf == mf.NF and gf.ks > 1 and gs.ks == 1 and m % L == 0 and (not write_a or nvb_m <= nvb_a + 1) and (not write_m or nvb_m < nbk)
    # the up and in_proj finishes read and zero linbf tiles
    assert gu.ks > 1 and gi.ks > 1
    pl, pp, po = plan(m, cfg, L, nvb_a, nvb_m, nf, capture)
    offs, half = pl.peer_offsets
    soffs, _ = pl.scratch_offsets
    pre_offs = {r.name: offs[r.name] for r in pp.regions if r.peer}
    if capture:
        pre_offs.update(cap=offs["cap"], capf=offs["capf"])
    post_offs = {r.name: offs["o_" + r.name] for r in po.regions if r.peer}
    fo = pl.flags_offset()
    state_len = L + k1.CONV_W - 1
    nin = k1.NPROJ
    # pre half
    rb_u, ct_u = gu.bm // gu.ks, SL // gu.bn
    ktu, kti = LAT // gu.ks, H // gi.ks
    nown_u, nown_i = max(ktu // LSL, 1), max(kti // SL, 1)
    tnu, tmu = SL // gu.bn, (m + gu.bm - 1) // gu.bm
    tni, tmi = (nin + gi.bn - 1) // gi.bn, (m + gi.bm - 1) // gi.bm
    n_u, n_i = tmu * tnu * gu.ks, tmi * tni * gi.ks
    # post half
    rb_o, ct_o = go.bm // go.ks, SL // go.bn
    kto, ktf = KO // go.ks, H // gf.ks
    assert kto % k1.HD == 0
    nown_f = max(ktf // SL, 1)
    tno, tmo = H // go.bn, (m + go.bm - 1) // go.bm
    tnf, tmf = (nf + gf.bn - 1) // gf.bn, (m + gf.bm - 1) // gf.bm
    n_o, n_f = tmo * tno * go.ks, tmf * tnf * gf.ks
    cnt_i = tmu * tnu
    cnt_o = cnt_i + tmi * tni
    cnt_f = cnt_o + tmo * tno
    rbf = gf.bm // gf.ks
    nbf = (m + rbf - 1) // rbf
    tns, tms = H // gs.bn, (m + gs.bm - 1) // gs.bm
    n_s = tms * tns
    gems = (gu, gi, go, gf, gs)
    bmx, bnx = max(g.bm for g in gems), max(g.bn for g in gems)
    cwords = max(g.bm * (g.bn + 4) for g in gems)
    kwl = kdaw.lds_floats(L)

    @fx.struct
    class AbLds:
        xs: fx.Array[fx.Int32, 2 * bmx * RW, 16]
        ws: fx.Array[fx.Int32, 2 * bnx * RW, 16]

    @fx.struct
    class CLds:
        c: fx.Array[fx.Float32, cwords, 16]

    @fx.struct
    class KdwLds:
        v: fx.Array[fx.Float32, kwl, 16]

    @fx.union
    class Smem:
        ab: AbLds
        c: CLds
        kw: KdwLds

    @fx.struct
    class Small:
        slot: fx.Array[fx.Int32, 4, 16]
        rms: fx.Array[fx.Float32, gu.bm, 16]
        red: fx.Array[fx.Float32, max(red_words(nvb_a), red_words(nvb_m)), 16]

    name = (f"k31_kda_layer_m{m}_{'_'.join('x'.join(map(str, g)) for g in cfg.key())}_L{L}_n{nf}_a{nvb_a}{int(write_a)}"
            f"_m{nvb_m}{int(write_m)}of{nbk}_l{launches}_p{kpt}_c{int(capture)}")

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def layer_kernel(lat_p: Int64, sh_p: Int64, w_up: Int64, prefix: Int64, mid: Int64, prefix_out: Int64,
                     bank: Int64, cw_a: Int64, w_in: Int64, w_ip: Int64, ip_buf: Int64, acc_u: Int64, kda_tab: Int64,
                     core: Int64, wo: Int64, cw_m: Int64, w_ln: Int64, wf: Int64, acc_o: Int64, bias: Int64,
                     lat_out: Int64, topk_ids: Int64, topk_w: Int64, w_sd: Int64, sh_out: Int64, cap_out: Int64,
                     cap_ld: Int32, cnt: Int64, scratch: Int64, peers: Int64, rank: Int32, epoch: Int64,
                     slot_i: Int32):
        own = load_ptr64(peers, rank)
        with guard.scope(guard.Guard(epoch, own=own, peers=peers, rank=rank, npes=TP, flags_offset=fo)):
            tid, bid = fx.thread_idx.x, fx.block_idx.x
            alloc = fx.SharedAllocator()
            st = alloc.allocate(Smem)
            small = alloc.allocate(Small).peek()
            slot, rmsl, red = small.slot.ptr, small.rms.ptr, small.red.ptr
            lds, cl = st.ab.peek(), st.c.peek().c.ptr
            ca = coll.region_context(peers, rank, epoch, slot_i, pre_offs, half, tid, bid, launches)
            co = coll.region_context(peers, rank, epoch, slot_i, post_offs, half, tid, bid, launches)
            shs = scratch + fx.Int64(soffs["shs"])
            pd_a = sreg(scratch, soffs["pdone"], "pdone")
            pd_m = sreg(scratch, soffs["o_pdone"], "o_pdone")
            proj = sreg(scratch, soffs["proj"], "proj")
            kdone = sreg(scratch, soffs["kdone"], "kdone")
            tail_offs = {k: offs[k] for k in ("rgat", "lgat", "fgf")}
            cf = coll.region_context(peers, rank, epoch, slot_i, tail_offs, half, tid, bid, launches)
            fbuf = scratch + fx.Int64(soffs["fbuf"])
            sgu = scratch + fx.Int64(soffs["sgu"])
            hbuf = scratch + fx.Int64(soffs["hbuf"])
            smark = sreg(scratch, soffs["smark"], "smark")
            hmark = sreg(scratch, soffs["hmark"], "hmark")

            # ---- T1: MoE tail
            for t in range(bid, m * TP, BLOCKS):
                pre._push(ca, t // TP, t % TP, m, lat_p, sh_p)
            for row in range(bid, m, BLOCKS):
                pre._lred(ca, row, m, shs)
            lagx = ca["lag_own"].value
            for task in range(bid, n_u, BLOCKS):
                sk = task % gu.ks
                tile = task // gu.ks
                m0 = (tile // tnu) * gu.bm
                n0 = (tile % tnu) * gu.bn
                pre._rows_wait(ca, "lagf_own", m0, gu.bm, m, (sk * ktu) // LSL, nown_u)
                pre._row_rms(ca, m0, gu.bm, m, eps_n, rmsl)
                e_, add = atomic_lin(acc_u, SL, n0, m, SL, True)

                def scaled(row, col, vals, add=add, m0=m0):
                    s = fx.ptr_load(rmsl + fx.min(row - m0, gu.bm - 1))
                    add(row, col, [v * s for v in vals])

                mma_tile(tid, lagx, LAT, m, m0, w_up, LAT, SL, n0, sk * ktu, ktu, gu.bm, gu.bn, gu.wm, gu.wn,
                         lds.xs.ptr, lds.ws.ptr, None, DEPTH, (e_, scaled), cl)
                _arrive_all(tid, cnt, tile, gu.ks, slot)
                r0 = m0 + sk * rb_u
                pre._finish(ca, acc_u, shs, prefix, mid, m, r0, rb_u, n0, gu.bn, gu.ks, pd_a,
                            (r0 // rb_u) * ct_u + n0 // gu.bn)
            # ---- T2: attention seam
            for row in range(bid, m, BLOCKS):
                pre._seam(ca, row, m, rb_u, ct_u, nvb_a, nbk, mid, bank, cw_a, w_in, eps, pd_a, red, write_a, capture)
            # ---- T3: in_proj -> KDA
            xa = ca["ag_own"].value
            f0 = first_task(bid, max(BLOCKS - n_i, 0))
            for task in range(f0, n_i, BLOCKS):
                sk = task % gi.ks
                tile = task // gi.ks
                m0 = (tile // tni) * gi.bm
                n0 = (tile % tni) * gi.bn
                pre._rows_wait(ca, "agf_own", m0, gi.bm, m, (sk * kti) // SL, nown_i)
                mma_tile(tid, xa, H, m, m0, w_ip, H, nin, n0, sk * kti, kti, gi.bm, gi.bn, gi.wm, gi.wn, lds.xs.ptr,
                         lds.ws.ptr, None, DEPTH, atomic_lin(ip_buf, nin, n0, m, nin, True), cl)
                _arrive_all(tid, cnt + fx.Int64(4 * cnt_i), tile, gi.ks, slot)
                rbi = gi.bm // gi.ks
                pre._finish_proj(ca, ip_buf, m, nin, m0 + sk * rbi, rbi, n0, gi.bn, gi.ks, proj)
            ca.update({
                "S": m, "L": L, "SL": state_len, "lane": tid % 64, "wave": tid // 64, "lb": -5.0, "onorm_eps": eps_n,
                "proj": proj, "args": pre._kda_args(kda_tab), "kdone": kdone,
            })
            kdaw.kda_heads(ca, st.kw.peek().v.ptr, 0, kpt)
            # ---- T4: o_proj (core_out of this launch)
            for task in range(bid, n_o, BLOCKS):
                sk = task % go.ks
                tile = task // go.ks
                m0 = (tile // tno) * go.bm
                n0 = (tile % tno) * go.bn
                _heads_wait(ca, m0, go.bm, m, L, (sk * kto) // k1.HD, kto // k1.HD)
                mma_tile(tid, core, KO, m, m0, wo, KO, H, n0, sk * kto, kto, go.bm, go.bn, go.wm, go.wn, lds.xs.ptr,
                         lds.ws.ptr, dev_emit(acc_o, H, n0, m, H, False) if go.ks == 1 else None, DEPTH,
                         None if go.ks == 1 else atomic_lin(acc_o, H, n0, m, H, True), cl, CM_DEV)
                _arrive_all(tid, cnt + fx.Int64(4 * cnt_o), tile, go.ks, slot)
                piece = ((m0 + sk * rb_o) // rb_o) * ct_o + (n0 % SL) // go.bn
                _push_band_flag(co, acc_o, m, H, m0 + sk * rb_o, rb_o, n0, go.bn, go.ks, "linbf", SL, piece)
            # ---- T5: RS + residual, MLP seam
            for piece in range(bid, ((m + rb_o - 1) // rb_o) * ct_o, BLOCKS):
                post._rs_piece(co, piece, m, SL, rb_o, go.bn, mid, prefix_out, pd_m, False, None, False, CM_DEV)
            for row in range(bid, m, BLOCKS):
                post._seam_row(co, row, m, SL, rb_o, ct_o, nvb_m, nbk, prefix_out, bank, cw_m, w_ln, eps, pd_m, red,
                               write_m, False, None, False, CM_DEV if write_a else 0)
            # ---- T6: MoE front
            xm = co["ag_own"].value
            g0 = first_task(bid, max(BLOCKS - n_f, 0))
            for task in range(g0, n_f, BLOCKS):
                sk = task % gf.ks
                tile = task // gf.ks
                m0 = (tile // tnf) * gf.bm
                n0 = (tile % tnf) * gf.bn
                post._front_wait(co, m0, gf.bm, m, (sk * ktf) // SL, nown_f, False, None, False)
                mma_tile(tid, xm, H, m, m0, wf, H, nf, n0, sk * ktf, ktf, gf.bm, gf.bn, gf.wm, gf.wn, lds.xs.ptr,
                         lds.ws.ptr, None, DEPTH, atomic_lin(fbuf, nf, n0, m, nf, False), cl)
                _arrive_all(tid, cnt + fx.Int64(4 * cnt_f), tile, gf.ks, slot)
                r0 = m0 + sk * rbf
                mf.finish_front(cf, fbuf, sgu, m, r0, rbf, n0, gf.bn, r0 // rbf, n0 // gf.bn, nbf, smark, tnf)
            # ---- T6 tail: routing + latent for the experts, shared SiTU + down
            for row in range(bid, m, BLOCKS):
                mf.route_row(cf, row, m, rbf, nbf, gf.bn, bias, lat_out, topk_ids, topk_w)
            for row in range(first_task(bid, max(BLOCKS - m, 0)), m, BLOCKS):
                mf.situ_row(cf, row, m, rbf, tnf, gf.bn, sgu, smark, hbuf, hmark)
            for task in range(bid, n_s, BLOCKS):
                m0 = (task // tns) * gs.bm
                n0 = (task % tns) * gs.bn
                rows = fx.min(m0 + gs.bm, m) - m0
                if tid < rows:
                    cf["poll"]([(hmark, m0 + tid, 1)])
                gpu.barrier()
                mma_tile(tid, hbuf, mf.SI, m, m0, w_sd, mf.SI, H, n0, 0, mf.SI, gs.bm, gs.bn, gs.wm, gs.wn,
                         lds.xs.ptr, lds.ws.ptr, bf16_emit(sh_out, H, n0, m, H), DEPTH, None, cl, CM_DEV)
            # ---- DSpark capture of the previous layer
            if const_expr(capture):
                for row in range(first_task(bid, max(BLOCKS - m, 0)), m, BLOCKS):
                    pre.capture_row(ca, row, m, cap_out, cap_ld)

    @flyc.jit
    def launch(lat_p: Int64, sh_p: Int64, w_up: Int64, prefix: Int64, mid: Int64, prefix_out: Int64, bank: Int64,
               cw_a: Int64, w_in: Int64, w_ip: Int64, ip_buf: Int64, acc_u: Int64, kda_tab: Int64, core: Int64,
               wo: Int64, cw_m: Int64, w_ln: Int64, wf: Int64, acc_o: Int64, bias: Int64, lat_out: Int64,
               topk_ids: Int64, topk_w: Int64, w_sd: Int64, sh_out: Int64, cap_out: Int64, cap_ld: Int32, cnt: Int64,
               scratch: Int64, peers: Int64, rank: Int32, epoch: Int64, slot_i: Int32,
               stream: fx.Stream = fx.Stream(None)):  # noqa: B008
        layer_kernel(lat_p, sh_p, w_up, prefix, mid, prefix_out, bank, cw_a, w_in, w_ip, ip_buf, acc_u, kda_tab, core,
                     wo, cw_m, w_ln, wf, acc_o, bias, lat_out, topk_ids, topk_w, w_sd, sh_out, cap_out, cap_ld, cnt,
                     scratch, peers, rank, epoch, slot_i).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)

    _BUILDS[key] = (launch, pl)
    return launch, pl


def counters(m, cfg, nf):
    """int32 tile counters a layer build uses (up, in_proj, o_proj tiles)."""
    gu, gi, go, gf = cfg.up, cfg.inp, cfg.oproj, cfg.front
    return (((m + gu.bm - 1) // gu.bm) * (SL // gu.bn) + ((m + gi.bm - 1) // gi.bm) * ((k1.NPROJ + gi.bn - 1) // gi.bn)
            + ((m + go.bm - 1) // go.bm) * (H // go.bn) + ((m + gf.bm - 1) // gf.bm) * ((nf + gf.bn - 1) // gf.bn))
