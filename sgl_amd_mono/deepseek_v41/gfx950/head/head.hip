// SPDX-License-Identifier: Apache-2.0
// The DSpark verify head after the target's LM head, on this rank's vocab shard, in four launches: the scan (per row
// and vocab chunk: the temperature softmax stats of the target and the draft, the drafted token's logits, the greedy
// argmax, the simulated bonus's Gumbel key), the verdict (a block a request: the ranks' partials swapped through peer
// buffers, the accepted length), the bonus race (per request and vocab chunk) and the finish (the ranks' race bests
// swapped, the simulated override and the accept outputs SGLang's static tail writes). The rules are SGLang's
// accept_sampling_sharded (sharded_accept_hip.py), AcceptGreedy and static_accept_tail (static_accept_tail.py); the
// simulated bonus is simulated_bonus.py's draw, key for key.
#include <climits>
#include <cmath>
#include <cstdint>

#include <ATen/ATen.h>
#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <c10/hip/HIPException.h>
#include <hip/hip_runtime.h>
#include <torch/library.h>

#ifndef C10_HIP_KERNEL_LAUNCH_CHECK
#define C10_HIP_KERNEL_LAUNCH_CHECK() C10_CUDA_KERNEL_LAUNCH_CHECK()
#endif

namespace {

using bfloat16 = __bf16;
using uint32x4 = uint32_t __attribute__((ext_vector_type(4)));

constexpr int MAX_SLOTS       = 8;
constexpr int MAX_TP          = 8;
constexpr int MAX_BS          = 64;
constexpr int CHUNK           = 2048; // a scan block's vocab columns
constexpr int SCAN_THREADS    = 256;
constexpr int VERDICT_THREADS = 64;
constexpr int SPT             = CHUNK / SCAN_THREADS; // a scan thread's columns
constexpr int STATE           = 8;  // a request's verdict words for the race and the finish
constexpr int PART            = 10; // a (row, chunk)'s partial words
constexpr int ROW_WORDS       = 8;  // a row's exchanged words: t_m, t_s, t_x, d_m, d_s, d_x, argmax value, argmax id
constexpr int PACKETS         = 32; // round 1's 16 B packets a (request, source): 3 words and the tag each
// the exchange: [round 2][request MAX_BS][parity 2][source MAX_TP][PACKETS] packets of 16 B (head.py's EXCHANGE_BYTES)
constexpr int64_t ROUND_PACKETS = int64_t(MAX_BS) * 2 * MAX_TP * PACKETS;

constexpr uint32_t ACCEPT_STEP = 0xFFFFFFFEu, BONUS_STEP = 0xFFFFFFFFu;
constexpr int      NO_INDEX    = 0x7FFFFFFF;

struct Params {
    const void    *target; // [rows, >= col0 + width] logits, f32 or bf16 (target_bf16)
    int64_t        target_stride;
    int            target_bf16;
    int            col0;  // the column of global id ``start`` in ``target``
    int            start; // this rank's first global id
    int            width; // this rank's ids
    int            vocab; // the simulated bonus's counter stride and "no index" (simulated_bonus.py's vocab)
    float          cap;   // > 0: the logits are softcapped here (cap * tanh(x / cap))
    const float   *draft;      // [bs, gamma, padded] the draft's corrected logits on this rank's shard
    const int64_t *draft_addr; // [1] or null: the address of the draft's logits, read at run time instead of draft
    int64_t        draft_padded;
    int            gamma;
    const int64_t *cand; // [bs, slots] the anchor and the drafts
    int64_t        cand_stride;
    int            slots;
    const float   *temps;     // [bs] the accept's temperatures; <= 0: the request takes the greedy rule
    const float   *sim_temps; // [bs] the simulated bonus's
    const int64_t *sim_len;   // [1] the simulated correct length, or null
    const int     *sim_seed;  // [1]
    const int64_t *key;       // [1] the draft noise's Philox key
    const int64_t *offset;    // [1] its block counter
    const void    *seq_lens;
    int            seq_lens_i64;
    const int64_t *verify_lens;
    const int     *gate;
    int            bs, nchunk, rank, tp;
    float         *part;  // [bs * slots, nchunk, PART]
    float         *state; // [MAX_BS, STATE]
    float         *race;  // [bs, nchunk, 2]
    int64_t       *correct_out, *bonus_out, *new_seq_out, *out_tokens;
    int64_t        out_tokens_stride;
    int           *cap_trim_out, *commit_out, *gated_out, *block_out;
    int           *calls; // [MAX_BS]
    void          *recv[MAX_TP];
    uint64_t       deadline_ticks;
};

// Philox4x32-10 (tl.philox's)
__device__ inline uint32x4 philox(uint32_t k0, uint32_t k1, uint32_t c0, uint32_t c1, uint32_t c2, uint32_t c3) {
#pragma unroll
    for (int r = 0; r < 10; ++r) {
        const uint32_t a0 = c0, a2 = c2;
        c0 = __umulhi(0xCD9E8D57u, a2) ^ c1 ^ k0;
        c2 = __umulhi(0xD2511F53u, a0) ^ c3 ^ k1;
        c1 = 0xCD9E8D57u * a2;
        c3 = 0xD2511F53u * a0;
        k0 += 0x9E3779B9u;
        k1 += 0xBB67AE85u;
    }
    return uint32x4{c0, c1, c2, c3};
}

// tl.uint_to_uniform_float of a uint32
__device__ inline float uniform(uint32_t w) {
    int32_t x = static_cast<int32_t>(w);
    x         = x < 0 ? ~x : x;
    return static_cast<float>(x) * 4.6566127342e-10f;
}

__device__ inline float load_target(const Params &p, int64_t row, int j) {
    const int64_t at = row * p.target_stride + p.col0 + j;
    float         x  = p.target_bf16 ? static_cast<float>(static_cast<const bfloat16 *>(p.target)[at])
                                     : static_cast<const float *>(p.target)[at];
    if (p.cap > 0.0f) {
        x = p.cap * tanhf(x / p.cap);
    }
    return x;
}

// 4 columns from j (j % 4 == 0); past hi -inf
__device__ inline void load_target4(const Params &p, int64_t row, int j, int hi, float (&v)[4]) {
    const int64_t at = row * p.target_stride + p.col0 + j;
    if (j + 4 <= hi && (at & 3) == 0) {
        if (p.target_bf16) {
            const uint2 w = *reinterpret_cast<const uint2 *>(static_cast<const bfloat16 *>(p.target) + at);
            v[0]          = __uint_as_float(w.x << 16);
            v[1]          = __uint_as_float(w.x & 0xFFFF0000u);
            v[2]          = __uint_as_float(w.y << 16);
            v[3]          = __uint_as_float(w.y & 0xFFFF0000u);
        } else {
            const float4 w = *reinterpret_cast<const float4 *>(static_cast<const float *>(p.target) + at);
            v[0] = w.x, v[1] = w.y, v[2] = w.z, v[3] = w.w;
        }
        if (p.cap > 0.0f) {
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                v[k] = p.cap * tanhf(v[k] / p.cap);
            }
        }
        return;
    }
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        v[k] = j + k < hi ? load_target(p, row, j + k) : -INFINITY;
    }
}

__device__ inline void load_draft4(const float *row, int j, int hi, float (&v)[4]) {
    if (j + 4 <= hi && (reinterpret_cast<uintptr_t>(row + j) & 15) == 0) {
        const float4 w = *reinterpret_cast<const float4 *>(row + j);
        v[0] = w.x, v[1] = w.y, v[2] = w.z, v[3] = w.w;
        return;
    }
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        v[k] = j + k < hi ? row[j + k] : -INFINITY;
    }
}

// _fold / _merge's running (max, sum of exp(v - max))
__device__ inline void fold(float &m, float &s, float m2, float s2) {
    const float big = fmaxf(m, m2);
    const float ref = big == -INFINITY ? 0.0f : big;
    s               = s * expf(m - ref) + s2 * expf(m2 - ref);
    m               = big;
}

// torch.argmax's order: NaN first, then the larger value, ties to the lower id
__device__ inline void keep_argmax(float &v, int &i, float v2, int i2) {
    const bool n = v != v, n2 = v2 != v2;
    if ((n2 && !n) || (n2 == n && (v2 > v || (v2 == v && i2 < i)))) {
        v = v2, i = i2;
    }
}

// the simulated bonus's select: the larger value, ties to the lower id
__device__ inline void keep_best(float &v, int &i, float v2, int i2) {
    if (v2 > v || (v2 == v && i2 < i)) {
        v = v2, i = i2;
    }
}

// the bonus race's finish (_markov_sample_finish_kernel): a NaN key wins, then the larger, ties to the lower id
__device__ inline void keep_race(float &v, int &i, float v2, int i2) {
    if (i2 == NO_INDEX) {
        return;
    }
    const bool n = v != v, n2 = v2 != v2;
    if (i == NO_INDEX || (n2 && !n) || (n2 == n && (v2 > v || (v2 == v && i2 < i)))) {
        v = v2, i = i2;
    }
}

template <int N, typename F> __device__ inline void block_reduce(float *sv, int *si, float &v, int &i, F keep) {
    sv[threadIdx.x] = v, si[threadIdx.x] = i;
    __syncthreads();
    for (int half = N / 2; half > 0; half /= 2) {
        if (threadIdx.x < half) {
            float a = sv[threadIdx.x];
            int   b = si[threadIdx.x];
            keep(a, b, sv[threadIdx.x + half], si[threadIdx.x + half]);
            sv[threadIdx.x] = a, si[threadIdx.x] = b;
        }
        __syncthreads();
    }
    v = sv[0], i = si[0];
    __syncthreads();
}

template <int N> __device__ inline void block_fold(float *sm, float *ss, float &m, float &s) {
    sm[threadIdx.x] = m, ss[threadIdx.x] = s;
    __syncthreads();
    for (int half = N / 2; half > 0; half /= 2) {
        if (threadIdx.x < half) {
            float a = sm[threadIdx.x], b = ss[threadIdx.x];
            fold(a, b, sm[threadIdx.x + half], ss[threadIdx.x + half]);
            sm[threadIdx.x] = a, ss[threadIdx.x] = b;
        }
        __syncthreads();
    }
    m = sm[0], s = ss[0];
    __syncthreads();
}

__device__ inline void store_nt(uint32x4 *ptr, uint32x4 val) {
    asm volatile("global_store_dwordx4 %0, %1, off sc0 sc1 nt\n\ts_nop 1" ::"v"(ptr), "v"(val) : "memory");
}

__device__ inline uint32x4 load_nt(const uint32x4 *ptr) {
    uint32x4 val;
    asm volatile("global_load_dwordx4 %0, %1, off sc0 sc1 nt\n\ts_waitcnt vmcnt(0)" : "=v"(val) : "v"(ptr) : "memory");
    return val;
}

// a peer's packet of this call; past the deadline the launch traps, stopping this rank's step
__device__ inline uint32x4 poll_tag(const uint32x4 *src, uint32_t tag, uint64_t deadline) {
    uint64_t t0 = 0;
    for (int n = 1;; ++n) {
        const uint32x4 pk = load_nt(src);
        if (pk.w == tag) {
            return pk;
        }
        __builtin_amdgcn_s_sleep(2);
        if (n % 1024 == 0) {
            const uint64_t now = __builtin_amdgcn_s_memrealtime();
            if (t0 == 0) {
                t0 = now;
            } else if (now - t0 > deadline) {
                __builtin_trap();
            }
        }
    }
}

__device__ inline int64_t slot_of(int round, int req, uint32_t call, int src, int packet) {
    return round * ROUND_PACKETS + ((int64_t(req) * 2 + call % 2) * MAX_TP + src) * PACKETS + packet;
}

__device__ inline const float *draft_of(const Params &p) {
    return p.draft_addr != nullptr ? reinterpret_cast<const float *>(p.draft_addr[0]) : p.draft;
}

// a block a (row, vocab chunk): a thread's SPT contiguous columns in registers
__global__ __launch_bounds__(SCAN_THREADS) void head_scan_kernel(const Params p) {
    __shared__ float sv[SCAN_THREADS];
    __shared__ int   si[SCAN_THREADS];
    __shared__ float sw[SCAN_THREADS];
    const int        r = blockIdx.y, c = blockIdx.x;
    const int        b = r / p.slots, i = r % p.slots;
    const bool       has_x   = i < p.slots - 1;
    const float      t_acc   = p.temps[b];
    const bool       greedy  = !(t_acc > 0.0f);
    const float      inv_t   = 1.0f / fmaxf(t_acc, 1e-5f);
    const bool       sim     = p.sim_len != nullptr && i == static_cast<int>(p.sim_len[0]);
    const float      t_sim   = p.sim_temps[b];
    const uint32_t   seed    = static_cast<uint32_t>(p.sim_seed[0]);
    const uint32_t   seed_hi = p.sim_seed[0] < 0 ? 0xFFFFFFFFu : 0u;
    const int        lo = c * CHUNK, hi = min(lo + CHUNK, p.width);
    const float     *drow = draft_of(p) + (static_cast<int64_t>(b) * p.gamma + i) * p.draft_padded;
    // column of x[k]: lo + (threadIdx.x + (k / 4) * SCAN_THREADS) * 4 + k % 4
    const auto col = [&](int k) { return lo + (threadIdx.x + (k / 4) * SCAN_THREADS) * 4 + k % 4; };

    float x[SPT];
#pragma unroll
    for (int q = 0; q < SPT / 4; ++q) {
        float v[4];
        load_target4(p, r, col(4 * q), hi, v);
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            x[4 * q + k] = v[k];
        }
    }
    float t_m = -INFINITY, t_s = 0.0f, d_m = -INFINITY, d_s = 0.0f;
    float a_v = -INFINITY, s_v = -INFINITY;
    int   a_i = INT_MAX, s_i = p.vocab;
    if (greedy) {
#pragma unroll
        for (int k = 0; k < SPT; ++k) {
            if (col(k) < hi) {
                keep_argmax(a_v, a_i, x[k], p.start + col(k));
            }
        }
    } else {
#pragma unroll
        for (int k = 0; k < SPT; ++k) {
            t_m = fmaxf(t_m, x[k] * inv_t);
        }
        if (t_m != -INFINITY) {
#pragma unroll
            for (int k = 0; k < SPT; ++k) {
                t_s += __expf(x[k] * inv_t - t_m);
            }
        }
        if (has_x) {
            float d[SPT];
#pragma unroll
            for (int q = 0; q < SPT / 4; ++q) {
                float v[4];
                load_draft4(drow, col(4 * q), hi, v);
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    d[4 * q + k] = v[k] * inv_t;
                    d_m          = fmaxf(d_m, d[4 * q + k]);
                }
            }
            if (d_m != -INFINITY) {
#pragma unroll
                for (int k = 0; k < SPT; ++k) {
                    d_s += __expf(d[k] - d_m);
                }
            }
        }
    }
    if (sim) {
#pragma unroll
        for (int k = 0; k < SPT; ++k) {
            if (col(k) < hi) {
                const int g   = p.start + col(k);
                float     key = x[k];
                if (t_sim > 0.0f) {
                    const uint32_t ctr = static_cast<uint32_t>(b * p.vocab + g);
                    const float    u   = uniform(philox(seed, seed_hi, ctr, 0u, 0u, 0u).x);
                    key                = x[k] / fmaxf(t_sim, 1e-5f) - logf(-logf(u));
                }
                keep_best(s_v, s_i, key, g);
            }
        }
    }
    if (!greedy) {
        block_fold<SCAN_THREADS>(sv, sw, t_m, t_s);
        if (has_x) {
            block_fold<SCAN_THREADS>(sv, sw, d_m, d_s);
        }
    } else {
        block_reduce<SCAN_THREADS>(sv, si, a_v, a_i, keep_argmax);
    }
    if (sim) {
        block_reduce<SCAN_THREADS>(sv, si, s_v, s_i, keep_best);
    }
    if (threadIdx.x != 0) {
        return;
    }
    float t_x = -INFINITY, d_x = -INFINITY;
    if (has_x && !greedy) {
        const int xi = static_cast<int>(p.cand[b * p.cand_stride + i + 1]) - p.start;
        if (xi >= lo && xi < hi) {
            t_x = load_target(p, r, xi) * inv_t;
            d_x = drow[xi] * inv_t;
        }
    }
    float *out = p.part + (static_cast<int64_t>(r) * p.nchunk + c) * PART;
    out[0]     = t_m;
    out[1]     = t_s;
    out[2]     = t_x;
    out[3]     = d_m;
    out[4]     = d_s;
    out[5]     = d_x;
    out[6]     = a_v;
    out[7]     = __int_as_float(a_i);
    out[8]     = s_v;
    out[9]     = __int_as_float(s_i);
}

// a block a request: the ranks' rows swapped and merged, the accepted length; the race's row stats to ``state``
__global__ __launch_bounds__(VERDICT_THREADS) void head_verdict_kernel(const Params p) {
    __shared__ float words[MAX_TP][PACKETS * 3];
    __shared__ float rows[MAX_SLOTS][ROW_WORDS];
    const int        b = blockIdx.x, S = p.slots, tid = threadIdx.x;
    const bool       greedy = !(p.temps[b] > 0.0f);
    uint32_t         call   = static_cast<uint32_t>(p.calls[b]) + 1;
    call                    = call ? call : 1;
    const int sl            = p.sim_len != nullptr ? static_cast<int>(p.sim_len[0]) : -1;

    // 1. this rank's row partials, chunk by chunk; the simulated row's best into the last two words
    float *mine = words[p.rank];
    if (tid < S) {
        const float *part = p.part + static_cast<int64_t>(b * S + tid) * p.nchunk * PART;
        float        t_m = -INFINITY, t_s = 0.0f, t_x = -INFINITY, d_m = -INFINITY, d_s = 0.0f, d_x = -INFINITY;
        float        a_v = -INFINITY;
        int          a_i = INT_MAX;
        for (int k = 0; k < p.nchunk; ++k, part += PART) {
            if (greedy) {
                keep_argmax(a_v, a_i, part[6], __float_as_int(part[7]));
            } else {
                fold(t_m, t_s, part[0], part[1]);
                t_x = fmaxf(t_x, part[2]);
                fold(d_m, d_s, part[3], part[4]);
                d_x = fmaxf(d_x, part[5]);
            }
        }
        float *w = mine + tid * ROW_WORDS;
        w[0] = t_m, w[1] = t_s, w[2] = t_x, w[3] = d_m, w[4] = d_s, w[5] = d_x, w[6] = a_v;
        w[7] = __int_as_float(a_i);
    }
    if (tid == MAX_SLOTS) {
        float s_v = -INFINITY;
        int   s_i = p.vocab;
        if (sl >= 0 && sl < S) {
            const float *part = p.part + static_cast<int64_t>(b * S + sl) * p.nchunk * PART;
            for (int k = 0; k < p.nchunk; ++k, part += PART) {
                keep_best(s_v, s_i, part[8], __float_as_int(part[9]));
            }
        }
        mine[S * ROW_WORDS]     = s_v;
        mine[S * ROW_WORDS + 1] = __int_as_float(s_i);
    }
    __syncthreads();

    // 2. round 1: every rank's rows to every rank
    const int npk = (S * ROW_WORDS + 2 + 2) / 3;
    if (tid < npk) {
        const uint32x4 pk{__float_as_uint(mine[3 * tid]), __float_as_uint(mine[3 * tid + 1]),
                          __float_as_uint(mine[3 * tid + 2]), call};
        for (int q = 0; q < p.tp; ++q) {
            store_nt(static_cast<uint32x4 *>(p.recv[q]) + slot_of(0, b, call, p.rank, tid), pk);
        }
        for (int q = 0; q < p.tp; ++q) {
            if (q == p.rank) {
                continue;
            }
            const uint32x4 got = poll_tag(static_cast<const uint32x4 *>(p.recv[p.rank]) + slot_of(0, b, call, q, tid),
                                          call, p.deadline_ticks);
            words[q][3 * tid]     = __uint_as_float(got.x);
            words[q][3 * tid + 1] = __uint_as_float(got.y);
            words[q][3 * tid + 2] = __uint_as_float(got.z);
        }
    }
    __syncthreads();

    // 3. the ranks' rows merged in rank order (_merge)
    if (tid < S) {
        float t_m = -INFINITY, t_x = -INFINITY, d_m = -INFINITY, d_x = -INFINITY, a_v = -INFINITY;
        int   a_i = INT_MAX;
        for (int q = 0; q < p.tp; ++q) {
            const float *w = words[q] + tid * ROW_WORDS;
            t_m            = fmaxf(t_m, w[0]);
            d_m            = fmaxf(d_m, w[3]);
            t_x            = fmaxf(t_x, w[2]);
            d_x            = fmaxf(d_x, w[5]);
            keep_argmax(a_v, a_i, w[6], __float_as_int(w[7]));
        }
        const float t_ref = t_m == -INFINITY ? 0.0f : t_m, d_ref = d_m == -INFINITY ? 0.0f : d_m;
        float       t_s = 0.0f, d_s = 0.0f;
        for (int q = 0; q < p.tp; ++q) {
            const float *w = words[q] + tid * ROW_WORDS;
            t_s += w[1] * expf(w[0] - t_ref);
            d_s += w[4] * expf(w[3] - d_ref);
        }
        float *o = rows[tid];
        o[0] = t_m, o[1] = t_s, o[2] = expf(t_x - t_m) / t_s; // p(x)
        o[3] = d_m, o[4] = d_s, o[5] = expf(d_x - d_m) / d_s; // q(x)
        o[6] = a_v, o[7] = __int_as_float(a_i);
    }
    __syncthreads();

    // 4. the accepted length: the greedy chain, or u * q(x) < p(x) row by row
    if (tid != 0) {
        return;
    }
    int accepted = S - 1;
    if (greedy) {
        for (int k = 0; k < S - 1; ++k) {
            if (p.cand[b * p.cand_stride + k + 1] != __float_as_int(rows[k][7])) {
                accepted = k;
                break;
            }
        }
    } else {
        const uint32_t k0 = static_cast<uint32_t>(static_cast<uint64_t>(p.key[0]) & 0xFFFFFFFFull);
        const uint32_t k1 = static_cast<uint32_t>(static_cast<uint64_t>(p.key[0]) >> 32);
        const uint32_t c3 = static_cast<uint32_t>(p.offset[0] & 0xFFFFFFFF);
        for (int k = 0; k < S - 1; ++k) {
            const float u  = uniform(philox(k0, k1, k, b, ACCEPT_STEP, c3).x);
            const bool  ok = rows[k][5] > 0.0f && u * rows[k][5] < rows[k][2];
            if (!ok) {
                accepted = k;
                break;
            }
        }
    }
    float *st = p.state + b * STATE;
    st[0]     = __int_as_float(accepted);
    st[1]     = rows[accepted][0];
    st[2]     = rows[accepted][1];
    st[3]     = rows[accepted][3];
    st[4]     = rows[accepted][4];
    st[5]     = __int_as_float(__float_as_int(rows[accepted][7]));                 // the greedy bonus
    st[6]     = __int_as_float(sl >= 0 ? __float_as_int(rows[sl][7]) : NO_INDEX); // the simulated greedy bonus
    float s_v = -INFINITY;
    int   s_i = p.vocab;
    for (int q = 0; q < p.tp; ++q) {
        keep_best(s_v, s_i, words[q][S * ROW_WORDS], __float_as_int(words[q][S * ROW_WORDS + 1]));
    }
    st[7] = __int_as_float(s_i); // the simulated sampled bonus
}

// a block a (request, vocab chunk): the bonus race, max(p - q, 0) after a rejection, p after a full chain; a thread
// a Philox quad of 4 tokens at a time
__global__ __launch_bounds__(SCAN_THREADS) void head_race_kernel(const Params p) {
    __shared__ float sv[SCAN_THREADS];
    __shared__ int   si[SCAN_THREADS];
    const int        c = blockIdx.x, b = blockIdx.y, S = p.slots;
    float            race_v = -INFINITY;
    int              race_i = NO_INDEX;
    const float      t_acc  = p.temps[b];
    if (t_acc > 0.0f) {
        const float   *st       = p.state + b * STATE;
        const int      accepted = __float_as_int(st[0]);
        const float    t_m = st[1], t_s = st[2], d_m = st[3], d_s = st[4];
        const float    inv_t    = 1.0f / fmaxf(t_acc, 1e-5f);
        const int64_t  row      = static_cast<int64_t>(b) * S + accepted;
        const bool     residual = accepted < S - 1;
        const float   *drow     = draft_of(p) + (static_cast<int64_t>(b) * p.gamma + accepted) * p.draft_padded;
        const uint32_t k0       = static_cast<uint32_t>(static_cast<uint64_t>(p.key[0]) & 0xFFFFFFFFull);
        const uint32_t k1       = static_cast<uint32_t>(static_cast<uint64_t>(p.key[0]) >> 32);
        const uint32_t c3       = static_cast<uint32_t>(p.offset[0] & 0xFFFFFFFF);
        const float    inv_ts = 1.0f / t_s, inv_ds = 1.0f / d_s;
        const int      lo = c * CHUNK, hi = min(lo + CHUNK, p.width);
        for (int j4 = lo + threadIdx.x * 4; j4 < hi; j4 += SCAN_THREADS * 4) {
            const uint32_t g4 = static_cast<uint32_t>(p.start + j4);
            const uint32x4 w  = philox(k0, k1, g4 >> 2, b, BONUS_STEP, c3);
            const uint32_t ws[4]{w.x, w.y, w.z, w.w};
            float          t4[4], d4[4];
            load_target4(p, row, j4, hi, t4);
            if (residual) {
                load_draft4(drow, j4, hi, d4);
            }
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                if (j4 + k >= hi) {
                    break;
                }
                const float pj   = __expf(t4[k] * inv_t - t_m) * inv_ts;
                float       mass = pj;
                if (residual) {
                    mass = fmaxf(pj - __expf(d4[k] * inv_t - d_m) * inv_ds, 0.0f);
                }
                const float noise = -log1pf(-(static_cast<float>(ws[k]) + 0.5f) * 2.3283064365386963e-10f);
                const float key   = mass > 0.0f ? __logf(mass) - __logf(noise) : -INFINITY;
                keep_race(race_v, race_i, key, key == -INFINITY ? NO_INDEX : static_cast<int>(g4) + k);
            }
        }
    }
    block_reduce<SCAN_THREADS>(sv, si, race_v, race_i, keep_race);
    if (threadIdx.x == 0) {
        float *out = p.race + (static_cast<int64_t>(b) * p.nchunk + c) * 2;
        out[0]     = race_v;
        out[1]     = __int_as_float(race_i);
    }
}

// a block a request: the ranks' race bests swapped, the simulated override, the accept outputs
__global__ __launch_bounds__(64) void head_finish_kernel(const Params p) {
    __shared__ float r_v[MAX_TP];
    __shared__ int   r_i[MAX_TP];
    const int        b = blockIdx.x, S = p.slots, tid = threadIdx.x;
    const bool       greedy = !(p.temps[b] > 0.0f);
    uint32_t         call   = static_cast<uint32_t>(p.calls[b]) + 1;
    call                    = call ? call : 1;
    if (tid == 0) {
        float        v = -INFINITY;
        int          i = NO_INDEX;
        const float *race = p.race + static_cast<int64_t>(b) * p.nchunk * 2;
        for (int k = 0; k < p.nchunk; ++k) {
            keep_race(v, i, race[2 * k], __float_as_int(race[2 * k + 1]));
        }
        const uint32x4 pk{__float_as_uint(v), static_cast<uint32_t>(i), 0u, call};
        for (int q = 0; q < p.tp; ++q) {
            store_nt(static_cast<uint32x4 *>(p.recv[q]) + slot_of(1, b, call, p.rank, 0), pk);
        }
        r_v[p.rank] = v, r_i[p.rank] = i;
    }
    __syncthreads();
    if (tid < p.tp && tid != p.rank) {
        const uint32x4 got = poll_tag(static_cast<const uint32x4 *>(p.recv[p.rank]) + slot_of(1, b, call, tid, 0),
                                      call, p.deadline_ticks);
        r_v[tid] = __uint_as_float(got.x), r_i[tid] = static_cast<int>(got.y);
    }
    __syncthreads();
    if (tid != 0) {
        return;
    }
    const float *st       = p.state + b * STATE;
    const int    accepted = __float_as_int(st[0]);
    int64_t      bonus;
    if (greedy) {
        bonus = __float_as_int(st[5]);
    } else {
        float v = -INFINITY;
        int   i = NO_INDEX;
        for (int q = 0; q < p.tp; ++q) {
            keep_race(v, i, r_v[q], r_i[q]);
        }
        bonus = i;
    }
    int correct = accepted;
    if (p.sim_len != nullptr) {
        correct = static_cast<int>(p.sim_len[0]);
        bonus   = greedy ? __float_as_int(st[6]) : __float_as_int(st[7]);
    }
    const int     commit = correct + 1;
    const int64_t seq    = p.seq_lens_i64 ? static_cast<const int64_t *>(p.seq_lens)[b]
                                          : static_cast<const int *>(p.seq_lens)[b];
    p.correct_out[b]  = correct;
    p.bonus_out[b]    = bonus;
    p.cap_trim_out[b] = 0;
    p.commit_out[b]   = commit;
    p.new_seq_out[b]  = seq + commit;
    if (p.gated_out != nullptr) {
        p.gated_out[b] = min(commit, static_cast<int>(p.verify_lens[b])) * p.gate[0];
    }
    if (p.block_out != nullptr) {
        p.block_out[b] = commit;
    }
    for (int k = 0; k < S; ++k) {
        p.out_tokens[b * p.out_tokens_stride + k] =
            k < correct ? p.cand[b * p.cand_stride + k + 1] : (k == correct ? bonus : 0);
    }
    p.calls[b] = static_cast<int>(call);
}

void check(const at::Tensor &t, const at::Tensor &ref, at::ScalarType dtype, const char *name) {
    TORCH_CHECK(t.device() == ref.device() && t.scalar_type() == dtype, name, " must be ", dtype, " on ",
                ref.device());
    TORCH_CHECK(t.numel() <= 1 || t.stride(-1) == 1, name, " must have unit inner stride");
}

// target [rows, >= col0 + width] f32 / bf16, draft [bs, gamma, padded] f32, cand [bs, slots] int64, temps / sim_temps
// [>= bs] f32, sim_len [1] int64 or none, sim_seed [1] int32, key / offset [1] int64, seq_lens [>= bs] int64 / int32,
// verify_lens [>= bs] int64, gate [1] int32, part f32 scratch (head.py's part_words), the outputs [>= bs] (out_tokens
// [>= bs, slots]), gated / block int32 or none, calls [64] int32, peers [tp] int64 on the host. draft_addr [1] int64
// or none: the address of a tensor shaped and laid out as draft, read by the launch instead of draft's (the caller
// keeps that tensor alive)
void verify_head(const at::Tensor &target, int64_t col0, int64_t start, int64_t width, int64_t vocab, double cap,
                 const at::Tensor &draft, const std::optional<at::Tensor> &draft_addr, const at::Tensor &cand,
                 const at::Tensor &temps, const at::Tensor &sim_temps,
                 const std::optional<at::Tensor> &sim_len, const at::Tensor &sim_seed, const at::Tensor &key,
                 const at::Tensor &offset, const at::Tensor &seq_lens, const at::Tensor &verify_lens,
                 const at::Tensor &gate, at::Tensor &part, at::Tensor &correct_out, at::Tensor &bonus_out,
                 at::Tensor &cap_trim_out, at::Tensor &commit_out, at::Tensor &new_seq_out, at::Tensor &out_tokens,
                 const std::optional<at::Tensor> &gated_out, const std::optional<at::Tensor> &block_out,
                 at::Tensor &calls, const at::Tensor &peers, int64_t rank, double deadline_s) {
    const int64_t bs = cand.size(0), slots = cand.size(1);
    TORCH_CHECK(cand.dim() == 2 && bs >= 1 && bs <= MAX_BS && slots >= 2 && slots <= MAX_SLOTS,
                "cand [1 .. ", MAX_BS, ", 2 .. ", MAX_SLOTS, "]");
    TORCH_CHECK(target.dim() == 2 && target.size(0) >= bs * slots && col0 + width <= target.size(1) && width > 0,
                "target [bs * slots, >= col0 + width]");
    TORCH_CHECK(target.scalar_type() == at::kFloat || target.scalar_type() == at::kBFloat16, "target f32 / bf16");
    TORCH_CHECK(target.stride(1) == 1, "target unit inner stride");
    TORCH_CHECK(draft.dim() == 3 && draft.size(0) >= bs && draft.size(1) >= slots - 1 && draft.size(2) >= width &&
                    draft.is_contiguous(),
                "draft [bs, gamma, padded] contiguous");
    check(draft, target, at::kFloat, "draft");
    if (draft_addr) {
        check(*draft_addr, target, at::kLong, "draft_addr");
    }
    check(cand, target, at::kLong, "cand");
    check(temps, target, at::kFloat, "temps");
    check(sim_temps, target, at::kFloat, "sim_temps");
    check(sim_seed, target, at::kInt, "sim_seed");
    check(key, target, at::kLong, "key");
    check(offset, target, at::kLong, "offset");
    TORCH_CHECK(seq_lens.scalar_type() == at::kLong || seq_lens.scalar_type() == at::kInt, "seq_lens int64 / int32");
    check(verify_lens, target, at::kLong, "verify_lens");
    check(gate, target, at::kInt, "gate");
    check(part, target, at::kFloat, "part");
    check(correct_out, target, at::kLong, "correct_out");
    check(bonus_out, target, at::kLong, "bonus_out");
    check(cap_trim_out, target, at::kInt, "cap_trim_out");
    check(commit_out, target, at::kInt, "commit_out");
    check(new_seq_out, target, at::kLong, "new_seq_out");
    check(out_tokens, target, at::kLong, "out_tokens");
    check(calls, target, at::kInt, "calls");
    TORCH_CHECK(calls.numel() >= MAX_BS, "calls [", MAX_BS, "]");
    TORCH_CHECK(peers.device().is_cpu() && peers.scalar_type() == at::kLong && peers.numel() >= 1 &&
                    peers.numel() <= MAX_TP && rank >= 0 && rank < peers.numel(),
                "peers: every rank's exchange buffer (int64, host)");
    Params p{};
    p.target        = target.data_ptr();
    p.target_stride = target.stride(0);
    p.target_bf16   = target.scalar_type() == at::kBFloat16;
    p.col0          = static_cast<int>(col0);
    p.start         = static_cast<int>(start);
    p.width         = static_cast<int>(width);
    p.vocab         = static_cast<int>(vocab);
    p.cap           = static_cast<float>(cap);
    p.draft         = draft.data_ptr<float>();
    p.draft_addr    = draft_addr ? draft_addr->data_ptr<int64_t>() : nullptr;
    p.draft_padded  = draft.size(2);
    p.gamma         = static_cast<int>(draft.size(1));
    p.cand          = cand.data_ptr<int64_t>();
    p.cand_stride   = cand.stride(0);
    p.slots         = static_cast<int>(slots);
    p.temps         = temps.data_ptr<float>();
    p.sim_temps     = sim_temps.data_ptr<float>();
    if (sim_len) {
        check(*sim_len, target, at::kLong, "sim_len");
        p.sim_len = sim_len->data_ptr<int64_t>();
    }
    p.sim_seed     = sim_seed.data_ptr<int>();
    p.key          = key.data_ptr<int64_t>();
    p.offset       = offset.data_ptr<int64_t>();
    p.seq_lens     = seq_lens.data_ptr();
    p.seq_lens_i64 = seq_lens.scalar_type() == at::kLong;
    p.verify_lens  = verify_lens.data_ptr<int64_t>();
    p.gate         = gate.data_ptr<int>();
    p.bs           = static_cast<int>(bs);
    p.nchunk       = static_cast<int>((width + CHUNK - 1) / CHUNK);
    const int64_t scan_words = bs * slots * p.nchunk * PART;
    TORCH_CHECK(part.numel() >= scan_words + MAX_BS * STATE + bs * p.nchunk * 2,
                "part [bs * slots * nchunk * ", PART, " + ", MAX_BS * STATE, " + bs * nchunk * 2]");
    p.rank              = static_cast<int>(rank);
    p.tp                = static_cast<int>(peers.numel());
    p.part              = part.data_ptr<float>();
    p.state             = p.part + scan_words;
    p.race              = p.state + MAX_BS * STATE;
    p.correct_out       = correct_out.data_ptr<int64_t>();
    p.bonus_out         = bonus_out.data_ptr<int64_t>();
    p.new_seq_out       = new_seq_out.data_ptr<int64_t>();
    p.out_tokens        = out_tokens.data_ptr<int64_t>();
    p.out_tokens_stride = out_tokens.stride(0);
    p.cap_trim_out      = cap_trim_out.data_ptr<int>();
    p.commit_out        = commit_out.data_ptr<int>();
    if (gated_out) {
        check(*gated_out, target, at::kInt, "gated_out");
        p.gated_out = gated_out->data_ptr<int>();
    }
    if (block_out) {
        check(*block_out, target, at::kInt, "block_out");
        p.block_out = block_out->data_ptr<int>();
    }
    p.calls          = calls.data_ptr<int>();
    p.deadline_ticks = static_cast<uint64_t>(deadline_s * 1e8); // the 100 MHz realtime clock
    for (int q = 0; q < p.tp; ++q) {
        p.recv[q] = reinterpret_cast<void *>(peers.data_ptr<int64_t>()[q]);
    }
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA guard(target.device());
    const hipStream_t stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA(target.device().index()).stream();
    head_scan_kernel<<<dim3(p.nchunk, bs * slots), SCAN_THREADS, 0, stream>>>(p);
    C10_HIP_KERNEL_LAUNCH_CHECK();
    head_verdict_kernel<<<bs, VERDICT_THREADS, 0, stream>>>(p);
    C10_HIP_KERNEL_LAUNCH_CHECK();
    head_race_kernel<<<dim3(p.nchunk, bs), SCAN_THREADS, 0, stream>>>(p);
    C10_HIP_KERNEL_LAUNCH_CHECK();
    head_finish_kernel<<<bs, 64, 0, stream>>>(p);
    C10_HIP_KERNEL_LAUNCH_CHECK();
}

} // namespace

#ifndef TORCH_LIBRARY_EXPAND
#define TORCH_LIBRARY_EXPAND(NAME, MODULE) TORCH_LIBRARY(NAME, MODULE)
#endif

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, m) {
    m.def("verify_head(Tensor target, int col0, int start, int width, int vocab, float cap, Tensor draft, "
          "Tensor? draft_addr, "
          "Tensor cand, Tensor temps, Tensor sim_temps, Tensor? sim_len, Tensor sim_seed, Tensor key, Tensor offset, "
          "Tensor seq_lens, Tensor verify_lens, Tensor gate, Tensor(a!) part, Tensor(b!) correct_out, "
          "Tensor(c!) bonus_out, Tensor(d!) cap_trim_out, Tensor(e!) commit_out, Tensor(f!) new_seq_out, "
          "Tensor(g!) out_tokens, Tensor(h!)? gated_out, Tensor(i!)? block_out, Tensor(j!) calls, Tensor peers, "
          "int rank, float deadline_s) -> ()");
    m.impl("verify_head", at::kCUDA, &verify_head);
}
