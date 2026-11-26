// SPDX-License-Identifier: Apache-2.0
// Adapted from AKLR (https://github.com/AMD-AGI/AKLR, MIT, Copyright (C) 2025-2026 Advanced Micro Devices, Inc.).
// A sampled DSpark draft's Markov step on this rank's vocab shard as SGLang's MarkovSampleStep: the bias GEMV, the
// Philox Gumbel-max scores and the shard's bests (step), then the ranks' bests swapped through peer buffers (finish).
#include <climits>
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

constexpr int DIM           = 256;
constexpr int THREADS       = 256;
constexpr int ROW_LANES     = 16; // a vocab row's DIM by 16 lanes, 16 values each
constexpr int ROWS_PER_PASS = THREADS / ROW_LANES;
constexpr int PASSES        = 4;
constexpr int BLOCK_ROWS    = ROWS_PER_PASS * PASSES;
constexpr int FINISH        = 512;
constexpr int MAX_SEQ       = 8;
constexpr int MAX_TP        = 8;
constexpr int SLOTS         = 16; // a sequence's packets: two halves by call parity, a rank's slot in each
// the exchange buffer: every sequence's slots, 16 B each (markov.py's EXCHANGE_BYTES)

struct Params {
    const bfloat16 *logits; // this step's base logits, sequence s's row at logits + s * logits_stride
    int64_t         logits_stride;
    const bfloat16 *embed; // markov_w1 [vocab, DIM]
    const bfloat16 *head;  // this rank's markov_w2 rows [width, DIM]
    const int64_t  *anchor; // sequence s's at anchor[s * anchor_stride]
    int64_t         anchor_stride;
    float          *step_logits; // sequence s's row of this step at + s * sl_stride (null: none kept)
    int64_t         sl_stride;
    const float    *temps; // [nseq]
    const int64_t  *offset;
    uint32_t        key_lo, key_hi;
    int             vocab_start, width, padded, step, nseq, blocks, rank, tp, bump_offset;
    float          *part; // [nseq, blocks, 2]
    int64_t        *out;  // sequence s's token of this step at out[s * out_stride + step]
    int64_t         out_stride;
    int64_t        *tok;   // [nseq] the previous step's token
    int            *calls; // [nseq]
    void           *recv[MAX_TP];
    uint64_t        deadline_ticks;
};

__device__ inline void keep_best(float &v, int &i, float v2, int i2) {
    if (v2 > v || (v2 == v && i2 < i)) {
        v = v2, i = i2;
    }
}

__device__ inline float round_bf16(float x) { return static_cast<float>(static_cast<bfloat16>(x)); }

__device__ inline void load16f(const bfloat16 *src, float (&v)[16]) {
    bfloat16 b[16];
    *reinterpret_cast<uint4 *>(b)     = *reinterpret_cast<const uint4 *>(src);
    *reinterpret_cast<uint4 *>(b + 8) = *reinterpret_cast<const uint4 *>(src + 8);
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        v[i] = static_cast<float>(b[i]);
    }
}

template <int WIDTH> __device__ inline float lane_sum(float v) {
#pragma unroll
    for (int m = WIDTH / 2; m > 0; m /= 2) {
        v += __shfl_xor(v, m, WIDTH);
    }
    return v;
}

// Philox4x32-10 (tl.philox's): the word k of the four
__device__ inline uint32_t philox_word(uint32_t k0, uint32_t k1, uint32_t c0, uint32_t c1, uint32_t c2, uint32_t c3,
                                       int k) {
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
    return k == 0 ? c0 : k == 1 ? c1 : k == 2 ? c2 : c3;
}

__device__ inline float select_key(const Params &p, int seq, int v, float value) {
    const float t = p.temps[seq];
    float       key = value;
    if (t != 0.0f) {
        const uint32_t g = static_cast<uint32_t>(p.vocab_start + v);
        const uint32_t w = philox_word(p.key_lo, p.key_hi, g >> 2, static_cast<uint32_t>(seq),
                                       static_cast<uint32_t>(p.step), static_cast<uint32_t>(p.offset[0]),
                                       static_cast<int>(g & 3u));
        const float u = (static_cast<float>(w) + 0.5f) * 2.3283064365386963e-10f;
        key           = value / t - logf(-log1pf(-u));
    }
    return key != key ? INFINITY : key;
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

// a block BLOCK_ROWS rows of the shard, held in registers, against every sequence in turn
__global__ __launch_bounds__(THREADS) void markov_sample_step_kernel(const Params p) {
    __shared__ float sv[BLOCK_ROWS];
    const int        lane = threadIdx.x % ROW_LANES, grp = threadIdx.x / ROW_LANES;
    float            e[16], h[PASSES][16];
    const auto prev_of = [&](int seq) { return p.step == 0 ? p.anchor[seq * p.anchor_stride] : p.tok[seq]; };
    load16f(p.embed + prev_of(0) * DIM + lane * 16, e);
#pragma unroll
    for (int ps = 0; ps < PASSES; ++ps) {
        const int v = blockIdx.x * BLOCK_ROWS + ps * ROWS_PER_PASS + grp;
        if (v < p.width) {
            load16f(p.head + static_cast<int64_t>(v) * DIM + lane * 16, h[ps]);
        }
    }
    for (int seq = 0; seq < p.nseq; ++seq) {
        if (seq > 0) {
            load16f(p.embed + prev_of(seq) * DIM + lane * 16, e);
        }
#pragma unroll
        for (int ps = 0; ps < PASSES; ++ps) {
            const int v   = blockIdx.x * BLOCK_ROWS + ps * ROWS_PER_PASS + grp;
            float     dot = 0.0f;
            if (v < p.width) {
#pragma unroll
                for (int i = 0; i < 16; ++i) {
                    dot += e[i] * h[ps][i];
                }
            }
            dot = lane_sum<ROW_LANES>(dot);
            if (lane == 0) {
                sv[ps * ROWS_PER_PASS + grp] = dot;
            }
        }
        __syncthreads();
        // the first wave: a row a lane
        if (threadIdx.x < BLOCK_ROWS) {
            const int v    = blockIdx.x * BLOCK_ROWS + threadIdx.x;
            float     best = -INFINITY;
            int       bi   = INT_MAX;
            if (v < p.padded) {
                const float bias  = v < p.width ? round_bf16(sv[threadIdx.x]) : 0.0f;
                const float value = static_cast<float>(p.logits[seq * p.logits_stride + v]) + bias;
                if (p.step_logits != nullptr) {
                    p.step_logits[seq * p.sl_stride + v] = value;
                }
                if (v < p.width) {
                    best = select_key(p, seq, v, value);
                    bi   = v;
                }
            }
#pragma unroll
            for (int m = BLOCK_ROWS / 2; m > 0; m /= 2) {
                keep_best(best, bi, __shfl_xor(best, m, BLOCK_ROWS), __shfl_xor(bi, m, BLOCK_ROWS));
            }
            if (threadIdx.x == 0) {
                float *part = p.part + 2 * (static_cast<int64_t>(seq) * p.blocks + blockIdx.x);
                part[0]     = best;
                part[1]     = __int_as_float(bi);
            }
        }
        __syncthreads(); // sv free for the next sequence
    }
}

// a block a sequence: the shard's best, then the ranks' swapped
__global__ __launch_bounds__(FINISH) void markov_sample_finish_kernel(const Params p) {
    __shared__ float sv[FINISH];
    __shared__ int   si[FINISH];
    const int        seq  = blockIdx.x;
    const float     *part = p.part + 2 * static_cast<int64_t>(seq) * p.blocks;
    float            best = -INFINITY;
    int              bi   = INT_MAX;
    for (int b = threadIdx.x; b < p.blocks; b += FINISH) {
        keep_best(best, bi, part[2 * b], __float_as_int(part[2 * b + 1]));
    }
    sv[threadIdx.x] = best, si[threadIdx.x] = bi;
    __syncthreads();
    for (int half = FINISH / 2; half > 0; half /= 2) {
        if (threadIdx.x < half) {
            keep_best(sv[threadIdx.x], si[threadIdx.x], sv[threadIdx.x + half], si[threadIdx.x + half]);
        }
        __syncthreads();
    }
    if (threadIdx.x != 0) {
        return;
    }
    uint32_t call = static_cast<uint32_t>(p.calls[seq]) + 1;
    call          = call ? call : 1;
    const auto slot = [&](int src) { return seq * SLOTS + static_cast<int>(call % 2) * 8 + src; };
    const uint32x4 mine{__builtin_bit_cast(uint32_t, sv[0]), static_cast<uint32_t>(si[0] + p.vocab_start), 0, call};
    for (int r = 0; r < p.tp; ++r) {
        store_nt(static_cast<uint32x4 *>(p.recv[r]) + slot(p.rank), mine);
    }
    float   top = -INFINITY;
    int64_t tok = INT_MAX;
    for (int r = 0; r < p.tp; ++r) { // the best key, the lowest index (rank) on ties
        const uint32x4 pk = poll_tag(static_cast<const uint32x4 *>(p.recv[p.rank]) + slot(r), call, p.deadline_ticks);
        const float    v  = __builtin_bit_cast(float, pk.x);
        if (r == 0 || v > top) {
            top = v, tok = static_cast<int64_t>(pk.y);
        }
    }
    p.out[seq * p.out_stride + p.step] = tok;
    p.tok[seq]                         = tok;
    p.calls[seq]                       = static_cast<int>(call);
    if (p.bump_offset && seq == 0) { // the block's last step: every step kernel read the counter
        const_cast<int64_t *>(p.offset)[0] += 1;
    }
}

// ---------------------------------------------------------------- the whole block in one launch
// SGLANG_AMD_MONO_DSV41_MARKOV_PERSIST: every step of a draft block in one launch of PBLOCKS persistent blocks (one a CU,
// so all resident). A block holds its G row groups of the W2 shard in registers for every step; a step's bests go
// block -> partial (this GPU) -> the sequence's leader block -> every rank's slot -> every block, as tagged 16 B packets
// (data and tag in one store; the tag is the launch's call and the step), with no grid barrier and no last-block count.
constexpr int PBLOCKS = 256;

struct BlockParams {
    const bfloat16 *logits; // [nseq, gamma, padded]: sequence s's step j row at + s * seq_stride + j * step_stride
    int64_t         seq_stride, step_stride;
    const bfloat16 *embed;
    const bfloat16 *head;
    const int64_t  *anchor;
    int64_t         anchor_stride;
    float          *step_logits; // same layout as logits, f32 (null: none kept)
    int64_t         sl_seq_stride, sl_step_stride;
    const float    *temps;
    int64_t        *offset;
    uint32_t        key_lo, key_hi;
    int             vocab_start, width, padded, gamma, nseq, blocks, rank, tp;
    uint32x4       *partials; // [2][MAX_SEQ][PBLOCKS] this GPU's block bests
    int64_t        *out;      // sequence s's step j token at out[s * out_stride + j]
    int64_t         out_stride;
    int            *calls; // [1] the launches so far
    void           *recv[MAX_TP]; // every rank's slots [2][MAX_SEQ][MAX_TP]
    uint64_t        deadline_ticks;
};

__device__ inline float block_key(const BlockParams &p, int seq, int step, int v, float value) {
    const float t = p.temps[seq];
    float       key = value;
    if (t != 0.0f) {
        const uint32_t g = static_cast<uint32_t>(p.vocab_start + v);
        const uint32_t w = philox_word(p.key_lo, p.key_hi, g >> 2, static_cast<uint32_t>(seq),
                                       static_cast<uint32_t>(step), static_cast<uint32_t>(p.offset[0]),
                                       static_cast<int>(g & 3u));
        const float u = (static_cast<float>(w) + 0.5f) * 2.3283064365386963e-10f;
        key           = value / t - logf(-log1pf(-u));
    }
    return key != key ? INFINITY : key;
}

template <int G> __global__ __launch_bounds__(THREADS) void markov_sample_block_kernel(const BlockParams p) {
    __shared__ float   sv[BLOCK_ROWS];
    __shared__ float   rv[PBLOCKS];
    __shared__ int     ri[PBLOCKS];
    __shared__ int64_t stok[MAX_SEQ];
    const int          lane = threadIdx.x % ROW_LANES, grp = threadIdx.x / ROW_LANES;
    float              e[16], h[G][PASSES][16];
#pragma unroll
    for (int g = 0; g < G; ++g) {
        const int group = blockIdx.x + g * PBLOCKS;
#pragma unroll
        for (int ps = 0; ps < PASSES; ++ps) {
            const int v = group * BLOCK_ROWS + ps * ROWS_PER_PASS + grp;
            if (group < p.blocks && v < p.width) {
                load16f(p.head + static_cast<int64_t>(v) * DIM + lane * 16, h[g][ps]);
            }
        }
    }
    const uint32_t call = static_cast<uint32_t>(p.calls[0]) + 1u;
    for (int j = 0; j < p.gamma; ++j) {
        const uint32_t tag = call * 8u + static_cast<uint32_t>(j) + 1u;
        const int      par = j & 1;
        for (int seq = 0; seq < p.nseq; ++seq) {
            const int64_t prev = j == 0 ? p.anchor[seq * p.anchor_stride] : stok[seq];
            load16f(p.embed + prev * DIM + lane * 16, e);
            float best = -INFINITY;
            int   bi   = INT_MAX;
#pragma unroll
            for (int g = 0; g < G; ++g) {
                const int group = blockIdx.x + g * PBLOCKS;
                if (group >= p.blocks) {
                    continue; // block-uniform
                }
#pragma unroll
                for (int ps = 0; ps < PASSES; ++ps) {
                    const int v   = group * BLOCK_ROWS + ps * ROWS_PER_PASS + grp;
                    float     dot = 0.0f;
                    if (v < p.width) {
#pragma unroll
                        for (int i = 0; i < 16; ++i) {
                            dot += e[i] * h[g][ps][i];
                        }
                    }
                    dot = lane_sum<ROW_LANES>(dot);
                    if (lane == 0) {
                        sv[ps * ROWS_PER_PASS + grp] = dot;
                    }
                }
                __syncthreads();
                if (threadIdx.x < BLOCK_ROWS) {
                    const int v = group * BLOCK_ROWS + threadIdx.x;
                    if (v < p.padded) {
                        const float bias  = v < p.width ? round_bf16(sv[threadIdx.x]) : 0.0f;
                        const float value = static_cast<float>(p.logits[seq * p.seq_stride + j * p.step_stride + v]) + bias;
                        if (p.step_logits != nullptr) {
                            p.step_logits[seq * p.sl_seq_stride + j * p.sl_step_stride + v] = value;
                        }
                        if (v < p.width) {
                            keep_best(best, bi, block_key(p, seq, j, v, value), v);
                        }
                    }
                }
                __syncthreads(); // sv free
            }
            if (threadIdx.x < BLOCK_ROWS) {
#pragma unroll
                for (int m = BLOCK_ROWS / 2; m > 0; m /= 2) {
                    keep_best(best, bi, __shfl_xor(best, m, BLOCK_ROWS), __shfl_xor(bi, m, BLOCK_ROWS));
                }
                if (threadIdx.x == 0) {
                    const uint32x4 pk{__builtin_bit_cast(uint32_t, best), static_cast<uint32_t>(bi), 0, tag};
                    store_nt(p.partials + (par * MAX_SEQ + seq) * PBLOCKS + blockIdx.x, pk);
                }
            }
        }
        // the leader of each sequence: this GPU's best of all blocks, to every rank
        for (int seq = 0; seq < p.nseq; ++seq) {
            if (static_cast<int>(blockIdx.x) != seq % PBLOCKS) {
                continue; // block-uniform
            }
            const uint32x4 pk = poll_tag(p.partials + (par * MAX_SEQ + seq) * PBLOCKS + threadIdx.x % PBLOCKS, tag,
                                         p.deadline_ticks);
            if (threadIdx.x < PBLOCKS) {
                rv[threadIdx.x] = __builtin_bit_cast(float, pk.x), ri[threadIdx.x] = static_cast<int>(pk.y);
            }
            __syncthreads();
            for (int half = PBLOCKS / 2; half > 0; half /= 2) {
                if (threadIdx.x < half) {
                    keep_best(rv[threadIdx.x], ri[threadIdx.x], rv[threadIdx.x + half], ri[threadIdx.x + half]);
                }
                __syncthreads();
            }
            if (threadIdx.x == 0) {
                const uint32x4 mine{__builtin_bit_cast(uint32_t, rv[0]), static_cast<uint32_t>(ri[0] + p.vocab_start), 0,
                                    tag};
                for (int r = 0; r < p.tp; ++r) {
                    store_nt(static_cast<uint32x4 *>(p.recv[r]) + (par * MAX_SEQ + seq) * MAX_TP + p.rank, mine);
                }
            }
            __syncthreads();
        }
        // every block: the ranks' bests of every sequence, the best key and the lowest rank on ties
        if (threadIdx.x < p.nseq) {
            const int seq = threadIdx.x;
            float     top = -INFINITY;
            int64_t   tok = INT_MAX;
            for (int r = 0; r < p.tp; ++r) {
                const uint32x4 pk = poll_tag(static_cast<const uint32x4 *>(p.recv[p.rank]) + (par * MAX_SEQ + seq) * MAX_TP + r,
                                             tag, p.deadline_ticks);
                const float v = __builtin_bit_cast(float, pk.x);
                if (r == 0 || v > top) {
                    top = v, tok = static_cast<int64_t>(pk.y);
                }
            }
            stok[seq] = tok;
            if (blockIdx.x == 0) {
                p.out[seq * p.out_stride + j] = tok;
            }
        }
        __syncthreads();
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) { // every block read the call and the counter at its first step
        p.calls[0] = static_cast<int>(call);
        p.offset[0] += 1;
    }
}

void check(const at::Tensor &t, const at::Tensor &ref, at::ScalarType dtype, const char *name) {
    TORCH_CHECK(t.device() == ref.device() && t.scalar_type() == dtype, name, " must be ", dtype, " on ",
                ref.device());
    TORCH_CHECK((t.stride(-1) == 1 || t.numel() == 1) && reinterpret_cast<uintptr_t>(t.data_ptr()) % 4 == 0, name,
                " must have unit inner stride");
}

// logits [nseq, padded] bf16 (this step's rows), embed [vocab, 256], head [width, 256] bf16, anchor / tok [nseq]
// int64, step_logits [nseq, >= padded] f32 rows or none, temps [nseq] f32, offset [1] int64, part [nseq, blocks, 2]
// f32 scratch, out int64 (sequence s's step token at s * out_stride + step), calls [nseq] int32, peers [tp] int64 on
// the host: every rank's exchange buffer
void markov_sample_step(const at::Tensor &logits, const at::Tensor &embed, const at::Tensor &head,
                        const at::Tensor &anchor, const std::optional<at::Tensor> &step_logits,
                        const at::Tensor &temps, const at::Tensor &offset, int64_t key, int64_t vocab_start,
                        int64_t step, at::Tensor &part, at::Tensor &out, int64_t out_stride, at::Tensor &tok,
                        at::Tensor &calls, const at::Tensor &peers, int64_t rank, bool bump_offset,
                        double deadline_s) {
    const int64_t nseq  = logits.size(0);
    const int64_t width = head.size(0);
    TORCH_CHECK(logits.dim() == 2 && nseq >= 1 && nseq <= MAX_SEQ, "logits [1 .. ", MAX_SEQ, " sequences, padded]");
    TORCH_CHECK(head.dim() == 2 && head.size(1) == DIM && head.is_contiguous() && embed.dim() == 2 &&
                    embed.size(1) == DIM && embed.is_contiguous() && width <= logits.size(1),
                "head [width, 256], embed [vocab, 256] contiguous");
    check(logits, logits, at::kBFloat16, "logits");
    check(embed, logits, at::kBFloat16, "embed");
    check(head, logits, at::kBFloat16, "head");
    TORCH_CHECK(anchor.device() == logits.device() && anchor.scalar_type() == at::kLong && anchor.dim() == 1 &&
                    anchor.size(0) >= nseq,
                "anchor [nseq] int64 (any stride)");
    check(temps, logits, at::kFloat, "temps");
    check(offset, logits, at::kLong, "offset");
    check(part, logits, at::kFloat, "part");
    check(out, logits, at::kLong, "out");
    check(tok, logits, at::kLong, "tok");
    check(calls, logits, at::kInt, "calls");
    TORCH_CHECK(peers.device().is_cpu() && peers.scalar_type() == at::kLong && peers.numel() >= 1 &&
                    peers.numel() <= MAX_TP,
                "peers: every rank's exchange buffer (int64, host)");
    Params p{};
    p.logits        = static_cast<const bfloat16 *>(logits.data_ptr());
    p.logits_stride = logits.stride(0);
    p.embed         = static_cast<const bfloat16 *>(embed.data_ptr());
    p.head          = static_cast<const bfloat16 *>(head.data_ptr());
    p.anchor        = anchor.data_ptr<int64_t>();
    p.anchor_stride = anchor.stride(0);
    if (step_logits) {
        check(*step_logits, logits, at::kFloat, "step_logits");
        TORCH_CHECK(step_logits->dim() == 2 && step_logits->size(0) >= nseq && step_logits->size(1) >= logits.size(1),
                    "step_logits [nseq, >= padded]");
        p.step_logits = step_logits->data_ptr<float>();
        p.sl_stride   = step_logits->stride(0);
    }
    p.temps       = temps.data_ptr<float>();
    p.offset      = offset.data_ptr<int64_t>();
    p.key_lo      = static_cast<uint32_t>(static_cast<uint64_t>(key) & 0xFFFFFFFFull);
    p.key_hi      = static_cast<uint32_t>(static_cast<uint64_t>(key) >> 32);
    p.vocab_start = static_cast<int>(vocab_start);
    p.width       = static_cast<int>(width);
    p.padded      = static_cast<int>(logits.size(1));
    p.step        = static_cast<int>(step);
    p.nseq        = static_cast<int>(nseq);
    p.blocks      = static_cast<int>((logits.size(1) + BLOCK_ROWS - 1) / BLOCK_ROWS);
    TORCH_CHECK(part.numel() >= 2 * nseq * p.blocks, "part [nseq, blocks, 2]");
    p.rank           = static_cast<int>(rank);
    p.tp             = static_cast<int>(peers.numel());
    p.bump_offset    = bump_offset ? 1 : 0;
    p.part           = part.data_ptr<float>();
    p.out            = out.data_ptr<int64_t>();
    p.out_stride     = out_stride;
    p.tok            = tok.data_ptr<int64_t>();
    p.calls          = calls.data_ptr<int>();
    p.deadline_ticks = static_cast<uint64_t>(deadline_s * 1e8); // the 100 MHz realtime clock
    for (int r = 0; r < p.tp; ++r) {
        p.recv[r] = reinterpret_cast<void *>(peers.data_ptr<int64_t>()[r]);
    }
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA guard(logits.device());
    const hipStream_t stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA(logits.device().index()).stream();
    markov_sample_step_kernel<<<p.blocks, THREADS, 0, stream>>>(p);
    C10_HIP_KERNEL_LAUNCH_CHECK();
    markov_sample_finish_kernel<<<p.nseq, FINISH, 0, stream>>>(p);
    C10_HIP_KERNEL_LAUNCH_CHECK();
}

// base [nseq, gamma, padded] bf16, step_logits [nseq, gamma, >= padded] f32 or none, partials [2 * MAX_SEQ * PBLOCKS * 4]
// int32 (zeroed once), out int64 (s * out_stride + j), calls [1] int32, peers [tp] int64 on the host: every rank's
// block-exchange buffer (2 * MAX_SEQ * MAX_TP packets)
void markov_sample_block(const at::Tensor &base, const at::Tensor &embed, const at::Tensor &head,
                         const at::Tensor &anchor, const std::optional<at::Tensor> &step_logits,
                         const at::Tensor &temps, at::Tensor &offset, int64_t key, int64_t vocab_start,
                         at::Tensor &partials, at::Tensor &out, int64_t out_stride, at::Tensor &calls,
                         const at::Tensor &peers, int64_t rank, double deadline_s) {
    const int64_t nseq  = base.size(0);
    const int64_t width = head.size(0);
    TORCH_CHECK(base.dim() == 3 && nseq >= 1 && nseq <= MAX_SEQ && base.stride(2) == 1,
                "base [1 .. ", MAX_SEQ, " sequences, gamma, padded]");
    TORCH_CHECK(head.dim() == 2 && head.size(1) == DIM && head.is_contiguous() && embed.dim() == 2 &&
                    embed.size(1) == DIM && embed.is_contiguous() && width <= base.size(2),
                "head [width, 256], embed [vocab, 256] contiguous");
    check(embed, base, at::kBFloat16, "embed");
    check(head, base, at::kBFloat16, "head");
    TORCH_CHECK(base.scalar_type() == at::kBFloat16, "base bf16");
    TORCH_CHECK(anchor.device() == base.device() && anchor.scalar_type() == at::kLong && anchor.dim() == 1 &&
                    anchor.size(0) >= nseq,
                "anchor [nseq] int64 (any stride)");
    check(temps, base, at::kFloat, "temps");
    check(offset, base, at::kLong, "offset");
    check(out, base, at::kLong, "out");
    check(calls, base, at::kInt, "calls");
    TORCH_CHECK(partials.device() == base.device() && partials.scalar_type() == at::kInt &&
                    partials.numel() >= 2 * MAX_SEQ * PBLOCKS * 4,
                "partials [2 * MAX_SEQ * PBLOCKS * 4] int32");
    TORCH_CHECK(peers.device().is_cpu() && peers.scalar_type() == at::kLong && peers.numel() >= 1 &&
                    peers.numel() <= MAX_TP,
                "peers: every rank's block-exchange buffer (int64, host)");
    BlockParams p{};
    p.logits      = static_cast<const bfloat16 *>(base.data_ptr());
    p.seq_stride  = base.stride(0);
    p.step_stride = base.stride(1);
    p.embed       = static_cast<const bfloat16 *>(embed.data_ptr());
    p.head        = static_cast<const bfloat16 *>(head.data_ptr());
    p.anchor      = anchor.data_ptr<int64_t>();
    p.anchor_stride = anchor.stride(0);
    if (step_logits) {
        TORCH_CHECK(step_logits->scalar_type() == at::kFloat && step_logits->dim() == 3 &&
                        step_logits->size(0) >= nseq && step_logits->size(1) >= base.size(1) &&
                        step_logits->size(2) >= base.size(2) && step_logits->stride(2) == 1,
                    "step_logits [nseq, gamma, >= padded] f32");
        p.step_logits    = step_logits->data_ptr<float>();
        p.sl_seq_stride  = step_logits->stride(0);
        p.sl_step_stride = step_logits->stride(1);
    }
    p.temps       = temps.data_ptr<float>();
    p.offset      = offset.data_ptr<int64_t>();
    p.key_lo      = static_cast<uint32_t>(static_cast<uint64_t>(key) & 0xFFFFFFFFull);
    p.key_hi      = static_cast<uint32_t>(static_cast<uint64_t>(key) >> 32);
    p.vocab_start = static_cast<int>(vocab_start);
    p.width       = static_cast<int>(width);
    p.padded      = static_cast<int>(base.size(2));
    p.gamma       = static_cast<int>(base.size(1));
    p.nseq        = static_cast<int>(nseq);
    p.blocks      = static_cast<int>((base.size(2) + BLOCK_ROWS - 1) / BLOCK_ROWS);
    p.rank        = static_cast<int>(rank);
    p.tp          = static_cast<int>(peers.numel());
    p.partials    = reinterpret_cast<uint32x4 *>(partials.data_ptr<int>());
    p.out         = out.data_ptr<int64_t>();
    p.out_stride  = out_stride;
    p.calls       = calls.data_ptr<int>();
    p.deadline_ticks = static_cast<uint64_t>(deadline_s * 1e8);
    for (int r = 0; r < p.tp; ++r) {
        p.recv[r] = reinterpret_cast<void *>(peers.data_ptr<int64_t>()[r]);
    }
    const int groups = (p.blocks + PBLOCKS - 1) / PBLOCKS;
    TORCH_CHECK(groups <= 4, "a vocab shard of at most ", 4 * PBLOCKS * BLOCK_ROWS, " rows");
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA guard(base.device());
    const hipStream_t stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA(base.device().index()).stream();
    if (groups == 1) {
        markov_sample_block_kernel<1><<<PBLOCKS, THREADS, 0, stream>>>(p);
    } else if (groups == 2) {
        markov_sample_block_kernel<2><<<PBLOCKS, THREADS, 0, stream>>>(p);
    } else {
        markov_sample_block_kernel<4><<<PBLOCKS, THREADS, 0, stream>>>(p);
    }
    C10_HIP_KERNEL_LAUNCH_CHECK();
}

} // namespace

#ifndef TORCH_LIBRARY_EXPAND
#define TORCH_LIBRARY_EXPAND(NAME, MODULE) TORCH_LIBRARY(NAME, MODULE)
#endif

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, m) {
    m.def("markov_sample_step(Tensor logits, Tensor embed, Tensor head, Tensor anchor, Tensor? step_logits, "
          "Tensor temps, Tensor offset, int key, int vocab_start, int step, Tensor(a!) part, Tensor(b!) out, "
          "int out_stride, Tensor(c!) tok, Tensor(d!) calls, Tensor peers, int rank, bool bump_offset, "
          "float deadline_s) -> ()");
    m.impl("markov_sample_step", at::kCUDA, &markov_sample_step);
    m.def("markov_sample_block(Tensor base, Tensor embed, Tensor head, Tensor anchor, Tensor? step_logits, "
          "Tensor temps, Tensor(a!) offset, int key, int vocab_start, Tensor(b!) partials, Tensor(c!) out, "
          "int out_stride, Tensor(d!) calls, Tensor peers, int rank, float deadline_s) -> ()");
    m.impl("markov_sample_block", at::kCUDA, &markov_sample_block);
}
