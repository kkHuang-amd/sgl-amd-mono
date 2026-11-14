// SPDX-License-Identifier: Apache-2.0
// The index sources' top-512 rows swapped across the TP ranks when each rank selects only its own requests' rows
// (spec.py's _select under SGLANG_AMD_MONO_DSV41_INDEX_SPLIT=1): every rank pushes its rows of SGLang's sparse raw
// and page indices into every peer's buffer, tags them, waits for the peers' tags and copies their rows in. A block
// owns a fixed slice of each rank's rows and its own call counter, so the ranks' blocks pair up without a grid sync;
// the call's parity keeps a rank that runs one call ahead off the buffer a slower peer still reads.
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

using uint32x4 = uint32_t __attribute__((ext_vector_type(4)));

constexpr int MAX_TP   = 8;
constexpr int MAX_ROWS = 48;
constexpr int TOPK     = 512;
constexpr int VECS     = TOPK / 4; // a row's 16 B vectors in one of the two arrays
constexpr int BLOCKS   = 16;
constexpr int THREADS  = 256;
// a rank's buffer: [parity 2][source MAX_TP][row MAX_ROWS][array 2][TOPK] int32, then [parity 2][source][BLOCKS]
// tag packets of 16 B (exchange.py's EXCHANGE_BYTES)
constexpr int64_t DATA_VECS = int64_t(2) * MAX_TP * MAX_ROWS * 2 * VECS;

struct Params {
    int32_t  *raw, *page; // [>= rows, TOPK] SGLang's sparse indices, this rank's rows already selected
    int64_t   raw_stride, page_stride;
    int       lo[MAX_TP], hi[MAX_TP]; // every rank's rows
    int       rank, tp;
    int      *calls; // [BLOCKS]
    void     *recv[MAX_TP];
    uint64_t  deadline_ticks;
};


__device__ inline void store_nt(uint32x4 *ptr, uint32x4 val) {
    asm volatile("global_store_dwordx4 %0, %1, off sc0 sc1 nt\n\ts_nop 1" ::"v"(ptr), "v"(val) : "memory");
}

__device__ inline uint32x4 load_nt(const uint32x4 *ptr) {
    uint32x4 val;
    asm volatile("global_load_dwordx4 %0, %1, off sc0 sc1 nt\n\ts_waitcnt vmcnt(0)" : "=v"(val) : "v"(ptr) : "memory");
    return val;
}

// a peer's tag of this call; past the deadline the launch traps, stopping this rank's step
__device__ inline void poll_tag(const uint32x4 *src, uint32_t tag, uint64_t deadline) {
    uint64_t t0 = 0;
    for (int n = 1;; ++n) {
        if (load_nt(src).w == tag) {
            return;
        }
        __builtin_amdgcn_s_sleep(1);
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

__device__ inline int64_t data_vec(uint32_t parity, int src, int row, int array, int vec) {
    return (((int64_t(parity) * MAX_TP + src) * MAX_ROWS + row) * 2 + array) * VECS + vec;
}

// the slice [a, b) of n vectors block i owns
__device__ inline void slice(int n, int i, int &a, int &b) {
    a = int(int64_t(n) * i / BLOCKS);
    b = int(int64_t(n) * (i + 1) / BLOCKS);
}

__global__ __launch_bounds__(THREADS) void index_exchange_kernel(const Params p) {
    const int i = blockIdx.x;
    uint32_t  call = static_cast<uint32_t>(p.calls[i]) + 1;
    call           = call ? call : 1;
    const uint32_t parity = call & 1u;

    // 1. this rank's rows to every peer
    const int own = p.hi[p.rank] - p.lo[p.rank];
    int       a, b;
    slice(own * 2 * VECS, i, a, b);
    for (int v = a + threadIdx.x; v < b; v += THREADS) {
        const int row = v / (2 * VECS), array = v / VECS % 2, vec = v % VECS;
        const int32_t *src = array ? p.page + (p.lo[p.rank] + row) * p.page_stride
                                   : p.raw + (p.lo[p.rank] + row) * p.raw_stride;
        const uint32x4 val = reinterpret_cast<const uint32x4 *>(src)[vec];
        for (int q = 0; q < p.tp; ++q) {
            if (q != p.rank) {
                store_nt(static_cast<uint32x4 *>(p.recv[q]) + data_vec(parity, p.rank, row, array, vec), val);
            }
        }
    }
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    __syncthreads();

    // 2. the tags: this block's slice has landed on every peer; 3. the peers' tags
    uint32x4 *tags = nullptr;
    if (threadIdx.x < p.tp && threadIdx.x != p.rank) {
        const int q = threadIdx.x;
        tags        = static_cast<uint32x4 *>(p.recv[q]) + DATA_VECS;
        store_nt(tags + (int64_t(parity) * MAX_TP + p.rank) * BLOCKS + i, uint32x4{0u, 0u, 0u, call});
        const uint32x4 *mine = static_cast<const uint32x4 *>(p.recv[p.rank]) + DATA_VECS;
        poll_tag(mine + (int64_t(parity) * MAX_TP + q) * BLOCKS + i, call, p.deadline_ticks);
    }
    __syncthreads();

    // 4. the peers' rows in (the buffer is uncached: plain loads see the peers' stores)
    const uint32x4 *mine = static_cast<const uint32x4 *>(p.recv[p.rank]);
    for (int q = 0; q < p.tp; ++q) {
        if (q == p.rank) {
            continue;
        }
        slice((p.hi[q] - p.lo[q]) * 2 * VECS, i, a, b);
        for (int v = a + threadIdx.x; v < b; v += THREADS) {
            const int row = v / (2 * VECS), array = v / VECS % 2, vec = v % VECS;
            int32_t *dst = array ? p.page + (p.lo[q] + row) * p.page_stride : p.raw + (p.lo[q] + row) * p.raw_stride;
            reinterpret_cast<uint32x4 *>(dst)[vec] = mine[data_vec(parity, q, row, array, vec)];
        }
    }
    if (threadIdx.x == 0) {
        p.calls[i] = static_cast<int>(call);
    }
}

// the check's row compare: a / b [rows, TOPK] int32 (each row sorted, as a REINDEX layer's top-k is unordered);
// rows in [own_lo, own_hi) count as own, the rest as peers'
__global__ __launch_bounds__(THREADS) void index_compare_kernel(const int32_t *a, const int32_t *b, int64_t a_stride,
                                                                int64_t b_stride, int rows, int own_lo, int own_hi,
                                                                int64_t *check) {
    __shared__ int bad_own_vecs, bad_peer_vecs, all_vecs;
    if (threadIdx.x == 0) {
        bad_own_vecs = bad_peer_vecs = all_vecs = 0;
    }
    __syncthreads();
    const int i = blockIdx.x;
    int       lo, hi, bad_own = 0, bad_peer = 0, seen = 0;
    slice(rows * VECS, i, lo, hi);
    for (int v = lo + threadIdx.x; v < hi; v += THREADS) {
        const int      row = v / VECS, vec = v % VECS;
        const uint32x4 x   = reinterpret_cast<const uint32x4 *>(a + row * a_stride)[vec];
        const uint32x4 y   = reinterpret_cast<const uint32x4 *>(b + row * b_stride)[vec];
        const int      d   = (x.x != y.x) | (x.y != y.y) | (x.z != y.z) | (x.w != y.w);
        (row >= own_lo && row < own_hi ? bad_own : bad_peer) += d;
        ++seen;
    }
    atomicAdd(&bad_own_vecs, bad_own);
    atomicAdd(&bad_peer_vecs, bad_peer);
    atomicAdd(&all_vecs, seen);
    __syncthreads();
    if (threadIdx.x == 0) {
        volatile int64_t *c = check; // host memory, this block's words only
        c[i] += bad_own_vecs;
        c[BLOCKS + i] += bad_peer_vecs;
        c[2 * BLOCKS + i] += all_vecs;
    }
}

void index_compare(const at::Tensor &a, const at::Tensor &b, int64_t own_lo, int64_t own_hi, at::Tensor &check) {
    for (const at::Tensor *t : {&a, &b}) {
        TORCH_CHECK(t->is_cuda() && t->scalar_type() == at::kInt && t->dim() == 2 && t->size(1) == TOPK &&
                        t->stride(1) == 1 && t->stride(0) % 4 == 0 &&
                        reinterpret_cast<uintptr_t>(t->data_ptr()) % 16 == 0,
                    "a / b: [rows, ", TOPK, "] int32, 16 B aligned rows");
    }
    TORCH_CHECK(a.size(0) == b.size(0), "a / b: the same rows");
    TORCH_CHECK(check.is_cpu() && check.is_pinned() && check.scalar_type() == at::kLong && check.numel() >= 3 * BLOCKS,
                "check: [", 3 * BLOCKS, "] int64, pinned host memory");
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA guard(a.device());
    const hipStream_t stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA(a.device().index()).stream();
    index_compare_kernel<<<BLOCKS, THREADS, 0, stream>>>(a.data_ptr<int32_t>(), b.data_ptr<int32_t>(), a.stride(0),
                                                         b.stride(0), static_cast<int>(a.size(0)),
                                                         static_cast<int>(own_lo), static_cast<int>(own_hi),
                                                         check.data_ptr<int64_t>());
    C10_HIP_KERNEL_LAUNCH_CHECK();
}

// raw / page [>= rows, TOPK] int32 (16 B aligned rows), lo / hi every rank's rows (host), calls [BLOCKS] int32,
// peers [tp] int64 on the host: every rank's exchange buffer
void index_exchange(at::Tensor &raw, at::Tensor &page, const std::vector<int64_t> &lo, const std::vector<int64_t> &hi,
                    at::Tensor &calls, const at::Tensor &peers, int64_t rank, double deadline_s) {
    const int64_t tp = peers.numel();
    TORCH_CHECK(peers.device().is_cpu() && peers.scalar_type() == at::kLong && tp >= 2 && tp <= MAX_TP &&
                    rank >= 0 && rank < tp,
                "peers: every rank's exchange buffer (int64, host)");
    TORCH_CHECK(int64_t(lo.size()) == tp && int64_t(hi.size()) == tp, "lo / hi: a range a rank");
    for (const at::Tensor *t : {&raw, &page}) {
        TORCH_CHECK(t->is_cuda() && t->scalar_type() == at::kInt && t->dim() == 2 && t->size(1) == TOPK &&
                        t->stride(1) == 1 && t->stride(0) % 4 == 0 &&
                        reinterpret_cast<uintptr_t>(t->data_ptr()) % 16 == 0,
                    "raw / page: [rows, ", TOPK, "] int32, 16 B aligned rows");
    }
    TORCH_CHECK(calls.is_cuda() && calls.scalar_type() == at::kInt && calls.numel() >= BLOCKS, "calls [", BLOCKS,
                "] int32");
    Params p{};
    p.raw         = raw.data_ptr<int32_t>();
    p.page        = page.data_ptr<int32_t>();
    p.raw_stride  = raw.stride(0);
    p.page_stride = page.stride(0);
    for (int q = 0; q < tp; ++q) {
        TORCH_CHECK(0 <= lo[q] && lo[q] <= hi[q] && hi[q] - lo[q] <= MAX_ROWS && hi[q] <= raw.size(0) &&
                        hi[q] <= page.size(0),
                    "rank ", q, "'s rows [", lo[q], ", ", hi[q], ")");
        p.lo[q] = static_cast<int>(lo[q]);
        p.hi[q] = static_cast<int>(hi[q]);
        p.recv[q] = reinterpret_cast<void *>(peers.data_ptr<int64_t>()[q]);
    }
    p.rank           = static_cast<int>(rank);
    p.tp             = static_cast<int>(tp);
    p.calls          = calls.data_ptr<int>();
    p.deadline_ticks = static_cast<uint64_t>(deadline_s * 1e8); // the 100 MHz realtime clock
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA guard(raw.device());
    const hipStream_t stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA(raw.device().index()).stream();
    index_exchange_kernel<<<BLOCKS, THREADS, 0, stream>>>(p);
    C10_HIP_KERNEL_LAUNCH_CHECK();
}

} // namespace

#ifndef TORCH_LIBRARY_EXPAND
#define TORCH_LIBRARY_EXPAND(NAME, MODULE) TORCH_LIBRARY(NAME, MODULE)
#endif

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, m) {
    m.def("index_exchange(Tensor(a!) raw, Tensor(b!) page, int[] lo, int[] hi, Tensor(c!) calls, Tensor peers, "
          "int rank, float deadline_s) -> ()");
    m.impl("index_exchange", at::kCUDA, &index_exchange);
    m.def("index_compare(Tensor a, Tensor b, int own_lo, int own_hi, Tensor(a!) check) -> ()");
    m.impl("index_compare", at::kCUDA, &index_compare);
}
