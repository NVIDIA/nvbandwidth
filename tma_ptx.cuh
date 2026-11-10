/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef TMA_PTX_CUH_
#define TMA_PTX_CUH_

/**
 * Generic TMA / cp.async.bulk / mbarrier PTX primitives
 * Shared by TMA pointer kernels (kernels.cu) and fabric LE kernels (kernels_le_tma.cu)
 */

#include <cuda_runtime.h>
#include <cstdint>

#if __CUDA_ARCH__ >= 900
#include <cuda_awbarrier_primitives.h>

// Counted S→G bulk copy: dstMem (global), barMem (global completion counter), srcMem (shared).
extern "C" __device__ void __nv_ptx_builtin_ocg_cp_async_bulk_counted_global_shared(
    uint64_t dstMem, uint64_t barMem, uint64_t srcMem, uint32_t size, uint64_t desc);

// Async bulk copy: global -> shared
__device__ __forceinline__ void cp_async_bulk_global_to_shared(
    void *dest, void *src, __mbarrier_t *barrier, int size
) {
    uint32_t smem_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(dest));
    uint64_t gmem_ptr = static_cast<uint64_t>(__cvta_generic_to_global(src));
    uint32_t smem_barrier_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(barrier));
    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
        : : "r"(smem_ptr), "l"(gmem_ptr), "r"(size), "r"(smem_barrier_ptr) : "memory");
}

// Async bulk copy: shared -> global
__device__ __forceinline__ void cp_async_bulk_shared_to_global(void *dest, void *src, int size) {
    uint64_t dest_gmem_ptr = static_cast<uint64_t>(__cvta_generic_to_global(dest));
    uint32_t src_smem_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(src));
    asm volatile(
        "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;\n"
        : : "l"(dest_gmem_ptr), "r"(src_smem_ptr), "r"(size) : "memory");
}

__device__ __forceinline__ void cp_async_bulk_commit_group() {
    asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
}

__device__ __forceinline__ void cp_async_bulk_wait_all_read() {
    asm volatile("cp.async.bulk.wait_group.read 0;\n" ::: "memory");
}

// Keep at most 1 bulk group in flight — used for double-buffer pipelining where
// one group copies smem[prev]→gmem while the warp fills smem[cur] via ld_reduce.
__device__ __forceinline__ void cp_async_bulk_wait_group_read_1() {
    asm volatile("cp.async.bulk.wait_group.read 1;\n" ::: "memory");
}

__device__ __forceinline__ void barrier_red_arrive1_tx_relaxed(__mbarrier_t *barrier, uint32_t expected_tx_count) {
    asm volatile(
        "mbarrier.arrive.expect_tx.relaxed.cta.shared::cta.b64 _, [%0], %1;"
        : : "r"(static_cast<unsigned int>(__cvta_generic_to_shared(barrier))), "r"(expected_tx_count) : "memory");
}

__device__ __forceinline__ void barrier_expect_tx(__mbarrier_t *barrier, uint32_t txCount) {
    asm volatile(
        "mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [%0], %1;"
        : : "r"(static_cast<unsigned int>(__cvta_generic_to_shared(barrier))), "r"(txCount) : "memory");
}

__device__ __forceinline__ __mbarrier_token_t barrier_arrive1_tx(__mbarrier_t *barrier, uint32_t expected_tx_count) {
    __mbarrier_token_t token;
    asm volatile(
        "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 %0, [%1], %2;"
        : "=l"(token)
        : "r"(static_cast<unsigned int>(__cvta_generic_to_shared(barrier))), "r"(expected_tx_count) : "memory");
    return token;
}

__device__ __forceinline__ bool barrier_try_wait_token(__mbarrier_t *barrier, __mbarrier_token_t token) {
    int ready;
    asm volatile(
        "{\n\t.reg .pred p;\n\t"
        "mbarrier.try_wait.acquire.cta.shared::cta.b64 p, [%1], %2;\n\t"
        "selp.b32 %0, 1, 0, p;\n\t}"
        : "=r"(ready) : "r"(static_cast<unsigned int>(__cvta_generic_to_shared(barrier))), "l"(token) : "memory");
    return ready != 0;
}

__device__ __forceinline__ bool barrier_try_wait_parity_relaxed(__mbarrier_t *barrier, int parity) {
    int ready;
    asm volatile(
        "{\n\t.reg .pred p;\n\t"
        "mbarrier.try_wait.parity.relaxed.cta.shared::cta.b64 p, [%1], %2;\n\t"
        "selp.b32 %0, 1, 0, p;\n\t}"
        : "=r"(ready) : "r"(static_cast<unsigned int>(__cvta_generic_to_shared(barrier))), "r"(parity) : "memory");
    return ready != 0;
}

#endif  // __CUDA_ARCH__ >= 900

#endif  // TMA_PTX_CUH_
