/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef FABRIC_PTX_CUH_
#define FABRIC_PTX_CUH_

/**
 * Fabric-specific PTX (fabric.try_put, fabric.try_get, etc.).
 * TMA/cp.async.bulk/mbarrier helpers are in tma_ptx.cuh.
 */
#include "tma_ptx.cuh"

#if __CUDA_ARCH__ >= 900

// ============================================================================
// Shared Memory Pipe Structure (sm_90+)
// ============================================================================
template <int chunk_size_bytes, int pipe_len = 1>
struct SMEMPIPE {
    alignas(16) char buf[pipe_len][chunk_size_bytes];
};

// ============================================================================
// Handle Barrier: wrapper for fabric ops (kernels_le_tma uses this)
// ============================================================================

struct handle_barrier {
    alignas(8) __mbarrier_t bar;

    __device__ __forceinline__ void init(int arvCnt) {
        __mbarrier_init(&bar, arvCnt);
    }

    __device__ __forceinline__ void arrive_relaxed(int txCnt) {
        barrier_red_arrive1_tx_relaxed(&bar, (txCnt / 16));
    }

    // arrive_relaxed_bytes: raw byte count (used by fabric.try_get which reports in bytes)
    __device__ __forceinline__ void arrive_relaxed_bytes(int bytes) {
        barrier_red_arrive1_tx_relaxed(&bar, bytes);
    }

    __device__ __forceinline__ void wait_relaxed(bool phase) {
#pragma unroll 1
        while (!barrier_try_wait_parity_relaxed(&bar, phase)) {}
    }

    // Token-based (used by G→S bulk copy where arrival is signaled by a tx-count token).
    __device__ __forceinline__ __mbarrier_token_t arrive_tx(uint32_t bytes) {
        return barrier_arrive1_tx(&bar, bytes);
    }
    __device__ __forceinline__ void wait_token(__mbarrier_token_t token) {
        while (!barrier_try_wait_token(&bar, token)) {}
    }
};

// Overload: G→S bulk copy taking handle_barrier& instead of __mbarrier_t*.
__device__ __forceinline__ void cp_async_bulk_global_to_shared(
    void *dest, void *src, handle_barrier &bar, int size) {
    cp_async_bulk_global_to_shared(dest, src, &bar.bar, size);
}

// ============================================================================
// Fabric Put/Get Operations
// ============================================================================
#if __CUDA_ARCH__ >= 1000
__device__ __forceinline__ void fabric_try_put(uint32_t le, uint64_t offset, void *src, int size, handle_barrier &bar) {
    uint32_t src_smem_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(src));
    uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&bar.bar));

    asm volatile(
        "fabric.try_put.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.b128 "
        "[%0, %1], [%2], %3, [%4];"
        :
        : "r"(le), "l"(offset), "r"(src_smem_ptr), "r"(size), "r"(mbar_ptr)
        : "memory");
}

/** Counted put: consumer can wait on counter_offset (bytes completed). */
__device__ __forceinline__ void fabric_try_put_counted(uint32_t le, uint64_t offset, uint64_t counter_offset, void *src, int size, handle_barrier &bar) {
    uint32_t src_smem_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(src));
    uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&bar.bar));
    asm volatile(
        "fabric.try_put.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.counted::bytes.relaxed.sys.b128 "
        "[%0, %1, %2], [%3], %4, [%5];"
        :
        : "r"(le), "l"(offset), "l"(counter_offset), "r"(src_smem_ptr), "r"(size), "r"(mbar_ptr)
        : "memory");
}

// Fabric try_get from LE address (sm_100+)
// Do NOT call barrier_expect_tx before this — mbarrier::complete_tx::bytes reports the
// completion count directly; doubling via expect_tx causes a hang.
__device__ __forceinline__ void fabric_try_get(uint32_t le, uint64_t offset, void *dst, int size, handle_barrier &bar) {
    uint32_t dst_smem_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
    uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&bar.bar));

    asm volatile(
        "fabric.try_get.async.shared::cta.relaxed.sys.mbarrier::complete_tx::bytes.mbarrier::report::fabric.b128 "
        "[%0], [%1, %2], %3, [%4];"
        :
        : "r"(dst_smem_ptr), "r"(le), "l"(offset), "r"(size), "r"(mbar_ptr)
        : "memory");
}

// Fabric submit fence: commits fabric ops, restricts completion tracking to fetching (read) ops.
__device__ __forceinline__ void fabric_submit() {
    asm volatile("fabric.submit.op_restrict::fetching;" ::: "memory");
}

/** Put to multicast LE (multimem fabric, non-counted). */
__device__ __forceinline__ void fabric_try_put_multimem(uint32_t le, uint64_t offset, void *src, int size, handle_barrier &bar) {
    uint32_t src_smem_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(src));
    uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&bar.bar));
    asm volatile(
        "fabric.try_put.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.b128 "
        "[%0, %1], [%2], %3, [%4];"
        :
        : "r"(le), "l"(offset), "r"(src_smem_ptr), "r"(size), "r"(mbar_ptr)
        : "memory");
}

/**
 * Reduction read from multicast LE into shared memory (add.u64).
 * fabric.try_pullred sums N backing memories into dst_smem — the result at each
 * 8-byte element position equals the sum across all N participants.
 * complete_tx::bytes reports completion in bytes (same as try_get, not 16B-counted).
 * Using u64 matches multimem.ld_reduce.u64 and halves the request count vs u32.
 */
__device__ __forceinline__ void fabric_try_pullred_multimem_add_u64_sync(
    uint32_t le, uint64_t offset, void *dst, int size, __mbarrier_t *mbar) {
    uint64_t dst_smem_ptr = static_cast<uint64_t>(__cvta_generic_to_shared(dst));
    uint32_t mbar_ptr     = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
    asm volatile(
        "fabric.try_pullred.async.multimem.shared::cta.mbarrier::complete_tx::bytes.mbarrier::report::fabric.relaxed.sys.add.u64.sync "
        "[%0], [%1, %2], %3, [%4], 0xFFFFFFFF;"
        :
        : "l"(dst_smem_ptr), "r"(le), "l"(offset), "r"(size), "r"(mbar_ptr)
        : "memory");
}

#endif  // __CUDA_ARCH__ >= 1000

// Fence to order stores w.r.t TMA operations
__device__ __forceinline__ void fence_proxy_async_shared_cta() {
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}

/** Fence: order fabric alias operations through shared-memory proxy (release then acquire). */
__device__ __forceinline__ void fence_proxy_fabric_alias() {
    asm volatile("fence.proxy.generic::fabric.alias.release.sys;" ::: "memory");
    asm volatile("fence.proxy.generic::fabric.alias.acquire.sys;" ::: "memory");
}

#endif  // __CUDA_ARCH__ >= 900

#endif  // FABRIC_PTX_CUH_
