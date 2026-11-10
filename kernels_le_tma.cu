/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

/**
 * TMA-based Logical Endpoint kernels using fabric_try_put/get protocol
 *
 * Fabric kernels (fabric.try_put/get) require Blackwell (sm_100+).
 * For arch < Blackwell (or a toolkit without LE API headers), CMake builds
 * kernels_le_tma_stub.cu instead and leaves NVBW_HAVE_FABRIC_KERNELS undefined.
 * The stubs are protected by a mix of compile-time and runtime guards depending
 * on which test reaches a given kernel:
 *   - LE tests are waived at runtime by filterSupportsLogicalEndpoint().
 *   - The multicast-write (TMA) path is waived at compile time via
 *     NVBW_HAVE_FABRIC_KERNELS in MultinodeMulticastFabricPut::filter().
 */

#include "kernels_le_tma.hpp"
#include "common.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <iostream>
#include "error_handling.h"
#include "fabric_ptx.cuh"
#include "multimem_ptx.cuh"
// ============================================================================
// Constants
// ============================================================================

// TMA alignment requirement: all transfer sizes must be multiples of this
constexpr size_t LE_TMA_ALIGNMENT_BYTES = 16;

// Chunk size for TMA transfers - must be LE_TMA_ALIGNMENT_BYTES aligned
constexpr size_t LE_CHUNK_SIZE = 16384;  // 16KB per transfer chunk; legacy, still used by memset kernel
constexpr size_t LE_WRITE_CHUNK_SIZE = 28672;  // 28KB per warp for try_put
constexpr size_t LE_READ_CHUNK_SIZE  = 28672;  // 28KB per warp for try_get
static_assert(LE_CHUNK_SIZE % LE_TMA_ALIGNMENT_BYTES == 0 && LE_WRITE_CHUNK_SIZE % LE_TMA_ALIGNMENT_BYTES == 0 && LE_READ_CHUNK_SIZE % LE_TMA_ALIGNMENT_BYTES == 0, "Chunk sizes must be TMA-aligned");
constexpr int    LE_PULLRED_OPT_CHUNK = 8192;
constexpr size_t LE_PULLRED_OPT_SMEM  =
    ((size_t)4 * 2 * LE_PULLRED_OPT_CHUNK   // 4 warps × 2 slots × CHUNK
     + (size_t)4 * 64 + 128                 // mbarrier array + alignment padding
     + 255) & ~(size_t)255;                 // round up to 256-byte HW granularity → 66 048 B
constexpr int    LE_PULLRED_OPT_BPS   = 8;

static const int LE_WRITE_THREADS    = 128;
static const int LE_WRITE_WARPS      = LE_WRITE_THREADS / 32;  // 4
static const int LE_WRITE_BLK_PER_SM = 2;
static const int LE_READ_THREADS     = 128;
static const int LE_READ_WARPS       = LE_READ_THREADS / 32;   // 4
static const int LE_READ_BLK_PER_SM  = 2;

// kDeviceToDeviceLatencyPutSize is defined in common.h.
constexpr size_t FABRIC_SMEM_SIZE_LATENCY = 256;  // default 16 + 2*64 + padding (backward compat)

// ============================================================================
// Forward declarations for fabric kernels (needed for cudaLaunchKernel)
// ============================================================================
__global__ void leFabricWriteKernel(char *src_gmem, uint32_t dst_le, size_t num_bytes, unsigned long long loop_count);
__global__ void leFabricWriteMulticastKernel(char *src_gmem, uint32_t dst_le, size_t num_bytes, unsigned long long loop_count);
__global__ void leFabricReadKernel(uint32_t src_le, char *dst_gmem, size_t num_bytes, unsigned long long loop_count);
__global__ void leFabricMemsetKernel(uint32_t dst_le, unsigned int value, size_t num_bytes);
__global__ void leFabricReduceMulticastKernel(uint32_t src_le, char *dst_gmem, size_t num_bytes, unsigned long long loop_count);
__global__ void multimemWriteKernel(char *src_gmem, char *dst_multicast_va, size_t num_bytes, unsigned long long loop_count);
// device-to-device latency: producer waits for consumer ack (true device-to-device over NVLink).
__global__ void leFabricDeviceToDeviceConsumerMultimemKernel(char *consumer_buf, uint32_t sequence_offset, uint64_t ack_ptr, unsigned long long num_iterations);
__global__ void leFabricDeviceToDeviceProducerUnicastKernel(char *src_gmem, uint32_t dst_le, uint64_t ack_ptr, unsigned long long num_iterations);
__global__ void leFabricDeviceToDeviceProducerUnicastCountedKernel(char *src_gmem, uint32_t dst_le, uint64_t ack_ptr, unsigned long long num_iterations);
// Fabric / LE operation → raw-pointer PTX counterpart:
//   try_put.{counted}.{multimem}.{ascending}  →  UBLKCP
//   try_get                                →  UBLKCP
//   try_red.{counted}.{multimem}           →  UBLKRED
//   try_pullred.multimem                   →  LDGMC
//
// Semantics (fabric / TMA):
//   try_put     — does not guarantee the op is submitted; it only enqueues.
//   submit      — guarantees the op is issued out of the TMA → memsys.
//   barrier     — guarantees the op is completed from the programming-model perspective.
//   fence       — guarantees the op is pushed far enough for consumer threads/GPU to observe it;
//                 also orders subsequent ops (in our use case we need to order the 2 puts).
//
// Pointers to the device-to-device cases (producer kernel, consumer kernel, PTX/fabric commands):
//
//   Case 1 — Fabric unicast non-counted (2 put + fence)
//     Producer:  leFabricDeviceToDeviceProducerUnicastKernel
//     Consumer:  leFabricDeviceToDeviceConsumerMultimemKernel
//     PTX/LE:    cp.async.bulk G→S (mbarrier); fabric_try_put (put 1); fence_proxy_fabric_alias;
//               fabric_try_put (put 2); fabric_submit; mbarrier wait(2).
//     Consumer:  poll consumer_buf+sequence_offset; __threadfence_system; store ack.
//     Why 2 puts? Non-counted fabric_try_put provides no hardware arrival signal at the
//     destination. Put 2 carries a monotonically increasing sequence word that acts as a
//     software-visible arrival indicator; the consumer polls it instead of the payload,
//     which could initially hold any value. This is the standard software protocol for
//     signalling data arrival over LE when counted writes are unavailable. Case 2 replaces
//     the two-put sequence with a single counted put and exposes the resulting latency saving.
//
//   Case 2 — Fabric unicast counted (1 put_counted)
//     Producer:  leFabricDeviceToDeviceProducerUnicastCountedKernel
//     Consumer:  leFabricDeviceToDeviceConsumerMultimemKernel
//     PTX/LE:    cp.async.bulk G→S (mbarrier); fabric_try_put_counted; fabric_submit; mbarrier wait.
//     Consumer:  same as Case 1.
//
//   Case 3 — Raw-pointer non-counted (2 put + fence)
//     Producer:  ublkcpGSDeviceToDeviceProducerKernel
//     Consumer:  ublkcpGSDeviceToDeviceConsumerKernel
//     PTX:       cp.async.bulk G→S (mbarrier); cp.async.bulk S→G (put 1); __threadfence_system;
//               cp.async.bulk S→G (put 2); __threadfence_system.
//     Consumer:  poll mailbox+0 (put1), ack; poll mailbox+16 (put2), ack.
//
//   Case 4 — Raw-pointer counted (1 put)
//     Producer:  ublkcpGSCountedDeviceToDeviceProducerKernel
//     Consumer:  ublkcpGSCountedDeviceToDeviceConsumerKernel
//     PTX:       cp.async.bulk G→S (mbarrier); clear barMem; counted S→G; poll barMem==size;
//               __threadfence_system; store *sequence.
//     Mailbox:   [data 16B][counter 8B]; sequence_ptr = mailbox (16B aligned).
//     Consumer:  poll sequence_ptr (written by producer after counted copy completes); __threadfence; store ack.
//
__global__ void ublkcpGSCountedDeviceToDeviceProducerKernel(char *src_gmem, char *mailbox_gmem, uint64_t sequence_ptr, uint64_t ack_ptr, unsigned long long num_iterations);
__global__ void ublkcpGSDeviceToDeviceProducerKernel(char *src_gmem, char *mailbox_gmem, uint64_t ack_ptr, unsigned long long num_iterations);
__global__ void ublkcpGSDeviceToDeviceConsumerKernel(char *mailbox_gmem, uint64_t ack_ptr, unsigned long long num_iterations);
__global__ void ublkcpGSCountedDeviceToDeviceConsumerKernel(char *mailbox_gmem, uint64_t sequence_ptr, uint64_t ack_ptr, unsigned long long num_iterations);

// multimemReadKernelV4F32 / launchMultimemReadKernelV4F32 live in kernels_multimem.cu
// (always compiled), since multimem.ld_reduce only needs sm_90+ and must not be gated
// behind NVBW_HAVE_LE_API.

#if __CUDA_ARCH__ >= 1000

/**
 * Fabric-based LE write kernel (bulk write to LE, no round-trip sync).
 * Writes from device global memory to a Logical Endpoint using fabric_try_put.
 * Same building blocks as SIMPLE protocol sample but without waiting for remote ack.
 */
__global__ void leFabricWriteKernel(
    char *src_gmem,
    uint32_t dst_le,
    size_t num_bytes,
    unsigned long long loop_count
) {
    extern __shared__ char smem_bytes[];
    using SMEMPIPE_T = SMEMPIPE<LE_WRITE_CHUNK_SIZE, LE_WRITE_WARPS>;
    SMEMPIPE_T &smem = reinterpret_cast<SMEMPIPE_T&>(smem_bytes);

    __shared__ handle_barrier bar[LE_WRITE_WARPS];
    __shared__ handle_barrier load_bar[LE_WRITE_WARPS];

    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;

    if (lane == 0) {
        bar[warp].init(1);
        load_bar[warp].init(1);
    }

    size_t warp_offset = (size_t)(blockIdx.x * LE_WRITE_WARPS + warp) * LE_WRITE_CHUNK_SIZE;
    size_t grid_stride = (size_t)gridDim.x * LE_WRITE_WARPS * LE_WRITE_CHUNK_SIZE;
    int phase = 0;

    for (unsigned long long loop = 0; loop < loop_count; loop++) {
        #pragma unroll 4
        for (size_t offset = warp_offset; offset < num_bytes; offset += grid_stride) {
            size_t chunk_size = min((size_t)LE_WRITE_CHUNK_SIZE, num_bytes - offset);
            if (chunk_size == 0) continue;

            if (lane == 0) {
                cp_async_bulk_global_to_shared(&smem.buf[warp][0], src_gmem + offset, load_bar[warp], (int)chunk_size);
                __mbarrier_token_t token = load_bar[warp].arrive_tx((uint32_t)chunk_size);
                load_bar[warp].wait_token(token);
                fabric_try_put(dst_le, (uint64_t)offset, &smem.buf[warp][0], (int)chunk_size, bar[warp]);
                fabric_submit();
                bar[warp].arrive_relaxed((int)chunk_size);
                bar[warp].wait_relaxed(phase & 1);
            }
            phase++;
        }
    }
}

/**
 * Fabric-based write to multicast LE using fabric.try_put.multimem.
 * PTX spec requires .multimem when dstLeId is a multicast logical endpoint.
 */
__global__ void leFabricWriteMulticastKernel(
    char *src_gmem,
    uint32_t dst_le,
    size_t num_bytes,
    unsigned long long loop_count
) {
    extern __shared__ char smem_bytes[];
    using SMEMPIPE_T = SMEMPIPE<LE_WRITE_CHUNK_SIZE, LE_WRITE_WARPS>;
    SMEMPIPE_T &smem = reinterpret_cast<SMEMPIPE_T&>(smem_bytes);

    __shared__ handle_barrier bar[LE_WRITE_WARPS];
    __shared__ handle_barrier load_bar[LE_WRITE_WARPS];

    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;

    if (lane == 0) {
        bar[warp].init(1);
        load_bar[warp].init(1);
    }

    size_t warp_offset = (size_t)(blockIdx.x * LE_WRITE_WARPS + warp) * LE_WRITE_CHUNK_SIZE;
    size_t grid_stride = (size_t)gridDim.x * LE_WRITE_WARPS * LE_WRITE_CHUNK_SIZE;
    int phase = 0;

    for (unsigned long long loop = 0; loop < loop_count; loop++) {
        #pragma unroll 4
        for (size_t offset = warp_offset; offset < num_bytes; offset += grid_stride) {
            size_t chunk_size = min((size_t)LE_WRITE_CHUNK_SIZE, num_bytes - offset);
            if (chunk_size == 0) continue;

            if (lane == 0) {
                cp_async_bulk_global_to_shared(&smem.buf[warp][0], src_gmem + offset, load_bar[warp], (int)chunk_size);
                __mbarrier_token_t token = load_bar[warp].arrive_tx((uint32_t)chunk_size);
                load_bar[warp].wait_token(token);
                fabric_try_put_multimem(dst_le, (uint64_t)offset, &smem.buf[warp][0], (int)chunk_size, bar[warp]);
                fabric_submit();
                bar[warp].arrive_relaxed((int)chunk_size);
                bar[warp].wait_relaxed(phase & 1);
            }
            phase++;
        }
    }
}

/**
 * Fabric-based LE read kernel (fabric.try_get → smem → global).
 */
__global__ void leFabricReadKernel(
    uint32_t src_le,
    char *dst_gmem,
    size_t num_bytes,
    unsigned long long loop_count
) {
    extern __shared__ char smem_bytes[];
    using SMEMPIPE_T = SMEMPIPE<LE_CHUNK_SIZE, 1>;
    SMEMPIPE_T &smem = reinterpret_cast<SMEMPIPE_T&>(smem_bytes);

    __shared__ handle_barrier bar;

    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;

    if (lane == 0 && warp == 0) {
        bar.init(1);
    }

    size_t block_offset = blockIdx.x * LE_CHUNK_SIZE;
    size_t grid_stride = gridDim.x * LE_CHUNK_SIZE;
    int phase = 0;

    for (unsigned long long loop = 0; loop < loop_count; loop++) {
        for (size_t offset = block_offset; offset < num_bytes; offset += grid_stride) {
            size_t chunk_size = min((size_t)LE_CHUNK_SIZE, num_bytes - offset);
            if (chunk_size == 0) continue;
            if (lane == 0 && warp == 0) {
                fabric_try_get(src_le, (uint64_t)offset, &smem.buf[0][0], (int)chunk_size, bar);
                fabric_submit();
                bar.arrive_relaxed_bytes((int)chunk_size);
                bar.wait_relaxed(phase & 1);
                cp_async_bulk_shared_to_global(dst_gmem + offset, &smem.buf[0][0], (int)chunk_size);
                cp_async_bulk_commit_group();
                cp_async_bulk_wait_all_read();
            }
            phase++;
        }
    }
}

/**
 * Pullred kernel using the measured-optimal chunk size (LE_PULLRED_OPT_CHUNK=1024 B).
 * Double-buffer pipeline: 2 DMA slots per warp overlap fabric fetch with gmem writeback.
 */
#if __CUDA_ARCH__ >= 1000
__global__ void leFabricReduceMulticastKernel(
    uint32_t src_le, char *dst_gmem, size_t num_bytes, unsigned long long loop_count
) {
    extern __shared__ char smem_bytes[];
    using DB_T = SMEMPIPE<LE_PULLRED_OPT_CHUNK, LE_READ_WARPS * 2>;
    DB_T &smem = reinterpret_cast<DB_T&>(smem_bytes);
    __shared__ __mbarrier_t bar[LE_READ_WARPS][2];
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    if (lane == 0) {
        __mbarrier_init(&bar[warp][0], 1);
        __mbarrier_init(&bar[warp][1], 1);
    }
    __syncwarp();  // memory fence: fabric_try_pullred_multimem uses .sync (all 32 threads); bar must be visible to all lanes before first use
    const size_t warp_id     = (size_t)(blockIdx.x * LE_READ_WARPS + warp);
    const size_t warp_off0   = warp_id * (size_t)LE_PULLRED_OPT_CHUNK;
    const size_t grid_stride = (size_t)gridDim.x * LE_READ_WARPS * (size_t)LE_PULLRED_OPT_CHUNK;
    int phase0 = 0, phase1 = 0;
    for (unsigned long long loop = 0; loop < loop_count; loop++) {
        size_t off = warp_off0;
        for (; off + grid_stride < num_bytes; off += 2 * grid_stride) {
            const size_t off1 = off + grid_stride;
            const size_t sz0 = min((size_t)LE_PULLRED_OPT_CHUNK, num_bytes - off);
            const size_t sz1 = min((size_t)LE_PULLRED_OPT_CHUNK, num_bytes - off1);
            fabric_try_pullred_multimem_add_u64_sync(src_le, off,  &smem.buf[warp*2+0][0], sz0, &bar[warp][0]);
            fabric_try_pullred_multimem_add_u64_sync(src_le, off1, &smem.buf[warp*2+1][0], sz1, &bar[warp][1]);
            fabric_submit();
            if (lane == 0) {
                barrier_red_arrive1_tx_relaxed(&bar[warp][0], (uint32_t)sz0);
                barrier_red_arrive1_tx_relaxed(&bar[warp][1], (uint32_t)sz1);
                while (!barrier_try_wait_parity_relaxed(&bar[warp][0], phase0 & 1)) {}
                cp_async_bulk_shared_to_global(dst_gmem + off,  &smem.buf[warp*2+0][0], (int)sz0);
                cp_async_bulk_commit_group();
                cp_async_bulk_wait_group_read_1();
                while (!barrier_try_wait_parity_relaxed(&bar[warp][1], phase1 & 1)) {}
                cp_async_bulk_shared_to_global(dst_gmem + off1, &smem.buf[warp*2+1][0], (int)sz1);

                cp_async_bulk_commit_group();
                cp_async_bulk_wait_all_read();
            }
            phase0++; phase1++;
        }
        if (off < num_bytes) {
            const size_t sz0 = min((size_t)LE_PULLRED_OPT_CHUNK, num_bytes - off);
            fabric_try_pullred_multimem_add_u64_sync(src_le, off, &smem.buf[warp*2+0][0], sz0, &bar[warp][0]);
            fabric_submit();
            if (lane == 0) {
                barrier_red_arrive1_tx_relaxed(&bar[warp][0], (uint32_t)sz0);
                while (!barrier_try_wait_parity_relaxed(&bar[warp][0], phase0 & 1)) {}
                cp_async_bulk_shared_to_global(dst_gmem + off, &smem.buf[warp*2+0][0], (int)sz0);
                cp_async_bulk_commit_group();
                cp_async_bulk_wait_all_read();
            }
            phase0++;
        }
    }
}
#else
// Stub for non-sm_100 / host compilation pass.
__global__ void leFabricReduceMulticastKernel(uint32_t, char*, size_t, unsigned long long) {}
#endif

/**
 * Fabric-based LE memset kernel
 * Fills a Logical Endpoint with a constant value
 */
__global__ void leFabricMemsetKernel(
    uint32_t dst_le,
    unsigned int value,
    size_t num_bytes
) {
    extern __shared__ char smem_bytes[];
    using SMEMPIPE_T = SMEMPIPE<LE_CHUNK_SIZE, 1>;
    SMEMPIPE_T &smem = reinterpret_cast<SMEMPIPE_T&>(smem_bytes);

    __shared__ handle_barrier bar;

    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;

    // Initialize shared memory with pattern
    unsigned int *smem_u32 = (unsigned int*)&smem.buf[0][0];
    for (size_t i = threadIdx.x; i < LE_CHUNK_SIZE / sizeof(unsigned int); i += blockDim.x) {
        smem_u32[i] = value;
    }

    if (lane == 0 && warp == 0) {
        bar.init(1);
    }
    __syncthreads();

    fence_proxy_async_shared_cta();

    size_t block_offset = blockIdx.x * LE_CHUNK_SIZE;
    size_t grid_stride = gridDim.x * LE_CHUNK_SIZE;
    int phase = 0;

    for (size_t offset = block_offset; offset < num_bytes; offset += grid_stride) {
        size_t chunk_size = min((size_t)LE_CHUNK_SIZE, num_bytes - offset);
        if (chunk_size == 0) continue;

        if (lane == 0 && warp == 0) {
            fabric_try_put(dst_le, (uint64_t)offset, &smem.buf[0][0], (int)chunk_size, bar);
            fabric_submit();
            bar.arrive_relaxed((int)chunk_size);
            bar.wait_relaxed(phase & 1);
        }
        phase++;
    }
}

// Raw-pointer write to multicast VA: TMA G→S then cp.async.bulk S→G (using multicast VA destination).
__global__ void multimemWriteKernel(
    char *src_gmem,
    char *dst_multicast_va,
    size_t num_bytes,
    unsigned long long loop_count
) {
    extern __shared__ char smem_bytes[];
    using SMEMPIPE_T = SMEMPIPE<LE_WRITE_CHUNK_SIZE, LE_WRITE_WARPS>;
    SMEMPIPE_T &smem = reinterpret_cast<SMEMPIPE_T&>(smem_bytes);
    __shared__ handle_barrier load_bar[LE_WRITE_WARPS];
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    if (lane == 0) {
        load_bar[warp].init(1);
    }
    size_t warp_offset = (size_t)(blockIdx.x * LE_WRITE_WARPS + warp) * LE_WRITE_CHUNK_SIZE;
    size_t grid_stride = (size_t)gridDim.x * LE_WRITE_WARPS * LE_WRITE_CHUNK_SIZE;
    for (unsigned long long loop = 0; loop < loop_count; loop++) {
        #pragma unroll 4
        for (size_t offset = warp_offset; offset < num_bytes; offset += grid_stride) {
            size_t chunk_size = min((size_t)LE_WRITE_CHUNK_SIZE, num_bytes - offset);
            if (chunk_size == 0) continue;
            if (lane == 0) {
                cp_async_bulk_global_to_shared(&smem.buf[warp][0], src_gmem + offset, load_bar[warp], (int)chunk_size);
                __mbarrier_token_t token = load_bar[warp].arrive_tx((uint32_t)chunk_size);
                load_bar[warp].wait_token(token);
                cp_async_bulk_shared_to_global(dst_multicast_va + offset, &smem.buf[warp][0], (int)chunk_size);
                cp_async_bulk_commit_group();
                cp_async_bulk_wait_all_read();
            }
        }
    }
}

// device-to-device consumer (LE path): polls consumer_buf[sequence_offset..+3] for the sequence word (i+1),
// then writes ack back to the producer. Sequences start at 1 so an uninitialized 0 won't match.
__global__ void leFabricDeviceToDeviceConsumerMultimemKernel(char *consumer_buf, uint32_t sequence_offset, uint64_t ack_ptr, unsigned long long num_iterations) {
    volatile int *ack = (volatile int *)(uintptr_t)ack_ptr;
    for (unsigned long long i = 0; i < num_iterations; i++) {
        uint32_t expected = (uint32_t)(i + 1);  // 1,2,3,... never 0
        while (*(volatile uint32_t *)(consumer_buf + sequence_offset) != expected) {}
        // Ensure the sequence load is globally ordered before the ack store reaches the
        // producer; without this fence the producer could observe the ack before the data.
        __threadfence_system();
        *ack = 1;
    }
}

// device-to-device producer unicast (non-counted): 2 put + fence per iteration, then poll ack (peer read).
// Two puts are required: put1 carries the data payload (bytes 0..15); put2 carries the sequence
// word (bytes 16..31) that the consumer polls to confirm both puts have arrived. The consumer
// acks after detecting the sequence word, completing one round trip per put.
__global__ void leFabricDeviceToDeviceProducerUnicastKernel(char *src_gmem, uint32_t dst_le, uint64_t ack_ptr, unsigned long long num_iterations) {
    // smem holds two consecutive put slots: [data 16B][sequence 16B].
    __shared__ alignas(16) char smem[2 * kDeviceToDeviceLatencyPutSize];
    __shared__ handle_barrier bar;
    __shared__ handle_barrier load_bar;
    volatile int *ack = (volatile int *)(uintptr_t)ack_ptr;
    bar.init(1);
    load_bar.init(1);
    int phase = 0;
    for (unsigned long long i = 0; i < num_iterations; i++) {
        *ack = 0;
        cp_async_bulk_global_to_shared(smem, src_gmem, load_bar, (uint32_t)kDeviceToDeviceLatencyPutSize);
        __mbarrier_token_t token = load_bar.arrive_tx((uint32_t)kDeviceToDeviceLatencyPutSize);
        load_bar.wait_token(token);
        // Write sequence word into the second slot; consumer polls this to detect put2 arrival.
        *(uint32_t *)(smem + kDeviceToDeviceLatencyPutSize) = (uint32_t)(i + 1);  // sequence 1,2,...
        fabric_try_put(dst_le, 0, smem, (int)kDeviceToDeviceLatencyPutSize, bar);
        fabric_submit();
        fence_proxy_fabric_alias();
        fabric_try_put(dst_le, kDeviceToDeviceLatencyPutSize, smem + kDeviceToDeviceLatencyPutSize, (int)kDeviceToDeviceLatencyPutSize, bar);
        fabric_submit();
        bar.arrive_relaxed_bytes(2);  // 2 puts issued above, each credits 1 arrival
        bar.wait_relaxed(phase & 1);
        while (*ack != 1) {}
        phase++;
    }
}

// device-to-device producer unicast (counted): fabric_try_put_counted + fabric_submit + mbarrier wait.
// bar.wait_relaxed drains the hardware counted-report descriptor allocated per put_counted call.
__global__ void leFabricDeviceToDeviceProducerUnicastCountedKernel(char *src_gmem, uint32_t dst_le, uint64_t ack_ptr, unsigned long long num_iterations) {
    __shared__ alignas(16) char smem[kDeviceToDeviceLatencyPutSize];
    __shared__ handle_barrier bar;
    __shared__ handle_barrier load_bar;
    // counter_le_offset: byte offset within the LE where the hardware completion counter lives.
    // Placed one put-size slot past the start so it does not overlap the data payload.
    const uint64_t counter_le_offset = kDeviceToDeviceLatencyPutSize;
    // put_offset: byte offset within the LE for the data payload.
    // Placed two put-size slots past the start so both the payload and the counter fit without overlap.
    const uint64_t put_offset = 2 * kDeviceToDeviceLatencyPutSize;
    volatile int *ack = (volatile int *)(uintptr_t)ack_ptr;
    bar.init(1);
    load_bar.init(1);
    int phase = 0;
    for (unsigned long long i = 0; i < num_iterations; i++) {
        *ack = 0;
        cp_async_bulk_global_to_shared(smem, src_gmem, load_bar, (uint32_t)kDeviceToDeviceLatencyPutSize);
        __mbarrier_token_t token = load_bar.arrive_tx((uint32_t)kDeviceToDeviceLatencyPutSize);
        load_bar.wait_token(token);
        // Embed sequence number in the payload so the consumer can detect arrival
        // without decoding the LE hardware counter (which is not directly pollable).
        *(uint32_t *)smem = (uint32_t)(i + 1);
        // Prime the barrier for the counted put's completion BEFORE issuing it:
        // complete_tx::16B credits one unit per 16B delivered, so a putSize-byte put
        // contributes putSize/16 units. Without this prime the barrier has nothing to
        // complete and the wait below spins forever.
        bar.arrive_tx((uint32_t)(kDeviceToDeviceLatencyPutSize / 16));
        fabric_try_put_counted(dst_le, put_offset, counter_le_offset, smem, (int)kDeviceToDeviceLatencyPutSize, bar);
        fabric_submit();
        // Wait on the put's own completion before reusing smem next iteration.
        bar.wait_relaxed(phase & 1);
        phase++;
        while (*ack != 1) {}
    }
}


// UBLKCP non-counted: 2 put + fence per iteration, mirroring the LE 2-put protocol for a fair
// comparison. Put1 carries the data payload; put2 carries a sequence word the consumer uses to
// detect arrival (same two-slot layout as the LE variant). Each put triggers a full round trip
// so the reported latency reflects one put+ack cycle, not the entire iteration.
// SM 100+ (Blackwell) only; Hopper not supported for this path.
// Mailbox layout: bytes 0..3 = put1 sequence (2*i+1), bytes 16..19 = put2 sequence (2*i+2).
__global__ void ublkcpGSDeviceToDeviceProducerKernel(char *src_gmem, char *mailbox_gmem, uint64_t ack_ptr, unsigned long long num_iterations) {
    // smem holds two consecutive put slots matching the mailbox layout.
    __shared__ alignas(16) char smem[2 * kDeviceToDeviceLatencyPutSize];
    __shared__ handle_barrier load_bar;
    volatile uint32_t *ack = (volatile uint32_t *)(uintptr_t)ack_ptr;
    load_bar.init(1);
    for (unsigned long long i = 0; i < num_iterations; i++) {
        cp_async_bulk_global_to_shared(smem, src_gmem, load_bar, (uint32_t)kDeviceToDeviceLatencyPutSize);
        __mbarrier_token_t token = load_bar.arrive_tx((uint32_t)kDeviceToDeviceLatencyPutSize);
        load_bar.wait_token(token);
        // Stamp monotonically increasing sequence numbers so the consumer can distinguish
        // each put's arrival: put1 → mailbox[0..3] = 2i+1, put2 → mailbox[16..19] = 2i+2.
        *(uint32_t *)smem = (uint32_t)(2 * i + 1);
        *(uint32_t *)(smem + kDeviceToDeviceLatencyPutSize) = (uint32_t)(2 * i + 2);
        // Put 1 (bytes 0-15) → round trip
        *ack = 0;
        cp_async_bulk_shared_to_global(mailbox_gmem, smem, (uint32_t)kDeviceToDeviceLatencyPutSize);
        cp_async_bulk_commit_group();
        cp_async_bulk_wait_all_read();
        // Ensure the UBLKCP write is visible to the remote GPU before the consumer's ack
        // write back is observed; without this fence the ack could be seen before the data.
        __threadfence_system();
        while (*ack != 1) {}
        // Put 2 (bytes 16-31) → round trip
        *ack = 0;
        cp_async_bulk_shared_to_global(mailbox_gmem + kDeviceToDeviceLatencyPutSize, smem + kDeviceToDeviceLatencyPutSize, (uint32_t)kDeviceToDeviceLatencyPutSize);
        cp_async_bulk_commit_group();
        cp_async_bulk_wait_all_read();
        __threadfence_system();
        while (*ack != 1) {}
    }
}

// Consumer for ublkcp_gs device-to-device: acks after each put.
// Polls mailbox[0..3] for put1 sequence, then mailbox[16..19] for put2 sequence.
__global__ void ublkcpGSDeviceToDeviceConsumerKernel(char *mailbox_gmem, uint64_t ack_ptr, unsigned long long num_iterations) {
    volatile uint32_t *ack = (volatile uint32_t *)(uintptr_t)ack_ptr;
    for (unsigned long long i = 0; i < num_iterations; i++) {
        uint32_t expect1 = (uint32_t)(2 * i + 1);
        uint32_t expect2 = (uint32_t)(2 * i + 2);
        while (*(volatile uint32_t *)mailbox_gmem != expect1) {}
        // Ensure the mailbox load is globally ordered before the ack store reaches the
        // producer; without this fence the producer could observe the ack before the data.
        __threadfence_system();
        *ack = 1;
        while (*(volatile uint32_t *)(mailbox_gmem + kDeviceToDeviceLatencyPutSize) != expect2) {}
        // Same ordering requirement as above: fence before ack so the producer
        // cannot observe the ack before the put2 data is visible.
        __threadfence_system();
        *ack = 1;
    }
}

// Watchdog: bound each cross-GPU spin with a %globaltimer deadline so a stalled
// handshake returns instead of hanging. The bound is far above any healthy run.
__device__ __forceinline__ unsigned long long d2dLatencyGlobalTimerNs() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}
static constexpr unsigned long long kDeviceToDeviceLatencyWatchdogNs = 5ULL * 1000ULL * 1000ULL * 1000ULL;  // 5 s

// UBLKCP.G.S.COUNTED: producer kernel (separate from consumer).
// Mailbox layout: [data 16B][barMem counter 8B].
//
//   dstMem (data, 16-byte aligned): the 16 bytes copied by the counted write.
//
//   barMem (counter at mailbox+16): a hardware-managed 8-byte arrival counter. The TMA
//     engine atomically *increments* it by the transfer size when each counted write
//     arrives at the destination. Counted copies are not tracked by cp.async.bulk
//     bulk_group, so commit_group/wait_all_read cannot complete them. The producer
//     therefore clears barMem to 0, issues the counted write, and polls until
//     barMem == transfer_size before publishing *sequence or reusing smem.
//     __threadfence_system() alone does not wait for the async copy: without the
//     barMem wait the consumer can ack the ordinary *sequence peer store while the
//     counted copy is still in flight, and the next iteration can overwrite smem
//     while the prior copy is still reading it.
//
//   sequence_ptr: an application-managed uint32 (overlapping mailbox base) that the
//     producer stamps with (i+1) only after barMem confirms the counted transfer completed.
//     The consumer polls this word; it only needs to wait for (i+1) each iteration with no
//     per-iteration reset.
// The cp.async.bulk.counted.global.shared::cta instruction below is not
// supported on consumer Blackwell (sm_12X, e.g. gb202) — it traps at runtime
// there. Gate the builtin off for the sm_12X family only; sm_100/103/110
// (datacenter Blackwell) and any other arch where the instruction is
// supported keep the real body.
#if !(__CUDA_ARCH__ >= 1200 && __CUDA_ARCH__ < 1300)
constexpr size_t UBLKCP_COUNTED_COUNTER_OFFSET = kDeviceToDeviceLatencyPutSize;  // barMem at mailbox+16
__global__ void ublkcpGSCountedDeviceToDeviceProducerKernel(char *src_gmem, char *mailbox_gmem, uint64_t sequence_ptr, uint64_t ack_ptr, unsigned long long num_iterations) {
    __shared__ alignas(16) char smem[kDeviceToDeviceLatencyPutSize];
    __shared__ handle_barrier load_bar;
    volatile uint32_t *sequence = (volatile uint32_t *)(uintptr_t)sequence_ptr;
    // ack_ptr layout: [0] handshake (consumer-writable), [1] producer-only watchdog status.
    volatile uint32_t *ack = (volatile uint32_t *)(uintptr_t)(ack_ptr + kDeviceToDeviceLatencyAckHandshakeOffset);
    volatile uint32_t *watchdogStatus = (volatile uint32_t *)(uintptr_t)(ack_ptr + kDeviceToDeviceLatencyAckStatusOffset);
    volatile uint64_t *barMem = (volatile uint64_t *)(mailbox_gmem + UBLKCP_COUNTED_COUNTER_OFFSET);
    load_bar.init(1);
    for (unsigned long long i = 0; i < num_iterations; i++) {
        *ack = 0;
        cp_async_bulk_global_to_shared(smem, src_gmem, load_bar, (uint32_t)kDeviceToDeviceLatencyPutSize);
        __mbarrier_token_t token = load_bar.arrive_tx((uint32_t)kDeviceToDeviceLatencyPutSize);
        load_bar.wait_token(token);
        *(uint32_t *)smem = (uint32_t)(i + 1);
        // Clear barMem before the counted write. HW increments it by the transfer size
        // on arrival; polling for == putSize is the completion signal.
        *barMem = 0;
        __threadfence_system();
        // S→G counted: dstMem and srcMem must be 16-byte aligned.
        __nv_ptx_builtin_ocg_cp_async_bulk_counted_global_shared(
            (uint64_t)(mailbox_gmem),                                     // dstMem: data (16B aligned)
            (uint64_t)(mailbox_gmem + UBLKCP_COUNTED_COUNTER_OFFSET),      // barMem: counter at mailbox+16
            (uint64_t)(&smem[0]),                                          // srcMem: shared
            (uint32_t)kDeviceToDeviceLatencyPutSize, 0);
        // Wait for counted-copy arrival (barMem incremented to putSize) before
        // publishing *sequence. Do not use wait_all_read — counted writes are outside
        // the bulk_group tracker.
        const unsigned long long barDeadline = d2dLatencyGlobalTimerNs() + kDeviceToDeviceLatencyWatchdogNs;
        while (*barMem != (uint64_t)kDeviceToDeviceLatencyPutSize) {
            if (d2dLatencyGlobalTimerNs() > barDeadline) {
                // Publish into the producer-only status word so a late consumer *ack=1
                // cannot clobber the sentinel after this kernel returns.
                *watchdogStatus = (uint32_t)kDeviceToDeviceLatencyWatchdogTimeoutAck;
                __threadfence_system();
                return;
            }
        }
        __threadfence_system();
        *sequence = (uint32_t)(i + 1);
        const unsigned long long ackDeadline = d2dLatencyGlobalTimerNs() + kDeviceToDeviceLatencyWatchdogNs;
        while (*ack != 1) {
            if (d2dLatencyGlobalTimerNs() > ackDeadline) {
                // Watchdog fired: the consumer never acked this put, so the cross-GPU
                // handshake has stalled. Publish into the producer-only status word (not
                // the handshake ack) so a delayed consumer cannot overwrite the sentinel
                // before the host reads it after both streams synchronize.
                *watchdogStatus = (uint32_t)kDeviceToDeviceLatencyWatchdogTimeoutAck;
                __threadfence_system();
                return;
            }
        }
    }
}
#else
__global__ void ublkcpGSCountedDeviceToDeviceProducerKernel(char *, char *, uint64_t, uint64_t, unsigned long long) {}
#endif

// UBLKCP.G.S.COUNTED: consumer kernel (separate from producer).
// Polls the application-managed sequence word (written by the producer after barMem
// confirms the counted write arrived). The consumer does not poll barMem: a monotonic
// sequence needs no per-iteration re-arm on the consumer side.
__global__ void ublkcpGSCountedDeviceToDeviceConsumerKernel(char *mailbox_gmem, uint64_t sequence_ptr, uint64_t ack_ptr, unsigned long long num_iterations) {
    (void)mailbox_gmem;
    volatile uint32_t *sequence = (volatile uint32_t *)(uintptr_t)sequence_ptr;
    volatile uint32_t *ack = (volatile uint32_t *)(uintptr_t)ack_ptr;
    for (unsigned long long i = 0; i < num_iterations; i++) {
        uint32_t expected = (uint32_t)(i + 1);
        const unsigned long long seqDeadline = d2dLatencyGlobalTimerNs() + kDeviceToDeviceLatencyWatchdogNs;
        while (*sequence != expected) {
            if (d2dLatencyGlobalTimerNs() > seqDeadline) return;  // watchdog
        }
        // Ensure the sequence load is globally ordered before the ack store reaches the
        // producer on a different GPU; __threadfence() would only cover the local GPU.
        __threadfence_system();
        *ack = 1;
    }
}

#else
// Stubs for non-sm_100 architectures in a multi-arch build (e.g. sm_90+sm_100).
// DeviceToDeviceLatency::filter() gates the UBLKCP path to sm_100+, so
// these are never launched; they exist only to satisfy the fatbin symbol table.
__global__ void ublkcpGSDeviceToDeviceProducerKernel(char *src_gmem, char *mailbox_gmem, uint64_t ack_ptr, unsigned long long num_iterations) {}
__global__ void ublkcpGSDeviceToDeviceConsumerKernel(char *mailbox_gmem, uint64_t ack_ptr, unsigned long long num_iterations) {}
__global__ void ublkcpGSCountedDeviceToDeviceProducerKernel(char *src_gmem, char *mailbox_gmem, uint64_t sequence_ptr, uint64_t ack_ptr, unsigned long long num_iterations) {}
__global__ void ublkcpGSCountedDeviceToDeviceConsumerKernel(char *mailbox_gmem, uint64_t sequence_ptr, uint64_t ack_ptr, unsigned long long num_iterations) {}
#endif  // __CUDA_ARCH__ >= 1000

// ============================================================================
// Host-side kernel launcher functions
// ============================================================================

// Shared memory size for fabric kernels (per-warp buffers + barriers; mbarrier ~64B)
// Write kernel uses two barriers per warp (handle_barrier bar + handle_barrier load_bar); read and
// pullred kernels use one mbarrier per warp.
constexpr size_t FABRIC_SMEM_SIZE_WRITE        = LE_WRITE_WARPS * LE_WRITE_CHUNK_SIZE + LE_WRITE_WARPS * 64 * 2 + 128;
constexpr size_t FABRIC_SMEM_SIZE_READ         = LE_READ_WARPS  * LE_READ_CHUNK_SIZE  + LE_READ_WARPS  * 64     + 128;
// Memset kernel still uses single-chunk smem
constexpr size_t FABRIC_SMEM_SIZE = LE_CHUNK_SIZE + 128;

// Host API: Write to LE
extern "C"
void leLaunchWriteKernel(
    CUdeviceptr src,
    unsigned int dst_leid,
    size_t num_bytes,
    unsigned long long loop_count,
    CUstream stream
) {
    int deviceId;
    CUDA_ASSERT(cudaGetDevice(&deviceId));
    int numSm;
    CUDA_ASSERT(cudaDeviceGetAttribute(&numSm, cudaDevAttrMultiProcessorCount, deviceId));

    size_t numChunks = (num_bytes + LE_WRITE_CHUNK_SIZE * LE_WRITE_WARPS - 1) / (LE_WRITE_CHUNK_SIZE * LE_WRITE_WARPS);
    int numBlocks = min((int)numChunks, numSm * LE_WRITE_BLK_PER_SM);
    if (numBlocks < 1) numBlocks = 1;
    uint32_t dst_le = (uint32_t)dst_leid;
    void* args[] = { &src, &dst_le, &num_bytes, &loop_count };
    CUDA_ASSERT(cudaFuncSetAttribute(leFabricWriteKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)FABRIC_SMEM_SIZE_WRITE));
    CUDA_ASSERT(cudaLaunchKernel((void*)leFabricWriteKernel, dim3(numBlocks), dim3(LE_WRITE_THREADS),
                     args, FABRIC_SMEM_SIZE_WRITE, stream));
}

// Host API: Write to multicast LE via fabric.try_put.multimem
void leLaunchWriteMulticastKernel(
    CUdeviceptr src,
    unsigned int dst_leid,
    size_t num_bytes,
    unsigned long long loop_count,
    CUstream stream
) {
    int deviceId;
    CUDA_ASSERT(cudaGetDevice(&deviceId));
    int numSm;
    CUDA_ASSERT(cudaDeviceGetAttribute(&numSm, cudaDevAttrMultiProcessorCount, deviceId));

    size_t numChunks = (num_bytes + LE_WRITE_CHUNK_SIZE * LE_WRITE_WARPS - 1) / (LE_WRITE_CHUNK_SIZE * LE_WRITE_WARPS);
    int numBlocks = min((int)numChunks, numSm * LE_WRITE_BLK_PER_SM);
    if (numBlocks < 1) numBlocks = 1;
    uint32_t dst_le = (uint32_t)dst_leid;
    void* args[] = { &src, &dst_le, &num_bytes, &loop_count };
    CUDA_ASSERT(cudaFuncSetAttribute(leFabricWriteMulticastKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)FABRIC_SMEM_SIZE_WRITE));
    CUDA_ASSERT(cudaLaunchKernel((void*)leFabricWriteMulticastKernel, dim3(numBlocks), dim3(LE_WRITE_THREADS),
                     args, FABRIC_SMEM_SIZE_WRITE, stream));
}

// Host API: Write to raw multicast VA via TMA-staged multimem.st
void launchMultimemWriteKernel(
    CUdeviceptr src,
    CUdeviceptr dst_multicast_va,
    size_t num_bytes,
    unsigned long long loop_count,
    CUstream stream
) {
    int deviceId;
    CUDA_ASSERT(cudaGetDevice(&deviceId));
    int numSm;
    CUDA_ASSERT(cudaDeviceGetAttribute(&numSm, cudaDevAttrMultiProcessorCount, deviceId));

    size_t numChunks = (num_bytes + LE_WRITE_CHUNK_SIZE * LE_WRITE_WARPS - 1) / (LE_WRITE_CHUNK_SIZE * LE_WRITE_WARPS);
    int numBlocks = min((int)numChunks, numSm * LE_WRITE_BLK_PER_SM);
    if (numBlocks < 1) numBlocks = 1;
    char* src_ptr = reinterpret_cast<char*>(src);
    char* dst_ptr = reinterpret_cast<char*>(dst_multicast_va);
    void* args[] = { &src_ptr, &dst_ptr, &num_bytes, &loop_count };
    CUDA_ASSERT(cudaFuncSetAttribute(multimemWriteKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)FABRIC_SMEM_SIZE_WRITE));
    CUDA_ASSERT(cudaLaunchKernel((void*)multimemWriteKernel, dim3(numBlocks), dim3(LE_WRITE_THREADS),
                     args, FABRIC_SMEM_SIZE_WRITE, stream));
}

// Shared boilerplate for all device-to-device latency launchers: create streams on each device,
// launch consumer kernel, time the producer kernel, then tear down.
// Caller handles peer-access enabling and is expected to leave the current device
// set to producerDevice on entry. timingDenominator controls ns-per-event scaling
// (pass num_iterations for 1 event/iter, 2*num_iterations for 2 events/iter).
static double runDeviceToDeviceLatency(
    int producerDevice, void *producerKernel, void **producerArgs, size_t producerSmem,
    int consumerDevice, void *consumerKernel, void **consumerArgs,
    unsigned long long num_iterations, unsigned long long timingDenominator
) {
    cudaStream_t producerStream, consumerStream;
    CUDA_ASSERT(cudaSetDevice(producerDevice));
    CUDA_ASSERT(cudaStreamCreate(&producerStream));
    CUDA_ASSERT(cudaSetDevice(consumerDevice));
    CUDA_ASSERT(cudaStreamCreate(&consumerStream));
    CUDA_ASSERT(cudaLaunchKernel(consumerKernel, dim3(1), dim3(1), consumerArgs, 0, consumerStream));
    CUDA_ASSERT(cudaSetDevice(producerDevice));
    cudaEvent_t startEvent, endEvent;
    CUDA_ASSERT(cudaEventCreate(&startEvent));
    CUDA_ASSERT(cudaEventCreate(&endEvent));
    CUDA_ASSERT(cudaFuncSetAttribute(producerKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)producerSmem));
    CUDA_ASSERT(cudaEventRecord(startEvent, producerStream));
    CUDA_ASSERT(cudaLaunchKernel(producerKernel, dim3(1), dim3(1), producerArgs, producerSmem, producerStream));
    CUDA_ASSERT(cudaEventRecord(endEvent, producerStream));
    CUDA_ASSERT(cudaStreamSynchronize(producerStream));
    CUDA_ASSERT(cudaSetDevice(consumerDevice));
    CUDA_ASSERT(cudaStreamSynchronize(consumerStream));
    float elapsedMs = 0.0f;
    CUDA_ASSERT(cudaSetDevice(producerDevice));
    CUDA_ASSERT(cudaEventElapsedTime(&elapsedMs, startEvent, endEvent));
    CUDA_ASSERT(cudaEventDestroy(startEvent));
    CUDA_ASSERT(cudaEventDestroy(endEvent));
    CUDA_ASSERT(cudaStreamDestroy(producerStream));
    CUDA_ASSERT(cudaSetDevice(consumerDevice));
    CUDA_ASSERT(cudaStreamDestroy(consumerStream));
    if (timingDenominator == 0) return 0.0;
    return (double)elapsedMs * 1e6 / (double)timingDenominator;
}

// Shared body for both unicast device-to-device launchers; only seqOffset and producerKernel differ.
// seqOffset is the byte offset within consumer_buf where the consumer polls for the sequence word.
// num_iterations is also the timing denominator (1 round trip per iteration for unicast).
static double runUnicastDeviceToDeviceLatency(
    CUdeviceptr producerBuf, uint32_t consumerLeId, CUdeviceptr consumerBuf, uint64_t ackVa,
    uint32_t seqOffset, void *producerKernel,
    unsigned long long num_iterations, int producerDevice, int consumerDevice
) {
    CUDA_ASSERT(cudaSetDevice(consumerDevice));
    (void)cudaDeviceEnablePeerAccess(producerDevice, 0);
    void *producerArgs[] = { &producerBuf, &consumerLeId, &ackVa, &num_iterations };
    void *consumerArgs[] = { &consumerBuf, &seqOffset, &ackVa, &num_iterations };
    return runDeviceToDeviceLatency(
        producerDevice, producerKernel, producerArgs, FABRIC_SMEM_SIZE_LATENCY,
        consumerDevice, (void*)leFabricDeviceToDeviceConsumerMultimemKernel, consumerArgs,
        num_iterations, num_iterations);
}

// device-to-device latency unicast (non-counted): producer waits for consumer ack. Returns mean ns per round-trip.
extern "C"
double leLaunchE2ELatencyUnicast(
    CUdeviceptr producerBuf,
    unsigned int consumerLeId,
    CUdeviceptr consumerBuf,
    CUdeviceptr ackPtr,
    unsigned long long num_iterations,
    int producerDevice,
    int consumerDevice
) {
    // seqOffset = kDeviceToDeviceLatencyPutSize: put2 lands at the second slot in the consumer buffer;
    // the consumer polls that slot's sequence word to confirm both puts have arrived.
    return runUnicastDeviceToDeviceLatency(producerBuf, (uint32_t)consumerLeId, consumerBuf, (uint64_t)ackPtr,
        kDeviceToDeviceLatencyPutSize,
        (void*)leFabricDeviceToDeviceProducerUnicastKernel,
        num_iterations, producerDevice, consumerDevice);
}

// device-to-device latency unicast (counted): producer uses fabric_try_put_counted (unicast LE).
extern "C"
double leLaunchE2ELatencyUnicastCounted(
    CUdeviceptr producerBuf,
    unsigned int consumerLeId,
    CUdeviceptr consumerBuf,
    CUdeviceptr ackPtr,
    unsigned long long num_iterations,
    int producerDevice,
    int consumerDevice
) {
    // seqOffset = 2 * kDeviceToDeviceLatencyPutSize: the put lands at put_offset within the LE, so the
    // sequence word arrives at that same offset in the consumer buffer.
    return runUnicastDeviceToDeviceLatency(producerBuf, (uint32_t)consumerLeId, consumerBuf, (uint64_t)ackPtr,
        2 * kDeviceToDeviceLatencyPutSize,
        (void*)leFabricDeviceToDeviceProducerUnicastCountedKernel,
        num_iterations, producerDevice, consumerDevice);
}

/*
 * UBLKCP device-to-device: mailbox and sequence on consumerDevice (consumer reads locally); ack on
 * producerDevice (producer reads locally, consumer peer-writes). Bidirectional peer
 * access required: producer writes mailbox/sequence on consumerDevice; consumer writes
 * ack on producerDevice.
 */
extern "C"
double leLaunchE2ELatencyUblkcpGSCounted(
    CUdeviceptr src,
    CUdeviceptr mailbox,
    CUdeviceptr sequence_ptr,
    CUdeviceptr ack_ptr,
    unsigned long long num_iterations,
    int producerDevice,
    int consumerDevice
) {
    uint64_t seq_va = (uint64_t)sequence_ptr;
    uint64_t ack_va = (uint64_t)ack_ptr;
    CUDA_ASSERT(cudaSetDevice(producerDevice));
    (void)cudaDeviceEnablePeerAccess(consumerDevice, 0);  // producer -> consumer for mailbox writes
    CUDA_ASSERT(cudaSetDevice(consumerDevice));
    (void)cudaDeviceEnablePeerAccess(producerDevice, 0);  // consumer -> producer for ack writes
    void *producerArgs[] = { &src, &mailbox, &seq_va, &ack_va, &num_iterations };
    void *consumerArgs[] = { &mailbox, &seq_va, &ack_va, &num_iterations };
    double latencyNs = runDeviceToDeviceLatency(
        producerDevice, (void*)ublkcpGSCountedDeviceToDeviceProducerKernel, producerArgs, FABRIC_SMEM_SIZE_LATENCY,
        consumerDevice, (void*)ublkcpGSCountedDeviceToDeviceConsumerKernel, consumerArgs,
        num_iterations, num_iterations);
    // If the producer's watchdog fired it published a sentinel into the producer-only
    // status word (ack_ptr + kDeviceToDeviceLatencyAckStatusOffset). Read that word —
    // not the handshake ack — so a late consumer *ack=1 cannot hide the failure after
    // both streams have synchronized.
    CUDA_ASSERT(cudaSetDevice(producerDevice));
    uint32_t statusValue = 0;
    CUDA_ASSERT(cudaMemcpy(&statusValue,
        (const void *)(uintptr_t)(ack_va + kDeviceToDeviceLatencyAckStatusOffset),
        sizeof(statusValue), cudaMemcpyDefault));
    if (statusValue == (uint32_t)kDeviceToDeviceLatencyWatchdogTimeoutAck)
        return kDeviceToDeviceLatencyWatchdogFailureNs;
    return latencyNs;
}

/** UBLKCP non-counted device-to-device: 2 put + fence, round trip after each put.
 *  Mailbox (kUblkcpMailboxBytes): put1 seq at offset 0, put2 seq at kUblkcpCounterOffset. */
extern "C"
double leLaunchE2ELatencyUblkcpGSE2E(
    CUdeviceptr src,
    CUdeviceptr mailbox,
    CUdeviceptr ack_ptr,
    unsigned long long num_iterations,
    int producerDevice,
    int consumerDevice
) {
    uint64_t ack_va = (uint64_t)ack_ptr;
    CUDA_ASSERT(cudaSetDevice(producerDevice));
    (void)cudaDeviceEnablePeerAccess(consumerDevice, 0);
    CUDA_ASSERT(cudaSetDevice(consumerDevice));
    (void)cudaDeviceEnablePeerAccess(producerDevice, 0);
    void *producerArgs[] = { &src, &mailbox, &ack_va, &num_iterations };
    void *consumerArgs[] = { &mailbox, &ack_va, &num_iterations };
    // Two round trips per iteration (one after each put); divide by 2*num_iterations to report ns per put.
    return runDeviceToDeviceLatency(
        producerDevice, (void*)ublkcpGSDeviceToDeviceProducerKernel, producerArgs, FABRIC_SMEM_SIZE_LATENCY,
        consumerDevice, (void*)ublkcpGSDeviceToDeviceConsumerKernel, consumerArgs,
        num_iterations, 2 * num_iterations);
}
// Host API: Read from LE
extern "C"
void leLaunchReadKernel(
    unsigned int src_leid,
    CUdeviceptr dst,
    size_t num_bytes,
    unsigned long long loop_count,
    CUstream stream
) {
    int deviceId;
    CUDA_ASSERT(cudaGetDevice(&deviceId));
    int numSm;
    CUDA_ASSERT(cudaDeviceGetAttribute(&numSm, cudaDevAttrMultiProcessorCount, deviceId));

    size_t numChunks = (num_bytes + LE_READ_CHUNK_SIZE * LE_READ_WARPS - 1) / (LE_READ_CHUNK_SIZE * LE_READ_WARPS);
    int numBlocks = min((int)numChunks, numSm * LE_READ_BLK_PER_SM);
    if (numBlocks < 1) numBlocks = 1;
    uint32_t src_le = (uint32_t)src_leid;
    void* args[] = { &src_le, &dst, &num_bytes, &loop_count };
    CUDA_ASSERT(cudaFuncSetAttribute(leFabricReadKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)FABRIC_SMEM_SIZE_READ));
    CUDA_ASSERT(cudaLaunchKernel((void*)leFabricReadKernel, dim3(numBlocks), dim3(LE_READ_THREADS),
                     args, FABRIC_SMEM_SIZE_READ, stream));
}

// Host API: two-slot fabric.try_pullred read with intra-iteration latency hiding (sm_100+ only).
// Issues both pullred fetches before polling either barrier, then overlaps slot-0's TMA
// writeback with the wait for slot-1's arrival in smem.
void leLaunchReduceMulticastKernel(
    unsigned int src_leid,
    CUdeviceptr dst,
    size_t num_bytes,
    unsigned long long loop_count,
    CUstream stream
) {
    int deviceId;
    CUDA_ASSERT(cudaGetDevice(&deviceId));
    int numSm;
    CUDA_ASSERT(cudaDeviceGetAttribute(&numSm, cudaDevAttrMultiProcessorCount, deviceId));

    size_t numChunks = (num_bytes + (size_t)LE_PULLRED_OPT_CHUNK * LE_READ_WARPS - 1)
                       / ((size_t)LE_PULLRED_OPT_CHUNK * LE_READ_WARPS);
    uint32_t src_le = (uint32_t)src_leid;

    constexpr size_t smem_alloc = LE_PULLRED_OPT_SMEM;
    CUDA_ASSERT(cudaFuncSetAttribute((void*)leFabricReduceMulticastKernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_alloc));
    int numBlocks = min((int)numChunks, numSm * LE_PULLRED_OPT_BPS);
    if (numBlocks < 1) numBlocks = 1;

    void* args[] = { &src_le, &dst, &num_bytes, &loop_count };
    CUDA_ASSERT(cudaLaunchKernel((void*)leFabricReduceMulticastKernel,
                     dim3(numBlocks), dim3(LE_READ_THREADS), args, smem_alloc, stream));
}

// Host API: Memset on LE
extern "C"
void leLaunchMemsetKernel(
    unsigned int dst_leid,
    unsigned int value,
    size_t num_bytes,
    CUstream stream
) {
    int deviceId;
    CUDA_ASSERT(cudaGetDevice(&deviceId));
    int numSm;
    CUDA_ASSERT(cudaDeviceGetAttribute(&numSm, cudaDevAttrMultiProcessorCount, deviceId));

    size_t numChunks = (num_bytes + LE_CHUNK_SIZE - 1) / LE_CHUNK_SIZE;
    int numBlocks = min((int)numChunks, numSm * LE_WRITE_BLK_PER_SM);
    if (numBlocks < 1) numBlocks = 1;
    uint32_t dst_le = (uint32_t)dst_leid;
    void* args[] = { &dst_le, &value, &num_bytes };
    CUDA_ASSERT(cudaLaunchKernel((void*)leFabricMemsetKernel, dim3(numBlocks), dim3(LE_WRITE_THREADS),
                     args, FABRIC_SMEM_SIZE, stream));
}

