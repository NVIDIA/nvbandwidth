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

#ifndef KERNELS_LE_TMA_HPP_
#define KERNELS_LE_TMA_HPP_

/**
 * TMA-based Logical Endpoint kernel declarations
 *
 * Provides fabric-based LE operations on Blackwell (sm_100+).
 * For arch < Blackwell, stub implementations are used (kernels_le_tma_stub.cu);
 */

#include <cuda.h>
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

void leLaunchWriteKernel(
    CUdeviceptr src,
    unsigned int dst_leid,
    size_t num_bytes,
    unsigned long long loop_count,
    CUstream stream
);

/** E2E latency (unicast LE, non-counted): producer waits for consumer ack. Returns mean ns per round-trip. */
double leLaunchE2ELatencyUnicast(
    CUdeviceptr producerBuf,
    unsigned int consumerLeId,
    CUdeviceptr consumerBuf,
    CUdeviceptr ackPtr,
    unsigned long long num_iterations,
    int producerDevice,
    int consumerDevice
);

/** E2E latency (unicast LE, counted): fabric_try_put_counted + fabric_submit; consumer polls sequence. */
double leLaunchE2ELatencyUnicastCounted(
    CUdeviceptr producerBuf,
    unsigned int consumerLeId,
    CUdeviceptr consumerBuf,
    CUdeviceptr ackPtr,
    unsigned long long num_iterations,
    int producerDevice,
    int consumerDevice
);

/** UBLKCP non-counted e2e: cross-GPU round-trip latency using 2 put+fence per iteration
 *  (raw-pointer equivalent of the LE fabric_try_put non-counted path).
 *  Mailbox (kUblkcpMailboxBytes): put1 sequence at offset 0, put2 sequence at kUblkcpCounterOffset.
 *  Returns mean ns per put (total time / 2*num_iterations). */
double leLaunchE2ELatencyUblkcpGSE2E(
    CUdeviceptr src,
    CUdeviceptr mailbox,
    CUdeviceptr ack_ptr,
    unsigned long long num_iterations,
    int producerDevice,
    int consumerDevice
);

/** UBLKCP counted e2e: cross-GPU round-trip latency using cp.async.bulk.G.S.COUNTED per iteration
 *  (raw-pointer equivalent of the LE fabric_try_put_counted path).
 *  Mailbox (kUblkcpCountedMailboxBytes): data at offset 0, barMem counter at kUblkcpCounterOffset.
 *  The producer clears/polls barMem for counted-copy arrival (HW increments by transfer size),
 *  then writes sequence_ptr; the consumer polls sequence_ptr.
 *  ack_ptr must be kDeviceToDeviceLatencyAckBytes: [0] handshake, [1] producer-only watchdog
 *  status (host reads [1] after both streams sync so a late consumer cannot hide a timeout).
 *  Returns mean ns per iteration. */
double leLaunchE2ELatencyUblkcpGSCounted(
    CUdeviceptr src,
    CUdeviceptr mailbox,
    CUdeviceptr sequence_ptr,
    CUdeviceptr ack_ptr,
    unsigned long long num_iterations,
    int producerDevice,
    int consumerDevice
);

void leLaunchReadKernel(
    unsigned int src_leid,
    CUdeviceptr dst,
    size_t num_bytes,
    unsigned long long loop_count,
    CUstream stream
);

void leLaunchMemsetKernel(
    unsigned int dst_leid,
    unsigned int value,
    size_t num_bytes,
    CUstream stream
);

#ifdef __cplusplus
}
#endif

void leLaunchWriteMulticastKernel(
    CUdeviceptr src,
    unsigned int dst_leid,
    size_t num_bytes,
    unsigned long long loop_count,
    CUstream stream
);

// Read from multicast LE via fabric.try_pullred (add.u32 reduction, sm_100+).
// Result[k] = sum of element[k] across all N backing memories; callers must scale
// the expected pattern by N when verifying.
void leLaunchReduceMulticastKernel(
    unsigned int src_leid,
    CUdeviceptr dst,
    size_t num_bytes,
    unsigned long long loop_count,
    CUstream stream
);

void launchMultimemWriteKernel(
    CUdeviceptr src,
    CUdeviceptr dst_multicast_va,
    size_t num_bytes,
    unsigned long long loop_count,
    CUstream stream
);

// multimem.ld_reduce.add.v4.f32, ILP=8, bPS=8.
// Backing memories must be filled with 1.0f per float.
void launchMultimemReadKernelV4F32(
    CUdeviceptr src_multicast_va,
    CUdeviceptr dst,
    size_t num_bytes,
    unsigned long long loop_count,
    CUstream stream
);

#endif  // KERNELS_LE_TMA_HPP_
