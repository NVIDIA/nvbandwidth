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

/**
 * Hardware-multicast (multimem) kernels that do NOT depend on the Logical Endpoint API.
 *
 * multimem.ld_reduce only requires sm_90+ and hardware multicast support, so these kernels
 * must be available regardless of whether the build toolkit exposes the LE API headers
 * (NVBW_HAVE_LE_API). Compiling this file unconditionally keeps the multimem path real on
 * every supported toolkit.
 */

#include "kernels_le_tma.hpp"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <sstream>
#include <iostream>
#include "error_handling.h"
#include "multimem_ptx.cuh"

// ============================================================================
// multimemReadKernelV4F32 grid/block constants
// ============================================================================
static const int    MC_READ_THREADS = 128;
static const int    MC_READ_OPT_BPS = 8;      // optimal blkPerSm for multimem.ld_reduce.v4.f32 (ILP=8, B200)
static const size_t MC_STEP         = 4 * sizeof(uint32_t);  // bytes per multimem_ld_reduce_v4f32 call

// multimemReadKernelV4F32: 1 instruction per 16 bytes (add.v4.f32).
// Optimal kernel for multimem.ld_reduce: ILP=8, bPS=8.
// Backing memory must be filled with 1.0f per float (0x3F800000).
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
__global__ void multimemReadKernelV4F32(
    char *src_multicast_va,
    char *dst_gmem,
    size_t num_bytes,
    unsigned long long loop_count
) {
    constexpr int ILP = 8;
    const size_t tid    = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t)gridDim.x * blockDim.x * MC_STEP;
    const uintptr_t src_global = __cvta_generic_to_global(src_multicast_va);

    for (unsigned long long loop = 0; loop < loop_count; loop++) {
        size_t base = tid * MC_STEP;

        // Bulk: all ILP groups in-bounds, no branches.
        for (; base + (size_t)(ILP - 1) * stride + MC_STEP <= num_bytes; base += (size_t)ILP * stride) {
            float v[ILP][4];
            #pragma unroll
            for (int k = 0; k < ILP; k++)
                multimem_ld_reduce_v4f32(src_global + base + (size_t)k * stride,
                                         v[k][0], v[k][1], v[k][2], v[k][3]);
            #pragma unroll
            for (int k = 0; k < ILP; k++) {
                uint4 out;
                memcpy(&out.x, &v[k][0], 4); memcpy(&out.y, &v[k][1], 4);
                memcpy(&out.z, &v[k][2], 4); memcpy(&out.w, &v[k][3], 4);
                *reinterpret_cast<uint4 *>(dst_gmem + base + (size_t)k * stride) = out;
            }
        }

        // Tail: half-window cascade (4, 2, 1).
        {
            constexpr int HALF = ILP / 2;  // 4
            if (base + (size_t)(HALF - 1) * stride + MC_STEP <= num_bytes) {
                float h[HALF][4];
                #pragma unroll
                for (int k = 0; k < HALF; k++)
                    multimem_ld_reduce_v4f32(src_global + base + (size_t)k * stride,
                                             h[k][0], h[k][1], h[k][2], h[k][3]);
                #pragma unroll
                for (int k = 0; k < HALF; k++) {
                    uint4 out;
                    memcpy(&out.x, &h[k][0], 4); memcpy(&out.y, &h[k][1], 4);
                    memcpy(&out.z, &h[k][2], 4); memcpy(&out.w, &h[k][3], 4);
                    *reinterpret_cast<uint4 *>(dst_gmem + base + (size_t)k * stride) = out;
                }
                base += (size_t)HALF * stride;
            }
        }
        {
            constexpr int QRTR = ILP / 4;  // 2
            if (base + (size_t)(QRTR - 1) * stride + MC_STEP <= num_bytes) {
                float q[QRTR][4];
                #pragma unroll
                for (int k = 0; k < QRTR; k++)
                    multimem_ld_reduce_v4f32(src_global + base + (size_t)k * stride,
                                             q[k][0], q[k][1], q[k][2], q[k][3]);
                #pragma unroll
                for (int k = 0; k < QRTR; k++) {
                    uint4 out;
                    memcpy(&out.x, &q[k][0], 4); memcpy(&out.y, &q[k][1], 4);
                    memcpy(&out.z, &q[k][2], 4); memcpy(&out.w, &q[k][3], 4);
                    *reinterpret_cast<uint4 *>(dst_gmem + base + (size_t)k * stride) = out;
                }
                base += (size_t)QRTR * stride;
            }
        }
        if (base < num_bytes) {
            float s0, s1, s2, s3;
            multimem_ld_reduce_v4f32(src_global + base, s0, s1, s2, s3);
            uint4 out;
            memcpy(&out.x, &s0, 4); memcpy(&out.y, &s1, 4);
            memcpy(&out.z, &s2, 4); memcpy(&out.w, &s3, 4);
            *reinterpret_cast<uint4 *>(dst_gmem + base) = out;
        }
    }
}
#else
__global__ void multimemReadKernelV4F32(char *, char *, size_t, unsigned long long) {}
#endif

// Shared block-count computation for multimem.ld_reduce launchers.
static int multimemReadNumBlocks(int numSm, size_t num_bytes) {
    size_t numThreads = (num_bytes + MC_STEP - 1) / MC_STEP;
    int maxBlocksByData = (int)((numThreads + MC_READ_THREADS - 1) / MC_READ_THREADS);
    if (maxBlocksByData < 1) maxBlocksByData = 1;
    int n = min((int)((size_t)numSm * MC_READ_OPT_BPS), maxBlocksByData);
    return (n < 1) ? 1 : n;
}

// Host API: multimem.ld_reduce.add.v4.f32 read (ILP=8, optimal for bandwidth).
// Backing memories must be filled with 1.0f per float (0x3F800000).
void launchMultimemReadKernelV4F32(
    CUdeviceptr src_multicast_va,
    CUdeviceptr dst,
    size_t num_bytes,
    unsigned long long loop_count,
    CUstream stream
) {
    int deviceId; CUDA_ASSERT(cudaGetDevice(&deviceId));
    int numSm;    CUDA_ASSERT(cudaDeviceGetAttribute(&numSm, cudaDevAttrMultiProcessorCount, deviceId));
    int numBlocks = multimemReadNumBlocks(numSm, num_bytes);
    char *src_ptr = reinterpret_cast<char *>(src_multicast_va);
    char *dst_ptr = reinterpret_cast<char *>(dst);
    void *args[] = { &src_ptr, &dst_ptr, &num_bytes, &loop_count };
    CUDA_ASSERT(cudaLaunchKernel((void*)multimemReadKernelV4F32, dim3(numBlocks),
                    dim3(MC_READ_THREADS), args, 0, stream));
}
