/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "kernels.cuh"

__global__ void simpleCopyKernel(unsigned long long loopCount, uint4 *dst, uint4 *src) {
    for (unsigned int i = 0; i < loopCount; i++) {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        size_t offset = idx * sizeof(uint4);
        uint4* dst_uint4 = reinterpret_cast<uint4*>((char*)dst + offset);
        uint4* src_uint4 = reinterpret_cast<uint4*>((char*)src + offset);
        __stcg(dst_uint4, __ldcg(src_uint4));
    }
}
__global__ void stridingMemcpyKernel(unsigned int totalThreadCount, unsigned long long loopCount, uint4* dst, uint4* src, size_t chunkSizeInElement) {
    unsigned long long from = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned long long bigChunkSizeInElement = chunkSizeInElement / 12;
    dst += from;
    src += from;
    uint4* dstBigEnd = dst + (bigChunkSizeInElement * 12) * totalThreadCount;
    uint4* dstEnd = dst + chunkSizeInElement * totalThreadCount;

    for (unsigned int i = 0; i < loopCount; i++) {
        uint4* cdst = dst;
        uint4* csrc = src;

        while (cdst < dstBigEnd) {
            uint4 pipe_0 = *csrc; csrc += totalThreadCount;
            uint4 pipe_1 = *csrc; csrc += totalThreadCount;
            uint4 pipe_2 = *csrc; csrc += totalThreadCount;
            uint4 pipe_3 = *csrc; csrc += totalThreadCount;
            uint4 pipe_4 = *csrc; csrc += totalThreadCount;
            uint4 pipe_5 = *csrc; csrc += totalThreadCount;
            uint4 pipe_6 = *csrc; csrc += totalThreadCount;
            uint4 pipe_7 = *csrc; csrc += totalThreadCount;
            uint4 pipe_8 = *csrc; csrc += totalThreadCount;
            uint4 pipe_9 = *csrc; csrc += totalThreadCount;
            uint4 pipe_10 = *csrc; csrc += totalThreadCount;
            uint4 pipe_11 = *csrc; csrc += totalThreadCount;

            *cdst = pipe_0; cdst += totalThreadCount;
            *cdst = pipe_1; cdst += totalThreadCount;
            *cdst = pipe_2; cdst += totalThreadCount;
            *cdst = pipe_3; cdst += totalThreadCount;
            *cdst = pipe_4; cdst += totalThreadCount;
            *cdst = pipe_5; cdst += totalThreadCount;
            *cdst = pipe_6; cdst += totalThreadCount;
            *cdst = pipe_7; cdst += totalThreadCount;
            *cdst = pipe_8; cdst += totalThreadCount;
            *cdst = pipe_9; cdst += totalThreadCount;
            *cdst = pipe_10; cdst += totalThreadCount;
            *cdst = pipe_11; cdst += totalThreadCount;
        }

        while (cdst < dstEnd) {
            *cdst = *csrc; cdst += totalThreadCount; csrc += totalThreadCount;
        }
    }
}
__global__ void ptrChasingKernel(struct LatencyNode *data, size_t size, unsigned int accesses, unsigned long long* time) {
    struct LatencyNode *p = data;
    auto start = clock64();
    for (auto i = 0; i < accesses; ++i) {
        p = p->next;
    }

    auto interval = static_cast<unsigned long long>(clock64() - start);
    atomicAdd(time, interval);

    // avoid compiler optimization
    if (p == nullptr) {
        __trap();
    }
}

double latencyPtrChaseKernel(const int srcId, void* data, size_t size, unsigned long long loopCount) {
    unsigned long long *hTime;
    CUstream stream;
    int device, clock_rate_khz;
    double latencySum = 0.0f, finalLatencyPerAccessNs = 0.0;
    CUcontext srcCtx;
    CU_ASSERT(cuDevicePrimaryCtxRetain(&srcCtx, srcId));
    CU_ASSERT(cuCtxSetCurrent(srcCtx));
    
    CU_ASSERT(cuStreamCreate(&stream, CU_STREAM_DEFAULT));
    CU_ASSERT(cuMemAllocManaged((CUdeviceptr*) &hTime, sizeof(unsigned long long), CU_MEM_ATTACH_GLOBAL));
    CU_ASSERT(cuCtxGetDevice(&device));
    CU_ASSERT(cuDeviceGetAttribute(&clock_rate_khz, CU_DEVICE_ATTRIBUTE_CLOCK_RATE, device));

    for ( int i = 0; i < loopCount; i++) {
        memset((void*)hTime, 0, sizeof(unsigned long long));
        ptrChasingKernel <<< 1, 1, 0, stream>>> ((struct LatencyNode*) data, size, latencyMemAccessCnt, hTime);
        CUDA_ASSERT(cudaGetLastError());
        CU_ASSERT(cuStreamSynchronize(stream));
        latencySum += static_cast<double>(hTime[0]) / (1.0E3 /* khz -> hz */ * static_cast<double>(clock_rate_khz));
    }
    finalLatencyPerAccessNs = (latencySum * 1.0E9) / (loopCount * latencyMemAccessCnt);
    return finalLatencyPerAccessNs;

}
size_t copyKernel(CUdeviceptr dstBuffer, CUdeviceptr srcBuffer, size_t size, CUstream stream, unsigned long long loopCount) {
    CUdevice dev;
    CUcontext ctx;

    CU_ASSERT(cuStreamGetCtx(stream, &ctx));
    CU_ASSERT(cuCtxGetDevice(&dev));

    int numSm;
    CU_ASSERT(cuDeviceGetAttribute(&numSm, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev));
    unsigned int totalThreadCount = numSm * numThreadPerBlock;

    // If the user provided buffer size is samller than default buffer size,
    // we use the simple copy kernel for our bandwidth test.
    // This is done so that no trucation of the buffer size occurs.
    // Please note that to achieve peak bandwidth, it is suggested to use the
    // default buffer size, which in turn triggers the use of the optimized
    // kernel.
    if (size < (defaultBufferSize * _MiB) ) {
        // copy size is rounded down to 16 bytes
        int numUint4 = size / sizeof(uint4);
        // we allow max 1024 threads per block, and then scale out the copy across multiple blocks
        dim3 block(std::min(numUint4, 1024));
        dim3 grid(numUint4/block.x);
        simpleCopyKernel <<<grid, block, 0 , stream>>> (loopCount, (uint4 *)dstBuffer, (uint4 *)srcBuffer);
        return numUint4 * sizeof(uint4);
    }

    // adjust size to elements (size is multiple of MB, so no truncation here)
    size_t sizeInElement = size / sizeof(uint4);
    // this truncates the copy
    sizeInElement = totalThreadCount * (sizeInElement / totalThreadCount);

    size_t chunkSizeInElement = sizeInElement / totalThreadCount;

    dim3 gridDim(numSm, 1, 1);
    dim3 blockDim(numThreadPerBlock, 1, 1);
    stridingMemcpyKernel<<<gridDim, blockDim, 0, stream>>> (totalThreadCount, loopCount, (uint4 *)dstBuffer, (uint4 *)srcBuffer, chunkSizeInElement);

    return sizeInElement * sizeof(uint4);
}

__global__ void spinKernelDevice(volatile int *latch, const unsigned long long timeoutClocks)
{
    register unsigned long long endTime = clock64() + timeoutClocks;
    while (!*latch) {
        if (timeoutClocks != ~0ULL && clock64() > endTime) {
            break;
        }
    }
}

CUresult spinKernel(volatile int *latch, CUstream stream, unsigned long long timeoutMs)
{
    int clocksPerMs = 0;
    CUcontext ctx;
    CUdevice dev;

    CU_ASSERT(cuStreamGetCtx(stream, &ctx));
    CU_ASSERT(cuCtxGetDevice(&dev));

    CU_ASSERT(cuDeviceGetAttribute(&clocksPerMs, CU_DEVICE_ATTRIBUTE_CLOCK_RATE, dev));

    unsigned long long timeoutClocks = clocksPerMs * timeoutMs;

    spinKernelDevice<<<1, 1, 0, stream>>>(latch, timeoutClocks);

    return CUDA_SUCCESS;
}

void preloadKernels(int deviceCount)
{
    cudaFuncAttributes unused;
    for (int iDev = 0; iDev < deviceCount; iDev++) {
        cudaSetDevice(iDev);
        cudaFuncGetAttributes(&unused, &stridingMemcpyKernel);
        cudaFuncGetAttributes(&unused, &spinKernelDevice);
        cudaFuncGetAttributes(&unused, &simpleCopyKernel);
    }
}

