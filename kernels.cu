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

// This kernel performs a split warp copy, alternating copy directions across warps.
__global__ void splitWarpCopyKernel(unsigned long long loopCount, uint4 *dst, uint4 *src) {
    for (unsigned int i = 0; i < loopCount; i++) {
        unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int globalWarpId = idx / warpSize;
        unsigned int warpLaneId = idx % warpSize;
        uint4* dst_uint4;
        uint4* src_uint4;

        // alternate copy directions across warps
        if (globalWarpId & 0x1) {
            // odd warp
            dst_uint4 = dst + (globalWarpId * warpSize + warpLaneId);
            src_uint4 = src + (globalWarpId * warpSize + warpLaneId);
        } else {
            // even warp
            dst_uint4 = src + (globalWarpId * warpSize + warpLaneId);
            src_uint4 = dst + (globalWarpId * warpSize + warpLaneId);
        }

        __stcg(dst_uint4, __ldcg(src_uint4));
    }
}

__global__ void ptrChasingKernel(struct LatencyNode *data, size_t size, unsigned int accesses, unsigned int targetBlock) {
    struct LatencyNode *p = data;
    if (blockIdx.x != targetBlock) return;
    for (auto i = 0; i < accesses; ++i) {
        p = p->next;
    }

    // avoid compiler optimization
    if (p == nullptr) {
        __trap();
    }
}

static __device__ __noinline__
void mc_st_u32(unsigned int *dst, unsigned int v) {
#if __CUDA_ARCH__ >= 900
    asm volatile ("multimem.st.u32 [%0], %1;" :: "l"(dst), "r" (v));
#endif
}

static __device__ __noinline__
void mc_ld_u32(unsigned int *dst, const unsigned int *src) {
#if __CUDA_ARCH__ >= 900
     asm volatile ("multimem.ld_reduce.and.b32 %0, [%1];" : "=r"((*dst)) : "l" (src));
#endif
}

// Writes from regular memory to multicast memory
__global__ void multicastCopyKernel(unsigned long long loopCount, unsigned int* __restrict__ dst, unsigned int* __restrict__ src, size_t nElems) {
    const size_t totalThreadCount = blockDim.x * gridDim.x;
    const size_t offset = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int* const enddst = dst + nElems;
    dst += offset;
    src += offset;

    for (unsigned int i = 0; i < loopCount; i++) {
        // Reset pointers to src and dst chunks.
        unsigned int* cur_src_ptr = src;
        unsigned int* cur_dst_ptr = dst;
        #pragma unroll(12)
        while (cur_dst_ptr < enddst) {
            mc_st_u32(cur_dst_ptr, *cur_src_ptr);
            cur_dst_ptr += totalThreadCount;
            cur_src_ptr += totalThreadCount;
        }
    }
}

double latencyPtrChaseKernel(const int srcId, void* data, size_t size, unsigned long long latencyMemAccessCnt, unsigned smCount) {
    CUstream stream;
    int device, clock_rate_khz;
    double latencySum = 0.0f, finalLatencyPerAccessNs = 0.0;
    CUcontext srcCtx;
    cudaEvent_t start, end;
    float latencyMs = 0;

    CUDA_ASSERT(cudaEventCreate(&start));
    CUDA_ASSERT(cudaEventCreate(&end));

    CU_ASSERT(cuDevicePrimaryCtxRetain(&srcCtx, srcId));
    CU_ASSERT(cuCtxSetCurrent(srcCtx));

    CU_ASSERT(cuStreamCreate(&stream, CU_STREAM_DEFAULT));
    CU_ASSERT(cuCtxGetDevice(&device));
    CU_ASSERT(cuDeviceGetAttribute(&clock_rate_khz, CU_DEVICE_ATTRIBUTE_CLOCK_RATE, device));

    for (int targetBlock = 0; targetBlock < smCount; ++targetBlock) {
        CUDA_ASSERT(cudaEventRecord(start, stream));
        ptrChasingKernel <<< smCount, 1, 0, stream>>> ((struct LatencyNode*) data, size, latencyMemAccessCnt / smCount, targetBlock);
        CUDA_ASSERT(cudaEventRecord(end, stream));
        CUDA_ASSERT(cudaGetLastError());
        CU_ASSERT(cuStreamSynchronize(stream));
        cudaEventElapsedTime(&latencyMs, start, end);
        latencySum += (latencyMs / 1000);
    }
    finalLatencyPerAccessNs = (latencySum * 1.0E9) / (latencyMemAccessCnt);

    CUDA_ASSERT(cudaEventDestroy(start));
    CUDA_ASSERT(cudaEventDestroy(end));

    return finalLatencyPerAccessNs;
}

size_t copyKernel(MemcpyDescriptor &desc) {
    CUdevice dev;
    CUcontext ctx;

    CU_ASSERT(cuStreamGetCtx(desc.stream, &ctx));
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
    if (desc.copySize < (smallBufferThreshold * _MiB)) {
        // copy size is rounded down to 16 bytes
        unsigned int numUint4 = desc.copySize / sizeof(uint4);
        // we allow max 1024 threads per block, and then scale out the copy across multiple blocks
        dim3 block(std::min(numUint4, static_cast<unsigned int>(1024)));
        dim3 grid(numUint4/block.x);
        simpleCopyKernel <<<grid, block, 0 , desc.stream>>> (desc.loopCount, (uint4 *)desc.dst, (uint4 *)desc.src);
        return numUint4 * sizeof(uint4);
    }

    // adjust size to elements (size is multiple of MB, so no truncation here)
    size_t sizeInElement = desc.copySize / sizeof(uint4);
    // this truncates the copy
    sizeInElement = totalThreadCount * (sizeInElement / totalThreadCount);

    size_t chunkSizeInElement = sizeInElement / totalThreadCount;

    dim3 gridDim(numSm, 1, 1);
    dim3 blockDim(numThreadPerBlock, 1, 1);
    stridingMemcpyKernel<<<gridDim, blockDim, 0, desc.stream>>> (totalThreadCount, desc.loopCount, (uint4 *)desc.dst, (uint4 *)desc.src, chunkSizeInElement);

    return sizeInElement * sizeof(uint4);
}

size_t copyKernelSplitWarp(MemcpyDescriptor &desc) {
    CUdevice dev;
    CUcontext ctx;

    CU_ASSERT(cuStreamGetCtx(desc.stream, &ctx));
    CU_ASSERT(cuCtxGetDevice(&dev));

    int numSm;
    CU_ASSERT(cuDeviceGetAttribute(&numSm, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev));

    // copy size is rounded down to 16 bytes
    unsigned int numUint4 = desc.copySize / sizeof(uint4);

    // we allow max 1024 threads per block, and then scale out the copy across multiple blocks
    dim3 block(std::min(numUint4, static_cast<unsigned int>(1024)));
    dim3 grid(numUint4/block.x);
    splitWarpCopyKernel <<<grid, block, 0 , desc.stream>>> (desc.loopCount, (uint4 *)desc.dst, (uint4 *)desc.src);
    return numUint4 * sizeof(uint4);
}

size_t multicastCopy(CUdeviceptr dstBuffer, CUdeviceptr srcBuffer, size_t size, CUstream stream, unsigned long long loopCount) {
    CUdevice dev;
    CUcontext ctx;

    CU_ASSERT(cuStreamGetCtx(stream, &ctx));
    CU_ASSERT(cuCtxGetDevice(&dev));

    int numSm;
    CU_ASSERT(cuDeviceGetAttribute(&numSm, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev));
    // adjust size to elements (size is multiple of MB, so no truncation here)
    size_t sizeInElement = size / sizeof(unsigned);
    dim3 gridDim(numSm, 1, 1);
    dim3 blockDim(numThreadPerBlock, 1, 1);
    multicastCopyKernel<<<gridDim, blockDim, 0, stream>>> (loopCount, (unsigned *)dstBuffer, (unsigned *)srcBuffer, sizeInElement);
    return sizeInElement * sizeof(unsigned);
}

__global__ void spinKernelDevice(volatile int *latch, const unsigned long long timeoutClocks) {
    register unsigned long long endTime = clock64() + timeoutClocks;
    while (!*latch) {
        if (timeoutClocks != ~0ULL && clock64() > endTime) {
            break;
        }
    }
}

CUresult spinKernel(volatile int *latch, CUstream stream, unsigned long long timeoutMs) {
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

__global__ void spinKernelDeviceMultistage(volatile int *latch1, volatile int *latch2, const unsigned long long timeoutClocks) {
    if (latch1) {
        register unsigned long long endTime = clock64() + timeoutClocks;
        while (!*latch1) {
            if (timeoutClocks != ~0ULL && clock64() > endTime) {
                return;
            }
        }

        *latch2 = 1;
    }

    register unsigned long long endTime = clock64() + timeoutClocks;
    while (!*latch2) {
        if (timeoutClocks != ~0ULL && clock64() > endTime) {
            break;
        }
    }
}

// Implement a 2-stage spin kernel for multi-node synchronization.
// One of the host nodes releases the first latch. Subsequently,
// the second latch is released, that is polled by all other devices
// latch1 argument is optional. If defined, kernel will spin on it until released, and then will release latch2.
// latch2 argument is mandatory. Kernel will spin on it until released.
// timeoutMs argument applies to each stage separately.
// However, since each kernel will spin on only one stage, total runtime is still limited by timeoutMs
CUresult spinKernelMultistage(volatile int *latch1, volatile int *latch2, CUstream stream, unsigned long long timeoutMs) {
    int clocksPerMs = 0;
    CUcontext ctx;
    CUdevice dev;

    ASSERT(latch2 != nullptr);

    CU_ASSERT(cuStreamGetCtx(stream, &ctx));
    CU_ASSERT(cuCtxGetDevice(&dev));

    CU_ASSERT(cuDeviceGetAttribute(&clocksPerMs, CU_DEVICE_ATTRIBUTE_CLOCK_RATE, dev));

    unsigned long long timeoutClocks = clocksPerMs * timeoutMs;

    spinKernelDeviceMultistage<<<1, 1, 0, stream>>>(latch1, latch2, timeoutClocks);

    return CUDA_SUCCESS;
}

__global__ void memsetKernelDevice(CUdeviceptr buffer, CUdeviceptr pattern, unsigned long long num_elements, unsigned int num_pattern_elements) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int* buf = reinterpret_cast<unsigned int*>(buffer);
    unsigned int* pat = reinterpret_cast<unsigned int*>(pattern);

    if (idx < num_elements) {
        buf[idx] = pat[idx % num_pattern_elements];
    }
}

// This kernel clears memory locations in the buffer based on warp parity.
// If clearOddWarpIndexed is true, it clears buffer locations indexed by odd warps.
// Otherwise, it clears buffer locations indexed by even warps.
__global__ void memclearKernelByWarpParityDevice(CUdeviceptr buffer, bool clearOddWarpIndexed) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint4* buf = reinterpret_cast<uint4*>(buffer);
    unsigned int globalWarpId = idx / warpSize;
    unsigned int thread_idx_in_warp = idx % warpSize;

    if (clearOddWarpIndexed) {
        // clear memory locations in buffer indexed by odd warps
        if (globalWarpId & 0x1) {
            buf[globalWarpId * warpSize + thread_idx_in_warp] = make_uint4(0x0, 0x0, 0x0, 0x0);
        }
    } else {
        // clear memory locations in buffer indexed by even warps
        if (!(globalWarpId & 0x1)) {
            buf[globalWarpId * warpSize + thread_idx_in_warp] = make_uint4(0x0, 0x0, 0x0, 0x0);
        }
    }
}

__global__ void memcmpKernelDevice(CUdeviceptr buffer, CUdeviceptr pattern, unsigned long long num_elements, unsigned int num_pattern_elements, CUdeviceptr errorFlag) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int* buf = reinterpret_cast<unsigned int*>(buffer);
    unsigned int* pat = reinterpret_cast<unsigned int*>(pattern);

    if (idx < num_elements) {
        if (buf[idx] != pat[idx % num_pattern_elements]) {
            if (atomicCAS((int*)errorFlag, 0, 1) == 0) {
                // have the first thread that detects a mismatch print the error message
                printf(" Invalid value when checking the pattern at %p\n", (void*)((char*)buffer));
                printf(" Current offset : %lu \n", idx);
                return;
            }
        }
    }
}

__global__ void multicastMemcmpKernelDevice(CUdeviceptr buffer, CUdeviceptr pattern, unsigned long long num_elements, unsigned int num_pattern_elements, CUdeviceptr errorFlag) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int* buf = reinterpret_cast<unsigned int*>(buffer);
    unsigned int* pat = reinterpret_cast<unsigned int*>(pattern);

    if (idx < num_elements) {
        unsigned buf_val;
        mc_ld_u32(&buf_val, &buf[idx]);
        if (buf_val != pat[idx % num_pattern_elements]) {
            if (atomicCAS((int*)errorFlag, 0, 1) == 0) {
                // have the first thread that detects a mismatch print the error message
                printf(" Invalid value when checking the pattern at %p\n", (void*)((char*)buffer));
                printf(" Current offset : %lu \n", idx);
                return;
            }
        }
    }
}

CUresult memsetKernel(CUstream stream, CUdeviceptr buffer, CUdeviceptr pattern, unsigned long long num_elements, unsigned int num_pattern_elements) {
    unsigned threadsPerBlock = 1024;
    unsigned long long blocks = (num_elements + threadsPerBlock - 1) / threadsPerBlock;

    memsetKernelDevice<<<blocks, threadsPerBlock, 0, stream>>>(buffer, pattern, num_elements, num_pattern_elements);
    CUDA_ASSERT(cudaGetLastError());
    return CUDA_SUCCESS;
}

CUresult memclearKernelByWarpParity(CUstream stream, CUdeviceptr buffer, size_t size, bool clearOddWarpIndexed) {
    CUdevice dev;
    CUcontext ctx;

    CU_ASSERT(cuStreamGetCtx(stream, &ctx));
    CU_ASSERT(cuCtxGetDevice(&dev));

    int numSm;
    CU_ASSERT(cuDeviceGetAttribute(&numSm, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev));
    // copy size is rounded down to 16 bytes
    unsigned int numUint4 = size / sizeof(uint4);

    // we allow max 1024 threads per block, and then scale out the copy across multiple blocks
    dim3 block(std::min(numUint4, static_cast<unsigned int>(1024)));

    dim3 grid(numUint4/block.x);
    memclearKernelByWarpParityDevice <<<grid, block, 0 , stream>>> (buffer, clearOddWarpIndexed);
    CUDA_ASSERT(cudaGetLastError());
    return CUDA_SUCCESS;
}

CUresult memcmpKernel(CUstream stream, CUdeviceptr buffer, CUdeviceptr pattern, unsigned long long num_elements, unsigned int num_pattern_elements, CUdeviceptr errorFlag) {
    unsigned threadsPerBlock = 1024;
    unsigned long long blocks = (num_elements + threadsPerBlock - 1) / threadsPerBlock;

    memcmpKernelDevice<<<blocks, threadsPerBlock, 0, stream>>>(buffer, pattern, num_elements, num_pattern_elements, errorFlag);
    CUDA_ASSERT(cudaGetLastError());
    return CUDA_SUCCESS;
}

CUresult multicastMemcmpKernel(CUstream stream, CUdeviceptr buffer, CUdeviceptr pattern, unsigned long long num_elements, unsigned int num_pattern_elements, CUdeviceptr errorFlag) {
    unsigned threadsPerBlock = 1024;
    unsigned long long blocks = (num_elements + threadsPerBlock - 1) / threadsPerBlock;

    multicastMemcmpKernelDevice<<<blocks, threadsPerBlock, 0, stream>>>(buffer, pattern, num_elements, num_pattern_elements, errorFlag);
    CUDA_ASSERT(cudaGetLastError());
    return CUDA_SUCCESS;
}

void preloadKernels(int deviceCount) {
    cudaFuncAttributes unused;
    for (int iDev = 0; iDev < deviceCount; iDev++) {
        cudaSetDevice(iDev);
        cudaFuncGetAttributes(&unused, &stridingMemcpyKernel);
        cudaFuncGetAttributes(&unused, &spinKernelDevice);
        cudaFuncGetAttributes(&unused, &spinKernelDeviceMultistage);
        cudaFuncGetAttributes(&unused, &simpleCopyKernel);
        cudaFuncGetAttributes(&unused, &splitWarpCopyKernel);
        cudaFuncGetAttributes(&unused, &multicastCopyKernel);
        cudaFuncGetAttributes(&unused, &ptrChasingKernel);
        cudaFuncGetAttributes(&unused, &multicastCopyKernel);
        cudaFuncGetAttributes(&unused, &memsetKernelDevice);
        cudaFuncGetAttributes(&unused, &memcmpKernelDevice);
        cudaFuncGetAttributes(&unused, &multicastMemcmpKernelDevice);
    }
}



