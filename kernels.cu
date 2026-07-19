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
#if __CUDA_ARCH__ >= 900
// Compile-time compute capability checks fail without this
#include <cuda_awbarrier_primitives.h>  // __mbarrier_*
#endif

// NVLink utilization scheduling support check (CUDA 13.0+)
#if CUDART_VERSION >= 13000
    #define NVLINK_UTIL_SCHEDULING_SUPPORTED 1
#else
    #define NVLINK_UTIL_SCHEDULING_SUPPORTED 0
#endif

#if NVLINK_UTIL_SCHEDULING_SUPPORTED
// Helper function to launch kernels with NVLink util scheduling attributes
template<typename KernelFunc, typename... Args>
static cudaError_t launchKernelWithNvlinkScheduling(
    dim3 gridDim,
    dim3 blockDim,
    CUstream stream,
    KernelFunc kernel,
    Args... args) {
    cudaLaunchAttribute cudaAttr;
    cudaAttr.id = cudaLaunchAttributeNvlinkUtilCentricScheduling;
    cudaAttr.val.nvlinkUtilCentricScheduling = 1;

    cudaLaunchConfig_t config = {0};
    config.gridDim = gridDim;
    config.blockDim = blockDim;
    config.dynamicSmemBytes = 0;
    config.stream = stream;
    config.attrs = &cudaAttr;
    config.numAttrs = 1;

    return cudaLaunchKernelEx(&config, kernel, args...);
}
#endif

__global__ void simpleCopyKernel(uint4 *dst, uint4 *src) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t offset = idx * sizeof(uint4);
    uint4* dst_uint4 = reinterpret_cast<uint4*>((char*)dst + offset);
    uint4* src_uint4 = reinterpret_cast<uint4*>((char*)src + offset);
    __stcg(dst_uint4, __ldcg(src_uint4));
}

__global__ void stridingMemcpyKernel(unsigned int totalThreadCount, unsigned long long loopCount, uint4* dst, uint4* src, size_t chunkSizeInElement) {
    unsigned long long from = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned long long bigChunkSizeInElement = chunkSizeInElement / 12;
    dst += from;
    src += from;
    uint4* dstBigEnd = dst + (bigChunkSizeInElement * 12) * totalThreadCount;
    uint4* dstEnd = dst + chunkSizeInElement * totalThreadCount;

    for (unsigned int long long i = 0; i < loopCount; i++) {
        uint4* cdst = dst;
        uint4* csrc = src;

        while (cdst < dstBigEnd) {
            uint4 pipe_0 = __ldcg(csrc); csrc += totalThreadCount;
            uint4 pipe_1 = __ldcg(csrc); csrc += totalThreadCount;
            uint4 pipe_2 = __ldcg(csrc); csrc += totalThreadCount;
            uint4 pipe_3 = __ldcg(csrc); csrc += totalThreadCount;
            uint4 pipe_4 = __ldcg(csrc); csrc += totalThreadCount;
            uint4 pipe_5 = __ldcg(csrc); csrc += totalThreadCount;
            uint4 pipe_6 = __ldcg(csrc); csrc += totalThreadCount;
            uint4 pipe_7 = __ldcg(csrc); csrc += totalThreadCount;
            uint4 pipe_8 = __ldcg(csrc); csrc += totalThreadCount;
            uint4 pipe_9 = __ldcg(csrc); csrc += totalThreadCount;
            uint4 pipe_10 = __ldcg(csrc); csrc += totalThreadCount;
            uint4 pipe_11 = __ldcg(csrc); csrc += totalThreadCount;

            __stcg(cdst, pipe_0); cdst += totalThreadCount;
            __stcg(cdst, pipe_1); cdst += totalThreadCount;
            __stcg(cdst, pipe_2); cdst += totalThreadCount;
            __stcg(cdst, pipe_3); cdst += totalThreadCount;
            __stcg(cdst, pipe_4); cdst += totalThreadCount;
            __stcg(cdst, pipe_5); cdst += totalThreadCount;
            __stcg(cdst, pipe_6); cdst += totalThreadCount;
            __stcg(cdst, pipe_7); cdst += totalThreadCount;
            __stcg(cdst, pipe_8); cdst += totalThreadCount;
            __stcg(cdst, pipe_9); cdst += totalThreadCount;
            __stcg(cdst, pipe_10); cdst += totalThreadCount;
            __stcg(cdst, pipe_11); cdst += totalThreadCount;
        }

        while (cdst < dstEnd) {
            *cdst = *csrc; cdst += totalThreadCount; csrc += totalThreadCount;
        }
    }
}

// Read-only kernel - only reads from source buffer, stores sample for deterministic verification
__global__ void stridingReadOnlyKernel(unsigned int totalThreadCount, unsigned long long loopCount, uint4* src, size_t chunkSizeInElement, uint4* verifyOut) {
    unsigned long long from = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned long long bigChunkSizeInElement = chunkSizeInElement / 12;
    src += from;
    uint4* srcBigEnd = src + (bigChunkSizeInElement * 12) * totalThreadCount;
    uint4* srcEnd = src + chunkSizeInElement * totalThreadCount;

    unsigned int accumulator = 0;

    for (unsigned long long i = 0; i < loopCount; i++) {
        uint4* csrc = src;

        while (csrc < srcBigEnd) {
            uint4 pipe_0 = __ldcg(csrc); csrc += totalThreadCount;
            uint4 pipe_1 = __ldcg(csrc); csrc += totalThreadCount;
            uint4 pipe_2 = __ldcg(csrc); csrc += totalThreadCount;
            uint4 pipe_3 = __ldcg(csrc); csrc += totalThreadCount;
            uint4 pipe_4 = __ldcg(csrc); csrc += totalThreadCount;
            uint4 pipe_5 = __ldcg(csrc); csrc += totalThreadCount;
            uint4 pipe_6 = __ldcg(csrc); csrc += totalThreadCount;
            uint4 pipe_7 = __ldcg(csrc); csrc += totalThreadCount;
            uint4 pipe_8 = __ldcg(csrc); csrc += totalThreadCount;
            uint4 pipe_9 = __ldcg(csrc); csrc += totalThreadCount;
            uint4 pipe_10 = __ldcg(csrc); csrc += totalThreadCount;
            uint4 pipe_11 = __ldcg(csrc); csrc += totalThreadCount;

            // Accumulate values to prevent compiler from optimizing away reads
            accumulator += pipe_0.x + pipe_1.x + pipe_2.x + pipe_3.x;
            accumulator += pipe_4.x + pipe_5.x + pipe_6.x + pipe_7.x;
            accumulator += pipe_8.x + pipe_9.x + pipe_10.x + pipe_11.x;
        }

        while (csrc < srcEnd) {
            uint4 val = *csrc;
            csrc += totalThreadCount;
            accumulator += val.x;
        }
    }

    // Only first thread stores deterministic samples for verification
    // Store first, middle, and last elements to verify pattern
    if (verifyOut && threadIdx.x == 0 && blockIdx.x == 0) {
        uint4* baseSrc = src - from;  // Get back to base address
        size_t totalElements = chunkSizeInElement * totalThreadCount;
        size_t midIdx = totalElements / 2;
        size_t lastIdx = totalElements > 0 ? totalElements - 1 : 0;

        verifyOut[0] = baseSrc[0];  // First element
        verifyOut[1] = baseSrc[midIdx];  // Middle element
        verifyOut[2] = baseSrc[lastIdx];  // Last element
        verifyOut[3] = make_uint4(accumulator, 0, 0, 0);  // Prevent optimization
    }
}

// Write-only kernel - only writes to destination buffer, no reads
__global__ void stridingWriteOnlyKernel(unsigned int totalThreadCount, unsigned long long loopCount, uint4* __restrict__ dst, size_t chunkSizeInElement) {
    unsigned long long from = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned long long bigChunkSizeInElement = chunkSizeInElement / 12;
    dst += from;
    uint4* dstBigEnd = dst + (bigChunkSizeInElement * 12) * totalThreadCount;
    uint4* dstEnd = dst + chunkSizeInElement * totalThreadCount;

    for (unsigned long long i = 0; i < loopCount; i++) {
        uint4* cdst = dst;
        // Include loop iteration in pattern to verify all iterations completed
        uint4 pattern = make_uint4(threadIdx.x, blockIdx.x, i, 0xCAFEBABE);

        while (cdst < dstBigEnd) {
            __stcg(cdst, pattern); cdst += totalThreadCount;
            __stcg(cdst, pattern); cdst += totalThreadCount;
            __stcg(cdst, pattern); cdst += totalThreadCount;
            __stcg(cdst, pattern); cdst += totalThreadCount;
            __stcg(cdst, pattern); cdst += totalThreadCount;
            __stcg(cdst, pattern); cdst += totalThreadCount;
            __stcg(cdst, pattern); cdst += totalThreadCount;
            __stcg(cdst, pattern); cdst += totalThreadCount;
            __stcg(cdst, pattern); cdst += totalThreadCount;
            __stcg(cdst, pattern); cdst += totalThreadCount;
            __stcg(cdst, pattern); cdst += totalThreadCount;
            __stcg(cdst, pattern); cdst += totalThreadCount;
        }

        while (cdst < dstEnd) {
            __stcg(cdst, pattern);
            cdst += totalThreadCount;
        }
    }
}

// Verification kernel for write-only test - checks pattern in parallel
__global__ void verifyWritePattern(uint4* dst, size_t numElements, unsigned int numThreadPerBlock,
                                    unsigned int expectedIteration, unsigned int* errorCount,
                                    unsigned int totalThreadCount) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long stride = blockDim.x * gridDim.x;

    for (unsigned long long i = idx; i < numElements; i += stride) {
        uint4 val = dst[i];

        // The write kernel uses strided access: each thread writes to positions offset by totalThreadCount
        // So element at position i was written by thread with globalThreadId = i % totalThreadCount
        unsigned int writingGlobalThreadId = i % totalThreadCount;
        unsigned int expectedThreadIdx = writingGlobalThreadId % numThreadPerBlock;
        unsigned int expectedBlockIdx = writingGlobalThreadId / numThreadPerBlock;

        if (val.x != expectedThreadIdx ||
            val.y != expectedBlockIdx ||
            val.z != expectedIteration ||
            val.w != 0xCAFEBABE) {
            atomicAdd(errorCount, 1);
            // Only report first few errors to avoid noisy output
            if (atomicAdd(errorCount, 0) <= 5) {
                printf("ERROR at element %llu: Expected (%u,%u,%u,0xCAFEBABE) Got (%u,%u,%u,0x%08X)\n",
                       i, expectedThreadIdx, expectedBlockIdx, expectedIteration,
                       val.x, val.y, val.z, val.w);
            }
        }
    }
}

// This kernel performs a split warp copy, alternating copy directions across warps.
__global__ void splitWarpCopyKernel(unsigned long long loopCount, uint4 *dst, uint4 *src) {
    for (unsigned long long i = 0; i < loopCount; i++) {
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

__global__ void ptrChasingKernelSM(void *data, size_t size, unsigned int accesses, unsigned int targetBlock) {
    struct LatencyNode *p = static_cast<struct LatencyNode*>(data);
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

    for (unsigned long long i = 0; i < loopCount; i++) {
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

// Forward declaration for TMA pointer chasing kernel (defined later after TMA helper functions)
__global__ void ptrChasingKernelTMA(void *data, size_t size, unsigned int accesses, unsigned int targetBlock);

// Kernel launcher functions for pointer chasing
void launchPtrChasingKernelSM(CUstream stream, void* data, size_t size,
                               unsigned int accesses, unsigned int targetBlock, unsigned int smCount) {
    ptrChasingKernelSM <<<smCount, 1, 0, stream>>> (data, size, accesses, targetBlock);
}

void launchPtrChasingKernelTMA(CUstream stream, void* data, size_t size,
                                unsigned int accesses, unsigned int targetBlock, unsigned int smCount) {
    ptrChasingKernelTMA <<<smCount, 1, 0, stream>>> (data, size, accesses, targetBlock);
}

double latencyPtrChaseKernel(const int srcId, void* data, size_t size, unsigned long long latencyMemAccessCnt,
                              unsigned smCount, PtrChasingKernel kernel) {
    CUstream stream;
    double latencySum = 0.0f, finalLatencyPerAccessNs = 0.0;
    CUcontext srcCtx;
    cudaEvent_t start, end;
    float latencyMs = 0;

    CU_ASSERT(cuDevicePrimaryCtxRetain(&srcCtx, srcId));
    CU_ASSERT(cuCtxSetCurrent(srcCtx));

    CUDA_ASSERT(cudaEventCreate(&start));
    CUDA_ASSERT(cudaEventCreate(&end));

    CU_ASSERT(cuStreamCreate(&stream, CU_STREAM_DEFAULT));

    for (int targetBlock = 0; targetBlock < smCount; ++targetBlock) {
        CUDA_ASSERT(cudaEventRecord(start, stream));
        kernel(stream, data, size, latencyMemAccessCnt / smCount, targetBlock, smCount);
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
    unsigned int totalThreadCount = numSm * desc.threadsPerBlock;

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

        // Use Runtime API with NVLink util scheduling if supported, otherwise standard runtime API
        bool useNvlinkUtilScheduling = NVLINK_UTIL_SCHEDULING_SUPPORTED && desc.useNvlinkUtilScheduling;

        if (useNvlinkUtilScheduling) {
#if NVLINK_UTIL_SCHEDULING_SUPPORTED
            for (unsigned long long i = 0; i < desc.loopCount; ++i) {
                CUDA_ASSERT(launchKernelWithNvlinkScheduling(grid, block, desc.stream, simpleCopyKernel, (uint4 *)desc.dst, (uint4 *)desc.src));
            }
#endif
        } else {
            // Use standard runtime API launch
            for (unsigned long long i = 0; i < desc.loopCount; ++i) {
                simpleCopyKernel <<<grid, block, 0, desc.stream>>> ((uint4 *)desc.dst, (uint4 *)desc.src);
            }
        }
        return numUint4 * sizeof(uint4);
    }

    // adjust size to elements (size is multiple of MB, so no truncation here)
    size_t sizeInElement = desc.copySize / sizeof(uint4);
    // this truncates the copy
    sizeInElement = totalThreadCount * (sizeInElement / totalThreadCount);

    size_t chunkSizeInElement = sizeInElement / totalThreadCount;

    dim3 gridDim(numSm, 1, 1);
    dim3 blockDim(desc.threadsPerBlock, 1, 1);

    // Use Runtime API with NVLink util scheduling if supported, otherwise standard runtime API
    bool useNvlinkUtilScheduling = NVLINK_UTIL_SCHEDULING_SUPPORTED && desc.useNvlinkUtilScheduling;

    if (useNvlinkUtilScheduling) {
#if NVLINK_UTIL_SCHEDULING_SUPPORTED
        CUDA_ASSERT(launchKernelWithNvlinkScheduling(gridDim, blockDim, desc.stream, stridingMemcpyKernel, totalThreadCount, desc.loopCount, (uint4 *)desc.dst, (uint4 *)desc.src, chunkSizeInElement));
#endif
    } else {
        // Use standard runtime API launch
        stridingMemcpyKernel<<<gridDim, blockDim, 0, desc.stream>>> (totalThreadCount, desc.loopCount, (uint4 *)desc.dst, (uint4 *)desc.src, chunkSizeInElement);
    }

    return sizeInElement * sizeof(uint4);
}

size_t readOnlyKernel(MemcpyDescriptor &desc, CUdeviceptr* d_crc_out) {
    CUdevice dev;
    CUcontext ctx;

    CU_ASSERT(cuStreamGetCtx(desc.stream, &ctx));
    CU_ASSERT(cuCtxGetDevice(&dev));

    int numSm;
    CU_ASSERT(cuDeviceGetAttribute(&numSm, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev));
    unsigned int totalThreadCount = numSm * numThreadPerBlock;

    // adjust size to elements (size is multiple of MB, so no truncation here)
    size_t sizeInElement = desc.copySize / sizeof(uint4);
    // this truncates the copy
    sizeInElement = totalThreadCount * (sizeInElement / totalThreadCount);

    size_t chunkSizeInElement = sizeInElement / totalThreadCount;

    dim3 gridDim(numSm, 1, 1);
    dim3 blockDim(numThreadPerBlock, 1, 1);

    // Allocate verification buffer only if caller wants it
    // We need 4 uint4 elements: [0]=first sample, [1]=middle sample, [2]=last sample, [3]=accumulator
    uint4* d_verify = nullptr;
    if (d_crc_out) {
        const size_t numElements = 4;
        CU_ASSERT(cuMemAlloc((CUdeviceptr*)&d_verify, sizeof(uint4) * numElements));
        // Zero-initialize: 4 uint4s = 4 elements × 4 words/element = 16 words
        CU_ASSERT(cuMemsetD32((CUdeviceptr)d_verify, 0, numElements * 4));
    }

    stridingReadOnlyKernel<<<gridDim, blockDim, 0, desc.stream>>> (totalThreadCount, desc.loopCount, (uint4 *)desc.src, chunkSizeInElement, d_verify);

    // Return the verification buffer pointer for later checking in memcmpPattern
    if (d_crc_out) {
        *d_crc_out = (CUdeviceptr)d_verify;
    }

    return sizeInElement * sizeof(uint4);
}

size_t writeOnlyKernel(MemcpyDescriptor &desc) {
    CUdevice dev;
    CUcontext ctx;

    CU_ASSERT(cuStreamGetCtx(desc.stream, &ctx));
    CU_ASSERT(cuCtxGetDevice(&dev));

    int numSm;
    CU_ASSERT(cuDeviceGetAttribute(&numSm, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev));
    unsigned int totalThreadCount = numSm * numThreadPerBlock;

    // adjust size to elements (size is multiple of MB, so no truncation here)
    size_t sizeInElement = desc.copySize / sizeof(uint4);
    // this truncates the copy
    sizeInElement = totalThreadCount * (sizeInElement / totalThreadCount);

    size_t chunkSizeInElement = sizeInElement / totalThreadCount;

    dim3 gridDim(numSm, 1, 1);
    dim3 blockDim(numThreadPerBlock, 1, 1);
    stridingWriteOnlyKernel<<<gridDim, blockDim, 0, desc.stream>>> (totalThreadCount, desc.loopCount, (uint4 *)desc.dst, chunkSizeInElement);

    return sizeInElement * sizeof(uint4);
}

// Wrapper function to launch verification kernel from C++ code
CUresult verifyWriteOnlyKernel(CUstream stream, CUdeviceptr dstBuffer, size_t numElements,
                                unsigned int numThreadPerBlock, unsigned int expectedIteration,
                                CUdeviceptr d_errorCount) {
    CUdevice dev;
    CUcontext ctx;
    CU_ASSERT(cuStreamGetCtx(stream, &ctx));
    CU_ASSERT(cuCtxGetDevice(&dev));

    int numSm;
    CU_ASSERT(cuDeviceGetAttribute(&numSm, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev));

    // Calculate totalThreadCount used by the write kernel
    unsigned int totalThreadCount = numSm * numThreadPerBlock;

    dim3 gridDim(numSm * 4, 1, 1);  // Use more blocks for better coverage
    dim3 blockDim(numThreadPerBlock, 1, 1);

    verifyWritePattern<<<gridDim, blockDim, 0, stream>>>(
        (uint4*)dstBuffer,
        numElements,
        numThreadPerBlock,
        expectedIteration,
        (unsigned int*)d_errorCount,
        totalThreadCount);

    CUDA_ASSERT(cudaGetLastError());
    return CUDA_SUCCESS;
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

    // Use Runtime API with NVLink util scheduling if supported, otherwise standard runtime API
    bool useNvlinkUtilScheduling = NVLINK_UTIL_SCHEDULING_SUPPORTED && desc.useNvlinkUtilScheduling;

    if (useNvlinkUtilScheduling) {
#if NVLINK_UTIL_SCHEDULING_SUPPORTED
        CUDA_ASSERT(launchKernelWithNvlinkScheduling(grid, block, desc.stream, splitWarpCopyKernel, desc.loopCount, (uint4 *)desc.dst, (uint4 *)desc.src));
#endif
    } else {
        // Use standard runtime API launch
        splitWarpCopyKernel <<<grid, block, 0, desc.stream>>> (desc.loopCount, (uint4 *)desc.dst, (uint4 *)desc.src);
    }

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

#if __CUDA_ARCH__ >= 900
inline __device__ __mbarrier_token_t barrier_arrive1_tx(
__mbarrier_t *barrier, uint32_t expected_tx_count
) {
    // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-arrive
    __mbarrier_token_t token;

    asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 %0, [%1], %2;"
                 : "=l"(token)
                 : "r"(static_cast<unsigned int>(__cvta_generic_to_shared(barrier))), "r"(expected_tx_count)
                 : "memory");
    return token;
}

inline __device__ bool barrier_try_wait_token(__mbarrier_t *barrier, __mbarrier_token_t token) {
    // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-test-wait-try-wait
    //
    // This function returns a bool, so that software can retry.
    //
    //  The HW only provides best-effort waiting support. The wait time is limited
    //  by the HW capability, after which a fail occurs, in which case the SW is
    //  responsible for retrying.
    int __ready;
    asm volatile("{\n\t"
                 ".reg .pred p;\n\t"
                 "mbarrier.try_wait.acquire.cta.shared::cta.b64 p, [%1], %2;\n\t"
                 "selp.b32 %0, 1, 0, p;\n\t"
                 "}"
                 : "=r"(__ready)
                 : "r"(static_cast<unsigned int>(__cvta_generic_to_shared(barrier))),
                 "l"(token)
                 : "memory");
    return __ready;
}

inline __device__ void cp_async_bulk_global_to_shared(void *dest, void * src, __mbarrier_t *barrier, int size) {
    // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk
    uint32_t smem_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(dest));
    uint64_t gmem_ptr = static_cast<uint64_t>(__cvta_generic_to_global(src));
    uint32_t smem_barrier_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(barrier));

    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
        :
        : "r"(smem_ptr),
          "l"(gmem_ptr),
          "r"(size),
          "r"(smem_barrier_ptr)
        : "memory");
}

inline __device__ void cp_async_bulk_shared_to_global(void *dest, void * src, int size) {
    // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk
    uint64_t dest_gmem_ptr = static_cast<uint64_t>(__cvta_generic_to_global(dest));
    uint32_t src_smem_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(src));

    asm volatile(
    "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;\n"
    :
    : "l"(dest_gmem_ptr),
      "r"(src_smem_ptr),
      "r"(size)
    : "memory");
}

inline __device__ void cp_async_bulk_commit_group() {
    // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-commit-group
    asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
}

inline __device__ void cp_async_bulk_wait_all_read() {
    // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-wait-group

    // The optional .read modifier indicates that the waiting has to be done until
    // all the bulk async operations in the specified bulk async-group have
    // completed reading from their source locations.
    asm volatile("cp.async.bulk.wait_group.read 0; \n" ::: "memory");
}
#endif  // __CUDA_ARCH__ >= 900

constexpr size_t smem_chunk_size = 32768;
#if __CUDA_ARCH__ >= 900
template <int chunk_size_bytes>
struct SMEMPIPE {
    alignas(16) char buf[chunk_size_bytes];
    __mbarrier_t bar;
};
#endif

__global__ void tma_memcpy_kernel(char* dest, char *src, int steps, int loopCount) {
#if __CUDA_ARCH__ >= 900

    __shared__ SMEMPIPE<smem_chunk_size> smem;
    __mbarrier_init(&smem.bar, 1);

    // We assume one thread per block here
    size_t block_stride = smem_chunk_size * size_t(blockIdx.x);
    size_t grid_stride = size_t(gridDim.x) * smem_chunk_size;

    for (unsigned long long loop = 0; loop < loopCount; loop++) {
        for (int step = 0; step < steps; step++) {
            // Thread bounds check
            size_t i = block_stride + step * grid_stride;
            if (i / smem_chunk_size >= steps) {
                break;
            }

             // TMA load
            cp_async_bulk_global_to_shared(&smem.buf[0], src + i, &smem.bar, smem_chunk_size);
            __mbarrier_token_t token = barrier_arrive1_tx(&smem.bar, smem_chunk_size);
            while (!barrier_try_wait_token(&smem.bar, token)) {}
            // Data in smem_buf is ready.

            // TMA store
            cp_async_bulk_shared_to_global(dest + i, &smem.buf[0], smem_chunk_size);
            cp_async_bulk_commit_group();
            cp_async_bulk_wait_all_read();
        }

        // This kernel doesn't provide any guarantee of dest. Explicit membar.sys + signal/wait based sync is needed to guarantee dest buffer readiness before consumption.
    }
#endif
}

__global__ void ptrChasingKernelTMA(void *data, size_t size, unsigned int accesses, unsigned int targetBlock) {
#if __CUDA_ARCH__ >= 900
    if (blockIdx.x != targetBlock) return;

    __shared__ SMEMPIPE<sizeof(struct TMALatencyNode)> smem;
    __mbarrier_init(&smem.bar, 1);

    struct TMALatencyNode* smem_aligned = reinterpret_cast<struct TMALatencyNode*>(&smem.buf[0]);

    struct TMALatencyNode *p = static_cast<struct TMALatencyNode*>(data);
    for (auto i = 0; i < accesses; ++i) {
        cp_async_bulk_global_to_shared(&smem_aligned->next, p, &smem.bar, sizeof(struct TMALatencyNode));
        __mbarrier_token_t token = barrier_arrive1_tx(&smem.bar, sizeof(struct TMALatencyNode));
        while (!barrier_try_wait_token(&smem.bar, token)) {}

        // Get the next pointer from shared memory
        p = smem_aligned->next;
    }

    // avoid compiler optimization
    if (p == nullptr) {
        __trap();
    }
#endif
}

size_t copyKernelTMA(MemcpyDescriptor &desc) {
    CUdevice dev;
    CUcontext ctx;

    CU_ASSERT(cuStreamGetCtx(desc.stream, &ctx));
    CU_ASSERT(cuCtxGetDevice(&dev));

    int numSm;
    CU_ASSERT(cuDeviceGetAttribute(&numSm, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev));

    // TMA works with 16-byte-sized chunks and 16-byte aligned addresses, check for this
    if (desc.copySize % 16 != 0 || (size_t) desc.src % 16 != 0 || (size_t) desc.dst % 16 != 0) {
        // Log error and return 0 to indicate failure
        std::cerr << "ERROR: TMA copy requires 16-byte alignment. copySize=" << desc.copySize
                  << " src=" << desc.src << " dst=" << desc.dst << std::endl;
        return 0;
    }

    // Calculate how many steps we need based on shared memory chunk size
    size_t totalSteps = (desc.copySize + smem_chunk_size - 1) / smem_chunk_size;

    tma_memcpy_kernel<<<numSm, 1, 0, desc.stream>>>(
        (char*)desc.dst,
        (char*)desc.src,
        totalSteps,
        desc.loopCount);

    return desc.copySize;
}

void preloadKernels(int deviceCount) {
    cudaFuncAttributes unused;
#ifdef MULTINODE
    // In multinode mode, only test the local GPU assigned to the process.
    const int startDevice = localDevice;
    const int endDevice = localDevice + 1;
#else
    // In single-node mode, preload kernels on all GPUs
    const int startDevice = 0;
    const int endDevice = deviceCount;
#endif
    for (int iDev = startDevice; iDev < endDevice; iDev++) {
        cudaSetDevice(iDev);
        cudaFuncGetAttributes(&unused, &stridingMemcpyKernel);
        cudaFuncGetAttributes(&unused, &spinKernelDevice);
        cudaFuncGetAttributes(&unused, &spinKernelDeviceMultistage);
        cudaFuncGetAttributes(&unused, &simpleCopyKernel);
        cudaFuncGetAttributes(&unused, &splitWarpCopyKernel);
        cudaFuncGetAttributes(&unused, &ptrChasingKernelSM);
        cudaFuncGetAttributes(&unused, &ptrChasingKernelTMA);
        cudaFuncGetAttributes(&unused, &multicastCopyKernel);
        cudaFuncGetAttributes(&unused, &memsetKernelDevice);
        cudaFuncGetAttributes(&unused, &memcmpKernelDevice);
        cudaFuncGetAttributes(&unused, &multicastMemcmpKernelDevice);
        cudaFuncGetAttributes(&unused, &tma_memcpy_kernel);
    }
}



