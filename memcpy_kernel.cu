#include "memcpy_kernel.cuh"
#include "common.h"

__global__ void sm_memcpy_kernel(unsigned int totalThreadCount, unsigned long long loopCount, int4* dst, int4* src,
    unsigned long long chunkSizeInElement) {
    
    volatile unsigned long long elapsed = 0;
    volatile unsigned long long start = 0;
    unsigned long long from = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned long long bigChunkSizeInElement = chunkSizeInElement / 12;
    dst += from;
    src += from;
    int4* dstBigEnd = dst + (bigChunkSizeInElement * 12) * totalThreadCount;
    int4* dstEnd = dst + chunkSizeInElement * totalThreadCount;

    for (unsigned int i = 0; i < loopCount; i++) {
        int4* cdst = dst;
        int4* csrc = src;

        while (cdst < dstBigEnd) {
            int4 _0 = *csrc; csrc += totalThreadCount;
            int4 _1 = *csrc; csrc += totalThreadCount;
            int4 _2 = *csrc; csrc += totalThreadCount;
            int4 _3 = *csrc; csrc += totalThreadCount;
            int4 _4 = *csrc; csrc += totalThreadCount;
            int4 _5 = *csrc; csrc += totalThreadCount;
            int4 _6 = *csrc; csrc += totalThreadCount;
            int4 _7 = *csrc; csrc += totalThreadCount;
            int4 _8 = *csrc; csrc += totalThreadCount;
            int4 _9 = *csrc; csrc += totalThreadCount;
            int4 _10 = *csrc; csrc += totalThreadCount;
            int4 _11 = *csrc; csrc += totalThreadCount;

            *cdst = _0; cdst += totalThreadCount;
            *cdst = _1; cdst += totalThreadCount;
            *cdst = _2; cdst += totalThreadCount;
            *cdst = _3; cdst += totalThreadCount;
            *cdst = _4; cdst += totalThreadCount;
            *cdst = _5; cdst += totalThreadCount;
            *cdst = _6; cdst += totalThreadCount;
            *cdst = _7; cdst += totalThreadCount;
            *cdst = _8; cdst += totalThreadCount;
            *cdst = _9; cdst += totalThreadCount;
            *cdst = _10; cdst += totalThreadCount;
            *cdst = _11; cdst += totalThreadCount;
        }

        while (cdst < dstEnd) {
            *cdst = *csrc; cdst += totalThreadCount; csrc += totalThreadCount;
        }
    }
}

__global__ void sm_spin_kernel(volatile int *latch, const unsigned long long timeout_clocks) {
    register unsigned long long end_time = clock64() + timeout_clocks;
    while (!*latch) {
        if (timeout_clocks != ~0ULL && clock64() > end_time) {
            break;
        }
    }
}

namespace Kernels {
	void memcpy_kernel( int4* dst, int4* src, CUstream stream, unsigned long long size, unsigned int numThreadPerBlock,
        unsigned long long loopCount) {
        CUdevice dev;
        CUcontext ctx;
        int numSm;

        CU_ASSERT(cuStreamGetCtx(stream, &ctx));
        CU_ASSERT(cuCtxGetDevice(&dev));
        CU_ASSERT(cuDeviceGetAttribute(&numSm, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev));
        
        unsigned int totalThreadCount = numSm * numThreadPerBlock;
        unsigned long long chunkSizeInElement = size / totalThreadCount;
        
        dim3 block(numThreadPerBlock, 1, 1); 
        dim3 grid(numSm, 1, 1); 

		sm_memcpy_kernel <<<grid, block>>> (totalThreadCount, loopCount, dst, src, chunkSizeInElement);
	}

    void spin_kernel(volatile int *latch, CUstream stream) {
        const unsigned long long timeout_ns = 10000000000ULL;   // 10 seconds
        int blockSize = 1, gridSize = 1;
        int clocks_per_ms = 0;
        CUcontext ctx;
        CUdevice currentDev;

        CU_ASSERT(cuCtxGetCurrent(&ctx));
        CU_ASSERT(cuCtxGetDevice(&currentDev));
        CU_ASSERT(cuDeviceGetAttribute(&clocks_per_ms, CU_DEVICE_ATTRIBUTE_CLOCK_RATE, currentDev));

        unsigned long long timeout_clocks = (clocks_per_ms * timeout_ns) / 1000;
    
        dim3 block(blockSize, 1, 1); 
        dim3 grid(gridSize, 1, 1); 
        sm_spin_kernel<<<grid, block>>> (latch, timeout_clocks);
    }
}
