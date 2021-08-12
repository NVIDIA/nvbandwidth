#ifndef BENCHMARKS_H
#define BENCHMARKS_H

#include "common.h"
#include "memory_utils.h"
#include <map>

typedef void (*benchfn_t)(unsigned long long, unsigned long long);

class Benchmark {
    std::string key;
    std::string desc;
    benchfn_t benchmark_func;

public:
    Benchmark() {}

    Benchmark(std::string key, benchfn_t benchmark_func, std::string desc): key(key), benchmark_func(benchmark_func), desc(desc) {}

    std::string benchKey() { return key; }

    benchfn_t benchFn() { return benchmark_func; }

    std::string benchDesc() { return desc; }
};

inline void benchmark_prepare(CUcontext *ctx, int *deviceCount) {
    CU_ASSERT(cuCtxGetCurrent(ctx));
    CU_ASSERT(cuDeviceGetCount(deviceCount));
}

inline void benchmark_clean(void *srcBuffer, CUcontext *ctx, bool d2d = false, int currentDevice = 0) {
    CU_ASSERT(cuCtxSetCurrent(*ctx));
    if (d2d) {
        CU_ASSERT(cuMemFree((CUdeviceptr)srcBuffer));
        CU_ASSERT(cuDevicePrimaryCtxRelease(currentDevice));
    } else {
        freeHostMemory(srcBuffer);
    }
}

inline void benchmark_clean_bidir_h2d(CUcontext *ctx, int currentDevice, void *hostBuffer, void *gpuBuffer) {
    CU_ASSERT(cuCtxSetCurrent(*ctx));
    freeHostMemory(hostBuffer);
    freeHostMemory(gpuBuffer);
}

inline void benchmark_clean_bidir(CUcontext *ctx, int currentDevice, void *gpuBuffer0, void *gpuBuffer1) {
    CU_ASSERT(cuCtxSetCurrent(*ctx));
    CU_ASSERT(cuMemFree((CUdeviceptr)gpuBuffer0));
    CU_ASSERT(cuMemFree((CUdeviceptr)gpuBuffer1));
    CU_ASSERT(cuDevicePrimaryCtxRelease(currentDevice));
}

// CE Benchmarks
void launch_HtoD_memcpy_CE(unsigned long long size = defaultBufferSize, unsigned long long loopCount = defaultLoopCount);
void launch_DtoH_memcpy_CE(unsigned long long size = defaultBufferSize, unsigned long long loopCount = defaultLoopCount);
void launch_HtoD_memcpy_bidirectional_CE(unsigned long long size = defaultBufferSize, unsigned long long loopCount = defaultLoopCount);
void launch_DtoH_memcpy_bidirectional_CE(unsigned long long size = defaultBufferSize, unsigned long long loopCount = defaultLoopCount);
void launch_DtoD_memcpy_read_CE(unsigned long long size, unsigned long long loopCount);
void launch_DtoD_memcpy_write_CE(unsigned long long size, unsigned long long loopCount);
void launch_DtoD_memcpy_bidirectional_CE(unsigned long long size, unsigned long long loopCount);
// SM Benchmarks
void launch_HtoD_memcpy_SM(unsigned long long size, unsigned long long loopCount);
void launch_DtoH_memcpy_SM(unsigned long long size, unsigned long long loopCount);
void launch_DtoD_memcpy_read_SM(unsigned long long size, unsigned long long loopCount);
void launch_DtoD_memcpy_write_SM(unsigned long long size, unsigned long long loopCount);
void launch_DtoD_memcpy_bidirectional_read_SM(unsigned long long size, unsigned long long loopCount);
void launch_DtoD_memcpy_bidirectional_write_SM(unsigned long long size, unsigned long long loopCount);

#endif
