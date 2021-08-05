#ifndef BENCHMARKS_H
#define BENCHMARKS_H

#include "common.h"
#include <map>

typedef void (*benchfn_t)(unsigned long long, unsigned long long);

class Benchmark {
    benchfn_t benchmark_func;
    std::string desc;

public:
    Benchmark() {}

    Benchmark(benchfn_t benchmark_func, std::string desc): benchmark_func(benchmark_func), desc(desc) {}

    benchfn_t bench_fn() { return benchmark_func; }

    std::string description() { return desc; }
};

inline void benchmark_prepare(CUcontext *ctx, int *deviceCount) {
    CU_ASSERT(cuCtxGetCurrent(ctx));
    CU_ASSERT(cuDeviceGetCount(deviceCount));
}

inline void benchmark_prepare_bidir(CUcontext *srcCtx, int currentDevice, void *gpuBuffer0, void *gpuBuffer1, unsigned long long size) {
    CU_ASSERT(cuDevicePrimaryCtxRetain(srcCtx, currentDevice));
    CU_ASSERT(cuCtxSetCurrent(*srcCtx));

    CU_ASSERT(cuCtxGetDevice(&currentDevice));
    CU_ASSERT(cuMemAlloc((CUdeviceptr *)&gpuBuffer0, (size_t)size));
    CU_ASSERT(cuMemAlloc((CUdeviceptr *)&gpuBuffer1, (size_t)size));
}

inline void benchmark_clean(void *srcBuffer, CUcontext *ctx, bool d2d = false, int currentDevice = 0) {
    CU_ASSERT(cuCtxSetCurrent(*ctx));
    if (d2d) {
        CU_ASSERT(cuMemFree((CUdeviceptr)srcBuffer));
        CU_ASSERT(cuDevicePrimaryCtxRelease(currentDevice));
    } else {
        CU_ASSERT(cuMemFreeHost(srcBuffer));
    }
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
