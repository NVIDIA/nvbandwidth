#pragma once
#include <cstdio>
#include <cuda.h>

namespace Kernels {
    void memcpy_kernel(int4* dst, int4* src, CUstream stream, unsigned long long size, unsigned int numThreadPerBlock,
        unsigned long long loopCount);
    void spin_kernel(volatile int *latch, CUstream stream);
}
