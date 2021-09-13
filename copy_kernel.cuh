#ifndef NVBANDWIDTH__COPY_KERNEL_CUH
#define NVBANDWIDTH__COPY_KERNEL_CUH

#include <cuda.h>
#include "common.h"

CUresult copyKernel(CUdeviceptr dstBuffer, CUdeviceptr srcBuffer, size_t sizeInElement, CUstream stream);

#endif //NVBANDWIDTH__COPY_KERNEL_CUH
