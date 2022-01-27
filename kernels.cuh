/*
 * Copyright 1993-2021 by NVIDIA Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef NVBANDWIDTH__KERNELS_CUH
#define NVBANDWIDTH__KERNELS_CUH

#include <cuda.h>
#include "common.h"

const unsigned long long DEFAULT_SPIN_KERNEL_TIMEOUT = 10000000000ULL;   // 10 seconds

CUresult copyKernel(CUdeviceptr dstBuffer, CUdeviceptr srcBuffer, size_t sizeInElement, CUstream stream, unsigned long long loopCount);
CUresult spinKernel(volatile int *latch, CUstream stream, unsigned long long timeoutNs = DEFAULT_SPIN_KERNEL_TIMEOUT);

#endif //NVBANDWIDTH__KERNELS_CUH
