/*
 * Copyright 1993-2021 by NVIDIA Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef NVBANDWIDTH__COPY_KERNEL_CUH
#define NVBANDWIDTH__COPY_KERNEL_CUH

#include <cuda.h>
#include "common.h"

CUresult copyKernel(CUdeviceptr dstBuffer, CUdeviceptr srcBuffer, size_t sizeInElement, CUstream stream, unsigned long long loopCount);

#endif //NVBANDWIDTH__COPY_KERNEL_CUH
