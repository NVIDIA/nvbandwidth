/*
 * Copyright 1993-2021 by NVIDIA Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#include <cuda.h>

extern "C" __global__ void striding_memcpy_kernel(unsigned int totalThreadCount, unsigned long long loopCount, uint4* dst, uint4* src, unsigned long long chunkSizeInElement) {
    volatile unsigned long long elapsed = 0;
    volatile unsigned long long start = 0;
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
