/*
 * Copyright 1993-2021 by NVIDIA Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef MEMORY_UTILS_H
#define MEMORY_UTILS_H

#include <cstddef>

// Allocate host memory (including pageable)
void *allocateHostMemory(size_t size, bool isPageable);

// Free host memory
void freeHostMemory(void *memory);

// If memory is CUDA memory
bool isMemoryOwnedByCUDA(void *memory);

/// memset_pattern set a pattern generated with 'seed' into buffer.
/// This call is using the current cuda context to perform the set
/// and it causes a ctxSynchronize
void memset_pattern(void *buffer, unsigned long long size, unsigned int seed);

/// memcmp_pattern compare the buffer with the reference pattern generated with
/// 'seed'. if the comparison fail, the call will raise a dfontaine/testsuite
/// assert This call is using the current cuda context to perform the set and it
/// causes a ctxSynchronize
void memcmp_pattern(void *buffer, unsigned long long size, unsigned int seed);

#endif
