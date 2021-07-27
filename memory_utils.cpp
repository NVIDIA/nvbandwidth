#include <string.h>

#include "common.h"
#include "memory_utils.h"

static void xorshift2MBPattern(unsigned int *buffer, unsigned int seed) {
    unsigned int oldValue = seed;
    unsigned int n = 0;
    for (n = 0; n < _2MB / sizeof(unsigned int); n++) {
        unsigned int value = oldValue;
        value = value ^ (value << 13);
        value = value ^ (value >> 17);
        value = value ^ (value << 5);
        oldValue = value;
        buffer[n] = oldValue;
    }
}

void memset_pattern(void *buffer, unsigned long long size, unsigned int seed) {
    unsigned int *pattern;
    unsigned int n = 0;
    unsigned long long _2MBchunkCount = size / _2MB;
    unsigned long long remaining = size - (_2MBchunkCount * _2MB);

    // Allocate 2MB of pattern
    CU_ASSERT(cuMemHostAlloc((void **)&pattern, sizeof(char) * _2MB, CU_MEMHOSTALLOC_PORTABLE));
    xorshift2MBPattern(pattern, seed);

    for (n = 0; n < _2MBchunkCount; n++) {
        CU_ASSERT(cuMemcpy((CUdeviceptr)buffer, (CUdeviceptr)pattern, _2MB), "cuMemcpy failed.");
        buffer = (char *)buffer + (_2MB);
    }
    if (remaining) {
        CU_ASSERT(cuMemcpy((CUdeviceptr)buffer, (CUdeviceptr)pattern, (size_t)remaining), "cuMemcpy failed.");
    }

    CU_ASSERT(cuCtxSynchronize());
    CU_ASSERT(cuMemFreeHost((void *)pattern));
}

void memcmp_pattern(void *buffer, unsigned long long size, unsigned int seed) {
    unsigned int *devicePattern;
    unsigned int *pattern;
    unsigned long long _2MBchunkCount = size / _2MB;
    unsigned long long remaining = size - (_2MBchunkCount * _2MB);
    unsigned int n = 0;
    unsigned int x = 0;
    void *cpyBuffer = buffer;

    // Allocate 2MB of pattern
    CU_ASSERT(cuMemHostAlloc((void **)&devicePattern, sizeof(char) * _2MB, CU_MEMHOSTALLOC_PORTABLE));
    pattern = (unsigned int *)malloc(sizeof(char) * _2MB);
    xorshift2MBPattern(pattern, seed);

    for (n = 0; n < _2MBchunkCount; n++) {
        CU_ASSERT(cuMemcpy((CUdeviceptr)devicePattern, (CUdeviceptr)buffer, _2MB));
        CU_ASSERT(cuCtxSynchronize());
        if (memcmp(pattern, devicePattern, _2MB) != 0) {
            for (x = 0; x < _2MB / sizeof(unsigned int); x++) {
                if (devicePattern[x] != pattern[x] && verbose) {
                    std::cout << " Invalid value when checking the pattern at <"
                        << (void *)((char *)buffer + n * _2MB + x * sizeof(unsigned int))
                        << ">" << std::endl << " Current offset [ "
                        << (unsigned long long)((char *)buffer - (char *)cpyBuffer) + 
                        (unsigned long long)(x * sizeof(unsigned int))
                        << "/" << (size) << "]" << std::endl;
                }
            }
        }

        buffer = (char *)buffer + _2MB;
    }
    if (remaining) {
        CU_ASSERT(cuMemcpy((CUdeviceptr)devicePattern, (CUdeviceptr)buffer, (size_t)remaining));
        if (memcmp(pattern, devicePattern, (size_t)remaining) != 0) {
            for (x = 0; x < _2MB / sizeof(unsigned int); x++) {
                if (devicePattern[x] != pattern[x] && verbose) {
                    std::cout << " Invalid value when checking the pattern at <"
                        << (void *)((char *)buffer + n * _2MB + x * sizeof(unsigned int))
                        << ">" << std::endl << " Current offset [ "
                        << (unsigned long long)((char *)buffer - (char *)cpyBuffer) + 
                        (unsigned long long)(x * sizeof(unsigned int))
                        << "/" << (size) << "]" << std::endl;
                }
            }
        }
    }

    CU_ASSERT(cuCtxSynchronize());
    CU_ASSERT(cuMemFreeHost((void *)devicePattern));
    free(pattern);
}

bool isMemoryOwnedByCUDA(void *memory) {
    CUmemorytype memorytype;
    CUresult status = cuPointerGetAttribute(&memorytype, CU_POINTER_ATTRIBUTE_MEMORY_TYPE, (CUdeviceptr)memory);
    if (status == CUDA_ERROR_INVALID_VALUE) {
        return false;
    } else {
        CU_ASSERT(status);
        return true;
    }
}

void *allocateHostMemory(size_t size, bool isPageable) {
    void *memory;
    if (isPageable) {
        memory = malloc(size);
    } else {
        CU_ASSERT(cuMemHostAlloc(&memory, size, CU_MEMHOSTALLOC_PORTABLE));
    }
    return memory;
}

void freeHostMemory(void *memory) {
    if (isMemoryOwnedByCUDA(memory)) {
        CU_ASSERT(cuMemFreeHost(memory));
    } else {
        free(memory);
    }
}
