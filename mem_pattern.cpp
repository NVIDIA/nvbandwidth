#include <cuda.h>
#include <iostream>

#include "nvbw_os.h"
#include "mem_pattern.h"

static void xorshift2MBPattern(unsigned int* buffer, unsigned int seed)
{
    unsigned int oldValue = seed;
    unsigned int n = 0;
    for (n = 0; n < (1024 * 1024 * 2) / sizeof(unsigned int); n++)
    {
        unsigned int value = oldValue;
        value = value ^ (value << 13);
        value = value ^ (value >> 17);
        value = value ^ (value << 5);
        oldValue = value;
        buffer[n] = oldValue;
    }
}

void memset_pattern(void* buffer, unsigned long long size, unsigned int seed)
{
    unsigned int* pattern;
    unsigned int n = 0;
    unsigned long long _2MBchunkCount = size / (1024 * 1024 * 2);
    unsigned long long remaining = size - (_2MBchunkCount * 1024 * 1024 * 2);

    // Allocate 2MB of pattern
    cuMemHostAlloc((void**)&pattern, sizeof(char) * 1024 * 1024 * 2, CU_MEMHOSTALLOC_PORTABLE); // ASSERT_DRV
    xorshift2MBPattern(pattern, seed);

    for (n = 0; n < _2MBchunkCount; n++)
    {
        cuMemcpy((CUdeviceptr)buffer, (CUdeviceptr)pattern, 1024 * 1024 * 2); // ASSERT_DRV
        buffer = (char*)buffer + (1024 * 1024 * 2);
    }
    if (remaining) {
        cuMemcpy((CUdeviceptr)buffer, (CUdeviceptr)pattern, (size_t)remaining); // ASSERT_DRV
    }

    cuCtxSynchronize(); // ASSERT_DRV
    cuMemFreeHost((void*)pattern); // ASSERT_DRV
}

void memcmp_pattern(void* buffer, unsigned long long size, unsigned int seed)
{
    unsigned int* devicePattern;
    unsigned int* pattern;
    unsigned long long _2MBchunkCount = size / (1024 * 1024 * 2);
    unsigned long long remaining = size - (_2MBchunkCount * 1024 * 1024 * 2);
    unsigned int n = 0;
    unsigned int x = 0;
    void* cpyBuffer = buffer;

    // Allocate 2MB of pattern
    cuMemHostAlloc((void**)&devicePattern, sizeof(char) * 1024 * 1024 * 2, CU_MEMHOSTALLOC_PORTABLE); // ASSERT_DRV
    pattern = (unsigned int*)malloc(sizeof(char) * 1024 * 1024 * 2);
    xorshift2MBPattern(pattern, seed);

    for (n = 0; n < _2MBchunkCount; n++)
    {
        cuMemcpy((CUdeviceptr)devicePattern, (CUdeviceptr)buffer, 1024 * 1024 * 2); // ASSERT_DRV
        cuCtxSynchronize(); // ASSERT_DRV
        if(memcmp(pattern, devicePattern, 1024 * 1024 * 2) != 0)
        {
            for (x = 0; x < (1024 * 1024 * 2) / sizeof(unsigned int); x++)
            {
                if (devicePattern[x] != pattern[x]) std::cout << " Invalid value when checking the pattern at <" << (void*)((char*)buffer + n * (1024 * 1024 * 2) + x * sizeof(unsigned int)) << ">" << std::endl
                                                        << " Current offset [ " << (unsigned long long)((char*)buffer - (char*)cpyBuffer) + (unsigned long long)(x * sizeof(unsigned int)) << "/" << (size) << "]" << std::endl; // ASSERT_EQ
            }
        }

        buffer = (char*)buffer + (1024 * 1024 * 2);
    }
    if (remaining)
    {
        cuMemcpy((CUdeviceptr)devicePattern, (CUdeviceptr)buffer, (size_t)remaining); // ASSERT_DRV
        if (memcmp(pattern, devicePattern, (size_t)remaining) != 0)
        {
            for (x = 0; x < remaining / sizeof(unsigned int); x++)
            {
                if (devicePattern[x] != pattern[x]) std::cout << " Invalid value when checking the pattern at <" << (void*)((char*)buffer + n * (1024 * 1024 * 2) + x * sizeof(unsigned int)) << ">" << std::endl
                                                        << " Current offset [ " << (unsigned long long)((char*)buffer - (char*)cpyBuffer) + (unsigned long long)(x * sizeof(unsigned int)) << "/" << (size) << "]" << std::endl; // ASSERT_EQ
            }
        }
    }

    cuCtxSynchronize(); // ASSERT_DRV
    cuMemFreeHost((void*)devicePattern); // ASSERT_DRV
    free(pattern);
}
