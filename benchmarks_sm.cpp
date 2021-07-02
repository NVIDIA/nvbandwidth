#include <iostream>
#include <vector>
#include <iomanip>
#include <string.h>

#include "memory_utils.h"
#include "benchmarks.h"


static bool ignore_memset = true;

static void sm_xorshift2MBPattern(unsigned int* buffer, unsigned int seed)
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

void sm_memset_pattern(void* buffer, unsigned long long size, unsigned int seed)
{
    // if (ignore_memset) return; // TODO : SWITCH

    unsigned int* pattern;
    unsigned int n = 0;
    unsigned long long _2MBchunkCount = size / (1024 * 1024 * 2);
    unsigned long long remaining = size - (_2MBchunkCount * 1024 * 1024 * 2);

    // Allocate 2MB of pattern
    cuAssert(cuMemHostAlloc((void**)&pattern, sizeof(char) * 1024 * 1024 * 2, CU_MEMHOSTALLOC_PORTABLE), "cuMemHostAlloc failed.");
    sm_xorshift2MBPattern(pattern, seed);

    for (n = 0; n < _2MBchunkCount; n++)
    {
        cuAssert(cuMemcpy((CUdeviceptr)buffer, (CUdeviceptr)pattern, 1024 * 1024 * 2), "cuMemcpy failed.");
        buffer = (char*)buffer + (1024 * 1024 * 2);
    }
    if (remaining) {
        cuAssert(cuMemcpy((CUdeviceptr)buffer, (CUdeviceptr)pattern, (size_t)remaining), "cuMemcpy failed.");
    }

    cuAssert(cuCtxSynchronize(), "cuCtxSynchronize failed.");
    cuAssert(cuMemFreeHost((void*)pattern), "cuMemFreeHost failed.");
}

void sm_memcmp_pattern(void* buffer, unsigned long long size, unsigned int seed)
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
    sm_xorshift2MBPattern(pattern, seed);

    for (n = 0; n < _2MBchunkCount; n++)
    {
        cuMemcpy((CUdeviceptr)devicePattern, (CUdeviceptr)buffer, 1024 * 1024 * 2); // ASSERT_DRV
        cuCtxSynchronize(); // ASSERT_DRV

/* TODO : IGNORING TO SEE WHAT HAPPENS
        int cmp = memcmp(pattern, devicePattern, 1024 * 1024 * 2);
        if(cmp != 0)
        {
            for (x = 0; x < (1024 * 1024 * 2) / sizeof(unsigned int); x++)
            {
                // TODO : FIX was ASSERT_EQ
                if (devicePattern[x] != pattern[x]) {
                    std::cout << " Invalid value when checking the pattern at <" << (void*)((char*)buffer + n * (1024 * 1024 * 2) + x * sizeof(unsigned int)) << ">" << std::endl                                                        
                        << " Current offset [ " << (unsigned long long)((char*)buffer - (char*)cpyBuffer) + (unsigned long long)(x * sizeof(unsigned int)) << "/" << (size) << "]" << std::endl;
                    throw 1; // TODO : FIX was ASSERT_EQ
                }
            }
        }
*/
        buffer = (char*)buffer + (1024 * 1024 * 2);
    }
    if (remaining)
    {
        cuMemcpy((CUdeviceptr)devicePattern, (CUdeviceptr)buffer, (size_t)remaining); // ASSERT_DRV
    }

    cuCtxSynchronize(); // ASSERT_DRV
    cuMemFreeHost((void*)devicePattern); // ASSERT_DRV
    free(pattern);
}


static void memcpyAsync(void *dstBuffer, void *srcBuffer, CUcontext srcCtx, unsigned long long sizeInElement, unsigned int numThreadPerBlock, unsigned long long* bandwidth, bool stride, unsigned long long loopCount = defaultLoopCount)
{
    CUdevice device;

    CUstream stream;
    CUevent startEvent;
    CUevent endEvent;
    unsigned long long adjustedSizeInElement;

    volatile int *blockingVar = NULL;

    cuMemHostAlloc((void **)&blockingVar, sizeof(*blockingVar), CU_MEMHOSTALLOC_PORTABLE); // ASSERT_DRV
    *blockingVar = 0;

    *bandwidth = 0;

    cuCtxSetCurrent(srcCtx); // ASSERT_DRV
    cuCtxGetDevice(&device); // ASSERT_DRV

    int multiProcessorCount;
    cuDeviceGetAttribute(&multiProcessorCount, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device); // ASSERT_DRV
    unsigned long long totalThreadCount = (unsigned long long)(multiProcessorCount * numThreadPerBlock);
    adjustedSizeInElement = totalThreadCount * (sizeInElement / totalThreadCount);

    cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING); // ASSERT_DRV
    cuEventCreate(&startEvent, CU_EVENT_DEFAULT); // ASSERT_DRV
    cuEventCreate(&endEvent, CU_EVENT_DEFAULT); // ASSERT_DRV

    // TODO : Spin kernel stuff; launch_spin_kernel(blockingVar, stream, true); // ASSERT_DRV

    // launch the memcpy twice first as a warmup
    // TODO : Spin kernel stuff; memcpy_kernel((int *) dstBuffer, (int *) srcBuffer, stream, adjustedSizeInElement, numThreadPerBlock, stride, 4); // ASSERT_DRV

    cuEventRecord(startEvent, stream); // ASSERT_DRV

    cuStreamWaitEvent(stream, startEvent, 0); // ASSERT_DRV
    cuEventRecord(startEvent, stream); // ASSERT_DRV

    // TODO : Spin kernel stuff; memcpy_kernel((int *) dstBuffer, (int *) srcBuffer, stream, adjustedSizeInElement, numThreadPerBlock, stride, loopCount); // ASSERT_DRV
    cuEventRecord(endEvent, stream); // ASSERT_DRV

    *blockingVar = 1;

    cuStreamSynchronize(stream); // ASSERT_DRV

    float timeWithEvents = 0.0f;
    cuCtxSetCurrent(srcCtx); // ASSERT_DRV
    cuEventElapsedTime(&timeWithEvents, startEvent, endEvent); // ASSERT_DRV
    unsigned long long elapsedWithEventsInUs = (unsigned long long)(timeWithEvents * 1000.0f);

    *bandwidth += (adjustedSizeInElement * sizeof(int) * loopCount * 1000ull * 1000ull) / elapsedWithEventsInUs; // Bandwidth in Bytes per second

    cuMemcpy((CUdeviceptr)(((int *) dstBuffer) + adjustedSizeInElement),
                        (CUdeviceptr)(((int *) srcBuffer) + adjustedSizeInElement),
                        (size_t)((sizeInElement - adjustedSizeInElement) * sizeof(int))); // ASSERT_DRV
    cuCtxSynchronize(); // ASSERT_DRV
    cuStreamDestroy(stream); // ASSERT_DRV

    cuMemFreeHost((void*)blockingVar); // ASSERT_DRV
}

static void memcpyAsync_and_check(void *dstBuffer, void *srcBuffer, CUcontext srcCtx, unsigned long long sizeInElement, unsigned int numThreadPerBlock, unsigned long long* bandwidth, bool stride, unsigned long long loopCount = defaultLoopCount)
{
    CUdevice device;
    int kernelTimeout = 0;
    cuCtxGetDevice(&device); // ASSERT_DRV
    cuDeviceGetAttribute(&kernelTimeout, CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT, device); // ASSERT_DRV
    if (kernelTimeout)
    {
        double timeout = 1.8;
        unsigned long long expectedBandwidth = 0;
        unsigned long long smallSizeInElement = sizeInElement > 1024 * 1024 * 128 ? 1024 * 1024 * 128 : sizeInElement;
        
        sm_memset_pattern(srcBuffer, smallSizeInElement * sizeof(int), 0xCAFEBABE);
        sm_memset_pattern(dstBuffer, smallSizeInElement * sizeof(int), 0xBAADF00D);
        
        memcpyAsync(dstBuffer, srcBuffer, srcCtx, smallSizeInElement, numThreadPerBlock, &expectedBandwidth, stride, 1);
        sm_memcmp_pattern(dstBuffer, smallSizeInElement * sizeof(int), 0xCAFEBABE);

        unsigned long long maxBytes = (unsigned long long)((double)expectedBandwidth * timeout * 0.25);
        unsigned long long maxLoopcount = maxBytes / (sizeInElement * sizeof(int));
        maxLoopcount = maxLoopcount == 0 ? 1 : maxLoopcount;
        if (maxLoopcount < loopCount)
        {
            loopCount = maxLoopcount;
            if (maxLoopcount == 1 && maxBytes < (sizeInElement * sizeof(int)))
            {
                sizeInElement = maxBytes / (sizeInElement * sizeof(int));
                if (sizeInElement == 0)
                {
                    *bandwidth = 0;
                    return;
                }
            }
        }
    }
    
    sm_memset_pattern(srcBuffer, sizeInElement * sizeof(int), 0xCAFEBABE);
    sm_memset_pattern(dstBuffer, sizeInElement * sizeof(int), 0xBAADF00D);
    
    memcpyAsync(dstBuffer, srcBuffer, srcCtx, sizeInElement, numThreadPerBlock, bandwidth, stride, loopCount);
    sm_memcmp_pattern(dstBuffer, sizeInElement * sizeof(int), 0xCAFEBABE);
}


void launch_HtoD_memcpy_SM(const std::string &test_name, unsigned long long size, unsigned long long loopCount)
{

    int deviceCount = 0;
    bool stride = false;
    void* dstBuffer;
    void* srcBuffer;
    unsigned long long bandwidth;
    double device_bandwidth_sum = 0.0;

    cuDeviceGetCount(&deviceCount); // ASSERT_DRV

    std::vector<double> bandwidthValues(deviceCount);

    CUcontext srcCtx;
    int device;
    cudaGetDevice(&device);

    /* TODO :
        This piece is added because the lines below allocating portable host memory would procude a 
        201 (CUDA_ERROR_INVALID_CONTEXT) which is due no context being set. This is defaulting to the current
        context of the first device (devices[0])
        I am taking note of this because I am uncertain on this.
    */
    cuDevicePrimaryCtxRetain(&srcCtx, device);
    cuCtxSetCurrent(srcCtx);

    cuAssert(cuMemHostAlloc(&srcBuffer, (size_t)size, CU_MEMHOSTALLOC_PORTABLE), "cuMemHostAlloc failed.");
    for (int currentDevice = 0; currentDevice < deviceCount; currentDevice++)
    {
        cuDevicePrimaryCtxRetain(&srcCtx, currentDevice); // ASSERT_DRV
        cuCtxSetCurrent(srcCtx); // ASSERT_DRV

        cuMemAlloc((CUdeviceptr*)&dstBuffer, (size_t)size); // ASSERT_DRV

        unsigned long long device_current_bandwidth = 0;
        unsigned long long bandwidth_current;
        unsigned long long bandwidth_sum;

        unsigned int num_threads_per_sm = 512;

        bandwidth_sum = 0;
        for (unsigned int n = 0; n < loopCount; n++) {
            memcpyAsync_and_check(dstBuffer, srcBuffer, srcCtx, size / sizeof(int), (unsigned int)num_threads_per_sm, &bandwidth_current, stride, loopCount);
            bandwidth_sum += bandwidth_current;
        }
        bandwidth_sum /= loopCount;

        if (bandwidth_sum > device_current_bandwidth) device_current_bandwidth = bandwidth_sum;
        
        bandwidthValues[currentDevice] = bandwidth * 1e-9;
        device_bandwidth_sum += bandwidth * 1e-9;

        cuMemFree((CUdeviceptr)dstBuffer); // ASSERT_DRV
        cuDevicePrimaryCtxRelease(currentDevice); // ASSERT_DRV
    }
    cuMemFreeHost(srcBuffer); // ASSERT_DRV

    std::cout << "memcpy SM GPU <- CPU bandwidth (GB/s):" << std::endl;
    printIndexVector(std::cout << std::fixed << std::setprecision(2), bandwidthValues);
    std::string bandwidth_sum_name = test_name + "_sum";
    // testutils::performance::addPerfValue(bandwidth_sum_name.c_str(), bandwidth_sum, "GB/s", true);
}
