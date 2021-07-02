#include <iostream>
#include <vector>
#include <iomanip>
#include <string.h>

#include "memory_utils.h"
#include "benchmarks.h"

static void memcpyAsync(std::vector<BenchParams> params, unsigned long long sizeInElement, unsigned int numThreadPerBlock, unsigned long long* bandwidth, bool stride, unsigned long long loopCount = defaultLoopCount)
{
    CUdevice device;

    std::vector<CUstream> stream(params.size());
    std::vector<CUevent> startEvent(params.size());
    std::vector<CUevent> endEvent(params.size());
    std::vector<unsigned long long> adjustedSizeInElement(params.size());

    volatile int *blockingVar = NULL;

    cuAssert(cuMemHostAlloc((void **)&blockingVar, sizeof(*blockingVar), CU_MEMHOSTALLOC_PORTABLE));
    *blockingVar = 0;

    *bandwidth = 0;

    for (unsigned i = 0; i < params.size(); ++i) {
        cuAssert(cuCtxSetCurrent(params[i].ctx));
        cuAssert(cuCtxGetDevice(&device));

        int multiProcessorCount;
        cuAssert(cuDeviceGetAttribute(&multiProcessorCount, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device));
        unsigned long long totalThreadCount = (unsigned long long)(multiProcessorCount * numThreadPerBlock);
        adjustedSizeInElement[i] = totalThreadCount * (sizeInElement / totalThreadCount);

        cuAssert(cuStreamCreate(&stream[i], CU_STREAM_NON_BLOCKING));
        cuAssert(cuEventCreate(&startEvent[i], CU_EVENT_DEFAULT));
        cuAssert(cuEventCreate(&endEvent[i], CU_EVENT_DEFAULT));
    }

    cuAssert(cuEventRecord(startEvent[0], stream[0]));
    for (unsigned i = 1; i < params.size(); ++i) {
        // ensuring that all copies are launched at the same time
        cuAssert(cuStreamWaitEvent(stream[i], startEvent[0], 0));
        cuAssert(cuEventRecord(startEvent[i], stream[i]));
    }

    for (unsigned i = 0; i < params.size(); ++i) {
        cuAssert(cuEventRecord(endEvent[i], stream[i]));
    }

    *blockingVar = 1;

    for (unsigned i = 0; i < params.size(); ++i) {
        cuAssert(cuStreamSynchronize(stream[i]));
    }

    for (unsigned i = 0; i < params.size(); ++i) {
        float timeWithEvents = 0.0f;
        cuAssert(cuCtxSetCurrent(params[i].ctx));
        cuAssert(cuEventElapsedTime(&timeWithEvents, startEvent[i], endEvent[i]));
        unsigned long long elapsedWithEventsInUs = (unsigned long long)(timeWithEvents * 1000.0f);

        *bandwidth += (adjustedSizeInElement[i] * sizeof(int) * loopCount * 1000ull * 1000ull) / elapsedWithEventsInUs; // Bandwidth in Bytes per second

        cuAssert(cuMemcpy((CUdeviceptr)(((int *) params[i].dst) + adjustedSizeInElement[i]),
                            (CUdeviceptr)(((int *) params[i].src) + adjustedSizeInElement[i]),
                            (size_t)((sizeInElement - adjustedSizeInElement[i]) * sizeof(int))));
        cuAssert(cuCtxSynchronize());
        cuAssert(cuStreamDestroy(stream[i]));
    }

    cuAssert(cuMemFreeHost((void*)blockingVar));
}

static void memcpyAsync_and_check(std::vector<BenchParams> params, unsigned long long sizeInElement, unsigned int numThreadPerBlock, unsigned long long* bandwidth, bool stride, unsigned long long loopCount = defaultLoopCount)
{
    CUdevice device;
    int kernelTimeout = 0;
    cuAssert(cuCtxGetDevice(&device));
    cuAssert(cuDeviceGetAttribute(&kernelTimeout, CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT, device));
    if (kernelTimeout)
    {
        double timeout = 1.8;
        unsigned long long expectedBandwidth = 0;
        unsigned long long smallSizeInElement = sizeInElement > 1024 * 1024 * 128 ? 1024 * 1024 * 128 : sizeInElement;
        
        for (unsigned i = 0; i < params.size(); ++i) {
            memset_pattern(params[i].src, smallSizeInElement * sizeof(int), 0xCAFEBABE);
            memset_pattern(params[i].dst, smallSizeInElement * sizeof(int), 0xBAADF00D);
        }
        
        memcpyAsync(params, smallSizeInElement, numThreadPerBlock, &expectedBandwidth, stride, 1);
        for (unsigned i = 0; i < params.size(); ++i) {
            memcmp_pattern(params[i].dst, smallSizeInElement * sizeof(int), 0xCAFEBABE);
        }

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
    
    for (unsigned i = 0; i < params.size(); ++i) {
        memset_pattern(params[i].src, sizeInElement * sizeof(int), 0xCAFEBABE);
        memset_pattern(params[i].dst, sizeInElement * sizeof(int), 0xBAADF00D);
    }
    memcpyAsync(params, sizeInElement, numThreadPerBlock, bandwidth, stride, loopCount);
    for (unsigned i = 0; i < params.size(); ++i) {
        memcmp_pattern(params[0].dst, sizeInElement * sizeof(int), 0xCAFEBABE);
    }
}

void launch_HtoD_memcpy_SM(const std::string &test_name, unsigned long long size, unsigned long long loopCount)
{

    int deviceCount = 0;
    bool stride = false;
    void* dstBuffer;
    void* srcBuffer;
    unsigned long long bandwidth;
    double device_bandwidth_sum = 0.0;

    cuAssert(cuDeviceGetCount(&deviceCount));

    std::vector<double> bandwidthValues(deviceCount);

    CUcontext srcCtx;
    int device;
    cudaGetDevice(&device);

    cuAssert(cuDevicePrimaryCtxRetain(&srcCtx, device));
    cuAssert(cuCtxSetCurrent(srcCtx));

    cuAssert(cuMemHostAlloc(&srcBuffer, (size_t)size, CU_MEMHOSTALLOC_PORTABLE), "cuMemHostAlloc failed.");
    for (int currentDevice = 0; currentDevice < deviceCount; currentDevice++)
    {
        cuAssert(cuDevicePrimaryCtxRetain(&srcCtx, currentDevice));
        cuAssert(cuCtxSetCurrent(srcCtx));
        cuAssert(cuMemAlloc((CUdeviceptr*)&dstBuffer, (size_t)size));

        std::vector<BenchParams> params;
        params.push_back(BenchParams(dstBuffer, srcBuffer, srcCtx));

        //
        unsigned long long device_current_bandwidth = 0;
        unsigned long long bandwidth_current;
        unsigned long long bandwidth_sum;

        unsigned int num_threads_per_sm = 512;

        bandwidth_sum = 0;
        for (unsigned int n = 0; n < loopCount; n++) {
            memcpyAsync_and_check(params, size / sizeof(int), (unsigned int)num_threads_per_sm, &bandwidth_current, stride, loopCount);
            bandwidth_sum += bandwidth_current;
        }
        bandwidth_sum /= loopCount;

        if (bandwidth_sum > device_current_bandwidth) device_current_bandwidth = bandwidth_sum;
        //

        bandwidthValues[currentDevice] = bandwidth * 1e-9;
        device_bandwidth_sum += bandwidth * 1e-9;

        cuAssert(cuMemFree((CUdeviceptr)dstBuffer));
        cuAssert(cuDevicePrimaryCtxRelease(currentDevice));
    }
    cuAssert(cuMemFreeHost(srcBuffer));

    std::cout << "memcpy SM GPU <- CPU bandwidth (GB/s):" << std::endl;
    printIndexVector(std::cout << std::fixed << std::setprecision(2), bandwidthValues);
}
void launch_DtoH_memcpy_SM(const std::string &test_name, unsigned long long size, unsigned long long loopCount)
{
    int deviceCount = 0;

    void* dstBuffer;
    void* srcBuffer;
    unsigned long long bandwidth;
    double bandwidth_sum = 0.0;

    cuAssert(cuDeviceGetCount(&deviceCount));

    std::vector<double> bandwidthValues(deviceCount);

    cuAssert(cuMemHostAlloc(&dstBuffer, (size_t)size, CU_MEMHOSTALLOC_PORTABLE));
    for (int currentDevice = 0; currentDevice < deviceCount; currentDevice++)
    {
        CUcontext srcCtx;

        cuAssert(cuDevicePrimaryCtxRetain(&srcCtx, currentDevice));
        cuAssert(cuCtxSetCurrent(srcCtx));

        cuAssert(cuMemAlloc((CUdeviceptr*)&srcBuffer, (size_t)size));

        std::vector<BenchParams> params;
        params.push_back(BenchParams(dstBuffer, srcBuffer, srcCtx));

        unsigned long long bandwidth_current;
        unsigned long long bandwidth_sum = 0;
        unsigned int num_threads_per_sm = 512;
        bandwidth = 0;
        for (unsigned int n = 0; n < averageLoopCount; n++)
        {
            memcpyAsync_and_check(params, size / sizeof(int), (unsigned int)num_threads_per_sm, &bandwidth_current, true, loopCount);
            bandwidth_sum += bandwidth_current;
        }
        bandwidth_sum /= averageLoopCount;
        if (bandwidth_sum > bandwidth) { bandwidth = bandwidth_sum; }
        
        cuAssert(cuMemFree((CUdeviceptr)srcBuffer));

        bandwidthValues[currentDevice] = bandwidth * 1e-9;
        bandwidth_sum += bandwidth * 1e-9;

        cuAssert(cuDevicePrimaryCtxRelease(currentDevice));
    }
    cuAssert(cuMemFreeHost(dstBuffer));

    std::cout << "memcpy SM GPU -> CPU bandwidth (GB/s):" << std::endl;
    printIndexVector(std::cout << std::fixed << std::setprecision(2), bandwidthValues);
}

void launch_DtoD_memcpy_SM(const std::string &test_name, bool read, unsigned long long size, unsigned long long loopCount)
{
    void* peerBuffer;
    void* srcBuffer;
    unsigned long long bandwidth;
    int deviceCount = 0;
    double bandwidth_sum = 0.0;
    bool anyGA100 = false;

    cuAssert(cuDeviceGetCount(&deviceCount));
    PeerValueMatrix<double> bandwidth_matrix(deviceCount);

    for (int currentDevice = 0; currentDevice < deviceCount; currentDevice++)
    {
        CUcontext srcCtx;
        cuAssert(cuDevicePrimaryCtxRetain(&srcCtx, currentDevice));
        cuAssert(cuCtxSetCurrent(srcCtx));

        cuAssert(cuCtxGetDevice(&currentDevice));

        cuAssert(cuMemAlloc((CUdeviceptr*)&srcBuffer, (size_t)size));
        for (int peer = 0; peer < deviceCount; peer++)
        {
            CUcontext peerCtx;

            if (currentDevice == peer) {
                continue;
            }

            int canAccessPeer = 0;
            cuAssert(cuDeviceCanAccessPeer(&canAccessPeer, currentDevice, peer));

            if (canAccessPeer)
            {
                cuAssert(cuDevicePrimaryCtxRetain(&peerCtx, peer));
                cuAssert(cuCtxSetCurrent(peerCtx));
                
                cuAssert(cuCtxEnablePeerAccess(srcCtx, 0));
                cuAssert(cuMemAlloc((CUdeviceptr*)&peerBuffer, (size_t)size));
                cuAssert(cuCtxSetCurrent(srcCtx));
                cuAssert(cuCtxEnablePeerAccess(peerCtx, 0));
                
                std::vector<BenchParams> params;
                if (read) params.push_back(BenchParams(srcBuffer, peerBuffer, srcCtx));
                else params.push_back(BenchParams(peerBuffer, srcBuffer, srcCtx));

                unsigned long long device_current_bandwidth = 0;
                unsigned long long bandwidth_current;
                unsigned long long bandwidth_sum;

                unsigned int num_threads_per_sm = 512;

                bandwidth_sum = 0;
                for (unsigned int n = 0; n < loopCount; n++) {
                    memcpyAsync_and_check(params, size / sizeof(int), (unsigned int)num_threads_per_sm, &bandwidth_current, true, loopCount);
                    bandwidth_sum += bandwidth_current;
                }
                bandwidth_sum /= loopCount;

                if (bandwidth_sum > device_current_bandwidth) device_current_bandwidth = bandwidth_sum;
                bandwidth_matrix.value(currentDevice, peer) = bandwidth * 1e-9;
                
                bandwidth_sum += bandwidth * 1e-9;
                cuAssert(cuCtxSetCurrent(srcCtx));
                cuAssert(cuCtxDisablePeerAccess(peerCtx));
                cuAssert(cuCtxSetCurrent(peerCtx));
                cuAssert(cuCtxDisablePeerAccess(srcCtx));
                cuAssert(cuMemFree((CUdeviceptr)peerBuffer));
                cuAssert(cuDevicePrimaryCtxRelease(peer));
            }
        }
        cuAssert(cuMemFree((CUdeviceptr)srcBuffer));
        cuAssert(cuDevicePrimaryCtxRelease(currentDevice));
    }

    std::cout << "memcpy SM GPU(row) " << (read ? "->" : "<-") << " GPU(column) bandwidth (GB/s):" << std::endl;
    std::cout << std::fixed << std::setprecision(2) << bandwidth_matrix << std::endl;
}

void launch_DtoD_memcpy_bidirectional_SM(const std::string &test_name, bool read, unsigned long long size, unsigned long long loopCount)
{
    void* gpuAbuffer0;
    void* gpuAbuffer1;
    void* gpuBbuffer0;
    void* gpuBbuffer1;

    unsigned long long bandwidth;
    int deviceCount = 0;
    double bandwidth_sum = 0.0;

    cuAssert(cuDeviceGetCount(&deviceCount));
    PeerValueMatrix<double> bandwidth_matrix(deviceCount);

    for (int currentDevice = 0; currentDevice < deviceCount; currentDevice++)
    {
        CUcontext srcCtx;
        cuAssert(cuDevicePrimaryCtxRetain(&srcCtx, currentDevice));
        cuAssert(cuCtxSetCurrent(srcCtx));

        cuAssert(cuCtxGetDevice(&currentDevice));
        cuAssert(cuMemAlloc((CUdeviceptr*)&gpuAbuffer0, (size_t)size));
        cuAssert(cuMemAlloc((CUdeviceptr*)&gpuAbuffer1, (size_t)size));

        for (int peer = 0; peer < deviceCount; peer++)
        {
            CUcontext peerCtx;
            std::vector<BenchParams> params;

            if (currentDevice == peer) {
                continue;
            }

            int canAccessPeer = 0;
            cuAssert(cuDeviceCanAccessPeer(&canAccessPeer, currentDevice, peer));

            if (canAccessPeer)
            {
                cuAssert(cuDevicePrimaryCtxRetain(&peerCtx, peer));
                cuAssert(cuCtxSetCurrent(peerCtx));

                cuAssert(cuCtxEnablePeerAccess(srcCtx, 0));
                cuAssert(cuMemAlloc((CUdeviceptr*)&gpuBbuffer0, (size_t)size));
                cuAssert(cuMemAlloc((CUdeviceptr*)&gpuBbuffer1, (size_t)size));
                cuAssert(cuCtxSetCurrent(srcCtx));
                cuAssert(cuCtxEnablePeerAccess(peerCtx, 0));

                if (read) {
                    params.push_back(BenchParams(gpuAbuffer0, gpuBbuffer0, srcCtx));
                    params.push_back(BenchParams(gpuBbuffer1, gpuAbuffer1, peerCtx));
                }
                else {
                    params.push_back(BenchParams(gpuBbuffer0, gpuAbuffer0, srcCtx));
                    params.push_back(BenchParams(gpuAbuffer1, gpuBbuffer1, peerCtx));
                }

                unsigned long long device_current_bandwidth = 0;
                unsigned long long bandwidth_current;
                unsigned long long bandwidth_sum;

                unsigned int num_threads_per_sm = 512;

                bandwidth_sum = 0;
                for (unsigned int n = 0; n < loopCount; n++) {
                    memcpyAsync_and_check(params, size / sizeof(int), (unsigned int)num_threads_per_sm, &bandwidth_current, true, loopCount);
                    bandwidth_sum += bandwidth_current;
                }
                bandwidth_sum /= loopCount;

                if (bandwidth_sum > device_current_bandwidth) device_current_bandwidth = bandwidth_sum;

                bandwidth_matrix.value(currentDevice, peer) = bandwidth * 1e-9;
                bandwidth_sum += bandwidth * 1e-9;

                cuAssert(cuCtxSetCurrent(srcCtx));
                cuAssert(cuCtxDisablePeerAccess(peerCtx));
                cuAssert(cuCtxSetCurrent(peerCtx));
                cuAssert(cuCtxDisablePeerAccess(srcCtx));

                cuAssert(cuMemFree((CUdeviceptr)gpuBbuffer0));
                cuAssert(cuMemFree((CUdeviceptr)gpuBbuffer1));
                cuAssert(cuDevicePrimaryCtxRelease(peer));
            }
        }
        cuAssert(cuMemFree((CUdeviceptr)gpuAbuffer0));
        cuAssert(cuMemFree((CUdeviceptr)gpuAbuffer1));
        cuAssert(cuDevicePrimaryCtxRelease(currentDevice));
    }
    std::cout << "memcpy SM GPU(row) " << (read ? "->" : "<-") << " GPU(column) bandwidth (GB/s):" << std::endl;
    std::cout << std::fixed << std::setprecision(2) << bandwidth_matrix << std::endl;
}
