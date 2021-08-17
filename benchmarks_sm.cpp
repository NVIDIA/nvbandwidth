#include <cstddef>
#include <iomanip>
#include <iostream>
#include <string.h>
#include <vector>

#include "benchmarks.h"
#include "common.h"
#include "memory_utils.h"

CUresult memcpy_kernel(int4* dst, int4* src, CUstream stream, unsigned long long sizeInElement, unsigned int numThreadPerBlock,
    bool stride, unsigned long long loopCount) {

    CUdevice dev;
    CUcontext ctx;
    CUfunction func;
    CUmodule cuModule;

    CU_ASSERT(cuStreamGetCtx(stream, &ctx));
    CU_ASSERT(cuCtxGetDevice(&dev));

    int numSm;
    CU_ASSERT(cuDeviceGetAttribute(&numSm, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev));
    unsigned int totalThreadCount = numSm * numThreadPerBlock;
    unsigned long long chunkSizeInElement = sizeInElement / totalThreadCount;

    if (sizeInElement % totalThreadCount != 0) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    CU_ASSERT(cuModuleLoad(&cuModule, "sm_memcpy_kernel.ptx"));
    CU_ASSERT(cuModuleGetFunction(&func, cuModule, "striding_memcpy_kernel"));

    void* params[] = {&totalThreadCount, &loopCount, &dst, &src, &chunkSizeInElement};
    return cuLaunchKernel(func, numSm, 1, 1, numThreadPerBlock, 1, 1, 0, stream, params, 0);
}

static void memcpy_sm(void *dst, void *src, CUcontext *ctx, unsigned long long sizeInElement, unsigned long long* bandwidth,
    unsigned long long loopCount = defaultLoopCount, CUcontext *peerCtx = nullptr) {

    unsigned int numThreadPerBlock = 512;
    CUdevice device;
    int kernelTimeout = 0;

    CU_ASSERT(cuCtxGetDevice(&device));
    CU_ASSERT(cuDeviceGetAttribute(&kernelTimeout, CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT, device));

    CUstream stream, streamPeer;
    CUevent startEvent, startEventPeer;
    CUevent endEvent, endEventPeer;
    unsigned long long adjustedSizeInElement;

    volatile int *blockingVar = NULL;

    CU_ASSERT(cuMemHostAlloc((void **)&blockingVar, sizeof(*blockingVar), CU_MEMHOSTALLOC_PORTABLE));
    *blockingVar = 0;

    *bandwidth = 0;

    CU_ASSERT(cuCtxSetCurrent(*ctx));
    CU_ASSERT(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
    CU_ASSERT(cuEventCreate(&startEvent, CU_EVENT_DEFAULT));
    CU_ASSERT(cuEventCreate(&endEvent, CU_EVENT_DEFAULT));

    if (peerCtx != nullptr) {
        CU_ASSERT(cuCtxSetCurrent(*ctx));
        CU_ASSERT(cuStreamCreate(&streamPeer, CU_STREAM_NON_BLOCKING));
        CU_ASSERT(cuEventCreate(&startEventPeer, CU_EVENT_DEFAULT));
        CU_ASSERT(cuEventCreate(&endEventPeer, CU_EVENT_DEFAULT));
    }

    CU_ASSERT(cuCtxGetDevice(&device));
    int multiProcessorCount;
    CU_ASSERT(cuDeviceGetAttribute(&multiProcessorCount,CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device));
    unsigned long long totalThreadCount = (unsigned long long)(multiProcessorCount * numThreadPerBlock);
    adjustedSizeInElement = totalThreadCount * (sizeInElement / totalThreadCount);

    // launch the memcpy twice first as a warmup
    CU_ASSERT(memcpy_kernel((int4 *)dst, (int4 *)src, stream, adjustedSizeInElement, numThreadPerBlock, true, loopCount));

    CU_ASSERT(cuEventRecord(startEvent, stream));

    // ensuring that all copies are launched at the same time
    CU_ASSERT(cuStreamWaitEvent(stream, startEvent, 0));
    CU_ASSERT(cuEventRecord(startEvent, stream));
    if (peerCtx != nullptr) {
        CU_ASSERT(cuStreamWaitEvent(streamPeer, startEventPeer, 0));
        CU_ASSERT(cuEventRecord(startEventPeer, streamPeer));
    }

    CU_ASSERT(memcpy_kernel((int4 *)dst, (int4 *)src, stream, adjustedSizeInElement, numThreadPerBlock, true, loopCount));
    CU_ASSERT(cuEventRecord(endEvent, stream));
    if (peerCtx != nullptr) {
        CU_ASSERT(memcpy_kernel((int4 *)src, (int4 *)dst, streamPeer, adjustedSizeInElement, numThreadPerBlock, true, loopCount));
        CU_ASSERT(cuEventRecord(endEventPeer, streamPeer));
    }

    *blockingVar = 1;

    CU_ASSERT(cuStreamSynchronize(stream));
    if (peerCtx != nullptr) {
        CU_ASSERT(cuStreamSynchronize(streamPeer));
    }

    float timeWithEvents = 0.0f;
    CU_ASSERT(cuCtxSetCurrent(*ctx));
    CU_ASSERT(cuEventElapsedTime(&timeWithEvents, startEvent, endEvent));
    unsigned long long elapsedWithEventsInUs = (unsigned long long)(timeWithEvents * 1000.0f);

    // Bandwidth in Bytes per second
    *bandwidth += (adjustedSizeInElement * sizeof(int4) * loopCount * 1000ull * 1000ull) / elapsedWithEventsInUs;

    CU_ASSERT(cuMemcpy((CUdeviceptr)(((int4 *)dst) + adjustedSizeInElement), (CUdeviceptr)(((int4 *)src) + adjustedSizeInElement),
        (size_t)((sizeInElement - adjustedSizeInElement) * sizeof(int4))));
    CU_ASSERT(cuCtxSynchronize());
    CU_ASSERT(cuStreamDestroy(stream));
    if (peerCtx != nullptr) {
        timeWithEvents = 0.0;
        CU_ASSERT(cuCtxSetCurrent(*peerCtx));
        CU_ASSERT(cuEventElapsedTime(&timeWithEvents, startEventPeer, endEventPeer));
        elapsedWithEventsInUs = (unsigned long long)(timeWithEvents * 1000.0f);

        // Bandwidth in Bytes per second
        *bandwidth += (adjustedSizeInElement * sizeof(int4) * loopCount * 1000ull * 1000ull) / elapsedWithEventsInUs;

        CU_ASSERT(cuMemcpy((CUdeviceptr)(((int4 *)src) + adjustedSizeInElement), (CUdeviceptr)(((int4 *)dst) + adjustedSizeInElement), 
            (size_t)((sizeInElement - adjustedSizeInElement) * sizeof(int4))));
        CU_ASSERT(cuCtxSynchronize());
        CU_ASSERT(cuStreamDestroy(streamPeer));
    }

    CU_ASSERT(cuMemFreeHost((void*)blockingVar));
}

void launch_HtoD_memcpy_SM(unsigned long long size, unsigned long long loopCount) {
    int deviceCount = 0;

    void* dstBuffer;
    void* srcBuffer;
    unsigned long long bandwidth;
    double bandwidth_sum = 0.0;
    CUcontext benchCtx;
    benchmark_prepare(&benchCtx, &deviceCount);

    std::vector<double> bandwidthValues(deviceCount);

    CU_ASSERT(cuMemHostAlloc(&srcBuffer, (size_t)size, CU_MEMHOSTALLOC_PORTABLE));
    for (int currentDevice = 0; currentDevice < deviceCount; currentDevice++) {
        setOptimalCpuAffinity(currentDevice);

        CUcontext srcCtx;
        CU_ASSERT(cuDevicePrimaryCtxRetain(&srcCtx, currentDevice));
        CU_ASSERT(cuCtxSetCurrent(srcCtx));

        CU_ASSERT(cuMemAlloc((CUdeviceptr*)&dstBuffer, (size_t)size));

        memcpy_sm(dstBuffer, srcBuffer, &srcCtx, size / sizeof(int4), &bandwidth, loopCount);
        bandwidthValues[currentDevice] = bandwidth * 1e-9;
        bandwidth_sum += bandwidth * 1e-9;

        CU_ASSERT(cuMemFree((CUdeviceptr)dstBuffer));
        CU_ASSERT(cuDevicePrimaryCtxRelease(currentDevice));
    }
    benchmark_clean(srcBuffer, &benchCtx);

    std::cout << "memcpy SM GPU <- CPU bandwidth (GB/s):" << std::endl;
    printIndexVector(std::cout << std::fixed << std::setprecision(2), bandwidthValues);
}

void launch_DtoH_memcpy_SM(unsigned long long size, unsigned long long loopCount) {
    int deviceCount = 0;

    void* dstBuffer;
    void* srcBuffer;
    unsigned long long bandwidth;
    double bandwidth_sum = 0.0;
    CUcontext benchCtx;
    benchmark_prepare(&benchCtx, &deviceCount);

    std::vector<double> bandwidthValues(deviceCount);

    CU_ASSERT(cuMemHostAlloc(&dstBuffer, (size_t)size, CU_MEMHOSTALLOC_PORTABLE));
    for (int currentDevice = 0; currentDevice < deviceCount; currentDevice++) {
        setOptimalCpuAffinity(currentDevice);

        CUcontext srcCtx;
        CU_ASSERT(cuDevicePrimaryCtxRetain(&srcCtx, currentDevice));
        CU_ASSERT(cuCtxSetCurrent(srcCtx));

        CU_ASSERT(cuMemAlloc((CUdeviceptr*)&srcBuffer, (size_t)size));

        memcpy_sm(dstBuffer, srcBuffer, &srcCtx, size / sizeof(int4), &bandwidth, loopCount);
        CU_ASSERT(cuMemFree((CUdeviceptr)srcBuffer));

        bandwidthValues[currentDevice] = bandwidth * 1e-9;
        bandwidth_sum += bandwidth * 1e-9;

        CU_ASSERT(cuDevicePrimaryCtxRelease(currentDevice));
    }
    benchmark_clean(dstBuffer, &benchCtx);

    std::cout << "memcpy SM GPU -> CPU bandwidth (GB/s):" << std::endl;
    printIndexVector(std::cout << std::fixed << std::setprecision(2), bandwidthValues);
}

static void launch_DtoD_memcpy_SM(bool read, unsigned long long size, unsigned long long loopCount) {
    void* dstBuffer;
    void* srcBuffer;
    unsigned long long bandwidth;
    double value_sum = 0.0;
    int deviceCount = 0;
    CUcontext benchCtx;
    benchmark_prepare(&benchCtx, &deviceCount);
    PeerValueMatrix<double> bandwidth_matrix(deviceCount);

    for (int currentDevice = 0; currentDevice < deviceCount; currentDevice++) {
        setOptimalCpuAffinity(currentDevice);

        unsigned long long currentSize = size;
        CUcontext srcCtx;
        CU_ASSERT(cuDevicePrimaryCtxRetain(&srcCtx, currentDevice));
        CU_ASSERT(cuCtxSetCurrent(srcCtx));

        CU_ASSERT(cuMemAlloc((CUdeviceptr*)&srcBuffer, (size_t)currentSize));

        for (int peer = 0; peer < deviceCount; peer++) {
            CUcontext peerCtx;
            int canAccessPeer = 0;
            CU_ASSERT(cuDeviceCanAccessPeer(&canAccessPeer, currentDevice, peer));

            if (canAccessPeer) {
                CU_ASSERT(cuDevicePrimaryCtxRetain(&peerCtx, peer));
                CU_ASSERT(cuCtxSetCurrent(peerCtx));

                CU_ASSERT(cuCtxEnablePeerAccess(srcCtx, 0));
                CU_ASSERT(cuMemAlloc((CUdeviceptr*)&dstBuffer, currentSize));
                CU_ASSERT(cuCtxSetCurrent(srcCtx));
                CU_ASSERT(cuCtxEnablePeerAccess(peerCtx, 0));

                if (read) {
                    memcpy_sm(dstBuffer, srcBuffer, &srcCtx, size / sizeof(int4), &bandwidth, loopCount);
                    bandwidth_matrix.value(currentDevice, peer) = bandwidth * 1e-9;
                } else {
                    memcpy_sm(srcBuffer, dstBuffer, &srcCtx, size / sizeof(int4), &bandwidth, loopCount);
                    bandwidth_matrix.value(currentDevice, peer) = bandwidth * 1e-9;
                }

                unsigned long long device_current_bandwidth = 0;
                unsigned long long bandwidth_current;
                unsigned long long bandwidth_sum;

                if (bandwidth_sum > device_current_bandwidth) device_current_bandwidth = bandwidth_sum;
                bandwidth_matrix.value(currentDevice, peer) = bandwidth * 1e-9;

                bandwidth_sum += bandwidth * 1e-9;
                CU_ASSERT(cuCtxSetCurrent(srcCtx));
                CU_ASSERT(cuCtxDisablePeerAccess(peerCtx));
                CU_ASSERT(cuCtxSetCurrent(peerCtx));
                CU_ASSERT(cuCtxDisablePeerAccess(srcCtx));
                CU_ASSERT(cuMemFree((CUdeviceptr)dstBuffer));
                CU_ASSERT(cuDevicePrimaryCtxRelease(peer));
            }
        }
        benchmark_clean(srcBuffer, &benchCtx, true, currentDevice);
    }

    std::cout << "memcpy SM GPU(row) " << (read ? "->" : "<-") << " GPU(column) bandwidth (GB/s):" << std::endl;
    std::cout << std::fixed << std::setprecision(2) << bandwidth_matrix << std::endl;
}


static void launch_DtoD_memcpy_bidirectional_SM(bool read, unsigned long long size, unsigned long long loopCount) {
    void* gpuAbuffer0;
    void* gpuAbuffer1;
    void* gpuBbuffer0;
    void* gpuBbuffer1;

    unsigned long long bandwidth;
    int deviceCount = 0;
    double bandwidth_sum = 0.0;
    CUcontext benchCtx;
    benchmark_prepare(&benchCtx, &deviceCount);

    PeerValueMatrix<double> bandwidth_matrix(deviceCount);

    for (int currentDevice = 0; currentDevice < deviceCount; currentDevice++) {
        setOptimalCpuAffinity(currentDevice);
        
        CUcontext srcCtx;
        CU_ASSERT(cuDevicePrimaryCtxRetain(&srcCtx, currentDevice));
        CU_ASSERT(cuCtxSetCurrent(srcCtx));

        CU_ASSERT(cuCtxGetDevice(&currentDevice));
        CU_ASSERT(cuMemAlloc((CUdeviceptr*)&gpuAbuffer0, (size_t)size));
        CU_ASSERT(cuMemAlloc((CUdeviceptr*)&gpuAbuffer1, (size_t)size));

        for (int peer = 0; peer < deviceCount; peer++) {
            CUcontext peerCtx;
            int canAccessPeer = 0;
            CU_ASSERT(cuDeviceCanAccessPeer(&canAccessPeer, currentDevice, peer));

            if (canAccessPeer) {
                CU_ASSERT(cuDevicePrimaryCtxRetain(&peerCtx, peer));
                CU_ASSERT(cuCtxSetCurrent(peerCtx));
    
                CU_ASSERT(cuCtxEnablePeerAccess(srcCtx, 0));
                CU_ASSERT(cuMemAlloc((CUdeviceptr*)&gpuBbuffer0, (size_t)size));
                CU_ASSERT(cuMemAlloc((CUdeviceptr*)&gpuBbuffer1, (size_t)size));
                CU_ASSERT(cuCtxSetCurrent(srcCtx));
                CU_ASSERT(cuCtxEnablePeerAccess(peerCtx, 0));

                unsigned long long bandwidth_src = 0, bandwidth_peer = 0;
                if (read) {
                    memcpy_sm(gpuAbuffer0, gpuBbuffer0, &srcCtx, size / sizeof(int4), &bandwidth, loopCount, &peerCtx);
                    bandwidth_matrix.value(currentDevice, peer) = bandwidth * 1e-9;
                } else {
                    unsigned long long bandwidth_src = 0, bandwidth_peer = 0;
                    memcpy_sm(gpuBbuffer0, gpuAbuffer0, &srcCtx, size / sizeof(int4), &bandwidth, loopCount, &peerCtx);
                    bandwidth_matrix.value(currentDevice, peer) = bandwidth * 1e-9;
                }
                bandwidth_sum += bandwidth * 1e-9;
                
                CU_ASSERT(cuCtxSetCurrent(srcCtx));
                CU_ASSERT(cuCtxDisablePeerAccess(peerCtx));
                CU_ASSERT(cuCtxSetCurrent(peerCtx));
                CU_ASSERT(cuCtxDisablePeerAccess(srcCtx));
            
                benchmark_clean_bidir(&benchCtx, peer, gpuBbuffer0, gpuBbuffer1);
            }
        }
        benchmark_clean_bidir(&benchCtx, currentDevice, gpuAbuffer0, gpuAbuffer1);
    }
    std::cout << "memcpy SM GPU(row) " << (read ? "->" : "<-") << " GPU(column) bandwidth (GB/s):" << std::endl;
    std::cout << std::fixed << std::setprecision(2) << bandwidth_matrix << std::endl;
}

void launch_DtoD_memcpy_bidirectional_read_SM(unsigned long long size, unsigned long long loopCount) {
    launch_DtoD_memcpy_bidirectional_SM(true, size, loopCount);
}
void launch_DtoD_memcpy_bidirectional_write_SM(unsigned long long size, unsigned long long loopCount) {
    launch_DtoD_memcpy_bidirectional_SM(false, size, loopCount);
}
void launch_DtoD_memcpy_read_SM(unsigned long long size, unsigned long long loopCount) {
    launch_DtoD_memcpy_SM(true, size, loopCount);
}
void launch_DtoD_memcpy_write_SM(unsigned long long size, unsigned long long loopCount) {
    launch_DtoD_memcpy_SM(false, size, loopCount);
}
