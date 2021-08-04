#include <cstddef>
#include <iomanip>
#include <iostream>
#include <string.h>
#include <vector>

#include "benchmarks.h"
#include "common.h"
#include "memory_utils.h"
#include "memcpy_kernel.cuh"

static void memcpy_sm(void *dst, void *src, CUcontext ctx, unsigned long long *bandwidth, unsigned long long size, bool isPageable, unsigned long long loopCount = defaultLoopCount, bool doubleBandwidth = false) {
    CUdevice device;

    CUstream stream;
    CUevent startEvent;
    CUevent endEvent;
    unsigned long long adjustedSizeInElement;
    volatile int *blockingVar = NULL;
    unsigned int numThreadPerBlock = 512;
    CUmodule cuModule;
    CUfunction func;
    unsigned long long totalThreadCount;

    CU_ASSERT(cuMemHostAlloc((void **)&blockingVar, sizeof(*blockingVar), CU_MEMHOSTALLOC_PORTABLE));
    *blockingVar = 0;
    *bandwidth = 0;

    for (unsigned i = 0; i < 1; ++i) {
        CU_ASSERT(cuCtxGetDevice(&device));

        int multiProcessorCount;
        CU_ASSERT(cuDeviceGetAttribute(&multiProcessorCount, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device));
        totalThreadCount = (unsigned long long)(multiProcessorCount * numThreadPerBlock);
        adjustedSizeInElement = totalThreadCount * (size / totalThreadCount);

        CU_ASSERT(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
        CU_ASSERT(cuEventCreate(&startEvent, CU_EVENT_DEFAULT));
        CU_ASSERT(cuEventCreate(&endEvent, CU_EVENT_DEFAULT));

        Kernels::spin_kernel(blockingVar, stream);

        // Two memcpy kernels to warm up
        Kernels::memcpy_kernel((int4 *)dst, (int4 *)src, stream, adjustedSizeInElement, numThreadPerBlock, loopCount);
        CU_ASSERT(cuEventRecord(startEvent, stream));
        // ensuring that all copies are launched at the same time
        CU_ASSERT(cuStreamWaitEvent(stream, startEvent, 0));
        CU_ASSERT(cuEventRecord(startEvent, stream));
        CU_ASSERT(cuEventRecord(endEvent, stream));

        Kernels::memcpy_kernel((int4 *)dst, (int4 *)src, stream, adjustedSizeInElement, numThreadPerBlock, loopCount);
        CU_ASSERT(cuEventRecord(endEvent, stream));
    }

    *blockingVar = 1;

    for (unsigned i = 0; i < 1; ++i) {
        CU_ASSERT(cuStreamSynchronize(stream));
    }

    for (unsigned i = 0; i < 1; ++i) {
        float timeWithEvents = 0.0f;
        CU_ASSERT(cuCtxSetCurrent(ctx));
        CU_ASSERT(cuEventElapsedTime(&timeWithEvents, startEvent, endEvent));
        unsigned long long elapsedWithEventsInUs = (unsigned long long)(timeWithEvents * 1000.0f);

        *bandwidth += (adjustedSizeInElement * sizeof(int4) * loopCount * 1000ull * 1000ull) / elapsedWithEventsInUs; // Bandwidth in Bytes per second

        CUdeviceptr dstBuffer = (CUdeviceptr)((int4 *)(dst) + adjustedSizeInElement);
        CUdeviceptr srcBuffer = (CUdeviceptr)((int4 *)(src) + adjustedSizeInElement);
        size_t adjustedSize = size - adjustedSizeInElement;

        CU_ASSERT(cuMemcpy(dstBuffer, srcBuffer, adjustedSize * sizeof(int4)));
        CU_ASSERT(cuCtxSynchronize());
        CU_ASSERT(cuStreamDestroy(stream));
    }

    CU_ASSERT(cuMemFreeHost((void*)blockingVar));
}

void launch_HtoD_memcpy_SM(unsigned long long size, unsigned long long loopCount) {
    int deviceCount = 0;
    void* dstBuffer;
    void* srcBuffer;
    unsigned long long bandwidth;
    double bandwidth_sum;
    CUcontext benchCtx;
    benchmark_prepare(&benchCtx, &deviceCount);
    std::vector<double> bandwidthValues(deviceCount);

    CU_ASSERT(cuMemHostAlloc(&srcBuffer, (size_t)size, CU_MEMHOSTALLOC_PORTABLE));
    for (int currentDevice = 0; currentDevice < deviceCount; currentDevice++) {
        CUcontext srcCtx;

        CU_ASSERT(cuDevicePrimaryCtxRetain(&srcCtx, currentDevice));
        CU_ASSERT(cuCtxSetCurrent(srcCtx));

        CU_ASSERT(cuMemAlloc((CUdeviceptr*)&dstBuffer, (size_t)size));
        memcpy_sm(dstBuffer, srcBuffer, srcCtx, &bandwidth, size / sizeof(int4), loopCount);
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
    double bandwidth_sum;
    CUcontext benchCtx;
    benchmark_prepare(&benchCtx, &deviceCount);
    std::vector<double> bandwidthValues(deviceCount);

    CU_ASSERT(cuMemHostAlloc(&srcBuffer, (size_t)size, CU_MEMHOSTALLOC_PORTABLE));
    for (int currentDevice = 0; currentDevice < deviceCount; currentDevice++) {
        CUcontext srcCtx;

        CU_ASSERT(cuDevicePrimaryCtxRetain(&srcCtx, currentDevice));
        CU_ASSERT(cuCtxSetCurrent(srcCtx));

        CU_ASSERT(cuMemAlloc((CUdeviceptr*)&srcBuffer, (size_t)size));

        memcpy_sm(dstBuffer, srcBuffer, srcCtx, &bandwidth, size, loopCount);
        CU_ASSERT(cuMemFree((CUdeviceptr)srcBuffer));

        bandwidthValues[currentDevice] = bandwidth * 1e-9;
        bandwidth_sum += bandwidth * 1e-9;

        CU_ASSERT(cuDevicePrimaryCtxRelease(currentDevice));
    }
    
    benchmark_clean(srcBuffer, &benchCtx);

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
    PeerValueMatrix<double> value_matrix(deviceCount);

    for (int currentDevice = 0; currentDevice < deviceCount; currentDevice++) {
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
                    memcpy_sm(dstBuffer, srcBuffer, srcCtx, &bandwidth, currentSize, loopCount);
                    value_matrix.value(currentDevice, peer) = bandwidth * 1e-9;
                } else {
                    memcpy_sm(srcBuffer, dstBuffer, srcCtx, &bandwidth, currentSize, loopCount);
                    value_matrix.value(currentDevice, peer) = bandwidth * 1e-9;
                }

                unsigned long long device_current_bandwidth = 0;
                unsigned long long bandwidth_current;
                unsigned long long bandwidth_sum;

                if (bandwidth_sum > device_current_bandwidth) device_current_bandwidth = bandwidth_sum;
                value_matrix.value(currentDevice, peer) = bandwidth * 1e-9;

                bandwidth_sum += bandwidth * 1e-9;
                CU_ASSERT(cuCtxSetCurrent(srcCtx));
                CU_ASSERT(cuCtxDisablePeerAccess(peerCtx));
                CU_ASSERT(cuCtxSetCurrent(peerCtx));
                CU_ASSERT(cuCtxDisablePeerAccess(srcCtx));
                CU_ASSERT(cuMemFree((CUdeviceptr)dstBuffer));
                CU_ASSERT(cuDevicePrimaryCtxRelease(peer));
            }
        }

        CU_ASSERT(cuCtxSetCurrent(benchCtx));
        CU_ASSERT(cuMemFree((CUdeviceptr)srcBuffer));
        CU_ASSERT(cuDevicePrimaryCtxRelease(currentDevice));
        benchmark_clean(srcBuffer, &benchCtx, true, currentDevice);
    }

    std::cout << "memcpy SM GPU(row) " << (read ? "->" : "<-") << " GPU(column) bandwidth (GB/s):" << std::endl;
    std::cout << std::fixed << std::setprecision(2) << value_matrix << std::endl;
}


static void launch_DtoD_memcpy_bidirectional_SM(bool read, unsigned long long size, unsigned long long loopCount) {
    CUcontext srcCtx;
    void *gpuAbuffer0;
    void *gpuAbuffer1;
    void *gpuBbuffer0;
    void *gpuBbuffer1;
    CUcontext benchCtx;
    unsigned long long bandwidth;
    double bandwidth_sum;
    int deviceCount = 0;
    benchmark_prepare(&benchCtx, &deviceCount);
    PeerValueMatrix<double> bandwidth_matrix(deviceCount);

    for (int currentDevice = 0; currentDevice < deviceCount; currentDevice++) {
        benchmark_prepare_bidir(&srcCtx, currentDevice, gpuAbuffer0, gpuAbuffer1, size);

        for (int peer = 0; peer < deviceCount; peer++) {
            bandwidth_sum = 0.0;
            CUcontext peerCtx;

            if (currentDevice == peer) {
                continue;
            }

            int canAccessPeer = 0;
            CU_ASSERT(cuDeviceCanAccessPeer(&canAccessPeer, currentDevice, peer));

            if (canAccessPeer) {
                CU_ASSERT(cuDevicePrimaryCtxRetain(&peerCtx, peer));

                CU_ASSERT(cuCtxSetCurrent(peerCtx));
                CU_ASSERT(cuCtxEnablePeerAccess(srcCtx, 0));

                CU_ASSERT(cuMemAlloc((CUdeviceptr *)&gpuBbuffer0, (size_t)size));
                CU_ASSERT(cuMemAlloc((CUdeviceptr *)&gpuBbuffer1, (size_t)size));
                
                CU_ASSERT(cuCtxSetCurrent(srcCtx));
                CU_ASSERT(cuCtxEnablePeerAccess(peerCtx, 0));

                for (unsigned int n = 0; n < loopCount; n++) {
                    if (read) {
                        memcpy_sm(gpuBbuffer0, gpuAbuffer0, srcCtx, &bandwidth, size / sizeof(int), loopCount);
                        bandwidth_sum += bandwidth;
                        memcpy_sm(gpuAbuffer1, gpuBbuffer1, peerCtx, &bandwidth, size / sizeof(int), loopCount);
                        bandwidth_sum += bandwidth;
                    } else {
                        memcpy_sm(gpuAbuffer0, gpuBbuffer0, srcCtx, &bandwidth, size / sizeof(int), loopCount);
                        bandwidth_sum += bandwidth;
                        memcpy_sm(gpuBbuffer1, gpuAbuffer1, peerCtx, &bandwidth, size / sizeof(int), loopCount);
                        bandwidth_sum += bandwidth;
                    }
                }

                bandwidth_matrix.value(currentDevice, peer) = (bandwidth_sum * 1e-9) / loopCount;
                CU_ASSERT(cuCtxSetCurrent(srcCtx));
                CU_ASSERT(cuCtxDisablePeerAccess(peerCtx));
                CU_ASSERT(cuCtxSetCurrent(peerCtx));
                CU_ASSERT(cuCtxDisablePeerAccess(srcCtx));

                CU_ASSERT(cuMemFree((CUdeviceptr)gpuBbuffer0));
                CU_ASSERT(cuMemFree((CUdeviceptr)gpuBbuffer1));
                CU_ASSERT(cuDevicePrimaryCtxRelease(peer));
            }
        }
        benchmark_clean_bidir(&benchCtx, currentDevice, gpuAbuffer0, gpuAbuffer1, size);
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
