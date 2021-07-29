#include <cstddef>
#include <iomanip>
#include <iostream>
#include <string.h>
#include <vector>

#include "benchmarks.h"
#include "common.h"
#include "memory_utils.h"

static void memcpy_sm(void *dst, void *src, unsigned long long *bandwidth, unsigned long long size, bool isPageable, unsigned long long loopCount = defaultLoopCount, bool doubleBandwidth = false) {
    CUstream stream;
    CUevent startEvent;
    CUevent endEvent;
    volatile int *blockingVar = NULL;
    *bandwidth = 0;

    CU_ASSERT(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
    CU_ASSERT(cuEventCreate(&startEvent, CU_EVENT_DEFAULT));
    CU_ASSERT(cuEventCreate(&endEvent, CU_EVENT_DEFAULT));

    CU_ASSERT(cuEventRecord(startEvent, stream));
    for (unsigned int n = 0; n < loopCount; n++) {
        CU_ASSERT(cuMemcpyAsync((CUdeviceptr)dst, (CUdeviceptr)src, (size_t)size, stream));
    }
    CU_ASSERT(cuEventRecord(endEvent, stream));

    if (!isPageable && !disableP2P) {
        *blockingVar = 1;
    }

    CU_ASSERT(cuStreamSynchronize(stream));

    float timeWithEvents = 0.0f;
    CU_ASSERT(cuEventElapsedTime(&timeWithEvents, startEvent, endEvent));
    unsigned long long elapsedWithEventsInUs = (unsigned long long)(timeWithEvents * 1000.0f);
    *bandwidth = (size * loopCount * 1000ull * 1000ull) / elapsedWithEventsInUs; // Bandwidth in Bytes per second

    if (!isPageable && !disableP2P) {
        CU_ASSERT(cuMemFreeHost((void *)blockingVar));
    }

    CU_ASSERT(cuCtxSynchronize());
    if (doubleBandwidth) *bandwidth = 2 * (*bandwidth);
}

void launch_HtoD_memcpy_SM(unsigned long long size, unsigned long long loopCount) {
    int deviceCount = 0;
    void* dstBuffer;
    void* srcBuffer;
    unsigned long long bandwidth;
    double bandwidth_sum;
    CUcontext benchCtx;
    std::vector<double> bandwidthValues(deviceCount);

    benchmark_prepare(&benchCtx, &deviceCount, false, srcBuffer, size);

    for (int currentDevice = 0; currentDevice < deviceCount; currentDevice++) {
        CUcontext srcCtx;

        CU_ASSERT(cuDevicePrimaryCtxRetain(&srcCtx, currentDevice));
        CU_ASSERT(cuCtxSetCurrent(srcCtx));

        CU_ASSERT(cuMemAlloc((CUdeviceptr*)&dstBuffer, (size_t)size));

        memcpy_sm(dstBuffer, srcBuffer, &bandwidth, size, loopCount);
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
    std::vector<double> bandwidthValues(deviceCount);

    benchmark_prepare(&benchCtx, &deviceCount, false, srcBuffer, size);

    for (int currentDevice = 0; currentDevice < deviceCount; currentDevice++) {
        CUcontext srcCtx;

        CU_ASSERT(cuDevicePrimaryCtxRetain(&srcCtx, currentDevice));
        CU_ASSERT(cuCtxSetCurrent(srcCtx));

        CU_ASSERT(cuMemAlloc((CUdeviceptr*)&srcBuffer, (size_t)size));

        memcpy_sm(dstBuffer, srcBuffer, &bandwidth, size, loopCount);
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
    PeerValueMatrix<double> value_matrix(deviceCount);

    benchmark_prepare(&benchCtx, &deviceCount, true);

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
                    memcpy_sm(dstBuffer, srcBuffer, &bandwidth, currentSize, loopCount);
                    value_matrix.value(currentDevice, peer) = bandwidth * 1e-9;
                } else {
                    memcpy_sm(srcBuffer, dstBuffer, &bandwidth, currentSize, loopCount);
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
    PeerValueMatrix<double> bandwidth_matrix(deviceCount);

    benchmark_prepare(&benchCtx, &deviceCount, true);

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
                        memcpy_sm(gpuBbuffer0, gpuAbuffer0, &bandwidth, size / sizeof(int), loopCount);
                        bandwidth_sum += bandwidth;
                        memcpy_sm(gpuAbuffer1, gpuBbuffer1, &bandwidth, size / sizeof(int), loopCount);
                        bandwidth_sum += bandwidth;
                    } else {
                        memcpy_sm(gpuAbuffer0, gpuBbuffer0, &bandwidth, size / sizeof(int), loopCount);
                        bandwidth_sum += bandwidth;
                        memcpy_sm(gpuBbuffer1, gpuAbuffer1, &bandwidth, size / sizeof(int), loopCount);
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
