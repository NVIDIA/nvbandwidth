#include <algorithm>
#include <cstddef>
#include <cuda.h>
#include <iomanip>
#include <iostream>
#include <stdio.h>
#include <string.h>
#include <time.h>

#include "benchmarks.h"
#include "memory_utils.h"

static void memcpy_ce(void *dst, void *src, unsigned long long size, unsigned long long *bandwidth,
    unsigned long long loopCount = defaultLoopCount) {

    CUstream stream;
    CUevent startEvent;
    CUevent endEvent;
    volatile int *blockingVar = NULL;

    bool isPageable = !isMemoryOwnedByCUDA(dst) || !isMemoryOwnedByCUDA(src);

    CU_ASSERT(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
    CU_ASSERT(cuEventCreate(&startEvent, CU_EVENT_DEFAULT));
    CU_ASSERT(cuEventCreate(&endEvent, CU_EVENT_DEFAULT));

    // Spend time on the GPU so we finish submitting everything before the benchmark starts
    // Also events are tied to the last submission channel so we want to be sure it is copy and not compute
    const size_t WARMUP_COUNT = 32;
    for (unsigned int n = 0; n < WARMUP_COUNT; n++) {
        // As latency benchmarks do 1 byte copies, we have to ensure we're not doing 0 byte copies
        cuMemcpyAsync((CUdeviceptr) dst, (CUdeviceptr) src, (size_t)((size + 7) / 8), stream); // ASSERT_DRV
    }

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
}

static void memcpy_ce_bidirectional(void *dst1, void *src1, CUcontext ctx1, void *dst2, void *src2, CUcontext ctx2, unsigned long long size,
    unsigned long long *bandwidth, unsigned long long loopCount = defaultLoopCount) {

    volatile int *blockingVar = NULL;

    int dev1, dev2;

    CUstream stream_dir1;
    CUevent startEvent_dir1;
    CUevent endEvent_dir1;

    CUstream stream_dir2;
    CUevent startEvent_dir2;
    CUevent endEvent_dir2;

    void *markerDst = NULL, *markerSrc = NULL, *additionalMarkerLocation = NULL;

    CU_ASSERT(cuCtxSetCurrent(ctx1));
    CU_ASSERT(cuCtxGetDevice(&dev1));
    CU_ASSERT(cuStreamCreate(&stream_dir1, CU_STREAM_NON_BLOCKING));
    CU_ASSERT(cuEventCreate(&startEvent_dir1, CU_EVENT_DEFAULT));
    CU_ASSERT(cuEventCreate(&endEvent_dir1, CU_EVENT_DEFAULT));

    CU_ASSERT(cuCtxSetCurrent(ctx2));
    CU_ASSERT(cuCtxGetDevice(&dev2));
    CU_ASSERT(cuStreamCreate(&stream_dir2, CU_STREAM_NON_BLOCKING));
    CU_ASSERT(cuEventCreate(&startEvent_dir2, CU_EVENT_DEFAULT));
    CU_ASSERT(cuEventCreate(&endEvent_dir2, CU_EVENT_DEFAULT));

    CU_ASSERT(cuEventRecord(startEvent_dir1, stream_dir1));
    CU_ASSERT(cuStreamWaitEvent(stream_dir2, startEvent_dir1, 0));

    for (unsigned int n = 0; n < loopCount; n++) {
        CU_ASSERT(cuMemcpyAsync((CUdeviceptr)dst1, (CUdeviceptr)src1, (size_t)size, stream_dir1));
        CU_ASSERT(cuMemcpyAsync((CUdeviceptr)dst2, (CUdeviceptr)src2, (size_t)size, stream_dir2));
    }

    CU_ASSERT(cuEventRecord(endEvent_dir1, stream_dir1));

    if (!disableP2P) {
        *blockingVar = 1;
    }

    // Now, we need to ensure there is always work in the stream2 pending, to
    // ensure there always is intereference to the stream1.
    unsigned int extraIters = loopCount > 1 ? (unsigned int)loopCount / 2 : 1;
    do {
        // Enqueue extra work
        for (unsigned int n = 0; n < extraIters; n++) {
            CU_ASSERT(cuMemcpyAsync((CUdeviceptr)dst2, (CUdeviceptr)src2, (size_t)size, stream_dir2));
        }

        // Record the event in the middle of interfering flow, to ensure the next
        // batch starts enqueuing before the previous one finishes.
        CU_ASSERT(cuEventRecord(endEvent_dir2, stream_dir2));

        // Add more iterations to hide latency of scheduling more work in the next
        // iteration of loop.
        for (unsigned int n = 0; n < extraIters; n++) {
            CU_ASSERT(cuMemcpyAsync((CUdeviceptr)dst2, (CUdeviceptr)src2, (size_t)size, stream_dir2));
        }

        // Wait until the flow in the interference stream2 is finished.
        CU_ASSERT(cuEventSynchronize(endEvent_dir2));
    } while (cuStreamQuery(stream_dir1) == CUDA_ERROR_NOT_READY);

    CU_ASSERT(cuStreamSynchronize(stream_dir1));
    CU_ASSERT(cuStreamSynchronize(stream_dir2));

    float timeWithEvents = 0.0f;
    CU_ASSERT(cuEventElapsedTime(&timeWithEvents, startEvent_dir1, endEvent_dir1));
    double elapsedWithEventsInUs = ((double)timeWithEvents * 1000.0);

    *bandwidth = (size * loopCount * 1000ull * 1000ull) / (unsigned long long)elapsedWithEventsInUs;

    if (!disableP2P) {
        CU_ASSERT(cuMemFreeHost((void *)blockingVar));
    }

    CU_ASSERT(cuCtxSynchronize());
}

static void unidirectional_memcpy(void *src, void *dst, unsigned long long *bandwidth, unsigned long long size,
    unsigned long long loopCount) {

    unsigned long long bandwidth_current;
    cudaStat bandwidthStat;

    *bandwidth = 0;
    for (unsigned int n = 0; n < averageLoopCount; n++) {
        memcpy_ce(dst, src, size, &bandwidth_current, loopCount);
        bandwidthStat((double)bandwidth_current);
        VERBOSE << "\tSample " << n << ' ' << std::fixed << std::setprecision (2) << bandwidth_current * 1e-9 << " GB/s\n";
    }
    VERBOSE << "       bandwidth: " << std::fixed << std::setprecision (2) << STAT_MEAN(bandwidthStat) * 1e-9 << "(+/- " << STAT_ERROR(bandwidthStat) * 1e-9 << ") GB/s\n";
    *bandwidth = (unsigned long long)(STAT_MEAN(bandwidthStat));
}

static void bidirectional_memcpy(void *dst1, void *src1, CUcontext ctx1, void *dst2, void *src2, CUcontext ctx2,
    unsigned long long *bandwidth, unsigned long long size, unsigned long long loopCount) {

    unsigned long long bandwidth_current;
    cudaStat bandwidthStat;

    *bandwidth = 0;
    for (unsigned int n = 0; n < averageLoopCount; n++) {
        memcpy_ce_bidirectional(dst1, src1, ctx1, dst2, src2, ctx2, size,  &bandwidth_current, loopCount);
        bandwidthStat((double)bandwidth_current);
        VERBOSE << "\tSample " << n << ' ' << std::fixed << std::setprecision (2) << bandwidth_current * 1e-9 << " GB/s\n";
    }
    VERBOSE << "       bandwidth: " << std::fixed << std::setprecision (2) << STAT_MEAN(bandwidthStat) * 1e-9 << "(+/- " << STAT_ERROR(bandwidthStat) * 1e-9 << ") GB/s\n";
    *bandwidth = (unsigned long long)(STAT_MEAN(bandwidthStat));
}

void launch_HtoD_memcpy_CE(unsigned long long size, unsigned long long loopCount) {
    void* dstBuffer;
    void* srcBuffer;
    double perf_value_sum = 0.0;
    int deviceCount = 0;
    size_t *procMask = NULL;
    size_t firstEnabledCPU = getFirstEnabledCPU();
    CUcontext benchCtx;
    benchmark_prepare(&benchCtx, &deviceCount);

    PeerValueMatrix<double> bandwidthValues(1, deviceCount);
    procMask = (size_t *)calloc(1, PROC_MASK_SIZE);

    PROC_MASK_SET(procMask, firstEnabledCPU);

    cuMemHostAlloc(&srcBuffer, (size_t)size, CU_MEMHOSTALLOC_PORTABLE);

    for (size_t currentDevice = 0; currentDevice < deviceCount; currentDevice++) {
        setOptimalCpuAffinity(currentDevice);

        CUcontext srcCtx;
        CU_ASSERT(cuDevicePrimaryCtxRetain(&srcCtx, currentDevice));
        CU_ASSERT(cuCtxSetCurrent(srcCtx));
            
        CU_ASSERT(cuMemAlloc((CUdeviceptr*)&dstBuffer, (size_t)size));

        unsigned long long bandwidth;
        unidirectional_memcpy(srcBuffer, dstBuffer, &bandwidth, size, loopCount);
        bandwidthValues.value(0, currentDevice) = bandwidth * 1e-9;
        perf_value_sum += bandwidth * 1e-9;

        CU_ASSERT(cuMemFree((CUdeviceptr)dstBuffer));

        CU_ASSERT(cuDevicePrimaryCtxRelease(currentDevice));

        PROC_MASK_CLEAR(procMask, 0);
    }

    benchmark_clean(srcBuffer, &benchCtx);

    free(procMask);

    std::cout << "memcpy CE GPU(columns) <- CPU(rows) bandwidth (GB/s)" << std::endl;
    std::cout << std::fixed << std::setprecision(2) << bandwidthValues << std::endl;
}

void launch_DtoH_memcpy_CE(unsigned long long size, unsigned long long loopCount) {
    void* dstBuffer;
    void* srcBuffer;
    double perf_value_sum = 0.0;
    int deviceCount = 0;
    size_t *procMask = NULL;
    size_t firstEnabledCPU = getFirstEnabledCPU();
    CUcontext benchCtx;
    benchmark_prepare(&benchCtx, &deviceCount);

    PeerValueMatrix<double> bandwidthValues(1, deviceCount);
    procMask = (size_t *)calloc(1, PROC_MASK_SIZE);

    PROC_MASK_SET(procMask, firstEnabledCPU);

    CU_ASSERT(cuMemHostAlloc(&dstBuffer, (size_t)size, CU_MEMHOSTALLOC_PORTABLE));
        
    for (size_t currentDevice = 0; currentDevice < deviceCount; currentDevice++) {
        setOptimalCpuAffinity(currentDevice);

        CUcontext srcCtx;
        CU_ASSERT(cuDevicePrimaryCtxRetain(&srcCtx, currentDevice));
        CU_ASSERT(cuCtxSetCurrent(srcCtx));
        CU_ASSERT(cuMemAlloc((CUdeviceptr*)&srcBuffer, size));

        unsigned long long bandwidth;
        unidirectional_memcpy(srcBuffer, dstBuffer, &bandwidth, size, loopCount);
        bandwidthValues.value(0, currentDevice) = bandwidth * 1e-9;
        perf_value_sum += bandwidth * 1e-9;

        CU_ASSERT(cuMemFree((CUdeviceptr)srcBuffer));
        CU_ASSERT(cuDevicePrimaryCtxRelease(currentDevice));
    }

    benchmark_clean(dstBuffer, &benchCtx);
    PROC_MASK_CLEAR(procMask, 0);

    free(procMask);

    std::cout << "memcpy CE GPU(columns) -> CPU(rows) bandwidth (GB/s)" << std::endl;
    std::cout << std::fixed << std::setprecision(2) << bandwidthValues << std::endl;
}

void launch_HtoD_memcpy_bidirectional_CE(unsigned long long size, unsigned long long loopCount) {
    CUcontext srcCtx;
    void *HtoD_dstBuffer;
    void *HtoD_srcBuffer;
    void *DtoH_dstBuffer;
    void *DtoH_srcBuffer;
    unsigned long long bandwidth;
    double bandwidth_sum = 0.0;
    size_t *procMask = NULL;
    size_t firstEnabledCPU = getFirstEnabledCPU();
    int deviceCount;
    CUcontext benchCtx;
    benchmark_prepare(&benchCtx, &deviceCount);

    PeerValueMatrix<double> bandwidthValues(1, deviceCount);
    procMask = (size_t *)calloc(1, PROC_MASK_SIZE);

    PROC_MASK_SET(procMask, firstEnabledCPU);

    /* The NUMA location of the calling thread determines the physical
        location of the pinned memory allocation, which can have different
        performance characteristics */
    CU_ASSERT(cuMemHostAlloc(&HtoD_srcBuffer, (size_t)size, CU_MEMHOSTALLOC_PORTABLE));
    CU_ASSERT(cuMemHostAlloc(&DtoH_dstBuffer, (size_t)size, CU_MEMHOSTALLOC_PORTABLE));

    for (size_t currentDevice = 0; currentDevice < deviceCount; currentDevice++) {
        setOptimalCpuAffinity(currentDevice);

        CU_ASSERT(cuDevicePrimaryCtxRetain(&srcCtx, currentDevice));
        CU_ASSERT(cuCtxSetCurrent(srcCtx));

        CU_ASSERT(cuMemAlloc((CUdeviceptr *)&HtoD_dstBuffer, (size_t)size));
        CU_ASSERT(cuMemAlloc((CUdeviceptr *)&DtoH_srcBuffer, (size_t)size));

        bidirectional_memcpy(HtoD_dstBuffer, HtoD_srcBuffer, srcCtx, DtoH_dstBuffer, DtoH_srcBuffer, srcCtx, &bandwidth, size, loopCount);

        bandwidthValues.value(0, currentDevice) = bandwidth * 1e-9;
        bandwidth_sum += bandwidth * 1e-9;

        CU_ASSERT(cuMemFree((CUdeviceptr)DtoH_srcBuffer));
        CU_ASSERT(cuMemFree((CUdeviceptr)HtoD_dstBuffer));
        CU_ASSERT(cuDevicePrimaryCtxRelease(currentDevice));
    }

    benchmark_clean_bidir_h2d(&benchCtx, 0, HtoD_srcBuffer, DtoH_dstBuffer);

    PROC_MASK_CLEAR(procMask, 0);

    free(procMask);

    std::cout << "memcpy CE GPU(columns) <- CPU(rows) bandwidth (GB/s)" << std::endl;
    std::cout << std::fixed << std::setprecision(2) << bandwidthValues << std::endl;
}

void launch_DtoH_memcpy_bidirectional_CE(unsigned long long size, unsigned long long loopCount) {
    CUcontext srcCtx;
    void *HtoD_dstBuffer;
    void *HtoD_srcBuffer;
    void *DtoH_dstBuffer;
    void *DtoH_srcBuffer;
    unsigned long long bandwidth;
    double bandwidth_sum = 0.0;
    size_t *procMask = NULL;
    size_t firstEnabledCPU = getFirstEnabledCPU();
    int deviceCount;
    CUcontext benchCtx;
    benchmark_prepare(&benchCtx, &deviceCount);

    PeerValueMatrix<double> bandwidthValues(1, deviceCount);
    procMask = (size_t *)calloc(1, PROC_MASK_SIZE);

    PROC_MASK_SET(procMask, firstEnabledCPU);

    /* The NUMA location of the calling thread determines the physical
        location of the pinned memory allocation, which can have different
        performance characteristics */
    CU_ASSERT(cuMemHostAlloc(&HtoD_srcBuffer, (size_t)size, CU_MEMHOSTALLOC_PORTABLE));
    CU_ASSERT(cuMemHostAlloc(&DtoH_dstBuffer, (size_t)size, CU_MEMHOSTALLOC_PORTABLE));

    for (size_t currentDevice = 0; currentDevice < deviceCount; currentDevice++) {
        setOptimalCpuAffinity(currentDevice);

        CU_ASSERT(cuDevicePrimaryCtxRetain(&srcCtx, currentDevice));
        CU_ASSERT(cuCtxSetCurrent(srcCtx));

        CU_ASSERT(cuMemAlloc((CUdeviceptr *)&HtoD_dstBuffer, (size_t)size));
        CU_ASSERT(cuMemAlloc((CUdeviceptr *)&DtoH_srcBuffer, (size_t)size));

        bidirectional_memcpy(DtoH_dstBuffer, DtoH_srcBuffer, srcCtx, HtoD_dstBuffer, HtoD_srcBuffer, srcCtx, &bandwidth, size, loopCount);
        bandwidthValues.value(0, currentDevice) = bandwidth * 1e-9;
        bandwidth_sum += bandwidth * 1e-9;

        CU_ASSERT(cuMemFree((CUdeviceptr)DtoH_srcBuffer));
        CU_ASSERT(cuMemFree((CUdeviceptr)HtoD_dstBuffer));
        CU_ASSERT(cuDevicePrimaryCtxRelease(currentDevice));
    }

    benchmark_clean_bidir_h2d(&benchCtx, 0, HtoD_srcBuffer, DtoH_dstBuffer);

    PROC_MASK_CLEAR(procMask, 0);

    free(procMask);

    std::cout << "memcpy CE GPU(columns) <- CPU(rows) bandwidth (GB/s)" << std::endl;
    std::cout << std::fixed << std::setprecision(2) << bandwidthValues << std::endl;
}

void launch_DtoD_memcpy_CE(bool read, unsigned long long size, unsigned long long loopCount) {
    void* dstBuffer;
    void* srcBuffer;
    unsigned long long bandwidth;
    double value_sum = 0.0;
    int deviceCount = 0;
    CUcontext benchCtx;
    benchmark_prepare(&benchCtx, &deviceCount);

    PeerValueMatrix<double> value_matrix(deviceCount);

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
                    unidirectional_memcpy(dstBuffer, srcBuffer, &bandwidth, currentSize, loopCount);
                    value_matrix.value(currentDevice, peer) = bandwidth * 1e-9;
                } else {
                    unidirectional_memcpy(srcBuffer, dstBuffer, &bandwidth, currentSize, loopCount);
                    value_matrix.value(currentDevice, peer) = bandwidth * 1e-9;
                }

                value_sum += value_matrix.value(currentDevice, peer);
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
    std::cout << "memcpy CE GPU(row) " << (read ? "->" : "<-") << " GPU(column) bandwidth (GB/s):" << std::endl;
    std::cout << std::fixed << std::setprecision(2) << value_matrix << std::endl;
}

void launch_DtoD_memcpy_read_CE(unsigned long long size, unsigned long long loopCount) {
    launch_DtoD_memcpy_CE(true, size, loopCount);
}
void launch_DtoD_memcpy_write_CE(unsigned long long size, unsigned long long loopCount) {
    launch_DtoD_memcpy_CE(false, size, loopCount);
}

void launch_DtoD_memcpy_bidirectional_CE(unsigned long long size, unsigned long long loopCount) {
    void* dst1Buffer;
    void* src1Buffer;
    void* dst2Buffer;
    void* src2Buffer;
    unsigned long long bandwidth;
    double bandwidth_sum = 0.0;
    int deviceCount = 0;
    CUcontext benchCtx;
    benchmark_prepare(&benchCtx, &deviceCount);

    PeerValueMatrix<double> bandwidth_matrix(deviceCount);

    for (int currentDevice = 0; currentDevice < deviceCount; currentDevice++) {
        setOptimalCpuAffinity(currentDevice);
        
        CUcontext srcCtx;
        CU_ASSERT(cuDevicePrimaryCtxRetain(&srcCtx, currentDevice));
        CU_ASSERT(cuCtxSetCurrent(srcCtx));
        CU_ASSERT(cuMemAlloc((CUdeviceptr*)&src1Buffer, (size_t)size));
        CU_ASSERT(cuMemAlloc((CUdeviceptr*)&dst2Buffer, (size_t)size));

        for (int peer = 0; peer < deviceCount; peer++) {
            CUcontext peerCtx;

            int canAccessPeer = 0;
            CU_ASSERT(cuDeviceCanAccessPeer(&canAccessPeer, currentDevice, peer));

            if (canAccessPeer) {
                CU_ASSERT(cuDevicePrimaryCtxRetain(&peerCtx, peer));
                CU_ASSERT(cuCtxSetCurrent(peerCtx));
                CU_ASSERT(cuCtxEnablePeerAccess(srcCtx, 0));
                CU_ASSERT(cuMemAlloc((CUdeviceptr*)&dst1Buffer, size));
                CU_ASSERT(cuMemAlloc((CUdeviceptr*)&src2Buffer, size));
                CU_ASSERT(cuCtxSetCurrent(srcCtx));
                CU_ASSERT(cuCtxEnablePeerAccess(peerCtx, 0));
           
                bidirectional_memcpy(dst1Buffer, src1Buffer, srcCtx, dst2Buffer, src2Buffer, peerCtx, &bandwidth, size, loopCount);

                bandwidth_matrix.value(peer, currentDevice) = bandwidth * 1e-9;
                bandwidth_sum += bandwidth * 1e-9;

                CU_ASSERT(cuCtxSetCurrent(srcCtx));
                CU_ASSERT(cuCtxDisablePeerAccess(peerCtx));
                CU_ASSERT(cuCtxSetCurrent(peerCtx));
                CU_ASSERT(cuCtxDisablePeerAccess(srcCtx));
                CU_ASSERT(cuMemFree((CUdeviceptr)dst1Buffer));
                CU_ASSERT(cuMemFree((CUdeviceptr)src2Buffer));
                CU_ASSERT(cuDevicePrimaryCtxRelease(peer));
            }
        }

        benchmark_clean_bidir(&benchCtx, currentDevice, src1Buffer, dst2Buffer);
    }
    std::cout << "memcpy CE GPU <-> GPU bandwidth (GB/s)" << std::endl;
    std::cout << std::fixed << std::setprecision(2) << bandwidth_matrix << std::endl;
}
