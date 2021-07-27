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

static void memcpyAsync(void *dst, void *src, unsigned long long size, unsigned long long *bandwidth, bool isPageable, unsigned long long loopCount = defaultLoopCount) {
  	CUstream stream;
  	CUevent startEvent;
  	CUevent endEvent;
  	volatile int *blockingVar = NULL;

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
}

static void
memcpyAsync_bidirectional(void *dst1, void *src1, CUcontext ctx1, void *dst2, void *src2, CUcontext ctx2, unsigned long long size, unsigned long long *bandwidth, unsigned long long loopCount = defaultLoopCount) {
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

static void memcpy_and_check(void *dst, void *src, unsigned long long size, unsigned long long *bandwidth, unsigned long long loopCount = defaultLoopCount) {
  	memset_pattern(src, size, 0xCAFEBABE);
  	memset_pattern(dst, size, 0xBAADF00D);

  	bool isPageable = !isMemoryOwnedByCUDA(dst) || !isMemoryOwnedByCUDA(src);
  	memcpyAsync(dst, src, size, bandwidth, isPageable, loopCount);
  	memcmp_pattern(dst, size, 0xCAFEBABE);
}

static void memcpyAsync_and_check_bidirectional(void *dst1, void *src1, CUcontext ctx1, void *dst2, void *src2, CUcontext ctx2,
	unsigned long long size, unsigned long long *bandwidth, unsigned long long loopCount = defaultLoopCount) {
	memset_pattern(src1, size, 0xCAFEBABE);
  	memset_pattern(dst1, size, 0xBAADF00D);
  	memset_pattern(src2, size, 0xFEEEFEEE);
  	memset_pattern(dst2, size, 0xFACEFEED);
  	memcpyAsync_bidirectional(dst1, src1, ctx1, dst2, src2, ctx2, size, bandwidth, loopCount);
  	memcmp_pattern(dst1, size, 0xCAFEBABE);
  	memcmp_pattern(dst2, size, 0xFEEEFEEE);
}

static void find_best_memcpy(void *src, void *dst, unsigned long long *bandwidth, unsigned long long size, unsigned long long loopCount) {
  	unsigned long long bandwidth_current;
  	cudaStat bandwidthStat;

  	*bandwidth = 0;
  	for (unsigned int n = 0; n < averageLoopCount; n++) {
    	memcpy_and_check(dst, src, size, &bandwidth_current, loopCount);
    	bandwidthStat((double)bandwidth_current);
  	}
  	*bandwidth = (unsigned long long)(STAT_MEAN(bandwidthStat));
}

static void find_memcpy_time(void *src, void *dst, double *time_us, unsigned long long size, unsigned long long loopCount) {
  	unsigned long long bandwidth;
  	find_best_memcpy(src, dst, &bandwidth, size, loopCount);
  	*time_us = size * 1e6 / bandwidth;
}

static void find_best_memcpy_bidirectional(void *dst1, void *src1, CUcontext ctx1, void *dst2, void *src2, CUcontext ctx2, 
	unsigned long long *bandwidth, unsigned long long size, unsigned long long loopCount) {
  	
	unsigned long long bandwidth_current;
  	cudaStat bandwidthStat;

  	*bandwidth = 0;
  	for (unsigned int n = 0; n < averageLoopCount; n++) {
    	memcpyAsync_and_check_bidirectional(dst1, src1, ctx1, dst2, src2, ctx2, size, 
                                        &bandwidth_current, loopCount);
    	bandwidthStat((double)bandwidth_current);
  	}
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

	CU_ASSERT(cuCtxGetCurrent(&benchCtx));
    CU_ASSERT(cuDeviceGetCount(&deviceCount));
  	PeerValueMatrix<double> bandwidthValues(1, deviceCount);
    procMask = (size_t *)calloc(1, PROC_MASK_SIZE);

	PROC_MASK_SET(procMask, firstEnabledCPU);

	cuMemHostAlloc(&srcBuffer, (size_t)size, CU_MEMHOSTALLOC_PORTABLE);

    for (size_t devIdx = 0; devIdx < deviceCount; devIdx++) {
        int currentDevice = devIdx;
        CUcontext srcCtx;

        CU_ASSERT(cuDevicePrimaryCtxRetain(&srcCtx, currentDevice));
        CU_ASSERT(cuCtxSetCurrent(srcCtx));
            
        CU_ASSERT(cuMemAlloc((CUdeviceptr*)&dstBuffer, (size_t)size));

        unsigned long long bandwidth;
        find_best_memcpy(srcBuffer, dstBuffer, &bandwidth, size, loopCount);
        bandwidthValues.value(0, currentDevice) = bandwidth * 1e-9;
        perf_value_sum += bandwidth * 1e-9;

        CU_ASSERT(cuMemFree((CUdeviceptr)dstBuffer));

        CU_ASSERT(cuDevicePrimaryCtxRelease(currentDevice));

        PROC_MASK_CLEAR(procMask, 0);
    }

	CU_ASSERT(cuCtxSetCurrent(benchCtx));
    freeHostMemory(srcBuffer);

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

	CU_ASSERT(cuCtxGetCurrent(&benchCtx));
    CU_ASSERT(cuDeviceGetCount(&deviceCount));
  	PeerValueMatrix<double> bandwidthValues(1, deviceCount);
    procMask = (size_t *)calloc(1, PROC_MASK_SIZE);

    PROC_MASK_SET(procMask, firstEnabledCPU);

	CU_ASSERT(cuMemHostAlloc(&dstBuffer, (size_t)size, CU_MEMHOSTALLOC_PORTABLE));
        
	for (size_t devIdx = 0; devIdx < deviceCount; devIdx++) {
        int currentDevice = devIdx;
        CUcontext srcCtx;

        CU_ASSERT(cuDevicePrimaryCtxRetain(&srcCtx, currentDevice));
        CU_ASSERT(cuCtxSetCurrent(srcCtx));
        CU_ASSERT(cuMemAlloc((CUdeviceptr*)&srcBuffer, size));

        unsigned long long bandwidth;
        find_best_memcpy(srcBuffer, dstBuffer, &bandwidth, size, loopCount);
        bandwidthValues.value(0, currentDevice) = bandwidth * 1e-9;
        perf_value_sum += bandwidth * 1e-9;

        CU_ASSERT(cuMemFree((CUdeviceptr)srcBuffer));

        CU_ASSERT(cuDevicePrimaryCtxRelease(currentDevice));
    }

	CU_ASSERT(cuCtxSetCurrent(benchCtx));
    CU_ASSERT(cuMemFreeHost(dstBuffer));
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

	CU_ASSERT(cuCtxGetCurrent(&benchCtx));
  	CU_ASSERT(cuDeviceGetCount(&deviceCount));
  	PeerValueMatrix<double> bandwidthValues(1, deviceCount);
  	procMask = (size_t *)calloc(1, PROC_MASK_SIZE);

    PROC_MASK_SET(procMask, firstEnabledCPU);

    /* The NUMA location of the calling thread determines the physical
       	location of the pinned memory allocation, which can have different
       	performance characteristics */
    CU_ASSERT(cuMemHostAlloc(&HtoD_srcBuffer, (size_t)size, CU_MEMHOSTALLOC_PORTABLE));
    CU_ASSERT(cuMemHostAlloc(&DtoH_dstBuffer, (size_t)size, CU_MEMHOSTALLOC_PORTABLE));

    for (size_t devIdx = 0; devIdx < deviceCount; devIdx++) {
      	int currentDevice = devIdx;

      	CU_ASSERT(cuDevicePrimaryCtxRetain(&srcCtx, currentDevice));
      	CU_ASSERT(cuCtxSetCurrent(srcCtx));

      	CU_ASSERT(cuMemAlloc((CUdeviceptr *)&HtoD_dstBuffer, (size_t)size));
      	CU_ASSERT(cuMemAlloc((CUdeviceptr *)&DtoH_srcBuffer, (size_t)size));

      	find_best_memcpy_bidirectional(HtoD_dstBuffer, HtoD_srcBuffer, srcCtx, DtoH_dstBuffer, DtoH_srcBuffer, srcCtx, &bandwidth, size, loopCount);

      	bandwidthValues.value(0, currentDevice) = bandwidth * 1e-9;
      	bandwidth_sum += bandwidth * 1e-9;

      	CU_ASSERT(cuMemFree((CUdeviceptr)DtoH_srcBuffer));
      	CU_ASSERT(cuMemFree((CUdeviceptr)HtoD_dstBuffer));
      	CU_ASSERT(cuDevicePrimaryCtxRelease(currentDevice));
    }
    
	CU_ASSERT(cuCtxSetCurrent(benchCtx));
    CU_ASSERT(cuMemFreeHost(HtoD_srcBuffer));
    CU_ASSERT(cuMemFreeHost(DtoH_dstBuffer));

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
	CU_ASSERT(cuCtxGetCurrent(&benchCtx));

  	CU_ASSERT(cuDeviceGetCount(&deviceCount));
  	PeerValueMatrix<double> bandwidthValues(1, deviceCount);
  	procMask = (size_t *)calloc(1, PROC_MASK_SIZE);

    PROC_MASK_SET(procMask, firstEnabledCPU);

    /* The NUMA location of the calling thread determines the physical
       	location of the pinned memory allocation, which can have different
       	performance characteristics */
    CU_ASSERT(cuMemHostAlloc(&HtoD_srcBuffer, (size_t)size, CU_MEMHOSTALLOC_PORTABLE));
    CU_ASSERT(cuMemHostAlloc(&DtoH_dstBuffer, (size_t)size, CU_MEMHOSTALLOC_PORTABLE));

    for (size_t devIdx = 0; devIdx < deviceCount; devIdx++) {
      	int currentDevice = devIdx;

      	CU_ASSERT(cuDevicePrimaryCtxRetain(&srcCtx, currentDevice));
      	CU_ASSERT(cuCtxSetCurrent(srcCtx));

      	CU_ASSERT(cuMemAlloc((CUdeviceptr *)&HtoD_dstBuffer, (size_t)size));
      	CU_ASSERT(cuMemAlloc((CUdeviceptr *)&DtoH_srcBuffer, (size_t)size));

      	find_best_memcpy_bidirectional(DtoH_dstBuffer, DtoH_srcBuffer, srcCtx, HtoD_dstBuffer, HtoD_srcBuffer, srcCtx, &bandwidth, size, loopCount);
      	bandwidthValues.value(0, currentDevice) = bandwidth * 1e-9;
      	bandwidth_sum += bandwidth * 1e-9;

      	CU_ASSERT(cuMemFree((CUdeviceptr)DtoH_srcBuffer));
      	CU_ASSERT(cuMemFree((CUdeviceptr)HtoD_dstBuffer));
    	CU_ASSERT(cuDevicePrimaryCtxRelease(currentDevice));
    }

	CU_ASSERT(cuCtxSetCurrent(benchCtx));
    CU_ASSERT(cuMemFreeHost(HtoD_srcBuffer));
    CU_ASSERT(cuMemFreeHost(DtoH_dstBuffer));

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

	CU_ASSERT(cuCtxGetCurrent(&benchCtx));
    CU_ASSERT(cuDeviceGetCount(&deviceCount));
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
                    find_best_memcpy(dstBuffer, srcBuffer, &bandwidth, currentSize, loopCount);
                    value_matrix.value(currentDevice, peer) = bandwidth * 1e-9;
                } else {
                    find_best_memcpy(srcBuffer, dstBuffer, &bandwidth, currentSize, loopCount);
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
		CU_ASSERT(cuCtxSetCurrent(benchCtx));
        CU_ASSERT(cuMemFree((CUdeviceptr)srcBuffer));
        CU_ASSERT(cuDevicePrimaryCtxRelease(currentDevice));
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

	CU_ASSERT(cuCtxGetCurrent(&benchCtx));
    CU_ASSERT(cuDeviceGetCount(&deviceCount));
    PeerValueMatrix<double> bandwidth_matrix(deviceCount);

    for (int currentDevice = 0; currentDevice < deviceCount; currentDevice++) {
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
           
                find_best_memcpy_bidirectional(dst1Buffer, src1Buffer, srcCtx, dst2Buffer, src2Buffer, peerCtx, &bandwidth, size, loopCount);

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
		CU_ASSERT(cuCtxSetCurrent(benchCtx));
        CU_ASSERT(cuMemFree((CUdeviceptr)src1Buffer));
        CU_ASSERT(cuMemFree((CUdeviceptr)dst2Buffer));
        CU_ASSERT(cuDevicePrimaryCtxRelease(currentDevice));
    }
    std::cout << "memcpy CE GPU <-> GPU bandwidth (GB/s)" << std::endl;
    std::cout << std::fixed << std::setprecision(2) << bandwidth_matrix << std::endl;
}
