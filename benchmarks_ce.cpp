#include <algorithm>
#include <cuda.h>
#include <iomanip>
#include <iostream>
#include <stdio.h>
#include <string.h>
#include <time.h>

#include "benchmarks.h"
#include "memory_utils.h"

const size_t WARMUP_COUNT = 32;
const size_t START_COPY_MARKER_SIZE = 17;

static void memcpyAsync(void *dst, void *src, unsigned long long size,
                        unsigned long long *bandwidth, bool isPageable,
                        unsigned long long loopCount = defaultLoopCount) {
  	CUstream stream;
  	CUevent startEvent;
  	CUevent endEvent;
  	volatile int *blockingVar = NULL;
  	void *markerDst = NULL, *markerSrc = NULL, *additionalMarkerLocation = NULL;

  	CU_ASSERT(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
  	CU_ASSERT(cuEventCreate(&startEvent, CU_EVENT_DEFAULT));
  	CU_ASSERT(cuEventCreate(&endEvent, CU_EVENT_DEFAULT));

  	// Spend time on the GPU so we finish submitting everything before the
  	// benchmark starts Also events are tied to the last submission channel so we
  	// want to be sure it is copy and not compute
  	for (unsigned int n = 0; n < WARMUP_COUNT; n++) {
    	// As latency benchmarks do 1 byte copies, we have to ensure we're not doing
    	// 0 byte copies
    	CU_ASSERT(cuMemcpyAsync((CUdeviceptr)dst, (CUdeviceptr)src, (size_t)((size + 7) / 8), stream));
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

static void
memcpyAsync_bidirectional(void *dst1, void *src1, CUcontext ctx1, void *dst2,
                        void *src2, CUcontext ctx2, unsigned long long size,
                        unsigned long long *bandwidth,
                        unsigned long long loopCount = defaultLoopCount) {
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

  	// Spend time on the GPU so we finish submitting everything before the
  	// benchmark starts Also events are tied to the last submission channel so we
  	// want to be sure it is copy and not compute
  	for (unsigned int n = 0; n < WARMUP_COUNT; n++) {
    	// As latency benchmarks do 1 byte copies, we have to ensure we're not doing
    	// 0 byte copies
    	CU_ASSERT(cuMemcpyAsync((CUdeviceptr)dst1, (CUdeviceptr)src1,
                            (size_t)((size + 7) / 8), stream_dir1));
    	CU_ASSERT(cuMemcpyAsync((CUdeviceptr)dst2, (CUdeviceptr)src2,
                            (size_t)((size + 7) / 8), stream_dir2));
  	}

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

static void memcpy_and_check(void *dst, void *src, unsigned long long size,
                            unsigned long long *bandwidth,
                            unsigned long long loopCount = defaultLoopCount) {
  	memset_pattern(src, size, 0xCAFEBABE);
  	memset_pattern(dst, size, 0xBAADF00D);

  	bool isPageable = !isMemoryOwnedByCUDA(dst) || !isMemoryOwnedByCUDA(src);
  	memcpyAsync(dst, src, size, bandwidth, isPageable, loopCount);
  	memcmp_pattern(dst, size, 0xCAFEBABE);
}

static void memcpyAsync_and_check_bidirectional(void *dst1, void *src1, CUcontext ctx1, void *dst2, void *src2,
    										CUcontext ctx2, unsigned long long size, unsigned long long *bandwidth,
											unsigned long long loopCount = defaultLoopCount) {
	memset_pattern(src1, size, 0xCAFEBABE);
  	memset_pattern(dst1, size, 0xBAADF00D);
  	memset_pattern(src2, size, 0xFEEEFEEE);
  	memset_pattern(dst2, size, 0xFACEFEED);
  	memcpyAsync_bidirectional(dst1, src1, ctx1, dst2, src2, ctx2, size, bandwidth, loopCount);
  	memcmp_pattern(dst1, size, 0xCAFEBABE);
  	memcmp_pattern(dst2, size, 0xFEEEFEEE);
}

static void find_best_memcpy(void *src, void *dst, unsigned long long *bandwidth,
                            unsigned long long size, unsigned long long loopCount) {
  	unsigned long long bandwidth_current;
  	cudaStat bandwidthStat;

  	*bandwidth = 0;
  	for (unsigned int n = 0; n < averageLoopCount; n++) {
    	memcpy_and_check(dst, src, size, &bandwidth_current, loopCount);
    	bandwidthStat((double)bandwidth_current);
  	}
  	*bandwidth = (unsigned long long)(STAT_MEAN(bandwidthStat));
}

static void find_memcpy_time(void *src, void *dst, double *time_us, unsigned long long size,
                            unsigned long long loopCount) {
  	unsigned long long bandwidth;
  	find_best_memcpy(src, dst, &bandwidth, size, loopCount);
  	*time_us = size * 1e6 / bandwidth;
}

static void find_best_memcpy_bidirectional(void *dst1, void *src1, CUcontext ctx1, void *dst2,
                                        void *src2, CUcontext ctx2, unsigned long long *bandwidth,
                                        unsigned long long size, unsigned long long loopCount) {
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

size_t getFirstEnabledCPU() {
  	size_t firstEnabledCPU = 0;
  	size_t *procMask = (size_t *)calloc(1, PROC_MASK_SIZE);
  	for (size_t i = 0; i < PROC_MASK_SIZE * 8; ++i) {
    	if (PROC_MASK_QUERY_BIT(procMask, i)) {
      		firstEnabledCPU = i;
      	break;
    	}
  	}
  	free(procMask);
  	return firstEnabledCPU;
}

void launch_HtoD_memcpy_bidirectional_CE(const std::string &test_name, unsigned long long size,
                                        unsigned long long loopCount) {
  	CUcontext srcCtx;
  	void *HtoD_dstBuffer;
  	void *HtoD_srcBuffer;
  	void *DtoH_dstBuffer;
  	void *DtoH_srcBuffer;
  	unsigned long long bandwidth;
  	double bandwidth_sum = 0.0;
  	size_t *procMask = NULL;
  	size_t firstEnabledCPU = getFirstEnabledCPU();
  	size_t procCount = 1;
  	int deviceCount;

  	CU_ASSERT(cuDeviceGetCount(&deviceCount));

  	PeerValueMatrix<double> bandwidthValues((int)procCount, deviceCount);

  	procMask = (size_t *)calloc(1, PROC_MASK_SIZE);

  	for (size_t procId = 0; procId < procCount; procId++) {
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

      		find_best_memcpy_bidirectional(HtoD_dstBuffer, HtoD_srcBuffer, srcCtx, DtoH_dstBuffer,
                                    DtoH_srcBuffer, srcCtx, &bandwidth, size, loopCount);

      		bandwidthValues.value((int)procId, currentDevice) = bandwidth * 1e-9;
      		bandwidth_sum += bandwidth * 1e-9;

      		CU_ASSERT(cuMemFree((CUdeviceptr)DtoH_srcBuffer));
      		CU_ASSERT(cuMemFree((CUdeviceptr)HtoD_dstBuffer));
      		CU_ASSERT(cuDevicePrimaryCtxRelease(currentDevice));
    	}

    	retain_ctx();
    
    	CU_ASSERT(cuMemFreeHost(HtoD_srcBuffer));
    	CU_ASSERT(cuMemFreeHost(DtoH_dstBuffer));

    	PROC_MASK_CLEAR(procMask, procId);
  	}

  	free(procMask);

  	std::cout << "memcpy CE GPU(columns) <- CPU(rows) bandwidth (GB/s)" << std::endl;
  	std::cout << std::fixed << std::setprecision(2) << bandwidthValues << std::endl;
}

void launch_DtoH_memcpy_bidirectional_CE(const std::string &test_name, unsigned long long size,
                                        unsigned long long loopCount) {
  	CUcontext srcCtx;
  	void *HtoD_dstBuffer;
  	void *HtoD_srcBuffer;
  	void *DtoH_dstBuffer;
  	void *DtoH_srcBuffer;
  	unsigned long long bandwidth;
  	double bandwidth_sum = 0.0;
  	size_t *procMask = NULL;
  	size_t firstEnabledCPU = getFirstEnabledCPU();
  	size_t procCount = 1;
  	int deviceCount;

  	CU_ASSERT(cuDeviceGetCount(&deviceCount));

  	PeerValueMatrix<double> bandwidthValues((int)procCount, deviceCount);

  	procMask = (size_t *)calloc(1, PROC_MASK_SIZE);

  	for (size_t procId = 0; procId < procCount; procId++) {
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

      		find_best_memcpy_bidirectional(DtoH_dstBuffer, DtoH_srcBuffer, srcCtx, HtoD_dstBuffer, HtoD_srcBuffer,
                                    srcCtx, &bandwidth, size, loopCount);
      		bandwidthValues.value((int)procId, currentDevice) = bandwidth * 1e-9;
      		bandwidth_sum += bandwidth * 1e-9;

      		CU_ASSERT(cuMemFree((CUdeviceptr)DtoH_srcBuffer));
      		CU_ASSERT(cuMemFree((CUdeviceptr)HtoD_dstBuffer));
      		CU_ASSERT(cuDevicePrimaryCtxRelease(currentDevice));
    	}

    	retain_ctx();

    	CU_ASSERT(cuMemFreeHost(HtoD_srcBuffer));
    	CU_ASSERT(cuMemFreeHost(DtoH_dstBuffer));

    	PROC_MASK_CLEAR(procMask, procId);
  	}

  	free(procMask);

  	std::cout << "memcpy CE GPU(columns) <- CPU(rows) bandwidth (GB/s)" << std::endl;
  	std::cout << std::fixed << std::setprecision(2) << bandwidthValues << std::endl;
}
