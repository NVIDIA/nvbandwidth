#include <cstddef>
#include <iomanip>
#include <iostream>
#include <string.h>
#include <vector>

#include "benchmarks.h"
#include "common.h"
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


static void memcpyAsync_and_check(void *src, void *dst, CUcontext ctx, unsigned long long sizeInElement, unsigned int numThreadPerBlock,
	unsigned long long *bandwidth, bool stride, unsigned long long loopCount = defaultLoopCount) {
  	CUdevice device;
  	int kernelTimeout = 0;
  	CU_ASSERT(cuCtxGetDevice(&device));
  	CU_ASSERT(cuDeviceGetAttribute(&kernelTimeout, CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT, device));
  	if (kernelTimeout) {
    	double timeout = 1.8;
    	unsigned long long expectedBandwidth = 0;
    	unsigned long long smallSizeInElement = sizeInElement > 1024 * 1024 * 128 ? 1024 * 1024 * 128
			: sizeInElement;

      	memset_pattern(src, smallSizeInElement * sizeof(int4), 0xCAFEBABE);
      	memset_pattern(dst, smallSizeInElement * sizeof(int4), 0xBAADF00D);

		memcpyAsync(src, dst, smallSizeInElement, &expectedBandwidth, stride, 1);

    	unsigned long long maxBytes = (unsigned long long)((double)expectedBandwidth) * timeout * 0.25;
    	unsigned long long maxLoopcount = maxBytes / (sizeInElement * sizeof(int4));
    	maxLoopcount = maxLoopcount == 0 ? 1 : maxLoopcount;
    	if (maxLoopcount < loopCount) {
      		loopCount = maxLoopcount;
      		if (maxLoopcount == 1 && maxBytes < (sizeInElement * sizeof(int4))) {
        		sizeInElement = maxBytes / (sizeInElement * sizeof(int4));
        		if (sizeInElement == 0) {
          			*bandwidth = 0;
          			return;
        		}
      		}
    	}
  	}

    memset_pattern(src, sizeInElement * sizeof(int), 0xCAFEBABE);
    memset_pattern(dst, sizeInElement * sizeof(int), 0xBAADF00D);
	
	memcpyAsync(src, dst, sizeInElement, bandwidth, stride, loopCount);
}

static void find_best_memcpy_threadcount_per_sm(void *src, void *dst, CUcontext ctx,  unsigned long long* bandwidth, bool stride = false,
	unsigned long long size = defaultBufferSize, unsigned long long loopCount = defaultLoopCount, bool doubleBandwidth = false) {
    unsigned long long bandwidth_current;
    unsigned long long bandwidth_sum;
	std::vector<int> num_threads_per_sm;

    *bandwidth = 0;

    if (num_threads_per_sm.empty()) {
        num_threads_per_sm.push_back(512);
    }

    for (unsigned int i = 0; i < num_threads_per_sm.size(); i++) {
        memcpyAsync(src, dst, size, &bandwidth_current, stride, loopCount);
        if (doubleBandwidth) bandwidth_current *= 2;
		*bandwidth = bandwidth_current;
    }
}

static void find_best_memcpy(void *src, void *dst, CUcontext ctx, unsigned long long* bandwidth, unsigned long long size = defaultBufferSize,
	unsigned long long loopCount = defaultLoopCount, bool doubleBandwidth = false) {
    unsigned long long bandwidth_current = 0;
    *bandwidth = 0;
    find_best_memcpy_threadcount_per_sm(src, dst, ctx, &bandwidth_current, true, size, loopCount, doubleBandwidth);
	
	if (bandwidth_current > *bandwidth) { *bandwidth = bandwidth_current; }
}

void launch_HtoD_memcpy_SM(unsigned long long size, unsigned long long loopCount) {
  	int deviceCount = 0;
    void* dstBuffer;
    void* srcBuffer;
    unsigned long long bandwidth;
    double bandwidth_sum = 0.0;
	CUcontext benchCtx;

	CU_ASSERT(cuCtxGetCurrent(&benchCtx));
    CU_ASSERT(cuDeviceGetCount(&deviceCount));
    std::vector<double> bandwidthValues(deviceCount);
    CU_ASSERT(cuMemHostAlloc(&srcBuffer, (size_t)size, CU_MEMHOSTALLOC_PORTABLE));
	
    for (int currentDevice = 0; currentDevice < deviceCount; currentDevice++) {
		CUcontext srcCtx;

        CU_ASSERT(cuDevicePrimaryCtxRetain(&srcCtx, currentDevice));
        CU_ASSERT(cuCtxSetCurrent(srcCtx));

        CU_ASSERT(cuMemAlloc((CUdeviceptr*)&dstBuffer, (size_t)size));

        find_best_memcpy(dstBuffer, srcBuffer, srcCtx, &bandwidth, size, loopCount);
        bandwidthValues[currentDevice] = bandwidth * 1e-9;
        bandwidth_sum += bandwidth * 1e-9;

        CU_ASSERT(cuMemFree((CUdeviceptr)dstBuffer));
        CU_ASSERT(cuDevicePrimaryCtxRelease(currentDevice));
    }

	CU_ASSERT(cuCtxSetCurrent(benchCtx));
    CU_ASSERT(cuMemFreeHost(srcBuffer));

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

	CU_ASSERT(cuCtxGetCurrent(&benchCtx));
    CU_ASSERT(cuDeviceGetCount(&deviceCount));
    std::vector<double> bandwidthValues(deviceCount);
    CU_ASSERT(cuMemHostAlloc(&dstBuffer, (size_t)size, CU_MEMHOSTALLOC_PORTABLE));

    for (int currentDevice = 0; currentDevice < deviceCount; currentDevice++) {
        CUcontext srcCtx;

        CU_ASSERT(cuDevicePrimaryCtxRetain(&srcCtx, currentDevice));
        CU_ASSERT(cuCtxSetCurrent(srcCtx));

        CU_ASSERT(cuMemAlloc((CUdeviceptr*)&srcBuffer, (size_t)size));

        find_best_memcpy(dstBuffer, srcBuffer, srcCtx, &bandwidth, size, loopCount);
        CU_ASSERT(cuMemFree((CUdeviceptr)srcBuffer));

        bandwidthValues[currentDevice] = bandwidth * 1e-9;
        bandwidth_sum += bandwidth * 1e-9;

        CU_ASSERT(cuDevicePrimaryCtxRelease(currentDevice));
    }
	
	CU_ASSERT(cuCtxSetCurrent(benchCtx));
    CU_ASSERT(cuMemFreeHost(dstBuffer));

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
                    find_best_memcpy(dstBuffer, srcBuffer, srcCtx, &bandwidth, currentSize, loopCount);
                    value_matrix.value(currentDevice, peer) = bandwidth * 1e-9;
                } else {
                    find_best_memcpy(srcBuffer, dstBuffer, srcCtx, &bandwidth, currentSize, loopCount);
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

	CU_ASSERT(cuCtxGetCurrent(&benchCtx));
  	unsigned long long bandwidth;
  	int deviceCount = 0;
  	double bandwidth_sum = 0.0;

  	CU_ASSERT(cuDeviceGetCount(&deviceCount));
  	PeerValueMatrix<double> bandwidth_matrix(deviceCount);

  	for (int currentDevice = 0; currentDevice < deviceCount; currentDevice++) {
    	CU_ASSERT(cuDevicePrimaryCtxRetain(&srcCtx, currentDevice));
    	CU_ASSERT(cuCtxSetCurrent(srcCtx));

    	CU_ASSERT(cuCtxGetDevice(&currentDevice));
    	CU_ASSERT(cuMemAlloc((CUdeviceptr *)&gpuAbuffer0, (size_t)size));
    	CU_ASSERT(cuMemAlloc((CUdeviceptr *)&gpuAbuffer1, (size_t)size));

    	for (int peer = 0; peer < deviceCount; peer++) {
      		CUcontext peerCtx;
      		std::vector<BenchParams> params;

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

        		if (read) {
          			params.push_back(BenchParams(gpuAbuffer0, gpuBbuffer0, srcCtx));
          			params.push_back(BenchParams(gpuBbuffer1, gpuAbuffer1, peerCtx));
        		} else {
          			params.push_back(BenchParams(gpuBbuffer0, gpuAbuffer0, srcCtx));
          			params.push_back(BenchParams(gpuAbuffer1, gpuBbuffer1, peerCtx));
        		}

        		unsigned long long device_current_bandwidth = 0;
        		unsigned long long bandwidth_current;
        		unsigned long long bandwidth_sum;

        		unsigned int num_threads_per_sm = 512;

        		bandwidth_sum = 0;
        		for (unsigned int n = 0; n < loopCount; n++) {
					for (BenchParams param : params) {
						find_best_memcpy(param.src, param.dst, param.ctx, &bandwidth, size / sizeof(int), loopCount);
					}
          			bandwidth_sum += bandwidth_current;
        		}
        		bandwidth_sum /= loopCount;

        		if (bandwidth_sum > device_current_bandwidth) device_current_bandwidth = bandwidth_sum;

		        bandwidth_matrix.value(currentDevice, peer) = bandwidth * 1e-9;
        		bandwidth_sum += bandwidth * 1e-9;

        		CU_ASSERT(cuCtxSetCurrent(srcCtx));
        		CU_ASSERT(cuCtxDisablePeerAccess(peerCtx));
        		CU_ASSERT(cuCtxSetCurrent(peerCtx));
        		CU_ASSERT(cuCtxDisablePeerAccess(srcCtx));

        		CU_ASSERT(cuMemFree((CUdeviceptr)gpuBbuffer0));
        		CU_ASSERT(cuMemFree((CUdeviceptr)gpuBbuffer1));
        		CU_ASSERT(cuDevicePrimaryCtxRelease(peer));
      		}
    	}
		CU_ASSERT(cuCtxSetCurrent(benchCtx));
    	CU_ASSERT(cuMemFree((CUdeviceptr)gpuAbuffer0));
    	CU_ASSERT(cuMemFree((CUdeviceptr)gpuAbuffer1));
    	CU_ASSERT(cuDevicePrimaryCtxRelease(currentDevice));
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
