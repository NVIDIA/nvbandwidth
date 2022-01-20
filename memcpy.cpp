/*
 * Copyright 1993-2021 by NVIDIA Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#include "memcpy.h"

MemcpyNode::MemcpyNode(): buffer(nullptr) {}

void *MemcpyNode::getBuffer() {
    return buffer;
}

HostNode::HostNode(size_t bufferSize, int targetDeviceId): MemcpyNode() {
    CUcontext targetCtx;

    // Before allocating host memory, set correct NUMA affinity
    setOptimalCpuAffinity(targetDeviceId);
    CU_ASSERT(cuDevicePrimaryCtxRetain(&targetCtx, targetDeviceId));
    CU_ASSERT(cuCtxSetCurrent(targetCtx));

    CU_ASSERT(cuMemHostAlloc(&buffer, bufferSize, CU_MEMHOSTALLOC_PORTABLE));
}

HostNode::~HostNode() {
    if (isMemoryOwnedByCUDA(buffer)) {
        CU_ASSERT(cuMemFreeHost(buffer));
    } else {
        free(buffer);
    }
}

// Host Nodes always return zero as they always represent one row in the bandwidth matrix
int HostNode::getNodeIdx() const {
    return 0;
}

DeviceNode::DeviceNode(size_t bufferSize, int deviceIdx): deviceIdx(deviceIdx), MemcpyNode() {
    CU_ASSERT(cuDevicePrimaryCtxRetain(&primaryCtx, deviceIdx));
    CU_ASSERT(cuCtxSetCurrent(primaryCtx));
    CU_ASSERT(cuMemAlloc((CUdeviceptr*)&buffer, bufferSize));
}

DeviceNode::~DeviceNode() {
    CU_ASSERT(cuCtxSetCurrent(primaryCtx));
    CU_ASSERT(cuMemFree((CUdeviceptr)buffer));
    CU_ASSERT(cuDevicePrimaryCtxRelease(deviceIdx));
}

CUcontext DeviceNode::getPrimaryCtx() const {
    return primaryCtx;
}

int DeviceNode::getNodeIdx() const {
    return deviceIdx;
}

Memcpy::Memcpy(MemcpyCEFunc memcpyFunc, size_t copySize, unsigned long long loopCount): ceFunc(memcpyFunc), copySize(copySize), loopCount(loopCount) {
    procMask = (size_t *)calloc(1, PROC_MASK_SIZE);
    PROC_MASK_SET(procMask, getFirstEnabledCPU());
}

Memcpy::Memcpy(MemcpySMFunc memcpyFunc, size_t copySize, unsigned long long loopCount): smFunc(memcpyFunc), copySize(copySize), loopCount(loopCount) {
    procMask = (size_t *)calloc(1, PROC_MASK_SIZE);
    PROC_MASK_SET(procMask, getFirstEnabledCPU());
}

Memcpy::~Memcpy() {
    delete bandwidthValues;
    PROC_MASK_CLEAR(procMask, 0);
}

// H2D and bidir H2D
void Memcpy::doMemcpy(HostNode *srcNode, DeviceNode *dstNode, HostNode *biDirSrc, DeviceNode *biDirDst) {
    unsigned long long bandwidth, bandwidthBiDir = 0;
    allocateBandwidthMatrix(true);
    // Set context of the copy
    copyCtx = dstNode->getPrimaryCtx();
    bandwidth = doMemcpy((CUdeviceptr)srcNode->getBuffer(), (CUdeviceptr)dstNode->getBuffer());
    if (biDirSrc) bandwidthBiDir = doMemcpy((CUdeviceptr)biDirSrc->getBuffer(), (CUdeviceptr)biDirDst->getBuffer());
    bandwidthValues->value(0, dstNode->getNodeIdx()) = (double)(biDirSrc? bandwidthBiDir : bandwidth) * 1e-9;
}

// D2H and bidir D2H
void Memcpy::doMemcpy(DeviceNode *srcNode, HostNode *dstNode, DeviceNode *biDirSrc, HostNode *biDirDst) {
    unsigned long long bandwidth, bandwidthBiDir = 0;
    allocateBandwidthMatrix(true);
    // Set context of the copy
    copyCtx = srcNode->getPrimaryCtx();
    bandwidth = doMemcpy((CUdeviceptr)srcNode->getBuffer(), (CUdeviceptr)dstNode->getBuffer());
    if (biDirSrc) bandwidthBiDir = doMemcpy((CUdeviceptr)biDirSrc->getBuffer(), (CUdeviceptr)biDirDst->getBuffer());
    bandwidthValues->value(0, srcNode->getNodeIdx()) = (double)(biDirSrc? bandwidthBiDir : bandwidth) * 1e-9;
}

// D2D and bidir D2D
void Memcpy::doMemcpy(DeviceNode *srcNode, DeviceNode *dstNode, DeviceNode *biDirSrc, DeviceNode *biDirDst) {
    unsigned long long bandwidth, bandwidthBiDir = 0;

    allocateBandwidthMatrix();
    // Set context of the copy
    copyCtx = srcNode->getPrimaryCtx();

    /**
     * The `skip` variable is needed because of OpenMP barriers. Barrier expects as many threads as deviceCount,
     * therefore we need to call doMemcpy deviceCount times but we'll skip the memcpy ops that aren't
     * supposed to happen (such as no peer access or src == dst).
     */
    bool skip = true;
    // Benchmark against yourself?
    if (srcNode->getNodeIdx() != dstNode->getNodeIdx()) {
        int canAccessPeer = 0;
        CU_ASSERT(cuDeviceCanAccessPeer(&canAccessPeer, srcNode->getNodeIdx(), dstNode->getNodeIdx()));
        if (canAccessPeer) {
            CUresult res;
            CU_ASSERT(cuCtxSetCurrent(dstNode->getPrimaryCtx()));
            res = cuCtxEnablePeerAccess(srcNode->getPrimaryCtx(), 0);
            if (res != CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED) CU_ASSERT(res);
            CU_ASSERT(cuCtxSetCurrent(srcNode->getPrimaryCtx()));
            res = cuCtxEnablePeerAccess(dstNode->getPrimaryCtx(), 0);
            if (res != CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED) CU_ASSERT(res);

            cuCtxSetCurrent(srcNode->getPrimaryCtx());
            skip = false;
        }
    }

    bandwidth = doMemcpy((CUdeviceptr)srcNode->getBuffer(), (CUdeviceptr)dstNode->getBuffer(), skip);
    if (biDirSrc) bandwidthBiDir = doMemcpy((CUdeviceptr)biDirSrc->getBuffer(), (CUdeviceptr)biDirDst->getBuffer(), skip);
    bandwidthValues->value(dstNode->getNodeIdx(), srcNode->getNodeIdx()) = (double)(bandwidth + bandwidthBiDir) * 1e-9;
}

unsigned long long Memcpy::doMemcpy(CUdeviceptr src, CUdeviceptr dst, bool skip) {
    CUstream stream;
    CUevent startEvent;
    CUevent endEvent;
    PerformanceStatistic bandwidthStat;

    // This loop is for sampling the benchmark (which itself has a loop count)
    for (unsigned int n = 0; n < averageLoopCount; n++) {
        volatile int *blockingVar = nullptr;
        bool useSM = smFunc != nullptr;

        // Set context and create stream and events
        CU_ASSERT(cuCtxSetCurrent(copyCtx));
        CU_ASSERT(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
        CU_ASSERT(cuEventCreate(&startEvent, CU_EVENT_DEFAULT));
        CU_ASSERT(cuEventCreate(&endEvent, CU_EVENT_DEFAULT));

        #pragma omp critical
        {
            if (parallel && !skip) {
                // Synchronize all copy operations to start at the same time
                if (masterStream == nullptr) {
                    masterStream = &stream;
                    masterEvent = &startEvent;
                } else {
                    // Ensure we start the copy when the master instance signals
                    CU_ASSERT(cuStreamWaitEvent(stream, *masterEvent, 0));
                }
                CU_ASSERT(cuEventRecord(startEvent, stream));
            } else if (!parallel) {
                CU_ASSERT(cuEventRecord(startEvent, stream));
            }
        }
        #pragma omp barrier

        if (!skip) {
            // Run and record memcpy
            if (useSM) CU_ASSERT(smFunc(dst, src, smCopySize(), stream, loopCount));
            else for (unsigned int l = 0; l < loopCount; l++) CU_ASSERT(ceFunc(dst, src, copySize, stream));
            CU_ASSERT(cuEventRecord(endEvent, stream));

            // Finish memcpy
            if (!disableP2P) {
                CU_ASSERT(cuMemHostAlloc((void **) &blockingVar, sizeof(*blockingVar), CU_MEMHOSTALLOC_PORTABLE));
                *blockingVar = 1;
            }
            CU_ASSERT(cuStreamSynchronize(stream));

            // Calculate bandwidth
            CU_ASSERT(cuCtxSetCurrent(copyCtx));
            float timeWithEvents = 0.0f;
            CU_ASSERT(cuEventElapsedTime(&timeWithEvents, startEvent, endEvent));
            double elapsedWithEventsInUs = ((double) timeWithEvents * 1000.0);
            unsigned long long bandwidth = (copySize * loopCount * 1000ull * 1000ull) / (unsigned long long) elapsedWithEventsInUs;
            bandwidthStat((double) bandwidth);
            VERBOSE << "\tSample " << n << ' ' << std::fixed << std::setprecision(2) << (double) bandwidth * 1e-9 << " GB/s\n";

            // Clean up
            if (!disableP2P) CU_ASSERT(cuMemFreeHost((void *) blockingVar));
            CU_ASSERT(cuCtxSynchronize());
        }

        #pragma omp barrier
        // Reset master stream and event for next iteration
        masterStream = nullptr;
        masterEvent = nullptr;
        #pragma omp barrier
    }

    return (unsigned long long)(STAT_MEAN(bandwidthStat));
}

void Memcpy::printBenchmarkMatrix(bool reverse) {
    std::cout << "memcpy CE GPU(row) " << (reverse ? "->" : "<-") << " GPU(column) bandwidth (GB/s):" << std::endl;
    std::cout << std::fixed << std::setprecision(2) << *bandwidthValues << std::endl;
}

size_t Memcpy::smCopySize() const {
    CUdevice cudaDevice;
    int multiProcessorCount;
    size_t size = copySize;

    size /= sizeof(int4);
    CU_ASSERT(cuCtxGetDevice(&cudaDevice));
    CU_ASSERT(cuDeviceGetAttribute(&multiProcessorCount,CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, cudaDevice));
    unsigned long long totalThreadCount = (unsigned long long)(multiProcessorCount * numThreadPerBlock);
    size = totalThreadCount * (size / totalThreadCount);
    return size;
}

void Memcpy::allocateBandwidthMatrix(bool hostVector) {
    #pragma omp critical
    {
        if (bandwidthValues == nullptr) {
            int rows = hostVector ? 1 : deviceCount;
            bandwidthValues = new PeerValueMatrix<double>(rows, deviceCount);
        }
    }
}
