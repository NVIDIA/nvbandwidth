/*
 * Copyright 1993-2021 by NVIDIA Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#include "memcpy.h"
#include "kernels.cuh"

MemcpyNode::MemcpyNode(): buffer(nullptr) {}

CUdeviceptr MemcpyNode::getBuffer() {
    return (CUdeviceptr)buffer;
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

// Host nodes don't have a context, return null
CUcontext HostNode::getPrimaryCtx() const {
    return nullptr;
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

MemcpyOperation::MemcpyOperation(MemcpyCEFunc memcpyFunc, size_t copySize, unsigned long long loopCount): ceFunc(memcpyFunc), copySize(copySize), loopCount(loopCount) {
    procMask = (size_t *)calloc(1, PROC_MASK_SIZE);
    PROC_MASK_SET(procMask, getFirstEnabledCPU());
}

MemcpyOperation::MemcpyOperation(MemcpySMFunc memcpyFunc, size_t copySize, unsigned long long loopCount): smFunc(memcpyFunc), copySize(copySize), loopCount(loopCount) {
    procMask = (size_t *)calloc(1, PROC_MASK_SIZE);
    PROC_MASK_SET(procMask, getFirstEnabledCPU());
}

MemcpyOperation::~MemcpyOperation() {
    delete bandwidthValues;
    PROC_MASK_CLEAR(procMask, 0);
}

void MemcpyOperation::doMemcpy(MemcpyNode* srcNode, MemcpyNode* dstNode) {
    std::vector<MemcpyNode*> srcNodes = {srcNode};
    std::vector<MemcpyNode*> dstNodes = {dstNode};
    doMemcpy(srcNodes, dstNodes);
}

void MemcpyOperation::doMemcpy(std::vector<MemcpyNode*> srcNodes, std::vector<MemcpyNode*> dstNodes) {
    bool isAnyHost = false;
    unsigned long long avgBandwidth = 0;
    PerformanceStatistic bandwidthStat;
    volatile int* blockingVar;
    std::vector<CUcontext> contexts(srcNodes.size());
    std::vector<CUstream> streams(srcNodes.size());
    std::vector<CUevent> startEvents(srcNodes.size());
    std::vector<CUevent> endEvents(srcNodes.size());

    // TODO this could be better
    for (int i = 0; i < srcNodes.size(); i++) {
        if (srcNodes[i]->getPrimaryCtx() != nullptr || dstNodes[i]->getPrimaryCtx() != nullptr) {
            isAnyHost = true;
            break;
        }
    }

    // allocate resources
    allocateBandwidthMatrix(isAnyHost);

    CU_ASSERT(cuMemHostAlloc((void **)&blockingVar, sizeof(*blockingVar), CU_MEMHOSTALLOC_PORTABLE));

    for (int i = 0; i < srcNodes.size(); i++) {
        // prefer source context
        if (srcNodes[i]->getPrimaryCtx() != nullptr) {
            CU_ASSERT(cuCtxSetCurrent(srcNodes[i]->getPrimaryCtx()));
            contexts[i] = srcNodes[i]->getPrimaryCtx();
        } else if (dstNodes[i]->getPrimaryCtx() != nullptr) {
            CU_ASSERT(cuCtxSetCurrent(dstNodes[i]->getPrimaryCtx()));
            contexts[i] = dstNodes[i]->getPrimaryCtx();
        }

        // allocate the per simulaneous copy resources
        CU_ASSERT(cuStreamCreate(&streams[i], CU_STREAM_NON_BLOCKING));
        CU_ASSERT(cuEventCreate(&startEvents[i], CU_EVENT_DEFAULT));
        CU_ASSERT(cuEventCreate(&endEvents[i], CU_EVENT_DEFAULT));
    }

    // This loop is for sampling the benchmark (which itself has a loop count)
    for (unsigned int n = 0; n < averageLoopCount; n++) {
        *blockingVar = 0;
        // block stream, and enqueue copy
        for (int i = 0; i < srcNodes.size(); i++) {
            CU_ASSERT(cuCtxSetCurrent(contexts[i]));

            // start the spin kernel on the stream
            CU_ASSERT(spinKernel(blockingVar, streams[i]));

            CU_ASSERT(cuEventRecord(startEvents[i], streams[i]));
        
            if (smFunc != nullptr) {
                CU_ASSERT(smFunc(dstNodes[i]->getBuffer(), srcNodes[i]->getBuffer(), smCopySize(), streams[i], loopCount));
            } else {
                for (unsigned int l = 0; l < loopCount; l++) CU_ASSERT(ceFunc(dstNodes[i]->getBuffer(), srcNodes[i]->getBuffer(), copySize, streams[i]));
            }

            CU_ASSERT(cuEventRecord(endEvents[i], streams[i]));
        }

        // unblock the streams
        *blockingVar = 1;

        for (CUstream stream : streams) {
            CU_ASSERT(cuStreamSynchronize(stream));
        }

        // TODO - do we only want the first copy, or the sum of all?
        float timeWithEvents = 0.0f;
        CU_ASSERT(cuEventElapsedTime(&timeWithEvents, startEvents[0], endEvents[0]));
        double elapsedWithEventsInUs = ((double) timeWithEvents * 1000.0);
        unsigned long long bandwidth = (copySize * loopCount * 1000ull * 1000ull) / (unsigned long long) elapsedWithEventsInUs;
        bandwidthStat((double) bandwidth);
        VERBOSE << "\tSample " << n << ' ' << std::fixed << std::setprecision(2) << (double) bandwidth * 1e-9 << " GB/s\n";
    }

    // cleanup
    CU_ASSERT(cuMemFreeHost((void*)blockingVar));

    for (int i = 0; i < srcNodes.size(); i++) {
        CU_ASSERT(cuStreamDestroy(streams[i]));
        CU_ASSERT(cuEventDestroy(startEvents[i]));
        CU_ASSERT(cuEventDestroy(endEvents[i]));
    }

    avgBandwidth = (unsigned long long)(STAT_MEAN(bandwidthStat));

    if (isAnyHost) {
        bandwidthValues->value(0, std::max(srcNodes[0]->getNodeIdx(), dstNodes[0]->getNodeIdx())) = (double)avgBandwidth * 1e-9;
    } else {
        bandwidthValues->value(srcNodes[0]->getNodeIdx(), dstNodes[0]->getNodeIdx()) = (double)avgBandwidth * 1e-9;
    }
}

void MemcpyOperation::printBenchmarkMatrix(bool reverse) {
    // TODO this is wrong
    std::cout << "memcpy CE GPU(row) " << (reverse ? "->" : "<-") << " GPU(column) bandwidth (GB/s):" << std::endl;
    std::cout << std::fixed << std::setprecision(2) << *bandwidthValues << std::endl;
}

size_t MemcpyOperation::smCopySize() const {
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

void MemcpyOperation::allocateBandwidthMatrix(bool hostVector) {
    #pragma omp critical
    {
        if (bandwidthValues == nullptr) {
            int rows = hostVector ? 1 : deviceCount;
            bandwidthValues = new PeerValueMatrix<double>(rows, deviceCount);
        }
    }
}
