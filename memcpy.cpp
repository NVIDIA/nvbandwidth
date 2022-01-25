/*
 * Copyright 1993-2021 by NVIDIA Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#include "memcpy.h"
#include "kernels.cuh"

#define WARMUP_COUNT 4

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

bool DeviceNode::enablePeerAcess(const DeviceNode *peerNode) {
    int canAccessPeer = 0;
    CU_ASSERT(cuDeviceCanAccessPeer(&canAccessPeer, getNodeIdx(), peerNode->getNodeIdx()));
    if (canAccessPeer) {
        CUresult res;
        CU_ASSERT(cuCtxSetCurrent(peerNode->getPrimaryCtx()));
        res = cuCtxEnablePeerAccess(getPrimaryCtx(), 0);
        if (res != CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED)
            CU_ASSERT(res);

        CU_ASSERT(cuCtxSetCurrent(getPrimaryCtx()));
        res = cuCtxEnablePeerAccess(peerNode->getPrimaryCtx(), 0);
        if (res != CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED)
            CU_ASSERT(res);

        return true;
    }
    return false;
}

MemcpyOperation::MemcpyOperation(size_t copySize, unsigned long long loopCount, bool preferSrcCtx, bool sumResults) : 
        copySize(copySize), loopCount(loopCount), preferSrcCtx(preferSrcCtx), sumResults(sumResults)
{
    procMask = (size_t *)calloc(1, PROC_MASK_SIZE);
    PROC_MASK_SET(procMask, getFirstEnabledCPU());
}

MemcpyOperation::~MemcpyOperation() {
    PROC_MASK_CLEAR(procMask, 0);
}

double MemcpyOperation::doMemcpy(MemcpyNode* srcNode, MemcpyNode* dstNode) {
    std::vector<MemcpyNode*> srcNodes = {srcNode};
    std::vector<MemcpyNode*> dstNodes = {dstNode};
    return doMemcpy(srcNodes, dstNodes);
}

double MemcpyOperation::doMemcpy(std::vector<MemcpyNode*> srcNodes, std::vector<MemcpyNode*> dstNodes) {
    volatile int* blockingVar;
    std::vector<CUcontext> contexts(srcNodes.size());
    std::vector<CUstream> streams(srcNodes.size());
    std::vector<CUevent> startEvents(srcNodes.size());
    std::vector<CUevent> endEvents(srcNodes.size());
    std::vector<PerformanceStatistic> bandwidthStats(srcNodes.size());

    CU_ASSERT(cuMemHostAlloc((void **)&blockingVar, sizeof(*blockingVar), CU_MEMHOSTALLOC_PORTABLE));

    for (int i = 0; i < srcNodes.size(); i++) {
        // prefer source context
        if (preferSrcCtx && srcNodes[i]->getPrimaryCtx() != nullptr) {
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
            
            // warmup
            CU_ASSERT(memcpyFunc(dstNodes[i]->getBuffer(), srcNodes[i]->getBuffer(), streams[i], WARMUP_COUNT));
        }

        CU_ASSERT(cuCtxSetCurrent(contexts[0]));
        CU_ASSERT(cuEventRecord(startEvents[0], streams[0]));
        for (int i = 1; i < srcNodes.size(); i++) {
            // ensure that all copies are launched at the same time
            CU_ASSERT(cuCtxSetCurrent(contexts[i]));
            CU_ASSERT(cuStreamWaitEvent(streams[i], startEvents[0], 0));
            CU_ASSERT(cuEventRecord(startEvents[i], streams[i]));
        }

        for (int i = 0; i < srcNodes.size(); i++) {
            CU_ASSERT(cuCtxSetCurrent(contexts[i]));
            CU_ASSERT(memcpyFunc(dstNodes[i]->getBuffer(), srcNodes[i]->getBuffer(), streams[i], loopCount));
            CU_ASSERT(cuEventRecord(endEvents[i], streams[i]));
        }

        // unblock the streams
        *blockingVar = 1;

        for (CUstream stream : streams) {
            CU_ASSERT(cuStreamSynchronize(stream));
        }

        for (int i = 0; i < bandwidthStats.size(); i++) {
        float timeWithEvents = 0.0f;
        CU_ASSERT(cuEventElapsedTime(&timeWithEvents, startEvents[0], endEvents[0]));
            double elapsedWithEventsInUs = ((double) timeWithEvents * 1000.0);
            unsigned long long bandwidth = (copySize * loopCount * 1000ull * 1000ull) / (unsigned long long) elapsedWithEventsInUs;
            bandwidthStats[i]((double) bandwidth);
            VERBOSE << "\tSample " << n << ' ' << std::fixed << std::setprecision(2) << (double) bandwidth * 1e-9 << " GB/s\n";
        }
    }

    // cleanup
    CU_ASSERT(cuMemFreeHost((void*)blockingVar));

    for (int i = 0; i < srcNodes.size(); i++) {
        CU_ASSERT(cuStreamDestroy(streams[i]));
        CU_ASSERT(cuEventDestroy(startEvents[i]));
        CU_ASSERT(cuEventDestroy(endEvents[i]));
    }

    if (sumResults) {
        double sum = 0.0;
        for (auto stat : bandwidthStats) {
            sum += (double)(STAT_MEAN(bandwidthStats[0]) * 1e-9);
        }
        return sum;
    } else {
        return (double)(STAT_MEAN(bandwidthStats[0]) * 1e-9);
    }
}

MemcpyOperationSM::MemcpyOperationSM(size_t copySize, unsigned long long loopCount, bool preferSrcCtx, bool sumResults) : 
        MemcpyOperation(copySize, loopCount, preferSrcCtx, sumResults) {}

CUresult MemcpyOperationSM::memcpyFunc(CUdeviceptr dst, CUdeviceptr src, CUstream stream, unsigned long long loopCount) {
    CUdevice cudaDevice;
    int multiProcessorCount;
    size_t size = copySize;

    size /= sizeof(int4);
    CU_ASSERT(cuCtxGetDevice(&cudaDevice));
    CU_ASSERT(cuDeviceGetAttribute(&multiProcessorCount, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, cudaDevice));
    unsigned long long totalThreadCount = (unsigned long long)(multiProcessorCount * numThreadPerBlock);
    size = totalThreadCount * (size / totalThreadCount);

    CU_ASSERT(copyKernel(dst, src, size, stream, loopCount));

    return CUDA_SUCCESS;
}

MemcpyOperationCE::MemcpyOperationCE(size_t copySize, unsigned long long loopCount, bool preferSrcCtx, bool sumResults) : 
        MemcpyOperation(copySize, loopCount, preferSrcCtx, sumResults) {}

CUresult MemcpyOperationCE::memcpyFunc(CUdeviceptr dst, CUdeviceptr src, CUstream stream, unsigned long long loopCount) {
    for (unsigned int l = 0; l < loopCount; l++) {
        CU_ASSERT(cuMemcpyAsync(dst, src, copySize, stream));
    }

    return CUDA_SUCCESS;
}
