/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "memcpy.h"
#include "kernels.cuh"

#define WARMUP_COUNT 4

MemcpyNode::MemcpyNode(size_t bufferSize): bufferSize(bufferSize), buffer(nullptr) {}

CUdeviceptr MemcpyNode::getBuffer() const {
    return (CUdeviceptr)buffer;
}

size_t MemcpyNode::getBufferSize() const {
    return bufferSize;
}

HostNode::HostNode(size_t bufferSize, int targetDeviceId): MemcpyNode(bufferSize) {
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

std::string HostNode::getNodeString() const {
    return "Host";
}

DeviceNode::DeviceNode(size_t bufferSize, int deviceIdx): deviceIdx(deviceIdx), MemcpyNode(bufferSize) {
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

std::string DeviceNode::getNodeString() const {
    return "Device " + std::to_string(deviceIdx);
}

bool DeviceNode::enablePeerAcess(const DeviceNode &peerNode) {
    int canAccessPeer = 0;
    CU_ASSERT(cuDeviceCanAccessPeer(&canAccessPeer, getNodeIdx(), peerNode.getNodeIdx()));
    if (canAccessPeer) {
        CUresult res;
        CU_ASSERT(cuCtxSetCurrent(peerNode.getPrimaryCtx()));
        res = cuCtxEnablePeerAccess(getPrimaryCtx(), 0);
        if (res != CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED)
            CU_ASSERT(res);

        CU_ASSERT(cuCtxSetCurrent(getPrimaryCtx()));
        res = cuCtxEnablePeerAccess(peerNode.getPrimaryCtx(), 0);
        if (res != CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED)
            CU_ASSERT(res);

        return true;
    }
    return false;
}

MemcpyOperation::MemcpyOperation(unsigned long long loopCount, ContextPreference ctxPreference, BandwidthValue bandwidthValue) : 
        loopCount(loopCount), ctxPreference(ctxPreference), bandwidthValue(bandwidthValue)
{
    procMask = (size_t *)calloc(1, PROC_MASK_SIZE);
    PROC_MASK_SET(procMask, getFirstEnabledCPU());
}

MemcpyOperation::~MemcpyOperation() {
    PROC_MASK_CLEAR(procMask, 0);
}

double MemcpyOperation::doMemcpy(const MemcpyNode &srcNode, const MemcpyNode &dstNode) {
    std::vector<const MemcpyNode*> srcNodes = {&srcNode};
    std::vector<const MemcpyNode*> dstNodes = {&dstNode};
    return doMemcpy(srcNodes, dstNodes);
}

double MemcpyOperation::doMemcpy(const std::vector<const MemcpyNode*> &srcNodes, const std::vector<const MemcpyNode*> &dstNodes) {
    volatile int* blockingVar;
    std::vector<CUcontext> contexts(srcNodes.size());
    std::vector<CUstream> streams(srcNodes.size());
    std::vector<CUevent> startEvents(srcNodes.size());
    std::vector<CUevent> endEvents(srcNodes.size());
    std::vector<PerformanceStatistic> bandwidthStats(srcNodes.size());
    std::vector<size_t> adjustedCopySizes(srcNodes.size());

    CU_ASSERT(cuMemHostAlloc((void **)&blockingVar, sizeof(*blockingVar), CU_MEMHOSTALLOC_PORTABLE));

    for (int i = 0; i < srcNodes.size(); i++) {
        // prefer source context
        if (ctxPreference == MemcpyOperation::PREFER_SRC_CONTEXT && srcNodes[i]->getPrimaryCtx() != nullptr) {
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
            memcpyFunc(dstNodes[i]->getBuffer(), srcNodes[i]->getBuffer(), streams[i], srcNodes[i]->getBufferSize(), WARMUP_COUNT);
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
            assert(srcNodes[i]->getBufferSize() == dstNodes[i]->getBufferSize());
            adjustedCopySizes[i] = memcpyFunc(dstNodes[i]->getBuffer(), srcNodes[i]->getBuffer(), streams[i], srcNodes[i]->getBufferSize(), loopCount);
            CU_ASSERT(cuEventRecord(endEvents[i], streams[i]));
        }

        // unblock the streams
        *blockingVar = 1;

        for (CUstream stream : streams) {
            CU_ASSERT(cuStreamSynchronize(stream));
        }

        for (int i = 0; i < bandwidthStats.size(); i++) {
            float timeWithEvents = 0.0f;
            CU_ASSERT(cuEventElapsedTime(&timeWithEvents, startEvents[i], endEvents[i]));
            double elapsedWithEventsInUs = ((double) timeWithEvents * 1000.0);
            unsigned long long bandwidth = (adjustedCopySizes[i] * loopCount * 1000ull * 1000ull) / (unsigned long long) elapsedWithEventsInUs;
            bandwidthStats[i]((double) bandwidth);

            if (bandwidthValue == BandwidthValue::SUM_BW || i == 0) {
                // Verbose print only the values that are used for the final output
                VERBOSE << "\tSample " << n << ": " << srcNodes[i]->getNodeString() << " -> " << dstNodes[i]->getNodeString() << ": " <<
                    std::fixed << std::setprecision(2) << (double)bandwidth * 1e-9 << " GB/s\n";
            }
        }
    }

    // cleanup
    CU_ASSERT(cuMemFreeHost((void*)blockingVar));

    for (int i = 0; i < srcNodes.size(); i++) {
        CU_ASSERT(cuStreamDestroy(streams[i]));
        CU_ASSERT(cuEventDestroy(startEvents[i]));
        CU_ASSERT(cuEventDestroy(endEvents[i]));
    }

    if (bandwidthValue == BandwidthValue::SUM_BW) {
        double sum = 0.0;
        for (auto stat : bandwidthStats) {
            sum += stat.median() * 1e-9;;
        }
        return sum;
    } else {
        return bandwidthStats[0].median() * 1e-9;
    }
}

MemcpyOperationSM::MemcpyOperationSM(unsigned long long loopCount, ContextPreference ctxPreference, BandwidthValue bandwidthValue) : 
        MemcpyOperation(loopCount, ctxPreference, bandwidthValue) {}

size_t MemcpyOperationSM::memcpyFunc(CUdeviceptr dst, CUdeviceptr src, CUstream stream, size_t copySize, unsigned long long loopCount) {
    return copyKernel(dst, src, copySize, stream, loopCount);
}

MemcpyOperationCE::MemcpyOperationCE(unsigned long long loopCount, ContextPreference ctxPreference, BandwidthValue bandwidthValue) : 
        MemcpyOperation(loopCount, ctxPreference, bandwidthValue) {}

size_t MemcpyOperationCE::memcpyFunc(CUdeviceptr dst, CUdeviceptr src, CUstream stream, size_t copySize, unsigned long long loopCount) {
    for (unsigned int l = 0; l < loopCount; l++) {
        CU_ASSERT(cuMemcpyAsync(dst, src, copySize, stream));
    }

    return copySize;
}
