/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "common.h"
#include <cuda_runtime.h>
#include <cuda.h>
#include "inline_common.h"
#include "memcpy.h"
#include "output.h"
#include "kernels.cuh"
#include "vector_types.h"
#define WARMUP_COUNT 4
#include <cassert>

#ifndef _WIN32
#include <sys/mman.h>
#endif

MemcpyBuffer::MemcpyBuffer(size_t bufferSize): bufferSize(bufferSize), buffer(nullptr) {}

CUdeviceptr MemcpyBuffer::getBuffer() const {
    return (CUdeviceptr)buffer;
}

size_t MemcpyBuffer::getBufferSize() const {
    return bufferSize;
}

void xorshift2MBPattern(unsigned int* buffer, unsigned int seed) {
    unsigned int oldValue = seed;
    unsigned int n = 0;
    for (n = 0; n < (1024 * 1024 * 2) / sizeof(unsigned int); n++) {
        unsigned int value = oldValue;
        value = value ^ (value << 13);
        value = value ^ (value >> 17);
        value = value ^ (value << 5);
        oldValue = value;
        buffer[n] = oldValue;
    }
}

void memsetPatternHelper(CUstream stream, CUdeviceptr buffer, unsigned long long size, unsigned int seed, std::shared_ptr<NodeHelper> nodeHelper) {
    unsigned int* h_pattern;
    CUdeviceptr d_pattern;

    unsigned long long num_elements = size / sizeof(unsigned int);
    unsigned long long num_pattern_elements = _2MiB / sizeof(unsigned int);

    // Allocate 2MB of pattern
    CU_ASSERT(cuMemHostAlloc((void**)&h_pattern, sizeof(char) * _2MiB, CU_MEMHOSTALLOC_PORTABLE));
    xorshift2MBPattern(h_pattern, seed);

    // Copy the pattern to a device buffer
    CU_ASSERT(cuMemAlloc(&d_pattern, sizeof(char) * _2MiB));
    CU_ASSERT(cuMemcpyAsync(d_pattern, (CUdeviceptr)h_pattern, sizeof(char) * _2MiB, CU_STREAM_PER_THREAD));

    // Launch the memset kernel
    CU_ASSERT(memsetKernel(CU_STREAM_PER_THREAD, buffer, d_pattern, num_elements, num_pattern_elements));
    CU_ASSERT(nodeHelper->streamSynchronizeWrapper(CU_STREAM_PER_THREAD));

    CU_ASSERT(cuMemFreeHost((void*)h_pattern));
    CU_ASSERT(cuMemFree(d_pattern));
}

void memclearByWarpParity(CUstream stream, CUdeviceptr buffer, unsigned long long size, bool clearOddWarpIndexed, std::shared_ptr<NodeHelper> nodeHelper) {
    CU_ASSERT(memclearKernelByWarpParity(CU_STREAM_PER_THREAD, buffer, size, clearOddWarpIndexed));
    CU_ASSERT(nodeHelper->streamSynchronizeWrapper(CU_STREAM_PER_THREAD));
}

void MemcpyInitiatorCE::memsetPattern(MemcpyDispatchInfo &info) const {
    for (int i = 0; i < info.srcBuffers.size(); i++) {
        CU_ASSERT(cuCtxSetCurrent(info.contexts[i]));
        memsetPatternHelper(info.streams[i], info.dstBuffers[i]->getBuffer(), info.adjustedCopySizes[i], 0xCAFEBABE, info.nodeHelper);
        memsetPatternHelper(info.streams[i], info.srcBuffers[i]->getBuffer(), info.adjustedCopySizes[i], 0xBAADF00D, info.nodeHelper);
    }
}

void MemcpyInitiatorSM::memsetPattern(MemcpyDispatchInfo &info) const {
    for (int i = 0; i < info.srcBuffers.size(); i++) {
        CU_ASSERT(cuCtxSetCurrent(info.contexts[i]));
        memsetPatternHelper(info.streams[i], info.dstBuffers[i]->getBuffer(), info.adjustedCopySizes[i], 0xCAFEBABE, info.nodeHelper);
        memsetPatternHelper(info.streams[i], info.srcBuffers[i]->getBuffer(), info.adjustedCopySizes[i], 0xBAADF00D, info.nodeHelper);
    }
}

void MemcpyInitiatorMulticastWrite::memsetPattern(MemcpyDispatchInfo &info) const {
    for (int i = 0; i < info.srcBuffers.size(); i++) {
        CU_ASSERT(cuCtxSetCurrent(info.contexts[i]));
        memsetPatternHelper(info.streams[i], info.dstBuffers[i]->getBuffer(), info.adjustedCopySizes[i], 0xCAFEBABE, info.nodeHelper);
        memsetPatternHelper(info.streams[i], info.srcBuffers[i]->getBuffer(), info.adjustedCopySizes[i], 0xBAADF00D, info.nodeHelper);
    }
}

void MemcpyInitiatorSMSplitWarp::memsetPattern(MemcpyDispatchInfo &info) const {
    for (int i = 0; i < info.srcBuffers.size(); i++) {
        CU_ASSERT(cuCtxSetCurrent(info.contexts[i]));
        memsetPatternHelper(info.streams[i], info.dstBuffers[i]->getBuffer(), info.adjustedCopySizes[i], 0xBAADF00D, info.nodeHelper);
        memsetPatternHelper(info.streams[i], info.srcBuffers[i]->getBuffer(), info.adjustedCopySizes[i], 0xBAADF00D, info.nodeHelper);
        memclearByWarpParity(info.streams[i], info.dstBuffers[i]->getBuffer(), info.adjustedCopySizes[i], true /* clearOddWarpIndexed */, info.nodeHelper);
        memclearByWarpParity(info.streams[i], info.srcBuffers[i]->getBuffer(), info.adjustedCopySizes[i], false /* clearOddWarpIndexed */, info.nodeHelper);
    }
}

unsigned long long MemcpyInitiatorCE::getAdjustedBandwidth(unsigned long long bandwidth) {
    return bandwidth;
}

unsigned long long MemcpyInitiatorSM::getAdjustedBandwidth(unsigned long long bandwidth) {
    return bandwidth;
}

unsigned long long MemcpyInitiatorMulticastWrite::getAdjustedBandwidth(unsigned long long bandwidth) {
    return bandwidth;
}

unsigned long long MemcpyInitiatorSMSplitWarp::getAdjustedBandwidth(unsigned long long bandwidth) {
    // For split warp copies, we estimate bandwidth in each direction as 1/2 of measured bandwidth
    return bandwidth / 2;
}

// Add this new typedef for the comparison function pointer
typedef CUresult (*CompareKernelFunc)(CUstream, CUdeviceptr, CUdeviceptr, unsigned long long, unsigned int, CUdeviceptr);

void memcmpPatternHelper(CUstream stream, CUdeviceptr buffer, unsigned long long size, unsigned int seed, CompareKernelFunc compareKernel, std::shared_ptr<NodeHelper> nodeHelper, uint32_t patternScale = 1) {
    unsigned int* h_pattern;
    CUdeviceptr d_pattern;
    int h_errorFlag = 0;
    CUdeviceptr d_errorFlag;
    unsigned long long num_elements = size / sizeof(unsigned int);
    unsigned long long num_pattern_elements = _2MiB / sizeof(unsigned int);

    // Allocate 2MB of pattern
    CU_ASSERT(cuMemHostAlloc((void**)&h_pattern, sizeof(char) * _2MiB, CU_MEMHOSTALLOC_PORTABLE));
    xorshift2MBPattern(h_pattern, seed);
    if (patternScale != 1) {
        for (unsigned long long k = 0; k < num_pattern_elements; k++) {
            h_pattern[k] *= patternScale;
        }
    }
    CU_ASSERT(cuMemAlloc(&d_pattern, sizeof(char) * _2MiB));
    CU_ASSERT(cuMemcpyAsync(d_pattern, (CUdeviceptr)h_pattern, sizeof(char) * _2MiB, CU_STREAM_PER_THREAD));

    // setup error flags
    CU_ASSERT(cuMemAlloc(&d_errorFlag, sizeof(int)));
    CU_ASSERT(cuMemcpyAsync(d_errorFlag, (CUdeviceptr)&h_errorFlag, sizeof(int), CU_STREAM_PER_THREAD));

    // launch kernel to compare
    CU_ASSERT(compareKernel(CU_STREAM_PER_THREAD, buffer, d_pattern, num_elements, num_pattern_elements, d_errorFlag));
    CU_ASSERT(nodeHelper->streamSynchronizeWrapper(CU_STREAM_PER_THREAD));
    CU_ASSERT(cuMemcpyAsync((CUdeviceptr)&h_errorFlag, d_errorFlag, sizeof(int), CU_STREAM_PER_THREAD));

    CU_ASSERT(cuMemFreeHost((void*)h_pattern));
    CU_ASSERT(cuMemFree(d_errorFlag));
    CU_ASSERT(cuMemFree(d_pattern));

    ASSERT(h_errorFlag == 0);
}

void MemcpyInitiatorCE::memcmpPattern(MemcpyDispatchInfo &info) const {
    for (int i = 0; i < info.srcBuffers.size(); i++) {
        CU_ASSERT(cuCtxSetCurrent(info.contexts[i]));
        memcmpPatternHelper(info.streams[i], info.dstBuffers[i]->getBuffer(), info.adjustedCopySizes[i], 0xBAADF00D, memcmpKernel, info.nodeHelper);
    }
}

void MemcpyInitiatorSM::memcmpPattern(MemcpyDispatchInfo &info) const {
    for (int i = 0; i < info.srcBuffers.size(); i++) {
        CU_ASSERT(cuCtxSetCurrent(info.contexts[i]));
        memcmpPatternHelper(info.streams[i], info.dstBuffers[i]->getBuffer(), info.adjustedCopySizes[i], 0xBAADF00D, memcmpKernel, info.nodeHelper);
    }
}

void MemcpyInitiatorMulticastWrite::memcmpPattern(MemcpyDispatchInfo &info) const {
    for (int i = 0; i < info.srcBuffers.size(); i++) {
        CU_ASSERT(cuCtxSetCurrent(info.contexts[i]));
        memcmpPatternHelper(info.streams[i], info.dstBuffers[i]->getBuffer(), info.adjustedCopySizes[i], 0xBAADF00D, multicastMemcmpKernel, info.nodeHelper);
    }
}


void MemcpyInitiatorSMSplitWarp::memcmpPattern(MemcpyDispatchInfo &info) const {
    for (int i = 0; i < info.srcBuffers.size(); i++) {
        CU_ASSERT(cuCtxSetCurrent(info.contexts[i]));
        // src and dst buffer contents must match after the bidirectional split warp copy
        memcmpPatternHelper(info.streams[i], info.dstBuffers[i]->getBuffer(), info.adjustedCopySizes[i], 0xBAADF00D, memcmpKernel, info.nodeHelper);
        memcmpPatternHelper(info.streams[i], info.srcBuffers[i]->getBuffer(), info.adjustedCopySizes[i], 0xBAADF00D, memcmpKernel, info.nodeHelper);
    }
}

// Non-multinode MemcpyBuffers will always return Rank 0
// Semantically, we can assume that non-MPI runs will have world rank = 0 and world size = 1
int MemcpyBuffer::getMPIRank() const {
    return 0;
}

CUresult MemcpyBuffer::streamSynchronizeWrapper(CUstream stream) const {
    return cuStreamSynchronize(stream);
}

HostBuffer::HostBuffer(size_t bufferSize, int targetDeviceId): MemcpyBuffer(bufferSize) {
    CUcontext targetCtx;

    // Before allocating host memory, set correct NUMA affinity
    setOptimalCpuAffinity(targetDeviceId);
    CU_ASSERT(cuDevicePrimaryCtxRetain(&targetCtx, targetDeviceId));
    CU_ASSERT(cuCtxSetCurrent(targetCtx));

#if !defined(_WIN32)
    if (gSettings.useHugePages) {
        buffer = aligned_alloc(2 * 1024 * 1024, bufferSize);
        int err = madvise(buffer, bufferSize, MADV_HUGEPAGE);
        ASSERT(err == 0);
        CU_ASSERT(cuMemHostRegister(buffer, bufferSize, CU_MEMHOSTREGISTER_PORTABLE));
        VERBOSE << "Allocated THP memory [" << buffer << ", " << (void*)((char*)buffer + bufferSize)
                << ") size " << bufferSize << "\n";
    } else {
        CU_ASSERT(cuMemHostAlloc(&buffer, bufferSize, CU_MEMHOSTALLOC_PORTABLE));
    }
#else
    CU_ASSERT(cuMemHostAlloc(&buffer, bufferSize, CU_MEMHOSTALLOC_PORTABLE));
#endif
}

HostBuffer::~HostBuffer() {
    if (isMemoryOwnedByCUDA(buffer)) {
        if (gSettings.useHugePages) {
            CU_ASSERT(cuMemHostUnregister(buffer));
            free(buffer);
        } else {
            CU_ASSERT(cuMemFreeHost(buffer));
        }
    } else {
        free(buffer);
    }
}

// Host nodes don't have a context, return null
CUcontext HostBuffer::getPrimaryCtx() const {
    return nullptr;
}

// Host buffers always return zero as they always represent one row in the bandwidth matrix
int HostBuffer::getBufferIdx() const {
    return 0;
}

std::string HostBuffer::getBufferString() const {
    return "Host";
}

DeviceBuffer::DeviceBuffer(size_t bufferSize, int deviceIdx): deviceIdx(deviceIdx), MemcpyBuffer(bufferSize) {
    CU_ASSERT(cuDevicePrimaryCtxRetain(&primaryCtx, deviceIdx));
    CU_ASSERT(cuCtxSetCurrent(primaryCtx));
    CU_ASSERT(cuMemAlloc((CUdeviceptr*)&buffer, bufferSize));
}

DeviceBuffer::~DeviceBuffer() {
    CU_ASSERT(cuCtxSetCurrent(primaryCtx));
    CU_ASSERT(cuMemFree((CUdeviceptr)buffer));
    CU_ASSERT(cuDevicePrimaryCtxRelease(deviceIdx));
}

CUcontext DeviceBuffer::getPrimaryCtx() const {
    return primaryCtx;
}

int DeviceBuffer::getBufferIdx() const {
    return deviceIdx;
}

std::string DeviceBuffer::getBufferString() const {
    return "Device " + std::to_string(deviceIdx);
}

// Enable peer access by device ID. Used when buffers don't exist yet (e.g. E2E tests
// before buffer allocation, or in filter()). DeviceBuffer::enablePeerAcess uses this internally.
bool enablePeerAccessBetweenDevices(int srcDeviceId, int dstDeviceId) {
    int canAccessPeer = 0;
    CU_ASSERT(cuDeviceCanAccessPeer(&canAccessPeer, srcDeviceId, dstDeviceId));
    if (!canAccessPeer) {
        return false;
    }

    CUcontext srcCtx, dstCtx;
    CU_ASSERT(cuDevicePrimaryCtxRetain(&srcCtx, srcDeviceId));
    CU_ASSERT(cuDevicePrimaryCtxRetain(&dstCtx, dstDeviceId));

    // Enable peer access from src to dst
    CU_ASSERT(cuCtxSetCurrent(srcCtx));
    CUresult res = cuCtxEnablePeerAccess(dstCtx, 0);
    if (res != CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED) {
        CU_ASSERT(res);
    }

    // Enable peer access from dst to src
    CU_ASSERT(cuCtxSetCurrent(dstCtx));
    res = cuCtxEnablePeerAccess(srcCtx, 0);
    if (res != CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED) {
        CU_ASSERT(res);
    }

    CU_ASSERT(cuDevicePrimaryCtxRelease(srcDeviceId));
    CU_ASSERT(cuDevicePrimaryCtxRelease(dstDeviceId));

    return true;
}

bool MemcpyBuffer::enablePeerAcess(const MemcpyBuffer &peerBuffer) const {
    if (!supportsPeerAccess() || !peerBuffer.supportsPeerAccess()) {
        return false;
    }
    return enablePeerAccessBetweenDevices(getBufferIdx(), peerBuffer.getBufferIdx());
}

bool DeviceBuffer::enablePeerAcess(const MemcpyBuffer &peerBuffer) const {
    if (!peerBuffer.supportsPeerAccess()) return false;
    return enablePeerAccessBetweenDevices(getBufferIdx(), peerBuffer.getBufferIdx());
}


MemcpyDescriptor::MemcpyDescriptor(CUdeviceptr dst, CUdeviceptr src, CUstream stream, size_t copySize,
                                   unsigned long long loopCount) :
    dst(dst), src(src), stream(stream), copySize(copySize), loopCount(loopCount) {}

MemcpyOperation::MemcpyOperation(unsigned long long loopCount, MemcpyInitiator* memcpyInitiator, ContextPreference ctxPreference, BandwidthValue bandwidthValue) :
    MemcpyOperation(loopCount, memcpyInitiator, new NodeHelperSingle(), ctxPreference, bandwidthValue) {}

MemcpyOperation::MemcpyOperation(unsigned long long loopCount, MemcpyInitiator* memcpyInitiator, NodeHelper* nodeHelper, ContextPreference ctxPreference, BandwidthValue bandwidthValue) :
        loopCount(loopCount), memcpyInitiator(memcpyInitiator), nodeHelper(nodeHelper), ctxPreference(ctxPreference), bandwidthValue(bandwidthValue) {
    procMask = (size_t *)calloc(1, PROC_MASK_SIZE);
    PROC_MASK_SET(procMask, getFirstEnabledCPU());
}

MemcpyOperation::~MemcpyOperation() {
    PROC_MASK_CLEAR(procMask, 0);
    free(procMask);
}

double MemcpyOperation::doMemcpy(const MemcpyBuffer &srcBuffer, const MemcpyBuffer &dstBuffer) {
    std::vector<const MemcpyBuffer*> srcBuffers = {&srcBuffer};
    std::vector<const MemcpyBuffer*> dstBuffers = {&dstBuffer};
    return doMemcpy(srcBuffers, dstBuffers);
}

MemcpyDispatchInfo::MemcpyDispatchInfo(std::vector<const MemcpyBuffer*> srcBuffers, std::vector<const MemcpyBuffer*> dstBuffers, std::vector<CUcontext> contexts, std::vector<int> originalRanks) :
    srcBuffers(srcBuffers), dstBuffers(dstBuffers), contexts(contexts), originalRanks(originalRanks),
    allSrcBuffers(srcBuffers), allDstBuffers(dstBuffers) {
}

NodeHelperSingle::NodeHelperSingle() {
    CU_ASSERT(cuMemHostAlloc((void **)&blockingVarHost, sizeof(*blockingVarHost), CU_MEMHOSTALLOC_PORTABLE));
}

NodeHelperSingle::~NodeHelperSingle() {
    CU_ASSERT(cuMemFreeHost((void*)blockingVarHost));
}

const MemcpyBuffer* NodeHelper::selectExecutingBuffer(
        const MemcpyBuffer* src, const MemcpyBuffer* dst, ContextPreference ctxPreference) {
    if (ctxPreference == PREFER_SRC_CONTEXT && src->getPrimaryCtx() != nullptr) return src;
    if (dst->getPrimaryCtx() != nullptr) return dst;
    return nullptr;
}

MemcpyDispatchInfo NodeHelperSingle::dispatchMemcpy(const std::vector<const MemcpyBuffer*> &srcBuffers, const std::vector<const MemcpyBuffer*> &dstBuffers, ContextPreference ctxPreference) {
    std::vector<CUcontext> contexts(srcBuffers.size());

    for (int i = 0; i < srcBuffers.size(); i++) {
        const MemcpyBuffer* exec = NodeHelper::selectExecutingBuffer(srcBuffers[i], dstBuffers[i], ctxPreference);
        if (exec) contexts[i] = exec->getPrimaryCtx();
    }

    return MemcpyDispatchInfo(srcBuffers, dstBuffers, contexts);
}

double NodeHelperSingle::calculateTotalBandwidth(double totalTime, double totalSize, size_t loopCount) {
    return (totalSize * loopCount * 1000ull * 1000ull) / totalTime;
}

double NodeHelperSingle::calculateSumBandwidth(std::vector<PerformanceStatistic> &bandwidthStats) {
    double sum = 0.0;
    for (auto stat : bandwidthStats) {
        sum += stat.returnAppropriateMetric() * 1e-9;
    }
    return sum;
}

double NodeHelperSingle::calculateFirstBandwidth(std::vector<PerformanceStatistic> &bandwidthStats) {
    return bandwidthStats[0].returnAppropriateMetric() * 1e-9;
}

std::vector<double> NodeHelperSingle::calculateVectorBandwidth(std::vector<double> &results, std::vector<int> originalRanks) {
    return results;
}

void NodeHelperSingle::synchronizeProcess() {
    // NOOP
}

CUresult NodeHelperSingle::streamSynchronizeWrapper(CUstream stream) const {
    return cuStreamSynchronize(stream);
}

void NodeHelperSingle::barrierAllRanks() {
    // No-op in single-node mode.
}

void NodeHelperSingle::streamBlockerReset() {
    *blockingVarHost = 0;
}

void NodeHelperSingle::streamBlockerRelease() {
    *blockingVarHost = 1;
}

void NodeHelperSingle::streamBlockerBlock(CUstream stream) {
    // start the spin kernel on the stream
    CU_ASSERT(spinKernel(blockingVarHost, stream));
}

double MemcpyOperation::doMemcpy(const std::vector<const MemcpyBuffer*> &srcBuffers, const std::vector<const MemcpyBuffer*> &dstBuffers) {
    MemcpyDispatchInfo dispatchInfo = nodeHelper->dispatchMemcpy(srcBuffers, dstBuffers, ctxPreference);
    auto result = doMemcpyCore(dispatchInfo);
    return result[0];
}

std::vector<double> MemcpyOperation::doMemcpyVector(const std::vector<const MemcpyBuffer*> &srcBuffers, const std::vector<const MemcpyBuffer*> &dstBuffers) {
    MemcpyDispatchInfo dispatchInfo = nodeHelper->dispatchMemcpy(srcBuffers, dstBuffers, ctxPreference);
    auto results = doMemcpyCore(dispatchInfo);

    return nodeHelper->calculateVectorBandwidth(results, dispatchInfo.originalRanks);
}

static bool shouldBlockStream(const MemcpyBuffer *dst, const MemcpyBuffer *src) {
    if (gSettings.bounceBufferConfComputeEnabled) {
        // With confidential computing using encrypted bounce buffers for CPU<->GPU transfers,
        // host-to-device and device-to-host copies become synchronous.
        // HostDeviceTransfer with cuMemcpyAsync are not waived when bounce-buffer CC is enabled.
        // Avoid blocking the stream in this case.
        if ((dst->getPrimaryCtx() == nullptr) || (src->getPrimaryCtx() == nullptr)) {
            return false;
        }
    }

    return true;
}

void MemcpyOperation::recordCoefficientOfVariation(double cv) {
    if (cv > worstCoeffOfVariation_) {
        worstCoeffOfVariation_ = cv;
    }
}

std::vector<double> MemcpyOperation::doMemcpyCore(MemcpyDispatchInfo &info) {
    std::vector<CUstream> streams(info.srcBuffers.size());
    std::vector<CUevent> startEvents(info.srcBuffers.size());
    std::vector<CUevent> endEvents(info.srcBuffers.size());
    std::vector<PerformanceStatistic> bandwidthStats(info.srcBuffers.size());
    std::vector<size_t> adjustedCopySizes(info.srcBuffers.size());
    PerformanceStatistic totalBandwidth;
    // Tracks the summed bandwidth across all simultaneous copies per sample, so
    // the run-to-run coefficient of variation of the SUM_BW metric can be
    // computed from the same data used to report it.
    PerformanceStatistic sumBandwidthStat;
    CUevent totalEnd;
    std::vector<size_t> finalCopySize(info.srcBuffers.size());

    for (int i = 0; i < info.srcBuffers.size(); i++) {
        CU_ASSERT(cuCtxSetCurrent(info.contexts[i]));
        // allocate the per simulaneous copy resources
        CU_ASSERT(cuStreamCreate(&streams[i], CU_STREAM_NON_BLOCKING));
        info.streams.push_back(streams[i]);
        CU_ASSERT(cuEventCreate(&startEvents[i], CU_EVENT_DEFAULT));
        CU_ASSERT(cuEventCreate(&endEvents[i], CU_EVENT_DEFAULT));
        // Get the final copy size that will be used.
        // CE and SM copy sizes will differ due to possible truncation
        // during SM copies.
        finalCopySize[i] = memcpyInitiator->getAdjustedCopySize(
            info.srcBuffers[i]->getBufferSize(),
            info.dstBuffers[i]->getBufferSize(),
            streams[i]);
        info.adjustedCopySizes.push_back(finalCopySize[i]);
    }
    info.nodeHelper = nodeHelper;
    info.loopCount = loopCount;  // Store loop count for verification

    if (info.contexts.size() > 0) {
        CU_ASSERT(cuCtxSetCurrent(info.contexts[0]));
    }
    // If no memcpy operations are happening on this node, let's still record a totalEnd event to simplify code
    CU_ASSERT(cuEventCreate(&totalEnd, CU_EVENT_DEFAULT));

    // This loop is for sampling the testcase (which itself has a loop count)
    for (unsigned int n = 0; n < gSettings.averageLoopCount; n++) {
        nodeHelper->streamBlockerReset();
        nodeHelper->synchronizeProcess();

        memcpyInitiator->memsetPattern(info);
        // block stream, and enqueue copy
        for (int i = 0; i < info.srcBuffers.size(); i++) {
            CU_ASSERT(cuCtxSetCurrent(info.contexts[i]));

            if (shouldBlockStream(info.dstBuffers[i], info.srcBuffers[i])) {
                nodeHelper->streamBlockerBlock(info.streams[i]);
            }

            // warmup: pass nullptr so verification buffers are not polluted (memcmpPattern checks measurement data)
            MemcpyDescriptor desc(info.dstBuffers[i]->getBuffer(), info.srcBuffers[i]->getBuffer(),
                                  info.streams[i], info.srcBuffers[i]->getBufferSize(), WARMUP_COUNT);
            desc.useNvlinkUtilScheduling = useNvlinkUtilScheduling_;
            memcpyInitiator->memcpyFunc(desc, nullptr);
        }

        if (info.srcBuffers.size() > 0) {
            CU_ASSERT(cuCtxSetCurrent(info.contexts[0]));
            CU_ASSERT(cuEventRecord(startEvents[0], info.streams[0]));
        }

        for (int i = 1; i < info.srcBuffers.size(); i++) {
            // ensure that all copies are launched at the same time
            CU_ASSERT(cuCtxSetCurrent(info.contexts[i]));
            CU_ASSERT(cuStreamWaitEvent(info.streams[i], startEvents[0], 0));
            CU_ASSERT(cuEventRecord(startEvents[i], info.streams[i]));
        }

        for (int i = 0; i < info.srcBuffers.size(); i++) {
            CU_ASSERT(cuCtxSetCurrent(info.contexts[i]));
            size_t copySize = info.adjustedCopySizes[i];

            MemcpyDescriptor desc(info.dstBuffers[i]->getBuffer(), info.srcBuffers[i]->getBuffer(),
                                  info.streams[i], copySize, loopCount);
            desc.useNvlinkUtilScheduling = useNvlinkUtilScheduling_;
            adjustedCopySizes[i] = memcpyInitiator->memcpyFunc(desc, &info);
            info.adjustedCopySizes[i] = adjustedCopySizes[i];  // Update with actual truncated size
            CU_ASSERT(cuEventRecord(endEvents[i], info.streams[i]));

            if (bandwidthValue == BandwidthValue::TOTAL_BW && i != 0) {
                // make stream0 wait on the all the others so we can measure total completion time
                CU_ASSERT(cuStreamWaitEvent(info.streams[0], endEvents[i], 0));
            }
        }

        // record the total end - only valid if BandwidthValue::TOTAL_BW is used due to StreamWaitEvent above
        if (info.srcBuffers.size() > 0) {
            CU_ASSERT(cuCtxSetCurrent(info.contexts[0]));
            CU_ASSERT(cuEventRecord(totalEnd, info.streams[0]));
        }

        // unblock the streams
        nodeHelper->streamBlockerRelease();

        for (CUstream stream : info.streams) {
            CU_ASSERT(nodeHelper->streamSynchronizeWrapper(stream));
        }

        nodeHelper->synchronizeProcess();

        if (!gSettings.skipVerification) {
            memcpyInitiator->memcmpPattern(info);
        }

        double sampleSumBandwidth = 0.0;
        for (int i = 0; i < bandwidthStats.size(); i++) {
            float timeWithEvents = 0.0f;
            CU_ASSERT(cuEventElapsedTime(&timeWithEvents, startEvents[i], endEvents[i]));
            double elapsedWithEventsInUs = ((double) timeWithEvents * 1000.0);
            unsigned long long bandwidth = (adjustedCopySizes[i] * loopCount * 1000ull * 1000ull) / (unsigned long long) elapsedWithEventsInUs;

            bandwidth = memcpyInitiator->getAdjustedBandwidth(bandwidth);

            bandwidthStats[i]((double) bandwidth);
            sampleSumBandwidth += (double) bandwidth;

            if (bandwidthValue == BandwidthValue::SUM_BW || bandwidthValue == BandwidthValue::TOTAL_BW || i == 0) {
                // Verbose print only the values that are used for the final output
                VERBOSE << "\tSample " << n << ": " << info.srcBuffers[i]->getBufferString() << " -> " << info.dstBuffers[i]->getBufferString() << ": " <<
                    std::fixed << std::setprecision(2) << (double)bandwidth * 1e-9 << " GB/s\n";
            }
        }
        sumBandwidthStat(sampleSumBandwidth);

        if (bandwidthValue == BandwidthValue::TOTAL_BW) {
            float totalTime = 0.0f;
            if (startEvents.size() > 0) {
                CU_ASSERT(cuEventElapsedTime(&totalTime, startEvents[0], totalEnd));
            }
            double elapsedTotalInUs = ((double) totalTime * 1000.0);

            // get total bytes copied
            double totalSize = 0;
            for (double size : adjustedCopySizes) {
                totalSize += size;
            }

            double bandwidth = nodeHelper->calculateTotalBandwidth(elapsedTotalInUs, totalSize, loopCount);
            totalBandwidth(bandwidth);
            VERBOSE << "\tSample " << n << ": Total Bandwidth : " <<
                std::fixed << std::setprecision(2) << (double)bandwidth * 1e-9 << " GB/s\n";
        }
    }

    // cleanup
    CU_ASSERT(cuEventDestroy(totalEnd));

    for (int i = 0; i < info.srcBuffers.size(); i++) {
        CU_ASSERT(cuStreamDestroy(info.streams[i]));
        CU_ASSERT(cuEventDestroy(startEvents[i]));
        CU_ASSERT(cuEventDestroy(endEvents[i]));
    }

    // Record the run-to-run coefficient of variation for the metric actually
    // reported, measuring stability across repeats of the same transfer.
    if (bandwidthValue == BandwidthValue::SUM_BW) {
        recordCoefficientOfVariation(sumBandwidthStat.coefficientOfVariation());
        return {nodeHelper->calculateSumBandwidth(bandwidthStats)};
    } else if (bandwidthValue == BandwidthValue::TOTAL_BW) {
        recordCoefficientOfVariation(totalBandwidth.coefficientOfVariation());
        return {totalBandwidth.returnAppropriateMetric() * 1e-9};
    } else if (bandwidthValue == BandwidthValue::VECTOR_BW) {
        std::vector<double> ret;
        for (auto stat : bandwidthStats) {
            recordCoefficientOfVariation(stat.coefficientOfVariation());
            ret.push_back(stat.returnAppropriateMetric() * 1e-9);
        }
        return ret;
    } else {
        if (!bandwidthStats.empty()) {
            recordCoefficientOfVariation(bandwidthStats[0].coefficientOfVariation());
        }
        return {nodeHelper->calculateFirstBandwidth(bandwidthStats)};
    }
}

size_t MemcpyInitiatorSM::memcpyFunc(MemcpyDescriptor &desc, MemcpyDispatchInfo* info) {
    return copyKernel(desc);
}

size_t MemcpyInitiatorSMReadOnly::getAdjustedCopySize(size_t srcSize, size_t dstSize, CUstream stream) {
    // Read-only test: only src buffer is accessed, so adjust based on srcSize only
    return MemcpyInitiatorSM::getAdjustedCopySize(srcSize, srcSize, stream);
}

size_t MemcpyInitiatorSMReadOnly::memcpyFunc(MemcpyDescriptor &desc, MemcpyDispatchInfo* info) {
    CUdeviceptr d_verify = 0;
    size_t result = readOnlyKernel(desc, &d_verify);
    if (info) {
        info->verificationBuffers.push_back(d_verify);
        info->verificationSizes.push_back(result);
    }
    return result;
}

void MemcpyInitiatorSMReadOnly::memsetPattern(MemcpyDispatchInfo &info) const {
    // Clear the verification buffers vector for new test run
    info.verificationBuffers.clear();
    info.verificationSizes.clear();

    // Only need to initialize source buffer for read-only test
    for (int i = 0; i < info.srcBuffers.size(); i++) {
        CU_ASSERT(cuCtxSetCurrent(info.contexts[i]));
        // Use a varied pattern (not all same value) to avoid XOR cancellation in CRC
        // Initialize with sequential uint4 pattern with distinct values per component
        // Pattern: (j, j+0x12345678, j+0xABCDEF01, j+0xDEADBEEF)
        // Use max of src/dst buffer sizes since adjustedCopySizes isn't populated yet
        size_t bufferSize = std::max(info.srcBuffers[i]->getBufferSize(), info.dstBuffers[i]->getBufferSize());
        uint4* h_pattern = nullptr;
        CU_ASSERT(cuMemAllocHost((void**)&h_pattern, bufferSize));
        size_t numElements = bufferSize / sizeof(uint4);
        for (size_t j = 0; j < numElements; j++) {
            h_pattern[j] = make_uint4(j, j + 0x12345678, j + 0xABCDEF01, j + 0xDEADBEEF);
        }
        CU_ASSERT(cuMemcpyHtoDAsync(info.srcBuffers[i]->getBuffer(), h_pattern, bufferSize, info.streams[i]));
        CU_ASSERT(info.nodeHelper->streamSynchronizeWrapper(info.streams[i]));
        CU_ASSERT(cuMemFreeHost(h_pattern));
    }
}

void MemcpyInitiatorSMReadOnly::memcmpPattern(MemcpyDispatchInfo &info) const {
    // For read-only tests, verify that data was actually read by checking sample values
    // The pattern is: (j, j+0x12345678, j+0xABCDEF01, j+0xDEADBEEF) for element j
    for (size_t i = 0; i < info.verificationBuffers.size() && i < info.contexts.size(); i++) {
        CU_ASSERT(cuCtxSetCurrent(info.contexts[i]));
        // The kernel pre-sampled 3 elements (first, middle, last) from the source buffer
        // and stored them in verifyOut[0..2], with verifyOut[3] being an accumulator
        // We just need to read these 4 pre-sampled values from device memory
        uint4 h_samples[4];
        CU_ASSERT(cuMemcpyDtoHAsync(h_samples, info.verificationBuffers[i], sizeof(uint4) * 4, info.streams[i]));
        CU_ASSERT(info.nodeHelper->streamSynchronizeWrapper(info.streams[i]));

        // Calculate indices using the truncated size returned by readOnlyKernel
        // The adjustedCopySize is already the truncated size (sizeInElement * sizeof(uint4))
        size_t adjustedCopySize = info.verificationSizes[i];
        size_t totalElements = adjustedCopySize / sizeof(uint4);

        // Sample at first, middle, and last positions
        size_t firstIdx = 0;
        size_t midIdx = totalElements / 2;
        size_t lastIdx = totalElements > 0 ? totalElements - 1 : 0;
        const size_t indices[] = {firstIdx, midIdx, lastIdx};

        // Verify sample values against expected pattern
        bool patternCorrect = true;

        for (int j = 0; j < 3; j++) {
            size_t idx = indices[j];
            uint32_t expected_x = (uint32_t)idx;
            uint32_t expected_y = (uint32_t)idx + 0x12345678;
            uint32_t expected_z = (uint32_t)idx + 0xABCDEF01;
            uint32_t expected_w = (uint32_t)idx + 0xDEADBEEF;

            if (h_samples[j].x != expected_x || h_samples[j].y != expected_y ||
                h_samples[j].z != expected_z || h_samples[j].w != expected_w) {
                std::cerr << "ERROR: Read-only pattern mismatch at element " << idx << std::endl;
                std::cerr << "  Expected: (" << expected_x << ", " << expected_y << ", "
                          << expected_z << ", 0x" << std::hex << expected_w << std::dec << ")" << std::endl;
                std::cerr << "  Got:      (" << h_samples[j].x << ", " << h_samples[j].y << ", "
                          << h_samples[j].z << ", 0x" << std::hex << h_samples[j].w << std::dec << ")" << std::endl;
                patternCorrect = false;
            }
        }

        VERBOSE << "Read-only test: Verified " << totalElements << " elements (sampled indices: 0, "
                << midIdx << ", " << lastIdx << ")\n";

        if (!patternCorrect) {
            std::cerr << "ERROR: Read-only test failed - incorrect data read from source buffer!" << std::endl;
            ASSERT(false);
        }

        // Free the verification buffer
        CU_ASSERT(cuMemFree(info.verificationBuffers[i]));
    }

    // Clear the buffers after verification
    info.verificationBuffers.clear();
    info.verificationSizes.clear();
}

size_t MemcpyInitiatorSMWriteOnly::getAdjustedCopySize(size_t srcSize, size_t dstSize, CUstream stream) {
    // Write-only test: only dst buffer is accessed, so adjust based on dstSize only
    return MemcpyInitiatorSM::getAdjustedCopySize(dstSize, dstSize, stream);
}

size_t MemcpyInitiatorSMWriteOnly::memcpyFunc(MemcpyDescriptor &desc, MemcpyDispatchInfo* info) {
    return writeOnlyKernel(desc);
}

void MemcpyInitiatorSMWriteOnly::memsetPattern(MemcpyDispatchInfo &info) const {
    // Initialize destination buffer to a known pattern (different from what kernel writes)
    // This helps verify the kernel actually wrote data
    for (int i = 0; i < info.dstBuffers.size(); i++) {
        CU_ASSERT(cuCtxSetCurrent(info.contexts[i]));
        // Fill with 0xFFFFFFFF to distinguish from unwritten (0x00) or kernel-written data
        size_t bufferSize = info.dstBuffers[i]->getBufferSize();
        CU_ASSERT(cuMemsetD32Async(info.dstBuffers[i]->getBuffer(), 0xFFFFFFFF,
                                    bufferSize / sizeof(unsigned int), info.streams[i]));
        CU_ASSERT(info.nodeHelper->streamSynchronizeWrapper(info.streams[i]));
    }
}

void MemcpyInitiatorSMWriteOnly::memcmpPattern(MemcpyDispatchInfo &info) const {
    // For write-only tests, verify that the correct pattern was written
    // The kernel writes: make_uint4(threadIdx.x, blockIdx.x, iteration_counter, 0xCAFEBABE)
    // The iteration counter allows us to verify all loop iterations completed

    CUdevice dev;
    CUcontext ctx;

    for (int i = 0; i < info.dstBuffers.size(); i++) {
        CU_ASSERT(cuCtxSetCurrent(info.contexts[i]));
        CU_ASSERT(cuStreamGetCtx(info.streams[i], &ctx));
        CU_ASSERT(cuCtxGetDevice(&dev));

        int numSm;
        CU_ASSERT(cuDeviceGetAttribute(&numSm, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev));
        unsigned int totalThreadCount = numSm * numThreadPerBlock;

        // Verify ALL written elements in parallel using GPU kernel
        size_t bufferElements = info.adjustedCopySizes[i] / sizeof(uint4);

        if (bufferElements == 0) {
            VERBOSE << "Write-only test: Buffer too small to verify\n";
            continue;
        }

        // The kernel writes the loop iteration counter (i) in the .z component
        // After all iterations complete, .z should contain (loopCount - 1)
        unsigned int expectedIteration = info.loopCount - 1;

        // Allocate device memory for error counter
        unsigned int* d_errorCount = nullptr;
        CU_ASSERT(cuMemAlloc((CUdeviceptr*)&d_errorCount, sizeof(unsigned int)));
        CU_ASSERT(cuMemsetD32((CUdeviceptr)d_errorCount, 0, 1));

        // Launch verification kernel in parallel using wrapper function
        CU_ASSERT(verifyWriteOnlyKernel(
            info.streams[i],
            info.dstBuffers[i]->getBuffer(),
            bufferElements,
            numThreadPerBlock,
            expectedIteration,
            (CUdeviceptr)d_errorCount));
        CU_ASSERT(info.nodeHelper->streamSynchronizeWrapper(info.streams[i]));

        // Read back error count
        unsigned int h_errorCount = 0;
        CU_ASSERT(cuMemcpyDtoH(&h_errorCount, (CUdeviceptr)d_errorCount, sizeof(unsigned int)));
        CU_ASSERT(cuMemFree((CUdeviceptr)d_errorCount));

        if (h_errorCount > 0) {
            std::cerr << "ERROR: Write-only test found " << h_errorCount << " mismatched elements!" << std::endl;
            ASSERT(false);
        }

        VERBOSE << "Write-only test: Verified ALL " << bufferElements << " elements correctly written (in parallel)\n";
        VERBOSE << "  Last iteration counter: " << expectedIteration << " (loop completed "
                << (expectedIteration + 1) << " times)\n";
    }
}

size_t MemcpyInitiatorSM::getAdjustedCopySize(size_t srcSize, size_t dstSize, CUstream stream) {
    CUdevice dev;
    CUcontext ctx;

    ASSERT(srcSize == dstSize);
    CU_ASSERT(cuStreamGetCtx(stream, &ctx));
    CU_ASSERT(cuCtxGetDevice(&dev));
    int numSm;
    CU_ASSERT(cuDeviceGetAttribute(&numSm, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev));
    unsigned int totalThreadCount = numSm * threadsPerBlock;
    // We want to calculate the exact copy sizes that will be
    // used by the copy kernels.
    if (srcSize < (smallBufferThreshold * _MiB)) {
        // copy size is rounded down to 16 bytes
        int numUint4 = srcSize / sizeof(uint4);
        return numUint4 * sizeof(uint4);
    }
    // adjust size to elements (size is multiple of MB, so no truncation here)
    size_t sizeInElement = srcSize / sizeof(uint4);
    // this truncates the copy
    sizeInElement = totalThreadCount * (sizeInElement / totalThreadCount);
    return sizeInElement * sizeof(uint4);
}

size_t MemcpyInitiatorCE::memcpyFunc(MemcpyDescriptor &desc, MemcpyDispatchInfo* info) {
    for (unsigned int l = 0; l < desc.loopCount; l++) {
        CU_ASSERT(cuMemcpyAsync(desc.dst, desc.src, desc.copySize, desc.stream));
    }

    return desc.copySize;
}

size_t MemcpyInitiatorCE::getAdjustedCopySize(size_t srcSize, size_t dstSize, CUstream stream) {
    ASSERT(srcSize == dstSize);
    // CE does not change/truncate buffer size
    return srcSize;
}

size_t MemcpyInitiatorMulticastWrite::memcpyFunc(MemcpyDescriptor &desc, MemcpyDispatchInfo* info) {
    return multicastCopy(desc.dst, desc.src, desc.copySize, desc.stream, desc.loopCount);
}

size_t MemcpyInitiatorMulticastWrite::getAdjustedCopySize(size_t srcSize, size_t dstSize, CUstream stream) {
    ASSERT(srcSize == dstSize);
    ASSERT(srcSize % 16 == 0);  // multimem_st_128 requires 16-byte granularity
    return srcSize;
}



size_t MemcpyInitiatorSMSplitWarp::memcpyFunc(MemcpyDescriptor &desc, MemcpyDispatchInfo* info) {
    return copyKernelSplitWarp(desc);
}

size_t MemcpyInitiatorTMA::memcpyFunc(MemcpyDescriptor &desc, MemcpyDispatchInfo* info) {
    return copyKernelTMA(desc);
}

size_t MemcpyInitiatorTMA::getAdjustedCopySize(size_t srcSize, size_t dstSize, CUstream stream) {
    ASSERT(srcSize == dstSize);
    return srcSize;
}

void MemcpyInitiatorTMA::memsetPattern(MemcpyDispatchInfo &info) const {
    MemcpyInitiatorSM smInitiator;
    smInitiator.memsetPattern(info);
}

void MemcpyInitiatorTMA::memcmpPattern(MemcpyDispatchInfo &info) const {
    MemcpyInitiatorSM smInitiator;
    smInitiator.memcmpPattern(info);
}

unsigned long long MemcpyInitiatorTMA::getAdjustedBandwidth(unsigned long long bandwidth) {
    return bandwidth;
}

size_t MemcpyInitiatorSMOneToAll::memcpyFunc(MemcpyDescriptor &desc, MemcpyDispatchInfo* info) {
    desc.threadsPerBlock = threadsPerBlock;
    return copyKernel(desc);
}

MemPtrChaseOperation::MemPtrChaseOperation(unsigned long long loopCount) : loopCount(loopCount) {
    cudaDeviceProp prop;
    CUDA_ASSERT(cudaGetDeviceProperties(&prop, 0));
    smCount = prop.multiProcessorCount;
}

double MemPtrChaseOperation::doPtrChase(const int srcId, const MemcpyBuffer &peerBuffer, PtrChasingKernel kernel) {
    double lat = 0.0;
    lat = latencyPtrChaseKernel(srcId, (void*)peerBuffer.getBuffer(), peerBuffer.getBufferSize(), latencyMemAccessCnt, smCount, kernel);
    return lat;
}
