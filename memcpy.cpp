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

#include "common.h"
#include "inline_common.h"
#include "memcpy.h"
#include "output.h"
#include "kernels.cuh"
#include "vector_types.h"
#include "common.h"

#define WARMUP_COUNT 4

MemcpyBuffer::MemcpyBuffer(size_t bufferSize): bufferSize(bufferSize), buffer(nullptr) {}

CUdeviceptr MemcpyBuffer::getBuffer() const {
    return (CUdeviceptr)buffer;
}

size_t MemcpyBuffer::getBufferSize() const {
    return bufferSize;
}
void MemcpyBuffer::memsetPattern(CUdeviceptr buffer, unsigned long long size, unsigned int seed) const {
    unsigned int* pattern;
    unsigned int n = 0;
    void * _buffer = (void*) buffer;
    unsigned long long _2MBchunkCount = size / (1024 * 1024 * 2);
    unsigned long long remaining = size - (_2MBchunkCount * 1024 * 1024 * 2);

    // Allocate 2MB of pattern
    CU_ASSERT(cuMemHostAlloc((void**)&pattern, sizeof(char) * 1024 * 1024 * 2, CU_MEMHOSTALLOC_PORTABLE));
    xorshift2MBPattern(pattern, seed);

    for (n = 0; n < _2MBchunkCount; n++) {
        CU_ASSERT(cuMemcpyAsync((CUdeviceptr)_buffer, (CUdeviceptr)pattern, 1024 * 1024 * 2, CU_STREAM_PER_THREAD));
        _buffer = (char*)_buffer + (1024 * 1024 * 2);
    }
    if (remaining) {
        CU_ASSERT(cuMemcpyAsync((CUdeviceptr)_buffer, (CUdeviceptr)pattern, (size_t)remaining, CU_STREAM_PER_THREAD));
    }

    CU_ASSERT(streamSynchronizeWrapper(CU_STREAM_PER_THREAD));
    CU_ASSERT(cuMemFreeHost((void*)pattern));
}

void MemcpyBuffer::memcmpPattern(CUdeviceptr buffer, unsigned long long size, unsigned int seed) const {
     unsigned int* devicePattern;
    unsigned int* pattern;
    unsigned long long _2MBchunkCount = size / (1024 * 1024 * 2);
    unsigned long long remaining = size - (_2MBchunkCount * 1024 * 1024 * 2);
    unsigned int n = 0;
    unsigned int x = 0;
    void * _buffer = (void*) buffer;

    // Allocate 2MB of pattern
    CU_ASSERT(cuMemHostAlloc((void**)&devicePattern, sizeof(char) * 1024 * 1024 * 2, CU_MEMHOSTALLOC_PORTABLE));
    pattern = (unsigned int*)malloc(sizeof(char) * 1024 * 1024 * 2);
    xorshift2MBPattern(pattern, seed);

    for (n = 0; n < _2MBchunkCount; n++) {
        CU_ASSERT(cuMemcpyAsync((CUdeviceptr)devicePattern, (CUdeviceptr)_buffer, 1024 * 1024 * 2, CU_STREAM_PER_THREAD));
        CU_ASSERT(streamSynchronizeWrapper(CU_STREAM_PER_THREAD));

        if (memcmp(pattern, devicePattern, 1024 * 1024 * 2) != 0) {
            for (x = 0; x < (1024 * 1024 * 2) / sizeof(unsigned int); x++) {
                if (devicePattern[x] != pattern[x]) {
                    std::stringstream errmsg1;
                    std::stringstream errmsg2;
                    errmsg1 << " Invalid value when checking the pattern at <" << (void*)((char*)_buffer + n * (1024 * 1024 * 2) + x * sizeof(unsigned int)) << ">";
                    errmsg2 << " Current offset [ " << (unsigned long long)((char*)_buffer - (char*)buffer) + (unsigned long long)(x * sizeof(unsigned int)) << "/" << (size) << "]";
                    output->recordErrorCurrentTest(errmsg1.str(), errmsg2.str());
                    output->print();
                    std::abort();
                }
            
            }
        }

        _buffer = (char*)_buffer + (1024 * 1024 * 2);
    }
    if (remaining) {
        CU_ASSERT(cuMemcpyAsync((CUdeviceptr)devicePattern, (CUdeviceptr)_buffer, (size_t)remaining, CU_STREAM_PER_THREAD));
        CU_ASSERT(streamSynchronizeWrapper(CU_STREAM_PER_THREAD));
        if (memcmp(pattern, devicePattern, (size_t)remaining) != 0) {
            for (x = 0; x < remaining / sizeof(unsigned int); x++) {
                if (devicePattern[x] != pattern[x]) {
                    std::stringstream errmsg1;
                    std::stringstream errmsg2;
                    errmsg1 << " Invalid value when checking the pattern at <" << (void*)((char*)buffer + n * (1024 * 1024 * 2) + x * sizeof(unsigned int)) << ">";
                    errmsg2 << " Current offset [ " << (unsigned long long)((char*)_buffer - (char*)buffer) + (unsigned long long)(x * sizeof(unsigned int)) << "/" << (size) << "]";
                    output->recordErrorCurrentTest(errmsg1.str(), errmsg2.str());
                    output->print();
                    std::abort();
                }
            }
        }
    }

    CU_ASSERT(cuMemFreeHost((void*)devicePattern));
    free(pattern);
}

void MemcpyBuffer::xorshift2MBPattern(unsigned int* buffer, unsigned int seed) const {
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

CUresult MemcpyBuffer::streamSynchronizeWrapper(CUstream stream) const {
    return cuStreamSynchronize(stream);
}

HostBuffer::HostBuffer(size_t bufferSize, int targetDeviceId): MemcpyBuffer(bufferSize) {
    CUcontext targetCtx;

    // Before allocating host memory, set correct NUMA affinity
    setOptimalCpuAffinity(targetDeviceId);
    CU_ASSERT(cuDevicePrimaryCtxRetain(&targetCtx, targetDeviceId));
    CU_ASSERT(cuCtxSetCurrent(targetCtx));

    CU_ASSERT(cuMemHostAlloc(&buffer, bufferSize, CU_MEMHOSTALLOC_PORTABLE));
}

HostBuffer::~HostBuffer() {
    if (isMemoryOwnedByCUDA(buffer)) {
        CU_ASSERT(cuMemFreeHost(buffer));
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

bool DeviceBuffer::enablePeerAcess(const DeviceBuffer &peerBuffer) {
    int canAccessPeer = 0;
    CU_ASSERT(cuDeviceCanAccessPeer(&canAccessPeer, getBufferIdx(), peerBuffer.getBufferIdx()));
    if (canAccessPeer) {
        CUresult res;
        CU_ASSERT(cuCtxSetCurrent(peerBuffer.getPrimaryCtx()));
        res = cuCtxEnablePeerAccess(getPrimaryCtx(), 0);
        if (res != CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED)
            CU_ASSERT(res);

        CU_ASSERT(cuCtxSetCurrent(getPrimaryCtx()));
        res = cuCtxEnablePeerAccess(peerBuffer.getPrimaryCtx(), 0);
        if (res != CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED)
            CU_ASSERT(res);

        return true;
    }
    return false;
}

MemcpyOperation::MemcpyOperation(unsigned long long loopCount, MemcpyInitiator* memcpyInitiator, ContextPreference ctxPreference, BandwidthValue bandwidthValue) :
    MemcpyOperation(loopCount, memcpyInitiator, new NodeHelperSingle(), ctxPreference, bandwidthValue)
{
}

MemcpyOperation::MemcpyOperation(unsigned long long loopCount, MemcpyInitiator* memcpyInitiator, NodeHelper* nodeHelper, ContextPreference ctxPreference, BandwidthValue bandwidthValue) :
        loopCount(loopCount), memcpyInitiator(memcpyInitiator), nodeHelper(nodeHelper), ctxPreference(ctxPreference), bandwidthValue(bandwidthValue)
{
    procMask = (size_t *)calloc(1, PROC_MASK_SIZE);
    PROC_MASK_SET(procMask, getFirstEnabledCPU());
}

MemcpyOperation::~MemcpyOperation() {
    PROC_MASK_CLEAR(procMask, 0);
}

double MemcpyOperation::doMemcpy(const MemcpyBuffer &srcBuffer, const MemcpyBuffer &dstBuffer) {
    std::vector<const MemcpyBuffer*> srcBuffers = {&srcBuffer};
    std::vector<const MemcpyBuffer*> dstBuffers = {&dstBuffer};
    return doMemcpy(srcBuffers, dstBuffers);
}

MemcpyDispatchInfo::MemcpyDispatchInfo(std::vector<const MemcpyBuffer*> srcBuffers, std::vector<const MemcpyBuffer*> dstBuffers, std::vector<CUcontext> contexts) :
    srcBuffers(srcBuffers), dstBuffers(dstBuffers), contexts(contexts)
{}

NodeHelperSingle::NodeHelperSingle() {
    CU_ASSERT(cuMemHostAlloc((void **)&blockingVarHost, sizeof(*blockingVarHost), CU_MEMHOSTALLOC_PORTABLE));
}

NodeHelperSingle::~NodeHelperSingle() {
    CU_ASSERT(cuMemFreeHost((void*)blockingVarHost));
}

MemcpyDispatchInfo NodeHelperSingle::dispatchMemcpy(const std::vector<const MemcpyBuffer*> &srcBuffers, const std::vector<const MemcpyBuffer*> &dstBuffers, ContextPreference ctxPreference) {
    std::vector<CUcontext> contexts(srcBuffers.size());

    for (int i = 0; i < srcBuffers.size(); i++) {
        // prefer source context
        if (ctxPreference == PREFER_SRC_CONTEXT && srcBuffers[i]->getPrimaryCtx() != nullptr) {
            contexts[i] = srcBuffers[i]->getPrimaryCtx();
        } else if (dstBuffers[i]->getPrimaryCtx() != nullptr) {
            contexts[i] = dstBuffers[i]->getPrimaryCtx();
        }
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

void NodeHelperSingle::synchronizeProcess() {
    // NOOP
}

CUresult NodeHelperSingle::streamSynchronizeWrapper(CUstream stream) const {
    return cuStreamSynchronize(stream);
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
    return doMemcpyCore(dispatchInfo.srcBuffers, dispatchInfo.dstBuffers, dispatchInfo.contexts);
}

double MemcpyOperation::doMemcpyCore(const std::vector<const MemcpyBuffer*> &srcBuffers, const std::vector<const MemcpyBuffer*> &dstBuffers, const std::vector<CUcontext> &contexts) {
    std::vector<CUstream> streams(srcBuffers.size());
    std::vector<CUevent> startEvents(srcBuffers.size());
    std::vector<CUevent> endEvents(srcBuffers.size());
    std::vector<PerformanceStatistic> bandwidthStats(srcBuffers.size());
    std::vector<size_t> adjustedCopySizes(srcBuffers.size());
    PerformanceStatistic totalBandwidth;
    CUevent totalEnd;
    std::vector<size_t> finalCopySize(srcBuffers.size());

    for (int i = 0; i < srcBuffers.size(); i++) {
        CU_ASSERT(cuCtxSetCurrent(contexts[i]));
        // allocate the per simulaneous copy resources
        CU_ASSERT(cuStreamCreate(&streams[i], CU_STREAM_NON_BLOCKING));
        CU_ASSERT(cuEventCreate(&startEvents[i], CU_EVENT_DEFAULT));
        CU_ASSERT(cuEventCreate(&endEvents[i], CU_EVENT_DEFAULT));
        // Get the final copy size that will be used.
        // CE and SM copy sizes will differ due to possible truncation
        // during SM copies.
        finalCopySize[i] = memcpyInitiator->getAdjustedCopySize(srcBuffers[i]->getBufferSize(), streams[i]);
    }
    
    if (contexts.size() > 0) {
        CU_ASSERT(cuCtxSetCurrent(contexts[0]));
    }
    // If no memcpy operations are happening on this node, let's still record a totalEnd event to simplify code
    CU_ASSERT(cuEventCreate(&totalEnd, CU_EVENT_DEFAULT));

    // This loop is for sampling the testcase (which itself has a loop count)
    for (unsigned int n = 0; n < averageLoopCount; n++) {
        nodeHelper->streamBlockerReset();
        nodeHelper->synchronizeProcess();

        // Set the memory patterns correctly before spin kernel launch etc.
        for (int i = 0; i < srcBuffers.size(); i++) {
            dstBuffers[i]->memsetPattern(dstBuffers[i]->getBuffer(), finalCopySize[i], 0xCAFEBABE);
            srcBuffers[i]->memsetPattern(srcBuffers[i]->getBuffer(), finalCopySize[i], 0xBAADF00D);
        }        
        // block stream, and enqueue copy
        for (int i = 0; i < srcBuffers.size(); i++) {
            CU_ASSERT(cuCtxSetCurrent(contexts[i]));

            nodeHelper->streamBlockerBlock(streams[i]);

            // warmup
            memcpyInitiator->memcpyFunc(dstBuffers[i]->getBuffer(), srcBuffers[i]->getBuffer(), streams[i], srcBuffers[i]->getBufferSize(), WARMUP_COUNT);
        }

        if (srcBuffers.size() > 0) {
            CU_ASSERT(cuCtxSetCurrent(contexts[0]));
            CU_ASSERT(cuEventRecord(startEvents[0], streams[0]));
        }

        for (int i = 1; i < srcBuffers.size(); i++) {
            // ensure that all copies are launched at the same time
            CU_ASSERT(cuCtxSetCurrent(contexts[i]));
            CU_ASSERT(cuStreamWaitEvent(streams[i], startEvents[0], 0));
            CU_ASSERT(cuEventRecord(startEvents[i], streams[i]));
        }

        for (int i = 0; i < srcBuffers.size(); i++) {
            CU_ASSERT(cuCtxSetCurrent(contexts[i]));
            ASSERT(srcBuffers[i]->getBufferSize() == dstBuffers[i]->getBufferSize());
            adjustedCopySizes[i] = memcpyInitiator->memcpyFunc(dstBuffers[i]->getBuffer(), srcBuffers[i]->getBuffer(), streams[i], srcBuffers[i]->getBufferSize(), loopCount);
            CU_ASSERT(cuEventRecord(endEvents[i], streams[i]));
            if (bandwidthValue == BandwidthValue::TOTAL_BW && i != 0) {
                // make stream0 wait on the all the others so we can measure total completion time
                CU_ASSERT(cuStreamWaitEvent(streams[0], endEvents[i], 0));
            }
        }

        // record the total end - only valid if BandwidthValue::TOTAL_BW is used due to StreamWaitEvent above
        if (srcBuffers.size() > 0) {
            CU_ASSERT(cuCtxSetCurrent(contexts[0]));
            CU_ASSERT(cuEventRecord(totalEnd, streams[0]));
        }

        // unblock the streams
        nodeHelper->streamBlockerRelease();

        for (CUstream stream : streams) {
            CU_ASSERT(nodeHelper->streamSynchronizeWrapper(stream));
        }

        nodeHelper->synchronizeProcess();

        if (!skipVerification) {
            for (int i = 0; i < srcBuffers.size(); i++) {            
                dstBuffers[i]->memcmpPattern(dstBuffers[i]->getBuffer(), finalCopySize[i], 0xBAADF00D);
            }
        }

        for (int i = 0; i < bandwidthStats.size(); i++) {
            float timeWithEvents = 0.0f;
            CU_ASSERT(cuEventElapsedTime(&timeWithEvents, startEvents[i], endEvents[i]));
            double elapsedWithEventsInUs = ((double) timeWithEvents * 1000.0);
            unsigned long long bandwidth = (adjustedCopySizes[i] * loopCount * 1000ull * 1000ull) / (unsigned long long) elapsedWithEventsInUs;
            bandwidthStats[i]((double) bandwidth);

            if (bandwidthValue == BandwidthValue::SUM_BW || BandwidthValue::TOTAL_BW || i == 0) {
                // Verbose print only the values that are used for the final output
                VERBOSE << "\tSample " << n << ": " << srcBuffers[i]->getBufferString() << " -> " << dstBuffers[i]->getBufferString() << ": " <<
                    std::fixed << std::setprecision(2) << (double)bandwidth * 1e-9 << " GB/s\n";
            }
        }

        if (bandwidthValue == BandwidthValue::TOTAL_BW) {
            float totalTime = 0.0f;
            CU_ASSERT(cuEventElapsedTime(&totalTime, startEvents[0], totalEnd));
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

    for (int i = 0; i < srcBuffers.size(); i++) {
        CU_ASSERT(cuStreamDestroy(streams[i]));
        CU_ASSERT(cuEventDestroy(startEvents[i]));
        CU_ASSERT(cuEventDestroy(endEvents[i]));
    }

    if (bandwidthValue == BandwidthValue::SUM_BW) {
        return nodeHelper->calculateSumBandwidth(bandwidthStats);
    } else if (bandwidthValue == BandwidthValue::TOTAL_BW) {
        return totalBandwidth.returnAppropriateMetric() * 1e-9;
    } else {
        return nodeHelper->calculateFirstBandwidth(bandwidthStats);
    }
}

size_t MemcpyInitiatorSM::memcpyFunc(CUdeviceptr dst, CUdeviceptr src, CUstream stream, size_t copySize, unsigned long long loopCount) {
    return copyKernel(dst, src, copySize, stream, loopCount);
}
size_t MemcpyInitiatorSM::getAdjustedCopySize(size_t size, CUstream stream) {
    CUdevice dev;
    CUcontext ctx;

    CU_ASSERT(cuStreamGetCtx(stream, &ctx));
    CU_ASSERT(cuCtxGetDevice(&dev));
    int numSm;
    CU_ASSERT(cuDeviceGetAttribute(&numSm, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev));
    unsigned int totalThreadCount = numSm * numThreadPerBlock;
    // We want to calculate the exact copy sizes that will be
    // used by the copy kernels.
    if (size < (defaultBufferSize * _MiB) ) {
        // copy size is rounded down to 16 bytes
        int numUint4 = size / sizeof(uint4);
        return numUint4 * sizeof(uint4);
    }
    // adjust size to elements (size is multiple of MB, so no truncation here)
    size_t sizeInElement = size / sizeof(uint4);
    // this truncates the copy
    sizeInElement = totalThreadCount * (sizeInElement / totalThreadCount);
    return sizeInElement * sizeof(uint4);
}

size_t MemcpyInitiatorCE::memcpyFunc(CUdeviceptr dst, CUdeviceptr src, CUstream stream, size_t copySize, unsigned long long loopCount) {
    for (unsigned int l = 0; l < loopCount; l++) {
        CU_ASSERT(cuMemcpyAsync(dst, src, copySize, stream));
    }

    return copySize;
}

size_t MemcpyInitiatorCE::getAdjustedCopySize(size_t size, CUstream stream) {
    //CE does not change/truncate buffer size
    return size;
}

MemPtrChaseOperation::MemPtrChaseOperation(unsigned long long loopCount) : loopCount(loopCount)
{
}

double MemPtrChaseOperation::doPtrChase(const int srcId, const MemcpyBuffer &peerBuffer) {
    double lat = 0.0;
    lat = latencyPtrChaseKernel(srcId, (void*)peerBuffer.getBuffer(), peerBuffer.getBufferSize(), loopCount);
    return lat;
}
