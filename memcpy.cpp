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
#include "memcpy_le.h"
#include "output.h"
#include "kernels.cuh"
#include "kernels_le_tma.hpp"
#include "vector_types.h"
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

// Verify that every uint32 in `buffer` (num_elements words) equals `expected_bits`.
// Uses a tiled 2 MiB device pattern and memcmpKernel with a device-side error flag.
static void memcmpConstantHelper(CUdeviceptr buffer, unsigned long long num_elements,
                                  uint32_t expected_bits, std::shared_ptr<NodeHelper> nodeHelper) {
    unsigned long long num_pattern_elements = _2MiB / sizeof(unsigned int);
    unsigned int *h_pattern;
    CUdeviceptr d_pattern;
    int h_errorFlag = 0;
    CUdeviceptr d_errorFlag;
    CU_ASSERT(cuMemHostAlloc((void **)&h_pattern, _2MiB, CU_MEMHOSTALLOC_PORTABLE));
    for (unsigned long long k = 0; k < num_pattern_elements; k++) h_pattern[k] = expected_bits;
    CU_ASSERT(cuMemAlloc(&d_pattern, _2MiB));
    CU_ASSERT(cuMemcpyAsync(d_pattern, (CUdeviceptr)h_pattern, _2MiB, CU_STREAM_PER_THREAD));
    CU_ASSERT(cuMemAlloc(&d_errorFlag, sizeof(int)));
    CU_ASSERT(cuMemcpyAsync(d_errorFlag, (CUdeviceptr)&h_errorFlag, sizeof(int), CU_STREAM_PER_THREAD));
    CU_ASSERT(memcmpKernel(CU_STREAM_PER_THREAD, buffer, d_pattern,
                           num_elements, num_pattern_elements, d_errorFlag));
    CU_ASSERT(nodeHelper->streamSynchronizeWrapper(CU_STREAM_PER_THREAD));
    CU_ASSERT(cuMemcpyAsync((CUdeviceptr)&h_errorFlag, d_errorFlag, sizeof(int), CU_STREAM_PER_THREAD));
    CU_ASSERT(cuStreamSynchronize(CU_STREAM_PER_THREAD));
    CU_ASSERT(cuMemFreeHost((void *)h_pattern));
    CU_ASSERT(cuMemFree(d_errorFlag));
    CU_ASSERT(cuMemFree(d_pattern));
    ASSERT(h_errorFlag == 0);
}

// Fill `buffer` with a uniform constant `value` (uint32) via SM kernel.
// Must be used instead of cuMemsetD32Async for multicast VAs, which the CE cannot access.
static void memsetConstantHelper(CUdeviceptr buffer, unsigned long long size,
                                  uint32_t value, std::shared_ptr<NodeHelper> nodeHelper) {
    unsigned long long num_pattern_elements = _2MiB / sizeof(unsigned int);
    unsigned int *h_pattern;
    CUdeviceptr   d_pattern;
    CU_ASSERT(cuMemHostAlloc((void **)&h_pattern, _2MiB, CU_MEMHOSTALLOC_PORTABLE));
    for (unsigned long long k = 0; k < num_pattern_elements; k++) h_pattern[k] = value;
    CU_ASSERT(cuMemAlloc(&d_pattern, _2MiB));
    CU_ASSERT(cuMemcpyAsync(d_pattern, (CUdeviceptr)h_pattern, _2MiB, CU_STREAM_PER_THREAD));
    CU_ASSERT(memsetKernel(CU_STREAM_PER_THREAD, buffer, d_pattern,
                           size / sizeof(unsigned int),
                           (unsigned int)num_pattern_elements));
    CU_ASSERT(nodeHelper->streamSynchronizeWrapper(CU_STREAM_PER_THREAD));
    CU_ASSERT(cuMemFreeHost((void *)h_pattern));
    CU_ASSERT(cuMemFree(d_pattern));
}

void memsetPatternHelper(CUstream stream, const MemcpyBuffer* memcpyBuffer, unsigned long long size, unsigned int seed, std::shared_ptr<NodeHelper> nodeHelper, bool LE = false) {
    unsigned int* h_pattern;
    CUdeviceptr d_pattern;
    CUdeviceptr buffer;
    if (LE) {
        buffer = memcpyBuffer->getDevicePtr();
    } else {
        buffer = memcpyBuffer->getBuffer();
    }

    unsigned long long num_elements = size / sizeof(unsigned int);
    unsigned long long num_pattern_elements = _2MiB / sizeof(unsigned int);

    // Allocate 2MB of pattern
    CU_ASSERT(cuMemHostAlloc((void**)&h_pattern, sizeof(char) * _2MiB, CU_MEMHOSTALLOC_PORTABLE));
    xorshift2MBPattern(h_pattern, seed);

    // Copy the pattern to a device buffer
    CU_ASSERT(cuMemAlloc(&d_pattern, sizeof(char) * _2MiB));
    CU_ASSERT(cuMemcpyAsync(d_pattern, (CUdeviceptr)h_pattern, sizeof(char) * _2MiB, stream));

    CUdeviceptr memsetBuffer = buffer;

    // When there is a staging buffer for the memcpyBuffer in bounce-buffer CC modes, memset
    // the staging buffer with the memsetKernel and then copy it to the target buffer.
    if (memcpyBuffer->getStagingBuffer()) {
        memsetBuffer = memcpyBuffer->getStagingBuffer();
    }

    // Launch the memset kernel
    CU_ASSERT(memsetKernel(stream, memsetBuffer, d_pattern, num_elements, num_pattern_elements));

    if (memcpyBuffer->getStagingBuffer()) {
        CU_ASSERT(cuMemcpyAsync(buffer, memsetBuffer, size, stream));
    }

    CU_ASSERT(nodeHelper->streamSynchronizeWrapper(stream));

    CU_ASSERT(cuMemFreeHost((void*)h_pattern));
    CU_ASSERT(cuMemFree(d_pattern));
}

static void memsetPatternHelperLE(CUstream stream, const MemcpyBuffer* memcpyBuffer, unsigned long long size, unsigned int seed, std::shared_ptr<NodeHelper> nodeHelper) {
    return memsetPatternHelper(stream, memcpyBuffer, size, seed, nodeHelper, true);
}

// True when the first source or destination buffer is backed by a unicast
// Logical Endpoint. Shared detection for the LE-unicast memset/memcmp helpers.
// Multicast LEs must not match: non-dispatched ranks have empty filtered
// src/dst lists but allSrcBuffers/allDstBuffers still carry the multicast LE,
// and getLeId()!=NONE alone would send them into barrierAllRanks() while the
// executor skips it.
static bool isUnicastLEBuffer(const MemcpyBuffer *buf) {
    return buf->getLeId() != MemcpyBuffer::LE_ID_NONE && !buf->isMulticastBuffer();
}

static bool isLEUnicastDispatch(const MemcpyDispatchInfo &info) {
    return (!info.allSrcBuffers.empty() && isUnicastLEBuffer(info.allSrcBuffers[0])) ||
           (!info.allDstBuffers.empty() && isUnicastLEBuffer(info.allDstBuffers[0]));
}

// Detects a unicast Logical Endpoint (fabric "Handle") dispatch and, if present,
// initialises the copy pattern in each buffer's own context. Shared by the CE and
// TMA initiators so both can drive Handle-backed device-to-device copies.
// Returns true when the LE-unicast path handled the memset (caller should return).
static bool memsetPatternLEUnicastIfNeeded(MemcpyDispatchInfo &info) {
    if (!isLEUnicastDispatch(info)) return false;

    // Seed LE source buffers (LE read case): only the LE owner has getDevicePtr()!=0.
    for (const auto* buf : info.allSrcBuffers) {
        if (buf->getLeId() != MemcpyBuffer::LE_ID_NONE && buf->getDevicePtr() != 0) {
            CU_ASSERT(cuCtxSetCurrent(buf->getPrimaryCtx()));
            memsetPatternHelperLE(CU_STREAM_PER_THREAD, buf,
                                  buf->getBufferSize(), 0xBAADF00D, info.nodeHelper);
        }
    }
    // Poison LE destination buffers (LE write case).
    for (const auto* buf : info.allDstBuffers) {
        if (buf->getLeId() != MemcpyBuffer::LE_ID_NONE && buf->getDevicePtr() != 0) {
            CU_ASSERT(cuCtxSetCurrent(buf->getPrimaryCtx()));
            memsetPatternHelperLE(CU_STREAM_PER_THREAD, buf,
                                  buf->getBufferSize(), 0xCAFEBABE, info.nodeHelper);
        }
    }
    // Seed raw (non-LE) source buffers (LE write case: src is a plain device buffer).
    // Use dispatch ctx/stream: src may be a green-context LocalizedBufferChunkView.
    // CU_STREAM_PER_THREAD is invalid on green contexts (device_local_copy_tma --force-le).
    for (int i = 0; i < (int)info.srcBuffers.size(); i++) {
        if (info.srcBuffers[i]->getLeId() == MemcpyBuffer::LE_ID_NONE &&
            info.srcBuffers[i]->getDevicePtr() != 0) {
            CU_ASSERT(cuCtxSetCurrent(info.contexts[i]));
            memsetPatternHelperLE(info.streams[i], info.srcBuffers[i],
                                  info.adjustedCopySizes[i], 0xBAADF00D, info.nodeHelper);
        }
    }
    // Poison raw (non-LE) destination buffers (LE read case: dst is a plain device buffer).
    for (int i = 0; i < (int)info.dstBuffers.size(); i++) {
        if (info.dstBuffers[i]->getLeId() == MemcpyBuffer::LE_ID_NONE &&
            info.dstBuffers[i]->getDevicePtr() != 0) {
            CU_ASSERT(cuCtxSetCurrent(info.contexts[i]));
            memsetPatternHelperLE(info.streams[i], info.dstBuffers[i],
                                  info.adjustedCopySizes[i], 0xCAFEBABE, info.nodeHelper);
        }
    }
    info.nodeHelper->barrierAllRanks();
    return true;
}

void memclearByWarpParity(CUstream stream, CUdeviceptr buffer, unsigned long long size, bool clearOddWarpIndexed, std::shared_ptr<NodeHelper> nodeHelper) {
    CU_ASSERT(memclearKernelByWarpParity(stream, buffer, size, clearOddWarpIndexed));
    CU_ASSERT(nodeHelper->streamSynchronizeWrapper(stream));
}

void MemcpyInitiatorCE::memsetPattern(MemcpyDispatchInfo &info) const {
    if (memsetPatternLEUnicastIfNeeded(info)) return;
    for (int i = 0; i < info.srcBuffers.size(); i++) {
        CU_ASSERT(cuCtxSetCurrent(info.contexts[i]));
        memsetPatternHelper(info.streams[i], info.dstBuffers[i], info.adjustedCopySizes[i], 0xCAFEBABE, info.nodeHelper);
        memsetPatternHelper(info.streams[i], info.srcBuffers[i], info.adjustedCopySizes[i], 0xBAADF00D, info.nodeHelper);
    }
}

void MemcpyInitiatorSM::memsetPattern(MemcpyDispatchInfo &info) const {
    for (int i = 0; i < info.srcBuffers.size(); i++) {
        CU_ASSERT(cuCtxSetCurrent(info.contexts[i]));
        memsetPatternHelper(info.streams[i], info.dstBuffers[i], info.adjustedCopySizes[i], 0xCAFEBABE, info.nodeHelper);
        memsetPatternHelper(info.streams[i], info.srcBuffers[i], info.adjustedCopySizes[i], 0xBAADF00D, info.nodeHelper);
    }
}

void MemcpyInitiatorMultimemWrite::memsetPattern(MemcpyDispatchInfo &info) const {
    for (int i = 0; i < info.srcBuffers.size(); i++) {
        CU_ASSERT(cuCtxSetCurrent(info.contexts[i]));
        memsetPatternHelper(info.streams[i], info.dstBuffers[i], info.adjustedCopySizes[i], 0xCAFEBABE, info.nodeHelper);
        memsetPatternHelper(info.streams[i], info.srcBuffers[i], info.adjustedCopySizes[i], 0xBAADF00D, info.nodeHelper);
    }
}

void MemcpyInitiatorSMSplitWarp::memsetPattern(MemcpyDispatchInfo &info) const {
    for (int i = 0; i < info.srcBuffers.size(); i++) {
        CU_ASSERT(cuCtxSetCurrent(info.contexts[i]));
        memsetPatternHelper(info.streams[i], info.dstBuffers[i], info.adjustedCopySizes[i], 0xBAADF00D, info.nodeHelper);
        memsetPatternHelper(info.streams[i], info.srcBuffers[i], info.adjustedCopySizes[i], 0xBAADF00D, info.nodeHelper);
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

unsigned long long MemcpyInitiatorMultimemWrite::getAdjustedBandwidth(unsigned long long bandwidth) {
    return bandwidth;
}

unsigned long long MemcpyInitiatorSMSplitWarp::getAdjustedBandwidth(unsigned long long bandwidth) {
    // For split warp copies, we estimate bandwidth in each direction as 1/2 of measured bandwidth
    return bandwidth / 2;
}

// Add this new typedef for the comparison function pointer
typedef CUresult (*CompareKernelFunc)(CUstream, CUdeviceptr, CUdeviceptr, unsigned long long, unsigned int, CUdeviceptr);

constexpr CUdeviceptr MEMCMP_NO_STAGING = 0;

void memcmpPatternHelper(CUstream stream, CUdeviceptr buffer, CUdeviceptr stagingBuffer, unsigned long long size, unsigned int seed, CompareKernelFunc compareKernel, std::shared_ptr<NodeHelper> nodeHelper, uint32_t patternScale = 1) {
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
        // fabric.try_pullred.add.u64 sums whole 64-bit words across backing memories.
        // Scale the expected pattern at u64 granularity so low/high halves carry correctly.
        unsigned long long num_u64_elements = num_pattern_elements / 2;
        const uint64_t scale_u64 = static_cast<uint64_t>(patternScale);
        for (unsigned long long k = 0; k < num_u64_elements; k++) {
            uint64_t word = 0;
            std::memcpy(&word, &h_pattern[2 * k], sizeof(word));
            word *= scale_u64;
            std::memcpy(&h_pattern[2 * k], &word, sizeof(word));
        }
    }
    CU_ASSERT(cuMemAlloc(&d_pattern, sizeof(char) * _2MiB));
    CU_ASSERT(cuMemcpyAsync(d_pattern, (CUdeviceptr)h_pattern, sizeof(char) * _2MiB, stream));

    // setup error flags
    CU_ASSERT(cuMemAlloc(&d_errorFlag, sizeof(int)));
    CU_ASSERT(cuMemcpyAsync(d_errorFlag, (CUdeviceptr)&h_errorFlag, sizeof(int), stream));

    CUdeviceptr memcmpBuffer = buffer;

    // When there is a staging buffer for the memcpyBuffer in bounce-buffer CC modes, copy
    // the buffer to the staging buffer and then validate the data there with the compareKernel.
    if (stagingBuffer) {
        memcmpBuffer = stagingBuffer;

        CU_ASSERT(cuMemcpyAsync(memcmpBuffer, buffer, size, stream));
    }

    // launch kernel to compare
    CU_ASSERT(compareKernel(stream, memcmpBuffer, d_pattern, num_elements, num_pattern_elements, d_errorFlag));
    CU_ASSERT(nodeHelper->streamSynchronizeWrapper(stream));
    CU_ASSERT(cuMemcpyDtoH(&h_errorFlag, d_errorFlag, sizeof(int)));

    CU_ASSERT(cuMemFreeHost((void*)h_pattern));
    CU_ASSERT(cuMemFree(d_errorFlag));
    CU_ASSERT(cuMemFree(d_pattern));

    ASSERT(h_errorFlag == 0);
}

void memcmpBufferPatternHelper(CUstream stream, const MemcpyBuffer *memcpyBuffer, unsigned long long size, unsigned int seed, CompareKernelFunc compareKernel, std::shared_ptr<NodeHelper> nodeHelper) {
    return memcmpPatternHelper(stream, memcpyBuffer->getBuffer(), memcpyBuffer->getStagingBuffer(), size, seed, compareKernel, nodeHelper);
}

void memsetLocalizedConcurrentOwnedBuffer(const MemcpyBuffer& buf, size_t size,
                                          unsigned int seed) {
    auto nodeHelper = std::make_shared<NodeHelperSingle>();
    CUcontext prevCtx = nullptr;
    CU_ASSERT(cuCtxGetCurrent(&prevCtx));
    CU_ASSERT(cuCtxSetCurrent(buf.getPrimaryCtx()));
    memsetPatternHelper(CU_STREAM_PER_THREAD, &buf, size, seed, nodeHelper);
    CU_ASSERT(cuCtxSetCurrent(prevCtx));
}

void memcmpLocalizedConcurrentOwnedBuffer(const MemcpyBuffer& buf, size_t size,
                                          unsigned int seed) {
    auto nodeHelper = std::make_shared<NodeHelperSingle>();
    CUcontext prevCtx = nullptr;
    CU_ASSERT(cuCtxGetCurrent(&prevCtx));
    CU_ASSERT(cuCtxSetCurrent(buf.getPrimaryCtx()));
    memcmpBufferPatternHelper(CU_STREAM_PER_THREAD, &buf, size, seed, memcmpKernel, nodeHelper);
    CU_ASSERT(cuCtxSetCurrent(prevCtx));
}

void prepareLocalizedConcurrentCeCopy(const MemcpyBuffer& memcpySrc,
                                      const MemcpyBuffer& memcpyDst,
                                      size_t copySize) {
    if (gSettings.skipVerification) return;
    auto nodeHelper = std::make_shared<NodeHelperSingle>();
    CUcontext prevCtx = nullptr;
    CU_ASSERT(cuCtxGetCurrent(&prevCtx));
    CU_ASSERT(cuCtxSetCurrent(memcpyDst.getPrimaryCtx()));
    memsetPatternHelper(CU_STREAM_PER_THREAD, &memcpyDst, copySize, 0xCAFEBABE, nodeHelper);
    CU_ASSERT(cuCtxSetCurrent(memcpySrc.getPrimaryCtx()));
    memsetPatternHelper(CU_STREAM_PER_THREAD, &memcpySrc, copySize, 0xBAADF00D, nodeHelper);
    CU_ASSERT(cuCtxSetCurrent(prevCtx));
}

void verifyLocalizedConcurrentCeCopy(const MemcpyBuffer& memcpySrc,
                                     const MemcpyBuffer& memcpyDst,
                                     size_t copySize) {
    (void)memcpySrc;
    if (gSettings.skipVerification) return;
    auto nodeHelper = std::make_shared<NodeHelperSingle>();
    CUcontext prevCtx = nullptr;
    CU_ASSERT(cuCtxGetCurrent(&prevCtx));
    CU_ASSERT(cuCtxSetCurrent(memcpyDst.getPrimaryCtx()));
    memcmpBufferPatternHelper(CU_STREAM_PER_THREAD, &memcpyDst, copySize, 0xBAADF00D,
                              memcmpKernel, nodeHelper);
    CU_ASSERT(cuCtxSetCurrent(prevCtx));
}

// Verifies a unicast Logical Endpoint ("Handle") dispatch by comparing the pattern
// in each destination buffer's own context. Shared by the CE and TMA initiators.
// Returns true when the LE-unicast path handled the verification (caller should return).
static bool memcmpPatternLEUnicastIfNeeded(MemcpyDispatchInfo &info) {
    if (!isLEUnicastDispatch(info)) return false;

    // Data always lands in the destination buffer; verify only the copied prefix
    // using its native ctx. TMA LE-unicast truncates to 16-byte granularity and
    // stores the real length in adjustedCopySizes after memcpyFunc — comparing
    // getBufferSize() would read poisoned tail bytes that were never written.
    for (const auto* buf : info.allDstBuffers) {
        if (buf->getDevicePtr() == 0) continue;

        size_t verifySize = (buf->getBufferSize() / 16u) * 16u;
        for (int j = 0; j < (int)info.dstBuffers.size(); j++) {
            if (info.dstBuffers[j] == buf) {
                verifySize = info.adjustedCopySizes[j];
                break;
            }
        }

        // LE devicePtr is owned on the LE primary context (not the green
        // executing context of a LocalizedBufferChunkView peer).
        CU_ASSERT(cuCtxSetCurrent(buf->getPrimaryCtx()));
        memcmpPatternHelper(CU_STREAM_PER_THREAD, buf->getDevicePtr(), MEMCMP_NO_STAGING,
                            verifySize, 0xBAADF00D, memcmpKernel,
                            info.nodeHelper);
    }
    return true;
}

void MemcpyInitiatorCE::memcmpPattern(MemcpyDispatchInfo &info) const {
    if (memcmpPatternLEUnicastIfNeeded(info)) return;
    for (int i = 0; i < info.srcBuffers.size(); i++) {
        CU_ASSERT(cuCtxSetCurrent(info.contexts[i]));
        memcmpBufferPatternHelper(info.streams[i], info.dstBuffers[i], info.adjustedCopySizes[i], 0xBAADF00D, memcmpKernel, info.nodeHelper);
    }
}

void MemcpyInitiatorSM::memcmpPattern(MemcpyDispatchInfo &info) const {
    for (int i = 0; i < info.srcBuffers.size(); i++) {
        CU_ASSERT(cuCtxSetCurrent(info.contexts[i]));
        memcmpBufferPatternHelper(info.streams[i], info.dstBuffers[i], info.adjustedCopySizes[i], 0xBAADF00D, memcmpKernel, info.nodeHelper);
    }
}

void MemcpyInitiatorMultimemWrite::memcmpPattern(MemcpyDispatchInfo &info) const {
    for (int i = 0; i < info.srcBuffers.size(); i++) {
        CU_ASSERT(cuCtxSetCurrent(info.contexts[i]));
        memcmpBufferPatternHelper(info.streams[i], info.dstBuffers[i], info.adjustedCopySizes[i], 0xBAADF00D, multicastMemcmpKernel, info.nodeHelper);
    }
}


void MemcpyInitiatorSMSplitWarp::memcmpPattern(MemcpyDispatchInfo &info) const {
    for (int i = 0; i < info.srcBuffers.size(); i++) {
        CU_ASSERT(cuCtxSetCurrent(info.contexts[i]));
        // src and dst buffer contents must match after the bidirectional split warp copy
        memcmpBufferPatternHelper(info.streams[i], info.dstBuffers[i], info.adjustedCopySizes[i], 0xBAADF00D, memcmpKernel, info.nodeHelper);
        memcmpBufferPatternHelper(info.streams[i], info.srcBuffers[i], info.adjustedCopySizes[i], 0xBAADF00D, memcmpKernel, info.nodeHelper);
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

    if (gSettings.bounceBufferConfComputeEnabled) {
        CU_ASSERT(cuMemAlloc((CUdeviceptr*)&stagingBuffer, bufferSize));
    }
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

    if (gSettings.bounceBufferConfComputeEnabled) {
        CU_ASSERT(cuMemFree(stagingBuffer));
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

// Enable peer access by device ID. Used when buffers don't exist yet (e.g. device-to-device tests
// before LE allocation, or in filter()). DeviceBuffer::enablePeerAccess uses this internally.
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

bool MemcpyBuffer::enablePeerAccess(const MemcpyBuffer &peerBuffer) const {
    if (!supportsPeerAccess() || !peerBuffer.supportsPeerAccess()) {
        return false;
    }
    return enablePeerAccessBetweenDevices(getBufferIdx(), peerBuffer.getBufferIdx());
}

bool DeviceBuffer::enablePeerAccess(const MemcpyBuffer &peerBuffer) const {
    if (!peerBuffer.supportsPeerAccess()) return false;
    return enablePeerAccessBetweenDevices(getBufferIdx(), peerBuffer.getBufferIdx());
}


MemcpyDescriptor::MemcpyDescriptor(CUdeviceptr dst, CUdeviceptr src, CUstream stream, size_t copySize,
                                   unsigned long long loopCount, uint32_t srcLeId, uint32_t dstLeId) :
    dst(dst), src(src), stream(stream), copySize(copySize), loopCount(loopCount),
    srcLeId(srcLeId), dstLeId(dstLeId) {
}

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
    // For LE transfers the fabric PTX instructions are context-agnostic, but the plain
    // DeviceBuffer must be directly accessible from the running context.  Always run on
    // the non-LE side so the kernel can read/write the DeviceBuffer locally.
    if (src->getLeId() != MemcpyBuffer::LE_ID_NONE) return dst;  // src is LE: kernel writes locally to dst
    if (dst->getLeId() != MemcpyBuffer::LE_ID_NONE) return src;  // dst is LE: kernel reads locally from src
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
    std::vector<CUevent> warmupDoneEvents(info.srcBuffers.size());
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
#ifdef NVBW_HAVE_UGPU_MEMORY_NODE_SPLIT
        const MemcpyBuffer* exec = NodeHelper::selectExecutingBuffer(
            info.srcBuffers[i], info.dstBuffers[i], ctxPreference);
        CUgreenCtx greenCtx = exec ? exec->getGreenCtx() : nullptr;
        if (greenCtx) {
            CU_ASSERT(cuGreenCtxStreamCreate(&streams[i], greenCtx, CU_STREAM_NON_BLOCKING, 0));
        } else {
            CU_ASSERT(cuStreamCreate(&streams[i], CU_STREAM_NON_BLOCKING));
        }
#else
        CU_ASSERT(cuStreamCreate(&streams[i], CU_STREAM_NON_BLOCKING));
#endif
        info.streams.push_back(streams[i]);
        CU_ASSERT(cuEventCreate(&warmupDoneEvents[i], CU_EVENT_DEFAULT));
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
    memcpyInitiator->prepareResources(info);

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
                                  info.streams[i], info.srcBuffers[i]->getBufferSize(), kWarmupCount,
                                  info.srcBuffers[i]->getLeId(), info.dstBuffers[i]->getLeId());
            desc.useNvlinkUtilScheduling = useNvlinkUtilScheduling_;
            memcpyInitiator->memcpyFunc(desc, nullptr);
            CU_ASSERT(cuEventRecord(warmupDoneEvents[i], info.streams[i]));
        }

        if (info.srcBuffers.size() > 0) {
            CU_ASSERT(cuCtxSetCurrent(info.contexts[0]));
            if (bandwidthValue == BandwidthValue::TOTAL_BW) {
                // Ensure all streams have completed warmup before the global timing starts.
                for (int i = 1; i < info.srcBuffers.size(); i++) {
                    CU_ASSERT(cuStreamWaitEvent(info.streams[0], warmupDoneEvents[i], 0));
                }
            }
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
                                  info.streams[i], copySize, loopCount,
                                  info.srcBuffers[i]->getLeId(), info.dstBuffers[i]->getLeId());
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

        memcpyInitiator->validateResult(info);
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
            bandwidth = static_cast<unsigned long long>(bandwidth * bandwidthScale_);

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

            double bandwidth = nodeHelper->calculateTotalBandwidth(elapsedTotalInUs, totalSize, loopCount) * bandwidthScale_;
            totalBandwidth(bandwidth);
            VERBOSE << "\tSample " << n << ": Total Bandwidth : " <<
                std::fixed << std::setprecision(2) << (double)bandwidth * 1e-9 << " GB/s\n";
        }
    }

    // cleanup
    CU_ASSERT(cuEventDestroy(totalEnd));
    memcpyInitiator->releaseResources(info);

    for (int i = 0; i < info.srcBuffers.size(); i++) {
        CU_ASSERT(cuStreamDestroy(info.streams[i]));
        CU_ASSERT(cuEventDestroy(warmupDoneEvents[i]));
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

    for (int i = 0; i < info.dstBuffers.size(); i++) {
        CU_ASSERT(cuCtxSetCurrent(info.contexts[i]));

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
    ASSERT(srcSize == dstSize);
    int numSm = smCountForStream(stream);
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
    (void)info;
    // CE copies raw pointers only. A Handle (LE) operand would need
    // cuFabricCpyBatchAsync, which is not used here, and handing a peer fabric VA
    // to cuMemcpyAsync is not a supported substitute. Handle buffers are reserved
    // for the TMA initiator, so CE should never be dispatched against one.
    ASSERT(desc.srcLeId == MemcpyBuffer::LE_ID_NONE &&
           desc.dstLeId == MemcpyBuffer::LE_ID_NONE &&
           "CE copies do not support Handle (LE) buffers");
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


size_t MemcpyInitiatorMultimemWrite::memcpyFunc(MemcpyDescriptor &desc, MemcpyDispatchInfo* info) {
    return multicastCopy(desc.dst, desc.src, desc.copySize, desc.stream, desc.loopCount);
}

size_t MemcpyInitiatorMultimemWrite::getAdjustedCopySize(size_t srcSize, size_t dstSize, CUstream stream) {
    ASSERT(srcSize == dstSize);
    ASSERT(srcSize % 16 == 0);  // multimem_st_128 requires 16-byte granularity
    return srcSize;
}

// ============================================================================
// MemcpyInitiatorMultimemReadV4F32
// ============================================================================

size_t MemcpyInitiatorMultimemReadV4F32::memcpyFunc(MemcpyDescriptor &desc, MemcpyDispatchInfo*) {
    launchMultimemReadKernelV4F32(desc.src, desc.dst, desc.copySize, desc.loopCount, desc.stream);
    return desc.copySize;
}

size_t MemcpyInitiatorMultimemReadV4F32::getAdjustedCopySize(size_t srcSize, size_t dstSize, CUstream) {
    ASSERT(srcSize == dstSize);
    ASSERT(srcSize % 16 == 0);
    return srcSize;
}

unsigned long long MemcpyInitiatorMultimemReadV4F32::getAdjustedBandwidth(unsigned long long bw) {
    return bw;
}

void MemcpyInitiatorMultimemReadV4F32::memsetPattern(MemcpyDispatchInfo &info) const {
    // Fill each rank's backing memory with 1.0f (0x3F800000) per float.
    // Must use SM helper — CE cannot access multicast VAs.
    for (int i = 0; i < (int)info.srcBuffers.size(); i++) {
        CU_ASSERT(cuCtxSetCurrent(info.contexts[i]));
        memsetConstantHelper(info.srcBuffers[i]->getBuffer(),
                             info.adjustedCopySizes[i], 0x3F800000u, info.nodeHelper);
    }
}

void MemcpyInitiatorMultimemReadV4F32::memcmpPattern(MemcpyDispatchInfo &info) const {
    // src was filled with 1.0f per float; after add.v4.f32 reduction over numBacking GPUs,
    // each float = numBacking × 1.0f = float(numBacking), exact in fp32 for numBacking ≤ 2^24.
    for (int i = 0; i < (int)info.dstBuffers.size(); i++) {
        CU_ASSERT(cuCtxSetCurrent(info.contexts[i]));
        uint32_t numBacking = info.srcBuffers[i]->getNumBackingMemories();
        float expected_f = (float)numBacking;
        uint32_t expected_bits;
        memcpy(&expected_bits, &expected_f, sizeof(uint32_t));
        unsigned long long num_elements = info.adjustedCopySizes[i] / sizeof(unsigned int);
        memcmpConstantHelper(info.dstBuffers[i]->getBuffer(), num_elements,
                             expected_bits, info.nodeHelper);
    }
}

size_t MemcpyInitiatorSMSplitWarp::memcpyFunc(MemcpyDescriptor &desc, MemcpyDispatchInfo* info) {
    return copyKernelSplitWarp(desc);
}

size_t MemcpyInitiatorTMA::memcpyFunc(MemcpyDescriptor &desc, MemcpyDispatchInfo* info) {
    // Warmup passes info=nullptr; skip LE/multicast paths to avoid using peer VAs
    // that are inaccessible in the current context (would cause sticky cudaErrorIllegalAddress).
    if (gSettings.bufferType == BufferType::LE && !info) {
        return 0;
    }
    if (desc.copySize == 0) return 0;

    if (info && !info->srcBuffers.empty() && !info->dstBuffers.empty()) {
        bool dstIsMulticast = info->dstBuffers[0]->isMulticastBuffer();
        bool srcIsMulticast = info->srcBuffers[0]->isMulticastBuffer();

        if (desc.dstLeId != MemcpyBuffer::LE_ID_NONE && dstIsMulticast) {
            // Multicast LE: fabric.try_put.multimem (PTX spec requires .multimem for multicast LEs)
            size_t copySize = (desc.copySize / 16u) * 16u;
            leLaunchWriteMulticastKernel(desc.src, desc.dstLeId, copySize, desc.loopCount, desc.stream);
            return copySize;
        } else if (desc.srcLeId != MemcpyBuffer::LE_ID_NONE && srcIsMulticast) {
            // Multicast LE read: fabric.try_pullred (add.u64 reduction, sm_100+)
            ASSERT(desc.copySize % 16 == 0);  // fabric.try_pullred requires 16-byte granularity
            size_t copySize = desc.copySize;
            if (copySize == 0) return 0;
            leLaunchReduceMulticastKernel(desc.srcLeId, desc.dst, copySize, desc.loopCount, desc.stream);
            return copySize;
        } else if (srcIsMulticast && desc.srcLeId == MemcpyBuffer::LE_ID_NONE) {
            ASSERT(desc.copySize % 16 == 0);
            launchMultimemReadKernelV4F32(desc.src, desc.dst, desc.copySize, desc.loopCount, desc.stream);
            return desc.copySize;
        } else if (desc.srcLeId != MemcpyBuffer::LE_ID_NONE || desc.dstLeId != MemcpyBuffer::LE_ID_NONE) {
            // Unicast LE: direction is encoded in which slot carries the LE ID.
            // LEBufferStrategy puts the LE on the peer slot; desc.src/dst carry the
            // corresponding device pointers set by doMemcpyCore from getBuffer().
            size_t copySize = (desc.copySize / 16u) * 16u;
            if (desc.srcLeId != MemcpyBuffer::LE_ID_NONE) {
                leLaunchReadKernel(desc.srcLeId, desc.dst, copySize, desc.loopCount, desc.stream);
            } else {
                leLaunchWriteKernel(desc.src, desc.dstLeId, copySize, desc.loopCount, desc.stream);
            }
            return copySize;
        } else if (dstIsMulticast) {
            // Raw multicast VA: TMA G->S then multimem_st_128 (16B/iter)
            ASSERT(desc.copySize % 16 == 0);  // multimem_st_128 requires 16-byte granularity
            launchMultimemWriteKernel(desc.src, desc.dst, desc.copySize, desc.loopCount, desc.stream);
            return desc.copySize;
        }
    }
    return copyKernelTMA(desc);
}

size_t MemcpyInitiatorTMA::getAdjustedCopySize(size_t srcSize, size_t dstSize, CUstream stream) {
    ASSERT(srcSize == dstSize);
    return srcSize;
}

void MemcpyInitiatorTMA::memsetPattern(MemcpyDispatchInfo &info) const {
    // Multicast read (src is multicast VA or LE): set up src pattern then poison dst.
    if (!info.srcBuffers.empty() && info.srcBuffers[0]->isMulticastBuffer()) {
        if (info.srcBuffers[0]->getLeId() == MemcpyBuffer::LE_ID_NONE) {
            // Raw multicast VA — delegate to V4F32 initiator which fills each backing
            // memory with 1.0f via SM stores (CE cannot access multicast VAs).
            MemcpyInitiatorMultimemReadV4F32 mcReadInitiator;
            mcReadInitiator.memsetPattern(info);
        } else {
            // LE-backed multicast src: push constant 1u into the LE via
            // fabric.try_put.multimem so every backing memory holds 1u per
            // uint32_t word.  After fabric.try_pullred.add.u64 over N GPUs,
            // each 32-bit half of every output uint64_t = N = numBacking
            // (no carry: 8×1 = 8 << 2^32).  Using a constant avoids xorshift
            // pattern issues with u64 carry across 32-bit boundaries.
            uint32_t leId = info.srcBuffers[0]->getLeId();
            for (int i = 0; i < (int)info.srcBuffers.size(); i++) {
                CU_ASSERT(cuCtxSetCurrent(info.contexts[i]));
                size_t copySize = info.adjustedCopySizes[i];
                CUdeviceptr tempSrc = 0;
                CU_ASSERT(cuMemAlloc(&tempSrc, copySize));
                CU_ASSERT(cuMemsetD32Async(tempSrc, 1u, copySize / sizeof(uint32_t), info.streams[i]));
                leLaunchWriteMulticastKernel(tempSrc, leId, copySize, 1, info.streams[i]);
                CU_ASSERT(cuStreamSynchronize(info.streams[i]));
                CU_ASSERT(cuMemFree(tempSrc));
            }
            for (int i = 0; i < (int)info.dstBuffers.size(); i++) {
                CU_ASSERT(cuCtxSetCurrent(info.contexts[i]));
                memsetPatternHelper(info.streams[i], info.dstBuffers[i],
                                    info.adjustedCopySizes[i], 0xCAFEBABE, info.nodeHelper);
            }

            // doMemcpyCore calls synchronizeProcess() after memset.
        }
        return;
    }
    if (!info.dstBuffers.empty() && info.dstBuffers[0]->isMulticastBuffer()) {
        if (info.dstBuffers[0]->getLeId() == MemcpyBuffer::LE_ID_NONE) {
            MemcpyInitiatorMultimemWrite mcInitiator;
            mcInitiator.memsetPattern(info);
        } else {
            // LE-backed multicast: dst getDevicePtr()==0 so we cannot memset it directly.
            // Initialise only src with the standard seeded pattern so it can be verified
            // after reading back from the LE in memcmpPattern.
            for (int i = 0; i < (int)info.srcBuffers.size(); i++) {
                CU_ASSERT(cuCtxSetCurrent(info.contexts[i]));
                memsetPatternHelper(info.streams[i], info.srcBuffers[i], info.adjustedCopySizes[i], 0xBAADF00D, info.nodeHelper);
            }
        }
        return;
    }
    // LE read (src is LE) or LE write (dst is LE, unicast): switch to each buffer's own
    // device context before calling memsetPatternHelper. memsetPatternHelper always uses
    // CU_STREAM_PER_THREAD internally, so the context must be set correctly — falling
    // through to MemcpyInitiatorSM would run the kernel in the source context trying to
    // access the destination LE buffer, causing CUDA_ERROR_ILLEGAL_ADDRESS.
    //
    // Use unfiltered lists so every rank sees the LE context.
    // In multinode mode the dispatched rank is filtered by dst/src context preference, so
    // a non-dispatched rank sees empty info.dstBuffers/srcBuffers and can't detect the LE.
    // allSrcBuffers/allDstBuffers are always populated for all ranks.
    if (memsetPatternLEUnicastIfNeeded(info)) return;
    MemcpyInitiatorSM smInitiator;
    smInitiator.memsetPattern(info);
}

void MemcpyInitiatorTMA::memcmpPattern(MemcpyDispatchInfo &info) const {
    // Multicast read: src is multicast, dst holds the reduction result (numBacking × pattern).
    if (!info.srcBuffers.empty() && info.srcBuffers[0]->isMulticastBuffer()) {
        if (info.srcBuffers[0]->getLeId() == MemcpyBuffer::LE_ID_NONE) {
            MemcpyInitiatorMultimemReadV4F32 mcReadInitiator;
            mcReadInitiator.memcmpPattern(info);
        } else {
            // Src was filled with constant 1u per uint32_t; after add.u64 pullred
            // over numBacking GPUs each uint32_t half = numBacking (no carry).
            uint32_t numBacking = info.srcBuffers[0]->getNumBackingMemories();
            for (int i = 0; i < (int)info.dstBuffers.size(); i++) {
                CU_ASSERT(cuCtxSetCurrent(info.contexts[i]));
                unsigned long long num_elements = info.adjustedCopySizes[i] / sizeof(unsigned int);
                memcmpConstantHelper(info.dstBuffers[i]->getBuffer(), num_elements,
                                     numBacking, info.nodeHelper);
            }
        }
        return;
    }
    if (!info.dstBuffers.empty() && info.dstBuffers[0]->isMulticastBuffer()) {
        if (info.dstBuffers[0]->getLeId() == MemcpyBuffer::LE_ID_NONE) {
            MemcpyInitiatorMultimemWrite mcInitiator;
            mcInitiator.memcmpPattern(info);
        } else {
            // LE-backed multicast write: read back via fabric.try_pullred (add.u64 reduction).
            // fabric.try_get is invalid on multicast LEs; pullred is the correct instruction
            // for multicast LE readback (sm_100+).
            // pullred sums N backing memories: result[k] = N * pattern[k], so scale expected by N.
            uint32_t leId = info.dstBuffers[0]->getLeId();
            uint32_t numBacking = info.dstBuffers[0]->getNumBackingMemories();
            for (int i = 0; i < (int)info.srcBuffers.size(); i++) {
                CU_ASSERT(cuCtxSetCurrent(info.contexts[i]));
                size_t copySize = info.adjustedCopySizes[i];
                CUdeviceptr tempBuf = 0;
                CU_ASSERT(cuMemAlloc(&tempBuf, copySize));
                leLaunchReduceMulticastKernel(leId, tempBuf, copySize, 1, CU_STREAM_PER_THREAD);
                memcmpPatternHelper(CU_STREAM_PER_THREAD, tempBuf, MEMCMP_NO_STAGING, copySize, 0xBAADF00D, memcmpKernel, info.nodeHelper, numBacking);
                CU_ASSERT(cuMemFree(tempBuf));
            }
        }
        return;
    }
    // Unicast LE (read or write): verify in each destination buffer's own context.
    //   LE read  (src=LE, dst=DevBuf): kernel ran in dst's context, data landed in dst.
    //   LE write (src=DevBuf, dst=LE): kernel ran in src's context, data landed in LE owner.
    // In both cases data resides in the destination buffer; verify there using its native ctx.
    // Use allDstBuffers so non-dispatched ranks also verify LE buffers they own.
    if (memcmpPatternLEUnicastIfNeeded(info)) return;
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

#ifdef NVBW_HAVE_UGPU_MEMORY_NODE_SPLIT

// Map a locality-domain VMM allocation at `base` and grant RW access to all
// devices. Hard-fails only if setting access for owningDeviceIdx fails.
static void mapLocalizedAllocationWithDeviceAccess(
    CUdeviceptr base, size_t allocSize, const CUmemAllocationProp& prop,
    int owningDeviceIdx) {
    CUmemGenericAllocationHandle handle = 0;
    CU_ASSERT(cuMemCreate(&handle, allocSize, &prop, 0));
    CU_ASSERT(cuMemMap(base, allocSize, 0, handle, 0));

    int nDevices = 0;
    CU_ASSERT(cuDeviceGetCount(&nDevices));
    for (int d = 0; d < nDevices; d++) {
        CUmemAccessDesc accessDesc = {};
        accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        accessDesc.location.id = d;
        accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        CUresult rc = cuMemSetAccess(base, allocSize, &accessDesc, 1);
        if (rc != CUDA_SUCCESS && d == owningDeviceIdx) {
            CU_ASSERT(rc);
        }
    }
    CU_ASSERT(cuMemRelease(handle));
}

LocalizedDeviceBuffer::LocalizedDeviceBuffer(
    size_t bufferSize, int deviceIdx, unsigned int memoryNodeId)
    : MemcpyBuffer(bufferSize), deviceIdx_(deviceIdx), memoryNodeId_(memoryNodeId) {
    CUdevice cuDev = static_cast<CUdevice>(deviceIdx_);

    CUcontext callerCtx = nullptr;
    cuCtxGetCurrent(&callerCtx);

    CU_ASSERT(cuDevicePrimaryCtxRetain(&primaryCtx_, cuDev));
    CU_ASSERT(cuCtxSetCurrent(primaryCtx_));

    // Public VMM allocation pinned to the requested uGPU locality domain. The
    // driver routes localized P2P CE copies of this memory to the per-uGPU LCE.
    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE_LOCALITY_DOMAIN;
    prop.location.localized.deviceId = static_cast<unsigned char>(deviceIdx_);
    prop.location.localized.localityDomainId = static_cast<unsigned char>(memoryNodeId_);

    // Use RECOMMENDED granularity to match the rest of the VMM allocation paths
    // (see multinode_memcpy.cpp); the buffer size is rounded up to it below.
    size_t granularity = 0;
    CU_ASSERT(cuMemGetAllocationGranularity(
        &granularity, &prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
    const size_t allocSize =
        ((bufferSize + granularity - 1) / granularity) * granularity;

    CUdeviceptr base = 0;
    CU_ASSERT(cuMemAddressReserve(&base, allocSize, 0, 0, 0));
    mapLocalizedAllocationWithDeviceAccess(base, allocSize, prop, deviceIdx_);
    vmmAllocSize_ = allocSize;
    buffer = reinterpret_cast<void*>(base);

    cuCtxSetCurrent(callerCtx);
}

LocalizedDeviceBuffer::~LocalizedDeviceBuffer() {
    CUcontext callerCtx = nullptr;
    cuCtxGetCurrent(&callerCtx);
    CU_ASSERT(cuCtxSetCurrent(primaryCtx_));

    const CUdeviceptr ptr = reinterpret_cast<CUdeviceptr>(buffer);
    CU_ASSERT(cuMemUnmap(ptr, vmmAllocSize_));
    CU_ASSERT(cuMemAddressFree(ptr, vmmAllocSize_));

    CU_ASSERT(cuDevicePrimaryCtxRelease(static_cast<CUdevice>(deviceIdx_)));
    cuCtxSetCurrent(callerCtx);
}

bool LocalizedDeviceBuffer::enablePeerAccess(const MemcpyBuffer& peerBuffer) const {
    if (!peerBuffer.supportsPeerAccess()) return false;
    return enablePeerAccessBetweenDevices(deviceIdx_, peerBuffer.getBufferIdx());
}

SplitLocalizedDeviceBuffer::SplitLocalizedDeviceBuffer(
    size_t totalSize, int deviceIdx, unsigned int numDomains)
    : deviceIdx_(deviceIdx), numDomains_(numDomains) {
    CUdevice cuDev = static_cast<CUdevice>(deviceIdx_);

    CUcontext callerCtx = nullptr;
    cuCtxGetCurrent(&callerCtx);

    CU_ASSERT(cuDevicePrimaryCtxRetain(&primaryCtx_, cuDev));
    CU_ASSERT(cuCtxSetCurrent(primaryCtx_));

    // Caller (DeviceLocalCopyTransfer::run()) guarantees totalSize % numDomains == 0.
    logicalChunkSize_ = totalSize / numDomains_;

    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE_LOCALITY_DOMAIN;
    prop.location.localized.deviceId = static_cast<unsigned char>(deviceIdx_);
    prop.location.localized.localityDomainId = 0;  // granularity doesn't vary by domain id

    // Use RECOMMENDED granularity to match LocalizedDeviceBuffer/multinode_memcpy.cpp;
    // each chunk's allocation is rounded up to it below.
    size_t granularity = 0;
    CU_ASSERT(cuMemGetAllocationGranularity(
        &granularity, &prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
    chunkAllocSize_ = ((logicalChunkSize_ + granularity - 1) / granularity) * granularity;

    // One VA reservation spanning all domains' chunks (the "stitched VA" pattern).
    CU_ASSERT(cuMemAddressReserve(&base_, chunkAllocSize_ * numDomains_, 0, 0, 0));

    for (unsigned int i = 0; i < numDomains_; i++) {
        CUmemAllocationProp chunkProp = prop;
        chunkProp.location.localized.localityDomainId = static_cast<unsigned char>(i);
        mapLocalizedAllocationWithDeviceAccess(chunkPtr(i), chunkAllocSize_, chunkProp, deviceIdx_);
    }

    cuCtxSetCurrent(callerCtx);
}

SplitLocalizedDeviceBuffer::~SplitLocalizedDeviceBuffer() {
    CUcontext callerCtx = nullptr;
    cuCtxGetCurrent(&callerCtx);
    CU_ASSERT(cuCtxSetCurrent(primaryCtx_));

    // Unmap each domain's chunk individually, then free the single stitched VA
    // reservation that spans all of them.
    for (unsigned int i = 0; i < numDomains_; i++) {
        CU_ASSERT(cuMemUnmap(chunkPtr(i), chunkAllocSize_));
    }
    CU_ASSERT(cuMemAddressFree(base_, chunkAllocSize_ * numDomains_));

    CU_ASSERT(cuDevicePrimaryCtxRelease(static_cast<CUdevice>(deviceIdx_)));
    cuCtxSetCurrent(callerCtx);
}

LocalizedGreenContexts::LocalizedGreenContexts(int deviceIdx, unsigned int numDomains)
    : numDomains_(numDomains) {
    CUdevice cuDev = static_cast<CUdevice>(deviceIdx);

    CUdevResource smResource;
    CU_ASSERT(cuDeviceGetDevResource(cuDev, &smResource, CU_DEV_RESOURCE_TYPE_SM));

    std::vector<CU_DEV_SM_RESOURCE_GROUP_PARAMS> params(numDomains_);
    for (unsigned int i = 0; i < numDomains_; i++) {
        params[i] = {};
        params[i].flags = CU_DEV_SM_RESOURCE_GROUP_LOCALITY_DOMAIN_ID;
        params[i].localityDomainId = i;
    }

    std::vector<CUdevResource> localizedSms(numDomains_);
    CUdevResource smRemainder;
    CU_ASSERT(cuDevSmResourceSplit(localizedSms.data(), numDomains_, &smResource, &smRemainder, 0, params.data()));

    // Prefer non-overlapping workqueues across these green contexts so
    // concurrent per-domain SM copies do not serialize on a shared queue.
    CUcontext primaryCtx = nullptr;
    CU_ASSERT(cuDevicePrimaryCtxRetain(&primaryCtx, cuDev));
    CUdevResource primaryWqConfig = {};
    CU_ASSERT(cuCtxGetDevResource(primaryCtx, &primaryWqConfig, CU_DEV_RESOURCE_TYPE_WORKQUEUE_CONFIG));
    CU_ASSERT(cuDevicePrimaryCtxRelease(cuDev));

    unsigned int wqConcurrencyLimit = primaryWqConfig.wqConfig.wqConcurrencyLimit / numDomains_;
    if (wqConcurrencyLimit == 0) {
        wqConcurrencyLimit = 1;
    }

    greenCtxs_.resize(numDomains_);
    ctxs_.resize(numDomains_);
    for (unsigned int i = 0; i < numDomains_; i++) {
        CUdevResource resources[2] = {};
        resources[0] = localizedSms[i];
        resources[1].type = CU_DEV_RESOURCE_TYPE_WORKQUEUE_CONFIG;
        resources[1].wqConfig.device = cuDev;
        resources[1].wqConfig.wqConcurrencyLimit = wqConcurrencyLimit;
        resources[1].wqConfig.sharingScope = CU_WORKQUEUE_SCOPE_GREEN_CTX_BALANCED;

        CUdevResourceDesc desc;
        CU_ASSERT(cuDevResourceGenerateDesc(&desc, resources, 2));
        CU_ASSERT(cuGreenCtxCreate(&greenCtxs_[i], desc, cuDev, CU_GREEN_CTX_DEFAULT_STREAM));
        CU_ASSERT(cuCtxFromGreenCtx(&ctxs_[i], greenCtxs_[i]));
    }
}

LocalizedGreenContexts::~LocalizedGreenContexts() {
    for (CUgreenCtx gc : greenCtxs_) {
        CU_ASSERT(cuGreenCtxDestroy(gc));
    }
}

#endif  // NVBW_HAVE_UGPU_MEMORY_NODE_SPLIT

MemPtrChaseOperation::MemPtrChaseOperation(unsigned long long loopCount) : loopCount(loopCount) {
    cudaDeviceProp prop;
    CUDA_ASSERT(cudaGetDeviceProperties(&prop, 0));
    smCount = prop.multiProcessorCount;
}

double MemPtrChaseOperation::doPtrChase(const int srcId, const MemcpyBuffer &peerBuffer, PtrChasingKernel kernel) {
    const unsigned int samples = gSettings.averageLoopCount > 0 ? gSettings.averageLoopCount : 1;
    PerformanceStatistic latencyNs;

    // Splitting one access budget across the samples inflates
    // host_device_latency_sm by ~7.6%.
    for (unsigned int n = 0; n < samples; n++) {
        latencyNs(latencyPtrChaseKernel(srcId, (void*)peerBuffer.getBuffer(), peerBuffer.getBufferSize(),
                                        latencyMemAccessCnt, smCount, kernel));
    }

    double cv = latencyNs.coefficientOfVariation();
    if (cv > worstCoeffOfVariation_) {
        worstCoeffOfVariation_ = cv;
    }
    return latencyNs.returnAppropriateMetric();
}
