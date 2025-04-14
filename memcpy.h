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

#ifndef MEMCPY_H_
#define MEMCPY_H_

#include <memory>
#include "common.h"

class MemcpyBuffer {
 protected:
    void* buffer{};
    size_t bufferSize;
 public:
    MemcpyBuffer(size_t bufferSize);
    virtual ~MemcpyBuffer() {}
    CUdeviceptr getBuffer() const;
    size_t getBufferSize() const;

    virtual int getBufferIdx() const = 0;
    virtual CUcontext getPrimaryCtx() const = 0;
    virtual std::string getBufferString() const = 0;
    // In MPI configuration we want to avoid using blocking functions such as cuStreamSynchronize to adhere to MPI notion of progress
    // For more details see https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/mpi.html#mpi-progress
    virtual CUresult streamSynchronizeWrapper(CUstream stream) const;
    virtual int getMPIRank() const;
};

// Represents the host buffer abstraction
class HostBuffer : public MemcpyBuffer {
 public:
    // NUMA affinity is set here through allocation of memory in the socket group where `targetDeviceId` resides
    HostBuffer(size_t bufferSize, int targetDeviceId);
    ~HostBuffer();

    int getBufferIdx() const override;
    CUcontext getPrimaryCtx() const override;
    virtual std::string getBufferString() const override;
};

// Represents the device buffer and context abstraction
class DeviceBuffer : public MemcpyBuffer {
 private:
    int deviceIdx;
    CUcontext primaryCtx{};
 public:
    DeviceBuffer(size_t bufferSize, int deviceIdx);
    ~DeviceBuffer();

    int getBufferIdx() const override;
    CUcontext getPrimaryCtx() const override;
    virtual std::string getBufferString() const override;

    bool enablePeerAcess(const DeviceBuffer &peerBuffer);
};

// Specifies the preferred node's context to do the operation from
// It's only a preference because if the preferred node is a HostBuffer, it has no context and will fall back to the other node
enum ContextPreference {
        PREFER_SRC_CONTEXT,    // Prefer the source buffer's context if available
        PREFER_DST_CONTEXT     // Prefer the destination buffer's context if available
};

class MemcpyOperation;

// forward declaration
class NodeHelper;

class MemcpyDispatchInfo {
 public:
    std::vector<CUcontext> contexts;
    std::vector<CUstream> streams;
    std::vector<const MemcpyBuffer*> srcBuffers;
    std::vector<const MemcpyBuffer*> dstBuffers;
    std::vector<int> originalRanks;
    std::vector<size_t> adjustedCopySizes;
    std::shared_ptr<NodeHelper> nodeHelper;
    MemcpyDispatchInfo(std::vector<const MemcpyBuffer*> srcBuffers, std::vector<const MemcpyBuffer*> dstBuffers, std::vector<CUcontext> contexts, std::vector<int> originalRanks = {});
};

class NodeHelper {
 public:
    virtual MemcpyDispatchInfo dispatchMemcpy(const std::vector<const MemcpyBuffer*> &srcBuffers, const std::vector<const MemcpyBuffer*> &dstBuffers, ContextPreference ctxPreference) = 0;
    virtual double calculateTotalBandwidth(double totalTime, double totalSize, size_t loopCount) = 0;
    virtual double calculateSumBandwidth(std::vector<PerformanceStatistic> &bandwidthStats) = 0;
    virtual double calculateFirstBandwidth(std::vector<PerformanceStatistic> &bandwidthStats) = 0;
    virtual std::vector<double> calculateVectorBandwidth(std::vector<double> &results, std::vector<int> originalRanks) = 0;
    virtual void synchronizeProcess() = 0;
    // In MPI configuration we want to avoid using blocking functions such as cuStreamSynchronize to adhere to MPI notion of progress
    // For more details see https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/mpi.html#mpi-progress
    virtual CUresult streamSynchronizeWrapper(CUstream stream) const = 0;

    // stream blocking functions
    virtual void streamBlockerReset() = 0;
    virtual void streamBlockerRelease() = 0;
    virtual void streamBlockerBlock(CUstream stream) = 0;
};

class NodeHelperSingle : public NodeHelper {
 private:
    volatile int* blockingVarHost;
 public:
    NodeHelperSingle();
    ~NodeHelperSingle();
    MemcpyDispatchInfo dispatchMemcpy(const std::vector<const MemcpyBuffer*> &srcBuffers, const std::vector<const MemcpyBuffer*> &dstBuffers, ContextPreference ctxPreference);
    double calculateTotalBandwidth(double totalTime, double totalSize, size_t loopCount);
    double calculateSumBandwidth(std::vector<PerformanceStatistic> &bandwidthStats);
    double calculateFirstBandwidth(std::vector<PerformanceStatistic> &bandwidthStats);
    std::vector<double> calculateVectorBandwidth(std::vector<double> &results, std::vector<int> originalRanks);
    void synchronizeProcess();
    CUresult streamSynchronizeWrapper(CUstream stream) const;

    // stream blocking functions
    void streamBlockerReset();
    void streamBlockerRelease();
    void streamBlockerBlock(CUstream stream);
};

class MemcpyInitiator {
 public:
    // Pure virtual function for implementation of the actual memcpy function
    // return actual bytes copied
    // This can vary from copySize due to SM copies truncated the copy to achieve max bandwidth
    virtual size_t memcpyFunc(MemcpyDescriptor &memcpyDescriptor) = 0;
    // Calculate the truncated sizes used by copy kernels
    virtual size_t getAdjustedCopySize(size_t size, CUstream stream) = 0;
    // Fill buffer with a pattern
    virtual void memsetPattern(MemcpyDispatchInfo &info) const = 0;
    // Compare buffer with a pattern
    virtual void memcmpPattern(MemcpyDispatchInfo &info) const = 0;
    // Adjust the bandwidth before final reporting
    virtual unsigned long long getAdjustedBandwidth(unsigned long long bandwidth) = 0;
};

class MemcpyInitiatorSM : public MemcpyInitiator {
 public:
    size_t memcpyFunc(MemcpyDescriptor &memcpyDescriptor);
    // Calculate the truncated sizes used by copy kernels
    size_t getAdjustedCopySize(size_t size, CUstream stream);
    // Fill buffer with a pattern
    void memsetPattern(MemcpyDispatchInfo &info) const;
    // Compare buffer with a pattern
    void memcmpPattern(MemcpyDispatchInfo &info) const;
    // Adjust the bandwidth before final reporting
    unsigned long long getAdjustedBandwidth(unsigned long long bandwidth);
};

class MemcpyInitiatorCE : public MemcpyInitiator  {
 public:
    size_t memcpyFunc(MemcpyDescriptor &memcpyDescriptor);
    // Calculate the truncated sizes used by copy kernels
    size_t getAdjustedCopySize(size_t size, CUstream stream);
    // Fill buffer with a pattern
    void memsetPattern(MemcpyDispatchInfo &info) const;
    // Compare buffer with a pattern
    void memcmpPattern(MemcpyDispatchInfo &info) const;
    // Adjust the bandwidth before final reporting
    unsigned long long getAdjustedBandwidth(unsigned long long bandwidth);
};

class MemcpyInitiatorMulticastWrite : public MemcpyInitiator {
 public:
    size_t memcpyFunc(MemcpyDescriptor &memcpyDescriptor);
    // Calculate the truncated sizes used by copy kernels
    size_t getAdjustedCopySize(size_t size, CUstream stream);
    // Fill buffer with a pattern
    void memsetPattern(MemcpyDispatchInfo &info) const;
    // Compare buffer with a pattern
    void memcmpPattern(MemcpyDispatchInfo &info) const;
    // Adjust the bandwidth before final reporting
    unsigned long long getAdjustedBandwidth(unsigned long long bandwidth);
};

class MemcpyInitiatorSMSplitWarp : public MemcpyInitiatorSM {
 public:
    size_t memcpyFunc(MemcpyDescriptor &memcpyDescriptor);
    // Fill buffer with a pattern
    void memsetPattern(MemcpyDispatchInfo &info) const;
    // Compare buffer with a pattern
    void memcmpPattern(MemcpyDispatchInfo &info) const;
    // Adjust the bandwidth before final reporting
    unsigned long long getAdjustedBandwidth(unsigned long long bandwidth);
};

// Abstraction of a memory Operation.
class MemoryOperation {
 public:
    MemoryOperation() = default;
    ~MemoryOperation() = default;
};

// Abstraction of a memcpy operation
class MemcpyOperation : public MemoryOperation {
 public:
    // Specifies which bandwidths to use for the final result of simultaneous copies
    enum BandwidthValue {
            USE_FIRST_BW,      // Use the bandwidth of the first copy in the simultaneous copy list
            SUM_BW,            // Use the sum of all bandwidths from the simultaneous copy list
            TOTAL_BW,          // Use the total bandwidth of all copies, based on total time and total bytes copied
            VECTOR_BW,         // Return bandwidths of each copy separately
    };

    ContextPreference ctxPreference;

 private:
    unsigned long long loopCount;

 protected:
    size_t *procMask;
    BandwidthValue bandwidthValue;

    std::shared_ptr<NodeHelper> nodeHelper;
    std::shared_ptr<MemcpyInitiator> memcpyInitiator;

 public:
    MemcpyOperation(unsigned long long loopCount, MemcpyInitiator *_memcpyInitiator, ContextPreference ctxPreference = ContextPreference::PREFER_SRC_CONTEXT, BandwidthValue bandwidthValue = BandwidthValue::USE_FIRST_BW);
    MemcpyOperation(unsigned long long loopCount, MemcpyInitiator *_memcpyInitiator, NodeHelper *_nodeHelper, ContextPreference ctxPreference = ContextPreference::PREFER_SRC_CONTEXT, BandwidthValue bandwidthValue = BandwidthValue::USE_FIRST_BW);
    virtual ~MemcpyOperation();

    // Lists of paired nodes will be executed sumultaneously
    // context of srcBuffers is preferred (if not host) unless otherwise specified
    std::vector<double> doMemcpyCore(MemcpyDispatchInfo &info);
    std::vector<double> doMemcpyVector(const std::vector<const MemcpyBuffer*> &srcBuffers, const std::vector<const MemcpyBuffer*> &dstBuffers);
    double doMemcpy(const std::vector<const MemcpyBuffer*> &srcBuffers, const std::vector<const MemcpyBuffer*> &dstBuffers);
    double doMemcpy(const MemcpyBuffer &srcBuffer, const MemcpyBuffer &dstBuffer);
};

class MemPtrChaseOperation : public MemoryOperation {
 public:
    MemPtrChaseOperation(unsigned long long loopCount);
    ~MemPtrChaseOperation() = default;
    double doPtrChase(const int srcId, const MemcpyBuffer &peerBuffer);
 private:
    unsigned long long loopCount;
    unsigned int smCount;
};

#endif  // MEMCPY_H_
