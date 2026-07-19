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

#ifndef MEMCPY_H_
#define MEMCPY_H_

#include <memory>
#include "common.h"
#include "kernels.cuh"

// Helper function to enable peer access between two devices
// Returns true if peer access was successfully enabled (or already enabled)
// Returns false if peer access is not possible between the devices
bool enablePeerAccessBetweenDevices(int srcDeviceId, int dstDeviceId);

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

    virtual bool enablePeerAcess(const MemcpyBuffer &peerBuffer) const;
    virtual bool supportsPeerAccess() const { return false; }
    virtual bool isMulticastBuffer() const { return false; }
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

    bool enablePeerAcess(const MemcpyBuffer &peerBuffer) const override;
    bool supportsPeerAccess() const override { return true; }
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
    // Unfiltered source/destination buffers: all pairs regardless of which rank is
    // dispatched.  Used so every rank can detect the unicast context (for the
    // barrierAllRanks() call) and initialise buffers it owns even when it is not
    // the dispatched copy rank for a given pair.
    std::vector<const MemcpyBuffer*> allSrcBuffers;
    std::vector<const MemcpyBuffer*> allDstBuffers;
    std::vector<CUdeviceptr> verificationBuffers;  // For read-only/write-only tests
    std::vector<size_t> verificationSizes;         // Adjusted sizes for verification
    std::vector<int> originalRanks;
    std::vector<size_t> adjustedCopySizes;
    std::shared_ptr<NodeHelper> nodeHelper;
    unsigned long long loopCount;  // Number of iterations the kernel performs
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

    // Synchronize all ranks (MPI_Barrier in multinode, no-op in single-node).
    // Use before any cross-rank data dependency — e.g. after a source buffer is
    // initialised by its owner rank and before the reader rank issues fabric.try_get.
    virtual void barrierAllRanks() = 0;

    static const MemcpyBuffer* selectExecutingBuffer(const MemcpyBuffer* src, const MemcpyBuffer* dst, ContextPreference ctxPreference);

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
    void barrierAllRanks();

    // stream blocking functions
    void streamBlockerReset();
    void streamBlockerRelease();
    void streamBlockerBlock(CUstream stream);
};

class MemcpyInitiator {
 public:
    virtual ~MemcpyInitiator() = default;

    // Pure virtual function for implementation of the actual memcpy function
    // return actual bytes copied
    // This can vary from copySize due to SM copies truncated the copy to achieve max bandwidth
    virtual size_t memcpyFunc(MemcpyDescriptor &memcpyDescriptor, MemcpyDispatchInfo* info = nullptr) = 0;
    // Calculate the truncated sizes used by copy kernels
    virtual size_t getAdjustedCopySize(size_t src_size, size_t dst_size, CUstream stream) = 0;
    // Fill buffer with a pattern
    virtual void memsetPattern(MemcpyDispatchInfo &info) const = 0;
    // Compare buffer with a pattern
    virtual void memcmpPattern(MemcpyDispatchInfo &info) const = 0;
    // Adjust the bandwidth before final reporting
    virtual unsigned long long getAdjustedBandwidth(unsigned long long bandwidth) = 0;
    // Get the name of the initiator for output labels
    virtual std::string getName() const = 0;
};

class MemcpyInitiatorSM : public MemcpyInitiator {
 protected:
    unsigned int threadsPerBlock = 512;
 public:
    size_t memcpyFunc(MemcpyDescriptor &memcpyDescriptor, MemcpyDispatchInfo* info = nullptr);
    // Calculate the truncated sizes used by copy kernels
    size_t getAdjustedCopySize(size_t src_size, size_t dst_size, CUstream stream);
    // Fill buffer with a pattern
    void memsetPattern(MemcpyDispatchInfo &info) const;
    // Compare buffer with a pattern
    void memcmpPattern(MemcpyDispatchInfo &info) const;
    // Adjust the bandwidth before final reporting
    unsigned long long getAdjustedBandwidth(unsigned long long bandwidth);
    // Get the name of the initiator for output labels
    std::string getName() const override { return "SM"; }
};

class MemcpyInitiatorCE : public MemcpyInitiator  {
 public:
    size_t memcpyFunc(MemcpyDescriptor &memcpyDescriptor, MemcpyDispatchInfo* info = nullptr);
    // Calculate the truncated sizes used by copy kernels
    size_t getAdjustedCopySize(size_t src_size, size_t dst_size, CUstream stream);
    // Fill buffer with a pattern
    void memsetPattern(MemcpyDispatchInfo &info) const;
    // Compare buffer with a pattern
    void memcmpPattern(MemcpyDispatchInfo &info) const;
    // Adjust the bandwidth before final reporting
    unsigned long long getAdjustedBandwidth(unsigned long long bandwidth);
    // Get the name of the initiator for output labels
    std::string getName() const override { return "CE"; }
};

class MemcpyInitiatorMulticastWrite : public MemcpyInitiator {
 public:
    size_t memcpyFunc(MemcpyDescriptor &memcpyDescriptor, MemcpyDispatchInfo* info = nullptr);
    size_t getAdjustedCopySize(size_t src_size, size_t dst_size, CUstream stream);
    void memsetPattern(MemcpyDispatchInfo &info) const;
    void memcmpPattern(MemcpyDispatchInfo &info) const;
    unsigned long long getAdjustedBandwidth(unsigned long long bandwidth);
    std::string getName() const override { return "Multicast"; }
};


class MemcpyInitiatorSMSplitWarp : public MemcpyInitiatorSM {
 public:
    size_t memcpyFunc(MemcpyDescriptor &memcpyDescriptor, MemcpyDispatchInfo* info = nullptr);
    // Fill buffer with a pattern
    void memsetPattern(MemcpyDispatchInfo &info) const;
    // Compare buffer with a pattern
    void memcmpPattern(MemcpyDispatchInfo &info) const;
    // Adjust the bandwidth before final reporting
    unsigned long long getAdjustedBandwidth(unsigned long long bandwidth);
};


// TMA (Tensor Memory Accelerator) memcpy initiator
class MemcpyInitiatorTMA : public MemcpyInitiator {
 public:
    MemcpyInitiatorTMA() = default;
    size_t memcpyFunc(MemcpyDescriptor &memcpyDescriptor, MemcpyDispatchInfo* info = nullptr);
    size_t getAdjustedCopySize(size_t src_size, size_t dst_size, CUstream stream);
    void memsetPattern(MemcpyDispatchInfo &info) const;
    void memcmpPattern(MemcpyDispatchInfo &info) const;
    unsigned long long getAdjustedBandwidth(unsigned long long bandwidth);
    std::string getName() const override { return "TMA"; }
};

class MemcpyInitiatorSMReadOnly : public MemcpyInitiatorSM {
 public:
    size_t memcpyFunc(MemcpyDescriptor &memcpyDescriptor, MemcpyDispatchInfo* info = nullptr);
    // Calculate the truncated sizes used by copy kernels (read-only: only src buffer is accessed)
    size_t getAdjustedCopySize(size_t src_size, size_t dst_size, CUstream stream);
    // Fill buffer with a pattern
    void memsetPattern(MemcpyDispatchInfo &info) const;
    // Compare buffer with a pattern - verifies samples
    void memcmpPattern(MemcpyDispatchInfo &info) const;
};

class MemcpyInitiatorSMWriteOnly : public MemcpyInitiatorSM {
 public:
    size_t memcpyFunc(MemcpyDescriptor &memcpyDescriptor, MemcpyDispatchInfo* info = nullptr);
    // Calculate the truncated sizes used by copy kernels (write-only: only dst buffer is accessed)
    size_t getAdjustedCopySize(size_t src_size, size_t dst_size, CUstream stream);
    // Fill buffer with a pattern
    void memsetPattern(MemcpyDispatchInfo &info) const;
    // Compare buffer with a pattern
    void memcmpPattern(MemcpyDispatchInfo &info) const;
};

// SM-based memcpy initiator optimized for OneToAll patterns.
// Uses 32 threads per block instead of the default 512 to improve
// performance characteristics for one-to-all GPU communication patterns.
class MemcpyInitiatorSMOneToAll : public MemcpyInitiatorSM {
 public:
    MemcpyInitiatorSMOneToAll() { threadsPerBlock = 32; }
    size_t memcpyFunc(MemcpyDescriptor &memcpyDescriptor, MemcpyDispatchInfo* info = nullptr);
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
    bool useNvlinkUtilScheduling_ = false;

    // Worst (largest) run-to-run coefficient of variation observed across all
    // measurements performed by this operation. Each test uses a dedicated
    // MemcpyOperation, so this tracks the least stable measurement in the test.
    // Negative means no measurement has been recorded yet.
    double worstCoeffOfVariation_ = -1.0;
    void recordCoefficientOfVariation(double cv);

 protected:
    size_t *procMask;
    BandwidthValue bandwidthValue;

    std::shared_ptr<NodeHelper> nodeHelper;
    std::shared_ptr<MemcpyInitiator> memcpyInitiator;

 public:
    MemcpyOperation(unsigned long long loopCount, MemcpyInitiator *_memcpyInitiator, ContextPreference ctxPreference = ContextPreference::PREFER_SRC_CONTEXT, BandwidthValue bandwidthValue = BandwidthValue::USE_FIRST_BW);
    MemcpyOperation(unsigned long long loopCount, MemcpyInitiator *_memcpyInitiator, NodeHelper *_nodeHelper, ContextPreference ctxPreference = ContextPreference::PREFER_SRC_CONTEXT, BandwidthValue bandwidthValue = BandwidthValue::USE_FIRST_BW);
    virtual ~MemcpyOperation();

    // Set whether to use NVLink utilization scheduling for SM kernels
    void setUseNvlinkUtilScheduling(bool enable) {
        useNvlinkUtilScheduling_ = enable;
    }

    // Worst run-to-run coefficient of variation across all measurements done by
    // this operation, or a negative value if nothing has been measured. Used to
    // report the COEFFICIENT_OF_VARIATION stability metric for a test.
    double worstCoefficientOfVariation() const { return worstCoeffOfVariation_; }

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
    double doPtrChase(const int srcId, const MemcpyBuffer &peerBuffer, PtrChasingKernel kernel);
 private:
    unsigned long long loopCount;
    unsigned int smCount;
};

#endif  // MEMCPY_H_
