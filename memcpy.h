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
#include <unordered_map>
#include "common.h"
#include "error_handling.h"
#include "kernels.cuh"

// Helper function to enable peer access between two devices
// Returns true if peer access was successfully enabled (or already enabled)
// Returns false if peer access is not possible between the devices
bool enablePeerAccessBetweenDevices(int srcDeviceId, int dstDeviceId);

class MemcpyBuffer;

// Pattern setup / check for localized concurrent paths that bypass MemcpyOperation
// (runConcurrentLocalizedStreams). Uses the same seeds as MemcpyInitiatorCE and
// verifies each destination in its owning context.
void prepareLocalizedConcurrentCeCopy(const MemcpyBuffer& memcpySrc,
                                      const MemcpyBuffer& memcpyDst,
                                      size_t copySize);
void verifyLocalizedConcurrentCeCopy(const MemcpyBuffer& memcpySrc,
                                     const MemcpyBuffer& memcpyDst,
                                     size_t copySize);

// Single-buffer memset/memcmp in the buffer's owning context.
void memsetLocalizedConcurrentOwnedBuffer(const MemcpyBuffer& buf, size_t size,
                                          unsigned int seed);
void memcmpLocalizedConcurrentOwnedBuffer(const MemcpyBuffer& buf, size_t size,
                                          unsigned int seed);

class MemcpyBuffer {
 protected:
    void* buffer{};
    size_t bufferSize;

    CUdeviceptr stagingBuffer{};

 public:
    MemcpyBuffer(size_t bufferSize);
    virtual ~MemcpyBuffer() {}
    CUdeviceptr getBuffer() const;
    size_t getBufferSize() const;

    CUdeviceptr getStagingBuffer() const { return stagingBuffer; }

    virtual int getBufferIdx() const = 0;
    virtual CUcontext getPrimaryCtx() const = 0;
#ifdef NVBW_HAVE_UGPU_MEMORY_NODE_SPLIT
    // Non-null when work for this buffer should run under a green context.
    virtual CUgreenCtx getGreenCtx() const { return nullptr; }
#endif
    virtual std::string getBufferString() const = 0;
    // In MPI configuration we want to avoid using blocking functions such as cuStreamSynchronize to adhere to MPI notion of progress
    // For more details see https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/mpi.html#mpi-progress
    virtual CUresult streamSynchronizeWrapper(CUstream stream) const;
    virtual int getMPIRank() const;

    // Returns the local device pointer; LE importers (remote side) override to return 0.
    virtual CUdeviceptr getDevicePtr() const { return getBuffer(); }

    // Logical Endpoint ID for LE-backed buffers; LE_ID_NONE for non-LE buffers.
    virtual uint32_t getLeId() const { return LE_ID_NONE; }
    static constexpr uint32_t LE_ID_NONE = static_cast<uint32_t>(-1);

    // True for DeviceBufferMulticast (raw multicast VA); used by MemcpyInitiatorTMA
    // to dispatch launchMultimemWriteKernel instead of copyKernelTMA.
    // Multicast LE is not used in the single-node tests.
    virtual bool isMulticastBuffer() const { return false; }

    // Number of physical backing memories behind this buffer.
    // For multicast LE buffers, returns the number of participant devices (N).
    // fabric.try_pullred sums all N backing memories, so callers multiply the
    // expected pattern by this value when verifying readback.
    virtual uint32_t getNumBackingMemories() const { return 1; }

    // Peer access: override supportsPeerAccess() in buffer types that support P2P (e.g. DeviceBuffer, LogicalEndpointBuffer)
    virtual bool supportsPeerAccess() const { return false; }
    virtual bool enablePeerAccess(const MemcpyBuffer &peerBuffer) const;
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

    bool supportsPeerAccess() const override { return true; }
    bool enablePeerAccess(const MemcpyBuffer &peerBuffer) const override;
};

#ifdef NVBW_HAVE_UGPU_MEMORY_NODE_SPLIT
// Device buffer pinned to a specific uGPU memory node via the public CUDA VMM API
// (cuMemCreate with CU_MEM_LOCATION_TYPE_DEVICE_LOCALITY_DOMAIN).
// memoryNodeId maps to localityDomainId (0-based: 0 = first die, 1 = second die, ...).
// Requires CUDA 13.4+ (CU_MEM_LOCATION_TYPE_DEVICE_LOCALITY_DOMAIN support).
class LocalizedDeviceBuffer : public MemcpyBuffer {
 private:
    int deviceIdx_;
    unsigned int memoryNodeId_;
    CUcontext primaryCtx_{};
    size_t vmmAllocSize_{0};

 public:
    LocalizedDeviceBuffer(size_t bufferSize, int deviceIdx, unsigned int memoryNodeId);
    ~LocalizedDeviceBuffer();

    int getBufferIdx() const override { return deviceIdx_; }
    CUcontext getPrimaryCtx() const override { return primaryCtx_; }
    std::string getBufferString() const override { return std::string("LocalizedNode") + std::to_string(memoryNodeId_); }
    bool supportsPeerAccess() const override { return true; }
    bool enablePeerAccess(const MemcpyBuffer& peerBuffer) const override;
};

// One stitched VA allocation split across numDomains uGPU locality domains:
// a single cuMemAddressReserve spanning numDomains chunks, with a separate
// cuMemCreate/cuMemMap/cuMemSetAccess per domain mapping domain i's chunk at
// offset i * chunkAllocSize. Unlike LocalizedDeviceBuffer above (one
// independent reservation per single-domain buffer), this backs numDomains
// logical buffers from one VA range.
class SplitLocalizedDeviceBuffer {
 private:
    int deviceIdx_;
    unsigned int numDomains_;
    CUcontext primaryCtx_{};
    CUdeviceptr base_{};
    size_t chunkAllocSize_{0};    // per-domain chunk allocation size, rounded up to granularity
    size_t logicalChunkSize_{0};  // caller-requested per-domain chunk size (unrounded)

 public:
    SplitLocalizedDeviceBuffer(size_t totalSize, int deviceIdx, unsigned int numDomains);
    ~SplitLocalizedDeviceBuffer();

    int getBufferIdx() const { return deviceIdx_; }
    CUcontext getPrimaryCtx() const { return primaryCtx_; }
    unsigned int getNumDomains() const { return numDomains_; }
    size_t getChunkSize() const { return logicalChunkSize_; }
    CUdeviceptr chunkPtr(unsigned int domain) const {
        ASSERT(domain < numDomains_);
        return base_ + static_cast<size_t>(domain) * chunkAllocSize_;
    }
};

// Per-device provider of green contexts localized to each uGPU locality domain.
class LocalizedGreenContexts {
 private:
    unsigned int numDomains_;
    std::vector<CUgreenCtx> greenCtxs_;
    std::vector<CUcontext> ctxs_;

 public:
    LocalizedGreenContexts(int deviceIdx, unsigned int numDomains);
    ~LocalizedGreenContexts();

    CUcontext ctxForDomain(unsigned int domain) const {
        ASSERT(domain < numDomains_);
        return ctxs_[domain];
    }
    CUgreenCtx greenCtxForDomain(unsigned int domain) const {
        ASSERT(domain < numDomains_);
        return greenCtxs_[domain];
    }
};

// Thin, non-owning adapter presenting one domain's chunk of a
// SplitLocalizedDeviceBuffer as an independent MemcpyBuffer, so it can be
// used with the existing doMemcpy(vector<>, vector<>) path unchanged.
// If greenCtxs is non-null, getPrimaryCtx() returns that domain's green
// context instead of the buffer's ordinary primary context, so the copy
// also executes only on the SMs local to that domain.
class LocalizedBufferChunkView : public MemcpyBuffer {
 private:
    const SplitLocalizedDeviceBuffer& owner_;
    unsigned int domain_;
    const LocalizedGreenContexts* greenCtxs_;

 public:
    LocalizedBufferChunkView(const SplitLocalizedDeviceBuffer& owner, unsigned int domain,
                             const LocalizedGreenContexts* greenCtxs = nullptr)
        : MemcpyBuffer(owner.getChunkSize()), owner_(owner), domain_(domain), greenCtxs_(greenCtxs) {
        buffer = reinterpret_cast<void*>(owner_.chunkPtr(domain_));
    }

    int getBufferIdx() const override { return owner_.getBufferIdx(); }
    // Despite the MemcpyBuffer API name, this may return a green context when
    // greenCtxs_ is set (domain-localized SM resources), not the device primary context.
    CUcontext getPrimaryCtx() const override {
        return greenCtxs_ ? greenCtxs_->ctxForDomain(domain_) : owner_.getPrimaryCtx();
    }
    CUgreenCtx getGreenCtx() const override {
        return greenCtxs_ ? greenCtxs_->greenCtxForDomain(domain_) : nullptr;
    }
    std::string getBufferString() const override {
        return std::string("SplitLocalizedNode") + std::to_string(domain_);
    }
    bool supportsPeerAccess() const override { return true; }
    bool enablePeerAccess(const MemcpyBuffer& peerBuffer) const override {
        return enablePeerAccessBetweenDevices(getBufferIdx(), peerBuffer.getBufferIdx());
    }
};
#endif  // NVBW_HAVE_UGPU_MEMORY_NODE_SPLIT

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
    // dispatched.  Used so every rank can detect the LE unicast context (for the
    // barrierAllRanks() call) and initialise LE buffers it owns even when it is not
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
    virtual ~NodeHelper() = default;

    // Returns the buffer on whose rank/context the kernel should run.
    // LE buffers are always remote; the non-LE side always hosts the kernel.
    static const MemcpyBuffer* selectExecutingBuffer(
        const MemcpyBuffer* src, const MemcpyBuffer* dst, ContextPreference ctxPreference);

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
    // Use before any cross-rank data dependency — e.g. after an LE source buffer is
    // initialised by its owner rank and before the reader rank issues fabric.try_get.
    virtual void barrierAllRanks() = 0;

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
    // Optional per-dispatch lifecycle for initiator-specific resources and status.
    virtual void prepareResources(MemcpyDispatchInfo &info) {}
    virtual void validateResult(MemcpyDispatchInfo &info) const {}
    virtual void releaseResources(MemcpyDispatchInfo &info) {}
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

class MemcpyInitiatorMultimemWrite : public MemcpyInitiator {
 public:
    size_t memcpyFunc(MemcpyDescriptor &memcpyDescriptor, MemcpyDispatchInfo* info = nullptr);
    size_t getAdjustedCopySize(size_t src_size, size_t dst_size, CUstream stream);
    void memsetPattern(MemcpyDispatchInfo &info) const;
    void memcmpPattern(MemcpyDispatchInfo &info) const;
    unsigned long long getAdjustedBandwidth(unsigned long long bandwidth);
    std::string getName() const override { return "Multicast"; }
};

// multimem.ld_reduce.add.v4.f32, ILP=8, bPS=8.
// Backing memory must be filled with 1.0f per float (0x3F800000).
class MemcpyInitiatorMultimemReadV4F32 : public MemcpyInitiator {
 public:
    size_t memcpyFunc(MemcpyDescriptor &desc, MemcpyDispatchInfo* info = nullptr) override;
    size_t getAdjustedCopySize(size_t srcSize, size_t dstSize, CUstream stream) override;
    void memsetPattern(MemcpyDispatchInfo &info) const override;
    void memcmpPattern(MemcpyDispatchInfo &info) const override;
    unsigned long long getAdjustedBandwidth(unsigned long long bandwidth) override;
    std::string getName() const override { return "MulticastReduceV4F32"; }
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
    // Multiplier applied to measured bandwidth before verbose output and the
    // returned metric. Default 1.0; device_local_copy uses 2.0 so reported
    // bandwidth counts read+write device-memory traffic.
    double bandwidthScale_ = 1.0;

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

    void setBandwidthScale(double scale) {
        bandwidthScale_ = scale;
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

    // Worst CV across this operation's pointer chases, negative if nothing has
    // been measured.
    double worstCoefficientOfVariation() const { return worstCoeffOfVariation_; }

 private:
    unsigned long long loopCount;
    unsigned int smCount;
    double worstCoeffOfVariation_ = -1.0;
};

#endif  // MEMCPY_H_
