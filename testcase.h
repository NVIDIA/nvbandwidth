/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef TESTCASE_H_
#define TESTCASE_H_


#include "common.h"
#include "inline_common.h"
#include "memcpy.h"
#include <map>
#include <regex>
#include <memory>
#include <functional>
#include <utility>

// Localization only helps where a single copy cannot saturate the link.
inline bool deviceArchUsesLocalization(int deviceId) {
    static std::map<int, bool> cache;
    auto cached = cache.find(deviceId);
    if (cached != cache.end()) return cached->second;

    bool enabled = false;
    char pciBusId[NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE] = {};
    nvmlDevice_t nvmlDev;
    nvmlDeviceArchitecture_t arch = NVML_DEVICE_ARCH_UNKNOWN;
    if (cuDeviceGetPCIBusId(pciBusId, sizeof(pciBusId), deviceId) == CUDA_SUCCESS &&
        nvmlDeviceGetHandleByPciBusId(pciBusId, &nvmlDev) == NVML_SUCCESS &&
        nvmlDeviceGetArchitecture(nvmlDev, &arch) == NVML_SUCCESS) {
        enabled = (arch == NVML_DEVICE_ARCH_RUBIN);
    }

    cache[deviceId] = enabled;
    if (!enabled) {
        VERBOSE << "device " << deviceId << ": architecture does not localize; "
                   "reporting non-localized bandwidth\n";
    }
    return enabled;
}

#ifdef NVBW_HAVE_UGPU_MEMORY_NODE_SPLIT
// Number of locality domains the driver reports for deviceId, or 0 if the
// query fails. Single-domain devices report 1.
inline int memoryNodeDomainCount(int deviceId) {
    int count = 0;
    if (cuDeviceGetAttribute(&count, CU_DEVICE_ATTRIBUTE_LOCALITY_DOMAIN_COUNT,
                             static_cast<CUdevice>(deviceId)) != CUDA_SUCCESS) {
        VERBOSE << "cuDeviceGetAttribute(LOCALITY_DOMAIN_COUNT) failed for device "
                << deviceId << "; reporting non-localized bandwidth\n";
        return 0;
    }
    return count;
}

// Whether deviceId is measured with localized concurrent copies: it must report
// more than one locality domain and be an architecture that gains from the split.
// numDomains receives that count when true, and 0 otherwise.
inline bool shouldLocalize(int deviceId, int &numDomains) {
    numDomains = 0;
    if (!deviceArchUsesLocalization(deviceId)) return false;

    const int count = memoryNodeDomainCount(deviceId);
    if (count < 2) return false;

    numDomains = count;
    return true;
}
#endif  // NVBW_HAVE_UGPU_MEMORY_NODE_SPLIT

// ============================================================================
// Configuration Enums for Consolidated Test Classes
// ============================================================================

// Specifies the copy initiator/engine to use.
enum class CopyInitiator {
    CE,     // Copy Engine (cuMemcpyAsync)
    SM,     // Shader/SM copy kernel
    TMA     // Tensor Memory Accelerator
};

// Specifies the transfer direction for host-device copies
enum class Direction {
    HostToDevice,
    DeviceToHost
};

// ============================================================================
// Host-Device Transfer Direction Strategy (for buffer ordering)
// ============================================================================

class HostDeviceTransferDirectionStrategy {
 public:
    virtual ~HostDeviceTransferDirectionStrategy() = default;

    // Returns (src, dst) pair in correct order for the transfer direction
    virtual std::pair<const MemcpyBuffer*, const MemcpyBuffer*>
        getOrderedBuffers(const HostBuffer& host, const DeviceBuffer& device) const = 0;

    // For output label formatting
    virtual std::string getDirectionArrow() const = 0;
    virtual std::string getName() const = 0;
};

class HostToDeviceDirection : public HostDeviceTransferDirectionStrategy {
 public:
    std::pair<const MemcpyBuffer*, const MemcpyBuffer*>
    getOrderedBuffers(const HostBuffer& host, const DeviceBuffer& device) const override {
        return {&host, &device};
    }
    std::string getDirectionArrow() const override { return "->"; }
    std::string getName() const override { return "host_to_device"; }
};

class DeviceToHostDirection : public HostDeviceTransferDirectionStrategy {
 public:
    std::pair<const MemcpyBuffer*, const MemcpyBuffer*>
    getOrderedBuffers(const HostBuffer& host, const DeviceBuffer& device) const override {
        return {&device, &host};
    }
    std::string getDirectionArrow() const override { return "<-"; }
    std::string getName() const override { return "device_to_host"; }
};

// ============================================================================
// Device-to-Device Access Type (Read vs Write)
// ============================================================================

// Specifies the access type for device-to-device copies
enum class AccessType {
    Read,   // Copy from peer to target (PREFER_DST_CONTEXT)
    Write   // Copy from target to peer (PREFER_SRC_CONTEXT)
};

// Strategy interface for device-to-device access type (for buffer ordering and context preference)
class DeviceToDeviceAccessStrategy {
 public:
    virtual ~DeviceToDeviceAccessStrategy() = default;

    // Returns (src, dst) pair in correct order for the access type
    virtual std::pair<const MemcpyBuffer*, const MemcpyBuffer*>
        getOrderedBuffers(const MemcpyBuffer& srcBuffer, const MemcpyBuffer& peerBuffer) const = 0;

    // Returns context preference for MemcpyOperation
    virtual ContextPreference getContextPreference() const = 0;
};

class ReadAccessStrategy : public DeviceToDeviceAccessStrategy {
 public:
    std::pair<const MemcpyBuffer*, const MemcpyBuffer*>
    getOrderedBuffers(const MemcpyBuffer& srcBuffer, const MemcpyBuffer& peerBuffer) const override {
        return {&peerBuffer, &srcBuffer};  // Read FROM peer TO src
    }
    ContextPreference getContextPreference() const override { return PREFER_DST_CONTEXT; }
};

class WriteAccessStrategy : public DeviceToDeviceAccessStrategy {
 public:
    std::pair<const MemcpyBuffer*, const MemcpyBuffer*>
    getOrderedBuffers(const MemcpyBuffer& srcBuffer, const MemcpyBuffer& peerBuffer) const override {
        return {&srcBuffer, &peerBuffer};  // Write FROM src TO peer
    }
    ContextPreference getContextPreference() const override { return PREFER_SRC_CONTEXT; }
};

// Factory for the access-type strategy (defined in testcases.cpp). Declared here
// so other translation units (e.g. multinode_testcases.cpp) can construct the
// strategy used for device-to-device buffer ordering.
std::unique_ptr<DeviceToDeviceAccessStrategy> createDeviceToDeviceAccessStrategy(AccessType accessType);

// Shown in --list generateDesc for TMA paths that use LEBufferStrategy under
// --force-le-variant.
inline const char* handlePeerBufferConventionNote() {
    return "With --force-le-variant (Handle), the peer buffer is a Logical Endpoint "
           "and the local/executing buffer is a raw device pointer. Write copies "
           "ptr->LE; Read copies LE->ptr. LE-LE copies are unsupported.";
}

// Handle (Logical Endpoint) buffers for single-node TMA tests.
// Auto-selected from the device architecture (see nvbandwidth.cpp) or forced
// via --force-le-variant / --force-raw-pointer.
//
// CE is deliberately excluded: a CE copy against a Handle operand requires
// cuFabricCpyBatchAsync (CUDA 13.5+), so CE stays on raw pointers here.
inline bool initiatorUsesHandleBuffers(CopyInitiator initiator) {
    return gSettings.bufferType == BufferType::LE &&
           initiator == CopyInitiator::TMA;
}

// Handle (Logical Endpoint) for multinode_* TMA tests. Use this from
// multinode_testcases.cpp instead of initiatorUsesHandleBuffers().
inline bool multinodeInitiatorUsesHandleBuffers(CopyInitiator initiator) {
    return gSettings.bufferType == BufferType::LE &&
           initiator == CopyInitiator::TMA;
}

// One raw CE copy. Shared by the single-node and multinode localized concurrent
// paths, which drive the copies themselves instead of going through
// MemcpyOperation.
inline void submitLocalizedCeCopy(const MemcpyBuffer& src, const MemcpyBuffer& dst,
                                  size_t size, CUstream stream) {
    CU_ASSERT(cuMemcpyAsync(dst.getBuffer(), src.getBuffer(), size, stream));
}

// Runs N concurrent streams and returns their aggregate bandwidth (GB/s). The
// caller owns all buffer/context setup and supplies submit(i, stream), which must
// enqueue exactly one `size`-byte copy for stream index i on the given stream.
//
// Bandwidth is total bytes over total time, as MemcpyOperation::TOTAL_BW does, so
// streams that do not overlap cannot report more than was achieved.
//
// streamsPerGroup: contiguous group size to measure over, 0 for one group of every
// stream. A group must share one device, since its span is timed on its first stream.
// perStreamBwOut: per-stream GB/s in stream order, or null. Diagnostic only.
// perGroupBwOut: aggregate GB/s per group, or null.
// streamContexts: context for stream i, or null to use the caller's current context.
// Non-null must have size >= streamCount. Stream create, blocker, and submit must run
// with that context current.
template <typename SubmitFn>
inline double runConcurrentLocalizedStreams(int streamCount, unsigned long long size,
                                            unsigned long long loopCount,
                                            const std::string& verboseTag,
                                            SubmitFn&& submit,
                                            std::vector<double>* perStreamBwOut = nullptr,
                                            const std::vector<CUcontext>* streamContexts = nullptr,
                                            int streamsPerGroup = 0,
                                            std::vector<double>* perGroupBwOut = nullptr) {
    if (perStreamBwOut) perStreamBwOut->clear();
    if (perGroupBwOut) perGroupBwOut->clear();
    if (streamCount <= 0) return 0.0;

    ASSERT(!streamContexts ||
           static_cast<int>(streamContexts->size()) >= streamCount);

    const int groupSize = (streamsPerGroup > 0) ? streamsPerGroup : streamCount;
    ASSERT(groupSize > 0 && streamCount % groupSize == 0);
    const int groupCount = streamCount / groupSize;

    CUcontext entryCtx = nullptr;
    CU_ASSERT(cuCtxGetCurrent(&entryCtx));

    CUcontext activeCtx = entryCtx;
    auto enterStreamCtx = [&](int i) {
        if (!streamContexts) return;
        CUcontext want = (*streamContexts)[i];
        if (want != activeCtx) {
            CU_ASSERT(cuCtxSetCurrent(want));
            activeCtx = want;
        }
    };

    std::vector<CUstream> streams(streamCount, nullptr);
    std::vector<CUevent>  startEvts(streamCount, nullptr), endEvts(streamCount, nullptr),
                          warmupEvts(streamCount, nullptr);
    // Bound each group's span; recorded on the group's first stream.
    std::vector<CUevent> groupStartEvts(groupCount, nullptr), groupEndEvts(groupCount, nullptr);
    for (int i = 0; i < streamCount; i++) {
        enterStreamCtx(i);
        CU_ASSERT(cuStreamCreate(&streams[i], CU_STREAM_NON_BLOCKING));
        CU_ASSERT(cuEventCreate(&startEvts[i],  CU_EVENT_DEFAULT));
        CU_ASSERT(cuEventCreate(&endEvts[i],    CU_EVENT_DEFAULT));
        CU_ASSERT(cuEventCreate(&warmupEvts[i], CU_EVENT_DEFAULT));
    }
    for (int g = 0; g < groupCount; g++) {
        enterStreamCtx(g * groupSize);
        CU_ASSERT(cuEventCreate(&groupStartEvts[g], CU_EVENT_DEFAULT));
        CU_ASSERT(cuEventCreate(&groupEndEvts[g],   CU_EVENT_DEFAULT));
    }

    // Prime each stream so its copy engine is co-scheduled before the timed
    // window. Uses the shared single-stream kWarmupCount: an A/B sweep (4..64
    // warmup iters) showed the summed bandwidth is flat within run-to-run noise
    // from 4 upward, so no extra priming margin is warranted here.
    const int kWarmupIters = kWarmupCount;
    const int kTimedCopies = static_cast<int>(loopCount);

    const bool useStreamBlocker = !gSettings.bounceBufferConfComputeEnabled;
    NodeHelperSingle nodeHelper;
    if (useStreamBlocker) {
        nodeHelper.streamBlockerReset();
        for (int i = 0; i < streamCount; i++) {
            enterStreamCtx(i);
            nodeHelper.streamBlockerBlock(streams[i]);
        }
    }

    // Interleaved warmup: one copy per stream per outer iteration.
    for (int k = 0; k < kWarmupIters; k++)
        for (int i = 0; i < streamCount; i++) {
            enterStreamCtx(i);
            submit(i, streams[i]);
        }
    for (int i = 0; i < streamCount; i++) {
        enterStreamCtx(i);
        CU_ASSERT(cuEventRecord(warmupEvts[i], streams[i]));
    }

    // Start only after every stream has primed, so warmup stays out of the window.
    enterStreamCtx(0);
    for (int i = 1; i < streamCount; i++)
        CU_ASSERT(cuStreamWaitEvent(streams[0], warmupEvts[i], 0));
    CU_ASSERT(cuEventRecord(groupStartEvts[0], streams[0]));
    CU_ASSERT(cuEventRecord(startEvts[0], streams[0]));

    // All groups start from that point, so they overlap in time.
    for (int g = 1; g < groupCount; g++) {
        const int ref = g * groupSize;
        enterStreamCtx(ref);
        CU_ASSERT(cuStreamWaitEvent(streams[ref], groupStartEvts[0], 0));
        CU_ASSERT(cuEventRecord(groupStartEvts[g], streams[ref]));
        CU_ASSERT(cuEventRecord(startEvts[ref], streams[ref]));
    }
    for (int g = 0; g < groupCount; g++) {
        const int ref = g * groupSize;
        for (int i = ref + 1; i < ref + groupSize; i++) {
            enterStreamCtx(i);
            CU_ASSERT(cuStreamWaitEvent(streams[i], groupStartEvts[g], 0));
            CU_ASSERT(cuEventRecord(startEvts[i], streams[i]));
        }
    }

    // Timed batch: interleaved one copy per stream per iteration.
    for (int l = 0; l < kTimedCopies; l++)
        for (int i = 0; i < streamCount; i++) {
            enterStreamCtx(i);
            submit(i, streams[i]);
        }
    for (int i = 0; i < streamCount; i++) {
        enterStreamCtx(i);
        CU_ASSERT(cuEventRecord(endEvts[i], streams[i]));
    }

    // Close a group's window once every stream in it has finished.
    for (int g = 0; g < groupCount; g++) {
        const int ref = g * groupSize;
        enterStreamCtx(ref);
        for (int i = ref + 1; i < ref + groupSize; i++)
            CU_ASSERT(cuStreamWaitEvent(streams[ref], endEvts[i], 0));
        CU_ASSERT(cuEventRecord(groupEndEvts[g], streams[ref]));
    }

    // Release the blocker; everything queued above now runs.
    if (useStreamBlocker)
        nodeHelper.streamBlockerRelease();

    for (int i = 0; i < streamCount; i++)
        CU_ASSERT(cuStreamSynchronize(streams[i]));

    const double bytesPerStream = static_cast<double>(size) * loopCount;
    // cuEventElapsedTime can round to 0.0 ms at its ~0.5 us resolution; clamp so the
    // divisions below can never divide by zero.
    auto elapsedMs = [](CUevent from, CUevent to) {
        float ms = 0.0f;
        CU_ASSERT(cuEventElapsedTime(&ms, from, to));
        return static_cast<double>(std::max(ms, 1e-6f));
    };

    for (int i = 0; i < streamCount; i++) {
        const double ms = elapsedMs(startEvts[i], endEvts[i]);
        const double bw = bytesPerStream * 1e3 / (ms * 1e9);
        if (perStreamBwOut) perStreamBwOut->push_back(bw);
        if (gSettings.verbose)
            VERBOSE << "  [" << verboseTag << "] stream" << i << "  elapsed: "
                    << std::fixed << std::setprecision(3) << ms << " ms  -> "
                    << std::setprecision(2) << bw << " GB/s\n";
    }

    double totalBwGBs = 0.0;
    for (int g = 0; g < groupCount; g++) {
        const double ms = elapsedMs(groupStartEvts[g], groupEndEvts[g]);
        const double bw = bytesPerStream * groupSize * 1e3 / (ms * 1e9);
        totalBwGBs += bw;
        if (perGroupBwOut) perGroupBwOut->push_back(bw);
        if (gSettings.verbose)
            VERBOSE << "  [" << verboseTag << "] group" << g << "  span: "
                    << std::fixed << std::setprecision(3) << ms << " ms  -> "
                    << std::setprecision(2) << bw << " GB/s\n";
    }
    if (gSettings.verbose)
        VERBOSE << "  [" << verboseTag << "] reported (total bytes / total time): "
                << std::setprecision(2) << totalBwGBs << " GB/s\n";

    for (int g = 0; g < groupCount; g++) {
        if (groupStartEvts[g]) cuEventDestroy(groupStartEvts[g]);
        if (groupEndEvts[g])   cuEventDestroy(groupEndEvts[g]);
    }
    for (int i = 0; i < streamCount; i++) {
        if (startEvts[i])  cuEventDestroy(startEvts[i]);
        if (endEvts[i])    cuEventDestroy(endEvts[i]);
        if (warmupEvts[i]) cuEventDestroy(warmupEvts[i]);
        if (streams[i])    cuStreamDestroy(streams[i]);
    }
    if (activeCtx != entryCtx) CU_ASSERT(cuCtxSetCurrent(entryCtx));
    return totalBwGBs;
}

// ============================================================================
// Buffer Type Strategy (for polymorphic buffer creation)
// ============================================================================

// Holds a pair of buffers for src and peer devices
class BufferPair {
 public:
    std::unique_ptr<MemcpyBuffer> src;
    std::unique_ptr<MemcpyBuffer> peer;

    BufferPair(std::unique_ptr<MemcpyBuffer> s, std::unique_ptr<MemcpyBuffer> p)
        : src(std::move(s)), peer(std::move(p)) {}
};

// Abstract interface for buffer type-specific operations
class BufferTypeStrategy {
 public:
    virtual ~BufferTypeStrategy() = default;

    // Check if this buffer type is supported on the system
    virtual bool isSupported() const = 0;

    // Create a pair of buffers for the given devices
    virtual std::unique_ptr<BufferPair> createBufferPair(
        size_t size, int srcDeviceId, int peerDeviceId) const = 0;

    // Enable peer access between buffers, returns false if not possible
    virtual bool enablePeerAccess(BufferPair& pair) const = 0;

    // Label for output/error messages
    virtual std::string getLabel() const = 0;
};

// Implementation for standard DeviceBuffer
class DeviceBufferStrategy : public BufferTypeStrategy {
 public:
    bool isSupported() const override { return true; }

    std::unique_ptr<BufferPair> createBufferPair(
            size_t size, int srcDeviceId, int peerDeviceId) const override {
        return std::make_unique<BufferPair>(
            std::unique_ptr<MemcpyBuffer>(new DeviceBuffer(size, srcDeviceId)),
            std::unique_ptr<MemcpyBuffer>(new DeviceBuffer(size, peerDeviceId)));
    }

    bool enablePeerAccess(BufferPair& pair) const override {
        return pair.src->enablePeerAccess(*pair.peer);
    }

    std::string getLabel() const override { return "Device"; }
};

// Forward declaration for LEBufferStrategy (defined in memcpy_le.h)
class LogicalEndpointBuffer;

// Implementation for LogicalEndpointBuffer (LE)
class LEBufferStrategy : public BufferTypeStrategy {
 public:
    bool isSupported() const override;

    std::unique_ptr<BufferPair> createBufferPair(
            size_t size, int srcDeviceId, int peerDeviceId) const override;

    bool enablePeerAccess(BufferPair& pair) const override;

    std::string getLabel() const override { return "Logical Endpoint"; }
};


/* Carries waiver reason when a test cannot run on the current system */
struct [[nodiscard]] FilterResult {
    bool passed;
    std::string reason;
    operator bool() const { return passed; }
    static FilterResult pass() { return {true, ""}; }
    static FilterResult waive(std::string reason) { return {false, std::move(reason)}; }
    // Note: overloaded operator&& does not short-circuit; both operands are always
    // evaluated. This is fine here since filter helpers are cheap attribute queries.
    friend FilterResult operator&&(FilterResult a, FilterResult b) { return a ? b : a; }
};

// ============================================================================
// Base Testcase Class
// ============================================================================

class Testcase {
 protected:
    std::string key;
    std::string desc;

    // Type alias for all-to-one and one-to-all buffer pairing callback
    using allToOneOneToAllCallback = std::function<void(int oneDeviceId,
                                                         std::vector<const MemcpyBuffer*>& allBuffers,
                                                         std::vector<const MemcpyBuffer*>& oneBuffers)>;

    static FilterResult filterHasAccessiblePeerPairs();
    static FilterResult filterHasMultipleGPUs();
    static FilterResult filterSupportsMulticast();
    static FilterResult filterSupportsTMA();
    static FilterResult filterSupportsUBLKCP();
    static FilterResult filterSupportsLogicalEndpoint();
    static FilterResult filterSupportsHandleCopy();
    static FilterResult filterBounceBufferConfComputeEnabled();
#ifdef MULTINODE
    static void addRequiredFabricCliqueType(std::vector<FabricCliqueType> &types, FabricCliqueType type);
#endif
#ifdef MULTINODE
    static FilterResult filterHasMultipleGPUsMultinode();
#endif
#ifdef NVBW_HAVE_UGPU_MEMORY_NODE_SPLIT
#endif

    // helper functions
    void forEachAllToOneBufferPairing(unsigned long long size, allToOneOneToAllCallback callback);
    void allToOneHelper(unsigned long long size, MemcpyOperation &memcpyInstance, PeerValueMatrix<double> &bandwidthValues);
    void oneToAllHelper(unsigned long long size, MemcpyOperation &memcpyInstance, PeerValueMatrix<double> &bandwidthValues);
    void allHostHelper(unsigned long long size, MemcpyOperation &memcpyInstance, PeerValueMatrix<double> &bandwidthValues, bool sourceIsHost);
    void allHostBidirHelper(unsigned long long size, MemcpyOperation &memcpyInstance, PeerValueMatrix<double> &bandwidthValues, bool sourceIsHost);

    // Copy the operation's worst coefficient of variation onto the result
    // matrix for reporting. No-op if nothing was measured. When a test measures
    // through several operations, call this for each; the worst CV wins.
    template <typename OperationT>
    static void recordStability(PeerValueMatrix<double> &bandwidthValues, const OperationT &operation) {
        double cv = operation.worstCoefficientOfVariation();
        if (cv < 0.0) {
            return;
        }
        if (!bandwidthValues.coefficientOfVariation || cv > *bandwidthValues.coefficientOfVariation) {
            bandwidthValues.coefficientOfVariation = cv;
        }
    }

    // Same, for a test whose devices can take either of two measurement paths in
    // one run: each path accumulates into its own MemcpyOperation, so reporting
    // just one of them would drop the other path's samples. Keeps the worse of
    // the two. A path with no measurements reports negative and is ignored.
    static void recordStability(PeerValueMatrix<double> &bandwidthValues,
                                const MemcpyOperation &firstInstance,
                                const MemcpyOperation &secondInstance) {
        double cv = firstInstance.worstCoefficientOfVariation();
        const double secondCv = secondInstance.worstCoefficientOfVariation();
        if (secondCv > cv) {
            cv = secondCv;
        }
        if (cv >= 0.0) {
            bandwidthValues.coefficientOfVariation = cv;
        }
    }

    template<typename AlignedLatencyNode = struct LatencyNode>
    void latencyHelper(const MemcpyBuffer &dataBuffer, bool measureDeviceToDeviceLatency);

    /**
     * Global cache for bidirectional results cross-test sharing
     * Outer Key: The test that PRODUCED the cached result (e.g., "host_to_device_bidirectional_memcpy_ce")
     * Inner Key: The test that can CONSUME the cached result (e.g., "device_to_host_bidirectional_memcpy_ce")
     * Cached Value:
     *    - Type: std::shared_ptr<PeerValueMatrix<double>>
     *    - Content: Complete bandwidth measurement matrix between all device pairs
     *    - Format: PeerValueMatrix[src_device][dst_device] = BW (GB/s)
     *    - Memory: Shared pointer allows sharing between multiple test instances
     **/
    static std::map<std::string, std::map<std::string, std::shared_ptr<PeerValueMatrix<double>>>> globalTestCache;

    // Helper to get complementary test cache key
    static std::string getComplementaryCacheKey(const std::string& testKey) {
        // Define symmetric pairs for bidirectional transfers.
        static std::vector<std::pair<std::string, std::string>> pairs = {
            {"all_to_host", "host_to_all"},
            {"host_to_device", "device_to_host"},
            {"_read", "_write"}
        };

        for (const auto& pair : pairs) {
            if (testKey.find(pair.first) != std::string::npos) {
                return std::regex_replace(testKey, std::regex(pair.first), pair.second);
            } else if (testKey.find(pair.second) != std::string::npos) {
                return std::regex_replace(testKey, std::regex(pair.second), pair.first);
            }
        }
        return "";
    }

    // Unified caching mechanism for all bidirectional tests
    template<typename TestExecutor>
    void runWithCache(PeerValueMatrix<double>& result, TestExecutor executor) {
        // Check if symmetric test has cached results we can reuse
        std::string complementaryKey = getComplementaryCacheKey(key);
        if (gSettings.performanceCacheEnabled() && !complementaryKey.empty() && globalTestCache.count(complementaryKey) &&
            globalTestCache[complementaryKey].count(key) &&
            globalTestCache[complementaryKey][key] != nullptr) {
            result = *globalTestCache[complementaryKey][key];
            return;
        }

        // Execute test with potential symmetric caching
        executor();

        // Store our result for future complementary tests to reuse:
        // globalTestCache[current_test][complementary_test] = our_result
        // This allows the symmetric test to find and reuse our measurement
        globalTestCache[key][key] = std::make_shared<PeerValueMatrix<double>>(result);
    }

    // Symmetric caching for device-to-device pairs
    template<typename TestExecutor>
    void cachedPairLoop(const std::vector<std::pair<int, int>>& pairs,
                       PeerValueMatrix<double>& matrix1,
                       PeerValueMatrix<double>& matrix2,
                       PeerValueMatrix<double>& matrixTotal,
                       TestExecutor executor) {
        for (const auto& pair : pairs) {
            int src = pair.first, dst = pair.second;

            // Use cached symmetric result if available
            if (src > dst && matrix1.value(dst, src).has_value()) {
                matrix1.value(src, dst) = matrix1.value(dst, src);
                matrix2.value(src, dst) = matrix2.value(dst, src);
                matrixTotal.value(src, dst) = matrixTotal.value(dst, src);
                continue;
            }

            auto results = executor(src, dst);
            ASSERT(results.size() == 2);

            matrix1.value(src, dst) = results[0];
            matrix2.value(src, dst) = results[1];
            matrixTotal.value(src, dst) = results[0] + results[1];
        }
    }

 public:
    Testcase(std::string key, std::string desc);
    virtual ~Testcase() {}

    std::string testKey();
    std::string testDesc();

    // Checks if the testcase can be run on the current system
    virtual FilterResult filter() { return FilterResult::pass(); }

#ifdef MULTINODE
    virtual void addRequiredFabricCliqueTypes(std::vector<FabricCliqueType> &types) const;
#endif

    // Runs the testcase
    virtual void run(unsigned long long size, unsigned long long loopCount) = 0;
};

// For D2D latency (and similar), skip peer-side allocation when the peer cannot be an endpoint of the --pair sweep.
// peerId is a CUDA device ordinal. When gpuPairDev* are unset (e.g. MPI --pair), no skipping.
inline bool peerIsGpuPairEndpoint(int peerId) {
    if (!hasGpuPair()) return true;
    if (gpuPairDev0 < 0 || gpuPairDev1 < 0) return true;
    return peerId == gpuPairDev0 || peerId == gpuPairDev1;
}

// Host-Device transfer test
class HostDeviceTransfer : public Testcase {
    CopyInitiator initiator_;
    std::unique_ptr<HostDeviceTransferDirectionStrategy> directionStrategy_;

 public:
    HostDeviceTransfer(CopyInitiator initiator, Direction direction);
    virtual ~HostDeviceTransfer() {}
    void run(unsigned long long size, unsigned long long loopCount) override;
    FilterResult filter() override;

 private:
    MemcpyInitiator* createMemcpyInitiator() const;
    static std::unique_ptr<HostDeviceTransferDirectionStrategy> createDirectionStrategy(Direction dir);
    static std::string generateKey(CopyInitiator initiator, Direction direction);
    static std::string generateDesc(CopyInitiator initiator, Direction direction);
    std::string buildOutputLabel() const;
};

// CE Testcase classes
// Host-Device bidirectional transfer test (CE/SM)
class HostDeviceBidirTransfer : public Testcase {
    CopyInitiator initiator_;
    Direction direction_;

 public:
    HostDeviceBidirTransfer(CopyInitiator initiator, Direction direction);
    virtual ~HostDeviceBidirTransfer() {}
    void run(unsigned long long size, unsigned long long loopCount) override;

    // Waive for bounce-buffer CC since cuMemcpyAsync becomes synchronous for
    // host<->device copies and the test does not produce bidirectional
    // traffic.
    FilterResult filter() override { return filterBounceBufferConfComputeEnabled(); }

 private:
    MemcpyInitiator* createMemcpyInitiator() const;
    static std::string generateKey(CopyInitiator initiator, Direction direction);
    static std::string generateDesc(CopyInitiator initiator, Direction direction);
    std::string buildOutputLabel() const;
};

// Device-to-device transfer test (CE/SM/TMA) - Read/Write
class DeviceToDeviceTransfer : public Testcase {
    CopyInitiator initiator_;
    AccessType accessType_;
    std::unique_ptr<DeviceToDeviceAccessStrategy> accessStrategy_;
    std::unique_ptr<BufferTypeStrategy> bufferStrategy_;

 public:
    DeviceToDeviceTransfer(CopyInitiator initiator, AccessType accessType);
    virtual ~DeviceToDeviceTransfer() {}
    void run(unsigned long long size, unsigned long long loopCount) override;
    FilterResult filter() override;

 private:
    MemcpyInitiator* createMemcpyInitiator() const;
    std::unique_ptr<BufferTypeStrategy> createBufferStrategy() const;
    static std::string generateKey(CopyInitiator initiator, AccessType accessType);
    static std::string generateDesc(CopyInitiator initiator, AccessType accessType);
    std::string buildOutputLabel() const;
};

// Device-to-device bidirectional transfer test (CE/SM/TMA) - Read/Write
class DeviceToDeviceBidirTransfer : public Testcase {
    CopyInitiator initiator_;
    AccessType accessType_;
    std::unique_ptr<DeviceToDeviceAccessStrategy> accessStrategy_;
    std::unique_ptr<BufferTypeStrategy> bufferStrategy_;

 public:
    DeviceToDeviceBidirTransfer(CopyInitiator initiator, AccessType accessType);
    virtual ~DeviceToDeviceBidirTransfer() {}
    void run(unsigned long long size, unsigned long long loopCount) override;
    FilterResult filter() override;

 private:
    MemcpyInitiator* createMemcpyInitiator() const;
    std::unique_ptr<BufferTypeStrategy> createBufferStrategy() const;
    static std::string generateKey(CopyInitiator initiator, AccessType accessType);
    static std::string generateDesc(CopyInitiator initiator, AccessType accessType);
    std::string buildOutputLabel(const std::string& suffix) const;
};

// Selects between non-counted (fabric_try_put / UBLKCP 2 put+fence) and
// counted (put_counted+fabric_submit / UBLKCP counted) latency variants.
enum class CountedWrite { No, Yes };

// Device-to-device latency test for all local GPU pairs (single-process only).
// LE variant (--force-le-variant): fabric_try_put[_counted] + consumer ack.
// Raw variant (default):           UBLKCP 2 put+fence / UBLKCP counted.
class DeviceToDeviceLatency : public Testcase {
 public:
    explicit DeviceToDeviceLatency(CountedWrite counted);
    virtual ~DeviceToDeviceLatency() {}
    void run(unsigned long long size, unsigned long long loopCount) override;
    FilterResult filter() override;

 private:
    CountedWrite counted_;
    std::string buildOutputLabel() const;
    static std::string makeKey(CountedWrite counted);
    static std::string makeDesc(CountedWrite counted);
};

// Device local copy transfer test
class DeviceLocalCopyTransfer : public Testcase {
    CopyInitiator initiator_;

 public:
    explicit DeviceLocalCopyTransfer(CopyInitiator initiator);
    virtual ~DeviceLocalCopyTransfer() {}
    void run(unsigned long long size, unsigned long long loopCount) override;
    FilterResult filter() override;

 private:
    // These tests always use raw pointers: a same-device Handle destination is an
    // LE owned by the copying device, so the copy would leave the die and return
    // through the switch, which is not local bandwidth.
    MemcpyInitiator* createMemcpyInitiator() const;
    static std::string generateKey(CopyInitiator initiator);
    static std::string generateDesc(CopyInitiator initiator);
    std::string buildOutputLabel() const;
};

// Device local SM read memcpy using a copy kernel
class DeviceLocalReadSM: public Testcase {
 public:
    DeviceLocalReadSM() : Testcase("device_local_read_sm",
            "\tMeasures bandwidth of a kernel reading from a device buffer local to the GPU.") {}
    virtual ~DeviceLocalReadSM() {}
    void run(unsigned long long size, unsigned long long loopCount);
};

// Device local SM write memcpy using a copy kernel
class DeviceLocalWriteSM: public Testcase {
 public:
    DeviceLocalWriteSM() : Testcase("device_local_write_sm",
            "\tMeasures bandwidth of a kernel writing to a device buffer local to the GPU.\n") {}
    virtual ~DeviceLocalWriteSM() {}
    void run(unsigned long long size, unsigned long long loopCount);
};

// All-to-host transfer test
class AllToHostTransfer : public Testcase {
    CopyInitiator initiator_;

 public:
    explicit AllToHostTransfer(CopyInitiator initiator);
    virtual ~AllToHostTransfer() {}
    void run(unsigned long long size, unsigned long long loopCount) override;

    // Waive for bounce-buffer CC since cuMemcpyAsync becomes synchronous for
    // host<->device copies and the test does not produce an all-to-host
    // traffic pattern.
    FilterResult filter() override { return filterBounceBufferConfComputeEnabled(); }

 private:
    MemcpyInitiator* createMemcpyInitiator() const;
    static std::string generateKey(CopyInitiator initiator);
    static std::string generateDesc(CopyInitiator initiator);
    std::string buildOutputLabel() const;
};

// All-to-host bidirectional transfer test
class AllToHostBidirTransfer : public Testcase {
    CopyInitiator initiator_;

 public:
    explicit AllToHostBidirTransfer(CopyInitiator initiator);
    virtual ~AllToHostBidirTransfer() {}
    void run(unsigned long long size, unsigned long long loopCount) override;

    // Waive for bounce-buffer CC since cuMemcpyAsync becomes synchronous for
    // host<->device copies and the test does not produce a bidirectional
    // all-to-host traffic pattern.
    FilterResult filter() override { return filterBounceBufferConfComputeEnabled(); }

 private:
    MemcpyInitiator* createMemcpyInitiator() const;
    static std::string generateKey(CopyInitiator initiator);
    static std::string generateDesc(CopyInitiator initiator);
    std::string buildOutputLabel() const;
};

// Host-to-all transfer test
class HostToAllTransfer : public Testcase {
    CopyInitiator initiator_;

 public:
    explicit HostToAllTransfer(CopyInitiator initiator);
    virtual ~HostToAllTransfer() {}
    void run(unsigned long long size, unsigned long long loopCount) override;

    // Waive for bounce-buffer CC since cuMemcpyAsync becomes synchronous for
    // host<->device copies and the test does not produce a host-to-all traffic
    // pattern.
    FilterResult filter() override { return filterBounceBufferConfComputeEnabled(); }

 private:
    MemcpyInitiator* createMemcpyInitiator() const;
    static std::string generateKey(CopyInitiator initiator);
    static std::string generateDesc(CopyInitiator initiator);
    std::string buildOutputLabel() const;
};

// Host-to-all bidirectional transfer test
class HostToAllBidirTransfer : public Testcase {
    CopyInitiator initiator_;

 public:
    explicit HostToAllBidirTransfer(CopyInitiator initiator);
    virtual ~HostToAllBidirTransfer() {}
    void run(unsigned long long size, unsigned long long loopCount) override;

    // Waive for bounce-buffer CC since cuMemcpyAsync becomes synchronous for
    // host<->device copies and the test does not produce a bidirectional
    // host-to-all traffic pattern.
    FilterResult filter() override { return filterBounceBufferConfComputeEnabled(); }

 private:
    MemcpyInitiator* createMemcpyInitiator() const;
    static std::string generateKey(CopyInitiator initiator);
    static std::string generateDesc(CopyInitiator initiator);
    std::string buildOutputLabel() const;
};

// All-to-one transfer test
class AllToOneTransfer : public Testcase {
    CopyInitiator initiator_;
    AccessType accessType_;
    std::unique_ptr<DeviceToDeviceAccessStrategy> accessStrategy_;

 public:
    AllToOneTransfer(CopyInitiator initiator, AccessType accessType);
    virtual ~AllToOneTransfer() {}
    void run(unsigned long long size, unsigned long long loopCount) override;
    FilterResult filter() override;

 private:
    MemcpyInitiator* createMemcpyInitiator() const;
    static std::string generateKey(CopyInitiator initiator, AccessType accessType);
    static std::string generateDesc(CopyInitiator initiator, AccessType accessType);
    std::string buildOutputLabel() const;
};

// One-to-all transfer test
class OneToAllTransfer : public Testcase {
    CopyInitiator initiator_;
    AccessType accessType_;
    std::unique_ptr<DeviceToDeviceAccessStrategy> accessStrategy_;

 public:
    OneToAllTransfer(CopyInitiator initiator, AccessType accessType);
    virtual ~OneToAllTransfer() {}
    void run(unsigned long long size, unsigned long long loopCount) override;
    FilterResult filter() override;

 private:
    MemcpyInitiator* createMemcpyInitiator() const;
    static std::string generateKey(CopyInitiator initiator, AccessType accessType);
    static std::string generateDesc(AccessType accessType);
    std::string buildOutputLabel() const;
};

// SM Testcase classes
// Host to device SM latency using a ptr chase kernel
class HostDeviceLatencySM: public Testcase {
 public:
    HostDeviceLatencySM() : Testcase("host_device_latency_sm",
            "\tHost - device access latency using a pointer chase kernel\n"
            "\tA 2MB buffer is allocated on the host and is accessed by the GPU") {}
    virtual ~HostDeviceLatencySM() {}
    void run(unsigned long long size, unsigned long long loopCount);

    // Waive for bounce-buffer CC since SM accesses to sysmem in bounce-buffer
    // CC modes result in page faults and migration of the accessed pages to
    // vidmem, so this does not measure host->device latency.
    FilterResult filter() override { return filterBounceBufferConfComputeEnabled(); }
};

// Device to Device SM Latency ptr chase kernel
class DeviceToDeviceLatencySM: public Testcase {
 public:
    DeviceToDeviceLatencySM() : Testcase("device_to_device_latency_sm",
            "\tMeasures latency of a pointer derefernce operation between each pair of accessible peers.\n"
            "\tA 2MB buffer is allocated on a GPU and is accessed by the peer GPU to determine latency.\n"
            "\t--bufferSize flag is ignored") {}
    virtual ~DeviceToDeviceLatencySM() {}
    void run(unsigned long long size, unsigned long long loopCount);
    FilterResult filter() override { return Testcase::filterHasAccessiblePeerPairs(); }
};

// Device to Device TMA Latency ptr chase kernel
class DeviceToDeviceLatencyTMA: public Testcase {
 public:
    DeviceToDeviceLatencyTMA() : Testcase("device_to_device_latency_tma",
            "\tMeasures latency of a pointer dereference operation between each pair of accessible peers using TMA.\n"
            "\tA 2MB buffer is allocated on a GPU and is accessed by the peer GPU using TMA to determine latency.\n"
            "\t--bufferSize flag is ignored") {}
    virtual ~DeviceToDeviceLatencyTMA() {}
    void run(unsigned long long size, unsigned long long loopCount);
    FilterResult filter() override {
        auto r = Testcase::filterHasAccessiblePeerPairs();
        if (!r) return r;
        return Testcase::filterSupportsTMA();
    }
};

// ============================================================================
// uGPU localization tests
// ============================================================================

#ifdef MULTINODE
// Multinode device-to-device transfer test (CE/SM/TMA) - Read/Write
class MultinodeDeviceToDeviceTransfer : public Testcase {
    CopyInitiator initiator_;
    AccessType accessType_;

 public:
    MultinodeDeviceToDeviceTransfer(CopyInitiator initiator, AccessType accessType);
    virtual ~MultinodeDeviceToDeviceTransfer() {}
    void run(unsigned long long size, unsigned long long loopCount) override;
    FilterResult filter() override;
    void addRequiredFabricCliqueTypes(std::vector<FabricCliqueType> &types) const override;

 private:
    ContextPreference getContextPreference() const;
    MemcpyInitiator* createMemcpyInitiator() const;
    static std::string generateKey(CopyInitiator initiator, AccessType accessType);
    static std::string generateDesc(CopyInitiator initiator, AccessType accessType);
    std::string buildOutputLabel() const;
};

// Multinode device-to-device bidirectional transfer test (CE/SM/TMA) - Read/Write
class MultinodeDeviceToDeviceBidirTransfer : public Testcase {
    CopyInitiator initiator_;
    AccessType accessType_;

 public:
    MultinodeDeviceToDeviceBidirTransfer(CopyInitiator initiator, AccessType accessType);
    virtual ~MultinodeDeviceToDeviceBidirTransfer() {}
    void run(unsigned long long size, unsigned long long loopCount) override;
    FilterResult filter() override;
    void addRequiredFabricCliqueTypes(std::vector<FabricCliqueType> &types) const override;

 private:
    ContextPreference getContextPreference() const;
    MemcpyInitiator* createMemcpyInitiator() const;
    static std::string generateKey(CopyInitiator initiator, AccessType accessType);
    static std::string generateDesc(CopyInitiator initiator, AccessType accessType);
    std::string buildOutputLabel(const std::string& suffix) const;
};

class MultinodeAllToOneWriteSM: public Testcase {
 public:
    MultinodeAllToOneWriteSM() : Testcase("multinode_device_to_device_all_to_one_write_sm",
            "\tMeasures the total bandwidth of copies from all accessible peers to a single device, for each\n"
            "\tdevice. Bandwidth is reported as the total inbound bandwidth for each device.\n"
            "\tWrite tests launch a copy from the peer to the target device using the peer's context.") {}
    virtual ~MultinodeAllToOneWriteSM() {}
    void run(unsigned long long size, unsigned long long loopCount);
    FilterResult filter() override { return Testcase::filterHasMultipleGPUsMultinode(); }
    void addRequiredFabricCliqueTypes(std::vector<FabricCliqueType> &types) const override;
};

class MultinodeAllFromOneReadSM: public Testcase {
 public:
    MultinodeAllFromOneReadSM() : Testcase("multinode_device_to_device_all_from_one_read_sm",
            "\tMeasures the total bandwidth of copies from a single device to all accessible peers, for each\n"
            "\tdevice. Bandwidth is reported as the total outbound bandwidth for each device.\n"
            "\tRead tests launch a copy from the target device to the peer using the peer's context.") {}
    virtual ~MultinodeAllFromOneReadSM() {}
    void run(unsigned long long size, unsigned long long loopCount);
    FilterResult filter() override { return Testcase::filterHasMultipleGPUsMultinode(); }
    void addRequiredFabricCliqueTypes(std::vector<FabricCliqueType> &types) const override;
};

class MultinodeBroadcastOneToAllSM: public Testcase {
 public:
    MultinodeBroadcastOneToAllSM() : Testcase("multinode_device_to_device_broadcast_one_to_all_sm",
            "\tMeasures bandwidth of a copy kernel copying data from device memory to multicast allocated memory\n"
            "\tthat's mapped on all accessible peers.\n"
            "\tTests launch a copy from the target device to the multicast memory on target using the target's context.") {}
    virtual ~MultinodeBroadcastOneToAllSM() {}
    void run(unsigned long long size, unsigned long long loopCount);
    FilterResult filter() override {
        auto r = Testcase::filterHasMultipleGPUsMultinode();
        if (!r) return r;
        return Testcase::filterSupportsMulticast();
    }
    void addRequiredFabricCliqueTypes(std::vector<FabricCliqueType> &types) const override;
};

class MultinodeBroadcastAllToAllSM: public Testcase {
 public:
    MultinodeBroadcastAllToAllSM() : Testcase("multinode_device_to_device_broadcast_all_to_all_sm",
            "\tMeasures bandwidth of a copy kernels copying data from device memory to multicast allocated memory\n"
            "\tthat's mapped on all accessible peers."
            "\tAll devices are doing copies at the same time.\n"
            "\tTests launch copies from the target device to the multicast memory on target using the target's context.") {}
    virtual ~MultinodeBroadcastAllToAllSM() {}
    void run(unsigned long long size, unsigned long long loopCount);
    FilterResult filter() override {
        auto r = Testcase::filterHasMultipleGPUsMultinode();
        if (!r) return r;
        return Testcase::filterSupportsMulticast();
    }
    void addRequiredFabricCliqueTypes(std::vector<FabricCliqueType> &types) const override;
};

// TODO(pgumienny) - add remaining combination of Read/Write CE/SM once the tooling is in
class MultinodeBisectWriteCE: public Testcase {
 public:
    MultinodeBisectWriteCE() : Testcase("multinode_bisect_write_ce",
            "\tMeasures bandwidths of simultaneous copies.\n"
            "\tFor N GPU system there will be N copies occuring at the same time\n"
            "\tGPU owned by rank A will be writing to GPU owned by rank (A + N/2) % N).") {}
    virtual ~MultinodeBisectWriteCE() {}
    void run(unsigned long long size, unsigned long long loopCount);
    FilterResult filter() override { return Testcase::filterHasMultipleGPUsMultinode(); }
    void addRequiredFabricCliqueTypes(std::vector<FabricCliqueType> &types) const override;
};

// Multinode multicast fabric put: write to multicast LE (TMA, --force-le-variant) or raw multicast VA.
class MultinodeMulticastFabricPut : public Testcase {
 public:
    explicit MultinodeMulticastFabricPut(CopyInitiator initiator);
    virtual ~MultinodeMulticastFabricPut() {}
    void run(unsigned long long size, unsigned long long loopCount) override;
    FilterResult filter() override;
    void addRequiredFabricCliqueTypes(std::vector<FabricCliqueType> &types) const override;

 private:
    CopyInitiator initiator_;
    std::string buildOutputLabel() const;
};

// Multinode multicast read and reduce: fabric.try_pullred via LE (--force-le-variant)
// or multimem.ld_reduce via raw multicast VA (default / --force-raw-pointer).
class MultinodeMultimemReduce : public Testcase {
 public:
    MultinodeMultimemReduce();
    virtual ~MultinodeMultimemReduce() {}
    void run(unsigned long long size, unsigned long long loopCount) override;
    FilterResult filter() override;
    void addRequiredFabricCliqueTypes(std::vector<FabricCliqueType> &types) const override;

 private:
    std::string buildOutputLabel() const;
};

#endif

#endif  // TESTCASE_H_
