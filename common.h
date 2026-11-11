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

#ifndef COMMON_H_
#define COMMON_H_

#include <cmath>
#include <cstdlib>
#include <cuda.h>
#include <nvml.h>
// Some CI toolchains ship an nvml.h predating this enum; value matches current nvml.h.
#ifndef NVML_DEVICE_ARCH_RUBIN
#define NVML_DEVICE_ARCH_RUBIN 13
#endif
#include <algorithm>
#include <cctype>
#include <float.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <ostream>
#include <string>
#include <sstream>
#include <thread>
#include <vector>
#include <unordered_set>
#include <limits.h>
#include <optional>
#include <cstring>
#include <type_traits>

#include "test_settings.h"

#define STRING_LENGTH 256

// Copy src into a fixed-size MPI/exchange buffer and force null-termination.
template <size_t N>
inline void copyToFixedBuffer(char (&dst)[N], const std::string &src) {
    static_assert(N > 0, "fixed buffer must be non-empty");
    std::strncpy(dst, src.c_str(), N - 1);
    dst[N - 1] = '\0';
}

// Non-settings constants
constexpr int kWarmupCount = 4;
const unsigned long long smallBufferThreshold = 64;
const unsigned int _MiB = 1024 * 1024;
const unsigned int _2MiB = 2 * _MiB;
const unsigned int numThreadPerBlock = 512;
const unsigned int strideLen = 16; /* cacheLine size 128 Bytes, 16 words */
const unsigned long latencyMemAccessCnt = 1000000; /* 1M total read accesses to gauge latency */
// Device-to-device latency tests: fixed iteration count (independent of --loop-count).
constexpr unsigned long long kDeviceToDeviceLatencyIterations = 10000;
// Granularity of each fabric/UBLKCP put in the device-to-device latency kernels (bytes).
// Must satisfy the TMA 16-byte alignment requirement.
constexpr size_t kDeviceToDeviceLatencyPutSize = 16;
// Device-to-device latency buffers must cover at least two put-size slots; fall back to 64 KB
// if the caller-supplied size is smaller (e.g. when the user omits --size).
constexpr size_t kDeviceToDeviceLatencyMinBufSize = 65536;
// UBLKCP mailbox layout for device-to-device latency (raw-pointer path).
// The counter offset is one put-size slot past the data payload.
constexpr size_t kUblkcpCounterOffset = kDeviceToDeviceLatencyPutSize;
// Non-counted: [data kDeviceToDeviceLatencyPutSize B][sequence kDeviceToDeviceLatencyPutSize B] = 2 slots.
constexpr size_t kUblkcpMailboxBytes = 2 * kDeviceToDeviceLatencyPutSize;
// Hardware barMem counter size for cp.async.bulk.counted (TMA-managed, 8 bytes).
constexpr size_t kUblkcpBarMemBytes = 8;
// Counted: [data kDeviceToDeviceLatencyPutSize B][barMem counter kUblkcpBarMemBytes B].
constexpr size_t kUblkcpCountedMailboxBytes = kDeviceToDeviceLatencyPutSize + kUblkcpBarMemBytes;
// Device-to-device latency ack buffer (allocated on the producer device):
//   [0] handshake ack — producer clears to 0 each iteration; consumer writes 1
//   [1] producer-only watchdog status — 0, or kDeviceToDeviceLatencyWatchdogTimeoutAck
// The status word must not share storage with the handshake ack: a late consumer can still
// observe the current sequence and write 1 after the producer has timed out. If that write
// landed on the same word as the sentinel, runDeviceToDeviceLatency()'s post-sync host read
// would miss the failure and report a valid latency.
constexpr size_t kDeviceToDeviceLatencyAckHandshakeOffset = 0;
constexpr size_t kDeviceToDeviceLatencyAckStatusOffset = sizeof(uint32_t);
constexpr size_t kDeviceToDeviceLatencyAckBytes = 2 * sizeof(uint32_t);
// When the producer watchdog fires it publishes this marker into the status word. The
// launcher reads that word back and, on seeing it, returns
// kDeviceToDeviceLatencyWatchdogFailureNs so the testcase fails instead of recording a
// stalled run as a normal completion. The marker is distinct from the normal handshake
// values (0 and 1); a valid latency is always >= 0.
constexpr unsigned int kDeviceToDeviceLatencyWatchdogTimeoutAck = 0xDEADBEEFu;
constexpr double kDeviceToDeviceLatencyWatchdogFailureNs = -1.0;

// Runtime state (not settings)
extern int deviceCount;

// Pairwise GPU test: run only between these two GPUs (by UUID). Same or different nodes.
extern std::string gpuPairUuid0, gpuPairUuid1;
extern int gpuPairDev0, gpuPairDev1;  // Resolved device indices (single-node); -1 when not set
inline bool hasGpuPair() { return !gpuPairUuid0.empty(); }

#ifdef MULTINODE
extern int localDevice;
extern int localRank;
extern int ranksPerNode;
extern int worldRank;
extern int worldSize;
#endif

#ifdef MULTINODE
// 4 clique types are currently defined, leave space for 16 for good measure
constexpr unsigned int kMaxRankFabricCliques = 16;
constexpr unsigned int kFabricClusterUuidBytes = 16;

enum class FabricCliqueType : unsigned int {
    UnicastPointer = 0,
    MulticastPointer = 1,
    UnicastLogicalEndpoint = 2,
    MulticastLogicalEndpoint = 3,
};
#ifdef NVBW_HAVE_NVML_GPU_FABRIC_INFO
static_assert(static_cast<unsigned int>(FabricCliqueType::UnicastPointer) ==
              NVML_GPU_FABRIC_CLIQUE_TYPE_UNICAST_POINTER, "NVML clique enum drift");
#endif

// Per-rank GPU assignment, MPI_BYTE-exchanged across the world.
struct RankDeviceInfo {
    int rank = 0;
    int localDevice = -1;
    char hostname[STRING_LENGTH] = {};
    char deviceName[STRING_LENGTH] = {};
};
static_assert(std::is_trivially_copyable<RankDeviceInfo>::value,
              "RankDeviceInfo is exchanged via MPI_BYTE and must stay trivially copyable");

enum class RankFabricInfoSource : int {
    None = 0,
    CudaClique = 1,
    Nvml = 2,
};

struct RankFabricCliqueInfo {
    FabricCliqueType type = FabricCliqueType::UnicastPointer;
    unsigned int id = 0;
};

struct RankFabricInfo {
    int rank = 0;
    int localDevice = 0;
    RankFabricInfoSource infoSource = RankFabricInfoSource::None;
    int runtimeSymbolAvailable = 0;
    int queryResult = CUDA_ERROR_NOT_SUPPORTED;
    RankFabricCliqueInfo cliques[kMaxRankFabricCliques] = {};
    unsigned int numCliques = 0;
    unsigned long long reportedNumCliques = 0;
    int fabricStatus = -1;
    unsigned int state = 0;
    unsigned int healthMask = 0;
    unsigned char healthSummary = 0;
    unsigned char clusterUuid[kFabricClusterUuidBytes] = {};
    char hostname[STRING_LENGTH] = {};
};
static_assert(std::is_trivially_copyable<RankFabricInfo>::value,
              "RankFabricInfo is exchanged via MPI_BYTE and must stay trivially copyable");
#endif

class Verbosity {
 public:
    bool &controlVariable;

    Verbosity(bool &controlVariable): controlVariable(controlVariable) {}

    template<typename T>
    Verbosity& operator<<(T input) {
        if (!gSettings.isJsonOutput() && controlVariable) std::cout << input;
        return *this;
    }

    using StreamType = decltype(std::cout);
    Verbosity &operator<<(StreamType &(*func)(StreamType &)) {
        if (!gSettings.isJsonOutput() && controlVariable) {
            func(std::cout);
        }
        return *this;
    }
};
extern Verbosity VERBOSE;
extern Verbosity OUTPUT;

#ifdef _MSC_VER
#define __PRETTY_FUNCTION__ __FUNCTION__
#endif

// Rounds n up to the nearest multiple of "multiple".
// if n is already a multiple of "multiple", n is returned unchanged.
// works for arbitrary value of "multiple".
#define ROUND_UP(n, multiple)                                                           \
    (((n) + ((multiple)-1)) - (((n) + ((multiple)-1)) % (multiple)))

#define PROC_MASK_WORD_BITS (8 * sizeof(size_t))

#define PROC_MASK_SIZE                                                                  \
    ROUND_UP(std::thread::hardware_concurrency(), PROC_MASK_WORD_BITS) / 8

#define PROC_MASK_QUERY_BIT(mask, proc)                                                 \
    (mask[proc / PROC_MASK_WORD_BITS] &                                                 \
        ((size_t)1 << (proc % PROC_MASK_WORD_BITS)))                                    \
            ? 1                                                                         \
            : 0

/* Set a bit in an affinity mask */
#define PROC_MASK_SET(mask, proc)                                                       \
    do {                                                                                \
        size_t _proc = (proc);                                                          \
        (mask)[_proc / PROC_MASK_WORD_BITS] |= (size_t)1                                \
                                        << (_proc % PROC_MASK_WORD_BITS);               \
    } while (0)

/* Clear a bit in an affinity mask */
#define PROC_MASK_CLEAR(mask, proc)                                                     \
    do {                                                                                \
        size_t _proc = (proc);                                                          \
        (mask)[_proc / PROC_MASK_WORD_BITS] &=                                          \
            ~((size_t)1 << (_proc % PROC_MASK_WORD_BITS));                              \
    } while (0)

inline size_t getFirstEnabledCPU() {
    size_t firstEnabledCPU = 0;
    size_t *procMask = (size_t *)calloc(1, PROC_MASK_SIZE);
    for (size_t i = 0; i < PROC_MASK_SIZE * 8; ++i) {
        if (PROC_MASK_QUERY_BIT(procMask, i)) {
            firstEnabledCPU = i;
        break;
        }
    }
    free(procMask);
    return firstEnabledCPU;
}

// Calculation and display of performance statistics
// Basic online running statistics calculator, modeled after a less templated
// version of boost::accumulators.
class PerformanceStatistic {
    std::vector<double> values;

 public:
    void operator()(const double &sample) { recordSample(sample); }

    void recordSample(const double &sample) {
        auto it = std::lower_bound(values.begin(), values.end(), sample);
        values.insert(it, sample);
    }

    void reset(void) { values.clear(); }

    double sum(void) const {
        double total = 0.0;
        for (double val : values) {
            total += val;
        }
        return total;
    }

    size_t count(void) const { return values.size(); }

    double mean(void) const {
        return sum() / count();
    }

    double variance(void) const {
        double calculated_mean = mean();
        double sum_diff_squared = 0.0;
        for (double val : values) {
            double diff = val - calculated_mean;
            sum_diff_squared += diff * diff;
        }
        return (values.size() > 1 ? sum_diff_squared / (values.size() - 1) : 0.0);
    }

    double stddev(void) const {
        return (variance() > 0.0 ? std::sqrt(variance()) : 0.0);
    }

    // Run-to-run coefficient of variation (stddev / mean) over the recorded
    // samples. Measures temporal stability of a single measurement; returns 0
    // when there are not enough samples or the mean is non-positive.
    double coefficientOfVariation(void) const {
        double m = mean();
        return (count() > 1 && m > 0.0) ? stddev() / m : 0.0;
    }

    double largest(void) const { return values.size() > 0 ? values[values.size() - 1] : 0.0; }

    double smallest(void) const { return values.size() > 0 ? values[0] : 0.0; }

    double median(void) const {
        if (values.size() == 0) {
            return 0.0;
        } else if (values.size() % 2 == 0) {
            int idx = values.size() / 2;
            return (values[idx] + values[idx - 1]) / 2.0;
        } else {
            return values[values.size() / 2];
        }
    }

    double returnAppropriateMetric(void) const {
        if (gSettings.useMean) {
            return mean();
        } else {
            return median();
        }
    }
};

#ifdef MULTINODE
inline std::string getPaddedProcessId(int id) {
    // max printed number will be worldSize - 1
    int paddingSize = (int) log10(worldSize - 1) + 1;
    std::stringstream s;
    s << std::setfill(' ') << std::setw(paddingSize) << id;
    return s.str();
}

inline std::string formatRankDeviceInfo(const RankDeviceInfo &info) {
    std::stringstream s;
    s << "Process " << getPaddedProcessId(info.rank) << " (" << info.hostname << "): device "
      << info.localDevice << ": " << info.deviceName;
    return s.str();
}
#endif

struct LatencyNode {
    struct LatencyNode *next;
};

// 16-byte aligned node for TMA use cases
struct alignas(16) TMALatencyNode {
    struct TMALatencyNode *next;
};

enum UnitType {
    BANDWIDTH,
    LATENCY
};

inline std::string getUnitString(UnitType unitType) {
    switch (unitType) {
        case BANDWIDTH:
            return " +GB/s";
        case LATENCY:
            return " -ns";
        default:
            return "";
    }
}

inline bool hugePagesEnabled() {
    // Huge pages not supported in Windows version
#ifndef _WIN32
    // Check if THP (Transparent Huge Pages) is enabled
    std::ifstream thp_enabled("/sys/kernel/mm/transparent_hugepage/enabled");
    if (thp_enabled.is_open()) {
        std::string line;
        std::getline(thp_enabled, line);
        // THP is enabled if "always" or "madvise" is available (indicated by brackets)
        return (line.find("[always]") != std::string::npos ||
                line.find("[madvise]") != std::string::npos);
    }
#endif
    return false;
}

// Describe attributes of a single memcpy operation
class MemcpyDescriptor {
 public:
    CUdeviceptr dst;
    CUdeviceptr src;
    CUstream stream;
    size_t copySize;
    unsigned long long loopCount;
    bool useNvlinkUtilScheduling = false;
    unsigned int threadsPerBlock = 512;
    uint32_t srcLeId = ~0u;  // LE ID of the src buffer, or ~0u if not an LE (matches MemcpyBuffer::LE_ID_NONE)
    uint32_t dstLeId = ~0u;  // LE ID of the dst buffer, or ~0u if not an LE

    MemcpyDescriptor(CUdeviceptr dst, CUdeviceptr src, CUstream stream, size_t copySize,
                     unsigned long long loopCount, uint32_t srcLeId = ~0u, uint32_t dstLeId = ~0u);
};


#endif  // COMMON_H_
