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
#include <float.h>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <string>
#include <sstream>
#include <thread>
#include <vector>
#include <unordered_set>
#include <limits.h>
#include <optional>
#include <cstring>

#define STRING_LENGTH 256

// Default constants
const unsigned long long defaultLoopCount = 16;
const unsigned long long smallBufferThreshold = 64;
const unsigned long long defaultBufferSize = 512;  // 512 MiB
const unsigned int defaultAverageLoopCount = 3;
const unsigned int _MiB = 1024 * 1024;
const unsigned int _2MiB = 2 * _MiB;
const unsigned int numThreadPerBlock = 512;
const unsigned int strideLen = 16; /* cacheLine size 128 Bytes, 16 words */
const unsigned long latencyMemAccessCnt = 1000000; /* 1M total read accesses to gauge latency */
extern int deviceCount;
extern unsigned int averageLoopCount;
extern bool disableAffinity;
extern bool skipVerification;
extern bool useMean;
extern bool jsonOutput;
// Verbosity
extern bool verbose;
extern bool perfFormatter;

#ifdef MULTINODE
extern int localDevice;
extern int localRank;
extern int worldRank;
extern int worldSize;
#endif
extern char localHostname[STRING_LENGTH];

class Verbosity {
 public:
    bool &controlVariable;

    Verbosity(bool &controlVariable): controlVariable(controlVariable) {}

    template<typename T>
    Verbosity& operator<<(T input) {
        if (!jsonOutput && controlVariable) std::cout << input;
        return *this;
    }

    using StreamType = decltype(std::cout);
    Verbosity &operator<<(StreamType &(*func)(StreamType &)) {
        if (!jsonOutput && controlVariable) {
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
        if (useMean) {
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
#endif

struct LatencyNode {
    struct LatencyNode *next;
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

// Describe attributes of a single memcpy operation
class MemcpyDescriptor {
 public:
    CUdeviceptr dst;
    CUdeviceptr src;
    CUstream stream;
    size_t copySize;
    unsigned long long loopCount;

    MemcpyDescriptor(CUdeviceptr dst, CUdeviceptr src, CUstream stream, size_t copySize, unsigned long long loopCount);
};


#endif  // COMMON_H_
