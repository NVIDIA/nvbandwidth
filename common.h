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

#ifndef COMMON_H
#define COMMON_H

#include <cassert>
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

// Default constants
const unsigned long long defaultLoopCount = 16;
const unsigned long long defaultBufferSize = 64; // 64MB
const unsigned int defaultAverageLoopCount = 3;
const unsigned int _MiB = 1024 * 1024;
const unsigned int numThreadPerBlock = 512;

extern int deviceCount;
extern unsigned int averageLoopCount;
extern bool disableAffinity;
extern bool skipVerification;
// Verbosity
extern bool verbose;
class Verbosity {
public:    
    Verbosity() = default;
    template<typename T>
    Verbosity& operator<<(T input) {
      if (verbose) std::cout << input;
      return *this;
    }
};
extern Verbosity VERBOSE;

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
    
    double average(void) const { 
        return sum() / count();
    }
    
    double variance(void) const {
        double mean = average();
        double sum_diff_squared = 0.0;
        for (double val : values) {
            double diff = val - mean;
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
};

template <class T> struct PeerValueMatrix {
    std::optional <T> *m_matrix;
    int m_rows, m_columns;
    std::string key;

    PeerValueMatrix(int rows, int columns, std::string key = ""): m_matrix(new std::optional <T>[rows * columns]()), m_rows(rows), m_columns(columns), key(key) {}

    ~PeerValueMatrix() { delete[] m_matrix; }
    std::optional <T> &value(int src, int dst) {
        assert(src >= 0 && src < m_rows);
        assert(dst >= 0 && dst < m_columns);
        return m_matrix[src * m_columns + dst];
    }
    const std::optional <T> &value(int src, int dst) const {
        assert(src >= 0 && src < m_rows);
        assert(dst >= 0 && dst < m_columns);
        return m_matrix[src * m_columns + dst];
    }
};

template <class T>
std::ostream &operator<<(std::ostream &o, const PeerValueMatrix<T> &matrix) {
    // This assumes T is numeric
    T maxVal = std::numeric_limits<T>::min();
    T minVal = std::numeric_limits<T>::max();
    T sum = 0;
    int count = 0;

    o << ' ';
    for (int currentDevice = 0; currentDevice < matrix.m_columns; currentDevice++) {
        o << std::setw(10) << currentDevice;
    }
    o << std::endl;
    for (int currentDevice = 0; currentDevice < matrix.m_rows; currentDevice++) {
        o << currentDevice;
        for (int peer = 0; peer < matrix.m_columns; peer++) {
            std::optional <T> val = matrix.value(currentDevice, peer);
            if (val) {
                o << std::setw(10) << val.value();
            }
            else {
                o << std::setw(10) << "N/A";
            }
            sum += val.value_or(0.0);
            maxVal = std::max(maxVal, val.value_or(0.0));
            minVal = std::min(minVal, val.value_or(0.0));
            if (val.value_or(0.0) > 0) count++;
        }
        o << std::endl;
    }
    o << std::endl;
    o << "SUM " << matrix.key << " " << sum << std::endl;

    VERBOSE << "MIN " << matrix.key << " " << minVal << '\n';
    VERBOSE << "MAX " << matrix.key << " " << maxVal << '\n';
    VERBOSE << "AVG " << matrix.key << " " << sum / count << '\n';
    return o;
}

// CUDA Error handling
inline void CU_ASSERT(CUresult cuResult, const char *msg = nullptr) {
    if (cuResult != CUDA_SUCCESS) {
        const char *errDescStr, *errNameStr;
        cuGetErrorString(cuResult, &errDescStr);
        cuGetErrorName(cuResult, &errNameStr);
        std::cout << "[" << errNameStr << "] " << errDescStr;
        if (msg != nullptr) std::cout << ":\n\t" << msg;
        std::cout << std::endl;
        std::exit(1);
  }
}

// NVML Error handling
inline void NVML_ASSERT(nvmlReturn_t nvmlResult, const char *msg = nullptr) {
    if (nvmlResult != NVML_SUCCESS) {
        std::cout << "NVML_ERROR: [" << nvmlErrorString(nvmlResult) << "]";
        if (msg != nullptr) std::cout << ":\n\t" << msg;
        std::cout << std::endl;
        std::exit(1);
    }
}

// NUMA optimal affinity
inline void setOptimalCpuAffinity(int cudaDeviceID) {
#ifdef _WIN32
    // NVML doesn't support setting affinity on Windows
    return;
#endif
    if (disableAffinity) {
        return;
    }

    nvmlDevice_t device;
    CUuuid dev_uuid;

    std::stringstream s;
    std::unordered_set <unsigned char> dashPos {0, 4, 6, 8, 10};

    CU_ASSERT(cuDeviceGetUuid(&dev_uuid, cudaDeviceID));

    s << "GPU";
    for (int i = 0; i < 16; i++) {
        if (dashPos.count(i)) {
            s << '-';
        }
        s << std::hex << std::setfill('0') << std::setw(2) << (0xFF & (int)dev_uuid.bytes[i]);
    }

    NVML_ASSERT(nvmlDeviceGetHandleByUUID(s.str().c_str(), &device));
    nvmlReturn_t result = nvmlDeviceSetCpuAffinity(device);
    if (result != NVML_ERROR_NOT_SUPPORTED) {
        NVML_ASSERT(result);
    }
}

inline bool isMemoryOwnedByCUDA(void *memory) {
    CUmemorytype memorytype;
    CUresult status = cuPointerGetAttribute(&memorytype, CU_POINTER_ATTRIBUTE_MEMORY_TYPE, (CUdeviceptr)memory);
    if (status == CUDA_ERROR_INVALID_VALUE) {
        return false;
    } else {
        CU_ASSERT(status);
        return true;
    }
}

#endif
