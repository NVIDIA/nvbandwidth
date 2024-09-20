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

#ifndef INLINE_COMMON_H_
#define INLINE_COMMON_H_

#include "common.h"
#include "error_handling.h"

template <class T> struct PeerValueMatrix {
    std::vector<std::optional <T>> m_matrix;
    int m_rows, m_columns;
    std::string key;
    std::vector<std::string> column_labels;
    std::vector<std::string> row_labels;
    bool pFormatter;
    UnitType uType;

    PeerValueMatrix(int rows, int columns, std::string key = "", bool pFormatter = perfFormatter, UnitType uType = BANDWIDTH): m_matrix(rows * columns), m_rows(rows), m_columns(columns), key(key), pFormatter(perfFormatter), uType(uType) {}

    std::optional <T> &value(int src, int dst) {
        ASSERT(src >= 0 && src < m_rows);
        ASSERT(dst >= 0 && dst < m_columns);
        return m_matrix[src * m_columns + dst];
    }
    const std::optional <T> &value(int src, int dst) const {
        ASSERT(src >= 0 && src < m_rows);
        ASSERT(dst >= 0 && dst < m_columns);
        return m_matrix[src * m_columns + dst];
    }

    void setRowLabels(std::vector<std::string> _row_labels) {
        row_labels = _row_labels;
    }

    void setColumnLabels(std::vector<std::string> _column_labels) {
        column_labels = _column_labels;
    }
};

template <class T>
std::ostream &operator<<(std::ostream &o, const PeerValueMatrix<T> &matrix) {
    // This assumes T is numeric
    T maxVal = std::numeric_limits<T>::min();
    T minVal = std::numeric_limits<T>::max();
    T sum = 0;
    int count = 0;

    // First square of the table should be blank, calculate and print appropriately many spaces
    int columnIdWidth = 2;
    for (auto s : matrix.row_labels) {
        columnIdWidth = std::max(columnIdWidth, (int) s.size());
    }

    for (int i = 0; i < columnIdWidth; i++) {
        o << " ";
    }

    for (int currentDevice = 0; currentDevice < matrix.m_columns; currentDevice++) {
        if (matrix.column_labels.size() > 0) {
            o << std::setw(10) << matrix.column_labels[currentDevice];
        } else {
            o << std::setw(10) << currentDevice;
        }
    }
    o << std::endl;

    for (int currentDevice = 0; currentDevice < matrix.m_rows; currentDevice++) {
        if (matrix.row_labels.size() > 0) {
            o << std::setw(columnIdWidth) << matrix.row_labels[currentDevice];
        } else {
            o << std::setw(2) << currentDevice;
        }

        for (int peer = 0; peer < matrix.m_columns; peer++) {
            std::optional <T> val = matrix.value(currentDevice, peer);
            if (val) {
                o << std::setw(10) << val.value();
            } else {
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
    if (matrix.pFormatter) {
        o << "&&&& PERF " << matrix.key << " " << sum << getUnitString(matrix.uType) << std::endl;
    } else {
        o << "SUM " << matrix.key << " " << sum << std::endl;
    }

    VERBOSE << "MIN " << matrix.key << " " << minVal << '\n';
    VERBOSE << "MAX " << matrix.key << " " << maxVal << '\n';
    VERBOSE << "AVG " << matrix.key << " " << sum / count << '\n';
    return o;
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

#endif  // INLINE_COMMON_H_
