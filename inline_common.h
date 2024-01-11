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

#pragma once

#include "common.h"
#include "nvbandwidth_json.h"


// CUDA Error handling
inline void CU_ASSERT(CUresult cuResult, const char *msg = nullptr) {
    if (cuResult != CUDA_SUCCESS) {
        const char *errDescStr, *errNameStr;
        cuGetErrorString(cuResult, &errDescStr);
        cuGetErrorName(cuResult, &errNameStr);
        std::stringstream errmsg;
        errmsg << "[" << errNameStr << "] " << errDescStr;
        if (msg != nullptr) errmsg << ":\n\t" << msg;
        if (!jsonOutput) {
            std::cout << errmsg.str() << std::endl;
        } else {
            jsonMgr.recordError(errmsg.str());
            jsonMgr.printJson();
        }
        std::exit(1);
  }
}

// NVML Error handling
inline void NVML_ASSERT(nvmlReturn_t nvmlResult, const char *msg = nullptr) {
    if (nvmlResult != NVML_SUCCESS) {
        std::stringstream errmsg;
        errmsg << "NVML_ERROR: [" << nvmlErrorString(nvmlResult) << "]";
        if (msg != nullptr) errmsg << ":\n\t" << msg;
        if (!jsonOutput) {
            std::cout << errmsg.str() << std::endl;
        } else {
            jsonMgr.recordError(errmsg.str());
            jsonMgr.printJson();
        }
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
