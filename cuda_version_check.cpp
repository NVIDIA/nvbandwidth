/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "cuda_version_check.h"

#include "output.h"

#include <cuda_runtime.h>

#include <sstream>

std::string formatCudaVersionCode(int cudaVersionCode) {
    const int major = cudaVersionCode / 1000;
    const int minor = (cudaVersionCode % 1000) / 10;
    std::ostringstream oss;
    oss << major << '.' << minor;
    return oss.str();
}

bool validateCudaRuntimeAndDriver(Output *output, int *cudaRuntimeVersionOut, int *cudaDriverApiVersionOut) {
    int runtimeVersion = 0;
    cudaError_t err = cudaRuntimeGetVersion(&runtimeVersion);
    if (err != cudaSuccess) {
        std::ostringstream msg;
        msg << "CUDA runtime version check failed: cudaRuntimeGetVersion returned " << cudaGetErrorString(err);
        output->recordError(msg.str());
        return false;
    }

    // Warn rather than hard-fail: within the same major version CUDA's ABI is
    // backward-compatible, so a slightly older libcudart may still work fine.
    // Let CUDA itself reject any calls that genuinely require a newer runtime.
    if (runtimeVersion < CUDART_VERSION) {
        std::ostringstream msg;
        msg << "Warning: CUDA runtime library version " << formatCudaVersionCode(runtimeVersion)
            << " is older than the toolkit nvbandwidth was built against ("
            << formatCudaVersionCode(CUDART_VERSION) << "). "
            << "Results may be unexpected if required runtime symbols are absent.";
        output->recordWarning(msg.str());
    }

    int driverVersion = 0;
    err = cudaDriverGetVersion(&driverVersion);
    if (err != cudaSuccess) {
        std::ostringstream msg;
        msg << "CUDA driver version check failed: cudaDriverGetVersion returned " << cudaGetErrorString(err);
        output->recordError(msg.str());
        return false;
    }

    // Warn rather than hard-fail.  cudaDriverGetVersion() reports the version
    // the driver self-advertises, which understates actual support in two common
    // supported configurations:
    //   - Minor Version Compatibility (MVC, CUDA 11.3+): a driver advertising
    //     12.2 can run binaries built against any 12.x toolkit.
    //   - Forward Compatibility (cuda-compat package): even a cross-major driver
    //     can run newer-toolkit binaries via the compat shim.
    // Proceed and let the driver reject genuinely unsupported calls.
    if (driverVersion < CUDART_VERSION) {
        const int driverMajor = driverVersion / 1000;
        const int buildMajor  = CUDART_VERSION / 1000;
        std::ostringstream msg;
        msg << "Warning: driver reports maximum supported CUDA version "
            << formatCudaVersionCode(driverVersion)
            << ", which is below the toolkit nvbandwidth was built against ("
            << formatCudaVersionCode(CUDART_VERSION) << "). ";
        if (driverMajor == buildMajor) {
            msg << "Within the same major version this is usually handled by "
                << "CUDA Minor Version Compatibility (MVC).";
        } else {
            msg << "If the cuda-compat Forward Compatibility package is installed "
                << "this configuration may still work. "
                << "If CUDA calls subsequently fail, upgrade the driver to "
                << formatCudaVersionCode(CUDART_VERSION) << "+.";
        }
        output->recordWarning(msg.str());
    }

    if (cudaRuntimeVersionOut) {
        *cudaRuntimeVersionOut = runtimeVersion;
    }
    if (cudaDriverApiVersionOut) {
        *cudaDriverApiVersionOut = driverVersion;
    }
    return true;
}
