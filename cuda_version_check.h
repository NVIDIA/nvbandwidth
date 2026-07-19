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

#ifndef CUDA_VERSION_CHECK_H_
#define CUDA_VERSION_CHECK_H_

#include <string>

class Output;

// Human-readable form of CUDA toolkit encoding (1000*major + 10*minor).
std::string formatCudaVersionCode(int cudaVersionCode);

// Checks libcudart and driver versions against the CUDA toolkit nvbandwidth was
// built against (CUDART_VERSION) and writes the detected versions to the out
// parameters.  Version mismatches are reported as warnings rather than errors
// because CUDA Minor Version Compatibility (MVC) and the Forward Compatibility
// (cuda-compat) package both legitimately allow driver/runtime versions that are
// lower than the build toolkit; let CUDA itself reject truly incompatible calls.
// Returns false (hard failure) only when the version query APIs themselves fail,
// i.e. when CUDA is not present at all.
bool validateCudaRuntimeAndDriver(Output *output, int *cudaRuntimeVersionOut, int *cudaDriverApiVersionOut);

#endif  // CUDA_VERSION_CHECK_H_
