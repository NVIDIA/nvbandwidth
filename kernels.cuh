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

#ifndef NVBANDWIDTH__KERNELS_CUH
#define NVBANDWIDTH__KERNELS_CUH

#include <cuda.h>
#include "common.h"

const unsigned long long DEFAULT_SPIN_KERNEL_TIMEOUT = 10000000000ULL;   // 10 seconds

CUresult copyKernel(CUdeviceptr dstBuffer, CUdeviceptr srcBuffer, size_t sizeInElement, CUstream stream, unsigned long long loopCount);
CUresult spinKernel(volatile int *latch, CUstream stream, unsigned long long timeoutNs = DEFAULT_SPIN_KERNEL_TIMEOUT);

#endif //NVBANDWIDTH__KERNELS_CUH
