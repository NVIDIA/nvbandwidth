/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Stub implementations for LE/fabric kernels when building for architectures < Blackwell (sm_100)
 * or with a toolkit whose headers predate the Logical Endpoint API (NVBW_HAVE_LE_API).
 *
 * These no-ops are never reached because callers are guarded:
 *   - LE tests are waived at runtime by filterSupportsLogicalEndpoint().
 *   - The multicast-write (TMA) path is waived at compile time via NVBW_HAVE_FABRIC_KERNELS
 *     in MultinodeMulticastFabricPut::filter() (undefined whenever this stub is built).
 *
 * NOTE: the multimem.ld_reduce read path (multimemReadKernelV4F32 / launchMultimemReadKernelV4F32)
 * only requires sm_90+ and is NOT an LE test, so it is NOT stubbed here — it lives in the
 * always-compiled kernels_multimem.cu instead.
 */

#include <cuda.h>
#include <cuda_runtime.h>

// All functions must match the extern "C" linkage declared in kernels_le_tma.hpp.
extern "C" {

void leLaunchWriteKernel(CUdeviceptr, unsigned int, size_t, unsigned long long, CUstream) {}
void leLaunchReadKernel(unsigned int, CUdeviceptr, size_t, unsigned long long, CUstream) {}
void leLaunchMemsetKernel(unsigned int, unsigned int, size_t, CUstream) {}

double leLaunchE2ELatencyUnicast(CUdeviceptr, unsigned int, CUdeviceptr, CUdeviceptr,
                                 unsigned long long, int, int) { return 0; }
double leLaunchE2ELatencyUnicastCounted(CUdeviceptr, unsigned int, CUdeviceptr, CUdeviceptr,
                                        unsigned long long, int, int) { return 0; }
// UBLKCP path is always compiled (no NVBW_HAVE_LE_API guard at the call site).
double leLaunchE2ELatencyUblkcpGSE2E(CUdeviceptr, CUdeviceptr, CUdeviceptr,
                                     unsigned long long, int, int) { return 0; }
double leLaunchE2ELatencyUblkcpGSCounted(CUdeviceptr, CUdeviceptr, CUdeviceptr, CUdeviceptr,
                                         unsigned long long, int, int) { return 0; }

}  // extern "C"

void leLaunchWriteMulticastKernel(CUdeviceptr, unsigned int, size_t, unsigned long long, CUstream) {}
void leLaunchReduceMulticastKernel(unsigned int, CUdeviceptr, size_t, unsigned long long, CUstream) {}
void launchMultimemWriteKernel(CUdeviceptr, CUdeviceptr, size_t, unsigned long long, CUstream) {}
