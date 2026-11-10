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

#ifndef MULTIMEM_PTX_CUH_
#define MULTIMEM_PTX_CUH_

/**
 * PTX helpers for multimem (multicast VA) operations.
 *   multimem_st_128:          16-byte store  (multimem.st.v4.f32)    — write to all backing memories.
 *   multimem_ld_reduce_v4f32: 16-byte load+reduce (multimem.ld_reduce.add.v4.f32) — read and sum across backing memories.
 */
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900

/** 16-byte store to multicast VA (multimem.st.global.v4.f32). */
__device__ __forceinline__ void multimem_st_128(unsigned int* dst, unsigned int v0, unsigned int v1, unsigned int v2, unsigned int v3) {
    uintptr_t addr = __cvta_generic_to_global(dst);
    asm volatile("multimem.st.global.v4.f32 [%0], {%1,%2,%3,%4};"
                 :: "l"(addr), "r"(v0), "r"(v1), "r"(v2), "r"(v3) : "memory");
}

/** 4 × f32 reduction-read via multimem.ld_reduce.add.v4.f32 (16 bytes, 1 instruction).
 *
 *  This is the primary multimem read helper used by multimemReadKernelV4F32.
 *  A single vector instruction issues one NVLink request for 16 bytes across all
 *  backing memories, performing an fp32 add-reduction in hardware.  This halves
 *  the NVLink request count versus four scalar u32 loads and measures ~4× higher
 *  bandwidth on B200 (~397 GB/s vs ~97 GB/s).
 *
 *  Precondition: backing memory must be filled with fp32-safe values (e.g. 1.0f).
 *  An arbitrary bit pattern may produce non-deterministic results due to fp32
 *  rounding when exponents differ across backing memories.
 *
 *  Scope: .gpu — sufficient for a bandwidth benchmark where all backing memories
 *  are fully initialised before the kernel runs (no concurrent remote writers).
 *  Use .sys if repurposed for a live producer/consumer pattern. */
__device__ __forceinline__ void multimem_ld_reduce_v4f32(
        uintptr_t addr,
        float &v0, float &v1, float &v2, float &v3) {
    asm volatile(
        "multimem.ld_reduce.relaxed.gpu.global.add.v4.f32 {%0,%1,%2,%3}, [%4];"
        : "=f"(v0), "=f"(v1), "=f"(v2), "=f"(v3)
        : "l"(addr)
        : "memory");
}

#endif  // __CUDA_ARCH__ >= 900

#endif  // MULTIMEM_PTX_CUH_
