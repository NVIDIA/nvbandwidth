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

#ifndef MEMCPY_LE_H_
#define MEMCPY_LE_H_

/**
 * Logical Endpoint (LE) buffer support for nvbandwidth
 *
 * Provides LogicalEndpointBuffer class for LE-based P2P memory transfers.
 * The full implementation requires CUDA driver headers that expose the LE API
 * (CUlogicalEndpointProp, cuLogicalEndpointCreate, etc.).  On older drivers
 * that lack these symbols, NVBW_HAVE_LE_API is not defined by CMake and this
 * header compiles to stubs that report LE as unsupported.
 */

#include "memcpy.h"
#include "kernels_le_tma.hpp"
#include "output.h"
#include <cuda.h>
#include <cuda_runtime.h>

#ifdef NVBW_HAVE_LE_API

#include <cstdio>
#include <cstdlib>
#include <set>
#include <thread>
#include <chrono>
#include <vector>

// ============================================================================
// Helper Functions
// ============================================================================

// Forward declaration; defined below.
inline bool waitLeReadyWithTimeout(CUlogicalEndpointId leId, std::chrono::seconds timeout);

// Time to wait for an LE to reach the ready state before bind on the per-buffer
// create path, where the device is already known to be on the selected LE path.
inline constexpr auto kLeReadyTimeout = std::chrono::seconds(15);

// Shorter timeout for the one-shot support probe. On a box without working
// fabric the endpoint can create but never converge to ready, so the probe
// would otherwise stall startup for the full kLeReadyTimeout before the LE
// path is waived. A working fabric converges in well under a second, so a
// small budget fails fast without false-negatives on healthy devices.
inline constexpr auto kLeProbeReadyTimeout = std::chrono::seconds(2);

inline bool probeLogicalEndpointBind(CUdevice dev, int deviceId);

// Some toolkits ship the LE API structs but not these device-attribute
// enumerators yet; fall back to their numeric values (cuDeviceGetAttribute
// rejects unknown attributes at runtime, so this stays safe on old drivers).
#ifdef NVBW_HAVE_LE_DEVICE_ATTRS
inline constexpr CUdevice_attribute kLeUnicastSupportedAttr =
    CU_DEVICE_ATTRIBUTE_LOGICAL_ENDPOINT_UNICAST_SUPPORTED;
inline constexpr CUdevice_attribute kLeSupportedHandleTypesAttr =
    CU_DEVICE_ATTRIBUTE_LOGICAL_ENDPOINT_SUPPORTED_HANDLE_TYPES;
#else
inline constexpr CUdevice_attribute kLeUnicastSupportedAttr = (CUdevice_attribute)153;
inline constexpr CUdevice_attribute kLeSupportedHandleTypesAttr = (CUdevice_attribute)158;
#endif
// Same-device / loopback unicast LE access (GFM hook). Always use the numeric
// value: the enumerator may lag the LE API structs the same way 153/158 do,
// and enum members are not preprocessor macros for #ifdef.
inline constexpr CUdevice_attribute kLeUnicastAccessOnOwnerDeviceSupportedAttr =
    (CUdevice_attribute)156;  // CU_DEVICE_ATTRIBUTE_LOGICAL_ENDPOINT_UNICAST_ACCESS_ON_OWNER_DEVICE_SUPPORTED

// Check if a device supports Logical Endpoint with fabric handles (Blackwell+ only).
inline bool supportsLogicalEndpoint(int deviceId) {
    CUdevice dev;
    if (cuDeviceGet(&dev, deviceId) != CUDA_SUCCESS) {
        return false;
    }
    int major = 0;
    if (cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev) != CUDA_SUCCESS) {
        return false;
    }
    // Fabric/LE handles are available on Blackwell+ (sm_100), not on Hopper (sm_90).
    if (major < 10) {
        return false;
    }

    // Query unicast-LE support; the driver gates this on FLA-mappable memory,
    // nvswitchPeerCount > 0, and sufficient UCFLA heap.
    int unicastSupported = 0;
    if (cuDeviceGetAttribute(&unicastSupported,
                             kLeUnicastSupportedAttr, dev) != CUDA_SUCCESS
            || !unicastSupported) {
        return false;
    }

    // Confirm the FABRIC IPC handle type is actually usable. Querying this
    // attribute also drives the IMEX channel subscribe in the driver.
    int handleTypes = 0;
    if (cuDeviceGetAttribute(&handleTypes,
                             kLeSupportedHandleTypesAttr, dev) != CUDA_SUCCESS
            || (handleTypes & CU_LOGICAL_ENDPOINT_IPC_HANDLE_TYPE_FABRIC) == 0) {
        return false;
    }

    // Attributes can over-report, so confirm with a real create->ready->bind probe.
    return probeLogicalEndpointBind(dev, deviceId);
}

// Run the full unicast-LE path once (create -> ready -> bind) on a small fabric
// allocation, then tear it down. Returns false on failure.
inline bool probeLogicalEndpointBind(CUdevice dev, int deviceId) {
    CUcontext ctx = nullptr;
    if (cuDevicePrimaryCtxRetain(&ctx, dev) != CUDA_SUCCESS) {
        return false;
    }
    if (cuCtxSetCurrent(ctx) != CUDA_SUCCESS) {
        cuDevicePrimaryCtxRelease(dev);
        return false;
    }

    bool ready = false;

    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = deviceId;
    prop.requestedHandleTypes = (CUmemAllocationHandleType)(CU_MEM_HANDLE_TYPE_FABRIC);

    size_t granularity = 0;
    if (cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED) == CUDA_SUCCESS
            && granularity > 0) {
        CUmemGenericAllocationHandle handle = 0;
        if (cuMemCreate(&handle, granularity, &prop, 0) == CUDA_SUCCESS) {
            CUlogicalEndpointId leId = ~0u;  // let the driver pick
            if (cuLogicalEndpointIdReserve(&leId, 1) == CUDA_SUCCESS) {
                CUlogicalEndpointProp leProp = {};
                leProp.type = CU_LOGICAL_ENDPOINT_TYPE_UNICAST;
                leProp.size = granularity;  // must be the granularity-aligned size
                leProp.unicast.device = dev;
                leProp.ipcHandleTypes = CU_LOGICAL_ENDPOINT_IPC_HANDLE_TYPE_FABRIC;
                leProp.flags = CU_LOGICAL_ENDPOINT_FLAG_NONE;

                if (cuLogicalEndpointCreate(leId, &leProp) == CUDA_SUCCESS) {
                    // Query the endpoint to the ready state before binding memory to it.
                    // Use the short probe timeout so a non-converging device fails fast.
                    if (waitLeReadyWithTimeout(leId, kLeProbeReadyTimeout)
                            && cuLogicalEndpointBindMem(leId, dev, 0, handle, 0, granularity, 0) == CUDA_SUCCESS) {
                        ready = true;
                        cuLogicalEndpointUnbind(leId, dev, 0, granularity);
                    }
                    cuLogicalEndpointDestroy(leId);
                }
                cuLogicalEndpointIdRelease(leId, 1);
            }
            cuMemRelease(handle);
        }
    }

    cuDevicePrimaryCtxRelease(dev);
    return ready;
}

// Check if all devices support LE buffers (for test filtering).
// If any device lacks LE support (e.g. Hopper in a mixed system), the whole
// test falls back to plain device pointers so the result matrix is uniform.
// Result is cached after the first call since the probe involves a real allocation.
inline bool leBufferSupportedOnAllDevices() {
    static int cachedAllDevicesResult = -1;  // -1 = not yet computed, 0 = false, 1 = true
    if (cachedAllDevicesResult != -1) return cachedAllDevicesResult;

    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        cachedAllDevicesResult = 0;
        return false;
    }
    for (int i = 0; i < deviceCount; i++) {
        if (!supportsLogicalEndpoint(i)) {
            cachedAllDevicesResult = 0;
            return false;
        }
    }
    cachedAllDevicesResult = 1;
    return true;
}

// True when every device reports non-zero
// CU_DEVICE_ATTRIBUTE_LOGICAL_ENDPOINT_UNICAST_ACCESS_ON_OWNER_DEVICE_SUPPORTED.
// Required for same-device (loopback) LE copies; cross-device / peer LE does not
// need this. Non-zero values are treated as supported (drivers may return a
// capability mask, e.g. 4, rather than a boolean 1).
inline bool leOwnerDeviceAccessSupportedOnAllDevices() {
    static int cached = -1;
    if (cached != -1) return cached;

    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        cached = 0;
        return false;
    }
    for (int i = 0; i < deviceCount; i++) {
        CUdevice dev;
        if (cuDeviceGet(&dev, i) != CUDA_SUCCESS) {
            cached = 0;
            return false;
        }
        int supported = 0;
        if (cuDeviceGetAttribute(&supported,
                                 kLeUnicastAccessOnOwnerDeviceSupportedAttr,
                                 dev) != CUDA_SUCCESS ||
            supported == 0) {
            cached = 0;
            return false;
        }
    }
    cached = 1;
    return true;
}

// Retain and set CUDA context for a device
inline void retainAndSetContext(CUcontext* ctx, int deviceId) {
    CU_ASSERT(cuDevicePrimaryCtxRetain(ctx, deviceId));
    CU_ASSERT(cuCtxSetCurrent(*ctx));
}

// Release CUDA context for a device
inline void releaseContext(int deviceId) {
    cuDevicePrimaryCtxRelease(deviceId);
}

// ============================================================================
// Memory Allocation with Fabric Handles
// ============================================================================

// memoryNodeId < 0: ordinary device allocation.
// memoryNodeId >= 0: pin to that uGPU locality domain (requires
// NVBW_HAVE_UGPU_MEMORY_NODE_SPLIT).
inline CUdeviceptr allocateAndMapMemory(CUmemGenericAllocationHandle* handle, size_t size, int deviceId,
                                        bool useFabricHandles, size_t* outMappedSize = nullptr,
                                        int memoryNodeId = -1) {
    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    if (memoryNodeId >= 0) {
#ifdef NVBW_HAVE_UGPU_MEMORY_NODE_SPLIT
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE_LOCALITY_DOMAIN;
        prop.location.localized.deviceId =
            static_cast<unsigned char>(deviceId);
        prop.location.localized.localityDomainId =
            static_cast<unsigned char>(memoryNodeId);
#else
        ASSERT(false && "locality-domain allocation requires UGPU memory-node split support");
#endif
    } else {
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id = deviceId;
    }

    if (useFabricHandles) {
        prop.requestedHandleTypes = (CUmemAllocationHandleType)(CU_MEM_HANDLE_TYPE_FABRIC);
    }

    // Get granularity
    size_t granularity = 0;
    CU_ASSERT(cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));

    // Round up size to granularity
    size_t alignedSize = ((size + granularity - 1) / granularity) * granularity;

    // Allocate memory
    CU_ASSERT(cuMemCreate(handle, alignedSize, &prop, 0));

    // Reserve VA range
    CUdeviceptr ptr = 0;
    CU_ASSERT(cuMemAddressReserve(&ptr, alignedSize, granularity, 0, 0));

    // Map the allocation
    CU_ASSERT(cuMemMap(ptr, alignedSize, 0, *handle, 0));

    // Set access. Grant RW to the owner plus every peer that can reach it, not just
    // the owner: fabric-backed LE buffers are peer-accessed by kernels on other
    // devices (e.g. the D2D latency consumer peer-reads the producer-owned LE
    // buffer).
    int accessDevCount = 1;
    CU_ASSERT(cuDeviceGetCount(&accessDevCount));
    std::vector<CUmemAccessDesc> accessDescs;
    accessDescs.reserve(accessDevCount);
    for (int d = 0; d < accessDevCount; d++) {
        if (d != deviceId) {
            int canAccess = 0;
            if (cuDeviceCanAccessPeer(&canAccess, d, deviceId) != CUDA_SUCCESS || !canAccess)
                continue;
        }
        CUmemAccessDesc desc = {};
        desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        desc.location.id = d;
        desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        accessDescs.push_back(desc);
    }
    CU_ASSERT(cuMemSetAccess(ptr, alignedSize, accessDescs.data(), accessDescs.size()));

    if (outMappedSize) *outMappedSize = alignedSize;
    return ptr;
}

inline void unmapAndFreeMemory(CUdeviceptr ptr, CUmemGenericAllocationHandle handle, size_t size) {
    if (ptr) {
        cuMemUnmap(ptr, size);
        cuMemAddressFree(ptr, size);
    }
    if (handle) {
        cuMemRelease(handle);
    }
}

// Wait for LE to be ready with a timeout. Returns true if ready, false on timeout.
inline bool waitLeReadyWithTimeout(CUlogicalEndpointId leId, std::chrono::seconds timeout) {
    auto deadline = std::chrono::steady_clock::now() + timeout;
    while (std::chrono::steady_clock::now() < deadline) {
        int is_ready = 0;
        CUresult res = cuLogicalEndpointQuery(leId, 1, &is_ready);
        if (res != CUDA_SUCCESS) return false;
        if (is_ready) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return false;
}

// After cuLogicalEndpointBindMem, fabric route programming continues
// asynchronously: the driver returns before the switch is programmed, and work
// launched in that window fails with cudaErrorDevicesUnavailable. Spin on a
// trivial device op until the device accepts work, mirroring the multinode
// fabric paths. cuLogicalEndpointQuery reaching the ready state is a separate,
// pre-bind gate and does not cover this post-bind window.
//
// Returns cudaSuccess once the device accepts work, or the last cudaMemset
// error if it never does within the retry budget so the caller can waive the LE
// path (like the other setup failures) and name the actual error instead of
// aborting the whole run. This differs from the multinode
// waitForFabricProgramming() (multinode_memcpy.cpp), which asserts on a stall:
// single-node has no collective barrier, so waiving here is safe.
inline cudaError_t waitForLeFabricProgramming() {
    constexpr int kMaxRetries = 1000;
    constexpr auto kRetryDelay = std::chrono::milliseconds(5);
    CUdeviceptr scratch = 0;
    CU_ASSERT(cuMemAlloc(&scratch, sizeof(int)));
    cudaError_t err = cudaErrorDevicesUnavailable;
    for (int i = 0; i < kMaxRetries && err == cudaErrorDevicesUnavailable; ++i) {
        cudaGetLastError();
        err = cudaMemset(reinterpret_cast<void*>(scratch), 0, sizeof(int));
        if (err == cudaErrorDevicesUnavailable)
            std::this_thread::sleep_for(kRetryDelay);
    }
    // Drain the thread-local last-error so a waived path cannot trip a later
    // CUDA_ASSERT(cudaGetLastError()) launch check in an unrelated test.
    cudaGetLastError();
    CU_ASSERT(cuMemFree(scratch));
    return err;
}

// Wait for fabric programming on the device that will launch work against a
// peer LE. Shared by LEBufferStrategy, single-node localized paths, and
// multinode localized concurrent (after LE readiness is already confirmed).
// Restores the caller's CUDA context. Returns false (and records a warning)
// on a persistent stall so callers can skip the pair rather than abort.
inline bool enableLeExecutingDeviceAccess(CUcontext execCtx, int execDeviceIdx,
                                          int peerDeviceIdx) {
    if (execCtx == nullptr) return true;
    CUcontext prevCtx = nullptr;
    CU_ASSERT(cuCtxGetCurrent(&prevCtx));
    CU_ASSERT(cuCtxSetCurrent(execCtx));
    cudaError_t execErr = waitForLeFabricProgramming();
    // Restore unconditionally; cuCtxSetCurrent(nullptr) detaches the context,
    // which is the correct state to restore when there was no prior context.
    CU_ASSERT(cuCtxSetCurrent(prevCtx));
    if (execErr != cudaSuccess) {
        output->recordWarning(
            std::string("LE: executing device ")
            + std::to_string(execDeviceIdx)
            + " did not accept work for the LE on device "
            + std::to_string(peerDeviceIdx)
            + " (" + cudaGetErrorString(execErr) + "); skipping pair");
        return false;
    }
    return true;
}

// ============================================================================
// LogicalEndpointBuffer: LE buffer for P2P tests
// ============================================================================

class LogicalEndpointBuffer : public MemcpyBuffer {
 private:
    CUlogicalEndpointId leId;
    CUdeviceptr devicePtr;
    size_t mappedSize;
    int deviceIdx;
    int memoryNodeId;  // -1 = device-wide; >=0 = uGPU locality domain
    CUdevice cuDevice;
    CUcontext primaryCtx;
    CUmemGenericAllocationHandle memHandle;
    bool isCreated;

    // Reset to a uniformly "dead" state after an LE setup failure (caller must
    // have already freed/released the memory and LE id). getLeId()/getBuffer()/
    // getDevicePtr() then all report unusable, so dispatch never touches a freed
    // VA / released id (the CUDA_ERROR_ILLEGAL_ADDRESS). isCreated stays false.
    void markUnusable() {
        devicePtr = 0;
        memHandle = 0;
        leId = LE_ID_NONE;
        buffer = nullptr;
        // Error paths release the context explicitly; null it so the destructor's
        // `if (primaryCtx) releaseContext(...)` doesn't release a second time.
        primaryCtx = nullptr;
    }

 public:
    // memoryNodeId < 0: ordinary fabric-backed LE (device-wide placement).
    // memoryNodeId >= 0: pin the LE's backing allocation to that uGPU locality
    // domain so localized copies can be routed to the matching LCE.
    LogicalEndpointBuffer(size_t size, int deviceId, int memoryNodeIdArg = -1)
        : MemcpyBuffer(size),
          leId(0),
          devicePtr(0),
          mappedSize(0),
          deviceIdx(deviceId),
          memoryNodeId(memoryNodeIdArg),
          cuDevice(0),
          primaryCtx(nullptr),
          memHandle(0),
          isCreated(false) {
        // Set device and get context
        CUDA_ASSERT(cudaSetDevice(deviceIdx));
        retainAndSetContext(&primaryCtx, deviceIdx);

        // Reserve LE ID
        leId = ~0u;  // Let driver pick
        CU_ASSERT(cuLogicalEndpointIdReserve(&leId, 1));

        // Get CUdevice handle for the device first (needed for prop and bind)
        CU_ASSERT(cuDeviceGet(&cuDevice, deviceIdx));

        // Allocate FABRIC-backed memory, optionally locality-domain pinned.
        devicePtr = allocateAndMapMemory(&memHandle, size, deviceIdx, true /* useFabricHandles */,
                                         &mappedSize, memoryNodeId);
        buffer = (void*)devicePtr;

        // Create LE properties; prop.size must be the granularity-aligned size, not the
        // caller-requested size — passing the raw size causes CUDA_ERROR_INVALID_VALUE when
        // the buffer is smaller than the fabric granularity (e.g. 64KB < 2MB).
        CUlogicalEndpointProp prop = {};
        prop.type = CU_LOGICAL_ENDPOINT_TYPE_UNICAST;
        prop.size = mappedSize;
        prop.unicast.device = cuDevice;  // Use CUdevice, not int deviceIdx
        prop.ipcHandleTypes = CU_LOGICAL_ENDPOINT_IPC_HANDLE_TYPE_FABRIC;
        prop.flags = CU_LOGICAL_ENDPOINT_FLAG_NONE;

        // Create LE — non-fatal: fabric may not be configured on all systems.
        // NOTE: cuLogicalEndpointAddDevice is for multicast LEs to add secondary devices.
        // For unicast LEs, the device is already set via prop.unicast.device.
        CUresult leCreateResult = cuLogicalEndpointCreate(leId, &prop);
        if (leCreateResult != CUDA_SUCCESS) {
            output->recordWarning("LogicalEndpointBuffer: cuLogicalEndpointCreate failed "
                                  "(fabric not configured?); skipping LE path");
            cuLogicalEndpointIdRelease(leId, 1);
            unmapAndFreeMemory(devicePtr, memHandle, mappedSize);
            markUnusable();
            releaseContext(deviceIdx);
            return;
        }

        // Query the endpoint to the ready state before binding memory to it.
        if (!waitLeReadyWithTimeout(leId, kLeReadyTimeout)) {
            output->recordError("LogicalEndpointBuffer: LE did not become ready within timeout");
            cuLogicalEndpointDestroy(leId);
            cuLogicalEndpointIdRelease(leId, 1);
            unmapAndFreeMemory(devicePtr, memHandle, mappedSize);
            markUnusable();
            releaseContext(deviceIdx);
            return;
        }

        // Bind memory to LE at offset 0; size must be the granularity-aligned mappedSize.
        CUresult leBindResult = cuLogicalEndpointBindMem(leId, cuDevice, 0, memHandle, 0, mappedSize, 0);
        if (leBindResult != CUDA_SUCCESS) {
            output->recordWarning("LogicalEndpointBuffer: cuLogicalEndpointBindMem failed "
                                  "(fabric not configured?); skipping LE path");
            cuLogicalEndpointDestroy(leId);
            cuLogicalEndpointIdRelease(leId, 1);
            unmapAndFreeMemory(devicePtr, memHandle, mappedSize);
            markUnusable();
            releaseContext(deviceIdx);
            return;
        }

        // Bind kicks off asynchronous fabric route programming; wait until this
        // (owner) device accepts work. The fabric kernels actually launch from
        // the non-LE executing device (selectExecutingBuffer), which is the one
        // that hits the per-device cudaErrorDevicesUnavailable window at first
        // launch, so LEBufferStrategy::enablePeerAccess waits on that device too.
        // A persistent stall here waives this device's LE path like the other
        // setup failures rather than aborting the run. Memory is bound at this
        // point, so unbind before tearing down the LE id and allocation.
        cudaError_t fabricErr = waitForLeFabricProgramming();
        if (fabricErr != cudaSuccess) {
            output->recordWarning(std::string("LogicalEndpointBuffer: device did not accept work after bind (")
                                  + cudaGetErrorString(fabricErr) + "); skipping LE path");
            cuLogicalEndpointUnbind(leId, cuDevice, 0, mappedSize);
            cuLogicalEndpointDestroy(leId);
            cuLogicalEndpointIdRelease(leId, 1);
            unmapAndFreeMemory(devicePtr, memHandle, mappedSize);
            markUnusable();
            releaseContext(deviceIdx);
            return;
        }

        isCreated = true;
    }

    ~LogicalEndpointBuffer() override {
        if (isCreated) {
            // Unbind memory (leId, device, offset, size)
            cuLogicalEndpointUnbind(leId, cuDevice, 0, mappedSize);
            // Destroy LE
            cuLogicalEndpointDestroy(leId);
            // Release LE ID
            cuLogicalEndpointIdRelease(leId, 1);
        }

        // Free memory
        unmapAndFreeMemory(devicePtr, memHandle, mappedSize);

        if (primaryCtx) {
            releaseContext(deviceIdx);
        }
    }

    // Accessors
    int getBufferIdx() const override { return deviceIdx; }
    CUcontext getPrimaryCtx() const override { return primaryCtx; }
    std::string getBufferString() const override {
        std::string s = "LE_GPU_" + std::to_string(deviceIdx);
        if (memoryNodeId >= 0)
            s += "_Node" + std::to_string(memoryNodeId);
        return s;
    }

    // Only report a usable LE id once the create/ready/bind sequence succeeded.
    // A failed setup leaves leId == LE_ID_NONE via markUnusable(), but guarding on
    // isCreated is belt-and-suspenders against any partially-initialized state.
    uint32_t getLeId() const override { return isCreated ? leId : LE_ID_NONE; }
    CUdeviceptr getDevicePtr() const override { return devicePtr; }

    // True only when the LE was fully created, bound, and reached the ready state.
    bool isReady() const { return isCreated; }
};

// Skip if the LE failed create/bind; otherwise wait for fabric programming on
// the executing (non-LE) device before the first copy. Defined after
// LogicalEndpointBuffer because it needs the complete type.
inline bool enableLePairAccess(const MemcpyBuffer& execBuf,
                               LogicalEndpointBuffer& peerLE) {
    if (!peerLE.isReady()) return false;
    return enableLeExecutingDeviceAccess(execBuf.getPrimaryCtx(),
                                         execBuf.getBufferIdx(),
                                         peerLE.getBufferIdx());
}

// Memset all locally-accessible buffers in an LE dispatch.
// Buffers on the remote side (LE importers) have getDevicePtr()==0 and are skipped.
inline void memsetPatternLE(MemcpyDispatchInfo &info) {
    for (size_t i = 0; i < info.srcBuffers.size(); i++) {
        const MemcpyBuffer* srcBuf = info.srcBuffers[i];
        const MemcpyBuffer* dstBuf = info.dstBuffers[i];
        if (srcBuf->getDevicePtr() != 0) {
            CUDA_ASSERT(cudaSetDevice(srcBuf->getBufferIdx()));
            CUDA_ASSERT(cudaMemset((void*)srcBuf->getDevicePtr(), 0xAB, srcBuf->getBufferSize()));
        }
        if (dstBuf->getDevicePtr() != 0) {
            CUDA_ASSERT(cudaSetDevice(dstBuf->getBufferIdx()));
            CUDA_ASSERT(cudaMemset((void*)dstBuf->getDevicePtr(), 0, dstBuf->getBufferSize()));
        }
    }
    std::set<int> devicesToSync;
    for (size_t i = 0; i < info.srcBuffers.size(); i++) {
        devicesToSync.insert(info.srcBuffers[i]->getBufferIdx());
    }
    for (size_t i = 0; i < info.dstBuffers.size(); i++) {
        devicesToSync.insert(info.dstBuffers[i]->getBufferIdx());
    }
    for (int dev : devicesToSync) {
        CUDA_ASSERT(cudaSetDevice(dev));
        CUDA_ASSERT(cudaDeviceSynchronize());
    }
}

#else  // !NVBW_HAVE_LE_API -----------------------------------------------
// Stubs for builds against older CUDA driver headers that lack the LE API.
// leBufferSupportedOnAllDevices() returns false so all LE tests are filtered out.

inline bool supportsLogicalEndpoint(int) { return false; }
inline bool leBufferSupportedOnAllDevices() { return false; }
inline bool leOwnerDeviceAccessSupportedOnAllDevices() { return false; }

// Minimal stub so that LEBufferStrategy::createBufferPair compiles;
// it is never reached because isSupported() returns false.
class LogicalEndpointBuffer : public MemcpyBuffer {
 public:
    LogicalEndpointBuffer(size_t size, int, int = -1) : MemcpyBuffer(size) {}
    int getBufferIdx() const override { return 0; }
    CUcontext getPrimaryCtx() const override { return nullptr; }
    std::string getBufferString() const override { return "LE_UNAVAIL"; }
    bool isReady() const { return false; }
};

inline void memsetPatternLE(MemcpyDispatchInfo &) {}

#endif  // NVBW_HAVE_LE_API

#endif  // MEMCPY_LE_H_
