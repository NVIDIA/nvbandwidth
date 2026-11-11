/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifdef MULTINODE
#include <algorithm>
#include <mpi.h>
#include <set>
#include <random>
#include <unistd.h>

#include "kernels.cuh"
#include "memcpy_le.h"
#include "multinode_memcpy.h"

// ============================================================================
// Timing and retry constants
// ============================================================================
static constexpr int kFabricMaxRetries                     = 1000;
static constexpr auto kFabricRetryDelay                    = std::chrono::milliseconds(5);
static constexpr auto kLeUnicastReadyTimeout               = std::chrono::seconds(15);
static constexpr auto kLeMulticastReadyTimeout             = std::chrono::seconds(30);

// fabric route programming is asynchronous — the driver returns before the switch is
// done, and kernel launches during that window fail with cudaErrorDevicesUnavailable.
// Call this helper (once, after the final MPI_Barrier of any fabric setup sequence)
// to spin until the device accepts work.
//
// Failure model: this asserts (aborts the run) on a persistent stall, unlike the
// single-node LogicalEndpointBuffer path (memcpy_le.h) which soft-waives the LE path
// and keeps running. That is intentional — every rank calls this in lockstep after a
// collective barrier, so one rank can't unilaterally waive without deadlocking the
// others; a soft path here would need a collective agreement on the outcome first.
static void waitForFabricProgramming() {
    CUdeviceptr scratch = 0;
    CU_ASSERT(cuMemAlloc(&scratch, sizeof(int)));
    cudaError_t err = cudaErrorDevicesUnavailable;
    for (int i = 0; i < kFabricMaxRetries && err == cudaErrorDevicesUnavailable; ++i) {
        cudaGetLastError();
        err = cudaMemset(reinterpret_cast<void*>(scratch), 0, sizeof(int));
        if (err == cudaErrorDevicesUnavailable)
            std::this_thread::sleep_for(kFabricRetryDelay);
    }
    CU_ASSERT(cuMemFree(scratch));
    if (err != cudaSuccess)
        CUDA_ASSERT(err);
}

MultinodeMemoryAllocation::MultinodeMemoryAllocation(size_t bufferSize, int MPI_rank): bufferSize(bufferSize), MPI_rank(MPI_rank) {
    CUDA_ASSERT(cudaSetDevice(localDevice));
}

static CUresult MPIstreamSyncHelper(CUstream stream) {
    CUresult err = CUDA_ERROR_NOT_READY;
    int flag;
    while (err == CUDA_ERROR_NOT_READY) {
        err = cuStreamQuery(stream);
        MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, MPI_STATUS_IGNORE);
    }
    return err;
}

CUresult MultinodeMemoryAllocation::streamSynchronizeWrapper(CUstream stream) const {
    return MPIstreamSyncHelper(stream);
}

MultinodeMemoryAllocationUnicast::MultinodeMemoryAllocationUnicast(size_t bufferSize, int MPI_rank): MultinodeMemoryAllocation(bufferSize, MPI_rank) {
    handleType = CU_MEM_HANDLE_TYPE_FABRIC;
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = localDevice;
    prop.requestedHandleTypes = handleType;

    size_t granularity = 0;
    CU_ASSERT(cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));

    roundedUpAllocationSize = ROUND_UP(bufferSize, granularity);

    if (MPI_rank == worldRank) {
        // Allocate the memory
        CU_ASSERT(cuMemCreate(&handle, roundedUpAllocationSize, &prop, 0 /*flags*/));

        // Export the allocation to the importing process
        CU_ASSERT(cuMemExportToShareableHandle(&fh, handle, handleType, 0 /*flags*/));
    }

    MPI_Bcast(&fh, sizeof(fh), MPI_BYTE, MPI_rank, MPI_COMM_WORLD);

    if (MPI_rank != worldRank) {
        CU_ASSERT(cuMemImportFromShareableHandle(&handle, (void *)&fh, handleType));
    }

    // Map the memory
    CU_ASSERT(cuMemAddressReserve((CUdeviceptr *) &buffer, roundedUpAllocationSize, 0, 0 /*baseVA*/, 0 /*flags*/));

    CU_ASSERT(cuMemMap((CUdeviceptr) buffer, roundedUpAllocationSize, 0 /*offset*/, handle, 0 /*flags*/));
    desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    desc.location.id = localDevice;
    desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    CU_ASSERT(cuMemSetAccess((CUdeviceptr) buffer, roundedUpAllocationSize, &desc, 1 /*count*/));

    // Make sure that everyone is done with mapping the fabric allocation
    MPI_Barrier(MPI_COMM_WORLD);
    waitForFabricProgramming();
}

MultinodeMemoryAllocationUnicast::~MultinodeMemoryAllocationUnicast() {
    // Make sure that everyone is done using the memory
    MPI_Barrier(MPI_COMM_WORLD);

    CU_ASSERT(cuMemUnmap((CUdeviceptr) buffer, roundedUpAllocationSize));
    CU_ASSERT(cuMemRelease(handle));
    CU_ASSERT(cuMemAddressFree((CUdeviceptr) buffer, roundedUpAllocationSize));
}

#ifdef NVBW_HAVE_UGPU_MEMORY_NODE_SPLIT
MultinodeLocalizedMemoryAllocationUnicast::MultinodeLocalizedMemoryAllocationUnicast(
        size_t bufferSize, int MPI_rank, unsigned int memoryNodeId)
    : MultinodeMemoryAllocation(bufferSize, MPI_rank) {
    constexpr CUmemAllocationHandleType handleType = CU_MEM_HANDLE_TYPE_FABRIC;

    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE_LOCALITY_DOMAIN;
    prop.location.localized.deviceId         = static_cast<unsigned char>(localDevice);
    prop.location.localized.localityDomainId = static_cast<unsigned char>(memoryNodeId);
    prop.requestedHandleTypes = handleType;

    size_t granularity = 0;
    CU_ASSERT(cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
    roundedUpAllocationSize_ = ROUND_UP(bufferSize, granularity);

    if (MPI_rank == worldRank) {
        CU_ASSERT(cuMemCreate(&handle_, roundedUpAllocationSize_, &prop, 0));
        CU_ASSERT(cuMemExportToShareableHandle(&fh_, handle_, handleType, 0));
    }

    MPI_Bcast(&fh_, sizeof(fh_), MPI_BYTE, MPI_rank, MPI_COMM_WORLD);

    if (MPI_rank != worldRank) {
        CU_ASSERT(cuMemImportFromShareableHandle(&handle_, (void*)&fh_, handleType));
    }

    CU_ASSERT(cuMemAddressReserve((CUdeviceptr*)&buffer, roundedUpAllocationSize_, 0, 0, 0));
    CU_ASSERT(cuMemMap((CUdeviceptr)buffer, roundedUpAllocationSize_, 0, handle_, 0));

    CUmemAccessDesc desc = {};
    desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    desc.location.id   = localDevice;
    desc.flags         = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    CU_ASSERT(cuMemSetAccess((CUdeviceptr)buffer, roundedUpAllocationSize_, &desc, 1));

    MPI_Barrier(MPI_COMM_WORLD);
    waitForFabricProgramming();
}

MultinodeLocalizedMemoryAllocationUnicast::~MultinodeLocalizedMemoryAllocationUnicast() {
    MPI_Barrier(MPI_COMM_WORLD);
    CU_ASSERT(cuMemUnmap((CUdeviceptr)buffer, roundedUpAllocationSize_));
    CU_ASSERT(cuMemRelease(handle_));
    CU_ASSERT(cuMemAddressFree((CUdeviceptr)buffer, roundedUpAllocationSize_));
}

MultinodeLocalizedDeviceBufferUnicast::MultinodeLocalizedDeviceBufferUnicast(
        size_t bufferSize, int MPI_rank, unsigned int memoryNodeId)
    : MultinodeDeviceBuffer(bufferSize, MPI_rank)
    , alloc_(bufferSize, MPI_rank, memoryNodeId) {
    buffer = alloc_.getBuffer();
}
#endif  // NVBW_HAVE_UGPU_MEMORY_NODE_SPLIT

MultinodeMemoryAllocationMulticast::MultinodeMemoryAllocationMulticast(size_t bufferSize, int MPI_rank): MultinodeMemoryAllocation(bufferSize, MPI_rank) {
    handleType = CU_MEM_HANDLE_TYPE_FABRIC;
    multicastProp.numDevices = worldSize;
    multicastProp.handleTypes = handleType;
    size_t gran;
    CU_ASSERT(cuMulticastGetGranularity(&gran, &multicastProp, CU_MULTICAST_GRANULARITY_RECOMMENDED));
    roundedUpAllocationSize = ROUND_UP(bufferSize, gran);
    multicastProp.size = roundedUpAllocationSize;

    if (MPI_rank == worldRank) {
        // Allocate the memory
        CU_ASSERT(cuMulticastCreate(&multicastHandle, &multicastProp));

        // Export the allocation to the importing process
        CU_ASSERT(cuMemExportToShareableHandle(&fh, multicastHandle, handleType, 0 /*flags*/));
    }

    MPI_Bcast(&fh, sizeof(fh), MPI_BYTE, MPI_rank, MPI_COMM_WORLD);

    if (MPI_rank != worldRank) {
        CU_ASSERT(cuMemImportFromShareableHandle(&multicastHandle, (void *)&fh, handleType));
    }

    CUdevice dev;
    CU_ASSERT(cuDeviceGet(&dev, localDevice));
    CU_ASSERT(cuMulticastAddDevice(multicastHandle, dev));

    // Ensure all devices in this process are added BEFORE binding mem on any device
    MPI_Barrier(MPI_COMM_WORLD);

    // Allocate the memory (same as unicast) and bind to MC handle
    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = localDevice;
    prop.requestedHandleTypes = handleType;
    CU_ASSERT(cuMemCreate(&handle, roundedUpAllocationSize, &prop, 0 /*flags*/));
    CU_ASSERT(cuMulticastBindMem(multicastHandle, 0, handle, 0, roundedUpAllocationSize, 0));

    // Map the memory
    CU_ASSERT(cuMemAddressReserve((CUdeviceptr *) &buffer, roundedUpAllocationSize, 0, 0 /*baseVA*/, 0 /*flags*/));

    CU_ASSERT(cuMemMap((CUdeviceptr) buffer, roundedUpAllocationSize, 0 /*offset*/, multicastHandle, 0 /*flags*/));
    desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    desc.location.id = localDevice;
    desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    CU_ASSERT(cuMemSetAccess((CUdeviceptr) buffer, roundedUpAllocationSize, &desc, 1 /*count*/));

    // Make sure that everyone is done with mapping the fabric allocation
    MPI_Barrier(MPI_COMM_WORLD);
    waitForFabricProgramming();
}

MultinodeMemoryAllocationMulticast::~MultinodeMemoryAllocationMulticast() {
    // Make sure that everyone is done using the memory
    MPI_Barrier(MPI_COMM_WORLD);

    CUdevice dev;
    CU_ASSERT(cuDeviceGet(&dev, localDevice));
    CU_ASSERT(cuMulticastUnbind(multicastHandle, dev, 0, roundedUpAllocationSize));
    CU_ASSERT(cuMemRelease(handle));

    CU_ASSERT(cuMemUnmap((CUdeviceptr) buffer, roundedUpAllocationSize));
    CU_ASSERT(cuMemRelease(multicastHandle));
    CU_ASSERT(cuMemAddressFree((CUdeviceptr) buffer, roundedUpAllocationSize));
}

MultinodeDeviceBuffer::MultinodeDeviceBuffer(size_t bufferSize, int MPI_rank):
    MPI_rank(MPI_rank),
    MemcpyBuffer(bufferSize) {
}

int MultinodeDeviceBuffer::getBufferIdx() const {
    return localDevice;
}

std::string MultinodeDeviceBuffer::getBufferString() const {
    return "Multinode node " + std::to_string(MPI_rank);
}

CUcontext MultinodeDeviceBuffer::getPrimaryCtx() const {
    CUcontext primaryCtx;
    CU_ASSERT(cuDevicePrimaryCtxRetain(&primaryCtx, localDevice));
    return primaryCtx;
}

int MultinodeDeviceBuffer::getMPIRank() const {
    return MPI_rank;
}

MultinodeDeviceBufferUnicast::MultinodeDeviceBufferUnicast(size_t bufferSize, int MPI_rank):
    MultinodeDeviceBuffer(bufferSize, MPI_rank),
    MemoryAllocation(bufferSize, MPI_rank) {
    buffer = MemoryAllocation.getBuffer();
}

// ============================================================================
// MultinodeLEBufferUnicast / MultinodeLEBufferMulticast  (LE API required)
// ============================================================================
#ifdef NVBW_HAVE_LE_API

// NOTE: LE buffer constructors use output->recordError() rather than CU_ASSERT/CUDA_ASSERT.
// CU_ASSERT calls exit() which kills one MPI rank while others block on MPI_Barrier,
// causing a deadlock.  Every error path here must reach its corresponding MPI_Barrier
// so all ranks can fail and clean up symmetrically.
MultinodeLEBufferUnicast::MultinodeLEBufferUnicast(size_t bufferSize, int MPI_rank,
                                                   int memoryNodeId)
    : MultinodeDeviceBuffer(bufferSize, MPI_rank)
    , leId_(0)
    , devicePtr_(0)
    , cuDevice_(0)
    , primaryCtx_(nullptr)
    , memHandle_(0)
    , mappedSize_(0)
    , isOwner_(MPI_rank == worldRank)
    , isCreated_(false)
    , memoryNodeId_(memoryNodeId) {
    CUDA_ASSERT(cudaSetDevice(localDevice));
    CU_ASSERT(cuDeviceGet(&cuDevice_, localDevice));
    CU_ASSERT(cuDevicePrimaryCtxRetain(&primaryCtx_, cuDevice_));

    if (isOwner_) {
        devicePtr_ = allocateAndMapMemory(&memHandle_, bufferSize, localDevice, true,
                                          &mappedSize_, memoryNodeId_);
        buffer = (void*)devicePtr_;

        CUlogicalEndpointProp prop = {};
        prop.type = CU_LOGICAL_ENDPOINT_TYPE_UNICAST;
        // Size the LE with the granularity-aligned mappedSize_; a sub-granularity
        // size (e.g. 64KB < 2MB) fails the bind with CUDA_ERROR_INVALID_VALUE and
        // cascades into every importer timing out in waitLeReadyWithTimeout().
        prop.size = mappedSize_;
        prop.unicast.device = cuDevice_;
        prop.ipcHandleTypes = CU_LOGICAL_ENDPOINT_IPC_HANDLE_TYPE_FABRIC;
        prop.flags = CU_LOGICAL_ENDPOINT_FLAG_NONE;

        CUlogicalEndpointFabricHandle leFh = {};
        bool leCreated = false, leBound = false;

        CU_ASSERT(cuLogicalEndpointIdReserve(&leId_, 1));

        if (cuLogicalEndpointCreate(leId_, &prop) != CUDA_SUCCESS) {
            output->recordError("MultinodeLEBufferUnicast: cuLogicalEndpointCreate failed");
            goto owner_cleanup;
        }
        leCreated = true;

        // Query the endpoint to the ready state before binding memory to it.
        if (!waitLeReadyWithTimeout(leId_, kLeUnicastReadyTimeout)) {
            output->recordError("MultinodeLEBufferUnicast: owner LE did not become ready within timeout");
            goto owner_cleanup;
        }

        if (cuLogicalEndpointBindMem(leId_, cuDevice_, 0, memHandle_, 0, mappedSize_, 0) != CUDA_SUCCESS) {
            output->recordError("MultinodeLEBufferUnicast: cuLogicalEndpointBindMem failed");
            goto owner_cleanup;
        }
        leBound = true;

        if (cuLogicalEndpointExport(&leFh, leId_, CU_LOGICAL_ENDPOINT_IPC_HANDLE_TYPE_FABRIC) != CUDA_SUCCESS) {
            output->recordError("MultinodeLEBufferUnicast: cuLogicalEndpointExport failed");
            goto owner_cleanup;
        }

        isCreated_ = true;
        goto owner_broadcast;

    owner_cleanup:
        if (leBound) {
            cuLogicalEndpointUnbind(leId_, cuDevice_, 0, mappedSize_);
        }
        if (leCreated) {
            cuLogicalEndpointDestroy(leId_);
        }
        cuLogicalEndpointIdRelease(leId_, 1);
        unmapAndFreeMemory(devicePtr_, memHandle_, mappedSize_);
        // Clear the base buffer pointer so a failed buffer (which reports
        // getLeId()==LE_ID_NONE) can't be memset against freed VA.
        devicePtr_ = 0; memHandle_ = 0; buffer = nullptr;

    owner_broadcast:
        // Broadcast real handle on success, zero handle on failure so importers don't deadlock.
        MPI_Bcast(&leFh, sizeof(leFh), MPI_BYTE, MPI_rank, MPI_COMM_WORLD);
    } else {
        CUlogicalEndpointFabricHandle leFh = {};
        MPI_Bcast(&leFh, sizeof(leFh), MPI_BYTE, MPI_rank, MPI_COMM_WORLD);

        CU_ASSERT(cuLogicalEndpointIdReserve(&leId_, 1));
        CUresult res = cuLogicalEndpointImport(leId_, &leFh, CU_LOGICAL_ENDPOINT_IPC_HANDLE_TYPE_FABRIC);
        if (res != CUDA_SUCCESS) {
            output->recordError("MultinodeLEBufferUnicast: importer failed to import LE fabric handle");
            cuLogicalEndpointIdRelease(leId_, 1);
            // Fall through to MPI_Barrier so other ranks don't deadlock.
        } else {
            bool ready = waitLeReadyWithTimeout(leId_, kLeUnicastReadyTimeout);
            if (!ready) {
                output->recordError("MultinodeLEBufferUnicast: importer LE did not become ready within timeout");
                cuLogicalEndpointDestroy(leId_);
                cuLogicalEndpointIdRelease(leId_, 1);
                // Fall through to MPI_Barrier so other ranks don't deadlock.
            } else {
                buffer = nullptr;
                isCreated_ = true;
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    waitForFabricProgramming();
}

MultinodeLEBufferUnicast::~MultinodeLEBufferUnicast() {
    MPI_Barrier(MPI_COMM_WORLD);

    if (isCreated_ && isOwner_) {
        cuLogicalEndpointUnbind(leId_, cuDevice_, 0, mappedSize_);
        cuLogicalEndpointDestroy(leId_);
        cuLogicalEndpointIdRelease(leId_, 1);
        unmapAndFreeMemory(devicePtr_, memHandle_, mappedSize_);
    } else if (isCreated_ && !isOwner_) {
        cuLogicalEndpointDestroy(leId_);
        cuLogicalEndpointIdRelease(leId_, 1);
    }

    if (primaryCtx_) {
        releaseContext(localDevice);
    }
}

uint32_t MultinodeLEBufferUnicast::getLeId() const {
    return isCreated_ ? leId_ : LE_ID_NONE;
}

CUdeviceptr MultinodeLEBufferUnicast::getDevicePtr() const {
    return devicePtr_;
}

// ============================================================================
// MultinodeLEBufferMulticast
// ============================================================================

// Lifecycle mirrors the single-node multicast flow and the canonical LE IPC reference:
//   Create (rank 0) → Export/Import → AddDevice → cuMemCreate → poll ready → BindMem
//   → wait for NVSwitch programming → kernels
// No VA mapping (cuMemMap/cuMemSetAccess) is done: kernels write and read via leId only.
MultinodeLEBufferMulticast::MultinodeLEBufferMulticast(size_t bufferSize, int /*deviceCount*/)
    : MultinodeDeviceBuffer(bufferSize, worldRank)
    , leId_(0)
    , cuDevice_(0)
    , primaryCtx_(nullptr)
    , memHandle_(0)
    , mappedSize_(0)
    , numRanks_(worldSize)
    , isCreated_(false) {
    CUDA_ASSERT(cudaSetDevice(localDevice));
    CU_ASSERT(cuDeviceGet(&cuDevice_, localDevice));
    CU_ASSERT(cuDevicePrimaryCtxRetain(&primaryCtx_, cuDevice_));
    CU_ASSERT(cuCtxSetCurrent(primaryCtx_));

    // Granularity-align the LE/allocation size up front so the owner's multicast LE
    // create, every rank's cuMemCreate, and the bind all use the same size; a
    // sub-granularity size fails the bind with CUDA_ERROR_INVALID_VALUE (the unicast
    // path hit the same root cause).
    {
        CUmemAllocationProp gprop = {};
        gprop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        gprop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        gprop.location.id = localDevice;
        gprop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC;
        size_t granularity = 0;
        if (cuMemGetAllocationGranularity(&granularity, &gprop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED) == CUDA_SUCCESS
                && granularity > 0) {
            mappedSize_ = ((bufferSize + granularity - 1) / granularity) * granularity;
        } else {
            mappedSize_ = bufferSize;
        }
    }

    const bool isOwner = (worldRank == 0);
    CUlogicalEndpointFabricHandle leFh = {};
    CU_ASSERT(cuLogicalEndpointIdReserve(&leId_, 1));

    // Track teardown state so the single cleanup block at the bottom
    // knows exactly what to undo.
    bool leHandleOk = false;  // LE object created (owner) or imported (non-owner)
    bool leAdded    = false;  // cuLogicalEndpointAddDevice succeeded
    bool memAlloced = false;  // cuMemCreate succeeded
    bool leBound    = false;  // cuLogicalEndpointBindMem succeeded

    // ---- Owner/non-owner setup: create+export or import the LE handle ----
    // MPI_Bcast is always called by both sides so neither blocks the other.
    if (isOwner) {
        CUlogicalEndpointProp prop = {};
        prop.type = CU_LOGICAL_ENDPOINT_TYPE_MULTICAST;
        prop.size = mappedSize_;
        prop.multicast.numDevices = (unsigned int)worldSize;
        prop.ipcHandleTypes = CU_LOGICAL_ENDPOINT_IPC_HANDLE_TYPE_FABRIC;
        prop.flags = CU_LOGICAL_ENDPOINT_FLAG_NONE;

        bool leCreated = false;
        CUresult res = cuLogicalEndpointCreate(leId_, &prop);
        if (res != CUDA_SUCCESS) {
            output->recordError("MultinodeLEBufferMulticast: cuLogicalEndpointCreate failed");
            goto owner_bcast;
        }
        leCreated = true;

        res = cuLogicalEndpointExport(&leFh, leId_, CU_LOGICAL_ENDPOINT_IPC_HANDLE_TYPE_FABRIC);
        if (res != CUDA_SUCCESS) {
            output->recordError("MultinodeLEBufferMulticast: cuLogicalEndpointExport failed");
            goto owner_bcast;
        }
        leHandleOk = true;

    owner_bcast:
        // Broadcast real handle on success, zero handle on failure so importers don't deadlock.
        MPI_Bcast(&leFh, sizeof(leFh), MPI_BYTE, 0, MPI_COMM_WORLD);
        if (!leHandleOk && leCreated) {
            cuLogicalEndpointDestroy(leId_);
        }
    } else {
        MPI_Bcast(&leFh, sizeof(leFh), MPI_BYTE, 0, MPI_COMM_WORLD);

        const CUlogicalEndpointFabricHandle zeroFh = {};
        if (memcmp(&leFh, &zeroFh, sizeof(leFh)) == 0) {
            output->recordError("MultinodeLEBufferMulticast: owner broadcast zero handle, aborting");
        } else {
            CUresult res = cuLogicalEndpointImport(leId_, &leFh, CU_LOGICAL_ENDPOINT_IPC_HANDLE_TYPE_FABRIC);
            if (res != CUDA_SUCCESS) {
                output->recordError("MultinodeLEBufferMulticast: cuLogicalEndpointImport failed");
            } else {
                leHandleOk = true;
            }
        }
    }

    // ---- Shared phase: AddDevice → alloc → ready-poll → BindMem ----
    // Every rank calls every MPI_Barrier regardless of failure so no rank
    // deadlocks waiting for a peer that already returned early.

    if (leHandleOk) {
        CUresult res = cuLogicalEndpointAddDevice(leId_, cuDevice_);
        if (res != CUDA_SUCCESS) {
            output->recordError("MultinodeLEBufferMulticast: cuLogicalEndpointAddDevice failed");
        } else {
            leAdded = true;
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);  // all ranks must have called AddDevice before any polls

    if (leAdded) {
        CUmemAllocationProp prop = {};
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id = localDevice;
        prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC;
        // mappedSize_ was granularity-aligned up front (shared with prop.size and bind).
        CUresult res = cuMemCreate(&memHandle_, mappedSize_, &prop, 0);
        if (res != CUDA_SUCCESS) {
            output->recordError("MultinodeLEBufferMulticast: cuMemCreate failed");
        } else {
            memAlloced = true;
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);  // all ranks must have allocated before any binds

    bool leReady = false;
    if (memAlloced) {
        bool ready = waitLeReadyWithTimeout(leId_, kLeMulticastReadyTimeout);
        if (!ready) {
            output->recordError("MultinodeLEBufferMulticast: LE not ready after AddDevice (timeout)");
        } else {
            leReady = true;
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);  // all ranks must be ready before binding

    if (leReady) {
        CUresult res = cuLogicalEndpointBindMem(leId_, cuDevice_, 0, memHandle_, 0, mappedSize_, 0);
        if (res != CUDA_SUCCESS) {
            output->recordError("MultinodeLEBufferMulticast: cuLogicalEndpointBindMem failed");
        } else {
            leBound = true;
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);  // all ranks must have finished BindMem before any proceeds

    if (leBound) {
        waitForFabricProgramming();
        buffer = (void*)(uintptr_t)leId_;
        isCreated_ = true;
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // ---- Single cleanup point on any failure ----
    if (!isCreated_) {
        if (leBound)    cuLogicalEndpointUnbind(leId_, cuDevice_, 0, mappedSize_);
        if (leHandleOk) cuLogicalEndpointDestroy(leId_);
        if (memAlloced) cuMemRelease(memHandle_);
        cuLogicalEndpointIdRelease(leId_, 1);
        leId_ = 0;
        memHandle_ = 0;
    }
}

MultinodeLEBufferMulticast::~MultinodeLEBufferMulticast() {
    MPI_Barrier(MPI_COMM_WORLD);
    if (isCreated_) {
        cuLogicalEndpointUnbind(leId_, cuDevice_, 0, mappedSize_);
        cuLogicalEndpointDestroy(leId_);
        cuLogicalEndpointIdRelease(leId_, 1);
        if (memHandle_) cuMemRelease(memHandle_);
    } else if (leId_ != 0) {
        cuLogicalEndpointIdRelease(leId_, 1);
    }
    if (primaryCtx_) {
        releaseContext(localDevice);
    }
}

bool MultinodeLEBufferMulticast::isMulticastBuffer() const {
    return isCreated_;
}

uint32_t MultinodeLEBufferMulticast::getLeId() const {
    return isCreated_ ? leId_ : LE_ID_NONE;
}

uint32_t MultinodeLEBufferMulticast::getNumBackingMemories() const {
    return isCreated_ ? (uint32_t)numRanks_ : 1;
}

int MultinodeLEBufferMulticast::getMPIRank() const {
    return isCreated_ ? worldRank : -1;
}

bool MultinodeLEBufferMulticast::isCreationSuccess() const {
    return isCreated_;
}

#endif  // NVBW_HAVE_LE_API

MultinodeDeviceBufferMulticast::MultinodeDeviceBufferMulticast(size_t bufferSize, int MPI_rank):
    MultinodeDeviceBuffer(bufferSize, MPI_rank),
    MemoryAllocation(bufferSize, MPI_rank) {
    buffer = MemoryAllocation.getBuffer();
}

MultinodeDeviceBufferLocal::MultinodeDeviceBufferLocal(size_t bufferSize, int MPI_rank):
    MultinodeDeviceBuffer(bufferSize, MPI_rank) {
    buffer = nullptr;
    if (worldRank == MPI_rank) {
        CU_ASSERT(cuDevicePrimaryCtxRetain(&primaryCtx, localDevice));
        CU_ASSERT(cuCtxSetCurrent(primaryCtx));
        if (bufferSize) {
            CU_ASSERT(cuMemAlloc((CUdeviceptr*)&buffer, bufferSize));
        }
    }
}

MultinodeDeviceBufferLocal::~MultinodeDeviceBufferLocal() {
    if (buffer) {
        CU_ASSERT(cuCtxSetCurrent(primaryCtx));
        CU_ASSERT(cuMemFree((CUdeviceptr)buffer));
        CU_ASSERT(cuDevicePrimaryCtxRelease(localDevice));
    }
}

// Helper function to generate test pairs using NodeHelperMulti sampling with
// target number of pairs
std::vector<std::pair<int, int>>
generateTestPairs(int worldSize, long long targetNumPairs) {
    // Members are static, we only need to do the pair-wise sampling once.
    static std::vector<std::pair<int, int>> allValidPairs;  // All possible (src ≠ dst) GPU pairs
    static std::vector<std::pair<int, int>> selectedPairs;  // Subset of pairs chosen via sampling
    static bool pairsCalculated = false;  // Ensure the chosen pairs are computed once per run

    // Handle special cases
    if (targetNumPairs == 0) {
        return {};
    }
    if (targetNumPairs == -1) {
        targetNumPairs = ((worldSize) * (worldSize - 1));
    }

    bool useDecisionSampling = targetNumPairs < ((worldSize) * (worldSize - 1));

    // If pairs were already calculated and we're not doing decision sampling, return all pairs
    if (pairsCalculated && !useDecisionSampling) {
        return allValidPairs;
    }

    // If pairs were already calculated and we are doing decision sampling, return selected pairs.
    // Currently, we only need to calculate the pairs once for all tests.
    if (pairsCalculated && useDecisionSampling && !selectedPairs.empty()) {
        return selectedPairs;
    }

    if (worldRank == 0) {
        std::vector<bool> gpuIncluded(worldSize, false);

        if (!pairsCalculated) {
            // Generate all valid pairs
            for (int srcDeviceId = 0; srcDeviceId < worldSize; srcDeviceId++) {
                for (int peerDeviceId = 0; peerDeviceId < worldSize; peerDeviceId++) {
                    if (peerDeviceId == srcDeviceId) {
                        continue;
                    }
                    allValidPairs.push_back({srcDeviceId, peerDeviceId});
                }
            }

            // Use fixed seed for reproducible results
            std::mt19937 gen(0xBAADF00D);
            std::shuffle(allValidPairs.begin(), allValidPairs.end(), gen);

            if (useDecisionSampling) {
                selectedPairs.clear();
                std::set<std::pair<int, int>> selectedPairsSet;

                std::vector<int> gpuOrder;
                for (int i = 0; i < worldSize; i++) {
                    gpuOrder.push_back(i);
                }
                std::shuffle(gpuOrder.begin(), gpuOrder.end(), gen);

                // First pass: try to include each GPU at least once
                for (int gpu : gpuOrder) {
                    if (selectedPairsSet.size() >= static_cast<size_t>(targetNumPairs)) {
                        break;
                    }
                    if (gpuIncluded[gpu]) {
                        continue;
                    }
                    // Find a pair that includes this GPU
                    for (const auto &pair : allValidPairs) {
                        if ((pair.first == gpu || pair.second == gpu) &&
                            selectedPairsSet.size() < static_cast<size_t>(targetNumPairs)) {
                            if (selectedPairsSet.find(pair) == selectedPairsSet.end()) {
                                selectedPairsSet.insert(pair);
                                selectedPairs.push_back(pair);
                                gpuIncluded[pair.first] = true;
                                gpuIncluded[pair.second] = true;
                                break;  // Found one pair for this GPU, move to next
                            }
                        }
                    }
                }

                // Second pass: fill remaining slots with random pairs
                if (selectedPairsSet.size() < static_cast<size_t>(targetNumPairs)) {
                    // Create a list of remaining pairs (excluding already selected ones)
                    std::vector<std::pair<int, int>> remainingPairs;
                    for (const auto &pair : allValidPairs) {
                        if (selectedPairsSet.find(pair) == selectedPairsSet.end()) {
                            remainingPairs.push_back(pair);
                        }
                    }

                    // Shuffle remaining pairs and add them
                    if (!remainingPairs.empty()) {
                        std::shuffle(remainingPairs.begin(), remainingPairs.end(), gen);

                        int remainingSlots = targetNumPairs - selectedPairsSet.size();
                        int pairsToAdd = std::min(
                            remainingSlots,
                            static_cast<int>(remainingPairs.size()));

                        for (int i = 0; i < pairsToAdd; i++) {
                            selectedPairsSet.insert(remainingPairs[i]);
                            selectedPairs.push_back(remainingPairs[i]);
                        }
                    }
                }
            }
            pairsCalculated = true;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    pairsCalculated = true;

    std::vector<std::pair<int, int>> finalPairs;
    if (worldRank == 0) {
        if (useDecisionSampling) {
            finalPairs = selectedPairs;
        } else {
            finalPairs = allValidPairs;
        }
    }

    // Broadcast the size first
    long pair_vector_size = finalPairs.size();
    MPI_Bcast(&pair_vector_size, 1, MPI_LONG, 0, MPI_COMM_WORLD);

    // Serialize pairs to flat integer array for safe MPI communication
    std::vector<int> serializedPairs;
    if (worldRank == 0) {
        serializedPairs.reserve(finalPairs.size() * 2);
        for (const auto& pair : finalPairs) {
            serializedPairs.push_back(pair.first);
            serializedPairs.push_back(pair.second);
        }
    } else {
        serializedPairs.resize(pair_vector_size * 2);
        finalPairs.resize(pair_vector_size);
    }

    // Broadcast the serialized data
    MPI_Bcast(serializedPairs.data(), serializedPairs.size(), MPI_INT, 0, MPI_COMM_WORLD);

    // Deserialize back to pairs on non-rank-0 processes
    if (worldRank != 0) {
        for (size_t i = 0; i < finalPairs.size(); ++i) {
            finalPairs[i] = {serializedPairs[i * 2], serializedPairs[i * 2 + 1]};
        }
        if (useDecisionSampling) {
            selectedPairs = finalPairs;
        } else {
            allValidPairs = finalPairs;
        }
    }

    return finalPairs;
}

NodeHelperMulti::NodeHelperMulti() : blockingVarDeviceAllocation(sizeof(*blockingVarDevice), 0)  {
    CU_ASSERT(cuMemHostAlloc((void **)&blockingVarHost, sizeof(*blockingVarHost), CU_MEMHOSTALLOC_PORTABLE));
    blockingVarDevice = (volatile int*) blockingVarDeviceAllocation.getBuffer();
}

NodeHelperMulti::~NodeHelperMulti() {
    CU_ASSERT(cuMemFreeHost((void*)blockingVarHost));
}

MemcpyDispatchInfo NodeHelperMulti::dispatchMemcpy(const std::vector<const MemcpyBuffer*> &srcNodesUnfiltered, const std::vector<const MemcpyBuffer*> &dstNodesUnfiltered, ContextPreference ctxPreference) {
    std::vector<int> ranksUnfiltered(srcNodesUnfiltered.size(), -1);
    std::vector<CUcontext> contextsUnfiltered(srcNodesUnfiltered.size());
    std::vector<const MemcpyBuffer*> srcNodes;
    std::vector<const MemcpyBuffer*> dstNodes;
    std::vector<CUcontext> contexts;

    for (int i = 0; i < srcNodesUnfiltered.size(); i++) {
        const MemcpyBuffer* exec = NodeHelper::selectExecutingBuffer(
            srcNodesUnfiltered[i], dstNodesUnfiltered[i], ctxPreference);
        if (exec) {
            contextsUnfiltered[i] = exec->getPrimaryCtx();
            ranksUnfiltered[i]    = exec->getMPIRank();
        }
    }

    for (int i = 0; i < srcNodesUnfiltered.size(); i++) {
        if (ranksUnfiltered[i] == worldRank) {
            srcNodes.push_back(srcNodesUnfiltered[i]);
            dstNodes.push_back(dstNodesUnfiltered[i]);
            contexts.push_back(contextsUnfiltered[i]);
        }
    }

    // Don't crash if there are no memcopies to do
    if (ranksUnfiltered.size() > 0) {
        rankOfFirstMemcpy = ranksUnfiltered[0];
    }

    MemcpyDispatchInfo info(srcNodes, dstNodes, contexts, ranksUnfiltered);
    info.allSrcBuffers = srcNodesUnfiltered;
    info.allDstBuffers = dstNodesUnfiltered;
    return info;
}

double NodeHelperMulti::calculateTotalBandwidth(double totalTime, double totalSize, size_t loopCount) {
    double totalMax = 0;
    MPI_Allreduce(&totalTime, &totalMax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    totalTime = totalMax;

    double totalSum = 0;
    MPI_Allreduce(&totalSize, &totalSum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    totalSize = totalSum;

    return (totalSize * loopCount * 1000ull * 1000ull) / totalTime;
}

double NodeHelperMulti::calculateSumBandwidth(std::vector<PerformanceStatistic> &bandwidthStats) {
    double sum = 0.0;
    for (auto stat : bandwidthStats) {
        sum += stat.returnAppropriateMetric() * 1e-9;
    }
    // Calculate total BW sum across all nodes and memcopies
    double totalSum = 0;
    MPI_Allreduce(&sum, &totalSum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return totalSum;
}

double NodeHelperMulti::calculateFirstBandwidth(std::vector<PerformanceStatistic> &bandwidthStats) {
    // Broadcast bandwidth of "first" memcopy to other nodes
    double retval = 0;
    if (worldRank == rankOfFirstMemcpy) {
        retval = bandwidthStats[0].returnAppropriateMetric() * 1e-9;
    }
    MPI_Bcast(&retval, 1, MPI_DOUBLE, rankOfFirstMemcpy, MPI_COMM_WORLD);
    return retval;
}

std::vector<double> NodeHelperMulti::calculateVectorBandwidth(std::vector<double> &results, std::vector<int> originalRanks) {
    std::vector<double> retval;
    int current_local_elem = 0;
    for (int i = 0; i < originalRanks.size(); i++) {
        double tmp = 0;
        if (worldRank == originalRanks[i]) {
            tmp = results[current_local_elem];
            current_local_elem++;
        }
        MPI_Bcast(&tmp, 1, MPI_DOUBLE, originalRanks[i], MPI_COMM_WORLD);
        retval.push_back(tmp);
    }
    return retval;
}

void NodeHelperMulti::synchronizeProcess() {
    MPI_Barrier(MPI_COMM_WORLD);
}

CUresult NodeHelperMulti::streamSynchronizeWrapper(CUstream stream) const {
    return MPIstreamSyncHelper(stream);
}

void NodeHelperMulti::barrierAllRanks() {
    MPI_Barrier(MPI_COMM_WORLD);
}

void NodeHelperMulti::streamBlockerReset() {
    *blockingVarHost = 0;
    CU_ASSERT(cuMemsetD32((CUdeviceptr) blockingVarDevice, 0, 1));
}

void NodeHelperMulti::streamBlockerRelease() {
    *blockingVarHost = 1;
}

void NodeHelperMulti::streamBlockerBlock(CUstream stream) {
    // MPI rank ranks[0] is released by blockingVar, writes to blockingVarDevice, releasing other ranks
    CU_ASSERT(spinKernelMultistage((worldRank == rankOfFirstMemcpy) ? blockingVarHost : nullptr, blockingVarDevice, stream));
}

#endif  // MULTINODE
