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

#ifndef MULTINODE_MEMCPY_H_
#define MULTINODE_MEMCPY_H_

#include <cuda.h>
#include <cuda_runtime_api.h>

#include "common.h"
#include "memcpy.h"
#include "memcpy_le.h"

std::vector<std::pair<int, int>> generateTestPairs(int worldSize, long long targetNumPairs = -1);

class MultinodeMemoryAllocation {
 protected:
    void* buffer = nullptr;
    size_t bufferSize;
    int MPI_rank;

 public:
    MultinodeMemoryAllocation(size_t bufferSize, int MPI_rank);
    void *getBuffer() { return (void *) buffer; }
    CUresult streamSynchronizeWrapper(CUstream stream) const;
};

// Class responsible for allocating memory that is shareable on a NVLink system, following RAII principles
// Constructor takes as parameters:
// - bufferSize: size of requested allocation
// - MPI_rank: node on which the allocation physically resides.
//      All other nodes will have this allocation mapped and accessible remotely.
class MultinodeMemoryAllocationUnicast : public MultinodeMemoryAllocation {
 private:
    CUmemGenericAllocationHandle handle = {};
    CUmemFabricHandle fh = {};
    CUmemAllocationHandleType handleType = {};
    CUmemAllocationProp prop = {};
    CUmemAccessDesc desc = {};
    size_t roundedUpAllocationSize;

 public:
    MultinodeMemoryAllocationUnicast(size_t bufferSize, int MPI_rank);
    ~MultinodeMemoryAllocationUnicast();
};

// Class responsible for allocating multicast object, following RAII principles
// Constructor takes as parameters:
// - bufferSize: size of requested allocation
// - MPI_rank: node driving the allocation process and exporting memory handle.
//      All nodes will have this allocation mapped and accessible.
class MultinodeMemoryAllocationMulticast : public MultinodeMemoryAllocation {
 private:
    CUmemGenericAllocationHandle handle = {};
    CUmemGenericAllocationHandle multicastHandle = {};
    CUmemFabricHandle fh = {};
    CUmemAllocationHandleType handleType = {};
    CUmulticastObjectProp multicastProp = {};
    CUmemAccessDesc desc = {};
    size_t roundedUpAllocationSize;
 public:
    MultinodeMemoryAllocationMulticast(size_t bufferSize, int MPI_rank);
    ~MultinodeMemoryAllocationMulticast();
};

// Class responsible for implementing Multinode MemcpyBuffer
// Each instance has information about which node owns the memory
class MultinodeDeviceBuffer : public MemcpyBuffer {
 private:
    int MPI_rank;
 public:
    MultinodeDeviceBuffer(size_t bufferSize, int MPI_rank);

    virtual CUcontext getPrimaryCtx() const override;
    virtual int getBufferIdx() const override;
    virtual std::string getBufferString() const override;
    virtual int getMPIRank() const override;
};

// MemcpyBuffer containing memory accessible from a different node in a multi-node NVLink connected system
// MPI_rank node owns the memory allocation, other nodes have it mapped
// Writes/reads to that memory from other nodes happen over NVLink
class MultinodeDeviceBufferUnicast : public MultinodeDeviceBuffer {
 private:
    MultinodeMemoryAllocationUnicast MemoryAllocation;
 public:
    MultinodeDeviceBufferUnicast(size_t bufferSize, int MPI_rank);
};

#ifdef NVBW_HAVE_LE_API
// Unicast LE across ranks: owner has VA + LE id; importers have LE id only (getDevicePtr()==0).
// memoryNodeId >= 0 pins backing memory to that uGPU locality domain.
class MultinodeLEBufferUnicast : public MultinodeDeviceBuffer {
 private:
    CUlogicalEndpointId leId_;
    CUdeviceptr devicePtr_;
    CUdevice cuDevice_;
    CUcontext primaryCtx_;
    CUmemGenericAllocationHandle memHandle_;
    size_t mappedSize_;
    bool isOwner_;
    bool isCreated_;
    int memoryNodeId_;

 public:
    MultinodeLEBufferUnicast(size_t bufferSize, int MPI_rank, int memoryNodeId = -1);
    ~MultinodeLEBufferUnicast();

    uint32_t getLeId() const override;
    CUdeviceptr getDevicePtr() const override;

    // True only when this rank's LE was fully created/bound (owner) or imported
    // (importer) and reached the ready state. Callers must AND-reduce this across
    // all ranks before dispatching, otherwise a transfer/memset would run against
    // a dead buffer and raise CUDA_ERROR_ILLEGAL_ADDRESS.
    bool isCreationSuccess() const { return isCreated_; }
};

// Shared multicast LE spanning all MPI ranks.
// Rank 0 creates with numDevices=worldSize, exports the fabric handle; all ranks import
// it, add their device, allocate a fabric memory handle, poll ready, then bind.
// Only the memory handle is allocated — no VA mapping is needed because kernels write
// via leId (fabric.try_put.multimem) and verify via leId (fabric.try_pullred).
// getMPIRank() returns worldRank on success or -1 on failure.
// getNumBackingMemories() returns worldSize for pullred pattern scaling.
class MultinodeLEBufferMulticast : public MultinodeDeviceBuffer {
    CUlogicalEndpointId leId_;
    CUdevice cuDevice_;
    CUcontext primaryCtx_;
    CUmemGenericAllocationHandle memHandle_;
    size_t mappedSize_;
    int numRanks_;
    bool isCreated_;
 public:
    MultinodeLEBufferMulticast(size_t bufferSize, int deviceCount);
    ~MultinodeLEBufferMulticast();
    bool isMulticastBuffer() const override;
    uint32_t getLeId() const override;
    uint32_t getNumBackingMemories() const override;
    int getMPIRank() const override;
    bool isCreationSuccess() const;
};
#endif  // NVBW_HAVE_LE_API

// MemcpyBuffer containing memory bound to multicast object
// Each node has its own copy of the memory, and the copies are the same
// Writes to this memory are instantly propagated to other nodes (conforming to P2P writes memory model)
class MultinodeDeviceBufferMulticast : public MultinodeDeviceBuffer {
 private:
    MultinodeMemoryAllocationMulticast MemoryAllocation;
 public:
    MultinodeDeviceBufferMulticast(size_t bufferSize, int MPI_rank);
    bool isMulticastBuffer() const override { return true; }
    // N backing memories (one per MPI rank) contribute to multimem.ld_reduce results.
    uint32_t getNumBackingMemories() const override { return (uint32_t)worldSize; }
};

// MemcpyBuffer containing regular device memory
// Only available on one node, exists primarily to simplify writing testcases
class MultinodeDeviceBufferLocal : public MultinodeDeviceBuffer {
 private:
    CUcontext primaryCtx {};
 public:
    MultinodeDeviceBufferLocal(size_t bufferSize, int MPI_rank);
    ~MultinodeDeviceBufferLocal();
};

#ifdef NVBW_HAVE_UGPU_MEMORY_NODE_SPLIT
// Unicast fabric buffer pinned to a specific uGPU locality domain.
// The owning rank allocates with CU_MEM_LOCATION_TYPE_DEVICE_LOCALITY_DOMAIN +
// CU_MEM_HANDLE_TYPE_FABRIC; all other ranks import and map the fabric handle.
class MultinodeLocalizedMemoryAllocationUnicast : public MultinodeMemoryAllocation {
    CUmemGenericAllocationHandle handle_{};
    CUmemFabricHandle fh_{};
    size_t roundedUpAllocationSize_{};
 public:
    MultinodeLocalizedMemoryAllocationUnicast(size_t bufferSize, int MPI_rank,
                                              unsigned int memoryNodeId);
    ~MultinodeLocalizedMemoryAllocationUnicast();
};

class MultinodeLocalizedDeviceBufferUnicast : public MultinodeDeviceBuffer {
    MultinodeLocalizedMemoryAllocationUnicast alloc_;
 public:
    MultinodeLocalizedDeviceBufferUnicast(size_t bufferSize, int MPI_rank,
                                          unsigned int memoryNodeId);
};
#endif  // NVBW_HAVE_UGPU_MEMORY_NODE_SPLIT

class NodeHelperMulti : public NodeHelper {
 private:
    int rankOfFirstMemcpy;

    // streamBlocker
    volatile int* blockingVarHost;
    volatile int* blockingVarDevice;
    MultinodeMemoryAllocationUnicast blockingVarDeviceAllocation;
 public:
    NodeHelperMulti();
    ~NodeHelperMulti();
    MemcpyDispatchInfo dispatchMemcpy(const std::vector<const MemcpyBuffer*> &srcBuffers, const std::vector<const MemcpyBuffer*> &dstBuffers, ContextPreference ctxPreference);
    double calculateTotalBandwidth(double totalTime, double totalSize, size_t loopCount);
    double calculateSumBandwidth(std::vector<PerformanceStatistic> &bandwidthStats);
    double calculateFirstBandwidth(std::vector<PerformanceStatistic> &bandwidthStats);
    std::vector<double> calculateVectorBandwidth(std::vector<double> &results, std::vector<int> originalRanks);
    void synchronizeProcess();
    CUresult streamSynchronizeWrapper(CUstream stream) const;
    void barrierAllRanks();

    // stream blocking functions
    void streamBlockerReset();
    void streamBlockerRelease();
    void streamBlockerBlock(CUstream stream);
};

#endif  // MULTINODE_MEMCPY_H_
