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

#ifndef MULTINODE_MEMCPY_H_
#define MULTINODE_MEMCPY_H_
#ifdef MULTINODE

#include <cuda.h>
#include <cuda_runtime_api.h>

#include "common.h"
#include "memcpy.h"

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

// MemcpyBuffer containing memory bound to multicast object
// Each node has its own copy of the memory, and the copies are the same
// Writes to this memory are instantly propagated to other nodes (conforming to P2P writes memory model)
class MultinodeDeviceBufferMulticast : public MultinodeDeviceBuffer {
 private:
    MultinodeMemoryAllocationMulticast MemoryAllocation;
 public:
    MultinodeDeviceBufferMulticast(size_t bufferSize, int MPI_rank);
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

    // stream blocking functions
    void streamBlockerReset();
    void streamBlockerRelease();
    void streamBlockerBlock(CUstream stream);
};

#endif  // MULTINODE
#endif  // MULTINODE_MEMCPY_H_
