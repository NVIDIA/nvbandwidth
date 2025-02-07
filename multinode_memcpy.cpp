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

#ifdef MULTINODE
#include <mpi.h>
#include <unistd.h>

#include "kernels.cuh"
#include "multinode_memcpy.h"

MultinodeMemoryAllocation::MultinodeMemoryAllocation(size_t bufferSize, int MPI_rank): bufferSize(bufferSize), MPI_rank(MPI_rank) {
    cudaSetDevice(localDevice);
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
}

MultinodeMemoryAllocationUnicast::~MultinodeMemoryAllocationUnicast() {
    // Make sure that everyone is done using the memory
    MPI_Barrier(MPI_COMM_WORLD);

    CU_ASSERT(cuMemUnmap((CUdeviceptr) buffer, roundedUpAllocationSize));
    CU_ASSERT(cuMemRelease(handle));
    CU_ASSERT(cuMemAddressFree((CUdeviceptr) buffer, roundedUpAllocationSize));
}

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
    // only single-GPU supported for now
    return 0;
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
        // prefer source context
        // determine which ranks executes given operation
        if (ctxPreference == PREFER_SRC_CONTEXT && srcNodesUnfiltered[i]->getPrimaryCtx() != nullptr) {
            contextsUnfiltered[i] = srcNodesUnfiltered[i]->getPrimaryCtx();
            ranksUnfiltered[i] = srcNodesUnfiltered[i]->getMPIRank();
        } else if (dstNodesUnfiltered[i]->getPrimaryCtx() != nullptr) {
            contextsUnfiltered[i] = dstNodesUnfiltered[i]->getPrimaryCtx();
            ranksUnfiltered[i] = dstNodesUnfiltered[i]->getMPIRank();
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

    return MemcpyDispatchInfo(srcNodes, dstNodes, contexts, ranksUnfiltered);
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
