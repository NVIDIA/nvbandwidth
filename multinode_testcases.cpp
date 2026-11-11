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

#include <cuda.h>
#include <algorithm>
#include <iomanip>
#include <memory>
#include <vector>

#include "testcase.h"
#include "memcpy.h"
#include "memcpy_le.h"
#include "common.h"
#include "output.h"
#include "kernels_le_tma.hpp"
#ifdef MULTINODE
#include <mpi.h>
#include "multinode_memcpy.h"

static FabricCliqueType getRequiredFabricCliqueType(bool isMulticast, bool usesLogicalEndpoint) {
    if (isMulticast) {
        return usesLogicalEndpoint
            ? FabricCliqueType::MulticastLogicalEndpoint
            : FabricCliqueType::MulticastPointer;
    }
    return usesLogicalEndpoint
        ? FabricCliqueType::UnicastLogicalEndpoint
        : FabricCliqueType::UnicastPointer;
}

static FabricCliqueType getRequiredFabricCliqueType(bool isMulticast, CopyInitiator initiator) {
    return getRequiredFabricCliqueType(isMulticast, multinodeInitiatorUsesHandleBuffers(initiator));
}

#ifdef NVBW_HAVE_LE_API
// AND-reduce LE readiness across ranks (MPI_MIN) so one rank's failure makes every
// rank skip in lockstep, instead of dispatching against a dead LE buffer (freed VA
// -> CUDA_ERROR_ILLEGAL_ADDRESS). Used by both the unicast and multicast LE paths.
static bool allRanksLeReady(bool localReady) {
    int created = localReady ? 1 : 0;
    int allCreated = 0;
    MPI_Allreduce(&created, &allCreated, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    return allCreated != 0;
}

// Rank-0-only warning for a unicast-LE pair skipped because it wasn't ready everywhere.
static void warnLeSkip(const std::string &testKey, int srcDeviceId, int peerDeviceId) {
    if (worldRank == 0)
        output->recordWarning("Skipping " + testKey + " pair (" +
            std::to_string(srcDeviceId) + "," + std::to_string(peerDeviceId) +
            "): unicast LE not ready on all ranks.");
}
#endif  // NVBW_HAVE_LE_API

#ifdef NVBW_HAVE_UGPU_MEMORY_NODE_SPLIT
// ============================================================================
// Multinode uGPU localization helpers
// ============================================================================

// N localized remote buffers on remoteRank (one per memory node).
static void allocateMultinodeLocalizedRemoteBuffers(
        unsigned long long size, int remoteRank, int memNodeCount,
        std::vector<std::unique_ptr<MemcpyBuffer>>& remoteBufs) {
    remoteBufs.clear();
    remoteBufs.reserve(memNodeCount);
    for (int n = 0; n < memNodeCount; n++) {
        remoteBufs.push_back(std::make_unique<MultinodeLocalizedDeviceBufferUnicast>(
            size, remoteRank, static_cast<unsigned int>(n)));
    }
}

// Executing-rank concurrent localized CE. Local prepare/verify only; remote seed/check on owner.
static double runLocalizedConcurrentOnRank(
        const DeviceToDeviceAccessStrategy& accessStrategy,
        const std::vector<std::unique_ptr<MemcpyBuffer>>& remoteBufs,
        unsigned long long size, unsigned long long loopCount,
        const char* verboseTag) {
    const int localityDomains = static_cast<int>(remoteBufs.size());
    if (localityDomains == 0) return 0.0;

    CUcontext callerCtx = nullptr;
    CU_ASSERT(cuCtxGetCurrent(&callerCtx));

    std::vector<std::unique_ptr<LocalizedDeviceBuffer>> localBufs;
    localBufs.reserve(localityDomains);
    for (int i = 0; i < localityDomains; i++)
        localBufs.push_back(std::make_unique<LocalizedDeviceBuffer>(
            size, localDevice, static_cast<unsigned int>(i)));

    CU_ASSERT(cuCtxSetCurrent(localBufs[0]->getPrimaryCtx()));

    auto submitOn = [&](int i, CUstream stream) {
        auto [memcpySrc, memcpyDst] =
            accessStrategy.getOrderedBuffers(*localBufs[i], *remoteBufs[i]);
        submitLocalizedCeCopy(*memcpySrc, *memcpyDst, size, stream);
    };

    if (!gSettings.skipVerification) {
        const bool isWrite =
            (accessStrategy.getContextPreference() == PREFER_SRC_CONTEXT);
        for (int i = 0; i < localityDomains; i++) {
            memsetLocalizedConcurrentOwnedBuffer(*localBufs[i], size,
                                                 isWrite ? 0xBAADF00D : 0xCAFEBABE);
        }
    }

    // A rank drives a single access direction, so it runs one stream per domain.
    const int streamCount = localityDomains;
    double result = runConcurrentLocalizedStreams(streamCount, size, loopCount, verboseTag,
                                                  submitOn);

    if (!gSettings.skipVerification) {
        const bool isWrite =
            (accessStrategy.getContextPreference() == PREFER_SRC_CONTEXT);
        if (!isWrite) {
            for (int i = 0; i < localityDomains; i++) {
                memcmpLocalizedConcurrentOwnedBuffer(*localBufs[i], size, 0xBAADF00D);
            }
        }
    }

    CU_ASSERT(cuCtxSetCurrent(callerCtx));
    return result;
}

static void prepareRemoteLocalizedBuffers(
        const DeviceToDeviceAccessStrategy& accessStrategy,
        const std::vector<std::unique_ptr<MemcpyBuffer>>& remoteBufs,
        unsigned long long size) {
    if (gSettings.skipVerification || remoteBufs.empty()) return;
    const bool isWrite =
        (accessStrategy.getContextPreference() == PREFER_SRC_CONTEXT);
    const unsigned int seed = isWrite ? 0xCAFEBABE : 0xBAADF00D;
    for (const auto& buf : remoteBufs) {
        memsetLocalizedConcurrentOwnedBuffer(*buf, size, seed);
    }
}

static void verifyRemoteLocalizedBuffers(
        const DeviceToDeviceAccessStrategy& accessStrategy,
        const std::vector<std::unique_ptr<MemcpyBuffer>>& remoteBufs,
        unsigned long long size) {
    if (gSettings.skipVerification || remoteBufs.empty()) return;
    const bool isWrite =
        (accessStrategy.getContextPreference() == PREFER_SRC_CONTEXT);
    if (!isWrite) return;
    for (const auto& buf : remoteBufs) {
        memcmpLocalizedConcurrentOwnedBuffer(*buf, size, 0xBAADF00D);
    }
}

// Returns the sum of per-stream bandwidths (GB/s).
static double runConcurrentLocalizedMultinode(
        int executingRank, int peerRank,
        const DeviceToDeviceAccessStrategy& accessStrategy,
        const std::vector<std::unique_ptr<MemcpyBuffer>>& remoteBufs,
        unsigned long long size, unsigned long long loopCount) {
    double result = 0.0;

    if (worldRank == peerRank) {
        prepareRemoteLocalizedBuffers(accessStrategy, remoteBufs, size);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    if (worldRank == executingRank && !remoteBufs.empty()) {
        if (gSettings.verbose)
            VERBOSE << "[multinode_localized_concurrent src=" << executingRank
                    << " raw]\n";
        result = runLocalizedConcurrentOnRank(accessStrategy, remoteBufs, size, loopCount,
                                              "multinode_localized_concurrent");
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (worldRank == peerRank) {
        verifyRemoteLocalizedBuffers(accessStrategy, remoteBufs, size);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Bcast(&result, 1, MPI_DOUBLE, executingRank, MPI_COMM_WORLD);
    return result;
}

static std::vector<double> runConcurrentLocalizedBidirMultinode(
        int srcRank, int peerRank,
        const DeviceToDeviceAccessStrategy& accessStrategy,
        const std::vector<std::unique_ptr<MemcpyBuffer>>& srcRemoteBufs,
        const std::vector<std::unique_ptr<MemcpyBuffer>>& peerRemoteBufs,
        unsigned long long size, unsigned long long loopCount) {
    double localResult = 0.0;
    const bool executing =
        (worldRank == srcRank || worldRank == peerRank) &&
        !srcRemoteBufs.empty() && !peerRemoteBufs.empty();

    if (worldRank == srcRank) {
        prepareRemoteLocalizedBuffers(accessStrategy, srcRemoteBufs, size);
    }
    if (worldRank == peerRank) {
        prepareRemoteLocalizedBuffers(accessStrategy, peerRemoteBufs, size);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (executing) {
        const auto& remoteBufs = (worldRank == srcRank) ? peerRemoteBufs : srcRemoteBufs;
        if (gSettings.verbose)
            VERBOSE << "[multinode_localized_concurrent_bidir rank=" << worldRank
                    << " src=" << srcRank << " peer=" << peerRank << " raw]\n";
        localResult = runLocalizedConcurrentOnRank(accessStrategy, remoteBufs, size, loopCount,
                                                   "multinode_localized_concurrent_bidir");
    }
    MPI_Barrier(MPI_COMM_WORLD);

    if (worldRank == srcRank) {
        verifyRemoteLocalizedBuffers(accessStrategy, srcRemoteBufs, size);
    }
    if (worldRank == peerRank) {
        verifyRemoteLocalizedBuffers(accessStrategy, peerRemoteBufs, size);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    double legsLocal[2] = {0.0, 0.0};
    if (worldRank == srcRank)  legsLocal[0] = localResult;
    if (worldRank == peerRank) legsLocal[1] = localResult;
    double legs[2] = {0.0, 0.0};
    MPI_Allreduce(legsLocal, legs, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return {legs[0], legs[1]};
}

// Domain count for a multinode CE test to localize across, or 0 to stay on the
// non-localized path.
//
// Every rank must reach the same answer. The localized path runs MPI collectives,
// so ranks disagreeing about whether to take it would deadlock rather than merely
// report inconsistently -- which is why the count is agreed here rather than each
// rank querying its own device. A mismatched world (a device reporting a different
// number of domains, or an architecture that does not localize) stays non-localized
// altogether.
//
// This is itself collective and must be called unconditionally by every rank.
static int multinodeLocalizedDomainCount(const std::string& key) {
    int local = 0;
    shouldLocalize(localDevice, local);

    int minCount = 0, maxCount = 0;
    MPI_Allreduce(&local, &minCount, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&local, &maxCount, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

    if (minCount != maxCount) {
        VERBOSE << "[" << key << "] locality domain count differs across ranks ("
                << minCount << ".." << maxCount << "); reporting non-localized bandwidth\n";
        return 0;
    }
    if (minCount < 2) return 0;
    return minCount;
}
#endif  // NVBW_HAVE_UGPU_MEMORY_NODE_SPLIT

// ============================================================================
// MultinodeDeviceToDeviceTransfer
// ============================================================================
MultinodeDeviceToDeviceTransfer::MultinodeDeviceToDeviceTransfer(CopyInitiator initiator, AccessType accessType)
    : Testcase(generateKey(initiator, accessType), generateDesc(initiator, accessType))
    , initiator_(initiator)
    , accessType_(accessType) {}

FilterResult MultinodeDeviceToDeviceTransfer::filter() {
    auto r = Testcase::filterHasMultipleGPUsMultinode();
    if (!r) return r;
    if (initiator_ == CopyInitiator::TMA) {
        auto tma = filterSupportsTMA();
        if (!tma) return tma;
    }
    if (multinodeInitiatorUsesHandleBuffers(initiator_))
        return filterSupportsHandleCopy();
    return FilterResult::pass();
}

void MultinodeDeviceToDeviceTransfer::addRequiredFabricCliqueTypes(std::vector<FabricCliqueType> &types) const {
    addRequiredFabricCliqueType(types, getRequiredFabricCliqueType(false, initiator_));
}

ContextPreference MultinodeDeviceToDeviceTransfer::getContextPreference() const {
    return (accessType_ == AccessType::Read) ? PREFER_DST_CONTEXT : PREFER_SRC_CONTEXT;
}

void MultinodeDeviceToDeviceTransfer::run(unsigned long long size, unsigned long long loopCount) {
    const bool useHandle = multinodeInitiatorUsesHandleBuffers(initiator_);

    PeerValueMatrix<double> bandwidthValues(worldSize, worldSize, key);
    MemcpyOperation memcpyInstance(loopCount, createMemcpyInitiator(),
                                   new NodeHelperMulti(), getContextPreference());

    auto pairs = generateTestPairs(worldSize, gSettings.targetNumPairs);

#ifdef NVBW_HAVE_UGPU_MEMORY_NODE_SPLIT
    // Collective, so every rank evaluates it even when the answer is "do not localize".
    const int localizedDomains = (initiator_ == CopyInitiator::CE)
        ? multinodeLocalizedDomainCount(testKey()) : 0;
    std::unique_ptr<DeviceToDeviceAccessStrategy> localizedStrategy;
    if (localizedDomains > 0)
        localizedStrategy = createDeviceToDeviceAccessStrategy(accessType_);
#endif

    for (const auto& pair : pairs) {
        int srcDeviceId = pair.first;
        int peerDeviceId = pair.second;

#ifdef NVBW_HAVE_UGPU_MEMORY_NODE_SPLIT
        if (localizedDomains > 0) {
            std::vector<std::unique_ptr<MemcpyBuffer>> remoteBufs;
            allocateMultinodeLocalizedRemoteBuffers(size, peerDeviceId,
                                                    localizedDomains, remoteBufs);
            bandwidthValues.value(srcDeviceId, peerDeviceId) = runConcurrentLocalizedMultinode(
                srcDeviceId, peerDeviceId, *localizedStrategy, remoteBufs,
                size, loopCount);
            continue;
        }
#endif

        if (useHandle) {
#ifdef NVBW_HAVE_LE_API
            if (accessType_ == AccessType::Read) {
                // Use LE for source (fabric try_get), raw device buffer for destination
                MultinodeLEBufferUnicast peerLE(size, peerDeviceId);
                MultinodeDeviceBufferUnicast srcNode(size, srcDeviceId);
                if (!allRanksLeReady(peerLE.isCreationSuccess())) {
                    warnLeSkip(testKey(), srcDeviceId, peerDeviceId);
                    bandwidthValues.value(srcDeviceId, peerDeviceId) = 0.0;
                    continue;
                }
                bandwidthValues.value(srcDeviceId, peerDeviceId) = memcpyInstance.doMemcpy(peerLE, srcNode);
            } else {
                MultinodeDeviceBufferUnicast srcNode(size, srcDeviceId);
                MultinodeLEBufferUnicast peerLE(size, peerDeviceId);
                if (!allRanksLeReady(peerLE.isCreationSuccess())) {
                    warnLeSkip(testKey(), srcDeviceId, peerDeviceId);
                    bandwidthValues.value(srcDeviceId, peerDeviceId) = 0.0;
                    continue;
                }
                bandwidthValues.value(srcDeviceId, peerDeviceId) = memcpyInstance.doMemcpy(srcNode, peerLE);
            }
#else
            ASSERT(false && "CE/TMA Handle path requires NVBW_HAVE_LE_API");
            bandwidthValues.value(srcDeviceId, peerDeviceId) = 0.0;
#endif
        } else {
            MultinodeDeviceBufferUnicast srcNode(size, srcDeviceId);
            MultinodeDeviceBufferUnicast peerNode(size, peerDeviceId);
            if (accessType_ == AccessType::Read) {
                bandwidthValues.value(srcDeviceId, peerDeviceId) = memcpyInstance.doMemcpy(peerNode, srcNode);
            } else {
                bandwidthValues.value(srcDeviceId, peerDeviceId) = memcpyInstance.doMemcpy(srcNode, peerNode);
            }
        }
    }

    recordStability(bandwidthValues, memcpyInstance);
    output->addTestcaseResults(bandwidthValues, buildOutputLabel());
}

MemcpyInitiator* MultinodeDeviceToDeviceTransfer::createMemcpyInitiator() const {
    switch (initiator_) {
        case CopyInitiator::CE:  return new MemcpyInitiatorCE();
        case CopyInitiator::SM:  return new MemcpyInitiatorSM();
        case CopyInitiator::TMA: return new MemcpyInitiatorTMA();
    }
    return nullptr;
}

std::string MultinodeDeviceToDeviceTransfer::generateKey(CopyInitiator initiator, AccessType accessType) {
    std::string result = "multinode_device_to_device_memcpy_";
    result += (accessType == AccessType::Read) ? "read_" : "write_";
    switch (initiator) {
        case CopyInitiator::CE:  result += "ce"; break;
        case CopyInitiator::SM:  result += "sm"; break;
        case CopyInitiator::TMA: result += "tma"; break;
    }
    return result;
}

std::string MultinodeDeviceToDeviceTransfer::generateDesc(CopyInitiator initiator, AccessType accessType) {
    std::string result = "\tMeasures bandwidth of ";
    switch (initiator) {
        case CopyInitiator::CE:
            result += "CE copies (cuMemcpyAsync)";
            break;
        case CopyInitiator::SM:  result += "SM copy kernel"; break;
        case CopyInitiator::TMA: result += "Tensor Memory Accelerator"; break;
    }
    result += " between each pair of accessible peers.\n\t";
    if (accessType == AccessType::Read) {
        result += "Read tests launch a copy from the peer device to the target using the target's context.";
    } else {
        result += "Write tests launch a copy from the target device to the peer using the target's context.";
    }
    if (initiator == CopyInitiator::TMA) {
        result += std::string("\n\t") + handlePeerBufferConventionNote();
    }
    return result;
}

std::string MultinodeDeviceToDeviceTransfer::buildOutputLabel() const {
    std::string initiatorName;
    switch (initiator_) {
        case CopyInitiator::CE:  initiatorName = "CE"; break;
        case CopyInitiator::SM:  initiatorName = "SM"; break;
        case CopyInitiator::TMA: initiatorName = "TMA"; break;
    }
    std::string arrow = (accessType_ == AccessType::Read) ? "->" : "<-";
    return "memcpy " + initiatorName + " GPU(row) " + arrow + " GPU(column) bandwidth (GB/s)";
}

// ============================================================================
// MultinodeDeviceToDeviceBidirTransfer
// ============================================================================

MultinodeDeviceToDeviceBidirTransfer::MultinodeDeviceToDeviceBidirTransfer(CopyInitiator initiator, AccessType accessType)
    : Testcase(generateKey(initiator, accessType), generateDesc(initiator, accessType))
    , initiator_(initiator)
    , accessType_(accessType) {}

FilterResult MultinodeDeviceToDeviceBidirTransfer::filter() {
    auto r = Testcase::filterHasMultipleGPUsMultinode();
    if (!r) return r;
    if (initiator_ == CopyInitiator::TMA) {
        auto tma = filterSupportsTMA();
        if (!tma) return tma;
    }
    if (multinodeInitiatorUsesHandleBuffers(initiator_))
        return filterSupportsHandleCopy();
    return FilterResult::pass();
}

void MultinodeDeviceToDeviceBidirTransfer::addRequiredFabricCliqueTypes(std::vector<FabricCliqueType> &types) const {
    addRequiredFabricCliqueType(types, getRequiredFabricCliqueType(false, initiator_));
}

ContextPreference MultinodeDeviceToDeviceBidirTransfer::getContextPreference() const {
    return (accessType_ == AccessType::Read) ? PREFER_DST_CONTEXT : PREFER_SRC_CONTEXT;
}

void MultinodeDeviceToDeviceBidirTransfer::run(unsigned long long size, unsigned long long loopCount) {
    std::string suffix = (accessType_ == AccessType::Read) ? "read" : "write";
    PeerValueMatrix<double> bandwidthValues1(worldSize, worldSize, key + "_" + suffix + "1");
    PeerValueMatrix<double> bandwidthValues2(worldSize, worldSize, key + "_" + suffix + "2");
    PeerValueMatrix<double> bandwidthValuesTotal(worldSize, worldSize, key + "_total");

    MemcpyOperation memcpyInstance(loopCount, createMemcpyInitiator(),
                                   new NodeHelperMulti(), getContextPreference(),
                                   MemcpyOperation::VECTOR_BW);

    auto pairs = generateTestPairs(worldSize, gSettings.targetNumPairs);

    const bool useHandle = multinodeInitiatorUsesHandleBuffers(initiator_);

#ifdef NVBW_HAVE_UGPU_MEMORY_NODE_SPLIT
    // Collective, so every rank evaluates it even when the answer is "do not localize".
    const int localizedDomains = (initiator_ == CopyInitiator::CE)
        ? multinodeLocalizedDomainCount(testKey()) : 0;
    std::unique_ptr<DeviceToDeviceAccessStrategy> localizedStrategy;
    if (localizedDomains > 0)
        localizedStrategy = createDeviceToDeviceAccessStrategy(accessType_);
#endif

    cachedPairLoop(pairs, bandwidthValues1, bandwidthValues2, bandwidthValuesTotal,
        [&](int srcDeviceId, int peerDeviceId) -> std::vector<double> {
#ifdef NVBW_HAVE_UGPU_MEMORY_NODE_SPLIT
            if (localizedDomains > 0) {
                std::vector<std::unique_ptr<MemcpyBuffer>> srcRemoteBufs, peerRemoteBufs;
                allocateMultinodeLocalizedRemoteBuffers(size, srcDeviceId,
                                                        localizedDomains, srcRemoteBufs);
                allocateMultinodeLocalizedRemoteBuffers(size, peerDeviceId,
                                                        localizedDomains, peerRemoteBufs);
                return runConcurrentLocalizedBidirMultinode(
                    srcDeviceId, peerDeviceId, *localizedStrategy, srcRemoteBufs,
                    peerRemoteBufs, size, loopCount);
            }
#endif

            if (useHandle) {
#ifdef NVBW_HAVE_LE_API
                // Each copy must have its LE on a different device so
                // selectExecutingBuffer() routes them to different ranks.
                // Read: data flows LE->device (LE is src) and Write: device->LE (LE is dst).
                MultinodeLEBufferUnicast peerLE(size, peerDeviceId), srcLE(size, srcDeviceId);
                MultinodeDeviceBufferUnicast srcDev(size, srcDeviceId), peerDev(size, peerDeviceId);
                if (!allRanksLeReady(peerLE.isCreationSuccess() && srcLE.isCreationSuccess())) {
                    warnLeSkip(testKey(), srcDeviceId, peerDeviceId);
                    return {0.0, 0.0};
                }
                std::vector<const MemcpyBuffer*> leVec  = {&peerLE, &srcLE};
                std::vector<const MemcpyBuffer*> devVec = {&srcDev, &peerDev};
                const bool isRead = (accessType_ == AccessType::Read);
                return memcpyInstance.doMemcpyVector(isRead ? leVec : devVec,
                                                     isRead ? devVec : leVec);
#else
                return {};  // unreachable: useHandle requires NVBW_HAVE_LE_API
#endif
            } else {
                MultinodeDeviceBufferUnicast src1(size, srcDeviceId), src2(size, srcDeviceId);
                MultinodeDeviceBufferUnicast peer1(size, peerDeviceId), peer2(size, peerDeviceId);
                std::vector<const MemcpyBuffer*> srcNodes = {&peer1, &src1};
                std::vector<const MemcpyBuffer*> peerNodes = {&src2, &peer2};
                return memcpyInstance.doMemcpyVector(srcNodes, peerNodes);
            }
        });

    std::string suffix1 = (accessType_ == AccessType::Read) ? "Read1" : "Write1";
    std::string suffix2 = (accessType_ == AccessType::Read) ? "Read2" : "Write2";
    recordStability(bandwidthValuesTotal, memcpyInstance);
    output->addTestcaseResults(bandwidthValues1, buildOutputLabel(suffix1));
    output->addTestcaseResults(bandwidthValues2, buildOutputLabel(suffix2));
    output->addTestcaseResults(bandwidthValuesTotal, buildOutputLabel("Total"));
}

MemcpyInitiator* MultinodeDeviceToDeviceBidirTransfer::createMemcpyInitiator() const {
    switch (initiator_) {
        case CopyInitiator::CE:  return new MemcpyInitiatorCE();
        case CopyInitiator::SM:  return new MemcpyInitiatorSM();
        case CopyInitiator::TMA: return new MemcpyInitiatorTMA();
    }
    return nullptr;
}

std::string MultinodeDeviceToDeviceBidirTransfer::generateKey(CopyInitiator initiator, AccessType accessType) {
    std::string result = "multinode_device_to_device_bidirectional_memcpy_";
    result += (accessType == AccessType::Read) ? "read_" : "write_";
    switch (initiator) {
        case CopyInitiator::CE:  result += "ce"; break;
        case CopyInitiator::SM:  result += "sm"; break;
        case CopyInitiator::TMA: result += "tma"; break;
    }
    return result;
}

std::string MultinodeDeviceToDeviceBidirTransfer::generateDesc(CopyInitiator initiator, AccessType accessType) {
    std::string result;
    if (initiator == CopyInitiator::CE) {
        result = "\tMeasures bandwidth of CE copies (cuMemcpyAsync)\n"
                 "\tbetween each pair of accessible peers.\n"
                 "\tA copy in the opposite direction of the measured copy is run simultaneously but not measured.\n\t";
    } else if (initiator == CopyInitiator::SM) {
        result = "\tMeasures bandwidth of a copy kernel between each pair of accessible peers.\n"
                 "\tCopies are run in both directions between each pair, and the sum is reported.\n\t";
    } else if (initiator == CopyInitiator::TMA) {
        result = "\tMeasures bandwidth of TMA bidirectional copies between each pair of accessible peers.\n\t";
    }
    result += (accessType == AccessType::Read)
        ? "Read tests launch a copy from the peer device to the target using the target's context."
        : "Write tests launch a copy from the target device to the peer using the target's context.";
    if (initiator == CopyInitiator::CE || initiator == CopyInitiator::TMA) {
        result += std::string("\n\t") + handlePeerBufferConventionNote();
    }
    return result;
}

std::string MultinodeDeviceToDeviceBidirTransfer::buildOutputLabel(const std::string& suffix) const {
    std::string initiatorName;
    switch (initiator_) {
        case CopyInitiator::CE:  initiatorName = "CE"; break;
        case CopyInitiator::SM:  initiatorName = "SM"; break;
        case CopyInitiator::TMA: initiatorName = "TMA"; break;
    }
    return "memcpy " + initiatorName + " GPU(row) <-> GPU(column) " + suffix + " bandwidth (GB/s)";
}

void MultinodeAllToOneWriteSM::addRequiredFabricCliqueTypes(std::vector<FabricCliqueType> &types) const {
    addRequiredFabricCliqueType(types, FabricCliqueType::UnicastPointer);
}

void MultinodeAllToOneWriteSM::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bandwidthValues(1, worldSize, key);
    MemcpyOperation memcpyInstance(loopCount, new MemcpyInitiatorSM(), new NodeHelperMulti(), PREFER_SRC_CONTEXT, MemcpyOperation::SUM_BW);

    for (int dstDeviceId = 0; dstDeviceId < worldSize; dstDeviceId++) {
        std::vector<const MemcpyBuffer*> srcNodes;
        std::vector<const MemcpyBuffer*> dstNodes;

        for (int srcDeviceId = 0; srcDeviceId < worldSize; srcDeviceId++) {
            if (dstDeviceId == srcDeviceId) {
                continue;
            }

            srcNodes.push_back(new MultinodeDeviceBufferLocal(size, srcDeviceId));
            dstNodes.push_back(new MultinodeDeviceBufferUnicast(size, dstDeviceId));
        }

        bandwidthValues.value(0, dstDeviceId) = memcpyInstance.doMemcpy(srcNodes, dstNodes);

        for (auto node : dstNodes) {
            delete node;
        }
        for (auto node : srcNodes) {
            delete node;
        }
    }

    recordStability(bandwidthValues, memcpyInstance);
    output->addTestcaseResults(bandwidthValues, "memcpy SM All Gpus -> GPU(column) total bandwidth (GB/s)");
}

void MultinodeAllFromOneReadSM::addRequiredFabricCliqueTypes(std::vector<FabricCliqueType> &types) const {
    addRequiredFabricCliqueType(types, FabricCliqueType::UnicastPointer);
}

void MultinodeAllFromOneReadSM::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bandwidthValues(1, worldSize, key);
    MemcpyOperation memcpyInstance(loopCount, new MemcpyInitiatorSM(), new NodeHelperMulti(), PREFER_DST_CONTEXT, MemcpyOperation::SUM_BW);

    for (int srcDeviceId = 0; srcDeviceId < worldSize; srcDeviceId++) {
        std::vector<const MemcpyBuffer*> srcNodes;
        std::vector<const MemcpyBuffer*> dstNodes;

        for (int dstDeviceId = 0; dstDeviceId < worldSize; dstDeviceId++) {
            if (dstDeviceId == srcDeviceId) {
                continue;
            }

            srcNodes.push_back(new MultinodeDeviceBufferUnicast(size, srcDeviceId));
            dstNodes.push_back(new MultinodeDeviceBufferLocal(size, dstDeviceId));
        }

        bandwidthValues.value(0, srcDeviceId) = memcpyInstance.doMemcpy(srcNodes, dstNodes);

        for (auto node : dstNodes) {
            delete node;
        }
        for (auto node : srcNodes) {
            delete node;
        }
    }

    recordStability(bandwidthValues, memcpyInstance);
    output->addTestcaseResults(bandwidthValues, "memcpy SM All Gpus <- GPU(column) total bandwidth (GB/s)");
}

void MultinodeBroadcastOneToAllSM::addRequiredFabricCliqueTypes(std::vector<FabricCliqueType> &types) const {
    addRequiredFabricCliqueType(types, FabricCliqueType::MulticastPointer);
}

void MultinodeBroadcastOneToAllSM::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bandwidthValues(1, worldSize, key);
    MemcpyOperation memcpyInstance(loopCount, new MemcpyInitiatorMultimemWrite(), new NodeHelperMulti(), PREFER_DST_CONTEXT, MemcpyOperation::SUM_BW);

    for (int dstDeviceId = 0; dstDeviceId < worldSize; dstDeviceId++) {
        std::vector<const MemcpyBuffer*> srcNodes;
        std::vector<const MemcpyBuffer*> dstNodes;

        srcNodes.push_back(new MultinodeDeviceBufferLocal(size, dstDeviceId));
        dstNodes.push_back(new MultinodeDeviceBufferMulticast(size, dstDeviceId));

        bandwidthValues.value(0, dstDeviceId) = memcpyInstance.doMemcpy(srcNodes, dstNodes);

        for (auto node : dstNodes) {
            delete node;
        }
        for (auto node : srcNodes) {
            delete node;
        }
    }

    recordStability(bandwidthValues, memcpyInstance);
    output->addTestcaseResults(bandwidthValues, "multicast SM GPU(column) -> All Gpus total bandwidth (GB/s)");
}

void MultinodeBroadcastAllToAllSM::addRequiredFabricCliqueTypes(std::vector<FabricCliqueType> &types) const {
    addRequiredFabricCliqueType(types, FabricCliqueType::MulticastPointer);
}

void MultinodeBroadcastAllToAllSM::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bandwidthValues(1, 1, key);
    MemcpyOperation memcpyInstance(loopCount, new MemcpyInitiatorMultimemWrite(), new NodeHelperMulti(), PREFER_DST_CONTEXT, MemcpyOperation::SUM_BW);
    std::vector<const MemcpyBuffer*> srcNodes;
    std::vector<const MemcpyBuffer*> dstNodes;

    for (int dstDeviceId = 0; dstDeviceId < worldSize; dstDeviceId++) {
        srcNodes.push_back(new MultinodeDeviceBufferLocal(size, dstDeviceId));
        dstNodes.push_back(new MultinodeDeviceBufferMulticast(size, dstDeviceId));
    }

    bandwidthValues.value(0, 0) = memcpyInstance.doMemcpy(srcNodes, dstNodes);

    for (auto node : dstNodes) {
        delete node;
    }
    for (auto node : srcNodes) {
        delete node;
    }

    recordStability(bandwidthValues, memcpyInstance);
    output->addTestcaseResults(bandwidthValues, "multicast SM All -> All Gpus total bandwidth (GB/s)");
}

void MultinodeBisectWriteCE::addRequiredFabricCliqueTypes(std::vector<FabricCliqueType> &types) const {
    addRequiredFabricCliqueType(types, FabricCliqueType::UnicastPointer);
}

void MultinodeBisectWriteCE::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bandwidthValues(worldSize, 1, key);
    MemcpyOperation memcpyInstance(loopCount, new MemcpyInitiatorCE(), new NodeHelperMulti(), PREFER_SRC_CONTEXT, MemcpyOperation::VECTOR_BW);
    std::vector<std::string> rowLabels;
    std::vector<const MemcpyBuffer*> srcNodes, dstNodes;

    for (int i = 0; i < worldSize; i++) {
        int peer = (i + worldSize / 2) % worldSize;
        srcNodes.push_back(new MultinodeDeviceBufferUnicast(size, i));
        dstNodes.push_back(new MultinodeDeviceBufferUnicast(size, peer));

        std::stringstream s;
        s << getPaddedProcessId(i) << "->" << getPaddedProcessId(peer);
        rowLabels.push_back(s.str());
    }

    auto results = memcpyInstance.doMemcpyVector(dstNodes, srcNodes);

    for (int i = 0; i < results.size(); i++) {
        bandwidthValues.value(i, 0) = results[i];
    }
    bandwidthValues.setRowLabels(rowLabels);

    for (auto node : dstNodes) {
        delete node;
    }
    for (auto node : srcNodes) {
        delete node;
    }

    recordStability(bandwidthValues, memcpyInstance);
    output->addTestcaseResults(bandwidthValues, "Bisect benchmarking, simultaneous write CE BW");
}

// ============================================================================
// MultinodeMulticastFabricPut
// ============================================================================

MultinodeMulticastFabricPut::MultinodeMulticastFabricPut(CopyInitiator initiator)
    : Testcase(
        std::string("multinode_multicast_fabric_put_") + (initiator == CopyInitiator::TMA ? "tma" : "sm"),
        initiator == CopyInitiator::TMA
            ? "\tMeasures bandwidth of TMA multicast fabric write from rank 0."
            : "\tMeasures bandwidth of SM multicast fabric write from rank 0.")
    , initiator_(initiator) {}

FilterResult MultinodeMulticastFabricPut::filter() {
    auto r = Testcase::filterHasMultipleGPUsMultinode();
    if (!r) return r;
    if (initiator_ == CopyInitiator::TMA) {
#ifndef NVBW_HAVE_FABRIC_KERNELS
        // NVBW_HAVE_FABRIC_KERNELS is undefined whenever the real fabric/TMA kernels were not
        // built — i.e. the target arch is < Blackwell (sm_100+) OR the toolkit lacks the LE API
        // headers (e.g. CUDA 13.0). In that case launchMultimemWriteKernel (TMA raw path) and
        // leLaunchWriteMulticastKernel (TMA LE path) are no-op stubs, so the destination is never
        // written. Waive here instead of verifying an unwritten buffer.
        //
        // This runtime waive — not a compile-time guard — is what protects the LE path: run()'s
        // LE branch is gated by NVBW_HAVE_LE_API, which can be defined while NVBW_HAVE_FABRIC_KERNELS
        // is not (e.g. an sm_90 build with LE headers present). There the LE branch still compiles
        // live against the stub, so it relies on this filter() waive to keep run() from reaching it.
        //
        // The SM variant is unaffected — it uses multicastCopyKernel (kernels.cu, always compiled)
        // and falls through below.
        return FilterResult::waive("multicast write (TMA) kernel not built (toolkit lacks fabric/LE kernel support)");
#else
        auto tma = filterSupportsTMA();
        if (!tma) return tma;
        if (gSettings.bufferType == BufferType::LE) {
            // Each rank maps to localDevice = localRank % deviceCount, so the requirement
            // is one distinct local GPU per rank on a node (ranksPerNode <= deviceCount).
            // The multicast LE itself spans all ranks across the fabric.
            if (deviceCount < ranksPerNode)
                return FilterResult::waive("requires at least one local GPU per rank (ranks-per-node exceeds local GPU count)");
            return Testcase::filterSupportsMulticast() && Testcase::filterSupportsLogicalEndpoint();
        }
        return Testcase::filterSupportsMulticast();
#endif  // NVBW_HAVE_FABRIC_KERNELS
    }
    return Testcase::filterSupportsMulticast();
}

void MultinodeMulticastFabricPut::addRequiredFabricCliqueTypes(std::vector<FabricCliqueType> &types) const {
    addRequiredFabricCliqueType(types, getRequiredFabricCliqueType(true, initiator_));
}

std::string MultinodeMulticastFabricPut::buildOutputLabel() const {
    return (initiator_ == CopyInitiator::TMA)
        ? "multicast fabric put TMA (GB/s)"
        : "multicast fabric put SM (GB/s)";
}

void MultinodeMulticastFabricPut::run(unsigned long long size, unsigned long long loopCount) {
    if (initiator_ == CopyInitiator::TMA && gSettings.bufferType == BufferType::LE) {
#ifdef NVBW_HAVE_LE_API
        PeerValueMatrix<double> bandwidthValues(1, 1, key);
        MultinodeLEBufferMulticast multicastDst(size, deviceCount);
        // All ranks must join the shared multicast LE: a partial membership never
        // becomes ready (waitLeReadyWithTimeout times out for the rest) and writes
        // would target a LE with missing backing memories, so skip in lockstep.
        if (!allRanksLeReady(multicastDst.isCreationSuccess())) {
            if (worldRank == 0) {
                output->recordWarning("Skipping " + testKey() + ": one or more ranks could not create multicast LE.");
            }
            bandwidthValues.value(0, 0) = 0.0;
            output->addTestcaseResults(bandwidthValues, buildOutputLabel());
            return;
        }
        MultinodeDeviceBufferLocal src(size, /*MPI_rank=*/0);
        MemcpyOperation memcpyInstance(loopCount, new MemcpyInitiatorTMA(), new NodeHelperMulti(),
                                       PREFER_SRC_CONTEXT, MemcpyOperation::SUM_BW);
        bandwidthValues.value(0, 0) = memcpyInstance.doMemcpy(src, multicastDst);
        recordStability(bandwidthValues, memcpyInstance);
        output->addTestcaseResults(bandwidthValues, buildOutputLabel());
        return;
#endif  // NVBW_HAVE_LE_API
    }

    // Raw path: one writer (rank 0) to multicast VA.
    PeerValueMatrix<double> bandwidthValues(1, 1, key);
    MemcpyInitiator* initiator = (initiator_ == CopyInitiator::TMA)
        ? static_cast<MemcpyInitiator*>(new MemcpyInitiatorTMA())
        : static_cast<MemcpyInitiator*>(new MemcpyInitiatorMultimemWrite());
    MemcpyOperation memcpyInstance(loopCount, initiator, new NodeHelperMulti(), PREFER_DST_CONTEXT, MemcpyOperation::SUM_BW);

    MultinodeDeviceBufferLocal src(size, /*MPI_rank=*/0);
    MultinodeDeviceBufferMulticast dst(size, /*MPI_rank=*/0);
    bandwidthValues.value(0, 0) = memcpyInstance.doMemcpy(src, dst);

    recordStability(bandwidthValues, memcpyInstance);
    output->addTestcaseResults(bandwidthValues, buildOutputLabel());
}

// ============================================================================
// MultinodeMultimemReduce
// ============================================================================

MultinodeMultimemReduce::MultinodeMultimemReduce()
    : Testcase("multinode_multicast_reduce",
               "\tMeasures multicast reduction-read bandwidth per rank "
               "(fabric.try_pullred with --force-le-variant, "
               "multimem.ld_reduce.v4.f32 otherwise).") {}

FilterResult MultinodeMultimemReduce::filter() {
    auto r = Testcase::filterHasMultipleGPUsMultinode();
    if (!r) return r;
    if (gSettings.bufferType == BufferType::LE) {
        // Each rank maps to localDevice = localRank % deviceCount, so the requirement
        // is one distinct local GPU per rank on a node (ranksPerNode <= deviceCount).
        if (deviceCount < ranksPerNode)
            return FilterResult::waive("requires at least one local GPU per rank (ranks-per-node exceeds local GPU count)");
        // LE path uses fabric.try_pullred via a multicast LogicalEndpoint; it does not
        // use cuMulticastCreate / multimem.ld_reduce, so hardware multicast support
        // (CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED) is not required.
        return Testcase::filterSupportsLogicalEndpoint();
    }
    // Default path uses multimem.ld_reduce which requires hardware multicast support.
    return Testcase::filterSupportsMulticast();
}

void MultinodeMultimemReduce::addRequiredFabricCliqueTypes(std::vector<FabricCliqueType> &types) const {
    addRequiredFabricCliqueType(types, getRequiredFabricCliqueType(true, CopyInitiator::TMA));
}

std::string MultinodeMultimemReduce::buildOutputLabel() const {
    return (gSettings.bufferType == BufferType::LE)
        ? "multicast read and reduce fabric.try_pullred (GB/s)"
        : "multicast read and reduce multimem.ld_reduce.v4.f32 (GB/s)";
}

void MultinodeMultimemReduce::run(unsigned long long size, unsigned long long loopCount) {
    if (gSettings.bufferType == BufferType::LE) {
#ifdef NVBW_HAVE_LE_API
        PeerValueMatrix<double> bandwidthValues(1, 1, key);
        MultinodeLEBufferMulticast multicastSrc(size, deviceCount);
        if (!allRanksLeReady(multicastSrc.isCreationSuccess())) {
            if (worldRank == 0)
                output->recordWarning("Skipping " + testKey() + ": one or more ranks could not create multicast LE.");
            bandwidthValues.value(0, 0) = 0.0;
            output->addTestcaseResults(bandwidthValues, buildOutputLabel());
            return;
        }
        // One receiver (rank 0), all ranks provide backing memory.
        MultinodeDeviceBufferLocal dst(size, 0);
        MemcpyOperation memcpyInstance(loopCount, new MemcpyInitiatorTMA(), new NodeHelperMulti(),
                                       PREFER_SRC_CONTEXT, MemcpyOperation::SUM_BW);
        bandwidthValues.value(0, 0) = memcpyInstance.doMemcpy(multicastSrc, dst);
        recordStability(bandwidthValues, memcpyInstance);
        output->addTestcaseResults(bandwidthValues, buildOutputLabel());
        return;
#endif  // NVBW_HAVE_LE_API
    }

    // One receiver (rank 0), all ranks provide backing memory — true one_from_all pullreduce.
    // MultinodeDeviceBufferMulticast(size, 0): all ranks bind their memory to the multicast VA.
    // MultinodeDeviceBufferLocal(size, 0): only rank 0 allocates a dst buffer.
    // → only rank 0 runs the kernel; others participate in MPI collectives and provide HBM.
    PeerValueMatrix<double> bandwidthValues(1, 1, key);
    MultinodeDeviceBufferMulticast src(size, 0);
    MultinodeDeviceBufferLocal     dst(size, 0);
    MemcpyOperation memcpyInstance(loopCount, new MemcpyInitiatorMultimemReadV4F32(), new NodeHelperMulti(),
                                   PREFER_SRC_CONTEXT, MemcpyOperation::SUM_BW);
    bandwidthValues.value(0, 0) = memcpyInstance.doMemcpy(src, dst);
    recordStability(bandwidthValues, memcpyInstance);
    output->addTestcaseResults(bandwidthValues, buildOutputLabel());
}

#endif  // MULTINODE

// ============================================================================
// DeviceToDeviceLatency: local device-to-device latency, all GPU pairs.
// LE variant (--force-le-variant): fabric_try_put[_counted] + consumer ack.
// Raw variant (default): UBLKCP 2 put+fence / UBLKCP counted.
// Single-process only (base test list); not part of the multinode suite.
// ============================================================================

std::string DeviceToDeviceLatency::makeKey(CountedWrite counted) {
    return (counted == CountedWrite::Yes) ? "device_to_device_latency_counted" : "device_to_device_latency";
}

std::string DeviceToDeviceLatency::makeDesc(CountedWrite counted) {
    return (counted == CountedWrite::Yes)
        ? "\tLocal device-to-device latency (counted), all GPU pairs.\n"
          "\t--force-le-variant: put_counted -> fabric_submit (unicast LE).\n"
          "\tDefault: UBLKCP counted (raw pointer)."
        : "\tLocal device-to-device latency (non-counted), all GPU pairs.\n"
          "\t--force-le-variant: fabric_try_put + consumer ack (unicast LE).\n"
          "\tDefault: UBLKCP 2 put + fence (raw pointer).";
}

DeviceToDeviceLatency::DeviceToDeviceLatency(CountedWrite counted)
    : Testcase(makeKey(counted), makeDesc(counted)), counted_(counted) {}

FilterResult DeviceToDeviceLatency::filter() {
    auto r = Testcase::filterHasMultipleGPUs();
    if (!r) return r;
    if (gSettings.bufferType == BufferType::LE)
        return Testcase::filterSupportsLogicalEndpoint();
    auto u = Testcase::filterSupportsUBLKCP();
    if (!u) return u;
    // The counted UBLKCP path uses cp.async.bulk.counted.global.shared::cta,
    // which is not supported on consumer Blackwell (sm_12X, e.g. gb202) and
    // traps there. Waive when any device in the system is sm_12X.
    if (counted_ == CountedWrite::Yes) {
        int deviceCount = 0;
        CU_ASSERT(cuDeviceGetCount(&deviceCount));
        for (int currentDevice = 0; currentDevice < deviceCount; currentDevice++) {
            CUdevice dev;
            CU_ASSERT(cuDeviceGet(&dev, currentDevice));
            int major = 0, minor = 0;
            CU_ASSERT(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev));
            CU_ASSERT(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev));
            if (major == 12) {
                return FilterResult::waive(
                    "device_to_device_latency_counted not supported on sm_12"
                    + std::to_string(minor) + " (device " + std::to_string(currentDevice) + ")");
            }
        }
    }
    return FilterResult::pass();
}

std::string DeviceToDeviceLatency::buildOutputLabel() const {
    const bool useLE = (gSettings.bufferType == BufferType::LE);
    if (counted_ == CountedWrite::Yes) {
        return useLE
            ? "fabric device-to-device counted (put_counted -> fabric_submit, unicast LE) latency GPU(row->col) (ns)"
            : "UBLKCP device-to-device counted latency GPU(row->col) (ns)";
    }
    return useLE
        ? "fabric device-to-device (put + consumer ack, unicast LE) latency GPU(row->col) (ns)"
        : "UBLKCP device-to-device (2 put+fence, round trip) latency GPU(row->col) (ns) per put";
}

void DeviceToDeviceLatency::run(unsigned long long size, unsigned long long loopCount) {
    // Iteration count is fixed at kDeviceToDeviceLatencyIterations; the standard loop-count parameter is unused.
    (void)loopCount;
    int numDevices = 0;
    CU_ASSERT(cuDeviceGetCount(&numDevices));
    PeerValueMatrix<double> latencyValues(numDevices, numDevices, key, gSettings.perfFormatter, LATENCY);
    // Use caller-supplied size when large enough to cover all put slots; otherwise default to
    // kDeviceToDeviceLatencyMinBufSize so the kernel's shared-memory loads always stay in bounds.
    size_t bufSize = (size >= kDeviceToDeviceLatencyPutSize) ? (size_t)size : kDeviceToDeviceLatencyMinBufSize;
    const bool useLE = (gSettings.bufferType == BufferType::LE);
    for (int producerDevice = 0; producerDevice < numDevices; producerDevice++) {
        for (int consumerDevice = 0; consumerDevice < numDevices; consumerDevice++) {
            if (producerDevice == consumerDevice)
                continue;
            int p2c = 0, c2p = 0;
            if (cuDeviceCanAccessPeer(&p2c, producerDevice, consumerDevice) != CUDA_SUCCESS ||
                cuDeviceCanAccessPeer(&c2p, consumerDevice, producerDevice) != CUDA_SUCCESS)
                continue;
            if (!p2c || !c2p)
                continue;
            double latencyNs = 0.0;
            if (useLE) {
#ifdef NVBW_HAVE_LE_API
                // The destination LE buffer must live on the consumer device: the
                // consumer kernel reads it locally, and the producer reaches it via
                // fabric_try_put (the A->B NVLink hop). It is a VMM/fabric allocation,
                // which cudaDeviceEnablePeerAccess does not map into a peer context, so
                // it must be owned by the device that dereferences it (the consumer).
                LogicalEndpointBuffer leDst(bufSize, consumerDevice);
                if (leDst.getDevicePtr() == 0)
                    continue;
                DeviceBuffer src(bufSize, producerDevice);
                CU_ASSERT(cuCtxSetCurrent(src.getPrimaryCtx()));
                CUdeviceptr ackPtr = 0;
                if (cuMemAlloc(&ackPtr, kDeviceToDeviceLatencyAckBytes) != CUDA_SUCCESS)
                    continue;
                latencyNs = (counted_ == CountedWrite::Yes)
                    ? leLaunchE2ELatencyUnicastCounted(
                          src.getBuffer(), leDst.getLeId(), leDst.getDevicePtr(), ackPtr,
                          kDeviceToDeviceLatencyIterations, producerDevice, consumerDevice)
                    : leLaunchE2ELatencyUnicast(
                          src.getBuffer(), leDst.getLeId(), leDst.getDevicePtr(), ackPtr,
                          kDeviceToDeviceLatencyIterations, producerDevice, consumerDevice);
                cuMemFree(ackPtr);
#endif
            } else {
                // Raw-pointer path: the producer writes to a "mailbox" buffer allocated on the
                // consumer device via UBLKCP (shared→global async copy).  Unlike a plain
                // destination buffer, the mailbox serves dual duty: it holds the put payload *and*
                // an application-managed sequence word that the consumer polls to detect arrival.
                // The term "mailbox" captures this combined data+notification role — the producer
                // deposits data and stamps a sequence number, the consumer polls the sequence word
                // to know the mail has arrived, then writes an ack back to the producer device.
                DeviceBuffer src(bufSize, producerDevice);
                CU_ASSERT(cuCtxSetCurrent(src.getPrimaryCtx()));
                CUdeviceptr ackPtr = 0;
                // Two words: [0] handshake ack, [1] producer-only watchdog status.
                if (cuMemAlloc(&ackPtr, kDeviceToDeviceLatencyAckBytes) != CUDA_SUCCESS)
                    continue;
                CUdevice devConsumer;
                CUcontext ctxConsumer;
                CU_ASSERT(cuDeviceGet(&devConsumer, consumerDevice));
                CU_ASSERT(cuDevicePrimaryCtxRetain(&ctxConsumer, devConsumer));
                CU_ASSERT(cuCtxSetCurrent(ctxConsumer));
                // Mailbox layout:
                //   Non-counted (kUblkcpMailboxBytes=32): [data 16B][sequence 16B]
                //     - two UBLKCP puts, sequence word at offset kUblkcpCounterOffset.
                //   Counted (kUblkcpCountedMailboxBytes=24): [data 16B][barMem counter 8B]
                //     - barMem is incremented by hardware TMA engine on counted-write arrival.
                size_t mailboxBytes = (counted_ == CountedWrite::Yes)
                    ? kUblkcpCountedMailboxBytes
                    : kUblkcpMailboxBytes;
                CUdeviceptr mailbox = 0;
                if (cuMemAlloc(&mailbox, mailboxBytes) != CUDA_SUCCESS) {
                    CU_ASSERT(cuCtxSetCurrent(src.getPrimaryCtx()));
                    cuMemFree(ackPtr);
                    cuDevicePrimaryCtxRelease(devConsumer);
                    continue;
                }
                CUstream streamConsumer;
                CU_ASSERT(cuStreamCreate(&streamConsumer, CU_STREAM_NON_BLOCKING));
                // Zero the entire mailbox before the run so consumer's i=0 sequence polls
                // (expect1=1, expect2=2) cannot spuriously match leftover bytes — don't rely
                // on cuMemAlloc returning zeroed memory.
                CU_ASSERT(cuMemsetD8Async(mailbox, 0, mailboxBytes, streamConsumer));
                CU_ASSERT(cuStreamSynchronize(streamConsumer));
                cuStreamDestroy(streamConsumer);
                CU_ASSERT(cuCtxSetCurrent(src.getPrimaryCtx()));
                CUstream streamProducer;
                CU_ASSERT(cuStreamCreate(&streamProducer, CU_STREAM_NON_BLOCKING));
                CU_ASSERT(cuMemsetD8Async(ackPtr, 0, kDeviceToDeviceLatencyAckBytes, streamProducer));
                CU_ASSERT(cuStreamSynchronize(streamProducer));
                cuStreamDestroy(streamProducer);
                latencyNs = (counted_ == CountedWrite::Yes)
                    ? leLaunchE2ELatencyUblkcpGSCounted(
                          src.getBuffer(), mailbox, mailbox, ackPtr,
                          kDeviceToDeviceLatencyIterations, producerDevice, consumerDevice)
                    : leLaunchE2ELatencyUblkcpGSE2E(
                          src.getBuffer(), mailbox, ackPtr,
                          kDeviceToDeviceLatencyIterations, producerDevice, consumerDevice);
                CU_ASSERT(cuCtxSetCurrent(ctxConsumer));
                cuMemFree(mailbox);
                cuDevicePrimaryCtxRelease(devConsumer);
                CU_ASSERT(cuCtxSetCurrent(src.getPrimaryCtx()));
                cuMemFree(ackPtr);
            }
            // A negative latency is the device-to-device watchdog failure sentinel: the
            // producer kernel's ack watchdog fired (the cross-GPU handshake stalled). Fail
            // the testcase instead of recording a stalled run as a normal completion.
            if (latencyNs < 0.0) {
                std::stringstream errmsg;
                errmsg << "DeviceToDeviceLatency: cross-GPU handshake stalled (watchdog timeout) for GPU "
                       << producerDevice << " -> " << consumerDevice << HOST_INFO << std::endl;
                RecordError(errmsg);
                MPI_ABORT;
                std::exit(1);
            }
            latencyValues.value(producerDevice, consumerDevice) = latencyNs;
        }
    }
    output->addTestcaseResults(latencyValues, buildOutputLabel());
}
