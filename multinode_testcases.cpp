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

#include <cuda.h>

#include "testcase.h"
#include "memcpy.h"
#include "common.h"
#include "output.h"
#ifdef MULTINODE
#include <mpi.h>
#include "multinode_memcpy.h"

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
        return Testcase::filterSupportsTMA();
    }
    return FilterResult::pass();
}

ContextPreference MultinodeDeviceToDeviceTransfer::getContextPreference() const {
    return (accessType_ == AccessType::Read) ? PREFER_DST_CONTEXT : PREFER_SRC_CONTEXT;
}

void MultinodeDeviceToDeviceTransfer::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bandwidthValues(worldSize, worldSize, key);
    MemcpyOperation memcpyInstance(loopCount, createMemcpyInitiator(),
                                   new NodeHelperMulti(), getContextPreference());

    auto pairs = generateTestPairs(worldSize, gSettings.targetNumPairs);

    for (const auto& pair : pairs) {
        int srcDeviceId = pair.first;
        int peerDeviceId = pair.second;

        MultinodeDeviceBufferUnicast srcNode(size, srcDeviceId);
        MultinodeDeviceBufferUnicast peerNode(size, peerDeviceId);

        if (accessType_ == AccessType::Read) {
            bandwidthValues.value(srcDeviceId, peerDeviceId) = memcpyInstance.doMemcpy(peerNode, srcNode);
        } else {
            bandwidthValues.value(srcDeviceId, peerDeviceId) = memcpyInstance.doMemcpy(srcNode, peerNode);
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
        case CopyInitiator::CE:  result += "cuMemcpyAsync"; break;
        case CopyInitiator::SM:  result += "SM copy kernel"; break;
        case CopyInitiator::TMA: result += "Tensor Memory Accelerator"; break;
    }
    result += " between each pair of accessible peers.\n\t";
    if (accessType == AccessType::Read) {
        result += "Read tests launch a copy from the peer device to the target using the target's context.";
    } else {
        result += "Write tests launch a copy from the target device to the peer using the target's context.";
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
        return Testcase::filterSupportsTMA();
    }
    return FilterResult::pass();
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

    cachedPairLoop(pairs, bandwidthValues1, bandwidthValues2, bandwidthValuesTotal,
        [&](int srcDeviceId, int peerDeviceId) -> std::vector<double> {
            MultinodeDeviceBufferUnicast src1(size, srcDeviceId), src2(size, srcDeviceId);
            MultinodeDeviceBufferUnicast peer1(size, peerDeviceId), peer2(size, peerDeviceId);

            std::vector<const MemcpyBuffer*> srcNodes = {&peer1, &src2};
            std::vector<const MemcpyBuffer*> peerNodes = {&src1, &peer2};

            return memcpyInstance.doMemcpyVector(srcNodes, peerNodes);
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
        result = "\tMeasures bandwidth of cuMemcpyAsync between each pair of accessible peers.\n"
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

void MultinodeBroadcastOneToAllSM::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bandwidthValues(1, worldSize, key);
    MemcpyOperation memcpyInstance(loopCount, new MemcpyInitiatorMulticastWrite(), new NodeHelperMulti(), PREFER_DST_CONTEXT, MemcpyOperation::SUM_BW);

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

void MultinodeBroadcastAllToAllSM::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bandwidthValues(1, 1, key);
    MemcpyOperation memcpyInstance(loopCount, new MemcpyInitiatorMulticastWrite(), new NodeHelperMulti(), PREFER_DST_CONTEXT, MemcpyOperation::SUM_BW);
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
            ? "\tMeasures bandwidth of TMA multicast fabric write from each rank."
            : "\tMeasures bandwidth of SM multicast fabric write from each rank.")
    , initiator_(initiator) {}

FilterResult MultinodeMulticastFabricPut::filter() {
    auto r = Testcase::filterHasMultipleGPUsMultinode();
    if (!r) return r;
    if (initiator_ == CopyInitiator::TMA) {
        auto tma = filterSupportsTMA();
        if (!tma) return tma;
    }
    return Testcase::filterSupportsMulticast();
}

std::string MultinodeMulticastFabricPut::buildOutputLabel() const {
    return (initiator_ == CopyInitiator::TMA)
        ? "multicast fabric put TMA (GB/s)"
        : "multicast fabric put SM (GB/s)";
}

void MultinodeMulticastFabricPut::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bandwidthValues(1, 1, key);

    // Use SM multicast write initiator for both SM and TMA variants (raw multicast VA path).
    MemcpyOperation memcpyInstance(loopCount, new MemcpyInitiatorMulticastWrite(),
                                   new NodeHelperMulti(), PREFER_DST_CONTEXT, MemcpyOperation::SUM_BW);

    double sumBw = 0.0;
    for (int dstDeviceId = 0; dstDeviceId < worldSize; dstDeviceId++) {
        std::vector<const MemcpyBuffer*> srcNodes;
        std::vector<const MemcpyBuffer*> dstNodes;

        srcNodes.push_back(new MultinodeDeviceBufferLocal(size, dstDeviceId));
        dstNodes.push_back(new MultinodeDeviceBufferMulticast(size, dstDeviceId));

        sumBw += memcpyInstance.doMemcpy(srcNodes, dstNodes);

        for (auto node : dstNodes) delete node;
        for (auto node : srcNodes) delete node;
    }

    bandwidthValues.value(0, 0) = (worldSize > 0) ? (sumBw / worldSize) : 0.0;
    recordStability(bandwidthValues, memcpyInstance);
    output->addTestcaseResults(bandwidthValues, buildOutputLabel());
}

#endif
