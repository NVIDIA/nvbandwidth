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

// DtoD Read test - copy from dst to src (backwards) using src contxt
void MultinodeDeviceToDeviceReadCE::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bandwidthValues(worldSize, worldSize, key);
    MemcpyOperation memcpyInstance(loopCount, new MemcpyInitiatorCE(), new NodeHelperMulti(), PREFER_DST_CONTEXT);

    for (int srcDeviceId = 0; srcDeviceId < worldSize; srcDeviceId++) {
        for (int peerDeviceId = 0; peerDeviceId < worldSize; peerDeviceId++) {
            if (peerDeviceId == srcDeviceId) {
                continue;
            }
            MultinodeDeviceBufferUnicast srcNode(size, srcDeviceId);
            MultinodeDeviceBufferUnicast peerNode(size, peerDeviceId);

            // swap src and peer nodes, but use srcNodes (the copy's destination) context
            bandwidthValues.value(srcDeviceId, peerDeviceId) = memcpyInstance.doMemcpy(peerNode, srcNode);
        }
    }

    output->addTestcaseResults(bandwidthValues, "memcpy CE GPU(row) -> GPU(column) bandwidth (GB/s)");
}


// DtoD Write test - copy from src to dst using src context
void MultinodeDeviceToDeviceWriteCE::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bandwidthValues(worldSize, worldSize, key);
    MemcpyOperation memcpyInstance(loopCount, new MemcpyInitiatorCE(), new NodeHelperMulti());

    for (int srcDeviceId = 0; srcDeviceId < worldSize; srcDeviceId++) {
        for (int peerDeviceId = 0; peerDeviceId < worldSize; peerDeviceId++) {
            if (peerDeviceId == srcDeviceId) {
                continue;
            }

            MultinodeDeviceBufferUnicast srcNode(size, srcDeviceId);
            MultinodeDeviceBufferUnicast peerNode(size, peerDeviceId);

            bandwidthValues.value(srcDeviceId, peerDeviceId) = memcpyInstance.doMemcpy(srcNode, peerNode);
        }
    }

    output->addTestcaseResults(bandwidthValues, "memcpy CE GPU(row) <- GPU(column) bandwidth (GB/s)");
}

// DtoD Bidir Read test - copy from dst to src (backwards) using src contxt
void MultinodeDeviceToDeviceBidirReadCE::run(unsigned long long size, unsigned long long loopCount) {
    MemcpyOperation memcpyInstance(loopCount, new MemcpyInitiatorCE(), new NodeHelperMulti(), PREFER_DST_CONTEXT, MemcpyOperation::VECTOR_BW);
    PeerValueMatrix<double> bandwidthValuesRead1(worldSize, worldSize, key + "_read1");
    PeerValueMatrix<double> bandwidthValuesRead2(worldSize, worldSize, key + "_read2");
    PeerValueMatrix<double> bandwidthValuesTotal(worldSize, worldSize, key + "_total");

    for (int srcDeviceId = 0; srcDeviceId < worldSize; srcDeviceId++) {
        for (int peerDeviceId = 0; peerDeviceId < worldSize; peerDeviceId++) {
            if (peerDeviceId == srcDeviceId) {
                continue;
            }

            // Double the size of the interference copy to ensure it interferes correctly
            MultinodeDeviceBufferUnicast src1(size, srcDeviceId), src2(size, srcDeviceId);
            MultinodeDeviceBufferUnicast peer1(size, peerDeviceId), peer2(size, peerDeviceId);

            // swap src and peer nodes, but use srcNodes (the copy's destination) context
            std::vector<const MemcpyBuffer*> srcNodes = {&peer1, &src2};
            std::vector<const MemcpyBuffer*> peerNodes = {&src1, &peer2};

            auto results = memcpyInstance.doMemcpyVector(srcNodes, peerNodes);
            bandwidthValuesRead1.value(srcDeviceId, peerDeviceId) = results[0];
            bandwidthValuesRead2.value(srcDeviceId, peerDeviceId) = results[1];
            bandwidthValuesTotal.value(srcDeviceId, peerDeviceId) = results[0] + results[1];
        }
    }

    output->addTestcaseResults(bandwidthValuesRead1, "memcpy CE CPU(row) <-> GPU(column) Read1 bandwidth (GB/s)");
    output->addTestcaseResults(bandwidthValuesRead2, "memcpy CE CPU(row) <-> GPU(column) Read2 bandwidth (GB/s)");
    output->addTestcaseResults(bandwidthValuesTotal, "memcpy CE CPU(row) <-> GPU(column) Total bandwidth (GB/s)");
}

// DtoD Bidir Write test - copy from src to dst using src context
void MultinodeDeviceToDeviceBidirWriteCE::run(unsigned long long size, unsigned long long loopCount) {
    MemcpyOperation memcpyInstance(loopCount, new MemcpyInitiatorCE(), new NodeHelperMulti(), PREFER_SRC_CONTEXT, MemcpyOperation::VECTOR_BW);
    PeerValueMatrix<double> bandwidthValuesWrite1(worldSize, worldSize, key + "_write1");
    PeerValueMatrix<double> bandwidthValuesWrite2(worldSize, worldSize, key + "_write2");
    PeerValueMatrix<double> bandwidthValuesTotal(worldSize, worldSize, key + "_total");

    for (int srcDeviceId = 0; srcDeviceId < worldSize; srcDeviceId++) {
        for (int peerDeviceId = 0; peerDeviceId < worldSize; peerDeviceId++) {
            if (peerDeviceId == srcDeviceId) {
                continue;
            }

            // Double the size of the interference copy to ensure it interferes correctly
            MultinodeDeviceBufferUnicast src1(size, srcDeviceId), src2(size, srcDeviceId);
            MultinodeDeviceBufferUnicast peer1(size, peerDeviceId), peer2(size, peerDeviceId);

            std::vector<const MemcpyBuffer*> srcNodes = {&src1, &peer2};
            std::vector<const MemcpyBuffer*> peerNodes = {&peer1, &src2};

            auto results = memcpyInstance.doMemcpyVector(srcNodes, peerNodes);
            bandwidthValuesWrite1.value(srcDeviceId, peerDeviceId) = results[0];
            bandwidthValuesWrite2.value(srcDeviceId, peerDeviceId) = results[1];
            bandwidthValuesTotal.value(srcDeviceId, peerDeviceId) = results[0] + results[1];
        }
    }

    output->addTestcaseResults(bandwidthValuesWrite1, "memcpy CE CPU(row) <-> GPU(column) Read1 bandwidth (GB/s)");
    output->addTestcaseResults(bandwidthValuesWrite2, "memcpy CE CPU(row) <-> GPU(column) Read2 bandwidth (GB/s)");
    output->addTestcaseResults(bandwidthValuesTotal, "memcpy CE CPU(row) <-> GPU(column) Total bandwidth (GB/s)");
}


// DtoD Read test - copy from dst to src (backwards) using src contxt
void MultinodeDeviceToDeviceReadSM::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bandwidthValues(worldSize, worldSize, key);
    MemcpyOperation memcpyInstance(loopCount, new MemcpyInitiatorSM(), new NodeHelperMulti(), PREFER_DST_CONTEXT);

    for (int srcDeviceId = 0; srcDeviceId < worldSize; srcDeviceId++) {
        for (int peerDeviceId = 0; peerDeviceId < worldSize; peerDeviceId++) {
            if (peerDeviceId == srcDeviceId) {
                continue;
            }

            MultinodeDeviceBufferUnicast srcNode(size, srcDeviceId);
            MultinodeDeviceBufferUnicast peerNode(size, peerDeviceId);

            // swap src and peer nodes, but use srcNodes (the copy's destination) context
            bandwidthValues.value(srcDeviceId, peerDeviceId) = memcpyInstance.doMemcpy(peerNode, srcNode);
        }
    }

    output->addTestcaseResults(bandwidthValues, "memcpy CE GPU(row) -> GPU(column) bandwidth (GB/s)");
}

// DtoD Write test - copy from src to dst using src context
void MultinodeDeviceToDeviceWriteSM::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bandwidthValues(worldSize, worldSize, key);
    MemcpyOperation memcpyInstance(loopCount, new MemcpyInitiatorSM(), new NodeHelperMulti());

    for (int srcDeviceId = 0; srcDeviceId < worldSize; srcDeviceId++) {
        for (int peerDeviceId = 0; peerDeviceId < worldSize; peerDeviceId++) {
            if (peerDeviceId == srcDeviceId) {
                continue;
            }

            MultinodeDeviceBufferUnicast srcNode(size, srcDeviceId);
            MultinodeDeviceBufferUnicast peerNode(size, peerDeviceId);

            bandwidthValues.value(srcDeviceId, peerDeviceId) = memcpyInstance.doMemcpy(srcNode, peerNode);
        }
    }

    output->addTestcaseResults(bandwidthValues, "memcpy SM GPU(row) <- GPU(column) bandwidth (GB/s)");
}

// DtoD Bidir Read test - copy from dst to src (backwards) using src contxt
void MultinodeDeviceToDeviceBidirReadSM::run(unsigned long long size, unsigned long long loopCount) {
    MemcpyOperation memcpyInstance(loopCount, new MemcpyInitiatorSM(), new NodeHelperMulti(), PREFER_DST_CONTEXT, MemcpyOperation::VECTOR_BW);
    PeerValueMatrix<double> bandwidthValuesRead1(worldSize, worldSize, key + "_read1");
    PeerValueMatrix<double> bandwidthValuesRead2(worldSize, worldSize, key + "_read2");
    PeerValueMatrix<double> bandwidthValuesTotal(worldSize, worldSize, key + "_total");

    for (int srcDeviceId = 0; srcDeviceId < worldSize; srcDeviceId++) {
        for (int peerDeviceId = 0; peerDeviceId < worldSize; peerDeviceId++) {
            if (peerDeviceId == srcDeviceId) {
                continue;
            }

            MultinodeDeviceBufferUnicast src1(size, srcDeviceId), src2(size, srcDeviceId);
            MultinodeDeviceBufferUnicast peer1(size, peerDeviceId), peer2(size, peerDeviceId);

            // swap src and peer nodes, but use srcNodes (the copy's destination) context
            std::vector<const MemcpyBuffer*> srcNodes = {&peer1, &src2};
            std::vector<const MemcpyBuffer*> peerNodes = {&src1, &peer2};

            auto results = memcpyInstance.doMemcpyVector(srcNodes, peerNodes);
            bandwidthValuesRead1.value(srcDeviceId, peerDeviceId) = results[0];
            bandwidthValuesRead2.value(srcDeviceId, peerDeviceId) = results[1];
            bandwidthValuesTotal.value(srcDeviceId, peerDeviceId) = results[0] + results[1];
        }
    }

    output->addTestcaseResults(bandwidthValuesRead1, "memcpy SM CPU(row) <-> GPU(column) Read1 bandwidth (GB/s)");
    output->addTestcaseResults(bandwidthValuesRead2, "memcpy SM CPU(row) <-> GPU(column) Read2 bandwidth (GB/s)");
    output->addTestcaseResults(bandwidthValuesTotal, "memcpy SM CPU(row) <-> GPU(column) Total bandwidth (GB/s)");
}

// DtoD Bidir Write test - copy from src to dst using src context
void MultinodeDeviceToDeviceBidirWriteSM::run(unsigned long long size, unsigned long long loopCount) {
    MemcpyOperation memcpyInstance(loopCount, new MemcpyInitiatorSM(), new NodeHelperMulti(), PREFER_SRC_CONTEXT, MemcpyOperation::VECTOR_BW);
    PeerValueMatrix<double> bandwidthValuesWrite1(worldSize, worldSize, key + "_write1");
    PeerValueMatrix<double> bandwidthValuesWrite2(worldSize, worldSize, key + "_write2");
    PeerValueMatrix<double> bandwidthValuesTotal(worldSize, worldSize, key + "_total");

    for (int srcDeviceId = 0; srcDeviceId < worldSize; srcDeviceId++) {
        for (int peerDeviceId = 0; peerDeviceId < worldSize; peerDeviceId++) {
            if (peerDeviceId == srcDeviceId) {
                continue;
            }

            MultinodeDeviceBufferUnicast src1(size, srcDeviceId), src2(size, srcDeviceId);
            MultinodeDeviceBufferUnicast peer1(size, peerDeviceId), peer2(size, peerDeviceId);

            std::vector<const MemcpyBuffer*> srcNodes = {&src1, &peer2};
            std::vector<const MemcpyBuffer*> peerNodes = {&peer1, &src2};

            auto results = memcpyInstance.doMemcpyVector(srcNodes, peerNodes);
            bandwidthValuesWrite1.value(srcDeviceId, peerDeviceId) = results[0];
            bandwidthValuesWrite2.value(srcDeviceId, peerDeviceId) = results[1];
            bandwidthValuesTotal.value(srcDeviceId, peerDeviceId) = results[0] + results[1];
        }
    }

    output->addTestcaseResults(bandwidthValuesWrite1, "memcpy SM CPU(row) <-> GPU(column) Write1 bandwidth (GB/s)");
    output->addTestcaseResults(bandwidthValuesWrite2, "memcpy SM CPU(row) <-> GPU(column) Write2 bandwidth (GB/s)");
    output->addTestcaseResults(bandwidthValuesTotal, "memcpy SM CPU(row) <-> GPU(column) Total bandwidth (GB/s)");
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

    output->addTestcaseResults(bandwidthValues, "Bisect benchmarking, simultaneous write CE BW");
}

#endif
