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

#include "testcase.h"
#include "kernels.cuh"
#include "memcpy.h"
#include "common.h"
#include "output.h"

void HostToDeviceSM::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bandwidthValues(1, deviceCount, key);
    MemcpyOperation memcpyInstance(loopCount, new MemcpyInitiatorSM());

    for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
        HostBuffer hostNode(size, deviceId);
        DeviceBuffer deviceNode(size, deviceId);

        bandwidthValues.value(0, deviceId) = memcpyInstance.doMemcpy(hostNode, deviceNode);
    }

    output->addTestcaseResults(bandwidthValues, "memcpy SM CPU(row) -> GPU(column) bandwidth (GB/s)");
}

void DeviceToHostSM::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bandwidthValues(1, deviceCount, key);
    MemcpyOperation memcpyInstance(loopCount, new MemcpyInitiatorSM());

    for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
        HostBuffer hostNode(size, deviceId);
        DeviceBuffer deviceNode(size, deviceId);

        bandwidthValues.value(0, deviceId) = memcpyInstance.doMemcpy(deviceNode, hostNode);
    }

    output->addTestcaseResults(bandwidthValues, "memcpy SM CPU(row) -> GPU(column) bandwidth (GB/s)");
}

void HostToDeviceBidirSM::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bandwidthValues(1, deviceCount, key);
    MemcpyOperation memcpyInstance(loopCount, new MemcpyInitiatorSM());

    for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
        // Double the size of the interference copy to ensure it interferes correctly
        HostBuffer host1(size, deviceId), host2(size * 2, deviceId);
        DeviceBuffer dev1(size, deviceId), dev2(size * 2, deviceId);

        std::vector<const MemcpyBuffer*> srcNodes = {&host1, &dev2};
        std::vector<const MemcpyBuffer*> dstNodes = {&dev1, &host2};

        bandwidthValues.value(0, deviceId) = memcpyInstance.doMemcpy(srcNodes, dstNodes);
    }

    output->addTestcaseResults(bandwidthValues, "memcpy SM CPU(row) <- GPU(column) bandwidth (GB/s)");
}

void DeviceToHostBidirSM::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bandwidthValues(1, deviceCount, key);
    MemcpyOperation memcpyInstance(loopCount, new MemcpyInitiatorSM());

    for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
        // Double the size of the interference copy to ensure it interferes correctly
        HostBuffer host1(size, deviceId), host2(size * 2, deviceId);
        DeviceBuffer dev1(size, deviceId), dev2(size * 2, deviceId);

        std::vector<const MemcpyBuffer*> srcNodes = {&dev1, &host2};
        std::vector<const MemcpyBuffer*> dstNodes = {&host1, &dev2};

        bandwidthValues.value(0, deviceId) = memcpyInstance.doMemcpy(srcNodes, dstNodes);
    }

    output->addTestcaseResults(bandwidthValues, "memcpy SM CPU(row) <- GPU(column) bandwidth (GB/s)");
}

// DtoD Read test - copy from dst to src (backwards) using src contxt
void DeviceToDeviceReadSM::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bandwidthValues(deviceCount, deviceCount, key);
    MemcpyOperation memcpyInstance(loopCount, new MemcpyInitiatorSM(), PREFER_DST_CONTEXT);

    for (int srcDeviceId = 0; srcDeviceId < deviceCount; srcDeviceId++) {
        for (int peerDeviceId = 0; peerDeviceId < deviceCount; peerDeviceId++) {
            if (peerDeviceId == srcDeviceId) {
                continue;
            }

            DeviceBuffer srcNode(size, srcDeviceId);
            DeviceBuffer peerNode(size, peerDeviceId);

            if (!srcNode.enablePeerAcess(peerNode)) {
                continue;
            }

            // swap src and peer nodes, but use srcNodes (the copy's destination) context
            bandwidthValues.value(srcDeviceId, peerDeviceId) = memcpyInstance.doMemcpy(peerNode, srcNode);
        }
    }

    output->addTestcaseResults(bandwidthValues, "memcpy SM CPU(row) -> GPU(column) bandwidth (GB/s)");
}

// DtoD Write test - copy from src to dst using src context
void DeviceToDeviceWriteSM::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bandwidthValues(deviceCount, deviceCount, key);
    MemcpyOperation memcpyInstance(loopCount, new MemcpyInitiatorSM());

    for (int srcDeviceId = 0; srcDeviceId < deviceCount; srcDeviceId++) {
        for (int peerDeviceId = 0; peerDeviceId < deviceCount; peerDeviceId++) {
            if (peerDeviceId == srcDeviceId) {
                continue;
            }

            DeviceBuffer srcNode(size, srcDeviceId);
            DeviceBuffer peerNode(size, peerDeviceId);

            if (!srcNode.enablePeerAcess(peerNode)) {
                continue;
            }

            bandwidthValues.value(srcDeviceId, peerDeviceId) = memcpyInstance.doMemcpy(srcNode, peerNode);
        }
    }

    output->addTestcaseResults(bandwidthValues, "memcpy SM CPU(row) -> GPU(column) bandwidth (GB/s)");
}

// DtoD Bidir Read test - copy from dst to src (backwards) using src contxt
void DeviceToDeviceBidirReadSM::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bandwidthValues(deviceCount, deviceCount, key);
    MemcpyOperation memcpyInstance(loopCount, new MemcpyInitiatorSM(), PREFER_DST_CONTEXT, MemcpyOperation::TOTAL_BW);


    for (int srcDeviceId = 0; srcDeviceId < deviceCount; srcDeviceId++) {
        for (int peerDeviceId = 0; peerDeviceId < deviceCount; peerDeviceId++) {
            if (peerDeviceId == srcDeviceId) {
                continue;
            }

            DeviceBuffer src1(size, srcDeviceId), src2(size, srcDeviceId);
            DeviceBuffer peer1(size, peerDeviceId), peer2(size, peerDeviceId);

            if (!src1.enablePeerAcess(peer1)) {
                continue;
            }

            // swap src and peer nodes, but use srcNodes (the copy's destination) context
            std::vector<const MemcpyBuffer*> srcNodes = {&peer1, &src2};
            std::vector<const MemcpyBuffer*> peerNodes = {&src1, &peer2};

            bandwidthValues.value(srcDeviceId, peerDeviceId) = memcpyInstance.doMemcpy(srcNodes, peerNodes);
        }
    }

    output->addTestcaseResults(bandwidthValues, "memcpy SM CPU(row) -> GPU(column) bandwidth (GB/s)");
}

// DtoD Bidir Write test - copy from src to dst using src context
void DeviceToDeviceBidirWriteSM::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bandwidthValues(deviceCount, deviceCount, key);
    MemcpyOperation memcpyInstance(loopCount, new MemcpyInitiatorSM(), PREFER_SRC_CONTEXT, MemcpyOperation::TOTAL_BW);

    for (int srcDeviceId = 0; srcDeviceId < deviceCount; srcDeviceId++) {
        for (int peerDeviceId = 0; peerDeviceId < deviceCount; peerDeviceId++) {
            if (peerDeviceId == srcDeviceId) {
                continue;
            }

            DeviceBuffer src1(size, srcDeviceId), src2(size, srcDeviceId);
            DeviceBuffer peer1(size, peerDeviceId), peer2(size, peerDeviceId);

            if (!src1.enablePeerAcess(peer1)) {
                continue;
            }

            std::vector<const MemcpyBuffer*> srcNodes = {&src1, &peer2};
            std::vector<const MemcpyBuffer*> peerNodes = {&peer1, &src2};

            bandwidthValues.value(srcDeviceId, peerDeviceId) = memcpyInstance.doMemcpy(srcNodes, peerNodes);
        }
    }

    output->addTestcaseResults(bandwidthValues, "memcpy SM CPU(row) -> GPU(column) bandwidth (GB/s)");
}

void AllToHostSM::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bandwidthValues(1, deviceCount, key);
    MemcpyOperation memcpyInstance(loopCount, new MemcpyInitiatorSM(), PREFER_SRC_CONTEXT, MemcpyOperation::USE_FIRST_BW);

    allHostHelper(size, memcpyInstance, bandwidthValues, false);

    output->addTestcaseResults(bandwidthValues, "memcpy SM CPU(row) -> GPU(column) bandwidth (GB/s)");
}

void AllToHostBidirSM::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bandwidthValues(1, deviceCount, key);
    MemcpyOperation memcpyInstance(loopCount, new MemcpyInitiatorSM(), PREFER_SRC_CONTEXT, MemcpyOperation::USE_FIRST_BW);

    allHostBidirHelper(size, memcpyInstance, bandwidthValues, false);

    output->addTestcaseResults(bandwidthValues, "memcpy SM CPU(row) -> GPU(column) bandwidth (GB/s)");
}

void HostToAllSM::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bandwidthValues(1, deviceCount, key);
    MemcpyOperation memcpyInstance(loopCount, new MemcpyInitiatorSM(), PREFER_SRC_CONTEXT, MemcpyOperation::USE_FIRST_BW);

    allHostHelper(size, memcpyInstance, bandwidthValues, true);

    output->addTestcaseResults(bandwidthValues, "memcpy SM CPU(row) -> GPU(column) bandwidth (GB/s)");
}

void HostToAllBidirSM::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bandwidthValues(1, deviceCount, key);
    MemcpyOperation memcpyInstance(loopCount, new MemcpyInitiatorSM(), PREFER_SRC_CONTEXT, MemcpyOperation::USE_FIRST_BW);

    allHostBidirHelper(size, memcpyInstance, bandwidthValues, true);

    output->addTestcaseResults(bandwidthValues, "memcpy SM CPU(row) -> GPU(column) bandwidth (GB/s)");
}

// Write test - copy from src to dst using src context
void AllToOneWriteSM::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bandwidthValues(1, deviceCount, key);
    MemcpyOperation memcpyInstance(loopCount, new MemcpyInitiatorSM(), PREFER_SRC_CONTEXT, MemcpyOperation::TOTAL_BW);
    
    allToOneHelper(size, memcpyInstance, bandwidthValues, false);

    output->addTestcaseResults(bandwidthValues, "memcpy SM CPU(row) -> GPU(column) bandwidth (GB/s)");
}

// Read test - copy from dst to src (backwards) using src contxt
void AllToOneReadSM::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bandwidthValues(1, deviceCount, key);
    MemcpyOperation memcpyInstance(loopCount, new MemcpyInitiatorSM(), PREFER_SRC_CONTEXT, MemcpyOperation::TOTAL_BW);
    allToOneHelper(size, memcpyInstance, bandwidthValues, true);

    output->addTestcaseResults(bandwidthValues, "memcpy SM CPU(row) -> GPU(column) bandwidth (GB/s)");
}

// Write test - copy from src to dst using src context
void OneToAllWriteSM::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bandwidthValues(1, deviceCount, key);
    MemcpyOperation memcpyInstance(loopCount, new MemcpyInitiatorSM(), PREFER_SRC_CONTEXT, MemcpyOperation::TOTAL_BW);
    oneToAllHelper(size, memcpyInstance, bandwidthValues, false);

    output->addTestcaseResults(bandwidthValues, "memcpy SM CPU(row) -> GPU(column) bandwidth (GB/s)");
}

// Read test - copy from dst to src (backwards) using src contxt
void OneToAllReadSM::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bandwidthValues(1, deviceCount, key);
    MemcpyOperation memcpyInstance(loopCount, new MemcpyInitiatorSM(), PREFER_SRC_CONTEXT, MemcpyOperation::TOTAL_BW);
    oneToAllHelper(size, memcpyInstance, bandwidthValues, true);

    output->addTestcaseResults(bandwidthValues, "memcpy SM CPU(row) -> GPU(column) bandwidth (GB/s)");
}
