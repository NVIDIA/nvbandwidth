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

void HostToDeviceSM::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bandwidthValues(1, deviceCount, key);
    MemcpyOperationSM memcpyInstance(loopCount);

    for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
        HostNode hostNode(size, deviceId);
        DeviceNode deviceNode(size, deviceId);

        bandwidthValues.value(0, deviceId) = memcpyInstance.doMemcpy(hostNode, deviceNode);
    }

    std::cout << "memcpy SM CPU(row) -> GPU(column) bandwidth (GB/s)" << std::endl;
    std::cout << std::fixed << std::setprecision(2) << bandwidthValues << std::endl;
}

void DeviceToHostSM::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bandwidthValues(1, deviceCount, key);
    MemcpyOperationSM memcpyInstance(loopCount);

    for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
        HostNode hostNode(size, deviceId);
        DeviceNode deviceNode(size, deviceId);

        bandwidthValues.value(0, deviceId) = memcpyInstance.doMemcpy(deviceNode, hostNode);
    }

    std::cout << "memcpy SM CPU(row) <- GPU(column) bandwidth (GB/s)" << std::endl;
    std::cout << std::fixed << std::setprecision(2) << bandwidthValues << std::endl;
}

// DtoD Read test - copy from dst to src (backwards) using src contxt
void DeviceToDeviceReadSM::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bandwidthValues(deviceCount, deviceCount, key);
    MemcpyOperationSM memcpyInstance(loopCount, MemcpyOperation::PREFER_DST_CONTEXT);

    for (int srcDeviceId = 0; srcDeviceId < deviceCount; srcDeviceId++) {
        for (int peerDeviceId = 0; peerDeviceId < deviceCount; peerDeviceId++) {
            if (peerDeviceId == srcDeviceId) {
                continue;
            }

            DeviceNode srcNode(size, srcDeviceId);
            DeviceNode peerNode(size, peerDeviceId);

            if (!srcNode.enablePeerAcess(peerNode)) {
                continue;
            }

            // swap src and peer nodes, but use srcNodes (the copy's destination) context
            bandwidthValues.value(srcDeviceId, peerDeviceId) = memcpyInstance.doMemcpy(peerNode, srcNode);
        }
    }

    std::cout << "memcpy CE GPU(row) -> GPU(column) bandwidth (GB/s)" << std::endl;
    std::cout << std::fixed << std::setprecision(2) << bandwidthValues << std::endl;
}

// DtoD Write test - copy from src to dst using src context
void DeviceToDeviceWriteSM::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bandwidthValues(deviceCount, deviceCount, key);
    MemcpyOperationSM memcpyInstance(loopCount);

    for (int srcDeviceId = 0; srcDeviceId < deviceCount; srcDeviceId++) {
        for (int peerDeviceId = 0; peerDeviceId < deviceCount; peerDeviceId++) {
            if (peerDeviceId == srcDeviceId) {
                continue;
            }

            DeviceNode srcNode(size, srcDeviceId);
            DeviceNode peerNode(size, peerDeviceId);

            if (!srcNode.enablePeerAcess(peerNode)) {
                continue;
            }

            bandwidthValues.value(srcDeviceId, peerDeviceId) = memcpyInstance.doMemcpy(srcNode, peerNode);
        }
    }

    std::cout << "memcpy SM GPU(row) <- GPU(column) bandwidth (GB/s)" << std::endl;
    std::cout << std::fixed << std::setprecision(2) << bandwidthValues << std::endl;
}

// DtoD Bidir Read test - copy from dst to src (backwards) using src contxt
void DeviceToDeviceBidirReadSM::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bandwidthValues(deviceCount, deviceCount, key);
    MemcpyOperationSM memcpyInstance(loopCount, MemcpyOperation::PREFER_DST_CONTEXT);

    for (int srcDeviceId = 0; srcDeviceId < deviceCount; srcDeviceId++) {
        for (int peerDeviceId = 0; peerDeviceId < deviceCount; peerDeviceId++) {
            if (peerDeviceId == srcDeviceId) {
                continue;
            }

            DeviceNode src1(size, srcDeviceId), src2(size, srcDeviceId);
            DeviceNode peer1(size, peerDeviceId), peer2(size, peerDeviceId);

            if (!src1.enablePeerAcess(peer1)) {
                continue;
            }

            // swap src and peer nodes, but use srcNodes (the copy's destination) context
            std::vector<const MemcpyNode*> srcNodes = {&peer1, &src2};
            std::vector<const MemcpyNode*> peerNodes = {&src1, &peer2};

            bandwidthValues.value(srcDeviceId, peerDeviceId) = memcpyInstance.doMemcpy(srcNodes, peerNodes);
        }
    }

    std::cout << "memcpy SM GPU(row) -> GPU(column) bandwidth (GB/s)" << std::endl;
    std::cout << std::fixed << std::setprecision(2) << bandwidthValues << std::endl;
}

// DtoD Bidir Write test - copy from src to dst using src context
void DeviceToDeviceBidirWriteSM::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bandwidthValues(deviceCount, deviceCount, key);
    MemcpyOperationSM memcpyInstance(loopCount);

    for (int srcDeviceId = 0; srcDeviceId < deviceCount; srcDeviceId++) {
        for (int peerDeviceId = 0; peerDeviceId < deviceCount; peerDeviceId++) {
            if (peerDeviceId == srcDeviceId) {
                continue;
            }

            DeviceNode src1(size, srcDeviceId), src2(size, srcDeviceId);
            DeviceNode peer1(size, peerDeviceId), peer2(size, peerDeviceId);

            if (!src1.enablePeerAcess(peer1)) {
                continue;
            }

            std::vector<const MemcpyNode*> srcNodes = {&src1, &peer2};
            std::vector<const MemcpyNode*> peerNodes = {&peer1, &src2};

            bandwidthValues.value(srcDeviceId, peerDeviceId) = memcpyInstance.doMemcpy(srcNodes, peerNodes);
        }
    }

    std::cout << "memcpy SM GPU(row) <- GPU(column) bandwidth (GB/s)" << std::endl;
    std::cout << std::fixed << std::setprecision(2) << bandwidthValues << std::endl;
}

void AllToHostSM::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bandwidthValues(1, deviceCount, key);
    MemcpyOperationSM memcpyInstance(loopCount, MemcpyOperation::PREFER_SRC_CONTEXT, MemcpyOperation::USE_FIRST_BW);

    allHostHelper(size, memcpyInstance, bandwidthValues, false);

    std::cout << "memcpy SM CPU(row) <- GPU(column) bandwidth (GB/s)" << std::endl;
    std::cout << std::fixed << std::setprecision(2) << bandwidthValues << std::endl;
}

void HostToAllSM::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bandwidthValues(1, deviceCount, key);
    MemcpyOperationSM memcpyInstance(loopCount, MemcpyOperation::PREFER_SRC_CONTEXT, MemcpyOperation::USE_FIRST_BW);

    allHostHelper(size, memcpyInstance, bandwidthValues, true);

    std::cout << "memcpy SM CPU(row) -> GPU(column) bandwidth (GB/s)" << std::endl;
    std::cout << std::fixed << std::setprecision(2) << bandwidthValues << std::endl;
}

// Write test - copy from src to dst using src context
void AllToOneWriteSM::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bandwidthValues(1, deviceCount, key);
    MemcpyOperationSM memcpyInstance(loopCount, MemcpyOperation::PREFER_SRC_CONTEXT, MemcpyOperation::TOTAL_BW);
    allToOneHelper(size, memcpyInstance, bandwidthValues, false);

    std::cout << "memcpy SM All Gpus -> GPU(column) total bandwidth (GB/s)" << std::endl;
    std::cout << std::fixed << std::setprecision(2) << bandwidthValues << std::endl;
}

// Read test - copy from dst to src (backwards) using src contxt
void AllToOneReadSM::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bandwidthValues(1, deviceCount, key);
    MemcpyOperationSM memcpyInstance(loopCount, MemcpyOperation::PREFER_DST_CONTEXT, MemcpyOperation::TOTAL_BW);
    allToOneHelper(size, memcpyInstance, bandwidthValues, true);

    std::cout << "memcpy SM All GPUs <- GPU(column) total bandwidth (GB/s)" << std::endl;
    std::cout << std::fixed << std::setprecision(2) << bandwidthValues << std::endl;
}

// Write test - copy from src to dst using src context
void OneToAllWriteSM::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bandwidthValues(1, deviceCount, key);
    MemcpyOperationSM memcpyInstance(loopCount, MemcpyOperation::PREFER_SRC_CONTEXT, MemcpyOperation::TOTAL_BW);
    oneToAllHelper(size, memcpyInstance, bandwidthValues, false);

    std::cout << "memcpy SM GPU(column) -> All GPUs total bandwidth (GB/s)" << std::endl;
    std::cout << std::fixed << std::setprecision(2) << bandwidthValues << std::endl;
}

// Read test - copy from dst to src (backwards) using src contxt
void OneToAllReadSM::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bandwidthValues(1, deviceCount, key);
    MemcpyOperationSM memcpyInstance(loopCount, MemcpyOperation::PREFER_DST_CONTEXT, MemcpyOperation::TOTAL_BW);
    oneToAllHelper(size, memcpyInstance, bandwidthValues, true);

    std::cout << "memcpy SM GPU(column) <- All GPUs total bandwidth (GB/s)" << std::endl;
    std::cout << std::fixed << std::setprecision(2) << bandwidthValues << std::endl;
}
