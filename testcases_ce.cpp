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

void HostToDeviceCE::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bandwidthValues(1, deviceCount, key);
    MemcpyOperationCE memcpyInstance(loopCount);

    for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
        HostNode hostNode(size, deviceId);
        DeviceNode deviceNode(size, deviceId);

        bandwidthValues.value(0, deviceId) = memcpyInstance.doMemcpy(hostNode, deviceNode);
    }

    std::cout << "memcpy CE CPU(row) -> GPU(column) bandwidth (GB/s)" << std::endl;
    std::cout << std::fixed << std::setprecision(2) << bandwidthValues << std::endl;
}

void DeviceToHostCE::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bandwidthValues(1, deviceCount, key);
    MemcpyOperationCE memcpyInstance(loopCount);

    for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
        HostNode hostNode(size, deviceId);
        DeviceNode deviceNode(size, deviceId);

        bandwidthValues.value(0, deviceId) = memcpyInstance.doMemcpy(deviceNode, hostNode);
    }

    std::cout << "memcpy CE CPU(row) <- GPU(column) bandwidth (GB/s)" << std::endl;
    std::cout << std::fixed << std::setprecision(2) << bandwidthValues << std::endl;
}

void HostToDeviceBidirCE::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bandwidthValues(1, deviceCount, key);
    MemcpyOperationCE memcpyInstance(loopCount);

    for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
        // Double the size of the interference copy to ensure it interferes correctly
        HostNode host1(size, deviceId), host2(size * 2, deviceId);
        DeviceNode dev1(size, deviceId), dev2(size * 2, deviceId);

        std::vector<const MemcpyNode*> srcNodes = {&host1, &dev2};
        std::vector<const MemcpyNode*> dstNodes = {&dev1, &host2};

        bandwidthValues.value(0, deviceId) = memcpyInstance.doMemcpy(srcNodes, dstNodes);
    }

    std::cout << "memcpy CE CPU(row) <-> GPU(column) bandwidth (GB/s)" << std::endl;
    std::cout << std::fixed << std::setprecision(2) << bandwidthValues << std::endl;
}

void DeviceToHostBidirCE::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bandwidthValues(1, deviceCount, key);
    MemcpyOperationCE memcpyInstance(loopCount);

    for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
        // Double the size of the interference copy to ensure it interferes correctly
        HostNode host1(size, deviceId), host2(size * 2, deviceId);
        DeviceNode dev1(size, deviceId), dev2(size * 2, deviceId);

        std::vector<const MemcpyNode*> srcNodes = {&dev1, &host2};
        std::vector<const MemcpyNode*> dstNodes = {&host1, &dev2};

        bandwidthValues.value(0, deviceId) = memcpyInstance.doMemcpy(srcNodes, dstNodes);
    }

    std::cout << "memcpy CE CPU(row) <-> GPU(column) bandwidth (GB/s)" << std::endl;
    std::cout << std::fixed << std::setprecision(2) << bandwidthValues << std::endl;
}

// DtoD Read test - copy from dst to src (backwards) using src contxt
void DeviceToDeviceReadCE::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bandwidthValues(deviceCount, deviceCount, key);
    MemcpyOperationCE memcpyInstance(loopCount, MemcpyOperation::PREFER_DST_CONTEXT);

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
void DeviceToDeviceWriteCE::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bandwidthValues(deviceCount, deviceCount, key);
    MemcpyOperationCE memcpyInstance(loopCount);

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

    std::cout << "memcpy CE GPU(row) <- GPU(column) bandwidth (GB/s)" << std::endl;
    std::cout << std::fixed << std::setprecision(2) << bandwidthValues << std::endl;
}

// DtoD Bidir Read test - copy from dst to src (backwards) using src contxt
void DeviceToDeviceBidirReadCE::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bandwidthValues(deviceCount, deviceCount, key);
    MemcpyOperationCE memcpyInstance(loopCount, MemcpyOperation::PREFER_DST_CONTEXT);

    for (int srcDeviceId = 0; srcDeviceId < deviceCount; srcDeviceId++) {
        for (int peerDeviceId = 0; peerDeviceId < deviceCount; peerDeviceId++) {
            if (peerDeviceId == srcDeviceId) {
                continue;
            }

            // Double the size of the interference copy to ensure it interferes correctly
            DeviceNode src1(size, srcDeviceId), src2(size * 2, srcDeviceId);
            DeviceNode peer1(size, peerDeviceId), peer2(size * 2, peerDeviceId);

            if (!src1.enablePeerAcess(peer1)) {
                continue;
            }

            // swap src and peer nodes, but use srcNodes (the copy's destination) context
            std::vector<const MemcpyNode*> srcNodes = {&peer1, &src2};
            std::vector<const MemcpyNode*> peerNodes = {&src1, &peer2};

            bandwidthValues.value(srcDeviceId, peerDeviceId) = memcpyInstance.doMemcpy(srcNodes, peerNodes);
        }
    }

    std::cout << "memcpy CE GPU(row) <-> GPU(column) bandwidth (GB/s)" << std::endl;
    std::cout << std::fixed << std::setprecision(2) << bandwidthValues << std::endl;
}

// DtoD Bidir Write test - copy from src to dst using src context
void DeviceToDeviceBidirWriteCE::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bandwidthValues(deviceCount, deviceCount, key);
    MemcpyOperationCE memcpyInstance(loopCount);

    for (int srcDeviceId = 0; srcDeviceId < deviceCount; srcDeviceId++) {
        for (int peerDeviceId = 0; peerDeviceId < deviceCount; peerDeviceId++) {
            if (peerDeviceId == srcDeviceId) {
                continue;
            }

            // Double the size of the interference copy to ensure it interferes correctly
            DeviceNode src1(size, srcDeviceId), src2(size * 2, srcDeviceId);
            DeviceNode peer1(size, peerDeviceId), peer2(size * 2, peerDeviceId);

            if (!src1.enablePeerAcess(peer1)) {
                continue;
            }

            std::vector<const MemcpyNode*> srcNodes = {&src1, &peer2};
            std::vector<const MemcpyNode*> peerNodes = {&peer1, &src2};

            bandwidthValues.value(srcDeviceId, peerDeviceId) = memcpyInstance.doMemcpy(srcNodes, peerNodes);
        }
    }

    std::cout << "memcpy CE GPU(row) <-> GPU(column) bandwidth (GB/s)" << std::endl;
    std::cout << std::fixed << std::setprecision(2) << bandwidthValues << std::endl;
}

void AllToHostCE::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bandwidthValues(1, deviceCount, key);
    MemcpyOperationCE memcpyInstance(loopCount);

    allHostHelper(size, memcpyInstance, bandwidthValues, false);

    std::cout << "memcpy CE CPU(row) <- GPU(column) bandwidth (GB/s)" << std::endl;
    std::cout << std::fixed << std::setprecision(2) << bandwidthValues << std::endl;
}

void HostToAllCE::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bandwidthValues(1, deviceCount, key);
    MemcpyOperationCE memcpyInstance(loopCount);

    allHostHelper(size, memcpyInstance, bandwidthValues, true);

    std::cout << "memcpy CE CPU(row) -> GPU(column) bandwidth (GB/s)" << std::endl;
    std::cout << std::fixed << std::setprecision(2) << bandwidthValues << std::endl;
}

// Write test - copy from src to dst using src context
void AllToOneWriteCE::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bandwidthValues(1, deviceCount, key);
    MemcpyOperationCE memcpyInstance(loopCount, MemcpyOperation::PREFER_SRC_CONTEXT, MemcpyOperation::TOTAL_BW);
    allToOneHelper(size, memcpyInstance, bandwidthValues, false);

    std::cout << "memcpy CE All Gpus -> GPU(column) total bandwidth (GB/s)" << std::endl;
    std::cout << std::fixed << std::setprecision(2) << bandwidthValues << std::endl;
}

// Read test - copy from dst to src (backwards) using src contxt
void AllToOneReadCE::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bandwidthValues(1, deviceCount, key);
    MemcpyOperationCE memcpyInstance(loopCount, MemcpyOperation::PREFER_DST_CONTEXT, MemcpyOperation::TOTAL_BW);
    allToOneHelper(size, memcpyInstance, bandwidthValues, true);

    std::cout << "memcpy CE All Gpus <- GPU(column) total bandwidth (GB/s)" << std::endl;
    std::cout << std::fixed << std::setprecision(2) << bandwidthValues << std::endl;
}

// Write test - copy from src to dst using src context
void OneToAllWriteCE::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bandwidthValues(1, deviceCount, key);
    MemcpyOperationCE memcpyInstance(loopCount, MemcpyOperation::PREFER_SRC_CONTEXT, MemcpyOperation::TOTAL_BW);
    oneToAllHelper(size, memcpyInstance, bandwidthValues, false);

    std::cout << "memcpy CE GPU(column) -> All GPUs total bandwidth (GB/s)" << std::endl;
    std::cout << std::fixed << std::setprecision(2) << bandwidthValues << std::endl;
}

// Read test - copy from dst to src (backwards) using src contxt
void OneToAllReadCE::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bandwidthValues(1, deviceCount, key);
    MemcpyOperationCE memcpyInstance(loopCount, MemcpyOperation::PREFER_DST_CONTEXT, MemcpyOperation::TOTAL_BW);
    oneToAllHelper(size, memcpyInstance, bandwidthValues, true);

    std::cout << "memcpy CE GPU(column) <- All GPUs total bandwidth (GB/s)" << std::endl;
    std::cout << std::fixed << std::setprecision(2) << bandwidthValues << std::endl;
}
