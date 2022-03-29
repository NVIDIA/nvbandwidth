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

Testcase::Testcase(std::string key, std::string desc) : 
    key(std::move(key)), desc(std::move(desc))
{}

std::string Testcase::testKey() { return key; }
std::string Testcase::testDesc() { return desc; }

bool Testcase::filterHasAccessiblePeerPairs() {
    int deviceCount = 0;
    CU_ASSERT(cuDeviceGetCount(&deviceCount));

    for (int currentDevice = 0; currentDevice < deviceCount; currentDevice++) {
        for (int peer = 0; peer < deviceCount; peer++) {
            int canAccessPeer = 0;

            if (peer == currentDevice) {
                continue;
            }

            CU_ASSERT(cuDeviceCanAccessPeer(&canAccessPeer, currentDevice, peer));
            if (canAccessPeer) {
                return true;
            }
        }
    }

    return false;
}

void Testcase::allToOneHelper(unsigned long long size, MemcpyOperation &memcpyInstance, PeerValueMatrix<double> &bandwidthValues, bool isRead) {
    std::vector<const DeviceNode*> allSrcNodes;

    //allocate all src nodes up front, re-use to avoid reallocation
    for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
        allSrcNodes.push_back(new DeviceNode(size, deviceId));
    }

    for (int dstDeviceId = 0; dstDeviceId < deviceCount; dstDeviceId++) {
        std::vector<const MemcpyNode*> dstNodes;
        std::vector<const MemcpyNode*> srcNodes;

        for (int srcDeviceId = 0; srcDeviceId < deviceCount; srcDeviceId++) {
            if (srcDeviceId == dstDeviceId) {
                continue;
            }

            DeviceNode* dstNode = new DeviceNode(size, dstDeviceId);

            if (!dstNode->enablePeerAcess(*allSrcNodes[srcDeviceId])) {
                delete dstNode;
                continue;
            }

            srcNodes.push_back(allSrcNodes[srcDeviceId]);
            dstNodes.push_back(dstNode);
        }

        if (isRead) {
            // swap dst and src for read tests
            bandwidthValues.value(0, dstDeviceId) = memcpyInstance.doMemcpy(dstNodes, srcNodes);
        } else {
            bandwidthValues.value(0, dstDeviceId) = memcpyInstance.doMemcpy(srcNodes, dstNodes);
        }

        for (auto node : dstNodes) {
            delete node;
        }
    }

    for (auto node : allSrcNodes) {
        delete node;
    }
}

void Testcase::oneToAllHelper(unsigned long long size, MemcpyOperation &memcpyInstance, PeerValueMatrix<double> &bandwidthValues, bool isRead) {
    std::vector<const DeviceNode*> allDstNodes;

    //allocate all src nodes up front, re-use to avoid reallocation
    for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
        allDstNodes.push_back(new DeviceNode(size, deviceId));
    }

    for (int srcDeviceId = 0; srcDeviceId < deviceCount; srcDeviceId++) {
        std::vector<const MemcpyNode*> dstNodes;
        std::vector<const MemcpyNode*> srcNodes;

        for (int dstDeviceId = 0; dstDeviceId < deviceCount; dstDeviceId++) {
            if (srcDeviceId == dstDeviceId) {
                continue;
            }

            DeviceNode* srcNode = new DeviceNode(size, srcDeviceId);

            if (!srcNode->enablePeerAcess(*allDstNodes[dstDeviceId])) {
                delete srcNode;
                continue;
            }

            srcNodes.push_back(srcNode);
            dstNodes.push_back(allDstNodes[dstDeviceId]);
        }

        if (isRead) {
            // swap dst and src for read tests
            bandwidthValues.value(0, srcDeviceId) = memcpyInstance.doMemcpy(dstNodes, srcNodes);
        } else {
            bandwidthValues.value(0, srcDeviceId) = memcpyInstance.doMemcpy(srcNodes, dstNodes);
        }

        for (auto node : srcNodes) {
            delete node;
        }
    }

    for (auto node : allDstNodes) {
        delete node;
    }
}
