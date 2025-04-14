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

#include "common.h"
#include "output.h"
#include "testcase.h"
#include "inline_common.h"

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

bool Testcase::filterSupportsMulticast() {
    int deviceCount = 0;
    CU_ASSERT(cuDeviceGetCount(&deviceCount));

    for (int currentDevice = 0; currentDevice < deviceCount; currentDevice++) {
        CUdevice dev;
        CU_ASSERT(cuDeviceGet(&dev, currentDevice));
        int supportsMulticast = 0;

        CU_ASSERT(cuDeviceGetAttribute(&supportsMulticast, CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED, dev));
        if (!supportsMulticast) {
            return false;
        }
    }

    return true;
}

#ifdef MULTINODE
// Each MPI rank handles one GPU, so we simply have to check if we have more than 1 process
bool Testcase::filterHasMultipleGPUsMultinode() {
    return worldSize > 1;
}
#endif

void Testcase::latencyHelper(const MemcpyBuffer &dataBuffer, bool measureDeviceToDeviceLatency) {
    uint64_t n_ptrs = dataBuffer.getBufferSize() / sizeof(struct LatencyNode);

    if (measureDeviceToDeviceLatency) {
        // For device-to-device latency, create and initialize pattern on device
        for (uint64_t i = 0; i < n_ptrs; i++) {
            struct LatencyNode node;
            size_t nextOffset = ((i + strideLen) % n_ptrs) * sizeof(struct LatencyNode);
            // Set up pattern with device addresses
            node.next = (struct LatencyNode*)(dataBuffer.getBuffer() + nextOffset);
            CU_ASSERT(cuMemcpyHtoD(dataBuffer.getBuffer() + i*sizeof(struct LatencyNode),
                                 &node, sizeof(struct LatencyNode)));
        }
    } else {
        // For host-device latency, initialize pattern with host addresses
        struct LatencyNode* hostMem = (struct LatencyNode*)dataBuffer.getBuffer();
        for (uint64_t i = 0; i < n_ptrs; i++) {
            hostMem[i].next = &hostMem[(i + strideLen) % n_ptrs];
        }
    }
}

void Testcase::allToOneHelper(unsigned long long size, MemcpyOperation &memcpyInstance, PeerValueMatrix<double> &bandwidthValues, bool isRead) {
    std::vector<const DeviceBuffer*> allSrcBuffers;

    // allocate all src nodes up front, re-use to avoid reallocation
    for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
        allSrcBuffers.push_back(new DeviceBuffer(size, deviceId));
    }

    for (int dstDeviceId = 0; dstDeviceId < deviceCount; dstDeviceId++) {
        std::vector<const MemcpyBuffer*> dstBuffers;
        std::vector<const MemcpyBuffer*> srcBuffers;

        for (int srcDeviceId = 0; srcDeviceId < deviceCount; srcDeviceId++) {
            if (srcDeviceId == dstDeviceId) {
                continue;
            }

            DeviceBuffer* dstBuffer = new DeviceBuffer(size, dstDeviceId);

            if (!dstBuffer->enablePeerAcess(*allSrcBuffers[srcDeviceId])) {
                delete dstBuffer;
                continue;
            }

            srcBuffers.push_back(allSrcBuffers[srcDeviceId]);
            dstBuffers.push_back(dstBuffer);
        }
        // If no peer GPUs, skip measurements.
        if (!srcBuffers.empty()) {
            if (isRead) {
                // swap dst and src for read tests
                bandwidthValues.value(0, dstDeviceId) = memcpyInstance.doMemcpy(dstBuffers, srcBuffers);
            } else {
                bandwidthValues.value(0, dstDeviceId) = memcpyInstance.doMemcpy(srcBuffers, dstBuffers);
            }
        }

        for (auto node : dstBuffers) {
            delete node;
        }
    }

    for (auto node : allSrcBuffers) {
        delete node;
    }
}

void Testcase::oneToAllHelper(unsigned long long size, MemcpyOperation &memcpyInstance, PeerValueMatrix<double> &bandwidthValues, bool isRead) {
    std::vector<const DeviceBuffer*> allDstBuffers;

    // allocate all src nodes up front, re-use to avoid reallocation
    for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
        allDstBuffers.push_back(new DeviceBuffer(size, deviceId));
    }

    for (int srcDeviceId = 0; srcDeviceId < deviceCount; srcDeviceId++) {
        std::vector<const MemcpyBuffer*> dstBuffers;
        std::vector<const MemcpyBuffer*> srcBuffers;

        for (int dstDeviceId = 0; dstDeviceId < deviceCount; dstDeviceId++) {
            if (srcDeviceId == dstDeviceId) {
                continue;
            }

            DeviceBuffer* srcBuffer = new DeviceBuffer(size, srcDeviceId);

            if (!srcBuffer->enablePeerAcess(*allDstBuffers[dstDeviceId])) {
                delete srcBuffer;
                continue;
            }

            srcBuffers.push_back(srcBuffer);
            dstBuffers.push_back(allDstBuffers[dstDeviceId]);
        }
        // If no peer GPUs, skip measurements.
        if ( !srcBuffers.empty() ) {
            if (isRead) {
                // swap dst and src for read tests
                bandwidthValues.value(0, srcDeviceId) = memcpyInstance.doMemcpy(dstBuffers, srcBuffers);
            } else {
                bandwidthValues.value(0, srcDeviceId) = memcpyInstance.doMemcpy(srcBuffers, dstBuffers);
            }
        }

        for (auto node : srcBuffers) {
            delete node;
        }
    }

    for (auto node : allDstBuffers) {
        delete node;
    }
}

void Testcase::allHostHelper(unsigned long long size, MemcpyOperation &memcpyInstance, PeerValueMatrix<double> &bandwidthValues, bool sourceIsHost) {
    for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
        std::vector<const MemcpyBuffer*> deviceBuffers;
        std::vector<const MemcpyBuffer*> hostBuffers;

        deviceBuffers.push_back(new DeviceBuffer(size, deviceId));
        hostBuffers.push_back(new HostBuffer(size, deviceId));

        for (int interferenceDeviceId = 0; interferenceDeviceId < deviceCount; interferenceDeviceId++) {
            if (interferenceDeviceId == deviceId) {
                continue;
            }

            // Double the size of the interference copy to ensure it interferes correctly
            deviceBuffers.push_back(new DeviceBuffer(size * 2, interferenceDeviceId));
            hostBuffers.push_back(new HostBuffer(size * 2, interferenceDeviceId));
        }

        if (sourceIsHost) {
            bandwidthValues.value(0, deviceId) = memcpyInstance.doMemcpy(hostBuffers, deviceBuffers);
        } else {
            bandwidthValues.value(0, deviceId) = memcpyInstance.doMemcpy(deviceBuffers, hostBuffers);
        }

        for (auto node : deviceBuffers) {
            delete node;
        }

        for (auto node : hostBuffers) {
            delete node;
        }
    }
}

void Testcase::allHostBidirHelper(unsigned long long size, MemcpyOperation &memcpyInstance, PeerValueMatrix<double> &bandwidthValues, bool sourceIsHost) {
    for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
        std::vector<const MemcpyBuffer*> srcBuffers;
        std::vector<const MemcpyBuffer*> dstBuffers;

        if (sourceIsHost) {
            srcBuffers.push_back(new HostBuffer(size, deviceId));
            dstBuffers.push_back(new DeviceBuffer(size, deviceId));

            // Double the size of the interference copy to ensure it interferes correctly
            srcBuffers.push_back(new DeviceBuffer(size * 2, deviceId));
            dstBuffers.push_back(new HostBuffer(size * 2, deviceId));
        } else {
            srcBuffers.push_back(new DeviceBuffer(size, deviceId));
            dstBuffers.push_back(new HostBuffer(size, deviceId));

            // Double the size of the interference copy to ensure it interferes correctly
            srcBuffers.push_back(new HostBuffer(size * 2, deviceId));
            dstBuffers.push_back(new DeviceBuffer(size * 2, deviceId));
        }

        for (int interferenceDeviceId = 0; interferenceDeviceId < deviceCount; interferenceDeviceId++) {
            if (interferenceDeviceId == deviceId) {
                continue;
            }

            // Double the size of the interference copy to ensure it interferes correctly
            srcBuffers.push_back(new DeviceBuffer(size * 2, interferenceDeviceId));
            dstBuffers.push_back(new HostBuffer(size * 2, interferenceDeviceId));

            srcBuffers.push_back(new HostBuffer(size * 2, interferenceDeviceId));
            dstBuffers.push_back(new DeviceBuffer(size * 2, interferenceDeviceId));
        }

        bandwidthValues.value(0, deviceId) = memcpyInstance.doMemcpy(srcBuffers, dstBuffers);

        for (auto node : srcBuffers) {
            delete node;
        }

        for (auto node : dstBuffers) {
            delete node;
        }
    }
}

