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


std::map<std::string, std::map<std::string, std::shared_ptr<PeerValueMatrix<double>>>> Testcase::globalTestCache;

Testcase::Testcase(std::string key, std::string desc) :
    key(std::move(key)), desc(std::move(desc))
{}

std::string Testcase::testKey() { return key; }
std::string Testcase::testDesc() { return desc; }

FilterResult Testcase::filterHasMultipleGPUs() {
    int n = 0;
    CU_ASSERT(cuDeviceGetCount(&n));
    if (n < 2) return FilterResult::waive("requires at least 2 GPUs");
    return FilterResult::pass();
}

FilterResult Testcase::filterHasAccessiblePeerPairs() {
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
                return FilterResult::pass();
            }
        }
    }

    return FilterResult::waive("no accessible peer GPU pairs (P2P not supported between any devices)");
}

FilterResult Testcase::filterSupportsMulticast() {
    int deviceCount = 0;
    CU_ASSERT(cuDeviceGetCount(&deviceCount));

    for (int currentDevice = 0; currentDevice < deviceCount; currentDevice++) {
        CUdevice dev;
        CU_ASSERT(cuDeviceGet(&dev, currentDevice));
        int supportsMulticast = 0;

        CU_ASSERT(cuDeviceGetAttribute(&supportsMulticast, CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED, dev));
        if (!supportsMulticast) {
            return FilterResult::waive("multicast not supported on device " + std::to_string(currentDevice));
        }
    }

    return FilterResult::pass();
}

FilterResult Testcase::filterSupportsTMA() {
    int deviceCount = 0;
    CU_ASSERT(cuDeviceGetCount(&deviceCount));

    for (int currentDevice = 0; currentDevice < deviceCount; currentDevice++) {
        CUdevice dev;
        CU_ASSERT(cuDeviceGet(&dev, currentDevice));
        int major = 0;

        CU_ASSERT(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev));
        if (major < 9) {
            return FilterResult::waive(
                "TMA requires compute capability 9.0+ (Hopper or later), device "
                + std::to_string(currentDevice) + " has " + std::to_string(major) + ".x");
        }
    }

    return FilterResult::pass();
}


FilterResult Testcase::filterBounceBufferConfComputeEnabled() {
    if (gSettings.bounceBufferConfComputeEnabled) {
        return FilterResult::waive("bounce-buffer confidential computing enabled");
    }

    return FilterResult::pass();
}

#ifdef MULTINODE
// Each MPI rank handles one GPU, so we simply have to check if we have more than 1 process
FilterResult Testcase::filterHasMultipleGPUsMultinode() {
    if (worldSize > 1) return FilterResult::pass();
    return FilterResult::waive("requires multiple MPI ranks (only 1 process)");
}
#endif

template<typename AlignedLatencyNode>
void Testcase::latencyHelper(const MemcpyBuffer &dataBuffer, bool measureDeviceToDeviceLatency) {
    uint64_t n_ptrs = dataBuffer.getBufferSize() / sizeof(AlignedLatencyNode);

    if (measureDeviceToDeviceLatency) {
        // For device-to-device latency, create and initialize pattern on device
        for (uint64_t i = 0; i < n_ptrs; i++) {
            AlignedLatencyNode alignedNode;
            size_t nextOffset = ((i + strideLen) % n_ptrs) * sizeof(AlignedLatencyNode);
            // Set up pattern with device addresses
            alignedNode.next = (AlignedLatencyNode *)(dataBuffer.getBuffer() + nextOffset);
            CU_ASSERT(cuMemcpyHtoD(dataBuffer.getBuffer() + i*sizeof(AlignedLatencyNode),
                                 &alignedNode, sizeof(AlignedLatencyNode)));
        }
    } else {
        // For host-device latency, initialize pattern with host addresses
        AlignedLatencyNode* hostMem = (AlignedLatencyNode*)dataBuffer.getBuffer();
        for (uint64_t i = 0; i < n_ptrs; i++) {
            hostMem[i].next = &hostMem[(i + strideLen) % n_ptrs];
        }
    }
}

template void Testcase::latencyHelper<struct LatencyNode>(const MemcpyBuffer &dataBuffer, bool measureDeviceToDeviceLatency);
template void Testcase::latencyHelper<struct TMALatencyNode>(const MemcpyBuffer &dataBuffer, bool measureDeviceToDeviceLatency);

void Testcase::forEachAllToOneBufferPairing(unsigned long long size, allToOneOneToAllCallback callback) {
    std::vector<const DeviceBuffer*> allBuffersPreallocated;

    // allocate the all-group buffers up front, re-use to avoid reallocation
    for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
        allBuffersPreallocated.push_back(new DeviceBuffer(size, deviceId));
    }

    for (int oneDeviceId = 0; oneDeviceId < deviceCount; oneDeviceId++) {
        std::vector<const MemcpyBuffer*> oneBuffers;
        std::vector<const MemcpyBuffer*> allBuffers;

        for (int allDeviceId = 0; allDeviceId < deviceCount; allDeviceId++) {
            if (allDeviceId == oneDeviceId) {
                continue;
            }

            DeviceBuffer* oneBuffer = new DeviceBuffer(size, oneDeviceId);

            if (!oneBuffer->enablePeerAcess(*allBuffersPreallocated[allDeviceId])) {
                delete oneBuffer;
                continue;
            }

            // Pairs a buffer on a GPU in the all group, with a buffer on a GPU in the one group
            allBuffers.push_back(allBuffersPreallocated[allDeviceId]);
            oneBuffers.push_back(oneBuffer);
        }
        // If no peer GPUs, skip measurements.
        if (!allBuffers.empty()) {
            callback(oneDeviceId, allBuffers, oneBuffers);
        }

        for (auto node : oneBuffers) {
            delete node;
        }
    }

    for (auto node : allBuffersPreallocated) {
        delete node;
    }
}

void Testcase::allToOneHelper(unsigned long long size, MemcpyOperation &memcpyInstance, PeerValueMatrix<double> &bandwidthValues) {
    forEachAllToOneBufferPairing(size, [&](int oneDeviceId,
                                            std::vector<const MemcpyBuffer*>& allBuffers,
                                            std::vector<const MemcpyBuffer*>& oneBuffers) {
        if (memcpyInstance.ctxPreference == PREFER_DST_CONTEXT) {
            // Do a read from each GPU in the all group, by reading from the one GPU
            bandwidthValues.value(0, oneDeviceId) = memcpyInstance.doMemcpy(oneBuffers, allBuffers);
        } else {
            // Do a write from each GPU in the all group, by writing to the one GPU
            bandwidthValues.value(0, oneDeviceId) = memcpyInstance.doMemcpy(allBuffers, oneBuffers);
        }
    });
    recordStability(bandwidthValues, memcpyInstance);
}

void Testcase::oneToAllHelper(unsigned long long size, MemcpyOperation &memcpyInstance, PeerValueMatrix<double> &bandwidthValues) {
    forEachAllToOneBufferPairing(size, [&](int oneDeviceId,
                                            std::vector<const MemcpyBuffer*>& allBuffers,
                                            std::vector<const MemcpyBuffer*>& oneBuffers) {
        if (memcpyInstance.ctxPreference == PREFER_DST_CONTEXT) {
            // Do a read from the GPU in the one group, by reading from each GPU in the all group
            bandwidthValues.value(0, oneDeviceId) = memcpyInstance.doMemcpy(allBuffers, oneBuffers);
        } else {
            // Do a write from the GPU in the one group, by writing to each GPU in the all group
            bandwidthValues.value(0, oneDeviceId) = memcpyInstance.doMemcpy(oneBuffers, allBuffers);
        }
    });
    recordStability(bandwidthValues, memcpyInstance);
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
    recordStability(bandwidthValues, memcpyInstance);
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
    recordStability(bandwidthValues, memcpyInstance);
}
