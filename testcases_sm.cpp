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

void HostDeviceLatencySM::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> latencyValues(1, deviceCount, key, perfFormatter, LATENCY);
    MemPtrChaseOperation ptrChaseOp(latencyMemAccessCnt);

    for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
        HostBuffer dataBuffer(size, deviceId);
        latencyHelper(dataBuffer, false);
        latencyValues.value(0, deviceId) = ptrChaseOp.doPtrChase(deviceId, dataBuffer);
    }

    output->addTestcaseResults(latencyValues, "memory latency SM CPU(row) <-> GPU(column) (ns)");
}

void HostToDeviceSM::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bandwidthValues(1, deviceCount, key);
    MemcpyOperation memcpyInstance(loopCount, new MemcpyInitiatorSM());

    for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
        HostBuffer hostBuffer(size, deviceId);
        DeviceBuffer deviceBuffer(size, deviceId);

        bandwidthValues.value(0, deviceId) = memcpyInstance.doMemcpy(hostBuffer, deviceBuffer);
    }

    output->addTestcaseResults(bandwidthValues, "memcpy SM CPU(row) -> GPU(column) bandwidth (GB/s)");
}

void DeviceToHostSM::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bandwidthValues(1, deviceCount, key);
    MemcpyOperation memcpyInstance(loopCount, new MemcpyInitiatorSM());

    for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
        HostBuffer hostBuffer(size, deviceId);
        DeviceBuffer deviceBuffer(size, deviceId);

        bandwidthValues.value(0, deviceId) = memcpyInstance.doMemcpy(deviceBuffer, hostBuffer);
    }

    output->addTestcaseResults(bandwidthValues, "memcpy SM CPU(row) <- GPU(column) bandwidth (GB/s)");
}

void HostToDeviceBidirSM::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bandwidthValues(1, deviceCount, key);
    MemcpyOperation memcpyInstance(loopCount, new MemcpyInitiatorSMSplitWarp());

    for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
        HostBuffer hostBuffer(size, deviceId);
        DeviceBuffer deviceBuffer(size, deviceId);

        bandwidthValues.value(0, deviceId) = memcpyInstance.doMemcpy(hostBuffer, deviceBuffer);
    }

    output->addTestcaseResults(bandwidthValues, "memcpy SM CPU(row) <-> GPU(column) bandwidth (GB/s)");
}

void DeviceToHostBidirSM::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bandwidthValues(1, deviceCount, key);
    MemcpyOperation memcpyInstance(loopCount, new MemcpyInitiatorSMSplitWarp());

    for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
        HostBuffer hostBuffer(size, deviceId);
        DeviceBuffer deviceBuffer(size, deviceId);

        bandwidthValues.value(0, deviceId) = memcpyInstance.doMemcpy(hostBuffer, deviceBuffer);
    }

    output->addTestcaseResults(bandwidthValues, "memcpy SM CPU(row) <-> GPU(column) bandwidth (GB/s)");
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

            DeviceBuffer srcBuffer(size, srcDeviceId);
            DeviceBuffer peerBuffer(size, peerDeviceId);

            if (!srcBuffer.enablePeerAcess(peerBuffer)) {
                continue;
            }

            // swap src and peer nodes, but use srcBuffers (the copy's destination) context
            bandwidthValues.value(srcDeviceId, peerDeviceId) = memcpyInstance.doMemcpy(peerBuffer, srcBuffer);
        }
    }

    output->addTestcaseResults(bandwidthValues, "memcpy SM GPU(row) -> GPU(column) bandwidth (GB/s)");
}

void DeviceToDeviceLatencySM::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> latencyValues(deviceCount, deviceCount, key, perfFormatter, LATENCY);
    MemPtrChaseOperation ptrChaseOp(latencyMemAccessCnt);

    for (int peerDeviceId = 0; peerDeviceId < deviceCount; peerDeviceId++) {
        DeviceBuffer peerBuffer(size, peerDeviceId);
        latencyHelper(peerBuffer, true);

        for (int srcDeviceId = 0; srcDeviceId < deviceCount; srcDeviceId++) {
            if (peerDeviceId == srcDeviceId) {
                continue;
            }

            // Note: srcBuffer is not used in the pointer chase operation
            // It is simply used here to enable peer access
            DeviceBuffer srcBuffer(size, srcDeviceId);
            if (!srcBuffer.enablePeerAcess(peerBuffer)) {
                continue;
            }
            latencyValues.value(srcDeviceId, peerDeviceId) = ptrChaseOp.doPtrChase(srcDeviceId, peerBuffer);
        }
    }
    output->addTestcaseResults(latencyValues, "Device to Device Latency SM GPU(row) <-> GPU(column) (ns)");
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

            DeviceBuffer srcBuffer(size, srcDeviceId);
            DeviceBuffer peerBuffer(size, peerDeviceId);

            if (!srcBuffer.enablePeerAcess(peerBuffer)) {
                continue;
            }

            bandwidthValues.value(srcDeviceId, peerDeviceId) = memcpyInstance.doMemcpy(srcBuffer, peerBuffer);
        }
    }

    output->addTestcaseResults(bandwidthValues, "memcpy SM GPU(row) <- GPU(column) bandwidth (GB/s)");
}

// DtoD Bidir Read test - copy to dst from src (backwards) using dst contxt
void DeviceToDeviceBidirReadSM::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bandwidthValuesRead1(deviceCount, deviceCount, key + "_read1");
    PeerValueMatrix<double> bandwidthValuesRead2(deviceCount, deviceCount, key + "_read2");
    PeerValueMatrix<double> bandwidthValuesTotal(deviceCount, deviceCount, key + "_total");
    MemcpyOperation memcpyInstance(loopCount, new MemcpyInitiatorSM(), PREFER_DST_CONTEXT, MemcpyOperation::VECTOR_BW);


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

            // swap src and peer nodes, but use srcBuffers (the copy's destination) context
            std::vector<const MemcpyBuffer*> srcBuffers = {&peer1, &src2};
            std::vector<const MemcpyBuffer*> peerBuffers = {&src1, &peer2};

            auto results = memcpyInstance.doMemcpyVector(srcBuffers, peerBuffers);
            bandwidthValuesRead1.value(srcDeviceId, peerDeviceId) = results[0];
            bandwidthValuesRead2.value(srcDeviceId, peerDeviceId) = results[1];
            bandwidthValuesTotal.value(srcDeviceId, peerDeviceId) = results[0] + results[1];
        }
    }

    output->addTestcaseResults(bandwidthValuesRead1, "memcpy SM GPU(row) <-> GPU(column) Read1 bandwidth (GB/s)");
    output->addTestcaseResults(bandwidthValuesRead2, "memcpy SM GPU(row) <-> GPU(column) Read2 bandwidth (GB/s)");
    output->addTestcaseResults(bandwidthValuesTotal, "memcpy SM GPU(row) <-> GPU(column) Total bandwidth (GB/s)");
}

// DtoD Bidir Write test - copy from  src to dst using src contxt
void DeviceToDeviceBidirWriteSM::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bandwidthValuesWrite1(deviceCount, deviceCount, key + "_write1");
    PeerValueMatrix<double> bandwidthValuesWrite2(deviceCount, deviceCount, key + "_write2");
    PeerValueMatrix<double> bandwidthValuesTotal(deviceCount, deviceCount, key + "_total");
    MemcpyOperation memcpyInstance(loopCount, new MemcpyInitiatorSM(), PREFER_SRC_CONTEXT, MemcpyOperation::VECTOR_BW);


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

            // swap src and peer nodes, but use srcBuffers (the copy's destination) context
            std::vector<const MemcpyBuffer*> srcBuffers = {&peer1, &src2};
            std::vector<const MemcpyBuffer*> peerBuffers = {&src1, &peer2};

            auto results = memcpyInstance.doMemcpyVector(srcBuffers, peerBuffers);
            bandwidthValuesWrite1.value(srcDeviceId, peerDeviceId) = results[0];
            bandwidthValuesWrite2.value(srcDeviceId, peerDeviceId) = results[1];
            bandwidthValuesTotal.value(srcDeviceId, peerDeviceId) = results[0] + results[1];
        }
    }

    output->addTestcaseResults(bandwidthValuesWrite1, "memcpy SM GPU(row) <-> GPU(column) Write1 bandwidth (GB/s)");
    output->addTestcaseResults(bandwidthValuesWrite2, "memcpy SM GPU(row) <-> GPU(column) Write2 bandwidth (GB/s)");
    output->addTestcaseResults(bandwidthValuesTotal, "memcpy SM GPU(row) <-> GPU(column) Total bandwidth (GB/s)");
}

void AllToHostSM::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bandwidthValues(1, deviceCount, key);
    MemcpyOperation memcpyInstance(loopCount, new MemcpyInitiatorSM(), PREFER_SRC_CONTEXT, MemcpyOperation::USE_FIRST_BW);

    allHostHelper(size, memcpyInstance, bandwidthValues, false);

    output->addTestcaseResults(bandwidthValues, "memcpy SM CPU(row) <- GPU(column) bandwidth (GB/s)");
}

void AllToHostBidirSM::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bandwidthValues(1, deviceCount, key);
    MemcpyOperation memcpyInstance(loopCount, new MemcpyInitiatorSMSplitWarp(), PREFER_SRC_CONTEXT, MemcpyOperation::USE_FIRST_BW);

    for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
        std::vector<const MemcpyBuffer*> srcBuffers;
        std::vector<const MemcpyBuffer*> dstBuffers;

        srcBuffers.push_back(new DeviceBuffer(size, deviceId));
        dstBuffers.push_back(new HostBuffer(size, deviceId));

        for (int interferenceDeviceId = 0; interferenceDeviceId < deviceCount; interferenceDeviceId++) {
            if (interferenceDeviceId == deviceId) {
                continue;
            }

            srcBuffers.push_back(new DeviceBuffer(size, interferenceDeviceId));
            dstBuffers.push_back(new HostBuffer(size, interferenceDeviceId));
        }

        bandwidthValues.value(0, deviceId) = memcpyInstance.doMemcpy(srcBuffers, dstBuffers);

        for (auto node : srcBuffers) {
            delete node;
        }

        for (auto node : dstBuffers) {
            delete node;
        }
    }

    output->addTestcaseResults(bandwidthValues, "memcpy SM CPU(row) <-> GPU(column) bandwidth (GB/s)");
}

void HostToAllSM::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bandwidthValues(1, deviceCount, key);
    MemcpyOperation memcpyInstance(loopCount, new MemcpyInitiatorSM(), PREFER_SRC_CONTEXT, MemcpyOperation::USE_FIRST_BW);

    allHostHelper(size, memcpyInstance, bandwidthValues, true);

    output->addTestcaseResults(bandwidthValues, "memcpy SM CPU(row) -> GPU(column) bandwidth (GB/s)");
}

void HostToAllBidirSM::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bandwidthValues(1, deviceCount, key);
    MemcpyOperation memcpyInstance(loopCount, new MemcpyInitiatorSMSplitWarp(), PREFER_SRC_CONTEXT, MemcpyOperation::USE_FIRST_BW);

    for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
        std::vector<const MemcpyBuffer*> srcBuffers;
        std::vector<const MemcpyBuffer*> dstBuffers;

        srcBuffers.push_back(new HostBuffer(size, deviceId));
        dstBuffers.push_back(new DeviceBuffer(size, deviceId));

        for (int interferenceDeviceId = 0; interferenceDeviceId < deviceCount; interferenceDeviceId++) {
            if (interferenceDeviceId == deviceId) {
                continue;
            }

            srcBuffers.push_back(new DeviceBuffer(size, interferenceDeviceId));
            dstBuffers.push_back(new HostBuffer(size, interferenceDeviceId));
        }

        bandwidthValues.value(0, deviceId) = memcpyInstance.doMemcpy(srcBuffers, dstBuffers);

        for (auto node : srcBuffers) {
            delete node;
        }

        for (auto node : dstBuffers) {
            delete node;
        }
    }

    output->addTestcaseResults(bandwidthValues, "memcpy SM CPU(row) <-> GPU(column) bandwidth (GB/s)");
}

// Write test - copy from src to dst using src context
void AllToOneWriteSM::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bandwidthValues(1, deviceCount, key);
    MemcpyOperation memcpyInstance(loopCount, new MemcpyInitiatorSM(), PREFER_SRC_CONTEXT, MemcpyOperation::TOTAL_BW);

    allToOneHelper(size, memcpyInstance, bandwidthValues, false);

    output->addTestcaseResults(bandwidthValues, "memcpy SM GPU(row) <- GPU(column) bandwidth (GB/s)");
}

// Read test - copy from dst to src (backwards) using src contxt
void AllToOneReadSM::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bandwidthValues(1, deviceCount, key);
    MemcpyOperation memcpyInstance(loopCount, new MemcpyInitiatorSM(), PREFER_SRC_CONTEXT, MemcpyOperation::TOTAL_BW);
    allToOneHelper(size, memcpyInstance, bandwidthValues, true);

    output->addTestcaseResults(bandwidthValues, "memcpy SM GPU(row) -> GPU(column) bandwidth (GB/s)");
}

// Write test - copy from src to dst using src context
void OneToAllWriteSM::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bandwidthValues(1, deviceCount, key);
    MemcpyOperation memcpyInstance(loopCount, new MemcpyInitiatorSM(), PREFER_SRC_CONTEXT, MemcpyOperation::TOTAL_BW);
    oneToAllHelper(size, memcpyInstance, bandwidthValues, false);

    output->addTestcaseResults(bandwidthValues, "memcpy SM GPU(row) -> GPU(column) bandwidth (GB/s)");
}

// Read test - copy from dst to src (backwards) using src contxt
void OneToAllReadSM::run(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bandwidthValues(1, deviceCount, key);
    MemcpyOperation memcpyInstance(loopCount, new MemcpyInitiatorSM(), PREFER_SRC_CONTEXT, MemcpyOperation::TOTAL_BW);
    oneToAllHelper(size, memcpyInstance, bandwidthValues, true);

    output->addTestcaseResults(bandwidthValues, "memcpy SM GPU(row) <- GPU(column) bandwidth (GB/s)");
}
