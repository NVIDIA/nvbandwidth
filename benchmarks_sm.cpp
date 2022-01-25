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

#include <omp.h>

#include "benchmark.h"
#include "kernels.cuh"

void launch_HtoD_memcpy_SM(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bandwidthValues(1, deviceCount);
    MemcpyOperationSM memcpyInstance(size, loopCount);

    for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
        HostNode hostNode(size, deviceId);
        DeviceNode deviceNode(size, deviceId);

        bandwidthValues.value(0, deviceId) = memcpyInstance.doMemcpy(&hostNode, &deviceNode);
    }

    std::cout << "memcpy SM CPU -> GPU bandwidth (GB/s)" << std::endl;
    std::cout << std::fixed << std::setprecision(2) << bandwidthValues << std::endl;
}

void launch_DtoH_memcpy_SM(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bandwidthValues(1, deviceCount);
    MemcpyOperationSM memcpyInstance(size, loopCount);

    for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
        HostNode hostNode(size, deviceId);
        DeviceNode deviceNode(size, deviceId);

        bandwidthValues.value(0, deviceId) = memcpyInstance.doMemcpy(&deviceNode, &hostNode);
    }

    std::cout << "memcpy SM GPU -> CPU bandwidth (GB/s)" << std::endl;
    std::cout << std::fixed << std::setprecision(2) << bandwidthValues << std::endl;
}

// DtoD Read test - copy from dst to src (backwards) using src contxt
void launch_DtoD_memcpy_read_SM(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bandwidthValues(deviceCount, deviceCount);
    MemcpyOperationSM memcpyInstance(size, loopCount, false);

    for (int srcDeviceId = 0; srcDeviceId < deviceCount; srcDeviceId++) {
        for (int peerDeviceId = 0; peerDeviceId < deviceCount; peerDeviceId++) {
            if (peerDeviceId == srcDeviceId) {
                continue;
            }

            DeviceNode srcNode(size, srcDeviceId);
            DeviceNode peerNode(size, peerDeviceId);

            if (!srcNode.enablePeerAcess(&peerNode)) {
                continue;
            }

            // swap src and peer nodes, but use srcNodes (the copy's destination) context
            bandwidthValues.value(srcDeviceId, peerDeviceId) = memcpyInstance.doMemcpy(&peerNode, &srcNode);
        }
    }

    std::cout << "memcpy CE GPU(row) -> GPU(column) bandwidth (GB/s)" << std::endl;
    std::cout << std::fixed << std::setprecision(2) << bandwidthValues << std::endl;
}

// DtoD Write test - copy from src to dst using src context
void launch_DtoD_memcpy_write_SM(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bandwidthValues(deviceCount, deviceCount);
    MemcpyOperationSM memcpyInstance(size, loopCount);

    for (int srcDeviceId = 0; srcDeviceId < deviceCount; srcDeviceId++) {
        for (int peerDeviceId = 0; peerDeviceId < deviceCount; peerDeviceId++) {
            if (peerDeviceId == srcDeviceId) {
                continue;
            }

            DeviceNode srcNode(size, srcDeviceId);
            DeviceNode peerNode(size, peerDeviceId);

            if (!srcNode.enablePeerAcess(&peerNode)) {
                continue;
            }

            bandwidthValues.value(srcDeviceId, peerDeviceId) = memcpyInstance.doMemcpy(&srcNode, &peerNode);
        }
    }

    std::cout << "memcpy CE GPU(row) <- GPU(column) bandwidth (GB/s)" << std::endl;
    std::cout << std::fixed << std::setprecision(2) << bandwidthValues << std::endl;
}

// DtoD Bidir Read test - copy from dst to src (backwards) using src contxt
void launch_DtoD_memcpy_bidirectional_read_SM(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bandwidthValues(deviceCount, deviceCount);
    MemcpyOperationSM memcpyInstance(size, loopCount, false, true);

    for (int srcDeviceId = 0; srcDeviceId < deviceCount; srcDeviceId++) {
        for (int peerDeviceId = 0; peerDeviceId < deviceCount; peerDeviceId++) {
            if (peerDeviceId == srcDeviceId) {
                continue;
            }

            DeviceNode src1(size, srcDeviceId), src2(size, srcDeviceId);
            DeviceNode peer1(size, peerDeviceId), peer2(size, peerDeviceId);

            if (!src1.enablePeerAcess(&peer1)) {
                continue;
            }

            // swap src and peer nodes, but use srcNodes (the copy's destination) context
            std::vector<MemcpyNode*> srcNodes = {&peer1, &src1};
            std::vector<MemcpyNode*> peerNodes = {&src2, &peer2};

            bandwidthValues.value(srcDeviceId, peerDeviceId) = memcpyInstance.doMemcpy(srcNodes, peerNodes);
        }
    }

    std::cout << "memcpy SM GPU(row) -> GPU(column) bandwidth (GB/s)" << std::endl;
    std::cout << std::fixed << std::setprecision(2) << bandwidthValues << std::endl;
}


// DtoD Bidir Write test - copy from src to dst using src context
void launch_DtoD_memcpy_bidirectional_write_SM(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bandwidthValues(deviceCount, deviceCount);
    MemcpyOperationSM memcpyInstance(size, loopCount, true, true);

    for (int srcDeviceId = 0; srcDeviceId < deviceCount; srcDeviceId++) {
        for (int peerDeviceId = 0; peerDeviceId < deviceCount; peerDeviceId++) {
            if (peerDeviceId == srcDeviceId) {
                continue;
            }

            DeviceNode src1(size, srcDeviceId), src2(size, srcDeviceId);
            DeviceNode peer1(size, peerDeviceId), peer2(size, peerDeviceId);

            if (!src1.enablePeerAcess(&peer1)) {
                continue;
            }

            std::vector<MemcpyNode*> srcNodes = {&src1, &peer1};
            std::vector<MemcpyNode*> peerNodes = {&peer2, &src2};

            bandwidthValues.value(srcDeviceId, peerDeviceId) = memcpyInstance.doMemcpy(srcNodes, peerNodes);
        }
    }

    std::cout << "memcpy SM GPU(row) <- GPU(column) bandwidth (GB/s)" << std::endl;
    std::cout << std::fixed << std::setprecision(2) << bandwidthValues << std::endl;
}

//void launch_DtoD_paired_memcpy_read_SM(unsigned long long size, unsigned long long loopCount) {
    // Memcpy memcpyInstance = Memcpy(copyKernel, size, loopCount);

    // std::vector<DeviceNode *> devices;
    // for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
    //     devices.push_back(new DeviceNode(size, deviceId));
    // }

    // if (parallel) {
    //     #pragma omp parallel num_threads(deviceCount / 2)
    //     {
    //         int deviceId = omp_get_thread_num();
    //         memcpyInstance.doMemcpy(devices[deviceId], devices[deviceId + (deviceCount / 2)]);
    //     }
    // } else {
    //     parallel = 1;
    //     for (int deviceId = 0; deviceId < deviceCount / 2; deviceId++) {
    //         memcpyInstance.doMemcpy(devices[deviceId], devices[deviceId + (deviceCount / 2)]);
    //     }
    //     parallel = 0;
    // }

    // for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
    //     delete devices[deviceId];
    // }

    // memcpyInstance.printBenchmarkMatrix();
//}

//void launch_DtoD_paired_memcpy_write_SM(unsigned long long size, unsigned long long loopCount) {
    // Memcpy memcpyInstance = Memcpy(copyKernel, size, loopCount);

    // std::vector<DeviceNode *> devices;
    // for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
    //     devices.push_back(new DeviceNode(size, deviceId));
    // }

    // if (parallel) {
    //     #pragma omp parallel num_threads(deviceCount / 2)
    //     {
    //         int deviceId = omp_get_thread_num();
    //         memcpyInstance.doMemcpy(devices[deviceId + (deviceCount / 2)], devices[deviceId]);
    //     }
    // } else {
    //     parallel = 1;
    //     for (int deviceId = 0; deviceId < deviceCount / 2; deviceId++) {
    //         memcpyInstance.doMemcpy(devices[deviceId + (deviceCount / 2)], devices[deviceId]);
    //     }
    //     parallel = 0;
    // }

    // for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
    //     delete devices[deviceId];
    // }

    // memcpyInstance.printBenchmarkMatrix();
//}
