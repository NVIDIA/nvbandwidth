/*
 * Copyright 1993-2021 by NVIDIA Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
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
    MemcpyOperationSM memcpyInstance(size, loopCount);

    for (int srcDeviceId = 0; srcDeviceId < deviceCount; srcDeviceId++) {
        for (int peerDeviceId = 0; peerDeviceId < deviceCount; peerDeviceId++) {
            if (peerDeviceId == srcDeviceId) {
                continue;
            }

            DeviceNode srcNode(size, srcDeviceId);
            DeviceNode peerNode(size, peerDeviceId);

            // swap src and peer nodes, but use srcNodes (the copy's destination) context
            bandwidthValues.value(srcDeviceId, peerDeviceId) = memcpyInstance.doMemcpy(&peerNode, &srcNode, false);
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

            bandwidthValues.value(srcDeviceId, peerDeviceId) = memcpyInstance.doMemcpy(&srcNode, &peerNode);
        }
    }

    std::cout << "memcpy CE GPU(row) <- GPU(column) bandwidth (GB/s)" << std::endl;
    std::cout << std::fixed << std::setprecision(2) << bandwidthValues << std::endl;
}

// DtoD Bidir Read test - copy from dst to src (backwards) using src contxt
void launch_DtoD_memcpy_bidirectional_read_SM(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bandwidthValues(deviceCount, deviceCount);
    MemcpyOperationSM memcpyInstance(size, loopCount);

    for (int srcDeviceId = 0; srcDeviceId < deviceCount; srcDeviceId++) {
        for (int peerDeviceId = 0; peerDeviceId < deviceCount; peerDeviceId++) {
            if (peerDeviceId == srcDeviceId) {
                continue;
            }

            // swap src and peer nodes, but use srcNodes (the copy's destination) context
            std::vector<MemcpyNode *> srcNodes = {new DeviceNode(size, peerDeviceId), new DeviceNode(size, srcDeviceId)};
            std::vector<MemcpyNode *> peerNodes = {new DeviceNode(size, srcDeviceId), new DeviceNode(size, peerDeviceId)};

            bandwidthValues.value(srcDeviceId, peerDeviceId) = memcpyInstance.doMemcpy(srcNodes, peerNodes, false);

            for (auto node : srcNodes) delete node;
            for (auto node : peerNodes) delete node;
        }
    }

    std::cout << "memcpy SM GPU(row) -> GPU(column) bandwidth (GB/s)" << std::endl;
    std::cout << std::fixed << std::setprecision(2) << bandwidthValues << std::endl;
}


// DtoD Bidir Write test - copy from src to dst using src context
void launch_DtoD_memcpy_bidirectional_write_SM(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bandwidthValues(deviceCount, deviceCount);
    MemcpyOperationSM memcpyInstance(size, loopCount);

    for (int srcDeviceId = 0; srcDeviceId < deviceCount; srcDeviceId++) {
        for (int peerDeviceId = 0; peerDeviceId < deviceCount; peerDeviceId++) {
            if (peerDeviceId == srcDeviceId) {
                continue;
            }

            // swap src and peer nodes, but use srcNodes (the copy's destination) context
            std::vector<MemcpyNode *> srcNodes = {new DeviceNode(size, srcDeviceId), new DeviceNode(size, peerDeviceId)};
            std::vector<MemcpyNode *> peerNodes = {new DeviceNode(size, peerDeviceId), new DeviceNode(size, srcDeviceId)};

            bandwidthValues.value(srcDeviceId, peerDeviceId) = memcpyInstance.doMemcpy(srcNodes, peerNodes);

            for (auto node : srcNodes) delete node;
            for (auto node : peerNodes) delete node;
        }
    }

    std::cout << "memcpy SM GPU(row) <- GPU(column) bandwidth (GB/s)" << std::endl;
    std::cout << std::fixed << std::setprecision(2) << bandwidthValues << std::endl;
}

void launch_DtoD_paired_memcpy_read_SM(unsigned long long size, unsigned long long loopCount) {
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
}

void launch_DtoD_paired_memcpy_write_SM(unsigned long long size, unsigned long long loopCount) {
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
}
