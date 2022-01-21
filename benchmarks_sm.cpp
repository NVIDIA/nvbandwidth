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

void launch_DtoD_memcpy_read_SM(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bandwidthValues(deviceCount, deviceCount);
    MemcpyOperationSM memcpyInstance(size, loopCount);

    for (int srcDeviceId = 0; srcDeviceId < deviceCount; srcDeviceId++) {
        for (int dstDeviceId = 0; dstDeviceId < deviceCount; dstDeviceId++) {
            if (dstDeviceId == srcDeviceId) {
                continue;
            }

            DeviceNode srcNode(size, srcDeviceId);
            DeviceNode dstNode(size, dstDeviceId);

            bandwidthValues.value(srcDeviceId, dstDeviceId) = memcpyInstance.doMemcpy(&srcNode, &dstNode);
        }
    }

    //TODO fix arrow for read/write
    std::cout << "memcpy SM GPU(row) -> GPU(column) bandwidth (GB/s)" << std::endl;
    std::cout << std::fixed << std::setprecision(2) << bandwidthValues << std::endl;
}

void launch_DtoD_memcpy_write_SM(unsigned long long size, unsigned long long loopCount) {
    // this is no different than the read test, other than src and dst and swapped, but the matrix printed is identical
    // I assume the intent was to swap the context, such that the copy context was the dst instead of the source, but that is not the case here
    // Memcpy memcpyInstance = Memcpy(copyKernel, size, loopCount);

    // for (int targetDeviceId = 0; targetDeviceId < deviceCount; targetDeviceId++) {
    //     std::vector<DeviceNode *> devices, dstDevices;
    //     for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
    //         devices.push_back(new DeviceNode(size, targetDeviceId));
    //         dstDevices.push_back(new DeviceNode(size, deviceId));
    //     }

    //     if (parallel) {
    //         #pragma omp parallel num_threads(deviceCount)
    //         {
    //             int deviceId = omp_get_thread_num();
    //             memcpyInstance.doMemcpy(devices[deviceId], dstDevices[deviceId]);
    //         }
    //     } else {
    //         for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
    //             memcpyInstance.doMemcpy(devices[deviceId], dstDevices[deviceId]);
    //         }
    //     }

    //     for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
    //         delete devices[deviceId];
    //         delete dstDevices[deviceId];
    //     }
    // }

    // memcpyInstance.printBenchmarkMatrix();
}

void launch_DtoD_memcpy_bidirectional_read_SM(unsigned long long size, unsigned long long loopCount) {
    PeerValueMatrix<double> bandwidthValues(deviceCount, deviceCount);
    MemcpyOperationSM memcpyInstance(size, loopCount);

    for (int srcDeviceId = 0; srcDeviceId < deviceCount; srcDeviceId++) {
        for (int dstDeviceId = 0; dstDeviceId < deviceCount; dstDeviceId++) {
            if (dstDeviceId == srcDeviceId) {
                continue;
            }

            std::vector<MemcpyNode *> srcNodes = {new DeviceNode(size, srcDeviceId), new DeviceNode(size, dstDeviceId)};
            std::vector<MemcpyNode *> dstNodes = {new DeviceNode(size, dstDeviceId), new DeviceNode(size, srcDeviceId)};

            bandwidthValues.value(srcDeviceId, dstDeviceId) = memcpyInstance.doMemcpy(srcNodes, dstNodes);

            for (auto node : srcNodes) delete node;
            for (auto node : dstNodes) delete node;
        }
    }

    std::cout << "memcpy SM GPU(row) <-> GPU(column) bandwidth (GB/s)" << std::endl;
    std::cout << std::fixed << std::setprecision(2) << bandwidthValues << std::endl;
}

void launch_DtoD_memcpy_bidirectional_write_SM(unsigned long long size, unsigned long long loopCount) {
    // this is no different than the read test, other than src and dst and swapped, but the matrix printed is identical
    // I assume the intent was to swap the context, such that the copy context was the dst instead of the source, but that is not the case here
    // Memcpy memcpyInstance = Memcpy(copyKernel, size, loopCount);

    // for (int targetDeviceId = 0; targetDeviceId < deviceCount; targetDeviceId++) {
    //     std::vector<DeviceNode *> devicesDir1, devicesDir2, dstDevicesDir1, dstDevicesDir2;
    //     for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
    //         devicesDir1.push_back(new DeviceNode(size, targetDeviceId));
    //         devicesDir2.push_back(new DeviceNode(size, targetDeviceId));
    //         dstDevicesDir1.push_back(new DeviceNode(size, deviceId));
    //         dstDevicesDir2.push_back(new DeviceNode(size, deviceId));
    //     }

    //     if (parallel) {
    //         #pragma omp parallel num_threads(deviceCount)
    //         {
    //             int deviceId = omp_get_thread_num();
    //             memcpyInstance.doMemcpy(devicesDir1[deviceId], dstDevicesDir1[deviceId], devicesDir2[deviceId], dstDevicesDir2[deviceId]);
    //         }
    //     } else {
    //         parallel = 1;
    //         for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
    //             memcpyInstance.doMemcpy(devicesDir1[deviceId], dstDevicesDir1[deviceId], devicesDir2[deviceId], dstDevicesDir2[deviceId]);
    //         }
    //         parallel = 0;
    //     }

    //     for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
    //         delete devicesDir1[deviceId], devicesDir2[deviceId];
    //         delete dstDevicesDir1[deviceId], dstDevicesDir2[deviceId];
    //     }
    // }

    // memcpyInstance.printBenchmarkMatrix();
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
