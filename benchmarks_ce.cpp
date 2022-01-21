/*
 * Copyright 1993-2021 by NVIDIA Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#include <cuda.h>
#include <omp.h>

#include "benchmarks.h"

void launch_HtoD_memcpy_CE(unsigned long long size, unsigned long long loopCount) {
    MemcpyOperation memcpyInstance(cuMemcpyAsync, size, loopCount);

    for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
        HostNode hostNode(size, deviceId);
        DeviceNode deviceNode(size, deviceId);

        memcpyInstance.doMemcpy(&hostNode, &deviceNode);
    }

    memcpyInstance.printBenchmarkMatrix();
}

void launch_DtoH_memcpy_CE(unsigned long long size, unsigned long long loopCount) {
    MemcpyOperation memcpyInstance(cuMemcpyAsync, size, loopCount);

    for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
        HostNode hostNode(size, deviceId);
        DeviceNode deviceNode(size, deviceId);

        memcpyInstance.doMemcpy(&deviceNode, &hostNode);
    }

    memcpyInstance.printBenchmarkMatrix();
}

void launch_HtoD_memcpy_bidirectional_CE(unsigned long long size, unsigned long long loopCount) {
    MemcpyOperation memcpyInstance(cuMemcpyAsync, size, loopCount);

    for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
        std::vector<MemcpyNode *> srcNodes = {new HostNode(size, deviceId), new DeviceNode(size, deviceId)};
        std::vector<MemcpyNode *> dstNodes = {new DeviceNode(size, deviceId), new HostNode(size, deviceId)};

        memcpyInstance.doMemcpy(srcNodes, dstNodes);

        for (auto node : srcNodes) delete node;
        for (auto node : dstNodes) delete node;
    }

    memcpyInstance.printBenchmarkMatrix();
}

void launch_DtoH_memcpy_bidirectional_CE(unsigned long long size, unsigned long long loopCount) {
    MemcpyOperation memcpyInstance(cuMemcpyAsync, size, loopCount);

    for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
        std::vector<MemcpyNode *> srcNodes = {new DeviceNode(size, deviceId), new HostNode(size, deviceId)};
        std::vector<MemcpyNode *> dstNodes = {new HostNode(size, deviceId), new DeviceNode(size, deviceId)};

        memcpyInstance.doMemcpy(srcNodes, dstNodes);

        for (auto node : srcNodes) delete node;
        for (auto node : dstNodes) delete node;
    }

    memcpyInstance.printBenchmarkMatrix();
}

void launch_DtoD_memcpy_read_CE(unsigned long long size, unsigned long long loopCount) {
    MemcpyOperation memcpyInstance(cuMemcpyAsync, size, loopCount);

    for (int srcDeviceId = 0; srcDeviceId < deviceCount; srcDeviceId++) {
        for (int dstDeviceId = 0; dstDeviceId < deviceCount; dstDeviceId++) {
            if (dstDeviceId == srcDeviceId) {
                continue;
            }

            DeviceNode srcNode(size, srcDeviceId);
            DeviceNode dstNode(size, dstDeviceId);

            memcpyInstance.doMemcpy(&srcNode, &dstNode);
        }
    }

    memcpyInstance.printBenchmarkMatrix();
}

void launch_DtoD_memcpy_write_CE(unsigned long long size, unsigned long long loopCount) {
    // this is no different than the read test, other than src and dst and swapped, but the matrix printed is identical
    // I assume the intent was to swap the context, such that the copy context was the dst instead of the source, but that is not the case here
    // Memcpy memcpyInstance = Memcpy(cuMemcpyAsync, size, loopCount);

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

void launch_DtoD_memcpy_bidirectional_CE(unsigned long long size, unsigned long long loopCount) {
    MemcpyOperation memcpyInstance = MemcpyOperation(cuMemcpyAsync, size, loopCount);

    for (int srcDeviceId = 0; srcDeviceId < deviceCount; srcDeviceId++) {
        for (int dstDeviceId = 0; dstDeviceId < deviceCount; dstDeviceId++) {
            if (dstDeviceId == srcDeviceId) {
                continue;
            }

            std::vector<MemcpyNode *> srcNodes = {new DeviceNode(size, srcDeviceId), new DeviceNode(size, dstDeviceId)};
            std::vector<MemcpyNode *> dstNodes = {new DeviceNode(size, dstDeviceId), new DeviceNode(size, srcDeviceId)};

            memcpyInstance.doMemcpy(srcNodes, dstNodes);

            for (auto node : srcNodes) delete node;
            for (auto node : dstNodes) delete node;
        }
    }

    memcpyInstance.printBenchmarkMatrix();
}

void launch_DtoD_paired_memcpy_read_CE(unsigned long long size, unsigned long long loopCount) {
    //TODO fix me
    // Memcpy memcpyInstance = Memcpy(cuMemcpyAsync, size, loopCount);

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

void launch_DtoD_paired_memcpy_write_CE(unsigned long long size, unsigned long long loopCount) {
    // TODO fix me
//     Memcpy memcpyInstance = Memcpy(cuMemcpyAsync, size, loopCount);

//     std::vector<DeviceNode *> devices;
//     for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
//         devices.push_back(new DeviceNode(size, deviceId));
//     }

//     if (parallel) {
//         #pragma omp parallel num_threads(deviceCount / 2)
//         {
//             int deviceId = omp_get_thread_num();
//             memcpyInstance.doMemcpy(devices[deviceId + (deviceCount / 2)], devices[deviceId]);
//         }
//     } else {
//         parallel = 1;
//         for (int deviceId = 0; deviceId < deviceCount / 2; deviceId++) {
//             memcpyInstance.doMemcpy(devices[deviceId + (deviceCount / 2)], devices[deviceId]);
//         }
//         parallel = 0;
//     }

//     for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
//         delete devices[deviceId];
//     }

//     memcpyInstance.printBenchmarkMatrix();
}
