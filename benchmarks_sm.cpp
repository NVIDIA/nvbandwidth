#include <omp.h>

#include "benchmarks.h"
#include "copy_kernel.cuh"

void launch_HtoD_memcpy_SM(unsigned long long size, unsigned long long loopCount) {
    Memcpy memcpyInstance = Memcpy(copyKernel, size, loopCount);

    std::vector<HostNode *> hosts;
    std::vector<DeviceNode *> dstDevices;
    for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
        hosts.push_back(new HostNode(size, deviceId));
        dstDevices.push_back(new DeviceNode(size, deviceId));
    }

    if (parallel) {
        #pragma omp parallel num_threads(deviceCount)
        {
            int deviceId = omp_get_thread_num();
            memcpyInstance.doMemcpy(hosts[deviceId], dstDevices[deviceId]);
        }
    } else {
        for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
            memcpyInstance.doMemcpy(hosts[deviceId], dstDevices[deviceId]);
        }
    }

    for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
        delete hosts[deviceId];
        delete dstDevices[deviceId];
    }
    memcpyInstance.printBenchmarkMatrix();
}

void launch_DtoH_memcpy_SM(unsigned long long size, unsigned long long loopCount) {
    Memcpy memcpyInstance = Memcpy(copyKernel, size, loopCount);

    std::vector<HostNode *> hosts;
    std::vector<DeviceNode *> dstDevices;
    for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
        hosts.push_back(new HostNode(size, deviceId));
        dstDevices.push_back(new DeviceNode(size, deviceId));
    }

    if (parallel) {
        #pragma omp parallel num_threads(deviceCount)
        {
            int deviceId = omp_get_thread_num();
            memcpyInstance.doMemcpy(dstDevices[deviceId], hosts[deviceId]);
        }
    } else {
        for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
            memcpyInstance.doMemcpy(dstDevices[deviceId], hosts[deviceId]);
        }
    }

    for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
        delete hosts[deviceId];
        delete dstDevices[deviceId];
    }

    memcpyInstance.printBenchmarkMatrix();
}

void launch_DtoD_memcpy_read_SM(unsigned long long size, unsigned long long loopCount) {
    Memcpy memcpyInstance = Memcpy(copyKernel, size, loopCount);

    for (int targetDeviceId = 0; targetDeviceId < deviceCount; targetDeviceId++) {
        std::vector<DeviceNode *> devices, dstDevices;
        for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
            devices.push_back(new DeviceNode(size, targetDeviceId));
            dstDevices.push_back(new DeviceNode(size, deviceId));
        }

        if (parallel) {
            #pragma omp parallel num_threads(deviceCount)
            {
                int deviceId = omp_get_thread_num();
                memcpyInstance.doMemcpy(dstDevices[deviceId], devices[deviceId]);
            }
        } else {
            for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
                memcpyInstance.doMemcpy(dstDevices[deviceId], devices[deviceId]);
            }
        }

        for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
            delete devices[deviceId];
            delete dstDevices[deviceId];
        }
    }

    memcpyInstance.printBenchmarkMatrix();
}

void launch_DtoD_memcpy_write_SM(unsigned long long size, unsigned long long loopCount) {
    Memcpy memcpyInstance = Memcpy(copyKernel, size, loopCount);

    for (int targetDeviceId = 0; targetDeviceId < deviceCount; targetDeviceId++) {
        std::vector<DeviceNode *> devices, dstDevices;
        for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
            devices.push_back(new DeviceNode(size, targetDeviceId));
            dstDevices.push_back(new DeviceNode(size, deviceId));
        }

        if (parallel) {
            #pragma omp parallel num_threads(deviceCount)
            {
                int deviceId = omp_get_thread_num();
                memcpyInstance.doMemcpy(devices[deviceId], dstDevices[deviceId]);
            }
        } else {
            for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
                memcpyInstance.doMemcpy(devices[deviceId], dstDevices[deviceId]);
            }
        }

        for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
            delete devices[deviceId];
            delete dstDevices[deviceId];
        }
    }

    memcpyInstance.printBenchmarkMatrix();
}

void launch_DtoD_memcpy_bidirectional_read_SM(unsigned long long size, unsigned long long loopCount) {
    Memcpy memcpyInstance = Memcpy(copyKernel, size, loopCount);

    for (int targetDeviceId = 0; targetDeviceId < deviceCount; targetDeviceId++) {
        std::vector<DeviceNode *> devicesDir1, devicesDir2, dstDevicesDir1, dstDevicesDir2;
        for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
            devicesDir1.push_back(new DeviceNode(size, targetDeviceId));
            devicesDir2.push_back(new DeviceNode(size, targetDeviceId));
            dstDevicesDir1.push_back(new DeviceNode(size, deviceId));
            dstDevicesDir2.push_back(new DeviceNode(size, deviceId));
        }

        if (parallel) {
            #pragma omp parallel num_threads(deviceCount)
            {
                int deviceId = omp_get_thread_num();
                memcpyInstance.doMemcpy(dstDevicesDir1[deviceId], devicesDir1[deviceId], dstDevicesDir2[deviceId], devicesDir2[deviceId]);
            }
        } else {
            parallel = 1;
            for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
                memcpyInstance.doMemcpy(dstDevicesDir1[deviceId], devicesDir1[deviceId], dstDevicesDir2[deviceId], devicesDir2[deviceId]);
            }
            parallel = 0;
        }

        for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
            delete devicesDir1[deviceId], devicesDir2[deviceId];
            delete dstDevicesDir1[deviceId], dstDevicesDir2[deviceId];
        }
    }

    memcpyInstance.printBenchmarkMatrix();
}

void launch_DtoD_memcpy_bidirectional_write_SM(unsigned long long size, unsigned long long loopCount) {
    Memcpy memcpyInstance = Memcpy(copyKernel, size, loopCount);

    for (int targetDeviceId = 0; targetDeviceId < deviceCount; targetDeviceId++) {
        std::vector<DeviceNode *> devicesDir1, devicesDir2, dstDevicesDir1, dstDevicesDir2;
        for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
            devicesDir1.push_back(new DeviceNode(size, targetDeviceId));
            devicesDir2.push_back(new DeviceNode(size, targetDeviceId));
            dstDevicesDir1.push_back(new DeviceNode(size, deviceId));
            dstDevicesDir2.push_back(new DeviceNode(size, deviceId));
        }

        if (parallel) {
            #pragma omp parallel num_threads(deviceCount)
            {
                int deviceId = omp_get_thread_num();
                memcpyInstance.doMemcpy(devicesDir1[deviceId], dstDevicesDir1[deviceId], devicesDir2[deviceId], dstDevicesDir2[deviceId]);
            }
        } else {
            parallel = 1;
            for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
                memcpyInstance.doMemcpy(devicesDir1[deviceId], dstDevicesDir1[deviceId], devicesDir2[deviceId], dstDevicesDir2[deviceId]);
            }
            parallel = 0;
        }

        for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
            delete devicesDir1[deviceId], devicesDir2[deviceId];
            delete dstDevicesDir1[deviceId], dstDevicesDir2[deviceId];
        }
    }

    memcpyInstance.printBenchmarkMatrix();
}

void launch_DtoD_paired_memcpy_read_SM(unsigned long long size, unsigned long long loopCount) {
    Memcpy memcpyInstance = Memcpy(copyKernel, size, loopCount);

    std::vector<DeviceNode *> devices;
    for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
        devices.push_back(new DeviceNode(size, deviceId));
    }

    if (parallel) {
        #pragma omp parallel num_threads(deviceCount / 2)
        {
            int deviceId = omp_get_thread_num();
            memcpyInstance.doMemcpy(devices[deviceId], devices[deviceId + (deviceCount / 2)]);
        }
    } else {
        parallel = 1;
        for (int deviceId = 0; deviceId < deviceCount / 2; deviceId++) {
            memcpyInstance.doMemcpy(devices[deviceId], devices[deviceId + (deviceCount / 2)]);
        }
        parallel = 0;
    }

    for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
        delete devices[deviceId];
    }

    memcpyInstance.printBenchmarkMatrix();
}

void launch_DtoD_paired_memcpy_write_SM(unsigned long long size, unsigned long long loopCount) {
    Memcpy memcpyInstance = Memcpy(copyKernel, size, loopCount);

    std::vector<DeviceNode *> devices;
    for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
        devices.push_back(new DeviceNode(size, deviceId));
    }

    if (parallel) {
        #pragma omp parallel num_threads(deviceCount / 2)
        {
            int deviceId = omp_get_thread_num();
            memcpyInstance.doMemcpy(devices[deviceId + (deviceCount / 2)], devices[deviceId]);
        }
    } else {
        parallel = 1;
        for (int deviceId = 0; deviceId < deviceCount / 2; deviceId++) {
            memcpyInstance.doMemcpy(devices[deviceId + (deviceCount / 2)], devices[deviceId]);
        }
        parallel = 0;
    }

    for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
        delete devices[deviceId];
    }

    memcpyInstance.printBenchmarkMatrix();
}
