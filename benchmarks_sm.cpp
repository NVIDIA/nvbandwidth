#include <omp.h>

#include "benchmarks.h"
#include "copy_kernel.cuh"

void launch_HtoD_memcpy_SM(unsigned long long size, unsigned long long loopCount) {
    std::vector<HostNode *> hosts = std::vector<HostNode *>(deviceCount);
    std::vector<DeviceNode *> devices = std::vector<DeviceNode *>(deviceCount);
    for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
        hosts[deviceId] = new HostNode(size, deviceId);
        devices[deviceId] = new DeviceNode(size, deviceId);
    }

    Memcpy memcpyInstance = Memcpy(copyKernel, size, loopCount);
    if (parallel) {
        #pragma omp parallel num_threads(deviceCount)
        {
            int deviceId = omp_get_thread_num();
            memcpyInstance.doMemcpy(hosts[deviceId], devices[deviceId]);
        }
    } else {
        for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
            memcpyInstance.doMemcpy(hosts[deviceId], devices[deviceId]);
        }
    }

    memcpyInstance.printBenchmarkMatrix();
}

void launch_DtoH_memcpy_SM(unsigned long long size, unsigned long long loopCount) {
    std::vector<HostNode *> hosts = std::vector<HostNode *>(deviceCount);
    std::vector<DeviceNode *> devices = std::vector<DeviceNode *>(deviceCount);
    for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
        hosts[deviceId] = new HostNode(size, deviceId);
        devices[deviceId] = new DeviceNode(size, deviceId);
    }

    Memcpy memcpyInstance = Memcpy(copyKernel                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          , size, loopCount);
    if (parallel) {
        #pragma omp parallel num_threads(deviceCount)
        {
            int deviceId = omp_get_thread_num();
            memcpyInstance.doMemcpy(devices[deviceId], hosts[deviceId]);
        }
    } else {
        for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
            memcpyInstance.doMemcpy(devices[deviceId], hosts[deviceId]);
        }
    }

    memcpyInstance.printBenchmarkMatrix();
}

void launch_DtoD_memcpy_read_SM(unsigned long long size, unsigned long long loopCount) {
    std::vector<DeviceNode *> devices = std::vector<DeviceNode *>(deviceCount);
    for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
        devices[deviceId] = new DeviceNode(size, deviceId);
    }

    Memcpy memcpyInstance = Memcpy(copyKernel, size, loopCount);
    for (int targetDeviceId = 0; targetDeviceId < deviceCount; targetDeviceId++) {
        if (parallel) {
            #pragma omp parallel num_threads(deviceCount)
            {
                int deviceId = omp_get_thread_num();
                DeviceNode *dst = new DeviceNode(size, deviceId);
                memcpyInstance.doMemcpy(devices[targetDeviceId], dst);
				delete dst;
            }
        } else {
            for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
                DeviceNode *dst = new DeviceNode(size, deviceId);
                memcpyInstance.doMemcpy(devices[targetDeviceId], dst);
				delete dst;
            }
        }
    }

    memcpyInstance.printBenchmarkMatrix();
}

void launch_DtoD_memcpy_write_SM(unsigned long long size, unsigned long long loopCount) {
    std::vector<DeviceNode *> devices = std::vector<DeviceNode *>(deviceCount);
    for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
        devices[deviceId] = new DeviceNode(size, deviceId);
    }

    Memcpy memcpyInstance = Memcpy(copyKernel, size, loopCount);
    for (int targetDeviceId = 0; targetDeviceId < deviceCount; targetDeviceId++) {
        if (parallel) {
            #pragma omp parallel num_threads(deviceCount)
            {
                int deviceId = omp_get_thread_num();
                DeviceNode *dst = new DeviceNode(size, deviceId);
                memcpyInstance.doMemcpy(dst, devices[targetDeviceId]);
				delete dst;
            }
        } else {
            for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
                DeviceNode *dst = new DeviceNode(size, deviceId);
                memcpyInstance.doMemcpy(dst, devices[targetDeviceId]);
				delete dst;
            }
        }
    }

    memcpyInstance.printBenchmarkMatrix();
}

void launch_DtoD_memcpy_bidirectional_read_SM(unsigned long long size, unsigned long long loopCount) {
    std::vector<DeviceNode *> devicesDir1 = std::vector<DeviceNode *>(deviceCount);
    std::vector<DeviceNode *> devicesDir2 = std::vector<DeviceNode *>(deviceCount);
    for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
        devicesDir1[deviceId] = new DeviceNode(size, deviceId);
        devicesDir2[deviceId] = new DeviceNode(size, deviceId);
    }

    Memcpy memcpyInstance = Memcpy(copyKernel, size, loopCount);
    for (int targetDeviceId = 0; targetDeviceId < deviceCount; targetDeviceId++) {
        if (parallel) {
        #pragma omp parallel num_threads(deviceCount)
            {
                int deviceId = omp_get_thread_num();
                DeviceNode *dst = new DeviceNode(size, deviceId);
                DeviceNode *src = new DeviceNode(size, deviceId);
                memcpyInstance.doMemcpy(devicesDir1[targetDeviceId], dst);
                memcpyInstance.doMemcpy(src, devicesDir2[targetDeviceId]);
                delete dst, delete src;
            }
        } else {
            for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
                DeviceNode *dst = new DeviceNode(size, deviceId);
                DeviceNode *src = new DeviceNode(size, deviceId);
                memcpyInstance.doMemcpy(devicesDir1[targetDeviceId], dst);
                memcpyInstance.doMemcpy(src, devicesDir2[targetDeviceId]);
                delete dst, delete src;
            }
        }
    }

    memcpyInstance.printBenchmarkMatrix();
}

void launch_DtoD_memcpy_bidirectional_write_SM(unsigned long long size, unsigned long long loopCount) {
    std::vector<DeviceNode *> devicesDir1 = std::vector<DeviceNode *>(deviceCount);
    std::vector<DeviceNode *> devicesDir2 = std::vector<DeviceNode *>(deviceCount);
    for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
        devicesDir1[deviceId] = new DeviceNode(size, deviceId);
        devicesDir2[deviceId] = new DeviceNode(size, deviceId);
    }

    Memcpy memcpyInstance = Memcpy(copyKernel, size, loopCount);
    for (int targetDeviceId = 0; targetDeviceId < deviceCount; targetDeviceId++) {
        if (parallel) {
            #pragma omp parallel num_threads(deviceCount)
            {
                int deviceId = omp_get_thread_num();
                DeviceNode *dst = new DeviceNode(size, deviceId);
                DeviceNode *src = new DeviceNode(size, deviceId);
                memcpyInstance.doMemcpy(dst, devicesDir1[targetDeviceId]);
                memcpyInstance.doMemcpy(devicesDir2[targetDeviceId], src);
				delete dst, delete src;
            }
        } else {
            for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
                DeviceNode *dst = new DeviceNode(size, deviceId);
                DeviceNode *src = new DeviceNode(size, deviceId);
                memcpyInstance.doMemcpy(dst, devicesDir1[targetDeviceId]);
                memcpyInstance.doMemcpy(devicesDir2[targetDeviceId], src);
				delete dst, delete src;
            }
        }
    }

    memcpyInstance.printBenchmarkMatrix();
}
