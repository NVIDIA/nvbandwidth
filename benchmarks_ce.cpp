#include <cuda.h>
#include <omp.h>

#include "benchmarks.h"

void launch_HtoD_memcpy_CE(unsigned long long size, unsigned long long loopCount) {
    std::vector<HostNode *> hosts = std::vector<HostNode *>(deviceCount);
    std::vector<DeviceNode *> devices = std::vector<DeviceNode *>(deviceCount);
    for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
        hosts[deviceId] = new HostNode(size, deviceId);
        devices[deviceId] = new DeviceNode(size, deviceId);
    }

    Memcpy memcpyInstance = Memcpy(cuMemcpyAsync, size, loopCount);
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

void launch_DtoH_memcpy_CE(unsigned long long size, unsigned long long loopCount) {
    std::vector<HostNode *> hosts = std::vector<HostNode *>(deviceCount);
    std::vector<DeviceNode *> devices = std::vector<DeviceNode *>(deviceCount);
    for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
        hosts[deviceId] = new HostNode(size, deviceId);
        devices[deviceId] = new DeviceNode(size, deviceId);
    }

    Memcpy memcpyInstance = Memcpy(cuMemcpyAsync, size, loopCount);
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

void launch_HtoD_memcpy_bidirectional_CE(unsigned long long size, unsigned long long loopCount) {
    std::vector<HostNode *> hostsDir1 = std::vector<HostNode *>(deviceCount);
    std::vector<HostNode *> hostsDir2 = std::vector<HostNode *>(deviceCount);
    std::vector<DeviceNode *> devicesDir1 = std::vector<DeviceNode *>(deviceCount);
    std::vector<DeviceNode *> devicesDir2 = std::vector<DeviceNode *>(deviceCount);
    for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
        hostsDir1[deviceId] = new HostNode(size, deviceId);
        hostsDir2[deviceId] = new HostNode(size, deviceId);
        devicesDir1[deviceId] = new DeviceNode(size, deviceId);
        devicesDir2[deviceId] = new DeviceNode(size, deviceId);
    }

    Memcpy memcpyInstance = Memcpy(cuMemcpyAsync, size, loopCount);
    if (parallel) {
        #pragma omp parallel num_threads(deviceCount)
        {
            int deviceId = omp_get_thread_num();
            memcpyInstance.doMemcpy(devicesDir1[deviceId], hostsDir1[deviceId]);
            memcpyInstance.doMemcpy(devicesDir2[deviceId], hostsDir2[deviceId]);
        }
    } else {
        for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
            memcpyInstance.doMemcpy(devicesDir1[deviceId], hostsDir1[deviceId]);
            memcpyInstance.doMemcpy(devicesDir2[deviceId], hostsDir2[deviceId]);
        }
    }

    memcpyInstance.printBenchmarkMatrix();
}

void launch_DtoH_memcpy_bidirectional_CE(unsigned long long size, unsigned long long loopCount) {
    std::vector<HostNode *> hostsDir1 = std::vector<HostNode *>(deviceCount);
    std::vector<HostNode *> hostsDir2 = std::vector<HostNode *>(deviceCount);
    std::vector<DeviceNode *> devicesDir1 = std::vector<DeviceNode *>(deviceCount);
    std::vector<DeviceNode *> devicesDir2 = std::vector<DeviceNode *>(deviceCount);
    for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
        hostsDir1[deviceId] = new HostNode(size, deviceId);
        hostsDir2[deviceId] = new HostNode(size, deviceId);
        devicesDir1[deviceId] = new DeviceNode(size, deviceId);
        devicesDir2[deviceId] = new DeviceNode(size, deviceId);
    }

    Memcpy memcpyInstance = Memcpy(cuMemcpyAsync, size, loopCount);
    if (parallel) {
#pragma omp parallel num_threads(deviceCount)
        {
            int deviceId = omp_get_thread_num();
            memcpyInstance.doMemcpy(hostsDir1[deviceId], devicesDir1[deviceId]);
            memcpyInstance.doMemcpy(hostsDir2[deviceId], devicesDir2[deviceId]);
        }
    } else {
        for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
            memcpyInstance.doMemcpy(hostsDir1[deviceId], devicesDir1[deviceId]);
            memcpyInstance.doMemcpy(hostsDir2[deviceId], devicesDir2[deviceId]);
        }
    }

    memcpyInstance.printBenchmarkMatrix();
}

void launch_DtoD_memcpy_read_CE(unsigned long long size, unsigned long long loopCount) {
    std::vector<DeviceNode *> devices = std::vector<DeviceNode *>(deviceCount);
    for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
        devices[deviceId] = new DeviceNode(size, deviceId);
    }

    Memcpy memcpyInstance = Memcpy(cuMemcpyAsync, size, loopCount);
    for (int targetDeviceId = 0; targetDeviceId < deviceCount; targetDeviceId++) {
        if (parallel) {
            #pragma omp parallel num_threads(deviceCount)
            {
                int deviceId = omp_get_thread_num();
                DeviceNode *dst = new DeviceNode(size, deviceId);
                memcpyInstance.doMemcpy(devices[targetDeviceId], dst);
            }
        } else {
            for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
                DeviceNode *dst = new DeviceNode(size, deviceId);
                memcpyInstance.doMemcpy(devices[targetDeviceId], dst);
            }
        }
    }

    memcpyInstance.printBenchmarkMatrix();
}

void launch_DtoD_memcpy_write_CE(unsigned long long size, unsigned long long loopCount) {
    std::vector<DeviceNode *> devices = std::vector<DeviceNode *>(deviceCount);
    for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
        devices[deviceId] = new DeviceNode(size, deviceId);
    }

    Memcpy memcpyInstance = Memcpy(cuMemcpyAsync, size, loopCount);
    for (int targetDeviceId = 0; targetDeviceId < deviceCount; targetDeviceId++) {
        if (parallel) {
            #pragma omp parallel num_threads(deviceCount)
            {
                int deviceId = omp_get_thread_num();
                DeviceNode *dst = new DeviceNode(size, deviceId);
                memcpyInstance.doMemcpy(devices[targetDeviceId], dst);
            }
        } else {
            for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
                DeviceNode *dst = new DeviceNode(size, deviceId);
                memcpyInstance.doMemcpy(devices[targetDeviceId], dst);
            }
        }
    }

    memcpyInstance.printBenchmarkMatrix();
}

void launch_DtoD_memcpy_bidirectional_CE(unsigned long long size, unsigned long long loopCount) {
    std::vector<DeviceNode *> devicesDir1 = std::vector<DeviceNode *>(deviceCount);
    std::vector<DeviceNode *> devicesDir2 = std::vector<DeviceNode *>(deviceCount);
    for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
        devicesDir1[deviceId] = new DeviceNode(size, deviceId);
        devicesDir2[deviceId] = new DeviceNode(size, deviceId);
    }

    Memcpy memcpyInstance = Memcpy(cuMemcpyAsync, size, loopCount);
    for (int targetDeviceId = 0; targetDeviceId < deviceCount; targetDeviceId++) {
        if (parallel) {
            #pragma omp parallel num_threads(deviceCount)
            {
                int deviceId = omp_get_thread_num();
                DeviceNode *dst = new DeviceNode(size, deviceId);
                DeviceNode *src = new DeviceNode(size, deviceId);
                memcpyInstance.doMemcpy(devicesDir1[targetDeviceId], dst);
                memcpyInstance.doMemcpy(src, devicesDir2[targetDeviceId]);
            }
        } else {
            for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
                DeviceNode *dst = new DeviceNode(size, deviceId);
                DeviceNode *src = new DeviceNode(size, deviceId);
                memcpyInstance.doMemcpy(devicesDir1[targetDeviceId], dst);
                memcpyInstance.doMemcpy(src, devicesDir2[targetDeviceId]);
            }
        }
    }

    memcpyInstance.printBenchmarkMatrix();
}
