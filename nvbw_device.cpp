#include "nvbw_device.h"

std::vector<cudaDeviceProp> deviceProps;

bool alwaysTrueDeviceFilter(int deviceId, cudaDeviceProp *prop) {
    return true;
}

std::vector<int> filterDevices(DeviceFilter filter)
{
    int deviceCount = (int)deviceProps.size();
    std::vector<int> filtered;

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp prop = deviceProps[dev];
        bool pass = filter(dev, &prop);
        if (pass) {
            filtered.push_back(dev);
        }
    }

    return filtered;
}
