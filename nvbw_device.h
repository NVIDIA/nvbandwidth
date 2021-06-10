#include <cuda_runtime_api.h>
#include <cuda.h>
#include <vector>
#include <map>
#include <functional>
#include <stdlib.h>

#include "nvbw_os.h"


typedef std::function<bool (int deviceId, cudaDeviceProp *prop)> DeviceFilter;
extern std::vector<cudaDeviceProp> deviceProps;

class UtilityModuleFixture {
    static std::map<CUcontext, CUmodule>& getModuleMap() {
        static std::map<CUcontext, CUmodule> modules;
        return modules;
    }

public:
    static CUmodule getUtilityModule(CUcontext ctx) {
        return getModuleMap()[ctx];
    }

    static void addUtilityModule(CUcontext ctx, CUmodule module) {
        getModuleMap()[ctx] = module;
    }

    static bool checkIfUtilityModuleIsLoaded(CUcontext ctx) {
        return getModuleMap().find(ctx) != getModuleMap().end();
    }

    static void removeUtilityModule(CUcontext ctx) {
        getModuleMap().erase(ctx);
    }
};

bool alwaysTrueDeviceFilter(int deviceId, cudaDeviceProp *prop);

std::vector<int> filterDevices(DeviceFilter filter);
