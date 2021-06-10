#include "spinKernel.h"
#include "nvbw_device.h"

CUresult launch_spin_kernel(volatile int *latch, CUstream stream, bool single, unsigned long long timeout_ns)
{
    int blockSize = 1, gridSize = 1;
    int clocks_per_ms = 0;
    CUcontext ctx;
    CUdevice currentDev;
    CUfunction spin_kernel_fn;

    cuCtxGetCurrent(&ctx); // ASSERT_DRV
    cuCtxGetDevice(&currentDev); // ASSERT_DRV
    cuDeviceGetAttribute(&clocks_per_ms, CU_DEVICE_ATTRIBUTE_CLOCK_RATE, currentDev); // ASSERT_DRV
    cuModuleGetFunction(&spin_kernel_fn, UtilityModuleFixture::getUtilityModule(ctx), "spinKernel"); // ASSERT_DRV

    if (!single) {
        cuOccupancyMaxPotentialBlockSize(&gridSize, &blockSize, spin_kernel_fn, 0, 0, 0); // ASSERT_DRV
    }

    unsigned long long timeout_clocks = (clocks_per_ms * timeout_ns) / 1000;

    void* params[] = {&latch, &timeout_clocks};

    return cuLaunchKernel(
             spin_kernel_fn,
             gridSize, 1, 1,
             blockSize, 1, 1,
             0,
             stream,
             params,
             0);
}
