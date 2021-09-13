#include <cstddef>

#include "benchmarks.h"
#include "memcpy.h"
#include "copy_kernel.cuh"

void launch_HtoD_memcpy_SM(unsigned long long size, unsigned long long loopCount) {
    Memcpy memcpyOp = Memcpy(size, copyKernel, loopCount);
    memcpyOp.memcpy(true);
    memcpyOp.printBenchmarkMatrix();
}

void launch_DtoH_memcpy_SM(unsigned long long size, unsigned long long loopCount) {
    Memcpy memcpyOp = Memcpy(size, copyKernel, loopCount);
    memcpyOp.memcpy(true, true);
    memcpyOp.printBenchmarkMatrix();
}

static void launch_DtoD_memcpy_SM(bool read, unsigned long long size, unsigned long long loopCount) {
    // Setting target to null outside the loop to set it to each device in the loop
    Memcpy memcpyOp = Memcpy(size, copyKernel, loopCount,true);
    for (size_t currentDevice = 0; currentDevice < memcpyOp.getDeviceCount(); currentDevice++) {
        DeviceNode dev = DeviceNode(currentDevice, size, &memcpyOp);
        memcpyOp.setTarget(&dev);
        memcpyOp.memcpy(true, !read);
    }
    memcpyOp.printBenchmarkMatrix();
}


static void launch_DtoD_memcpy_bidirectional_SM(bool read, unsigned long long size, unsigned long long loopCount) {
    // Setting target to null outside the loop to set it to each device in the loop
    Memcpy memcpyDir0 = Memcpy(size, copyKernel, loopCount, true);
    Memcpy memcpyDir1 = Memcpy(size, copyKernel, loopCount, true);
    for (int currentDevice = 0; currentDevice < memcpyDir0.getDeviceCount(); currentDevice++) {
        DeviceNode deviceNode0 = DeviceNode(currentDevice, size, &memcpyDir0);
        DeviceNode deviceNode1 = DeviceNode(currentDevice, size, &memcpyDir1);
        memcpyDir0.setTarget(&deviceNode0);
        memcpyDir1.setTarget(&deviceNode1);

        memcpyDir0.memcpy(true, read, &memcpyDir1);
    }

    memcpyDir0.printBenchmarkMatrix(true);
}

void launch_DtoD_memcpy_bidirectional_read_SM(unsigned long long size, unsigned long long loopCount) {
    launch_DtoD_memcpy_bidirectional_SM(true, size, loopCount);
}
void launch_DtoD_memcpy_bidirectional_write_SM(unsigned long long size, unsigned long long loopCount) {
    launch_DtoD_memcpy_bidirectional_SM(false, size, loopCount);
}
void launch_DtoD_memcpy_read_SM(unsigned long long size, unsigned long long loopCount) {
    launch_DtoD_memcpy_SM(true, size, loopCount);
}
void launch_DtoD_memcpy_write_SM(unsigned long long size, unsigned long long loopCount) {
    launch_DtoD_memcpy_SM(false, size, loopCount);
}
