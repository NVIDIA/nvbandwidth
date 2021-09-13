#include <cstddef>
#include <cuda.h>

#include "benchmarks.h"
#include "memcpy.h"


void launch_HtoD_memcpy_CE(unsigned long long size, unsigned long long loopCount) {
    Memcpy memcpyOp = Memcpy(size, cuMemcpyAsync, loopCount);
    memcpyOp.memcpy(false);
    memcpyOp.printBenchmarkMatrix();
}

void launch_DtoH_memcpy_CE(unsigned long long size, unsigned long long loopCount) {
    Memcpy memcpyOp = Memcpy(size, cuMemcpyAsync, loopCount);

    memcpyOp.memcpy(false, true);
    memcpyOp.printBenchmarkMatrix(true);
}

void launch_HtoD_memcpy_bidirectional_CE(unsigned long long size, unsigned long long loopCount) {
    Memcpy memcpyDir0 = Memcpy(size, cuMemcpyAsync, loopCount);
    Memcpy memcpyDir1 = Memcpy(size, cuMemcpyAsync, loopCount);

    memcpyDir0.memcpy(false, false, &memcpyDir1);
    memcpyDir0.printBenchmarkMatrix();
}

void launch_DtoH_memcpy_bidirectional_CE(unsigned long long size, unsigned long long loopCount) {
    HostNode hostNode0 = HostNode(size);
    HostNode hostNode1 = HostNode(size);
    Memcpy memcpyDir0 = Memcpy(size, cuMemcpyAsync, loopCount);
    Memcpy memcpyDir1 = Memcpy(size, cuMemcpyAsync, loopCount);

    memcpyDir0.memcpy(false, true, &memcpyDir1);
    memcpyDir0.printBenchmarkMatrix(true);
}

void launch_DtoD_memcpy_CE(bool read, unsigned long long size, unsigned long long loopCount) {
    // Setting target to null outside the loop to set it to each device in the loop
    Memcpy memcpyOp = Memcpy(size, cuMemcpyAsync, loopCount,true);
    for (size_t currentDevice = 0; currentDevice < memcpyOp.getDeviceCount(); currentDevice++) {
        DeviceNode dev = DeviceNode(currentDevice, size, &memcpyOp);
        memcpyOp.setTarget(&dev);
        memcpyOp.memcpy(false, !read);
    }
    memcpyOp.printBenchmarkMatrix();
}


void launch_DtoD_memcpy_read_CE(unsigned long long size, unsigned long long loopCount) {
    launch_DtoD_memcpy_CE(true, size, loopCount);
}
void launch_DtoD_memcpy_write_CE(unsigned long long size, unsigned long long loopCount) {
    launch_DtoD_memcpy_CE(false, size, loopCount);
}

void launch_DtoD_memcpy_bidirectional_CE(unsigned long long size, unsigned long long loopCount) {
    // Setting target to null outside the loop to set it to each device in the loop
    Memcpy memcpyDir0 = Memcpy(size, cuMemcpyAsync, loopCount, true);
    Memcpy memcpyDir1 = Memcpy(size, cuMemcpyAsync, loopCount, true);
    for (int currentDevice = 0; currentDevice < memcpyDir0.getDeviceCount(); currentDevice++) {
        DeviceNode deviceNode0 = DeviceNode(currentDevice, size, &memcpyDir0);
        DeviceNode deviceNode1 = DeviceNode(currentDevice, size, &memcpyDir1);
        memcpyDir0.setTarget(&deviceNode0);
        memcpyDir1.setTarget(&deviceNode1);

        memcpyDir0.memcpy(false, true, &memcpyDir1);
    }

    memcpyDir0.printBenchmarkMatrix(true);
}
