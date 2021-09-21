#include "memcpy.h"

MemcpyNode::MemcpyNode(): buffer(nullptr) {}

void *MemcpyNode::getBuffer() {
    return buffer;
}

HostNode::HostNode(size_t bufferSize): MemcpyNode() {
    CU_ASSERT(cuMemHostAlloc(&buffer, bufferSize, CU_MEMHOSTALLOC_PORTABLE));
}

HostNode::~HostNode() {
    if (isMemoryOwnedByCUDA(buffer)) {
        CU_ASSERT(cuMemFreeHost(buffer));
    } else {
        free(buffer);
    }
}

// Host Nodes always return zero as they always represent one row in the bandwidth matrix
int HostNode::getNodeIdx() const {
    return 0;
}

DeviceNode::DeviceNode(int deviceIdx, size_t bufferSize, Memcpy *copy): deviceIdx(deviceIdx), MemcpyNode(), copy(copy) {
    setOptimalCpuAffinity(deviceIdx);
    CU_ASSERT(cuDevicePrimaryCtxRetain(&primaryCtx, deviceIdx));
    CU_ASSERT(cuCtxSetCurrent(primaryCtx));
    CU_ASSERT(cuMemAlloc((CUdeviceptr*)&buffer, bufferSize));

    // Host allocations and device peer access operations have to happen after device initialization
    copy->prepareNodes(this);
}

DeviceNode::~DeviceNode() {
    CU_ASSERT(cuCtxSetCurrent(primaryCtx));

    // Host node is allocated/deallocated per device
    if (!copy->isD2d() && copy->getTarget() != nullptr) {
        delete (HostNode *)(copy->getTarget());
        copy->setTarget(nullptr);
    }

    CU_ASSERT(cuMemFree((CUdeviceptr)buffer));
    CU_ASSERT(cuDevicePrimaryCtxRelease(deviceIdx));
}

CUcontext DeviceNode::getPrimaryCtx() const {
    return primaryCtx;
}

int DeviceNode::getNodeIdx() const {
    return deviceIdx;
}

Memcpy::Memcpy(size_t copySize, MemcpyFunc memcpyFunc, unsigned long long loopCount, bool d2d): target(nullptr),
    copySize(copySize), memcpyFunc(memcpyFunc), loopCount(loopCount), firstEnabledCPU(getFirstEnabledCPU()), d2d(d2d) {
    CU_ASSERT(cuDeviceGetCount(&deviceCount));
    bandwidthValues = new PeerValueMatrix<double>(d2d ? deviceCount : 1, deviceCount);
    procMask = (size_t *)calloc(1, PROC_MASK_SIZE);
    PROC_MASK_SET(procMask, firstEnabledCPU);
}

Memcpy::~Memcpy() {
    delete bandwidthValues;
    PROC_MASK_CLEAR(procMask, 0);
}

void Memcpy::memcpy(bool useSM, bool reverse, Memcpy *biDirCopy) {
    std::vector<DeviceNode *> devices;
    // Sequential loop to initiate devices and allocate buffers
    for (size_t currentDevice = 0; currentDevice < deviceCount; currentDevice++) {
        devices.push_back(new DeviceNode(currentDevice, copySize, this));
    }
    // Parallel loop to run memcpy
    #pragma omp parallel for if (parallel)
    for (size_t currentDevice = 0; currentDevice < deviceCount; currentDevice++) {
        if (d2d) {
            if (target->getNodeIdx() == currentDevice) continue;
            if (!canAccessTarget[currentDevice]) continue;
        }

        unsigned long long bandwidth = 0;
        cudaStat bandwidthStat;

        for (unsigned int n = 0; n < averageLoopCount; n++) {
            doMemcpy(devices[currentDevice], &bandwidth, useSM, reverse, biDirCopy);
            bandwidthStat((double)bandwidth);
            VERBOSE << "\tSample " << n << ' ' << std::fixed << std::setprecision (2) <<
                (double)bandwidth * 1e-9 << " GB/s\n";
        }
        VERBOSE << "       bandwidth: " << std::fixed << std::setprecision (2) << STAT_MEAN(bandwidthStat) * 1e-9 <<
            "(+/- " << STAT_ERROR(bandwidthStat) * 1e-9 << ") GB/s\n";
        bandwidth = (unsigned long long)(STAT_MEAN(bandwidthStat));
        bandwidthValues->value(target->getNodeIdx(), devices[currentDevice]->getNodeIdx()) = (double)bandwidth * 1e-9;
    }
    // Sequential loop to delete devices and deallocate buffers
    for (size_t currentDevice = 0; currentDevice < deviceCount; currentDevice++) {
        canAccessTarget.pop_back();
        delete devices[currentDevice];
    }
}

void Memcpy::doMemcpy(DeviceNode *device, unsigned long long *bandwidth, bool useSM, bool reverse, Memcpy *biDirCopy) {
    int dev1, dev2;
    CUstream stream_dir1, stream_dir2;
    CUevent startEvent_dir1, startEvent_dir2;
    CUevent endEvent_dir1, endEvent_dir2;
    DeviceNode *deviceBiDir;

    CUdevice cudaDevice;
    int multiProcessorCount;

    *bandwidth = 0;
    size_t size = copySize;
    if (useSM) {
        size /= sizeof(int4);
        CU_ASSERT(cuCtxGetDevice(&cudaDevice));
        CU_ASSERT(cuDeviceGetAttribute(&multiProcessorCount,CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, cudaDevice));
        unsigned long long totalThreadCount = (unsigned long long)(multiProcessorCount * numThreadPerBlock);
        size = totalThreadCount * (size / totalThreadCount);
    }

    volatile int *blockingVar = nullptr;

    CU_ASSERT(cuCtxSetCurrent(device->getPrimaryCtx()));
    CU_ASSERT(cuCtxGetDevice(&dev1));
    CU_ASSERT(cuStreamCreate(&stream_dir1, CU_STREAM_NON_BLOCKING));
    CU_ASSERT(cuEventCreate(&startEvent_dir1, CU_EVENT_DEFAULT));
    CU_ASSERT(cuEventCreate(&endEvent_dir1, CU_EVENT_DEFAULT));

    if (biDirCopy != nullptr) {
        deviceBiDir = new DeviceNode(device->getNodeIdx(), copySize, biDirCopy);
        CU_ASSERT(cuCtxSetCurrent(deviceBiDir->getPrimaryCtx()));
        CU_ASSERT(cuCtxGetDevice(&dev2));
        CU_ASSERT(cuStreamCreate(&stream_dir2, CU_STREAM_NON_BLOCKING));
        CU_ASSERT(cuEventCreate(&startEvent_dir2, CU_EVENT_DEFAULT));
        CU_ASSERT(cuEventCreate(&endEvent_dir2, CU_EVENT_DEFAULT));
    }

    CU_ASSERT(cuEventRecord(startEvent_dir1, stream_dir1));
    if (biDirCopy != nullptr)  {
        CU_ASSERT(cuStreamWaitEvent(stream_dir2, startEvent_dir1, 0));
        CU_ASSERT(cuEventRecord(startEvent_dir2, stream_dir2));
    }

    for (unsigned int n = 0; n < loopCount; n++) {
        if (reverse) {
            CU_ASSERT(memcpyFunc((CUdeviceptr)target->getBuffer(), (CUdeviceptr)device->getBuffer(), size, stream_dir1));
            if (biDirCopy != nullptr)  {
                CU_ASSERT(memcpyFunc((CUdeviceptr)deviceBiDir->getBuffer(), (CUdeviceptr)biDirCopy->target->getBuffer(), size, stream_dir2));
            }
        } else {
            CU_ASSERT(memcpyFunc((CUdeviceptr)device->getBuffer(), (CUdeviceptr)target->getBuffer(), size, stream_dir1));
            if (biDirCopy != nullptr)  {
                CU_ASSERT(memcpyFunc((CUdeviceptr)biDirCopy->target->getBuffer(), (CUdeviceptr)deviceBiDir->getBuffer(), size, stream_dir2));
            }
        }
    }

    CU_ASSERT(cuEventRecord(endEvent_dir1, stream_dir1));

    if (!disableP2P) {
        CU_ASSERT(cuMemHostAlloc((void **)&blockingVar, sizeof(*blockingVar), CU_MEMHOSTALLOC_PORTABLE));
        *blockingVar = 1;
    }

    if (biDirCopy != nullptr) {
        // Now, we need to ensure there is always work in the stream2 pending, to
        // ensure there always is interference to the stream1.
        unsigned int extraIters = loopCount > 1 ? (unsigned int)loopCount / 2 : 1;
        do {
            // Enqueue extra work
            for (unsigned int n = 0; n < extraIters; n++) {
                CU_ASSERT(memcpyFunc((CUdeviceptr)biDirCopy->target->getBuffer(), (CUdeviceptr)device->getBuffer(), size, stream_dir2));
            }

            // Record the event in the middle of interfering flow, to ensure the next
            // batch starts enqueuing before the previous one finishes.
            CU_ASSERT(cuEventRecord(endEvent_dir2, stream_dir2));

            // Add more iterations to hide latency of scheduling more work in the next
            // iteration of loop.
            for (unsigned int n = 0; n < extraIters; n++) {
                CU_ASSERT(memcpyFunc((CUdeviceptr)biDirCopy->target->getBuffer(), (CUdeviceptr)deviceBiDir->getBuffer(), size, stream_dir2));
            }

            // Wait until the flow in the interference stream2 is finished.
            CU_ASSERT(cuEventSynchronize(endEvent_dir2));
        } while (cuStreamQuery(stream_dir1) == CUDA_ERROR_NOT_READY);
    }

    CU_ASSERT(cuStreamSynchronize(stream_dir1));
    if (biDirCopy != nullptr) CU_ASSERT(cuStreamSynchronize(stream_dir2));

    CU_ASSERT(cuCtxSetCurrent(device->getPrimaryCtx()));
    float timeWithEvents = 0.0f;
    CU_ASSERT(cuEventElapsedTime(&timeWithEvents, startEvent_dir1, endEvent_dir1));
    double elapsedWithEventsInUs = ((double)timeWithEvents * 1000.0);

    *bandwidth += (size * (useSM ? sizeof(int4) : 1 ) * loopCount * 1000ull * 1000ull) / (unsigned long long)elapsedWithEventsInUs;

    if (useSM && biDirCopy != nullptr) {
        CU_ASSERT(cuCtxSetCurrent(deviceBiDir->getPrimaryCtx()));
        timeWithEvents = 0.0f;
        CU_ASSERT(cuEventElapsedTime(&timeWithEvents, startEvent_dir2, endEvent_dir2));
        elapsedWithEventsInUs = ((double)timeWithEvents * 1000.0);

        *bandwidth += (size * sizeof(int4) * loopCount * 1000ull * 1000ull) / (unsigned long long)elapsedWithEventsInUs;
    }

    if (!disableP2P) {
        CU_ASSERT(cuMemFreeHost((void *)blockingVar));
    }

    CU_ASSERT(cuCtxSynchronize());
}

int Memcpy::getDeviceCount() const {
    return deviceCount;
}

void Memcpy::setTarget(MemcpyNode *newTarget) {
    target = newTarget;
}

MemcpyNode *Memcpy::getTarget() const {
    return target;
}

void Memcpy::printBenchmarkMatrix(bool reverse) {
    std::cout << "memcpy CE GPU(row) " << (reverse ? "->" : "<-") << " GPU(column) bandwidth (GB/s):" << std::endl;
    std::cout << std::fixed << std::setprecision(2) << *bandwidthValues << std::endl;
}

bool Memcpy::isD2d() const {
    return d2d;
}

void Memcpy::prepareNodes(DeviceNode *peer) {
    if (d2d && target != nullptr && target != peer) { // Setup d2d peer access
        CUresult res;
        int canAccessPeer = 0;

        CU_ASSERT(cuDeviceCanAccessPeer(&canAccessPeer, target->getNodeIdx(), peer->getNodeIdx()));
        if (!canAccessPeer) {
            canAccessTarget.push_back(false);
        } else {
            res = cuCtxEnablePeerAccess(((DeviceNode *)target)->getPrimaryCtx(), 0);
            if (res != CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED) CU_ASSERT(res);
            CU_ASSERT(cuCtxSetCurrent(((DeviceNode *)target)->getPrimaryCtx()));
            res = cuCtxEnablePeerAccess(peer->getPrimaryCtx(), 0);
            if (res != CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED) CU_ASSERT(res);

            cuCtxSetCurrent(((DeviceNode *)target)->getPrimaryCtx());
            canAccessTarget.push_back(true);
        }
    } else { // Setup host for h2d and d2h
        if (target == nullptr) target = new HostNode(copySize);
    }
}
