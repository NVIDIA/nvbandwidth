#ifndef MEMCPY_H
#define MEMCPY_H

#include "common.h"
#include "memory_utils.h"

typedef CUresult (*MemcpyFunc)(CUdeviceptr dst, CUdeviceptr src, size_t sizeInElement, CUstream stream);

class MemcpyNode {
protected:
  void *buffer{};
public:
    explicit MemcpyNode();
    void *getBuffer();
    virtual int getNodeIdx() const = 0;
};
class Memcpy;

class HostNode : public MemcpyNode {
public:
    explicit HostNode(size_t bufferSize);
    ~HostNode();
    int getNodeIdx() const override;
};

class DeviceNode : public MemcpyNode {
private:
    int deviceIdx;
    CUcontext primaryCtx{};
    Memcpy *copy;
public:
    DeviceNode(int deviceIdx, size_t bufferSize, Memcpy *copy);
    ~DeviceNode();
    CUcontext getPrimaryCtx() const;
    int getNodeIdx() const override;
};

class Memcpy {
private:
    MemcpyNode *target;
    size_t copySize;
    MemcpyFunc memcpyFunc;
    unsigned long long loopCount;
    int deviceCount{};

    size_t firstEnabledCPU;
    size_t *procMask;
    bool d2d;
    std::vector<bool> canAccessTarget;
    PeerValueMatrix<double> *bandwidthValues{};
    void doMemcpy(DeviceNode *device, unsigned long long *bandwidth, bool useSM, bool reverse, Memcpy *biDirCopy);
public:
    Memcpy(size_t copySize, MemcpyFunc memcpyFunc, unsigned long long loopCount, bool d2d = false);
    ~Memcpy();
    void memcpy(bool useSM, bool reverse = false, Memcpy *biDirCopy = nullptr);
    void setTarget(MemcpyNode *newTarget);
    MemcpyNode *getTarget() const;
    int getDeviceCount() const;
    void printBenchmarkMatrix(bool reverse = false);
    bool isD2d() const;
    void prepareNodes(DeviceNode *peer);
};

#endif
