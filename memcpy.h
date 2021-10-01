#ifndef MEMCPY_H
#define MEMCPY_H

#include "common.h"
#include "memory_utils.h"

typedef CUresult (*MemcpyCEFunc)(CUdeviceptr dst, CUdeviceptr src, size_t sizeInElement, CUstream stream);
typedef CUresult (*MemcpySMFunc)(CUdeviceptr dst, CUdeviceptr src, size_t sizeInElement, CUstream stream, unsigned long long loopCount);


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
    HostNode(size_t bufferSize, int targetDeviceId);
    ~HostNode();

    int getNodeIdx() const override;
};

class DeviceNode : public MemcpyNode {
private:
    int deviceIdx;
    CUcontext primaryCtx{};
public:
    DeviceNode(size_t bufferSize, int deviceIdx);
    ~DeviceNode();

    CUcontext getPrimaryCtx() const;
    int getNodeIdx() const override;
};

class Memcpy {
private:
    size_t copySize;
    MemcpyCEFunc ceFunc{nullptr};
    MemcpySMFunc smFunc{nullptr};
    CUcontext copyCtx;
    CUstream *masterStream{nullptr};
    CUevent *masterEvent{nullptr}; // Other memcpy operations wait on this event to start at the same time

    unsigned long long loopCount;
    size_t *procMask;

    PeerValueMatrix<double> *bandwidthValues{nullptr};

    void allocateBandwidthMatrix(bool hostVector = false);
    size_t smCopySize() const;
    unsigned long long doMemcpy(CUdeviceptr src, CUdeviceptr dst, bool skip = false);
public:

    Memcpy(MemcpyCEFunc memcpyFunc, size_t copySize, unsigned long long loopCount);
    Memcpy(MemcpySMFunc memcpyFunc, size_t copySize, unsigned long long loopCount);

    ~Memcpy();

    // To infer copy recipe
    void doMemcpy(HostNode *srcNode, DeviceNode *dstNode, HostNode *biDirSrc = nullptr, DeviceNode *biDirDst = nullptr);
    void doMemcpy(DeviceNode *srcNode, HostNode *dstNode, DeviceNode *biDirSrc = nullptr, HostNode *biDirDst = nullptr);
    void doMemcpy(DeviceNode *srcNode, DeviceNode *dstNode, DeviceNode *biDirSrc = nullptr, DeviceNode *biDirDst = nullptr);

    void printBenchmarkMatrix(bool reverse = false);
};

#endif
