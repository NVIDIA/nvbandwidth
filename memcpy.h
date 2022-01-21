/*
 * Copyright 1993-2021 by NVIDIA Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef MEMCPY_H
#define MEMCPY_H

#include "common.h"
#include "memory_utils.h"

// Signature of a CE copy function (e.g. cuMemcpyAsync)
typedef CUresult (*MemcpyCEFunc)(CUdeviceptr dst, CUdeviceptr src, size_t sizeInElement, CUstream stream);
// Signature of an SM copy function (e.g. copyKernel)
typedef CUresult (*MemcpySMFunc)(CUdeviceptr dst, CUdeviceptr src, size_t sizeInElement, CUstream stream, unsigned long long loopCount);


class MemcpyNode {
protected:
    void* buffer{};
public:
    explicit MemcpyNode();
    CUdeviceptr getBuffer();

    virtual int getNodeIdx() const = 0;
    virtual CUcontext getPrimaryCtx() const = 0;
};

// Represents the host buffer abstraction
class HostNode : public MemcpyNode {
public:
    // NUMA affinity is set here through allocation of memory in the socket group where `targetDeviceId` resides
    HostNode(size_t bufferSize, int targetDeviceId);
    ~HostNode();

    int getNodeIdx() const override;
    CUcontext getPrimaryCtx() const override;
};

// Represents the device buffer and context abstraction
class DeviceNode : public MemcpyNode {
private:
    int deviceIdx;
    CUcontext primaryCtx{};
public:
    DeviceNode(size_t bufferSize, int deviceIdx);
    ~DeviceNode();

    int getNodeIdx() const override;
    CUcontext getPrimaryCtx() const override;
};

// Abstraction of a memcpy operation
class MemcpyOperation {
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

    // Allocate the bandwidth values matrix
    void allocateBandwidthMatrix(bool hostVector = false);
    // Because of the parallel nature of copy kernel, the size of data passed is different from cuMemcpyAsync
    size_t smCopySize() const;
    // The main memcpy abstraction, it calls ceFunc/smFunc
    unsigned long long _memcpy(MemcpyNode* src, MemcpyNode* dst, bool skip = false);
public:

    MemcpyOperation(MemcpyCEFunc memcpyFunc, size_t copySize, unsigned long long loopCount);
    MemcpyOperation(MemcpySMFunc memcpyFunc, size_t copySize, unsigned long long loopCount);

    ~MemcpyOperation();

    // Copy direction is determined by node type
    // lists of paired nodes will be executed sumultaneously
    void doMemcpy(std::vector<MemcpyNode*> srcNodes, std::vector<MemcpyNode*> dstNodes);
    void doMemcpy(MemcpyNode* srcNode, MemcpyNode* dstNode);

    void printBenchmarkMatrix(bool reverse = false);
};

#endif
