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
protected:
    size_t copySize;
    unsigned long long loopCount;
    size_t *procMask;

    PeerValueMatrix<double> *bandwidthValues{nullptr};

    // Allocate the bandwidth values matrix
    void allocateBandwidthMatrix(bool hostVector = false);

    // Pure virtual function for implementation of the actual memcpy function
    virtual CUresult memcpyFunc(CUdeviceptr dst, CUdeviceptr src, CUstream stream) = 0;
public:
    MemcpyOperation(size_t copySize, unsigned long long loopCount);
    virtual ~MemcpyOperation();

    // Copy direction is determined by node type
    // lists of paired nodes will be executed sumultaneously
    void doMemcpy(std::vector<MemcpyNode*> srcNodes, std::vector<MemcpyNode*> dstNodes);
    void doMemcpy(MemcpyNode* srcNode, MemcpyNode* dstNode);

    void printBenchmarkMatrix(bool reverse = false);
};

class MemcpyOperationSM : public MemcpyOperation {
private:
    CUresult memcpyFunc(CUdeviceptr dst, CUdeviceptr src, CUstream stream);
public:
    MemcpyOperationSM(size_t copySize, unsigned long long loopCount);
};

class MemcpyOperationCE : public MemcpyOperation {
private:
    CUresult memcpyFunc(CUdeviceptr dst, CUdeviceptr src, CUstream stream);
public:
    MemcpyOperationCE(size_t copySize, unsigned long long loopCount);
};

#endif
