/*
 * Copyright 1993-2021 by NVIDIA Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef MEMCPY_H
#define MEMCPY_H

#include "common.h"

class MemcpyNode {
protected:
    void* buffer{};
    size_t bufferSize;
public:
    MemcpyNode(size_t bufferSize);
    CUdeviceptr getBuffer();
    size_t getBufferSize();

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

    bool enablePeerAcess(const DeviceNode *peerNode);
};

// Abstraction of a memcpy operation
class MemcpyOperation {
private:
    unsigned long long loopCount;

protected:
    size_t *procMask;
    bool preferSrcCtx;
    bool sumResults;

    // Pure virtual function for implementation of the actual memcpy function
    virtual CUresult memcpyFunc(CUdeviceptr dst, CUdeviceptr src, CUstream stream, size_t copySize, unsigned long long loopCount) = 0;
public:
    MemcpyOperation(unsigned long long loopCount, bool preferSrcCtx = true, bool sumResults = false);
    virtual ~MemcpyOperation();

    // Lists of paired nodes will be executed sumultaneously
    // context of srcNodes is preferred (if not host) unless otherwise specified
    double doMemcpy(std::vector<MemcpyNode*> srcNodes, std::vector<MemcpyNode*> dstNodes);
    double doMemcpy(MemcpyNode* srcNode, MemcpyNode* dstNode);
};

class MemcpyOperationSM : public MemcpyOperation {
private:
    CUresult memcpyFunc(CUdeviceptr dst, CUdeviceptr src, CUstream stream, size_t copySize, unsigned long long loopCount);
public:
    MemcpyOperationSM(unsigned long long loopCount, bool preferSrcCtx = true, bool sumResults = false);
};

class MemcpyOperationCE : public MemcpyOperation {
private:
    CUresult memcpyFunc(CUdeviceptr dst, CUdeviceptr src, CUstream stream, size_t copySize, unsigned long long loopCount);
public:
    MemcpyOperationCE(unsigned long long loopCount, bool preferSrcCtx = true, bool sumResults = false);
};

#endif
