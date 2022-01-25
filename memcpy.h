/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef MEMCPY_H
#define MEMCPY_H

#include "common.h"

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

    bool enablePeerAcess(const DeviceNode *peerNode);
};

// Abstraction of a memcpy operation
class MemcpyOperation {
private:
    unsigned long long loopCount;

protected:
    size_t copySize;
    size_t *procMask;
    bool preferSrcCtx;
    bool sumResults;

    // Pure virtual function for implementation of the actual memcpy function
    virtual CUresult memcpyFunc(CUdeviceptr dst, CUdeviceptr src, CUstream stream, unsigned long long loopCount) = 0;
public:
    MemcpyOperation(size_t copySize, unsigned long long loopCount, bool preferSrcCtx = true, bool sumResults = false);
    virtual ~MemcpyOperation();

    // Lists of paired nodes will be executed sumultaneously
    // context of srcNodes is preferred (if not host) unless otherwise specified
    double doMemcpy(std::vector<MemcpyNode*> srcNodes, std::vector<MemcpyNode*> dstNodes);
    double doMemcpy(MemcpyNode* srcNode, MemcpyNode* dstNode);
};

class MemcpyOperationSM : public MemcpyOperation {
private:
    CUresult memcpyFunc(CUdeviceptr dst, CUdeviceptr src, CUstream stream, unsigned long long loopCount);
public:
    MemcpyOperationSM(size_t copySize, unsigned long long loopCount, bool preferSrcCtx = true, bool sumResults = false);
};

class MemcpyOperationCE : public MemcpyOperation {
private:
    CUresult memcpyFunc(CUdeviceptr dst, CUdeviceptr src, CUstream stream, unsigned long long loopCount);
public:
    MemcpyOperationCE(size_t copySize, unsigned long long loopCount, bool preferSrcCtx = true, bool sumResults = false);
};

#endif
