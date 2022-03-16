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
    size_t bufferSize;
public:
    MemcpyNode(size_t bufferSize);
    virtual ~MemcpyNode() {}
    CUdeviceptr getBuffer() const;
    size_t getBufferSize() const;

    virtual int getNodeIdx() const = 0;
    virtual CUcontext getPrimaryCtx() const = 0;
    virtual std::string getNodeString() const = 0;
};

// Represents the host buffer abstraction
class HostNode : public MemcpyNode {
public:
    // NUMA affinity is set here through allocation of memory in the socket group where `targetDeviceId` resides
    HostNode(size_t bufferSize, int targetDeviceId);
    ~HostNode();

    int getNodeIdx() const override;
    CUcontext getPrimaryCtx() const override;
    virtual std::string getNodeString() const override;
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
    virtual std::string getNodeString() const override;

    bool enablePeerAcess(const DeviceNode &peerNode);
};

// Abstraction of a memcpy operation
class MemcpyOperation {
public:
    // Specifies the preferred node's context to do the operation from
    // It's only a preference because if the preferred node is a HostNode, it has no context and will fall back to the other node
    enum ContextPreference { 
            PREFER_SRC_CONTEXT,    // Prefer the source Node's context if available
            PREFER_DST_CONTEXT     // Prefer the destination Node's context if available
    };

    // Specifies which bandwidths to use for the final result of simultaneous copies
    enum BandwidthValue { 
            USE_FIRST_BW,      // Use the bandwidth of the first copy in the simultaneous copy list
            SUM_BW             // Use the sum of all bandwidths from the simultaneous copy list
    };

private:
    unsigned long long loopCount;

protected:
    size_t *procMask;
    ContextPreference ctxPreference;
    BandwidthValue bandwidthValue;

    // Pure virtual function for implementation of the actual memcpy function
    // return actual bytes copied
    // This can vary from copySize due to SM copies truncated the copy to achieve max bandwidth
    virtual size_t memcpyFunc(CUdeviceptr dst, CUdeviceptr src, CUstream stream, size_t copySize, unsigned long long loopCount) = 0;
public:
    MemcpyOperation(unsigned long long loopCount, ContextPreference ctxPreference = ContextPreference::PREFER_SRC_CONTEXT, BandwidthValue bandwidthValue = BandwidthValue::USE_FIRST_BW);
    virtual ~MemcpyOperation();

    // Lists of paired nodes will be executed sumultaneously
    // context of srcNodes is preferred (if not host) unless otherwise specified
    double doMemcpy(const std::vector<const MemcpyNode*> &srcNodes, const std::vector<const MemcpyNode*> &dstNodes);
    double doMemcpy(const MemcpyNode &srcNode, const MemcpyNode &dstNode);
};

class MemcpyOperationSM : public MemcpyOperation {
private:
    size_t memcpyFunc(CUdeviceptr dst, CUdeviceptr src, CUstream stream, size_t copySize, unsigned long long loopCount);
public:
    MemcpyOperationSM(unsigned long long loopCount, ContextPreference ctxPreference = ContextPreference::PREFER_SRC_CONTEXT, BandwidthValue bandwidthValue = BandwidthValue::SUM_BW);
};

class MemcpyOperationCE : public MemcpyOperation {
private:
    size_t memcpyFunc(CUdeviceptr dst, CUdeviceptr src, CUstream stream, size_t copySize, unsigned long long loopCount);
public:
    MemcpyOperationCE(unsigned long long loopCount, ContextPreference ctxPreference = ContextPreference::PREFER_SRC_CONTEXT, BandwidthValue bandwidthValue = BandwidthValue::USE_FIRST_BW);
};

#endif
