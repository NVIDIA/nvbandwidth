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

#ifndef TESTCASE_H_
#define TESTCASE_H_


#include "common.h"
#include "inline_common.h"
#include "memcpy.h"

class Testcase {
 protected:
    std::string key;
    std::string desc;

    static bool filterHasAccessiblePeerPairs();
    static bool filterSupportsMulticast();
#ifdef MULTINODE
    static bool filterHasMultipleGPUsMultinode();
#endif

    // helper functions
    void allToOneHelper(unsigned long long size, MemcpyOperation &memcpyInstance, PeerValueMatrix<double> &bandwidthValues, bool isRead);
    void oneToAllHelper(unsigned long long size, MemcpyOperation &memcpyInstance, PeerValueMatrix<double> &bandwidthValues, bool isRead);
    void allHostHelper(unsigned long long size, MemcpyOperation &memcpyInstance, PeerValueMatrix<double> &bandwidthValues, bool sourceIsHost);
    void allHostBidirHelper(unsigned long long size, MemcpyOperation &memcpyInstance, PeerValueMatrix<double> &bandwidthValues, bool sourceIsHost);
    void latencyHelper(const MemcpyBuffer &dataBuffer, bool measureDeviceToDeviceLatency);

 public:
    Testcase(std::string key, std::string desc);
    virtual ~Testcase() {}

    std::string testKey();
    std::string testDesc();

    // Returns true if the testcase can be run on the current system
    virtual bool filter() { return true; }

    // Runs the testcase
    virtual void run(unsigned long long size, unsigned long long loopCount) = 0;
};


// CE Testcase classes

// Host to device CE memcpy using cuMemcpyAsync
class HostToDeviceCE: public Testcase {
 public:
    HostToDeviceCE() : Testcase("host_to_device_memcpy_ce",
            "\tHost to device CE memcpy using cuMemcpyAsync") {}
    virtual ~HostToDeviceCE() {}
    void run(unsigned long long size, unsigned long long loopCount);
};

// Device to host CE memcpy using cuMemcpyAsync
class DeviceToHostCE: public Testcase {
 public:
    DeviceToHostCE() : Testcase("device_to_host_memcpy_ce",
            "\tDevice to host CE memcpy using cuMemcpyAsync") {}
    virtual ~DeviceToHostCE() {}
    void run(unsigned long long size, unsigned long long loopCount);
};

// Host to device bidirectional CE memcpy using cuMemcpyAsync
class HostToDeviceBidirCE: public Testcase {
 public:
    HostToDeviceBidirCE() : Testcase("host_to_device_bidirectional_memcpy_ce",
            "\tA host to device copy is measured while a device to host copy is run simultaneously.\n"
            "\tOnly the host to device copy bandwidth is reported.") {}
    virtual ~HostToDeviceBidirCE() {}
    void run(unsigned long long size, unsigned long long loopCount);
};

// Device to host bidirectional CE memcpy using cuMemcpyAsync
class DeviceToHostBidirCE: public Testcase {
 public:
    DeviceToHostBidirCE() : Testcase("device_to_host_bidirectional_memcpy_ce",
            "\tA device to host copy is measured while a host to device copy is run simultaneously.\n"
            "\tOnly the device to host copy bandwidth is reported.") {}
    virtual ~DeviceToHostBidirCE() {}
    void run(unsigned long long size, unsigned long long loopCount);
};

// Host to device bidirectional SM memcpy
class HostToDeviceBidirSM: public Testcase {
 public:
    HostToDeviceBidirSM() : Testcase("host_to_device_bidirectional_memcpy_sm",
            "\tA host to device copy is measured while a device to host copy is run simultaneously.\n"
            "\tOnly the host to device copy bandwidth is reported.") {}
    virtual ~HostToDeviceBidirSM() {}
    void run(unsigned long long size, unsigned long long loopCount);
};

// Device to host bidirectional SM memcpy
class DeviceToHostBidirSM: public Testcase {
 public:
    DeviceToHostBidirSM() : Testcase("device_to_host_bidirectional_memcpy_sm",
            "\tA device to host copy is measured while a host to device copy is run simultaneously.\n"
            "\tOnly the device to host copy bandwidth is reported.") {}
    virtual ~DeviceToHostBidirSM() {}
    void run(unsigned long long size, unsigned long long loopCount);
};

// Device to Device CE Read memcpy using cuMemcpyAsync
class DeviceToDeviceReadCE: public Testcase {
 public:
    DeviceToDeviceReadCE() : Testcase("device_to_device_memcpy_read_ce",
            "\tMeasures bandwidth of cuMemcpyAsync between each pair of accessible peers.\n"
            "\tRead tests launch a copy from the peer device to the target using the target's context.") {}
    virtual ~DeviceToDeviceReadCE() {}
    void run(unsigned long long size, unsigned long long loopCount);
    bool filter() { return Testcase::filterHasAccessiblePeerPairs(); }
};

// Device to Device CE Write memcpy using cuMemcpyAsync
class DeviceToDeviceWriteCE: public Testcase {
 public:
    DeviceToDeviceWriteCE() : Testcase("device_to_device_memcpy_write_ce",
            "\tMeasures bandwidth of cuMemcpyAsync between each pair of accessible peers.\n"
            "\tWrite tests launch a copy from the target device to the peer using the target's context.") {}
    virtual ~DeviceToDeviceWriteCE() {}
    void run(unsigned long long size, unsigned long long loopCount);
    bool filter() { return Testcase::filterHasAccessiblePeerPairs(); }
};

// Device to Device Bidirectional Read CE memcpy using cuMemcpyAsync
class DeviceToDeviceBidirReadCE: public Testcase {
 public:
    DeviceToDeviceBidirReadCE() : Testcase("device_to_device_bidirectional_memcpy_read_ce",
            "\tMeasures bandwidth of cuMemcpyAsync between each pair of accessible peers.\n"
            "\tA copy in the opposite direction of the measured copy is run simultaneously but not measured.\n"
            "\tRead tests launch a copy from the peer device to the target using the target's context.") {}
    virtual ~DeviceToDeviceBidirReadCE() {}
    void run(unsigned long long size, unsigned long long loopCount);
    bool filter() { return Testcase::filterHasAccessiblePeerPairs(); }
};

// Device to Device Bidirectional Write CE memcpy using cuMemcpyAsync
class DeviceToDeviceBidirWriteCE: public Testcase {
 public:
    DeviceToDeviceBidirWriteCE() : Testcase("device_to_device_bidirectional_memcpy_write_ce",
            "\tMeasures bandwidth of cuMemcpyAsync between each pair of accessible peers.\n"
            "\tA copy in the opposite direction of the measured copy is run simultaneously but not measured.\n"
            "\tWrite tests launch a copy from the target device to the peer using the target's context.") {}
    virtual ~DeviceToDeviceBidirWriteCE() {}
    void run(unsigned long long size, unsigned long long loopCount);
    bool filter() { return Testcase::filterHasAccessiblePeerPairs(); }
};

// Device local memcpy using cuMemcpyAsync
class DeviceLocalCopy: public Testcase {
 public:
    DeviceLocalCopy() : Testcase("device_local_copy",
            "\tMeasures bandwidth of cuMemcpyAsync between device buffers local to the GPU.\n") {}
    virtual ~DeviceLocalCopy() {}
    void run(unsigned long long size, unsigned long long loopCount);
};

// All to Host CE memcpy using cuMemcpyAsync
class AllToHostCE: public Testcase {
 public:
    AllToHostCE() : Testcase("all_to_host_memcpy_ce",
            "\tMeasures bandwidth of cuMemcpyAsync between a single device and the host while simultaneously\n"
            "\trunning copies from all other devices to the host.") {}
    virtual ~AllToHostCE() {}
    void run(unsigned long long size, unsigned long long loopCount);
};

// All to Host bidirectional CE memcpy using cuMemcpyAsync
class AllToHostBidirCE: public Testcase {
 public:
    AllToHostBidirCE() : Testcase("all_to_host_bidirectional_memcpy_ce",
            "\tA device to host copy is measured while a host to device copy is run simultaneously.\n"
            "\tOnly the device to host copy bandwidth is reported.\n"
            "\tAll other devices generate simultaneous host to device and device to host interferring traffic.") {}
    virtual ~AllToHostBidirCE() {}
    void run(unsigned long long size, unsigned long long loopCount);
};

// Host to All CE memcpy using cuMemcpyAsync
class HostToAllCE: public Testcase {
 public:
    HostToAllCE() : Testcase("host_to_all_memcpy_ce",
            "\tMeasures bandwidth of cuMemcpyAsync between the host to a single device while simultaneously\n"
            "\trunning copies from the host to all other devices.") {}
    virtual ~HostToAllCE() {}
    void run(unsigned long long size, unsigned long long loopCount);
};

// Host to All bidirectional CE memcpy using cuMemcpyAsync
class HostToAllBidirCE: public Testcase {
 public:
    HostToAllBidirCE() : Testcase("host_to_all_bidirectional_memcpy_ce",
            "\tA host to device copy is measured while a device to host copy is run simultaneously.\n"
            "\tOnly the host to device copy bandwidth is reported.\n"
            "\tAll other devices generate simultaneous host to device and device to host interferring traffic.") {}
    virtual ~HostToAllBidirCE() {}
    void run(unsigned long long size, unsigned long long loopCount);
};


// All to One CE Write memcpy using cuMemcpyAsync
class AllToOneWriteCE: public Testcase {
 public:
    AllToOneWriteCE() : Testcase("all_to_one_write_ce",
            "\tMeasures the total bandwidth of copies from all accessible peers to a single device, for each\n"
            "\tdevice. Bandwidth is reported as the total inbound bandwidth for each device.\n"
            "\tWrite tests launch a copy from the target device to the peer using the target's context.") {}
    virtual ~AllToOneWriteCE() {}
    void run(unsigned long long size, unsigned long long loopCount);
    bool filter() { return Testcase::filterHasAccessiblePeerPairs(); }
};

// All to One CE Read memcpy using cuMemcpyAsync
class AllToOneReadCE: public Testcase {
 public:
    AllToOneReadCE() : Testcase("all_to_one_read_ce",
            "\tMeasures the total bandwidth of copies from all accessible peers to a single device, for each\n"
            "\tdevice. Bandwidth is reported as the total outbound bandwidth for each device.\n"
            "\tRead tests launch a copy from the peer device to the target using the target's context.") {}
    virtual ~AllToOneReadCE() {}
    void run(unsigned long long size, unsigned long long loopCount);
    bool filter() { return Testcase::filterHasAccessiblePeerPairs(); }
};

// One to All CE Write memcpy using cuMemcpyAsync
class OneToAllWriteCE: public Testcase {
 public:
    OneToAllWriteCE() : Testcase("one_to_all_write_ce",
            "\tMeasures the total bandwidth of copies from a single device to all accessible peers, for each\n"
            "\tdevice. Bandwidth is reported as the total outbound bandwidth for each device.\n"
            "\tWrite tests launch a copy from the target device to the peer using the target's context.") {}
    virtual ~OneToAllWriteCE() {}
    void run(unsigned long long size, unsigned long long loopCount);
    bool filter() { return Testcase::filterHasAccessiblePeerPairs(); }
};

// One to All CE Read memcpy using cuMemcpyAsync
class OneToAllReadCE: public Testcase {
 public:
    OneToAllReadCE() : Testcase("one_to_all_read_ce",
            "\tMeasures the total bandwidth of copies from a single device to all accessible peers, for each\n"
            "\tdevice. Bandwidth is reported as the total inbound bandwidth for each device.\n"
            "\tRead tests launch a copy from the peer device to the target using the target's context.") {}
    virtual ~OneToAllReadCE() {}
    void run(unsigned long long size, unsigned long long loopCount);
    bool filter() { return Testcase::filterHasAccessiblePeerPairs(); }
};

// SM Testcase classes
// Host to device SM latency using a ptr chase kernel
class HostDeviceLatencySM: public Testcase {
 public:
    HostDeviceLatencySM() : Testcase("host_device_latency_sm",
            "\tHost - device access latency using a pointer chase kernel\n"
            "\tA 2MB buffer is allocated on the host and is accessed by the GPU") {}
    virtual ~HostDeviceLatencySM() {}
    void run(unsigned long long size, unsigned long long loopCount);
};

// Host to device SM memcpy using a copy kernel
class HostToDeviceSM: public Testcase {
 public:
    HostToDeviceSM() : Testcase("host_to_device_memcpy_sm",
            "\tHost to device SM memcpy using a copy kernel") {}
    virtual ~HostToDeviceSM() {}
    void run(unsigned long long size, unsigned long long loopCount);
};

// Device to host SM memcpy using a copy kernel
class DeviceToHostSM: public Testcase {
 public:
    DeviceToHostSM() : Testcase("device_to_host_memcpy_sm",
            "\tDevice to host SM memcpy using a copy kernel") {}
    virtual ~DeviceToHostSM() {}
    void run(unsigned long long size, unsigned long long loopCount);
};

// Device to Device SM Read memcpy using a copy kernel
class DeviceToDeviceReadSM: public Testcase {
 public:
    DeviceToDeviceReadSM() : Testcase("device_to_device_memcpy_read_sm",
            "\tMeasures bandwidth of a copy kernel between each pair of accessible peers.\n"
            "\tRead tests launch a copy from the peer device to the target using the target's context.") {}
    virtual ~DeviceToDeviceReadSM() {}
    void run(unsigned long long size, unsigned long long loopCount);
    bool filter() { return Testcase::filterHasAccessiblePeerPairs(); }
};

// Device to Device SM Latency ptr chase kernel
class DeviceToDeviceLatencySM: public Testcase {
 public:
    DeviceToDeviceLatencySM() : Testcase("device_to_device_latency_sm",
            "\tMeasures latency of a pointer derefernce operation between each pair of accessible peers.\n"
            "\tA 2MB buffer is allocated on a GPU and is accessed by the peer GPU to determine latency.\n"
            "\t--bufferSize flag is ignored") {}
    virtual ~DeviceToDeviceLatencySM() {}
    void run(unsigned long long size, unsigned long long loopCount);
    bool filter() { return Testcase::filterHasAccessiblePeerPairs(); }
};

// Device to Device SM Write memcpy using a copy kernel
class DeviceToDeviceWriteSM: public Testcase {
 public:
    DeviceToDeviceWriteSM() : Testcase("device_to_device_memcpy_write_sm",
            "\tMeasures bandwidth of a copy kernel between each pair of accessible peers.\n"
            "\tWrite tests launch a copy from the target device to the peer using the target's context.") {}
    virtual ~DeviceToDeviceWriteSM() {}
    void run(unsigned long long size, unsigned long long loopCount);
    bool filter() { return Testcase::filterHasAccessiblePeerPairs(); }
};

// Device to Device bidirectional SM Read memcpy using a copy kernel
class DeviceToDeviceBidirReadSM: public Testcase {
 public:
    DeviceToDeviceBidirReadSM() : Testcase("device_to_device_bidirectional_memcpy_read_sm",
            "\tMeasures bandwidth of a copy kernel between each pair of accessible peers. Copies are run\n"
            "\tin both directions between each pair, and the sum is reported.\n"
            "\tRead tests launch a copy from the peer device to the target using the target's context.") {}
    virtual ~DeviceToDeviceBidirReadSM() {}
    void run(unsigned long long size, unsigned long long loopCount);
    bool filter() { return Testcase::filterHasAccessiblePeerPairs(); }
};

// Device to Device bidirectional SM Write memcpy using a copy kernel
class DeviceToDeviceBidirWriteSM: public Testcase {
 public:
    DeviceToDeviceBidirWriteSM() : Testcase("device_to_device_bidirectional_memcpy_write_sm",
            "\tMeasures bandwidth of a copy kernel between each pair of accessible peers. Copies are run\n"
            "\tin both directions between each pair, and the sum is reported.\n"
            "\tWrite tests launch a copy from the target device to the peer using the target's context.") {}
    virtual ~DeviceToDeviceBidirWriteSM() {}
    void run(unsigned long long size, unsigned long long loopCount);
    bool filter() { return Testcase::filterHasAccessiblePeerPairs(); }
};

// All to Host SM memcpy using a copy kernel
class AllToHostSM: public Testcase {
 public:
    AllToHostSM() : Testcase("all_to_host_memcpy_sm",
            "\tMeasures bandwidth of a copy kernel between a single device and the host while simultaneously\n"
            "\trunning copies from all other devices to the host.") {}
    virtual ~AllToHostSM() {}
    void run(unsigned long long size, unsigned long long loopCount);
};

// All to Host bidirectional SM memcpy using a copy kernel
class AllToHostBidirSM: public Testcase {
 public:
    AllToHostBidirSM() : Testcase("all_to_host_bidirectional_memcpy_sm",
            "\tA device to host bandwidth of a copy kernel is measured while a host to device copy is run simultaneously.\n"
            "\tOnly the device to host copy bandwidth is reported.\n"
            "\tAll other devices generate simultaneous host to device and device to host interferring traffic using copy kernels.") {}
    virtual ~AllToHostBidirSM() {}
    void run(unsigned long long size, unsigned long long loopCount);
};

// Host to All SM memcpy using a copy kernel
class HostToAllSM: public Testcase {
 public:
    HostToAllSM() : Testcase("host_to_all_memcpy_sm",
            "\tMeasures bandwidth of a copy kernel between the host to a single device while simultaneously\n"
            "\trunning copies from the host to all other devices.") {}
    virtual ~HostToAllSM() {}
    void run(unsigned long long size, unsigned long long loopCount);
};

// Host to All bidirectional SM memcpy using a copy kernel
class HostToAllBidirSM: public Testcase {
 public:
    HostToAllBidirSM() : Testcase("host_to_all_bidirectional_memcpy_sm",
            "\tA host to device bandwidth of a copy kernel is measured while a device to host copy is run simultaneously.\n"
            "\tOnly the host to device copy bandwidth is reported.\n"
            "\tAll other devices generate simultaneous host to device and device to host interferring traffic using copy kernels.") {}
    virtual ~HostToAllBidirSM() {}
    void run(unsigned long long size, unsigned long long loopCount);
};

// All to One SM Write memcpy using a copy kernel
class AllToOneWriteSM: public Testcase {
 public:
    AllToOneWriteSM() : Testcase("all_to_one_write_sm",
            "\tMeasures the total bandwidth of copies from all accessible peers to a single device, for each\n"
            "\tdevice. Bandwidth is reported as the total inbound bandwidth for each device.\n"
            "\tWrite tests launch a copy from the target device to the peer using the target's context.") {}
    virtual ~AllToOneWriteSM() {}
    void run(unsigned long long size, unsigned long long loopCount);
    bool filter() { return Testcase::filterHasAccessiblePeerPairs(); }
};

// All to One SM Read memcpy using a copy kernel
class AllToOneReadSM: public Testcase {
 public:
    AllToOneReadSM() : Testcase("all_to_one_read_sm",
            "\tMeasures the total bandwidth of copies from all accessible peers to a single device, for each\n"
            "\tdevice. Bandwidth is reported as the total outbound bandwidth for each device.\n"
            "\tRead tests launch a copy from the peer device to the target using the target's context.") {}
    virtual ~AllToOneReadSM() {}
    void run(unsigned long long size, unsigned long long loopCount);
    bool filter() { return Testcase::filterHasAccessiblePeerPairs(); }
};

// One to All SM Write memcpy using a copy kernel
class OneToAllWriteSM: public Testcase {
 public:
    OneToAllWriteSM() : Testcase("one_to_all_write_sm",
            "\tMeasures the total bandwidth of copies from a single device to all accessible peers, for each\n"
            "\tdevice. Bandwidth is reported as the total outbound bandwidth for each device.\n"
            "\tWrite tests launch a copy from the target device to the peer using the target's context.") {}
    virtual ~OneToAllWriteSM() {}
    void run(unsigned long long size, unsigned long long loopCount);
    bool filter() { return Testcase::filterHasAccessiblePeerPairs(); }
};

// One to All SM Read memcpy using a copy kernel
class OneToAllReadSM: public Testcase {
 public:
    OneToAllReadSM() : Testcase("one_to_all_read_sm",
            "\tMeasures the total bandwidth of copies from a single device to all accessible peers, for each\n"
            "\tdevice. Bandwidth is reported as the total inbound bandwidth for each device.\n"
            "\tRead tests launch a copy from the peer device to the target using the target's context.") {}
    virtual ~OneToAllReadSM() {}
    void run(unsigned long long size, unsigned long long loopCount);
    bool filter() { return Testcase::filterHasAccessiblePeerPairs(); }
};

#ifdef MULTINODE
// Device to Device CE Read memcpy using cuMemcpyAsync
class MultinodeDeviceToDeviceReadCE: public Testcase {
 public:
    MultinodeDeviceToDeviceReadCE() : Testcase("multinode_device_to_device_memcpy_read_ce",
            "\tMeasures bandwidth of cuMemcpyAsync between each pair of accessible peers.\n"
            "\tRead tests launch a copy from the peer device to the target using the target's context.") {}
    virtual ~MultinodeDeviceToDeviceReadCE() {}
    void run(unsigned long long size, unsigned long long loopCount);
    bool filter() { return Testcase::filterHasMultipleGPUsMultinode(); }
};

// Device to Device CE Write memcpy using cuMemcpyAsync
class MultinodeDeviceToDeviceWriteCE: public Testcase {
 public:
    MultinodeDeviceToDeviceWriteCE() : Testcase("multinode_device_to_device_memcpy_write_ce",
            "\tMeasures bandwidth of cuMemcpyAsync between each pair of accessible peers.\n"
            "\tWrite tests launch a copy from the target device to the peer using the target's context.") {}
    virtual ~MultinodeDeviceToDeviceWriteCE() {}
    void run(unsigned long long size, unsigned long long loopCount);
    bool filter() { return Testcase::filterHasMultipleGPUsMultinode(); }
};

// Device to Device Bidirectional Read CE memcpy using cuMemcpyAsync
class MultinodeDeviceToDeviceBidirReadCE: public Testcase {
 public:
    MultinodeDeviceToDeviceBidirReadCE() : Testcase("multinode_device_to_device_bidirectional_memcpy_read_ce",
            "\tMeasures bandwidth of cuMemcpyAsync between each pair of accessible peers.\n"
            "\tA copy in the opposite direction of the measured copy is run simultaneously but not measured.\n"
            "\tRead tests launch a copy from the peer device to the target using the target's context.") {}
    virtual ~MultinodeDeviceToDeviceBidirReadCE() {}
    void run(unsigned long long size, unsigned long long loopCount);
    bool filter() { return Testcase::filterHasMultipleGPUsMultinode(); }
};

// Device to Device Bidirectional Write CE memcpy using cuMemcpyAsync
class MultinodeDeviceToDeviceBidirWriteCE: public Testcase {
 public:
    MultinodeDeviceToDeviceBidirWriteCE() : Testcase("multinode_device_to_device_bidirectional_memcpy_write_ce",
            "\tMeasures bandwidth of cuMemcpyAsync between each pair of accessible peers.\n"
            "\tA copy in the opposite direction of the measured copy is run simultaneously but not measured.\n"
            "\tWrite tests launch a copy from the target device to the peer using the target's context.") {}
    virtual ~MultinodeDeviceToDeviceBidirWriteCE() {}
    void run(unsigned long long size, unsigned long long loopCount);
    bool filter() { return Testcase::filterHasMultipleGPUsMultinode(); }
};

// Device to Device SM Read memcpy using a copy kernel
class MultinodeDeviceToDeviceReadSM: public Testcase {
 public:
    MultinodeDeviceToDeviceReadSM() : Testcase("multinode_device_to_device_memcpy_read_sm",
            "\tMeasures bandwidth of a copy kernel between each pair of accessible peers.\n"
            "\tRead tests launch a copy from the peer device to the target using the target's context.") {}
    virtual ~MultinodeDeviceToDeviceReadSM() {}
    void run(unsigned long long size, unsigned long long loopCount);
    bool filter() { return Testcase::filterHasMultipleGPUsMultinode(); }
};

// Device to Device SM Write memcpy using a copy kernel
class MultinodeDeviceToDeviceWriteSM: public Testcase {
 public:
    MultinodeDeviceToDeviceWriteSM() : Testcase("multinode_device_to_device_memcpy_write_sm",
            "\tMeasures bandwidth of a copy kernel between each pair of accessible peers.\n"
            "\tWrite tests launch a copy from the target device to the peer using the target's context.") {}
    virtual ~MultinodeDeviceToDeviceWriteSM() {}
    void run(unsigned long long size, unsigned long long loopCount);
    bool filter() { return Testcase::filterHasMultipleGPUsMultinode(); }
};

// Device to Device bidirectional SM Read memcpy using a copy kernel
class MultinodeDeviceToDeviceBidirReadSM: public Testcase {
 public:
    MultinodeDeviceToDeviceBidirReadSM() : Testcase("multinode_device_to_device_bidirectional_memcpy_read_sm",
            "\tMeasures bandwidth of a copy kernel between each pair of accessible peers. Copies are run\n"
            "\tin both directions between each pair, and the sum is reported.\n"
            "\tRead tests launch a copy from the peer device to the target using the target's context.") {}
    virtual ~MultinodeDeviceToDeviceBidirReadSM() {}
    void run(unsigned long long size, unsigned long long loopCount);
    bool filter() { return Testcase::filterHasMultipleGPUsMultinode(); }
};

// Device to Device bidirectional SM Write memcpy using a copy kernel
class MultinodeDeviceToDeviceBidirWriteSM: public Testcase {
 public:
    MultinodeDeviceToDeviceBidirWriteSM() : Testcase("multinode_device_to_device_bidirectional_memcpy_write_sm",
            "\tMeasures bandwidth of a copy kernel between each pair of accessible peers. Copies are run\n"
            "\tin both directions between each pair, and the sum is reported.\n"
            "\tWrite tests launch a copy from the target device to the peer using the target's context.") {}
    virtual ~MultinodeDeviceToDeviceBidirWriteSM() {}
    void run(unsigned long long size, unsigned long long loopCount);
    bool filter() { return Testcase::filterHasMultipleGPUsMultinode(); }
};

class MultinodeAllToOneWriteSM: public Testcase {
 public:
    MultinodeAllToOneWriteSM() : Testcase("multinode_device_to_device_all_to_one_write_sm",
            "\tMeasures the total bandwidth of copies from all accessible peers to a single device, for each\n"
            "\tdevice. Bandwidth is reported as the total inbound bandwidth for each device.\n"
            "\tWrite tests launch a copy from the peer to the target device using the peer's context.") {}
    virtual ~MultinodeAllToOneWriteSM() {}
    void run(unsigned long long size, unsigned long long loopCount);
    bool filter() { return Testcase::filterHasMultipleGPUsMultinode(); }
};

class MultinodeAllFromOneReadSM: public Testcase {
 public:
    MultinodeAllFromOneReadSM() : Testcase("multinode_device_to_device_all_from_one_read_sm",
            "\tMeasures the total bandwidth of copies from a single device to all accessible peers, for each\n"
            "\tdevice. Bandwidth is reported as the total outbound bandwidth for each device.\n"
            "\tRead tests launch a copy from the target device to the peer using the peer's context.") {}
    virtual ~MultinodeAllFromOneReadSM() {}
    void run(unsigned long long size, unsigned long long loopCount);
    bool filter() { return Testcase::filterHasMultipleGPUsMultinode(); }
};

class MultinodeBroadcastOneToAllSM: public Testcase {
 public:
    MultinodeBroadcastOneToAllSM() : Testcase("multinode_device_to_device_broadcast_one_to_all_sm",
            "\tMeasures bandwidth of a copy kernel copying data from device memory to multicast allocated memory\n"
            "\tthat's mapped on all accessible peers.\n"
            "\tTests launch a copy from the target device to the multicast memory on target using the target's context.") {}
    virtual ~MultinodeBroadcastOneToAllSM() {}
    void run(unsigned long long size, unsigned long long loopCount);
    bool filter() { return Testcase::filterHasMultipleGPUsMultinode() && Testcase::filterSupportsMulticast(); }
};

class MultinodeBroadcastAllToAllSM: public Testcase {
 public:
    MultinodeBroadcastAllToAllSM() : Testcase("multinode_device_to_device_broadcast_all_to_all_sm",
            "\tMeasures bandwidth of a copy kernels copying data from device memory to multicast allocated memory\n"
            "\tthat's mapped on all accessible peers."
            "\tAll devices are doing copies at the same time.\n"
            "\tTests launch copies from the target device to the multicast memory on target using the target's context.") {}
    virtual ~MultinodeBroadcastAllToAllSM() {}
    void run(unsigned long long size, unsigned long long loopCount);
    bool filter() { return Testcase::filterHasMultipleGPUsMultinode() && Testcase::filterSupportsMulticast(); }
};

// TODO(pgumienny) - add remaining combination of Read/Write CE/SM once the tooling is in
class MultinodeBisectWriteCE: public Testcase {
 public:
    MultinodeBisectWriteCE() : Testcase("multinode_bisect_write_ce",
            "\tMeasures bandwidths of simultaneous copies.\n"
            "\tFor N GPU system there will be N copies occuring at the same time\n"
            "\tGPU owned by rank A will be writing to GPU owned by rank (A + N/2) % N).") {}
    virtual ~MultinodeBisectWriteCE() {}
    void run(unsigned long long size, unsigned long long loopCount);
    bool filter() { return Testcase::filterHasMultipleGPUsMultinode(); }
};
#endif

#endif  // TESTCASE_H_
