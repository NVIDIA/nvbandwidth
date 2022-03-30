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

#ifndef BENCHMARK_H
#define BENCHMARK_H

#include "common.h"
#include "memcpy.h"

class Testcase {
protected:
    std::string key;
    std::string desc;

    static bool filterHasAccessiblePeerPairs();

    // helper functions
    void allToOneHelper(unsigned long long size, MemcpyOperation &memcpyInstance, PeerValueMatrix<double> &bandwidthValues, bool isRead);
    void oneToAllHelper(unsigned long long size, MemcpyOperation &memcpyInstance, PeerValueMatrix<double> &bandwidthValues, bool isRead);

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

/************
 * Host to device CE memcpy using cuMemcpyAsync
 ************/
class HostToDeviceCE: public Testcase {
public:
    HostToDeviceCE() : Testcase("host_to_device_memcpy_ce", "Host to device memcpy using the Copy Engine") {}
    virtual ~HostToDeviceCE() {}
    void run(unsigned long long size, unsigned long long loopCount);
};

/************
 * Device to host CE memcpy using cuMemcpyAsync
 ************/
class DeviceToHostCE: public Testcase {
public:
    DeviceToHostCE() : Testcase("device_to_host_memcpy_ce", "Device to host memcpy using the Copy Engine") {}
    virtual ~DeviceToHostCE() {}
    void run(unsigned long long size, unsigned long long loopCount);
};

/************
 * Host to device bidirectional CE memcpy using cuMemcpyAsync
 * 
 * A host to device copy is measured while a device to host copy is run simultaneously.
 * Only the host to device copy bandwidth is reported.
 ************/
class HostToDeviceBidirCE: public Testcase {
public:
    HostToDeviceBidirCE() : Testcase("host_to_device_bidirectional_memcpy_ce", "Bidirectional host to device memcpy using the Copy Engine") {}
    virtual ~HostToDeviceBidirCE() {}
    void run(unsigned long long size, unsigned long long loopCount);
};

/************
 * Device to host bidirectional CE memcpy using cuMemcpyAsync
 * 
 * A device to host copy is measured while a host to device copy is run simultaneously.
 * Only the device to host copy bandwidth is reported.
 ************/
class DeviceToHostBidirCE: public Testcase {
public:
    DeviceToHostBidirCE() : Testcase("device_to_host_bidirectional_memcpy_ce", "Bidirectional device to host memcpy using the Copy Engine") {}
    virtual ~DeviceToHostBidirCE() {}
    void run(unsigned long long size, unsigned long long loopCount);
};

/************
 * Device to Device CE Read memcpy using cuMemcpyAsync
 * 
 * Measures bandwidth of cuMemcpyAsync between each pair of accessible peers.
 * Read tests launch a copy from the peer device to the target using the target's context.
 ************/
class DeviceToDeviceReadCE: public Testcase {
public:
    DeviceToDeviceReadCE() : Testcase("device_to_device_memcpy_read_ce", "Device to device memcpy using the Copy Engine (read)") {}
    virtual ~DeviceToDeviceReadCE() {}
    void run(unsigned long long size, unsigned long long loopCount);
    bool filter() { return Testcase::filterHasAccessiblePeerPairs(); }
};

/************
 * Device to Device CE Write memcpy using cuMemcpyAsync
 * 
 * Measures bandwidth of cuMemcpyAsync between each pair of accessible peers.
 * Write tests launch a copy from the source device to the peer using the source's context.
 ************/
class DeviceToDeviceWriteCE: public Testcase {
public:
    DeviceToDeviceWriteCE() : Testcase("device_to_device_memcpy_write_ce", "Device to device memcpy using the Copy Engine (write)") {}
    virtual ~DeviceToDeviceWriteCE() {}
    void run(unsigned long long size, unsigned long long loopCount);
    bool filter() { return Testcase::filterHasAccessiblePeerPairs(); }
};

/************
 * Device to Device Bidirectional CE memcpy using cuMemcpyAsync
 * 
 * Measures bandwidth of cuMemcpyAsync between each pair of accessible peers.
 * A copy in the opposite direction of the measured copy is run simultaneously but not measured.
 ************/
class DeviceToDeviceBidirCE: public Testcase {
public:
    DeviceToDeviceBidirCE() : Testcase("device_to_device_bidirectional_memcpy_ce", "Bidirectional device to device memcpy using the Copy Engine") {}
    virtual ~DeviceToDeviceBidirCE() {}
    void run(unsigned long long size, unsigned long long loopCount);
    bool filter() { return Testcase::filterHasAccessiblePeerPairs(); }
};

/************
 * All to Host CE memcpy using cuMemcpyAsync
 * 
 * Measures bandwidth of cuMemcpyAsync between a single device and the host while simultaneously
 * running copies from all other devices to the host. 
 ************/
class AllToHostCE: public Testcase {
public:
    AllToHostCE() : Testcase("all_to_host_memcpy_ce", "All devices to host memcpy using the Copy Engine") {}
    virtual ~AllToHostCE() {}
    void run(unsigned long long size, unsigned long long loopCount);
};

/************
 * Host to All CE memcpy using cuMemcpyAsync
 * 
 * Measures bandwidth of cuMemcpyAsync between the host to a single device while simultaneously
 * running copies from the host to all other devices. 
 ************/
class HostToAllCE: public Testcase {
public:
    HostToAllCE() : Testcase("host_to_all_memcpy_ce", "Host to all devices memcpy using the Copy Engine") {}
    virtual ~HostToAllCE() {}
    void run(unsigned long long size, unsigned long long loopCount);
};

/************
 * All to One CE memcpy using cuMemcpyAsync
 * 
 * Measures the total bandwidth of copies from all accessible peers to a single device, for each
 * device. Bandwidth is reported as the sum of all inbound bandwidths for each device.
 ************/
class AllToOneCE: public Testcase {
public:
    AllToOneCE() : Testcase("all_to_one_ce", "All devices to one device memcpy using the Copy Engine") {}
    void run(unsigned long long size, unsigned long long loopCount);
    bool filter() { return Testcase::filterHasAccessiblePeerPairs(); }
};

/************
 * One to All CE memcpy using cuMemcpyAsync
 * 
 * Measures the total bandwidth of copies from a single device to all accessible peers, for each
 * device. Bandwidth is reported as the sum of all outbound bandwidths for each device.
 ************/
class OneToAllCE: public Testcase {
public:
    OneToAllCE() : Testcase("one_to_all_ce", "One device to all devices memcpy using the Copy Engine") {}
    void run(unsigned long long size, unsigned long long loopCount);
    bool filter() { return Testcase::filterHasAccessiblePeerPairs(); }
};

// SM Testcase classes

/************
 * Host to device SM memcpy using a copy kernel
 ************/
class HostToDeviceSM: public Testcase {
public:
    HostToDeviceSM() : Testcase("host_to_device_memcpy_sm", "Host to device memcpy using the Stream Multiprocessor") {}
    virtual ~HostToDeviceSM() {}
    void run(unsigned long long size, unsigned long long loopCount);
};

/************
 * Device to host SM memcpy using a copy kernel
 ************/
class DeviceToHostSM: public Testcase {
public:
    DeviceToHostSM() : Testcase("device_to_host_memcpy_sm", "Device to host memcpy using the Stream Multiprocessor") {}
    virtual ~DeviceToHostSM() {}
    void run(unsigned long long size, unsigned long long loopCount);
};

/************
 * Device to Device SM Read memcpy using a copy kernel
 * 
 * Measures bandwidth of a copy kernel between each pair of accessible peers.
 * Read tests launch a copy from the peer device to the target using the target's context.
 ************/
class DeviceToDeviceReadSM: public Testcase {
public:
    DeviceToDeviceReadSM() : Testcase("device_to_device_memcpy_read_sm", "Device to device memcpy using the Stream Multiprocessor (read)") {}
    virtual ~DeviceToDeviceReadSM() {}
    void run(unsigned long long size, unsigned long long loopCount);
    bool filter() { return Testcase::filterHasAccessiblePeerPairs(); }
};

/************
 * Device to Device SM Write memcpy using a copy kernel
 * 
 * Measures bandwidth of a copy kernel between each pair of accessible peers.
 * Write tests launch a copy from the source device to the peer using the source's context.
 ************/
class DeviceToDeviceWriteSM: public Testcase {
public:
    DeviceToDeviceWriteSM() : Testcase("device_to_device_memcpy_write_sm", "Device to device memcpy using the Stream Multiprocessor (write)") {}
    virtual ~DeviceToDeviceWriteSM() {}
    void run(unsigned long long size, unsigned long long loopCount);
    bool filter() { return Testcase::filterHasAccessiblePeerPairs(); }
};

/************
 * Device to Device bidirectional SM Read memcpy using a copy kernel
 * 
 * Measures bandwidth of a copy kernel between each pair of accessible peers. Copies are run
 * in both directions between each pair, and the sum is reported.
 * Read tests launch a copy from the peer device to the target using the target's context.
 ************/
class DeviceToDeviceBidirReadSM: public Testcase {
public:
    DeviceToDeviceBidirReadSM() : Testcase("device_to_device_bidirectional_memcpy_read_sm", "Bidirectional device to device memcpy using the Stream Multiprocessor (read)") {}
    virtual ~DeviceToDeviceBidirReadSM() {}
    void run(unsigned long long size, unsigned long long loopCount);
    bool filter() { return Testcase::filterHasAccessiblePeerPairs(); }
};

/************
 * Device to Device bidirectional SM Write memcpy using a copy kernel
 * 
 * Measures bandwidth of a copy kernel between each pair of accessible peers. Copies are run
 * in both directions between each pair, and the sum is reported.
 * Write tests launch a copy from the source device to the peer using the source's context.
 ************/
class DeviceToDeviceBidirWriteSM: public Testcase {
public:
    DeviceToDeviceBidirWriteSM() : Testcase("device_to_device_bidirectional_memcpy_write_sm", "Bidirectional device to device memcpy using the Stream Multiprocessor (write)") {}
    virtual ~DeviceToDeviceBidirWriteSM() {}
    void run(unsigned long long size, unsigned long long loopCount);
    bool filter() { return Testcase::filterHasAccessiblePeerPairs(); }
};

/************
 * All to One SM Write memcpy using a copy kernel
 * 
 * Measures the total bandwidth of copies from all accessible peers to a single device, for each
 * device. Bandwidth is reported as the sum of all inbound bandwidths for each device.
 * Write tests launch a copy from the source device to the peer using the source's context.
 ************/
class AllToOneWriteSM: public Testcase {
public:
    AllToOneWriteSM() : Testcase("all_to_one_write_sm", "All devices to one device memcpy using the Streaming Multiprocessor (write)") {}
    void run(unsigned long long size, unsigned long long loopCount);
    bool filter() { return Testcase::filterHasAccessiblePeerPairs(); }
};

/************
 * All to One SM Read memcpy using a copy kernel
 * 
 * Measures the total bandwidth of copies from all accessible peers to a single device, for each
 * device. Bandwidth is reported as the sum of all inbound bandwidths for each device.
 * Read tests launch a copy from the peer device to the target using the target's context.
 ************/
class AllToOneReadSM: public Testcase {
public:
    AllToOneReadSM() : Testcase("all_to_one_read_sm", "All devices to one device memcpy using the Streaming Multiprocessor (read)") {}
    void run(unsigned long long size, unsigned long long loopCount);
    bool filter() { return Testcase::filterHasAccessiblePeerPairs(); }
};

/************
 * One to All SM Write memcpy using a copy kernel
 * 
 * Measures the total bandwidth of copies from a single device to all accessible peers, for each
 * device. Bandwidth is reported as the sum of all outbound bandwidths for each device.
 * Write tests launch a copy from the source device to the peer using the source's context.
 ************/
class OneToAllWriteSM: public Testcase {
public:
    OneToAllWriteSM() : Testcase("one_to_all_write_sm", "One device to all devices memcpy using the Streaming Multiprocessor (write)") {}
    void run(unsigned long long size, unsigned long long loopCount);
    bool filter() { return Testcase::filterHasAccessiblePeerPairs(); }
};

/************
 * One to All SM Read memcpy using a copy kernel
 * 
 * Measures the total bandwidth of copies from a single device to all accessible peers, for each
 * device. Bandwidth is reported as the sum of all outbound bandwidths for each device.
 * Read tests launch a copy from the peer device to the target using the target's context.
 ************/
class OneToAllReadSM: public Testcase {
public:
    OneToAllReadSM() : Testcase("one_to_all_read_sm", "One device to all devices memcpy using the Streaming Multiprocessor (read)") {}
    void run(unsigned long long size, unsigned long long loopCount);
    bool filter() { return Testcase::filterHasAccessiblePeerPairs(); }
};

#endif
