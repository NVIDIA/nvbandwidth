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

class Benchmark {
protected:
    std::string key;
    std::string desc;

    static bool filterHasAccessiblePeerPairs();

public:
    Benchmark(std::string key, std::string desc);
    virtual ~Benchmark() {}

    std::string benchKey();
    std::string benchDesc();

    // Returns true if the benchmark can be run on the current system
    virtual bool filter() { return true; }

    // Runs the benchmark
    virtual void run(unsigned long long size, unsigned long long loopCount) = 0;
};

// CE Benchmark classes
class HostToDeviceCE: public Benchmark {
public:
    HostToDeviceCE() : Benchmark("host_to_device_memcpy_ce", "Host to device memcpy using the Copy Engine") {}
    virtual ~HostToDeviceCE() {}
    void run(unsigned long long size, unsigned long long loopCount);
};

class DeviceToHostCE: public Benchmark {
public:
    DeviceToHostCE() : Benchmark("device_to_host_memcpy_ce", "Device to host memcpy using the Copy Engine") {}
    virtual ~DeviceToHostCE() {}
    void run(unsigned long long size, unsigned long long loopCount);
};

class HostToDeviceBidirCE: public Benchmark {
public:
    HostToDeviceBidirCE() : Benchmark("host_to_device_bidirectional_memcpy_ce", "Bidirectional host to device memcpy using the Copy Engine") {}
    virtual ~HostToDeviceBidirCE() {}
    void run(unsigned long long size, unsigned long long loopCount);
};

class DeviceToHostBidirCE: public Benchmark {
public:
    DeviceToHostBidirCE() : Benchmark("device_to_host_bidirectional_memcpy_ce", "Bidirectional device to host memcpy using the Copy Engine") {}
    virtual ~DeviceToHostBidirCE() {}
    void run(unsigned long long size, unsigned long long loopCount);
};

class DeviceToDeviceReadCE: public Benchmark {
public:
    DeviceToDeviceReadCE() : Benchmark("device_to_device_memcpy_read_ce", "Device to device memcpy using the Copy Engine (read)") {}
    virtual ~DeviceToDeviceReadCE() {}
    void run(unsigned long long size, unsigned long long loopCount);
    bool filter() { return Benchmark::filterHasAccessiblePeerPairs(); }
};

class DeviceToDeviceWriteCE: public Benchmark {
public:
    DeviceToDeviceWriteCE() : Benchmark("device_to_device_memcpy_write_ce", "Device to device memcpy using the Copy Engine (write)") {}
    virtual ~DeviceToDeviceWriteCE() {}
    void run(unsigned long long size, unsigned long long loopCount);
    bool filter() { return Benchmark::filterHasAccessiblePeerPairs(); }
};

class DeviceToDeviceBidirCE: public Benchmark {
public:
    DeviceToDeviceBidirCE() : Benchmark("device_to_device_bidirectional_memcpy_ce", "Bidirectional device to device memcpy using the Copy Engine") {}
    virtual ~DeviceToDeviceBidirCE() {}
    void run(unsigned long long size, unsigned long long loopCount);
    bool filter() { return Benchmark::filterHasAccessiblePeerPairs(); }
};

class AllToHostCE: public Benchmark {
public:
    AllToHostCE() : Benchmark("all_to_host_memcpy_ce", "All devices to host memcpy using the Copy Engine") {}
    virtual ~AllToHostCE() {}
    void run(unsigned long long size, unsigned long long loopCount);
};

class HostToAllCE: public Benchmark {
public:
    HostToAllCE() : Benchmark("host_to_all_memcpy_ce", "Host to all devices memcpy using the Copy Engine") {}
    virtual ~HostToAllCE() {}
    void run(unsigned long long size, unsigned long long loopCount);
};

// SM Benchmark classes
class HostToDeviceSM: public Benchmark {
public:
    HostToDeviceSM() : Benchmark("host_to_device_memcpy_sm", "Host to device memcpy using the Stream Multiprocessor") {}
    virtual ~HostToDeviceSM() {}
    void run(unsigned long long size, unsigned long long loopCount);
};

class DeviceToHostSM: public Benchmark {
public:
    DeviceToHostSM() : Benchmark("device_to_host_memcpy_sm", "Device to host memcpy using the Stream Multiprocessor") {}
    virtual ~DeviceToHostSM() {}
    void run(unsigned long long size, unsigned long long loopCount);
};

class DeviceToDeviceReadSM: public Benchmark {
public:
    DeviceToDeviceReadSM() : Benchmark("device_to_device_memcpy_read_sm", "Device to device memcpy using the Stream Multiprocessor (read)") {}
    virtual ~DeviceToDeviceReadSM() {}
    void run(unsigned long long size, unsigned long long loopCount);
    bool filter() { return Benchmark::filterHasAccessiblePeerPairs(); }
};

class DeviceToDeviceWriteSM: public Benchmark {
public:
    DeviceToDeviceWriteSM() : Benchmark("device_to_device_memcpy_write_sm", "Device to device memcpy using the Stream Multiprocessor (write)") {}
    virtual ~DeviceToDeviceWriteSM() {}
    void run(unsigned long long size, unsigned long long loopCount);
    bool filter() { return Benchmark::filterHasAccessiblePeerPairs(); }
};

class DeviceToDeviceBidirReadSM: public Benchmark {
public:
    DeviceToDeviceBidirReadSM() : Benchmark("device_to_device_bidirectional_memcpy_read_sm", "Bidirectional device to device memcpy using the Stream Multiprocessor (read)") {}
    virtual ~DeviceToDeviceBidirReadSM() {}
    void run(unsigned long long size, unsigned long long loopCount);
    bool filter() { return Benchmark::filterHasAccessiblePeerPairs(); }
};

class DeviceToDeviceBidirWriteSM: public Benchmark {
public:
    DeviceToDeviceBidirWriteSM() : Benchmark("device_to_device_bidirectional_memcpy_write_sm", "Bidirectional device to device memcpy using the Stream Multiprocessor (write)") {}
    virtual ~DeviceToDeviceBidirWriteSM() {}
    void run(unsigned long long size, unsigned long long loopCount);
    bool filter() { return Benchmark::filterHasAccessiblePeerPairs(); }
};

#endif
