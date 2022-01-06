/*
 * Copyright 1993-2021 by NVIDIA Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef BENCHMARKS_H
#define BENCHMARKS_H

#include "common.h"
#include "memory_utils.h"
#include "memcpy.h"
#include <map>
#include <utility>

typedef void (*benchfn_t)(unsigned long long, unsigned long long);
typedef bool (*filterfn_t)();

class Benchmark {
    std::string key;
    std::string desc;
    benchfn_t benchmark_func{};
    filterfn_t filter_func{};

public:
    static bool filter_default() { return true; }

    Benchmark() = default;
    Benchmark(std::string key, benchfn_t benchmark_func, std::string desc, filterfn_t filter_func = Benchmark::filter_default): key(std::move(key)), benchmark_func(benchmark_func), desc(std::move(desc)), filter_func(filter_func) {}
    std::string benchKey() { return key; }
    benchfn_t benchFn() { return benchmark_func; }
    filterfn_t filterFn() { return filter_func; }
    std::string benchDesc() { return desc; }
};

// CE Benchmarks
void launch_HtoD_memcpy_CE(unsigned long long size = defaultBufferSize, unsigned long long loopCount = defaultLoopCount);
void launch_DtoH_memcpy_CE(unsigned long long size = defaultBufferSize, unsigned long long loopCount = defaultLoopCount);
void launch_HtoD_memcpy_bidirectional_CE(unsigned long long size = defaultBufferSize, unsigned long long loopCount = defaultLoopCount);
void launch_DtoH_memcpy_bidirectional_CE(unsigned long long size = defaultBufferSize, unsigned long long loopCount = defaultLoopCount);
void launch_DtoD_memcpy_read_CE(unsigned long long size, unsigned long long loopCount);
void launch_DtoD_memcpy_write_CE(unsigned long long size, unsigned long long loopCount);
void launch_DtoD_memcpy_bidirectional_CE(unsigned long long size, unsigned long long loopCount);
void launch_DtoD_paired_memcpy_read_CE(unsigned long long size, unsigned long long loopCount);
void launch_DtoD_paired_memcpy_write_CE(unsigned long long size, unsigned long long loopCount);
// SM Benchmarks
void launch_HtoD_memcpy_SM(unsigned long long size, unsigned long long loopCount);
void launch_DtoH_memcpy_SM(unsigned long long size, unsigned long long loopCount);
void launch_DtoD_memcpy_read_SM(unsigned long long size, unsigned long long loopCount);
void launch_DtoD_memcpy_write_SM(unsigned long long size, unsigned long long loopCount);
void launch_DtoD_memcpy_bidirectional_read_SM(unsigned long long size, unsigned long long loopCount);
void launch_DtoD_memcpy_bidirectional_write_SM(unsigned long long size, unsigned long long loopCount);
void launch_DtoD_paired_memcpy_read_SM(unsigned long long size, unsigned long long loopCount);
void launch_DtoD_paired_memcpy_write_SM(unsigned long long size, unsigned long long loopCount);

// Benchmark filters
// TODO define in a cpp file, but benchmarks are declared here, defined in two different CPPs, and constructed in a different cpp
static bool filter_has_accessible_peer_pairs() {
    int deviceCount = 0;
    CU_ASSERT(cuDeviceGetCount(&deviceCount));

    for (int currentDevice = 0; currentDevice < deviceCount; currentDevice++) {
        for (int peer = 0; peer < deviceCount; peer++) {
            int canAccessPeer = 0;

            if (peer == currentDevice) {
                continue;
            }

            CU_ASSERT(cuDeviceCanAccessPeer(&canAccessPeer, currentDevice, peer));
            if (canAccessPeer) {
                return true;
            }
        }
    }

    return false;
}

#endif
