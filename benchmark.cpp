/*
 * Copyright 1993-2021 by NVIDIA Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#include "benchmark.h"

Benchmark::Benchmark(std::string key, benchfn_t benchmark_func, std::string desc, filterfn_t filter_func) : 
    key(std::move(key)), benchmark_func(benchmark_func), desc(std::move(desc)), isHost(isHost), filter_func(filter_func)
{}

std::string Benchmark::benchKey() { return key; }
bool Benchmark::filter() { return filter_func(); }
std::string Benchmark::benchDesc() { return desc; }

void Benchmark::run(unsigned long long size, unsigned long long loopCount) {
    benchmark_func(size, loopCount);
}

// Benchmark filters
bool filter_has_accessible_peer_pairs() {
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
