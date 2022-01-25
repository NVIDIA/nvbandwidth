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
