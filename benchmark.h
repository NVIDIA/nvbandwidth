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
#include <map>
#include <utility>

typedef void (*benchfn_t)(unsigned long long, unsigned long long);
typedef bool (*filterfn_t)();

class Benchmark {
private:
    std::string key;
    std::string desc;
    bool isHost;
    benchfn_t benchmark_func{};
    filterfn_t filter_func{};

public:
    static bool filter_default() { return true; }

    Benchmark(std::string key, benchfn_t benchmark_func, std::string desc, filterfn_t filter_func = Benchmark::filter_default);

    std::string benchKey();
    bool filter();
    std::string benchDesc();
    void run(unsigned long long size, unsigned long long loopCount);
};

// CE Benchmarks
void launch_HtoD_memcpy_CE(unsigned long long size, unsigned long long loopCount);
void launch_DtoH_memcpy_CE(unsigned long long size, unsigned long long loopCount);
void launch_HtoD_memcpy_bidirectional_CE(unsigned long long size, unsigned long long loopCount);
void launch_DtoH_memcpy_bidirectional_CE(unsigned long long size, unsigned long long loopCount);
void launch_DtoD_memcpy_read_CE(unsigned long long size, unsigned long long loopCount);
void launch_DtoD_memcpy_write_CE(unsigned long long size, unsigned long long loopCount);
void launch_DtoD_memcpy_bidirectional_CE(unsigned long long size, unsigned long long loopCount);
//void launch_DtoD_paired_memcpy_read_CE(unsigned long long size, unsigned long long loopCount);
//void launch_DtoD_paired_memcpy_write_CE(unsigned long long size, unsigned long long loopCount);
// SM Benchmarks
void launch_HtoD_memcpy_SM(unsigned long long size, unsigned long long loopCount);
void launch_DtoH_memcpy_SM(unsigned long long size, unsigned long long loopCount);
void launch_DtoD_memcpy_read_SM(unsigned long long size, unsigned long long loopCount);
void launch_DtoD_memcpy_write_SM(unsigned long long size, unsigned long long loopCount);
void launch_DtoD_memcpy_bidirectional_read_SM(unsigned long long size, unsigned long long loopCount);
void launch_DtoD_memcpy_bidirectional_write_SM(unsigned long long size, unsigned long long loopCount);
//void launch_DtoD_paired_memcpy_read_SM(unsigned long long size, unsigned long long loopCount);
//void launch_DtoD_paired_memcpy_write_SM(unsigned long long size, unsigned long long loopCount);

// Benchmark filters
bool filter_has_accessible_peer_pairs();

#endif
