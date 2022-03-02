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

#include <boost/program_options.hpp>
#include <cuda.h>
#include <nvml.h>
#include <iostream>

#include "benchmark.h"
#include "version.h"

namespace opt = boost::program_options;

int deviceCount;
unsigned int averageLoopCount;
unsigned long long bufferSize;
unsigned long long loopCount;
bool disableP2P;
bool verbose;
Verbosity VERBOSE;

// Define benchmarks here
std::vector<Benchmark> createBenchmarks() {
    return {
        Benchmark("host_to_device_memcpy_ce", launch_HtoD_memcpy_CE, "Host to device memcpy using the Copy Engine"),
        Benchmark("device_to_host_memcpy_ce", launch_DtoH_memcpy_CE, "Device to host memcpy using the Copy Engine"),
        Benchmark("host_to_device_bidirectional_memcpy_ce", launch_HtoD_memcpy_bidirectional_CE, "Bidirectional host to device memcpy using the Copy Engine"),
        Benchmark("device_to_host_bidirectional_memcpy_ce", launch_DtoH_memcpy_bidirectional_CE, "Bidirectional device to host memcpy using the Copy Engine"),
        Benchmark("device_to_device_memcpy_read_ce", launch_DtoD_memcpy_read_CE, "Device to device memcpy using the Copy Engine (read)", filter_has_accessible_peer_pairs),
        Benchmark("device_to_device_memcpy_write_ce", launch_DtoD_memcpy_write_CE, "Device to device memcpy using the Copy Engine (write)", filter_has_accessible_peer_pairs),
        Benchmark("device_to_device_bidirectional_memcpy_ce", launch_DtoD_memcpy_bidirectional_CE, "Bidirectional device to device memcpy using the Copy Engine", filter_has_accessible_peer_pairs),
        Benchmark("all_to_host_memcpy_ce", launch_AlltoH_memcpy_CE, "All devices to host memcpy using the Copy Engine"),
        Benchmark("host_to_all_memcpy_ce", launch_HtoAll_memcpy_CE, "Host to all devices memcpy using the Copy Engine"),
        Benchmark("host_to_device_memcpy_sm", launch_HtoD_memcpy_SM, "Host to device memcpy using the Stream Multiprocessor"),
        Benchmark("device_to_host_memcpy_sm", launch_DtoH_memcpy_SM, "Device to host memcpy using the Stream Multiprocessor"),
        Benchmark("device_to_device_memcpy_read_sm", launch_DtoD_memcpy_read_SM, "Device to device memcpy using the Stream Multiprocessor (read)", filter_has_accessible_peer_pairs),
        Benchmark("device_to_device_memcpy_write_sm", launch_DtoD_memcpy_write_SM, "Device to device memcpy using the Stream Multiprocessor (write)", filter_has_accessible_peer_pairs),
        Benchmark("device_to_device_bidirectional_memcpy_read_sm", launch_DtoD_memcpy_bidirectional_read_SM, "Bidirectional device to device memcpy using the Stream Multiprocessor (read)", filter_has_accessible_peer_pairs),
        Benchmark("device_to_device_bidirectional_memcpy_write_sm", launch_DtoD_memcpy_bidirectional_write_SM, "Bidirectional device to device memcpy using the Stream Multiprocessor (write)", filter_has_accessible_peer_pairs),
    };
}

Benchmark findBenchmark(std::vector<Benchmark> &benchmarks, std::string id) {
    // Check if benchmark ID is index
    char* p;
    long index = strtol(id.c_str(), &p, 10);
    if (*p) {
        // Conversion failed so key is ID
        auto it = find_if(benchmarks.begin(), benchmarks.end(), [&id](Benchmark& bench) {return bench.benchKey() == id;});
        if (it != benchmarks.end()) {
            return benchmarks.at(std::distance(benchmarks.begin(), it));
        } else {
            throw "Benchmark " + id + " not found!";
        }
    } else {
        // ID is index
        if (index >= benchmarks.size()) throw "Benchmark index " + id + " out of bound!";
        return benchmarks.at(index);
    }
}

void runBenchmark(std::vector<Benchmark> &benchmarks, const std::string &benchmarkID) {
    CUcontext benchCtx;

    try {
        Benchmark bench = findBenchmark(benchmarks, benchmarkID);
        if (!bench.filter()) {
            std::cout << "Waiving benchmark " << bench.benchKey() << "." << std::endl << std::endl;
            return;
        }
        std::cout << "Running benchmark " << bench.benchKey() << ".\n";

        CU_ASSERT(cuCtxCreate(&benchCtx, 0, 0));
        CU_ASSERT(cuCtxSetCurrent(benchCtx));
        // Run the launch_* benchmark
        bench.run(bufferSize * _MiB, loopCount);
        CU_ASSERT(cuCtxDestroy(benchCtx));
    } catch (std::string &s) {
        std::cout << "ERROR: " << s << std::endl;
    }
}

int main(int argc, char **argv) {
    averageLoopCount = defaultAverageLoopCount;
    disableP2P = true;

    std::cout << "nvbandwidth Version: " << NVBANDWIDTH_VERSION << std::endl;
    std::cout << "Built from: " << GIT_VERSION << std::endl << std::endl;
    
    std::vector<Benchmark> benchmarks = createBenchmarks();
    std::vector<std::string> benchmarksToRun;

    // Args parsing
    opt::options_description desc("NVBandwidth CLI");
    desc.add_options()
        ("help,h", "Produce help message")
        ("bufferSize", opt::value<unsigned long long int>(&bufferSize)->default_value(defaultBufferSize), "Memcpy buffer size in MiB")
        ("loopCount", opt::value<unsigned long long int>(&loopCount)->default_value(defaultLoopCount), "Iterations of memcpy to be performed")
        ("list,l", "List available benchmarks")
        ("benchmark,b", opt::value<std::vector<std::string>>(&benchmarksToRun)->multitoken(), "Benchmark(s) to doMemcpy (by name or index)")
        ("verbose,v", opt::bool_switch(&verbose)->default_value(false), "Verbose output");

    opt::variables_map vm;
    try {
        opt::store(opt::parse_command_line(argc, argv, desc), vm);
        opt::notify(vm);
    } catch (...) {
        std::cout << desc << "\n";
        return 1;
    }

    if (vm.count("help")) {
        std::cout << desc << "\n";
        return 1;
    }

    if (vm.count("list")) {
        size_t numBenchmarks = benchmarks.size();
        std::cout << "Index, Name:\n\tDescription\n";
        std::cout << "=======================\n";
        for (unsigned int i = 0; i < numBenchmarks; i++) {
            std::cout << i << ", " << benchmarks.at(i).benchKey() << ":\n\t" << benchmarks.at(i).benchDesc() << "\n\n";
        }
        return 0;
    }

    cuInit(0);
    nvmlInit();
    CU_ASSERT(cuDeviceGetCount(&deviceCount));

    int cudaVersion;
    CU_ASSERT(cuDriverGetVersion(&cudaVersion));
    std::cout << "Using CUDA Driver Version: " << cudaVersion << std::endl;

    char driverVersion[NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE];
    NVML_ASSERT(nvmlSystemGetDriverVersion(driverVersion, NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE));
    std::cout << "Using Driver Version: " << driverVersion << std::endl << std::endl;

    if (benchmarksToRun.size() == 0) {
        // run all benchmarks
        for (auto benchmark : benchmarks) {
            runBenchmark(benchmarks, benchmark.benchKey());
        }
    } else {
        for (const auto& benchmarkIndex : benchmarksToRun) {
            runBenchmark(benchmarks, benchmarkIndex);
        }
    }

    return 0;
}
