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
#include <cuda_runtime_api.h>
#include <nvml.h>
#include <iostream>

#include "testcase.h"
#include "version.h"

namespace opt = boost::program_options;

int deviceCount;
unsigned int averageLoopCount;
unsigned long long bufferSize;
unsigned long long loopCount;
bool verbose;
bool disableAffinity;
Verbosity VERBOSE;

// Define testcases here
std::vector<Testcase*> createTestcases() {
    return {
        new HostToDeviceCE(),
        new DeviceToHostCE(),
        new HostToDeviceBidirCE(),
        new DeviceToHostBidirCE(),
        new DeviceToDeviceReadCE(),
        new DeviceToDeviceWriteCE(),
        new DeviceToDeviceBidirReadCE(),
        new DeviceToDeviceBidirWriteCE(),
        new AllToHostCE(),
        new HostToAllCE(),
        new AllToOneWriteCE(),
        new AllToOneReadCE(),
        new OneToAllWriteCE(),
        new OneToAllReadCE(),
        new HostToDeviceSM(),
        new DeviceToHostSM(),
        new DeviceToDeviceReadSM(),
        new DeviceToDeviceWriteSM(),
        new DeviceToDeviceBidirReadSM(),
        new DeviceToDeviceBidirWriteSM(),
        new AllToHostSM(),
        new HostToAllSM(),
        new AllToOneWriteSM(),
        new AllToOneReadSM(),
        new OneToAllWriteSM(),
        new OneToAllReadSM()
    };
}

Testcase* findTestcase(std::vector<Testcase*> &testcases, std::string id) {
    // Check if testcase ID is index
    char* p;
    long index = strtol(id.c_str(), &p, 10);
    if (*p) {
        // Conversion failed so key is ID
        auto it = find_if(testcases.begin(), testcases.end(), [&id](Testcase* test) {return test->testKey() == id;});
        if (it != testcases.end()) {
            return testcases.at(std::distance(testcases.begin(), it));
        } else {
            throw "Testcase " + id + " not found!";
        }
    } else {
        // ID is index
        if (index >= testcases.size()) throw "Testcase index " + id + " out of bound!";
        return testcases.at(index);
    }
}

void runTestcase(std::vector<Testcase*> &testcases, const std::string &testcaseID) {
    CUcontext testCtx;

    try {
        Testcase* test = findTestcase(testcases, testcaseID);
        if (!test->filter()) {
            std::cout << "Waiving " << test->testKey() << "." << std::endl << std::endl;
            return;
        }
        std::cout << "Running " << test->testKey() << ".\n";

        CU_ASSERT(cuCtxCreate(&testCtx, 0, 0));
        CU_ASSERT(cuCtxSetCurrent(testCtx));
        // Run the testcase
        test->run(bufferSize * _MiB, loopCount);
        CU_ASSERT(cuCtxDestroy(testCtx));
    } catch (std::string &s) {
        std::cout << "ERROR: " << s << std::endl;
    }
}

int main(int argc, char **argv) {
    averageLoopCount = defaultAverageLoopCount;

    std::cout << "nvbandwidth Version: " << NVBANDWIDTH_VERSION << std::endl;
    std::cout << "Built from Git version: " << GIT_VERSION << std::endl << std::endl;
    
    std::vector<Testcase*> testcases = createTestcases();
    std::vector<std::string> testcasesToRun;

    // Args parsing
    opt::options_description desc("nvbandwidth CLI");
    desc.add_options()
        ("help,h", "Produce help message")
        ("bufferSize", opt::value<unsigned long long int>(&bufferSize)->default_value(defaultBufferSize), "Memcpy buffer size in MiB")
        ("loopCount", opt::value<unsigned long long int>(&loopCount)->default_value(defaultLoopCount), "Iterations of memcpy to be performed")
        ("list,l", "List available testcases")
        ("testcase,t", opt::value<std::vector<std::string>>(&testcasesToRun)->multitoken(), "Testcase(s) to run (by name or index)")
        ("verbose,v", opt::bool_switch(&verbose)->default_value(false), "Verbose output")
        ("disableAffinity,d", opt::bool_switch(&disableAffinity)->default_value(false), "Disable automatic CPU affinity control");

    opt::variables_map vm;
    try {
        opt::store(opt::parse_command_line(argc, argv, desc), vm);
        opt::notify(vm);
    } catch (...) {
        std::cout << "ERROR: Invalid Arguments " << std::endl;
        for (int i = 0; i < argc; i++) {
            std::cout << argv[i] << " ";
        }
        std::cout << std::endl << std::endl;
        std::cout << desc << "\n";
        return 1;
    }

    if (vm.count("help")) {
        std::cout << desc << "\n";
        return 1;
    }

    if (vm.count("list")) {
        size_t numTestcases = testcases.size();
        std::cout << "Index, Name:\n\tDescription\n";
        std::cout << "=======================\n";
        for (unsigned int i = 0; i < numTestcases; i++) {
            std::cout << i << ", " << testcases.at(i)->testKey() << ":\n" << testcases.at(i)->testDesc() << "\n\n";
        }
        return 0;
    }

    std::cout << "NOTE: This tool reports current measured bandwidth on your system." << std::endl 
              << "Additional system-specific tuning may be required to achieve maximal peak bandwidth." << std::endl << std::endl;

    cuInit(0);
    nvmlInit();
    CU_ASSERT(cuDeviceGetCount(&deviceCount));

    int cudaVersion;
    cudaRuntimeGetVersion(&cudaVersion);
    std::cout << "CUDA Runtime Version: " << cudaVersion << std::endl;

    CU_ASSERT(cuDriverGetVersion(&cudaVersion));
    std::cout << "CUDA Driver Version: " << cudaVersion << std::endl;

    char driverVersion[NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE];
    NVML_ASSERT(nvmlSystemGetDriverVersion(driverVersion, NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE));
    std::cout << "Driver Version: " << driverVersion << std::endl << std::endl;

    for (int iDev = 0; iDev < deviceCount; iDev++) {
        CUdevice dev;
        char name[256];

        CU_ASSERT(cuDeviceGet(&dev, iDev));
        CU_ASSERT(cuDeviceGetName(name, 256, dev));

        std::cout << "Device " << iDev << ": " << name << std::endl;
    }
    std::cout << std::endl;

    // This triggers the loading of all kernels on all devices, even with lazy loading enabled.
    // Some tests can create complex dependencies between devices and function loading requires a
    // device synchronization, so loading in the middle of a test can deadlock.
    preloadKernels(deviceCount);

    if (testcasesToRun.size() == 0) {
        // run all testcases
        for (auto testcase : testcases) {
            runTestcase(testcases, testcase->testKey());
        }
    } else {
        for (const auto& testcaseIndex : testcasesToRun) {
            runTestcase(testcases, testcaseIndex);
        }
    }

    for (auto testcase : testcases) { delete testcase; }

    return 0;
}
