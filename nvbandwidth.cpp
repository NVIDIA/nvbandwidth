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

#include "json_output.h"
#include "kernels.cuh"
#include "output.h"
#include "testcase.h"
#include "version.h"
#include "inline_common.h"

namespace opt = boost::program_options;

int deviceCount;
unsigned int averageLoopCount;
unsigned long long bufferSize;
unsigned long long loopCount;
bool verbose;
bool shouldOutput = true;
bool disableAffinity;
bool skipVerification;
bool useMean;

Verbosity VERBOSE(verbose);
Verbosity OUTPUT(shouldOutput);

bool jsonOutput;
Output *output;

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
        new AllToHostBidirCE(),
        new HostToAllCE(),
        new HostToAllBidirCE(),
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
        new AllToHostBidirSM(),
        new HostToAllSM(),
        new HostToAllBidirSM(),
        new AllToOneWriteSM(),
        new AllToOneReadSM(),
        new OneToAllWriteSM(),
        new OneToAllReadSM(),
        new HostDeviceLatencySM(),
        new DeviceToDeviceLatencySM(),
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
        if (index < 0 || index >= static_cast<long>(testcases.size())) throw "Testcase index " + id + " out of bound!";
        return testcases.at(index);
    }
}

std::vector<std::string> expandTestcases(std::vector<Testcase*> &testcases, std::vector<std::string> prefixes) {
    std::vector<std::string> testcasesToRun;
    for (auto testcase: testcases){ 
         auto it = find_if(prefixes.begin(), prefixes.end(), [&testcase](std::string prefix) {return testcase->testKey().compare(0, prefix.size(), prefix) == 0;});
            if (it != prefixes.end()) {
                testcasesToRun.push_back(testcase->testKey());
            }
    }
    return testcasesToRun;  
}

void runTestcase(std::vector<Testcase*> &testcases, const std::string &testcaseID) {
    CUcontext testCtx;
    Testcase* test{nullptr};
    try {
        test = findTestcase(testcases, testcaseID);
    } catch (std::string &s) {
        output->addTestcase(testcaseID, "ERROR", s);
        return;
    }

    try {
        if (!test->filter()) {
            output->addTestcase(test->testKey(), NVB_WAIVED);
            return;
        }

        output->addTestcase(test->testKey(), NVB_RUNNING);

        CU_ASSERT(cuCtxCreate(&testCtx, 0, 0));
        CU_ASSERT(cuCtxSetCurrent(testCtx));
        // Run the testcase
        test->run(bufferSize * _MiB, loopCount);
        CU_ASSERT(cuCtxDestroy(testCtx));
    } catch (std::string &s) {
        output->setTestcaseStatusAndAddIfNeeded(test->testKey(), NVB_ERROR_STATUS, s);
    }
}

int main(int argc, char **argv) {
    std::vector<Testcase*> testcases = createTestcases();
    std::vector<std::string> testcasesToRun;
    std::vector<std::string> testcasePrefixes;
    output = new Output();

    // Args parsing
    opt::options_description visible_opts("nvbandwidth CLI");
    visible_opts.add_options()
        ("help,h", "Produce help message")
        ("bufferSize,b", opt::value<unsigned long long int>(&bufferSize)->default_value(defaultBufferSize), "Memcpy buffer size in MiB")
        ("list,l", "List available testcases")
        ("testcase,t", opt::value<std::vector<std::string>>(&testcasesToRun)->multitoken(), "Testcase(s) to run (by name or index)")
        ("testcasePrefixes,p", opt::value<std::vector<std::string>>(&testcasePrefixes)->multitoken(), "Testcase(s) to run (by prefix))")
        ("verbose,v", opt::bool_switch(&verbose)->default_value(false), "Verbose output")
        ("skipVerification,s", opt::bool_switch(&skipVerification)->default_value(false), "Skips data verification after copy")
        ("disableAffinity,d", opt::bool_switch(&disableAffinity)->default_value(false), "Disable automatic CPU affinity control")
        ("testSamples,i", opt::value<unsigned int>(&averageLoopCount)->default_value(defaultAverageLoopCount), "Iterations of the benchmark")
        ("useMean,m", opt::bool_switch(&useMean)->default_value(false), "Use mean instead of median for results")
        ("json,j", opt::bool_switch(&jsonOutput)->default_value(false), "Print output in json format instead of plain text.");

    opt::options_description all_opts("");
    all_opts.add(visible_opts);
    all_opts.add_options()
        ("loopCount", opt::value<unsigned long long int>(&loopCount)->default_value(defaultLoopCount), "Iterations of memcpy to be performed within a test sample");

    opt::variables_map vm;
    try {
        opt::store(opt::parse_command_line(argc, argv, all_opts), vm);
        opt::notify(vm);
    
    } catch (...) {
        output->addVersionInfo();

        std::stringstream errmsg;
        errmsg << "ERROR: Invalid Arguments " << std::endl;
        for (int i = 0; i < argc; i++) {
            errmsg << argv[i] << " ";
        }
        std::vector<std::string> messageParts;
        std::stringstream buf;
        buf << visible_opts;
        messageParts.emplace_back(errmsg.str());
        messageParts.emplace_back(buf.str());
        output->recordError(messageParts);
        return 1;
    }

    if (jsonOutput) {
        delete output;
        output = new JsonOutput();
    }
    
    output->addVersionInfo();

    if (vm.count("help")) {
        OUTPUT << visible_opts << "\n";
        return 0;
    }

    if (vm.count("list")) {
        output->listTestcases(testcases);
        return 0;
    }

    if (testcasePrefixes.size() != 0 && testcasesToRun.size() != 0) {
        output->recordError("You cannot specify both testcase and testcasePrefix options at the same time");
        return 1;
    }

    output->printInfo();

    CU_ASSERT(cuInit(0));
    NVML_ASSERT(nvmlInit());
    CU_ASSERT(cuDeviceGetCount(&deviceCount));
    if (bufferSize < defaultBufferSize) {
        output->recordWarning("NOTE: You have chosen a buffer size that is smaller than the default buffer size. It is suggested to use the default buffer size (64MB) to achieve maximal peak bandwidth.");
    }

    int cudaVersion;
    cudaRuntimeGetVersion(&cudaVersion);

    CU_ASSERT(cuDriverGetVersion(&cudaVersion));

    char driverVersion[NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE];
    NVML_ASSERT(nvmlSystemGetDriverVersion(driverVersion, NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE));

    output->addCudaAndDriverInfo(cudaVersion, driverVersion);

    output->recordDevices(deviceCount);

    if (testcasePrefixes.size() > 0) {
        testcasesToRun = expandTestcases(testcases, testcasePrefixes);
        if (testcasesToRun.size() == 0) {
            output->recordError("Specified list of testcase prefixes did not match any testcases");
            return 1;
        }
    }

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

    output->print();

    for (auto testcase : testcases) { delete testcase; }

    return 0;
}
