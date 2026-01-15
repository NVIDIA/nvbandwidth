/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <argparse/argparse.hpp>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <nvml.h>
#include <iostream>
#ifndef _WIN32
#include <dlfcn.h>
#endif

#include "cuda_version_check.h"
#include "environment.h"
#include "json_output.h"
#include "perf_parser_output.h"
#include "kernels.cuh"
#include "nvbandwidth_version.h"
#include "output.h"
#include "testcase.h"
#include "test_settings.h"
#include "inline_common.h"

// Global settings instance
TestSettings gSettings;
std::string gpuPairUuid0, gpuPairUuid1;
int gpuPairDev0 = -1, gpuPairDev1 = -1;

int deviceCount;
bool shouldOutput = true;

Verbosity VERBOSE(gSettings.verbose);
Verbosity OUTPUT(shouldOutput);

// Device ordinal of the GPU owned by the process
int localDevice = 0;
int localRank = 0;
// Process rank within one OS
int worldRank = 0;
int worldSize = 0;
Output *output;

std::unique_ptr<Environment> env;

// Define testcases here
std::vector<Testcase*> createNodeTestcases() {
    std::vector<Testcase*> tests;
    if (worldRank == 0 && localRank == 0) {
         tests.insert(tests.end(), {
        // Base tests that always run, by a single node/process only.
        new HostDeviceTransfer(CopyInitiator::CE, Direction::HostToDevice),
        new HostDeviceTransfer(CopyInitiator::CE, Direction::DeviceToHost),
        new HostDeviceBidirTransfer(CopyInitiator::CE, Direction::HostToDevice),
        new HostDeviceBidirTransfer(CopyInitiator::CE, Direction::DeviceToHost),
        new DeviceToDeviceTransfer(CopyInitiator::CE, AccessType::Read),
        new DeviceToDeviceTransfer(CopyInitiator::CE, AccessType::Write),
        new DeviceToDeviceBidirTransfer(CopyInitiator::CE, AccessType::Read),
        new DeviceToDeviceBidirTransfer(CopyInitiator::CE, AccessType::Write),
        new AllToHostTransfer(CopyInitiator::CE),
        new AllToHostBidirTransfer(CopyInitiator::CE),
        new HostToAllTransfer(CopyInitiator::CE),
        new HostToAllBidirTransfer(CopyInitiator::CE),
        new AllToOneTransfer(CopyInitiator::CE, AccessType::Write),
        new AllToOneTransfer(CopyInitiator::CE, AccessType::Read),
        new OneToAllTransfer(CopyInitiator::CE, AccessType::Write),
        new OneToAllTransfer(CopyInitiator::CE, AccessType::Read),
        new HostDeviceTransfer(CopyInitiator::SM, Direction::HostToDevice),
        new HostDeviceTransfer(CopyInitiator::SM, Direction::DeviceToHost),
        new HostDeviceBidirTransfer(CopyInitiator::SM, Direction::HostToDevice),
        new HostDeviceBidirTransfer(CopyInitiator::SM, Direction::DeviceToHost),
        new DeviceToDeviceTransfer(CopyInitiator::SM, AccessType::Read),
        new DeviceToDeviceTransfer(CopyInitiator::SM, AccessType::Write),
        new DeviceToDeviceBidirTransfer(CopyInitiator::SM, AccessType::Read),
        new DeviceToDeviceBidirTransfer(CopyInitiator::SM, AccessType::Write),
        new AllToHostTransfer(CopyInitiator::SM),
        new AllToHostBidirTransfer(CopyInitiator::SM),
        new HostToAllTransfer(CopyInitiator::SM),
        new HostToAllBidirTransfer(CopyInitiator::SM),
        new AllToOneTransfer(CopyInitiator::SM, AccessType::Write),
        new AllToOneTransfer(CopyInitiator::SM, AccessType::Read),
        new OneToAllTransfer(CopyInitiator::SM, AccessType::Write),
        new OneToAllTransfer(CopyInitiator::SM, AccessType::Read),
        new HostDeviceLatencySM(),
        new DeviceToDeviceLatencySM(),
        new DeviceToDeviceTransfer(CopyInitiator::TMA, AccessType::Read),
        new DeviceToDeviceTransfer(CopyInitiator::TMA, AccessType::Write),
        new DeviceToDeviceBidirTransfer(CopyInitiator::TMA, AccessType::Read),
        new DeviceToDeviceBidirTransfer(CopyInitiator::TMA, AccessType::Write),
        new DeviceLocalCopyTransfer(CopyInitiator::CE),
        new DeviceLocalCopyTransfer(CopyInitiator::SM),
        new DeviceLocalReadSM(),
        new DeviceLocalWriteSM(),
        new DeviceToDeviceLatencyTMA(),
        new DeviceLocalCopyTransfer(CopyInitiator::TMA)
      });
    }
#ifdef MULTINODE
    // Add multinode tests only if we're running in multinode mode
    if (env->getSize() > 1) {
        tests.insert(tests.end(), {
            new MultinodeDeviceToDeviceTransfer(CopyInitiator::CE, AccessType::Read),
            new MultinodeDeviceToDeviceTransfer(CopyInitiator::CE, AccessType::Write),
            new MultinodeDeviceToDeviceBidirTransfer(CopyInitiator::CE, AccessType::Read),
            new MultinodeDeviceToDeviceBidirTransfer(CopyInitiator::CE, AccessType::Write),
            new MultinodeDeviceToDeviceTransfer(CopyInitiator::SM, AccessType::Read),
            new MultinodeDeviceToDeviceTransfer(CopyInitiator::SM, AccessType::Write),
            new MultinodeDeviceToDeviceBidirTransfer(CopyInitiator::SM, AccessType::Read),
            new MultinodeDeviceToDeviceBidirTransfer(CopyInitiator::SM, AccessType::Write),
            new MultinodeDeviceToDeviceTransfer(CopyInitiator::TMA, AccessType::Read),
            new MultinodeDeviceToDeviceTransfer(CopyInitiator::TMA, AccessType::Write),
            new MultinodeDeviceToDeviceBidirTransfer(CopyInitiator::TMA, AccessType::Read),
            new MultinodeDeviceToDeviceBidirTransfer(CopyInitiator::TMA, AccessType::Write),
            new MultinodeAllToOneWriteSM(),
            new MultinodeAllFromOneReadSM(),
            new MultinodeBroadcastOneToAllSM(),
            new MultinodeBroadcastAllToAllSM(),
            new MultinodeBisectWriteCE(),
            new MultinodeMulticastFabricPut(CopyInitiator::SM),
            new MultinodeMulticastFabricPut(CopyInitiator::TMA)
        });
    }
#endif
    return tests;
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

// Testcases whose device-to-device sweep is restricted (single-node) or GPU-bound (multinode) by --pair.
static bool testcaseKeyUsesGpuPairSelection(const std::string &k) {
    if (k == "device_to_device_latency_sm" || k == "device_to_device_latency_tma") return true;
    static const char *const prefixes[] = {
        "device_to_device_memcpy_",
        "device_to_device_bidirectional_memcpy_",
        "multinode_device_to_device_memcpy_",
        "multinode_device_to_device_bidirectional_memcpy_",
    };
    for (const char *pre : prefixes) {
        if (k.rfind(pre, 0) == 0) return true;
    }
    return false;
}

static bool selectedTestcasesIncludeGpuPairRelevant(std::vector<Testcase *> &testcases,
                                                    const std::vector<std::string> &testcasesToRun) {
    if (testcasesToRun.empty()) {
        return true;
    }
    for (const std::string &id : testcasesToRun) {
        try {
            Testcase *t = findTestcase(testcases, id);
            if (testcaseKeyUsesGpuPairSelection(t->testKey())) return true;
        } catch (const std::string &) {
            continue;
        }
    }
    return false;
}

std::vector<std::string> expandTestcases(std::vector<Testcase*> &testcases, std::vector<std::string> prefixes) {
    std::vector<std::string> testcasesToRun;
    for (auto testcase : testcases) {
         auto it = find_if(prefixes.begin(), prefixes.end(), [&testcase](std::string prefix) {return testcase->testKey().compare(0, prefix.size(), prefix) == 0;});
            if (it != prefixes.end()) {
                testcasesToRun.push_back(testcase->testKey());
            }
    }
    return testcasesToRun;
}

void runTestcase(std::vector<Testcase*> &testcases, const std::string &testcaseID) {
    Testcase* test{nullptr};
    try {
        test = findTestcase(testcases, testcaseID);
    } catch (std::string &s) {
        output->addTestcase(testcaseID, "ERROR", s);
        return;
    }

    try {
        auto filterResult = test->filter();
        if (!filterResult) {
            output->addTestcase(test->testKey(), NVB_WAIVED, filterResult.reason);
            return;
        }

        output->addTestcase(test->testKey(), NVB_RUNNING);

        // Run the testcase
        if (test->testKey().find("latency") != std::string::npos) {
            // use fixd-size buffer for latency tests
            test->run(2 * _MiB, gSettings.loopCount);
        } else {
            test->run(gSettings.bufferSizeMiB * _MiB, gSettings.loopCount);
        }
    } catch (std::string &s) {
        output->setTestcaseStatusAndAddIfNeeded(test->testKey(), NVB_ERROR_STATUS, s);
    }
}

static bool isBounceBufferConfComputeEnabled() {
    bool enabled = false;

#ifndef _WIN32
#ifdef NVBW_HAVE_NVML_CONF_COMPUTE_SETTINGS
    void* sym = dlsym(RTLD_DEFAULT, "nvmlSystemGetConfComputeSettings");
    if (sym != nullptr) {
        auto nvmlSystemGetConfComputeSettingsFn = reinterpret_cast<nvmlReturn_t (*)(nvmlSystemConfComputeSettings_v1_t*)>(sym);

        nvmlSystemConfComputeSettings_v1_t settings = {};
        settings.version = nvmlSystemConfComputeSettings_v1;

        nvmlReturn_t nvmlResult = nvmlSystemGetConfComputeSettingsFn(&settings);
        if (nvmlResult == NVML_SUCCESS) {
            enabled = (settings.ccFeature == NVML_CC_SYSTEM_FEATURE_ENABLED) ||
                      (settings.multiGpuMode == NVML_CC_SYSTEM_MULTIGPU_PROTECTED_PCIE);
        } else if (nvmlResult != NVML_ERROR_NOT_SUPPORTED) {
            std::stringstream errmsg;
            errmsg << "NOTE: nvmlSystemGetConfComputeSettings failed with " << nvmlResult << ": " << nvmlErrorString(nvmlResult);
            output->recordWarning(errmsg.str());
        }
    } else {
        std::stringstream errmsg;
        errmsg << "NOTE: nvmlSystemGetConfComputeSettings is not available ( " << dlerror() << "). nvbandwidth will not be able to detect bounce-buffer CC platforms.";
        output->recordWarning(errmsg.str());
    }
#endif
#endif

    return enabled;
}

int main(int argc, char **argv) {
    env = Environment::create(argc, argv);
    env->initialize(argc, argv);

#ifdef MULTINODE
    worldSize = env->getSize();
    worldRank = env->getRank();
    localRank = env->getLocalRank();
    // Avoid excessive output by limit output to rank 0
    shouldOutput = (worldRank == 0);
#endif

    std::vector<Testcase*> testcases = createNodeTestcases();
    std::vector<std::string> testcasesToRun;
    std::vector<std::string> testcasePrefixes;
    // Pre-scan argv for the output format flag before full argument parsing so
    // that the correct output handler (and its version block) is already in
    // place if parse_args throws.  --format / -F take precedence; fall back to
    // the legacy --json / -j flag.
    {
        std::string fmt = "text";
        for (int i = 1; i < argc; i++) {
            std::string arg(argv[i]);
            if ((arg == "--format" || arg == "-F") && i + 1 < argc) {
                fmt = std::string(argv[i + 1]);
                break;
            }
            const std::string prefix = "--format=";
            if (arg.substr(0, prefix.size()) == prefix) {
                fmt = arg.substr(prefix.size());
                break;
            }
        }
        if (fmt == "text") {
            for (int i = 1; i < argc; i++) {
                std::string arg(argv[i]);
                if (arg == "--json" || arg == "-j") {
                    fmt = "json";
                    break;
                }
            }
        }
        if (fmt == "json") {
            output = new JsonOutput(shouldOutput);
        } else if (fmt == "perf") {
            output = new PerfParserOutput(shouldOutput);
        } else {
            output = new Output();
        }
    }

    // Version info is added to the output object before any early-exit path
    // (parse errors, --version, validation failures) so the header block is
    // always present in the output document regardless of how we exit.
    output->addVersionInfo();

    // Args parsing - all settings stored directly into gSettings
    argparse::ArgumentParser program("nvbandwidth");

    program.add_description("A tool for bandwidth measurements on NVIDIA GPUs.");
    program.add_epilog("Examples:\n"
                       "  nvbandwidth                          # Run all testcases\n"
                       "  nvbandwidth -l                       # List available testcases\n"
                       "  nvbandwidth -t host_to_device_memcpy_ce device_to_host_memcpy_ce\n"
                       "  nvbandwidth -p device_to_device      # Run all device_to_device* tests\n"
                       "  nvbandwidth --format json            # Output in JSON format");

    bool printVersion = false;
    program.add_argument("--version")
        .help("Print nvbandwidth release version and exit")
        .flag()
        .store_into(printVersion);

    program.add_argument("-b", "--bufferSize")
        .help("Memcpy buffer size in MiB")
        .metavar("SIZE")
        .default_value(kDefaultBufferSizeMiB)
        .scan<'u', std::uint64_t>()
        .store_into(gSettings.bufferSizeMiB);
    program.add_argument("-l", "--list")
        .help("List available testcases")
        .flag();

    // Testcase selection - mutually exclusive
    auto &testcase_group = program.add_mutually_exclusive_group();
    testcase_group.add_argument("-t", "--testcase")
        .help("Testcase(s) to run (by name or index)")
        .metavar("TEST")
        .nargs(argparse::nargs_pattern::any)
        .append()
        .default_value(std::vector<std::string>{})
        .store_into(testcasesToRun);
    testcase_group.add_argument("-p", "--testcasePrefixes")
        .help("Testcase(s) to run (by prefix)")
        .metavar("PREFIX")
        .append()
        .store_into(testcasePrefixes);

    program.add_argument("-v", "--verbose")
        .help("Verbose output")
        .flag()
        .store_into(gSettings.verbose);
    program.add_argument("-s", "--skipVerification")
        .help("Skips data verification after copy")
        .flag()
        .store_into(gSettings.skipVerification);
    program.add_argument("-d", "--disableAffinity")
        .help("Disable automatic CPU affinity control")
        .flag()
        .store_into(gSettings.disableAffinity);
    program.add_argument("-i", "--testSamples")
        .help("Iterations of the benchmark")
        .metavar("N")
        .default_value(kDefaultAverageLoopCount)
        .scan<'u', std::uint32_t>()
        .store_into(gSettings.averageLoopCount);
    program.add_argument("-P", "--targetNumPairs")
        .help("Target pairs for multinode device-to-device tests")
        .metavar("N")
        .default_value(kDefaultTargetPairs)
        .scan<'i', std::int64_t>()
        .store_into(gSettings.targetNumPairs);
    program.add_argument("-m", "--useMean")
        .help("Use mean instead of median for results")
        .flag()
        .store_into(gSettings.useMean);
    program.add_argument("-H", "--useHugePages")
        .help("Use huge pages for host allocations")
        .flag()
        .store_into(gSettings.useHugePages);

    program.add_argument("-F", "--format")
        .help("Output format")
        .metavar("FORMAT")
        .default_value(std::string("text"))
        .choices("text", "json", "perf")
        .store_into(gSettings.outputFormat);
    // Legacy format flag (hidden, for backward compatibility) to ensure previous
    // existing scripts using the old format for output arguments work.
    bool legacyJson = false;
    program.add_argument("-j", "--json")
        .flag()
        .store_into(legacyJson)
        .hidden();

    // Hidden options (not shown in help)
    program.add_argument("--loopCount")
        .help("Iterations of memcpy to be performed within a test sample")
        .metavar("N")
        .default_value(kDefaultLoopCount)
        .scan<'u', std::uint64_t>()
        .store_into(gSettings.loopCount)
        .hidden();
    program.add_argument("--disablePerformanceCache")
        .help("Disable bidirectional result cache (enabled by default)")
        .flag()
        .store_into(gSettings.disablePerformanceCache)
        .hidden();
    program.add_argument("--perfFormatter")
        .help("Use perf formatter prefix in output")
        .flag()
        .store_into(gSettings.perfFormatter)
        .hidden();
    program.add_argument("--disableNvlinkSMScheduling")
        .help("Disable NVLink utilization scheduling for SM tests (Blackwell+)")
        .flag()
        .store_into(gSettings.disableNvlinkSMScheduling)
        .hidden();
    std::vector<std::string> pairUuids;
    program.add_argument("--pair")
        .help("Run tests only between these two GPUs (by UUID). GPUs may be on the same or different nodes.")
        .metavar("UUID")
        .nargs(2)
        .store_into(pairUuids);

    try {
        program.parse_args(argc, argv);
    } catch (const std::exception& err) {
        std::stringstream errmsg;
        errmsg << "Error parsing command line: " << err.what();
        output->recordError(errmsg.str());
        output->print();
        return 1;
    }

    // Handle legacy format flag (backward compatibility)
    if (legacyJson) {
        if (gSettings.outputFormat != "text") {
            output->recordError("Cannot use --json with --format; use --format json instead");
            return 1;
        }
        gSettings.outputFormat = "json";
    }

    if (printVersion) {
        // Version info is already in the tree; just flush and exit.
        output->print();
        return 0;
    }

    if (program.get<bool>("--list")) {
        output->listTestcases(testcases);
        return 0;
    }

    // Note: -t and -p mutual exclusivity is enforced by add_mutually_exclusive_group()

    if (pairUuids.size() >= 2) {
        gpuPairUuid0 = pairUuids[0];
        gpuPairUuid1 = pairUuids[1];
    }

    // Validate targetNumPairs argument
    if (gSettings.targetNumPairs < -1) {
        std::stringstream errmsg;
        errmsg << "ERROR: Invalid targetNumPairs value: " << gSettings.targetNumPairs
               << ". Must be -1 (all pairs), 0 (no pairs), or a positive number.";
        output->recordError(errmsg.str());
        return 1;
    }
#ifdef MULTINODE
    // In multinode mode, validate against maximum possible pairs
    if (gSettings.targetNumPairs > 0) {
        long long maxPairs = static_cast<long long>(worldSize) * (worldSize - 1);
        if (gSettings.targetNumPairs > maxPairs) {
            std::stringstream errmsg;
            errmsg << "ERROR: targetNumPairs (" << gSettings.targetNumPairs
                   << ") exceeds maximum possible pairs (" << maxPairs
                   << ") for worldSize " << worldSize
                   << ". Use -1 for all pairs or specify a value <= " << maxPairs << ".";
            output->recordError(errmsg.str());
            return 1;
        }
    }
#endif

    int cudaRuntimeVersion = 0;
    int cudaDriverApiVersion = 0;
    if (!validateCudaRuntimeAndDriver(output, &cudaRuntimeVersion, &cudaDriverApiVersion)) {
        return 1;
    }

    CU_ASSERT(cuInit(0));
    NVML_ASSERT(nvmlInit());
    CU_ASSERT(cuDeviceGetCount(&deviceCount));
#ifndef MULTINODE
    worldSize = deviceCount;
#endif

    if (hasGpuPair()) {
        if (normalizeUuidString(gpuPairUuid0) == normalizeUuidString(gpuPairUuid1)) {
            output->recordError("ERROR: --pair requires two distinct GPU UUIDs, got the same UUID twice: " +
                                gpuPairUuid0);
            return 1;
        }
        int idx0 = getDeviceIndexByUuid(gpuPairUuid0, deviceCount);
        int idx1 = getDeviceIndexByUuid(gpuPairUuid1, deviceCount);
#ifdef MULTINODE
        if (worldSize > 1) {
            if (worldSize != 2) {
                output->recordError("ERROR: --pair requires exactly 2 MPI ranks (one GPU per rank). Got worldSize=" +
                                    std::to_string(worldSize));
                return 1;
            }
            localDevice = (worldRank == 0) ? idx0 : idx1;
            if (localDevice < 0) {
                output->recordError("ERROR: GPU UUID not found on this node: " +
                                    (worldRank == 0 ? gpuPairUuid0 : gpuPairUuid1));
                return 1;
            }
        } else  // NOLINT(readability/braces)
#endif
        {
            if (idx0 < 0 || idx1 < 0) {
                output->recordError("ERROR: One or both GPU UUIDs not found on this machine: " +
                                    gpuPairUuid0 + ", " + gpuPairUuid1);
                return 1;
            }
            if (idx0 == idx1) {
                output->recordError("ERROR: Both UUIDs resolved to the same device index (" +
                                    std::to_string(idx0) + "); UUIDs: " +
                                    gpuPairUuid0 + ", " + gpuPairUuid1);
                return 1;
            }
            gpuPairDev0 = idx0;
            gpuPairDev1 = idx1;
        }
    }

    if (gSettings.bufferSizeMiB < kDefaultBufferSizeMiB) {
        output->recordWarning("NOTE: You have chosen a buffer size that is smaller than the default buffer size. It is suggested to use the default buffer size (512MB) to achieve maximal peak bandwidth.");
    }
#ifdef _WIN32
    if (gSettings.useHugePages) {
        // Disable huge pages on Windows
        output->recordWarning("NOTE: Huge pages are not supported on Windows. The option will be ignored.");
        gSettings.useHugePages = false;
    }
#else
    if (gSettings.useHugePages && !hugePagesEnabled()) {
        output->recordWarning("NOTE: Huge pages were requested, but Transparent Huge Pages (THP) are not enabled on this system. The option will be ignored. Enable THP with 'echo madvise > /sys/kernel/mm/transparent_hugepage/enabled' if you wish to use this feature.");
    }
#endif

    char driverVersion[NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE];
    NVML_ASSERT(nvmlSystemGetDriverVersion(driverVersion, NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE));

    gSettings.bounceBufferConfComputeEnabled = isBounceBufferConfComputeEnabled();

    output->addCudaAndDriverInfo(cudaRuntimeVersion, cudaDriverApiVersion, driverVersion);

    output->addConfComputeInfo(gSettings.bounceBufferConfComputeEnabled);

    // Print GPU information
    output->recordDevices(deviceCount);

    // Early CUDA runtime sanity check - test if we can create contexts and allocate memory
    // This catches driver/runtime issues before they manifest as confusing errors later
#ifdef MULTINODE
    // In multinode mode, only test the local GPU assigned to the process.
    const int testDeviceStart = localDevice;
    const int testDeviceEnd = localDevice + 1;
#else
    // In single-node mode, test all GPUs
    const int testDeviceStart = 0;
    const int testDeviceEnd = deviceCount;
#endif
    for (int i = testDeviceStart; i < testDeviceEnd; i++) {
        cudaError_t err = cudaSetDevice(i);
        if (err != cudaSuccess) {
            output->recordError("CUDA runtime sanity check failed: cudaSetDevice(" + std::to_string(i) + ") returned " + std::string(cudaGetErrorString(err)));
            return 1;
        }

        // Test basic memory allocation to verify that context creation works
        void* testPtr = nullptr;
        err = cudaMalloc(&testPtr, 1024);
        if (err != cudaSuccess) {
            output->recordError("CUDA runtime sanity check failed: cudaMalloc on device " + std::to_string(i) + " returned " + std::string(cudaGetErrorString(err)));
            return 1;
        }
        cudaFree(testPtr);
    }

    if (testcasePrefixes.size() > 0) {
        testcasesToRun = expandTestcases(testcases, testcasePrefixes);
        if (testcasesToRun.size() == 0) {
            output->recordError("Specified list of testcase prefixes did not match any testcases");
            return 1;
        }
    }

    if (hasGpuPair() && !selectedTestcasesIncludeGpuPairRelevant(testcases, testcasesToRun)) {
        output->recordWarning(
            "NOTE: --pair is set but none of the selected testcases are device-to-device tests that use the "
            "GPU pair (single-node memcpy/latency or multinode device-to-device). The pair filter has no effect "
            "on the current selection.");
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

    for (auto testcase : testcases) {
        delete testcase;
    }

    env->finalize();
    output->printInfo();
    return 0;
}
