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
#include <algorithm>
#include <cstring>
#include <iostream>
#include <map>
#ifndef _WIN32
#include <dlfcn.h>
#endif

#include "cuda_version_check.h"
#include "environment.h"
#include "json_output.h"
#include "perf_parser_output.h"
#include "kernels.cuh"
#include "memcpy_le.h"
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
// Number of ranks sharing this node (ranks per node)
int ranksPerNode = 1;
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
        new DeviceLocalCopyTransfer(CopyInitiator::TMA),
        new DeviceToDeviceLatency(CountedWrite::Yes),
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
            new MultinodeMulticastFabricPut(CopyInitiator::TMA),
            new MultinodeMultimemReduce()
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


#ifdef MULTINODE
static void addRequiredFabricCliqueTypesIfRunnable(Testcase *testcase, std::vector<FabricCliqueType> &types) {
    if (testcase->filter()) {
        testcase->addRequiredFabricCliqueTypes(types);
    }
}

static std::vector<FabricCliqueType> getRequiredFabricCliqueTypesForSelectedTests(
        std::vector<Testcase*> &testcases, const std::vector<std::string> &testcasesToRun) {
    std::vector<FabricCliqueType> types;
    if (testcasesToRun.empty()) {
        for (Testcase *testcase : testcases) {
            addRequiredFabricCliqueTypesIfRunnable(testcase, types);
        }
        return types;
    }

    for (const std::string &id : testcasesToRun) {
        try {
            Testcase *testcase = findTestcase(testcases, id);
            addRequiredFabricCliqueTypesIfRunnable(testcase, types);
        } catch (const std::string &) {
            continue;
        }
    }
    return types;
}
#endif

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
            test->run(gSettings.effectiveBufferSizeMiB(test->testKey()) * _MiB, gSettings.loopCount);
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

#ifdef MULTINODE
enum FabricInfoProviderCapability {
    kFabricInfoProviderCapabilityCuda = 1 << 0,
    kFabricInfoProviderCapabilityNvml = 1 << 1,
};

static int getLocalFabricInfoProviderCapabilities() {
    int capabilities = 0;
#ifdef NVBW_HAVE_CUDA_FABRIC_CLIQUE_INFO
    capabilities |= kFabricInfoProviderCapabilityCuda;
#endif
#ifdef NVBW_HAVE_NVML_GPU_FABRIC_INFO
    capabilities |= kFabricInfoProviderCapabilityNvml;
#endif
    return capabilities;
}

static std::vector<int> getAllRankFabricInfoProviderCapabilities() {
    int localCapabilities = getLocalFabricInfoProviderCapabilities();
    std::vector<int> allCapabilities(worldSize);
    MPI_Allgather(&localCapabilities, 1, MPI_INT,
                  allCapabilities.data(), 1, MPI_INT, MPI_COMM_WORLD);
    return allCapabilities;
}

static bool allRanksHaveFabricInfoProvider(const std::vector<int> &allCapabilities,
                                           FabricInfoProviderCapability capability) {
    for (int capabilities : allCapabilities) {
        if ((capabilities & capability) == 0) {
            return false;
        }
    }
    return true;
}

class FabricInfoProvider {
 public:
    virtual ~FabricInfoProvider() = default;

    virtual RankFabricInfoSource infoSource() const = 0;
    virtual RankFabricInfo getLocalRankFabricInfo() const = 0;
    virtual const char *missingRuntimeSymbolWarning() const = 0;
    virtual const char *validationFailureHeader() const = 0;
    virtual bool isRankValidForCliqueValidation(const RankFabricInfo &info) const = 0;
    virtual std::string rankValidationFailure(const RankFabricInfo &info) const = 0;
    virtual size_t clusterUuidBytes() const { return kFabricClusterUuidBytes; }
    virtual bool fallbackRecommended(const std::vector<RankFabricInfo> &allInfo) const {
        (void)allInfo;
        return false;
    }

 protected:
    RankFabricInfo makeLocalRankFabricInfo() const {
        RankFabricInfo info = {};
        info.rank = worldRank;
        info.localDevice = localDevice;
        info.infoSource = infoSource();
        copyToFixedBuffer(info.hostname, env->getHostname());
        return info;
    }

    template <typename CliqueInfoT>
    void copyFabricCliques(RankFabricInfo &info,
                           const CliqueInfoT *cliques,
                           size_t numCliques) const {
        info.reportedNumCliques = static_cast<unsigned long long>(numCliques);
        info.numCliques = static_cast<unsigned int>(
            std::min(numCliques, static_cast<size_t>(kMaxRankFabricCliques)));
        for (unsigned int i = 0; i < info.numCliques; i++) {
            info.cliques[i].type = static_cast<FabricCliqueType>(cliques[i].type);
            info.cliques[i].id = cliques[i].id;
        }
    }
};

#ifdef NVBW_HAVE_CUDA_FABRIC_CLIQUE_INFO
class CudaFabricInfoProvider : public FabricInfoProvider {
 public:
    RankFabricInfoSource infoSource() const override { return RankFabricInfoSource::CudaClique; }

    RankFabricInfo getLocalRankFabricInfo() const override {
        RankFabricInfo info = makeLocalRankFabricInfo();
        info.queryResult = CUDA_ERROR_NOT_SUPPORTED;
#ifndef _WIN32
        void *uuidSym = dlsym(RTLD_DEFAULT, "cuDeviceGetFabricClusterUuid");
        void *cliqueInfoSym = dlsym(RTLD_DEFAULT, "cuDeviceGetCliqueInfo");
        if (uuidSym == nullptr || cliqueInfoSym == nullptr) {
            return info;
        }
        info.runtimeSymbolAvailable = 1;

        auto cuDeviceGetFabricClusterUuidFn =
            reinterpret_cast<CUresult (*)(CUuuid *, CUdevice)>(uuidSym);
        auto cuDeviceGetCliqueInfoFn =
            reinterpret_cast<CUresult (*)(CUcliqueInfo *, size_t *, CUdevice)>(cliqueInfoSym);
#else
        info.runtimeSymbolAvailable = 1;
        auto cuDeviceGetFabricClusterUuidFn = &cuDeviceGetFabricClusterUuid;
        auto cuDeviceGetCliqueInfoFn = &cuDeviceGetCliqueInfo;
#endif

        CUdevice device;
        CUresult cudaResult = cuDeviceGet(&device, localDevice);
        if (cudaResult != CUDA_SUCCESS) {
            info.queryResult = cudaResult;
            return info;
        }

        CUuuid clusterUuid = {};
        cudaResult = cuDeviceGetFabricClusterUuidFn(&clusterUuid, device);
        info.queryResult = cudaResult;
        if (cudaResult != CUDA_SUCCESS) {
            return info;
        }
        std::memcpy(info.clusterUuid, clusterUuid.bytes, kFabricClusterUuidBytes);

        CUcliqueInfo cliqueInfo[kMaxRankFabricCliques] = {};
        size_t cliqueCount = kMaxRankFabricCliques;
        cudaResult = cuDeviceGetCliqueInfoFn(cliqueInfo, &cliqueCount, device);
        info.queryResult = cudaResult;
        if (cudaResult != CUDA_SUCCESS) {
            return info;
        }

        copyFabricCliques(info, cliqueInfo, cliqueCount);
        return info;
    }

    const char *missingRuntimeSymbolWarning() const override {
        return "NOTE: CUDA fabric clique query APIs are not available at runtime; skipping multinode fabric clique "
               "pre-flight validation. Cross-clique runs may fail later with CUDA_ERROR_NOT_PERMITTED.";
    }

    const char *validationFailureHeader() const override {
        return "ERROR: Multinode fabric pre-flight validation failed. All MPI ranks must be in the "
               "same NVLink fabric clique for each selected operation type before nvbandwidth can run multinode "
               "tests.";
    }

    bool isRankValidForCliqueValidation(const RankFabricInfo &info) const override {
        return info.queryResult == CUDA_SUCCESS;
    }

    std::string rankValidationFailure(const RankFabricInfo &info) const override {
        return "unable to query CUDA fabric clique info for " + formatRankFabricInfo(info);
    }

    bool fallbackRecommended(const std::vector<RankFabricInfo> &allInfo) const override {
        for (const RankFabricInfo &info : allInfo) {
            if (!info.runtimeSymbolAvailable ||
                info.queryResult == CUDA_ERROR_NOT_SUPPORTED) {
                return true;
            }
        }
        return false;
    }
};
#endif

#ifdef NVBW_HAVE_NVML_GPU_FABRIC_INFO
class NvmlFabricInfoProvider : public FabricInfoProvider {
 public:
    RankFabricInfoSource infoSource() const override { return RankFabricInfoSource::Nvml; }

    RankFabricInfo getLocalRankFabricInfo() const override {
        RankFabricInfo info = makeLocalRankFabricInfo();
        info.queryResult = NVML_ERROR_NOT_SUPPORTED;
        info.fabricStatus = NVML_ERROR_NOT_SUPPORTED;
        info.state = NVML_GPU_FABRIC_STATE_NOT_SUPPORTED;
        info.healthSummary = NVML_GPU_FABRIC_HEALTH_SUMMARY_NOT_SUPPORTED;
#ifndef _WIN32
        void *sym = dlsym(RTLD_DEFAULT, "nvmlDeviceGetGpuFabricInfo_v4");
        if (sym == nullptr) {
            return info;
        }
        info.runtimeSymbolAvailable = 1;

        auto getGpuFabricInfoV4Fn =
            reinterpret_cast<nvmlReturn_t (*)(nvmlDevice_t, nvmlGpuFabricInfo_v4_t *)>(sym);

        nvmlDevice_t device;
        nvmlReturn_t nvmlResult = nvmlDeviceGetHandleByUUID(getDeviceUuidString(localDevice).c_str(), &device);
        if (nvmlResult != NVML_SUCCESS) {
            info.queryResult = nvmlResult;
            return info;
        }

        nvmlGpuFabricInfo_v4_t fabricInfo = {};
        nvmlResult = getGpuFabricInfoV4Fn(device, &fabricInfo);
        info.queryResult = nvmlResult;
        if (nvmlResult != NVML_SUCCESS) {
            return info;
        }

        std::memcpy(info.clusterUuid, fabricInfo.clusterUuid, NVML_GPU_FABRIC_UUID_LEN);
        info.fabricStatus = fabricInfo.status;
        copyFabricCliques(info, fabricInfo.cliques, fabricInfo.numCliques);
        info.state = fabricInfo.state;
        info.healthMask = fabricInfo.healthMask;
        info.healthSummary = fabricInfo.healthSummary;
#endif

        return info;
    }

    const char *missingRuntimeSymbolWarning() const override {
        return "NOTE: nvmlDeviceGetGpuFabricInfo_v4 is not available at runtime; skipping NVML v4 fabric clique "
               "pre-flight validation. Cross-clique runs may fail later with CUDA_ERROR_NOT_PERMITTED.";
    }

    const char *validationFailureHeader() const override {
        return "ERROR: Multinode fabric pre-flight validation failed. All MPI ranks must be in the "
               "same completed NVLink fabric clique for each selected operation type before nvbandwidth can run "
               "multinode tests.";
    }

    bool isRankValidForCliqueValidation(const RankFabricInfo &info) const override {
        return info.queryResult == NVML_SUCCESS &&
               info.state == NVML_GPU_FABRIC_STATE_COMPLETED &&
               info.fabricStatus == NVML_SUCCESS;
    }

    std::string rankValidationFailure(const RankFabricInfo &info) const override {
        if (info.queryResult != NVML_SUCCESS) {
            return "unable to query fabric info for " + formatRankFabricInfo(info);
        }
        if (info.state != NVML_GPU_FABRIC_STATE_COMPLETED) {
            return "fabric state is not COMPLETED for " + formatRankFabricInfo(info);
        }
        if (info.fabricStatus != NVML_SUCCESS) {
            return "fabric registration status is not successful for " + formatRankFabricInfo(info);
        }
        return "";
    }

    size_t clusterUuidBytes() const override {
        return std::min(static_cast<size_t>(kFabricClusterUuidBytes),
                        static_cast<size_t>(NVML_GPU_FABRIC_UUID_LEN));
    }
};
#endif

static bool getFabricCliqueId(const RankFabricInfo &info, FabricCliqueType type, unsigned int *id) {
    const unsigned int count = std::min(info.numCliques, kMaxRankFabricCliques);
    for (unsigned int i = 0; i < count; i++) {
        if (info.cliques[i].type == type) {
            *id = info.cliques[i].id;
            return true;
        }
    }
    return false;
}

static bool hasTooManyFabricCliques(const RankFabricInfo &info) {
    return info.reportedNumCliques > kMaxRankFabricCliques;
}

static std::string formatTooManyFabricCliquesFailure(const RankFabricInfo &info) {
    return "rank reports " + std::to_string(info.reportedNumCliques) +
           " fabric clique types, but nvbandwidth can record at most " +
           std::to_string(kMaxRankFabricCliques) + ": " + formatRankFabricInfo(info);
}

static bool sameFabricClique(const FabricInfoProvider &provider,
                             const RankFabricInfo &a,
                             const RankFabricInfo &b,
                             FabricCliqueType type) {
    unsigned int aCliqueId = 0;
    unsigned int bCliqueId = 0;
    return getFabricCliqueId(a, type, &aCliqueId) &&
           getFabricCliqueId(b, type, &bCliqueId) &&
           aCliqueId == bCliqueId &&
           std::memcmp(a.clusterUuid, b.clusterUuid, provider.clusterUuidBytes()) == 0;
}

static std::vector<RankFabricInfo> getAllRankFabricInfo(const FabricInfoProvider &provider) {
    RankFabricInfo localInfo = provider.getLocalRankFabricInfo();
    std::vector<RankFabricInfo> allInfo(worldSize);
    MPI_Allgather(&localInfo, sizeof(RankFabricInfo), MPI_BYTE,
                  allInfo.data(), sizeof(RankFabricInfo), MPI_BYTE, MPI_COMM_WORLD);
    return allInfo;
}

static bool validateMultinodeFabricClique(const FabricInfoProvider &provider,
                                          const std::vector<RankFabricInfo> &allInfo,
                                          const std::vector<FabricCliqueType> &requiredCliqueTypes) {
    if (worldSize <= 1) {
        return true;
    }

    int validationFailed = 0;
    if (worldRank == 0) {
        bool missingInfoProvider = false;
        bool missingRuntimeSymbol = false;
        for (const RankFabricInfo &info : allInfo) {
            if (info.infoSource == RankFabricInfoSource::None) {
                missingInfoProvider = true;
                break;
            }
            if (!info.runtimeSymbolAvailable) {
                missingRuntimeSymbol = true;
                break;
            }
        }

        if (missingInfoProvider) {
            output->recordWarning(
                "NOTE: At least one MPI rank has no fabric info provider; skipping multinode fabric clique "
                "pre-flight validation. Cross-clique runs may fail later with CUDA_ERROR_NOT_PERMITTED.");
        } else if (missingRuntimeSymbol) {
            output->recordWarning(provider.missingRuntimeSymbolWarning());
        } else {
            std::vector<std::string> failures;
            const RankFabricInfo &rank0Info = allInfo[0];
            const bool rank0Valid = provider.isRankValidForCliqueValidation(rank0Info) &&
                                    !hasTooManyFabricCliques(rank0Info);
            if (rank0Valid) {
                for (FabricCliqueType type : requiredCliqueTypes) {
                    unsigned int rank0CliqueId = 0;
                    if (!getFabricCliqueId(rank0Info, type, &rank0CliqueId)) {
                        failures.push_back("rank 0 does not report required fabric clique type " +
                                           formatFabricCliqueType(type));
                    }
                }
            }

            for (const RankFabricInfo &info : allInfo) {
                if (hasTooManyFabricCliques(info)) {
                    failures.push_back(formatTooManyFabricCliquesFailure(info));
                    continue;
                }
                if (!provider.isRankValidForCliqueValidation(info)) {
                    failures.push_back(provider.rankValidationFailure(info));
                    continue;
                }
                if (!rank0Valid) {
                    continue;
                }
                for (FabricCliqueType type : requiredCliqueTypes) {
                    unsigned int rankCliqueId = 0;
                    if (!getFabricCliqueId(info, type, &rankCliqueId)) {
                        failures.push_back("rank does not report required fabric clique type " +
                                           formatFabricCliqueType(type) + ": " + formatRankFabricInfo(info));
                        continue;
                    }
                    if (!sameFabricClique(provider, rank0Info, info, type)) {
                        failures.push_back("rank is outside rank 0's " + formatFabricCliqueType(type) +
                                           " clique: " + formatRankFabricInfo(info));
                    }
                }
            }

            if (!failures.empty()) {
                validationFailed = 1;
                std::stringstream errmsg;
                errmsg << provider.validationFailureHeader() << "\n";
                errmsg << "Required clique types:";
                for (FabricCliqueType type : requiredCliqueTypes) {
                    errmsg << " " << formatFabricCliqueType(type);
                }
                errmsg << "\n";
                for (const RankFabricInfo &info : allInfo) {
                    errmsg << "  " << formatRankFabricInfo(info) << "\n";
                }
                errmsg << "Failures:\n";
                for (const std::string &failure : failures) {
                    errmsg << "  - " << failure << "\n";
                }
                output->recordError(errmsg.str());
            }
        }
    }

    MPI_Bcast(&validationFailed, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return validationFailed == 0;
}

#ifdef NVBW_HAVE_NVML_GPU_FABRIC_INFO
static bool checkFabricCliquesWithNvml(const std::vector<FabricCliqueType> &requiredCliqueTypes) {
    NvmlFabricInfoProvider provider;
    std::vector<RankFabricInfo> allFabricInfo = getAllRankFabricInfo(provider);

    output->recordGpuFabricInfo(allFabricInfo);
    return validateMultinodeFabricClique(provider, allFabricInfo, requiredCliqueTypes);
}
#endif

enum class FabricCliqueCheckResult {
    Passed,
    Failed,
    FallBack,
};

#ifdef NVBW_HAVE_CUDA_FABRIC_CLIQUE_INFO
static FabricCliqueCheckResult checkFabricCliquesWithCuda(
        const std::vector<FabricCliqueType> &requiredCliqueTypes,
        bool nvmlFallbackAvailable) {
    CudaFabricInfoProvider provider;
    std::vector<RankFabricInfo> allFabricInfo = getAllRankFabricInfo(provider);

    if (provider.fallbackRecommended(allFabricInfo)) {
        if (worldRank == 0) {
            if (nvmlFallbackAvailable) {
                output->recordWarning(
                    "NOTE: CUDA fabric clique APIs are unavailable or not supported on at least one rank; "
                    "trying NVML fabric-info fallback.");
            } else {
                output->recordWarning(
                    "NOTE: CUDA fabric clique APIs are unavailable or not supported on at least one rank, "
                    "and no common NVML v4 fabric-info fallback is available.");
            }
        }
        return FabricCliqueCheckResult::FallBack;
    }

    output->recordGpuFabricInfo(allFabricInfo);
    return validateMultinodeFabricClique(provider, allFabricInfo, requiredCliqueTypes)
        ? FabricCliqueCheckResult::Passed
        : FabricCliqueCheckResult::Failed;
}
#endif

static bool checkMultinodeFabricCliques(const std::vector<FabricCliqueType> &requiredCliqueTypes,
                                        bool forceNvmlFabricInfoFallback) {
    if (worldSize <= 1) {
        return true;
    }

    std::vector<int> providerCapabilities = getAllRankFabricInfoProviderCapabilities();
    const bool allRanksHaveCuda =
        allRanksHaveFabricInfoProvider(providerCapabilities, kFabricInfoProviderCapabilityCuda);
    const bool allRanksHaveNvml =
        allRanksHaveFabricInfoProvider(providerCapabilities, kFabricInfoProviderCapabilityNvml);

#ifdef NVBW_HAVE_CUDA_FABRIC_CLIQUE_INFO
    if (!forceNvmlFabricInfoFallback && allRanksHaveCuda) {
        switch (checkFabricCliquesWithCuda(requiredCliqueTypes, allRanksHaveNvml)) {
            case FabricCliqueCheckResult::Passed:
                return true;
            case FabricCliqueCheckResult::Failed:
                return false;
            case FabricCliqueCheckResult::FallBack:
                break;
        }
    } else if (forceNvmlFabricInfoFallback && worldRank == 0) {
        if (allRanksHaveNvml) {
            output->recordWarning(
                "NOTE: --force-nvml-fabric-info-fallback is set; skipping CUDA fabric clique APIs and using "
                "NVML fabric-info fallback.");
        } else {
            output->recordWarning(
                "NOTE: --force-nvml-fabric-info-fallback is set; skipping CUDA fabric clique APIs, "
                "but no common NVML v4 fabric-info fallback is available.");
        }
    }
#else
    (void)forceNvmlFabricInfoFallback;
    (void)allRanksHaveCuda;
    (void)allRanksHaveNvml;
#endif

#ifdef NVBW_HAVE_NVML_GPU_FABRIC_INFO
    if (allRanksHaveNvml) {
        if (worldRank == 0) {
            output->recordWarning(
                "NOTE: Using NVML v4 fabric-info fallback. This validates per-operation clique types plus "
                "fabric state and status.");
        }
        return checkFabricCliquesWithNvml(requiredCliqueTypes);
    }
#endif

    if (worldRank == 0) {
        output->recordWarning(
            "NOTE: At least one MPI rank has no fabric info provider; skipping multinode fabric clique "
            "pre-flight validation. Cross-clique runs may fail later with CUDA_ERROR_NOT_PERMITTED.");
    }
    return true;
}

static bool checkMultinodeFabricCliquesForSelectedTests(
        std::vector<Testcase*> &testcases,
        const std::vector<std::string> &testcasesToRun,
        bool forceNvmlFabricInfoFallback) {
    std::vector<FabricCliqueType> requiredFabricCliqueTypes =
        getRequiredFabricCliqueTypesForSelectedTests(testcases, testcasesToRun);
    return checkMultinodeFabricCliques(requiredFabricCliqueTypes, forceNvmlFabricInfoFallback);
}

// MPI exchange of each rank's assigned GPU for device listing.
static std::vector<RankDeviceInfo> gatherRankDevices() {
    RankDeviceInfo localInfo = {};
    localInfo.rank = worldRank;
    localInfo.localDevice = localDevice;
    std::string hostname = env->getHostname();
    ASSERT(hostname.size() < STRING_LENGTH);
    copyToFixedBuffer(localInfo.hostname, hostname);
    std::string deviceName = getDeviceDisplayInfo(localDevice);
    ASSERT(deviceName.size() < STRING_LENGTH);
    copyToFixedBuffer(localInfo.deviceName, deviceName);

    std::vector<RankDeviceInfo> rankDevices(worldSize);
    ASSERT(MPI_Allgather(&localInfo, sizeof(RankDeviceInfo), MPI_BYTE,
                         rankDevices.data(), sizeof(RankDeviceInfo), MPI_BYTE, MPI_COMM_WORLD) == MPI_SUCCESS);

    for (RankDeviceInfo &info : rankDevices) {
        info.hostname[STRING_LENGTH - 1] = '\0';
        info.deviceName[STRING_LENGTH - 1] = '\0';
    }

    std::vector<int> deviceCountExchange(worldSize);
    ASSERT(MPI_Allgather(&deviceCount, 1, MPI_INT, deviceCountExchange.data(), 1, MPI_INT, MPI_COMM_WORLD) == MPI_SUCCESS);

    std::map<std::string, int> gpuCounts;
    for (int i = 0; i < worldSize; i++) {
        std::string host(rankDevices[i].hostname);
        gpuCounts[host]++;
        if (gpuCounts[host] == deviceCountExchange[i] + 1) {
            std::stringstream warning;
            warning << "Warning: there are more processes than GPUs on " << host
                    << ". Please reduce number of processes to match GPU count.";
            output->recordWarning(warning.str());
        }
    }

    return rankDevices;
}

// Rank→GPU ownership is setup, not reporting — set before any device listing.
static void assignLocalDevice() {
    if (!hasGpuPair()) {
        localDevice = localRank % deviceCount;
    }
}
#endif  // MULTINODE

int main(int argc, char **argv) {
    env = Environment::create(argc, argv);
    env->initialize(argc, argv);

#ifdef MULTINODE
    worldSize = env->getSize();
    worldRank = env->getRank();
    localRank = env->getLocalRank();
    ranksPerNode = env->getRanksPerNode();
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
    program.add_argument("--force-le-variant")
        .help("Force Logical Endpoint (Handle) variant for TMA tests (auto-selected on future architectures). "
              "Peer buffer is the LE; local/executing buffer is a raw device pointer "
              "(Write: ptr->LE, Read: LE->ptr; LE-LE unsupported). CE always uses raw pointers.")
        .flag()
        .default_value(false);
    program.add_argument("--force-raw-pointer")
        .help("Force raw pointer variant for TMA tests (auto-selected on Blackwell and older)")
        .flag()
        .default_value(false);
    bool forceNvmlFabricInfoFallback = false;
    program.add_argument("--force-nvml-fabric-info-fallback")
        .help("Force NVML fabric-info fallback for multinode fabric pre-flight validation")
        .flag()
        .store_into(forceNvmlFabricInfoFallback);

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

    gSettings.bufferSizeUserOverride = program.is_used("--bufferSize");

    const bool forceLE = program.get<bool>("--force-le-variant");
    const bool forceRaw = program.get<bool>("--force-raw-pointer");
    if (forceLE && forceRaw) {
        output->recordError("--force-le-variant and --force-raw-pointer are mutually exclusive");
        output->print();
        return 1;
    }
    if (forceLE) {
        gSettings.bufferType = BufferType::LE;
    } else if (forceRaw) {
        gSettings.bufferType = BufferType::Pointer;
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

    if (!forceLE && !forceRaw && leBufferSupportedOnAllDevices()) {
        bool allLeDefaultArch = (deviceCount > 0);
        for (int i = 0; i < deviceCount && allLeDefaultArch; i++) {
            nvmlDevice_t nvmlDev;
            char pciBusId[NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE] = {};
            nvmlDeviceArchitecture_t arch = NVML_DEVICE_ARCH_UNKNOWN;
            if (cuDeviceGetPCIBusId(pciBusId, sizeof(pciBusId), i) == CUDA_SUCCESS &&
                nvmlDeviceGetHandleByPciBusId(pciBusId, &nvmlDev) == NVML_SUCCESS) {
                nvmlDeviceGetArchitecture(nvmlDev, &arch);
            }
            if (arch != NVML_DEVICE_ARCH_RUBIN) allLeDefaultArch = false;
        }
        if (allLeDefaultArch) gSettings.bufferType = BufferType::LE;
    }

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

#ifdef MULTINODE
    if (worldSize > 1) {
        int hasDevice = deviceCount > 0 ? 1 : 0;
        int allRanksHaveDevice = 0;
        MPI_Allreduce(&hasDevice, &allRanksHaveDevice, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        if (!allRanksHaveDevice) {
            if (worldRank == 0) {
                output->recordError("ERROR: At least one MPI rank has no CUDA device. Multinode mode requires one GPU per rank.");
            }
            env->finalize();
            return 1;
        }
        assignLocalDevice();
    }
#endif

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

    if (testcasePrefixes.size() > 0) {
        testcasesToRun = expandTestcases(testcases, testcasePrefixes);
        if (testcasesToRun.size() == 0) {
            output->recordError("Specified list of testcase prefixes did not match any testcases");
            return 1;
        }
    }

#ifdef MULTINODE
    if (worldSize > 1) {
        output->recordMultinodeDevices(gatherRankDevices());
    } else {
        output->recordDevices(deviceCount);
    }
#else
    output->recordDevices(deviceCount);
#endif

#ifdef MULTINODE
    if (!checkMultinodeFabricCliquesForSelectedTests(testcases, testcasesToRun,
                                                     forceNvmlFabricInfoFallback)) {
        env->finalize();
        return 1;
    }
#endif

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
