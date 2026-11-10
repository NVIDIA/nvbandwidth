/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "cuda_version_check.h"
#include "inline_common.h"
#include "nvbandwidth_version.h"
#include "output.h"

#include <math.h>

#ifdef MULTINODE
#include <mpi.h>
#endif

void Output::addVersionInfo() {
    OUTPUT << "nvbandwidth Version: " << NVBANDWIDTH_VERSION << "\n";
    OUTPUT << "Built from Git version: " << GIT_VERSION << "\n\n";

#ifdef MULTINODE
    char MPIVersion[MPI_MAX_LIBRARY_VERSION_STRING];
    int MPIVersionLen;
    MPI_Get_library_version(MPIVersion, &MPIVersionLen);

    OUTPUT << "MPI version: " << MPIVersion << "\n";
#endif
}

void Output::printInfo() {
    OUTPUT << "NOTE: The reported results may not reflect the full capabilities of the platform.\n"
           << "Performance can vary with software drivers, hardware clocks, and system topology.\n\n";
}

void Output::addCudaAndDriverInfo(int cudaRuntimeVersion, int cudaDriverApiVersion,
                                  const std::string &nvmlDriverVersion) {
    OUTPUT << "CUDA Runtime Version: " << formatCudaVersionCode(cudaRuntimeVersion) << " ("
           << cudaRuntimeVersion << ")" << "\n";
    OUTPUT << "CUDA Driver Version (API): " << formatCudaVersionCode(cudaDriverApiVersion) << " ("
           << cudaDriverApiVersion << ")" << "\n";
    OUTPUT << "Driver Version: " << nvmlDriverVersion << "\n\n";
}

void Output::addConfComputeInfo(bool bounceBufferConfComputeEnabled) {
    if (bounceBufferConfComputeEnabled) {
        OUTPUT << "Detected platform using encrypted bounce buffers for CPU<->GPU traffic.\n\n";
    }
}

void Output::recordError(const std::string &error) {
    std::cerr << error << std::endl;
}

void Output::recordError(const std::vector<std::string> &errorParts) {
    bool first = true;
    for (auto &part : errorParts) {
        if (first) {
            OUTPUT << part << ":\n\n";
            first = false;
        } else {
            OUTPUT << part << "\n";
        }
    }
}

void Output::listTestcases(const std::vector<Testcase*> &testcases) {
    size_t numTestcases = testcases.size();
    OUTPUT << "Index, Name:\n\tDescription\n";
    for (size_t i = 0; i < numTestcases; i++) {
        OUTPUT << i << ", " << testcases[i]->testKey() << ":\n\t" << testcases[i]->testDesc() << "\n";
    }
}

std::string getDeviceDisplayInfo(int deviceOrdinal) {
    std::stringstream sstream;
    CUdevice dev;
    char name[STRING_LENGTH];
    int busId, deviceId, domainId;
    CUdevice device;

    CU_ASSERT(cuDeviceGet(&dev, deviceOrdinal));
    CU_ASSERT(cuDeviceGetName(name, sizeof(name), dev));
    CU_ASSERT(cuDeviceGetAttribute(&domainId, CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID, dev));
    CU_ASSERT(cuDeviceGetAttribute(&busId, CU_DEVICE_ATTRIBUTE_PCI_BUS_ID, dev));
    CU_ASSERT(cuDeviceGetAttribute(&deviceId, CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID, dev));
    sstream << name << " (" <<
        std::hex << std::setw(8) << std::setfill('0') << domainId << ":" <<
        std::hex << std::setw(2) << std::setfill('0') << busId << ":" <<
        std::hex << std::setw(2) << std::setfill('0') << deviceId << ")" <<
        std::dec << std::setfill(' ') << std::setw(0);  // reset formatting

    return sstream.str();
}

#ifdef MULTINODE
std::string formatFabricCliqueType(FabricCliqueType type) {
    switch (type) {
        case FabricCliqueType::UnicastPointer:
            return "UNICAST_POINTER";
        case FabricCliqueType::MulticastPointer:
            return "MULTICAST_POINTER";
        case FabricCliqueType::UnicastLogicalEndpoint:
            return "UNICAST_LOGICAL_ENDPOINT";
        case FabricCliqueType::MulticastLogicalEndpoint:
            return "MULTICAST_LOGICAL_ENDPOINT";
        default:
            return "UNKNOWN(" + std::to_string(static_cast<unsigned int>(type)) + ")";
    }
}

static std::string formatFabricUuid(const unsigned char uuid[kFabricClusterUuidBytes]) {
    std::stringstream s;
    for (unsigned int i = 0; i < kFabricClusterUuidBytes; i++) {
        s << std::hex << std::setw(2) << std::setfill('0') << static_cast<unsigned int>(uuid[i]);
    }
    return s.str();
}

static std::string formatFabricCliques(const RankFabricInfo &info) {
    std::stringstream s;
    s << "{";
    const unsigned int count = std::min(info.numCliques, kMaxRankFabricCliques);
    for (unsigned int i = 0; i < count; i++) {
        if (i > 0) {
            s << ", ";
        }
        s << formatFabricCliqueType(info.cliques[i].type) << ":" << info.cliques[i].id;
    }
    if (info.reportedNumCliques > info.numCliques) {
        if (count > 0) {
            s << ", ";
        }
        s << "... reported=" << info.reportedNumCliques
          << ", max=" << kMaxRankFabricCliques;
    }
    s << "}";
    return s.str();
}

#ifdef NVBW_HAVE_NVML_GPU_FABRIC_INFO
static const char *fabricStateToString(unsigned int state) {
    switch (state) {
        case NVML_GPU_FABRIC_STATE_NOT_SUPPORTED:
            return "NOT_SUPPORTED";
        case NVML_GPU_FABRIC_STATE_NOT_STARTED:
            return "NOT_STARTED";
        case NVML_GPU_FABRIC_STATE_IN_PROGRESS:
            return "IN_PROGRESS";
        case NVML_GPU_FABRIC_STATE_COMPLETED:
            return "COMPLETED";
        default:
            return "UNKNOWN";
    }
}

static const char *fabricHealthSummaryToString(unsigned char healthSummary) {
    switch (healthSummary) {
        case NVML_GPU_FABRIC_HEALTH_SUMMARY_NOT_SUPPORTED:
            return "NOT_SUPPORTED";
        case NVML_GPU_FABRIC_HEALTH_SUMMARY_HEALTHY:
            return "HEALTHY";
        case NVML_GPU_FABRIC_HEALTH_SUMMARY_UNHEALTHY:
            return "UNHEALTHY";
        case NVML_GPU_FABRIC_HEALTH_SUMMARY_LIMITED_CAPACITY:
            return "LIMITED_CAPACITY";
        default:
            return "UNKNOWN";
    }
}

static const char *fabricHealthDegradedBwToString(unsigned int value) {
    switch (value) {
        case NVML_GPU_FABRIC_HEALTH_MASK_DEGRADED_BW_NOT_SUPPORTED:
            return "NOT_SUPPORTED";
        case NVML_GPU_FABRIC_HEALTH_MASK_DEGRADED_BW_TRUE:
            return "TRUE";
        case NVML_GPU_FABRIC_HEALTH_MASK_DEGRADED_BW_FALSE:
            return "FALSE";
        default:
            return "UNKNOWN";
    }
}

static const char *fabricHealthRouteRecoveryToString(unsigned int value) {
    switch (value) {
        case NVML_GPU_FABRIC_HEALTH_MASK_ROUTE_RECOVERY_NOT_SUPPORTED:
            return "NOT_SUPPORTED";
        case NVML_GPU_FABRIC_HEALTH_MASK_ROUTE_RECOVERY_TRUE:
            return "TRUE";
        case NVML_GPU_FABRIC_HEALTH_MASK_ROUTE_RECOVERY_FALSE:
            return "FALSE";
        default:
            return "UNKNOWN";
    }
}

static const char *fabricHealthRouteUnhealthyToString(unsigned int value) {
    switch (value) {
        case NVML_GPU_FABRIC_HEALTH_MASK_ROUTE_UNHEALTHY_NOT_SUPPORTED:
            return "NOT_SUPPORTED";
        case NVML_GPU_FABRIC_HEALTH_MASK_ROUTE_UNHEALTHY_TRUE:
            return "TRUE";
        case NVML_GPU_FABRIC_HEALTH_MASK_ROUTE_UNHEALTHY_FALSE:
            return "FALSE";
        default:
            return "UNKNOWN";
    }
}

static const char *fabricHealthAccessTimeoutRecoveryToString(unsigned int value) {
    switch (value) {
        case NVML_GPU_FABRIC_HEALTH_MASK_ACCESS_TIMEOUT_RECOVERY_NOT_SUPPORTED:
            return "NOT_SUPPORTED";
        case NVML_GPU_FABRIC_HEALTH_MASK_ACCESS_TIMEOUT_RECOVERY_TRUE:
            return "TRUE";
        case NVML_GPU_FABRIC_HEALTH_MASK_ACCESS_TIMEOUT_RECOVERY_FALSE:
            return "FALSE";
        default:
            return "UNKNOWN";
    }
}

static const char *fabricHealthIncorrectConfigurationToString(unsigned int value) {
    switch (value) {
        case NVML_GPU_FABRIC_HEALTH_MASK_INCORRECT_CONFIGURATION_NOT_SUPPORTED:
            return "NOT_SUPPORTED";
        case NVML_GPU_FABRIC_HEALTH_MASK_INCORRECT_CONFIGURATION_NONE:
            return "NONE";
        case NVML_GPU_FABRIC_HEALTH_MASK_INCORRECT_CONFIGURATION_INCORRECT_SYSGUID:
            return "INCORRECT_SYSGUID";
        case NVML_GPU_FABRIC_HEALTH_MASK_INCORRECT_CONFIGURATION_INCORRECT_CHASSIS_SN:
            return "INCORRECT_CHASSIS_SN";
        case NVML_GPU_FABRIC_HEALTH_MASK_INCORRECT_CONFIGURATION_NO_PARTITION:
            return "NO_PARTITION";
        case NVML_GPU_FABRIC_HEALTH_MASK_INCORRECT_CONFIGURATION_INSUFFICIENT_NVLINKS:
            return "INSUFFICIENT_NVLINKS";
        case NVML_GPU_FABRIC_HEALTH_MASK_INCORRECT_CONFIGURATION_INCOMPATIBLE_GPU_FW:
            return "INCOMPATIBLE_GPU_FW";
        case NVML_GPU_FABRIC_HEALTH_MASK_INCORRECT_CONFIGURATION_INVALID_LOCATION:
            return "INVALID_LOCATION";
        case NVML_GPU_FABRIC_HEALTH_MASK_INCORRECT_CONFIGURATION_GPU_STATE_INVALID:
            return "GPU_STATE_INVALID";
        default:
            return "UNKNOWN";
    }
}

static const char *fabricHealthPartitionAssignedToString(unsigned int value) {
    switch (value) {
        case NVML_GPU_FABRIC_HEALTH_MASK_PARTITION_ASSIGNED_NOT_SUPPORTED:
            return "NOT_SUPPORTED";
        case NVML_GPU_FABRIC_HEALTH_MASK_PARTITION_ASSIGNED_TRUE:
            return "TRUE";
        case NVML_GPU_FABRIC_HEALTH_MASK_PARTITION_ASSIGNED_FALSE:
            return "FALSE";
        default:
            return "UNKNOWN";
    }
}

static void appendFabricHealthIssue(std::stringstream &s, bool &hasIssue,
                                    const std::string &name, const char *value) {
    if (hasIssue) {
        s << ", ";
    }
    s << name << "=" << value;
    hasIssue = true;
}

static std::string formatFabricHealthIssues(unsigned int healthMask) {
    std::stringstream s;
    bool hasIssue = false;

    const unsigned int degradedBw = NVML_GPU_FABRIC_HEALTH_GET(healthMask, _DEGRADED_BW);
    if (degradedBw == NVML_GPU_FABRIC_HEALTH_MASK_DEGRADED_BW_TRUE ||
        (degradedBw != NVML_GPU_FABRIC_HEALTH_MASK_DEGRADED_BW_FALSE &&
         degradedBw != NVML_GPU_FABRIC_HEALTH_MASK_DEGRADED_BW_NOT_SUPPORTED)) {
        appendFabricHealthIssue(s, hasIssue, "degradedBw", fabricHealthDegradedBwToString(degradedBw));
    }

    const unsigned int routeRecovery = NVML_GPU_FABRIC_HEALTH_GET(healthMask, _ROUTE_RECOVERY);
    if (routeRecovery == NVML_GPU_FABRIC_HEALTH_MASK_ROUTE_RECOVERY_TRUE ||
        (routeRecovery != NVML_GPU_FABRIC_HEALTH_MASK_ROUTE_RECOVERY_FALSE &&
         routeRecovery != NVML_GPU_FABRIC_HEALTH_MASK_ROUTE_RECOVERY_NOT_SUPPORTED)) {
        appendFabricHealthIssue(s, hasIssue, "routeRecovery", fabricHealthRouteRecoveryToString(routeRecovery));
    }

    const unsigned int routeUnhealthy = NVML_GPU_FABRIC_HEALTH_GET(healthMask, _ROUTE_UNHEALTHY);
    if (routeUnhealthy == NVML_GPU_FABRIC_HEALTH_MASK_ROUTE_UNHEALTHY_TRUE ||
        (routeUnhealthy != NVML_GPU_FABRIC_HEALTH_MASK_ROUTE_UNHEALTHY_FALSE &&
         routeUnhealthy != NVML_GPU_FABRIC_HEALTH_MASK_ROUTE_UNHEALTHY_NOT_SUPPORTED)) {
        appendFabricHealthIssue(s, hasIssue, "routeUnhealthy", fabricHealthRouteUnhealthyToString(routeUnhealthy));
    }

    const unsigned int accessTimeoutRecovery = NVML_GPU_FABRIC_HEALTH_GET(healthMask, _ACCESS_TIMEOUT_RECOVERY);
    if (accessTimeoutRecovery == NVML_GPU_FABRIC_HEALTH_MASK_ACCESS_TIMEOUT_RECOVERY_TRUE ||
        (accessTimeoutRecovery != NVML_GPU_FABRIC_HEALTH_MASK_ACCESS_TIMEOUT_RECOVERY_FALSE &&
         accessTimeoutRecovery != NVML_GPU_FABRIC_HEALTH_MASK_ACCESS_TIMEOUT_RECOVERY_NOT_SUPPORTED)) {
        appendFabricHealthIssue(s, hasIssue, "accessTimeoutRecovery",
                                fabricHealthAccessTimeoutRecoveryToString(accessTimeoutRecovery));
    }

    const unsigned int incorrectConfiguration = NVML_GPU_FABRIC_HEALTH_GET(healthMask, _INCORRECT_CONFIGURATION);
    if (incorrectConfiguration != NVML_GPU_FABRIC_HEALTH_MASK_INCORRECT_CONFIGURATION_NONE &&
        incorrectConfiguration != NVML_GPU_FABRIC_HEALTH_MASK_INCORRECT_CONFIGURATION_NOT_SUPPORTED) {
        appendFabricHealthIssue(s, hasIssue, "incorrectConfiguration",
                                fabricHealthIncorrectConfigurationToString(incorrectConfiguration));
    }

    const unsigned int partitionAssigned = NVML_GPU_FABRIC_HEALTH_GET(healthMask, _PARTITION_ASSIGNED);
    if (partitionAssigned == NVML_GPU_FABRIC_HEALTH_MASK_PARTITION_ASSIGNED_FALSE ||
        (partitionAssigned != NVML_GPU_FABRIC_HEALTH_MASK_PARTITION_ASSIGNED_TRUE &&
         partitionAssigned != NVML_GPU_FABRIC_HEALTH_MASK_PARTITION_ASSIGNED_NOT_SUPPORTED)) {
        appendFabricHealthIssue(s, hasIssue, "partitionAssigned",
                                fabricHealthPartitionAssignedToString(partitionAssigned));
    }

    if (!hasIssue) {
        return "";
    }
    return "{" + s.str() + "}";
}

static bool hasUnhealthyFabricHealth(const RankFabricInfo &info) {
    if (info.infoSource != RankFabricInfoSource::Nvml) {
        return false;
    }
    if (!formatFabricHealthIssues(info.healthMask).empty()) {
        return true;
    }
    return info.healthSummary != NVML_GPU_FABRIC_HEALTH_SUMMARY_HEALTHY &&
           info.healthSummary != NVML_GPU_FABRIC_HEALTH_SUMMARY_NOT_SUPPORTED;
}

#endif

std::string formatRankFabricInfo(const RankFabricInfo &info) {
    std::stringstream s;
    s << "rank " << info.rank << " (" << info.hostname << "), device " << info.localDevice;
    if (info.infoSource == RankFabricInfoSource::None) {
        s << ": fabric query unavailable: no provider";
        return s.str();
    }
    if (!info.runtimeSymbolAvailable) {
        if (info.infoSource == RankFabricInfoSource::Nvml) {
            s << ": NVML fabric query unavailable";
        } else {
            s << ": CUDA fabric clique query unavailable";
        }
        return s.str();
    }
    if (info.infoSource == RankFabricInfoSource::Nvml) {
#ifdef NVBW_HAVE_NVML_GPU_FABRIC_INFO
        if (info.queryResult != NVML_SUCCESS) {
            s << ": NVML fabric query failed: "
              << nvmlErrorString(static_cast<nvmlReturn_t>(info.queryResult));
            return s.str();
        }

        s << ": fabricUuid=" << formatFabricUuid(info.clusterUuid)
          << ", cliques=" << formatFabricCliques(info)
          << ", state=" << fabricStateToString(info.state);
        if (info.state == NVML_GPU_FABRIC_STATE_COMPLETED) {
            s << ", status=" << nvmlErrorString(static_cast<nvmlReturn_t>(info.fabricStatus));
        }
        s << ", health=" << fabricHealthSummaryToString(info.healthSummary);
        std::string healthIssues = formatFabricHealthIssues(info.healthMask);
        if (!healthIssues.empty() ||
            (info.healthSummary != NVML_GPU_FABRIC_HEALTH_SUMMARY_HEALTHY &&
             info.healthSummary != NVML_GPU_FABRIC_HEALTH_SUMMARY_NOT_SUPPORTED)) {
            s << ", healthMask=0x" << std::hex << info.healthMask << std::dec;
            if (!healthIssues.empty()) {
                s << ", healthIssues=" << healthIssues;
            }
        }
        return s.str();
#else
        s << ": NVML fabric query unavailable";
        return s.str();
#endif
    }

    if (info.queryResult != CUDA_SUCCESS) {
        const char *errorName = nullptr;
        const char *errorString = nullptr;
        cuGetErrorName(static_cast<CUresult>(info.queryResult), &errorName);
        cuGetErrorString(static_cast<CUresult>(info.queryResult), &errorString);
        s << ": CUDA fabric clique query failed: "
          << (errorName != nullptr ? errorName : "CUDA_ERROR_UNKNOWN");
        if (errorString != nullptr) {
            s << " (" << errorString << ")";
        }
        return s.str();
    }

    s << ": fabricUuid=" << formatFabricUuid(info.clusterUuid)
      << ", cliques=" << formatFabricCliques(info);
    return s.str();
}
#endif

void Output::recordDevices(int deviceCount) {
    OUTPUT << env->getHostname() << "\n";
    for (int iDev = 0; iDev < deviceCount; iDev++) {
        OUTPUT << "Device " << iDev << ": " << getDeviceDisplayInfo(iDev) << "\n";
    }
    OUTPUT << "\n";
}

#ifdef MULTINODE
void Output::recordMultinodeDevices(const std::vector<RankDeviceInfo> &rankDevices) {
    for (const RankDeviceInfo &info : rankDevices) {
        OUTPUT << formatRankDeviceInfo(info) << "\n";
    }
    OUTPUT << "\n";
}

void Output::recordGpuFabricInfo(const std::vector<RankFabricInfo> &fabricInfo) {
    if (gSettings.outputFormat != "text") {
        return;
    }
    bool hasFabricInfo = false;
    for (const RankFabricInfo &info : fabricInfo) {
        if (info.runtimeSymbolAvailable) {
            hasFabricInfo = true;
            break;
        }
    }
    if (!hasFabricInfo) {
        return;
    }

    OUTPUT << "Fabric cluster, clique info:\n";
    for (const RankFabricInfo &info : fabricInfo) {
        OUTPUT << formatRankFabricInfo(info) << "\n";
    }
#ifdef NVBW_HAVE_NVML_GPU_FABRIC_INFO
    bool hasUnhealthyFabricNode = false;
    for (const RankFabricInfo &info : fabricInfo) {
        if (info.runtimeSymbolAvailable &&
            info.queryResult == NVML_SUCCESS &&
            hasUnhealthyFabricHealth(info)) {
            hasUnhealthyFabricNode = true;
            break;
        }
    }
    if (hasUnhealthyFabricNode) {
        recordWarning("\nUnhealthy fabric nodes:");
        for (const RankFabricInfo &info : fabricInfo) {
            if (info.runtimeSymbolAvailable &&
                info.queryResult == NVML_SUCCESS &&
                hasUnhealthyFabricHealth(info)) {
                recordWarning(formatRankFabricInfo(info));
            }
        }
    }
#endif
    OUTPUT << "\n";
}
#endif

void Output::addTestcase(const std::string &name, const std::string &status, const std::string &msg) {
    if (status == NVB_RUNNING) {
        OUTPUT << status << " " << name << ".\n";
    } else {
        OUTPUT << status << " " << name << ": " << msg << "\n";
    }
}

void Output::setTestcaseStatusAndAddIfNeeded(const std::string &name, const std::string &status, const std::string &msg) {
    // For plain text output, the name has always been printed already and therefore isn't needed here
    OUTPUT << status << ": " << msg << "\n";
}

void Output::addTestcaseResults(const PeerValueMatrix<double> &bandwidthValues, const std::string &description) {
    OUTPUT << description << "\n";
    OUTPUT << std::fixed << std::setprecision(2) << bandwidthValues << "\n";
    OUTPUT << std::flush;
}

void Output::print() {
    // NO-OP
}

void Output::recordErrorCurrentTest(const std::string &errorLine1, const std::string &errorLine2) {
    OUTPUT << errorLine1 << "\n" << errorLine2 << "\n";
}

void Output::recordWarning(const std::string &warning) {
    OUTPUT << warning << "\n";
}

void RecordError(const std::stringstream &errmsg) {
    output->recordError(errmsg.str());
}
