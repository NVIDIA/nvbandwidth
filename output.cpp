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
#include <unistd.h>
#include <mpi.h>
#include <map>
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
// Exchange and print information about all devices in MPI world
// Through this process each process learns about GPUs of other processes, as well as,
// determines its own GPU index
// Each process is allocated a dedicated GPU. It is advisable to initiate NUM_GPU processes per system,
// with each process autonomously selecting a GPU to utilize. To determine this selection,
// processes exchange their hostnames, and look for duplicates of own hostname among processes with lower value of worldRank.
// localRank is equal to number of processes with the same hostname, but lower worldRank.
static void printGPUsMultinode() {
    // Exchange hostnames
    char hostname[STRING_LENGTH];
    gethostname(hostname, STRING_LENGTH);
    std::vector<char> hostnameExchange(worldSize * STRING_LENGTH, 0);
    MPI_Allgather(hostname, STRING_LENGTH, MPI_BYTE, &hostnameExchange[0], STRING_LENGTH, MPI_BYTE, MPI_COMM_WORLD);

    if (!hasGpuPair()) {
        localDevice = localRank % deviceCount;
    }
    // else localDevice was already set in main from --pair UUID resolution

    std::vector<int> deviceCountExchange(worldSize);
    MPI_Allgather(&deviceCount, 1, MPI_INT, &deviceCountExchange[0], 1, MPI_INT, MPI_COMM_WORLD);

    // It's not recommended to run more ranks per node than GPU count, but we want to make sure we handle it gracefully
    std::map<std::string, int> gpuCounts;
    for (int i = 0; i < worldSize; i++) {
        std::string host(&hostnameExchange[i * STRING_LENGTH]);
        gpuCounts[host]++;
        if (gpuCounts[host] == deviceCountExchange[i] + 1) {
            // Unconditionally emitting a warning, once per node
            std::stringstream warning;
            warning << "Warning: there are more processes than GPUs on " << host << ". Please reduce number of processes to match GPU count.";
            output->recordWarning(warning.str());
        }
    }
    // Exchange device names
    std::string localDeviceName = getDeviceDisplayInfo(localDevice);
    ASSERT(localDeviceName.size() < STRING_LENGTH);
    localDeviceName.resize(STRING_LENGTH);

    std::vector<char> deviceNameExchange(worldSize * STRING_LENGTH, 0);
    MPI_Allgather(&localDeviceName[0], STRING_LENGTH, MPI_BYTE, &deviceNameExchange[0], STRING_LENGTH, MPI_BYTE, MPI_COMM_WORLD);

    // Exchange device ids
    std::vector<int> localDeviceIdExchange(worldSize, -1);
    MPI_Allgather(&localDevice, 1, MPI_INT, &localDeviceIdExchange[0], 1, MPI_INT, MPI_COMM_WORLD);

    // Print gathered info
    for (int i = 0; i < worldSize; i++) {
        char *deviceName = &deviceNameExchange[i * STRING_LENGTH];
        OUTPUT << "Process " << getPaddedProcessId(i) << " (" << &hostnameExchange[i * STRING_LENGTH] << "): device "
               << localDeviceIdExchange[i] << ": " << deviceName << "\n";
    }
    OUTPUT << "\n";
}
#endif

static void printGPUs() {
    OUTPUT << env->getHostname() << "\n";
    for (int iDev = 0; iDev < deviceCount; iDev++) {
        OUTPUT << "Device " << iDev << ": " << getDeviceDisplayInfo(iDev) << "\n";
    }
    OUTPUT << "\n";
}

void Output::recordDevices(int worldSize) {
    if (worldSize > 1) {
#ifdef MULTINODE
        printGPUsMultinode();
#endif
    } else {
        printGPUs();
    }
}

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
