/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "inline_common.h"
#include "output.h"
#include "version.h"

#include <math.h>

#ifdef MULTINODE
#include <unistd.h>
#include <mpi.h>
#include <map>
#endif

void Output::addVersionInfo() {
    OUTPUT << "nvbandwidth Version: " << NVBANDWIDTH_VERSION << std::endl;
    OUTPUT << "Built from Git version: " << GIT_VERSION << std::endl << std::endl;

#ifdef MULTINODE
    char MPIVersion[MPI_MAX_LIBRARY_VERSION_STRING];
    int MPIVersionLen;
    MPI_Get_library_version(MPIVersion, &MPIVersionLen);

    OUTPUT << "MPI version: " << MPIVersion << std::endl;
#endif
}

void Output::printInfo() {
    OUTPUT << "NOTE: The reported results may not reflect the full capabilities of the platform." << std::endl
           << "Performance can vary with software drivers, hardware clocks, and system topology." << std::endl << std::endl;
}

void Output::addCudaAndDriverInfo(int cudaVersion, const std::string &driverVersion) {
    OUTPUT << "CUDA Runtime Version: " << cudaVersion << std::endl;
    OUTPUT << "CUDA Driver Version: " << cudaVersion << std::endl;
    OUTPUT << "Driver Version: " << driverVersion << std::endl << std::endl;
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
            OUTPUT << part << std::endl;
        }
    }
}

void Output::listTestcases(const std::vector<Testcase*> &testcases) {
    size_t numTestcases = testcases.size();
    OUTPUT << "Index, Name:\n\tDescription\n";
    OUTPUT << "=======================\n";
    for (unsigned int i = 0; i < numTestcases; i++) {
        OUTPUT << i << ", " << testcases.at(i)->testKey() << ":\n" << testcases.at(i)->testDesc() << "\n\n";
    }
}

std::string getDeviceDisplayInfo(int deviceOrdinal) {
    std::stringstream sstream;
    CUdevice dev;
    char name[STRING_LENGTH];
    int busId, deviceId, domainId;

    CU_ASSERT(cuDeviceGet(&dev, deviceOrdinal));
    CU_ASSERT(cuDeviceGetName(name, STRING_LENGTH, dev));
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
static void printGPUsMultinode(int deviceCount) {
    // Exchange hostnames
    std::vector<char> hostnameExchange(worldSize * STRING_LENGTH);
    MPI_Allgather(localHostname, STRING_LENGTH, MPI_BYTE, &hostnameExchange[0], STRING_LENGTH, MPI_BYTE, MPI_COMM_WORLD);

    // Find local rank based on hostnames
    localRank = 0;
    for (int i = 0; i < worldRank; i++) {
        if (strncmp(localHostname, &hostnameExchange[i * STRING_LENGTH], STRING_LENGTH) == 0) {
            localRank++;
        }
    }

    std::vector<int> deviceCountExchange(worldSize);
    MPI_Allgather(&deviceCount, 1, MPI_INT, &deviceCountExchange[0], 1, MPI_INT, MPI_COMM_WORLD);

    localDevice = localRank % deviceCount;

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
        OUTPUT << "Process " << getPaddedProcessId(i) << " (" << &hostnameExchange[i * STRING_LENGTH] << "): device " << localDeviceIdExchange[i] << ": " << deviceName << std::endl;
    }
    OUTPUT << std::endl;
}
#endif

static void printGPUs() {
    OUTPUT << localHostname << std::endl;
    for (int iDev = 0; iDev < deviceCount; iDev++) {
        OUTPUT << "Device " << iDev << ": " << getDeviceDisplayInfo(iDev) << std::endl;
    }
    OUTPUT << std::endl;
}

void Output::recordDevices(int deviceCount) {
#ifdef MULTINODE
    printGPUsMultinode(deviceCount);
#else
    printGPUs();
#endif
}

void Output::addTestcase(const std::string &name, const std::string &status, const std::string &msg) {
    if (status == NVB_RUNNING) {
        OUTPUT << status << " " << name << ".\n";
    } else {
        OUTPUT << status << ": " << msg << std::endl;
    }
}

void Output::setTestcaseStatusAndAddIfNeeded(const std::string &name, const std::string &status, const std::string &msg) {
    // For plain text output, the name has always been printed already and therefore isn't needed here
    OUTPUT << status << ": " << msg << std::endl;
}

void Output::addTestcaseResults(const PeerValueMatrix<double> &bandwidthValues, const std::string &description) {
    OUTPUT << description << std::endl;
    OUTPUT << std::fixed << std::setprecision(2) << bandwidthValues << std::endl;
}

void Output::print() {
    // NO-OP
}

void Output::recordErrorCurrentTest(const std::string &errorLine1, const std::string &errorLine2) {
    OUTPUT << errorLine1 << std::endl << errorLine2 << std::endl;
}

void Output::recordWarning(const std::string &warning) {
    OUTPUT << warning << std::endl;
}

void RecordError(const std::stringstream &errmsg) {
    output->recordError(errmsg.str());
}
