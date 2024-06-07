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


void Output::addVersionInfo() {
    OUTPUT << "nvbandwidth Version: " << NVBANDWIDTH_VERSION << std::endl;
    OUTPUT << "Built from Git version: " << GIT_VERSION << std::endl << std::endl;
}

void Output::printInfo() {
    OUTPUT << "NOTE: This tool reports current measured bandwidth on your system." << std::endl 
              << "Additional system-specific tuning may be required to achieve maximal peak bandwidth." << std::endl << std::endl;
}

void Output::addCudaAndDriverInfo(int cudaVersion, const std::string &driverVersion) {
    OUTPUT << "CUDA Runtime Version: " << cudaVersion << std::endl;
    OUTPUT << "CUDA Driver Version: " << cudaVersion << std::endl;
    OUTPUT << "Driver Version: " << driverVersion << std::endl << std::endl;
}

void Output::recordError(const std::string &error) {
    OUTPUT << error << std::endl;
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

static void printGPUs(){
    for (int iDev = 0; iDev < deviceCount; iDev++) {
        CUdevice dev;
        char name[STRING_LENGTH];

        CU_ASSERT(cuDeviceGet(&dev, iDev));
        CU_ASSERT(cuDeviceGetName(name, STRING_LENGTH, dev));

        OUTPUT << "Device " << iDev << ": " << name << std::endl;
    }
    OUTPUT << std::endl;
}

void Output::recordDevices(int deviceCount) {
    printGPUs();
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
    recordError(warning);
}

void RecordError(const std::stringstream &errmsg) {
    output->recordError(errmsg.str());
}
