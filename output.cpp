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
    std::cout << "nvbandwidth Version: " << NVBANDWIDTH_VERSION << std::endl;
    std::cout << "Built from Git version: " << GIT_VERSION << std::endl << std::endl;
}

void Output::printInfo() {
    std::cout << "NOTE: This tool reports current measured bandwidth on your system." << std::endl 
              << "Additional system-specific tuning may be required to achieve maximal peak bandwidth." << std::endl << std::endl;
}

void Output::addCudaAndDriverInfo(int cudaVersion, const std::string &driverVersion) {
    std::cout << "CUDA Runtime Version: " << cudaVersion << std::endl;
    std::cout << "CUDA Driver Version: " << cudaVersion << std::endl;
    std::cout << "Driver Version: " << driverVersion << std::endl << std::endl;
}

void Output::recordError(const std::string &error) {
    std::cout << error << std::endl;
}

void Output::recordError(const std::vector<std::string> &errorParts) {
    bool first = true;
    for (auto &part : errorParts) {
        if (first) {
            std::cout << part << ":\n\n";
            first = false;
        } else {
            std::cout << part << std::endl;
        }
    }
}

void Output::listTestcases(const std::vector<Testcase*> &testcases) {
    size_t numTestcases = testcases.size();
    std::cout << "Index, Name:\n\tDescription\n";
    std::cout << "=======================\n";
    for (unsigned int i = 0; i < numTestcases; i++) {
        std::cout << i << ", " << testcases.at(i)->testKey() << ":\n" << testcases.at(i)->testDesc() << "\n\n";
    }
}

void Output::recordDevices(int deviceCount) {
    for (int iDev = 0; iDev < deviceCount; iDev++) {
        CUdevice dev;
        char name[256];

        CU_ASSERT(cuDeviceGet(&dev, iDev));
        CU_ASSERT(cuDeviceGetName(name, 256, dev));

        std::cout << "Device " << iDev << ": " << name << std::endl;
    }
    std::cout << std::endl;
}

void Output::addTestcase(const std::string &name, const std::string &status, const std::string &msg) {
    if (status == NVB_RUNNING) {
        std::cout << status << " " << name << ".\n";
    } else {
        std::cout << name << " " << status << "." << std::endl << std::endl;
    }
}

void Output::setTestcaseStatusAndAddIfNeeded(const std::string &name, const std::string &status, const std::string &msg) {
    // For plain text output, the name has always been printed already and therefore isn't needed here
    std::cout << status << ": " << msg << std::endl;
}

void Output::addTestcaseResults(const PeerValueMatrix<double> &bandwidthValues, const std::string &description) {
    std::cout << description << std::endl;
    std::cout << std::fixed << std::setprecision(2) << bandwidthValues << std::endl;
}

void Output::print() {
    // NO-OP
}

void Output::recordErrorCurrentTest(const std::string &errorLine1, const std::string &errorLine2) {
    std::cout << errorLine1 << std::endl << errorLine2 << std::endl;
}

void Output::recordWarning(const std::string &warning) {
    recordError(warning);
}
