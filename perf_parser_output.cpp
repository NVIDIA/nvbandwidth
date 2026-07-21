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

#include "perf_parser_output.h"
#include "cuda_version_check.h"
#include "inline_common.h"
#include "nvbandwidth_version.h"
#include <iostream>

PerfParserOutput::PerfParserOutput(bool _shouldOutput) {
    shouldOutput = _shouldOutput;
    testFailed = false;
}

void PerfParserOutput::addTestcase(const std::string &name, const std::string &status, const std::string &msg) {
    currentTestName = name;
}

void PerfParserOutput::setTestcaseStatusAndAddIfNeeded(const std::string &name, const std::string &status, const std::string &msg) {
    // Not needed for perf parser output
}

void PerfParserOutput::addTestcaseResults(const PeerValueMatrix<double> &matrix, const std::string &description) {
    // Iterate through each value in the matrix
    for (int row = 0; row < matrix.m_rows; row++) {
        for (int col = 0; col < matrix.m_columns; col++) {
            std::optional<double> val = matrix.value(row, col);

            if (!val.has_value()) {
                continue;
            }

            TestResult result;

            result.name = "nvbandwidth/" + currentTestName + "/GPU" + std::to_string(row) + "/GPU" + std::to_string(col);

            result.value = val.value();
            result.unitType = matrix.uType;
            testResults.push_back(result);
        }
    }
}

void PerfParserOutput::print() {
    if (!shouldOutput) return;

    for (const auto& result : testResults) {
        std::cout << "&&&& PERF " << result.name << " " << result.value << getUnitString(result.unitType) << "\n";
    }

    if (!testFailed) {
        std::cout << "&&&& nvbandwidth test PASSED\n";
    } else {
        std::cout << "&&&& nvbandwidth test FAILED\n";
    }
}

void PerfParserOutput::addVersionInfo() {
    if (!shouldOutput) return;
    std::cout << "nvbandwidth Version: " << NVBANDWIDTH_VERSION << "\n";
    std::cout << "Built from Git version: " << GIT_VERSION << "\n";
}

void PerfParserOutput::addCudaAndDriverInfo(int cudaRuntimeVersion, int cudaDriverApiVersion,
                                            const std::string &nvmlDriverVersion) {
    if (!shouldOutput) return;
    std::cout << "CUDA Runtime Version: " << formatCudaVersionCode(cudaRuntimeVersion) << " ("
              << cudaRuntimeVersion << ")" << "\n";
    std::cout << "CUDA Driver Version (API): " << formatCudaVersionCode(cudaDriverApiVersion) << " ("
              << cudaDriverApiVersion << ")" << "\n";
    std::cout << "Driver Version: " << nvmlDriverVersion << "\n";
}

void PerfParserOutput::addConfComputeInfo(bool bounceBufferConfComputeEnabled) {
    if (!shouldOutput) return;
    std::cout << "Bounce buffer confidential computing mode enabled: " << std::boolalpha << bounceBufferConfComputeEnabled << std::noboolalpha << "\n";
}

void PerfParserOutput::recordDevices(int deviceCount) {
    if (!shouldOutput) return;
    std::cout << "Devices: " << deviceCount << "\n";
}

void PerfParserOutput::recordError(const std::string &error) {
    testFailed = true;
    if (shouldOutput) {
        std::cout << "ERROR: " << error << "\n";
    }
    // Mirror JsonOutput::recordError: flush immediately so the "&&&& ... FAILED"
    // sentinel is emitted even when the caller returns without calling print().
    print();
}

void PerfParserOutput::recordError(const std::vector<std::string> &errorParts) {
    testFailed = true;
    if (shouldOutput) {
        for (const auto& part : errorParts) {
            std::cout << part << "\n";
        }
    }
    print();
}

void PerfParserOutput::recordErrorCurrentTest(const std::string &errorPart1, const std::string &errorPart2) {
    testFailed = true;
    if (shouldOutput) {
        std::cout << errorPart1 << " " << errorPart2 << "\n";
    }
}

void PerfParserOutput::recordWarning(const std::string &warning) {
    // No-op for perf parser
}

void PerfParserOutput::printInfo() {
    // No-op for perf parser
}

void PerfParserOutput::listTestcases(const std::vector<Testcase*> &testcases) {
    // Use default implementation from base class
    Output::listTestcases(testcases);
}
