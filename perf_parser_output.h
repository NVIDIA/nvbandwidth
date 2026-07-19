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

#ifndef PERF_PARSER_OUTPUT_H_
#define PERF_PARSER_OUTPUT_H_

#include "common.h"
#include "output.h"

#include <vector>

class PerfParserOutput : public Output {
 public:
    PerfParserOutput(bool _shouldOutput);

    void addTestcase(const std::string &name, const std::string &status, const std::string &msg = "");

    void setTestcaseStatusAndAddIfNeeded(const std::string &name, const std::string &status, const std::string &msg = "");

    void print();

    void recordError(const std::string &error);

    void recordError(const std::vector<std::string> &errorParts);

    void recordErrorCurrentTest(const std::string &errorPart1, const std::string &errorPart2);

    void recordWarning(const std::string &warning);

    void addCudaAndDriverInfo(int cudaRuntimeVersion, int cudaDriverApiVersion,
                              const std::string &nvmlDriverVersion);

    void addConfComputeInfo(bool bounceBufferConfComputeEnabled);

    void addTestcaseResults(const PeerValueMatrix<double> &matrix, const std::string &description);

    void addVersionInfo();

    void printInfo();

    void recordDevices(int deviceCount);

    void listTestcases(const std::vector<Testcase*> &testcases);

 private:
    bool shouldOutput;
    bool testFailed;
    std::string currentTestName;
    struct TestResult {
        std::string name;
        double value;
        UnitType unitType;
    };
    std::vector<TestResult> testResults;
};

#endif  // PERF_PARSER_OUTPUT_H_
