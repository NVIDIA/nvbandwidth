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

#ifndef JSON_OUTPUT_H_
#define JSON_OUTPUT_H_

#include <json/json.h>
#include <string>

#include "common.h"
#include "output.h"

class JsonOutput : public Output {
 public:
    JsonOutput(bool shouldOutput);

    void addTestcase(const std::string &name, const std::string &status, const std::string &msg = "");

    void setTestcaseStatusAndAddIfNeeded(const std::string &name, const std::string &status, const std::string &msg = "");

    void print();

    void recordError(const std::string &error);

    void recordError(const std::vector<std::string> &errorParts);

    void recordErrorCurrentTest(const std::string &errorPart1, const std::string &errorPart2);

    void recordWarning(const std::string &warning);

    void addVersionInfo();

    void addCudaAndDriverInfo(int cudaVersion, const std::string &driverVersion);

    void addTestcaseResults(const PeerValueMatrix<double> &matrix, const std::string &description);

    void printInfo();

    void recordDevices(int deviceCount);

 private:
    bool shouldOutput;
    Json::Value m_root;
};

#endif  // JSON_OUTPUT_H_
