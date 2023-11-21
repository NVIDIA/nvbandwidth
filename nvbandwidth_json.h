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

#pragma once

#include <json/json.h>
#include <string>

#include "common.h"

extern const std::string NVB_TITLE;
extern const std::string NVB_CUDA_RUNTIME_VERSION;
extern const std::string NVB_DRIVER_VERSION;
extern const std::string NVB_GIT_VERSION;
extern const std::string NVB_ERROR;
extern const std::string NVB_WARNING;
extern const std::string NVB_TESTCASES;
extern const std::string NVB_TESTCASE_NAME;
extern const std::string NVB_STATUS;
extern const std::string NVB_BW_DESCRIPTION;
extern const std::string NVB_BW_MATRIX;
extern const std::string NVB_BW_SUM;
extern const std::string NVB_BUFFER_SIZE;
extern const std::string NVB_TEST_SAMPLES;
extern const std::string NVB_USE_MEAN;
extern const std::string NVB_PASSED;
extern const std::string NVB_WAIVED;
extern const std::string NVB_NOT_FOUND;
extern const std::string NVB_ERROR_STATUS;

class NvBandwidthJson
{
public:
    void addTestcase(const std::string &name, const std::string &status);

    void addTestcaseIfNeeded(const std::string &name, const std::string &status);

    void printJson();

    void recordError(const std::string &error);

    void recordErrorCurrentTest(const std::string &error);

    void recordWarning(const std::string &warning);

    void init();

    void addCudaAndDriverInfo(int cudaVersion, const std::string &driverVersion);

    unsigned int addTestcaseResults(const PeerValueMatrix<double> &matrix, const std::string &description);

private:
    Json::Value m_root;
};

extern NvBandwidthJson jsonMgr;
