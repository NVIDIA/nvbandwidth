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

#include <cassert>
#include <string>

#include "common.h"
#include "json_output.h"
#include "nvbandwidth_version.h"

const std::string NVB_TITLE("nvbandwidth");
const std::string NVB_HOST_NAME("Hostname");
const std::string NVB_CUDA_RUNTIME_VERSION("CUDA Runtime Version");
const std::string NVB_DEVICE_INFO("GPU Device info");
const std::string NVB_DEVICE_LIST("GPU Device list");
const std::string NVB_DRIVER_VERSION("Driver Version");
const std::string NVB_GIT_VERSION("git_version");
const std::string NVB_VERSION("version");
const std::string NVB_VERSION_MAJOR("Version Major");
const std::string NVB_VERSION_MINOR("Version Minor");
const std::string NVB_VERSION_PATCH("Version Patch");
const std::string NVB_CUDA_DRIVER_API_VERSION("CUDA Driver API Version");
const std::string NVB_ERROR("error");
const std::string NVB_WARNING("warning");
const std::string NVB_TESTCASES("testcases");
const std::string NVB_TESTCASE_NAME("name");
const std::string NVB_TESTCASE_ERROR(NVB_ERROR);
const std::string NVB_STATUS("status");
const std::string NVB_BW_DESCRIPTION("bandwidth_description");
const std::string NVB_BW_MATRIX("bandwidth_matrix");
const std::string NVB_BW_SUM("sum");
const std::string NVB_BW_MAX("max");
const std::string NVB_BW_MIN("min");
const std::string NVB_BW_AVG("average");
const std::string NVB_BUFFER_SIZE("bufferSize");
const std::string NVB_TEST_SAMPLES("testSamples");
const std::string NVB_USE_MEAN("useMean");
const std::string NVB_BOUNCE_BUFFER_CONF_COMPUTE_MODE_ENABLED("bounceBufferConfComputeEnabled");
const std::string NVB_PASSED("Passed");
const std::string NVB_RUNNING("Running");
const std::string NVB_WAIVED("Waived");
const std::string NVB_NOT_FOUND("Not Found");
const std::string NVB_ERROR_STATUS("Error");

JsonOutput::JsonOutput(bool _shouldOutput) {
    shouldOutput = _shouldOutput;
}

void JsonOutput::addTestcaseResults(const PeerValueMatrix<double> &matrix, const std::string &description) {
    assert(m_root[NVB_TITLE][NVB_TESTCASES].isArray() && m_root[NVB_TITLE][NVB_TESTCASES].size() > 0);

    unsigned int size = m_root[NVB_TITLE][NVB_TESTCASES].size();
    Json::Value &testcase = m_root[NVB_TITLE][NVB_TESTCASES][size-1];

    double maxVal = std::numeric_limits<double>::min();
    double minVal = std::numeric_limits<double>::max();
    double sum = 0;
    int count = 0;

    for (int currentDevice = 0; currentDevice < matrix.m_rows; currentDevice++) {
        Json::Value row;
        for (int peer = 0; peer < matrix.m_columns; peer++) {
            std::optional <double> val = matrix.value(currentDevice, peer);
            if (val && val.value() > 0) {
                std::stringstream buf;
                buf << val.value();
                row.append(buf.str());
                sum += val.value();
                maxVal = std::max(maxVal, val.value());
                minVal = std::min(minVal, val.value());
                count++;
            } else {
                row.append("N/A");
            }
        }
        testcase[NVB_BW_MATRIX].append(row);
    }

    testcase[NVB_BW_SUM] = sum;
    testcase[NVB_BW_DESCRIPTION] = description;
    testcase[NVB_STATUS] = NVB_PASSED;

    if (gSettings.verbose) {
        testcase[NVB_BW_MIN] = minVal;
        testcase[NVB_BW_MAX] = maxVal;
        testcase[NVB_BW_AVG] = count > 0 ? sum/count : 0;
    }
}

void JsonOutput::addTestcase(const std::string &name, const std::string &status, const std::string &msg) {
    Json::Value testcase;
    testcase[NVB_TESTCASE_NAME] = name;
    testcase[NVB_STATUS] = status;
    if (!msg.empty() && status == NVB_WAIVED) {
        testcase["waiver_reason"] = msg;
    }
    m_root[NVB_TITLE][NVB_TESTCASES].append(testcase);
}

void JsonOutput::recordErrorCurrentTest(const std::string &errorPart1, const std::string &errorPart2) {
    bool testCaseExists = false;
    if (m_root[NVB_TESTCASES].isArray()) {
        Json::Value &testcases = m_root[NVB_TITLE][NVB_TESTCASES];
        unsigned int size = testcases.size();
        if (size > 0) {
            testcases[size-1][NVB_TESTCASE_ERROR] = errorPart1 + " " + errorPart2;
            testCaseExists = true;
        }
    }

    if (!testCaseExists) {
        std::vector<std::string> errors;
        errors.emplace_back(errorPart1);
        errors.emplace_back(errorPart2);
        recordError(errors);
    }
}

void JsonOutput::setTestcaseStatusAndAddIfNeeded(const std::string &name, const std::string &status, const std::string &msg) {
    bool testCaseExists = false;
    if (m_root[NVB_TESTCASES].isArray()) {
        Json::Value &testcases = m_root[NVB_TITLE][NVB_TESTCASES];
        unsigned int size = testcases.size();
        if (size > 0 && testcases[size-1][NVB_TESTCASE_NAME].asString() == name) {
            testcases[size-1][NVB_STATUS] = status;
            testCaseExists = true;
        }
    }

    if (!testCaseExists) {
        addTestcase(name, status);
    }
}

void JsonOutput::recordError(const std::string &error) {
    m_root[NVB_TITLE][NVB_ERROR] = error;
    print();
}

void JsonOutput::recordError(const std::vector<std::string> &errorParts) {
    std::stringstream buf;
    bool first = true;

    for (auto &part : errorParts) {
        if (first) {
            buf << part << ":";
            first = false;
        } else {
            buf << " " << part;
        }
    }
    m_root[NVB_TITLE][NVB_ERROR] = buf.str();
}

void JsonOutput::recordWarning(const std::string &warning) {
    m_root[NVB_TITLE][NVB_WARNING] = warning;
}

void JsonOutput::addVersionInfo() {
    m_root[NVB_TITLE][NVB_VERSION] = NVBANDWIDTH_VERSION;
    m_root[NVB_TITLE][NVB_VERSION_MAJOR] = NVBANDWIDTH_VERSION_MAJOR;
    m_root[NVB_TITLE][NVB_VERSION_MINOR] = NVBANDWIDTH_VERSION_MINOR;
    m_root[NVB_TITLE][NVB_VERSION_PATCH] = NVBANDWIDTH_VERSION_PATCH;
    m_root[NVB_TITLE][NVB_GIT_VERSION] = GIT_VERSION;
}

void JsonOutput::addCudaAndDriverInfo(int cudaRuntimeVersion, int cudaDriverApiVersion,
                                      const std::string &nvmlDriverVersion) {
    m_root[NVB_TITLE][NVB_CUDA_RUNTIME_VERSION] = cudaRuntimeVersion;
    m_root[NVB_TITLE][NVB_CUDA_DRIVER_API_VERSION] = cudaDriverApiVersion;
    m_root[NVB_TITLE][NVB_DRIVER_VERSION] = nvmlDriverVersion;
}

void JsonOutput::addConfComputeInfo(bool bounceBufferConfComputeEnabled) {
    m_root[NVB_TITLE][NVB_BOUNCE_BUFFER_CONF_COMPUTE_MODE_ENABLED] = bounceBufferConfComputeEnabled;
}

void JsonOutput::recordDevices(int deviceCount) {
    Json::Value deviceList;

    for (int iDev = 0; iDev < deviceCount; iDev++) {
        std::stringstream buf;
        buf << iDev << ": " << getDeviceDisplayInfo(iDev) << ": (" << env->getHostname() << ")";
        deviceList.append(buf.str());
    }
    m_root[NVB_TITLE][NVB_DEVICE_LIST] = deviceList;
}

void JsonOutput::print() {
    if (shouldOutput) {
        std::cout << m_root.toStyledString() << std::endl;
    }
}

void JsonOutput::printInfo() {
    // NO-OP
}
