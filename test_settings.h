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

#ifndef TEST_SETTINGS_H_
#define TEST_SETTINGS_H_

#include <cstdint>
#include <string>

// Default constants with explicit units and fixed-width types
constexpr std::uint64_t kDefaultLoopCount = 16;
constexpr std::uint64_t kDefaultBufferSizeMiB = 512;  // Buffer size in MiB
constexpr std::uint32_t kDefaultAverageLoopCount = 3;
constexpr std::int64_t kDefaultTargetPairs = -1;      // -1 means all pairs

// Minimum number of run-to-run samples (averageLoopCount) required before the
// COEFFICIENT_OF_VARIATION stability metric is emitted/gated. With fewer
// samples the coefficient of variation is too noisy to be meaningful.
constexpr std::uint32_t kMinAverageLoopCountForCV = 3;

// Consolidates all test configuration settings parsed from command line
struct TestSettings {
    // Buffer and loop configuration (sizes in MiB, counts are iterations)
    std::uint64_t bufferSizeMiB = kDefaultBufferSizeMiB;
    std::uint64_t loopCount = kDefaultLoopCount;
    std::uint32_t averageLoopCount = kDefaultAverageLoopCount;

    // Multinode configuration
    std::int64_t targetNumPairs = kDefaultTargetPairs;

    // Boolean flags
    bool verbose = false;
    bool skipVerification = false;
    bool disableAffinity = false;
    bool disablePerformanceCache = false;  // Cache enabled by default
    bool useMean = false;
    bool useHugePages = false;
    bool disableNvlinkSMScheduling = false;
    bool bounceBufferConfComputeEnabled = false;

    // Convenience accessor for cache (inverted for backward compat)
    bool performanceCacheEnabled() const { return !disablePerformanceCache; }

    // Output format (string-based, validated by argparse .choices())
    std::string outputFormat = "text";
    bool perfFormatter = false;  // Hidden: adds perf formatter prefix

    // Convenience accessors for output format
    bool isJsonOutput() const { return outputFormat == "json"; }
    bool isPerfOutput() const { return outputFormat == "perf"; }
    bool isTextOutput() const { return outputFormat == "text"; }
};

// Global settings instance (for backward compatibility during incremental refactor)
extern TestSettings gSettings;

#endif  // TEST_SETTINGS_H_

