/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.                                                                                                                                                  * SPDX-License-Identifier: Apache-2.0                                                                                                                                                                                                               *
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

#include "environment.h"

#include "error_handling.h"
#include <common.h>
#include <cuda_runtime.h>
#ifndef _WIN32
#include <unistd.h>
#endif

std::unique_ptr<Environment> Environment::create(int argc, char** argv) {
#ifdef MULTINODE
    // When compiled with MULTINODE, always use MultiNodeEnv
    return std::make_unique<MultiNodeEnv>();
#else
    return std::make_unique<SingleNodeEnv>();
#endif
}

std::string SingleNodeEnv::getHostname() const {
#ifdef _WIN32
    char hostname[STRING_LENGTH] = "unknown";
    strncpy(hostname, getenv("COMPUTERNAME"), STRING_LENGTH - 1);
    const char* computername = getenv("COMPUTERNAME");
    if (computername && computername[0] != '\0') {
        strncpy(hostname, computername, STRING_LENGTH - 1);
        hostname[STRING_LENGTH - 1] = '\0';
    }
#else
    char hostname[STRING_LENGTH];
    ASSERT(0 == gethostname(hostname, STRING_LENGTH - 1));
#endif
    return hostname;
}

#ifdef MULTINODE
void MultiNodeEnv::initialize(int argc, char** argv) {
    // Always initialize MPI when MULTINODE is defined
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    char host[STRING_LENGTH];
    gethostname(host, sizeof(host));
    hostname = host;

    // Get local rank
    const char* localRankStr = getenv("OMPI_COMM_WORLD_LOCAL_RANK");
    if (localRankStr) {
        localRank = atoi(localRankStr);
    } else {
        localRank = rank;
    }
}

void MultiNodeEnv::finalize() {
    MPI_Finalize();
}
#endif
