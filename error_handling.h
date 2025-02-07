/*
 * spdx-filecopyrighttext: copyright (c) 2024 nvidia corporation & affiliates. all rights reserved.
 * spdx-license-identifier: apache-2.0
 *
 * licensed under the apache license, version 2.0 (the "license");
 * you may not use this file except in compliance with the license.
 * you may obtain a copy of the license at
 *
 * http://www.apache.org/licenses/license-2.0
 *
 * unless required by applicable law or agreed to in writing, software
 * distributed under the license is distributed on an "as is" basis,
 * without warranties or conditions of any kind, either express or implied.
 * see the license for the specific language governing permissions and
 * limitations under the license.
 */

#ifndef ERROR_HANDLING_H_
#define ERROR_HANDLING_H_

void RecordError(const std::stringstream &errmsg);

#ifdef MULTINODE
#define HOST_INFO " on " << localHostname << ", rank = " << worldRank
#else
#define HOST_INFO ""
#endif

#ifdef MULTINODE
#include <mpi.h>
#define MPI_ABORT MPI_Abort(MPI_COMM_WORLD, 1)
#else
#define MPI_ABORT
#endif

// CUDA Error handling
#define CUDA_ASSERT(x) do { \
    cudaError_t cudaErr = (x); \
    if ((cudaErr) != cudaSuccess) { \
        std::stringstream errmsg; \
        errmsg << "[" << cudaGetErrorName(cudaErr) << "] " << cudaGetErrorString(cudaErr) << " in expression " << #x << HOST_INFO << " in " << __PRETTY_FUNCTION__ << "() : " << __FILE__ << ":" <<  __LINE__ << std::endl; \
        RecordError(errmsg); \
        MPI_ABORT; \
        std::exit(1); \
    }  \
} while ( 0 )

#define CU_ASSERT(x) do { \
    CUresult cuResult = (x); \
    if ((cuResult) != CUDA_SUCCESS) { \
        const char *errDescStr, *errNameStr; \
        cuGetErrorString(cuResult, &errDescStr); \
        cuGetErrorName(cuResult, &errNameStr); \
        std::stringstream errmsg; \
        errmsg << "[" << errNameStr << "] " << errDescStr << " in expression " << #x << HOST_INFO << " in " << __PRETTY_FUNCTION__ << "() : " << __FILE__ << ":" <<  __LINE__ << std::endl; \
        RecordError(errmsg); \
        MPI_ABORT; \
        std::exit(1); \
    }  \
} while ( 0 )

// NVML Error handling
#define NVML_ASSERT(x) do { \
    nvmlReturn_t nvmlResult = (x); \
    if ((nvmlResult) != NVML_SUCCESS) { \
        std::stringstream errmsg; \
        errmsg << "NVML_ERROR: [" << nvmlErrorString(nvmlResult) << "] in expression " << #x << HOST_INFO << " in " << __PRETTY_FUNCTION__ << "() : " << __FILE__ << ":" <<  __LINE__ << std::endl; \
        RecordError(errmsg); \
        MPI_ABORT; \
        std::exit(1); \
    }  \
} while ( 0 )

// Generic Error handling
#define ASSERT(x) do { \
    if (!(x)) { \
        std::stringstream errmsg; \
        errmsg << "ASSERT in expression " << #x << HOST_INFO << " in " << __PRETTY_FUNCTION__ << "() : " << __FILE__ << ":" <<  __LINE__  << std::endl; \
        RecordError(errmsg); \
        MPI_ABORT; \
        std::exit(1); \
    }  \
} while ( 0 )

#endif  // ERROR_HANDLING_H_
