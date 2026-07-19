cmake_minimum_required(VERSION 3.20)

# Check if user had set the target CUDA_ARCHITECTURE
if(DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CUDA_ARCH_DEFINED_BY_USER TRUE)
    message(STATUS "CMAKE_CUDA_ARCHITECTURES set by user: ${CMAKE_CUDA_ARCHITECTURES}")
endif()

project(nvbandwidth
    VERSION 0.10.0
    LANGUAGES CUDA CXX)

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/nvbandwidth_version.h.in
    ${CMAKE_CURRENT_BINARY_DIR}/nvbandwidth_version.h
    @ONLY)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Unless it's set by the user, ignore CMAKE_CUDA_ARCHITECTURES
if(NOT CUDA_ARCH_DEFINED_BY_USER)
    unset(CMAKE_CUDA_ARCHITECTURES)
    unset(CMAKE_CUDA_ARCHITECTURES CACHE)
endif()

if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "13.0")
    # 5.x and 7.0/7.2 architectures not supported since CUDA 13.0
    set(supported_archs "75" "80" "86" "89" "90" "100")
elseif(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "12.6")
    set(supported_archs "52" "70" "75" "80" "86" "89" "90" "100")
else()
    set(supported_archs "52" "70" "75" "80" "86" "89" "90")
endif()

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    message(STATUS "Detecting underlying CUDA Arch from GPU to set CMAKE_CUDA_ARCHITECTURES")
    include(detect_cuda_arch.cmake)
    # Set CMAKE_CUDA_ARCHITECURES based on the underlying device
    cuda_detect_architectures_from_gpu(supported_archs CMAKE_CUDA_ARCHITECTURES)
endif()

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  message(STATUS "Detecting underlying CUDA Arch from nvcc to set CMAKE_CUDA_ARCHITECTURES")
  # Set CMAKE_CUDA_ARCHITECURES based on the underlying device
  cuda_detect_architectures_from_nvcc(supported_archs CMAKE_CUDA_ARCHITECTURES)
endif()

include(CheckCXXSourceCompiles)
set(CMAKE_REQUIRED_INCLUDES "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")

check_cxx_source_compiles("
#include <nvml.h>
int main() {
    nvmlSystemConfComputeSettings_v1_t settings = {};
    settings.version = nvmlSystemConfComputeSettings_v1;
    (void)settings;
    auto x = NVML_CC_SYSTEM_FEATURE_ENABLED; auto y = NVML_CC_SYSTEM_MULTIGPU_PROTECTED_PCIE;
    (void)x; (void)y;
    return 0;
}
" NVBW_HAVE_NVML_CONF_COMPUTE_SETTINGS)

if(NVBW_HAVE_NVML_CONF_COMPUTE_SETTINGS)
  add_compile_definitions(NVBW_HAVE_NVML_CONF_COMPUTE_SETTINGS)
else()
  message(STATUS "nvmlSystemGetConfComputeSettings not found in nvml.h. nvbandwidth will skip detection of bounce-buffer confidential computing platforms.")
endif()

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()

include(FetchContent)
FetchContent_Declare(
        argparse
        URL https://github.com/p-ranav/argparse/archive/refs/tags/v3.2.tar.gz
        URL_HASH SHA256=9dcb3d8ce0a41b2a48ac8baa54b51a9f1b6a2c52dd374e28cc713bab0568ec98
        # Pinned version - provides .flag(), .store_into(), .choices()
)
FetchContent_MakeAvailable(argparse)

set(src
    cuda_version_check.cpp
    environment.cpp
    testcase.cpp
    testcases.cpp
    kernels.cu
    memcpy.cpp
    nvbandwidth.cpp
    multinode_memcpy.cpp
    multinode_testcases.cpp
    output.cpp
    json_output.cpp
    perf_parser_output.cpp
    json/jsoncpp.cpp
)

execute_process(
    COMMAND git describe --always --tags
    WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
    OUTPUT_VARIABLE GIT_VERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DGIT_VERSION=\\\"\"${GIT_VERSION}\"\\\"")

if(WIN32)
    set(NVML_LIB_NAME "nvml")
    # Suppress MSVC warnings about C runtime functions (sprintf, getenv, etc.)
    # that Microsoft considers deprecated in favor of *_s "secure" variants.
    # Sources: json/jsoncpp.cpp, environment.cpp
    add_compile_definitions("_CRT_SECURE_NO_WARNINGS" "1")
else()
    set(NVML_LIB_NAME "nvidia-ml")
endif()

add_executable(nvbandwidth ${src})
target_include_directories(nvbandwidth PRIVATE
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    .)
target_link_libraries(nvbandwidth argparse ${NVML_LIB_NAME} cuda)
if(NOT WIN32)
    target_link_libraries(nvbandwidth dl)
endif()

if (MULTINODE)
    find_package(MPI REQUIRED)
    include_directories(SYSTEM ${MPI_INCLUDE_PATH})
    target_link_libraries(nvbandwidth MPI::MPI_CXX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DMULTINODE")
endif()
