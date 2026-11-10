cmake_minimum_required(VERSION 3.20)

# Check if user had set the target CUDA_ARCHITECTURE
if(DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CUDA_ARCH_DEFINED_BY_USER TRUE)
    message(STATUS "CMAKE_CUDA_ARCHITECTURES set by user: ${CMAKE_CUDA_ARCHITECTURES}")
endif()

project(nvbandwidth
    VERSION 0.11.0
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

# Detect whether the CUDA driver headers expose the Logical Endpoint API
# (cuLogicalEndpointCreate and friends were added in a recent driver release).
# Older CI environments won't have these; gate all LE host code behind this flag.
include(CheckCXXSourceCompiles)
set(CMAKE_REQUIRED_INCLUDES "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
check_cxx_source_compiles("
#include <cuda.h>
void test() { CUlogicalEndpointProp p = {}; (void)p; }
int main() { return 0; }
" NVBW_HAVE_LE_API)
if(NVBW_HAVE_LE_API)
    add_compile_definitions(NVBW_HAVE_LE_API)
endif()

# The LE device-attribute enumerators were added later than the LE API structs,
# so a toolkit can have CUlogicalEndpointProp but not these. Fall back to the
# numeric attribute values (queried via cuDeviceGetAttribute, safe on any driver).
check_cxx_source_compiles("
#include <cuda.h>
int main() {
    auto a = CU_DEVICE_ATTRIBUTE_LOGICAL_ENDPOINT_UNICAST_SUPPORTED;
    auto b = CU_DEVICE_ATTRIBUTE_LOGICAL_ENDPOINT_SUPPORTED_HANDLE_TYPES;
    (void)a; (void)b;
    return 0;
}
" NVBW_HAVE_LE_DEVICE_ATTRS)
if(NVBW_HAVE_LE_DEVICE_ATTRS)
    add_compile_definitions(NVBW_HAVE_LE_DEVICE_ATTRS)
endif()

# kernels_le_tma.cu has fabric kernels requiring Blackwell or later (sm_100+).
# CMake keeps the special token "native" literal in CMAKE_CUDA_ARCHITECTURES (it is not
# expanded to numeric SM versions), so resolve it to the detected native archs before the
# Blackwell regex check; otherwise -DCMAKE_CUDA_ARCHITECTURES=native on a Blackwell host
# would miss the "^1[0-9][0-9]" match and silently fall back to the stub kernels.
set(_archs_for_blackwell_check ${CMAKE_CUDA_ARCHITECTURES})
if(_archs_for_blackwell_check MATCHES "native")
  if(DEFINED CMAKE_CUDA_ARCHITECTURES_NATIVE)
    list(APPEND _archs_for_blackwell_check ${CMAKE_CUDA_ARCHITECTURES_NATIVE})
  elseif(CMAKE_VERSION VERSION_LESS "3.24")
    # The "native" keyword and CMAKE_CUDA_ARCHITECTURES_NATIVE were only added in
    # CMake 3.24. On older CMake the token is never resolved, so the Blackwell
    # detection below would silently disable fabric kernels on capable hardware.
    message(WARNING "CMAKE_CUDA_ARCHITECTURES=native requires CMake >= 3.24 (you have ${CMAKE_VERSION}); "
                    "the 'native' keyword cannot be resolved on this version and "
                    "fabric/multicast-write kernels will be disabled even on Blackwell. "
                    "Upgrade CMake to 3.24+ or pass an explicit -DCMAKE_CUDA_ARCHITECTURES=100.")
  else()
    message(WARNING "CMAKE_CUDA_ARCHITECTURES=native could not be resolved to concrete SM versions; "
                    "fabric/multicast-write kernels may be disabled even on Blackwell. "
                    "Pass an explicit -DCMAKE_CUDA_ARCHITECTURES=100 if needed.")
  endif()
endif()
set(_has_blackwell_or_later FALSE)
foreach(_arch ${_archs_for_blackwell_check})
  if(_arch MATCHES "^1[0-9][0-9]")
    set(_has_blackwell_or_later TRUE)
    break()
  endif()
endforeach()
if(NOT _has_blackwell_or_later OR NOT NVBW_HAVE_LE_API)
  set(le_kernels_src kernels_le_tma_stub.cu)
  message(STATUS "LE fabric kernels disabled (arch < Blackwell or LE API unavailable): using kernels_le_tma_stub.cu (LE tests waived at runtime; multicast-write TMA test waived via NVBW_HAVE_FABRIC_KERNELS)")
else()
  set(le_kernels_src kernels_le_tma.cu)
  # Real fabric/TMA kernels (incl. the multimem multicast WRITE kernel) are built.
  # The multimem READ kernel lives in kernels_multimem.cu and is always available.
  add_compile_definitions(NVBW_HAVE_FABRIC_KERNELS)
  message(STATUS "LE fabric kernels enabled (sm_100+ and LE API available): using kernels_le_tma.cu")
endif()

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

if(MULTINODE)
  check_cxx_source_compiles("
  #include <cuda.h>
  int main() {
      CUuuid uuid = {};
      CUcliqueInfo cliqueInfo[1] = {};
      size_t count = 1;
      auto unicastPointer = CU_CLIQUE_TYPE_UNICAST_POINTER;
      auto multicastPointer = CU_CLIQUE_TYPE_MULTICAST_POINTER;
      auto unicastLogicalEndpoint = CU_CLIQUE_TYPE_UNICAST_LOGICAL_ENDPOINT;
      auto multicastLogicalEndpoint = CU_CLIQUE_TYPE_MULTICAST_LOGICAL_ENDPOINT;
      using UuidFn = decltype(&cuDeviceGetFabricClusterUuid);
      using CliqueInfoFn = decltype(&cuDeviceGetCliqueInfo);
      UuidFn uuidFn = nullptr;
      CliqueInfoFn cliqueInfoFn = nullptr;
      (void)uuid; (void)cliqueInfo; (void)count; (void)unicastPointer;
      (void)multicastPointer; (void)unicastLogicalEndpoint; (void)multicastLogicalEndpoint;
      (void)uuidFn; (void)cliqueInfoFn;
      return 0;
  }
  " NVBW_HAVE_CUDA_FABRIC_CLIQUE_INFO)

  if(NVBW_HAVE_CUDA_FABRIC_CLIQUE_INFO)
    add_compile_definitions(NVBW_HAVE_CUDA_FABRIC_CLIQUE_INFO)
  else()
    message(STATUS "CUDA fabric clique APIs not found in cuda.h. nvbandwidth will try the NVML fabric-info fallback if available.")
  endif()

  check_cxx_source_compiles("
  #include <nvml.h>
  int main() {
      nvmlGpuFabricInfo_v4_t fabricInfo = {};
      auto state = NVML_GPU_FABRIC_STATE_COMPLETED;
      auto cliqueType = NVML_GPU_FABRIC_CLIQUE_TYPE_UNICAST_POINTER;
      using FabricInfoFn = decltype(&nvmlDeviceGetGpuFabricInfo_v4);
      FabricInfoFn fn = nullptr;
      (void)fabricInfo; (void)state; (void)cliqueType; (void)fn;
      return 0;
  }
  " NVBW_HAVE_NVML_GPU_FABRIC_INFO)

  if(NVBW_HAVE_NVML_GPU_FABRIC_INFO)
    add_compile_definitions(NVBW_HAVE_NVML_GPU_FABRIC_INFO)
  else()
    message(STATUS "nvmlDeviceGetGpuFabricInfo_v4 support not found in nvml.h. nvbandwidth will skip the NVML v4 fabric-info fallback.")
  endif()
endif()

# Detect whether the CUDA driver headers expose uGPU memory-node split support
# (CU_MEM_LOCATION_TYPE_DEVICE_LOCALITY_DOMAIN was added in CUDA 13.4).
# Gate all localized-memory test code behind this flag.
check_cxx_source_compiles("
#include <cuda.h>
int main() {
    CUmemAllocationProp p = {};
    p.location.type = CU_MEM_LOCATION_TYPE_DEVICE_LOCALITY_DOMAIN;
    (void)p;
    return 0;
}
" NVBW_HAVE_UGPU_MEMORY_NODE_SPLIT)
if(NVBW_HAVE_UGPU_MEMORY_NODE_SPLIT)
    add_compile_definitions(NVBW_HAVE_UGPU_MEMORY_NODE_SPLIT)
    message(STATUS "uGPU memory-node split support detected: localized CE tests enabled.")
else()
    message(STATUS "CU_MEM_LOCATION_TYPE_DEVICE_LOCALITY_DOMAIN not found in cuda.h. Localized CE tests disabled.")
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
    kernels_multimem.cu
    ${le_kernels_src}
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
