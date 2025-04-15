cmake_minimum_required(VERSION 3.20)

project(nvbandwidth
    LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "13.0")
    # 5.2 architecture not supported since CUDA 13.0
    set(supported_archs "70" "75" "80" "86" "89" "90" "100")
else ()
    set(supported_archs "52" "70" "75" "80" "86" "89" "90" "100")
endif()

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    message(STATUS "Detecting underlying CUDA Arch to set CMAKE_CUDA_ARCHITECTURES")
    include(detect_cuda_arch.cmake)
    # Set CMAKE_CUDA_ARCHITECURES based on the underlying device
    cuda_detect_architectures(supported_archs CMAKE_CUDA_ARCHITECTURES)
endif()

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    file(READ "/etc/os-release" OS_RELEASE_CONTENT)
    # Skip static libs on Fedora - https://github.com/NVIDIA/nvbandwidth/issues/4
    if(NOT OS_RELEASE_CONTENT MATCHES "ID=.*fedora")
        set(Boost_USE_STATIC_LIBS ON)
    endif()
else()
    set(Boost_USE_STATIC_LIBS ON)
endif()
find_package(Boost COMPONENTS program_options REQUIRED)

set(src
    testcase.cpp
    testcases_ce.cpp
    testcases_sm.cpp
    kernels.cu
    memcpy.cpp
    nvbandwidth.cpp
    multinode_memcpy.cpp
    multinode_testcases.cpp
    output.cpp
    json_output.cpp
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
else()
    set(NVML_LIB_NAME "nvidia-ml")
endif()

add_executable(nvbandwidth ${src})
target_include_directories(nvbandwidth PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} .)
target_link_libraries(nvbandwidth Boost::program_options ${NVML_LIB_NAME} cuda)

if (MULTINODE)
    find_package(MPI REQUIRED)
    include_directories(SYSTEM ${MPI_INCLUDE_PATH})
    target_link_libraries(nvbandwidth MPI::MPI_CXX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DMULTINODE")
endif()
