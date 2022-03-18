cmake_minimum_required(VERSION 3.10)

project(nvbandwidth 
    LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()

set(Boost_USE_STATIC_LIBS ON)
find_package(Boost COMPONENTS program_options REQUIRED)

set(src
    testcase.cpp
    testcases_ce.cpp
    testcases_sm.cpp
    kernels.cu
    memcpy.cpp
    nvbandwidth.cpp
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
