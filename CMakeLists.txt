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

find_package(Boost COMPONENTS program_options REQUIRED)

set(src
    benchmark.cpp
    benchmarks_ce.cpp
    benchmarks_sm.cpp
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

add_executable(nvbandwidth ${src})
target_include_directories(nvbandwidth PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} .)
target_link_libraries(nvbandwidth Boost::program_options nvidia-ml cuda)
