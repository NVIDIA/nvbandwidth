cmake_minimum_required(VERSION 3.13)
project(nvbandwidth LANGUAGES CXX CUDA)

include_directories(.)
set(CMAKE_CXX_STANDARD 11)

set(Boost_USE_STATIC_LIBS ON)
find_package( Boost 1.58 COMPONENTS program_options REQUIRED )
find_package(OpenMP)

include_directories(.)
include_directories(/usr/local/cuda-11.4/include)
include_directories( ${Boost_INCLUDE_DIR} )

add_executable(nvbandwidth
        benchmarks.h
        benchmarks_ce.cpp
        benchmarks_sm.cpp
        common.h
        copy_kernel.cu
        copy_kernel.cuh
        memcpy.cpp
        memcpy.h
        memory_utils.cpp
        memory_utils.h
        nvbandwidth.cpp
        sm_memcpy_kernel.cu)

target_link_libraries(nvbandwidth ${Boost_LIBRARIES} cuda nvidia-ml pthread OpenMP::OpenMP_CXX)
