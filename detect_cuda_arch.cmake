include_guard(GLOBAL)

# Function uses the CUDA runtime API to query the compute capability of the device, so if a user
# doesn't pass any architecture options to CMake we only build the current architecture

# Adapted from https://github.com/rapidsai/rapids-cmake/blob/branch-24.04/rapids-cmake/cuda/detail/detect_architectures.cmake

function(cuda_detect_architectures possible_archs_var gpu_archs)

  set(__gpu_archs ${${possible_archs_var}})

  set(eval_file eval_gpu_archs.cu)
  set(eval_exe eval_gpu_archs)
  set(error_file eval_gpu_archs.stderr.log)

  if(NOT DEFINED CMAKE_CUDA_COMPILER)
    message(FATAL_ERROR "No CUDA compiler specified, unable to determine machine's GPUs.")
  endif()

  if(NOT EXISTS "${eval_exe}")
    file(WRITE ${eval_file}
         "
#include <cstdio>
#include <set>
#include <string>
using namespace std;
int main(int argc, char** argv) {
  set<string> archs;
  int nDevices;
  if((cudaGetDeviceCount(&nDevices) == cudaSuccess) && (nDevices > 0)) {
    for(int dev=0;dev<nDevices;++dev) {
      char buff[32];
      cudaDeviceProp prop;
      if(cudaGetDeviceProperties(&prop, dev) != cudaSuccess) continue;
      sprintf(buff, \"%d%d\", prop.major, prop.minor);
      archs.insert(buff);
    }
  }
  if(archs.empty()) {
    printf(\"${__gpu_archs}\");
  } else {
    bool first = true;
    for(const auto& arch : archs) {
      printf(first? \"%s\" : \";%s\", arch.c_str());
      first = false;
    }
  }
  printf(\"\\n\");
  return 0;
  }
  ")
    execute_process(COMMAND ${CMAKE_CUDA_COMPILER} -std=c++11 -o "${eval_exe}" "${eval_file}"
                    ERROR_FILE "${error_file}")
  endif()

  if(EXISTS "${eval_exe}")
    execute_process(COMMAND "./${eval_exe}" OUTPUT_VARIABLE __gpu_archs
                    OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_FILE "${error_file}")
    message(STATUS "Auto detection of gpu-archs: ${__gpu_archs}")
  else()
    message(STATUS "Failed auto detection of gpu-archs. Falling back to using ${__gpu_archs}.")
  endif()
  # remove the build artifacts
  file(REMOVE "${eval_file}" "${eval_exe}" "${error_file}")
  set(${gpu_archs} ${__gpu_archs} PARENT_SCOPE)

endfunction()
