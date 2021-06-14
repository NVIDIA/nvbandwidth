#include <iostream>

#include "memcpy_ce_tests.h"

typedef void (*BenchmarkFunc)(const std::string &, unsigned long long, unsigned long long, DeviceFilter);
typedef struct {
    BenchmarkFunc benchmark_func;
    std::string desc; 
} Benchmark;

typedef std::map<std::string, Benchmark> Benchmarks;

static Benchmarks benchmarks_map {
    { "host_to_device_bidirectional_memcpy",
        {
            launch_HtoD_memcpy_bidirectional_CE,
            "Host to device memcpy using the Copy Engine"
        }
    },
    { "device_to_host_bidirectional_memcpy",
        {
            launch_DtoH_memcpy_bidirectional_CE,
            "Device to host memcpy using the Copy Engine"
        }
    }
};

inline void list_benchmarks() {
    std::cout << "Available benchmarks:" << "\n";
    for  (const auto &bench : benchmarks_map) {
        std::cout << "\t" << bench.first << " : " << bench.second.desc << "\n";
    }
}
