#include <boost/exception/exception.hpp>
#include <boost/program_options.hpp>
#include <boost/program_options/options_description.hpp>
#include <cuda.h>
#include <iomanip>
#include <iostream>
#include <map>
#include <stdio.h>
#include <string.h>
#include <string>
#include <time.h>

#include "benchmarks.h"
#include "common.h"

namespace opt = boost::program_options;

unsigned int averageLoopCount;
unsigned long long bufferSize;
unsigned long long loopCount;

bool disableP2P;
bool verbose;

// Define benchmarks here
std::map<std::string, Benchmark> create_benchmarks() {
    return {
        {"host_to_device_memcpy_ce",
            Benchmark(launch_HtoD_memcpy_CE,
                    "Host to device memcpy using the Copy Engine")},
        {"device_to_host_memcpy_ce",
            Benchmark(launch_DtoH_memcpy_CE,
                    "Device to host memcpy using the Copy Engine")},
        {"host_to_device_bidirectional_memcpy_ce",
            Benchmark(launch_HtoD_memcpy_bidirectional_CE,
                    "Bidirectional host to device memcpy using the Copy Engine")},
        {"device_to_host_bidirectional_memcpy_ce",
            Benchmark(launch_DtoH_memcpy_bidirectional_CE,
                    "Bidirectional device to host memcpy using the Copy Engine")},
        {"device_to_device_memcpy_read_ce",
            Benchmark(launch_DtoD_memcpy_read_CE,
                    "Device to device memcpy using the Copy Engine (read)")},
        {"device_to_device_memcpy_write_ce",
            Benchmark(launch_DtoD_memcpy_write_CE,
                    "Device to device memcpy using the Copy Engine (write)")},
        {"device_to_device_bidirectional_memcpy_ce",
            Benchmark(launch_DtoD_memcpy_bidirectional_CE,
                    "Bidirectional device to device memcpy using the Copy Engine")},
        {"host_to_device_memcpy_sm",
            Benchmark(launch_HtoD_memcpy_SM,
                    "Host to device memcpy using the Stream Multiprocessor")},
        {"device_to_host_memcpy_sm",
            Benchmark(launch_DtoH_memcpy_SM,
                   "Device to host memcpy using the Stream Multiprocessor")},
        {"device_to_device_memcpy_read_sm",
            Benchmark(launch_DtoD_memcpy_read_SM,
                    "Device to device memcpy using the Stream Multiprocessor (read)")},
        {"device_to_device_memcpy_write_sm",
            Benchmark(launch_DtoD_memcpy_write_SM,
                    "Device to device memcpy using the Stream Multiprocessor (write)")},
        {"device_to_device_bidirectional_memcpy_read_sm",
            Benchmark(launch_DtoD_memcpy_bidirectional_read_SM,
                    "Bidirectional device to device memcpy using the Stream Multiprocessor (read)")},
        {"device_to_device_bidirectional_memcpy_write_sm",
            Benchmark(launch_DtoD_memcpy_bidirectional_write_SM,
                    "Bidirectional device to device memcpy using the Stream Multiprocessor (write)")}};
}

void init() {
    cuInit(0);
}

int main(int argc, char **argv) {
    averageLoopCount = defaultAverageLoopCount;
    disableP2P = true;

    std::map<std::string, Benchmark> benchmarks = create_benchmarks();
    std::string benchmark_name;

    // Args parsing
    opt::options_description desc("NVBandwidth CLI");
    desc.add_options()
        ("help,h", "Produce help message")
        ("bufferSize", opt::value<unsigned long long int>(), "Memcpy buffer size")
        ("loopCount", opt::value<unsigned long long int>(), "Iterations of memcpy to be performed")
        ("list,l", "List available benchmarks")
        ("benchmark,b", opt::value<std::string>(), "Benchmark to run")
        ("verbose,v", "Verbose output");

    opt::variables_map vm;
    try {
        opt::store(opt::parse_command_line(argc, argv, desc), vm);
        opt::notify(vm);
    } catch (...) {
        std::cout << desc << "\n";
        return 1;
    }

    if (vm.count("help")) {
        std::cout << desc << "\n";
        return 1;
    }

    if (vm.count("list")) {
        for (std::map<std::string, Benchmark>::iterator iter = benchmarks.begin();
            iter != benchmarks.end(); ++iter) {
            std::cout << iter->first << ":\n\t\t" << iter->second.description() << "\n";
        }
        return 0;
    }

    if (vm.count("bufferSize")) bufferSize = vm["bufferSize"].as<unsigned long long int>();;
    if (vm.count("loopCount")) loopCount = vm["loopCount"].as<unsigned long long int>();;

    if (vm.count("verbose")) verbose = true;
    else verbose = false;

    if (vm.count("benchmark")) {
        benchmark_name = vm["benchmark"].as<std::string>();
        std::cout << "Running benchmark " << benchmark_name << ".\n";
    }

    init();

    // Run benchmark
    try {
        CUcontext benchCtx;
        CU_ASSERT(cuCtxCreate(&benchCtx, 0, 0));
        CU_ASSERT(cuCtxSetCurrent(benchCtx));
        
        Benchmark bench = benchmarks[benchmark_name];
        bench.bench_fn()(defaultBufferSize, defaultLoopCount);

        CU_ASSERT(cuCtxDestroy(benchCtx));
    } catch (std::string s) {
        std::cout << s << std::endl;
    }
    return 0;
}
