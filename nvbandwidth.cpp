#include <boost/exception/exception.hpp>
#include <stdio.h>
#include <iostream>
#include <string.h>
#include <string>
#include <time.h>
#include <cuda.h>
#include <iomanip>
#include <boost/program_options.hpp>
#include <boost/program_options/options_description.hpp>

#include "benchmarks.h"

namespace opt = boost::program_options;

unsigned int averageLoopCount;
bool disableP2P;

int main(int argc, char**argv) {
    std::string benchmark_name;

    // Args parsing
    opt::options_description desc("NVBandwidth CLI");
    desc.add_options()
        ("help", "Produce help message")
        ("list", "List available benchmarks")
        ("benchmark", opt::value<std::string>(), "Benchmark to run")
    ;

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
        list_benchmarks();
        return 1;
    }

    if (vm.count("benchmark")) {
        benchmark_name = vm["benchmark"].as<std::string>();
        std::cout << "Running benchmark " << benchmark_name << ".\n";
    }

    // Setting some defaults (TODO : will be configurable via CLI)
    averageLoopCount = defaultAverageLoopCount;
    disableP2P = true;

    // Get device properties
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    for (int i = 0; i < nDevices; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        deviceProps.push_back(prop);
    }
    
    // Run benchmark
    benchmarks_map[benchmark_name].benchmark_func(benchmarks_map[benchmark_name].desc, defaultBufferSize, defaultLoopCount, alwaysTrueDeviceFilter);

    return 0;
}
