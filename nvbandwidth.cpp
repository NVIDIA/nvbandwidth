#include <boost/exception/exception.hpp>
#include <map>
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

// Define benchmarks here
std::map<std::string, Benchmark> create_benchmarks() {
    return {
        {
            "host_to_device_bidirectional_memcpy_ce",
            Benchmark(launch_HtoD_memcpy_bidirectional_CE, "Bidirectional host to device memcpy using the Copy Engine")
        },
        {
            "device_to_host_bidirectional_memcpy_ce",
            Benchmark(launch_DtoH_memcpy_bidirectional_CE, "Bidirectional device to host memcpy using the Copy Engine")
        },
        {
            "host_to_device_memcpy_sm",
            Benchmark(launch_HtoD_memcpy_SM, "Host to device memcpy using the Stream Multiprocessor")
        }
    };
}

int main(int argc, char**argv) {
    averageLoopCount = defaultAverageLoopCount;
    disableP2P = true;

    std::map<std::string, Benchmark> benchmarks = create_benchmarks();
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
        std::cout << "Available benchmarks:" << "\n";
        for  (std::map<std::string, Benchmark>::iterator iter = benchmarks.begin(); iter != benchmarks.end(); ++iter) {
            std::cout << "\t" << iter->first << " : " << iter->second.description() << "\n";
        }
        return 1;
    }

    if (vm.count("benchmark")) {
        benchmark_name = vm["benchmark"].as<std::string>();
        std::cout << "Running benchmark " << benchmark_name << ".\n";
    }

    cuInit(0);

    // Run benchmark
    try {
        Benchmark bench = benchmarks[benchmark_name];
        bench.benchFn()(benchmarks[benchmark_name].description(), defaultBufferSize, defaultLoopCount);
    } catch(std::string s) {
        std::cout << s << std::endl;
    }
    return 0;
}
