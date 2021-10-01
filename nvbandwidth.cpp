#include <boost/program_options.hpp>
#include <cuda.h>
#include <nvml.h>
#include <iostream>

#include "benchmarks.h"

namespace opt = boost::program_options;

int deviceCount;
unsigned int averageLoopCount;
unsigned long long bufferSize;
unsigned long long loopCount;
bool disableP2P;
bool parallel;
bool verbose;
Verbosity VERBOSE;

// Define benchmarks here
std::vector<Benchmark> createBenchmarks() {
    return {
        Benchmark("host_to_device_memcpy_ce", launch_HtoD_memcpy_CE, "Host to device memcpy using the Copy Engine"),
        Benchmark("device_to_host_memcpy_ce", launch_DtoH_memcpy_CE, "Device to host memcpy using the Copy Engine"),
        Benchmark("host_to_device_bidirectional_memcpy_ce", launch_HtoD_memcpy_bidirectional_CE, "Bidirectional host to device memcpy using the Copy Engine"),
        Benchmark("device_to_host_bidirectional_memcpy_ce", launch_DtoH_memcpy_bidirectional_CE, "Bidirectional device to host memcpy using the Copy Engine"),
        Benchmark("device_to_device_memcpy_read_ce", launch_DtoD_memcpy_read_CE, "Device to device memcpy using the Copy Engine (read)"),
        Benchmark("device_to_device_memcpy_write_ce", launch_DtoD_memcpy_write_CE, "Device to device memcpy using the Copy Engine (write)"),
        Benchmark("device_to_device_bidirectional_memcpy_ce", launch_DtoD_memcpy_bidirectional_CE, "Bidirectional device to device memcpy using the Copy Engine"),
        Benchmark("device_to_device_paired_memcpy_read_ce", launch_DtoD_paired_memcpy_read_CE, "Paired device to device memcpy using the Copy Engine (read)"),
        Benchmark("device_to_device_paired_memcpy_write_ce", launch_DtoD_paired_memcpy_write_CE, "Paired device to device memcpy using the Copy Engine (write)"),
        Benchmark("host_to_device_memcpy_sm", launch_HtoD_memcpy_SM, "Host to device memcpy using the Stream Multiprocessor"),
        Benchmark("device_to_host_memcpy_sm", launch_DtoH_memcpy_SM, "Device to host memcpy using the Stream Multiprocessor"),
        Benchmark("device_to_device_memcpy_read_sm", launch_DtoD_memcpy_read_SM, "Device to device memcpy using the Stream Multiprocessor (read)"),
        Benchmark("device_to_device_memcpy_write_sm", launch_DtoD_memcpy_write_SM, "Device to device memcpy using the Stream Multiprocessor (write)"),
        Benchmark("device_to_device_bidirectional_memcpy_read_sm", launch_DtoD_memcpy_bidirectional_read_SM, "Bidirectional device to device memcpy using the Stream Multiprocessor (read)"),
        Benchmark("device_to_device_bidirectional_memcpy_write_sm", launch_DtoD_memcpy_bidirectional_write_SM, "Bidirectional device to device memcpy using the Stream Multiprocessor (write)"),
        Benchmark("device_to_device_paired_memcpy_read_sm", launch_DtoD_paired_memcpy_read_SM, "Paired device to device memcpy using the Stream Multiprocessor (read)"),
        Benchmark("device_to_device_paired_memcpy_write_sm", launch_DtoD_paired_memcpy_write_SM, "Paired device to device memcpy using the Stream Multiprocessor (write)")
    };
}

Benchmark findBenchmark(std::vector<Benchmark> &benchmarks, std::string id) {
    // Check if benchmark ID is index
    char* p;
    long index = strtol(id.c_str(), &p, 10);
    if (*p) {
        // Conversion failed so key is ID
        auto it = find_if(benchmarks.begin(), benchmarks.end(), [&id](Benchmark& bench) {return bench.benchKey() == id;});
        if (it != benchmarks.end()) {
            return benchmarks.at(std::distance(benchmarks.begin(), it));
        } else {
            throw "Benchmark " + id + " not found!";
        }
    } else {
        // ID is index
        if (index >= benchmarks.size()) throw "Benchmark index " + id + " out of bound!";
        return benchmarks.at(index);
    }
}

void runBenchmark(std::vector<Benchmark> &benchmarks, const std::string &benchmarkID) {
    CUcontext benchCtx;

    try {
        Benchmark bench = findBenchmark(benchmarks, benchmarkID);
        std::cout << "Running benchmark " << bench.benchKey() << ".\n";

        CU_ASSERT(cuCtxCreate(&benchCtx, 0, 0));
        CU_ASSERT(cuCtxSetCurrent(benchCtx));

        bench.benchFn()(bufferSize, loopCount);

        CU_ASSERT(cuCtxDestroy(benchCtx));
    } catch (std::string &s) {
        std::cout << "ERROR: " << s << std::endl;
    }
}

int main(int argc, char **argv) {
    averageLoopCount = defaultAverageLoopCount;
    disableP2P = true;

    std::vector<Benchmark> benchmarks = createBenchmarks();
    std::vector<std::string> benchmarksToRun;

    // Args parsing
    opt::options_description desc("NVBandwidth CLI");
    desc.add_options()
        ("help,h", "Produce help message")
        ("bufferSize", opt::value<unsigned long long int>(), "Memcpy buffer size")
        ("loopCount", opt::value<unsigned long long int>(), "Iterations of memcpy to be performed")
        ("parallel,p", "Run benchmark on each device at the same time")
        ("list,l", "List available benchmarks")
        ("benchmark,b", opt::value<std::vector<std::string>>()->multitoken(), "Benchmark(s) to doMemcpy (by name or index)")
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
        size_t numBenchmarks = benchmarks.size();
        std::cout << "Index, Name:\n\tDescription\n";
        std::cout << "=======================\n";
        for (unsigned int i = 0; i < numBenchmarks; i++) {
            std::cout << i << ", " << benchmarks.at(i).benchKey() << ":\n\t" << benchmarks.at(i).benchDesc() << "\n\n";
        }
        return 0;
    }

    if (vm.count("bufferSize")) bufferSize = vm["bufferSize"].as<unsigned long long int>();
    else bufferSize = defaultBufferSize;
    if (vm.count("loopCount")) loopCount = vm["loopCount"].as<unsigned long long int>();
    else loopCount = defaultLoopCount;

    if (vm.count("parallel")) parallel = true;
    else parallel = false;

    if (vm.count("verbose")) verbose = true;
    else verbose = false;

    if (vm.count("benchmark")) {
        benchmarksToRun = vm["benchmark"].as<std::vector<std::string>>();
    }

    cuInit(0);
    nvmlInit();
    CU_ASSERT(cuDeviceGetCount(&deviceCount));
    for (const auto& benchmarkIndex : benchmarksToRun) {
        runBenchmark(benchmarks, benchmarkIndex);
    }

    return 0;
}
