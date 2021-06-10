#include <stdio.h>
#include <iostream>
#include <string.h>
#include <string>
#include <time.h>
#include <cuda.h>
#include <iomanip>
#include <boost/program_options.hpp>
#include <boost/program_options/options_description.hpp>

#include "memcpy_ce_tests.h"

// Beware, if the multiplier is too high, the benchmark can hang because of GPFIFO exhaustion.
// CE benchmarks launch a spin kernel which prevents GPFIFO from emptying
// (but it makes the benchmark more stable). Each p2p copy adds 3 entries to GPFIFO.
// The usual GPFIFO size of 1024 allows us to schedule around 330-340 copies before deadlocking.
// P2P latency benchmarks schedule WARMUP_COUNT + loopCount * LATENCY_COUNT_MULTIPLIER copies
#define LATENCY_COUNT_MULTIPLIER 16

namespace opt = boost::program_options;

unsigned long long bufferSize;
unsigned int averageLoopCount;
bool noStridingTests;
bool noCheckBandwidth;
int pairingMode;
bool noTraffic;
bool isRandom;
unsigned int startValue;
unsigned int endValue;
unsigned int loopCount;
std::string ipc_key;
std::vector<int> element_sizes;
std::vector<int> num_threads_per_sm;
bool skip_verif;
bool fullNumaTest;
bool doFullSweep;
bool disableP2P;
bool enableCECopyStartMarker;

int main(int argc, char**argv) {
    std::string benchmark_name;

    // Args parsing
    opt::options_description desc("NVBandwidth CLI");
    desc.add_options()
        ("help", "produce help message")
        ("benchmark", opt::value<std::string>(), "benchmark to run")
    ;

    opt::variables_map vm;
    opt::store(opt::parse_command_line(argc, argv, desc), vm);
    opt::notify(vm);    

    if (vm.count("help")) {
        std::cout << desc << "\n";
        return 1;
    }

    benchmark_name = vm["benchmark"].as<std::string>();
    if (vm.count("benchmark")) {
        std::cout << "Running benchmark " << benchmark_name << ".\n";
    } else {
        std::cout << "Please specify which benchmark to run.\n";
        return 1;
    }

    if (benchmark_name == "host_to_device_bidirectional_memcpy") launch_HtoD_memcpy_bidirectional_CE("Host ot Device Bidirectional memcpy");
    else if (benchmark_name == "device_to_host_bidirectional_memcpy") launch_DtoH_memcpy_bidirectional_CE("Device to Host Bidirectional memcpy");

    return 0;
}
