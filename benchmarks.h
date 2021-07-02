#include <iostream>
#include <string>
#include "common.h"

typedef void (*BenchmarkFunc)(const std::string &, unsigned long long, unsigned long long);

class Benchmark {
    BenchmarkFunc benchmark_func;
    std::string desc;
public:
    Benchmark() {}
    
    Benchmark(BenchmarkFunc benchmark_func, std::string desc): benchmark_func(benchmark_func), desc(desc) {}

    BenchmarkFunc benchFn() {
        return benchmark_func;
    }

    std::string description() {
        return desc;
    }
};

// CE Benchmarks
void launch_HtoD_memcpy_bidirectional_CE(const std::string &test_name, unsigned long long size = defaultBufferSize, unsigned long long loopCount = defaultLoopCount);
void launch_DtoH_memcpy_bidirectional_CE(const std::string &test_name, unsigned long long size = defaultBufferSize, unsigned long long loopCount = defaultLoopCount);

// SM Benchmarks
void launch_HtoD_memcpy_SM(const std::string &test_name, unsigned long long size, unsigned long long loopCount);
