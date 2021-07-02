#include <iostream>
#include <string>
#include "common.h"

typedef void (*benchfn_t)(const std::string &, unsigned long long, unsigned long long);

class Benchmark {
    benchfn_t benchmark_func;
    std::string desc;
public:
    Benchmark() {}

    Benchmark(benchfn_t benchmark_func, std::string desc): benchmark_func(benchmark_func), desc(desc) {}

    benchfn_t benchFn() {
        return benchmark_func;
    }

    std::string description() {
        return desc;
    }
};

class BenchParams {
public:
    // what should be copied on where on which device
    void *dst;
    void *src;
    CUcontext ctx;

    BenchParams(void *_dst, void *_src, CUcontext _ctx) {
        dst = _dst;
        src = _src;
        ctx = _ctx;
    }
};

// CE Benchmarks
void launch_HtoD_memcpy_bidirectional_CE(const std::string &test_name, unsigned long long size = defaultBufferSize, unsigned long long loopCount = defaultLoopCount);
void launch_DtoH_memcpy_bidirectional_CE(const std::string &test_name, unsigned long long size = defaultBufferSize, unsigned long long loopCount = defaultLoopCount);

// SM Benchmarks
void launch_HtoD_memcpy_SM(const std::string &test_name, unsigned long long size, unsigned long long loopCount);
void launch_DtoH_memcpy_SM(const std::string &test_name, unsigned long long size, unsigned long long loopCount);
void launch_DtoD_memcpy_read_SM(const std::string &test_name, unsigned long long size, unsigned long long loopCount);
void launch_DtoD_memcpy_write_SM(const std::string &test_name, unsigned long long size, unsigned long long loopCount);
void launch_DtoD_memcpy_bidirectional_read_SM(const std::string &test_name, unsigned long long size, unsigned long long loopCount);
void launch_DtoD_memcpy_bidirectional_write_SM(const std::string &test_name, unsigned long long size, unsigned long long loopCount);
