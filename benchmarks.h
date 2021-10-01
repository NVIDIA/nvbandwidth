#ifndef BENCHMARKS_H
#define BENCHMARKS_H

#include "common.h"
#include "memory_utils.h"
#include "memcpy.h"
#include <map>
#include <utility>

typedef void (*benchfn_t)(unsigned long long, unsigned long long);

class Benchmark {
    std::string key;
    std::string desc;
    benchfn_t benchmark_func{};

public:
    Benchmark() = default;
    Benchmark(std::string key, benchfn_t benchmark_func, std::string desc): key(std::move(key)), benchmark_func(benchmark_func), desc(std::move(desc)) {}
    std::string benchKey() { return key; }
    benchfn_t benchFn() { return benchmark_func; }
    std::string benchDesc() { return desc; }
};

// CE Benchmarks
void launch_HtoD_memcpy_CE(unsigned long long size = defaultBufferSize, unsigned long long loopCount = defaultLoopCount);
void launch_DtoH_memcpy_CE(unsigned long long size = defaultBufferSize, unsigned long long loopCount = defaultLoopCount);
void launch_HtoD_memcpy_bidirectional_CE(unsigned long long size = defaultBufferSize, unsigned long long loopCount = defaultLoopCount);
void launch_DtoH_memcpy_bidirectional_CE(unsigned long long size = defaultBufferSize, unsigned long long loopCount = defaultLoopCount);
void launch_DtoD_memcpy_read_CE(unsigned long long size, unsigned long long loopCount);
void launch_DtoD_memcpy_write_CE(unsigned long long size, unsigned long long loopCount);
void launch_DtoD_memcpy_bidirectional_CE(unsigned long long size, unsigned long long loopCount);
void launch_DtoD_paired_memcpy_read_CE(unsigned long long size, unsigned long long loopCount);
void launch_DtoD_paired_memcpy_write_CE(unsigned long long size, unsigned long long loopCount);
// SM Benchmarks
void launch_HtoD_memcpy_SM(unsigned long long size, unsigned long long loopCount);
void launch_DtoH_memcpy_SM(unsigned long long size, unsigned long long loopCount);
void launch_DtoD_memcpy_read_SM(unsigned long long size, unsigned long long loopCount);
void launch_DtoD_memcpy_write_SM(unsigned long long size, unsigned long long loopCount);
void launch_DtoD_memcpy_bidirectional_read_SM(unsigned long long size, unsigned long long loopCount);
void launch_DtoD_memcpy_bidirectional_write_SM(unsigned long long size, unsigned long long loopCount);
void launch_DtoD_paired_memcpy_read_SM(unsigned long long size, unsigned long long loopCount);
void launch_DtoD_paired_memcpy_write_SM(unsigned long long size, unsigned long long loopCount);

#endif
