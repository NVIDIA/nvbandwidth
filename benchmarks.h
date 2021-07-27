#ifndef BENCHMARKS_H
#define BENCHMARKS_H

#include "common.h"
#include <map>

typedef void (*benchfn_t)(unsigned long long, unsigned long long);

class Benchmark {
  	benchfn_t benchmark_func;
  	std::string desc;

public:
  	Benchmark() {}

  	Benchmark(benchfn_t benchmark_func, std::string desc): benchmark_func(benchmark_func), desc(desc) {}

  	benchfn_t bench_fn() { return benchmark_func; }

  	std::string description() { return desc; }
};

// CE Benchmarks
void launch_HtoD_memcpy_CE(unsigned long long size = defaultBufferSize, unsigned long long loopCount = defaultLoopCount);
void launch_DtoH_memcpy_CE(unsigned long long size = defaultBufferSize, unsigned long long loopCount = defaultLoopCount);
void launch_HtoD_memcpy_bidirectional_CE(unsigned long long size = defaultBufferSize, unsigned long long loopCount = defaultLoopCount);
void launch_DtoH_memcpy_bidirectional_CE(unsigned long long size = defaultBufferSize, unsigned long long loopCount = defaultLoopCount);
void launch_DtoD_memcpy_read_CE(unsigned long long size, unsigned long long loopCount);
void launch_DtoD_memcpy_write_CE(unsigned long long size, unsigned long long loopCount);
void launch_DtoD_memcpy_bidirectional_CE(unsigned long long size, unsigned long long loopCount);
// SM Benchmarks
void launch_HtoD_memcpy_SM(unsigned long long size, unsigned long long loopCount);
void launch_DtoH_memcpy_SM(unsigned long long size, unsigned long long loopCount);
void launch_DtoD_memcpy_read_SM(unsigned long long size, unsigned long long loopCount);
void launch_DtoD_memcpy_write_SM(unsigned long long size, unsigned long long loopCount);
void launch_DtoD_memcpy_bidirectional_read_SM(unsigned long long size, unsigned long long loopCount);
void launch_DtoD_memcpy_bidirectional_write_SM(unsigned long long size, unsigned long long loopCount);

#endif
