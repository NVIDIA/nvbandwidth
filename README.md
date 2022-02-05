# NVBandwidth
Memory copy performance microbenchmark.

## Dependencies
To build and run NVBandwidth please install the Boost program_options library (https://www.boost.org/doc/libs/1_66_0/doc/html/program_options.html).

Ubuntu/Debian users can run the following to install:
```
apt install libboost-program-options-dev
```

## Build
To build the `nvbandwidth` executable:
```
make
```
CUDA is assumed to be installed in /usr/local/cuda

## Usage:
```
./nvbandwidth -h
NVBandwidth CLI:
  -h [ --help ]                Produce help message
  --bufferSize arg (=67108864) Memcpy buffer size
  --loopCount arg (=16)        Iterations of memcpy to be performed
  -l [ --list ]                List available benchmarks
  -b [ --benchmark ] arg       Benchmark(s) to doMemcpy (by name or index)
  -v [ --verbose ]             Verbose output
```

To run all benchmarks:
```
./nvbandwidth
```

To run a specific benchmark:
```
./nvbandwidth -b host_to_device_memcpy_ce
```
Example output:
```
Running benchmark host_to_device_memcpy_ce.
memcpy CE CPU(row) -> GPU(column) bandwidth (GB/s)
          0         1
0      6.20     12.36
```

Set number of iterations and the buffer size for copies with --loopCount and --bufferSize

## Test Details
There are two types of copies implemented, Copy Engine (CE) or Steaming Multiprocessor (SM)

CE copies use memcpy APIs. SM copies use kernels.

### Unidirectional Bandwidth Tests
```
Running benchmark host_to_device_memcpy_ce.
memcpy CE CPU(row) -> GPU(column) bandwidth (GB/s)
          0         1
0      6.20     12.36
```

Unidirectional tests measure the bandwidth between each pair in the output matrix individually. Traffic is not sent simultaneously.

### Bidirectional Bandiwdth Tests
```
Running benchmark host_to_device_bidirectional_memcpy_ce.
memcpy CE CPU(row) <-> GPU(column) bandwidth (GB/s)
          0         1
0      5.64     11.16
```

The setup for bidirectional host to device bandwidth transfer is shown below:
![](diagrams/HtoDBidir.png)

Stream 0 (measured stream) performs writes to the device, while the interfering stream in the opposite direction produces reads. This pattern is reversed for measuring bidirectional device to host bandwidth as shown below.

![](diagrams/DtoHBidir.png)
