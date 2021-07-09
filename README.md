# NVBandwidth
Memory copy performance microbenchmark.

## Dependencies
To build and run NVBandwidth please install the Boost program_options (https://www.boost.org/doc/libs/1_66_0/doc/html/program_options.html) and Boost exceptions (https://www.boost.org/doc/libs/1_66_0/libs/exception/doc/boost-exception.html).

Ubuntu/Debian users can run the following to install:
```
apt install libboost-exception-dev libboost-program-options-dev
```

## Build
To build the `nvbandwidth` executable:
```
make
```

## Usage:
To list available benchmarks:
```
./nvbandwidth --list
```

To run a benchmarks:
```
./nvbandwidth --benchmark host_to_device_bidirectional_memcpy_ce
```
Example output:
```
Running benchmark host_to_device_bidirectional_memcpy_ce.
memcpy CE GPU(columns) <- CPU(rows) bandwidth (GB/s)
          0         1
0     12.33      0.34
```
