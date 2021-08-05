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

If you plan on making changes to `sm_memcpy_kernel.cu` which is the module containing the memcpy kernel used for SM tests, run
```
nvcc -ptx -c sm_memcpy_kernel.cu -o sm_memcpy_kernel.ptx
```

## Usage:
To list available benchmarks:
```
./nvbandwidth -l
```

To run a benchmarks:
```
./nvbandwidth -b host_to_device_bidirectional_memcpy_ce
```
Example output:
```
Running benchmark host_to_device_bidirectional_memcpy_ce.
memcpy CE GPU(columns) <- CPU(rows) bandwidth (GB/s)
          0         1
0     12.33      0.34
```
