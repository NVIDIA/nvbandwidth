# nvbandwidth
A tool for bandwidth measurements on NVIDIA GPUs.

Measures bandwidth for various memcpy patterns across different links using copy engine or kernel copy methods.
nvbandwidth does not guarantee accuracy across all systems. Sytem specific tuning may be required to achieve maximum bandwidth.

## Dependencies
To build and run nvbandwidth please install the Boost program_options library (https://www.boost.org/doc/libs/1_66_0/doc/html/program_options.html).

Ubuntu/Debian users can run the following to install:
```
apt install libboost-program-options-dev
```

## Build
To build the `nvbandwidth` executable:
```
cmake .
make
```
You may need to set the BOOST_ROOT environment variable on Windows to tell CMake where to find your Boost installation.

## Usage:
```
./nvbandwidth -h
nvbandwidth CLI:
  -h [ --help ]                Produce help message
  --bufferSize arg (=67108864) Memcpy buffer size in bytes
  --loopCount arg (=16)        Iterations of memcpy to be performed
  -l [ --list ]                List available testcases
  -t [ --testcase ] arg        Testcase(s) to doMemcpy (by name or index)
  -v [ --verbose ]             Verbose output
```

To run all testcases:
```
./nvbandwidth
```

To run a specific testcase:
```
./nvbandwidth -t host_to_device_memcpy_ce
```
Example output:
```
Running host_to_device_memcpy_ce.
memcpy CE CPU(row) -> GPU(column) bandwidth (GB/s)
          0         1
0      6.20     12.36
```

Set number of iterations and the buffer size for copies with --loopCount and --bufferSize

## Test Details
There are two types of copies implemented, Copy Engine (CE) or Steaming Multiprocessor (SM)

CE copies use memcpy APIs. SM copies use kernels.

SM copies will truncate the copy size to fit uniformly on the target device to correctly report the bandwidth. The actual byte size for the copy is:
```
(512 * deviceSMCount) * floor(copySize / (512 * deviceSMCount))
```

### Measurement Details
![](diagrams/measurement.png)

A blocking kernel and CUDA events are used to measure time to perform copies via SM or CE, and bandwidth is calculated from a series of copies.

First, we enqueue a spin kernel that spins on a flag in host memory. The spin kernel spins on the device until all events for measurement have been fully enqueued into the measurement streams. This ensures that the overhead of enqueuing operations is excluded from the measurement of actual transfer over the interconnect. Next, we enqueue a start event, one or more iterations of memcpy, depending on loopCount, and finally a stop event. Finally, we release the flag to start the measurement.

This process is repeated 3 times, and the median bandwidth for each trial is reported.

### Unidirectional Bandwidth Tests
```
Running host_to_device_memcpy_ce.
memcpy CE CPU(row) -> GPU(column) bandwidth (GB/s)
          0         1
0      6.20     12.36
```

Unidirectional tests measure the bandwidth between each pair in the output matrix individually. Traffic is not sent simultaneously.

### Bidirectional Host <-> Device Bandwidth Tests
```
Running host_to_device_bidirectional_memcpy_ce.
memcpy CE CPU(row) <-> GPU(column) bandwidth (GB/s)
          0         1
0      5.64     11.16
```

The setup for bidirectional host to device bandwidth transfer is shown below:
![](diagrams/HtoDBidir.png)

Stream 0 (measured stream) performs writes to the device, while the interfering stream in the opposite direction produces reads. This pattern is reversed for measuring bidirectional device to host bandwidth as shown below.

![](diagrams/DtoHBidir.png)

### Bidirectional Device <-> Device Bandwidth Tests
The setup for bidirectional device to device transfers is shown below:

![](diagrams/DtoDBidir.png)

The test launches traffic on two streams: stream 0 launched on device 0 performs writes from device 0 to device 1 while the interference stream 1 launches opposite traffic performing writes from device 1 to device 0.

CE bidirectional bandwidth tests calculate bandwidth on the measured stream:
```
CE bidir. bandwidth = (size of data on measured stream) / (time on measured stream)
```
However, SM bidirectional test launches memcpy kernels on source and peer GPUs as independent streams and calculates bandwidth as:
```
SM bidir. bandwidth = size/(time on stream1) + size/(time on stream2)
```
