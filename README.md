# nvbandwidth
A tool for bandwidth measurements on NVIDIA GPUs.

Measures bandwidth for various memcpy patterns across different links using copy engine or kernel copy methods.
nvbandwidth reports current measured bandwidth on your system. Additional system-specific tuning may be required to achieve maximal peak bandwidth.

## Requirements

Linux is the primary documented platform. Windows builds are possible but may require additional setup (see **Build** below).

The following components are required:

- **CUDA Toolkit** (version 11.x or above). Multinode version requires 12.3 toolkit and 550 driver or above.
- **C++17 compiler**. GCC 7.x or above on Linux; MSVC 2019 or above on Windows.
- **CMake** (version 3.20 or above). CMake version 3.24 or newer is encouraged.

On Linux, ensure the path to the `nvcc` binary (installed via the CUDA Toolkit) is available in `$PATH`.

The system must have a CUDA-enabled GPU and an NVIDIA display driver compatible with the CUDA Toolkit used to build nvbandwidth.
For more information, refer to https://docs.nvidia.com/deploy/cuda-compatibility/

## Build

To build the `nvbandwidth` executable for single-node:
```bash
cmake .
make
```

On Ubuntu/Debian, a utility script is provided that installs the required software components and builds the project:
```bash
sudo ./debian_install.sh
```

On Windows, ensure MSVC and the CUDA Toolkit are on your `PATH` before running CMake.

## Usage:
```
./nvbandwidth -h

Usage: nvbandwidth [options]

A tool for bandwidth measurements on NVIDIA GPUs.

Arguments:
  -h, --help                shows help message and exits
      --version             Print nvbandwidth release version and exit
  -b, --bufferSize SIZE     Memcpy buffer size in MiB [default: 512]
  -l, --list                List available testcases
  -t, --testcase TEST       Testcase(s) to run (by name or index)
  -p, --testcasePrefixes PREFIX  Testcase(s) to run (by prefix)
  -v, --verbose             Verbose output
  -s, --skipVerification    Skips data verification after copy
  -d, --disableAffinity     Disable automatic CPU affinity control
  -i, --testSamples N       Iterations of the benchmark [default: 3]
  -P, --targetNumPairs N    Target pairs for multinode tests [default: -1]
  -m, --useMean             Use mean instead of median for results
  -H, --useHugePages        Use huge pages for host allocations
  -F, --format FORMAT       Output format: text, json, or perf [default: text]
      --pair UUID1 UUID2    Run device-to-device tests only between these two GPUs (by UUID)

Examples:
  nvbandwidth                          # Run all testcases
  nvbandwidth -l                       # List available testcases
  nvbandwidth -t host_to_device_memcpy_ce device_to_host_memcpy_ce
  nvbandwidth -p device_to_device      # Run all device_to_device* tests
  nvbandwidth --format json            # Output in JSON format
```

To run all testcases:
```
./nvbandwidth
```

To run a specific testcase:
```
./nvbandwidth -t device_to_device_memcpy_write_ce
```
Example output:
```
Running device_to_device_memcpy_write_ce.
memcpy CE GPU(row) <- GPU(column) bandwidth (GB/s)
          0         1         2         3         4         5         6         7
0      0.00    276.07    276.36    276.14    276.29    276.48    276.55    276.33
1    276.19      0.00    276.29    276.29    276.57    276.48    276.38    276.24
2    276.33    276.29      0.00    276.38    276.50    276.50    276.29    276.31
3    276.19    276.62    276.24      0.00    276.29    276.60    276.29    276.55
4    276.03    276.55    276.45    276.76      0.00    276.45    276.36    276.62
5    276.17    276.57    276.19    276.50    276.31      0.00    276.31    276.15
6    274.89    276.41    276.38    276.67    276.41    276.26      0.00    276.33
7    276.12    276.45    276.12    276.36    276.00    276.57    276.45      0.00
```

Set number of iterations and the buffer size for copies with --testSamples and --bufferSize

## Pairwise GPU Testing

The `--pair` option restricts device-to-device tests to a single, explicitly specified pair of GPUs identified by their UUIDs. This is useful for isolating and measuring a specific link without running the full all-pairs sweep.

### Finding GPU UUIDs

A GPU UUID has the form `GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx`, where each `x` is a hexadecimal digit — for example, `GPU-9a6efb31-7320-ce3c-b72c-75d85b6d0cea`. The `GPU-` prefix is required.

On a single node, `nvidia-smi -L` lists every GPU with its UUID:

```
$ nvidia-smi -L
GPU 0: NVIDIA <GPU model> (UUID: GPU-<uuid-0>)
GPU 1: NVIDIA <GPU model> (UUID: GPU-<uuid-1>)
...
```

Across multiple nodes, use `mpirun` with your hostfile to query all nodes at once. The `--map-by ppr:1:node` mapping runs one process per node; increase the count to match the number of GPUs per node if you need all UUIDs:

```
# One process per GPU — shows all GPUs on every node (e.g. 4 GPUs/node, 2 nodes = -n 8)
mpirun --hostfile /etc/nvidia-imex/nodes_config.cfg \
    --map-by ppr:4:node -n 8 \
    nvidia-smi -L
```

Note that output from different nodes may be interleaved; redirect per-host output with `--output-filename` if you need clean separation.

`--pair` affects the following testcase families:
- `device_to_device_memcpy_*`
- `device_to_device_bidirectional_memcpy_*`
- `device_to_device_latency_sm` / `device_to_device_latency_tma`
- `multinode_device_to_device_memcpy_*`
- `multinode_device_to_device_bidirectional_memcpy_*`

### Buffer size

A small buffer (e.g. `-b 1`) is sufficient for quickly verifying connectivity between two GPUs. Use the default 512 MiB buffer (or omit `-b` entirely) when measuring peak bandwidth, as smaller buffers may underreport achievable throughput.

### Single-node pairwise test

Pass both UUIDs directly. The matrix output will show only the row/column entries for those two devices; all other pairs are skipped.

```
./nvbandwidth -t device_to_device_memcpy_read_sm \
    --pair <GPU-UUID-0> <GPU-UUID-1>
```

Example output:
```
Running device_to_device_memcpy_read_sm.
memcpy SM GPU(row) -> GPU(column) bandwidth (GB/s)
           0         1         2         3
 0       N/A       N/A       N/A    <BW>

SUM device_to_device_memcpy_read_sm <SUM>
COEFFICIENT_OF_VARIATION device_to_device_memcpy_read_sm <CV>
```

### Multinode pairwise test

For multinode tests, run with exactly 2 MPI ranks (one per GPU). Each rank's UUID must be the GPU that rank will use. A hostfile pointing to the two nodes of interest must already be configured for MPI.

```
mpirun -n 2 ./nvbandwidth -t multinode_device_to_device_memcpy_read_sm \
    --pair <GPU-UUID-0> <GPU-UUID-1>
```

Example output:
```
Running multinode_device_to_device_memcpy_read_sm.
memcpy SM GPU(row) -> GPU(column) bandwidth (GB/s)
           0         1
 0       N/A    <BW>
 1    <BW>       N/A

SUM multinode_device_to_device_memcpy_read_sm <SUM>
COEFFICIENT_OF_VARIATION multinode_device_to_device_memcpy_read_sm <CV>
```

## Multinode benchmarks

In order to build multinode version of nvbandwidth, execute

```
cmake -DMULTINODE=1 .
make
```

Multinode nvbandwidth requires MPI. CMake will find a local installation of MPI to build and link against. Multinode also requires installing, setting up the IMEX service, and creating the IMEX channels. IMEX (NVIDIA Internode Memory Exchange) runs on each compute tray to support GPU memory export and import operations across OS domains in an NVLink multi-node deployment. To start the IMEX service, run the following command: 

`sudo systemctl start nvidia-imex.service`
Specify the IP addresses of the cluster nodes in /etc/nvidia-imex/nodes_config.cfg file.

For example, to run multinode bandwidth on a system with 2 nodes and 4 GPUs per node run the command:
`mpirun --allow-run-as-root --map-by ppr:4:node --bind-to core -np 8 --report-bindings -q -mca btl_tcp_if_include enP5p9s0 --hostfile /etc/nvidia-imex/nodes_config.cfg  ./nvbandwidth -p multinode`

**Note**: `enP5p9s0` is a site-specific network interface name. Replace it with the appropriate interface on your cluster (e.g. the output of `ip link show`), or omit `-mca btl_tcp_if_include` entirely if your MPI environment or job scheduler already handles network binding.

### Pair Sampling for Multinode Tests

For large multinode systems, testing all possible GPU pairs can be time-consuming and resource-intensive. nvbandwidth provides a sampling option to reduce test time while maintaining good coverage of the GPU topology.

#### Sampling Options

- **`--targetNumPairs -1`** (default): Test all possible pairs (N×(N-1) for N GPUs)
- **`--targetNumPairs <number>`**: Test exactly `<number>` pairs using intelligent sampling

For example, to run multinode bandwidth on a system with 4 nodes and 8 GPUs per node (32 GPUs total), and select 8 pairs of GPUs run the command:
`mpirun --allow-run-as-root --map-by ppr:8:node --bind-to core -np 32 --report-bindings -q -mca btl_tcp_if_include enP5p9s0 --hostfile /etc/nvidia-imex/nodes_config.cfg  ./nvbandwidth -p multinode  --targetNumPairs 8`

**Note**: Replace `enP5p9s0` with your cluster's network interface name (see note above).

The command selects random GPU pairs, maximizing coverage of unique GPUs. Bandwidth (BW) for untested pairs will be set to NA.

#### Sampling Algorithm

When using sampling (`targetNumPairs > 0` and less than total pairs), nvbandwidth employs a two-phase approach:

1. **GPU Coverage Phase**: Ensures each GPU participates in at least one test pair
2. **Random Filling Phase**: Fills remaining slots with random pairs, prioritizing uncovered GPUs

This approach maximizes GPU topology coverage even with limited test pairs.

#### Examples

```bash
# Test all pairs (default/full coverage) on a 4-node, 8-GPU/node system (32 ranks)
mpirun -n 32 ./nvbandwidth -p multinode --targetNumPairs -1

# Test 20 carefully selected pairs
mpirun -n 32 ./nvbandwidth -p multinode --targetNumPairs 20

# For a 4-node, 8-GPU/node system: 992 total pairs available (32×31)
# Using --targetNumPairs 100 will test ~10% of pairs while covering all GPUs
```

**Note**: The `--targetNumPairs` parameter only affects multinode device-to-device tests. In single-node mode, this parameter is ignored and a warning will be displayed if a positive value is specified.

### Local testing

You can test it on a single-node machine (Ampere+ GPU required):

```
mpirun -n 4 ./nvbandwidth -p multinode
```
This command will spawn 4 processes, and run all tests with "multinode" prefix.

### Running it on a cluster

To run it on a cluster, submit a job to a workload scheduler that has MPI integration. Run one process per GPU.

Running less processes than GPU count is valid, processes will take consecutive GPUs, starting from GPU 0.

Running more processes than GPU count is not valid.

All ranks in the MPI batch must be part of one multinode clique. Run one MPI rank per GPU (typically one rank per GPU per node).

When running under MPI, only MPI rank 0 will output stdout to the console. Stderr, if needed, will be output by all processes.

It is recommended to only run "multinode*" testcases under MPI. While any testcase will succeed, results for non multinode testcases will only come from MPI rank 0.

## Test Details
There are two types of copies implemented, Copy Engine (CE) or Streaming Multiprocessor (SM)

CE copies use memcpy APIs. SM copies use kernels.

SM copies will truncate the copy size to fit uniformly on the target device to correctly report the bandwidth. The actual byte size for the copy is:
```
(threadsPerBlock * deviceSMCount) * floor(copySize / (threadsPerBlock * deviceSMCount))
```

threadsPerBlock is set to 512.

### Latency Measurements

nvbandwidth uses **pointer chasing** to measure memory latency rather than simple sequential access patterns. This methodology provides more realistic latency measurements by forcing random memory access patterns that prevent prefetching optimizations.

#### How Pointer Chasing Works

1. **Setup**: Memory is organized as a linked list where each node contains a pointer to the next node
2. **Pattern**: The chain follows a strided pattern through memory: `Node[i] -> Node[(i + stride) % total_nodes]`
3. **Execution**: The kernel follows this pointer chain for a specified number of accesses
4. **Measurement**: Total time is divided by number of accesses gives latency per access

#### Important Notes

- **TLB costs are not attributed to latency in these measurements**: the same buffer is reused across all accesses, keeping TLB entries warm so that address translation overhead does not appear in the reported latency numbers. TLB lookups still occur in hardware; they are simply not what this microbenchmark is measuring.
- **Data cache hits prevented**: Random pointer chasing patterns prevent data cache benefits by design

This approach provides latency measurements that benefit from TLB optimization while maintaining random memory access patterns that prevent data cache effects.

### Measurement Details
![](diagrams/measurement.png)

A blocking kernel and CUDA events are used to measure time to perform copies via SM or CE, and bandwidth is calculated from a series of copies.

First, we enqueue a spin kernel that spins on a flag in host memory. The spin kernel spins on the device until all events for measurement have been fully enqueued into the measurement streams. This ensures that the overhead of enqueuing operations is excluded from the measurement of actual transfer over the interconnect. Next, we enqueue a start event, certain count of memcpy iterations, and finally a stop event. Finally, we release the flag to start the measurement.

This process is repeated 3 times by default (controlled by `--testSamples`), and the median bandwidth for each trial is reported.

Number of repetitions can be overridden using the --testSamples option, and in order to use arithmetic mean instead of median you can specify --useMean option.

### Unidirectional Bandwidth Tests
```
Running host_to_device_memcpy_ce.
memcpy CE CPU(row) -> GPU(column) bandwidth (GB/s)
          0         1         2         3         4         5         6         7
0     26.03     25.94     25.97     26.00     26.19     25.95     26.00     25.97
```

Unidirectional tests measure the bandwidth between each pair in the output matrix individually. Traffic is not sent simultaneously.

### Bidirectional Host <-> Device Bandwidth Tests
```
Running host_to_device_bidirectional_memcpy_ce.
memcpy CE CPU(row) <-> GPU(column) bandwidth (GB/s)
          0         1         2         3         4         5         6         7
0     18.56     18.37     19.37     19.59     18.71     18.79     18.46     18.61
```

The setup for bidirectional host to device bandwidth transfer is shown below:
![](diagrams/HtoDBidir.png)

**CE copies**  
Stream 0 (measured stream) performs writes to the device, while the interfering stream in the opposite direction produces reads. This pattern is reversed for measuring bidirectional device to host bandwidth as shown below.


![](diagrams/DtoHBidir.png)

**SM copies**  
The test launches a kernel copy where alternating thread warps are copying data in alternating directions.

### Bidirectional Device <-> Device Bandwidth Tests
The setup for bidirectional device to device transfers is shown below:

![](diagrams/DtoDBidir.png)

**CE copies**  
Stream 0 (measured stream) performs writes to the device, while the interfering stream in the opposite direction produces reads.

**SM Copies**  
Similar to the HtoDBidir test above, the test launches a kernel where alternating thread warps copy data in alternating directions.

**Bandwidth calculation**  
CE bidirectional bandwidth tests calculate bandwidth on the measured stream:
```
CE bidir. bandwidth = (size of data on measured stream) / (time on measured stream)
```
However, SM bidirectional test launches a kernel where odd and even warps are copying data in different directions. Bandwidth is calculated as shown below:
```
SM bidir. bandwidth = size/(kernel time);
```

**Interpreting bidirectional results**
In bidirectional tests, nvbandwidth drives traffic in both directions concurrently,  one stream is the "measured stream" while another stream generates traffic in the opposite direction. 
The reported bidirectional GB/s represents per-direction bandwidth under concurrent opposite-direction copy, not an aggregated sum. 
 
## Known Issues

#### 1. Invalid latency measurements on confidential computing platforms

On platforms with [NVIDIA Confidential Computing](https://www.nvidia.com/en-us/data-center/solutions/confidential-computing/) enabled in a mode which uses bounce buffers for CPU<->GPU communication, the latency measurements reported by nvbandwidth
may be negative or otherwise invalid since [`cudaEventElapsedTime` is unreliable in these modes](https://docs.nvidia.com/nvidia-secure-ai-operations-guide.pdf).

This will be addressed with a future CUDA driver update.
