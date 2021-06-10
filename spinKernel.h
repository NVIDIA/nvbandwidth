#ifndef _SPINKERNEL_H_
#define _SPINKERNEL_H_

#include <cuda.h>

const unsigned long long DEFAULT_SPIN_KERNEL_TIMEOUT = 10000000000ULL;   // 10 seconds
const unsigned long long INFINITE_SPIN_KERNEL_TIMEOUT = ~0ULL;

CUresult launch_spin_kernel(volatile int *latch, CUstream stream, bool single=false, unsigned long long timeout_ns=DEFAULT_SPIN_KERNEL_TIMEOUT);

#endif // _SPINKERNEL_H_
