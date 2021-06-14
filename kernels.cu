__global__ void spinKernel(volatile int *latch, const unsigned long long timeout_clocks)
{
    register unsigned long long end_time = clock64() + timeout_clocks;
    while (!*latch) {
        if (timeout_clocks != ~0ULL && clock64() > end_time) {
            break;
        }
    }
}
