#include <cuda.h>
#include <stdio.h>

bool isMemoryOwnedByCUDA(void *memory) {
	CUmemorytype memorytype;
	CUresult status = cuPointerGetAttribute(&memorytype, CU_POINTER_ATTRIBUTE_MEMORY_TYPE, (CUdeviceptr) memory);
	if (status == CUDA_ERROR_INVALID_VALUE) {
		return false;
	} else {
		// ASSERT_DRV(status);
		return true;
	}
}

void* allocateHostMemory(size_t size, bool isPageable)
{
	void *memory;
	if (isPageable) {
		memory = malloc(size);
		// ASSERT(memory, "");
	} else {
		cuMemHostAlloc(&memory, size, CU_MEMHOSTALLOC_PORTABLE); // ASSERT_DRV
	}
	return memory;
}

void freeHostMemory(void *memory)
{
	if (isMemoryOwnedByCUDA(memory)) {
		cuMemFreeHost(memory); //ASSERT_DRV();
	} else {
		free(memory);
	}
}
