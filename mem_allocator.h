#ifndef MEM_ALLOCATOR
#define MEM_ALLOCATOR

void* allocateHostMemory(size_t size, bool isPageable);

void freeHostMemory(void *memory);

bool isMemoryOwnedByCUDA(void *memory);

#endif
