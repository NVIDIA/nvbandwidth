#ifndef _MEM_PATTERN_H_
#define _MEM_PATTERN_H_


/// memset_pattern set a pattern generated with 'seed' into buffer.
/// This call is using the current cuda context to perform the set
/// and it causes a ctxSynchronize
void memset_pattern(void* buffer, unsigned long long size, unsigned int seed);
/// memcmp_pattern compare the buffer with the reference pattern generated with 'seed'.
/// if the comparison fail, the call will raise a dfontaine/testsuite assert
/// This call is using the current cuda context to perform the set
/// and it causes a ctxSynchronize
void memcmp_pattern(void* buffer, unsigned long long size, unsigned int seed);

#endif // _MEM_PATTERN_H_
