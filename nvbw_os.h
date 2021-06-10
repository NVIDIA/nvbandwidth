#ifndef __NVBW_OS_H__
#define __NVBW_OS_H__
#include <cstdint>
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <semaphore.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>
#include <linux/limits.h> // PATH_MAX
#include <signal.h>

// TODO : Some compatibility stuff from the original code that I'll revisit
#ifdef __GNUC__
#define GCC_PRAGMA(x) GCC_PRAGMA_DO(GCC x)
#define GCC_PRAGMA_DO(x) _Pragma(#x)
#define GCC_DIAG_PRAGMA(x)
#endif

// TODO : Double check but seems to be <these values> everywhere
#define PROT_NONE 0x0
#define MAP_ANONYMOUS 0x20
#define MAP_FIXED 0x10

#define MS_PER_S 1000
#define NS_PER_US 1000
#define NS_PER_MS 1000000
#define NS_PER_S 1000000000

#define SYS_SHM_EX_TEMPLATE_MAX 40
#define MODULE_NAME_LOWERCASE "cuda" // TODO : Based on what I found
#define SYS_SHM_EX_TEMPLATE "/" MODULE_NAME_LOWERCASE ".shm.%x.%x.%llx"

#define SYS_INFINITE_TIMEOUT ((unsigned int)~0)

// "memory" arguments to inline assembly prevent the compiler reordering
// barrier intrinsics over any other loads or stores
#define StoreFence() asm volatile("sfence" : : : "memory")
#define LoadFence() asm volatile("lfence" : : : "memory")
#define MemFence() asm volatile("mfence" : : : "memory")

/* Clear a bit in an affinity mask */
#define SysProcessorMaskClear(mask, proc)                                      \
  do {                                                                         \
    size_t _proc = (proc);                                                     \
    (mask)[_proc / SysProcessorMaskWordBits()] &=                              \
        ~((size_t)1 << (_proc % SysProcessorMaskWordBits()));                  \
  } while (0)

/* Return the number of bits per word */
#define SysProcessorMaskWordBits() (8 * sizeof(size_t))

/* Zero-fill a mask */
#define SysProcessorMaskZero(mask) memset((mask), '\0', cuosProcessorMaskSize())

/* Set a bit in an affinity mask */
#define SysProcessorMaskSet(mask, proc)                                        \
  do {                                                                         \
    size_t _proc = (proc);                                                     \
    (mask)[_proc / SysProcessorMaskWordBits()] |=                              \
        (size_t)1 << (_proc % SysProcessorMaskWordBits());                     \
  } while (0)

/* Clear a bit in an affinity mask */
#define SysProcessorMaskClear(mask, proc)                                     \
  do {                                                                         \
    size_t _proc = (proc);                                                     \
    (mask)[_proc / SysProcessorMaskWordBits()] &=                              \
        ~((size_t)1 << (_proc % SysProcessorMaskWordBits()));                  \
  } while (0)

// TODO : Change comments
// Rounds n up to the nearest multiple of "multiple".
// if n is already a multiple of "multiple", n is returned unchanged.
// works for arbitrary value of "multiple".
#define SYS_ROUND_UP(n, multiple)                                              \
  (((n) + ((multiple)-1)) - (((n) + ((multiple)-1)) % (multiple)))

// Rounds n down to the nearest multiple of "multiple"
#define SYS_ROUND_DOWN(n, multiple) ((n) - ((n) % (multiple)))

// Round to the nearest multiple of 'a' assuming 'a' is a power of two
#define SYS_ROUND_DOWN_2(p, a) ((p) & ~((a)-1))
#define SYS_ROUND_UP_2(p, a) (SYS_ROUND_DOWN_2((p) + (a)-1, a))

#define SYS_MIN(a, b) ((a) < (b) ? (a) : (b))
#define SYS_MAX(a, b) (((a) > (b)) ? (a) : (b))

typedef volatile unsigned int cuosOnceWithArgsControl;
#define SYS_ONCE_WITH_ARGS_INIT 0

/* Return the number of bits per word */
#define SysProcessorMaskWordBits() (8 * sizeof(size_t))

// TODO : Double check and change comments
// These custom atomic implementations are done with macros instead of inline
// functions so they can be reused for multiple operand sizes.

// Atomically performs the compare-and-swap operation:
//
// temp = *ptr
// if(temp == old)
//     *ptr = new
// old = temp
//
// cmpxchg implicitly uses the eax/rax register for the old value.
// TODO : Used to be `__asm__ __volatile ("lock; cmpxchg %4, %0;"` but in this
// case I am yet to see difference
#define SYS_CMPXCHG(ptr, old, new)                                             \
  asm("lock; cmpxchg %4, %0;"                                                  \
      : "=m"(*ptr), "=a"(old)                                                  \
      : "m"(*ptr), "a"(old), "r"(new)                                          \
      : "cc", "memory")

// TODO : Double check and change comments
// Atomically performs the combined exchange-and-add operation:
//
// temp = *ptr
// *ptr += val
// val = temp
//
// xadd takes two operands, a register and a pointer, and performs the above
// operation on them. The lock prefix makes the operation atomic.
#define SYS_XADD(ptr, val)                                                     \
  __asm__ __volatile__("lock; xadd %0, %2;"                                    \
                       : "+r"(val), "=m"(*ptr)                                 \
                       : "m"(*ptr)                                             \
                       : "cc", "memory")

#define SYS_CLOCK_UNINITIALIZED 0xFFFFFFFF

// TODO : Change comments
/* Return codes indicating the status of cuos function calls. These are not yet
 * in standard use across the library, but they should be used whenever possible
 * to start converging towards a standard error reporting system. */
enum { SYS_SUCCESS = 0, SYS_ERROR = -1, SYS_TIMEOUT = -2, SYS_EOF = -3 };

/* File seek operation flags */
typedef enum {
  SYS_SEEK_SET = 0,
  SYS_SEEK_CUR = 1,
  SYS_SEEK_END = 3
} SysSeekEnum;

// Enabling lse redirection to legacy cuosInterlocked* // TODO : Still needed?
typedef unsigned long long (*fp_sync_lock_test_and_set)(
    volatile unsigned int *ptr, const unsigned int val);
// 32
typedef unsigned int (*fp_sync_val_compare_and_swap)(volatile unsigned int *v,
                                                     unsigned int compare,
                                                     unsigned int exchange);
typedef unsigned int (*fp_sync_fetch_and_and)(volatile unsigned int *v,
                                              unsigned int mask);
typedef unsigned int (*fp_sync_fetch_and_or)(volatile unsigned int *v,
                                             unsigned int mask);
typedef unsigned int (*fp_sync_add_and_fetch)(volatile unsigned int *v,
                                              const unsigned int val);
typedef unsigned int (*fp_sync_sub_and_fetch)(volatile unsigned int *v,
                                              const unsigned int val);
// 64
typedef unsigned long long (*fp_sync_val_compare_and_swap64)(
    volatile unsigned long long *v, unsigned long long compare,
    unsigned long long exchange);
typedef unsigned long long (*fp_sync_fetch_and_and64)(
    volatile unsigned long long *v, unsigned long long mask);
typedef unsigned long long (*fp_sync_fetch_and_or64)(
    volatile unsigned long long *v, unsigned long long mask);
typedef unsigned long long (*fp_sync_add_and_fetch64)(
    volatile unsigned long long *v, const unsigned int val);
typedef unsigned long long (*fp_sync_sub_and_fetch64)(
    volatile unsigned long long *v, const unsigned int val);

unsigned int SysGetProcessorCount(void);

// TODO : Not a declaration, move down later
typedef int (*sys_pthread_getaffinity_np_t)(pthread_t, size_t, unsigned long *);
static sys_pthread_getaffinity_np_t sys_pthread_getaffinity_np;

// Atomics
typedef struct SYSAtomicFunctions_st {
  // TODO : Assuming GCC builtin atomics
  fp_sync_lock_test_and_set sync_lock_test_and_set_func;

  fp_sync_val_compare_and_swap sync_val_compare_and_swap_func;
  fp_sync_fetch_and_and sync_fetch_and_and_func;
  fp_sync_fetch_and_or sync_fetch_and_or_func;
  fp_sync_add_and_fetch sync_add_and_fetch_func;
  fp_sync_sub_and_fetch sync_sub_and_fetch_func;

  fp_sync_val_compare_and_swap64 sync_val_compare_and_swap64_func;
  fp_sync_fetch_and_and64 sync_fetch_and_and64_func;
  fp_sync_fetch_and_or64 sync_fetch_and_or64_func;
  fp_sync_add_and_fetch64 sync_add_and_fetch64_func;
  fp_sync_sub_and_fetch64 sync_sub_and_fetch64_func;
} SYSAtomicFunctions;

// Synchornization primitives and structures
typedef sem_t SysSem;
typedef pid_t SysPid;
typedef pthread_mutex_t SysCriticalSection;
typedef pthread_cond_t SysCV;
typedef struct SysBarrier_st {
  SysCriticalSection mutex;
  SysCV cv;
  unsigned int limit;
  unsigned int count;
  volatile unsigned long long signalSeq;
} SysBarrier;

// Threading
#define SYS_ONCE_INIT PTHREAD_ONCE_INIT
typedef pthread_t SysThreadId;
typedef pthread_once_t SysOnceControl;
typedef pthread_key_t SysTLSEntry;
typedef struct SysThread_st {
  void *threadHandle;
  int (*userStartFunc)(void*);
  void *userArgs;
  int returnValue;
  unsigned int refCount;
  SysSem startSemaphore;
  pthread_t pthread;
} SysThread;

// Shared memory stuff
// TODO : Odd...
static volatile unsigned long long SysShmSerial;
// struct containing information about IPC shared memory
typedef struct SysShmInfo_st {
  void *hMapFile;    // handle of shared memory
  void *pViewOfFile; // mapped/attached view of file
  size_t size;       // total size of shared memory
} SysShmInfo;

typedef enum SysShmCloseExFlags_enum {
  SYS_SHM_CLOSE_EX_INVALID = 0,
  // Decommit already-committed memory
  SYS_SHM_CLOSE_EX_DECOMMIT = 1,
  // Free (and decommit if needed) reserved or committed memory
  SYS_SHM_CLOSE_EX_RELEASE = 2,
} SysShmCloseExFlags;

typedef struct ShmKey_st {
  unsigned long pid;
  unsigned long long serial;
} ShmKey;

typedef struct ShmInfoEx_st {
  char *name;  // Globally unique name, generated from key
  ShmKey key;  // Globally unique key, used to open in other processes
  void *addr;  // Process virtual address where shared memory is mapped
  size_t size; // Size of shared memory region
  int fd;      // TODO : Added later and I'm doubtful
  uid_t uid;   // TODO : Added later and I'm doubtful
} ShmInfoEx;

// Misc
typedef struct sysstimer {
    struct timespec t;
} SysTimer;

static uint32_t SysLinuxBestSystemClock = SYS_CLOCK_UNINITIALIZED;

// Misc
char *SysSprintfMalloc(const char *format, ...);
int SysGetCurrentProcessExecPath(char **path);
int SysOnce(SysOnceControl *onceControl, void (*initRoutine)(void));
SysTLSEntry SysTlsAlloc(void (*f)(void*));
void SysTlsFree(SysTLSEntry v);
void *SysTlsGetValue(SysTLSEntry v);
int SysTlsSetValue(SysTLSEntry e, void *v);
void  SysSleep  (unsigned int msec);
void SysResetTimer(SysTimer *timer);
float SysGetTimer(SysTimer *timer);
int SysGetcwd(char *buffer, size_t bufferSize);

// Atomics
unsigned int SysInterlockedDecrement(volatile unsigned int *v);
unsigned int SysInterlockedIncrement(volatile unsigned int *v);
unsigned long long SysInterlockedCompareExchange64(volatile unsigned long long *v, unsigned long long exchange, unsigned long long compare);
unsigned int SysInterlockedExchange(volatile unsigned int *v, unsigned int exchange);
unsigned long long SysInterlockedIncrement64(volatile unsigned long long *v);
unsigned long long SysInterlockedDecrement64(volatile unsigned long long *v);
void *SysInterlockedCompareExchangePointer(void *volatile*v, void *exchange, void *compare);
unsigned int SysInterlockedCompareExchange(volatile unsigned int *v, unsigned int exchange, unsigned int compare);
unsigned int SysInterlockedOr(volatile unsigned int *v, unsigned int mask);

// Shared memory
void SysShmCloseEx(ShmInfoEx *shmInfoEx, unsigned int shmCloseExFlags, unsigned int unlink);
int SysShmCreateEx(void *addr, ShmKey *key, size_t size, ShmInfoEx **shmInfoEx);
int SysShmCreateNamedEx(void *addr, const char *key, size_t size, ShmInfoEx **shmInfoEx);

// ============================= Stuff after IPC (memory CE stuff)
size_t SysGetMinProcessorMaskSize(void);
size_t SysProcessorMaskSize(void);

// Some thread stuff here
/* Creates a thread
 * - this thread structure will be destroyed only once both
 *   - the thread has exited and
 *   - the thread has had either join or detach called
 */
int SysThreadCreate(SysThread **thread, int (*startFunc)(void *), void *userData);
int SysThreadCreateWithName(SysThread **thread, int (*startFunc)(void *), void *userData, const char *name);

/* Wait for the thread to join, and then free the thread handle */
void SysThreadJoin(SysThread *thread, int *retCode);

/* Indicate that a thread will never be joined on, and its resources may be
 * reclaimed when it exits */
void SysThreadDetach(SysThread *thread);

/* Get the current thread's ID */
SysThreadId SysGetCurrentThreadId(void);

/* Return 1 if the specified thread is current */
int SysThreadIsCurrent(SysThread *thread);

/* Return 1 if the thread ids are equal */
int SysThreadIdEqual(SysThreadId tid1, SysThreadId tid2);

/* Return TRUE if the thread has exited; FALSE otherwise.*/
int SysHasThreadExited(SysThread *thread);

/* Get the procesor affinity for the specified thread */
/* If thread is NULL, this function operates on the calling thread */
void SysGetThreadAffinity(SysThread *thread, size_t *mask);

/* Set the processor affinity for the specified thread */
/* If thread is NULL, this function operates on the calling thread */
void SysSetThreadAffinity(SysThread *thread, const size_t *mask);

/* Query a bit in an affinity mask */
int SysProcessorMaskQueryBit(const size_t *mask, size_t proc);
unsigned int SysGetProcessorCount(void);

void SysThreadYield(void);
void SysThreadYieldHeavy(void);
int SysSetThreadName(SysThread *thread, const char *name);

/* Get the procesor affinity for the specified thread */
/* If thread is NULL, this function operates on the calling thread */
void SysGetThreadAffinity(SysThread *thread, size_t *mask);

/* Set the processor affinity for the specified thread */
/* If thread is NULL, this function operates on the calling thread */
void SysSetThreadAffinity(SysThread *thread, const size_t *mask);

// Semaphore stuff
int SysSemaphoreCreate(SysSem *sem, int value);
int SysSemaphoreDestroy(SysSem *sem);
int SysSemaphoreWait(SysSem *sem, unsigned int timeoutMs);
int SysSemaphoreWaitInfinite(SysSem *sem);
int SysSemaphoreSignal(SysSem *sem);

// Lock stuff moved up
void SysInitializeCriticalSectionWithSharedFlag(SysCriticalSection *x, int shared_flag);
void SysInitializeCriticalSection(SysCriticalSection *x);
void SysInitializeCriticalSectionShared(SysCriticalSection *x);
void SysDeleteCriticalSection(SysCriticalSection *x);
void SysEnterCriticalSection(SysCriticalSection *x);
int SysTryEnterCriticalSection(SysCriticalSection *x);
void SysLeaveCriticalSection(SysCriticalSection *x);

// CV stuff
int SysCondCreateWithSharedFlag(SysCV *cv, int shared_flag);
int SysCondCreate(SysCV *cv);
int SysCondCreateShared(SysCV *cv);
int SysCondWait(SysCV *cv, SysCriticalSection *mutex, unsigned int timeoutMs);
int SysCondSignal(SysCV *cv);
int SysCondBroadcast(SysCV *cv);
int SysCondDestroy(SysCV *cv);

// Barrier stuff
int SysBarrierCreate(SysBarrier *barrier, unsigned int limit);
int SysBarrierWait(SysBarrier *barrier);
int SysBarrierDestroy(SysBarrier *barrier);

#endif // __NVBW_OS_H__
