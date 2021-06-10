#include "nvbw_os.h"

int SysGetCurrentProcessExecPath(char **path) {
    char *buf = (char *)malloc(PATH_MAX);

    if (NULL == buf) {
        return SYS_ERROR;
    }

    if (NULL == realpath("/proc/self/exe", buf)) {
        free(buf);
        return SYS_ERROR;
    }

    *path = buf;

    return SYS_SUCCESS;
}

int SysOnce(SysOnceControl *onceControl, void (*initRoutine)(void)) {
    int ret = pthread_once(onceControl, initRoutine);
    if(ret != 0) {
        // CU_ERROR_PRINT(("pthread_once failed: %s (%d)\n", strerror(ret), ret));
        return SYS_ERROR;
    }
    return SYS_SUCCESS;
}

SysTLSEntry SysTlsAlloc(void (*f)(void*))
{
    SysTLSEntry k;
    int ret;

    ret = pthread_key_create(&k, f);

    if (ret != 0) {
        return 0;
    } else {
        return k+1;
    }
}

void SysTlsFree(SysTLSEntry v) {
    int ret = 0;
    // CU_ASSERT(v != 0);
    ret = pthread_key_delete(v-1);
    // CU_ASSERT(ret == 0);
    (void) ret;
}

void *SysTlsGetValue(SysTLSEntry v)
{
    return pthread_getspecific(v-1);
}

int SysTlsSetValue(SysTLSEntry e, void *v)
{
    if (!pthread_setspecific(e-1, v)) {
        return SYS_SUCCESS;
    }
    else {
        return SYS_ERROR;
    }
}


void  SysSleep  (unsigned int msec) {
    struct timespec t_req, t_rem;
    int ret = 0;

    t_req.tv_sec = msec / 1000;
    t_req.tv_nsec = (msec % 1000) * 1000000;

    ret = nanosleep(&t_req, &t_rem);

    // if interrupted by a non-blocked signal
    // copy remaining time to the requested time
    while(ret != 0 && errno == EINTR) {
        t_req = t_rem;
        ret = nanosleep(&t_req, &t_rem);
    }
}

void SysResetTimer(SysTimer *timer) {
    if (SysLinuxBestSystemClock == SYS_CLOCK_UNINITIALIZED) {
        return;
    }

    clock_gettime(SysLinuxBestSystemClock, &timer->t);
}

float SysGetTimer(SysTimer *timer) {
    struct timespec s;

    if (SysLinuxBestSystemClock == SYS_CLOCK_UNINITIALIZED) {
        return 0.0f;
    }

    clock_gettime(SysLinuxBestSystemClock, &s);

    return (int)(s.tv_sec - timer->t.tv_sec)*1000.0f +
           (int)(s.tv_nsec - timer->t.tv_nsec)/1000000.0f;
}

int SysGetcwd(char *buffer, size_t bufferSize)
{
    if (getcwd(buffer, bufferSize) != NULL) {
        return SYS_SUCCESS;
    }
    return SYS_ERROR;
}


// ========================
// ========================
// ========================
// ========================
// ========================
// ========================
// ========================
// ========================


// Misc
char *SysSprintfMalloc(const char *format, ...) {
  va_list args;
  size_t size;
  int ret;
  char *buf;

  va_start(args, format);
  // Get the required buffer size
  ret = vsnprintf(NULL, 0, format, args);
  va_end(args);

  if (ret < 0) {
    return NULL;
  }
  size = ret;

  // Account for '\0'
  size += 1;

  buf = (char *)malloc(size);
  if (!buf) {
    return NULL;
  }

  va_start(args, format);
  ret = vsnprintf(buf, size, format, args);
  va_end(args);

  if (ret < 0) {
    free(buf);
    return NULL;
  }

  return buf;
}

unsigned int SysInterlockedDecrement(volatile unsigned int *v) {
#if GCC_BUILTIN_ATOMICS
  return __sync_sub_and_fetch(v, 1);
#else
  // TODO : Removed register because C++17; volatile register unsigned int *uiptr = (volatile unsigned int *)v;
  volatile unsigned int *uiptr = (volatile unsigned int *)v;
  int incr = -1;
  /*
        lock: prefix is used for atomic operation
        xadd exchanges the values in its two operands, and
        then adds them together and writes the result into
        the destination operand.
        %0 : incr - register operand
        %1 : *uiptr - memory operand, destination operand
   */
  __asm__ __volatile__("lock; xaddl %0, %1;"
                       : "=r"(incr)             // output operands
                       : "m"(*uiptr), "r"(incr) // input operands
  );

  return incr - 1; // return old value minus 1
#endif
}

unsigned int SysInterlockedIncrement(volatile unsigned int *v) {
#if GCC_BUILTIN_ATOMICS
  return __sync_add_and_fetch(v, 1);
#else
  // TODO : Removed register because C++17; volatile register unsigned int *uiptr = (volatile unsigned int *)v;
  volatile unsigned int *uiptr = (volatile unsigned int *)v;
  unsigned int incr = 1;
  /*
        lock: prefix is used for atomic operation
        xadd exchanges the values in its two operands, and
        then adds them together and writes the result into
        the destination operand.
        %0 : incr - register operand
        %1 : *uiptr - memory operand, destination operand
   */
  __asm__ __volatile__("lock; xaddl %0, %1;"
                       : "=r"(incr)             // output operands
                       : "m"(*uiptr), "r"(incr) // input operands
  );

  return incr + 1; // return old value plus 1
#endif
}

void *SysInterlockedCompareExchangePointer(void *volatile*v,
                                            void *exchange,
                                            void *compare)
{
#if GCC_BUILTIN_ATOMICS
    return __sync_val_compare_and_swap(v, compare, exchange);
#else
    SYS_CMPXCHG(v, compare, exchange);
    return compare;
#endif
}

unsigned int SyssInterlockedCompareExchange(volatile unsigned int *v,
                                            unsigned int exchange,
                                            unsigned int compare)
{
#if GCC_BUILTIN_ATOMICS
    return CUOS_ATOMIC_FUNCTION_WRAPPER(sync_val_compare_and_swap, v, compare, exchange);
#else
    SYS_CMPXCHG(v, compare, exchange);
    return compare;
#endif
}

// Expected return is the value of *v before the atomic operation is applied.
unsigned int SysInterlockedOr(volatile unsigned int *v, unsigned int mask)
{
#if GCC_BUILTIN_ATOMICS
    // Use the "fetch_and_<operation>" variant of this function to achieve
    // return-previous-value behavior.
    return CUOS_ATOMIC_FUNCTION_WRAPPER(sync_fetch_and_or, v, mask);
#else
    // There is no way to retrieve the previous value from an atomic OR
    // operation on x86, so we must use a read-modify-compare-and-swap
    // loop to achieve return-previous-value behavior.
    unsigned int old;
    do {
        old = *v;
    } while (SysInterlockedCompareExchange(v, old | mask, old) != old);
    return old;
#endif
}

void SysThreadYield(void) {
    sched_yield();
}

void SysThreadYieldHeavy(void) {
    sched_yield();
}

// Shared memory
void SysShmCloseEx(ShmInfoEx *shmInfoEx, unsigned int shmCloseExFlags,
                   unsigned int unlink) {
  int status;
  int ret;
  void *retAddr;

  // TODO : Double check CU_ASSERT(shmInfoEx);

  if (shmInfoEx->addr) {
    switch (shmCloseExFlags) {
    case SYS_SHM_CLOSE_EX_DECOMMIT:
      retAddr = mmap(shmInfoEx->addr, shmInfoEx->size, PROT_NONE,
                     MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0);
      // TODO : Double check CU_ASSERT(retAddr == shmInfoEx->addr);
      break;
    case SYS_SHM_CLOSE_EX_RELEASE:
      ret = munmap(shmInfoEx->addr, shmInfoEx->size);
      // TODO : Double check CU_ASSERT(ret == 0);
      break;
    default:
      // CU_ASSERT(0); TODO : Double check
      break;
    }
  }

  // TODO : Has to be revisited
  /*
  if (shmInfoEx->fd != -1) {
      status = close(shmInfoEx->fd);
      CU_ASSERT(-1 != status); TODO : Double check

      if (unlink) {
          CU_ASSERT(shmInfoEx->name); TODO : Double check
          status = shm_unlink(shmInfoEx->name);
          CU_ASSERT(-1 != status); TODO : Double check
      }
  }
  */

  if (shmInfoEx->name) {
    free(shmInfoEx->name);
  }

  memset(shmInfoEx, 0, sizeof(*shmInfoEx));
  free(shmInfoEx);
}

int SysShmCreateEx(void *addr, ShmKey *key, size_t size,
                   ShmInfoEx **shmInfoEx) {
  char *name;
  unsigned long pid;
  unsigned long long serial;
  uid_t uid;
  int retcode = SYS_SUCCESS;

  // TODO : Double check
  // CU_ASSERT(size);
  // CU_ASSERT(shmInfoEx);

  if (key) {
    pid = key->pid;
    serial = key->serial;
  } else {
    pid = getpid(); // TODO : Later, replace this with an abstract call to
                    // remove OS dependency
    serial = SysInterlockedIncrement64(&SysShmSerial);
  }

  // This function is always successful by spec
  uid = getuid();

  name = SysSprintfMalloc(SYS_SHM_EX_TEMPLATE, (int)uid, (int)pid, serial);
  if (name == NULL) {
    // CU_ERROR_PRINT(("calloc failed in %s\n", __FUNCTION__)); TODO : Figure
    // out later
    return SYS_ERROR;
  }

  // The NAME_MAX limit on filename max length might be met here, but in
  // this case cuosShmCreateNamedEx() will fail. Let's handle the error
  // later rather than earlier to avoid platform dependent tests.
  if (SysShmCreateNamedEx(addr, name, size, shmInfoEx) != SYS_SUCCESS) {
    // CU_ERROR_PRINT(("cuosShmCreateEx failed in %s\n", __FUNCTION__)); TODO
    retcode = SYS_ERROR;
  } else {
    (*shmInfoEx)->key.pid = pid;
    (*shmInfoEx)->key.serial = serial;
  }

  free(name);

  return retcode;
}

// NEW batch

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

unsigned int SysInterlockedCompareExchange(volatile unsigned int *v,
                                            unsigned int exchange,
                                            unsigned int compare)
{
#if GCC_BUILTIN_ATOMICS
    return CUOS_ATOMIC_FUNCTION_WRAPPER(sync_val_compare_and_swap, v, compare, exchange);
#else
    SYS_CMPXCHG(v, compare, exchange);
    return compare;
#endif
}

unsigned long long
SysInterlockedCompareExchange64(volatile unsigned long long *v,
                                unsigned long long exchange,
                                unsigned long long compare) {
  // TODO : Needs more investigation
  /*
  #if GCC_BUILTIN_ATOMICS64
      return CUOS_ATOMIC_FUNCTION_WRAPPER_64(sync_val_compare_and_swap, v,
  compare, exchange); #elif defined(__x86_64__)
      // We don't have the builtin, but we match native size so regular cmpxchg
      // works
      NV_CMPXCHG(v, compare, exchange);
      return compare;
  #elif defined(__i386__)
      // Ugh, we're on i386 with no builtin to help us. Have to do it the hard
  way
      // with cmpxchg8b.
      return nv_cmpxchg8b(v, compare, exchange);
  #endif
  */
  SYS_CMPXCHG(v, compare, exchange);
  return compare;
}

unsigned int SysInterlockedExchange(volatile unsigned int *v,
                                    unsigned int exchange) {
  // TODO : Needs more investigation
  /*
  #if GCC_BUILTIN_ATOMICS
      // According to the gcc documentation this intrinsic doesn't guarantee a
      // full memory barrier. However, on x86 the xchg instruction used to
      // implement this does give us that guarantee.
      //
      // We ought to be able to use __sync_synchronize() here, but in testing on
      // ARM that doesn't actually generate any code.
  #if defined(__arm__) || defined(__aarch64__) || defined(__powerpc64__)
      cuosMemFence();
  #endif
      return CUOS_ATOMIC_FUNCTION_WRAPPER(sync_lock_test_and_set, v, exchange);
  #else
  */
  // The lock prefix is implicit on xchg. The "exchange" output uses "+r"
  // because we have to tell the compiler that the same register needs to act
  // as both an input and output. This doesn't apply to memory operands, which
  // need to be specified in both places.
  __asm__ __volatile__("xchg %0, %1;"
                       : "+r"(exchange), "=m"(*v)
                       : "m"(*v)
                       : "memory");
  return exchange;
  // #endif
}

unsigned long long SysInterlockedIncrement64(volatile unsigned long long *v) {
  /* TODO : like above
  #if GCC_BUILTIN_ATOMICS64
      return CUOS_ATOMIC_FUNCTION_WRAPPER_64(sync_add_and_fetch, v, 1);
  #elif defined(__x86_64__)
  */
  // We don't have the builtin, but we match native size so regular xadd works
  unsigned long long incr = 1;
  SYS_XADD(v, incr);
  return incr + 1;
  /*
  #elif defined(__i386__)
      // 32-bit x86 doesn't have a way to do a 64-bit xadd, but it does have a
  64-
      // bit cmpxchg. We can build the atomic inc on top of that.
      unsigned long long old, newv;
      do {
          old = *v;
          newv = old + 1;
      } while(cuosInterlockedCompareExchange64(v, newv, old) != old);
      return newv;
  #elif defined(__arm__)
      unsigned long long temp;
      cuosMemFence();
      do {
          temp = nv_ldrexd(v) + 1;
      } while (nv_strexd(v, temp));
      cuosMemFence();
      return temp;
  #endif
  */
}

unsigned long long SysInterlockedDecrement64(volatile unsigned long long *v) {
  /* TODO : Like above
  #if GCC_BUILTIN_ATOMICS64
      return CUOS_ATOMIC_FUNCTION_WRAPPER_64(sync_sub_and_fetch, v, 1);
  #elif defined(__x86_64__)
  */
  // We don't have the builtin, but we match native size so regular xadd works
  long long incr = -1;
  SYS_XADD(v, incr);
  return incr - 1;
  /*
  #elif defined(__i386__)
      // 32-bit x86 doesn't have a way to do a 64-bit xadd, but it does have a
  64-
      // bit cmpxchg. We can build the atomic inc on top of that.
      unsigned long long old, newv;
      do {
          old = *v;
          newv = old - 1;
      } while(cuosInterlockedCompareExchange64(v, newv, old) != old);
      return newv;
  #elif defined(__arm__)
      unsigned long long temp;
      cuosMemFence();
      do {
          temp = nv_ldrexd(v) - 1;
      } while (nv_strexd(v, temp));
      cuosMemFence();
      return temp;
  #endif
  */
}

int SysShmCreateNamedEx(void *addr, const char *key, size_t size,
                        ShmInfoEx **shmInfoEx) {
  ShmInfoEx *shm;
  int status;
  int prot = 0;
  int flags = 0;
  int open_flags = 0;
  struct stat s;

  // CU_ASSERT(key);
  // CU_ASSERT(size);
  // CU_ASSERT(shmInfoEx);

  if (!strlen(key)) {
    return SYS_ERROR;
  }

  shm = (ShmInfoEx *)calloc(1, sizeof(*shm));
  if (shm == NULL) {
    // CU_ERROR_PRINT(("calloc failed in %s\n", __FUNCTION__));
    goto Error;
  }

  shm->size = size;

  // Alloc space for the unique identifier string
  shm->name = strdup(key);
  if (shm->name == NULL) {
    // CU_ERROR_PRINT(("calloc failed in %s\n", __FUNCTION__));
    goto Error;
  }

  // Open a new shared memory region
  // require that the file is not already present, and symlinks are not followed
  open_flags = O_CREAT | O_EXCL | O_RDWR;

  // BUG 1311988: according to spec, O_TRUNC and O_EXCL are mutually exclusive.
  // Actually we expect to fail if the file already exists ( the EEXIST case
  // below ). and O_TRUNC doesn't work on Darwin...
  while (-1 == (shm->fd = shm_open(shm->name, open_flags, 0600))) {
    if (errno != EEXIST) {
      // CU_ERROR_PRINT(("shm_open failed in %s\n", __FUNCTION__));
      goto Error;
    } else if (shm_unlink(shm->name) == -1) {
      // if it exists, try to remove the file, error e.g. if it belongs to a
      // different user CU_ERROR_PRINT(("shm_unlink failed in %s\n",
      // __FUNCTION__));
      goto Error;
    }
  }

  status = fstat(shm->fd, &s);
  if (status != 0) {
    // CU_ERROR_PRINT(("fstat failed in %s\n", __FUNCTION__));
    goto Error;
  }
  shm->uid = s.st_uid;

  // Allocate space in the file
  status = ftruncate(shm->fd, shm->size);
  if (status == -1) {
    // CU_ERROR_PRINT(("ftruncate failed in %s\n", __FUNCTION__));
    goto Error;
  }

  prot = PROT_READ | PROT_WRITE;
  flags = MAP_SHARED;
  if (addr) {
    flags |= MAP_FIXED;
  }

  // Map the file into process address space
  shm->addr = mmap(addr, shm->size, prot, flags, shm->fd, 0);
  if (shm->addr == MAP_FAILED) {
    // CU_ERROR_PRINT(("mmap failed in %s\n", __FUNCTION__));
    goto Error;
  }

  *shmInfoEx = shm;

  return SYS_SUCCESS;

Error:
  if (shm) {
    SysShmCloseEx(shm, SYS_SHM_CLOSE_EX_RELEASE, 1);
  }
  return SYS_ERROR;
}

// ============================= Stuff after IPC (memory CE stuff)
size_t SysGetMinProcessorMaskSize(void) {
  const size_t max_mask_size = 128 * 1024; // Enough for 1M CPUs
  size_t default_size;
  unsigned long *cpu_mask;
  size_t probe_size;
  size_t pass_probe_size;
  size_t fail_probe_size;
  bool probe_error;
  int retval;

  // Original size calculation - this works the way CPU_ALLOC_SIZE works, one
  // bit per CPU rounded up to multiple of sizeof(size_t)
  default_size =
      SYS_ROUND_UP(SysGetProcessorCount(), SysProcessorMaskWordBits()) / 8;

  // The kernel mask size may be larger than what is needed to represent all of
  // the online CPUs (e.g. on systems with CPU hotplug enabled, it keeps track
  // of possible CPUs, not just present ones).

  // The suggested way of dealing with this is to use the sched_getaffinity
  // syscall to probe for a working size.  There's no access to
  // sched_getaffinity syscall or even glibc library call, so we'll probe with
  // pthread_getaffinity.  Unlike the syscall, which returns the exact size
  // needed once it passes, pthread_getaffinity just gives us a pass/fail, so
  // it's a bit more work to get a precise minimum.

  if (sys_pthread_getaffinity_np == NULL) {
    return default_size;
  }

  cpu_mask = (unsigned long *)malloc(max_mask_size);
  if (!cpu_mask) {
    // CU_ERROR_PRINT(("Unable to allocate %lu bytes memory to probe cpu mask
    // size", max_mask_size));
    return default_size;
  }

  // Check if default size is ok, if so skip the probe
  retval = sys_pthread_getaffinity_np(pthread_self(), default_size, cpu_mask);
  if (retval == 0) {
    free(cpu_mask);
    return default_size;
  }

  // Else binary search for smallest passing size (leave lower bound at 0, not
  // default_size so that we stick to power-of-2 sizes until another probe
  // fails)
  probe_error = false;
  pass_probe_size = max_mask_size;
  fail_probe_size = 0;
  probe_size = max_mask_size;
  while (pass_probe_size > (fail_probe_size + SysProcessorMaskWordBits() / 8)) {

    // Do the probe
    retval = sys_pthread_getaffinity_np(pthread_self(), probe_size, cpu_mask);

    // Adjust probe_size or terminate based on results
    if (retval == 0) {
      pass_probe_size = probe_size;
    } else if (retval == EINVAL) {
      if (probe_size == max_mask_size) {
        // Initial size not large enough
        // CU_ERROR_PRINT(("Max mask size of %lu too small for
        // pthread_getaffinity_np\n", max_mask_size));
        probe_error = true;
        break;
      }
      fail_probe_size = probe_size;
    } else {
      // Any error other than EINVAL
      // CU_ERROR_PRINT(("pthread_getaffinity_np() returned %d during mask size
      // probe\n", retval));
      probe_error = true;
      break;
    }

    probe_size = (pass_probe_size + fail_probe_size) / 2;
  }

  free(cpu_mask);

  // If for some reason we weren't able to probe the minimum mask size, fall
  // back to default calculation (and, while it shouldn't happen, return default
  // size if it ends up larger)
  return probe_error ? default_size : SYS_MAX(pass_probe_size, default_size);
}

size_t SysProcessorMaskSize(void) { return SysGetMinProcessorMaskSize(); }

// Some thread stuff here

/* Creates a thread
 * - this thread structure will be destroyed only once both
 *   - the thread has exited and
 *   - the thread has had either join or detach called
 */

static void SysThreadRelease(SysThread *thread)
{
    if (SysInterlockedDecrement(&thread->refCount) == 0) {
        memset(thread, 0, sizeof(*thread));
        free(thread);
    }
}

static void* SysPosixThreadStartFunc(void *data)
{
    SysThread *thread = (SysThread *)data;
    int result = 0;

    // wait for the calling thread to write all data to the thread handle structure
    result = SysSemaphoreWait(&thread->startSemaphore, SYS_INFINITE_TIMEOUT);
    // CU_ASSERT(result == SYS_SUCCESS);
    result = SysSemaphoreDestroy(&thread->startSemaphore);
    // CU_ASSERT(!result);
    (void) result;

    // call the user-specified threadproc
    thread->returnValue = thread->userStartFunc(thread->userArgs);

    // release the thread
    SysThreadRelease(thread);
    return NULL;
}

int SysThreadCreateWithName(
    SysThread **outThreadHandle,
    int (*startFunc)(void*),
    void *userData,
    const char *name)
{
    SysThread *thread = NULL;
    int result = 0;

    // CU_ASSERT(outThreadHandle);
    *outThreadHandle = NULL;

    // create the thread data
    thread = (SysThread *)malloc(sizeof(*thread));
    if (!thread) {
        // CU_ERROR_PRINT(("malloc failed\n"));
        return -1;
    }
    memset(thread, 0, sizeof(*thread));
    thread->userStartFunc = startFunc;
    thread->userArgs      = userData;
    thread->returnValue   = -1;
    result = SysSemaphoreCreate(&thread->startSemaphore, 0);
    if (result) {
        // CU_ERROR_PRINT(("cuosSemaphoreCreate failed\n"));
        free(thread);
        return -1;
    }
    // start the reference count at 2
    // - the thread itself has to decrement it once it has exited
    // - the thread must either be detached or joined
    thread->refCount = 2;

    // spawn the thread
    result = pthread_create(&thread->pthread, NULL, SysPosixThreadStartFunc, thread);
    if (result != 0) {
        // CU_ERROR_PRINT(("pthread_create failed with %d\n", result));
        free(thread);
        return -1;
    }

    if (name) {
        result = SysSetThreadName(thread, name);
        if (result) {
            // CU_ERROR_PRINT(("cuosSetThreadName failed\n"));
            // This should be non-fatal, ignore the error.
        }
    }

    // wake up the worker thread, now that we have written thread->pthread
    result = SysSemaphoreSignal(&thread->startSemaphore);
    if (result) {
        // CU_ERROR_PRINT(("cuosSemaphoreSignal failed\n"));
        // CU_ASSERT(0); // just leak everything and hope for the best...
        return -1;
    }

    *outThreadHandle = thread;
    return 0;
}

int SysThreadCreate(
    SysThread **outThreadHandle,
    int (*startFunc)(void*),
    void *userData)
{
    return SysThreadCreateWithName(outThreadHandle, startFunc, userData, NULL);
}

void SysThreadJoin(SysThread *thread, int *retCode)
{
    void *exitCode = NULL;
    int result = 0;

    result = pthread_join(thread->pthread, &exitCode);
    if (result != 0) {
        //CU_ERROR_PRINT(("pthread_join failed with %d\n", result));
        //CU_ASSERT(0);
    }
    // CU_ASSERT(exitCode == NULL);

    if (retCode) {
        *retCode = thread->returnValue;
    }

    SysThreadRelease(thread);
}

void SysThreadDetach(SysThread *thread)
{
    int result = 0;

    result = pthread_detach(thread->pthread);
    if (result != 0) {
        // CU_ERROR_PRINT(("pthread_detach failed with %d\n", result));
        // CU_ASSERT(0);
    }

    SysThreadRelease(thread);
}

int SysHasThreadExited(SysThread *thread) {
    return (pthread_kill(thread->pthread, 0) == ESRCH);
}

int SysThreadIsCurrent(SysThread *thread) {
    return pthread_equal(pthread_self(), thread->pthread);
}

SysThreadId SysGetCurrentThreadId(void)
{
    return pthread_self();
}

int SysThreadIdEqual(SysThreadId tid1, SysThreadId tid2)
{
    return pthread_equal(tid1, tid2);
}

// TODO : Dummy func
int SysSetThreadName(SysThread *thread, const char *name)
{
    int result = 0;
    if (result == 0) {
        return SYS_SUCCESS;
    }
    else {
        return SYS_ERROR;
    }
}

// TODO : Dummy function
void SysGetThreadAffinity(SysThread *thread, size_t *mask)
{
    (void)thread;
    (void)mask;
    /*
    if (_pthread_getaffinity_np != NULL) {
        pthread_t pthread = thread ? thread->pthread : pthread_self();
        int status = _pthread_getaffinity_np(pthread, cuosProcessorMaskSize(), (unsigned long *)mask);
        if (status != 0) {
            CU_ERROR_PRINT(("pthread_getaffinity_np() returned %d", status));
            // claim to be running on core zero upon failure
            *mask = 1;
        }
    }
    else {
        // CU_ERROR_PRINT(("pthread_getaffinity_np() not found"));
        *mask = 1;
    }
    */
}

// TODO : Dummy function
void SysSetThreadAffinity(SysThread *thread, const size_t *mask)
{
    (void)thread;
    (void)mask;
    /*
    if (_pthread_setaffinity_np != NULL) {
        pthread_t pthread = thread ? thread->pthread : pthread_self();
        int status = _pthread_setaffinity_np(pthread, cuosProcessorMaskSize(), (const unsigned long *)mask);
        if (status != 0) {
            // CU_ERROR_PRINT(("pthread_setaffinity_np() returned %d", status));
        }
    }
    else {
        // CU_ERROR_PRINT(("pthread_setaffinity_np() not found"));
    }
    */
}

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
#define cuosProcessorMaskClear(mask, proc)                                     \
  do {                                                                         \
    size_t _proc = (proc);                                                     \
    (mask)[_proc / SysProcessorMaskWordBits()] &=                              \
        ~((size_t)1 << (_proc % SysProcessorMaskWordBits()));                  \
  } while (0)

/* Query a bit in an affinity mask */
int SysProcessorMaskQueryBit(const size_t *mask, size_t proc) {
  return (mask[proc / SysProcessorMaskWordBits()] &
          ((size_t)1 << (proc % SysProcessorMaskWordBits())))
             ? 1
             : 0;
}

unsigned int SysGetProcessorCount(void) {
  return sysconf(_SC_NPROCESSORS_ONLN);
}

/* Clear a bit in an affinity mask */
#define SysProcessorMaskClear(mask, proc)                                      \
  do {                                                                         \
    size_t _proc = (proc);                                                     \
    (mask)[_proc / SysProcessorMaskWordBits()] &=                              \
        ~((size_t)1 << (_proc % SysProcessorMaskWordBits()));                  \
  } while (0)

// Semaphore stuff

int SysSemaphoreCreate(SysSem *sem, int value) {
  int result;
  result = sem_init(sem, 0, value);
  if (result != 0) {
    // CU_ERROR_PRINT(("sem_init failed with %d\n", result));
    return -1;
  }
  return 0;
}

int SysSemaphoreDestroy(SysSem *sem) {
  int result;
  result = sem_destroy(sem);
  if (result != 0) {
    // CU_ERROR_PRINT(("sem_destroy failed with %d\n", result));
    return -1;
  }
  return 0;
}

int SysSemaphoreWait(SysSem *sem, unsigned int timeoutMs) {
  int result;
  struct timespec tp;
  struct timeval tv;

  // Check whether we should block indefinitely.
  if (timeoutMs == SYS_INFINITE_TIMEOUT) {
    return SysSemaphoreWaitInfinite(sem);
  }
  // Or if we should return immediately with sem_trywait.
  else if (timeoutMs == 0) {
    result = sem_trywait(sem);
    if (result == 0) {
      return SYS_SUCCESS;
    } else if (result == -1 && errno == EAGAIN) {
      return SYS_TIMEOUT;
    } else {
      // CU_ERROR_PRINT(("sem_trywait failed with %d\n", errno));
      return SYS_ERROR;
    }
  }

  // We need to time out after a nonzero time, so set up a timer and use
  // sem_timedwait.
  result = gettimeofday(&tv, NULL);
  if (result == -1) {
    // CU_ERROR_PRINT(("gettimeofday failed with %d\n", errno));
    return SYS_ERROR;
  }
  tp.tv_sec = tv.tv_sec + (timeoutMs / MS_PER_S);
  tp.tv_nsec = tv.tv_usec * NS_PER_US + ((timeoutMs % MS_PER_S) * NS_PER_MS);
  tp.tv_sec += tp.tv_nsec / NS_PER_S;
  tp.tv_nsec %= NS_PER_S;

  while (1) {
    result = sem_timedwait(sem, &tp);
    if (result == 0) {
      return SYS_SUCCESS;
    } else if (result == -1 && errno == ETIMEDOUT) {
      return SYS_TIMEOUT;
    } else if (result == -1 && errno == EINTR) {
      // CU_DEBUG_PRINT(("sem_timedwait returned EINTR, interrupted by a
      // signal\n"));
    } else {
      // CU_ERROR_PRINT(("sem_timedwait failed with %d\n", errno));
      return SYS_ERROR;
    }
  }
}

int SysSemaphoreWaitInfinite(SysSem *sem) {
  int result;

  while (1) {
    result = sem_wait(sem);
    if (result == 0) {
      return SYS_SUCCESS;
    } else if (result == -1 && errno == EINTR) {
      // CU_DEBUG_PRINT(("sem_wait returned EINTR, interrupted by a signal\n"));
    } else {
      // CU_ERROR_PRINT(("sem_wait failed with %d\n", errno));
      return SYS_ERROR;
    }
  }
}

int SysSemaphoreSignal(SysSem *sem) {
  int result;
  result = sem_post(sem);
  if (result != 0) {
    // CU_ERROR_PRINT(("sem_post failed with %d\n", result));
    return -1;
  }
  return 0;
}

// Lock stuff moved up

void SysInitializeCriticalSectionWithSharedFlag(SysCriticalSection *x,
                                                int shared_flag) {
  int error;
  pthread_mutexattr_t attr;

  error = pthread_mutexattr_init(&attr);
  if (error) {
    // CU_ERROR_PRINT(("cuosInitializeCriticalSection: %s\n", strerror(error)));
    return;
  }
  error = pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_RECURSIVE);
  if (error) {
    // CU_ERROR_PRINT(("cuosInitializeCriticalSection: %s\n", strerror(error)));
    return;
  }
  error = pthread_mutexattr_setpshared(&attr, shared_flag);
  if (error) {
    // CU_ERROR_PRINT(("cuosInitializeCriticalSection: %s\n", strerror(error)));
    return;
  }

  // Priority inheritence(PI) is required on tegra platforms and is found to
  // have performance impacts on desktop and power PC. Though the following
  // block can be removed for tegra platforms also as most tegra platforms where
  // PI is absolutely necessary have PI enabled at kernel level (QNX has PI
  // enabled by default and Linux uses RT kernel where PI is enabled), this is
  // still being kept as L4T has not moved to RT kernel and has performace and
  // predictability requirements. Refer to Bug 2111672 for more information.
  // However, this should NOT be enabled on ARM64+dGPU systems like Balboa or
  // SHH.
  // __TEMP_WAR__ Bug 200223061 Android Headers incorrectly declare
  // _POSIX_THREAD_PRIO_INHERIT as supported but it is not.
  // TODO: Use proper macro Bug 2111672
  /* TODO : I'm enabling priority inheritence by default
  #if cuosPlatformIncludesMrm() && !(defined(__ANDROID__))
  #if defined(_POSIX_THREAD_PRIO_INHERIT) && _POSIX_THREAD_PRIO_INHERIT != -1
      error = pthread_mutexattr_setprotocol(&attr, PTHREAD_PRIO_INHERIT);
      if (error) {
          CU_ERROR_PRINT(("cuosInitializeCriticalSection: %s\n",
  strerror(error))); return;
      }
  #endif
  #endif
  */
  error = pthread_mutexattr_setprotocol(&attr, PTHREAD_PRIO_INHERIT);
  if (error) {
    // CU_ERROR_PRINT(("cuosInitializeCriticalSection: %s\n", strerror(error)));
    return;
  }

  error = pthread_mutex_init(x, &attr);
  if (error) {
    // CU_ERROR_PRINT(("cuosInitializeCriticalSection: %s\n", strerror(error)));
    return;
  }
  error = pthread_mutexattr_destroy(&attr);
  if (error) {
    // CU_ERROR_PRINT(("cuosInitializeCriticalSection: %s\n", strerror(error)));
    return;
  }
}

void SysInitializeCriticalSection(SysCriticalSection *x) {
  SysInitializeCriticalSectionWithSharedFlag(x, PTHREAD_PROCESS_PRIVATE);
}

void SysInitializeCriticalSectionShared(SysCriticalSection *x) {
  SysInitializeCriticalSectionWithSharedFlag(x, PTHREAD_PROCESS_SHARED);
}

void SysDeleteCriticalSection(SysCriticalSection *x) {
  int error;
  error = pthread_mutex_destroy(x);
  if (error) {
    // CU_ERROR_PRINT(("cuosDeleteCriticalSection: %s\n", strerror(error)));
  }
}

void SysEnterCriticalSection(SysCriticalSection *x) {
  int error;
  error = pthread_mutex_lock(x);
  if (error) {
    // CU_ERROR_PRINT(("cuosEnterCriticalSection: %s\n", strerror(error)));
  }
}

int SysTryEnterCriticalSection(SysCriticalSection *x) {
  int error;
  error = pthread_mutex_trylock(x);
  switch (error) {
  case 0:
    return SYS_SUCCESS;
  case EBUSY:
    return SYS_TIMEOUT;
  default:
    // CU_ERROR_PRINT(("cuosTryEnterCriticalSection: %s\n", strerror(error)));
    return SYS_ERROR;
  }
}

void SysLeaveCriticalSection(SysCriticalSection *x) {
  int error;
  error = pthread_mutex_unlock(x);
  if (error) {
    // CU_ERROR_PRINT(("cuosLeaveCriticalSection: %s\n", strerror(error)));
  }
}

// CV stuff
int SysCondCreateWithSharedFlag(SysCV *cv, int shared_flag) {
  int ret;
  pthread_condattr_t cond_attr;
  ret = pthread_condattr_init(&cond_attr);
  if (ret != 0) {
    // CU_ERROR_PRINT(("pthread_condattr_init failed: %s (%d)\n", strerror(ret),
    // ret));
    return SYS_ERROR;
  }
  ret = pthread_condattr_setpshared(&cond_attr, shared_flag);
  if (ret != 0) {
    // CU_ERROR_PRINT(("pthread_condattr_setpshared failed: %s (%d)\n",
    // strerror(ret), ret));
    return SYS_ERROR;
  }
  ret = pthread_cond_init(cv, &cond_attr);
  if (ret != 0) {
    // CU_ERROR_PRINT(("pthread_cond_init failed: %s (%d)\n", strerror(ret),
    // ret));
    return SYS_ERROR;
  }
  return SYS_SUCCESS;
}

int SysCondCreate(SysCV *cv) {
  return SysCondCreateWithSharedFlag(cv, PTHREAD_PROCESS_PRIVATE);
}

int SysCondCreateShared(SysCV *cv) {
  return SysCondCreateWithSharedFlag(cv, PTHREAD_PROCESS_SHARED);
}

int SysCondWait(SysCV *cv, SysCriticalSection *mutex, unsigned int timeoutMs) {
  int ret;

  if (timeoutMs == SYS_INFINITE_TIMEOUT) {
    ret = pthread_cond_wait(cv, mutex);
  } else {
    // Compute the absolute time for the wait function
    struct timespec tp;
    struct timeval tv;

    if (timeoutMs == 0) {
      // Don't call gettimeofday if we don't need to
      memset(&tp, 0, sizeof(tp));

      /* TODO : Not concerend rn
      #if defined(__APPLE__) || defined(__ANDROID__)
                  // Work around bug 804802. It appears that Apple's
      implementation of
                  // pthread_cond_timedwait doesn't release and re-acquire the
      mutex
                  // if the timeout has expired, leading to starvation. To WAR
      this
                  // we'll do it ourselves when we know that we're in a
      busy-wait
                  // situation (the timeout is 0).
                  // WAR bug 931076 as well. Android's pthread_cond_wait returns
                  // ETIMEDOUT if it sees that timeout has expired, without
      releasing
                  // and re-aquiring the mutex.
                  cuosLeaveCriticalSection(mutex);
                  cuosEnterCriticalSection(mutex);
      #endif // __APPLE__ || __ANDROID__
      */
    } else {
      ret = gettimeofday(&tv, NULL);
      if (ret != 0) {
        // CU_ERROR_PRINT(("gettimeofday failed: %s (%d)\n", strerror(errno),
        // errno));
        return SYS_ERROR;
      }

      tp.tv_sec = tv.tv_sec + timeoutMs / 1000;
      tp.tv_nsec = tv.tv_usec * 1000 + (timeoutMs % 1000) * 1000000;
      tp.tv_sec += tp.tv_nsec / 1000000000;
      tp.tv_nsec %= 1000000000;
    }
    ret = pthread_cond_timedwait(cv, mutex, &tp);
    if (ret == ETIMEDOUT) {
      return SYS_TIMEOUT;
    }
  }

  if (ret != 0) {
    // CU_ERROR_PRINT(("pthread_cond_[timed]wait failed: %s (%d)\n",
    // strerror(ret), ret));
    return SYS_ERROR;
  }

  return SYS_SUCCESS;
}

int SysCondSignal(SysCV *cv) {
  int ret = pthread_cond_signal(cv);
  if (ret != 0) {
    // CU_ERROR_PRINT(("pthread_cond_signal failed: %s (%d)\n", strerror(ret),
    // ret));
    return SYS_ERROR;
  }
  return SYS_SUCCESS;
}

int SysCondBroadcast(SysCV *cv) {
  int ret = pthread_cond_broadcast(cv);
  if (ret != 0) {
    // CU_ERROR_PRINT(("pthread_cond_broadcast failed: %s (%d)\n",
    // strerror(ret), ret));
    return SYS_ERROR;
  }
  return SYS_SUCCESS;
}

int SysCondDestroy(SysCV *cv) {
  int ret;
  ret = pthread_cond_destroy(cv);
  if (ret != 0) {
    // CU_ERROR_PRINT(("pthread_cond_destroy failed: %s (%d)\n", strerror(ret),
    // ret));
    return SYS_ERROR;
  }
  return SYS_SUCCESS;
}

// Barrier stuff
int SysBarrierCreate(SysBarrier *barrier, unsigned int limit) {
  int ret;

  if (limit == 0) {
    return SYS_ERROR;
  }

  memset(barrier, 0, sizeof(*barrier));
  barrier->limit = limit;

  SysInitializeCriticalSection(&barrier->mutex);

  ret = SysCondCreate(&barrier->cv);
  if (ret != SYS_SUCCESS) {
    // CU_ERROR_PRINT(("Failed to init condition variable\n"));
    return ret;
  }

  return SYS_SUCCESS;
}

int SysBarrierWait(SysBarrier *barrier) {
  unsigned long long expectedSeq;
  int ret;

  // CU_ASSERT(barrier->limit > 0);

  SysEnterCriticalSection(&barrier->mutex);

  // CU_ASSERT(barrier->count < barrier->limit);

  ++barrier->count;
  if (barrier->count == barrier->limit) {
    // Inform the waiting threads that a signal has actually happened
    ++barrier->signalSeq;
    // CU_ASSERT(barrier->signalSeq != 0); // Yeah right

    // Wake 'em up
    ret = SysCondBroadcast(&barrier->cv);
    if (ret != SYS_SUCCESS) {
      // CU_ERROR_PRINT(("cuosCondBroadcast failed\n"));
      // CU_ASSERT(0);
    }

    barrier->count = 0;
  } else {
    // The condition variable wait may wake up spuriously, so remember the
    // signal sequence number before we began waiting. That way we can check
    // it every wakeup to ensure that we don't proceed unless there's
    // actually been a signal since we began the wait.
    expectedSeq = barrier->signalSeq + 1;
    do {
      ret = SysCondWait(&barrier->cv, &barrier->mutex, SYS_INFINITE_TIMEOUT);
      if (ret != SYS_SUCCESS) {
        // CU_ASSERT(ret != CUOS_TIMEOUT); // Just in case
        // CU_ERROR_PRINT(("cuosCondWait failed\n"));
      }
    } while (ret == SYS_SUCCESS && expectedSeq > barrier->signalSeq);
  }

  SysLeaveCriticalSection(&barrier->mutex);

  return ret;
}

int SysBarrierDestroy(SysBarrier *barrier) {
  int tempRet, finalRet = SYS_SUCCESS;

  tempRet = SysCondDestroy(&barrier->cv);
  if (tempRet != SYS_SUCCESS) {
    // CU_ERROR_PRINT(("Failed to destroy condition variable\n"));
    if (finalRet == SYS_SUCCESS) {
      finalRet = tempRet;
    }
  }

  SysDeleteCriticalSection(&barrier->mutex);

  memset(barrier, 0, sizeof(*barrier));

  return finalRet;
}
