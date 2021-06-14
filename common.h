#include <thread>

// Rounds n up to the nearest multiple of "multiple".
// if n is already a multiple of "multiple", n is returned unchanged.
// works for arbitrary value of "multiple".
#define ROUND_UP(n, multiple)  (((n) + ((multiple)-1)) - (((n) + ((multiple)-1)) % (multiple)))

#define PROC_MASK_WORD_BITS (8 * sizeof(size_t))

#define PROC_MASK_SIZE ROUND_UP(std::thread::hardware_concurrency(), PROC_MASK_WORD_BITS) / 8

#define PROC_MASK_QUERY_BIT(mask, proc)  (mask[proc / PROC_MASK_WORD_BITS] & ((size_t)1 << (proc % PROC_MASK_WORD_BITS))) ? 1 : 0

/* Set a bit in an affinity mask */
#define PROC_MASK_SET(mask, proc)                                        \
  do {                                                                         \
    size_t _proc = (proc);                                                     \
    (mask)[_proc / PROC_MASK_WORD_BITS] |=                              \
        (size_t)1 << (_proc % PROC_MASK_WORD_BITS);                     \
  } while (0)

/* Clear a bit in an affinity mask */
#define PROC_MASK_CLEAR(mask, proc)                                      \
  do {                                                                         \
    size_t _proc = (proc);                                                     \
    (mask)[_proc / PROC_MASK_WORD_BITS] &=                              \
        ~((size_t)1 << (_proc % PROC_MASK_WORD_BITS));                  \
  } while (0)
