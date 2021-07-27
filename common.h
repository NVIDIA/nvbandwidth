#ifndef COMMON_H
#define COMMON_H

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime.h>
#include <float.h>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <string>
#include <thread>
#include <vector>

// Default constants
const unsigned long long defaultLoopCount = 16;
const unsigned long long defaultBufferSize = 1024 * 1024 * 64; // 64MB
const unsigned int defaultAverageLoopCount = 3;
const unsigned int _2MB = 1024 * 1024 * 2;
extern unsigned int averageLoopCount;
extern bool disableP2P;

// Verbosity
extern bool VERBOSE;

// Rounds n up to the nearest multiple of "multiple".
// if n is already a multiple of "multiple", n is returned unchanged.
// works for arbitrary value of "multiple".
#define ROUND_UP(n, multiple)															\
  	(((n) + ((multiple)-1)) - (((n) + ((multiple)-1)) % (multiple)))

#define PROC_MASK_WORD_BITS (8 * sizeof(size_t))

#define PROC_MASK_SIZE                                                         			\
  	ROUND_UP(std::thread::hardware_concurrency(), PROC_MASK_WORD_BITS) / 8

#define PROC_MASK_QUERY_BIT(mask, proc)                                        			\
  	(mask[proc / PROC_MASK_WORD_BITS] &                                          		\
   		((size_t)1 << (proc % PROC_MASK_WORD_BITS)))                                	\
      		? 1                                                                      	\
      		: 0

/* Set a bit in an affinity mask */
#define PROC_MASK_SET(mask, proc)                                              			\
  	do {                                                                         		\
    	size_t _proc = (proc);                                                     		\
    	(mask)[_proc / PROC_MASK_WORD_BITS] |= (size_t)1                           		\
                                        << (_proc % PROC_MASK_WORD_BITS);   			\
  	} while (0)

/* Clear a bit in an affinity mask */
#define PROC_MASK_CLEAR(mask, proc)                                            			\
  	do {                                                                         		\
    	size_t _proc = (proc);                                                     		\
    	(mask)[_proc / PROC_MASK_WORD_BITS] &=                                     		\
        	~((size_t)1 << (_proc % PROC_MASK_WORD_BITS));                         		\
  	} while (0)

inline size_t getFirstEnabledCPU() {
  	size_t firstEnabledCPU = 0;
  	size_t *procMask = (size_t *)calloc(1, PROC_MASK_SIZE);
  	for (size_t i = 0; i < PROC_MASK_SIZE * 8; ++i) {
    	if (PROC_MASK_QUERY_BIT(procMask, i)) {
      		firstEnabledCPU = i;
      	break;
    	}
  	}
  	free(procMask);
  	return firstEnabledCPU;
}

// Calculation and display of performance statistics
// Basic online running statistics calculator, modeled after a less templated
// version of boost::accumulators.
class PerformanceStatistic {
  	double m_smallest, m_largest;
  	double m_total, m_mean, m_var;
  	size_t m_cnt;

public:
  	PerformanceStatistic(): m_smallest(DBL_MAX), m_largest(-DBL_MAX), m_total(0.0), m_mean(0.0), m_var(0.0), m_cnt(0) {}
  	
	void operator()(const double &sample) { recordSample(sample); }
  	
	void recordSample(const double &sample) {
    	m_cnt++;
    	if (m_smallest > sample) {
      		m_smallest = sample;
    	}
    	if (m_largest < sample) {
      		m_largest = sample;
    	}
    	// Online variance calculation algorithm can be found here:
    	// https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online_algorithm
    	// Donald E. Knuth (1998). The Art of Computer Programming, volume 2:
    	// Seminumerical Algorithms, 3rd edn., p. 232. Boston: Addison-Wesley.
    	m_total += sample;
    	double delta = sample - m_mean;
    	m_mean += delta / m_cnt;
    	double delta2 = sample - m_mean;
    	m_var += delta * delta2;
  	}

  	// Aggregate the statistics from samples recorded in 'other' into this
  	// statistic (useful for combining multiple statistics from mulitple threads)
  	void aggregate(const PerformanceStatistic &other) {
    	if (m_smallest > other.m_smallest) {
      		m_smallest = other.m_smallest;
    	}
    	if (m_largest > other.m_largest) {
      		m_largest = other.m_largest;
    	}
    	m_total += other.m_total;
    	// This just calculates the ratios of the combined counts to keep the
    	// averages stable: avg1 * (cnt1 / (cnt1 + cnt2)) + avg2 * (cnt2 / (cnt1 +
    	// cnt2))
    	m_mean = m_mean * (((double)m_cnt) / (m_cnt + other.m_cnt)) +
             	other.m_mean * (((double)other.m_cnt) / (m_cnt + other.m_cnt));

    	// Since the variance is just a sum of squares and does not depend on
    	// the sample count, no need for extra math here
    	m_var += other.m_var;
    	m_cnt += other.m_cnt;
  	}

  	void reset(void) { *this = PerformanceStatistic(); }
  	
	double sum(void) const { return m_total; }
  	
	size_t count(void) const { return m_cnt; }
  	
	double average(void) const { return m_mean; }
  	
	double variance(void) const { return (m_cnt > 1 ? m_var / (m_cnt - 1) : 0); }
  	
	double stddev(void) const {
    	return (variance() > 0.0 ? std::sqrt(variance()) : 0.0);
  	}
  	
	double largest(void) const { return m_largest; }
	
	double smallest(void) const { return m_smallest; }
};

typedef PerformanceStatistic cudaStat;

#define STAT_MEAN(s) (s).average()
#define STAT_ERROR(s) (s).stddev()
#define STAT_MAX(s) (s).largest()
#define STAT_MIN(s) (s).smallest()

static std::ostream &operator<<(std::ostream &o, const cudaStat &s) {
	return o << STAT_MEAN(s) << "(+/- " << STAT_ERROR(s) << ')';
}

template <class T> struct PeerValueMatrix {
  	T *m_matrix;
  	int m_rows, m_columns;
  	PeerValueMatrix(int rows, int columns): m_matrix(new T[rows * columns]()), m_rows(rows), m_columns(columns) {}
  	PeerValueMatrix(int rows) : m_matrix(new T[rows * rows]()), m_rows(rows), m_columns(rows) {}

  	~PeerValueMatrix() { delete[] m_matrix; }
  	T &value(int src, int dst) {
    	assert(src >= 0 && src < m_rows);
    	assert(dst >= 0 && dst < m_columns);
    	return m_matrix[src * m_columns + dst];
  	}
  	const T &value(int src, int dst) const {
    	assert(src >= 0 && src < m_rows);
    	assert(dst >= 0 && dst < m_columns);
    	return m_matrix[src * m_columns + dst];
  	}
};

template <class T>
std::ostream &operator<<(std::ostream &o, const PeerValueMatrix<T> &matrix) {
  	o << ' ';
  	for (int currentDevice = 0; currentDevice < matrix.m_columns; currentDevice++) {
    	o << std::setw(10) << currentDevice;
  	}
  	o << std::endl;
  	for (int currentDevice = 0; currentDevice < matrix.m_rows; currentDevice++) {
    	o << currentDevice;
    	for (int peer = 0; peer < matrix.m_columns; peer++)
      		o << std::setw(10) << matrix.value(currentDevice, peer);
    	o << std::endl;
  }
  return o;
}

template <class T>
std::ostream &printIndexVector(std::ostream &o, std::vector<T> &v, int field_width = 10) {
  	for (size_t i = 0; i < v.size(); i++)
    	o << std::setw(field_width) << i;
  	o << std::endl;
  	for (size_t i = 0; i < v.size(); i++)
    	o << std::setw(field_width) << v[i];
  	o << std::endl;
  	return o;
}

/* Calculates the least-squares approximation of the linear data */
template <class T>
void calculateLeastSquares(const std::vector<T> &x, const std::vector<T> &y, double &slope, double &intercept) {
  	double x_avg = 0.0, y_avg = 0.0;
  	double rise = 0.0, run = 0.0;
  	ASSERT_GT(x.size(), 1U);
  	ASSERT_GT(y.size(), 1U);
  	ASSERT_EQ(x.size(), y.size());

  	for (size_t i = 0; i < x.size(); i++) {
    	x_avg += (double)x[i];
    	y_avg += (double)y[i];
  	}

  	x_avg /= x.size();
  	y_avg /= y.size();

  	for (size_t i = 0; i < x.size(); i++) {
    	rise += (x[i] - x_avg) * (y[i] - y_avg);
    	run += (x[i] - x_avg) * (x[i] - x_avg);
  	}

  	slope = rise / run;
  	intercept = y_avg - slope * x_avg;
}

// Error handling
inline void CU_ASSERT(CUresult cuResult, const char *msg = nullptr) {
  	if (cuResult != CUDA_SUCCESS) {
    	const char *errDescStr, *errNameStr;
    	cuGetErrorString(cuResult, &errDescStr);
    	cuGetErrorName(cuResult, &errNameStr);
    	std::cout << "[" << errNameStr << "] " << errDescStr;
    	if (msg != nullptr) std::cout << ":\n\t" << msg;
    	std::cout << std::endl;
    	std::exit(1);
  }
}

#endif
