#ifndef __RIPOSTE_COMMON
#define __RIPOSTE_COMMON

#include <stdint.h>
#include <string>
#include <string.h>
#include <sstream>
#include <complex>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <sys/time.h>
#include <stdio.h>

//#define GC_DEBUG
#define GC_THREADS
#include <gc/gc_cpp.h>
#include <gc/gc_allocator.h>

#ifdef __GNUC__
	#define ALWAYS_INLINE __attribute__((always_inline))
#else
	#define ALWAYS_INLINE
#endif

// Arg. Clang claims to be compatible with gcc 4.2, but it's not in this case.
#if defined __GNUC__ && (__GNUC__ < 4 || (__GNUC__ == 4 && __GNUC_MINOR__ < 3)) && !defined(__clang__)
#define SPECIALIZED_STATIC static
#else
#define SPECIALIZED_STATIC
#endif

#define USE_THREADED_INTERPRETER
#define TIMING

static inline std::string rawToStr( unsigned char n )
{
	std::ostringstream result;
	result << std::hex << std::setfill('0') << std::setw(2) << n;
	return result.str();
}

static inline std::string intToStr( int64_t n )
{
    std::ostringstream result;
    result << n;
    return result.str();
}

static inline std::string intToHexStr( uint64_t n )
{
    std::ostringstream result;
    result << "0x" << std::hex << n;
    return result.str();
}

static inline std::string doubleToStr( double n )
{
    std::ostringstream result;
    if(std::isnan(n)) return std::string("NaN");
    else if(n == std::numeric_limits<double>::infinity()) return std::string("Inf");
    else if(n == -std::numeric_limits<double>::infinity()) return std::string("-Inf");
    else if(n == 0) return std::string("0");    // handle the -0 case
    result << std::setprecision(7) << n;
    return result.str();
}

static inline std::string complexToStr( std::complex<double> n )
{
    std::ostringstream result;
    result << n.real();
    result.setf(std::ios::showpos);
    result << n.imag() << "i";
    return result.str();
}

static inline int64_t strToInt( std::string const& s) {
	int64_t r;
	std::istringstream(s) >> std::dec >> r;
	return r;
}

static inline int64_t strToHexInt( std::string const& s) {
	int64_t r;
	std::istringstream(s) >> std::hex >> r;
	return r;
}

static inline double strToDouble( std::string const& s) {
	double r;
	std::istringstream(s) >> r;
	return r;
}

static inline double time_diff (
    timespec const& end, timespec const& begin)
{
#ifdef TIMING
    double result;

    result = end.tv_sec - begin.tv_sec;
    result += (end.tv_nsec - begin.tv_nsec) / (double)1000000000;

    return result;
#else
    return 0;
#endif
}

static inline void get_time (timespec& ts)
{
#ifdef TIMING
    volatile long noskip;
    #if _POSIX_TIMERS > 0
        clock_gettime(CLOCK_REALTIME, &ts);
    #else
        struct timeval tv;
        gettimeofday(&tv, NULL);
        ts.tv_sec = tv.tv_sec;
        ts.tv_nsec = tv.tv_usec*1000;
    #endif
    noskip = ts.tv_nsec;
#endif
}

static inline timespec get_time()
{
#ifdef TIMING
    timespec t;
    get_time(t);
    return t;
#endif
}

static inline double time_elapsed(timespec const& begin)
{
#ifdef TIMING
    timespec now;
    get_time(now);
    return time_diff(now, begin);
#else
    return 0;
#endif
}

static inline void print_time (char const* prompt, timespec const& begin, timespec const& end)
{
#ifdef TIMING
    printf("%s : %.3f\n", prompt, time_diff(end, begin));
#endif
}

static inline void print_time_elapsed (char const* prompt, timespec const& begin)
{
#ifdef TIMING
    printf("%s : %.3f\n", prompt, time_elapsed(begin));
#endif
}

static inline uint64_t nextPow2(uint64_t i) {
       i--;
       i |= i >> 1;
       i |= i >> 2;
       i |= i >> 4;
       i |= i >> 8;
       i |= i >> 16;
       i |= i >> 32;
       i++;
       return i;
}

static inline uint32_t numSetBits(uint32_t i) {
	i = i - ((i >> 1) & 0x55555555);
	i = (i & 0x33333333) + ((i >> 2) & 0x33333333);
	return (((i + (i >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24;
}

#endif
