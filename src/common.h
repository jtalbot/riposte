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
#include <cstdlib>
#include <limits>
#ifdef __GNUC__
	#define ALWAYS_INLINE __inline__ __attribute__((always_inline))
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

std::string rawToStr( unsigned char n );
std::string intToStr( int64_t n );
std::string intToHexStr( uint64_t n );
std::string charToHexStr( char c );
std::string charToOctStr( char c );
std::string doubleToStr( double n, uint64_t decimals=7, bool fixed=false );
std::string complexToStr( std::complex<double> n );

int64_t strToInt( std::string const& s);
int64_t hexStrToInt( std::string const& s);
int64_t octStrToInt( std::string const& s);
double strToDouble( std::string const& s);

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
    #if _POSIX_TIMERS > 0
        clock_gettime(CLOCK_REALTIME, &ts);
    #else
        struct timeval tv;
        gettimeofday(&tv, NULL);
        ts.tv_sec = tv.tv_sec;
        ts.tv_nsec = tv.tv_usec*1000;
    #endif
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

static inline std::string escape(std::string const& s) {
    std::string::const_iterator i = s.begin();

    std::string r;
    while(i != s.end()) 
    {
        char c = *i++;
        switch(c) {
            case '\a': r += "\\a"; break;
	        case '\b': r += "\\b"; break;
			case '\f': r += "\\f"; break;
			case '\n': r += "\\n"; break;
			case '\r': r += "\\r"; break;
			case '\t': r += "\\t"; break;
			case '\v': r += "\\v"; break;
			case '\\': r += "\\\\"; break;
			case '"': r += "\\\""; break;
			default:
                if(c >= 0x20 && c <= 0x7e) r += c;
                else r += std::string("\\") + charToOctStr(c); 
        }
    }
    return r;
}

#endif
