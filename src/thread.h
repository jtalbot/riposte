
#ifndef THREAD_H
#define THREAD_H

#include <pthread.h>
#include <time.h>

static inline int fetch_and_add(int64_t * variable, int64_t value) {
	asm volatile( 
			"lock; xaddq %%rax, %2\n"
			:"=a" (value)                   //Output
			: "a" (value), "m" (*variable)  //Input
			:"memory" );
	return value;
}

static inline int atomic_xchg(int64_t* v, int64_t n) {
	asm volatile(
			"lock xchgq    %1, (%2)    \n"
			: "=r" (n) : "0" (n), "r" (v));
	return n;
}

class Lock
{
    pthread_mutex_t m;
public:
    Lock() {
        pthread_mutex_init(&m, NULL);
    }

    ~Lock() {
        pthread_mutex_destroy(&m);
    }

    void acquire() {
        pthread_mutex_lock(&m);
    }

    void release() {
        pthread_mutex_unlock(&m);
    }
};

static inline void sleep() {
	struct timespec sleepTime;
	struct timespec returnTime;
	sleepTime.tv_sec = 0;
	sleepTime.tv_nsec = 5000000;
	nanosleep(&sleepTime, &returnTime);
}

#endif
