
#ifndef THREAD_H
#define THREAD_H

#include <vector>
#include <deque>
#include <iostream>
#include <stdint.h>
#include <pthread.h>
#include <time.h>

#include "exceptions.h"

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
	sleepTime.tv_nsec = 500000;
	nanosleep(&sleepTime, &returnTime);
}

class TaskQueue;

class TaskQueues {
public:
	std::vector<TaskQueue*> queues;

    int64_t done;
	
	TaskQueues(uint64_t queues);

	~TaskQueues() {
		fetch_and_add(&done, 1);
		while(fetch_and_add(&done, 0) != (int64_t)queues.size()) { 
			sleep(); 
		}
	}

	TaskQueue& getMainThread() const {
		return *queues[0];
	}
};

class Thread;

class TaskQueue {
public:
	struct Task {
		typedef void* (*HeaderPtr)(void* args, uint64_t a, uint64_t b, Thread& thread);
		typedef void (*FunctionPtr)(void* args, void* header, uint64_t a, uint64_t b, Thread& thread);

		HeaderPtr header;
		FunctionPtr func;
		void* args;
		uint64_t a;	// start of range [a <= x < b]
		uint64_t b;	// end
		uint64_t alignment;
		uint64_t ppt;
		int64_t* done;
		Task() : header(0), func(0), args(0), a(0), b(0), alignment(0), ppt(0), done(0) {}
		Task(HeaderPtr header, FunctionPtr func, void* args, uint64_t a, uint64_t b, uint64_t alignment, uint64_t ppt) 
			: header(header), func(func), args(args), a(a), b(b), alignment(alignment), ppt(ppt) {
			done = new int64_t(1);
		}
	};

	TaskQueues& qs;
	uint64_t index;
	pthread_t pthread;
    Thread* thread;
	
	std::deque<Task> tasks;
	Lock tasksLock;
	int64_t steals;

	TaskQueue(TaskQueues& qs, uint64_t index);
	
    static void* start(void* ptr) {
		TaskQueue* p = (TaskQueue*)ptr;
		p->loop();
		return 0;
	}

	void doall(Thread& thread, Task::HeaderPtr header, Task::FunctionPtr func, void* args, uint64_t a, uint64_t b, uint64_t alignment=1, uint64_t ppt = 1);

private:
	void loop();
	
    void run(Thread& thread, Task& t);
	uint64_t split(Task const& t);
	bool dequeue(Task& out);
	bool steal(Task& out);
};

#endif
