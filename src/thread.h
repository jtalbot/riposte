
#ifndef THREAD_H
#define THREAD_H

#include <deque>
#include <pthread.h>
#include <time.h>

static inline int fetch_and_add( int * variable, int value ){
	asm volatile( 
			"lock; xaddl %%eax, %2;"
			:"=a" (value)                   //Output
			: "a" (value), "m" (*variable)  //Input
			:"memory" );
	return value;
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
/*
#ifdef _DARWIN_
#include <mach/mach.h>
#include <mach/semaphore.h>
#else
#include <semaphore.h>
#endif

class Semaphore
{
private:
#ifdef _DARWIN_
    semaphore_t sem;
#else
    sem_t sem;
#endif
public:
    Semaphore(int initialValue=0)
    {
        #ifdef _DARWIN_
        semaphore_create(mach_task_self(), &sem, SYNC_POLICY_FIFO, initialValue);
        #else
        sem_init(&sem, 0, initialValue);
        #endif
    }

    void post()
    {
        #ifdef _DARWIN_
        semaphore_signal(sem);
        #else
        sem_post(&sem);
        #endif
    }

    void wait()
    {
        #ifdef _DARWIN_
        semaphore_wait(sem);
        #else
        sem_wait(&sem);
        #endif
    }
};

class Thread;
class SharedState;

typedef void (*TaskFunctionPtr)(void* args, uint64_t a, uint64_t b, Thread* thread);

class Pool {
private:
public:
	std::vector<Thread*> workers;
	
	Pool(SharedState& shared, uint64_t workers);
	
	Thread* getMainThread() {
		return workers[0];
	}
};

class Thread {
	struct Task {
		TaskFunctionPtr func;
		void* args;
		uint64_t a;	// start of range [a <= x < b]
		uint64_t b;	// end
		uint64_t alignment;
		uint64_t ppt;
		int* done;
		Task() : func(0), args(0), done(0) {}
		Task(TaskFunctionPtr func, void* args, uint64_t a, uint64_t b, uint64_t alignment, uint64_t ppt) 
			: func(func), args(args), a(a), b(b), alignment(alignment), ppt(ppt) {
			done = new int(1);
		}
	};

	Pool* pool;
	uint64_t index;

	std::deque<Task> tasks;
	Lock tasksLock;

public:
	pthread_t thread;

	Thread(Pool* pool, uint64_t index) : pool(pool), index(index) {}

	
//		doall is called from thread x,
//		thread x immediately starts executing without pushing onto queue
//			if ppt is exceeded pushes half onto queue
//		...thread x will always execute the first element
//		thread x owns the task and stalls at the end of the job waiting for any stolen tasks to be finished and returned.
//		could alternatively attempt to steal other people's tasks while waiting for it's own task to finish.
//		thread returns when task is done. Either returns into own thread (in case of recursive call) or terminates thread if it was a child thread.
//		child threads are initialized with a pending task controlled by the parent. The pending task causes the thread to steal work. To end all child threads, the parent completes the pending task which allows the child threads to advance to completion.
//		Only interface is doall
	

	static void* start(void* ptr) {
		Thread* p = (Thread*)ptr;
		p->loop();
		return 0;
	}

	void doall(TaskFunctionPtr func, void* args, uint64_t a, uint64_t b, uint64_t alignment=1, uint64_t ppt = 1) {
		printf("%d: in doall\n", index);
		if(a < b && func != 0) {
			uint64_t tmp = ppt+alignment-1;
			ppt = std::max(1ULL, tmp - (tmp % alignment));

			// avoid an extra enqueue (expensive on leaf nodes) and the possibility of
			// of an unnecessary steal by directly executing task rather than queueing
			// and then unqueueing.

			Task t(func, args, a, b, alignment, ppt);
				
			tasksLock.acquire();
			tasks.push_front(t);
			tasksLock.release();
			printf("%d: queued up\n", index);	
			while(fetch_and_add(t.done, 0) != 0) {
				Task s;
				if(dequeue(s) || steal(s)) run(s);
				else sleep(); 
			}
			delete t.done;
		}
	}

	void loop() {
		while(true) {
			// pull stuff off my queue and run
			// or steal and run
			Task s;
			if(dequeue(s) || steal(s)) run(s);
			else sleep(); 
		}
	}

	void sleep() const {
		struct timespec sleepTime;
		struct timespec returnTime;
		sleepTime.tv_sec = 0;
		sleepTime.tv_nsec = 1000000;
		nanosleep(&sleepTime, &returnTime);
	}

	void run(Task& t) {
		while(t.a < t.b) {
			// check if we need to relinquish some of our chunk...
			if((t.b-t.a) > t.ppt) {
				tasksLock.acquire();
				if(tasks.size() == 0) {
					Task n = t;
					uint64_t half = split(t);
					t.b = half;
					n.a = half;
					if(n.a < n.b) {
						int a=fetch_and_add(n.done, 1); 
						printf("Thread %d relinquishing %d (%d %d) (%d)\n", index, n.b-n.a, t.a, t.b, a);
						tasks.push_front(n);
					}
				}
				tasksLock.release();
			}
			t.func(t.args, t.a, std::min(t.a+t.ppt,t.b), this);
			t.a += t.ppt;
		}
		printf("Thread %d finished %d %d (%d)\n", index, t.a, t.b, t.done);
		fetch_and_add(t.done, -1);
	}

private:

	uint64_t split(Task const& t) {
		uint64_t half = (t.a+t.b)/2;
		uint64_t r = half + (t.alignment/2);
		half = r - (r % t.alignment);
		if(half < t.a) half = t.a;
		if(half > t.b) half = t.b;
		return half;
	}

	bool dequeue(Task& out) {
		// if only one task and size is larger than ppt in queue pull half
		// otherwise pull the whole thing
		tasksLock.acquire();
		if(tasks.size() >= 1) {
			out = tasks.front();
			if(tasks.size() == 1 && (out.b-out.a) > out.ppt) {
				uint64_t half = split(out);
				printf("Thread %d dequeuing and splitting (%d %d %d) (%d)\n", index, out.a, half, out.b, out.done);
				out.b = half;
				tasks.front().a = half;
				fetch_and_add(tasks.front().done, 1); 
				// wake a sleeping thread
			}
			else {
				printf("Thread %d dequeuing the whole thing (%d %d) (%d)\n", index, out.a, out.b, out.done);
				tasks.pop_front();
			}
			tasksLock.release();
			return true;
		}
		tasksLock.release();
		return false;
	}

	bool steal(Task& out) {
		// check other threads for available tasks, don't check myself.
		bool found = false;
		for(uint64_t i = 0; i < pool->workers.size() && !found; i++) {
			if(i != index) {
				Thread& t = *(pool->workers[i]);
				t.tasksLock.acquire();
				if(t.tasks.size() > 0) {
				printf("Thread %d stealing from %d\n", index, t.index);
					out = t.tasks.back();
					t.tasks.pop_back();
					t.tasksLock.release();
					if(out.b-out.a > out.ppt) {
						uint64_t half = split(out);
						tasksLock.acquire();
						tasks.push_front(out);
						out.b = half;
						tasks.front().a = half;
						fetch_and_add(tasks.front().done, 1);
						tasksLock.release();
					}
					found = true;
				} else {
					t.tasksLock.release();
				}
			}
		}
		return found;
	}
};

	inline Pool::Pool(uint64_t workers) {
		pthread_attr_t  attr;
		pthread_attr_init (&attr);
		pthread_attr_setscope (&attr, PTHREAD_SCOPE_SYSTEM);
		pthread_attr_setdetachstate (&attr, PTHREAD_CREATE_DETACHED);

		Thread* t = new Thread(this, 0);
		this->workers.push_back(t);

		for(uint64_t i = 1; i < workers; i++) {
			Thread* t = new Thread(this, i);
			pthread_create (&t->thread, &attr, Thread::start, t);
			this->workers.push_back(t);
		}
	}
*/
#endif
