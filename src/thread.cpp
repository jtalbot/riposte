
#include "thread.h"

TaskQueues::TaskQueues(uint64_t threads)
    : done(0) {

    pthread_attr_t  attr;
    pthread_attr_init (&attr);
    pthread_attr_setscope (&attr, PTHREAD_SCOPE_SYSTEM);
    pthread_attr_setdetachstate (&attr, PTHREAD_CREATE_DETACHED);

    TaskQueue* t = new TaskQueue(*this, 0);
    this->queues.push_back(t);

    for(uint64_t i = 1; i < threads; i++) {
        TaskQueue* t = new TaskQueue(*this, i);
        pthread_create (&t->pthread, &attr, TaskQueue::start, t);
        this->queues.push_back(t);
    }
}

TaskQueue::TaskQueue(TaskQueues& qs, uint64_t index)
    : qs(qs)
    , index(index) {}

void TaskQueue::doall(
    Thread& thread,
    Task::HeaderPtr header,
    Task::FunctionPtr func,
    void* args,
    uint64_t a,
    uint64_t b,
    uint64_t alignment,
    uint64_t ppt) {
    if(a < b && func != 0) {
        uint64_t tmp = ppt+alignment-1;
        ppt = std::max((uint64_t)1, tmp - (tmp % alignment));

        Task t(header, func, args, a, b, alignment, ppt);
        run(thread, t);
	
        while(fetch_and_add(t.done, 0) != 0) {
            Task s;
            if(dequeue(s) || steal(s)) run(thread, s);
            else sleep();
        }
    }
}

void TaskQueue::loop() {
    while(fetch_and_add(&(qs.done), 0) == 0) {
        // pull stuff off my queue and run
        // or steal and run
        Task s;
        if(dequeue(s) || steal(s)) {
            try {
                run(*thread, s);
            } catch(RiposteException const& e) { 
                std::cout << "Error (" << e.kind() << ":" << (int)index << ") " << e.what();
            } 
        } else sleep(); 
    }
    fetch_and_add(&(qs.done), 1);
}

void TaskQueue::run(Thread& thread, Task& t) {
    void* h = t.header != NULL ? t.header(t.args, t.a, t.b, thread) : 0;
    while(t.a < t.b) {
        // check if we need to relinquish some of our chunk...
        int64_t s = atomic_xchg(&steals, 0);
        if(s > 0 && (t.b-t.a) > t.ppt) {
            Task n = t;
            if((t.b-t.a) > t.ppt*4) {
                uint64_t half = split(t);
                t.b = half;
                n.a = half;
            } else {
                t.b = t.a+t.ppt;
                n.a = t.a+t.ppt;
            }
            if(n.a < n.b) {
                //printf("Thread %d relinquishing %d (%d %d)\n", index, n.b-n.a, t.a, t.b);
                tasksLock.acquire();
                fetch_and_add(t.done, 1); 
                tasks.push_front(n);
                tasksLock.release();
            }
        }
        t.func(t.args, h, t.a, std::min(t.a+t.ppt,t.b), thread);
        t.a += t.ppt;
    }
    //printf("Thread %d finished %d %d (%d)\n", index, t.a, t.b, t.done);
    fetch_and_add(t.done, -1);
}

uint64_t TaskQueue::split(Task const& t) {
    uint64_t half = (t.a+t.b)/2;
    uint64_t r = half + (t.alignment/2);
    half = r - (r % t.alignment);
    if(half < t.a) half = t.a;
    if(half > t.b) half = t.b;
    return half;
}

bool TaskQueue::dequeue(Task& out) {
    tasksLock.acquire();
    if(tasks.size() >= 1) {
        out = tasks.front();
        tasks.pop_front();
        tasksLock.release();
        return true;
    }
    tasksLock.release();
    return false;
}

bool TaskQueue::steal(Task& out) {
    // check other queues for available tasks, don't check myself.
    bool found = false;
    for(uint64_t i = 0; i < qs.queues.size() && !found; i++) {
        if(i != index) {
            TaskQueue& t = *(qs.queues[i]);
            t.tasksLock.acquire();
            if(t.tasks.size() > 0) {
                out = t.tasks.back();
                t.tasks.pop_back();
                t.tasksLock.release();
                found = true;
            } else {
                fetch_and_add(&t.steals,1);
                t.tasksLock.release();
            }
        }
    }
    return found;
}

