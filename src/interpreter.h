#ifndef _RIPOSTE_INTERPRETER_H
#define _RIPOSTE_INTERPRETER_H

#include <map>
#include <set>
#include <deque>

#include "value.h"
#include "thread.h"
#include "random.h"

#ifdef EPEE
#include "epee/ir.h"
#include "epee/trace.h"
#endif

class Thread;

void marker(char* a);

////////////////////////////////////////////////////////////////////
// VM data structures
///////////////////////////////////////////////////////////////////


// TODO: Make this use a good concurrent map implementation 
class StringTable {
	std::map<std::string, String> stringTable;
	Lock lock;
public:
	StringTable() {
	#define ENUM_STRING_TABLE(name, string) \
		stringTable[string] = Strings::name; 
		STRINGS(ENUM_STRING_TABLE);
	}

	String in(std::string const& s) {
		lock.acquire();
		std::map<std::string, String>::const_iterator i = stringTable.find(s);
		if(i == stringTable.end()) {
			char* str = new char[s.size()+1];
			memcpy(str, s.c_str(), s.size()+1);
			String string = (String)str;
			stringTable[s] = string;
			lock.release();
			return string;
		} else {
			lock.release();
			return i->second;
		}
	}

	std::string out(String s) const {
		return std::string(s);
	}
};


struct CompiledCall {
	List call;

	PairList arguments;
	int64_t argumentsSize;
	int64_t dotIndex;
	bool named;
	
	explicit CompiledCall(List const& call, PairList arguments, int64_t dotIndex, bool named) 
		: call(call), arguments(arguments), argumentsSize(arguments.size()), dotIndex(dotIndex), named(named) {}
};

struct Prototype : public HeapObject {
	Value expression;
	String string;

	PairList parameters;
	int64_t parametersSize;
	int dotIndex;

	int registers;
	std::vector<Value> constants;
	std::vector<CompiledCall> calls; 

	std::vector<Instruction> bc;

	void visit() const;

	static void printByteCode(Prototype const* prototype, State const& state); 
};

struct StackFrame {
	Value* registers;
	Environment* environment;
	Prototype const* prototype;

	Instruction const* returnpc;
	
	int64_t dest;
	Environment* env;
};

// TODO: Careful, args and result might overlap!

struct InternalFunction {
    typedef void (*Ptr)(Thread& s, Value const* args, Value& result);
	
    Ptr ptr;
	int64_t params;
};


#define DEFAULT_NUM_REGISTERS 10000


////////////////////////////////////////////////////////////////////
// Global shared state 
///////////////////////////////////////////////////////////////////

class State {
public:
	StringTable strings;
	
	std::vector<InternalFunction> internalFunctions;
	std::map<String, int64_t> internalFunctionIndex;
	
	std::vector<Environment*> path;
	Environment* global;

	std::vector<Thread*> threads;

	bool verbose;
	bool epeeEnabled;

    enum Format {
        RiposteFormat,
        RFormat
    };
    Format format;
    
    int64_t done;
	
    Character arguments;

	State(uint64_t threads, int64_t argc, char** argv);

	~State() {
		fetch_and_add(&done, 1);
		while(fetch_and_add(&done, 0) != (int64_t)threads.size()) { 
			sleep(); 
		}
	}


	Thread& getMainThread() const {
		return *threads[0];
	}

	void registerInternalFunction(String s, InternalFunction::Ptr internalFunction, int64_t params) {
		InternalFunction i = { internalFunction, params };
		internalFunctions.push_back(i);
		internalFunctionIndex[s] = internalFunctions.size()-1;
	}

	void interpreter_init(Thread& state);
	
	std::string stringify(Value const& v) const;
	std::string deparse(Value const& v) const;

	String internStr(std::string s) {
		return strings.in(s);
	}

	std::string externStr(String s) const {
		return strings.out(s);
	}
};

////////////////////////////////////////////////////////////////////
// Per-thread state 
///////////////////////////////////////////////////////////////////

class Thread {
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

	State& state;
	uint64_t index;
	pthread_t thread;
	
	Value* registers;

	std::vector<StackFrame> stack;
	StackFrame frame;

	std::vector<std::string> warnings;

    std::vector<Value> gcStack;

#ifdef EPEE
	Traces traces;
#endif

	std::deque<Task> tasks;
	Lock tasksLock;
	Random random;	
	int64_t steals;

	int64_t assignment[64], set[64]; // temporary space for matching arguments
	
	Thread(State& state, uint64_t index);

	StackFrame& push() {
		stack.push_back(frame);
		return frame;
	}

	void pop() {
		frame = stack.back();
		stack.pop_back();
	}

	std::string stringify(Value const& v) const { return state.stringify(v); }
	std::string deparse(Value const& v) const { return state.deparse(v); }
	String internStr(std::string s) { return state.internStr(s); }
	std::string externStr(String s) const { return state.externStr(s); }

	static void* start(void* ptr) {
		Thread* p = (Thread*)ptr;
		p->loop();
		return 0;
	}

	Value eval(Prototype const* prototype, Environment* environment); 
	Value eval(Prototype const* prototype);
	
	void doall(Task::HeaderPtr header, Task::FunctionPtr func, void* args, uint64_t a, uint64_t b, uint64_t alignment=1, uint64_t ppt = 1) {
		if(a < b && func != 0) {
			uint64_t tmp = ppt+alignment-1;
			ppt = std::max((uint64_t)1, tmp - (tmp % alignment));

			Task t(header, func, args, a, b, alignment, ppt);
			run(t);
	
			while(fetch_and_add(t.done, 0) != 0) {
				Task s;
				if(dequeue(s) || steal(s)) run(s);
				else sleep(); 
			}
		}
	}

private:
	void loop() {
		while(fetch_and_add(&(state.done), 0) == 0) {
			// pull stuff off my queue and run
			// or steal and run
			Task s;
			if(dequeue(s) || steal(s)) {
				try {
					run(s);
				} catch(RiposteError& error) {
					printf("Error (riposte:%d): %s\n", (int)index, error.what().c_str());
				} catch(RuntimeError& error) {
					printf("Error (runtime:%d): %s\n", (int)index, error.what().c_str());
				} catch(CompileError& error) {
					printf("Error (compiler:%d): %s\n", (int)index, error.what().c_str());
				}
			} else sleep(); 
		}
		fetch_and_add(&(state.done), 1);
	}

	void run(Task& t) {
		void* h = t.header != NULL ? t.header(t.args, t.a, t.b, *this) : 0;
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
			t.func(t.args, h, t.a, std::min(t.a+t.ppt,t.b), *this);
			t.a += t.ppt;
		}
		//printf("Thread %d finished %d %d (%d)\n", index, t.a, t.b, t.done);
		fetch_and_add(t.done, -1);
	}

	uint64_t split(Task const& t) {
		uint64_t half = (t.a+t.b)/2;
		uint64_t r = half + (t.alignment/2);
		half = r - (r % t.alignment);
		if(half < t.a) half = t.a;
		if(half > t.b) half = t.b;
		return half;
	}

	bool dequeue(Task& out) {
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

	bool steal(Task& out) {
		// check other threads for available tasks, don't check myself.
		bool found = false;
		for(uint64_t i = 0; i < state.threads.size() && !found; i++) {
			if(i != index) {
				Thread& t = *(state.threads[i]);
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
};

inline State::State(uint64_t threads, int64_t argc, char** argv) 
	: verbose(false), epeeEnabled(true), format(State::RiposteFormat), done(0) {
	Environment* base = new Environment(1,0,0,Null::Singleton());
	this->global = new Environment(1,base,0,Null::Singleton());
	path.push_back(base);

	arguments = Character(argc);
	for(int64_t i = 0; i < argc; i++) {
		arguments[i] = internStr(std::string(argv[i]));
	}
	
	pthread_attr_t  attr;
	pthread_attr_init (&attr);
	pthread_attr_setscope (&attr, PTHREAD_SCOPE_SYSTEM);
	pthread_attr_setdetachstate (&attr, PTHREAD_CREATE_DETACHED);

	Thread* t = new Thread(*this, 0);
	this->threads.push_back(t);

	for(uint64_t i = 1; i < threads; i++) {
		Thread* t = new Thread(*this, i);
		pthread_create (&t->thread, &attr, Thread::start, t);
		this->threads.push_back(t);
	}

	interpreter_init(getMainThread());
}

#endif
