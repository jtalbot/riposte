#ifndef _RIPOSTE_INTERPRETER_H
#define _RIPOSTE_INTERPRETER_H

#include <map>
#include <set>
#include <deque>

#include "value.h"
#include "thread.h"
#include "random.h"
#include "vector_jit/trace.h"

class Thread;

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

struct CallSite {
	List call;

	PairList arguments;
	int64_t argumentsSize;
	int64_t dotIndex;
	bool hasNames;
	bool hasDots;

	explicit CallSite(
		List const& call, 
		PairList arguments, 
		int64_t dotIndex, 
		bool hasNames,
		bool hasDots) 
		: call(call)
		, arguments(arguments)
		, argumentsSize(arguments.size())
		, dotIndex(dotIndex)
		, hasNames(hasNames) 
		, hasDots(hasDots) {}
};

struct Shape {
	std::map<String, size_t> m;
	size_t add(String s) {
		std::map<String, size_t>::const_iterator i = m.find(s);
		if(i != m.end()) {
			return i->second;
		}
		else {
			size_t size = m.size();
			m[s] = size;
			return size;
		}
	}
};

struct IC {
	Shape* shape;
	size_t slot;
	String s;
};

struct Code {
	typedef Value (*Ptr)(Thread* thread);
	Ptr ptr;
	size_t registers;
	Shape* shape;
	
	std::vector<CallSite> calls;
	std::vector<IC> ics;
};

struct Prototype : public HeapObject {
	Value expression;
	String string;

	PairList parameters;
	int64_t parametersSize;
	int dotIndex;
	bool hasDots;

	Code const* code;

	void visit() const;
};

struct StackFrame {
	Value* registers;	// registers start here
	Value* reservedTo;	// end of registers
	Environment* environment;
	CallSite const* calls;
	IC* ics;
};

struct InternalFunction {
	// TODO: Careful, args and result might overlap!
	typedef void (*Ptr)(Thread& s, Value const* args, Value& result);

	Ptr ptr;
	int64_t params;
};

class Environment : public HeapObject {
public:
	Environment* lexical;
	Environment* dynamic;
	Value call;
	std::map<String, Value> dictionary;
	
	Shape* shape;
	size_t numDots;
	bool dotsNamed;
	Value* data;

	explicit Environment(
		Environment* lexical, 
		Environment* dynamic, 
		Shape* shape,
		int64_t dots,
		Value const& call) 
		: lexical(lexical)
		, dynamic(dynamic)
		, call(call)
		, shape(shape)
		, numDots(dots)
		, dotsNamed(false) {
		data = new Value[shape->m.size()+numDots*2];
		memset(data, 0, (shape->m.size()+numDots*2)*sizeof(Value));
	}

	void mutateShape(Shape* s) {
		// copy out old data
		std::map<String, size_t>::const_iterator i;
		for(i = shape->m.begin(); i != shape->m.end(); i++) {
			dictionary[i->first] = Slots()[i->second];
		}
	
		// make new room
		shape = s;
		data = new Value[shape->m.size()+numDots*2];
	
		// copy over stuff, assuming no dots for now.
		std::map<String, size_t>::const_iterator j;
		for(j = shape->m.begin(); j != shape->m.end(); j++) {
			std::map<String, Value>::const_iterator k;
			k = dictionary.find(j->first);
			if(k != dictionary.end()) {
				Slots()[j->second] = k->second;
				dictionary.erase(j->first);
			}
			else {
				Slots()[j->second] = Value::Nil();
			}
		}
			
	}
		
	Environment* LexicalScope() const { return lexical; }
	Environment* DynamicScope() const { return dynamic; }

	Value const* Slots() const {
		return data;
	}
	
	Value const* Dots() const {
		return data+shape->m.size();
	}

	Value* Slots() {
		return data;
	}
	
	Value* Dots() {
		return data+shape->m.size();
	}

	Value const* get(String name) const {
		std::map<String, size_t>::const_iterator i = 
			shape->m.find(name);
		if(i != shape->m.end())
			return Slots() + i->second;
	
		std::map<String, Value>::const_iterator j =
			dictionary.find(name);
		if(j != dictionary.end())
			return &j->second;
	
		return 0;
	}

	Value* set(String name, Value const& v) {
		Value* a = (Value*)get(name);
		if(!a) {
			a = &(dictionary[name]);
		}
		*a = v;
		return a;
	}

	bool has(String name) const ALWAYS_INLINE {
		return get(name) != 0;
	}

	void remove(String name) {
		_error("NYI: remove");
	}

	// Look up insertion location using R <<- rules
	// (i.e. find variable with same name in the lexical scope)
	// Note: doesn't actually insert anything.
	Value* insertRecursive(String name, Environment*& env) const ALWAYS_INLINE {
		env = (Environment*)this;
		Value* v = (Value*)env->get(name);
		while(!v && (env = env->LexicalScope())) {
			v = (Value*)env->get(name);
		}
		return v;
	}
	
	// Look up variable using standard R lexical scoping rules
	// Should be same as insertRecursive, but with extra constness
	Value const* getRecursive(String name, Environment*& env) const ALWAYS_INLINE {
		return insertRecursive(name, env);
	}

	void visit() const;

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
	bool jitEnabled;

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

#ifdef ENABLE_JIT
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

	Environment* beginEval(Environment* lexicalScope, Environment* dynamicScope);
	Value continueEval(Code const* code);
	Environment* endEval(bool liveOut);

	Value eval(Code const* code, Environment* lexicalScope, Environment* dynamicScope); 
	
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

public:
};

inline State::State(uint64_t threads, int64_t argc, char** argv) 
	: verbose(false), jitEnabled(true), done(0) {
	
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
	
	Environment* base = new Environment(NULL,NULL,new Shape(),0,Null::Singleton());
	path.push_back(base);
	
	this->global = getMainThread().beginEval(base, 0);
}

#endif
