#ifndef _RIPOSTE_INTERPRETER_H
#define _RIPOSTE_INTERPRETER_H

#include <map>
#include <set>
#include <deque>
#include <iostream>

#include "value.h"
#include "thread.h"
#include "random.h"
#include "bc.h"

#ifdef EPEE
#include "epee/ir.h"
#include "epee/trace.h"
#endif

////////////////////////////////////////////////////////////////////
// VM data structures
///////////////////////////////////////////////////////////////////

struct Instruction {
	int16_t a, b, c;
	ByteCode::Enum bc:16;

	Instruction(ByteCode::Enum bc, int16_t a=0, int16_t b=0, int16_t c=0) :
		a(a), b(b), c(c), bc(bc) {}
	
	std::string toString() const {
		return std::string("") 
            + ByteCode::toString(bc) 
            + "\t" + intToStr(a) 
            + "\t" + intToStr(b) 
            + "\t" + intToStr(c);
	}
};

class StringTable {
	std::map<std::string, String> stringTable;
	Lock lock;
public:

	String in(std::string const& s) {
		lock.acquire();
		std::map<std::string, String>::const_iterator i = stringTable.find(s);
		if(i == stringTable.end()) {
            String string = new (s.size()+1) StringImpl();
			memcpy((void*)string->s, s.c_str(), s.size()+1);
			stringTable[s] = string;
			lock.release();
			return string;
		} else {
            String ss = i->second;
			lock.release();
			return ss;
		}
	}

	std::string out(String s) const {
		return std::string(s->s);
	}

    std::map<std::string, String> const& table() const {
        return stringTable;
    }
};

struct CompiledCall {
	List call;

    List arguments;
    Character names;
	int64_t dotIndex;

    List extraArgs;
    Character extraNames;
	
	explicit CompiledCall(
        List const& call, 
        List const& arguments, 
        Character const& names,
        int64_t dotIndex,
        List const& extraArgs,
        Character const& extraNames) 
		: call(call)
        , arguments(arguments)
        , names(names)
        , dotIndex(dotIndex)
        , extraArgs(extraArgs)
        , extraNames(extraNames) {}
};

struct Code : public HeapObject {
    Value expression;

    std::vector<Instruction> bc;
    std::vector<Value> constants;
    std::vector<CompiledCall> calls;
    int registers;
    
	void printByteCode(State const& state) const;
	void visit() const;
};

struct Prototype : public HeapObject {
    Code const* code;
    String string;
    Value formals;

    Character parameters;
    List defaults;
	int dotIndex;

	void visit() const;
};

class Dictionary : public HeapObject {
protected:
	uint64_t size, load, ksize;

    struct Pair { String n; Value v; };
	
	struct Inner : public HeapObject {
		Pair d[];
	};

	Inner* d;

	// Returns the location of variable `name` in this environment or
	// an empty pair (String::NA, Value::Nil).
	// success is set to true if the variable is found. This boolean flag
	// is necessary for compiler optimizations to eliminate expensive control flow.
	Pair* find(String name, bool& success) const ALWAYS_INLINE {
		uint64_t i = ((uint64_t)name >> 3) & ksize;
		Pair* first = &d->d[i];
		if(__builtin_expect(first->n == name, true)) {
			success = true;
			return first;
		}
		uint64_t j = 0;
		while(d->d[i].n != Strings::NA) {
			i = (i+(++j)) & ksize;
			if(__builtin_expect(d->d[i].n == name, true)) {
				success = true;
				return &d->d[i];
			}
		}
		success = false;
		return &d->d[i];
	}

	// Returns the location where variable `name` should be inserted.
	// Assumes that `name` doesn't exist in the hash table yet.
	// Used for rehash and insert where this is known to be true.
	Pair* slot(String name) const ALWAYS_INLINE {
		uint64_t i = ((uint64_t)name >> 3) & ksize;
		if(__builtin_expect(d->d[i].n == Strings::NA, true)) {
			return &d->d[i];
		}
		uint64_t j = 0;
		while(d->d[i].n != Strings::NA) {
			i = (i+(++j)) & ksize;
		}
		return &d->d[i];
	}

	void rehash(uint64_t s) {
		uint64_t old_size = size;
		uint64_t old_load = load;
		Inner* old_d = d;

		d = new (sizeof(Pair)*s) Inner();
		size = s;
		ksize = s-1;
		clear();
		
		// copy over previous populated values...
		if(old_load > 0) {
			for(uint64_t i = 0; i < old_size; i++) {
				if(old_d->d[i].n != Strings::NA) {
					load++;
					*slot(old_d->d[i].n) = old_d->d[i];
				}
			}
		}
	}

public:
	Dictionary(int64_t initialLoad) : size(0), load(0), d(0) {
		rehash(std::max((uint64_t)1, nextPow2(initialLoad*2)));
	}

	bool has(String name) const ALWAYS_INLINE {
		bool success;
		find(name, success);
		return success;
	}

	Value const& get(String name) const ALWAYS_INLINE {
		bool success;
		return find(name, success)->v;
	}

	Value& insert(String name) {
		bool success;
		Pair* p = find(name, success);
		if(!success) {
			if(((load+1) * 2) > size)
				rehash((size*2));
			load++;
			p = slot(name);
			p->n = name;
		}
		return p->v;
	}

	void remove(String name) {
		bool success;
		Pair* p = find(name, success);
		if(success) {
			load--;
			memset(p, 0, sizeof(Pair));
		}
	}

	void clear() {
		memset(d->d, 0, sizeof(Pair)*size); 
		load = 0;
	}

	// clone with room for extra elements
	Dictionary* clone(uint64_t extra) const {
		Dictionary* clone = new Dictionary((load+extra)*2);
		// copy over elements
		if(load > 0) {
			for(uint64_t i = 0; i < size; i++) {
				if(d->d[i].n != Strings::NA) {
					clone->load++;
					*clone->slot(d->d[i].n) = d->d[i];
				}
			}
		}
		return clone;
	}

	class const_iterator {
		Dictionary const* d;
		int64_t i;
	public:
		const_iterator(Dictionary const* d, int64_t idx) {
			this->d = d;
			i = std::max((int64_t)0, std::min((int64_t)d->size, idx));
			while(d->d->d[i].n == Strings::NA && i < (int64_t)d->size) i++;
		}
		String string() const { return d->d->d[i].n; }	
		Value const& value() const { return d->d->d[i].v; }
		const_iterator& operator++() {
			while(d->d->d[++i].n == Strings::NA && i < (int64_t)d->size);
			return *this;
		}
		bool operator==(const_iterator const& o) {
			return d == o.d && i == o.i;
		}
		bool operator!=(const_iterator const& o) {
			return d != o.d || i != o.i;
		}
	};

	const_iterator begin() const {
		return const_iterator(this, 0);
	}

	const_iterator end() const {
		return const_iterator(this, size);
	}

	void visit() const;

    uint64_t Size() const { return load; }
};

class Environment : public Dictionary {

	Environment* enclosure;
    Dictionary* attributes;

public:
	explicit Environment(int64_t initialLoad, Environment* enclosure)
        : Dictionary(initialLoad)
        , enclosure(enclosure)
        , attributes(0) {}

	Environment* getEnclosure() const { return enclosure; }
	void setEnclosure(Environment* env) { enclosure = env; }

    Dictionary* getAttributes() const { return attributes; }
    void setAttributes(Dictionary* d) { attributes = d; }
    bool hasAttributes() const { return attributes != 0; }

	// Look up insertion location using R <<- rules
	// (i.e. find variable with same name in the lexical scope)
	Value& insertRecursive(String name, Environment*& env) const ALWAYS_INLINE {
		env = (Environment*)this;
		
        bool success;
		Pair* p = env->find(name, success);
		while(!success && (env = env->getEnclosure())) {
			p = env->find(name, success);
		}
		return p->v;
	}
	
	// Look up variable using standard R lexical scoping rules
	// Should be same as insertRecursive, but with extra constness
	Value const& getRecursive(String name, Environment*& env) const ALWAYS_INLINE {
		return insertRecursive(name, env);
	}

	void visit() const;
};

struct StackFrame {
	Value* registers;
	Environment* environment;
    Code const* code;
    bool isPromise;

	Instruction const* returnpc;
};

////////////////////////////////////////////////////////////////////
// Global shared state 
///////////////////////////////////////////////////////////////////

class Thread;

#define DEFAULT_NUM_REGISTERS 10000 

class State {
public:
	StringTable strings;

    std::map<std::string, void*> handles;
	
    Environment* empty;
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

    std::vector<Value> gcStack;
    Code* promiseCode;
    bool visible;

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

	Value eval(Code const* code, Environment* environment, int64_t resultSlot = 0); 
	Value eval(Code const* code);

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
                } catch(RiposteException const& e) { 
                    std::cout << "Error (" << e.kind() << ":" << (int)index << ") " << e.what();
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

#endif
