#ifndef _RIPOSTE_INTERPRETER_H
#define _RIPOSTE_INTERPRETER_H

#include <map>
#include <set>
#include <deque>
#include <list>
#include <iostream>

#include "riposte.h"
#include "value.h"
#include "thread.h"
#include "random.h"
#include "bc.h"

#ifdef EPEE
#include "epee/ir.h"
#include "epee/trace.h"
#endif

class Global;

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
    
	void printByteCode(Global const& global) const;
	void visit() const;

    static void Finalize(HeapObject* o);
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

// For R API support
struct SEXPREC : public HeapObject {
    Value v;
    SEXPREC(Value const& v) : v(v) {}
    void visit() const;
};

typedef SEXPREC* SEXP;

struct SEXPStack {
    int* size;
    SEXP* stack;
};

////////////////////////////////////////////////////////////////////
// Global state (shared across all threads) 
///////////////////////////////////////////////////////////////////

class State;

class Global {
public:
	StringTable strings;

    std::map<std::string, void*> handles;
	
    Environment* empty;
	Environment* global;
    Code* promiseCode;

    // For R API support
    Lock apiLock;
    SEXPStack* apiStack;
    // SEXPs that the API needs to have live between calls.
    std::list<SEXP> installedSEXPs;
    SEXP installSEXP(SEXP s) {
        installedSEXPs.push_back(s);
        return s;
    }
    SEXP installSEXP(Value const& v) {
        return installSEXP(new SEXPREC(v));
    }
    void uninstallSEXP(SEXP s) {
        // go back to front, assuming we're uninstalling something
        // we recently installed.
        for(std::list<SEXP>::reverse_iterator i = installedSEXPs.rbegin();
            i != installedSEXPs.rend(); ++i) {
            if(*i == s) {
                installedSEXPs.erase((++i).base());
                break;
            }
        }
    }

    std::list<State*> states;
    State* getState();
    void deleteState(State* s);

    bool profile;
	bool verbose;
	bool epeeEnabled;

    Riposte::Format format;
    
    Character arguments;

    TaskQueues queues;

    Global(uint64_t threads, int64_t argc, char** argv);

    void dumpProfile(std::string filename);
	std::string stringify(Value const& v) const;
	std::string deparse(Value const& v) const;

	String internStr(std::string s) {
		return strings.in(s);
	}

	std::string externStr(String s) const {
		return strings.out(s);
	}
};

// Global pointer, used by the R API, which
// doesn't get passed a per-thread State.
extern Global* global;

////////////////////////////////////////////////////////////////////
// Per-thread State 
///////////////////////////////////////////////////////////////////

#define DEFAULT_NUM_REGISTERS 10000 

class State {
public:
    // Shared global state
	Global& global;

    // Interpreter execution data structures
	Value* registers;
	std::vector<StackFrame> stack;
	StackFrame frame;
    bool visible;
    int64_t assignment[256], set[256]; // temporary space for matching arguments
	
#ifdef EPEE
	Traces traces;
#endif

	Random random;
    std::vector<Value> gcStack;
    TaskQueue* queue;

	State(Global& global, TaskQueue* queue);

	StackFrame& push() {
		stack.push_back(frame);
		return frame;
	}

	void pop() {
		frame = stack.back();
		stack.pop_back();
	}

	std::string stringify(Value const& v) const { return global.stringify(v); }
	std::string deparse(Value const& v) const { return global.deparse(v); }
	String internStr(std::string s) { return global.internStr(s); }
	std::string externStr(String s) const { return global.externStr(s); }

	Value evalTopLevel(Code const* code, Environment* environment, int64_t resultSlot = 0); 
	Value eval(Code const* code, Environment* environment, int64_t resultSlot = 0); 
	Value eval(Code const* code);
    Value eval(Promise const& p, int64_t resultSlot = 0);
};

#endif

