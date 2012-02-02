#ifndef _RIPOSTE_INTERPRETER_H
#define _RIPOSTE_INTERPRETER_H

#include <map>
#include <deque>

#include "value.h"
#include "thread.h"
#include "ir.h"

#include "recording.h"
#include "register_set.h"


class Thread;

////////////////////////////////////////////////////////////////////
// VM ops 
///////////////////////////////////////////////////////////////////

#define DECLARE_INTERPRETER_FNS(bc,name,...) \
		Instruction const * bc##_op(Thread& state, Instruction const& inst);

BYTECODES(DECLARE_INTERPRETER_FNS)

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
		if((int64_t)s < 0) return std::string("..") + intToStr(-(int64_t)s);
		else return std::string(s);
	}
};


struct CompiledCall : public gc {
	List call;

	List arguments;
	Character names;
	int64_t dots;
	
	explicit CompiledCall(List const& call, List const& arguments, Character const& names, int64_t dots) 
		: call(call), arguments(arguments), names(names), dots(dots) {}
};

struct Prototype : public gc {
	Value expression;
	String string;

	Character parameters;
	List defaults;

	int dots;

	int registers;
	std::vector<Value, traceable_allocator<Value> > constants;
	std::vector<Prototype*, traceable_allocator<Prototype*> > prototypes; 	
	std::vector<CompiledCall, traceable_allocator<CompiledCall> > calls; 

	std::vector<Instruction> bc;			// bytecode
	mutable std::vector<Instruction> tbc;		// threaded bytecode
};

struct StackFrame {
	Environment* environment;
	bool ownEnvironment;
	Prototype const* prototype;

	Instruction const* returnpc;
	Value* returnbase;
	Value* result;
};

// TODO: Careful, args and result might overlap!
typedef void (*InternalFunctionPtr)(Thread& s, Value const* args, Value& result);

struct InternalFunction {
	InternalFunctionPtr ptr;
	int64_t params;
};

#define TRACE_MAX_VECTOR_REGISTERS (32)
#define TRACE_VECTOR_WIDTH (64)
//maximum number of instructions to record before dropping out of the
//recording interpreter
#define TRACE_MAX_RECORDED (1024)

struct TraceCodeBuffer;
struct Trace : public gc {
	
	struct Location {
		enum Type {REG, VAR};
		Type type;
		union {
			Environment::Pointer pointer; //fat pointer to environment location
			struct {
				Value * base;
				int64_t offset;
			} reg;
		};
	};

	struct Output {
		Location location; //location where an output might exist
		                   //if that location is live and contains a future then that is a live output
		IRef ref;	   //location of the associated store
	};


	std::vector<IRNode, traceable_allocator<IRNode> > nodes;
	
	std::vector<Output> outputs;

	TraceCodeBuffer * code_buffer;

	Value * max_live_register_base;
	int64_t max_live_register;

	size_t n_recorded_since_last_exec;
	bool active;

	Trace() : active(false) { 
		Reset(); 
		code_buffer = NULL;
 	}

	Instruction const * BeginTracing(Thread & state, Instruction const * inst) {
		if(active) {
			_error("recursive record\n");
		}
		max_live_register = NULL;
		active = true;
		try {
			return recording_interpret(state,inst);
		} catch(...) {
			Reset();
			active = false;
			throw;
		}
	}

	void EndTracing(Thread & thread) {
		if(active) {
			Flush(thread);
			active = false;
		}
	}

	void Force(Thread& thread, Value& v) {
		if(!v.isFuture()) return;
		Execute(thread, v.future.ref);
	}

	void Flush(Thread & thread) {
		if(active) {
			n_recorded_since_last_exec = 0;
			Execute(thread);
		}
	}

	IRef EmitCoerce(IRef a, Type::Enum dst_type);
	IRef EmitUnary(IROpCode::Enum op, Type::Enum type, IRef a, int64_t data); 
	IRef EmitBinary(IROpCode::Enum op, Type::Enum type, IRef a, IRef b, int64_t data);
	IRef EmitTrinary(IROpCode::Enum op, Type::Enum type, IRef a, IRef b, IRef c);
	IRef EmitFold(IROpCode::Enum op, IRef a); 
	
	IRef EmitFilter(IRef a, IRef b);
	IRef EmitSplit(IRef x, IRef f, int64_t levels);
	
	IRef EmitSpecial(IROpCode::Enum op, Type::Enum type, int64_t length, int64_t a, int64_t b);
	IRef EmitConstant(Type::Enum type, int64_t c);
	IRef EmitLoad(Value const& v);

	void RegOutput(IRef ref, Value * base, int64_t id);
	void VarOutput(IRef ref, const Environment::Pointer & p);

	void SetMaxLiveRegister(Value * base, int64_t r) {
		max_live_register_base = base;
		max_live_register = r;
	}
	void UnionWithMaxLiveRegister(Value * base, int64_t r) {
		if(base < max_live_register_base
		   || (base == max_live_register_base && r > max_live_register)) {
			SetMaxLiveRegister(base,r);
		}
	}
	bool LocationIsDead(const Trace::Location & l) {
		bool dead = l.type == Trace::Location::REG &&
		( l.reg.base < max_live_register_base ||
		  ( l.reg.base == max_live_register_base &&
		    l.reg.offset > max_live_register
		  )
		);
		//if(dead)
		//	printf("r%d is dead! long live r%d\n",(int)l.reg.offset,(int)trace.max_live_register);
		return dead;
	}

	void Rollback() {
	}
	//commits the recorded instructions and outputs from the current op
	//if the trace does not have enough room to record another op, it is flushed
	//and the slot is freed for another trace
	void Commit(Thread& thread) {
		/*n_nodes = n_pending_nodes;
		if(n_nodes + TRACE_MAX_NODES_PER_COMMIT >= TRACE_MAX_NODES) {
			Flush(thread);
		}*/
		if(nodes.size() > 128) {
			Flush(thread);
		}
	}

private:
	void Reset();
	void DiscardDeadOutputs(Thread & state);
	void WriteOutputs(Thread & state);
	void Execute(Thread & state);
	void Execute(Thread & state, IRef ref);
	std::string toString(Thread & state);

	void Interpret(Thread & state);
	void JIT(Thread & state);

	void MarkLiveOutputs(Thread& thread);
	void SimplifyOps(Thread& thread);
	void AlgebraicSimplification(Thread& thread);
	void DeadCodeElimination(Thread& thread);
};

#define DEFAULT_NUM_REGISTERS 10000


////////////////////////////////////////////////////////////////////
// Global shared state 
///////////////////////////////////////////////////////////////////

struct State : public gc {
	StringTable strings;
	
	std::vector<InternalFunction> internalFunctions;
	std::map<String, int64_t> internalFunctionIndex;
	
	std::vector<Environment*, traceable_allocator<Environment*> > path;
	Environment* global;

	std::vector<Thread*, traceable_allocator<Thread*> > threads;
	int64_t nThreads;

	bool verbose;
	bool jitEnabled;

	int64_t done;

	State(uint64_t threads);

	~State() {
		fetch_and_add(&done, 1);
		while(fetch_and_add(&done, 0) != nThreads) { sleep(); }
	}


	Thread& getMainThread() const {
		return *threads[0];
	}

	void registerInternalFunction(String s, InternalFunctionPtr internalFunction, int64_t params) {
		InternalFunction i = { internalFunction, params };
		internalFunctions.push_back(i);
		internalFunctionIndex[s] = internalFunctions.size()-1;
	}

	void interpreter_init(Thread& state);
	
	std::string stringify(Value const& v) const;
	std::string stringify(Trace const & t) const;
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

typedef void* (*TaskHeaderPtr)(void* args, uint64_t a, uint64_t b, Thread& thread);
typedef void (*TaskFunctionPtr)(void* args, void* header, uint64_t a, uint64_t b, Thread& thread);

struct Thread : public gc {
	struct Task : public gc {
		TaskHeaderPtr header;
		TaskFunctionPtr func;
		void* args;
		uint64_t a;	// start of range [a <= x < b]
		uint64_t b;	// end
		uint64_t alignment;
		uint64_t ppt;
		int64_t* done;
		Task() : func(0), args(0), done(0) {}
		Task(TaskHeaderPtr header, TaskFunctionPtr func, void* args, uint64_t a, uint64_t b, uint64_t alignment, uint64_t ppt) 
			: header(header), func(func), args(args), a(a), b(b), alignment(alignment), ppt(ppt) {
			done = new (GC) int64_t(1);
		}
	};

	State& state;
	uint64_t index;
	pthread_t thread;
	
	Value* base;
	Value* registers;

	std::vector<StackFrame, traceable_allocator<StackFrame> > stack;
	StackFrame frame;
	std::vector<Environment*, traceable_allocator<Environment*> > environments;

	std::vector<std::string> warnings;

	Trace trace; //all state related to tracing compiler

	std::deque<Task> tasks;
	Lock tasksLock;
	int64_t steals;

	int64_t assignment[64], set[64]; // temporary space for matching arguments
	
	Thread(State& state, uint64_t index) : state(state), index(index), steals(1) {
		registers = new (GC) Value[DEFAULT_NUM_REGISTERS];
		this->base = registers + DEFAULT_NUM_REGISTERS;
	}

	StackFrame& push() {
		stack.push_back(frame);
		return frame;
	}

	void pop() {
		frame = stack.back();
		stack.pop_back();
	}

	std::string stringify(Value const& v) const { return state.stringify(v); }
	std::string stringify(Trace const & t) const { return state.stringify(t); }
	std::string deparse(Value const& v) const { return state.deparse(v); }
	String internStr(std::string s) { return state.internStr(s); }
	std::string externStr(String s) const { return state.externStr(s); }

	static void* start(void* ptr) {
		Thread* p = (Thread*)ptr;
		p->loop();
		return 0;
	}

	Value eval(Function const& function);
	Value eval(Prototype const* prototype, Environment* environment); 
	Value eval(Prototype const* prototype);
	
	void doall(TaskHeaderPtr header, TaskFunctionPtr func, void* args, uint64_t a, uint64_t b, uint64_t alignment=1, uint64_t ppt = 1) {
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

inline State::State(uint64_t threads) : nThreads(threads), verbose(false), jitEnabled(true), done(0) {
	Environment* base = new (GC) Environment(0);
	this->global = new (GC) Environment(base);
	path.push_back(base);
	
	pthread_attr_t  attr;
	pthread_attr_init (&attr);
	pthread_attr_setscope (&attr, PTHREAD_SCOPE_SYSTEM);
	pthread_attr_setdetachstate (&attr, PTHREAD_CREATE_DETACHED);

	Thread* t = new (GC) Thread(*this, 0);
	this->threads.push_back(t);

	for(uint64_t i = 1; i < threads; i++) {
		Thread* t = new Thread(*this, i);
		pthread_create (&t->thread, &attr, Thread::start, t);
		this->threads.push_back(t);
	}

	interpreter_init(getMainThread());
}

#endif
