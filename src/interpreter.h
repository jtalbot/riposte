#ifndef _RIPOSTE_INTERPRETER_H
#define _RIPOSTE_INTERPRETER_H

#include <map>
#include <set>
#include <deque>

#include "value.h"
#include "thread.h"
#include "ir.h"

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
		return std::string(s);
	}
};


struct CompiledCall : public gc {
	List call;

	PairList arguments;
	int64_t dotIndex;
	bool named;
	
	explicit CompiledCall(List const& call, PairList arguments, int64_t dotIndex, bool named) 
		: call(call), arguments(arguments), dotIndex(dotIndex), named(named) {}
};

struct Prototype : public gc {
	Value expression;
	String string;

	PairList parameters;
	int dotIndex;

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
	
	// result can go in a register or in an environment or in dots
	enum Destination {
		REG,
		MEMORY,
		DOTS
	};
	Destination dest;
	union {
		int64_t i;
		String s;
	};
	Environment* env;
};

// TODO: Careful, args and result might overlap!
typedef void (*InternalFunctionPtr)(Thread& s, Value const* args, Value& result);

struct InternalFunction {
	InternalFunctionPtr ptr;
	int64_t params;
};

#ifdef ENABLE_JIT

#define TRACE_MAX_VECTOR_REGISTERS (32)
#define TRACE_VECTOR_WIDTH (64)
//maximum number of instructions to record before dropping out of the
//recording interpreter
#define TRACE_MAX_RECORDED (1024)

struct TraceCodeBuffer;
class Trace : public gc {

	public:	

		std::vector<IRNode, traceable_allocator<IRNode> > nodes;
		std::set<Environment*> liveEnvironments;

		struct Output {
			enum Type { REG, MEMORY };
			Type type;
			union {
				Value* reg;
				Environment::Pointer pointer;
			};
			IRef ref;	   //location of the associated store
		};

		std::vector<Output> outputs;

		TraceCodeBuffer * code_buffer;

		size_t n_recorded_since_last_exec;

		Trace();

		void Bind(Thread& thread, Value const& v);
		void Flush(Thread & thread);

		IRef EmitCoerce(IRef a, Type::Enum dst_type);
		IRef EmitUnary(IROpCode::Enum op, Type::Enum type, IRef a, int64_t data); 
		IRef EmitBinary(IROpCode::Enum op, Type::Enum type, IRef a, IRef b, int64_t data);
		IRef EmitTrinary(IROpCode::Enum op, Type::Enum type, IRef a, IRef b, IRef c);

		IRef EmitFilter(IRef a, IRef b);
		IRef EmitSplit(IRef x, IRef f, int64_t levels);

		IRef EmitSequence(int64_t length, int64_t a, int64_t b);
		IRef EmitSequence(int64_t length, double a, double b);
		IRef EmitConstant(Type::Enum type, int64_t length, int64_t c);
		IRef EmitLoad(Value const& v);

		static Type::Enum futureType(Value const& v) {
			if(v.isFuture()) return v.future.typ;
			else return v.type;
		}

		struct LoadCache {
			IRef get(Trace & trace, const Value& v) {
				uint64_t idx = (int64_t) v.p;
				idx += idx >> 32;
				idx += idx >> 16;
				idx += idx >> 8;
				idx &= 0xFF;
				IRef cached = cache[idx];
				if(cached < trace.nodes.size() &&
						trace.nodes[cached].op == IROpCode::load &&
						trace.nodes[cached].out.p == v.p) {
					return cached;
				} else {
					return (cache[idx] = trace.EmitLoad(v));
				}
			}
			IRef cache[256];
		};
		LoadCache loadCache;

		IRef GetRef(Value const& v) {
			if(v.isFuture()) return v.future.ref;
			else if(v.length == 1) return EmitConstant(v.type, 1, v.i);
			else return loadCache.get(*this, v);
		}

		template< template<class X> class Group >
			Value EmitUnary(IROpCode::Enum op, Value const& a, int64_t data) {
				IRef r;
				if(futureType(a) == Type::Double) {
					r = EmitUnary(op, Group<Double>::R::VectorType, EmitCoerce(GetRef(a), Group<Double>::MA::VectorType), data);
				} else if(futureType(a) == Type::Integer) {
					r = EmitUnary(op, Group<Integer>::R::VectorType, EmitCoerce(GetRef(a), Group<Integer>::MA::VectorType), data);
				} else if(futureType(a) == Type::Logical) {
					r = EmitUnary(op, Group<Logical>::R::VectorType, EmitCoerce(GetRef(a), Group<Logical>::MA::VectorType), data);
				} else _error("Attempting to record invalid type in EmitUnary");
				Value v;
				Future::Init(v, nodes[r].type, nodes[r].shape.length, r);
				return v;
			}

		template< template<class X, class Y> class Group >
			Value EmitBinary(IROpCode::Enum op, Value const& a, Value const& b, int64_t data) {
				IRef r;
				if(futureType(a) == Type::Double) {
					if(futureType(b) == Type::Double)
						r = EmitBinary(op, Group<Double,Double>::R::VectorType, EmitCoerce(GetRef(a), Group<Double,Double>::MA::VectorType), EmitCoerce(GetRef(b), Group<Double,Double>::MB::VectorType), data);
					else if(futureType(b) == Type::Integer)
						r = EmitBinary(op, Group<Double,Integer>::R::VectorType, EmitCoerce(GetRef(a), Group<Double,Integer>::MA::VectorType), EmitCoerce(GetRef(b), Group<Double,Integer>::MB::VectorType), data);
					else if(futureType(b) == Type::Logical)
						r = EmitBinary(op, Group<Double,Logical>::R::VectorType, EmitCoerce(GetRef(a), Group<Double,Logical>::MA::VectorType), EmitCoerce(GetRef(b), Group<Double,Logical>::MB::VectorType), data);
					else _error("Attempting to record invalid type in EmitBinary");
				} else if(futureType(a) == Type::Integer) {
					if(futureType(b) == Type::Double)
						r = EmitBinary(op, Group<Integer,Double>::R::VectorType, EmitCoerce(GetRef(a), Group<Integer,Double>::MA::VectorType), EmitCoerce(GetRef(b), Group<Integer,Double>::MB::VectorType), data);
					else if(futureType(b) == Type::Integer)
						r = EmitBinary(op, Group<Integer,Integer>::R::VectorType, EmitCoerce(GetRef(a), Group<Integer,Integer>::MA::VectorType), EmitCoerce(GetRef(b), Group<Integer,Integer>::MB::VectorType), data);
					else if(futureType(b) == Type::Logical)
						r = EmitBinary(op, Group<Integer,Logical>::R::VectorType, EmitCoerce(GetRef(a), Group<Integer,Logical>::MA::VectorType), EmitCoerce(GetRef(b), Group<Integer,Logical>::MB::VectorType), data);
					else _error("Attempting to record invalid type in EmitBinary");
				} else if(futureType(a) == Type::Logical) {
					if(futureType(b) == Type::Double)
						r = EmitBinary(op, Group<Logical,Double>::R::VectorType, EmitCoerce(GetRef(a), Group<Logical,Double>::MA::VectorType), EmitCoerce(GetRef(b), Group<Logical,Double>::MB::VectorType), data);
					else if(futureType(b) == Type::Integer)
						r = EmitBinary(op, Group<Logical,Integer>::R::VectorType, EmitCoerce(GetRef(a), Group<Logical,Integer>::MA::VectorType), EmitCoerce(GetRef(b), Group<Logical,Integer>::MB::VectorType), data);
					else if(futureType(b) == Type::Logical)
						r = EmitBinary(op, Group<Logical,Logical>::R::VectorType, EmitCoerce(GetRef(a), Group<Logical,Logical>::MA::VectorType), EmitCoerce(GetRef(b), Group<Logical,Logical>::MB::VectorType), data);
					else _error("Attempting to record invalid type in EmitBinary");
				} else _error("Attempting to record invalid type in EmitBinary");
				Value v;
				Future::Init(v, nodes[r].type, nodes[r].shape.length, r);
				return v;
			}

		Value EmitSplit(Value const& a, Value const& b, int64_t data) {
			IRef r = EmitSplit(GetRef(a), EmitCoerce(GetRef(b), Type::Integer), data);
			Value v;
			Future::Init(v, nodes[r].type, nodes[r].shape.length, r);
			return v;
		}

		Value AddConstant(Type::Enum type, int64_t length, int64_t c) {
			IRef r = EmitConstant(type, length, c);
			Value v;
			Future::Init(v, nodes[r].type, nodes[r].shape.length, r);
			return v;
		}

		Value AddSequence(int64_t length, int64_t a, int64_t b) {
			IRef r = EmitSequence(length, a, b);
			Value v;
			Future::Init(v, nodes[r].type, nodes[r].shape.length, r);
			return v;
		}

		Value AddSequence(int64_t length, double a, double b) {
			IRef r = EmitSequence(length, a, b);
			Value v;
			Future::Init(v, nodes[r].type, nodes[r].shape.length, r);
			return v;
		}

		Value AddGather(Value const& a, Value const& i) {
			IRef r = EmitUnary(IROpCode::gather, a.type, EmitCoerce(GetRef(i), Type::Integer), ((int64_t)a.p)-8);
			Value v;
			Future::Init(v, nodes[r].type, nodes[r].shape.length, r);
			return v;
		}

		void addEnvironment(Environment* env) { 
			if(nodes.size() > 0)
				liveEnvironments.insert(env); 
		}

		void killEnvironment(Environment* env) {
			liveEnvironments.erase(env);
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

#endif

#define DEFAULT_NUM_REGISTERS 10000


////////////////////////////////////////////////////////////////////
// Global shared state 
///////////////////////////////////////////////////////////////////

class State : public gc {
public:
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
#ifdef ENABLE_JIT
	std::string stringify(Trace const & t) const;
#endif
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

class Thread : public gc {
public:
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

#ifdef ENABLE_JIT
	Trace trace; //all state related to tracing compiler
#endif

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
#ifdef ENABLE_JIT
	std::string stringify(Trace const & t) const { return state.stringify(t); }
#endif
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
