#ifndef _RIPOSTE_INTERPRETER_H
#define _RIPOSTE_INTERPRETER_H

#include <map>
#include <deque>

#include "value.h"
#include "thread.h"

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

#define TRACE_MAX_NODES (128)
#define TRACE_MAX_OUTPUTS (128)
#define TRACE_MAX_VECTOR_REGISTERS (32)
#define TRACE_VECTOR_WIDTH (64)
//maximum number of instructions to record before dropping out of the
//recording interpreter
#define TRACE_MAX_RECORDED (1024)
#define TRACE_MAX_TRACES (4)
#define TRACE_MAX_NODES_PER_COMMIT (4)
#define TRACE_MAX_OUTPUTS_PER_COMMIT (1)

struct TraceCodeBuffer;
struct Trace {
	IRNode nodes[TRACE_MAX_NODES];

	size_t n_nodes;
	size_t n_pending_nodes;

	int64_t length;

	int64_t uniqueShapes;
	
	struct Location {
		enum Type {REG, VAR};
		Type type;
		/*union { */ //union disabled because Pointer has a Symbol with constructor
			Environment::Pointer pointer; //fat pointer to environment location
			struct {
				Value * base;
				int64_t offset;
			} reg;
		/*};*/
	};

	struct Output {
		Location location; //location where an output might exist
		                   //if that location is live and contains a future then that is a live output
		Value * value; //pointer into output_values array
	};

	Output outputs[TRACE_MAX_OUTPUTS];
	size_t n_outputs;
	size_t n_pending_outputs;

	Value output_values[TRACE_MAX_OUTPUTS];
	size_t n_output_values;
	TraceCodeBuffer * code_buffer;

	Trace() { Reset(); code_buffer = NULL; }

	IRef EmitBinary(IROpCode::Enum op, Type::Enum type, int64_t a, int64_t b) {
		IRNode & n = nodes[n_pending_nodes];
		n.enc = IRNode::BINARY;
		n.op = op;
		n.type = type;
		n.binary.a = a;
		n.binary.b = b;
		return n_pending_nodes++;
	}
	IRef EmitSpecial(IROpCode::Enum op, Type::Enum type, int64_t a, int64_t b) {
		IRNode & n = nodes[n_pending_nodes];
		n.enc = IRNode::SPECIAL;
		n.op = op;
		n.type = type;
		n.special.a = a;
		n.special.b = b;
		return n_pending_nodes++;
	}
	IRef EmitUnary(IROpCode::Enum op, Type::Enum type, int64_t a, int64_t data=0) {
		IRNode & n = nodes[n_pending_nodes];
		n.enc = IRNode::UNARY;
		n.op = op;
		n.type = type;
		n.unary.a = a;
		n.unary.data = data;
		return n_pending_nodes++;
	}
	IRef EmitFold(IROpCode::Enum op, Type::Enum type, int64_t a, int64_t base) {
		IRNode & n = nodes[n_pending_nodes];
		n.enc = IRNode::FOLD;
		n.op = op;
		n.type = type;
		n.fold.a = a;
		n.fold.i = base;
		return n_pending_nodes++;
	}
	IRef EmitLoadC(Type::Enum type, int64_t c) {
		IRNode & n = nodes[n_pending_nodes];
		n.enc = IRNode::LOADC;
		n.op = IROpCode::loadc;
		n.type = type;
		n.loadc.i = c;
		return n_pending_nodes++;
	}
	IRef EmitLoadV(Type::Enum type,void * v) {
		IRNode & n = nodes[n_pending_nodes];
		n.enc = IRNode::LOADV;
		n.op = IROpCode::loadv;
		n.type = type;
		n.loadv.p = v;
		return n_pending_nodes++;
	}
	IRef EmitStoreV(Type::Enum type, Value * dst, int64_t a) {
		IRNode & n = nodes[n_pending_nodes];
		n.enc = IRNode::STORE;
		n.op = IROpCode::storev;
		n.type = type;
		n.store.a = a;
		n.store.dst = dst;
		return n_pending_nodes++;
	}
	IRef EmitStoreC(Type::Enum type, Value * dst, int64_t a) {
		IRNode & n = nodes[n_pending_nodes];
		n.enc = IRNode::STORE;
		n.op = IROpCode::storec;
		n.type = type;
		n.store.a = a;
		n.store.dst = dst;
		return n_pending_nodes++;
	}
	void EmitRegOutput(Value * base, int64_t id) {
		Trace::Output & out = outputs[n_pending_outputs++];
		out.location.type = Location::REG;
		out.location.reg.base = base;
		out.location.reg.offset = id;
	}
	void EmitVarOutput(Thread & state, const Environment::Pointer & p) {
		Trace::Output & out = outputs[n_pending_outputs++];
		out.location.type = Trace::Location::VAR;
		out.location.pointer = p;
	}
	void Reset();
	void InitializeOutputs(Thread & state);
	void WriteOutputs(Thread & state);
	void Execute(Thread & state);
	std::string toString(Thread & state);
private:
	void Interpret(Thread & state);
	void JIT(Thread & state);
};

//member of Thread, manages information for all traces
//and the currently recording trace (if any)
struct TraceThread {
	TraceThread()
	: live_traces(TRACE_MAX_TRACES) {
		active = false;
		config = DISABLED;
		n_recorded_since_last_exec = 0;
	}

	enum Mode {
		DISABLED,
		INTERPRET,
		COMPILE
	};
	Mode config;
	bool active;

	Trace traces[TRACE_MAX_TRACES];
	RegisterAllocator live_traces;


	Value * max_live_register_base;
	int64_t max_live_register;

	size_t n_recorded_since_last_exec;

	bool Enabled() { return DISABLED != config; }
	bool IsTracing() const { return active; }

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

	Instruction const * BeginTracing(Thread & state, Instruction const * inst) {
		if(active) {
			_error("recursive record\n");
		}
		max_live_register = NULL;
		active = true;

		return recording_interpret(state,inst);
	}

	void EndTracing(Thread & state) {
		if(active) {
			active = false;
			FlushAllTraces(state);
		}
	}

	Trace & AllocateTrace(Thread & state, int64_t shape) {
		int8_t reg;
		if(live_traces.allocate(&reg)) {
			traces[reg].Reset();
			traces[reg].length = shape;
			return traces[reg];
		} else {
			FlushAllTraces(state);
			return AllocateTrace(state,shape);
		}
	}
	int64_t TraceID(Trace & trace) {
		return &trace - &traces[0];
	}
	Trace & GetOrAllocateTrace(Thread & state, int64_t shape) {
		for(size_t i = 0; i < TRACE_MAX_TRACES; i++) {
			if(live_traces.is_live(i) && traces[i].length == shape)
				return traces[i];
		}
		return AllocateTrace(state,shape);
	}

	void Rollback(Trace & t) {
		t.n_pending_nodes = t.n_nodes;
		t.n_pending_outputs = t.n_output_values;
		if(t.n_nodes == 0) {
			size_t id = TraceID(t);
			live_traces.free(id);
		}
	}
	//commits the recorded instructions and outputs from the current op
	//if the trace does not have enough room to record another op, it is flushed
	//and the slot is freed for another trace
	void Commit(Thread & state, Trace & t) {
		t.n_nodes = t.n_pending_nodes;
		t.n_outputs = t.n_pending_outputs;
		if(t.n_nodes + TRACE_MAX_NODES_PER_COMMIT >= TRACE_MAX_NODES
		  || t.n_outputs + TRACE_MAX_OUTPUTS_PER_COMMIT >= TRACE_MAX_OUTPUTS) {
			Flush(state,t);
		}
	}
	void Flush(Thread & state, Trace & trace) {
		size_t id = TraceID(trace);
		if(live_traces.is_live(id)) {
			n_recorded_since_last_exec = 0;
			trace.Execute(state);
			assert(id < TRACE_MAX_TRACES);
			live_traces.free(TraceID(trace));
		}
	}
	void FlushAllTraces(Thread & state) {
		for(size_t i = 0; i < TRACE_MAX_TRACES; i++) {
			Flush(state,traces[i]);
		}
	}
};

#define DEFAULT_NUM_REGISTERS 10000


////////////////////////////////////////////////////////////////////
// Global shared state 
///////////////////////////////////////////////////////////////////

struct State {
	StringTable strings;
	
	std::vector<InternalFunction> internalFunctions;
	std::map<String, int64_t> internalFunctionIndex;
	
	std::vector<Environment*, traceable_allocator<Environment*> > path;
	Environment* global;

	std::vector<Thread*> threads;

	bool verbose;

	int64_t done;

	State(uint64_t threads, Environment* global, Environment* base);

	~State() {
		fetch_and_add(&done, 1);
		while(fetch_and_add(&done, 0) != (int64_t)threads.size()) { sleep(); }
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

struct Thread {
	struct Task {
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
			done = new int64_t(1);
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

	TraceThread tracing; //all state related to tracing compiler

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
			delete t.done;
		}
	}

private:
	void loop() {
		while(fetch_and_add(&(state.done), 0) == 0) {
			// pull stuff off my queue and run
			// or steal and run
			Task s;
			if(dequeue(s) || steal(s)) run(s);
			else sleep(); 
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
				uint64_t half = split(t);
				t.b = half;
				n.a = half;
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

inline State::State(uint64_t threads, Environment* global, Environment* base) : verbose(false), done(0) {
	this->global = global;
	path.push_back(base);
	
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
