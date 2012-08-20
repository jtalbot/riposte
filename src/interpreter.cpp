#include <string>
#include <sstream>
#include <stdexcept>
#include <string>

#include "value.h"
#include "type.h"
#include "bc.h"
#include "ops.h"
#include "runtime.h"
#include "interpreter.h"
#include "compiler.h"
#include "sse.h"
#include "call.h"

#include "primes.h"

Thread::RandomSeed Thread::seed[100];

Thread::Thread(State& state, uint64_t index) : state(state), index(index), steals(1) {
	registers = new (GC) Value[DEFAULT_NUM_REGISTERS];
	this->base = registers + DEFAULT_NUM_REGISTERS;
	RandomSeed& r = seed[index];

	r.v[0] = 1;
	r.v[1] = 1;
	r.m[0] = 0x27bb2ee687b0b0fd;
	r.m[1] = 0x27bb2ee687b0b0fd;
	r.a[0] = primes[index*2];
	r.a[1] = primes[index*2+1];
	// advance a few?
	for(int64_t i = 0; i < 1000; i++) {
		r.v[0] = r.v[0] * r.m[0] + r.a[0];
		r.v[1] = r.v[1] * r.m[1] + r.a[1];
	}
}

extern Instruction const* loop_op(Thread& thread, Instruction const& inst) ALWAYS_INLINE;
extern Instruction const* mov_op(Thread& thread, Instruction const& inst) ALWAYS_INLINE;
extern Instruction const* fastmov_op(Thread& thread, Instruction const& inst) ALWAYS_INLINE;
extern Instruction const* assign_op(Thread& thread, Instruction const& inst) ALWAYS_INLINE;
extern Instruction const* forend_op(Thread& thread, Instruction const& inst) ALWAYS_INLINE;
extern Instruction const* add_op(Thread& thread, Instruction const& inst) ALWAYS_INLINE;
extern Instruction const* gather_op(Thread& thread, Instruction const& inst) ALWAYS_INLINE;
extern Instruction const* gather1_op(Thread& thread, Instruction const& inst) ALWAYS_INLINE;
extern Instruction const* jc_op(Thread& thread, Instruction const& inst) ALWAYS_INLINE;
extern Instruction const* lt_op(Thread& thread, Instruction const& inst) ALWAYS_INLINE;
extern Instruction const* ret_op(Thread& thread, Instruction const& inst) ALWAYS_INLINE;
extern Instruction const* retp_op(Thread& thread, Instruction const& inst) ALWAYS_INLINE;
extern Instruction const* internal_op(Thread& thread, Instruction const& inst) ALWAYS_INLINE;
extern Instruction const* strip_op(Thread& thread, Instruction const& inst) ALWAYS_INLINE;

Instruction const* forceDot(Thread& thread, Instruction const& inst, Value const* a, Environment* env, int64_t index) {
	if(a->isPromise()) {
		Function const& f = (Function const&)(*a);
		assert(f.environment()->DynamicScope());
		return buildStackFrame(thread, f.environment()->DynamicScope(), f.prototype(), env, index, &inst);
	} else {
		_error(std::string("Object '..") + intToStr(index+1) + "' not found, missing argument?");
	}
}

Instruction const* forceReg(Thread& thread, Instruction const& inst, Value const* a, String name) {
	if(a->isPromise()) {
		Function const& f = (Function const&)(*a);
		assert(f.environment()->DynamicScope());
        if(thread.jit.state == JIT::RECORDING)
            thread.jit.emitPush(f.environment()->DynamicScope());
        return buildStackFrame(thread, f.environment()->DynamicScope(), f.prototype(), f.environment(), name, &inst);
	} else if(a->isDefault()) {
		Function const& f = (Function const&)(*a);
		assert(f.environment());
        if(thread.jit.state == JIT::RECORDING)
            thread.jit.emitPush(f.environment());
		return buildStackFrame(thread, f.environment(), f.prototype(), f.environment(), name, &inst);
	} else {
		_error(std::string("Object '") + thread.externStr(name) + "' not found");
	}
}

// Tracing stuff

//track the heat of back edge operations and invoke the recorder on hot traces
//unused until we begin tracing loops again
static Instruction const * profile_back_edge(Thread & thread, Instruction const * inst) {
	return inst;
}

// Control flow instructions

Instruction const* call_op(Thread& thread, Instruction const& inst) {
	OPERAND(f, inst.a); FORCE(f, inst.a); BIND(f);
	if(!f.isFunction())
		_error(std::string("Non-function (") + Type::toString(f.type) + ") as first parameter to call\n");
	Function const& func = (Function const&)f;
	
	CompiledCall const& call = thread.frame.prototype->calls[inst.b];
	Environment* fenv = CreateEnvironment(thread, func.environment(), thread.frame.environment, call.call);

    if(thread.jit.state == JIT::RECORDING) {
        JIT::IRRef a = thread.jit.load(thread, inst.a, &inst);
        thread.jit.emitCall(a, func, fenv, &inst);
    }
	
	MatchArgs(thread, thread.frame.environment, fenv, func, call);
	return buildStackFrame(thread, fenv, func.prototype(), inst.c, &inst+1);
}

Instruction const* ncall_op(Thread& thread, Instruction const& inst) {
	OPERAND(f, inst.a); FORCE(f, inst.a); BIND(f);
	if(!f.isFunction())
		_error(std::string("Non-function (") + Type::toString(f.type) + ") as first parameter to call\n");
	Function const& func = (Function const&)f;
	
	CompiledCall const& call = thread.frame.prototype->calls[inst.b];
	Environment* fenv = CreateEnvironment(thread, func.environment(), thread.frame.environment, call.call);
	
	MatchNamedArgs(thread, thread.frame.environment, fenv, func, call);
	return buildStackFrame(thread, fenv, func.prototype(), inst.c, &inst+1);
}

Instruction const* ret_op(Thread& thread, Instruction const& inst) {
	// we can return futures from functions, so don't BIND
	OPERAND(result, inst.a); FORCE(result, inst.a);	
	
	// We can free this environment for reuse
	// as long as we don't return a closure...
	// TODO: but also can't if an assignment to an out of scope variable occurs (<<-, assign) with a value of a closure!
	if(result.isClosureSafe()) {
		thread.environments.push_back(thread.frame.environment);
#ifdef ENABLE_JIT
		thread.KillEnvironment(thread.frame.environment);
#endif
	}

    if(thread.jit.state == JIT::RECORDING) {
        JIT::IRRef ira = thread.jit.load(thread, inst.a, &inst);
        thread.jit.store(thread, ira, 0);
        thread.jit.insert(thread.jit.trace, TraceOpCode::POP, 0, 0, 0, Type::Promise, 1);
    }
	
	REGISTER(0) = result;
	
	thread.base = thread.frame.returnbase;
	Instruction const* returnpc = thread.frame.returnpc;

	thread.pop();
	
#ifdef ENABLE_JIT
	thread.LiveEnvironment(thread.frame.environment, result);
#endif
	return returnpc;
}

Instruction const* rets_op(Thread& thread, Instruction const& inst) {
	// top-level statements can't return futures, so bind 
	OPERAND(result, inst.a); FORCE(result, inst.a); BIND(result);	
	
	REGISTER(0) = result;
	
	thread.base = thread.frame.returnbase;
	thread.pop();
	
	// there should always be a done_op after a rets
	return &inst+1;
}

Instruction const* done_op(Thread& thread, Instruction const& inst) {
	return 0;
}

Instruction const* retp_op(Thread& thread, Instruction const& inst) {
	// we can return futures from promises, so don't BIND
	OPERAND(result, inst.a); FORCE(result, inst.a);	
    
    JIT::IRRef ira;	
	
    if(thread.frame.dest > 0) {
		thread.frame.env->insert((String)thread.frame.dest) = result;
        if(thread.jit.state == JIT::RECORDING) {
            ira = thread.jit.load(thread, inst.a, &inst);
            thread.jit.estore(ira, thread.frame.env, (String)thread.frame.dest);
            thread.jit.insert(thread.jit.trace, TraceOpCode::POP, 0, 0, 0, Type::Promise, 1);
        }
	} else {
		thread.frame.env->dots[-thread.frame.dest].v = result;
	}
#ifdef ENABLE_JIT
	thread.LiveEnvironment(thread.frame.env, result);
#endif	
	thread.base = thread.frame.returnbase;
	Instruction const* returnpc = thread.frame.returnpc;
	thread.pop();
	
	return returnpc;
}

Instruction const* constant_op(Thread& thread, Instruction const& inst) {
    OUT(thread, inst.c) = thread.frame.prototype->constants[inst.a];
    return &inst+1;
}

Instruction const* jmp_op(Thread& thread, Instruction const& inst) {
	return &inst+inst.a;
}

Instruction const* loop_op(Thread& thread, Instruction const& inst) {
    return &inst+1;
}

Instruction const* jc_op(Thread& thread, Instruction const& inst) {
	OPERAND(c, inst.c);
	if(c.isLogical1()) {
		if(Logical::isTrue(c.c)) return &inst+inst.a;
		else if(Logical::isFalse(c.c)) return &inst+inst.b;
		else _error("NA where TRUE/FALSE needed"); 
	} else if(c.isInteger1()) {
		if(Integer::isNA(c.i)) _error("NA where TRUE/FALSE needed");
		else if(c.i != 0) return &inst + inst.a;
		else return & inst+inst.b;
	} else if(c.isDouble1()) {
		if(Double::isNA(c.d)) _error("NA where TRUE/FALSE needed");
		else if(c.d != 0) return &inst + inst.a;
		else return & inst+inst.b;
	}
	FORCE(c, inst.c); BIND(c);
	_error("Need single element logical in conditional jump");
}

Instruction const* branch_op(Thread& thread, Instruction const& inst) {
	OPERAND(a, inst.a);
	int64_t index = -1;
	if(a.isDouble1()) index = (int64_t)a.d;
	else if(a.isInteger1()) index = a.i;
	else if(a.isLogical1()) index = a.i;
	else if(a.isCharacter1()) {
		for(int64_t i = 1; i <= inst.b; i++) {
			if((String)(&inst+i)->a == a.s) {
				index = i;
				break;
			}
			if(index < 0 && (String)(&inst+i)->a == Strings::empty) {
				index = i;
			}
		}
	}
	if(index >= 1 && index <= inst.b) {
		return &inst + ((&inst+index)->c);
	} 
	FORCE(a, inst.a); BIND(a);
	return &inst+1+inst.b;
}

Instruction const* forbegin_op(Thread& thread, Instruction const& inst) {
	// a = loop variable (e.g. i), b = loop vector(e.g. 1:100), c = counter register
	// following instruction is a jmp that contains offset
	OPERAND(vec, inst.b); FORCE(vec, inst.b); BIND(vec);
	if((int64_t)vec.length <= 0) {
		return &inst+(&inst+1)->a;	// offset is in following JMP, dispatch together
	} else {
		Element2(vec, 0, thread.frame.environment->insert((String)inst.a));
		Value& counter = REGISTER(inst.c);
		counter.header = vec.length;	// warning: not a valid object, but saves a shift
		counter.i = 1;
		return &inst+2;			// skip over following JMP
	}
}
Instruction const* forend_op(Thread& thread, Instruction const& inst) {
	Value& counter = REGISTER(inst.c);
	if(__builtin_expect((counter.i) < counter.header, true)) {
		OPERAND(vec, inst.b); //FORCE(vec, inst.b); BIND(vec); // this must have necessarily been forced by the forbegin.
		Element2(vec, counter.i, thread.frame.environment->insert((String)inst.a));
		counter.i++;
		return profile_back_edge(thread,&inst+(&inst+1)->a);
	} else {
		return &inst+2;			// skip over following JMP
	}
}

Instruction const* dotslist_op(Thread& thread, Instruction const& inst) {
	PairList const& dots = thread.frame.environment->dots;
	
	Value& iter = REGISTER(inst.a);
	Value& out = OUT(thread, inst.c);
	
	// First time through, make a result vector...
	if(iter.i == 0) {
		out = List(dots.size());
	}
	
	if(iter.i < (int64_t)dots.size()) {
		DOTDOT(a, iter.i); FORCE_DOTDOT(a, iter.i); 
		BIND(a); // BIND since we don't yet support futures in lists
		((List&)out)[iter.i] = a;
		iter.i++;
	}
	
	// If we're all done, check to see if we need to add names and then exit
	if(iter.i >= (int64_t)dots.size()) {
		if(thread.frame.environment->named) {
			Character names(dots.size());
			for(int64_t i = 0; i < (int64_t)dots.size(); i++)
				names[i] = dots[i].n;
			Object::Init((Object&)out, out);
			((Object&)out).insertMutable(Strings::names, names);
		}
		return &inst+1;
	}
	
	// Loop on this instruction until done.
	return &inst;
}

Instruction const* list_op(Thread& thread, Instruction const& inst) {
	List out(inst.b);
	for(int64_t i = 0; i < inst.b; i++) {
		Value& r = REGISTER(inst.a-i);
		out[i] = r;
	}
	OUT(thread, inst.c) = out;
	return &inst+1;
}

// Memory access ops

Instruction const* assign_op(Thread& thread, Instruction const& inst) {
	OPERAND(value, inst.c); FORCE(value, inst.c); // don't BIND 
	thread.frame.environment->insert((String)inst.a) = value;
	return &inst+1;
}

Instruction const* assign2_op(Thread& thread, Instruction const& inst) {
	// assign2 is always used to assign up at least one scope level...
	// so start off looking up one level...
	assert(thread.frame.environment->LexicalScope() != 0);
	
	OPERAND(value, inst.c); FORCE(value, inst.c); /*BIND(value);*/
	
	String s = (String)inst.a;
	Value& dest = thread.frame.environment->LexicalScope()->insertRecursive(s);

	if(!dest.isNil()) {
		dest = value;
		// TODO: should add dest's environment to the liveEnvironments list
	}
	else {
		thread.state.global->insert(s) = value;
#ifdef ENABLE_JIT
		thread.LiveEnvironment(thread.state.global, dest);
#endif
	}
	return &inst+1;
}

Instruction const* mov_op(Thread& thread, Instruction const& inst) {
	OPERAND(value, inst.a); FORCE(value, inst.a); BIND(value);
	OUT(thread, inst.c) = value;
	return &inst+1;
}

Instruction const* fastmov_op(Thread& thread, Instruction const& inst) {
	OPERAND(value, inst.a); FORCE(value, inst.a); // fastmov assumes we don't need to bind. So next op better be able to handle a future 
	OUT(thread, inst.c) = value;
	return &inst+1;
}

Instruction const* dotdot_op(Thread& thread, Instruction const& inst) {
	if(inst.a >= (int64_t)thread.frame.environment->dots.size())
        	_error(std::string("The '...' list does not contain ") + intToStr(inst.a+1) + " elements");
	DOTDOT(a, inst.a); FORCE_DOTDOT(a, inst.a); // no need to bind since value is in a register
	OUT(thread, inst.c) = a;
	return &inst+1;
}

Instruction const* scatter_op(Thread& thread, Instruction const& inst) {
	// a = value, b = index, c = dest 
	OPERAND(value, inst.a); FORCE(value, inst.a); 
	OPERAND(index, inst.b); FORCE(index, inst.b); 
	OPERAND(dest, inst.c); FORCE(dest, inst.c); 

	BIND(dest);
	BIND(index);
	
#ifdef ENABLE_JIT
	if(value.isFuture() && (dest.isVector() || dest.isFuture())) {
		if(index.isInteger() && index.length == 1) {
			OUT(thread, inst.c) = thread.EmitSStore(thread.frame.environment, dest, ((Integer&)index)[0], value);
			return &inst+1;
		}
		else if(index.isDouble() && index.length == 1) {
			OUT(thread, inst.c) = thread.EmitSStore(thread.frame.environment, dest, ((Double&)index)[0], value);
			return &inst+1;
		}
	}
#endif

	BIND(value);
	SubsetAssign(thread, dest, true, index, value, OUT(thread,inst.c));
	return &inst+1;
}

Instruction const* scatter1_op(Thread& thread, Instruction const& inst) {
	// a = value, b = index, c = dest
	OPERAND(value, inst.a); FORCE(value, inst.a);
	OPERAND(index, inst.b); FORCE(index, inst.b);
	OPERAND(dest, inst.c); FORCE(dest, inst.c);

	BIND(index);
	
#ifdef ENABLE_JIT
	if(value.isFuture() && (dest.isVector() || dest.isFuture())) {
		if(index.isInteger() && index.length == 1) {
			OUT(thread, inst.c) = thread.EmitSStore(thread.frame.environment, dest, ((Integer&)index)[0], value);
			return &inst+1;
		}
		else if(index.isDouble() && index.length == 1) {
			OUT(thread, inst.c) = thread.EmitSStore(thread.frame.environment, dest, ((Double&)index)[0], value);
			return &inst+1;
		}
	}
#endif
	BIND(dest);
	BIND(value);
	Subset2Assign(thread, dest, true, index, value, OUT(thread,inst.c));
	return &inst+1; 
}

Instruction const* gather_op(Thread& thread, Instruction const& inst) {
	OPERAND(a, inst.a); 
	OPERAND(i, inst.b);

	if(a.isVector()) {
		if(i.isDouble1()) { Element(a, i.d-1, OUT(thread, inst.c)); return &inst+1; }
		else if(i.isInteger1()) { Element(a, i.i-1, OUT(thread, inst.c)); return &inst+1; }
		else if(i.isLogical1()) { Element(a, Logical::isTrue(i.c) ? 0 : -1, OUT(thread, inst.c)); return &inst+1; }
		else if(i.isCharacter1()) { _error("Subscript out of bounds"); }
	}

#ifdef ENABLE_JIT
	if(isTraceable(thread, a, i) 
		&& thread.futureType(i) == Type::Logical 
		&& thread.futureShape(a) == thread.futureShape(i)) {
		OUT(thread, inst.c) = thread.EmitFilter(thread.frame.environment, a, i);
		thread.OptBind(OUT(thread, inst.c));
		return &inst+1;
	}
#endif
	FORCE(a, inst.a);
	BIND(a);

#ifdef ENABLE_JIT
	if(isTraceable(thread, a, i) 
		&& (thread.futureType(i) == Type::Integer || thread.futureType(i) == Type::Double)) {
		OUT(thread, inst.c) = thread.EmitGather(thread.frame.environment, a, i);
		thread.OptBind(OUT(thread, inst.c));
		return &inst+1;
	}
#endif
	if(a.isObject()) { 
		return GenericDispatch(thread, inst, Strings::bracket, a, i, inst.c); 
	} 
	
	FORCE(i, inst.b); 
	BIND(i);

	if(i.isObject()) { 
		return GenericDispatch(thread, inst, Strings::bracket, a, i, inst.c); 
	} 
	
	SubsetSlow(thread, a, i, OUT(thread, inst.c)); 
	return &inst+1;
}

Instruction const* gather1_op(Thread& thread, Instruction const& inst) {
	OPERAND(a, inst.a); FORCE(a, inst.a); BIND(a);
	OPERAND(i, inst.b);
	if(a.isVector()) {
		int64_t index = 0;
		if(i.isDouble1()) { index = i.d-1; }
		else if(i.isInteger1()) { index = i.i-1; }
		else if(i.isLogical1() && Logical::isTrue(i.c)) { index = 1-1; }
		else if(i.isVector() && (i.length == 0 || i.length > 1)) { 
			_error("Attempt to select less or more than 1 element in subset2"); 
		}
		else { _error("Subscript out of bounds"); }
		Element2(a, index, OUT(thread, inst.c));
		return &inst+1;
	}
 	FORCE(i, inst.b); BIND(i);
	if(a.isObject() || i.isObject()) { return GenericDispatch(thread, inst, Strings::bb, a, i, inst.c); } 
	_error("Invalid subset2 operation");
}

#ifdef ENABLE_JIT
#define OP(Name, string, Group, Func) \
Instruction const* Name##_op(Thread& thread, Instruction const& inst) { \
	OPERAND(a, inst.a);	\
	Value & c = OUT(thread, inst.c);	\
	if(a.isDouble1())  { Name##VOp<Double>::Scalar(thread, a.d, c); return &inst+1; } \
	if(a.isInteger1()) { Name##VOp<Integer>::Scalar(thread, a.i, c); return &inst+1; } \
	if(a.isLogical1()) { Name##VOp<Logical>::Scalar(thread, a.c, c); return &inst+1; } \
	FORCE(a, inst.a); \
	if(isTraceable<Group>(thread,a)) { \
		c = thread.EmitUnary<Group>(thread.frame.environment, IROpCode::Name, a, 0); \
		thread.OptBind(c); \
 		return &inst+1; \
	} \
	BIND(a); \
	if(a.isObject()) { return GenericDispatch(thread, inst, Strings::Name, a, inst.c); } \
\
	Group##Dispatch<Name##VOp>(thread, a, c); \
	return &inst+1; \
}
#else
#define OP(Name, string, Group, Func) \
Instruction const* Name##_op(Thread& thread, Instruction const& inst) { \
	OPERAND(a, inst.a);	\
	Value & c = OUT(thread, inst.c);	\
	if(a.isDouble1())  { Name##VOp<Double>::Scalar(thread, a.d, c); return &inst+1; } \
	if(a.isInteger1()) { Name##VOp<Integer>::Scalar(thread, a.i, c); return &inst+1; } \
	if(a.isLogical1()) { Name##VOp<Logical>::Scalar(thread, a.c, c); return &inst+1; } \
	FORCE(a, inst.a); \
	BIND(a); \
	if(a.isObject()) { return GenericDispatch(thread, inst, Strings::Name, a, inst.c); } \
\
	Group##Dispatch<Name##VOp>(thread, a, c); \
	return &inst+1; \
}
#endif
UNARY_FOLD_SCAN_BYTECODES(OP)
#undef OP

#ifdef ENABLE_JIT
#define OP(Name, string, Group, Func) \
Instruction const* Name##_op(Thread& thread, Instruction const& inst) { \
	OPERAND(a, inst.a);	\
	OPERAND(b, inst.b);	\
	Value & c = OUT(thread, inst.c);	\
        if(a.isDouble1()) {			\
		if(b.isDouble1()) { Name##VOp<Double,Double>::Scalar(thread, a.d, b.d, c); return &inst+1; } \
		if(b.isInteger1()) { Name##VOp<Double,Integer>::Scalar(thread, a.d, b.i, c); return &inst+1; } \
		if(b.isLogical1()) { Name##VOp<Double,Logical>::Scalar(thread, a.d, b.c, c); return &inst+1; } \
        }	\
        else if(a.isInteger1()) {	\
		if(b.isDouble1()) { Name##VOp<Integer,Double>::Scalar(thread, a.i, b.d, c); return &inst+1; } \
		if(b.isInteger1()) { Name##VOp<Integer,Integer>::Scalar(thread, a.i, b.i, c); return &inst+1; } \
		if(b.isLogical1()) { Name##VOp<Integer,Logical>::Scalar(thread, a.i, b.c, c); return &inst+1; } \
        } \
        else if(a.isLogical1()) {	\
		if(b.isDouble1()) { Name##VOp<Logical,Double>::Scalar(thread, a.c, b.d, c); return &inst+1; } \
		if(b.isInteger1()) { Name##VOp<Logical,Integer>::Scalar(thread, a.c, b.i, c); return &inst+1; } \
		if(b.isLogical1()) { Name##VOp<Logical,Logical>::Scalar(thread, a.c, b.c, c); return &inst+1; } \
        } \
	FORCE(a, inst.a); FORCE(b, inst.b); \
	if(isTraceable<Group>(thread,a,b)) { \
		c = thread.EmitBinary<Group>(thread.frame.environment, IROpCode::Name, a, b, 0); \
		thread.OptBind(c); \
		return &inst+1; \
	} \
	BIND(a); BIND(b); \
	if(a.isObject() || b.isObject()) { return GenericDispatch(thread, inst, Strings::Name, a, b, inst.c); } \
\
	Group##Dispatch<Name##VOp>(thread, a, b, c);	\
	return &inst+1;	\
}
#else
#define OP(Name, string, Group, Func) \
Instruction const* Name##_op(Thread& thread, Instruction const& inst) { \
	OPERAND(a, inst.a);	\
	OPERAND(b, inst.b);	\
	Value & c = OUT(thread, inst.c);	\
        if(a.isDouble1()) {			\
		if(b.isDouble1()) { Name##VOp<Double,Double>::Scalar(thread, a.d, b.d, c); return &inst+1; } \
		if(b.isInteger1()) { Name##VOp<Double,Integer>::Scalar(thread, a.d, b.i, c); return &inst+1; } \
		if(b.isLogical1()) { Name##VOp<Double,Logical>::Scalar(thread, a.d, b.c, c); return &inst+1; } \
        }	\
        else if(a.isInteger1()) {	\
		if(b.isDouble1()) { Name##VOp<Integer,Double>::Scalar(thread, a.i, b.d, c); return &inst+1; } \
		if(b.isInteger1()) { Name##VOp<Integer,Integer>::Scalar(thread, a.i, b.i, c); return &inst+1; } \
		if(b.isLogical1()) { Name##VOp<Integer,Logical>::Scalar(thread, a.i, b.c, c); return &inst+1; } \
        } \
        else if(a.isLogical1()) {	\
		if(b.isDouble1()) { Name##VOp<Logical,Double>::Scalar(thread, a.c, b.d, c); return &inst+1; } \
		if(b.isInteger1()) { Name##VOp<Logical,Integer>::Scalar(thread, a.c, b.i, c); return &inst+1; } \
		if(b.isLogical1()) { Name##VOp<Logical,Logical>::Scalar(thread, a.c, b.c, c); return &inst+1; } \
        } \
	FORCE(a, inst.a); FORCE(b, inst.b); \
	BIND(a); BIND(b); \
	if(a.isObject() || b.isObject()) { return GenericDispatch(thread, inst, Strings::Name, a, b, inst.c); } \
\
	Group##Dispatch<Name##VOp>(thread, a, b, c);	\
	return &inst+1;	\
}
#endif
BINARY_BYTECODES(OP)
#undef OP

#ifdef TRACE_DEVELOPMENT
Instruction const* length_op(Thread& thread, Instruction const& inst) {
	OPERAND(a, inst.a); FORCE(a, inst.a); 
	if(a.isVector())
		Integer::InitScalar(OUT(thread, inst.c), a.length);
	else if(a.isFuture()) {
		IRNode::Shape shape = thread.futureShape(a);
		if(shape.split < 0 && shape.filter < 0) {
			Integer::InitScalar(OUT(thread, inst.c), shape.length);
		} else {
			OUT(thread, inst.c) = thread.EmitUnary<CountFold>(thread.frame.environment, IROpCode::length, a, 0);
			thread.OptBind(OUT(thread,inst.c));
		}
	}
	else if(a.isObject()) { 
		return GenericDispatch(thread, inst, Strings::length, a, inst.c); 
	} else {
		Integer::InitScalar(OUT(thread, inst.c), 1);
	}
	return &inst+1;
}

Instruction const* mean_op(Thread& thread, Instruction const& inst) {
	OPERAND(a, inst.a); FORCE(a, inst.a); 
	if(isTraceable<MomentFold>(thread,a)) {
		OUT(thread, inst.c) = thread.EmitUnary<MomentFold>(thread.frame.environment, IROpCode::mean, a, 0);
		thread.OptBind(OUT(thread,inst.c));
 		return &inst+1;
	}
	return &inst+1;
}

Instruction const* cm2_op(Thread& thread, Instruction const& inst) {
	OPERAND(a, inst.a); FORCE(a, inst.a); 
	OPERAND(b, inst.b); FORCE(b, inst.b); 
	if(isTraceable<Moment2Fold>(thread,a,b)) {
		OUT(thread, inst.c) = thread.EmitBinary<Moment2Fold>(thread.frame.environment, IROpCode::cm2, a, b, 0);
		thread.OptBind(OUT(thread,inst.c));
 		return &inst+1;
	}
	return &inst+1;
}

Instruction const* ifelse_op(Thread& thread, Instruction const& inst) {
	OPERAND(a, inst.a); FORCE(a, inst.a);
	OPERAND(b, inst.b); FORCE(b, inst.b);
	OPERAND(c, inst.c); FORCE(c, inst.c);
	if(c.isLogical1()) {
		OUT(thread, inst.c) = Logical::isTrue(c.c) ? b : a;
		return &inst+1; 
	}
	else if(c.isInteger1()) {
		OUT(thread, inst.c) = c.i ? b : a;
		return &inst+1; 
	}
	else if(c.isDouble1()) {
		OUT(thread, inst.c) = c.d ? b : a;
		return &inst+1; 
	}
	if(isTraceable<IfElse>(thread,a,b,c)) {
		OUT(thread, inst.c) = thread.EmitIfElse(thread.frame.environment, a, b, c);
		thread.OptBind(OUT(thread,inst.c));
		return &inst+1;
	}
	BIND(a); BIND(b); BIND(c);

	_error("ifelse not defined in scalar yet");
	return &inst+1; 
}

Instruction const* split_op(Thread& thread, Instruction const& inst) {
	OPERAND(a, inst.a); FORCE(a, inst.a); BIND(a);
	int64_t levels = As<Integer>(thread, a)[0];
	OPERAND(b, inst.b); FORCE(b, inst.b);
	OPERAND(c, inst.c); FORCE(c, inst.c);
	if(isTraceable<Split>(thread,b,c)) {
		OUT(thread, inst.c) = thread.EmitSplit(thread.frame.environment, c, b, levels);
		thread.OptBind(OUT(thread,inst.c));
		return &inst+1;
	}
	BIND(a); BIND(b); BIND(c);

	_error("split not defined in scalar yet");
	return &inst+1; 
}
#endif

Instruction const* function_op(Thread& thread, Instruction const& inst) {
	OPERAND(function, inst.a); FORCE(function, inst.a);
	Value& out = OUT(thread, inst.c);
	out.header = function.header;
	out.p = (void*)thread.frame.environment;
	return &inst+1;
}

#ifdef TRACE_DEVELOPMENT
Instruction const* vector_op(Thread& thread, Instruction const& inst) {
	OPERAND(a, inst.a); FORCE(a, inst.a); BIND(a);
	OPERAND(b, inst.b); FORCE(b, inst.b); BIND(b);
	Type::Enum type = string2Type( As<Character>(thread, a)[0] );
	int64_t l = As<Integer>(thread, b)[0];
	
	// TODO: replace with isTraceable...
	if(thread.state.jitEnabled 
		&& (type == Type::Double || type == Type::Integer || type == Type::Logical)
		&& l >= TRACE_VECTOR_WIDTH) {
		OUT(thread, inst.c) = thread.EmitConstant(thread.frame.environment, type, l, 0);
		thread.OptBind(OUT(thread,inst.c));
		return &inst+1;
	}

	if(type == Type::Logical) {
		Logical v(l);
		for(int64_t i = 0; i < v.length; i++) v[i] = Logical::FalseElement;
		OUT(thread, inst.c) = v;
	} else if(type == Type::Integer) {
		Integer v(l);
		for(int64_t i = 0; i < v.length; i++) v[i] = 0;
		OUT(thread, inst.c) = v;
	} else if(type == Type::Double) {
		Double v(l);
		for(int64_t i = 0; i < v.length; i++) v[i] = 0;
		OUT(thread, inst.c) = v;
	} else if(type == Type::Character) {
		Character v(l);
		for(int64_t i = 0; i < v.length; i++) v[i] = Strings::empty;
		OUT(thread, inst.c) = v;
	} else if(type == Type::Raw) {
		Raw v(l);
		for(int64_t i = 0; i < v.length; i++) v[i] = 0;
		OUT(thread, inst.c) = v;
	} else {
		_error("Invalid type in vector");
	} 
	return &inst+1;
}
#endif
Instruction const* seq_op(Thread& thread, Instruction const& inst) {
	// c = start, b = step, a = length
	OPERAND(a, inst.a); FORCE(a, inst.a); BIND(a);
	OPERAND(b, inst.b); FORCE(b, inst.b); BIND(b);
	OPERAND(c, inst.c); FORCE(c, inst.c); BIND(c);

	double start = As<Double>(thread, c)[0];
	double step = As<Double>(thread, b)[0];
	int64_t len = As<Integer>(thread, a)[0];

#ifdef ENABLE_JIT	
	if(len >= TRACE_VECTOR_WIDTH) {
		if(b.isDouble() || c.isDouble()) {
			OUT(thread, inst.c) = thread.EmitSequence(thread.frame.environment, len, start, step);
			thread.OptBind(OUT(thread,inst.c));
		} else {
			OUT(thread, inst.c) = thread.EmitSequence(thread.frame.environment, len, (int64_t)start, (int64_t)step);
			thread.OptBind(OUT(thread,inst.c));
		}
		return &inst+1;
	}
#endif
	if(b.isDouble() || c.isDouble())	
		OUT(thread, inst.c) = Sequence(start, step, len);
	else
		OUT(thread, inst.c) = Sequence((int64_t)start, (int64_t)step, len);
	return &inst+1;
}

#ifdef TRACE_DEVELOPMENT
Instruction const* rep_op(Thread& thread, Instruction const& inst) {
	// c = n, b = each, a = length
	OPERAND(a, inst.a); FORCE(a, inst.a); BIND(a);
	OPERAND(b, inst.b); FORCE(b, inst.b); BIND(b);
	OPERAND(c, inst.c); FORCE(c, inst.c); BIND(c);

	int64_t n = As<Integer>(thread, c)[0];
	int64_t each = As<Integer>(thread, b)[0];
	int64_t len = As<Integer>(thread, a)[0];
	
	if(len >= TRACE_VECTOR_WIDTH) {
		OUT(thread, inst.c) = thread.EmitRepeat(thread.frame.environment, len, (int64_t)n, (int64_t)each);
		thread.OptBind(OUT(thread,inst.c));
		return &inst+1;
	}

	OUT(thread, inst.c) = Repeat((int64_t)n, (int64_t)each, len);
	return &inst+1;
}

Instruction const* random_op(Thread& thread, Instruction const& inst) {
	OPERAND(a, inst.a); FORCE(a, inst.a); BIND(a);

	int64_t len = As<Integer>(thread, a)[0];
	
	/*if(len >= TRACE_VECTOR_WIDTH) {
		OUT(thread, inst.c) = thread.EmitRandom(thread.frame.environment, len);
		thread.OptBind(OUT(thread,inst.c));
		return &inst+1;
	}*/

	OUT(thread, inst.c) = Random(thread, len);
	return &inst+1;
}

Instruction const* type_op(Thread& thread, Instruction const& inst) {
	OPERAND(a, inst.a); FORCE(a, inst.a);
	switch(thread.futureType(a)) {
                #define CASE(name, str) case Type::name: OUT(thread, inst.c) = Character::c(Strings::name); break;
                TYPES(CASE)
                #undef CASE
                default: _error("Unknown type in type to string, that's bad!"); break;
        }
	return &inst+1;
}

Instruction const* missing_op(Thread& thread, Instruction const& inst) {
	// TODO: in R this is recursive. If this function was passed a parameter that
	// was missing in the outer scope, then it should be missing here too. But
	// missingness doesn't propogate through expressions, leading to strange behavior:
	// 	f <- function(x,y) g(x,y)
	//	g <- function(x,y) missing(y)
	//	f(1) => TRUE
	// but
	//	f <- function(x,y) g(x,y+1)
	//	g <- function(x,y) missing(y)
	//	f(1) => FALSE
	// but
	//	f <- function(x,y) g(x,y+1)
	//	g <- function(x,y) y
	//	f(1) => Error in y+1: 'y' is missing
	// For now I'll keep the simpler non-recursive semantics. Missing solely means
	// whether or not this scope was passed a value, irregardless of whether that
	// value is missing at a higher level.
	String s = (String)inst.a;
	Value const& v = thread.frame.environment->get(s);
	bool missing = v.isNil() || v.isDefault();
	Logical::InitScalar(OUT(thread, inst.c), missing ? Logical::TrueElement : Logical::FalseElement);
	return &inst+1;
}
Instruction const* strip_op(Thread& thread, Instruction const& inst) {
	OPERAND(a, inst.a); FORCE(a, inst.a);
	Value& c = OUT(thread, inst.c);
	if(a.isObject())
		c = ((Object const&)a).base();
	else
		c = a;
	return &inst+1;
}

Instruction const* internal_op(Thread& thread, Instruction const& inst) {
	if(inst.a < 0)
		_error("Attempting to use undefined internal function");
	int64_t nargs = thread.state.internalFunctions[inst.a].params;
	for(int64_t i = 0; i < nargs; i++) {
		BIND(REGISTER(inst.b-i));
	}
	thread.state.internalFunctions[inst.a].ptr(thread, &REGISTER(inst.b), OUT(thread, inst.c));
	return &inst+1;
}
#endif

/*struct _Load {
    int64_t i;
    _Load(int64_t i) : i(i) {}
    Value const& eval(Thread& thread, Instruction const*& bail) {
        OPERAND(r, i);
        return r;
    }
};

template<class T>
struct _Force {
    T t;
    _Force(T t) : t(t) {}
    Value const& eval(Thread& thread) {
        Value const& 
        FORCE(a, i);
    }
};

_Load Load(int64_t i) { return _Load(i); }

template<class T>
struct _Store {
    T t;
    int64_t i;
    _Store(T t, int64_t i) : t(t), i(i) {}
    Value const& eval(Thread& thread, Instruction const*& bail) {
        t.eval(thread, bail);
        if(bail != 0) return;
        t.eval(thread, r);
        Value & out = OUT(thread, i);
        out = t.eval(thread);
        return out;
    }
};

template<class T>
_Store<T> Store(T t, int64_t i) { return _Store<T>(t, i); }

Instruction const* mov_op(Thread& thread, Instruction const& inst) {
    auto op = Store(Load(inst.a), inst.c);
    op.eval(thread);
    return &inst+1;
}*/

//
//    Main interpreter loop 
//
//__attribute__((__noinline__,__noclone__)) 
void interpret(Thread& thread, Instruction const* pc) {

#ifdef USE_THREADED_INTERPRETER
    	#define LABELS_THREADED(name,type,...) (void*)&&name##_label,
	static const void* ops[] = {BYTECODES(LABELS_THREADED)};
	#undef LABELS_THREADED
    	#define RECORD_THREADED(name,type,...) (void*)&&name##_record,
	static const void* record[] = {BYTECODES(RECORD_THREADED)};
	#undef RECORD_THREADED

	const void** labels = ops;

	goto *(const void*)(labels[pc->bc]);
	
    #define LABELED_OP(name,type,...) \
		name##_label: \
			{ pc = name##_op(thread, *pc); goto *(const void*)labels[pc->bc]; } 
	STANDARD_BYTECODES(LABELED_OP)
	#undef LABELED_OP

	loop_label: 	{
        if(pc->a != 0) {
	        timespec a = get_time();
            pc = ((JIT::Ptr)pc->a)(thread);
	        printf("Execution time: %f\n", time_elapsed(a));
        }
        else { 
    	    unsigned short& counter = 
	    		thread.jit.counters[(((uintptr_t)pc)>>5) & (1024-1)];
    		counter++;
	    	if(counter > JIT::RECORD_TRIGGER) {
                printf("Starting to record at %li (counter: %li is %d)\n", pc, &counter, counter);
    			counter = 0;
	    		thread.jit.start_recording(pc);
		    	labels = record;
    		}
            pc++;
        }
		goto *(const void*)labels[pc->bc]; 
	}
    jc_label: {
        pc = jc_op(thread, *pc);
        goto *(const void*)labels[pc->bc]; 
    }
	jmp_label: 	{ 
        pc = jmp_op(thread, *pc); 
        goto *(const void*)labels[pc->bc]; 
    }
	call_label: 	{ 
        pc = call_op(thread, *pc); 
        goto *(const void*)labels[pc->bc]; 
    }
	ret_label: 	{ 
        pc = ret_op(thread, *pc); 
        goto *(const void*)labels[pc->bc]; 
    }
	retp_label: 	{ 
        pc = retp_op(thread, *pc); 
        goto *(const void*)labels[pc->bc]; 
    }
	rets_label: 	{ 
        pc = rets_op(thread, *pc); 
        goto *(const void*)labels[pc->bc]; 
    }
	

	
    #define RECORD_OP(name,...) \
		name##_record: { \
            Instruction const* old_pc = pc; \
            thread.jit.record(thread, old_pc); \
            pc = name##_op(thread, *pc); \
            goto *(const void*)labels[pc->bc]; \
    } 
	STANDARD_BYTECODES(RECORD_OP)
	#undef RECORD_OP
	
	loop_record: 	{ 
        // did we make a loop yet??
        if(pc->a != 0) {
            thread.jit.fail_recording();
		    labels = ops;
        }
        else if(thread.jit.loop(thread, pc)) {
            printf("Made loop at %li\n", pc);
		    ((Instruction*)pc)->a = (int64_t)thread.jit.end_recording(thread);
		    labels = ops;
        }
        else {
            pc++;
        }
		goto *(const void*)labels[pc->bc]; 
	}
    jc_record:  {
        Instruction const* old_pc = pc;
        pc = jc_op(thread, *pc); 
        thread.jit.record(thread, old_pc, pc==(old_pc+old_pc->a));
        goto *(const void*)labels[pc->bc]; 
    }
	jmp_record: 	{ 
        pc = jmp_op(thread, *pc); 
        goto *(const void*)labels[pc->bc]; 
    }
	call_record: 	{ 
        Instruction const* old_pc = pc;
        pc = call_op(thread, *pc);
        thread.jit.record(thread, old_pc);
        goto *(const void*)labels[pc->bc]; 
    }
	ret_record: 	{ 
        Instruction const* old_pc = pc;
        pc = ret_op(thread, *pc);
        goto *(const void*)labels[pc->bc]; 
    }
	retp_record: 	{ 
        Instruction const* old_pc = pc;
        pc = retp_op(thread, *pc);
        goto *(const void*)labels[pc->bc]; 
    }
	rets_record: 	{
		pc = rets_op(thread, *pc);
		// terminate recording
		labels = ops;
		goto *(const void*)labels[pc->bc]; 
	}
	
	done_label: {}
	done_record: {}
#else
	while(pc->bc != ByteCode::done) {
		switch(pc->bc) {
			#define SWITCH_OP(name,type,...) \
				case ByteCode::name: { pc = name##_op(thread, *pc); } break;
			BYTECODES(SWITCH_OP)
		};
	}
#endif

}

// ensure glabels is inited before we need it.
void State::interpreter_init(Thread& thread) {
#ifdef USE_THREADED_INTERPRETER
#endif
}

Value Thread::eval(Prototype const* prototype) {
	return eval(prototype, frame.environment);
}

Value Thread::eval(Prototype const* prototype, Environment* environment) {
	Value* old_base = base;
	int64_t stackSize = stack.size();

	//printCode(*this, prototype, environment);

	// make room for the result
	base--;	
	Instruction const* run = buildStackFrame(*this, environment, prototype, 0, (Instruction const*)0);
	try {
		interpret(*this, run);
		base++;
		assert(base == old_base);
		assert(stackSize == stack.size());
		return *(base-1);
	} catch(...) {
		base = old_base;
		stack.resize(stackSize);
		throw;
	}
}

