#include <string>
#include <sstream>
#include <stdexcept>
#include <string>

#include "value.h"
#include "type.h"
#include "bc.h"
#include "ops.h"
#include "internal.h"
#include "interpreter.h"
#include "compiler.h"
#include "sse.h"
#include "call.h"

#ifdef USE_THREADED_INTERPRETER
const void** glabels = 0;
#endif

const int64_t Random::primes[100] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43,
47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131,
137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223,
227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311,
313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409,
419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503,
509, 521, 523, 541};

Thread::Thread(State& state, uint64_t index) : state(state), index(index), random(index),steals(1) {
	registers = new Value[DEFAULT_NUM_REGISTERS];
	this->base = registers + DEFAULT_NUM_REGISTERS;
}

void Prototype::threadByteCode(Prototype*  prototype) {
#ifdef USE_THREADED_INTERPRETER
	for(int64_t i = 0; i < (int64_t)prototype->bc.size(); ++i) {
		Instruction const& inst = prototype->bc[i];
		inst.ibc = glabels[inst.bc];
	}
#endif
}

/*
extern Instruction const* mov_op(Thread& thread, Instruction const& inst) ALWAYS_INLINE;
extern Instruction const* fastmov_op(Thread& thread, Instruction const& inst) ALWAYS_INLINE;
extern Instruction const* assign_op(Thread& thread, Instruction const& inst) ALWAYS_INLINE;
extern Instruction const* forend_op(Thread& thread, Instruction const& inst) ALWAYS_INLINE;
extern Instruction const* add_op(Thread& thread, Instruction const& inst) ALWAYS_INLINE;
extern Instruction const* subset_op(Thread& thread, Instruction const& inst) ALWAYS_INLINE;
extern Instruction const* subset2_op(Thread& thread, Instruction const& inst) ALWAYS_INLINE;
extern Instruction const* jc_op(Thread& thread, Instruction const& inst) ALWAYS_INLINE;
extern Instruction const* lt_op(Thread& thread, Instruction const& inst) ALWAYS_INLINE;
extern Instruction const* ret_op(Thread& thread, Instruction const& inst) ALWAYS_INLINE;
extern Instruction const* retp_op(Thread& thread, Instruction const& inst) ALWAYS_INLINE;
extern Instruction const* internal_op(Thread& thread, Instruction const& inst) ALWAYS_INLINE;
extern Instruction const* strip_op(Thread& thread, Instruction const& inst) ALWAYS_INLINE;
*/

Instruction const* forceDot(Thread& thread, Instruction const& inst, Value const* a, Environment* env, int64_t index) {
	if(a->isPromise()) {
		Promise const& f = (Promise const&)(*a);
		assert(f.environment()->DynamicScope());
		return buildStackFrame(thread, f.environment()->DynamicScope(), f.prototype(), env, index, &inst);
	} else {
		_error(std::string("Object '..") + intToStr(index+1) + "' not found, missing argument?");
	}
}

Instruction const* forceReg(Thread& thread, Instruction const& inst, Value const* a, Environment* dest, String name) {
	if(a->isPromise()) {
		Promise const& f = (Promise const&)(*a);
		return buildStackFrame(thread, f.environment(), f.prototype(), dest, name, &inst);
	} else if(a->isDefault()) {
		Default const& f = (Default const&)(*a);
		assert(f.environment());
		return buildStackFrame(thread, f.environment(), f.prototype(), dest, name, &inst);
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
	Heap::Global.collect(thread.state);
	OPERAND(f, inst.a); FORCE(f, inst.a); BIND(f);
	if(!f.isFunction())
		_error(std::string("Non-function (") + Type::toString(f.type()) + ") as first parameter to call\n");
	Function const& func = (Function const&)f;
	
	CompiledCall const& call = thread.frame.prototype->calls[inst.b];
	Environment* fenv = CreateEnvironment(thread, func.environment(), thread.frame.environment, call.call);
	
	MatchArgs(thread, thread.frame.environment, fenv, func, call);
	return buildStackFrame(thread, fenv, func.prototype(), inst.c, &inst+1);
}

Instruction const* ncall_op(Thread& thread, Instruction const& inst) {
	Heap::Global.collect(thread.state);
	OPERAND(f, inst.a); FORCE(f, inst.a); BIND(f);
	if(!f.isFunction())
		_error(std::string("Non-function (") + Type::toString(f.type()) + ") as first parameter to call\n");
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
		thread.traces.KillEnvironment(thread.frame.environment);
	}

	REGISTER(0) = result;
	
	thread.base = thread.frame.returnbase;
	Instruction const* returnpc = thread.frame.returnpc;
	
	thread.pop();
	
	thread.traces.LiveEnvironment(thread.frame.environment, result);

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
	
	if(thread.frame.dest > 0) {
		thread.frame.env->insert((String)thread.frame.dest) = result;
	} else {
		thread.frame.env->dots[-thread.frame.dest].v = result;
	}
	thread.traces.LiveEnvironment(thread.frame.env, result);
	
	thread.base = thread.frame.returnbase;
	Instruction const* returnpc = thread.frame.returnpc;
	thread.pop();
	
	return returnpc;
}

Instruction const* jmp_op(Thread& thread, Instruction const& inst) {
	return &inst+inst.a;
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
	if(!vec.isVector())
		_error("Invalid for() loop sequence");
	Vector const& v = (Vector const&)vec;
	if((int64_t)v.length() <= 0) {
		return &inst+(&inst+1)->a;	// offset is in following JMP, dispatch together
	} else {
		Element2(v, 0, thread.frame.environment->insert((String)inst.a));
		Integer::InitScalar(REGISTER(inst.c), 1);
		Integer::InitScalar(REGISTER(inst.c-1), v.length());
		return &inst+2;			// skip over following JMP
	}
}
Instruction const* forend_op(Thread& thread, Instruction const& inst) {
	Value& counter = REGISTER(inst.c);
	Value& limit = REGISTER(inst.c-1);
	if(__builtin_expect(counter.i < limit.i, true)) {
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
		Heap::Global.collect(thread.state);
		out = List(dots.size());
		memset(((List&)out).v(), 0, dots.size()*sizeof(List::Element));
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
			Dictionary* d = new Dictionary(1);
			d->insert(Strings::names) = names;
			out.z((uint64_t)d);
		}
		return &inst+1;
	}
	
	// Loop on this instruction until done.
	return &inst;
}

Instruction const* list_op(Thread& thread, Instruction const& inst) {
	Heap::Global.collect(thread.state);
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
	Environment* penv;
	Value& dest = thread.frame.environment->LexicalScope()->insertRecursive(s, penv);

	if(!dest.isNil()) {
		dest = value;
		thread.traces.LiveEnvironment(penv, dest);
	}
	else {
		thread.state.global->insert(s) = value;
		thread.traces.LiveEnvironment(thread.state.global, dest);
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

Instruction const* iassign_op(Thread& thread, Instruction const& inst) {
	// a = value, b = index, c = dest 
	OPERAND(value, inst.a); FORCE(value, inst.a); 
	OPERAND(index, inst.b); FORCE(index, inst.b); 
	OPERAND(dest, inst.c); FORCE(dest, inst.c); 

	BIND(dest);
	BIND(index);
	
	if(value.isFuture() && (dest.isVector() || dest.isFuture())) {
		if(index.isInteger() && ((Integer const&)index).length() == 1) {
			OUT(thread, inst.c) = thread.traces.EmitSStore(thread.frame.environment, dest, ((Integer&)index)[0], value);
			return &inst+1;
		}
		else if(index.isDouble() && ((Double const&)index).length() == 1) {
			OUT(thread, inst.c) = thread.traces.EmitSStore(thread.frame.environment, dest, ((Double&)index)[0], value);
			return &inst+1;
		}
	}

	BIND(value);
	SubsetAssign(thread, dest, true, index, value, OUT(thread,inst.c));
	return &inst+1;
}
Instruction const* eassign_op(Thread& thread, Instruction const& inst) {
	// a = value, b = index, c = dest
	OPERAND(value, inst.a); FORCE(value, inst.a);
	OPERAND(index, inst.b); FORCE(index, inst.b);
	OPERAND(dest, inst.c); FORCE(dest, inst.c);

	BIND(index);
	
	if(value.isFuture() && (dest.isVector() || dest.isFuture())) {
		if(index.isInteger() && ((Integer const&)index).length() == 1) {
			OUT(thread, inst.c) = thread.traces.EmitSStore(thread.frame.environment, dest, ((Integer&)index)[0], value);
			return &inst+1;
		}
		else if(index.isDouble() && ((Double const&)index).length() == 1) {
			OUT(thread, inst.c) = thread.traces.EmitSStore(thread.frame.environment, dest, ((Double&)index)[0], value);
			return &inst+1;
		}
	}

	BIND(dest);
	BIND(value);
	Subset2Assign(thread, dest, true, index, value, OUT(thread,inst.c));
	return &inst+1; 
}

Instruction const* subset_op(Thread& thread, Instruction const& inst) {
	OPERAND(a, inst.a); 
	OPERAND(i, inst.b);

	if(a.isVector()) {
		if(i.isDouble1()) { Element(a, i.d-1, OUT(thread, inst.c)); return &inst+1; }
		else if(i.isInteger1()) { Element(a, i.i-1, OUT(thread, inst.c)); return &inst+1; }
		else if(i.isLogical1()) { Element(a, Logical::isTrue(i.c) ? 0 : -1, OUT(thread, inst.c)); return &inst+1; }
		else if(i.isCharacter1()) { _error("Subscript out of bounds"); }
	}

	if(thread.state.jitEnabled 
		&& thread.traces.isTraceable(a, i) 
		&& thread.traces.futureType(i) == Type::Logical 
		&& thread.traces.futureShape(a) == thread.traces.futureShape(i)) {
		OUT(thread, inst.c) = thread.traces.EmitFilter(thread.frame.environment, a, i);
		thread.traces.OptBind(thread, OUT(thread, inst.c));
		return &inst+1;
	}

	FORCE(a, inst.a);
	BIND(a);

	if(thread.traces.isTraceable(a, i) 
		&& (thread.traces.futureType(i) == Type::Integer 
			|| thread.traces.futureType(i) == Type::Double)) {
		OUT(thread, inst.c) = thread.traces.EmitGather(thread.frame.environment, a, i);
		thread.traces.OptBind(thread, OUT(thread, inst.c));
		return &inst+1;
	}

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

Instruction const* subset2_op(Thread& thread, Instruction const& inst) {
	OPERAND(a, inst.a); FORCE(a, inst.a); BIND(a);
	OPERAND(i, inst.b);
	if(a.isVector()) {
		int64_t index = 0;
		if(i.isDouble1()) { index = i.d-1; }
		else if(i.isInteger1()) { index = i.i-1; }
		else if(i.isLogical1() && Logical::isTrue(i.c)) { index = 1-1; }
		else if(i.isVector() && (((Vector const&)i).length() == 0 || ((Vector const&)i).length() > 1)) { 
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


#define OP(Name, string, Group, Func) \
Instruction const* Name##_op(Thread& thread, Instruction const& inst) { \
	OPERAND(a, inst.a);	\
	Value & c = OUT(thread, inst.c);	\
	if(a.isDouble1())  { Name##VOp<Double>::Scalar(thread, a.d, c); return &inst+1; } \
	if(a.isInteger1()) { Name##VOp<Integer>::Scalar(thread, a.i, c); return &inst+1; } \
	if(a.isLogical1()) { Name##VOp<Logical>::Scalar(thread, a.c, c); return &inst+1; } \
	FORCE(a, inst.a); \
	if(thread.traces.isTraceable<Group>(a)) { \
		c = thread.traces.EmitUnary<Group>(thread.frame.environment, IROpCode::Name, a, 0); \
		thread.traces.OptBind(thread, c); \
 		return &inst+1; \
	} \
	BIND(a); \
	if(a.isObject()) { return GenericDispatch(thread, inst, Strings::Name, a, inst.c); } \
\
	Group##Dispatch<Name##VOp>(thread, a, c); \
	return &inst+1; \
}
UNARY_FOLD_SCAN_BYTECODES(OP)
#undef OP


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
	if(thread.traces.isTraceable<Group>(a,b)) { \
		c = thread.traces.EmitBinary<Group>(thread.frame.environment, IROpCode::Name, a, b, 0); \
		thread.traces.OptBind(thread, c); \
		return &inst+1; \
	} \
	BIND(a); BIND(b); \
	if(a.isObject() || b.isObject()) { return GenericDispatch(thread, inst, Strings::Name, a, b, inst.c); } \
\
	Group##Dispatch<Name##VOp>(thread, a, b, c);	\
	return &inst+1;	\
}
BINARY_BYTECODES(OP)
#undef OP

Instruction const* length_op(Thread& thread, Instruction const& inst) {
	OPERAND(a, inst.a); FORCE(a, inst.a); 
	if(a.isVector())
		Integer::InitScalar(OUT(thread, inst.c), ((Vector const&)a).length());
	else if(a.isFuture()) {
		IRNode::Shape shape = thread.traces.futureShape(a);
		if(shape.split < 0 && shape.filter < 0) {
			Integer::InitScalar(OUT(thread, inst.c), shape.length);
		} else {
			OUT(thread, inst.c) = thread.traces.EmitUnary<CountFold>(thread.frame.environment, IROpCode::length, a, 0);
			thread.traces.OptBind(thread, OUT(thread,inst.c));
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
	if(thread.traces.isTraceable<MomentFold>(a)) {
		OUT(thread, inst.c) = thread.traces.EmitUnary<MomentFold>(thread.frame.environment, IROpCode::mean, a, 0);
		thread.traces.OptBind(thread, OUT(thread,inst.c));
 		return &inst+1;
	}
	return &inst+1;
}

Instruction const* cm2_op(Thread& thread, Instruction const& inst) {
	OPERAND(a, inst.a); FORCE(a, inst.a); 
	OPERAND(b, inst.b); FORCE(b, inst.b); 
	if(thread.traces.isTraceable<Moment2Fold>(a,b)) {
		OUT(thread, inst.c) = thread.traces.EmitBinary<Moment2Fold>(thread.frame.environment, IROpCode::cm2, a, b, 0);
		thread.traces.OptBind(thread, OUT(thread,inst.c));
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
	if(thread.traces.isTraceable<IfElse>(a,b,c)) {
		OUT(thread, inst.c) = thread.traces.EmitIfElse(thread.frame.environment, a, b, c);
		thread.traces.OptBind(thread, OUT(thread,inst.c));
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
	if(thread.traces.isTraceable<Split>(b,c)) {
		OUT(thread, inst.c) = thread.traces.EmitSplit(thread.frame.environment, c, b, levels);
		thread.traces.OptBind(thread, OUT(thread,inst.c));
		return &inst+1;
	}
	BIND(a); BIND(b); BIND(c);

	_error("split not defined in scalar yet");
	return &inst+1; 
}

Instruction const* function_op(Thread& thread, Instruction const& inst) {
	Value const& function = CONSTANT(inst.a);
	Value& out = OUT(thread, inst.c);
	Function::Init(out, ((Function const&)function).prototype(), thread.frame.environment);
	return &inst+1;
}

Instruction const* vector_op(Thread& thread, Instruction const& inst) {
	OPERAND(a, inst.a); FORCE(a, inst.a); BIND(a);
	OPERAND(b, inst.b); FORCE(b, inst.b); BIND(b);
	Type::Enum type = string2Type( As<Character>(thread, a)[0] );
	int64_t l = As<Integer>(thread, b)[0];
	
	// TODO: replace with isTraceable...
	if(thread.state.jitEnabled 
		&& (type == Type::Double || type == Type::Integer || type == Type::Logical)
		&& l >= TRACE_VECTOR_WIDTH) {
		OUT(thread, inst.c) = thread.traces.EmitConstant(thread.frame.environment, type, l, 0);
		thread.traces.OptBind(thread, OUT(thread,inst.c));
		return &inst+1;
	}

	if(type == Type::Logical) {
		Logical v(l);
		for(int64_t i = 0; i < l; i++) v[i] = Logical::FalseElement;
		OUT(thread, inst.c) = v;
	} else if(type == Type::Integer) {
		Integer v(l);
		for(int64_t i = 0; i < l; i++) v[i] = 0;
		OUT(thread, inst.c) = v;
	} else if(type == Type::Double) {
		Double v(l);
		for(int64_t i = 0; i < l; i++) v[i] = 0;
		OUT(thread, inst.c) = v;
	} else if(type == Type::Character) {
		Character v(l);
		for(int64_t i = 0; i < l; i++) v[i] = Strings::empty;
		OUT(thread, inst.c) = v;
	} else if(type == Type::Raw) {
		Raw v(l);
		for(int64_t i = 0; i < l; i++) v[i] = 0;
		OUT(thread, inst.c) = v;
	} else {
		_error("Invalid type in vector");
	} 
	return &inst+1;
}

Instruction const* seq_op(Thread& thread, Instruction const& inst) {
	// c = start, b = step, a = length
	OPERAND(a, inst.a); FORCE(a, inst.a); BIND(a);
	OPERAND(b, inst.b); FORCE(b, inst.b); BIND(b);
	OPERAND(c, inst.c); FORCE(c, inst.c); BIND(c);

	double start = As<Double>(thread, c)[0];
	double step = As<Double>(thread, b)[0];
	int64_t len = As<Integer>(thread, a)[0];
	
	if(len >= TRACE_VECTOR_WIDTH) {
		if(b.isDouble() || c.isDouble()) {
			OUT(thread, inst.c) = thread.traces.EmitSequence(thread.frame.environment, len, start, step);
			thread.traces.OptBind(thread, OUT(thread,inst.c));
		} else {
			OUT(thread, inst.c) = thread.traces.EmitSequence(thread.frame.environment, len, (int64_t)start, (int64_t)step);
			thread.traces.OptBind(thread, OUT(thread,inst.c));
		}
		return &inst+1;
	}

	if(b.isDouble() || c.isDouble())	
		OUT(thread, inst.c) = Sequence(start, step, len);
	else
		OUT(thread, inst.c) = Sequence((int64_t)start, (int64_t)step, len);
	return &inst+1;
}

Instruction const* rep_op(Thread& thread, Instruction const& inst) {
	// c = n, b = each, a = length
	OPERAND(a, inst.a); FORCE(a, inst.a); BIND(a);
	OPERAND(b, inst.b); FORCE(b, inst.b); BIND(b);
	OPERAND(c, inst.c); FORCE(c, inst.c); BIND(c);

	int64_t n = As<Integer>(thread, c)[0];
	int64_t each = As<Integer>(thread, b)[0];
	int64_t len = As<Integer>(thread, a)[0];
	
	if(len >= TRACE_VECTOR_WIDTH) {
		OUT(thread, inst.c) = thread.traces.EmitRepeat(thread.frame.environment, len, (int64_t)n, (int64_t)each);
		thread.traces.OptBind(thread, OUT(thread,inst.c));
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

	OUT(thread, inst.c) = RandomVector(thread, len);
	return &inst+1;
}

Instruction const* type_op(Thread& thread, Instruction const& inst) {
	OPERAND(a, inst.a); FORCE(a, inst.a);
	switch(thread.traces.futureType(a)) {
                #define CASE(name, str, ...) case Type::name: OUT(thread, inst.c) = Character::c(Strings::name); break;
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
	c = a;
	c.z(0);
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

//
//    Main interpreter loop 
//
//__attribute__((__noinline__,__noclone__)) 
void interpret(Thread& thread, Instruction const* pc) {

#ifdef USE_THREADED_INTERPRETER
	if(pc == 0) { 
    		#define LABELS_THREADED(name,type,...) (void*)&&name##_label,
		static const void* labels[] = {BYTECODES(LABELS_THREADED)};
		glabels = labels;
		return;
	}

	goto *(void*)(pc->ibc);
	#define LABELED_OP(name,type,...) \
		name##_label: \
			{ pc = name##_op(thread, *pc); goto *(void*)(pc->ibc); } 
	STANDARD_BYTECODES(LABELED_OP)
	done_label: {}
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
	interpret(thread, 0);
#endif
}

Value Thread::eval(Prototype const* prototype) {
	return eval(prototype, frame.environment);
}

Value Thread::eval(Prototype const* prototype, Environment* environment) {
	Value* old_base = base;
	int64_t stackSize = stack.size();

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

