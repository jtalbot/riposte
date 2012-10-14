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
extern Instruction const* lget_op(Thread& thread, Instruction const& inst) ALWAYS_INLINE;
extern Instruction const* lassign_op(Thread& thread, Instruction const& inst) ALWAYS_INLINE;
extern Instruction const* forend_op(Thread& thread, Instruction const& inst) ALWAYS_INLINE;
extern Instruction const* whileend_op(Thread& thread, Instruction const& inst) ALWAYS_INLINE;
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
        return buildStackFrame(thread, f.environment()->DynamicScope(), f.prototype(), f.environment(), name, &inst);
	} else if(a->isDefault()) {
		Function const& f = (Function const&)(*a);
		assert(f.environment());
		return buildStackFrame(thread, f.environment(), f.prototype(), f.environment(), name, &inst);
	} else {
		_error(std::string("Object '") + thread.externStr(name) + "' not found");
	}
}

// Control flow instructions

Instruction const* call_op(Thread& thread, Instruction const& inst) {
	Value const& f = IN(inst.a);
	if(!f.isFunction())
		_error(std::string("Non-function (") + Type::toString(f.type) + ") as first parameter to call\n");
	Function const& func = (Function const&)f;
	
	CompiledCall const& call = thread.frame.prototype->calls[inst.b];
	Environment* fenv = CreateEnvironment(thread, func.environment(), thread.frame.environment, call.call);

    if(thread.jit.state == JIT::RECORDING) {
        JIT::Var a = thread.jit.load(thread, inst.a, &inst);
        thread.jit.emitCall(a, func, fenv, call.call, &inst);
    }
	
	MatchArgs(thread, thread.frame.environment, fenv, func, call);
	return buildStackFrame(thread, fenv, func.prototype(), inst.c, &inst+1);
}

Instruction const* ncall_op(Thread& thread, Instruction const& inst) {
	Value const& f = IN(inst.a);
	if(!f.isFunction())
		_error(std::string("Non-function (") + Type::toString(f.type) + ") as first parameter to call\n");
	Function const& func = (Function const&)f;
	
	CompiledCall const& call = thread.frame.prototype->calls[inst.b];
	Environment* fenv = CreateEnvironment(thread, func.environment(), thread.frame.environment, call.call);
	
    if(thread.jit.state == JIT::RECORDING) {
        JIT::Var a = thread.jit.load(thread, inst.a, &inst);
        thread.jit.emitCall(a, func, fenv, call.call, &inst);
    }
	
	MatchNamedArgs(thread, thread.frame.environment, fenv, func, call);
	return buildStackFrame(thread, fenv, func.prototype(), inst.c, &inst+1);
}

Instruction const* ret_op(Thread& thread, Instruction const& inst) {
	// We can free this environment for reuse
	// as long as we don't return a closure...
	// TODO: but also can't if an assignment to an out of scope variable occurs (<<-, assign) with a value of a closure!
	Value const& result = IN(inst.a);

    if(result.isClosureSafe()) {
		thread.environments.push_back(thread.frame.environment);
	}

    if(thread.jit.state == JIT::RECORDING) {
        JIT::Var ira = thread.jit.load(thread, inst.a, &inst);
        thread.jit.store(thread, ira, 0);
        thread.jit.Emit( JIT::IR(TraceOpCode::pop, Type::Nil, JIT::Shape::Empty, JIT::Shape::Empty) );
    }
	
	OUT(0) = result;
	
	thread.base = thread.frame.returnbase;
	Instruction const* returnpc = thread.frame.returnpc;

	thread.pop();
	
	return returnpc;
}

Instruction const* rets_op(Thread& thread, Instruction const& inst) {
	
	OUT(0) = IN(inst.a);
	
	thread.base = thread.frame.returnbase;
	thread.pop();
	
	// there should always be a done_op after a rets
	return &inst+1;
}

Instruction const* done_op(Thread& thread, Instruction const& inst) {
	return 0;
}

Instruction const* retp_op(Thread& thread, Instruction const& inst) {
    if(thread.frame.dest > 0) {
		thread.frame.env->insert((String)thread.frame.dest) = IN(inst.a);
        if(thread.jit.state == JIT::RECORDING) {
            JIT::Var ira = thread.jit.load(thread, inst.a, &inst);
            thread.jit.estore(ira, thread.frame.env, (String)thread.frame.dest);
            thread.jit.Emit( JIT::IR( TraceOpCode::pop, Type::Nil, JIT::Shape::Empty, JIT::Shape::Empty) );
        }
	} else {
		thread.frame.env->dots[-thread.frame.dest].v = IN(inst.a);
	}
	thread.base = thread.frame.returnbase;
	Instruction const* returnpc = thread.frame.returnpc;
	thread.pop();
	
	return returnpc;
}

Instruction const* constant_op(Thread& thread, Instruction const& inst) {
    OUT(inst.c) = thread.frame.prototype->constants[inst.a];
    return &inst+1;
}

Instruction const* jmp_op(Thread& thread, Instruction const& inst) {
	return &inst+inst.a;
}

Instruction const* loop_op(Thread& thread, Instruction const& inst) {
    return &inst+1;
}

Instruction const* jc_op(Thread& thread, Instruction const& inst) {
	Value const& c = IN(inst.c);
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
	_error("Need single element logical in conditional jump");
}

Instruction const* branch_op(Thread& thread, Instruction const& inst) {
	Value const& a = IN(inst.a);
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
	return &inst+1+inst.b;
}

Instruction const* forbegin_op(Thread& thread, Instruction const& inst) {
	// a = loop variable (e.g. i), b = loop vector(e.g. 1:100), c = counter register
	// following instruction is a jmp that contains offset
	Value const& vec = IN(inst.b);
	if((int64_t)vec.length <= 0) {
		return &inst+(&inst+1)->a;	// offset is in following JMP, dispatch together
	} else {
		Element2(vec, 0, thread.frame.environment->insert((String)inst.a));
        OUT(inst.c-1) = Integer::c(vec.length);     // length register
        OUT(inst.c)   = Integer::c(1);              // iterator register
		return &inst+2;			// skip over following JMP
	}
}

Instruction const* forend_op(Thread& thread, Instruction const& inst) {
	Value const& length = IN(inst.c-1);
	Value& counter = OUT(inst.c);
	if(__builtin_expect((counter.i) < length.i, true)) {
		Element2(IN(inst.b), counter.i, thread.frame.environment->insert((String)inst.a));
		counter.i++;
		return &inst+(&inst+1)->a;
	} else {
		return &inst+2;			// skip over following JMP
	}
}

Instruction const* whileend_op(Thread& thread, Instruction const& inst) {
	Value const& c = IN(inst.c);
	if(c.isLogical1()) {
		if(Logical::isTrue(c.c)) return &inst+inst.a;
		else if(Logical::isFalse(c.c)) return &inst+1;
		else _error("NA where TRUE/FALSE needed"); 
	} else if(c.isInteger1()) {
		if(Integer::isNA(c.i)) _error("NA where TRUE/FALSE needed");
		else if(c.i != 0) return &inst + inst.a;
		else return & inst+1;
	} else if(c.isDouble1()) {
		if(Double::isNA(c.d)) _error("NA where TRUE/FALSE needed");
		else if(c.d != 0) return &inst + inst.a;
		else return & inst+1;
	}
	_error("Need single element logical in while condition");
}

Instruction const* dotslist_op(Thread& thread, Instruction const& inst) {
	PairList const& dots = thread.frame.environment->dots;
	
	Value& iter = OUT(inst.a);
	Value& out = OUT(inst.c);
	
	// First time through, make a result vector...
	if(iter.i == 0) {
		out = List(dots.size());
	}
	
	if(iter.i < (int64_t)dots.size()) {
		Value const& a = DOT(iter.i); 
        FORCE_DOTDOT(a, iter.i); 
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
		Value const& r = IN(inst.a-i);
		out[i] = r;
	}
	OUT(inst.c) = out;
	return &inst+1;
}

// Memory access ops

Instruction const* get_op(Thread& thread, Instruction const& inst) {
	Value const& env = IN(inst.b);
    
    if(!env.isEnvironment())
        _error("Getting from a non-environment is not supported");

    // TODO: force unevaluated promises

    if( ((REnvironment&)env).environment()->has((String)inst.a) )
        OUT(inst.c) = ((REnvironment&)env).environment()->get((String)inst.a);
	else
        OUT(inst.c) = Null::Singleton();

	return &inst+1;
}

Instruction const* assign_op(Thread& thread, Instruction const& inst) {
    Value const& env = IN(inst.b);
	
    if(!env.isEnvironment())
        _error("Assigning to a non-environment is not supported");

    ((REnvironment&)env).environment()->insert((String)inst.a) = IN(inst.c);

    // Implicitly: OUT(inst.c) = IN(inst.c);

	return &inst+1;
}

Instruction const* lget_op(Thread& thread, Instruction const& inst) {

    Value const& a = thread.frame.environment->getRecursive((String)inst.a);
    
    // If unevaluated promise, force.
    if(!a.isConcrete()) {
        if(a.isDotdot()) {
            Value const& t = ((Environment*)a.p)->dots[a.length].v;
            if(!t.isConcrete()) {
                return forceDot(thread, inst, &t, (Environment*)a.p, a.length);
            }
            thread.frame.environment->insert((String)inst.a) = t;
        }
        else {
            return forceReg(thread, inst, &a, (String)inst.a);
        }
    }

	OUT(inst.c) = a;
	return &inst+1;
}

Instruction const* lassign_op(Thread& thread, Instruction const& inst) {
	thread.frame.environment->insert((String)inst.a) = IN(inst.c);
	return &inst+1;
}

Instruction const* lassign2_op(Thread& thread, Instruction const& inst) {
	// assign2 is always used to assign up at least one scope level...
	// so start off looking up one level...
	assert(thread.frame.environment->LexicalScope() != 0);
	
	String s = (String)inst.a;
	Value& dest = thread.frame.environment->LexicalScope()->insertRecursive(s);

	if(!dest.isNil()) {
		dest = IN(inst.c);
		// TODO: should add dest's environment to the liveEnvironments list
	}
	else {
		thread.state.global->insert(s) = IN(inst.c);
	}
	return &inst+1;
}

Instruction const* dotdot_op(Thread& thread, Instruction const& inst) {
	if(inst.a >= (int64_t)thread.frame.environment->dots.size())
        	_error(std::string("The '...' list does not contain ") + intToStr(inst.a+1) + " elements");
	Value const& a = DOT(inst.a);
    FORCE_DOTDOT(a, inst.a); // no need to bind since value is in a register
	OUT(inst.c) = a;
	return &inst+1;
}

Instruction const* scatter_op(Thread& thread, Instruction const& inst) {
	// a = value, b = index, c = dest
	SubsetAssign(thread, IN(inst.c), true, IN(inst.b), IN(inst.a), OUT(inst.c));
	return &inst+1;
}

Instruction const* scatter1_op(Thread& thread, Instruction const& inst) {
	// a = value, b = index, c = dest
	Subset2Assign(thread, IN(inst.c), true, IN(inst.b), IN(inst.a), OUT(inst.c));
	return &inst+1; 
}

Instruction const* gather_op(Thread& thread, Instruction const& inst) {
	Value const& a = IN(inst.a);
	Value const& i = IN(inst.b);

	if(a.isVector()) {
		if(i.isDouble1()) { 
            Element(a, i.d-1, OUT(inst.c)); return &inst+1; 
        }
		else if(i.isInteger1()) { 
            Element(a, i.i-1, OUT(inst.c)); return &inst+1; 
        }
		else if(i.isLogical1()) { 
            Element(a, Logical::isTrue(i.c) ? 0 : -1, OUT(inst.c)); return &inst+1; 
        }
		else if(i.isCharacter1()) { 
            _error("Subscript out of bounds"); 
        }
	}

	if(a.isObject()) { 
		return GenericDispatch(thread, inst, Strings::bracket, a, i, inst.c); 
	} 
	
    if(i.isObject()) { 
		return GenericDispatch(thread, inst, Strings::bracket, a, i, inst.c); 
	} 
	
	SubsetSlow(thread, a, i, OUT(inst.c)); 
	return &inst+1;
}

Instruction const* gather1_op(Thread& thread, Instruction const& inst) {
    Value const& a = IN(inst.a);
    Value const& i = IN(inst.b);

	if(a.isVector()) {
		int64_t index = 0;
		if(i.isDouble1()) { 
            index = i.d-1; 
        }
		else if(i.isInteger1()) { 
            index = i.i-1; 
        }
		else if(i.isLogical1() && Logical::isTrue(i.c)) { 
            index = 1-1; 
        }
		else if(i.isVector() && (i.length == 0 || i.length > 1)) { 
			_error("Attempt to select less or more than 1 element in subset2"); 
		}
		else { 
            _error("Subscript out of bounds"); 
        }
		Element2(a, index, OUT(inst.c));
		return &inst+1;
	}

	if(a.isObject() || i.isObject()) { 
        return GenericDispatch(thread, inst, Strings::bb, a, i, inst.c); 
    }

	_error("Invalid subset2 operation");
}

#define OP(Name, string, Group, Func, Cost) \
Instruction const* Name##_op(Thread& thread, Instruction const& inst) { \
	Value const& a = IN(inst.a); \
	Value & c = OUT(inst.c);	 \
	if(a.isDouble1())  { Name##VOp<Double>::Scalar(thread, a.d, c); return &inst+1; } \
	if(a.isInteger1()) { Name##VOp<Integer>::Scalar(thread, a.i, c); return &inst+1; } \
	if(a.isLogical1()) { Name##VOp<Logical>::Scalar(thread, a.c, c); return &inst+1; } \
	if(a.isObject())   { return GenericDispatch(thread, inst, Strings::Name, a, inst.c); } \
\
	Group##Dispatch<Name##VOp>(thread, a, c); \
	return &inst+1; \
}
UNARY_FOLD_SCAN_BYTECODES(OP)
#undef OP

#define OP(Name, string, Group, Func, Cost) \
Instruction const* Name##_op(Thread& thread, Instruction const& inst) { \
	Value const& a = IN(inst.a); \
	Value const& b = IN(inst.b); \
	Value & c = OUT(inst.c);	\
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
	if(a.isObject() || b.isObject()) { return GenericDispatch(thread, inst, Strings::Name, a, b, inst.c); } \
\
	Group##Dispatch<Name##VOp>(thread, a, b, c);	\
	return &inst+1;	\
}
BINARY_BYTECODES(OP)
#undef OP

Instruction const* length_op(Thread& thread, Instruction const& inst) {
	Value const& a = IN(inst.a);
	if(a.isVector())
		Integer::InitScalar(OUT(inst.c), a.length);
	else if(a.isObject()) { 
		return GenericDispatch(thread, inst, Strings::length, a, inst.c); 
	} else {
		Integer::InitScalar(OUT(inst.c), 1);
	}
	return &inst+1;
}

Instruction const* mean_op(Thread& thread, Instruction const& inst) {
	//Value const& a = IN(inst.a);
    _error("No scalar implementation of mean");
	return &inst+1;
}

Instruction const* cm2_op(Thread& thread, Instruction const& inst) {
	//Value const& a = IN(inst.a);
	//Value const& b = IN(inst.b);
    _error("No scalar implementation of cm2");
	return &inst+1;
}

Instruction const* split_op(Thread& thread, Instruction const& inst) {
	//Value const& a = IN(inst.a);
	//int64_t levels = As<Integer>(thread, a)[0];
	//Value const& b = IN(inst.b);
	//Value const& c = IN(inst.c);
	_error("split not defined in scalar yet");
	return &inst+1; 
}

Instruction const* ifelse_op(Thread& thread, Instruction const& inst) {
	Value const& a = IN(inst.a);
	Value const& b = IN(inst.b);
	Value const& c = IN(inst.c);
	if(c.isLogical1()) {
		OUT(inst.c) = Logical::isTrue(c.c) ? b : a;
		return &inst+1; 
	}
	else if(c.isInteger1()) {
		OUT(inst.c) = c.i ? b : a;
		return &inst+1; 
	}
	else if(c.isDouble1()) {
		OUT(inst.c) = c.d ? b : a;
		return &inst+1; 
	}
	_error("ifelse not defined in scalar yet");
	return &inst+1; 
}

Instruction const* function_op(Thread& thread, Instruction const& inst) {
	Value const& function = IN(inst.a);
	Value& out = OUT(inst.c);
	out.header = function.header;
	out.p = (void*)thread.frame.environment;
	return &inst+1;
}

Instruction const* vector_op(Thread& thread, Instruction const& inst) {
	Value const& a = IN(inst.a);
    Value const& b = IN(inst.b);
	
    Type::Enum type = string2Type( As<Character>(thread, a)[0] );
	int64_t l = As<Integer>(thread, b)[0];
	
	if(type == Type::Logical) {
		Logical v(l);
		for(int64_t i = 0; i < v.length; i++) v[i] = Logical::FalseElement;
		OUT(inst.c) = v;
	} else if(type == Type::Integer) {
		Integer v(l);
		for(int64_t i = 0; i < v.length; i++) v[i] = 0;
		OUT(inst.c) = v;
	} else if(type == Type::Double) {
		Double v(l);
		for(int64_t i = 0; i < v.length; i++) v[i] = 0;
		OUT(inst.c) = v;
	} else if(type == Type::Character) {
		Character v(l);
		for(int64_t i = 0; i < v.length; i++) v[i] = Strings::empty;
		OUT(inst.c) = v;
	} else if(type == Type::Raw) {
		Raw v(l);
		for(int64_t i = 0; i < v.length; i++) v[i] = 0;
		OUT(inst.c) = v;
	} else {
		_error("Invalid type in vector");
	} 
	return &inst+1;
}

Instruction const* seq_op(Thread& thread, Instruction const& inst) {
	// c = start, b = step, a = length
	Value const& a = IN(inst.a);
    Value const& b = IN(inst.b);
    Value const& c = IN(inst.c);

	double start = As<Double>(thread, c)[0];
	double step = As<Double>(thread, b)[0];
	int64_t len = As<Integer>(thread, a)[0];

	if(b.isDouble() || c.isDouble())	
		OUT(inst.c) = Sequence(start, step, len);
	else
		OUT(inst.c) = Sequence((int64_t)start, (int64_t)step, len);
	return &inst+1;
}

Instruction const* rep_op(Thread& thread, Instruction const& inst) {
	// c = n, b = each, a = length
	Value const& a = IN(inst.a);
    Value const& b = IN(inst.b);
    Value const& c = IN(inst.c);

	int64_t n = As<Integer>(thread, c)[0];
	int64_t each = As<Integer>(thread, b)[0];
	int64_t len = As<Integer>(thread, a)[0];

	OUT(inst.c) = Repeat((int64_t)n, (int64_t)each, len);
	return &inst+1;
}

Instruction const* random_op(Thread& thread, Instruction const& inst) {
	Value const& a = IN(inst.a);

	int64_t len = As<Integer>(thread, a)[0];
	
	OUT(inst.c) = Random(thread, len);
	return &inst+1;
}

Instruction const* type_op(Thread& thread, Instruction const& inst) {
	Value const& a = IN(inst.a);
	
    switch(a.type) {
        #define CASE(name, str) \
            case Type::name: OUT(inst.c) = Character::c(Strings::name); break;
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
    // 
    // This is further exacerbated by the fact that defaults are not evaluated on
    // recursively missing arguments. Bad design choice R!
	String s = (String)inst.a;
	Value const& v = thread.frame.environment->get(s);
	bool missing = v.isNil() || v.isDefault();
	Logical::InitScalar(OUT(inst.c), missing ? Logical::TrueElement : Logical::FalseElement);
	return &inst+1;
}
Instruction const* strip_op(Thread& thread, Instruction const& inst) {
	Value const& a = IN(inst.a);
	Value& c = OUT(inst.c);
	if(a.isObject())
		c = ((Object const&)a).base();
	else
		c = a;
	return &inst+1;
}

Instruction const* internal_op(Thread& thread, Instruction const& inst) {
	if(inst.a < 0)
		_error("Attempting to use undefined internal function");
	thread.state.internalFunctions[inst.a].ptr(thread, &IN(inst.b), OUT(inst.c));
	return &inst+1;
}

Instruction const* nargs_op(Thread& thread, Instruction const& inst) {
    OUT(inst.c) = Integer::c(thread.frame.environment->call.length-1);
    return &inst+1;
}

Instruction const* attrget_op(Thread& thread, Instruction const& inst) {
	Value const& object = IN(inst.a);
	Value const& whichTmp = IN(inst.b);
    
    if(object.isObject()) {
        Character which = As<Character>(thread, whichTmp);
        OUT(inst.c) = ((Object const&)object).get(which[0]);
    }
    else {
        OUT(inst.c) = Null::Singleton();
    }
    return &inst+1;
}

Instruction const* attrset_op(Thread& thread, Instruction const& inst) {
	Value const& object = IN(inst.a);
	Value const& whichTmp = IN(inst.b);
	Value const& value = IN(inst.c);

    Character which = As<Character>(thread, whichTmp);
    if(!object.isObject()) {
        Value v;
        Object::Init((Object&)v, object);
        ((Object&)v).insertMutable(which[0], value);
        OUT(inst.c) = v;
    } else {
        OUT(inst.c) = ((Object&)object).insert(which[0], value);
    }
    return &inst+1;
}

Instruction const* newenv_op(Thread& thread, Instruction const& inst) {
	Value const& parent = IN(inst.a);
   
    Environment* p = ((REnvironment const&)parent).environment();
    Environment* e = new Environment(p,p);
    REnvironment::Init( OUT(inst.c), e );
    return &inst+1; 
}

Instruction const* parentframe_op(Thread& thread, Instruction const& inst) {
    Environment* e = thread.frame.environment->DynamicScope();
    REnvironment::Init( OUT(inst.c), e ); 
    return &inst+1;
}

bool traceLoop(Thread& thread, Instruction const* pc, JIT::Trace* trace) 
{
    unsigned short& counter = 
        trace == 0 ? thread.jit.counters[(((uintptr_t)pc)>>5) & (1024-1)] : trace->counter;
    counter++;
    if(counter >= JIT::RECORD_TRIGGER && counter == nextPow2(counter)) {
        if(trace == 0) {
            trace = new JIT::Trace();
            trace->traceIndex = JIT::Trace::traceCount++;
            trace->function = 0;
            trace->ptr = 0;
            trace->root = trace;
            trace->Reenter = pc;
            trace->counter = 0;
        }

        if(thread.state.verbose)
            printf("Starting to record trace %d at %li (counter: %li is %d)\n", trace->traceIndex, (uint64_t)pc, &counter, counter);

        thread.jit.start_recording(thread, pc, thread.frame.environment, 0, trace);
        thread.jit.cache[pc] = trace;
        return true;
    }
    return false;
}

bool traceSideExit(Thread& thread, Instruction const* pc, JIT::Trace* exit)
{
    if(++exit->counter >= JIT::RECORD_TRIGGER && exit->counter == nextPow2(exit->counter)) {
        if(thread.state.verbose)
            printf("Starting to record side exit %d at %li (counter: %d)\n", exit->traceIndex, (uint64_t)pc, exit->counter);
        thread.jit.start_recording(thread, pc, thread.frame.environment, exit->root, exit);
        return true;
    }
    return false;
}

// returns true if we should begin tracing
bool trace(Thread& thread, Instruction const* loopPC, Instruction const* loopExitPC, Instruction const*& pc) {
    std::map<Instruction const*, JIT::Trace*>::const_iterator t = thread.jit.cache.find(loopPC);
    if(t != thread.jit.cache.end() && t->second->ptr != 0) {
        JIT::Trace* rootTrace = t->second;
    
        GC_disable();
        JIT::Trace* exit = (JIT::Trace*)(rootTrace->ptr)(thread);
        GC_enable();

        // if exit==0, then the trace tree ended the loop along the normal exit, 
        // jump to the default exit, no need to attempt to start another trace.
        if(exit == 0) {
            pc = loopExitPC;
            return false;
        }
        // we exited on a side exit, jump to reentry instructon and check if we
        // want to trace the side exit
        else {
            pc = exit->Reenter;
            return traceSideExit(thread, exit->root->Reenter, exit); 
        }
    }
    // not traced yet, check if we want to start 
    else {
        return traceLoop(thread, loopPC, t != thread.jit.cache.end() ? t->second : 0); 
    }
}

void stopTrace(Thread& thread, char const* reason) {
    if(thread.state.verbose)
        printf("Stopped trace %d due to %s\n", thread.jit.dest->traceIndex, reason);
    thread.jit.fail_recording();
} 

// returns true if we should continue tracing
bool continueTrace(Thread& thread, Instruction const* loopPC, Instruction const* loopExitPC, Instruction const*& pc) {
    // did we make a loop yet??
    JIT::LoopResult loopResult = thread.jit.loop(thread, loopPC);
    
    if(loopResult == JIT::LOOP) {
        if(thread.state.verbose)
            printf("Made loop at %li\n", loopPC);
        thread.jit.end_recording(thread);
        return false;
    }

    if(loopResult == JIT::RECURSIVE) {
        stopTrace(thread, "recursive self loop");
        return false;
    }

    // otherwise, we've hit a nested loop.
    // if it's already traced, emit a nested call
    // otherwise bail tracing, and attempt to record the nested loop
    std::map<Instruction const*, JIT::Trace*>::const_iterator t = thread.jit.cache.find(loopPC);
    if(t != thread.jit.cache.end()) {
        JIT::Trace* rootTrace = t->second;
        if(rootTrace->ptr == 0) {
            stopTrace(thread, "blacklisted inner loop");
            return false;
        }
        else {
            // there is a good inner trace, execute it.
            GC_disable();
            JIT::Trace* exit = (JIT::Trace*)(rootTrace->ptr)(thread);
            GC_enable();

            if(exit == 0) {
                // we exited from the nested loop normally,
                // record nest and continue from the normal exit point
                thread.jit.EmitNest(thread, rootTrace);
                pc = loopExitPC;
                return true;
            }
            else {
                // we bailed prematurely due to an untraced side exit in the inner loop.
                // Stop recording the outer loop to give the inner loop time to pick up the
                //  the side exit
                stopTrace(thread, "untraced side exit in inner loop");
                pc = exit->Reenter;
                return false;
            }
        }
    }
    else {
        stopTrace(thread, "untraced inner loop");
        return false;
    }
}


//
//    Main interpreter loop 
//
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
	
    #define LABELED_OP(name,...) \
		name##_label: \
			{ pc = name##_op(thread, *pc); goto *(const void*)labels[pc->bc]; } 
	STANDARD_BYTECODES(LABELED_OP)
    LABELED_OP(ncall)
    LABELED_OP(forbegin)
    LABELED_OP(branch)
    LABELED_OP(list)
    LABELED_OP(dotslist)
	#undef LABELED_OP

    forend_label: {
        Instruction const* this_pc = pc;
        pc = forend_op(thread, *pc);
        bool startTrace = thread.state.jitEnabled
                            && pc < this_pc
                            && trace(thread, this_pc, this_pc+2, pc);
        if(startTrace)
            labels = record;
        goto *(const void*)labels[pc->bc]; 
    }

    whileend_label: {
        Instruction const* this_pc = pc;
        pc = whileend_op(thread, *pc);
        bool startTrace = thread.state.jitEnabled
                            && pc < this_pc
                            && trace(thread, this_pc, this_pc+1, pc);
        if(startTrace)
            labels = record;
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
            if(!thread.jit.record(thread, pc)) { \
                thread.jit.fail_recording(); \
		        labels = ops; \
            } \
            pc = name##_op(thread, *pc); \
            goto *(const void*)labels[pc->bc]; \
    } 
	STANDARD_BYTECODES(RECORD_OP)
    RECORD_OP(ncall)
    RECORD_OP(forbegin)
    RECORD_OP(branch)
    RECORD_OP(list)
    RECORD_OP(dotslist)
	#undef RECORD_OP
	
	forend_record: 	{
        thread.jit.record(thread, pc);
        bool contTrace = continueTrace(thread, pc, pc+2, pc);
        if(!contTrace)
            labels = ops;
        goto *(const void*)labels[pc->bc]; 
	}
	
    whileend_record: 	{
        thread.jit.record(thread, pc);
        bool contTrace = continueTrace(thread, pc, pc+1, pc);
        if(!contTrace)
            labels = ops;
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
        //Instruction const* old_pc = pc;
        pc = ret_op(thread, *pc);
        goto *(const void*)labels[pc->bc]; 
    }
	retp_record: 	{ 
        //Instruction const* old_pc = pc;
        pc = retp_op(thread, *pc);
        goto *(const void*)labels[pc->bc]; 
    }
	rets_record: 	{
		pc = rets_op(thread, *pc);
		// terminate recording
		labels = ops;
        thread.jit.fail_recording();
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

