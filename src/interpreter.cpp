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

extern Instruction const* kget_op(Thread& thread, Instruction const& inst) ALWAYS_INLINE;
extern Instruction const* get_op(Thread& thread, Instruction const& inst) ALWAYS_INLINE;
extern Instruction const* assign_op(Thread& thread, Instruction const& inst) ALWAYS_INLINE;
extern Instruction const* assign2_op(Thread& thread, Instruction const& inst) ALWAYS_INLINE;
extern Instruction const* forend_op(Thread& thread, Instruction const& inst) ALWAYS_INLINE;
extern Instruction const* add_op(Thread& thread, Instruction const& inst) ALWAYS_INLINE;
extern Instruction const* subset_op(Thread& thread, Instruction const& inst) ALWAYS_INLINE;
extern Instruction const* subset2_op(Thread& thread, Instruction const& inst) ALWAYS_INLINE;

#define REG(thread, i) (*(thread.base+i))

// Tracing stuff

//track the heat of back edge operations and invoke the recorder on hot traces
//unused until we begin tracing loops again
static Instruction const * profile_back_edge(Thread & thread, Instruction const * inst) {
	return inst;
}

bool isRecordableType(Type::Enum type) {
	return type == Type::Double || type == Type::Integer || type == Type::Logical;
}

static Instruction const* trace(Thread& thread, Instruction const& inst, Type::Enum type, int64_t length) {
#ifdef ENABLE_JIT
	if(thread.state.jitEnabled && isRecordableType(type) && length >= TRACE_VECTOR_WIDTH) {
		return thread.trace.BeginTracing(thread, &inst);
	}
#endif
	return 0;
}

static Instruction const* trace(Thread& thread, Instruction const& inst, Value const& a) {
	return trace(thread, inst, a.type, a.length);
}

static Instruction const* trace(Thread& thread, Instruction const& inst, Value const& a, Value const& b) {
#ifdef ENABLE_JIT
	if(thread.state.jitEnabled && isRecordableType(a.type) && isRecordableType(b.type) && 
		(a.length >= TRACE_VECTOR_WIDTH || b.length >= TRACE_VECTOR_WIDTH)) {
		return thread.trace.BeginTracing(thread, &inst);
	}
#endif
	return 0;
}



// Control flow instructions

Instruction const* call_op(Thread& thread, Instruction const& inst) {
	Value f = REG(thread, inst.a);
	if(!f.isFunction())
		_error(std::string("Non-function (") + Type::toString(f.type) + ") as first parameter to call\n");
	Function func(f);
	
	// TODO: using inst.b < 0 to indicate a normal call means that do.call can never use a ..# variable. Not common, but would surely be unexpected for users. Probably best to just have a separate op for do.call?
	
	List arguments;
	Character names;
	Environment* fenv;
	if(inst.b < 0) {
		CompiledCall const& call = thread.frame.prototype->calls[-(inst.b+1)];
		arguments = call.arguments;
		names = call.names;
		if(call.dots < arguments.length)
			ExpandDots(thread, arguments, names, call.dots);
		fenv = CreateEnvironment(thread, func.environment(), thread.frame.environment, call.call);
	} else {
		Value const& reg = REG(thread, inst.b);
		if(reg.isObject()) {
			arguments = List(((Object const&)reg).base());
			names = Character(((Object const&)reg).getNames());
		}
		else {
			arguments = List(reg);
		}
		fenv = CreateEnvironment(thread, func.environment(), thread.frame.environment, Null::Singleton());
	}

	MatchArgs(thread, thread.frame.environment, fenv, func, arguments, names);
	return buildStackFrame(thread, fenv, true, func.prototype(), &REG(thread, inst.c), &inst+1);
}

Instruction const* ret_op(Thread& thread, Instruction const& inst) {
	*(thread.frame.result) = REG(thread, inst.c);
	// if this stack frame owns the environment, we can free it for reuse
	// as long as we don't return a closure...
	// TODO: but also can't if an assignment to an out of scope variable occurs (<<-, assign) with a value of a closure!
	if(thread.frame.ownEnvironment && REG(thread, inst.c).isClosureSafe())
		thread.environments.push_back(thread.frame.environment);
	thread.base = thread.frame.returnbase;
	Instruction const* returnpc = thread.frame.returnpc;
	thread.pop();
	return returnpc;
}

Instruction const* UseMethod_op(Thread& thread, Instruction const& inst) {
	String generic = inst.s;

	CompiledCall const& call = thread.frame.prototype->calls[inst.b];
	List arguments = call.arguments;
	Character names = call.names;
	if(call.dots < arguments.length)
		ExpandDots(thread, arguments, names, call.dots);

	Value object = REG(thread, inst.c);
	Character type = klass(thread, object);

	String method;
	Value f = GenericSearch(thread, type, generic, method);

	if(!f.isFunction()) { 
		_error(std::string("no applicable method for '") + thread.externStr(generic) + "' applied to an object of class \"" + thread.externStr(type[0]) + "\"");
	}

	Function func(f);
	Environment* fenv = CreateEnvironment(thread, func.environment(), thread.frame.environment, call.call);
	MatchArgs(thread, thread.frame.environment, fenv, func, arguments, names);	
	fenv->assign(Strings::dotGeneric, CreateSymbol(generic));
	fenv->assign(Strings::dotMethod, CreateSymbol(method));
	fenv->assign(Strings::dotClass, type); 
	return buildStackFrame(thread, fenv, true, func.prototype(), &REG(thread, inst.c), &inst+1);
}

Instruction const* jmp_op(Thread& thread, Instruction const& inst) {
	return &inst+inst.a;
}

Instruction const* jc_op(Thread& thread, Instruction const& inst) {
	Value& c = REG(thread, inst.c);
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
	Value const& c = REG(thread, inst.a);
	int64_t index = -1;
	if(c.isDouble1()) index = (int64_t)c.d;
	else if(c.isInteger1()) index = c.i;
	else if(c.isLogical1()) index = c.i;
	else if(c.isCharacter1()) {
		for(int64_t i = 1; i <= inst.b; i++) {
			if((&inst+i)->s == c.s) {
				index = i;
				break;
			}
			if(index < 0 && (&inst+i)->s == Strings::empty) {
				index = i;
			}
		}
	}
	if(index >= 1 && index <= inst.b) {
		return &inst + ((&inst+index)->c);
	} else {
		return &inst+1+inst.b;
	}
}

Instruction const* forbegin_op(Thread& thread, Instruction const& inst) {
	// inst.b-1 holds the loopVector
	if((int64_t)REG(thread, inst.b-1).length <= 0) { return &inst+inst.a; }
	Element2(REG(thread, inst.b-1), 0, REG(thread, inst.c));
	REG(thread, inst.b).header = REG(thread, inst.b-1).length;	// warning: not a valid object, but saves a shift
	REG(thread, inst.b).i = 1;
	return &inst+1;
}
Instruction const* forend_op(Thread& thread, Instruction const& inst) {
	if(__builtin_expect((REG(thread,inst.b).i) < REG(thread,inst.b).header, true)) {
		Element2(REG(thread, inst.b-1), REG(thread, inst.b).i, REG(thread, inst.c));
		REG(thread, inst.b).i++;
		return profile_back_edge(thread,&inst+inst.a);
	} else return &inst+1;
}

Instruction const* list_op(Thread& thread, Instruction const& inst) {
	std::vector<String> const& dots = thread.frame.environment->dots;
	// First time through, make a result vector...
	if(REG(thread, inst.a).i == 0) {
		REG(thread, inst.c) = List(dots.size());
	}
	// Otherwise populate result vector with next element
	else {
		thread.frame.environment->assign((String)-REG(thread, inst.a).i, REG(thread, inst.b));
		((List&)REG(thread, inst.c))[REG(thread, inst.a).i-1] = REG(thread, inst.b);
	}

	// If we're all done, check to see if we need to add names and then exit
	if(REG(thread, inst.a).i == (int64_t)dots.size()) {
		bool nonEmptyName = false;
		for(int i = 0; i < (int64_t)dots.size(); i++) 
			if(dots[i] != Strings::empty) nonEmptyName = true;
		if(nonEmptyName) {
			// TODO: should really just use the names in the dots directly
			Character names(dots.size());
			for(int64_t i = 0; i < (int64_t)dots.size(); i++)
				names[i] = dots[i];
			Object::Init(REG(thread, inst.c), REG(thread, inst.c), names);
		}
		return &inst+1;
	}

	// Not done yet, increment counter, evaluate next ..#
	REG(thread, inst.a).i++;
	Value const& src = thread.frame.environment->get((String)-REG(thread, inst.a).i);
	if(!src.isPromise()) {
		REG(thread, inst.b) = src;
		return &inst;
	}
	else {
		Environment* env = Function(src).environment();
		if(env == 0) env = thread.frame.environment;
		Prototype* prototype = Function(src).prototype();
		return buildStackFrame(thread, env, false, prototype, &REG(thread, inst.b), &inst);
	}
}

// Memory access ops

Instruction const* get_op(Thread& thread, Instruction const& inst) {
	// gets are always generated as a sequence of 2 instructions...
	//	1) the get with source symbol in a and dest register in c.
	//	2) an assign with dest symbol in a and source register in c.
	//		(for use by the promise evaluation. If no promise, step over this instruction.)

	// otherwise, need to do a real look up starting from env
	Environment* env = thread.frame.environment;
	String s = inst.s;
	
	Value& dest = REG(thread, inst.c);
	if(env->fastGet(s, dest)) return &inst+2;

	dest = env->get(s);
	while(dest.isNil() && env->LexicalScope() != 0) {
		env = env->LexicalScope();
		dest = env->get(s);
	}

	if(dest.isConcrete()) {
		return &inst+2;
	} else if(dest.isPromise()) {
		Environment* env = Function(dest).environment();
		Prototype* prototype = Function(dest).prototype();
		assert(env != 0);
		return buildStackFrame(thread, env, false, prototype, &dest, &inst+1);
	}
	else
		throw RiposteError(std::string("object '") + thread.externStr(s) + "' not found");
}

Instruction const* kget_op(Thread& thread, Instruction const& inst) {
	REG(thread, inst.c) = thread.frame.prototype->constants[inst.a];
	return &inst+1;
}

Instruction const* assign_op(Thread& thread, Instruction const& inst) {
	if(thread.frame.environment->fastAssign(inst.s, REG(thread, inst.c))) return &inst+1;

	thread.frame.environment->assign(inst.s, REG(thread, inst.c));
	return &inst+1;
}
Instruction const* assign2_op(Thread& thread, Instruction const& inst) {
	// assign2 is always used to assign up at least one scope level...
	// so start off looking up one level...

	Environment* env = thread.frame.environment->LexicalScope();
	assert(env != 0);

	String s = inst.s;
	Value dest = env->get(s);
	while(dest.isNil() && env->LexicalScope() != 0) {
		env = env->LexicalScope();
		dest = env->get(s);
	}

	if(!dest.isNil()) {
		env->assign(s, REG(thread, inst.c));
	}
	else {
		thread.state.global->assign(s, REG(thread, inst.c));
	}
	return &inst+1;
}


// everything else should be in registers

Instruction const* iassign_op(Thread& thread, Instruction const& inst) {
	// a = value, b = index, c = dest 
	SubsetAssign(thread, REG(thread,inst.c), true, REG(thread,inst.b), REG(thread,inst.a), REG(thread,inst.c));
	return &inst+1;
}
Instruction const* eassign_op(Thread& thread, Instruction const& inst) {
	// a = value, b = index, c = dest
	Subset2Assign(thread, REG(thread,inst.c), true, REG(thread,inst.b), REG(thread,inst.a), REG(thread,inst.c));
	return &inst+1; 
}

Instruction const* subset_op(Thread& thread, Instruction const& inst) {
	Value& a = REG(thread, inst.a);
	Value& i = REG(thread, inst.b);
	if(a.isVector()) {
		if(i.isDouble1()) { Element(a, i.d-1, REG(thread, inst.c)); return &inst+1; }
		else if(i.isInteger1()) { Element(a, i.i-1, REG(thread, inst.c)); return &inst+1; }
		else if(i.isLogical1()) { Element(a, Logical::isTrue(i.c) ? 0 : -1, REG(thread, inst.c)); return &inst+1; }
		else if(i.isCharacter1()) { _error("Subscript out of bounds"); }
		else if(i.isVector()) { SubsetSlow(thread, a, i, REG(thread, inst.c)); return &inst+1; }
	}
	if(a.isObject() || i.isObject()) { return GenericDispatch(thread, inst, Strings::bracket, a, i, REG(thread, inst.c)); } 
	_error("Invalid subset operation");
}

Instruction const* subset2_op(Thread& thread, Instruction const& inst) {
	Value& a = REG(thread, inst.a);
	Value& i = REG(thread, inst.b);

	if(a.isVector()) {
		int64_t index = 0;
		if(i.isDouble1()) { index = i.d-1; }
		else if(i.isInteger1()) { index = i.i-1; }
		else if(i.isLogical1() && Logical::isTrue(i.c)) { index = 1-1; }
		else if(i.isVector() && (i.length == 0 || i.length > 1)) { 
			_error("Attempt to select less or more than 1 element in subset2"); 
		}
		else { _error("Subscript out of bounds"); }
		Element2(a, index, REG(thread, inst.c));
		return &inst+1;
	}
	if(a.isObject() || i.isObject()) { return GenericDispatch(thread, inst, Strings::bb, a, i, REG(thread, inst.c)); } 
	_error("Invalid subset2 operation");
}

Instruction const* colon_op(Thread& thread, Instruction const& inst) {
	double from = asReal1(REG(thread,inst.a));
	double to = asReal1(REG(thread,inst.b));
	REG(thread,inst.c) = Sequence(from, to>from?1:-1, fabs(to-from)+1);
	return &inst+1;
}


Instruction const* seq_op(Thread& thread, Instruction const& inst) {
	int64_t len = As<Integer>(thread, REG(thread, inst.a))[0];
	int64_t step = As<Integer>(thread, REG(thread, inst.b))[0];
	
	Instruction const* jit = trace(thread, inst, Type::Integer, len);
	if(jit) return jit;
	
	REG(thread, inst.c) = Sequence(len, 1, step);
	return &inst+1;
}


#define OP(Name, string, Op, Group, Func) \
Instruction const* Name##_op(Thread& thread, Instruction const& inst) { \
	Value & a =  REG(thread, inst.a);	\
	Value & c = REG(thread, inst.c);	\
	if(a.isDouble1())  { Name##VOp<Double>::Scalar(thread, a.d, c); return &inst+1; } \
	if(a.isInteger1()) { Name##VOp<Integer>::Scalar(thread, a.i, c); return &inst+1; } \
	if(a.isLogical1()) { Name##VOp<Logical>::Scalar(thread, a.c, c); return &inst+1; } \
	if(a.isObject()) { return GenericDispatch(thread, inst, Strings::Op, a, c); } \
	\
	Instruction const* jit = trace(thread, inst, a); \
	if(jit) return jit; \
	\
	Group##Dispatch<Name##VOp>(thread, a, c); \
	return &inst+1; \
}
UNARY_FOLD_SCAN_BYTECODES(OP)
#undef OP


#define OP(Name, string, Op, Group, Func) \
Instruction const* Name##_op(Thread& thread, Instruction const& inst) { \
	Value & a =  REG(thread, inst.a);	\
	Value & b =  REG(thread, inst.b);	\
	Value & c = REG(thread, inst.c);	\
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
	if(a.isObject() || b.isObject()) { return GenericDispatch(thread, inst, Strings::Op, a, b, c); } \
	\
	Instruction const* jit = trace(thread, inst, a, b); \
	if(jit) return jit; \
\
	Group##Dispatch<Name##VOp>(thread, a, b, c);	\
	return &inst+1;	\
}
BINARY_BYTECODES(OP)
#undef OP

Instruction const* ifelse_op(Thread& thread, Instruction const& inst) {
	Instruction const* jit = trace(thread, inst, REG(thread, inst.a));
	if(jit) return jit;

	_error("ifelse not defined in scalar yet");
	return &inst+2; 
}

Instruction const* split_op(Thread& thread, Instruction const& inst) {
	Instruction const* jit = trace(thread, inst, REG(thread, inst.a));
	if(jit) return jit;
		
	_error("split not defined in scalar yet");
	return &inst+2; 
}

Instruction const* function_op(Thread& thread, Instruction const& inst) {
	REG(thread, inst.c) = Function(thread.frame.prototype->prototypes[inst.a], thread.frame.environment);
	return &inst+1;
}
Instruction const* logical1_op(Thread& thread, Instruction const& inst) {
	Integer i = As<Integer>(thread, REG(thread, inst.a));
	REG(thread, inst.c) = Logical(i[0]);
	return &inst+1;
}
Instruction const* integer1_op(Thread& thread, Instruction const& inst) {
	Integer i = As<Integer>(thread, REG(thread, inst.a));
	REG(thread, inst.c) = Integer(i[0]);
	return &inst+1;
}
Instruction const* double1_op(Thread& thread, Instruction const& inst) {
	int64_t length = asReal1(REG(thread, inst.a));
	Double d(length);
	for(int64_t i = 0; i < length; i++) d[i] = 0;
	REG(thread, inst.c) = d;
	return &inst+1;
}
Instruction const* character1_op(Thread& thread, Instruction const& inst) {
	Integer i = As<Integer>(thread, REG(thread, inst.a));
	Character r = Character(i[0]);
	for(int64_t j = 0; j < r.length; j++) r[j] = Strings::empty;
	REG(thread, inst.c) = r;
	return &inst+1;
}
Instruction const* raw1_op(Thread& thread, Instruction const& inst) {
	Integer i = As<Integer>(thread, REG(thread, inst.a));
	REG(thread, inst.c) = Raw(i[0]);
	return &inst+1;
}
Instruction const* type_op(Thread& thread, Instruction const& inst) {
	Character c(1);
	// Should have a direct mapping from type to symbol.
	c[0] = thread.internStr(Type::toString(REG(thread, inst.a).type));
	REG(thread, inst.c) = c;
	return &inst+1;
}
Instruction const* length_op(Thread& thread, Instruction const& inst) {
	if(REG(thread,inst.a).isVector())
		Integer::InitScalar(REG(thread, inst.c), REG(thread,inst.a).length);
	else
		Integer::InitScalar(REG(thread, inst.c), 1);
	return &inst+1;
}
Instruction const* missing_op(Thread& thread, Instruction const& inst) {
	// This could be inline cached...or implemented in terms of something else?
	String s = inst.s;
	Value const& v = thread.frame.environment->get(s);
	bool missing = v.isNil() || (v.isPromise() && Function(v).environment() == thread.frame.environment);
	Logical::InitScalar(REG(thread, inst.c), missing);
	return &inst+1;
}
Instruction const* mmul_op(Thread& thread, Instruction const& inst) {
	REG(thread, inst.c) = MatrixMultiply(thread, REG(thread, inst.a), REG(thread, inst.b));
	return &inst+1;
}
Instruction const* strip_op(Thread& thread, Instruction const& inst) {
	REG(thread, inst.c) = REG(thread, inst.a);
	if(REG(thread, inst.c).isObject())
		REG(thread, inst.c) = ((Object const&)REG(thread, inst.c)).base();
	return &inst+1;
}

Instruction const* internal_op(Thread& thread, Instruction const& inst) {
       thread.state.internalFunctions[inst.a].ptr(thread, &REG(thread, inst.b), REG(thread, inst.c));
       return &inst+1;
}

Instruction const* done_op(Thread& thread, Instruction const& inst) {
	// not used. When this instruction is hit, interpreter exits.
	return 0;
}


static void printCode(Thread const& thread, Prototype const* prototype) {
	std::string r = "block:\nconstants: " + intToStr(prototype->constants.size()) + "\n";
	for(int64_t i = 0; i < (int64_t)prototype->constants.size(); i++)
		r = r + intToStr(i) + "=\t" + thread.stringify(prototype->constants[i]) + "\n";

	r = r + "code: " + intToStr(prototype->bc.size()) + "\n";
	for(int64_t i = 0; i < (int64_t)prototype->bc.size(); i++)
		r = r + intToHexStr((uint64_t)&(prototype->bc[i])) + "--: " + intToStr(i) + ":\t" + prototype->bc[i].toString() + "\n";

	std::cout << r << std::endl;
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

	goto *(pc->ibc);
	#define LABELED_OP(name,type,...) \
		name##_label: \
			{ pc = name##_op(thread, *pc); goto *(pc->ibc); } 
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

Value Thread::eval(Function const& function) {
	return eval(function.prototype(), function.environment());
}

Value Thread::eval(Prototype const* prototype) {
	return eval(prototype, frame.environment);
}

Value Thread::eval(Prototype const* prototype, Environment* environment) {
	Instruction done(ByteCode::done);
#ifdef USE_THREADED_INTERPRETER
	done.ibc = glabels[ByteCode::done];
#endif
	Value* old_base = base;
	int64_t stackSize = stack.size();
	
	// Build a half-hearted stack frame for the result. Necessary for the trace recorder.
	StackFrame& s = push();
	s.environment = 0;
	s.prototype = 0;
	s.returnbase = base;
	base -= 1;
	Value* result = base;
	
	Instruction const* run = buildStackFrame(*this, environment, false, prototype, result, &done);
	try {
		interpret(*this, run);
		base = old_base;
		pop();
	} catch(...) {
		base = old_base;
		stack.resize(stackSize);
		throw;
	}
	return *result;
}

