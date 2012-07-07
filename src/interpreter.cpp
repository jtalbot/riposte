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

static inline Instruction const* mov_op(Thread& thread, Instruction const& inst) ALWAYS_INLINE;
static inline Instruction const* fastmov_op(Thread& thread, Instruction const& inst) ALWAYS_INLINE;
static inline Instruction const* assign_op(Thread& thread, Instruction const& inst) ALWAYS_INLINE;
static inline Instruction const* forend_op(Thread& thread, Instruction const& inst) ALWAYS_INLINE;
static inline Instruction const* add_op(Thread& thread, Instruction const& inst) ALWAYS_INLINE;
static inline Instruction const* subset_op(Thread& thread, Instruction const& inst) ALWAYS_INLINE;
static inline Instruction const* subset2_op(Thread& thread, Instruction const& inst) ALWAYS_INLINE;
static inline Instruction const* jc_op(Thread& thread, Instruction const& inst) ALWAYS_INLINE;
static inline Instruction const* lt_op(Thread& thread, Instruction const& inst) ALWAYS_INLINE;
static inline Instruction const* ret_op(Thread& thread, Instruction const& inst) ALWAYS_INLINE;
static inline Instruction const* retp_op(Thread& thread, Instruction const& inst) ALWAYS_INLINE;
static inline Instruction const* internal_op(Thread& thread, Instruction const& inst) ALWAYS_INLINE;
static inline Instruction const* strip_op(Thread& thread, Instruction const& inst) ALWAYS_INLINE;


// forces a value stored in the Environments dotdot slot: dest[index]
// call through FORCE_DOTDOT macro which inlines some performance-important checks
static inline Instruction const* forceDot(Thread& thread, Instruction const& inst, Value const& v, Environment* dest, int64_t index) {
	if(v.isPromise()) {
		Promise const& a = (Promise const&)v;
		if(a.isPrototype()) {
			return buildStackFrame(thread, a.environment(), a.prototype(), dest, index, &inst);
		} 
		else if(a.isDotdot()) {
       	        	Value const& t = a.environment()->dots[a.dotIndex()].v;
			Instruction const* result = &inst;
			if(!t.isObject()) {
				result = forceDot(thread, inst, t, a.environment(), a.dotIndex());
			}
			if(t.isObject()) {
				dest->dots[index].v = t;
				thread.traces.LiveEnvironment(dest, t);
			}
			return result;
		}
		else {
			_error("Invalid promise type");
		}
	}
	else {
		_error(std::string("Object '..") + intToStr(index+1) + "' not found, missing argument?");
	} 
}

// forces a value stored in the Environment slot: dest->name
// call through FORCE macro which inlines some performance-important checks
//  for the common cases.
// Environments can have promises, defaults, or dotdots (references to ..n in the parent).
static inline Instruction const* force(Thread& thread, Instruction const& inst, Value const& v, Environment
* dest, String name) {
	if(v.isPromise()) {
		Promise const& a = (Promise const&)v;
		if(a.isPrototype()) {
       	        	return buildStackFrame(thread, a.environment(), a.prototype(), dest, name, &inst);
        	}
		else if(a.isDotdot()) {
                	Value const& t = a.environment()->dots[a.dotIndex()].v;
			Instruction const* result = &inst;
			// if this dotdot is a promise, attempt to force.
			// first time through this will return the address of the 
			//	promise's new stack frame.
			// second time through this will return the resulting value
			// => Thus, evaluating dotdot requires at most 2 sweeps up the dotdot chain
			if(!t.isObject()) {
				result = forceDot(thread, inst, (Promise const&)t, a.environment(), a.dotIndex());
			}
       	        	if(t.isObject()) {
       	                	dest->insert(name) = t;
       	                 	thread.traces.LiveEnvironment(dest, t);
                	}
                	return result;
        	}
		else {
			_error("Invalid promise type");
		}
	}
	else {
		_error(std::string("Object '") + thread.externStr(name) + "' not found"); 
	} 
}

// Control flow instructions

static inline Instruction const* call_op(Thread& thread, Instruction const& inst) {
	Heap::Global.collect(thread.state);
	DECODE(a); FORCE(a); BIND(a);
	if(!a.isFunction())
		_error(std::string("Non-function (") + Type::toString(a.type()) + ") as first parameter to call\n");
	Function const& func = (Function const&)a;
	
	CompiledCall const& call = thread.frame.prototype->calls[inst.b];
	Environment* fenv = new Environment((int64_t)call.arguments.size(), func.environment(), thread.frame.environment, call.call);
	
	MatchArgs(thread, thread.frame.environment, fenv, func, call);
	return buildStackFrame(thread, fenv, func.prototype(), inst.c, &inst+1);
}

static inline Instruction const* fastcall_op(Thread& thread, Instruction const& inst) {
	Heap::Global.collect(thread.state);
	DECODE(a); FORCE(a); BIND(a);
	if(!a.isFunction())
		_error(std::string("Non-function (") + Type::toString(a.type()) + ") as first parameter to call\n");
	Function const& func = (Function const&)a;
	
	CompiledCall const& call = thread.frame.prototype->calls[inst.b];
	Environment* fenv = new Environment((int64_t)call.arguments.size(), func.environment(), thread.frame.environment, call.call);
	
	FastMatchArgs(thread, thread.frame.environment, fenv, func, call);
	return buildStackFrame(thread, fenv, func.prototype(), inst.c, &inst+1);
}

static inline Instruction const* ret_op(Thread& thread, Instruction const& inst) {
	// we can return futures from functions, so don't BIND
	DECODE(a); FORCE(a);	
	
	// We can free this environment for reuse
	// as long as we don't return a closure...
	// TODO: but also can't if an assignment to an out of scope variable occurs (<<-, assign) with a value of a closure!
	if(!(a.isFunction() || a.isEnvironment() || a.isList())) {
		thread.traces.KillEnvironment(thread.frame.environment);
	}

	REGISTER(0) = a;
	Instruction const* returnpc = thread.frame.returnpc;
	thread.pop();
	
	thread.traces.LiveEnvironment(thread.frame.environment, a);

	return returnpc;
}

static inline Instruction const* retp_op(Thread& thread, Instruction const& inst) {
	// we can return futures from promises, so don't BIND
	DECODE(a); FORCE(a);	
	
	if(thread.frame.dest > 0) {
		thread.frame.env->insert((String)thread.frame.dest) = a;
	} else {
		thread.frame.env->dots[-thread.frame.dest].v = a;
	}
	thread.traces.LiveEnvironment(thread.frame.env, a);
	
	Instruction const* returnpc = thread.frame.returnpc;
	thread.pop();
	
	return returnpc;
}

static inline Instruction const* rets_op(Thread& thread, Instruction const& inst) {
	// top-level statements can't return futures, so bind 
	DECODE(a); FORCE(a); BIND(a);	
	
	REGISTER(0) = a;
	thread.pop();
	
	// there should always be a done_op after a rets
	return &inst+1;
}

static inline Instruction const* done_op(Thread& thread, Instruction const& inst) {
	return 0;
}

static inline Instruction const* jmp_op(Thread& thread, Instruction const& inst) {
	return &inst+inst.a;
}

static inline Instruction const* jc_op(Thread& thread, Instruction const& inst) {
	DECODE(c);
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
	FORCE(c); BIND(c);
	_error("Need single element logical in conditional jump");
}

static inline Instruction const* branch_op(Thread& thread, Instruction const& inst) {
	DECODE(a);
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
	FORCE(a); BIND(a);
	return &inst+1+inst.b;
}

static inline Instruction const* forbegin_op(Thread& thread, Instruction const& inst) {
	// a = loop variable (e.g. i), b = loop vector(e.g. 1:100), c = counter register
	// following instruction is a jmp that contains offset
	Value& b = REGISTER(inst.b);
	if(!b.isVector())
		_error("Invalid for() loop sequence");
	Vector const& v = (Vector const&)b;
	if((int64_t)v.length() <= 0) {
		return &inst+(&inst+1)->a;	// offset is in following JMP, dispatch together
	} else {
		Element2(v, 0, thread.frame.environment->insert((String)inst.a));
		Integer::InitScalar(REGISTER(inst.c), 1);
		Integer::InitScalar(REGISTER(inst.c-1), v.length());
		return &inst+2;			// skip over following JMP
	}
}

static inline Instruction const* forend_op(Thread& thread, Instruction const& inst) {
	Value& counter = REGISTER(inst.c);
	Value& limit = REGISTER(inst.c-1);
	if(__builtin_expect(counter.i < limit.i, true)) {
		Value& b = REGISTER(inst.b);
		Element2(b, counter.i, thread.frame.environment->insert((String)inst.a));
		counter.i++;
		return &inst+(&inst+1)->a;
	} else {
		return &inst+2;			// skip over following JMP
	}
}

static inline Instruction const* dotslist_op(Thread& thread, Instruction const& inst) {
	PairList const& dots = thread.frame.environment->dots;
	
	Value& iter = REGISTER(inst.a);
	Value& out = OUT(c);
	
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
			((Object&)out).attributes(d);
		}
		return &inst+1;
	}
	
	// Loop on this instruction until done.
	return &inst;
}

static inline Instruction const* list_op(Thread& thread, Instruction const& inst) {
	Heap::Global.collect(thread.state);
	List out(inst.b);
	for(int64_t i = 0; i < inst.b; i++) {
		Value& r = REGISTER(inst.a-i);
		out[i] = r;
	}
	OUT(c) = out;
	return &inst+1;
}

// Memory access ops

static inline Instruction const* assign_op(Thread& thread, Instruction const& inst) {
	DECODE(c); FORCE(c); // don't BIND 
	thread.frame.environment->insert((String)inst.a) = c;
	return &inst+1;
}

static inline Instruction const* assign2_op(Thread& thread, Instruction const& inst) {
	// assign2 is always used to assign up at least one scope level...
	// so start off looking up one level...
	assert(thread.frame.environment->LexicalScope() != 0);
	
	DECODE(c); FORCE(c); /*BIND(value);*/
	
	String s = (String)inst.a;
	Environment* penv;
	Value& dest = thread.frame.environment->LexicalScope()->insertRecursive(s, penv);

	if(!dest.isNil()) {
		dest = c;
		thread.traces.LiveEnvironment(penv, dest);
	}
	else {
		Value& global = thread.state.global->insert(s);
		global = c;
		thread.traces.LiveEnvironment(thread.state.global, global);
	}
	return &inst+1;
}

static inline Instruction const* mov_op(Thread& thread, Instruction const& inst) {
	DECODE(a); FORCE(a); BIND(a);
	OUT(c) = a;
	return &inst+1;
}

static inline Instruction const* fastmov_op(Thread& thread, Instruction const& inst) {
	DECODE(a); FORCE(a); // fastmov assumes we don't need to bind. So next op better be able to handle a future 
	OUT(c) = a;
	return &inst+1;
}

static inline Instruction const* dotdot_op(Thread& thread, Instruction const& inst) {
	if(inst.a >= (int64_t)thread.frame.environment->dots.size())
        	_error(std::string("The '...' list does not contain ") + intToStr(inst.a+1) + " elements");
	DOTDOT(a, inst.a); FORCE_DOTDOT(a, inst.a); // no need to bind since value is in a register
	OUT(c) = a;
	return &inst+1;
}

static inline Instruction const* iassign_op(Thread& thread, Instruction const& inst) {
	// a = value, b = index, c = dest 
	DECODE(a); FORCE(a); 
	DECODE(b); FORCE(b); BIND(b); 
	DECODE(c); FORCE(c); BIND(c); 
	
	if(a.isFuture() && (c.isVector() || c.isFuture())) {
		if(b.isInteger() && ((Integer const&)b).length() == 1) {
			OUT(c) = thread.traces.EmitSStore(thread.frame.environment, c, ((Integer&)b)[0], a);
			return &inst+1;
		}
		else if(b.isDouble() && ((Double const&)b).length() == 1) {
			OUT(c) = thread.traces.EmitSStore(thread.frame.environment, c, ((Double&)b)[0], a);
			return &inst+1;
		}
	}

	BIND(a);
	SubsetAssign(thread, c, true, b, a, OUT(c));
	return &inst+1;
}
static inline Instruction const* eassign_op(Thread& thread, Instruction const& inst) {
	// a = value, b = index, c = dest
	DECODE(a); FORCE(a);
	DECODE(b); FORCE(b); BIND(b);
	DECODE(c); FORCE(c);

	if(a.isFuture() && (c.isVector() || c.isFuture())) {
		if(b.isInteger() && ((Integer const&)b).length() == 1) {
			OUT(c) = thread.traces.EmitSStore(thread.frame.environment, c, ((Integer&)b)[0], a);
			return &inst+1;
		}
		else if(b.isDouble() && ((Double const&)b).length() == 1) {
			OUT(c) = thread.traces.EmitSStore(thread.frame.environment, c, ((Double&)b)[0], a);
			return &inst+1;
		}
	}

	BIND(a);
	BIND(c);
	Subset2Assign(thread, c, true, b, a, OUT(c));
	return &inst+1; 
}

static inline Instruction const* subset_op(Thread& thread, Instruction const& inst) {
	DECODE(a); 
	DECODE(b);

	if(a.isVector()) {
		if(b.isDouble1()) { Element(a, b.d-1, OUT(c)); return &inst+1; }
		else if(b.isInteger1()) { Element(a, b.i-1, OUT(c)); return &inst+1; }
		else if(b.isLogical1()) { Element(a, Logical::isTrue(b.c) ? 0 : -1, OUT(c)); return &inst+1; }
		else if(b.isCharacter1()) { _error("Subscript out of bounds"); }
	}

	if(thread.state.jitEnabled 
		&& thread.traces.isTraceable(a, b) 
		&& thread.traces.futureType(b) == Type::Logical 
		&& thread.traces.futureShape(a) == thread.traces.futureShape(b)) {
		OUT(c) = thread.traces.EmitFilter(thread.frame.environment, a, b);
		thread.traces.OptBind(thread, OUT(c));
		return &inst+1;
	}

	FORCE(a); BIND(a);

	if(thread.traces.isTraceable(a, b) 
		&& (thread.traces.futureType(b) == Type::Integer 
			|| thread.traces.futureType(b) == Type::Double)) {
		OUT(c) = thread.traces.EmitGather(thread.frame.environment, a, b);
		thread.traces.OptBind(thread, OUT(c));
		return &inst+1;
	}

	if(((Object const&)a).hasAttributes()) { 
		return GenericDispatch(thread, inst, Strings::bracket, a, b, inst.c); 
	} 
	
	FORCE(b); BIND(b);

	if(((Object const&)b).hasAttributes()) { 
		return GenericDispatch(thread, inst, Strings::bracket, a, b, inst.c); 
	} 
	
	SubsetSlow(thread, a, b, OUT(c)); 
	return &inst+1;
}

static inline Instruction const* subset2_op(Thread& thread, Instruction const& inst) {
	DECODE(a); FORCE(a); BIND(a);
	DECODE(b);
	if(a.isVector()) {
		int64_t index = 0;
		if(b.isDouble1()) { index = b.d-1; }
		else if(b.isInteger1()) { index = b.i-1; }
		else if(b.isLogical1() && Logical::isTrue(b.c)) { index = 0; }
		else if(b.isVector() && (((Vector const&)b).length() == 0 || ((Vector const&)b).length() > 1)) { 
			_error("Attempt to select less or more than 1 element in subset2"); 
		}
		else { _error("Subscript out of bounds"); }
		Element2(a, index, OUT(c));
		return &inst+1;
	}
 	FORCE(b); BIND(b);
	if(((Object const&)a).hasAttributes() || ((Object const&)b).hasAttributes()) { return GenericDispatch(thread, inst, Strings::bb, a, b, inst.c); } 
	_error("Invalid subset2 operation");
}

static inline Instruction const* attrget_op(Thread& thread, Instruction const& inst) {
	DECODE(a); FORCE(a);
	DECODE(b); FORCE(b); BIND(b);
	if(a.isObject() && b.isCharacter1()) {
		String name = ((Character const&)b)[0];
		Object const& o = (Object const&)a;
		if(o.hasAttributes() && o.attributes()->has(name))
			OUT(c) = o.attributes()->get(name);
		else
			OUT(c) = Null::Singleton();
		return &inst+1;
	}
	_error("Invalid attrget operation");
}

static inline Instruction const* attrset_op(Thread& thread, Instruction const& inst) {
	DECODE(c); FORCE(c);
	DECODE(b); FORCE(b); BIND(b);
	DECODE(a); FORCE(a); BIND(a);
	if(c.isObject() && b.isCharacter1()) {
		String name = ((Character const&)b)[0];
		Object o = (Object const&)c;
		Dictionary* d = o.hasAttributes()
                	? o.attributes()->clone(1)
                	: new Dictionary(1);
		d->insert(name) = a;
		o.attributes(d);
		OUT(c) = o;
		return &inst+1;
	}
	_error("Invalid attrset operation");
}


#define OP(Name, string, Group, Func) \
static inline Instruction const* Name##_op(Thread& thread, Instruction const& inst) { \
	DECODE(a);	\
	Value & c = OUT(c);	\
	if(a.isDouble1())  { Name##VOp<Double>::Scalar(thread, a.d, c); return &inst+1; } \
	if(a.isInteger1()) { Name##VOp<Integer>::Scalar(thread, a.i, c); return &inst+1; } \
	if(a.isLogical1()) { Name##VOp<Logical>::Scalar(thread, a.c, c); return &inst+1; } \
	FORCE(a); \
	if(thread.traces.isTraceable<Group>(a)) { \
		c = thread.traces.EmitUnary<Group>(thread.frame.environment, IROpCode::Name, a, 0); \
		thread.traces.OptBind(thread, c); \
 		return &inst+1; \
	} \
	BIND(a); \
	if(((Object const&)a).hasAttributes()) { return GenericDispatch(thread, inst, Strings::Name, a, inst.c); } \
\
	Group##Dispatch<Name##VOp>(thread, a, c); \
	return &inst+1; \
}
UNARY_FOLD_SCAN_BYTECODES(OP)
#undef OP


#define OP(Name, string, Group, Func) \
static inline Instruction const* Name##_op(Thread& thread, Instruction const& inst) { \
	DECODE(a);	\
	DECODE(b);	\
	Value & c = OUT(c);	\
        if(__builtin_expect(a.isDouble1(),true)) {			\
		if(__builtin_expect(b.isDouble1(),true)) { Name##VOp<Double,Double>::Scalar(thread, a.d, b.d, c); return &inst+1; } \
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
	FORCE(a); FORCE(b); \
	if(thread.traces.isTraceable<Group>(a,b)) { \
		c = thread.traces.EmitBinary<Group>(thread.frame.environment, IROpCode::Name, a, b, 0); \
		thread.traces.OptBind(thread, c); \
		return &inst+1; \
	} \
	BIND(a); BIND(b); \
	if(((Object const&)a).hasAttributes() || ((Object const&)b).hasAttributes()) { return GenericDispatch(thread, inst, Strings::Name, a, b, inst.c); } \
\
	Group##Dispatch<Name##VOp>(thread, a, b, c);	\
	return &inst+1;	\
}
BINARY_BYTECODES(OP)
#undef OP

static inline Instruction const* length_op(Thread& thread, Instruction const& inst) {
	DECODE(a); FORCE(a); 
	if(a.isVector())
		Integer::InitScalar(OUT(c), ((Vector const&)a).length());
	else if(a.isFuture()) {
		IRNode::Shape shape = thread.traces.futureShape(a);
		if(shape.split < 0 && shape.filter < 0) {
			Integer::InitScalar(OUT(c), shape.length);
		} else {
			OUT(c) = thread.traces.EmitUnary<CountFold>(thread.frame.environment, IROpCode::length, a, 0);
			thread.traces.OptBind(thread, OUT(c));
		}
	}
	else if(((Object const&)a).hasAttributes()) { 
		return GenericDispatch(thread, inst, Strings::length, a, inst.c); 
	} else {
		Integer::InitScalar(OUT(c), 1);
	}
	return &inst+1;
}

static inline Instruction const* mean_op(Thread& thread, Instruction const& inst) {
	DECODE(a); FORCE(a); 
	if(thread.traces.isTraceable<MomentFold>(a)) {
		OUT(c) = thread.traces.EmitUnary<MomentFold>(thread.frame.environment, IROpCode::mean, a, 0);
		thread.traces.OptBind(thread, OUT(c));
 		return &inst+1;
	}
	return &inst+1;
}

static inline Instruction const* cm2_op(Thread& thread, Instruction const& inst) {
	DECODE(a); FORCE(a); 
	DECODE(b); FORCE(b); 
	if(thread.traces.isTraceable<Moment2Fold>(a,b)) {
		OUT(c) = thread.traces.EmitBinary<Moment2Fold>(thread.frame.environment, IROpCode::cm2, a, b, 0);
		thread.traces.OptBind(thread, OUT(c));
 		return &inst+1;
	}
	return &inst+1;
}

static inline Instruction const* ifelse_op(Thread& thread, Instruction const& inst) {
	DECODE(a); FORCE(a);
	DECODE(b); FORCE(b);
	DECODE(c); FORCE(c);
	if(c.isLogical1()) {
		OUT(c) = Logical::isTrue(c.c) ? b : a;
		return &inst+1; 
	}
	else if(c.isInteger1()) {
		OUT(c) = c.i ? b : a;
		return &inst+1; 
	}
	else if(c.isDouble1()) {
		OUT(c) = c.d ? b : a;
		return &inst+1; 
	}
	if(thread.traces.isTraceable<IfElse>(a,b,c)) {
		OUT(c) = thread.traces.EmitIfElse(thread.frame.environment, a, b, c);
		thread.traces.OptBind(thread, OUT(c));
		return &inst+1;
	}
	BIND(a); BIND(b); BIND(c);

	_error("ifelse not defined in scalar yet");
	return &inst+1; 
}

static inline Instruction const* split_op(Thread& thread, Instruction const& inst) {
	DECODE(a); FORCE(a); BIND(a);
	DECODE(b); FORCE(b);
	DECODE(c); FORCE(c);
	int64_t levels = As<Integer>(thread, a)[0];
	if(thread.traces.isTraceable<Split>(b,c)) {
		OUT(c) = thread.traces.EmitSplit(thread.frame.environment, c, b, levels);
		thread.traces.OptBind(thread, OUT(c));
		return &inst+1;
	}
	BIND(a); BIND(b); BIND(c);

	_error("split not defined in scalar yet");
	return &inst+1; 
}

static inline Instruction const* function_op(Thread& thread, Instruction const& inst) {
	Value const& function = CONSTANT(inst.a);
	Value& out = OUT(c);
	Function::Init(out, ((Function const&)function).prototype(), thread.frame.environment);
	return &inst+1;
}

static inline Instruction const* vector_op(Thread& thread, Instruction const& inst) {
	DECODE(a); FORCE(a); BIND(a);
	DECODE(b); FORCE(b); BIND(b);
	Type::Enum type = string2Type( As<Character>(thread, a)[0] );
	int64_t l = As<Integer>(thread, b)[0];
	
	// TODO: replace with isTraceable...
	if(thread.state.jitEnabled 
		&& (type == Type::Double || type == Type::Integer || type == Type::Logical)
		&& l >= TRACE_VECTOR_WIDTH) {
		OUT(c) = thread.traces.EmitConstant(thread.frame.environment, type, l, 0);
		thread.traces.OptBind(thread, OUT(c));
		return &inst+1;
	}

	if(type == Type::Logical) {
		Logical v(l);
		for(int64_t i = 0; i < l; i++) v[i] = Logical::FalseElement;
		OUT(c) = v;
	} else if(type == Type::Integer) {
		Integer v(l);
		for(int64_t i = 0; i < l; i++) v[i] = 0;
		OUT(c) = v;
	} else if(type == Type::Double) {
		Double v(l);
		for(int64_t i = 0; i < l; i++) v[i] = 0;
		OUT(c) = v;
	} else if(type == Type::Character) {
		Character v(l);
		for(int64_t i = 0; i < l; i++) v[i] = Strings::empty;
		OUT(c) = v;
	} else if(type == Type::Raw) {
		Raw v(l);
		for(int64_t i = 0; i < l; i++) v[i] = 0;
		OUT(c) = v;
	} else {
		_error("Invalid type in vector");
	} 
	return &inst+1;
}

static inline Instruction const* seq_op(Thread& thread, Instruction const& inst) {
	// c = start, b = step, a = length
	DECODE(a); FORCE(a); BIND(a);
	DECODE(b); FORCE(b); BIND(b);
	DECODE(c); FORCE(c); BIND(c);

	double start = As<Double>(thread, c)[0];
	double step = As<Double>(thread, b)[0];
	int64_t len = As<Integer>(thread, a)[0];
	
	if(len >= TRACE_VECTOR_WIDTH) {
		if(b.isDouble() || c.isDouble()) {
			OUT(c) = thread.traces.EmitSequence(thread.frame.environment, len, start, step);
			thread.traces.OptBind(thread, OUT(c));
		} else {
			OUT(c) = thread.traces.EmitSequence(thread.frame.environment, len, (int64_t)start, (int64_t)step);
			thread.traces.OptBind(thread, OUT(c));
		}
		return &inst+1;
	}

	if(b.isDouble() || c.isDouble())	
		OUT(c) = Sequence(start, step, len);
	else
		OUT(c) = Sequence((int64_t)start, (int64_t)step, len);
	return &inst+1;
}

static inline Instruction const* rep_op(Thread& thread, Instruction const& inst) {
	// c = n, b = each, a = length
	DECODE(a); FORCE(a); BIND(a);
	DECODE(b); FORCE(b); BIND(b);
	DECODE(c); FORCE(c); BIND(c);

	int64_t n = As<Integer>(thread, c)[0];
	int64_t each = As<Integer>(thread, b)[0];
	int64_t len = As<Integer>(thread, a)[0];
	
	if(len >= TRACE_VECTOR_WIDTH) {
		OUT(c) = thread.traces.EmitRepeat(thread.frame.environment, len, (int64_t)n, (int64_t)each);
		thread.traces.OptBind(thread, OUT(c));
		return &inst+1;
	}

	OUT(c) = Repeat((int64_t)n, (int64_t)each, len);
	return &inst+1;
}

static inline Instruction const* random_op(Thread& thread, Instruction const& inst) {
	DECODE(a); FORCE(a); BIND(a);

	int64_t len = As<Integer>(thread, a)[0];
	
	/*if(len >= TRACE_VECTOR_WIDTH) {
		OUT(c) = thread.EmitRandom(thread.frame.environment, len);
		thread.OptBind(OUT(c));
		return &inst+1;
	}*/

	OUT(c) = RandomVector(thread, len);
	return &inst+1;
}

static inline Instruction const* type_op(Thread& thread, Instruction const& inst) {
	DECODE(a); FORCE(a);
	switch(thread.traces.futureType(a)) {
                #define CASE(name, str, ...) case Type::name: OUT(c) = Character::c(Strings::name); break;
                TYPES(CASE)
                #undef CASE
                default: _error("Unknown type in type to string, that's bad!"); break;
        }
	return &inst+1;
}

static inline Instruction const* missing_op(Thread& thread, Instruction const& inst) {
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
	bool missing = v.isNil() || (v.isPromise() && ((Promise const&)v).isDefault());
	Logical::InitScalar(OUT(c), missing ? Logical::TrueElement : Logical::FalseElement);
	return &inst+1;
}
static inline Instruction const* strip_op(Thread& thread, Instruction const& inst) {
	DECODE(a); FORCE(a);
	Value& c = OUT(c);
	c = a;
	((Object&)c).attributes(0);
	return &inst+1;
}

static inline Instruction const* internal_op(Thread& thread, Instruction const& inst) {
	if(inst.a < 0)
		_error("Attempting to use undefined internal function");
	int64_t nargs = thread.state.internalFunctions[inst.a].params;
	for(int64_t i = 0; i < nargs; i++) {
		BIND(REGISTER(inst.b-i));
	}
	thread.state.internalFunctions[inst.a].ptr(thread, &REGISTER(inst.b), OUT(c));
	return &inst+1;
}

//
//    Main interpreter loop 
//
//__attribute__((__noinline__,__noclone__)) 
void interpret(Thread& thread, Instruction const* pc) {

#ifdef USE_THREADED_INTERPRETER
    	#define LABELS_THREADED(name,type,...) (void*)&&name##_label,
	static const void* labels[] = {BYTECODES(LABELS_THREADED)};

	goto *(void*)(labels[pc->bc]);
	#define LABELED_OP(name,type,...) \
		name##_label: \
			{ pc = name##_op(thread, *pc); goto *(void*)(labels[pc->bc]); } 
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

void State::interpreter_init(Thread& thread) {
	// nothing for now
}

Value Thread::eval(Prototype const* prototype) {
	return eval(prototype, frame.environment);
}

Value Thread::eval(Prototype const* prototype, Environment* environment) {
	uint64_t stackSize = stack.size();

	// make room for the result
	Instruction const* run = buildStackFrame(*this, environment, prototype, 0, (Instruction const*)0);
	try {
		interpret(*this, run);
		assert(stackSize == stack.size());
		return frame.registers[0];
	} catch(...) {
		stack.resize(stackSize);
		throw;
	}
}




const int64_t Random::primes[100] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43,
47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131,
137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223,
227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311,
313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409,
419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503,
509, 521, 523, 541};

Thread::Thread(State& state, uint64_t index) : state(state), index(index), random(index),steals(1) {
	registers = new Value[DEFAULT_NUM_REGISTERS];
	frame.registers = registers;
}

void Prototype::printByteCode(Prototype const* prototype, State const& state) {
	std::cout << "Prototype: " << intToHexStr((int64_t)prototype) << std::endl;
	std::cout << "\tRegisters: " << prototype->registers << std::endl;
	if(prototype->constants.size() > 0) {
		std::cout << "\tConstants: " << std::endl;
		for(int64_t i = 0; i < (int64_t)prototype->constants.size(); i++)
			std::cout << "\t\t" << i << ":\t" << state.stringify(prototype->constants[i]) << std::endl;
	}
	if(prototype->bc.size() > 0) {
		std::cout << "\tCode: " << std::endl;
		for(int64_t i = 0; i < (int64_t)prototype->bc.size(); i++) {
			std::cout << std::hex << &prototype->bc[i] << std::dec << "\t" << i << ":\t" << prototype->bc[i].toString();
			if(prototype->bc[i].bc == ByteCode::call || prototype->bc[i].bc == ByteCode::fastcall) {
				std::cout << "\t\t(arguments: " << prototype->calls[prototype->bc[i].b].arguments.size() << ")";
			}
			std::cout << std::endl;
		}
	}
	std::cout << std::endl;
}


