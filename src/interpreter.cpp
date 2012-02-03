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
#include "recording.h"
#include "compiler.h"
#include "sse.h"

static Instruction const* buildStackFrame(Thread& thread, Environment* environment, bool ownEnvironment, Prototype const* prototype, Value* result, Instruction const* returnpc);

extern Instruction const* kget_op(Thread& thread, Instruction const& inst) ALWAYS_INLINE;
extern Instruction const* get_op(Thread& thread, Instruction const& inst) ALWAYS_INLINE;
extern Instruction const* assign_op(Thread& thread, Instruction const& inst) ALWAYS_INLINE;
extern Instruction const* assign2_op(Thread& thread, Instruction const& inst) ALWAYS_INLINE;
extern Instruction const* forend_op(Thread& thread, Instruction const& inst) ALWAYS_INLINE;
extern Instruction const* add_op(Thread& thread, Instruction const& inst) ALWAYS_INLINE;
extern Instruction const* subset_op(Thread& thread, Instruction const& inst) ALWAYS_INLINE;

#define REG(thread, i) (*(thread.base+i))

static void ExpandDots(Thread& thread, List& arguments, Character& names, int64_t dots) {
	Environment* environment = thread.frame.environment;
	uint64_t dotslength = environment->dots.size();
	// Expand dots into the parameter list...
	if(dots < arguments.length) {
		List a(arguments.length + dotslength - 1);
		for(int64_t i = 0; i < dots; i++) a[i] = arguments[i];
		for(uint64_t i = dots; i < dots+dotslength; i++) { a[i] = Function(Compiler::compile(thread.state, CreateSymbol((String)-(i-dots+1))), NULL).AsPromise(); } // TODO: should cache these.
		for(uint64_t i = dots+dotslength; i < arguments.length+dotslength-1; i++) a[i] = arguments[i-dotslength];

		arguments = a;
		
		uint64_t named = 0;
		for(uint64_t i = 0; i < dotslength; i++) if(environment->dots[i] != Strings::empty) named++;

		if(names.length > 0 || named > 0) {
			Character n(arguments.length + dotslength - 1);
			for(int64_t i = 0; i < n.length; i++) n[i] = Strings::empty;
			if(names.length > 0) {
				for(int64_t i = 0; i < dots; i++) n[i] = names[i];
				for(uint64_t i = dots+dotslength; i < arguments.length+dotslength-1; i++) n[i] = names[i-dotslength];
			}
			if(named > 0) {
				for(uint64_t i = dots; i < dots+dotslength; i++) n[i] = environment->dots[i]; 
			}
			names = n;
		}
	}
}

inline void argAssign(Environment* env, String n, Value const& v, Environment* execution) {
	Value w = v;
	if(w.isPromise() && w.p == 0) w.p = execution;
	env->assign(n, w);
}

static void MatchArgs(Thread& thread, Environment* env, Environment* fenv, Function const& func, List const& arguments, Character const& anames) {
	List const& defaults = func.prototype()->defaults;
	Character const& parameters = func.prototype()->parameters;
	int64_t fdots = func.prototype()->dots;

	// set defaults
	for(int64_t i = 0; i < defaults.length; ++i) {
		argAssign(fenv, parameters[i], defaults[i], fenv);
	}

	// call arguments are not named, do posititional matching
	if(anames.length == 0) {
		int64_t end = std::min(arguments.length, fdots);
		for(int64_t i = 0; i < end; ++i) {
			if(!arguments[i].isNil()) argAssign(fenv, parameters[i], arguments[i], env);
		}

		// set dots if necessary
		if(fdots < parameters.length) {
			int64_t idx = 1;
			for(int64_t i = fdots; i < arguments.length; i++) {
				argAssign(fenv, (String)-idx, arguments[i], env);
				fenv->dots.push_back(Strings::empty);
				idx++;
			}
			end++;
		}
	}
	// call arguments are named, do matching by name
	else {
		// we should be able to cache and reuse this assignment for pairs of functions and call sites.
		int64_t *assignment = thread.assignment, *set = thread.set;
		for(int64_t i = 0; i < arguments.length; i++) assignment[i] = -1;
		for(int64_t i = 0; i < parameters.length; i++) set[i] = -(i+1);

		// named args, search for complete matches
		for(int64_t i = 0; i < arguments.length; ++i) {
			if(anames[i] != Strings::empty) {
				for(int64_t j = 0; j < parameters.length; ++j) {
					if(j != fdots && anames[i] == parameters[j]) {
						assignment[i] = j;
						set[j] = i;
						break;
					}
				}
			}
		}
		// named args, search for incomplete matches
		for(int64_t i = 0; i < arguments.length; ++i) {
			if(anames[i] != Strings::empty && assignment[i] < 0) {
				std::string a = thread.externStr(anames[i]);
				for(int64_t j = 0; j < parameters.length; ++j) {
					if(set[j] < 0 && j != fdots &&
							thread.externStr(parameters[j]).compare( 0, a.size(), a ) == 0 ) {	
						assignment[i] = j;
						set[j] = i;
						break;
					}
				}
			}
		}
		// unnamed args, fill into first missing spot.
		int64_t firstEmpty = 0;
		for(int64_t i = 0; i < arguments.length; ++i) {
			if(anames[i] == Strings::empty) {
				for(; firstEmpty < fdots; ++firstEmpty) {
					if(set[firstEmpty] < 0) {
						assignment[i] = firstEmpty;
						set[firstEmpty] = i;
						break;
					}
				}
			}
		}

		// stuff that can't be cached...

		// assign all the arguments
		for(int64_t j = 0; j < parameters.length; ++j) 
			if(j != fdots && set[j] >= 0 && !arguments[set[j]].isNil()) 
				argAssign(fenv, parameters[j], arguments[set[j]], env);

		// put unused args into the dots
		if(fdots < parameters.length) {
			int64_t idx = 1;
			for(int64_t i = 0; i < arguments.length; i++) {
				if(assignment[i] < 0) {
					argAssign(fenv, (String)-idx, arguments[i], env);
					fenv->dots.push_back(anames[i]);
					idx++;
				}
			}
		}
	}
}

static Environment* CreateEnvironment(Thread& thread, Environment* l, Environment* d, Value const& call) {
	Environment* env;
	if(thread.environments.size() == 0) {
		env = new Environment();
	} else {
		env = thread.environments.back();
		thread.environments.pop_back();
	}
	env->init(l, d, call);
	return env;
}
//track the heat of back edge operations and invoke the recorder on hot traces
//unused until we begin tracing loops again
static Instruction const * profile_back_edge(Thread & thread, Instruction const * inst) {
	return inst;
}

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

Instruction const* apply_op(Thread& thread, Instruction const& inst) {
	return &inst+1;
}

Instruction const* internal_op(Thread& thread, Instruction const& inst) {
       thread.state.internalFunctions[inst.a].ptr(thread, &REG(thread, inst.b), REG(thread, inst.c));
       return &inst+1;
}

// Get a Value by Symbol from the current environment,
//  TODO: UseMethod also should search in some cached library locations.
static Value GenericGet(Thread& thread, String s) {
	Environment* environment = thread.frame.environment;
	Value value = environment->get(s);
	while(value.isNil() && environment->LexicalScope() != 0) {
		environment = environment->LexicalScope();
		value = environment->get(s);
	}
	if(value.isPromise()) {
		//value = force(thread, value);
		//environment->assign(s, value);
		_error("UseMethod does not yet support evaluating promises");
	}
	return value;
}

static Value GenericSearch(Thread& thread, Character klass, String generic, String& method) {
		
	// first search for type specific method
	Value func = Value::Nil();
	for(int64_t i = 0; i < klass.length && func.isNil(); i++) {
		method = thread.internStr(thread.externStr(generic) + "." + thread.externStr(klass[i]));
		func = GenericGet(thread, method);	
	}

	// TODO: look for group generics

	// look for default if necessary
	if(func.isNil()) {
		method = thread.internStr(thread.externStr(generic) + ".default");
		func = GenericGet(thread, method);
	}

	return func;
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
	Subset(thread, REG(thread, inst.a), REG(thread, inst.b), REG(thread, inst.c));
	return &inst+1;
}
Instruction const* subset2_op(Thread& thread, Instruction const& inst) {
	Subset2(thread, REG(thread, inst.a), REG(thread, inst.b), REG(thread, inst.c));
	return &inst+1;
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
/*Instruction const* iforbegin_op(Thread& thread, Instruction const& inst) {
	double m = asReal1(REG(thread, inst.c-1));
	double n = asReal1(REG(thread, inst.c));
	REG(thread, inst.c-1) = Integer::c(n > m ? 1 : -1);
	REG(thread, inst.c-1).length = (int64_t)n+1;	// danger! this register no longer holds a valid object, but it saves a register and makes the for and ifor cases more similar
	REG(thread, inst.c) = Integer::c((int64_t)m);
	if(REG(thread, inst.c).i >= (int64_t)REG(thread, inst.c-1).length) { return &inst+inst.a; }
	thread.frame.environment->hassign(Symbol(inst.b), Integer::c(m));
	REG(thread, inst.c).i += REG(thread, inst.c-1).i;
	return &inst+1;
}
Instruction const* iforend_op(Thread& thread, Instruction const& inst) {
	if(REG(thread, inst.c).i < REG(thread, inst.c-1).length) { 
		thread.frame.environment->hassign(Symbol(inst.b), REG(thread, inst.c));
		REG(thread, inst.c).i += REG(thread, inst.c-1).i;
		return profile_back_edge(thread,&inst+inst.a);
	} else return &inst+1;
}*/
Instruction const* jt_op(Thread& thread, Instruction const& inst) {
	Logical l = As<Logical>(thread, REG(thread,inst.b));
	if(l.length == 0) _error("condition is of zero length");
	if(l[0]) return &inst+inst.a;
	else return &inst+1;
}
Instruction const* jf_op(Thread& thread, Instruction const& inst) {
	Logical l = As<Logical>(thread, REG(thread, inst.b));
	if(l.length == 0) _error("condition is of zero length");
	if(l[0]) return &inst+1;
	else return &inst+inst.a;
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
Instruction const* colon_op(Thread& thread, Instruction const& inst) {
	double from = asReal1(REG(thread,inst.a));
	double to = asReal1(REG(thread,inst.b));
	REG(thread,inst.c) = Sequence(from, to>from?1:-1, fabs(to-from)+1);
	return &inst+1;
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

Instruction const* seq_op(Thread& thread, Instruction const& inst) {
	int64_t len = As<Integer>(thread, REG(thread, inst.a))[0];
	int64_t step = As<Integer>(thread, REG(thread, inst.b))[0];
	
	Instruction const* jit = trace(thread, inst, Type::Integer, len);
	if(jit) return jit;
	
	REG(thread, inst.c) = Sequence(len, 1, step);
	return &inst+1;
}

#define OP(name, string, Op, Func) \
Instruction const* name##_op(Thread& thread, Instruction const& inst) { \
	Value & a =  REG(thread, inst.a);	\
	Value & c = REG(thread, inst.c);	\
	if(a.isDouble1()) { Op<TDouble>::RV::InitScalar(c, Op<TDouble>::eval(thread, a.d)); return &inst+1; } \
	else if(a.isInteger1()) { Op<TDouble>::RV::InitScalar(c, Op<TInteger>::eval(thread, a.i)); return &inst+1; } \
	if(a.isObject() ) { \
		String method; \
		Value f = GenericSearch(thread, klass(thread, a), thread.internStr(Func), method); \
		if(f.isFunction()) { \
			Function func(f); \
			Environment* fenv = CreateEnvironment(thread, func.environment(), thread.frame.environment, Null::Singleton()); \
			MatchArgs(thread, thread.frame.environment, fenv, func, List::c(a), Character(0)); \
			return buildStackFrame(thread, fenv, true, func.prototype(), &REG(thread, inst.c), &inst+1); \
		}	\
	} \
	Instruction const* jit = trace(thread, inst, a); \
	if(jit) return jit; \
	\
	unaryArith<Zip1, Op>(thread, a, c); \
	return &inst+1; \
}
UNARY_ARITH_MAP_BYTECODES(OP)
#undef OP


#define OP(name, string, Op, Func) \
Instruction const* name##_op(Thread& thread, Instruction const& inst) { \
	Value & a = REG(thread,inst.a); \
	\
	Instruction const* jit = trace(thread, inst, a); \
	if(jit) return jit; \
	\
	unaryLogical<Zip1, Op>(thread, a, REG(thread, inst.c)); \
	return &inst+1; \
}
UNARY_LOGICAL_MAP_BYTECODES(OP)
#undef OP

#define OP(name, string, Op, Func) \
Instruction const* name##_op(Thread& thread, Instruction const& inst) { \
	Value & a =  REG(thread, inst.a);	\
	Value & b =  REG(thread, inst.b);	\
	Value & c = REG(thread, inst.c);	\
        if(a.isDouble1()) {			\
                if(b.isDouble1())		\
                        { Op<TDouble>::RV::InitScalar(c, Op<TDouble>::eval(thread, a.d, b.d)); return &inst+1; }	\
                else if(b.isInteger1())	\
                        { Op<TDouble>::RV::InitScalar(c, Op<TDouble>::eval(thread, a.d, (double)b.i));return &inst+1; }	\
        }	\
        else if(a.isInteger1()) {	\
                if(b.isDouble1())	\
                        { Op<TDouble>::RV::InitScalar(c, Op<TDouble>::eval(thread, (double)a.i, b.d)); return &inst+1; }	\
                else if(b.isInteger1())	\
                        { Op<TInteger>::RV::InitScalar(c, Op<TInteger>::eval(thread, a.i, b.i)); return &inst+1;} \
        } \
	if(a.isObject() || b.isObject()) { \
		Value f; \
		String method; \
		if(a.isObject() && b.isObject()) { \
			Value af = GenericSearch(thread, klass(thread, a), thread.internStr(Func), method); \
			Value bf = GenericSearch(thread, klass(thread, b), thread.internStr(Func), method); \
			if(af != bf) _error("Generic functions do not match on binary operator"); \
			f = af; \
		} \
		else if(a.isObject()) f = GenericSearch(thread, klass(thread, a), thread.internStr(Func), method); \
		else f = GenericSearch(thread, klass(thread, b), thread.internStr(Func), method); \
		if(f.isFunction()) { \
			Function func(f); \
			Environment* fenv = CreateEnvironment(thread, func.environment(), thread.frame.environment, Null::Singleton()); \
			MatchArgs(thread, thread.frame.environment, fenv, func, List::c(a, b), Character(0)); \
			return buildStackFrame(thread, fenv, true, func.prototype(), &REG(thread, inst.c), &inst+1); \
		}	\
	} \
	\
	Instruction const* jit = trace(thread, inst, a, b); \
	if(jit) return jit; \
\
	binaryArithSlow<Zip2, Op>(thread, a, b, c);	\
	return &inst+1;	\
}
BINARY_ARITH_MAP_BYTECODES(OP)
#undef OP

#define OP(name, string, Op, Func) \
Instruction const* name##_op(Thread& thread, Instruction const& inst) { \
	Value & a = REG(thread,inst.a); \
	Value & b = REG(thread,inst.b); \
	\
	Instruction const* jit = trace(thread, inst, a, b); \
	if(jit) return jit; \
	\
	binaryLogical<Zip2, Op>(thread, a, b, REG(thread, inst.c)); \
	return &inst+1; \
}
BINARY_LOGICAL_MAP_BYTECODES(OP)
#undef OP

#define OP(name, string, Op, Func) \
Instruction const* name##_op(Thread& thread, Instruction const& inst) { \
	Value & a =  REG(thread, inst.a);	\
	Value & b =  REG(thread, inst.b);	\
	Value & c = REG(thread, inst.c);	\
        if(a.isDouble1()) {			\
                if(b.isDouble1())		\
                        { Op<TDouble>::RV::InitScalar(c, Op<TDouble>::eval(thread, a.d, b.d)); return &inst+1; }	\
                else if(b.isInteger1())	\
                        { Op<TDouble>::RV::InitScalar(c, Op<TDouble>::eval(thread, a.d, (double)b.i));return &inst+1; }	\
        }	\
        else if(a.isInteger1()) {	\
                if(b.isDouble1())	\
                        { Op<TDouble>::RV::InitScalar(c, Op<TDouble>::eval(thread, (double)a.i, b.d)); return &inst+1; }	\
                else if(b.isInteger1())	\
                        { Op<TInteger>::RV::InitScalar(c, Op<TInteger>::eval(thread, a.i, b.i)); return &inst+1;} \
        } \
	\
	Instruction const* jit = trace(thread, inst, a, b); \
	if(jit) return jit; \
	\
	binaryOrdinal<Zip2, Op>(thread, REG(thread, inst.a), REG(thread, inst.b), REG(thread, inst.c)); \
	return &inst+1; \
}
BINARY_ORDINAL_MAP_BYTECODES(OP)
#undef OP

#define OP(name, string, Op, Func) \
Instruction const* name##_op(Thread& thread, Instruction const& inst) { \
	unaryArith<FoldLeft, Op>(thread, REG(thread, inst.a), REG(thread, inst.c)); \
	return &inst+1; \
}
ARITH_FOLD_BYTECODES(OP)
#undef OP

#define OP(name, string, Op, Func) \
Instruction const* name##_op(Thread& thread, Instruction const& inst) { \
	unaryLogical<FoldLeft, Op>(thread, REG(thread, inst.a), REG(thread, inst.c)); \
	return &inst+1; \
}
LOGICAL_FOLD_BYTECODES(OP)
#undef OP

#define OP(name, string, Op, Func) \
Instruction const* name##_op(Thread& thread, Instruction const& inst) { \
	unaryOrdinal<FoldLeft, Op>(thread, REG(thread, inst.a), REG(thread, inst.c)); \
	return &inst+1; \
}
ORDINAL_FOLD_BYTECODES(OP)
#undef OP

#define OP(name, string, Op, Func) \
Instruction const* name##_op(Thread& thread, Instruction const& inst) { \
	unaryArith<ScanLeft, Op>(thread, REG(thread, inst.a), REG(thread, inst.c)); \
	return &inst+1; \
}
ARITH_SCAN_BYTECODES(OP)
#undef OP

#define OP(name, string, Op, Func) \
Instruction const* name##_op(Thread& thread, Instruction const& inst) { \
	unaryLogical<ScanLeft, Op>(thread, REG(thread, inst.a), REG(thread, inst.c)); \
	return &inst+1; \
}
LOGICAL_SCAN_BYTECODES(OP)
#undef OP

#define OP(name, string, Op, Func) \
Instruction const* name##_op(Thread& thread, Instruction const& inst) { \
	unaryOrdinal<ScanLeft, Op>(thread, REG(thread, inst.a), REG(thread, inst.c)); \
	return &inst+1; \
}
ORDINAL_SCAN_BYTECODES(OP)
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

Instruction const* jmp_op(Thread& thread, Instruction const& inst) {
	return &inst+inst.a;
}
Instruction const* sland_op(Thread& thread, Instruction const& inst) {
	Logical l = As<Logical>(thread, REG(thread, inst.a));
	if(l.length == 0) _error("argument to && is zero length");
	if(Logical::isFalse(l[0])) {
		REG(thread, inst.c) = Logical::False();
		return &inst+1;
	} else {
		Logical r = As<Logical>(thread, REG(thread, inst.b));
		if(r.length == 0) _error("argument to && is zero length");
		if(Logical::isFalse(r[0])) REG(thread, inst.c) = Logical::False();
		else if(Logical::isNA(l[0]) || Logical::isNA(r[0])) REG(thread, inst.c) = Logical::NA();
		else REG(thread, inst.c) = Logical::True();
		return &inst+1;
	}
}
Instruction const* slor_op(Thread& thread, Instruction const& inst) {
	Logical l = As<Logical>(thread, REG(thread, inst.a));
	if(l.length == 0) _error("argument to || is zero length");
	if(Logical::isTrue(l[0])) {
		REG(thread, inst.c) = Logical::True();
		return &inst+1;
	} else {
		Logical r = As<Logical>(thread, REG(thread, inst.b));
		if(r.length == 0) _error("argument to || is zero length");
		if(Logical::isTrue(r[0])) REG(thread, inst.c) = Logical::True();
		else if(Logical::isNA(l[0]) || Logical::isNA(r[0])) REG(thread, inst.c) = Logical::NA();
		else REG(thread, inst.c) = Logical::False();
		return &inst+1;
	}
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

#ifdef USE_THREADED_INTERPRETER
static const void** glabels = 0;
#endif

static Instruction const* buildStackFrame(Thread& thread, Environment* environment, bool ownEnvironment, Prototype const* prototype, Value* result, Instruction const* returnpc) {
	//printCode(thread, prototype);
	StackFrame& s = thread.push();
	s.environment = environment;
	s.ownEnvironment = ownEnvironment;
	s.returnpc = returnpc;
	s.returnbase = thread.base;
	s.result = result;
	s.prototype = prototype;
	thread.base -= prototype->registers;
	if(thread.base < thread.registers)
		throw RiposteError("Register overflow");

#ifdef USE_THREADED_INTERPRETER
	// Initialize threaded bytecode if not yet done 
	if(prototype->bc[0].ibc == 0)
	{
		for(int64_t i = 0; i < (int64_t)prototype->bc.size(); ++i) {
			Instruction const& inst = prototype->bc[i];
			inst.ibc = glabels[inst.bc];
		}
	}
#endif
	return &(prototype->bc[0]);
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

