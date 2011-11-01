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

#define USE_THREADED_INTERPRETER
#define ALWAYS_INLINE __attribute__((always_inline))

static Instruction const* buildStackFrame(State& state, Environment* environment, bool ownEnvironment, Prototype const* prototype, Value* result, Instruction const* returnpc);

#ifndef __ICC
extern Instruction const* kget_op(State& state, Instruction const& inst) ALWAYS_INLINE;
extern Instruction const* get_op(State& state, Instruction const& inst) ALWAYS_INLINE;
extern Instruction const* assign_op(State& state, Instruction const& inst) ALWAYS_INLINE;
extern Instruction const* assign2_op(State& state, Instruction const& inst) ALWAYS_INLINE;
extern Instruction const* forend_op(State& state, Instruction const& inst) ALWAYS_INLINE;
extern Instruction const* add_op(State& state, Instruction const& inst) ALWAYS_INLINE;
extern Instruction const* subset_op(State& state, Instruction const& inst) ALWAYS_INLINE;
#endif

#define REG(state, i) (*(state.base+i))

static void ExpandDots(State& state, List& arguments, Character& names, int64_t dots) {
	Environment* environment = state.frame.environment;
	uint64_t dotslength = environment->dots.size();
	// Expand dots into the parameter list...
	if(dots < arguments.length) {
		List a(arguments.length + dotslength - 1);
		for(int64_t i = 0; i < dots; i++) a[i] = arguments[i];
		for(uint64_t i = dots; i < dots+dotslength; i++) { a[i] = Function(Compiler::compile(state, Symbol(String::Init(-(i-dots+1)))), NULL).AsPromise(); } // TODO: should cache these.
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

inline void argAssign(Environment* env, int64_t i, Value const& v, Environment* execution, Character const& parameters) {
	Value w = v;
	if(w.isPromise() && w.p == 0) w.p = execution;
	if(i >= 0)
		env->assign(parameters[i], w);
	else {
		env->assign(String::Init(i), w);
	}
}

static void MatchArgs(State& state, Environment* env, Environment* fenv, Function const& func, List const& arguments, Character const& anames) {
	List const& defaults = func.prototype()->defaults;
	Character const& parameters = func.prototype()->parameters;
	int64_t fdots = func.prototype()->dots;

	// set defaults
	for(int64_t i = 0; i < defaults.length; ++i) {
		argAssign(fenv, i, defaults[i], fenv, parameters);
	}

	// call arguments are not named, do posititional matching
	if(anames.length == 0) {
		int64_t end = std::min(arguments.length, fdots);
		for(int64_t i = 0; i < end; ++i) {
			if(!arguments[i].isNil()) argAssign(fenv, i, arguments[i], env, parameters);
		}

		// set dots if necessary
		if(fdots < parameters.length) {
			int64_t idx = 1;
			for(int64_t i = fdots; i < arguments.length; i++) {
				argAssign(fenv, -idx, arguments[i], env, parameters);
				fenv->dots.push_back(Strings::empty);
				idx++;
			}
			end++;
		}
	}
	// call arguments are named, do matching by name
	else {
		// we should be able to cache and reuse this assignment for pairs of functions and call sites.
		static char assignment[64], set[64];
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
				std::string a = state.externStr(anames[i]);
				for(int64_t j = 0; j < parameters.length; ++j) {
					if(set[j] < 0 && j != fdots &&
							state.externStr(parameters[j]).compare( 0, a.size(), a ) == 0 ) {	
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
		for(int64_t j = 0; j < parameters.length; ++j) if(j != fdots && set[j] >= 0 && !arguments[set[j]].isNil()) argAssign(fenv, j, arguments[set[j]], env, parameters);

		// put unused args into the dots
		if(fdots < parameters.length) {
			int64_t idx = 1;
			for(int64_t i = 0; i < arguments.length; i++) {
				if(assignment[i] < 0) {
					argAssign(fenv, -idx, arguments[i], env, parameters);
					fenv->dots.push_back(anames[i]);
					idx++;
				}
			}
		}
	}
}

static Environment* CreateEnvironment(State& state, Environment* l, Environment* d, Value const& call) {
	Environment* env;
	if(state.environments.size() == 0) {
		env = new Environment();
	} else {
		env = state.environments.back();
		state.environments.pop_back();
	}
	env->init(l, d, call);
	return env;
}
//track the heat of back edge operations and invoke the recorder on hot traces
//unused until we begin tracing loops again
static Instruction const * profile_back_edge(State & state, Instruction const * inst) {
	return inst;
}

Instruction const* call_op(State& state, Instruction const& inst) {
	Value f = REG(state, inst.a);
	if(!f.isFunction())
		_error(std::string("Non-function (") + Type::toString(f.type) + ") as first parameter to call\n");
	Function func(f);
	
	// TODO: using inst.b < 0 to indicate a normal call means that do.call can never use a ..# variable. Not common, but would surely be unexpected for users. Probably best to just have a separate op for do.call?
	
	List arguments;
	Character names;
	Environment* fenv;
	if(inst.b < 0) {
		CompiledCall const& call = state.frame.prototype->calls[-(inst.b+1)];
		arguments = call.arguments;
		names = call.names;
		if(call.dots < arguments.length)
			ExpandDots(state, arguments, names, call.dots);
		fenv = CreateEnvironment(state, func.environment(), state.frame.environment, call.call);
	} else {
		Value const& reg = REG(state, inst.b);
		if(reg.isObject()) {
			arguments = List(((Object const&)reg).base());
			names = Character(((Object const&)reg).getNames());
		}
		else {
			arguments = List(reg);
		}
		fenv = CreateEnvironment(state, func.environment(), state.frame.environment, Null::Singleton());
	}

	MatchArgs(state, state.frame.environment, fenv, func, arguments, names);
	return buildStackFrame(state, fenv, true, func.prototype(), &REG(state, inst.c), &inst+1);
}

Instruction const* icall_op(State& state, Instruction const& inst) {
	state.internalFunctions[inst.a].ptr(state, &REG(state, inst.b), REG(state, inst.c));
	return &inst+1;
}

// Get a Value by Symbol from the current environment,
//  TODO: UseMethod also should search in some cached library locations.
static Value GenericGet(State& state, String s) {
	Environment* environment = state.frame.environment;
	Value value = environment->get(s);
	while(value.isNil() && environment->LexicalScope() != 0) {
		environment = environment->LexicalScope();
		value = environment->get(s);
	}
	if(value.isPromise()) {
		//value = force(state, value);
		//environment->assign(s, value);
		_error("UseMethod does not yet support evaluating promises");
	}
	return value;
}

static Value GenericSearch(State& state, Character klass, String generic, String& method) {
	
	// first search for type specific method
	Value func = Value::Nil();
	for(int64_t i = 0; i < klass.length && func.isNil(); i++) {
		method = state.internStr(state.externStr(generic) + "." + state.externStr(klass[i]));
		func = GenericGet(state, method);	
	}

	// TODO: look for group generics

	// look for default if necessary
	if(func.isNil()) {
		method = state.internStr(state.externStr(generic) + ".default");
		func = GenericGet(state, method);
	}

	return func;
}

Instruction const* UseMethod_op(State& state, Instruction const& inst) {
	String generic = String::Init(inst.a);

	CompiledCall const& call = state.frame.prototype->calls[inst.b];
	List arguments = call.arguments;
	Character names = call.names;
	if(call.dots < arguments.length)
		ExpandDots(state, arguments, names, call.dots);

	Value object = REG(state, inst.c);
	Character type = klass(state, object);

	String method;
	Value f = GenericSearch(state, type, generic, method);

	if(!f.isFunction()) { 
		_error(std::string("no applicable method for '") + state.externStr(generic) + "' applied to an object of class \"" + state.externStr(type[0]) + "\"");
	}

	Function func(f);
	Environment* fenv = CreateEnvironment(state, func.environment(), state.frame.environment, call.call);
	MatchArgs(state, state.frame.environment, fenv, func, arguments, names);	
	fenv->assign(Strings::dotGeneric, Symbol(generic));
	fenv->assign(Strings::dotMethod, Symbol(method));
	fenv->assign(Strings::dotClass, type); 
	return buildStackFrame(state, fenv, true, func.prototype(), &REG(state, inst.c), &inst+1);
}

Instruction const* get_op(State& state, Instruction const& inst) {
	// gets are always generated as a sequence of 3 instructions...
	//	1) the get with source symbol in a and dest register in c.
	//	2) an assign with dest symbol in a and source register in c.
	//		(for use by the promise evaluation. If no promise, step over this instruction.)
	//	3) an invalid instruction containing inline caching info.	

	// check if we can get the value through inline caching...
	uint64_t icRevision = (&inst+2)->b;

	if(__builtin_expect(state.frame.environment->equalRevision(icRevision), true)) {
		REG(state, inst.c) = state.frame.environment->get((&inst+2)->a);
		return &inst+3;
	}

	// otherwise, need to do a real look up starting from env
	Environment* env = state.frame.environment;
	String s = String::Init(inst.a);
	
	Value& dest = REG(state, inst.c);
	dest = env->get(s);
	while(dest.isNil() && env->LexicalScope() != 0) {
		env = env->LexicalScope();
		dest = env->get(s);
	}

	if(dest.isConcrete()) {
		Environment::Pointer p = env->makePointer(s);
		((Instruction*)(&inst+2))->a = p.index;
		((Instruction*)(&inst+2))->b = p.revision;
		return &inst+3;
	} else if(dest.isPromise()) {
		Environment* env = Function(dest).environment();
		Prototype* prototype = Function(dest).prototype();
		assert(env != 0);
		// the inline cache info will be populated by the assignment instruction
		return buildStackFrame(state, env, false, prototype, &dest, &inst+1);
	}
	else
		throw RiposteError(std::string("object '") + state.externStr(s) + "' not found");
}

Instruction const* kget_op(State& state, Instruction const& inst) {
	REG(state, inst.c) = state.frame.prototype->constants[inst.a];
	return &inst+1;
}
Instruction const* iget_op(State& state, Instruction const& inst) {
	REG(state, inst.c) = state.path[0]->get(inst.a);
	if(REG(state, inst.c).isNil()) throw RiposteError(std::string("object '") + state.externStr(String::Init(inst.a)) + "' not found");
	return &inst+1;
}
Instruction const* assign_op(State& state, Instruction const& inst) {
	// check if we can assign through inline caching
	if(state.frame.environment->equalRevision((&inst+1)->b))
		state.frame.environment->assign((&inst+1)->a, REG(state, inst.c));
	else {
		state.frame.environment->assign(String::Init(inst.a), REG(state, inst.c));
		
		Environment::Pointer p = state.frame.environment->makePointer(String::Init(inst.a));
		((Instruction*)(&inst+1))->a = p.index;
		((Instruction*)(&inst+1))->b = p.revision;
		((Instruction*)(&inst+1))->c = 0;
	}
	return &inst+2;
}
Instruction const* assign2_op(State& state, Instruction const& inst) {
	// check if we can assign the value through inline caching...
	// assign2 is always used to assign up at least one scope level...
	// so start off looking up one level...

	Environment* env = state.frame.environment->LexicalScope();
	assert(env != 0);

	uint64_t icRevision = (&inst+1)->b;
	if(__builtin_expect(env->equalRevision(icRevision), true)) {
		env->assign((&inst+1)->a, REG(state, inst.c));
		return &inst+2;
	}

	// otherwise, need to do a real look up starting from env
	String s = String::Init(inst.a);
	Value dest = env->get(s);
	icRevision = std::max(icRevision, env->getRevision());
	while(dest.isNil() && env->LexicalScope() != 0) {
		env = env->LexicalScope();
		dest = env->get(s);
	}

	if(!dest.isNil()) {
		env->assign(s, REG(state, inst.c));
		Environment::Pointer p = env->makePointer(String::Init(inst.a));
		((Instruction*)(&inst+1))->a = p.index;
		((Instruction*)(&inst+1))->b = p.revision;
	}
	else {
		state.global->assign(s, REG(state, inst.c));
		// Global may not be in the static scope of function so we can't populate IC here.
		// However, if global is in the static scope, the second iteration of assign2 will find it on the path and set the IC in the immediately previous if block.
	}
	return &inst+2;
}


// everything else should be in registers

Instruction const* iassign_op(State& state, Instruction const& inst) {
	// a = value, b = index, c = dest 
	SubsetAssign(state, REG(state,inst.c), true, REG(state,inst.b), REG(state,inst.a), REG(state,inst.c));
	return &inst+1;
}
Instruction const* eassign_op(State& state, Instruction const& inst) {
	// a = value, b = index, c = dest
	Subset2Assign(state, REG(state,inst.c), true, REG(state,inst.b), REG(state,inst.a), REG(state,inst.c));
	return &inst+1; 
}
Instruction const* subset_op(State& state, Instruction const& inst) {
	Subset(state, REG(state, inst.a), REG(state, inst.b), REG(state, inst.c));
	return &inst+1;
}
Instruction const* subset2_op(State& state, Instruction const& inst) {
	Subset2(state, REG(state, inst.a), REG(state, inst.b), REG(state, inst.c));
	return &inst+1;
}
Instruction const* forbegin_op(State& state, Instruction const& inst) {
	// inst.b-1 holds the loopVector
	if((int64_t)REG(state, inst.b-1).length <= 0) { return &inst+inst.a; }
	Element2(REG(state, inst.b-1), 0, REG(state, inst.c));
	REG(state, inst.b).header = REG(state, inst.b-1).length;	// warning: not a valid object, but saves a shift
	REG(state, inst.b).i = 1;
	return &inst+1;
}
Instruction const* forend_op(State& state, Instruction const& inst) {
	if(__builtin_expect((REG(state,inst.b).i) < REG(state,inst.b).header, true)) {
		Element2(REG(state, inst.b-1), REG(state, inst.b).i, REG(state, inst.c));
		REG(state, inst.b).i++;
		return profile_back_edge(state,&inst+inst.a);
	} else return &inst+1;
}
/*Instruction const* iforbegin_op(State& state, Instruction const& inst) {
	double m = asReal1(REG(state, inst.c-1));
	double n = asReal1(REG(state, inst.c));
	REG(state, inst.c-1) = Integer::c(n > m ? 1 : -1);
	REG(state, inst.c-1).length = (int64_t)n+1;	// danger! this register no longer holds a valid object, but it saves a register and makes the for and ifor cases more similar
	REG(state, inst.c) = Integer::c((int64_t)m);
	if(REG(state, inst.c).i >= (int64_t)REG(state, inst.c-1).length) { return &inst+inst.a; }
	state.frame.environment->hassign(Symbol(inst.b), Integer::c(m));
	REG(state, inst.c).i += REG(state, inst.c-1).i;
	return &inst+1;
}
Instruction const* iforend_op(State& state, Instruction const& inst) {
	if(REG(state, inst.c).i < REG(state, inst.c-1).length) { 
		state.frame.environment->hassign(Symbol(inst.b), REG(state, inst.c));
		REG(state, inst.c).i += REG(state, inst.c-1).i;
		return profile_back_edge(state,&inst+inst.a);
	} else return &inst+1;
}*/
Instruction const* jt_op(State& state, Instruction const& inst) {
	Logical l = As<Logical>(state, REG(state,inst.b));
	if(l.length == 0) _error("condition is of zero length");
	if(l[0]) return &inst+inst.a;
	else return &inst+1;
}
Instruction const* jf_op(State& state, Instruction const& inst) {
	Logical l = As<Logical>(state, REG(state, inst.b));
	if(l.length == 0) _error("condition is of zero length");
	if(l[0]) return &inst+1;
	else return &inst+inst.a;
}
Instruction const* branch_op(State& state, Instruction const& inst) {
	Value const& c = REG(state, inst.a);
	int64_t index = -1;
	if(c.isDouble1()) index = (int64_t)c.d;
	else if(c.isInteger1()) index = c.i;
	else if(c.isLogical1()) index = c.i;
	else if(c.isCharacter1()) {
		for(int64_t i = 1; i <= inst.b; i++) {
			if((&inst+i)->a == c.s.i) {
				index = i;
				break;
			}
			if(index < 0 && (&inst+i)->a == Strings::empty.i) {
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
Instruction const* colon_op(State& state, Instruction const& inst) {
	double from = asReal1(REG(state,inst.a));
	double to = asReal1(REG(state,inst.b));
	REG(state,inst.c) = Sequence(from, to>from?1:-1, fabs(to-from)+1);
	return &inst+1;
}
Instruction const* list_op(State& state, Instruction const& inst) {
	std::vector<String> const& dots = state.frame.environment->dots;
	// First time through, make a result vector...
	if(REG(state, inst.a).i == 0) {
		REG(state, inst.c) = List(dots.size());
	}
	// Otherwise populate result vector with next element
	else {
		state.frame.environment->assign(String::Init(-REG(state, inst.a).i), REG(state, inst.b));
		((List&)REG(state, inst.c))[REG(state, inst.a).i-1] = REG(state, inst.b);
	}

	// If we're all done, check to see if we need to add names and then exit
	if(REG(state, inst.a).i == (int64_t)dots.size()) {
		bool nonEmptyName = false;
		for(int i = 0; i < (int64_t)dots.size(); i++) 
			if(dots[i] != Strings::empty) nonEmptyName = true;
		if(nonEmptyName) {
			// TODO: should really just use the names in the dots directly
			Character names(dots.size());
			for(int64_t i = 0; i < (int64_t)dots.size(); i++)
				names[i] = dots[i];
			Object::Init(REG(state, inst.c), REG(state, inst.c), names);
		}
		return &inst+1;
	}

	// Not done yet, increment counter, evaluate next ..#
	REG(state, inst.a).i++;
	Value const& src = state.frame.environment->get(String::Init(-REG(state, inst.a).i));
	if(!src.isPromise()) {
		REG(state, inst.b) = src;
		return &inst;
	}
	else {
		Environment* env = Function(src).environment();
		if(env == 0) env = state.frame.environment;
		Prototype* prototype = Function(src).prototype();
		return buildStackFrame(state, env, false, prototype, &REG(state, inst.b), &inst);
	}
}

bool isRecordable(Type::Enum type, int64_t length) {
	return (type == Type::Double || type == Type::Integer)
		&& length > TRACE_VECTOR_WIDTH
		&& length % TRACE_VECTOR_WIDTH == 0;
}
bool isRecordable(Value const& a) {
	return isRecordable(a.type, a.length);
}
bool isRecordable(Value const& a, Value const& b) {
	bool valid_types =   (a.isDouble() || a.isInteger())
				      && (b.isDouble() || b.isInteger());
	size_t length = std::max(a.length,b.length);
	bool compatible_lengths = a.length == 1 || b.length == 1 || a.length == b.length;
	bool should_record_length = length >= TRACE_VECTOR_WIDTH && length % TRACE_VECTOR_WIDTH == 0;
	return valid_types && compatible_lengths && should_record_length;
}

Instruction const* seq_op(State& state, Instruction const& inst) {
	int64_t len = As<Integer>(state, REG(state, inst.a))[0];
	int64_t step = As<Integer>(state, REG(state, inst.b))[0];
	if(state.tracing.enabled() && isRecordable(Type::Integer, len))
		return state.tracing.begin_tracing(state, &inst, len);
	else {
		REG(state, inst.c) = Sequence(len, 1, step);
		return &inst+1;
	}
}

#define OP(name, string, Op, Func) \
Instruction const* name##_op(State& state, Instruction const& inst) { \
	Value & a =  REG(state, inst.a);	\
	Value & c = REG(state, inst.c);	\
	if(a.isDouble1()) { Op<TDouble>::RV::InitScalar(c, Op<TDouble>::eval(state, a.d)); return &inst+1; } \
	else if(a.isInteger1()) { Op<TDouble>::RV::InitScalar(c, Op<TInteger>::eval(state, a.i)); return &inst+1; } \
	if(a.isObject() ) { \
		String method; \
		Value f = GenericSearch(state, klass(state, a), state.internStr(Func), method); \
		if(f.isFunction()) { \
			Function func(f); \
			Environment* fenv = CreateEnvironment(state, func.environment(), state.frame.environment, Null::Singleton()); \
			MatchArgs(state, state.frame.environment, fenv, func, List::c(a), Character(0)); \
			return buildStackFrame(state, fenv, true, func.prototype(), &REG(state, inst.c), &inst+1); \
		}	\
	} \
	else if(state.tracing.enabled() && isRecordable(a)) \
		return state.tracing.begin_tracing(state, &inst, a.length); \
	\
	unaryArith<Zip1, Op>(state, a, c); \
	return &inst+1; \
}
UNARY_ARITH_MAP_BYTECODES(OP)
#undef OP


#define OP(name, string, Op, Func) \
Instruction const* name##_op(State& state, Instruction const& inst) { \
	unaryLogical<Zip1, Op>(state, REG(state, inst.a), REG(state, inst.c)); \
	return &inst+1; \
}
UNARY_LOGICAL_MAP_BYTECODES(OP)
#undef OP

#define OP(name, string, Op, Func) \
Instruction const* name##_op(State& state, Instruction const& inst) { \
	Value & a =  REG(state, inst.a);	\
	Value & b =  REG(state, inst.b);	\
	Value & c = REG(state, inst.c);	\
        if(a.isDouble1()) {			\
                if(b.isDouble1())		\
                        { Op<TDouble>::RV::InitScalar(c, Op<TDouble>::eval(state, a.d, b.d)); return &inst+1; }	\
                else if(b.isInteger1())	\
                        { Op<TDouble>::RV::InitScalar(c, Op<TDouble>::eval(state, a.d, (double)b.i));return &inst+1; }	\
        }	\
        else if(a.isInteger1()) {	\
                if(b.isDouble1())	\
                        { Op<TDouble>::RV::InitScalar(c, Op<TDouble>::eval(state, (double)a.i, b.d)); return &inst+1; }	\
                else if(b.isInteger1())	\
                        { Op<TInteger>::RV::InitScalar(c, Op<TInteger>::eval(state, a.i, b.i)); return &inst+1;} \
        } \
	if(a.isObject() || b.isObject()) { \
		Value f; \
		String method; \
		if(a.isObject() && b.isObject()) { \
			Value af = GenericSearch(state, klass(state, a), state.internStr(Func), method); \
			Value bf = GenericSearch(state, klass(state, b), state.internStr(Func), method); \
			if(af != bf) _error("Generic functions do not match on binary operator"); \
			f = af; \
		} \
		else if(a.isObject()) f = GenericSearch(state, klass(state, a), state.internStr(Func), method); \
		else f = GenericSearch(state, klass(state, b), state.internStr(Func), method); \
		if(f.isFunction()) { \
			Function func(f); \
			Environment* fenv = CreateEnvironment(state, func.environment(), state.frame.environment, Null::Singleton()); \
			MatchArgs(state, state.frame.environment, fenv, func, List::c(a, b), Character(0)); \
			return buildStackFrame(state, fenv, true, func.prototype(), &REG(state, inst.c), &inst+1); \
		}	\
	} \
	else if(state.tracing.enabled() && isRecordable(a,b)) \
		return state.tracing.begin_tracing(state, &inst, std::max(a.length,b.length));	\
\
	binaryArithSlow<Zip2, Op>(state, a, b, c);	\
	return &inst+1;	\
}
BINARY_ARITH_MAP_BYTECODES(OP)
#undef OP

#define OP(name, string, Op, Func) \
Instruction const* name##_op(State& state, Instruction const& inst) { \
	binaryLogical<Zip2, Op>(state, REG(state, inst.a), REG(state, inst.b), REG(state, inst.c)); \
	return &inst+1; \
}
BINARY_LOGICAL_MAP_BYTECODES(OP)
#undef OP

#define OP(name, string, Op, Func) \
Instruction const* name##_op(State& state, Instruction const& inst) { \
	Value & a =  REG(state, inst.a);	\
	Value & b =  REG(state, inst.b);	\
	Value & c = REG(state, inst.c);	\
        if(a.isDouble1()) {			\
                if(b.isDouble1())		\
                        { Op<TDouble>::RV::InitScalar(c, Op<TDouble>::eval(state, a.d, b.d)); return &inst+1; }	\
                else if(b.isInteger1())	\
                        { Op<TDouble>::RV::InitScalar(c, Op<TDouble>::eval(state, a.d, (double)b.i));return &inst+1; }	\
        }	\
        else if(a.isInteger1()) {	\
                if(b.isDouble1())	\
                        { Op<TDouble>::RV::InitScalar(c, Op<TDouble>::eval(state, (double)a.i, b.d)); return &inst+1; }	\
                else if(b.isInteger1())	\
                        { Op<TInteger>::RV::InitScalar(c, Op<TInteger>::eval(state, a.i, b.i)); return &inst+1;} \
        } \
	binaryOrdinal<Zip2, Op>(state, REG(state, inst.a), REG(state, inst.b), REG(state, inst.c)); \
	return &inst+1; \
}
BINARY_ORDINAL_MAP_BYTECODES(OP)
#undef OP

#define OP(name, string, Op, Func) \
Instruction const* name##_op(State& state, Instruction const& inst) { \
	unaryArith<FoldLeft, Op>(state, REG(state, inst.a), REG(state, inst.c)); \
	return &inst+1; \
}
ARITH_FOLD_BYTECODES(OP)
#undef OP

#define OP(name, string, Op, Func) \
Instruction const* name##_op(State& state, Instruction const& inst) { \
	unaryLogical<FoldLeft, Op>(state, REG(state, inst.a), REG(state, inst.c)); \
	return &inst+1; \
}
LOGICAL_FOLD_BYTECODES(OP)
#undef OP

#define OP(name, string, Op, Func) \
Instruction const* name##_op(State& state, Instruction const& inst) { \
	unaryOrdinal<FoldLeft, Op>(state, REG(state, inst.a), REG(state, inst.c)); \
	return &inst+1; \
}
ORDINAL_FOLD_BYTECODES(OP)
#undef OP

#define OP(name, string, Op, Func) \
Instruction const* name##_op(State& state, Instruction const& inst) { \
	unaryArith<ScanLeft, Op>(state, REG(state, inst.a), REG(state, inst.c)); \
	return &inst+1; \
}
ARITH_SCAN_BYTECODES(OP)
#undef OP

#define OP(name, string, Op, Func) \
Instruction const* name##_op(State& state, Instruction const& inst) { \
	unaryLogical<ScanLeft, Op>(state, REG(state, inst.a), REG(state, inst.c)); \
	return &inst+1; \
}
LOGICAL_SCAN_BYTECODES(OP)
#undef OP

#define OP(name, string, Op, Func) \
Instruction const* name##_op(State& state, Instruction const& inst) { \
	unaryOrdinal<ScanLeft, Op>(state, REG(state, inst.a), REG(state, inst.c)); \
	return &inst+1; \
}
ORDINAL_SCAN_BYTECODES(OP)
#undef OP

Instruction const* jmp_op(State& state, Instruction const& inst) {
	return &inst+inst.a;
}
Instruction const* sland_op(State& state, Instruction const& inst) {
	Logical l = As<Logical>(state, REG(state, inst.a));
	if(l.length == 0) _error("argument to && is zero length");
	if(Logical::isFalse(l[0])) {
		REG(state, inst.c) = Logical::False();
		return &inst+1;
	} else {
		Logical r = As<Logical>(state, REG(state, inst.b));
		if(r.length == 0) _error("argument to && is zero length");
		if(Logical::isFalse(r[0])) REG(state, inst.c) = Logical::False();
		else if(Logical::isNA(l[0]) || Logical::isNA(r[0])) REG(state, inst.c) = Logical::NA();
		else REG(state, inst.c) = Logical::True();
		return &inst+1;
	}
}
Instruction const* slor_op(State& state, Instruction const& inst) {
	Logical l = As<Logical>(state, REG(state, inst.a));
	if(l.length == 0) _error("argument to || is zero length");
	if(Logical::isTrue(l[0])) {
		REG(state, inst.c) = Logical::True();
		return &inst+1;
	} else {
		Logical r = As<Logical>(state, REG(state, inst.b));
		if(r.length == 0) _error("argument to || is zero length");
		if(Logical::isTrue(r[0])) REG(state, inst.c) = Logical::True();
		else if(Logical::isNA(l[0]) || Logical::isNA(r[0])) REG(state, inst.c) = Logical::NA();
		else REG(state, inst.c) = Logical::False();
		return &inst+1;
	}
}
Instruction const* function_op(State& state, Instruction const& inst) {
	REG(state, inst.c) = Function(state.frame.prototype->prototypes[inst.a], state.frame.environment);
	return &inst+1;
}
Instruction const* logical1_op(State& state, Instruction const& inst) {
	Integer i = As<Integer>(state, REG(state, inst.a));
	REG(state, inst.c) = Logical(i[0]);
	return &inst+1;
}
Instruction const* integer1_op(State& state, Instruction const& inst) {
	Integer i = As<Integer>(state, REG(state, inst.a));
	REG(state, inst.c) = Integer(i[0]);
	return &inst+1;
}
Instruction const* double1_op(State& state, Instruction const& inst) {
	int64_t length = asReal1(REG(state, inst.a));
	Double d(length);
	for(int64_t i = 0; i < length; i++) d[i] = 0;
	REG(state, inst.c) = d;
	return &inst+1;
}
Instruction const* character1_op(State& state, Instruction const& inst) {
	Integer i = As<Integer>(state, REG(state, inst.a));
	Character r = Character(i[0]);
	for(int64_t j = 0; j < r.length; j++) r[j] = Strings::empty;
	REG(state, inst.c) = r;
	return &inst+1;
}
Instruction const* raw1_op(State& state, Instruction const& inst) {
	Integer i = As<Integer>(state, REG(state, inst.a));
	REG(state, inst.c) = Raw(i[0]);
	return &inst+1;
}
Instruction const* type_op(State& state, Instruction const& inst) {
	Character c(1);
	// Should have a direct mapping from type to symbol.
	c[0] = state.internStr(Type::toString(REG(state, inst.a).type));
	REG(state, inst.c) = c;
	return &inst+1;
}
Instruction const* length_op(State& state, Instruction const& inst) {
	if(REG(state,inst.a).isVector())
		Integer::InitScalar(REG(state, inst.c), REG(state,inst.a).length);
	else
		Integer::InitScalar(REG(state, inst.c), 1);
	return &inst+1;
}
Instruction const* missing_op(State& state, Instruction const& inst) {
	// This could be inline cached...or implemented in terms of something else?
	String s = String::Init(inst.a);
	Value const& v = state.frame.environment->get(s);
	bool missing = v.isNil() || (v.isPromise() && Function(v).environment() == state.frame.environment);
	Logical::InitScalar(REG(state, inst.c), missing);
	return &inst+1;
}
Instruction const* mmul_op(State& state, Instruction const& inst) {
	REG(state, inst.c) = MatrixMultiply(state, REG(state, inst.a), REG(state, inst.b));
	return &inst+1;
}
Instruction const* ret_op(State& state, Instruction const& inst) {
	*(state.frame.result) = REG(state, inst.c);
	// if this stack frame owns the environment, we can free it for reuse
	// as long as we don't return a closure...
	// TODO: but also can't if an assignment to an out of scope variable occurs (<<-, assign) with a value of a closure!
	if(state.frame.ownEnvironment && REG(state, inst.c).isClosureSafe())
		state.environments.push_back(state.frame.environment);
	state.base = state.frame.returnbase;
	Instruction const* returnpc = state.frame.returnpc;
	state.pop();
	return returnpc;
}
Instruction const* done_op(State& state, Instruction const& inst) {
	// not used. When this instruction is hit, interpreter exits.
	return 0;
}


static void printCode(State const& state, Prototype const* prototype) {
	std::string r = "block:\nconstants: " + intToStr(prototype->constants.size()) + "\n";
	for(int64_t i = 0; i < (int64_t)prototype->constants.size(); i++)
		r = r + intToStr(i) + "=\t" + state.stringify(prototype->constants[i]) + "\n";

	r = r + "code: " + intToStr(prototype->bc.size()) + "\n";
	for(int64_t i = 0; i < (int64_t)prototype->bc.size(); i++)
		r = r + intToHexStr((uint64_t)&(prototype->bc[i])) + "--: " + intToStr(i) + ":\t" + prototype->bc[i].toString() + "\n";

	std::cout << r << std::endl;
}

#ifdef USE_THREADED_INTERPRETER
static const void** glabels = 0;
#endif

static Instruction const* buildStackFrame(State& state, Environment* environment, bool ownEnvironment, Prototype const* prototype, Value* result, Instruction const* returnpc) {
	//printCode(state, prototype);
	StackFrame& s = state.push();
	s.environment = environment;
	s.ownEnvironment = ownEnvironment;
	s.returnpc = returnpc;
	s.returnbase = state.base;
	s.result = result;
	s.prototype = prototype;
	state.base -= prototype->registers;
	if(state.base < state.registers)
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
void interpret(State& state, Instruction const* pc) {

#ifdef USE_THREADED_INTERPRETER
    #define LABELS_THREADED(name,type,...) (void*)&&name##_label,
	static const void* labels[] = {BYTECODES(LABELS_THREADED)};
	if(pc == 0) { 
		glabels = labels;
		return;
	}

	goto *(pc->ibc);
	#define LABELED_OP(name,type,...) \
		name##_label: \
			{ pc = name##_op(state, *pc); goto *(pc->ibc); } 
	STANDARD_BYTECODES(LABELED_OP)
	done_label: {}
#else
	while(pc->bc != ByteCode::done) {
		switch(pc->bc) {
			#define SWITCH_OP(name,type,...) \
				case ByteCode::name: { pc = name##_op(state, *pc); } break;
			BYTECODES(SWITCH_OP)
		};
	}
#endif
}

// ensure glabels is inited before we need it.
void interpreter_init(State& state) {
#ifdef USE_THREADED_INTERPRETER
	interpret(state, 0);
#endif
}

Value eval(State& state, Function const& function) {
	return eval(state, function.prototype(), function.environment());
}

Value eval(State& state, Prototype const* prototype) {
	return eval(state, prototype, state.frame.environment);
}

Value eval(State& state, Prototype const* prototype, Environment* environment) {
	static const Instruction* done = new Instruction(ByteCode::done);
#ifdef USE_THREADED_INTERPRETER
	done->ibc = glabels[ByteCode::done];
#endif
	Value* old_base = state.base;
	int64_t stackSize = state.stack.size();
	
	// Build a half-hearted stack frame for the result. Necessary for the trace recorder.
	StackFrame& s = state.push();
	s.environment = 0;
	s.prototype = 0;
	s.returnbase = state.base;
	state.base -= 1;
	Value* result = state.base;
	
	Instruction const* run = buildStackFrame(state, environment, false, prototype, result, done);
	try {
		interpret(state, run);
		state.base = old_base;
		state.pop();
	} catch(...) {
		state.base = old_base;
		state.stack.resize(stackSize);
		throw;
	}
	return *result;
}

