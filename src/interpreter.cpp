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

#define ALWAYS_INLINE __attribute__((always_inline))

static Instruction const* buildStackFrame(State& state, Environment* environment, bool ownEnvironment, Code const* code, Value* result, Instruction const* returnpc);

Instruction const* kget_op(State& state, Instruction const& inst) ALWAYS_INLINE;
Instruction const* sget_op(State& state, Instruction const& inst) ALWAYS_INLINE;
Instruction const* forend_op(State& state, Instruction const& inst) ALWAYS_INLINE;


inline void forcePromise(State& state, Value& v) { 
	while(v.isPromise()) {
		Environment* env = Function(v).environment();
		v = eval(state, Function(v).code(), 
			env != 0 ? env : state.frame.environment); 
	} 
}

// Get a Value by Symbol from the current environment
static Value get(State& state, Symbol s) {
	Environment* environment = state.frame.environment;
	Value value = environment->get(s);
	while(value.isNil() && environment->StaticParent() != 0) {
		environment = environment->StaticParent();
		value = environment->get(s);
	}
	if(value.isPromise()) {
		forcePromise(state, value);
		environment->assign(s, value);
	}
	return value;
}

// Get a Value by slot from the current environment
static Value& sget(State& state, int64_t i) {
	Value& value = state.frame.environment->get(i);
	if(value.isPromise()) forcePromise(state, value);
	return value;
}

static void assign(Environment* env, Symbol s, Value const& v) {
	env->assign(s, v);
}

static void sassign(Environment* env, int64_t i, Value const& v) {
	env->get(i) = v;
}

static void assign(State& state, Symbol s, Value const& v) {
	assign(state.frame.environment, s, v);
}

static void sassign(State& state, int64_t i, Value const& v) {
	sassign(state.frame.environment, i, v);
}

#define REG(state, i) (*(state.base+i))

Value & interpreter_reg(State & state, int64_t i) { return REG(state,i); }
Value interpreter_get(State & state, Symbol s) { return get(state,s); }
Value interpreter_sget(State & state, int64_t i) { return sget(state,i); }
void interpreter_assign(State & state, Symbol s, Value v) { assign(state,s,v); }
void interpreter_sassign(State & state, int64_t s, Value v) { sassign(state,s,v); }

static Value const& constant(State& state, int64_t i) {
	return state.frame.code->constants[i];
}

static List BuildArgs(State& state, CompiledCall const& call) {
	// Expand dots into the parameter list...
	// If it's in the dots it must already be a promise, thus no need to make a promise again.
	// Need to do the same for the names...
	List arguments = call.arguments;
	Value v = get(state, Symbol::dots);
	if(!v.isNil()) {
		List dots(v);
		List expanded(arguments.length + dots.length - 1 /* -1 for dots that will be replaced */);
		Insert(state, arguments, 0, expanded, 0, call.dots);
		Insert(state, dots, 0, expanded, call.dots, dots.length);
		Insert(state, arguments, call.dots, expanded, call.dots+dots.length, arguments.length-call.dots-1);
		arguments = expanded;
		if(hasNames(arguments) || hasNames(dots)) {
			Character names(expanded.length);
			for(int64_t i = 0; i < names.length; i++) names[i] = Symbol::empty;
			if(hasNames(arguments)) {
				Character anames = Character(getNames(arguments));
				Insert(state, anames, 0, names, 0, call.dots);
				Insert(state, anames, call.dots, names, call.dots+dots.length, arguments.length-call.dots-1);
			}
			if(hasNames(dots)) {
				Character dnames = Character(getNames(dots));
				Insert(state, dnames, 0, names, call.dots, dots.length);
			}
			setNames(arguments, names);
		}
	
	}
	return arguments;
}

inline void argAssign(Environment* env, int64_t i, Value const& v, Environment* execution) {
	Value& slot = env->get(i);
	slot = v;
	if(v.isPromise() && slot.env == 0)
		slot.env = execution;
}

static void MatchArgs(State& state, Environment* env, Environment* fenv, Function const& func, List const& arguments) {
	List parameters = func.parameters();
	Character pnames = Character(getNames(parameters));

	// Set to nil slots beyond the parameters
	for(int64_t i = parameters.length; i < fenv->SlotCount(); i++) {
		sassign(fenv, i, Nil);
	}

	// call arguments are not named, do posititional matching
	if(!hasNames(arguments)) {
		int64_t end = std::min(arguments.length, func.dots());
		for(int64_t i = 0; i < end; ++i) {
			argAssign(fenv, i, arguments[i], env);
		}
		// set dots if necessary
		if(func.dots() < parameters.length && arguments.length-func.dots() > 0) {
			List dots(arguments.length - func.dots());
			for(int64_t i = 0; i < arguments.length-func.dots(); i++) {
				dots[i] = arguments[i+func.dots()];
				if(dots[i].isPromise() && dots[i].env == 0)
					dots[i].env = env;
			}
			sassign(fenv, func.dots(), dots);
			end++;
		}
		// set defaults
		for(int64_t i = end; i < parameters.length; ++i) {
			argAssign(fenv, i, parameters[i], fenv);
		}
		
	}
	// call arguments are named, do matching by name
	else {
		// we should be able to cache and reuse this assignment for pairs of functions and call sites.
		static char assignment[64], set[64];
		for(int64_t i = 0; i < arguments.length; i++) assignment[i] = -1;
		for(int64_t i = 0; i < parameters.length; i++) set[i] = -(i+1);
		
		Character anames = Character(getNames(arguments));
		// named args, search for complete matches
		for(int64_t i = 0; i < arguments.length; ++i) {
			if(anames[i] != Symbol::empty) {
				for(int64_t j = 0; j < parameters.length; ++j) {
					if(pnames[j] != Symbol::dots && anames[i] == pnames[j]) {
						assignment[i] = j;
						set[j] = i;
						break;
					}
				}
			}
		}
		// named args, search for incomplete matches
		for(int64_t i = 0; i < arguments.length; ++i) {
			if(anames[i] != Symbol::empty && assignment[i] < 0) {
				std::string a = state.SymToStr(anames[i]);
				for(int64_t j = 0; j < parameters.length; ++j) {
					if(set[j] < 0 && pnames[j] != Symbol::dots &&
						state.SymToStr(pnames[j]).compare( 0, a.size(), a ) == 0 ) {	
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
			if(anames[i] == Symbol::empty) {
				for(; firstEmpty < func.dots(); ++firstEmpty) {
					if(set[firstEmpty] < 0) {
						assignment[i] = firstEmpty;
						set[firstEmpty] = i;
						break;
					}
				}
			}
		}
		
		// count up unused parameters and assign names
		Character names(0); 
		int64_t unassigned = 0;
		if(func.dots() < parameters.length) {
			// count up the unassigned args
			for(int64_t j = 0; j < arguments.length; j++) if(assignment[j] < 0) unassigned++;
			names = Character(unassigned);
			int64_t idx = 0;
			for(int64_t j = 0; j < arguments.length; j++) if(assignment[j] < 0) names[idx++] = anames[j];
		}

		// stuff that can't be cached...

		// assign all the arguments
		for(int64_t j = 0; j < parameters.length; ++j) if(j != func.dots()) argAssign(fenv, j, set[j]>=0 ? arguments[set[j]] : parameters[-(set[j]+1)], env);

		// put unused args into the dots
		if(func.dots() < parameters.length) {
			List values(unassigned);
			int64_t idx = 0;
			for(int64_t j = 0; j < arguments.length; j++) {
				if(assignment[j] < 0) {
					values[idx] = arguments[j];
					if(values[idx].isPromise() && values[idx].env == 0)
						values[idx].env = env;
					idx++;
				}
			}
			setNames(values, names);
			sassign(fenv, func.dots(), values);
		}
	}
}

static Environment* CreateEnvironment(State& state, Environment* s, Environment* d, std::vector<Symbol> const& symbols) {
	Environment* env;
	if(state.environments.size() == 0) {
		env = new Environment(s, d, symbols);
	} else {
		env = state.environments.back();
		state.environments.pop_back();
		env->init(s, d, symbols);
	}
	return env;
}
//track the heat of back edge operations and invoke the recorder on hot traces
#define PROFILE_TABLE_SIZE 1021
static Instruction const * profile_back_edge(State & state, Instruction const * inst) {
#ifndef RIPOSTE_DISABLE_TRACING
	if(state.tracing.is_tracing())
		return inst;
	static int64_t hash_table[PROFILE_TABLE_SIZE];
	assert(sizeof(Instruction) == 32);
	int64_t value = ( (int64_t) inst >> 5 ) % PROFILE_TABLE_SIZE;
	int64_t heat = ++hash_table[value];
	if(heat >= state.tracing.start_count && heat < state.tracing.start_count + state.tracing.max_attempts) { //upper bound is to prevent trying many failed traces
		printf("trace beginning at %s\n", inst->bc.toString());
		return recording_interpret(state,inst);
	} else return inst;
#else
	return inst;
#endif
}

Instruction const* call_op(State& state, Instruction const& inst) {
	Value f = REG(state, inst.a);
	CompiledCall const& call = state.frame.code->calls[inst.b];
	List arguments = call.dots < call.arguments.length ? BuildArgs(state, call) : call.arguments;

	if(f.isFunction()) {
		Function func(f);
		Environment* fenv = CreateEnvironment(state, func.environment(), state.frame.environment, func.code()->slotSymbols);
		MatchArgs(state, state.frame.environment, fenv, func, arguments);
		return buildStackFrame(state, fenv, true, func.code(), &REG(state, inst.c), &inst+1);
	} else if(f.isBuiltIn()) {
		REG(state, inst.c) = BuiltIn(f).func(state, arguments);
		return &inst+1;
	} else {
		_error(std::string("Non-function (") + Type::toString(f.type) + ") as first parameter to call\n");
		return &inst+1;
	}	
}
Instruction const* UseMethod_op(State& state, Instruction const& inst) {
	Value v = REG(state, inst.a);
	Symbol generic = v.isCharacter() ? Character(v)[0] : Symbol(v);
	
	CompiledCall const& call = state.frame.code->calls[inst.b];
	List arguments = call.dots < call.arguments.length ? BuildArgs(state, call) : call.arguments;
	
	Value object = REG(state, inst.c);
	Character type = klass(state, object);

	//Search for type-specific method
	Symbol method = state.StrToSym(state.SymToStr(generic) + "." + state.SymToStr(type[0]));
	Value f = get(state, method);
	
	//Search for default
	if(f.isNil()) {
		method = state.StrToSym(state.SymToStr(generic) + ".default");
		f = get(state, method);
	}

	if(f.isFunction()) {
		Function func(f);
		assert(func.body().isClosure());
		Environment* fenv = CreateEnvironment(state, func.environment(), state.frame.environment, func.code()->slotSymbols);
		MatchArgs(state, state.frame.environment, fenv, func, arguments);
		assign(fenv, Symbol::dotGeneric, generic);
		assign(fenv, Symbol::dotMethod, method);
		assign(fenv, Symbol::dotClass, type); 
		return buildStackFrame(state, fenv, true, func.code(), &REG(state, inst.c), &inst+1);
	} else if(f.isBuiltIn()) {
		REG(state, inst.c) = BuiltIn(f).func(state, arguments);
		return &inst+1;
	} else {
		_error(std::string("no applicable method for '") + state.SymToStr(generic) + "' applied to an object of class \"" + state.SymToStr(type[0]) + "\"");
	}
}
Instruction const* get_flat(State& state, Symbol s, Value& value, Environment* environment, Instruction const& inst)
{
	while(value.isNil()) {
		environment = environment->StaticParent();
		if(environment == NULL)
			throw RiposteError(std::string("object '") + state.SymToStr(s) + "' not found");
		value = environment->get(s);
	}
	if(!value.isPromise()) {
		return &inst+1;
	}
	else {
		return buildStackFrame(state, Function(value).environment(),
			false, Function(value).code(), &environment->getLocation(s), &inst);
	}
}
Instruction const* get_op(State& state, Instruction const& inst) {
	/*Environment* environment = state.frame.environment;
	REG(state, inst.c) = environment->get(Symbol(inst.a));
	return get_flat(state, Symbol(inst.a), REG(state, inst.c), environment, inst);
	*/
	REG(state, inst.c) = get(state, Symbol(inst.a));
	if(REG(state, inst.c).isNil()) throw RiposteError(std::string("object '") + state.SymToStr(Symbol(inst.a)) + "' not found");
	return &inst+1;
}

Instruction const* sget_op(State& state, Instruction const& inst)
{
	/*Environment* environment = state.frame.environment;
	REG(state, inst.c) = environment->get(inst.a);
	return get_flat(state, environment->slotName(inst.a), REG(state, inst.c), environment, inst);
	*/
	Value& dest = REG(state, inst.c);
	dest = state.frame.environment->get(inst.a);
	if(!dest.isConcrete()) {
		if(dest.isNil()) {
			dest = get(state, state.frame.environment->slotName(inst.a));
			if(dest.isNil())
				_error(std::string("object '") + state.SymToStr(state.frame.environment->slotName(inst.a)) + "' not found");
		}
		if(dest.isPromise()) 
			forcePromise(state, dest);
	}
	return &inst+1;
}

Instruction const* kget_op(State& state, Instruction const& inst) {
	REG(state, inst.c) = constant(state, inst.a);
	return &inst+1;
}
Instruction const* iget_op(State& state, Instruction const& inst) {
	REG(state, inst.c) = state.path[0]->get(Symbol(inst.a));
	if(REG(state, inst.c).isNil()) throw RiposteError(std::string("object '") + state.SymToStr(Symbol(inst.a)) + "' not found");
	return &inst+1;
}
Instruction const* assign_op(State& state, Instruction const& inst) {
	assign(state, Symbol(inst.a), REG(state, inst.c));
	return &inst+1;
}
Instruction const* sassign_op(State& state, Instruction const& inst) {
	sassign(state, inst.a, REG(state, inst.c));
	return &inst+1;
}
// everything else should be in registers

Instruction const* iassign_op(State& state, Instruction const& inst) {
	// a = value, b = index, c = dest 
	subAssign(state, REG(state,inst.c), REG(state,inst.b), REG(state,inst.a), REG(state,inst.c));
	return &inst+1;
}
Instruction const* eassign_op(State& state, Instruction const& inst) {
	// a = value, b = index, c = dest 
	Value v = REG(state, inst.c);
	if(v.isList()) {
		List r = Clone(List(v));
		r[As<Integer>(state, REG(state,inst.b))[0]-1] = REG(state,inst.a);
		REG(state, inst.c) = r;
		return &inst+1;
	}
	else {
		subAssign(state, REG(state,inst.c), REG(state,inst.b), REG(state,inst.a), REG(state,inst.c));
		return &inst+1;
	}
}
Instruction const* forbegin_op(State& state, Instruction const& inst) {
	// inst.c-1 holds the loopVector
	if((int64_t)REG(state, inst.c-1).length <= 0) { return &inst+inst.a; }
	REG(state, inst.c) = Integer::c(1);
	Value v;
	Element(REG(state, inst.c-1), 0, v);
	assign(state, Symbol(inst.b), v);
	return &inst+1;
}
Instruction const* forend_op(State& state, Instruction const& inst) {
	if((REG(state,inst.c).i) < REG(state,inst.c-1).length) {
		Value v;
		Element(REG(state, inst.c-1), REG(state, inst.c).i, v);
		assign(state, Symbol(inst.b), v);
		REG(state, inst.c).i++;
		return profile_back_edge(state,&inst+inst.a);
	} else return &inst+1;
}
Instruction const* iforbegin_op(State& state, Instruction const& inst) {
	double m = asReal1(REG(state, inst.c-1));
	double n = asReal1(REG(state, inst.c));
	REG(state, inst.c-1) = Integer::c(n > m ? 1 : -1);
	REG(state, inst.c-1).length = (int64_t)n+1;	// danger! this register no longer holds a valid object, but it saves a register and makes the for and ifor cases more similar
	REG(state, inst.c) = Integer::c((int64_t)m);
	if(REG(state, inst.c).i >= (int64_t)REG(state, inst.c-1).length) { return &inst+inst.a; }
	assign(state, Symbol(inst.b), Integer::c(m));
	REG(state, inst.c).i += REG(state, inst.c-1).i;
	return &inst+1;
}
Instruction const* iforend_op(State& state, Instruction const& inst) {
	if(REG(state, inst.c).i < REG(state, inst.c-1).length) { 
		assign(state, Symbol(inst.b), REG(state, inst.c));
		REG(state, inst.c).i += REG(state, inst.c-1).i;
		return profile_back_edge(state,&inst+inst.a);
	} else return &inst+1;
}
Instruction const* whilebegin_op(State& state, Instruction const& inst) {
	Logical l(REG(state,inst.b));
	if(l[0]) return &inst+1;
	else return &inst+inst.a;
}
Instruction const* whileend_op(State& state, Instruction const& inst) {
	Logical l(REG(state,inst.b));
	if(l[0]) return profile_back_edge(state,&inst+inst.a);
	else return &inst+1;
}
Instruction const* repeatbegin_op(State& state, Instruction const& inst) {
	return &inst+1;
}
Instruction const* repeatend_op(State& state, Instruction const& inst) {
	return profile_back_edge(state,&inst+inst.a);
}
Instruction const* next_op(State& state, Instruction const& inst) {
	return &inst+inst.a;
}
Instruction const* break1_op(State& state, Instruction const& inst) {
	return &inst+inst.a;
}
Instruction const* if1_op(State& state, Instruction const& inst) {
	Logical l = As<Logical>(state, REG(state,inst.b));
	if(l.length == 0) _error("if argument is of zero length");
	if(l[0]) return &inst+1;
	else return &inst+inst.a;
}
Instruction const* if0_op(State& state, Instruction const& inst) {
	Logical l = As<Logical>(state, REG(state, inst.b));
	if(l.length == 0) _error("if argument is of zero length");
	if(!l[0]) return &inst+1;
	else return &inst+inst.a;
}
Instruction const* colon_op(State& state, Instruction const& inst) {
	double from = asReal1(REG(state,inst.a));
	double to = asReal1(REG(state,inst.b));
	REG(state,inst.c) = Sequence(from, to>from?1:-1, fabs(to-from)+1);
	return &inst+1;
}
Instruction const* seq_op(State& state, Instruction const& inst) {
	int64_t len = As<Integer>(state, REG(state, inst.a))[0];
	REG(state, inst.c) = Sequence(len);
	return &inst+1;
}
Instruction const* add_op(State& state, Instruction const& inst) {
	binaryArith<Zip2, AddOp>(state, REG(state, inst.a), REG(state, inst.b), REG(state, inst.c));
	return &inst+1;
}
Instruction const* pos_op(State& state, Instruction const& inst) {
	unaryArith<Zip1, PosOp>(state, REG(state, inst.a), REG(state, inst.c));
	return &inst+1;
}
Instruction const* sub_op(State& state, Instruction const& inst) {
	binaryArith<Zip2, SubOp>(state, REG(state, inst.a), REG(state, inst.b), REG(state, inst.c));
	return &inst+1;
}
Instruction const* neg_op(State& state, Instruction const& inst) {
	unaryArith<Zip1, NegOp>(state, REG(state, inst.a), REG(state, inst.c));
	return &inst+1;
}
Instruction const* mul_op(State& state, Instruction const& inst) {
	binaryArith<Zip2, MulOp>(state, REG(state, inst.a), REG(state, inst.b), REG(state, inst.c));
	return &inst+1;
}
Instruction const* div_op(State& state, Instruction const& inst) {
	binaryArith<Zip2, DivOp>(state, REG(state, inst.a), REG(state, inst.b), REG(state, inst.c));
	return &inst+1;
}
Instruction const* idiv_op(State& state, Instruction const& inst) {
	binaryArith<Zip2, IDivOp>(state, REG(state, inst.a), REG(state, inst.b), REG(state, inst.c));
	return &inst+1;
}
Instruction const* mod_op(State& state, Instruction const& inst) {
	binaryArith<Zip2, ModOp>(state, REG(state, inst.a), REG(state, inst.b), REG(state, inst.c));
	return &inst+1;
}
Instruction const* pow_op(State& state, Instruction const& inst) {
	binaryArith<Zip2, PowOp>(state, REG(state, inst.a), REG(state, inst.b), REG(state, inst.c));
	return &inst+1;
}
Instruction const* lnot_op(State& state, Instruction const& inst) {
	unaryLogical<Zip1, LNotOp>(state, REG(state, inst.a), REG(state, inst.c));
	return &inst+1;
}
Instruction const* land_op(State& state, Instruction const& inst) {
	binaryLogical<Zip2, AndOp>(state, REG(state, inst.a), REG(state, inst.b), REG(state, inst.c));
	return &inst+1;
}
Instruction const* lor_op(State& state, Instruction const& inst) {
	binaryLogical<Zip2, OrOp>(state, REG(state, inst.a), REG(state, inst.b), REG(state, inst.c));
	return &inst+1;
}
Instruction const* eq_op(State& state, Instruction const& inst) {
	binaryOrdinal<Zip2, EqOp>(state, REG(state, inst.a), REG(state, inst.b), REG(state, inst.c));
	return &inst+1;
}
Instruction const* neq_op(State& state, Instruction const& inst) {
	binaryOrdinal<Zip2, NeqOp>(state, REG(state, inst.a), REG(state, inst.b), REG(state, inst.c));
	return &inst+1;
}
Instruction const* lt_op(State& state, Instruction const& inst) {
	binaryOrdinal<Zip2, LTOp>(state, REG(state, inst.a), REG(state, inst.b), REG(state, inst.c));
	return &inst+1;
}
Instruction const* le_op(State& state, Instruction const& inst) {
	binaryOrdinal<Zip2, LEOp>(state, REG(state, inst.a), REG(state, inst.b), REG(state, inst.c));
	return &inst+1;
}
Instruction const* gt_op(State& state, Instruction const& inst) {
	binaryOrdinal<Zip2, GTOp>(state, REG(state, inst.a), REG(state, inst.b), REG(state, inst.c));
	return &inst+1;
}
Instruction const* ge_op(State& state, Instruction const& inst) {
	binaryOrdinal<Zip2, GEOp>(state, REG(state, inst.a), REG(state, inst.b), REG(state, inst.c));
	return &inst+1;
}
Instruction const* abs_op(State& state, Instruction const& inst) {
	unaryArith<Zip1, AbsOp >(state, REG(state, inst.a), REG(state, inst.c));
	return &inst+1;
}
Instruction const* sign_op(State& state, Instruction const& inst) {
	unaryArith<Zip1, SignOp >(state, REG(state, inst.a), REG(state, inst.c));
	return &inst+1;
}
Instruction const* sqrt_op(State& state, Instruction const& inst) {
	unaryArith<Zip1, SqrtOp >(state, REG(state, inst.a), REG(state, inst.c));
	return &inst+1;
}
Instruction const* floor_op(State& state, Instruction const& inst) {
	unaryArith<Zip1, FloorOp >(state, REG(state, inst.a), REG(state, inst.c));
	return &inst+1;
}
Instruction const* ceiling_op(State& state, Instruction const& inst) {
	unaryArith<Zip1, CeilingOp >(state, REG(state, inst.a), REG(state, inst.c));
	return &inst+1;
}
Instruction const* trunc_op(State& state, Instruction const& inst) {
	unaryArith<Zip1, TruncOp >(state, REG(state, inst.a), REG(state, inst.c));
	return &inst+1;
}
Instruction const* round_op(State& state, Instruction const& inst) {
	unaryArith<Zip1, RoundOp >(state, REG(state, inst.a), REG(state, inst.c));
	return &inst+1;
}
Instruction const* signif_op(State& state, Instruction const& inst) {
	unaryArith<Zip1, SignifOp >(state, REG(state, inst.a), REG(state, inst.c));
	return &inst+1;
}
Instruction const* exp_op(State& state, Instruction const& inst) {
	unaryArith<Zip1, ExpOp >(state, REG(state, inst.a), REG(state, inst.c));
	return &inst+1;
}
Instruction const* log_op(State& state, Instruction const& inst) {
	unaryArith<Zip1, LogOp >(state, REG(state, inst.a), REG(state, inst.c));
	return &inst+1;
}
Instruction const* cos_op(State& state, Instruction const& inst) {
	unaryArith<Zip1, CosOp >(state, REG(state, inst.a), REG(state, inst.c));
	return &inst+1;
}
Instruction const* sin_op(State& state, Instruction const& inst) {
	unaryArith<Zip1, SinOp >(state, REG(state, inst.a), REG(state, inst.c));
	return &inst+1;
}
Instruction const* tan_op(State& state, Instruction const& inst) {
	unaryArith<Zip1, TanOp >(state, REG(state, inst.a), REG(state, inst.c));
	return &inst+1;
}
Instruction const* acos_op(State& state, Instruction const& inst) {
	unaryArith<Zip1, ACosOp >(state, REG(state, inst.a), REG(state, inst.c));
	return &inst+1;
}
Instruction const* asin_op(State& state, Instruction const& inst) {
	unaryArith<Zip1, ASinOp >(state, REG(state, inst.a), REG(state, inst.c));
	return &inst+1;
}
Instruction const* atan_op(State& state, Instruction const& inst) {
	unaryArith<Zip1, ATanOp >(state, REG(state, inst.a), REG(state, inst.c));
	return &inst+1;
}
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
	REG(state, inst.c) = Function(state.frame.code->code[inst.a], state.frame.environment);
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
	Integer i = As<Integer>(state, REG(state, inst.a));
	REG(state, inst.c) = Double(i[0]);
	return &inst+1;
}
Instruction const* complex1_op(State& state, Instruction const& inst) {
	Integer i = As<Integer>(state, REG(state, inst.a));
	REG(state, inst.c) = Complex(i[0]);
	return &inst+1;
}
Instruction const* character1_op(State& state, Instruction const& inst) {
	Integer i = As<Integer>(state, REG(state, inst.a));
	Character r = Character(i[0]);
	for(int64_t j = 0; j < r.length; j++) r[j] = Symbol::empty;
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
	c[0] = state.StrToSym(Type::toString(REG(state, inst.a).type));
	REG(state, inst.c) = c;
	return &inst+1;
}
Instruction const * invoketrace_op(State& state, Instruction const & inst) {
	Trace * trace = state.frame.code->traces[inst.a];
	int64_t offset;
	TCStatus::Enum status = trace->compiled->execute(state,&offset);
	if(status != TCStatus::SUCCESS) {
		printf("trace: encountered error %s\n",TCStatus::toString(status));
	}
	if(status  != TCStatus::SUCCESS || offset == 0) { //we exited to the trace start instruction, invoke the original instruction here
		Instruction invoketrace = inst;
		const_cast<Instruction&>(inst) = trace->trace_inst;
		Instruction const * pc = 0;
#define BC_SWITCH(bc,str) case ByteCode::bc: pc = bc##_op(state,inst); break;
		switch(trace->trace_inst.bc) {
			BYTECODES(BC_SWITCH)
		}
		const_cast<Instruction&>(inst) = invoketrace;
		return pc;
#undef BC_SWITCH
	}

	return &inst + offset;
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

#define THREADED_INTERPRETER

#ifdef THREADED_INTERPRETER
static const void** glabels = 0;
#endif

static Instruction const* buildStackFrame(State& state, Environment* environment, bool ownEnvironment, Code const* code, Value* result, Instruction const* returnpc) {
	//std::cout << "Compiled code: " << state.stringify(Closure((Code*)code,NULL)) << std::endl;
	StackFrame& s = state.push();
	s.environment = environment;
	s.ownEnvironment = ownEnvironment;
	s.returnpc = returnpc;
	s.returnbase = state.base;
	s.result = result;
	s.code= code;
	state.base -= code->registers;
	if(state.base < state.registers)
		throw RiposteError("Register overflow");

#ifdef THREADED_INTERPRETER
	// Initialize threaded bytecode if not yet done 
	if(code->tbc.size() == 0)
	{
		for(int64_t i = 0; i < (int64_t)code->bc.size(); ++i) {
			Instruction const& inst = code->bc[i];
			code->tbc.push_back(
				Instruction(
					inst.bc == ByteCode::done ? glabels[0] : glabels[inst.bc+1],
					inst.a, inst.b, inst.c));
		}
	}
	return &(code->tbc[0]);
#else
	return &(code->bc[0]);
#endif
}

//
//    Main interpreter loop 
//
//__attribute__((__noinline__,__noclone__)) 
void interpret(State& state, Instruction const* pc) {
#ifdef THREADED_INTERPRETER
    #define LABELS_THREADED(name,type) (void*)&&name##_label,
	static const void* labels[] = {&&DONE, BYTECODES(LABELS_THREADED)};
	glabels = &labels[0];
	if(pc == 0) return;

	goto *(pc->ibc);
	#define LABELED_OP(name,type) \
		name##_label: \
			{ pc = name##_op(state, *pc); goto *(pc->ibc); } 
	BYTECODES(LABELED_OP)
	DONE: {}
#else
	while(pc->bc != ByteCode::done) {
		switch(pc->bc.Enum()) {
			#define SWITCH_OP(name,type) \
				case ByteCode::name: { pc = name##_op(state, *pc); } break;
			BYTECODES(SWITCH_OP,0)
		};
	}
#endif
}

// ensure glabels is inited before we need it.
void interpreter_init(State& state) {
#ifdef THREADED_INTERPRETER
	interpret(state, 0);
#endif
}

Value eval(State& state, Function const& function) {
	return eval(state, function.code(), function.environment());
}

Value eval(State& state, Code const* code) {
	return eval(state, code, state.frame.environment);
}

Value eval(State& state, Code const* code, Environment* environment) {
	//std::cout << "Executing: " << state.stringify(Closure((Code*)code,NULL)) << std::endl;
	Value result;
#ifdef THREADED_INTERPRETER
	static const Instruction* done = new Instruction(glabels[0]);
#else
	static const Instruction* done = new Instruction(ByteCode::done);
#endif
	Value* old_base = state.base;
	int64_t stackSize = state.stack.size();
	Instruction const* run = buildStackFrame(state, environment, false, code, &result, done);
	try {
		interpret(state, run);
	} catch(...) {
		state.base = old_base;
		state.stack.resize(stackSize);
		throw;
	}
	return result;
}

