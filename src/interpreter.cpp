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

static Instruction const* buildStackFrame(State& state, Environment* environment, bool ownEnvironment, Code const* code, Value* result, Instruction const* returnpc);


// Get a Value by Symbol from the current environment
static Value get(State& state, Symbol s) {
	Environment* environment = state.frame().environment;
	Value value = environment->get(s);
	while(value.isNil() && environment->StaticParent() != 0) {
		environment = environment->StaticParent();
		value = environment->get(s);
	}
	value = force(state, value);
	environment->assign(s, value);
	return value;
}

// Get a Value by slot from the current environment
static Value sget(State& state, int64_t i) {
	Environment* environment = state.frame().environment;
	Value value = environment->get(i);
	if(!value.isNil()) {
		value = force(state, value);
		environment->get(i) = value;
		return value;
	}
	Symbol s = environment->slotName(i);
	while(value.isNil() && environment->StaticParent() != 0) {
		environment = environment->StaticParent();
		value = environment->get(s);
	}
	value = force(state, value);
	environment->assign(s, value);
	return value;
}

static void assign(Environment* env, Symbol s, Value v) {
	env->assign(s, v);
}

static void sassign(Environment* env, int64_t i, Value v) {
	env->get(i) = v;
}

static void assign(State& state, Symbol s, Value v) {
	assign(state.frame().environment, s, v);
}

static void sassign(State& state, int64_t i, Value v) {
	sassign(state.frame().environment, i, v);
}

static Value& reg(State& state, int64_t i) {
	return *(state.base + i);
}


Value & interpreter_reg(State & state, int64_t i) { return reg(state,i); }
Value interpreter_get(State & state, Symbol s) { return get(state,s); }
Value interpreter_sget(State & state, int64_t i) { return sget(state,i); }
void interpreter_assign(State & state, Symbol s, Value v) { assign(state,s,v); }
void interpreter_sassign(State & state, int64_t s, Value v) { sassign(state,s,v); }

static Value constant(State& state, int64_t i) {
	return state.frame().code->constants[i];
}

static List BuildArgs(State& state, CompiledCall& call) {
	// Expand dots into the parameter list...
	// If it's in the dots it must already be a promise, thus no need to make a promise again.
	// Need to do the same for the names...
	List arguments = call.arguments();
	Value v = get(state, Symbol::dots);
	if(!v.isNil()) {
		List dots(v);
		List expanded(arguments.length + dots.length - 1 /* -1 for dots that will be replaced */);
		Insert(state, arguments, 0, expanded, 0, call.dots());
		Insert(state, dots, 0, expanded, call.dots(), dots.length);
		Insert(state, arguments, call.dots(), expanded, call.dots()+dots.length, arguments.length-call.dots()-1);
		arguments = expanded;
		if(hasNames(arguments) || hasNames(dots)) {
			Character names(expanded.length);
			for(int64_t i = 0; i < names.length; i++) names[i] = Symbol::empty;
			if(hasNames(arguments)) {
				Character anames = getNames(arguments);
				Insert(state, anames, 0, names, 0, call.dots());
				Insert(state, anames, call.dots(), names, call.dots()+dots.length, arguments.length-call.dots()-1);
			}
			if(hasNames(dots)) {
				Character dnames = getNames(dots);
				Insert(state, dnames, 0, names, call.dots(), dots.length);
			}
			setNames(arguments, names);
		}
	
	}
	return arguments;
}

inline void argAssign(Environment* env, int64_t i, Value const& v, Environment* execution) {
	Value& slot = env->get(i);
	slot = v;
	if((v.isPromise() || v.isDefault()) && slot.env == 0)
		slot.env = execution;
}

static void MatchArgs(State& state, Environment* env, Environment* fenv, Function const& func, List const& arguments) {
	List parameters = func.parameters();
	Character pnames = getNames(parameters);

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
				if((dots[i].isPromise() || dots[i].isDefault()) && dots[i].env == 0)
					dots[i].env = env;
			}
			sassign(fenv, func.dots(), dots);
			end++;
		}
		// set defaults (often these will be completely overridden. Can we delay or skip?
		for(int64_t i = end; i < parameters.length; ++i) {
			argAssign(fenv, i, parameters[i], fenv);
		}
		
		// set nil slots
		for(int64_t i = parameters.length; i < fenv->SlotCount(); i++) {
			sassign(fenv, i, Value::Nil);
		}
	}
	// call arguments are named, do matching by name
	else {
		// set defaults (often these will be completely overridden. Can we delay or skip?
		for(int64_t i = 0; i < parameters.length; ++i) {
			argAssign(fenv, i, parameters[i], fenv);
		}
		// set nils
		for(int64_t i = parameters.length; i < fenv->SlotCount(); i++) {
			sassign(fenv, i, Value::Nil);
		}
		Character anames = getNames(arguments);
		// we should be able to cache and reuse this assignment for pairs of functions and call sites.
		static char assignment[64], set[64];
		for(int64_t i = 0; i < arguments.length; i++) assignment[i] = -1;
		for(int64_t i = 0; i < parameters.length; i++) set[i] = -1;
		// named args, search for complete matches
		for(int64_t i = 0; i < arguments.length; ++i) {
			if(anames[i] != Symbol::empty) {
				for(int64_t j = 0; j < parameters.length; ++j) {
					if(pnames[j] != Symbol::dots && anames[i] == pnames[j]) {
						argAssign(fenv, j, arguments[i], env);
						assignment[i] = j;
						set[j] = i;
						break;
					}
				}
			}
		}
		// named args, search for incomplete matches
		for(int64_t i = 0; i < arguments.length; ++i) {
			if(anames[i] != Symbol::empty && assignment[i] == 0) {
				std::string a = state.SymToStr(anames[i]);
				for(int64_t j = 0; j < parameters.length; ++j) {
					if(set[j] < 0 && pnames[j] != Symbol::dots &&
						state.SymToStr(pnames[i]).compare( 0, a.size(), a ) == 0 ) {	
						argAssign(fenv, j, arguments[i], env);
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
						argAssign(fenv, firstEmpty, arguments[i], env);
						assignment[i] = firstEmpty;
						set[firstEmpty] = i;
						break;
					}
				}
			}
		}
		// put unused args into the dots
		if(func.dots() < parameters.length) {
			// count up the unassigned args
			int64_t unassigned = 0;
			for(int64_t j = 0; j < arguments.length; j++) if(assignment[j] < 0) unassigned++;
			List values(unassigned);
			Character names(unassigned);
			int64_t idx = 0;
			for(int64_t j = 0; j < arguments.length; j++) {
				if(assignment[j] < 0) {
					values[idx] = arguments[j];
					if(values[idx].isPromise() && values[idx].env == 0)
						values[idx].env = env;
					names[idx++] = anames[j];
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
	Value f = reg(state, inst.a);
	CompiledCall call(constant(state, inst.b));
	List arguments = call.dots() < call.arguments().length ? BuildArgs(state, call) : call.arguments();

	if(f.type == Type::R_function) {
		Function func(f);
		assert(func.body().type == Type::I_closure || func.body().type == Type::I_promise);
		Environment* fenv = CreateEnvironment(state, func.s(), state.frame().environment, Closure(func.body()).code()->slotSymbols);
		MatchArgs(state, state.frame().environment, fenv, func, arguments);
		return buildStackFrame(state, fenv, true, Closure(func.body()).code(), &reg(state, inst.c), &inst+1);
	} else if(f.type == Type::R_cfunction) {
		reg(state, inst.c) = CFunction(f).func(state, arguments);
		return &inst+1;
	} else {
		_error(std::string("Non-function (") + f.type.toString() + ") as first parameter to call\n");
		return &inst+1;
	}	
}
Instruction const* UseMethod_op(State& state, Instruction const& inst) {
	Value v = reg(state, inst.a);
	Symbol generic = v.isCharacter() ? Character(v)[0] : Symbol(v);
	
	CompiledCall call(constant(state, inst.b));
	List arguments = call.dots() < call.arguments().length ? BuildArgs(state, call) : call.arguments();
	
	Value object = reg(state, inst.c);
	Character type = klass(state, object);

	//Search for type-specific method
	Symbol method = state.StrToSym(state.SymToStr(generic) + "." + state.SymToStr(type[0]));
	Value f = get(state, method);
	
	//Search for default
	if(f.isNil()) {
		method = state.StrToSym(state.SymToStr(generic) + ".default");
		f = get(state, method);
	}

	if(f.type == Type::R_function) {
		Function func(f);
		assert(func.body().type == Type::I_closure || func.body().type == Type::I_promise);
		Environment* fenv = CreateEnvironment(state, func.s(), state.frame().environment, Closure(func.body()).code()->slotSymbols);
		MatchArgs(state, state.frame().environment, fenv, func, arguments);
		assign(fenv, Symbol::dotGeneric, generic);
		assign(fenv, Symbol::dotMethod, method);
		assign(fenv, Symbol::dotClass, type); 
		return buildStackFrame(state, fenv, true, Closure(func.body()).code(), &reg(state, inst.c), &inst+1);
	} else if(f.type == Type::R_cfunction) {
		reg(state, inst.c) = CFunction(f).func(state, arguments);
		return &inst+1;
	} else {
		_error(std::string("no applicable method for '") + state.SymToStr(generic) + "' applied to an object of class \"" + state.SymToStr(type[0]) + "\"");
	}
}
Instruction const* get_op(State& state, Instruction const& inst) {
	reg(state, inst.c) = get(state, Symbol(inst.a));
	if(reg(state, inst.c).isNil()) throw RiposteError(std::string("object '") + state.SymToStr(Symbol(inst.a)) + "' not found");
	return &inst+1;
}
Instruction const* sget_op(State& state, Instruction const& inst) {
	reg(state, inst.c) = sget(state, inst.a);
	if(reg(state, inst.c).isNil()) throw RiposteError(std::string("object '") + state.SymToStr(state.frame().environment->slotName(inst.a)) + "' not found");
	return &inst+1;
}
Instruction const* kget_op(State& state, Instruction const& inst) {
	reg(state, inst.c) = constant(state, inst.a);
	return &inst+1;
}
Instruction const* iget_op(State& state, Instruction const& inst) {
	reg(state, inst.c) = state.path[0]->get(Symbol(inst.a));
	if(reg(state, inst.c).isNil()) throw RiposteError(std::string("object '") + state.SymToStr(Symbol(inst.a)) + "' not found");
	return &inst+1;
}
Instruction const* assign_op(State& state, Instruction const& inst) {
	assign(state, Symbol(inst.a), reg(state, inst.c));
	return &inst+1;
}
Instruction const* sassign_op(State& state, Instruction const& inst) {
	sassign(state, inst.a, reg(state, inst.c));
	return &inst+1;
}
// everything else should be in registers

Instruction const* iassign_op(State& state, Instruction const& inst) {
	// a = value, b = index, c = dest 
	subAssign(state, reg(state,inst.c), reg(state,inst.b), reg(state,inst.a), reg(state,inst.c));
	return &inst+1;
}
Instruction const* eassign_op(State& state, Instruction const& inst) {
	// a = value, b = index, c = dest 
	Value v = reg(state, inst.c);
	if(v.isList()) {
		List r = Clone(List(v));
		r[As<Integer>(state, reg(state,inst.b))[0]-1] = reg(state,inst.a);
		reg(state, inst.c) = r;
		return &inst+1;
	}
	else {
		subAssign(state, reg(state,inst.c), reg(state,inst.b), reg(state,inst.a), reg(state,inst.c));
		return &inst+1;
	}
}
Instruction const* forbegin_op(State& state, Instruction const& inst) {
	//TODO: need to keep a stack of these...
	Value loopVector = reg(state, inst.c);
	reg(state, inst.c) = Null::singleton;
	reg(state, inst.c+1) = loopVector;
	reg(state, inst.c+2) = Integer::c(0);
	if(reg(state, inst.c+2).i >= (int64_t)reg(state, inst.c+1).length) { return &inst+inst.a; }
	assign(state, Symbol(inst.b), Element(Vector(loopVector), 0));
	return &inst+1;
}
Instruction const* forend_op(State& state, Instruction const& inst) {
	if(++reg(state, inst.c+2).i < (int64_t)reg(state, inst.c+1).length) {
		assign(state, Symbol(inst.b), Element(Vector(reg(state, inst.c+1)), reg(state, inst.c+2).i));
		return profile_back_edge(state,&inst+inst.a);
	} else return &inst+1;
}
Instruction const* iforbegin_op(State& state, Instruction const& inst) {
	double m = asReal1(reg(state, inst.c));
	double n = asReal1(reg(state, inst.c+1));
	reg(state, inst.c) = Null::singleton;
	reg(state, inst.c+1) = Integer::c(n > m ? 1 : -1);
	reg(state, inst.c+1).length = (int64_t)n+1;	// danger! this register no longer holds a valid object, but it saves a register and makes the for and ifor cases more similar
	reg(state, inst.c+2) = Integer::c((int64_t)m);
	if(reg(state, inst.c+2).i >= (int64_t)reg(state, inst.c+1).length) { return &inst+inst.a; }
	assign(state, Symbol(inst.b), Integer::c(m));
	return &inst+1;
}
Instruction const* iforend_op(State& state, Instruction const& inst) {
	if((reg(state, inst.c+2).i+=reg(state, inst.c+1).i) < (int64_t)reg(state, inst.c+1).length) { 
		assign(state, Symbol(inst.b), reg(state, inst.c+2));
		return profile_back_edge(state,&inst+inst.a);
	} else return &inst+1;
}
Instruction const* whilebegin_op(State& state, Instruction const& inst) {
	Logical l(reg(state,inst.b));
	reg(state, inst.c) = Null::singleton;
	if(l[0]) return &inst+1;
	else return &inst+inst.a;
}
Instruction const* whileend_op(State& state, Instruction const& inst) {
	Logical l(reg(state,inst.b));
	if(l[0]) return profile_back_edge(state,&inst+inst.a);
	else return &inst+1;
}
Instruction const* repeatbegin_op(State& state, Instruction const& inst) {
	reg(state,inst.c) = Null::singleton;
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
	Logical l = As<Logical>(state, reg(state,inst.b));
	if(l.length == 0) _error("if argument is of zero length");
	if(l[0]) return &inst+1;
	else return &inst+inst.a;
}
Instruction const* if0_op(State& state, Instruction const& inst) {
	Logical l = As<Logical>(state, reg(state, inst.b));
	if(l.length == 0) _error("if argument is of zero length");
	if(!l[0]) return &inst+1;
	else return &inst+inst.a;
}
Instruction const* colon_op(State& state, Instruction const& inst) {
	double from = asReal1(reg(state,inst.a));
	double to = asReal1(reg(state,inst.b));
	reg(state,inst.c) = Sequence(from, to>from?1:-1, fabs(to-from)+1);
	return &inst+1;
}
Instruction const* seq_op(State& state, Instruction const& inst) {
	int64_t len = As<Integer>(state, reg(state, inst.a))[0];
	reg(state, inst.c) = Sequence(len);
	return &inst+1;
}
Instruction const* add_op(State& state, Instruction const& inst) {
	binaryArith<Zip2, AddOp>(state, reg(state, inst.a), reg(state, inst.b), reg(state, inst.c));
	return &inst+1;
}
Instruction const* pos_op(State& state, Instruction const& inst) {
	unaryArith<Zip1, PosOp>(state, reg(state, inst.a), reg(state, inst.c));
	return &inst+1;
}
Instruction const* sub_op(State& state, Instruction const& inst) {
	binaryArith<Zip2, SubOp>(state, reg(state, inst.a), reg(state, inst.b), reg(state, inst.c));
	return &inst+1;
}
Instruction const* neg_op(State& state, Instruction const& inst) {
	unaryArith<Zip1, NegOp>(state, reg(state, inst.a), reg(state, inst.c));
	return &inst+1;
}
Instruction const* mul_op(State& state, Instruction const& inst) {
	binaryArith<Zip2, MulOp>(state, reg(state, inst.a), reg(state, inst.b), reg(state, inst.c));
	return &inst+1;
}
Instruction const* div_op(State& state, Instruction const& inst) {
	binaryArith<Zip2, DivOp>(state, reg(state, inst.a), reg(state, inst.b), reg(state, inst.c));
	return &inst+1;
}
Instruction const* idiv_op(State& state, Instruction const& inst) {
	binaryArith<Zip2, IDivOp>(state, reg(state, inst.a), reg(state, inst.b), reg(state, inst.c));
	return &inst+1;
}
Instruction const* mod_op(State& state, Instruction const& inst) {
	binaryArith<Zip2, ModOp>(state, reg(state, inst.a), reg(state, inst.b), reg(state, inst.c));
	return &inst+1;
}
Instruction const* pow_op(State& state, Instruction const& inst) {
	binaryArith<Zip2, PowOp>(state, reg(state, inst.a), reg(state, inst.b), reg(state, inst.c));
	return &inst+1;
}
Instruction const* lnot_op(State& state, Instruction const& inst) {
	unaryLogical<Zip1, LNotOp>(state, reg(state, inst.a), reg(state, inst.c));
	return &inst+1;
}
Instruction const* land_op(State& state, Instruction const& inst) {
	binaryLogical<Zip2, AndOp>(state, reg(state, inst.a), reg(state, inst.b), reg(state, inst.c));
	return &inst+1;
}
Instruction const* lor_op(State& state, Instruction const& inst) {
	binaryLogical<Zip2, OrOp>(state, reg(state, inst.a), reg(state, inst.b), reg(state, inst.c));
	return &inst+1;
}
Instruction const* eq_op(State& state, Instruction const& inst) {
	binaryOrdinal<Zip2, EqOp>(state, reg(state, inst.a), reg(state, inst.b), reg(state, inst.c));
	return &inst+1;
}
Instruction const* neq_op(State& state, Instruction const& inst) {
	binaryOrdinal<Zip2, NeqOp>(state, reg(state, inst.a), reg(state, inst.b), reg(state, inst.c));
	return &inst+1;
}
Instruction const* lt_op(State& state, Instruction const& inst) {
	binaryOrdinal<Zip2, LTOp>(state, reg(state, inst.a), reg(state, inst.b), reg(state, inst.c));
	return &inst+1;
}
Instruction const* le_op(State& state, Instruction const& inst) {
	binaryOrdinal<Zip2, LEOp>(state, reg(state, inst.a), reg(state, inst.b), reg(state, inst.c));
	return &inst+1;
}
Instruction const* gt_op(State& state, Instruction const& inst) {
	binaryOrdinal<Zip2, GTOp>(state, reg(state, inst.a), reg(state, inst.b), reg(state, inst.c));
	return &inst+1;
}
Instruction const* ge_op(State& state, Instruction const& inst) {
	binaryOrdinal<Zip2, GEOp>(state, reg(state, inst.a), reg(state, inst.b), reg(state, inst.c));
	return &inst+1;
}
Instruction const* abs_op(State& state, Instruction const& inst) {
	unaryArith<Zip1, AbsOp >(state, reg(state, inst.a), reg(state, inst.c));
	return &inst+1;
}
Instruction const* sign_op(State& state, Instruction const& inst) {
	unaryArith<Zip1, SignOp >(state, reg(state, inst.a), reg(state, inst.c));
	return &inst+1;
}
Instruction const* sqrt_op(State& state, Instruction const& inst) {
	unaryArith<Zip1, SqrtOp >(state, reg(state, inst.a), reg(state, inst.c));
	return &inst+1;
}
Instruction const* floor_op(State& state, Instruction const& inst) {
	unaryArith<Zip1, FloorOp >(state, reg(state, inst.a), reg(state, inst.c));
	return &inst+1;
}
Instruction const* ceiling_op(State& state, Instruction const& inst) {
	unaryArith<Zip1, CeilingOp >(state, reg(state, inst.a), reg(state, inst.c));
	return &inst+1;
}
Instruction const* trunc_op(State& state, Instruction const& inst) {
	unaryArith<Zip1, TruncOp >(state, reg(state, inst.a), reg(state, inst.c));
	return &inst+1;
}
Instruction const* round_op(State& state, Instruction const& inst) {
	unaryArith<Zip1, RoundOp >(state, reg(state, inst.a), reg(state, inst.c));
	return &inst+1;
}
Instruction const* signif_op(State& state, Instruction const& inst) {
	unaryArith<Zip1, SignifOp >(state, reg(state, inst.a), reg(state, inst.c));
	return &inst+1;
}
Instruction const* exp_op(State& state, Instruction const& inst) {
	unaryArith<Zip1, ExpOp >(state, reg(state, inst.a), reg(state, inst.c));
	return &inst+1;
}
Instruction const* log_op(State& state, Instruction const& inst) {
	unaryArith<Zip1, LogOp >(state, reg(state, inst.a), reg(state, inst.c));
	return &inst+1;
}
Instruction const* cos_op(State& state, Instruction const& inst) {
	unaryArith<Zip1, CosOp >(state, reg(state, inst.a), reg(state, inst.c));
	return &inst+1;
}
Instruction const* sin_op(State& state, Instruction const& inst) {
	unaryArith<Zip1, SinOp >(state, reg(state, inst.a), reg(state, inst.c));
	return &inst+1;
}
Instruction const* tan_op(State& state, Instruction const& inst) {
	unaryArith<Zip1, TanOp >(state, reg(state, inst.a), reg(state, inst.c));
	return &inst+1;
}
Instruction const* acos_op(State& state, Instruction const& inst) {
	unaryArith<Zip1, ACosOp >(state, reg(state, inst.a), reg(state, inst.c));
	return &inst+1;
}
Instruction const* asin_op(State& state, Instruction const& inst) {
	unaryArith<Zip1, ASinOp >(state, reg(state, inst.a), reg(state, inst.c));
	return &inst+1;
}
Instruction const* atan_op(State& state, Instruction const& inst) {
	unaryArith<Zip1, ATanOp >(state, reg(state, inst.a), reg(state, inst.c));
	return &inst+1;
}
Instruction const* jmp_op(State& state, Instruction const& inst) {
	return &inst+inst.a;
}
Instruction const* istrue_op(State& state, Instruction const& inst) {
	Logical l = As<Logical>(state, reg(state, inst.a));
	if(l.length == 0) _error("argument is of zero length");
	reg(state, inst.c) = l[0] ? Logical::True() : Logical::False();
	return &inst+1;
}
Instruction const* function_op(State& state, Instruction const& inst) {
	reg(state, inst.c) = Function(constant(state, inst.a), constant(state, inst.b), constant(state, inst.b+1), state.frame().environment);
	return &inst+1;
}
Instruction const* logical1_op(State& state, Instruction const& inst) {
	Integer i = As<Integer>(state, reg(state, inst.a));
	reg(state, inst.c) = Logical(i[0]);
	return &inst+1;
}
Instruction const* integer1_op(State& state, Instruction const& inst) {
	Integer i = As<Integer>(state, reg(state, inst.a));
	reg(state, inst.c) = Integer(i[0]);
	return &inst+1;
}
Instruction const* double1_op(State& state, Instruction const& inst) {
	Integer i = As<Integer>(state, reg(state, inst.a));
	reg(state, inst.c) = Double(i[0]);
	return &inst+1;
}
Instruction const* complex1_op(State& state, Instruction const& inst) {
	Integer i = As<Integer>(state, reg(state, inst.a));
	reg(state, inst.c) = Complex(i[0]);
	return &inst+1;
}
Instruction const* character1_op(State& state, Instruction const& inst) {
	Integer i = As<Integer>(state, reg(state, inst.a));
	Character r = Character(i[0]);
	for(int64_t j = 0; j < r.length; j++) r[j] = Symbol::empty;
	reg(state, inst.c) = r;
	return &inst+1;
}
Instruction const* raw1_op(State& state, Instruction const& inst) {
	Integer i = As<Integer>(state, reg(state, inst.a));
	reg(state, inst.c) = Raw(i[0]);
	return &inst+1;
}
Instruction const* type_op(State& state, Instruction const& inst) {
	Character c(1);
	// Should have a direct mapping from type to symbol.
	c[0] = state.StrToSym(reg(state, inst.a).type.toString());
	reg(state, inst.c) = c;
	return &inst+1;
}
Instruction const * invoketrace_op(State& state, Instruction const & inst) {
	Trace * trace = state.frame().code->traces[inst.a];
	int64_t offset;
	TCStatus status = trace->compiled->execute(state,&offset);
	if(status != TCStatus::SUCCESS) {
		printf("trace: encountered error %s\n",status.toString());
	}
	if(status  != TCStatus::SUCCESS || offset == 0) { //we exited to the trace start instruction, invoke the original instruction here
		Instruction invoketrace = inst;
		const_cast<Instruction&>(inst) = trace->trace_inst;
		Instruction const * pc;
#define BC_SWITCH(bc,str,p) case ByteCode::E_##bc: pc = bc##_op(state,inst); break;
		switch(trace->trace_inst.bc.Enum()) {
			BC_ENUM(BC_SWITCH,0)
		}
		const_cast<Instruction&>(inst) = invoketrace;
		return pc;
#undef BC_SWITCH
	}

	return &inst + offset;
}

Instruction const* ret_op(State& state, Instruction const& inst) {
	*(state.frame().result) = reg(state, inst.c);
	// if this stack frame owns the environment, we can free it for reuse
	// as long as we don't return a closure...
	// TODO: but also can't if an assignment to an out of scope variable occurs (<<-, assign) with a value of a closure!
	if(state.frame().ownEnvironment && reg(state, inst.c).isClosureSafe())
		state.environments.push_back(state.frame().environment);
	state.base = state.frame().returnbase;
	Instruction const* returnpc = state.frame().returnpc;
	state.pop();
	return returnpc;
}
Instruction const* done_op(State& state, Instruction const& inst) {
	// not used. When this instruction is hit, interpreter exits.
	return 0;
}

//#define THREADED_INTERPRETER

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
	state.base -= 32;
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
					inst.bc == ByteCode::done ? glabels[0] : glabels[inst.bc.Enum()+1],
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
    #define LABELS_THREADED(name,type,p) (void*)&&name##_label,
	static const void* labels[] = {&&DONE, BC_ENUM(LABELS_THREADED,0)};
	if(glabels == 0) {
		glabels = &labels[0];
		return;
	}

	if(pc == 0) return;

	goto *(pc->ibc);
	#define LABELED_OP(name,type,p) \
		name##_label: \
			{ pc = name##_op(state, *pc); goto *(pc->ibc); } 
	BC_ENUM(LABELED_OP,0)
	DONE: {}
#else
	while(pc->bc != ByteCode::done) {
		switch(pc->bc.Enum()) {
			#define SWITCH_OP(name,type,p) \
				case ByteCode::E_##name: { pc = name##_op(state, *pc); } break;
			BC_ENUM(SWITCH_OP,0)
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

Value eval(State& state, Closure const& closure) {
	return eval(state, closure.code(), closure.environment());
}

Value eval(State& state, Code const* code) {
	return eval(state, code, state.frame().environment);
}

Value eval(State& state, Code const* code, Environment* environment) {
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
		while((int64_t)state.stack.size() > stackSize)
			state.pop();
		throw;
	}
	return result;
}

