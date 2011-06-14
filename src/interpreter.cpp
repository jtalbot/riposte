#include <string>
#include <sstream>
#include <stdexcept>
#include <string>

#include "value.h"
#include "type.h"
#include "bc.h"
#include "ops.h"
#include "internal.h"

// (stack discipline is hard to prove in R, but can we do better?)
// 1. If a function is returned, must assume that it contains upvalues to anything in either
//    its static scope (true upvalues) or dynamic scope (promises)
// 2. Upvalues can be contained in eval'ed code! consider, function(x) return(function() eval(parse(text='x')))
// 3. Functions can be held in any non-basic datatype (though lists seem like the obvious possibility)
// 4. So by (2) can't statically check, 
//		by (3) we'd have to traverse entire returned data structure to check for a function.
// 5. More conservatively, we could flag the creation of any function within a scope,
//		if that scope returns a non-basic type, we'd have to move the environment off the stack.
//    ---but, what about updating references to the environment. Ugly since we don't know
//	  ---which function is the problem or which upvalues will be used.
//    How about: Upon creation of a function (call to `function`), we create a in heap
//      forwarding frame for everything in the dynamic and static scope of the function.
//      (In repeated use, they wouldn't have to be recreated.)
//      When the function is created it points to the forwarder, which passes requests
//      back to the stack implementation. When the stack instance is popped off the stack,
//      all state is copied to the on-heap forwarder which becomes the true instance.
//    Downside is that forwarders have to be created for all frames on stack even if the
//      created function is never returned.
//    Other downside is that accessing through forwarder adds an indirection.
// Conclusion for now: heap allocate environments. 
// Try to make that fast, maybe with a pooled allocator...

static void MatchArgs(State& state, Environment* fenv, Function const& func, List const& arguments) {
	List parameters = func.parameters();
	Character pnames = getNames(parameters);
	// call arguments are not named, do posititional matching
	if(!hasNames(arguments)) {
		for(int64_t i = 0; i < std::min(arguments.length, func.dots()); ++i) {
			fenv->assign(pnames[i], arguments[i]);
		}
		for(int64_t i = std::min(arguments.length, func.dots()); i < parameters.length; ++i) {
			if(!parameters[i].isNil()) fenv->assign(pnames[i], parameters[i]);
		}
		fenv->assign(Symbol::dots, Subset(arguments, func.dots(), std::max((int64_t)0, (int64_t)arguments.length-(int64_t)func.dots())));
	}
	// call arguments are named, do matching by name
	else {
		Character anames = getNames(arguments);
		// populate environment with default values
		for(int64_t i = 0; i < parameters.length; ++i) {
			if(!parameters[i].isNil()) fenv->assign(pnames[i], parameters[i]);
		}	
		// we should be able to cache and reuse this assignment for pairs of functions and call sites.
		static char assignment[64], set[64];
		for(int64_t i = 0; i < arguments.length; i++) assignment[i] = -1;
		for(int64_t i = 0; i < parameters.length; i++) set[i] = -1;
		// named args, search for complete matches
		for(int64_t i = 0; i < arguments.length; ++i) {
			if(anames[i] != Symbol::empty) {
				for(int64_t j = 0; j < parameters.length; ++j) {
					if(pnames[j] != Symbol::dots && anames[i] == pnames[j]) {
						fenv->assign(pnames[j], arguments[i]);
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
				std::string a = anames[i].toString(state);
				for(int64_t j = 0; j < parameters.length; ++j) {
					if(set[j] < 0 && pnames[j] != Symbol::dots &&
						pnames[i].toString(state).compare( 0, a.size(), a ) == 0 ) {	
						fenv->assign(pnames[j], arguments[i]);
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
						fenv->assign(pnames[firstEmpty], arguments[i]);
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
					names[idx++] = anames[j];
				}
			}
			setNames(values, names);

			fenv->assign(Symbol::dots, values);
		}
	}
}

static int64_t call_function(State& state, Function const& func, List const& arguments) {
	if(func.body().type == Type::I_closure || func.body().type == Type::I_promise) {
		//See note above about allocating Environment on heap...
		Environment* fenv = new Environment(func.s(), state.global);
		fenv->assign(Symbol(state, "..args"), arguments);

		MatchArgs(state, fenv, func, arguments);
	
		eval(state, Closure(func.body()).bind(fenv));
	}
	else
		state.registers[0] = func.body();
	return 1;
}

static int64_t call_op(State& state, Closure const& closure, Instruction const& inst) {
	Value func = state.registers[inst.c];
	CompiledCall call(closure.constants()[inst.a]);
	List arguments = Clone(call.arguments());
	// Specialize the precompiled promises to evaluate in the current scope
	for(int64_t i = 0; i < arguments.length; i++) {
		if(arguments[i].type == Type::I_promise)
			arguments[i].env = (void*)state.global;
	}

	if(call.dots() < arguments.length) {
		// Expand dots into the parameter list...
		// If it's in the dots it must already be a promise, thus no need to make a promise again.
		// Need to do the same for the names...
		Value v;
		state.global->get(state, Symbol::dots, v);
		if(!v.isNull()) {
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
	}
	if(func.type == Type::R_function) {
		Value* old_registers = state.registers;
		state.registers = &(state.registers[inst.c]);
		call_function(state, Function(func), arguments);
		state.registers = old_registers;
		return 1;
	} else if(func.type == Type::R_cfunction) {
		Value* old_registers = state.registers;
		state.registers = &(state.registers[inst.c]);
		CFunction(func).func(state, call.call(), arguments);
		state.registers = old_registers;
		return 1;
	} else {
		_error(std::string("Non-function (") + func.type.toString() + ") as first parameter to call\n");
		return 1;
	}	
}
static int64_t UseMethod_op(State& state, Closure const& closure, Instruction const& inst) {
	Value v = state.registers[inst.a];
	Symbol generic;
	if(v.isCharacter())
		generic = Character(v)[0];
	else
		generic = Symbol(v);
	
	Value arguments;
	state.global->get(state, Symbol(state, "..args"), arguments);
	
	Value object;	
	if(state.registers[inst.b].i == 1)
		object = state.registers[inst.a+1];
	else
		object = force(state, List(arguments)[0]);

	Character type = klass(state, object);

	Value* old_registers = state.registers;
	state.registers = &(state.registers[inst.c]);
	
	//Search for first method
	Symbol method = Symbol(state, generic.toString(state) + "." + type[0].toString(state));
	bool success = state.global->get(state, method, state.registers[0] /* because we've rebased the registers pointer*/);
	
	//Search for default
	if(!success) {
		method = Symbol(state, generic.toString(state) + ".default");
		bool success = state.global->get(state, method, state.registers[0] /* because we've rebased the registers pointer*/);
		if(!success) {
			if(!success) throw RiposteError(std::string("no applicable method for '") + generic.toString(state) + "' applied to an object of class \"" + type[0].toString(state) + "\"");
		}
	}
	state.registers = old_registers;

	Function func = state.registers[inst.c];
	
	if(func.body().type == Type::I_closure || func.body().type == Type::I_promise) {
		Environment* fenv = new Environment(func.s(), state.global);
		fenv->assign(Symbol(state, "..args"), arguments);

		MatchArgs(state, fenv, func, arguments);

		fenv->assign(Symbol(state, ".Generic"), generic);
		fenv->assign(Symbol(state, ".Method"), method);
		fenv->assign(Symbol(state, ".Class"), type);
	
		eval(state, Closure(func.body()).bind(fenv));
	}
	else
		state.registers[0] = func.body();
	return 1;
}
static int64_t get_op(State& state, Closure const& closure, Instruction const& inst) {
	Value* old_registers = state.registers;
	state.registers = &(state.registers[inst.c]);
	bool success = state.global->get(state, Symbol(inst.a), state.registers[0] /* because we've rebased the registers pointer*/);
	state.registers = old_registers;
	if(!success) throw RiposteError(std::string("object '") + Symbol(inst.a).toString(state) + "' not found");
	return 1;
}
static int64_t kget_op(State& state, Closure const& closure, Instruction const& inst) {
	state.registers[inst.c] = closure.constants()[inst.a];
	return 1;
}
static int64_t iget_op(State& state, Closure const& closure, Instruction const& inst) {
	state.path[0]->get(state, Symbol(inst.a), state.registers[inst.c]);
	return 1;
}
static int64_t assign_op(State& state, Closure const& closure, Instruction const& inst) {
	if(!Symbol(inst.c).isAssignable()) {
		throw RiposteError("cannot assign to that symbol");
	}
	state.global->assign(Symbol(inst.c), state.registers[inst.a]);
	// assign assumes that source is equal to destination
	return 1;
}
static int64_t iassign_op(State& state, Closure const& closure, Instruction const& inst) {
	// a = value, b = index, c = dest 
	subAssign(state, state.registers[inst.c], state.registers[inst.b], state.registers[inst.a], state.registers[inst.c]);
	return 1;
}
static int64_t eassign_op(State& state, Closure const& closure, Instruction const& inst) {
	// a = value, b = index, c = dest 
	Value v = state.registers[inst.c];
	if(v.isList()) {
		List r = Clone(List(v));
		r[As<Integer>(state, state.registers[inst.b])[0]-1] = state.registers[inst.a];
		state.registers[inst.c] = r;
		return 1;
	}
	else {
		subAssign(state, state.registers[inst.c], state.registers[inst.b], state.registers[inst.a], state.registers[inst.c]);
		return 1;
	}
}
static int64_t forbegin_op(State& state, Closure const& closure, Instruction const& inst) {
	//TODO: need to keep a stack of these...
	Value loopVector = state.registers[inst.c];
	state.registers[inst.c] = Null::singleton;
	state.registers[inst.c+1] = loopVector;
	state.registers[inst.c+2] = Integer::c(0);
	if(state.registers[inst.c+2].i >= (int64_t)state.registers[inst.c+1].length) { return inst.a; }
	state.global->assign(Symbol(inst.b), Element(loopVector, 0));
	return 1;
}
static int64_t forend_op(State& state, Closure const& closure, Instruction const& inst) {
	if(++state.registers[inst.c+2].i < (int64_t)state.registers[inst.c+1].length) { 
		state.global->assign(Symbol(inst.b), Element(state.registers[inst.c+1], state.registers[inst.c+2].i));
		return inst.a; 
	} else return 1;
}
static int64_t iforbegin_op(State& state, Closure const& closure, Instruction const& inst) {
	double m = asReal1(state.registers[inst.c]);
	double n = asReal1(state.registers[inst.c+1]);
	state.registers[inst.c] = Null::singleton;
	state.registers[inst.c+1] = Integer::c(n > m ? 1 : -1);
	state.registers[inst.c+1].length = (int64_t)n+1;	// danger! this register no longer holds a valid object, but it saves a register and makes the for and ifor cases more similar
	state.registers[inst.c+2] = Integer::c((int64_t)m);
	if(state.registers[inst.c+2].i >= (int64_t)state.registers[inst.c+1].length) { return inst.a; }
	state.global->assign(Symbol(inst.b), Integer::c(m));
	return 1;
}
static int64_t iforend_op(State& state, Closure const& closure, Instruction const& inst) {
	if((state.registers[inst.c+2].i+=state.registers[inst.c+1].i) < (int64_t)state.registers[inst.c+1].length) { 
		state.global->assign(Symbol(inst.b), state.registers[inst.c+2]);
		return inst.a; 
	} else return 1;
}
static int64_t whilebegin_op(State& state, Closure const& closure, Instruction const& inst) {
	Logical l(state.registers[inst.b]);
	state.registers[inst.c] = Null::singleton;
	if(l[0]) return 1;
	else return inst.a;
}
static int64_t whileend_op(State& state, Closure const& closure, Instruction const& inst) {
	Logical l(state.registers[inst.b]);
	if(l[0]) return inst.a;
	else return 1;
}
static int64_t repeatbegin_op(State& state, Closure const& closure, Instruction const& inst) {
	state.registers[inst.c] = Null::singleton;
	return 1;
}
static int64_t repeatend_op(State& state, Closure const& closure, Instruction const& inst) {
	return inst.a;
}
static int64_t next_op(State& state, Closure const& closure, Instruction const& inst) {
	return inst.a;
}
static int64_t break1_op(State& state, Closure const& closure, Instruction const& inst) {
	return inst.a;
}
static int64_t if1_op(State& state, Closure const& closure, Instruction const& inst) {
	Logical l = As<Logical>(state, state.registers[inst.b]);
	if(l.length == 0) _error("if argument is of zero length");
	if(l[0]) return 1;
	else return inst.a;
}
static int64_t if0_op(State& state, Closure const& closure, Instruction const& inst) {
	Logical l = As<Logical>(state, state.registers[inst.b]);
	if(l.length == 0) _error("if argument is of zero length");
	if(!l[0]) return 1;
	else return inst.a;
}
static int64_t colon_op(State& state, Closure const& closure, Instruction const& inst) {
	double from = asReal1(state.registers[inst.a]);
	double to = asReal1(state.registers[inst.b]);
	state.registers[inst.c] = Sequence(from, to>from?1:-1, fabs(to-from)+1);
	return 1;
}
static int64_t seq_op(State& state, Closure const& closure, Instruction const& inst) {
	int64_t len = As<Integer>(state, state.registers[inst.a])[0];
	state.registers[inst.c] = Sequence(len);
	return 1;
}
static int64_t add_op(State& state, Closure const& closure, Instruction const& inst) {
	//if(!isObject(stack.peek()) && !isObject(stack.peek(1)))
	binaryArith<Zip2, AddOp>(state, state.registers[inst.a], state.registers[inst.b], state.registers[inst.c]);
	//else
	//	groupGeneric2(state, stack, closure, inst);
	return 1;
}
static int64_t pos_op(State& state, Closure const& closure, Instruction const& inst) {
	unaryArith<Zip1, PosOp>(state,state.registers[inst.a], state.registers[inst.c]);
	return 1;
}
static int64_t sub_op(State& state, Closure const& closure, Instruction const& inst) {
	binaryArith<Zip2, SubOp>(state, state.registers[inst.a], state.registers[inst.b], state.registers[inst.c]);
	return 1;
}
static int64_t neg_op(State& state, Closure const& closure, Instruction const& inst) {
	unaryArith<Zip1, NegOp>(state,state.registers[inst.a], state.registers[inst.c]);
	return 1;
}
static int64_t mul_op(State& state, Closure const& closure, Instruction const& inst) {
	binaryArith<Zip2, MulOp>(state, state.registers[inst.a], state.registers[inst.b], state.registers[inst.c]);
	return 1;
}
static int64_t div_op(State& state, Closure const& closure, Instruction const& inst) {
	binaryArith<Zip2, DivOp>(state, state.registers[inst.a], state.registers[inst.b], state.registers[inst.c]);
	return 1;
}
static int64_t idiv_op(State& state, Closure const& closure, Instruction const& inst) {
	binaryArith<Zip2, IDivOp>(state, state.registers[inst.a], state.registers[inst.b], state.registers[inst.c]);
	return 1;
}
static int64_t mod_op(State& state, Closure const& closure, Instruction const& inst) {
	binaryArith<Zip2, ModOp>(state, state.registers[inst.a], state.registers[inst.b], state.registers[inst.c]);
	return 1;
}
static int64_t pow_op(State& state, Closure const& closure, Instruction const& inst) {
	binaryArith<Zip2, PowOp>(state, state.registers[inst.a], state.registers[inst.b], state.registers[inst.c]);
	return 1;
}
static int64_t lnot_op(State& state, Closure const& closure, Instruction const& inst) {
	unaryLogical<Zip1, LNotOp >(state,state.registers[inst.a], state.registers[inst.c]);
	return 1;
}
static int64_t land_op(State& state, Closure const& closure, Instruction const& inst) {
	binaryLogical<Zip2, AndOp>(state, state.registers[inst.a], state.registers[inst.b], state.registers[inst.c]);
	return 1;
}
static int64_t lor_op(State& state, Closure const& closure, Instruction const& inst) {
	binaryLogical<Zip2, OrOp>(state, state.registers[inst.a], state.registers[inst.b], state.registers[inst.c]);
	return 1;
}
static int64_t eq_op(State& state, Closure const& closure, Instruction const& inst) {
	binaryOrdinal<Zip2, EqOp>(state, state.registers[inst.a], state.registers[inst.b], state.registers[inst.c]);
	return 1;
}
static int64_t neq_op(State& state, Closure const& closure, Instruction const& inst) {
	binaryOrdinal<Zip2, NeqOp>(state, state.registers[inst.a], state.registers[inst.b], state.registers[inst.c]);
	return 1;
}
static int64_t lt_op(State& state, Closure const& closure, Instruction const& inst) {
	binaryOrdinal<Zip2, LTOp>(state, state.registers[inst.a], state.registers[inst.b], state.registers[inst.c]);
	return 1;
}
static int64_t le_op(State& state, Closure const& closure, Instruction const& inst) {
	binaryOrdinal<Zip2, LEOp>(state, state.registers[inst.a], state.registers[inst.b], state.registers[inst.c]);
	return 1;
}
static int64_t gt_op(State& state, Closure const& closure, Instruction const& inst) {
	binaryOrdinal<Zip2, GTOp>(state, state.registers[inst.a], state.registers[inst.b], state.registers[inst.c]);
	return 1;
}
static int64_t ge_op(State& state, Closure const& closure, Instruction const& inst) {
	binaryOrdinal<Zip2, GEOp>(state, state.registers[inst.a], state.registers[inst.b], state.registers[inst.c]);
	return 1;
}
static int64_t abs_op(State& state, Closure const& closure, Instruction const& inst) {
	unaryArith<Zip1, AbsOp >(state,state.registers[inst.a], state.registers[inst.c]);
	return 1;
}
static int64_t sign_op(State& state, Closure const& closure, Instruction const& inst) {
	unaryArith<Zip1, SignOp >(state,state.registers[inst.a], state.registers[inst.c]);
	return 1;
}
static int64_t sqrt_op(State& state, Closure const& closure, Instruction const& inst) {
	unaryArith<Zip1, SqrtOp >(state,state.registers[inst.a], state.registers[inst.c]);
	return 1;
}
static int64_t floor_op(State& state, Closure const& closure, Instruction const& inst) {
	unaryArith<Zip1, FloorOp >(state,state.registers[inst.a], state.registers[inst.c]);
	return 1;
}
static int64_t ceiling_op(State& state, Closure const& closure, Instruction const& inst) {
	unaryArith<Zip1, CeilingOp >(state,state.registers[inst.a], state.registers[inst.c]);
	return 1;
}
static int64_t trunc_op(State& state, Closure const& closure, Instruction const& inst) {
	unaryArith<Zip1, TruncOp >(state,state.registers[inst.a], state.registers[inst.c]);
	return 1;
}
static int64_t round_op(State& state, Closure const& closure, Instruction const& inst) {
	unaryArith<Zip1, RoundOp >(state,state.registers[inst.a], state.registers[inst.c]);
	return 1;
}
static int64_t signif_op(State& state, Closure const& closure, Instruction const& inst) {
	unaryArith<Zip1, SignifOp >(state,state.registers[inst.a], state.registers[inst.c]);
	return 1;
}
static int64_t exp_op(State& state, Closure const& closure, Instruction const& inst) {
	unaryArith<Zip1, ExpOp >(state,state.registers[inst.a], state.registers[inst.c]);
	return 1;
}
static int64_t log_op(State& state, Closure const& closure, Instruction const& inst) {
	unaryArith<Zip1, LogOp >(state,state.registers[inst.a], state.registers[inst.c]);
	return 1;
}
static int64_t cos_op(State& state, Closure const& closure, Instruction const& inst) {
	unaryArith<Zip1, CosOp >(state,state.registers[inst.a], state.registers[inst.c]);
	return 1;
}
static int64_t sin_op(State& state, Closure const& closure, Instruction const& inst) {
	unaryArith<Zip1, SinOp >(state,state.registers[inst.a], state.registers[inst.c]);
	return 1;
}
static int64_t tan_op(State& state, Closure const& closure, Instruction const& inst) {
	unaryArith<Zip1, TanOp >(state,state.registers[inst.a], state.registers[inst.c]);
	return 1;
}
static int64_t acos_op(State& state, Closure const& closure, Instruction const& inst) {
	unaryArith<Zip1, ACosOp >(state,state.registers[inst.a], state.registers[inst.c]);
	return 1;
}
static int64_t asin_op(State& state, Closure const& closure, Instruction const& inst) {
	unaryArith<Zip1, ASinOp >(state,state.registers[inst.a], state.registers[inst.c]);
	return 1;
}
static int64_t atan_op(State& state, Closure const& closure, Instruction const& inst) {
	unaryArith<Zip1, ATanOp >(state,state.registers[inst.a], state.registers[inst.c]);
	return 1;
}
static int64_t jmp_op(State& state, Closure const& closure, Instruction const& inst) {
	return (int64_t)inst.a;
}
static int64_t null_op(State& state, Closure const& closure, Instruction const& inst) {
	state.registers[inst.c] = Null::singleton;
	return 1;
}
static int64_t true1_op(State& state, Closure const& closure, Instruction const& inst) {
	state.registers[inst.c] = Logical::True();
	return 1;
}
static int64_t false1_op(State& state, Closure const& closure, Instruction const& inst) {
	state.registers[inst.c] = Logical::False();
	return 1;
}
static int64_t NA_op(State& state, Closure const& closure, Instruction const& inst) {
	state.registers[inst.c] = Logical::NA();
	return 1;
}
static int64_t istrue_op(State& state, Closure const& closure, Instruction const& inst) {
	Logical l = As<Logical>(state, state.registers[inst.a]);
	if(l.length == 0) _error("argument is of zero length");
	if(l[0]) state.registers[inst.c] = Logical::True();
	else state.registers[inst.c] = Logical::False();
	return 1;
}
static int64_t function_op(State& state, Closure const& closure, Instruction const& inst) {
	state.registers[inst.c] = Function(closure.constants()[inst.a], closure.constants()[inst.b], closure.constants()[inst.b+1], state.global);
	return 1;
}
static int64_t logical1_op(State& state, Closure const& closure, Instruction const& inst) {
	Integer i = As<Integer>(state, state.registers[inst.a]);
	state.registers[inst.c] = Logical(i[0]);
	return 1;
}
static int64_t integer1_op(State& state, Closure const& closure, Instruction const& inst) {
	Integer i = As<Integer>(state, state.registers[inst.a]);
	state.registers[inst.c] = Integer(i[0]);
	return 1;
}
static int64_t double1_op(State& state, Closure const& closure, Instruction const& inst) {
	Integer i = As<Integer>(state, state.registers[inst.a]);
	state.registers[inst.c] = Double(i[0]);
	return 1;
}
static int64_t complex1_op(State& state, Closure const& closure, Instruction const& inst) {
	Integer i = As<Integer>(state, state.registers[inst.a]);
	state.registers[inst.c] = Complex(i[0]);
	return 1;
}
static int64_t character1_op(State& state, Closure const& closure, Instruction const& inst) {
	Integer i = As<Integer>(state, state.registers[inst.a]);
	Character r = Character(i[0]);
	for(int64_t j = 0; j < r.length; j++) r[j] = Symbol::empty;
	state.registers[inst.c] = r;
	return 1;
}
static int64_t raw1_op(State& state, Closure const& closure, Instruction const& inst) {
	Integer i = As<Integer>(state, state.registers[inst.a]);
	state.registers[inst.c] = Raw(i[0]);
	return 1;
}
static int64_t type_op(State& state, Closure const& closure, Instruction const& inst) {
	Character c(1);
	// Should have a direct mapping from type to symbol.
	c[0] = Symbol(state, state.registers[inst.a].type.toString());
	state.registers[0] = c;
	return 1;
}
static int64_t ret_op(State& state, Closure const& closure, Instruction const& inst) {
	// not used. Jumps to done label instead
	return 0;
}


// 
//
//    Main interpreter loop 
//
#define THREADED_INTERPRETER
//
//

//__attribute__((__noinline__,__noclone__)) 
void eval(State& state, Closure const& closure) {
	//std::cout << "Compiled code: " << state.stringify(closure) << std::endl;
	Environment* oldenv = state.global;
	if(closure.environment() != 0) state.global = closure.environment();	

	Instruction const* pc;

#ifdef THREADED_INTERPRETER
    #define LABELS_THREADED(name,type,p) (void*)&&name##_label,
	static const void* labels[] = {BC_ENUM(LABELS_THREADED,0)};

	/* Initialize threadedCode in closure if not yet done */
	if(closure.threadedCode().size() == 0)
	{
		for(int64_t i = 0; i < (int64_t)closure.code().size(); ++i) {
			Instruction const& inst = closure.code()[i];
			closure.threadedCode().push_back(
				Instruction(
					inst.bc == ByteCode::ret ? (void*)&&DONE : labels[inst.bc.Enum()],
					inst.a, inst.b, inst.c));
		}
	}

	pc = &(closure.threadedCode()[0]);
	goto *(pc->ibc);
	#define LABELED_OP(name,type,p) \
		name##_label: \
			pc += name##_op(state, closure, *pc); goto *(pc->ibc); 
	BC_ENUM(LABELED_OP,0)
	DONE: {}	
#else
	pc = &(closure.code()[0]);
	while(pc->bc != ByteCode::ret) {
		Instruction const& inst = *pc;
		switch(inst.bc.internal()) {
			#define SWITCH_OP(name,type,p) \
				case ByteCode::E##name: pc += name##_op(state, closure, inst); break;
			BC_ENUM(SWITCH_OP,0)
		};
	}
#endif
	state.global = oldenv;
	state.registers[0] = state.registers[pc->c];
}

