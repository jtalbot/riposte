#include <string>
#include <sstream>
#include <stdexcept>
#include <string>

#include "value.h"
#include "type.h"
#include "bc.h"
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

void eval(State& state, Closure const& closure, Environment* env); 

static int64_t call_function(State& state, Function const& func, List const& args) {
	if(func.body().type == Type::I_closure || 
		func.body().type == Type::I_promise) {
		//See note above about allocating Environment on heap...
		Environment* fenv = new Environment(func.s(), state.env);
			
		List parameters = func.parameters();
		Character pnames(parameters.attributes->names);
		// populate environment with default values
		for(uint64_t i = 0; i < parameters.length(); ++i) {
			fenv->assign(pnames[i], parameters[i]);
		}	

		// call arguments are not named, do posititional matching
		if(args.attributes == 0 || args.attributes->names.type == Type::R_null)
		{
			uint64_t i = 0;
			for(i = 0; i < std::min(args.length(), pnames.length()); ++i) {
				if(Symbol(pnames[i]) == Symbol::dots)
					break; 
				fenv->assign(pnames[i], args[i]);
			}
			if(i < args.length() && Symbol(pnames[i]) == Symbol::dots) {
				fenv->assign(Symbol::dots, List(Subset(args, i, args.length()-i)));
			}
		}
		// call arguments are named, do matching by name
		else {
			// we should be able to cache and reuse this assignment for pairs of functions and call sites.
			static uint64_t assignment[64], set[64];
			for(uint64_t i = 0; i < args.length(); i++) assignment[i] = 0;
			for(uint64_t i = 0; i < parameters.length(); i++) set[i] = 0;
			Character argNames(args.attributes->names);
			// named args, search for complete matches
			for(uint64_t i = 0; i < args.length(); ++i) {
				if(argNames[i] != 0) {
					for(uint64_t j = 0; j < parameters.length(); ++j) {
						if(Symbol(pnames[j]) != Symbol::dots && argNames[i] == pnames[j]) {
							fenv->assign(pnames[j], args[i]);
							assignment[i] = j+1;
							set[j] = i+1;
							break;
						}
					}
				}
			}
			// named args, search for incomplete matches
			for(uint64_t i = 0; i < args.length(); ++i) {
				std::string a = argNames[i].toString(state);
				if(argNames[i] != 0 && assignment[i] == 0) {
					for(uint64_t j = 0; j < parameters.length(); ++j) {
						if(set[j] == 0 && pnames[j] != Symbol::dots &&
							pnames[i].toString(state).compare( 0, a.size(), a ) == 0 ) {	
							fenv->assign(pnames[j], args[i]);
							assignment[i] = j+1;
							set[j] = i+1;
							break;
						}
					}
				}
			}
			// unnamed args, fill into first missing spot.
			uint64_t firstEmpty = 0;
			for(uint64_t i = 0; i < args.length(); ++i) {
				if(argNames[i] == 0) {
					for(; firstEmpty < parameters.length(); ++firstEmpty) {
						if(pnames[firstEmpty] == Symbol::dots) {
							break;
						}
						fenv->assign(pnames[firstEmpty], args[i]);
						assignment[i] = firstEmpty+1;
						set[firstEmpty] = i+1;
						break;
					}
				}
			}
			// put unused args into the dots
			if(func.dots()) {
				// count up the unassigned args
				uint64_t unassigned = 0;
				for(uint64_t j = 0; j < args.length(); j++) if(assignment[j] == 0) unassigned++;
				List values(unassigned);
				Character names(unassigned);
				uint64_t idx = 0;
				for(uint64_t j = 0; j < args.length(); j++) {
					if(assignment[j] == 0) {
						values[idx] = args[j];
						names[idx++] = argNames[j];
					}
				}
				setNames(values.attributes, names);
				
				fenv->assign(Symbol::dots, values);
			}
		}
		//env_index++;
		eval(state, Closure(func.body()).bind(fenv));
		//env_index--;
	}
	else
		state.registers[0] = func.body();
	return 1;
}
static int64_t call_op(State& state, Closure const& closure, Instruction const& inst) {
	Value func = state.registers[inst.c];
	CompiledCall call(closure.constants()[inst.a]);
	List parameters(call.parameters().length());
	for(uint64_t i = 0; i < parameters.length(); i++) {
		parameters[i] = call.parameters()[i];
		if(parameters[i].type == Type::I_promise)
			parameters[i].attributes = (Attributes*)state.env;
		parameters.attributes = call.parameters().attributes;
	}
	if(call.dots() != 0) {
		// Expand dots into the parameter list...
		// If it's in the dots it must already be a promise, thus no need to make a promise again.
		// Need to do the same for the names...
		Value v;
		state.env->get(state, Symbol::dots, v);
		List dots(v);
		Vector expanded(Type::R_list, parameters.length() + dots.length()-1);
		Insert(parameters, 0, expanded, 0, call.dots()-1);
		Insert(dots, 0, expanded, call.dots()-1, dots.length());
		Insert(parameters, call.dots(), expanded, call.dots()-1+dots.length(), parameters.length()-call.dots());
		if( (parameters.attributes != 0 && getNames(parameters.attributes).type != Type::R_null) ||
		    (dots.attributes != 0 && getNames(dots.attributes).type != Type::R_null) ) {
			Character names(expanded.length());
			for(uint64_t i = 0; i < names.length(); i++) names[i] = 0;
			if(parameters.attributes != 0 && getNames(parameters.attributes).type != Type::R_null) {
				for(uint64_t i = 0; i < call.dots()-1; i++) names[i] = Character(getNames(parameters.attributes))[i];
				for(uint64_t i = 0; i < parameters.length()-call.dots(); i++) names[call.dots()-1+dots.length()+i] = Character(getNames(parameters.attributes))[call.dots()+i];
				//Insert(getNames(parameters.attributes), 0, names, 0, call.dots()-1);
				//Insert(getNames(parameters.attributes), call.dots(), names, call.dots()-1+dots.length(), parameters.length()-call.dots());
			}
		    	if(dots.attributes != 0 && getNames(dots.attributes).type != Type::R_null)
				for(uint64_t i = 0; i < dots.length(); i++) names[call.dots()-1+i] = Character(getNames(dots.attributes))[i];
				//Insert(getNames(dots.attributes), 0, names, call.dots()-1, dots.length());
			setNames(expanded.attributes, names);
		}
		parameters = List(expanded);
	}
	if(func.type == Type::R_function) {
		Value* old_registers = state.registers;
		state.registers = &(state.registers[inst.c]);
		call_function(state, Function(func), parameters);
		state.registers = old_registers;
		return 1;
	} else if(func.type == Type::R_cfunction) {
		Value* old_registers = state.registers;
		state.registers = &(state.registers[inst.c]);
		CFunction(func).func(state, call.call(), parameters);
		state.registers = old_registers;
		return 1;
	} else {
		printf("Non-function as first parameter to call\n");
		assert(false);
		return 1;
	}	
}
static int64_t get_op(State& state, Closure const& closure, Instruction const& inst) {
	state.env->get(state, Symbol(inst.a), state.registers[inst.c]);
	return 1;
}
static int64_t kget_op(State& state, Closure const& closure, Instruction const& inst) {
	state.registers[inst.c] = closure.constants()[inst.a];
	return 1;
}
static int64_t iget_op(State& state, Closure const& closure, Instruction const& inst) {
	state.baseenv->get(state, Symbol(inst.a), state.registers[inst.c]);
	return 1;
}
static int64_t assign_op(State& state, Closure const& closure, Instruction const& inst) {
	if(!Symbol(inst.a).isAssignable()) {
		throw RiposteError("cannot assign to that symbol");
	}
	state.env->assign(Symbol(inst.a), state.registers[inst.c]);
	// assign assumes that source is equal to destination
	return 1;
}
static int64_t classassign_op(State& state, Closure const& closure, Instruction const& inst) {
	Value k;
	state.env->get(state, Symbol(inst.a), k);
	setClass(k.attributes, state.registers[inst.c]);
	state.env->assign(Symbol(inst.a), k);
	state.registers[inst.c] = k;
	return 1;
}
static int64_t namesassign_op(State& state, Closure const& closure, Instruction const& inst) {
	Value k;
	state.env->get(state, Symbol(inst.a), k);
	setNames(k.attributes, state.registers[inst.c]);
	state.env->assign(Symbol(inst.a), k);
	state.registers[inst.c] = k;
	return 1;
}
static int64_t dimassign_op(State& state, Closure const& closure, Instruction const& inst) {
	Value k;
	state.env->get(state, Symbol(inst.a), k);
	setDim(k.attributes, state.registers[inst.c]);
	state.env->assign(Symbol(inst.a), k);
	state.registers[inst.c] = k;
	return 1;
}
static int64_t iassign_op(State& state, Closure const& closure, Instruction const& inst) {
	Value k;
	state.env->get(state, Symbol(inst.a), k);
	subAssign(state, k, state.registers[inst.b], state.registers[inst.c], state.registers[inst.c]);
	state.env->assign(Symbol(inst.a), state.registers[inst.c]);
	return 1;
}
static int64_t iclassassign_op(State& state, Closure const& closure, Instruction const& inst) {
	// TODO: needs indexing
	Value k;
	state.env->get(state, Symbol(inst.a), k);
	setClass(k.attributes, state.registers[inst.c]);
	state.env->assign(Symbol(inst.a), k);
	state.registers[inst.c] = k;
	return 1;
}
static int64_t inamesassign_op(State& state, Closure const& closure, Instruction const& inst) {
	// TODO: needs indexing
	Value k;
	state.env->get(state, Symbol(inst.a), k);
	setNames(k.attributes, state.registers[inst.c]);
	state.env->assign(Symbol(inst.a), k);
	state.registers[inst.c] = k;
	return 1;
}
static int64_t idimassign_op(State& state, Closure const& closure, Instruction const& inst) {
	// TODO: needs indexing
	Value k;
	state.env->get(state, Symbol(inst.a), k);
	setDim(k.attributes, state.registers[inst.c]);
	state.env->assign(Symbol(inst.a), k);
	state.registers[inst.c] = k;
	return 1;
}
static int64_t forbegin_op(State& state, Closure const& closure, Instruction const& inst) {
	//TODO: need to keep a stack of these...
	state.loopVector = state.registers[inst.c];
	state.loopIndex = (int64_t)0;
	state.loopEnd = (int64_t)Vector(state.loopVector).length();
	state.registers[inst.c] = Null::singleton;
	if(state.loopIndex >= state.loopEnd) { return inst.a; }
	state.env->assign(Symbol(inst.b), Element(state.loopVector, state.loopIndex));
	return 1;
}
static int64_t forend_op(State& state, Closure const& closure, Instruction const& inst) {
	if(++state.loopIndex < state.loopEnd) { 
		state.env->assign(Symbol(inst.b), Element(Vector(state.loopVector), state.loopIndex));
		return inst.a; 
	} else return 1;
}
static int64_t iforbegin_op(State& state, Closure const& closure, Instruction const& inst) {
	double m = asReal1(state.registers[inst.c]);
	double n = asReal1(state.registers[inst.c+1]);
	state.loopIndex = (int64_t)m;
	state.loopStep = n > m ? 1 : -1;
	state.loopEnd = (int64_t)n+1;
	state.registers[inst.c] = Null::singleton;
	if(state.loopIndex >= state.loopEnd) { return inst.a; }
	state.env->assign(Symbol(inst.b), Integer::c(state.loopIndex));
	return 1;
}
static int64_t iforend_op(State& state, Closure const& closure, Instruction const& inst) {
	if(++state.loopIndex < state.loopEnd) { 
		state.env->assign(Symbol(inst.b), Integer::c(state.loopIndex));
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
	Logical l(state.registers[inst.b]);
	if(l[0]) return 1;
	else return inst.a;
}
static int64_t colon_op(State& state, Closure const& closure, Instruction const& inst) {
	double from = asReal1(state.registers[inst.a]);
	double to = asReal1(state.registers[inst.b]);
	state.registers[inst.c] = Sequence(from, to>from?1:-1, fabs(to-from)+1);
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
	unaryArith<Zip1, PosOp >(state,state.registers[inst.a], state.registers[inst.c]);
	return 1;
}
static int64_t sub_op(State& state, Closure const& closure, Instruction const& inst) {
	binaryArith<Zip2, SubOp>(state, state.registers[inst.a], state.registers[inst.b], state.registers[inst.c]);
	return 1;
}
static int64_t neg_op(State& state, Closure const& closure, Instruction const& inst) {
	unaryArith<Zip1, NegOp >(state,state.registers[inst.a], state.registers[inst.c]);
	return 1;
}
static int64_t mul_op(State& state, Closure const& closure, Instruction const& inst) {
	binaryArith<Zip2, MulOp>(state, state.registers[inst.a], state.registers[inst.b], state.registers[inst.c]);
	return 1;
}
static int64_t div_op(State& state, Closure const& closure, Instruction const& inst) {
	binaryDoubleArith<Zip2, DivOp>(state, state.registers[inst.a], state.registers[inst.b], state.registers[inst.c]);
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
	binaryDoubleArith<Zip2, PowOp>(state, state.registers[inst.a], state.registers[inst.b], state.registers[inst.c]);
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
static int64_t sland_op(State& state, Closure const& closure, Instruction const& inst) {
	/* NYI */
	return 1;
}
static int64_t lor_op(State& state, Closure const& closure, Instruction const& inst) {
	binaryLogical<Zip2, OrOp>(state, state.registers[inst.a], state.registers[inst.b], state.registers[inst.c]);
	return 1;
}
static int64_t slor_op(State& state, Closure const& closure, Instruction const& inst) {
	/* NYI */
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
static int64_t function_op(State& state, Closure const& closure, Instruction const& inst) {
	state.registers[inst.c] = Function(closure.constants()[inst.a], closure.constants()[inst.b], Character::NA, state.env);
	return 1;
}
static int64_t ret_op(State& state, Closure const& closure, Instruction const& inst) {
	// not used. Jumps to done label instead
	return 0;
}

#define THREADED_INTERPRETER

//__attribute__((__noinline__,__noclone__)) 
void eval(State& state, Closure const& closure) {
	Environment* oldenv = state.env;
	if(closure.environment() != 0) state.env = closure.environment();	

	Instruction const* pc;

#ifdef THREADED_INTERPRETER
    #define LABELS_THREADED(name,type,p) (void*)&&name##_label,
	static const void* labels[] = {BC_ENUM(LABELS_THREADED,0)};

	/* Initialize threadedCode in closure if not yet done */
	if(closure.threadedCode().size() == 0)
	{
		for(uint64_t i = 0; i < closure.code().size(); ++i) {
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
	state.env = oldenv;
	state.registers[0] = state.registers[pc->a];
}

