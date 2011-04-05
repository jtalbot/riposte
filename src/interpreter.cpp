#include <string>
#include <sstream>
#include <stdexcept>
#include <string>

#include "value.h"
#include "type.h"
#include "bc.h"
#include "internal.h"

const Value Value::null = Value(Type::R_null, (void*)0, (Attributes*)0);
const Value Value::NIL = Value(Type::I_nil, (void*)0, (Attributes*)0);


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

void eval(State& state, Block const& block, Environment* env); 


static int64_t call_op(State& state, Stack& stack, Block const& block, Instruction const& inst) {
	//static Environment envs[1024];
	//static uint64_t env_index = 0;
	
	Value func(stack.pop());
	Call compiledCall(block.constants()[inst.a]);
	
	if(func.type == Type::R_function) {
		Function f(func);
		if(f.body().type == Type::I_bytecode || 
			f.body().type == Type::I_promise || 
			f.body().type == Type::I_sympromise) {
			//See note above about allocating Environment on heap...
			Environment* fenv = new Environment(f.s(), state.env);
			//Environment* fenv = &envs[env_index];
			//fenv->init(f.s(), state.env);
			
			PairList parameters = f.parameters();
			Character pnames(parameters.attributes->names);
			// populate environment with default values
			for(uint64_t i = 0; i < parameters.length(); ++i) {
				fenv->assign(pnames[i], parameters[i]);
			}	

			// call arguments are not named, do posititional matching
			if(compiledCall.attributes == 0 || compiledCall.attributes->names.type == Type::R_null)
			{
				for(uint64_t i = 1; i < compiledCall.length(); ++i) {
					fenv->assign(pnames[i-1], compiledCall[i]);
				}
			}
			// call arguments are named, do matching by name
			else {
				Character argNames(compiledCall.attributes->names);
				for(uint64_t i = 1; i < compiledCall.length(); ++i) {
					// named arg, search for match
					if(argNames[i] != 0) {
						for(uint64_t j = 0; j < parameters.length(); ++j) {
							if(argNames[i] == pnames[j]) {
								fenv->assign(pnames[j], compiledCall[i]);
							}
						}
					}
				}
				uint64_t firstEmpty = 0;
				for(uint64_t i = 1; i < compiledCall.length(); ++i) {
					// unnamed arg in a named argument list, fill in first missing spot.
					if(argNames[i] == 0) {
						for(; firstEmpty < parameters.length(); ++firstEmpty) {
							Value v;
							fenv->getRaw(pnames[firstEmpty], v);
							if(v.type == Type::I_default || v.type == Type::I_symdefault) {
								fenv->assign(pnames[firstEmpty], compiledCall[i]);
								break;
							}
						}
					}
				}
			}

			//env_index++;
			if(f.body().type == Type::I_sympromise)
				fenv->get(state, Symbol(f.body()), stack.reserve());
			else	
				eval(state, f.body(), fenv);
			//env_index--;
		}
		else
			stack.push(f.body());
	} else if(func.type == Type::R_cfunction) {
		CFunction f(func);
		for(uint64_t i = compiledCall.length()-1; i > 0; --i) {
			stack.push(compiledCall[i]);
		}
		f.func(state, compiledCall.length()-1);
	} else {
		printf("Non-function as first parameter to call\n");
		assert(false);
	}
	return 1;
}
static int64_t inlinecall_op(State& state, Stack& stack, Block const& block, Instruction const& inst) {
	Value specialized = block.constants()[inst.b];
	//printf("In guard %s == %s\n", function.toString().c_str(), specialized.toString().c_str());
	if(stack.peek() ==  specialized) {
		stack.pop();
		//printf("Passed guard\n");
		return 1;
	} else {
		//printf("Failed guard, adding %d\n", inst.c);
		call_op(state, stack, block, inst);
		return inst.c;
	}	
}
static int64_t get_op(State& state, Stack& stack, Block const& block, Instruction const& inst) {
	state.env->get(state, Symbol(inst.a), stack.reserve());
	return 1;
}
static int64_t kget_op(State& state, Stack& stack, Block const& block, Instruction const& inst) {
	stack.push(block.constants()[inst.a]);
	return 1;
}
static int64_t pop_op(State& state, Stack& stack, Block const& block, Instruction const& inst) {
	stack.pop();
	return 1;
}
static int64_t assign_op(State& state, Stack& stack, Block const& block, Instruction const& inst) {
	//Value arg0 = quoted(stack.pop());
	//Value arg1 = force(state, stack.pop());
	//Value arg0 = stack.pop();
	Value arg1 = stack.pop();
	stack.push(state.env->assign(Symbol(inst.a), arg1));
	return 1;
}
static int64_t classassign_op(State& state, Stack& stack, Block const& block, Instruction const& inst) {
	Value v = stack.peek();
	Value k;
	state.env->get(state, Symbol(inst.a), k);
	setClass(k.attributes, v);
	state.env->assign(Symbol(inst.a), k);
	return 1;
}
static int64_t namesassign_op(State& state, Stack& stack, Block const& block, Instruction const& inst) {
	Value v = stack.peek();
	Value k;
	state.env->get(state, Symbol(inst.a), k);
	setNames(k.attributes, v);
	state.env->assign(Symbol(inst.a), k);
	return 1;
}
static int64_t forbegin_op(State& state, Stack& stack, Block const& block, Instruction const& inst) {
	Value sym = stack.pop();
	Value lower = stack.pop();
	Value upper = stack.pop();
	double k = asReal1(upper)-asReal1(lower);
	stack.push(Value::null);
	stack.reserve().i = (int64_t)k;
	//env->assign(Symbol(inst.a), registers[inst.c]);
	//if(asReal1(registers[inst.c]) > asReal1(registers[inst.b]))
	//i = i + inst.op;
	return 1;
}
static int64_t forend_op(State& state, Stack& stack, Block const& block, Instruction const& inst) {
	// pop the results of the loop...
	stack.pop();
	// decrement the loop variable
	stack.peek().i -= 1;

	if(stack.peek().i < 0) { stack.pop(); return 1;	}
	else return -inst.a;
}
static int64_t whilebegin_op(State& state, Stack& stack, Block const& block, Instruction const& inst) {
	Logical l(stack.pop());
	stack.push(Value::null);
	if(l[0]) return 1;
	else return inst.a;
}
static int64_t whileend_op(State& state, Stack& stack, Block const& block, Instruction const& inst) {
	Logical l(stack.pop());
	// pop the results of the loop...
	stack.pop();
	if(l[0]) return -inst.a;
	else return 1;
}
static int64_t repeatbegin_op(State& state, Stack& stack, Block const& block, Instruction const& inst) {
	stack.push(Value::null);
	return 1;
}
static int64_t repeatend_op(State& state, Stack& stack, Block const& block, Instruction const& inst) {
	// pop the result of the loop...
	stack.pop();
	return -inst.a;
}
static int64_t if1_op(State& state, Stack& stack, Block const& block, Instruction const& inst) {
	Logical l(stack.pop());
	if(l[0]) return 1;
	else return inst.a;
}
static int64_t add_op(State& state, Stack& stack, Block const& block, Instruction const& inst) {
	if(!isObject(stack.peek()) && !isObject(stack.peek(1)))
		binaryArith<Zip2, AddOp>(state,inst.a);
	//else
	//	groupGeneric2(state, stack, block, inst);
	return 1;
}
static int64_t pos_op(State& state, Stack& stack, Block const& block, Instruction const& inst) {
	unaryArith<Zip1, PosOp>(state,inst.a);
	return 1;
}
static int64_t sub_op(State& state, Stack& stack, Block const& block, Instruction const& inst) {
	binaryArith<Zip2, SubOp>(state,inst.a);
	return 1;
}
static int64_t neg_op(State& state, Stack& stack, Block const& block, Instruction const& inst) {
	unaryArith<Zip1, NegOp>(state,inst.a);
	return 1;
}
static int64_t mul_op(State& state, Stack& stack, Block const& block, Instruction const& inst) {
	binaryArith<Zip2, MulOp>(state,inst.a);
	return 1;
}
static int64_t div_op(State& state, Stack& stack, Block const& block, Instruction const& inst) {
	binaryDoubleArith<Zip2, DivOp>(state,inst.a);
	return 1;
}
static int64_t idiv_op(State& state, Stack& stack, Block const& block, Instruction const& inst) {
	binaryArith<Zip2, IDivOp>(state,inst.a);
	return 1;
}
static int64_t mod_op(State& state, Stack& stack, Block const& block, Instruction const& inst) {
	binaryArith<Zip2, ModOp>(state,inst.a);
	return 1;
}
static int64_t pow_op(State& state, Stack& stack, Block const& block, Instruction const& inst) {
	binaryDoubleArith<Zip2, PowOp>(state,inst.a);
	return 1;
}
static int64_t lneg_op(State& state, Stack& stack, Block const& block, Instruction const& inst) {
	unaryLogical<Zip1, LNegOp>(state,inst.a);
	return 1;
}
static int64_t land_op(State& state, Stack& stack, Block const& block, Instruction const& inst) {
	binaryLogical<Zip2, AndOp>(state,inst.a);
	return 1;
}
static int64_t sland_op(State& state, Stack& stack, Block const& block, Instruction const& inst) {
	/* NYI */
	return 1;
}
static int64_t lor_op(State& state, Stack& stack, Block const& block, Instruction const& inst) {
	binaryLogical<Zip2, OrOp>(state,inst.a);
	return 1;
}
static int64_t slor_op(State& state, Stack& stack, Block const& block, Instruction const& inst) {
	/* NYI */
	return 1;
}
static int64_t eq_op(State& state, Stack& stack, Block const& block, Instruction const& inst) {
	binaryOrdinal<Zip2, EqOp>(state,inst.a);
	return 1;
}
static int64_t neq_op(State& state, Stack& stack, Block const& block, Instruction const& inst) {
	binaryOrdinal<Zip2, NeqOp>(state,inst.a);
	return 1;
}
static int64_t lt_op(State& state, Stack& stack, Block const& block, Instruction const& inst) {
	binaryOrdinal<Zip2, LTOp>(state,inst.a);
	return 1;
}
static int64_t le_op(State& state, Stack& stack, Block const& block, Instruction const& inst) {
	binaryOrdinal<Zip2, LEOp>(state,inst.a);
	return 1;
}
static int64_t gt_op(State& state, Stack& stack, Block const& block, Instruction const& inst) {
	binaryOrdinal<Zip2, GTOp>(state,inst.a);
	return 1;
}
static int64_t ge_op(State& state, Stack& stack, Block const& block, Instruction const& inst) {
	binaryOrdinal<Zip2, GEOp>(state,inst.a);
	return 1;
}
static int64_t abs_op(State& state, Stack& stack, Block const& block, Instruction const& inst) {
	unaryArith<Zip1, AbsOp>(state,inst.a);
	return 1;
}
static int64_t sign_op(State& state, Stack& stack, Block const& block, Instruction const& inst) {
	unaryArith<Zip1, SignOp>(state,inst.a);
	return 1;
}
static int64_t sqrt_op(State& state, Stack& stack, Block const& block, Instruction const& inst) {
	unaryArith<Zip1, SqrtOp>(state,inst.a);
	return 1;
}
static int64_t floor_op(State& state, Stack& stack, Block const& block, Instruction const& inst) {
	unaryArith<Zip1, FloorOp>(state,inst.a);
	return 1;
}
static int64_t ceiling_op(State& state, Stack& stack, Block const& block, Instruction const& inst) {
	unaryArith<Zip1, CeilingOp>(state,inst.a);
	return 1;
}
static int64_t trunc_op(State& state, Stack& stack, Block const& block, Instruction const& inst) {
	unaryArith<Zip1, TruncOp>(state,inst.a);
	return 1;
}
static int64_t round_op(State& state, Stack& stack, Block const& block, Instruction const& inst) {
	unaryArith<Zip1, RoundOp>(state,inst.a);
	return 1;
}
static int64_t signif_op(State& state, Stack& stack, Block const& block, Instruction const& inst) {
	unaryArith<Zip1, SignifOp>(state,inst.a);
	return 1;
}
static int64_t exp_op(State& state, Stack& stack, Block const& block, Instruction const& inst) {
	unaryArith<Zip1, ExpOp>(state,inst.a);
	return 1;
}
static int64_t log_op(State& state, Stack& stack, Block const& block, Instruction const& inst) {
	unaryArith<Zip1, LogOp>(state,inst.a);
	return 1;
}
static int64_t cos_op(State& state, Stack& stack, Block const& block, Instruction const& inst) {
	unaryArith<Zip1, CosOp>(state,inst.a);
	return 1;
}
static int64_t sin_op(State& state, Stack& stack, Block const& block, Instruction const& inst) {
	unaryArith<Zip1, SinOp>(state,inst.a);
	return 1;
}
static int64_t tan_op(State& state, Stack& stack, Block const& block, Instruction const& inst) {
	unaryArith<Zip1, TanOp>(state,inst.a);
	return 1;
}
static int64_t acos_op(State& state, Stack& stack, Block const& block, Instruction const& inst) {
	unaryArith<Zip1, ACosOp>(state,inst.a);
	return 1;
}
static int64_t asin_op(State& state, Stack& stack, Block const& block, Instruction const& inst) {
	unaryArith<Zip1, ASinOp>(state,inst.a);
	return 1;
}
static int64_t atan_op(State& state, Stack& stack, Block const& block, Instruction const& inst) {
	unaryArith<Zip1, ATanOp>(state,inst.a);
	return 1;
}
static int64_t jmp_op(State& state, Stack& stack, Block const& block, Instruction const& inst) {
	return (int64_t)inst.a;
}
static int64_t ret_op(State& state, Stack& stack, Block const& block, Instruction const& inst) {
	return 0;
}

#define THREADED_INTERPRETER

// Execute block in current environment...
__attribute__((__noinline__,__noclone__)) 
void eval(State& state, Block const& block) {
	
	Stack& stack = state.stack;

#ifdef THREADED_INTERPRETER
    #define LABELS_THREADED(name,type) (void*)&&name##_label,
	static const void* labels[] = {BC_ENUM(LABELS_THREADED)};

	/* Initialize threadedCode in block if not yet done */
	if(block.threadedCode().size() == 0)
	{
		for(uint64_t i = 0; i < block.code().size(); ++i) {
			Instruction const& inst = block.code()[i];
			block.threadedCode().push_back(
				Instruction(
					inst.bc == ByteCode::ret ? (void*)&&DONE : labels[inst.bc.internal()],
					inst.a, inst.b, inst.c));
		}
	}

	Instruction const* pc = &(block.threadedCode()[0]);
	goto *(pc->ibc);
	#define LABELED_OP(name,type) \
		name##_label: \
			pc += name##_op(state, stack, block, *pc); goto *(pc->ibc); 
	BC_ENUM(LABELED_OP)
	DONE:
	{}
#else
	int64_t pc = 0;
    while(block.code()[pc].bc != ByteCode::ret) {
		Instruction const& inst = block.code()[pc];
		switch(inst.bc.internal()) {
			#define SWITCH_OP(name,type) \
				case ByteCode::name: pc += name##_op(state, stack, block, inst); break;
			BC_ENUM(SWITCH_OP)
		};
	}
#endif
}

// Execute block in specified environment
void eval(State& state, Block const& block, Environment* env) {
	Environment* oldenv = state.env;
	state.env = env;
	eval(state, block);
	state.env = oldenv;
}

