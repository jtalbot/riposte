#include <string>
#include <sstream>
#include <stdexcept>
#include <string>

#include "value.h"
#include "type.h"
#include "bc.h"
#include "internal.h"

extern Value force(State& state, Value const& v);
extern Value quoted(Value const& v);
extern Value code(Value const& v);

extern void zip2(Value const& arg0, Value const& arg1, Value& result, Value const& f);
extern CFunction::Cffi AddInternal;

std::map<std::string, uint64_t> Symbol::symbolTable;
std::map<uint64_t, std::string> Symbol::reverseSymbolTable;
const Value Value::null = Value(Type::R_null, (void*)0);
const Value Value::NIL = Value(Type::I_nil, (void*)0);


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
	
	uint64_t nargs = inst.a;
	Value func(stack.pop());
			
	if(func.type() == Type::R_function) {
		Function f(func);
		if(f.body().type() == Type::I_bytecode || f.body().type() == Type::I_promise || f.body().type() == Type::I_sympromise) {
			//See note above about allocating Environment on heap...
			Environment* fenv = new Environment(f.s(), state.env);
			//Environment* fenv = &envs[env_index];
			//fenv->init(f.s(), state.env);
			Character names(f.args().names());
			for(uint64_t i = 0; i < nargs; ++i) {
				fenv->assign(names[i], stack.pop());
			}
			//env_index++;
			eval(state, f.body(), fenv);
			//env_index--;
		}
		else
			stack.push(f.body());
	} else if(func.type() == Type::R_cfunction) {
		CFunction f(func);
		f.func(state, nargs);
	} else {
		printf("Non-function as first parameter to call\n");
		assert(false);
	}
	return 1;
}
static int64_t get_op(State& state, Stack& stack, Block const& block, Instruction const& inst) {
	state.env->get(state, Symbol(inst.a), stack.reserve());
	return 1;
}
static int64_t kget_op(State& state, Stack& stack, Block const& block, Instruction const& inst) {
	stack.push(block.constants()[inst.a]);
	return 1;
}
static int64_t delay_op(State& state, Stack& stack, Block const& block, Instruction const& inst) {
	Value::set(stack.reserve(), Type::I_promise, block.constants()[inst.a].ptr()); 
	return 1;
}
static int64_t symdelay_op(State& state, Stack& stack, Block const& block, Instruction const& inst) {
	Value::set(stack.reserve(), Type::I_sympromise, block.constants()[inst.a].ptr()); 
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
static int64_t forbegin_op(State& state, Stack& stack, Block const& block, Instruction const& inst) {
	Value sym = stack.pop();
	Value lower = stack.pop();
	Value upper = stack.pop();
	double k = asReal1(upper)-asReal1(lower);
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

	if(stack.peek().i < 0) {
		Value::set(stack.peek(), Type::R_null, 0);
		return 1;
	}	
	else
		return -inst.a;
}
static int64_t fguard_op(State& state, Stack& stack, Block const& block, Instruction const& inst) {
	Value function = stack.pop();
	Value specialized = block.constants()[inst.a];
	//printf("In guard %s == %s\n", function.toString().c_str(), specialized.toString().c_str());
	if(function == specialized) {
		//printf("Passed guard\n");
		return 1;
	} else {
		//printf("Failed guard, adding %d\n", inst.c);
		eval(state, block.constants()[inst.b]);
		return inst.c;
	}	
}
static int64_t add_op(State& state, Stack& stack, Block const& block, Instruction const& inst) {
	AddInternal(state,inst.a);
	return 1;

	//Value a = stack.pop();
	//Value b = stack.pop();
	//if(a.type() == Type::R_double && b.type() == Type::R_double &&
	//	a.packed == 1 && b.packed == 1) {
	//	Value v(Type::R_double, a.d+b.d);
	//	stack.push(v);
	//	return 1;
	//}
	//return 0;
}
static int64_t ret_op(State& state, Stack& stack, Block const& block, Instruction const& inst) {
	return 0;
}

#define THREADED_INTERPRETER

// Execute block in current environment...
__attribute__((__noinline__,__noclone__)) 
void eval(State& state, Block const& block) {
	
	Stack& stack = *(state.stack);

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

