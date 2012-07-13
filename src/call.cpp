
#include "call.h"

/* Register stack usage for function calls:

high	|  ^  ^  ^  |
	|  grow up  |
	|-----------|  <- StackFrame::reservedTo
	| registers |
	|	    |
	|-----------|  <- StackFrame::registers
	|   slots   |
	|(parameters|
	|  & locals)|
	|-----------|  <- StackFrame::slots		<- Environment::slots
	|  varargs  |
	|(name, val)|
	|-----------|  <- StackFrame::dots		<- Environment::dots
low	|  previous |

	The current frame can overlap the registers of the previous frame
	if they are dead (known at compile time). Overlapping minimizes
	the number of false live variables we detect in GC and tracing.

	Built via:
		1) MatchArgs or FastMatchArgs, returns number of varargs
		2) BuildStackFrame, reserves space on the register stack,
			creates a stackframe & an environment
		   TODO: allocate environment in stackframe, copy off later?
		3) AssignArgs or FastAssignArgs, 
			places arguments in appropriate slot or varargs location
*/

void BuildStackFrame(
	Thread& thread, 
	Code const* code, 
	Environment* lexicalScope, 
	Environment* dynamicScope, 
	CallSite const& callSite,
	size_t liveRegistersInCaller,
	size_t varargs) {

	varargs *= 2; // since we need space for the names too.

	// make new stack frame
	StackFrame& s = thread.push();
	s.dots = s.registers + liveRegistersInCaller;
	s.slots = s.dots + varargs;
	s.registers = s.slots + code->layout->m.size();
	s.reservedTo = s.registers + code->registers;
	s.calls = &code->calls[0];

	if(s.reservedTo > thread.registers+DEFAULT_NUM_REGISTERS)
		throw RiposteError("Register overflow");

	// clear memory. We know the dots will be completely filled in
	//	by AssignArgs, so no need to clear that.
	// Could do better by not clearing parameter slots too.
	// Clearing registers is not needed for correctness, but
	//	avoids false roots for the GC.
	memset((void*)s.slots, 0, (s.reservedTo-s.slots)*sizeof(Value));

	// make a new environment
	Environment* env = new Environment(
		code->layout, lexicalScope, dynamicScope, callSite.call);
	env->dots = s.dots;
	env->slots = s.slots;
	s.environment = env;
}


int64_t numArguments(Environment* env, CallSite const& call) {
	if(call.dotIndex < (int64_t)call.arguments.size()) {
		// subtract 1 to not count the dots
		return call.arguments.size() - 1 + env->numDots();
	} else {
		return call.arguments.size();
	}
}

bool namedArguments(Environment* env, CallSite const& call) {
	if(call.dotIndex < (int64_t)call.arguments.size()) {
		return call.hasNames || env->dotsNamed;
	} else {
		return call.hasNames;
	}
}

Pair argument(int64_t index, Environment* env, CallSite const& call) {
	if(index < call.dotIndex) {
		return call.arguments[index];
	} else {
		index -= call.dotIndex;
		if(index < (int64_t)env->numDots()) {
			Pair p;
			p.n = env->dots[index*2].s;
			p.v = env->dots[index*2+1];
			// Promises in the dots can't be passed down 
			//     (general rule is that promises only
			//	occur once anywhere in the program). 
			// But everything else can be passed down as is.
			if(p.v.isPromise()) {
				Promise::Init(p.v, env, index, false);
			} 
			return p;
		}
		else {
			index -= env->numDots();
			return call.arguments[call.dotIndex+index+1];
		}
	}
}

// Generic argument matching
size_t MatchArgs(Thread& thread, Prototype const* prototype, CallSite const& call) {
	Environment* env = thread.frame.environment;
	PairList const& parameters = prototype->parameters;
	size_t parametersSize = prototype->parametersSize;
	size_t pDotIndex = (size_t)prototype->dotIndex;

	size_t numArgs = (size_t)numArguments(env, call);
	bool named = namedArguments(env, call);

	if(!named) {
		// if the arguments are not named, then we'll just
		// match up to the pDotIndex, after that, everything is varargs. 
		return (size_t)std::max(numArgs - pDotIndex, (size_t)0);
	}
	else {
		// call arguments are named, do matching by name
		// to figure out the assignment and therefore the # of varargs
		// we should be able to cache and reuse this assignment for 
		// pairs of functions and call sites.
	
		int64_t *assignment = thread.assignment, *set = thread.set;
		for(size_t i = 0; i < numArgs; i++) 
			assignment[i] = -1;
		for(size_t i = 0; i < parametersSize; i++)
			set[i] = -(i+1);

		// named args, search for complete matches
		for(size_t i = 0; i < numArgs; ++i) {
			Pair const& arg = argument(i, env, call);
			if(arg.n != Strings::empty) {
				for(size_t j = 0; j < parametersSize; ++j) {
					if(j != pDotIndex && arg.n == parameters[j].n) {
						assignment[i] = j;
						set[j] = i;
						break;
					}
				}
			}
		}
		// named args, search for incomplete matches
		for(size_t i = 0; i < numArgs; ++i) {
			Pair const& arg = argument(i, env, call);
			if(arg.n != Strings::empty && assignment[i] < 0) {
				for(size_t j = 0; j < parametersSize; ++j) {
					if(set[j] < 0 && j != pDotIndex && strncmp(arg.n, parameters[j].n, strlen(arg.n)) == 0) {
						assignment[i] = j;
						set[j] = i;
						break;
					}
				}
			}
		}
		// unnamed args, fill into first missing spot.
		size_t firstEmpty = 0;
		for(size_t i = 0; i < numArgs; ++i) {
			Pair const& arg = argument(i, env, call);
			if(arg.n == Strings::empty) {
				for(; firstEmpty < pDotIndex; ++firstEmpty) {
					if(set[firstEmpty] < 0) {
						assignment[i] = firstEmpty;
						set[firstEmpty] = i;
						break;
					}
				}
			}
		}

		// iterate and count unassigned arguments,
		//  those are varargs.
		size_t varargs = 0;
		for(size_t i = 0; i < numArgs; i++) {
			if(assignment[i] < 0) 
				varargs++;
		}

		return varargs;
	}
}

inline void assignArgument(size_t i, Value const& v, Environment* evalEnv, Environment* assignEnv) {
	assert(!v.isFuture());
	
	Value& w = assignEnv->slots[i];
	w = v;
	if(w.isPromise()) {
		((Promise&)w).environment(evalEnv);
	}
}

inline void assignDot(size_t i, String n, Value const& v, Environment* evalEnv, Environment* assignEnv) {
	assert(!v.isFuture());

	assignEnv->dots[i*2+0] = Character::c(n);

	Value& w = assignEnv->dots[i*2+1];
	w = v;
	if(w.isPromise()) {
		((Promise&)w).environment(evalEnv);
	}
	//else if(v.isFuture()) {
	//	thread.traces.LiveEnvironment(assignEnv, v);
	//}
}

// Generic argument matching
void AssignArgs(Thread& thread, Prototype const* prototype, CallSite const& call, size_t varargs) {
	Environment* fenv = thread.frame.environment;
	Environment* env = thread.frame.environment->DynamicScope();
	PairList const& parameters = prototype->parameters;
	size_t parametersSize = prototype->parametersSize;
	size_t pDotIndex = (size_t)prototype->dotIndex;

	size_t numArgs = (size_t)numArguments(env, call);
	bool named = namedArguments(env, call);

	// set defaults
	for(size_t i = 0; i < parametersSize; ++i) {
		assignArgument(i, parameters[i].v, fenv, fenv);
	}

	fenv->dotsNamed = false;

	if(!named) {
		// do posititional matching up to the prototype's dots
		size_t end = (size_t)std::min(numArgs, pDotIndex);
		for(size_t i = 0; i < end; ++i) {
			Pair const& arg = argument(i, env, call);
			if(!arg.v.isNil())
				assignArgument(i, arg.v, env, fenv);
		}

		// all unused args go into ...
		for(size_t i = end; i < numArgs; i++) {
			Pair const& arg = argument(i, env, call);
			assignDot(i, arg.n, arg.v, env, fenv);
		}
	}
	else {
		int64_t *assignment = thread.assignment, *set = thread.set;
		
		// assign all the arguments
		for(size_t j = 0; j < parametersSize; ++j) {
			if(j != pDotIndex && set[j] >= 0) {
				Pair const& arg = argument(set[j], env, call);
				if(!arg.v.isNil())
					assignArgument(j, arg.v, env, fenv);
			}
		}

		// put unused args into the dots
		for(size_t i = 0, j = 0; i < numArgs; i++) {
			if(assignment[i] < 0) {
				Pair const& arg = argument(i, env, call);
				if(arg.n != Strings::empty) 
					fenv->dotsNamed = true;
				assignDot(j++, arg.n, arg.v, env, fenv);
			}
		}
	}
}


// Assumes no names and no ... in the argument list.
// Supports ... in the parameter list.
size_t FastMatchArgs(Thread& thread, Prototype const* prototype, CallSite const& call) {
	size_t const argumentsSize = (size_t)call.argumentsSize;
	size_t const pDotIndex = (size_t)prototype->dotIndex;

	// Everything after the dot index is vararg
	return std::max(argumentsSize - pDotIndex, (size_t)0);
}

// Assumes no names and no ... in the argument list.
// Supports ... in the parameter list.
void FastAssignArgs(Thread& thread, Prototype const* prototype, CallSite const& call, size_t varargs) {
	Environment* fenv = thread.frame.environment;
	Environment* env = thread.frame.environment->DynamicScope();
	
	PairList const& parameters = prototype->parameters;
	PairList const& arguments = call.arguments;

	size_t const end = std::min((size_t)call.argumentsSize, (size_t)prototype->dotIndex);

	fenv->dotsNamed = false;
	
	// set parameters from arguments & defaults
	for(size_t i = 0; i < (size_t)prototype->parametersSize; i++) {
		if(i < end && !arguments[i].v.isNil())
			assignArgument(i, arguments[i].v, env, fenv);
		else
			assignArgument(i, parameters[i].v, fenv, fenv);
	}

	// called function has dots, all unused args go into ...
	for(size_t i = end; i < (size_t)call.argumentsSize; i++) {
		assignDot(i-end, arguments[i].n, arguments[i].v, env, fenv);
	}
}

void PrepareStack(
	Thread& thread, 
	Prototype const* prototype, 
	Environment* lexicalScope, 
	Environment* dynamicScope, 
	CallSite const& callSite,
	size_t liveRegistersInCaller) {

	bool fast = !callSite.hasNames && !callSite.hasDots;

	size_t varargs = fast
		? FastMatchArgs(thread, prototype, callSite)
		: MatchArgs(thread, prototype, callSite);

	if(varargs > 0 && !prototype->hasDots)
		_error("Unused arguments");

	BuildStackFrame(
		thread, 
		prototype->code, 
		lexicalScope, 
		dynamicScope, 
		callSite, 
		liveRegistersInCaller, 
		varargs);

	if(fast)
		FastAssignArgs(thread, prototype, callSite, varargs);
	else
		AssignArgs(thread, prototype, callSite, varargs);
}

/*
	A promise stack is much simpler. It has registers, but the dots and slots,
	point back up the stack to the lexical scope. It also shares its environment
	with the lexical scope.
*/

void PreparePromiseStack(
	Thread& thread,
	Environment* lexicalScope, 
	Code const* code) {

	// make new stack frame
	StackFrame& s = thread.push();
	s.dots = lexicalScope->dots;
	s.slots = lexicalScope->slots;
	s.registers = s.reservedTo;
	s.reservedTo = s.registers + code->registers;
	s.calls = &code->calls[0];

	if(s.reservedTo > thread.registers+DEFAULT_NUM_REGISTERS)
		throw RiposteError("Register overflow");

	// Clearing registers is not needed for correctness, but
	//	avoids false roots for the GC.
	memset((void*)s.registers, 0, (s.reservedTo-s.registers)*sizeof(Value));

	s.environment = lexicalScope;
}

Instruction const* GenericDispatch(Thread& thread, Instruction const& inst, String op, Value const& a, int64_t out) {
	/*Environment* penv;
	Value const& f = thread.frame.environment->getRecursive(op, penv);
	if(f.isFunction()) {
		Environment* fenv = new Environment(new Template(), ((Function const&)f).environment(), thread.frame.environment, Null::Singleton());
		List call(0);
		Pair p;
		p.n = Strings::empty;
		p.v = a;
		PairList args;
		args.push_back(p);
		CallSite cc(call, args, 1, false);
		MatchArgs(thread, thread.frame.environment, fenv, ((Function const&)f), cc);
		return buildStackFrame(thread, fenv, ((Function const&)f).prototype(), out, &inst+1);
	}*/
	_error("Failed to find generic for builtin op");
}

Instruction const* GenericDispatch(Thread& thread, Instruction const& inst, String op, Value const& a, Value const& b, int64_t out) {
	/*Environment* penv;
	Value const& f = thread.frame.environment->getRecursive(op, penv);
	if(f.isFunction()) { 
		Environment* fenv = new Environment(new Template(), ((Function const&)f).environment(), thread.frame.environment, Null::Singleton());
		List call(0);
		PairList args;
		Pair p;
		p.n = Strings::empty;
		p.v = a;
		args.push_back(p);
		p.v = b;
		args.push_back(p);
		CallSite cc(call, args, 2, false);
		MatchArgs(thread, thread.frame.environment, fenv, ((Function const&)f), cc);
		return buildStackFrame(thread, fenv, ((Function const&)f).prototype(), out, &inst+1);
	}*/
	_error("Failed to find generic for builtin op");
}

