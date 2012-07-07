
#include "call.h"

Instruction const* buildStackFrame(Thread& thread, Environment* environment, Prototype const* prototype, Instruction const* returnpc, int64_t stackOffset) {
	//std::cout "\t(Executing in " << intToHexStr((int64_t)env) << ")" << std::endl;
	//Prototype::printCode(prototype, thread.state);
	
	// make new stack frame
	StackFrame& s = thread.push();
	s.environment = environment;
	s.prototype = prototype;
	s.returnpc = returnpc;
	s.registers += stackOffset;
	
	if(s.registers+prototype->registers > thread.registers+DEFAULT_NUM_REGISTERS)
		throw RiposteError("Register overflow");
	
	return &(prototype->bc[0]);
}

Instruction const* buildStackFrame(Thread& thread, Environment* environment, Prototype const* prototype, int64_t resultSlot, Instruction const* returnpc) {
	return buildStackFrame(thread, environment, prototype, returnpc, -resultSlot);
}

Instruction const* buildStackFrame(Thread& thread, Environment* environment, Prototype const* prototype, Environment* env, String s, Instruction const* returnpc) {
	Instruction const* i = buildStackFrame(thread, environment, prototype, returnpc, thread.frame.prototype->registers);
	thread.frame.dest = (int64_t)s;
	thread.frame.env = env;
	return i;
}

Instruction const* buildStackFrame(Thread& thread, Environment* environment, Prototype const* prototype, Environment* env, int64_t resultSlot, Instruction const* returnpc) {
	Instruction const* i = buildStackFrame(thread, environment, prototype, returnpc, thread.frame.prototype->registers);
	thread.frame.dest = -resultSlot;
	thread.frame.env = env;
	return i;
}

inline void assignArgument(Thread& thread, Environment* evalEnv, Environment* assignEnv, String n, Value const& v) {
	assert(!v.isFuture());
	
	Value& w = assignEnv->insert(n);
	w = v;
	if(v.isPromise()) {
		((Promise&)w).environment(evalEnv);
	}
}

inline void assignDot(Thread& thread, Environment* evalEnv, Environment* assignEnv, String n, Value const& v) {
	Pair p;
	p.n = n;
	p.v = v;

	if(v.isPromise()) {
		((Promise&)p.v).environment(evalEnv);
	}
	assert(!v.isFuture());
	//else if(v.isFuture()) {
	//	thread.traces.LiveEnvironment(assignEnv, v);
	//}
	
	assignEnv->dots.push_back(p);
}


Pair argument(int64_t index, Environment* env, CompiledCall const& call) {
	if(index < call.dotIndex) {
		return call.arguments[index];
	} else {
		index -= call.dotIndex;
		if(index < (int64_t)env->dots.size()) {
			// Promises in the dots can't be passed down 
			//     (general rule is that promises only
			//	occur once anywhere in the program). 
			// But everything else can be passed down.
			if(env->dots[index].v.isPromise()) {
				Pair p;
				p.n = env->dots[index].n;
				Promise::Init(p.v, env, index, false);
				return p;
			} 
			else {
				return env->dots[index];
			}
		}
		else {
			index -= env->dots.size();
			return call.arguments[call.dotIndex+index+1];
		}
	}
}

int64_t numArguments(Environment* env, CompiledCall const& call) {
	if(call.dotIndex < (int64_t)call.arguments.size()) {
		// subtract 1 to not count the dots
		return call.arguments.size() - 1 + env->dots.size();
	} else {
		return call.arguments.size();
	}
}

bool namedArguments(Environment* env, CompiledCall const& call) {
	if(call.dotIndex < (int64_t)call.arguments.size()) {
		return call.named || env->named;
	} else {
		return call.named;
	}
}


// Generic argument matching
void MatchArgs(Thread& thread, Environment* env, Environment* fenv, Function const& func, CompiledCall const& call) {
	PairList const& parameters = func.prototype()->parameters;
	int64_t pDotIndex = func.prototype()->dotIndex;
	int64_t numArgs = numArguments(env, call);
	bool named = namedArguments(env, call);

	// set defaults
	for(int64_t i = 0; i < (int64_t)parameters.size(); ++i) {
		assignArgument(thread, fenv, fenv, parameters[i].n, parameters[i].v);
	}

	if(!named) {
		fenv->named = false; // if no arguments are named, no dots can be either

		// call arguments are not named, do posititional matching up to the prototype's dots
		int64_t end = std::min(numArgs, pDotIndex);
		for(int64_t i = 0; i < end; ++i) {
			Pair const& arg = argument(i, env, call);
			if(!arg.v.isNil())
				assignArgument(thread, env, fenv, parameters[i].n, arg.v);
		}

		// if we have left over arguments, but no parameter dots, error
		if(end < numArgs && pDotIndex >= (int64_t)parameters.size())
			_error("Unused args");
		
		// all unused args go into ...
		for(int64_t i = end; i < numArgs; i++) {
			Pair const& arg = argument(i, env, call);
			assignDot(thread, env, fenv, arg.n, arg.v);
		}
	}
	else {
		// call arguments are named, do matching by name
		// we should be able to cache and reuse this assignment for pairs of functions and call sites.
	
		int64_t *assignment = thread.assignment, *set = thread.set;
		for(int64_t i = 0; i < numArgs; i++) assignment[i] = -1;
		for(int64_t i = 0; i < (int64_t)parameters.size(); i++) set[i] = -(i+1);

		// named args, search for complete matches
		for(int64_t i = 0; i < numArgs; ++i) {
			Pair const& arg = argument(i, env, call);
			if(arg.n != Strings::empty) {
				for(int64_t j = 0; j < (int64_t)parameters.size(); ++j) {
					if(j != pDotIndex && arg.n == parameters[j].n) {
						assignment[i] = j;
						set[j] = i;
						break;
					}
				}
			}
		}
		// named args, search for incomplete matches
		for(int64_t i = 0; i < numArgs; ++i) {
			Pair const& arg = argument(i, env, call);
			if(arg.n != Strings::empty && assignment[i] < 0) {
				for(int64_t j = 0; j < (int64_t)parameters.size(); ++j) {
					if(set[j] < 0 && j != pDotIndex && strncmp(arg.n, parameters[j].n, strlen(arg.n)) == 0) {
						assignment[i] = j;
						set[j] = i;
						break;
					}
				}
			}
		}
		// unnamed args, fill into first missing spot.
		int64_t firstEmpty = 0;
		for(int64_t i = 0; i < numArgs; ++i) {
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

		// stuff that can't be cached...

		// assign all the arguments
		for(int64_t j = 0; j < (int64_t)parameters.size(); ++j) {
			if(j != pDotIndex && set[j] >= 0) {
				Pair const& arg = argument(set[j], env, call);
				if(!arg.v.isNil())
					assignArgument(thread, env, fenv, parameters[j].n, arg.v);
			}
		}

		// put unused args into the dots
		fenv->named = false;
		for(int64_t i = 0; i < numArgs; i++) {
			if(assignment[i] < 0) {
				// if we have left over arguments, but no parameter dots, error
				if(pDotIndex >= (int64_t)parameters.size()) _error("Unused args");	
				Pair const& arg = argument(i, env, call);
				if(arg.n != Strings::empty) fenv->named = true;
				assignDot(thread, env, fenv, arg.n, arg.v);
			}
		}
	}
}

// Assumes no names and no ... in the argument list.
// Supports ... in the parameter list.
void FastMatchArgs(Thread& thread, Environment* env, Environment* fenv, Function const& func, CompiledCall const& call) {
	Prototype const* prototype = func.prototype();
	PairList const& parameters = prototype->parameters;
	PairList const& arguments = call.arguments;

	int64_t const parametersSize = prototype->parametersSize;
	int64_t const argumentsSize = call.argumentsSize;

	int64_t const pDotIndex = prototype->dotIndex;
	int64_t const end = std::min(argumentsSize, pDotIndex);

	// set parameters from arguments & defaults
	for(int64_t i = 0; i < parametersSize; i++) {
		if(i < end && !arguments[i].v.isNil())
			assignArgument(thread, env, fenv, parameters[i].n, arguments[i].v);
		else
			assignArgument(thread, fenv, fenv, parameters[i].n, parameters[i].v);
	}

	// handle unused arguments
	if(pDotIndex >= parametersSize) {
		// called function doesn't take dots, unused args is an error 
		if(argumentsSize > parametersSize)
			_error("Unused arguments");
	}
	else {
		// called function has dots, all unused args go into ...
		fenv->named = false; // if no arguments are named, no dots can be either
		fenv->dots.reserve(argumentsSize-end);
		for(int64_t i = end; i < (int64_t)argumentsSize; i++) {
			assignDot(thread, env, fenv, arguments[i].n, arguments[i].v);
		}
	}
}

Instruction const* GenericDispatch(Thread& thread, Instruction const& inst, String op, Value const& a, int64_t out) {
	Environment* penv;
	Value const& f = thread.frame.environment->getRecursive(op, penv);
	if(f.isFunction()) {
		Environment* fenv = new Environment(1, ((Function const&)f).environment(), thread.frame.environment, Null::Singleton());
		List call(0);
		Pair p;
		p.n = Strings::empty;
		p.v = a;
		PairList args;
		args.push_back(p);
		CompiledCall cc(call, args, 1, false);
		MatchArgs(thread, thread.frame.environment, fenv, ((Function const&)f), cc);
		return buildStackFrame(thread, fenv, ((Function const&)f).prototype(), out, &inst+1);
	}
	_error("Failed to find generic for builtin op");
}

Instruction const* GenericDispatch(Thread& thread, Instruction const& inst, String op, Value const& a, Value const& b, int64_t out) {
	Environment* penv;
	Value const& f = thread.frame.environment->getRecursive(op, penv);
	if(f.isFunction()) { 
		Environment* fenv = new Environment(2, ((Function const&)f).environment(), thread.frame.environment, Null::Singleton());
		List call(0);
		PairList args;
		Pair p;
		p.n = Strings::empty;
		p.v = a;
		args.push_back(p);
		p.v = b;
		args.push_back(p);
		CompiledCall cc(call, args, 2, false);
		MatchArgs(thread, thread.frame.environment, fenv, ((Function const&)f), cc);
		return buildStackFrame(thread, fenv, ((Function const&)f).prototype(), out, &inst+1);
	}
	_error("Failed to find generic for builtin op");
}

