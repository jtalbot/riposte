
#ifndef RIPOSTE_CALL_H

// code for making function calls

#ifdef USE_THREADED_INTERPRETER
static const void** glabels = 0;
#endif

static void printCode(Thread const& thread, Prototype const* prototype, Environment* env) {
	std::cout << "Prototype: " << intToHexStr((int64_t)prototype) << "\t(executing in " << intToHexStr((int64_t)env) << ")" << std::endl;
	std::cout << "\tRegisters: " << prototype->registers << std::endl;
	if(prototype->constants.size() > 0) {
		std::cout << "\tConstants: " << std::endl;
		for(int64_t i = 0; i < (int64_t)prototype->constants.size(); i++)
			std::cout << "\t\t" << i << ":\t" << thread.stringify(prototype->constants[i]) << std::endl;
	}
	if(prototype->bc.size() > 0) {
		std::cout << "\tCode: " << std::endl;
		for(int64_t i = 0; i < (int64_t)prototype->bc.size(); i++) {
			std::cout << "\t\t" << i << ":\t" << prototype->bc[i].toString();
			if(prototype->bc[i].bc == ByteCode::call) {
				std::cout << "\t\t(arguments: " << prototype->calls[prototype->bc[i].b].arguments.size() << ")";
			}
			std::cout << std::endl;
		}
	}
	std::cout << std::endl;
}

static Instruction const* buildStackFrame(Thread& thread, Environment* environment, Prototype const* prototype, Instruction const* returnpc, int64_t stackOffset) {
	//printCode(thread, prototype, environment);
	StackFrame& s = thread.push();
	s.environment = environment;
	s.returnpc = returnpc;
	s.returnbase = thread.base;
	s.prototype = prototype;
	thread.base -= stackOffset;
	
	if(thread.base-prototype->registers < thread.registers)
		throw RiposteError("Register overflow");
	
	if(prototype->constants.size() > 0)
		memcpy(thread.base-(prototype->constants.size()-1), &prototype->constants[0], sizeof(Value)*prototype->constants.size());

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

static Instruction const* buildStackFrame(Thread& thread, Environment* environment, Prototype const* prototype, int64_t resultSlot, Instruction const* returnpc) {
	return buildStackFrame(thread, environment, prototype, returnpc, -resultSlot);
}

static Instruction const* buildStackFrame(Thread& thread, Environment* environment, Prototype const* prototype, Environment* env, String s, Instruction const* returnpc) {
	Instruction const* i = buildStackFrame(thread, environment, prototype, returnpc, thread.frame.prototype->registers);
	thread.frame.dest = (int64_t)s;
	thread.frame.env = env;
	return i;
}

static Instruction const* buildStackFrame(Thread& thread, Environment* environment, Prototype const* prototype, Environment* env, int64_t resultSlot, Instruction const* returnpc) {
	Instruction const* i = buildStackFrame(thread, environment, prototype, returnpc, thread.frame.prototype->registers);
	thread.frame.dest = -resultSlot;
	thread.frame.env = env;
	return i;
}

static Pair argument(int64_t index, Environment* env, CompiledCall const& call) {
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
				p.v.type = Type::Dotdot;
				p.v.length = index;
				p.v.p = env;
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

static int64_t numArguments(Environment* env, CompiledCall const& call) {
	if(call.dotIndex < (int64_t)call.arguments.size()) {
		// subtract 1 to not count the dots
		return call.arguments.size() - 1 + env->dots.size();
	} else {
		return call.arguments.size();
	}
}

static bool namedArguments(Environment* env, CompiledCall const& call) {
	if(call.dotIndex < (int64_t)call.arguments.size()) {
		return call.named || env->named;
	} else {
		return call.named;
	}
}

inline void argAssign(Thread& thread, Environment* env, Pair const& parameter, Pair const& argument) {
	Value w = argument.v;
	if(!w.isNil()) {
		if(w.isPromise() || w.isDefault()) {
			assert(w.p == 0);
			w.p = env;
		} else if(w.isFuture()) {
			thread.LiveEnvironment(env, w);
		}
		env->insert(parameter.n) = w;
	}
}

inline void dotAssign(Thread& thread, Environment* env, Pair const& argument) {
	Value w = argument.v;
	assert(!w.isDefault());
	if(w.isPromise()) {
		assert(w.p == 0);
		w.p = env;
	}
	else if(w.isFuture()) {
		thread.LiveEnvironment(env, w);
	}
	Pair p;
	p.n = argument.n;
	p.v = w;
	env->dots.push_back(p);
}

static void MatchArgs(Thread& thread, Environment const* env, Environment* fenv, Function const& func, CompiledCall const& call) {
	PairList const& parameters = func.prototype()->parameters;
	PairList const& arguments = call.arguments;

	int64_t const pDotIndex = func.prototype()->dotIndex;
	int64_t const end = std::min((int64_t)arguments.size(), pDotIndex);

	// set parameters from arguments & defaults
	for(int64_t i = 0; i < (int64_t)parameters.size(); i++) {
		argAssign(thread, fenv, parameters[i], 
			(i < end && !arguments[i].v.isNil()) ? arguments[i] : parameters[i]);
	}

	// handle unused arguments
	if(pDotIndex >= (int64_t)parameters.size()) {
		// called function doesn't take dots, unused args is an error 
		if(arguments.size() > parameters.size())
			_error("Unused arguments");
	}
	else {
		// called function has dots, all unused args go into ...
		fenv->named = false; // if no arguments are named, no dots can be either
		fenv->dots.reserve(arguments.size()-end);
		for(int64_t i = end; i < (int64_t)arguments.size(); i++) {
			dotAssign(thread, fenv, arguments[i]);
		}
	}
}

static void MatchNamedArgs(Thread& thread, Environment* env, Environment* fenv, Function const& func, CompiledCall const& call) {
	PairList const& parameters = func.prototype()->parameters;
	int64_t pDotIndex = func.prototype()->dotIndex;
	int64_t numArgs = numArguments(env, call);
	bool named = namedArguments(env, call);

	// set defaults
	for(int64_t i = 0; i < (int64_t)parameters.size(); ++i) {
		argAssign(thread, fenv, parameters[i], parameters[i]);
	}

	if(!named) {
		fenv->named = false; // if no arguments are named, no dots can be either

		// call arguments are not named, do posititional matching up to the prototype's dots
		int64_t end = std::min(numArgs, pDotIndex);
		for(int64_t i = 0; i < end; ++i) {
			Pair const& arg = argument(i, env, call);
			argAssign(thread, fenv, parameters[i], arg);
		}

		// if we have left over arguments, but no parameter dots, error
		if(end < numArgs && pDotIndex >= (int64_t)parameters.size())
			_error("Unused args");
		
		// all unused args go into ...
		for(int64_t i = end; i < numArgs; i++) {
			Pair const& arg = argument(i, env, call);
			dotAssign(thread, fenv, arg);
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
				argAssign(thread, fenv, parameters[j], arg);
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
				dotAssign(thread, fenv, arg);
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

static Instruction const* GenericDispatch(Thread& thread, Instruction const& inst, String op, Value const& a, int64_t out) {
	Value const& f = thread.frame.environment->getRecursive(op);
	if(f.isFunction()) {
		Function func(f);
		Environment* fenv = CreateEnvironment(thread, func.environment(), thread.frame.environment, Null::Singleton());
		List call(0);
		Pair p;
		p.n = Strings::empty;
		p.v = a;
		PairList args;
		args.push_back(p);
		CompiledCall cc(call, args, 1, false);
		MatchArgs(thread, thread.frame.environment, fenv, func, cc);
		return buildStackFrame(thread, fenv, func.prototype(), out, &inst+1);
	}
	_error("Failed to find generic for builtin op");
}

static Instruction const* GenericDispatch(Thread& thread, Instruction const& inst, String op, Value const& a, Value const& b, int64_t out) {
	Value const& f = thread.frame.environment->getRecursive(op);
	if(f.isFunction()) { 
		Function func(f);
		Environment* fenv = CreateEnvironment(thread, func.environment(), thread.frame.environment, Null::Singleton());
		List call(0);
		PairList args;
		Pair p;
		p.n = Strings::empty;
		p.v = a;
		args.push_back(p);
		p.v = b;
		args.push_back(p);
		CompiledCall cc(call, args, 2, false);
		MatchArgs(thread, thread.frame.environment, fenv, func, cc);
		return buildStackFrame(thread, fenv, func.prototype(), out, &inst+1);
	}
	_error("Failed to find generic for builtin op");
}

Instruction const* forceDot(Thread& thread, Instruction const& inst, Value const* a, Environment* env, int64_t index);

Instruction const* forceReg(Thread& thread, Instruction const& inst, Value const* a, String name);

#define REGISTER(i) (*(thread.base+(i)))

// Out register is currently always a register, not memory
#define OUT(thread, i) (*(thread.base+(i)))

#define OPERAND(a, i) \
Value const& a = __builtin_expect((i) <= 0, true) ? \
		*(thread.base+(i)) : \
		thread.frame.environment->getRecursive((String)(i)); 
	
#define FORCE(a, i) \
if(__builtin_expect((i) > 0 && !a.isConcrete(), false)) { \
	if(a.isDotdot()) { \
		Value const& t = ((Environment*)a.p)->dots[a.length].v; \
		if(t.isConcrete()) { \
			thread.frame.environment->insert((String)(i)) = t; \
			thread.LiveEnvironment(thread.frame.environment, t); \
			return &inst; \
		} \
		else return forceDot(thread, inst, &t, (Environment*)a.p, a.length); \
	} \
	else return forceReg(thread, inst, &a, (String)(i)); \
} \

#define DOTDOT(a, i) \
Value const& a = thread.frame.environment->dots[(i)].v;

#define FORCE_DOTDOT(a, i) \
if(!a.isConcrete()) { \
	if(a.isDotdot()) { \
		Value const& t = ((Environment*)a.p)->dots[a.length].v; \
		if(t.isConcrete()) { \
			thread.frame.environment->dots[(i)].v = t; \
			thread.LiveEnvironment(thread.frame.environment, t); \
			return &inst; \
		} \
		else return forceDot(thread, inst, &t, (Environment*)a.p, a.length); \
	} \
	else return forceDot(thread, inst, &a, thread.frame.environment, (i)); \
}

#define BIND(a) \
if(__builtin_expect(a.isFuture(), false)) { \
	thread.Bind(a); \
	return &inst; \
}

bool isTraceableType(Thread const& thread, Value const& a) {
	Type::Enum type = thread.futureType(a);
        return type == Type::Double || type == Type::Integer || type == Type::Logical;
}

bool isTraceableShape(Thread const& thread, Value const& a) {
	IRNode::Shape const& shape = thread.futureShape(a);
	return !shape.blocking && shape.length >= TRACE_VECTOR_WIDTH;
}

bool isTraceableShape(Thread const& thread, Value const& a, Value const& b) {
	IRNode::Shape const& shapea = thread.futureShape(a);
	IRNode::Shape const& shapeb = thread.futureShape(b);
	return 	!shapea.blocking &&
		!shapeb.blocking &&
		(shapea.length >= TRACE_VECTOR_WIDTH || shapeb.length >=TRACE_VECTOR_WIDTH) &&
		!(a.isFuture() && b.isFuture() && shapea.length != shapeb.length);
}

bool isTraceable(Thread const& thread, Value const& a) {
	return 	thread.state.jitEnabled && 
		isTraceableType(thread, a) &&
		isTraceableShape(thread, a);
}

bool isTraceable(Thread const& thread, Value const& a, Value const& b) {
	return  thread.state.jitEnabled &&
		isTraceableType(thread, a) && 
		isTraceableType(thread, b) && 
		isTraceableShape(thread, a, b);
}

template< template<class X> class Group>
bool isTraceable(Thread const& thread, Value const& a) { 
	return 	isTraceable(thread, a);
}

template<>
bool isTraceable<ArithScan>(Thread const& thread, Value const& a) { return false; }

template<>
bool isTraceable<UnifyScan>(Thread const& thread, Value const& a) { return false; }

template< template<class X, class Y> class Group>
bool isTraceable(Thread const& thread, Value const& a, Value const& b) {
	return  isTraceable(thread, a, b);
}

template< template<class X, class Y, class Z> class Group>
bool isTraceable(Thread const& thread, Value const& a, Value const& b, Value const& c) {
	return false;
}

template<>
bool isTraceable<IfElse>(Thread const& thread, Value const& a, Value const& b, Value const& c) { 
	return	thread.state.jitEnabled &&
		isTraceableType(thread, a) &&
		isTraceableType(thread, b) &&
		isTraceableType(thread, c) &&
		isTraceableShape(thread, a, c) &&
		isTraceableShape(thread, b, c);
}

#endif
