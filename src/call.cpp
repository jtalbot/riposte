
#include "call.h"
#include "frontend.h"

// forces a value stored in the Environments dotdot slot: dest[index]
// call through FORCE_DOTDOT macro which inlines some performance-important checks
Instruction const* forceDot(Thread& thread, Instruction const& inst, Value const& v, Environment* dest, int64_t index) {
	if(v.isPromise()) {
		Promise const& a = (Promise const&)v;
		if(a.isExpression()) {
            Instruction const* r = buildStackFrame(thread, a.environment(), a.code(), &inst, thread.frame.code->registers);
            
            REnvironment::Init(REGISTER(0), dest);
            Integer::InitScalar(REGISTER(1), index);
			
            thread.frame.isPromise = true;
            return r;
		} 
		else if(a.isDotdot()) {
            Value const& t = a.environment()->getContext()->dots[a.dotIndex()].v;
			Instruction const* result = &inst;
			if(!t.isObject()) {
				result = forceDot(thread, inst, t, a.environment(), a.dotIndex());
			}
			if(t.isObject()) {
				((Context*)dest->getContext())->dots[index].v = t;
				thread.traces.LiveEnvironment(dest, t);
			}
			return result;
		}
		else {
			_error("Invalid promise type");
		}
	}
	else {
		_error(std::string("Object '..") + intToStr(index+1) + "' not found, missing argument?");
	} 
}

// forces a value stored in the Environment slot: dest->name
// call through FORCE macro which inlines some performance-important checks
//  for the common cases.
// Environments can have promises, defaults, or dotdots (references to ..n in the parent).
Instruction const* force(Thread& thread, Instruction const& inst, Value const& v, Environment* dest, String name) {
	if(v.isPromise()) {
		Promise a = (Promise const&)v;
		if(a.isExpression()) {
       	    Instruction const* r = buildStackFrame(thread, a.environment(), a.code(), &inst, thread.frame.code->registers);
            
            REnvironment::Init(REGISTER(0), dest);
            Character::InitScalar(REGISTER(1), name);
			
            thread.frame.isPromise = true;
            return r;
        }
		else if(a.isDotdot()) {
            Value const& t = a.environment()->getContext()->dots[a.dotIndex()].v;
			Instruction const* result = &inst;
			// if this dotdot is a promise, attempt to force.
			// first time through this will return the address of the 
			//	promise's new stack frame.
			// second time through this will return the resulting value
			// => Thus, evaluating dotdot requires at most 2 sweeps up the dotdot chain
			if(!t.isObject()) {
				result = forceDot(thread, inst, (Promise const&)t, a.environment(), a.dotIndex());
			}
       	    if(t.isObject()) {
       	       	dest->insert(name) = t;
       	       	thread.traces.LiveEnvironment(dest, t);
            }
            return result;
       	}
		else {
			_error("Invalid promise type");
		}
	}
	else {
		_error(std::string("Object '") + thread.externStr(name) + "' not found"); 
	} 
}

/*Instruction const* force2(Thread& thread, Instruction const& inst, Value const& v, Environment* dest) {
    Promise const& a = (Promise const&)v;

    Prototype const* p = a.isPrototype() ? a.prototype() : allProto;

    return buildStackFrame(thread, a.environment(), p, dest, name, &inst);
}*/

Instruction const* buildStackFrame(Thread& thread, Environment* environment, Code const* code, Instruction const* returnpc, int64_t stackOffset) {
	/*std::cout << "\t(Executing in " << intToHexStr((int64_t)environment) << ")" << std::endl;
	code->printByteCode(thread.state);
    if(environment->getContext() && ((List const &)environment->getContext()->call).length() > 0)
    std::cout << "Call:   " << (((List const&)environment->getContext()->call)[0]).s << std::endl;
	*/
	// make new stack frame
	StackFrame& s = thread.push();
	s.environment = environment;
	s.code = code;
	s.returnpc = returnpc;
	s.registers += stackOffset;
    s.isPromise = false;
	
	if(s.registers+code->registers > thread.registers+DEFAULT_NUM_REGISTERS)
		_internalError("Register overflow");

    // avoid confusing the GC with bogus stuff in registers...
    // can we avoid this somehow?
    for(int64_t i = 0; i < code->registers; ++i) {
        s.registers[i] = Value::Nil();
    }

	return &(code->bc[0]);
}

Instruction const* buildStackFrame(Thread& thread, Environment* environment, Code const* code, int64_t resultSlot, Instruction const* returnpc) {
	return buildStackFrame(thread, environment, code, returnpc, resultSlot);
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
	
	((Context*)assignEnv->getContext())->dots.push_back(p);
}


Pair argument(int64_t index, Environment* env, CompiledCall const& call) {
	if(index < call.dotIndex) {
		return call.arguments[index];
	} else {
		index -= call.dotIndex;
        int64_t ndots = env->getContext() ? env->getContext()->dots.size() : 0;
		if(index < ndots) {
			// Promises in the dots can't be passed down 
			//     (general rule is that promises only
			//	occur once anywhere in the program). 
			// But everything else can be passed down.
			if(env->getContext()->dots[index].v.isPromise()) {
				Pair p;
				p.n = env->getContext()->dots[index].n;
				Promise::Init(p.v, env, index, false);
				return p;
			} 
			else {
				return env->getContext()->dots[index];
			}
		}
		else {
			index -= ndots;
			return call.arguments[call.dotIndex+index+1];
		}
	}
}

int64_t numArguments(Environment* env, CompiledCall const& call) {
	if(call.dotIndex < (int64_t)call.arguments.size() && env->getContext()) {
		// subtract 1 to not count the dots
		return call.arguments.size() - 1 + env->getContext()->dots.size();
	} else {
		return call.arguments.size();
	}
}

bool namedArguments(Environment* env, CompiledCall const& call) {
	if(call.dotIndex < (int64_t)call.arguments.size() && env->getContext()) {
		return call.named || env->getContext()->named;
	} else {
		return call.named;
	}
}


// Generic argument matching
Environment* MatchArgs(Thread& thread, Environment* env, Closure const& func, CompiledCall const& call) {
	
    PairList const& parameters = func.prototype()->parameters;
	int64_t pDotIndex = func.prototype()->dotIndex;
	int64_t numArgs = numArguments(env, call);
	bool named = namedArguments(env, call);

    Context* context = new Context();
    context->parent = env;
    context->call = call.call;
    context->function = func;
    context->nargs = numArgs;
    context->named = named;
    context->onexit = Null::Singleton();

    Environment* fenv = new Environment(
        (int64_t)call.arguments.size(),
        func.environment(),
        context);

    // set extra args
    for(size_t i = 0; i < call.extraArgs.size(); ++i) {
        assignArgument(thread, env, fenv, 
            call.extraArgs[i].n, call.extraArgs[i].v);
    }

	// set defaults
	for(int64_t i = 0; i < (int64_t)parameters.size(); ++i) {
		assignArgument(thread, fenv, fenv, parameters[i].n, parameters[i].v);
	}

	if(!named) {
		context->named = false; // if no arguments are named, no dots can be either

		// call arguments are not named, do posititional matching up to the prototype's dots
		int64_t end = std::min(numArgs, pDotIndex);
		for(int64_t i = 0; i < end; ++i) {
			Pair const& arg = argument(i, env, call);
			if(!arg.v.isNil())
				assignArgument(thread, env, fenv, parameters[i].n, arg.v);
		}

		// if we have left over arguments, but no parameter dots, error
		if(end < numArgs && pDotIndex >= (int64_t)parameters.size())
			_error(std::string("Unused args in call: ") + thread.state.deparse(call.call));
		
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
		// named args, search for incomplete matches but only to the ...
		for(int64_t i = 0; i < numArgs; ++i) {
			Pair const& arg = argument(i, env, call);
			if(arg.n != Strings::empty && assignment[i] < 0) {
				for(int64_t j = 0; j < pDotIndex; ++j) {
					if(set[j] < 0 && strncmp(arg.n, parameters[j].n, strlen(arg.n)) == 0) {
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
		context->named = false;
		for(int64_t i = 0; i < numArgs; i++) {
			if(assignment[i] < 0) {
				// if we have left over arguments, but no parameter dots, error
				if(pDotIndex >= (int64_t)parameters.size())
			        _error(std::string("Unused args in call: ") + thread.state.deparse(call.call));
				Pair const& arg = argument(i, env, call);
				if(arg.n != Strings::empty) context->named = true;
				assignDot(thread, env, fenv, arg.n, arg.v);
			}
		}
	}
    return fenv;
}

// Assumes no names and no ... in the argument list.
// Supports ... in the parameter list.
Environment* FastMatchArgs(Thread& thread, Environment* env, Closure const& func, CompiledCall const& call) {
	
    Prototype const* prototype = func.prototype();
	PairList const& parameters = prototype->parameters;
	PairList const& arguments = call.arguments;

	int64_t const parametersSize = prototype->parametersSize;
	int64_t const argumentsSize = call.argumentsSize;

	int64_t const pDotIndex = prototype->dotIndex;
	int64_t const end = std::min(argumentsSize, pDotIndex);

    Context* context = new Context();
    context->parent = env;
    context->call = call.call;
    context->function = func;
    context->nargs = argumentsSize;
    context->onexit = Null::Singleton();

    Environment* fenv = new Environment(
        (int64_t)call.arguments.size(),
        func.environment(),
        context);

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
			_error(std::string("Unused args in call: ") + thread.state.deparse(call.call));
	}
	else {
		// called function has dots, all unused args go into ...
		context->named = false; // if no arguments are named, no dots can be either
		context->dots.reserve(argumentsSize-end);
		for(int64_t i = end; i < (int64_t)argumentsSize; i++) {
			assignDot(thread, env, fenv, arguments[i].n, arguments[i].v);
		}
	}

    return fenv;
}

Value Quote(Thread& thread, Value const& v) {
    if(isSymbol(v) || isCall(v) || isExpression(v)) {
        List call(2);
        call[0] = CreateSymbol(thread.state.internStr("quote"));
        call[1] = v;
        return CreateCall(call);
    } else {
        return v;
    }
}

Instruction const* GenericDispatch(Thread& thread, Instruction const& inst, String op, Value const& a, int64_t out) {
	Environment* penv;
	Value const& f = thread.frame.environment->getRecursive(op, penv);
	if(f.isClosure()) {
		List call(2);
        call[0] = CreateSymbol(op);
        call[1] = Quote(thread, a);
		Pair p;
		p.n = Strings::empty;
		p.v = a;
		PairList args, extra;
		args.push_back(p);
		CompiledCall cc(CreateCall(call), args, 1, false, extra);
		Environment* fenv = FastMatchArgs(thread, thread.frame.environment, ((Closure const&)f), cc);
		return buildStackFrame(thread, fenv, ((Closure const&)f).prototype()->code, out, &inst+1);
	}
	_error(std::string("Failed to find generic for builtin op: ") + op);
}

Instruction const* GenericDispatch(Thread& thread, Instruction const& inst, String op, Value const& a, Value const& b, int64_t out) {
	Environment* penv;
	Value const& f = thread.frame.environment->getRecursive(op, penv);
	if(f.isClosure()) { 
		List call(3);
        call[0] = CreateSymbol(op);
        call[1] = Quote(thread, a);
        call[2] = Quote(thread, b);
		PairList args, extra;
		Pair p;
		p.n = Strings::empty;
		p.v = a;
		args.push_back(p);
		p.v = b;
		args.push_back(p);
		CompiledCall cc(CreateCall(call), args, 2, false, extra);
		Environment* fenv = FastMatchArgs(thread, thread.frame.environment, ((Closure const&)f), cc);
		return buildStackFrame(thread, fenv, ((Closure const&)f).prototype()->code, out, &inst+1);
	}
	_error(std::string("Failed to find generic for builtin op: ") + op + " type: " + Type::toString(a.type()) + " " + Type::toString(b.type()));
}

Instruction const* GenericDispatch(Thread& thread, Instruction const& inst, String op, Value const& a, Value const& b, Value const& c, int64_t out) {
	Environment* penv;
	Value const& f = thread.frame.environment->getRecursive(op, penv);
	if(f.isClosure()) { 
		List call(4);
        call[0] = CreateSymbol(op);
        call[1] = Quote(thread, a);
        call[2] = Quote(thread, b);
        call[3] = Quote(thread, c);
        Character names(4);
        names[0] = Strings::empty;
        names[1] = Strings::empty;
        names[2] = Strings::empty;
        names[3] = Strings::value;
		PairList args, extra;
		Pair p;
		p.n = Strings::empty;
		p.v = a;
		args.push_back(p);
		p.v = b;
		args.push_back(p);
        p.n = Strings::value;
		p.v = c;
		args.push_back(p);
		CompiledCall cc(CreateCall(call, names), args, 3, true, extra);
        Environment* fenv = MatchArgs(thread, thread.frame.environment, ((Closure const&)f), cc);
		return buildStackFrame(thread, fenv, ((Closure const&)f).prototype()->code, out, &inst+1);
	}
	_error(std::string("Failed to find generic for builtin op: ") + op);
}


Instruction const* StopDispatch(Thread& thread, Instruction const& inst, String msg, int64_t out) {
	Environment* penv;
	Value const& f = thread.frame.environment->getRecursive(thread.internStr("__stop__"), penv);
	if(f.isClosure()) {
        Character v(1);
        v[0] = msg;
		List call(2);
        call[0] = f;
        call[1] = v;
		Pair p;
		p.n = Strings::empty;
		p.v = v;
		PairList args, extra;
		args.push_back(p);
		CompiledCall cc(CreateCall(call), args, 1, false, extra);
		Environment* fenv = FastMatchArgs(thread, thread.frame.environment, ((Closure const&)f), cc);
		return buildStackFrame(thread, fenv, ((Closure const&)f).prototype()->code, out, &inst+1);
	}
	_error(std::string("Failed to find stop handler (__stop__)"));
}


template<>
bool EnvironmentBinaryDispatch< struct eqVOp<REnvironment, REnvironment> >
(Thread& thread, void* args, Value const& a, Value const& b, Value& c) {
    Logical::InitScalar(c,
        ((REnvironment const&)a).environment() == ((REnvironment const&)b).environment() ?
            Logical::TrueElement : Logical::FalseElement );
    return true;
}

template<>
bool EnvironmentBinaryDispatch< struct neqVOp<REnvironment, REnvironment> >
(Thread& thread, void* args, Value const& a, Value const& b, Value& c) {
    Logical::InitScalar(c,
        ((REnvironment const&)a).environment() != ((REnvironment const&)b).environment() ?
            Logical::TrueElement : Logical::FalseElement );
    return true;
}

void IfElseDispatch(Thread& thread, void* args, Value const& a, Value const& b, Value const& cond, Value& c) {
	if(a.isVector() && b.isVector())
	{
        if(a.isCharacter() || b.isCharacter())
		    Zip3< IfElseVOp<Character> >::eval(thread, args, As<Character>(thread, a), As<Character>(thread, b), As<Logical>(thread, cond), c);
	    else if(a.isDouble() || b.isDouble())
		    Zip3< IfElseVOp<Double> >::eval(thread, args, As<Double>(thread, a), As<Double>(thread, b), As<Logical>(thread, cond), c);
	    else if(a.isInteger() || b.isInteger())	
		    Zip3< IfElseVOp<Integer> >::eval(thread, args, As<Integer>(thread, a), As<Integer>(thread, b), As<Logical>(thread, cond), c);
	    else if(a.isLogical() || b.isLogical())	
		    Zip3< IfElseVOp<Logical> >::eval(thread, args, As<Logical>(thread, a), As<Logical>(thread, b), As<Logical>(thread, cond), c);
	    else if(a.isNull() || b.isNull() || cond.isNull())
		    c = Null::Singleton();
    }
    else {
	    _error("non-zippable argument to ifelse operator");
    }
}

template< template<class X> class Group, IROpCode::Enum Op>
bool RecordUnary(Thread& thread, Value const& a, Value& c) {
    // If we can record the instruction, we can delay execution
    if(thread.traces.isTraceable<Group>(a)) {
        c = thread.traces.EmitUnary<Group>(thread.frame.environment, Op, a, 0);
        thread.traces.OptBind(thread, c);
        return true;
    }
    // If we couldn't delay, and the argument is a future, then we need to evaluate it.
    if(a.isFuture()) {
        thread.traces.Bind(thread, a);
    }
    return false;
}

template< template<class X, class Y> class Group, IROpCode::Enum Op>
bool RecordBinary(Thread& thread, Value const& a, Value const& b, Value& c) {
    // If we can record the instruction, we can delay execution
    if(thread.traces.isTraceable<Group>(a, b)) {
        c = thread.traces.EmitBinary<Group>(thread.frame.environment, Op, a, b, 0);
        thread.traces.OptBind(thread, c);
        return true;
    }
    // If we couldn't delay, and the arguments are futures, then we need to evaluate them.
    if(a.isFuture()) {
        thread.traces.Bind(thread, a);
    }
    if(b.isFuture()) {
        thread.traces.Bind(thread, b);
    }
    return false;
}

#define SLOW_DISPATCH_DEFN(Name, String, Group, Func) \
Instruction const* Name##Slow(Thread& thread, Instruction const& inst, void* args, Value const& a, Value& c) { \
    if(RecordUnary<Group, IROpCode::Name>(thread, a, c)) \
        return &inst+1; \
    else if(!((Object const&)a).hasAttributes() \
            && Group##Dispatch<Name##VOp>(thread, args, a, c)) \
        return &inst+1; \
    else \
        return GenericDispatch(thread, inst, Strings::Name, a, inst.c); \
}
UNARY_FOLD_SCAN_BYTECODES(SLOW_DISPATCH_DEFN)
#undef SLOW_DISPATCH_DEFN

#define SLOW_DISPATCH_DEFN(Name, String, Group, Func) \
Instruction const* Name##Slow(Thread& thread, Instruction const& inst, void* args, Value const& a, Value const& b, Value& c) { \
    if(RecordBinary<Group, IROpCode::Name>(thread, a, b, c)) \
        return &inst+1; \
    else if(   !((Object const&)a).hasAttributes() \
            && !((Object const&)b).hasAttributes() \
            && Group##Dispatch<Name##VOp>(thread, args, a, b, c)) \
        return &inst+1; \
    else \
        return GenericDispatch(thread, inst, Strings::Name, a, b, inst.c); \
}
BINARY_BYTECODES(SLOW_DISPATCH_DEFN)
#undef SLOW_DISPATCH_DEFN

Instruction const* GetSlow(Thread& thread, Instruction const& inst, Value const& a, Value const& b, Value& c) {
    BIND(a); BIND(b);
    
    if(!((Object const&)a).hasAttributes()) {
	    if(a.isVector()) {
            Vector const& v = (Vector const&)a;
		    if(b.isInteger()) {
                if(  ((Integer const&)b).length() != 1
                  || (b.i-1) < 0 )
                    _error("attempt to select more or less than one element");
                if( (b.i-1) >= v.length() )
                    _error("subscript out of bounds");

                Element2(v, b.i-1, c);
                return &inst+1;
            }
		    else if(b.isDouble()) {
                if(  ((Double const&)b).length() != 1
                  || ((int64_t)b.d-1) < 0 )
                    _error("attempt to select more or less than one element");
                if( ((int64_t)b.d-1) >= v.length())
                    _error("subscript out of bounds");
                
                Element2(a, (int64_t)b.d-1, c);
                return &inst+1;
            }
	    }
        else if(a.isEnvironment()) {
            if( b.isCharacter()
                && ((Character const&)b).length() == 1) {
	            String s = ((Character const&)b).s;
                Value const& v = ((REnvironment&)a).environment()->get(s);
                if(v.isObject()) {
                    c = v;
                    return &inst+1;
                }
                else if(v.isNil()) {
                    c = Null::Singleton();
                    return &inst+1;
                }
                else {
                    return force(thread, inst, v, 
                        ((REnvironment&)a).environment(), s); 
                }
            }
        }
        else if(a.isClosure()) {
            if( b.isCharacter()
                && ((Character const&)b).length() == 1 ) {
                Closure const& f = (Closure const&)a;
	            String s = ((Character const&)b).s;
                if(s == Strings::body) {
                    c = f.prototype()->code->expression;
                    return &inst+1;
                }
                else if(s == Strings::formals) {
                    c = f.prototype()->formals;
                    return &inst+1;
                }
            }
        }
    }
 
    return GenericDispatch(thread, inst, Strings::bb, a, b, inst.c); 
}

