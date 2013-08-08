
#include "call.h"
#include "frontend.h"
#include "compiler.h"

Instruction const* force(
    Thread& thread, Promise const& p,
    Environment* targetEnv, Value targetIndex,
    int64_t outRegister, Instruction const* returnpc) {

    Code* code = p.isExpression() ? p.code() : thread.promiseCode;
    Instruction const* r = buildStackFrame(
        thread, p.environment(), code, outRegister, returnpc);
    thread.frame.isPromise = true;
            
    REnvironment::Init(REGISTER(0), targetEnv);
    REGISTER(1) = targetIndex;
	
    if(p.isDotdot())		
        Integer::InitScalar(REGISTER(2), p.dotIndex());

    return r;
}

void dumpStackFrame(Thread& thread) {
    std::cout << "\t(Executing in " << 
        intToHexStr((int64_t)thread.frame.environment) << ")" << std::endl;
	
    thread.frame.code->printByteCode(thread.state);
    
    if( thread.frame.environment->getContext() && 
        ((List const &)thread.frame.environment->getContext()->call).length() > 0) {
        std::cout << "Call:   " << 
            (((List const&)thread.frame.environment->getContext()->call)[0]).s << std::endl;
    }

    std::cout << "Returning to: " << 
        intToHexStr((int64_t)thread.frame.returnpc) << std::endl;
}

Instruction const* buildStackFrame(Thread& thread, 
    Environment* environment, Code const* code, 
    int64_t outRegister, Instruction const* returnpc) {

	// make new stack frame
	StackFrame& s = thread.push();
	s.environment = environment;
	s.code = code;
	s.returnpc = returnpc;
	s.registers += outRegister;
    s.isPromise = false;
	
    //dumpStackFrame(thread);
	
	if(s.registers+code->registers > thread.registers+DEFAULT_NUM_REGISTERS)
		_internalError("Register overflow");

    // avoid confusing the GC with bogus stuff in registers...
    // can we avoid this somehow?
    for(int64_t i = 0; i < code->registers; ++i) {
        s.registers[i] = Value::Nil();
    }

	return &(code->bc[0]);
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


Value argument(int64_t index, Environment* env, CompiledCall const& call) {
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
				Value p;
				Promise::Init(p, env, index);
				return p;
			} 
			else {
				return env->getContext()->dots[index].v;
			}
		}
		else {
			index -= ndots;
			return call.arguments[call.dotIndex+index+1];
		}
	}
}

String name(int64_t index, Environment* env, CompiledCall const& call) {
	if(index < call.dotIndex) {
		return call.names[index];
	} else {
		index -= call.dotIndex;
        int64_t ndots = env->getContext() ? env->getContext()->dots.size() : 0;
		if(index < ndots) {
		    return env->getContext()->dots[index].n;
		}
		else {
			index -= ndots;
			return call.names[call.dotIndex+index+1];
		}
	}
}

int64_t numArguments(Environment* env, CompiledCall const& call) {
	if(call.dotIndex < (int64_t)call.arguments.length() && env->getContext()) {
		// subtract 1 to not count the dots
		return call.arguments.length() - 1 + env->getContext()->dots.size();
	} else {
		return call.arguments.length();
	}
}

bool namedArguments(Environment* env, CompiledCall const& call) {
	if(call.dotIndex < (int64_t)call.arguments.length() && env->getContext()) {
		return call.names.length()>0 || env->getContext()->named;
	} else {
		return call.names.length()>0;
	}
}


// Generic argument matching
Environment* MatchArgs(Thread& thread, Environment* env, Closure const& func, CompiledCall const& call) {
    Character const& parameters = func.prototype()->parameters;
    List const& defaults = func.prototype()->defaults;
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
    context->missing = Logical(parameters.length());

    Environment* fenv = new Environment(
        (int64_t)std::min(numArgs, (int64_t)parameters.length()),
        func.environment(),
        context);

    // set extra args (they must be named)
    for(size_t i = 0; i < call.extraArgs.length(); ++i) {
        assignArgument(thread, env, fenv, 
            call.extraNames[i], call.extraArgs[i]);
    }

	// set defaults
	for(int64_t i = 0; i < (int64_t)parameters.length(); ++i) {
		assignArgument(thread, fenv, fenv, parameters[i], defaults[i]);
	    context->missing[i] = Logical::TrueElement;
    }

	if(!named) {
		// call arguments are not named, 
        // do posititional matching up to the prototype's dots
		int64_t end = std::min(numArgs, pDotIndex);
		for(int64_t i = 0; i < end; ++i) {
			Value arg = argument(i, env, call);
			if(!arg.isNil()) {
				assignArgument(thread, env, fenv, parameters[i], arg);
	            context->missing[i] = Logical::FalseElement;
            }
		}

        if(numArgs > end) {
            if(pDotIndex >= (int64_t)parameters.length()) {
		        // if we have left over arguments, but no parameter dots, error
                _error(std::string("Unused args in call: ") + thread.state.deparse(call.call));
		    }
            // all unused args go into ...
            context->missing[pDotIndex] = Logical::FalseElement;
            for(int64_t i = end; i < numArgs; i++) {
	            Value arg = argument(i, env, call);
                assignDot(thread, env, fenv, Strings::empty, arg);
            }
        }
    }
    // function only has dots, can just stick everything there
    else if(parameters.length() == 1 && pDotIndex == 0) {
        context->missing[pDotIndex] = Logical::FalseElement;
		for(int64_t i = 0; i < numArgs; i++) {
            Value arg = argument(i, env, call);
            String n = name(i, env, call);
            assignDot(thread, env, fenv, n, arg);
        }
    }
	else {
		// call arguments are named, do matching by name
		// we should be able to cache and reuse this assignment 
        // for pairs of functions and call sites.
	
		int64_t *assignment = thread.assignment, *set = thread.set;
		for(int64_t i = 0; i < numArgs; i++)
            assignment[i] = -1;
		for(int64_t i = 0; i < (int64_t)parameters.length(); i++)
            set[i] = -(i+1);

		// named args, search for complete matches
		for(int64_t i = 0; i < numArgs; ++i) {
			String n = name(i, env, call);
			if(n != Strings::empty) {
				for(int64_t j = 0; j < (int64_t)parameters.length(); ++j) {
					if(j != pDotIndex && n == parameters[j]) {
						assignment[i] = j;
						set[j] = i;
						break;
					}
				}
			}
		}
		// named args, search for incomplete matches but only to the ...
		for(int64_t i = 0; i < numArgs; ++i) {
			String n = name(i, env, call);
			if(n != Strings::empty && assignment[i] < 0) {
				for(int64_t j = 0; j < pDotIndex; ++j) {
					if(set[j] < 0 && 
                        strncmp(n, parameters[j], strlen(n)) == 0) {
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
			String n = name(i, env, call);
			if(n == Strings::empty) {
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
		for(int64_t j = 0; j < (int64_t)parameters.length(); ++j) {
			if(j != pDotIndex && set[j] >= 0) {
				Value arg = argument(set[j], env, call);
				if(!arg.isNil()) {
					assignArgument(thread, env, fenv, parameters[j], arg);
                    context->missing[j] = Logical::FalseElement;
			    }
            }
		}

		// put unused args into the dots
		context->named = false;
		for(int64_t i = 0; i < numArgs; i++) {
			if(assignment[i] < 0) {
				// if we have left over arguments, but no parameter dots, error
				if(pDotIndex >= (int64_t)parameters.length())
			        _error(std::string("Unused args in call: ") + thread.state.deparse(call.call));
                context->missing[pDotIndex] = Logical::FalseElement;
				Value arg = argument(i, env, call);
                String n = name(i, env, call);
				if(n != Strings::empty)
                    context->named = true;
				assignDot(thread, env, fenv, n, arg);
			}
		}
	}
    return fenv;
}

// Assumes no names and no ... in the argument list.
// Supports ... in the parameter list.
Environment* FastMatchArgs(Thread& thread, Environment* env, Closure const& func, CompiledCall const& call) {
    Prototype const* prototype = func.prototype();
	Character const& parameters = prototype->parameters;
    List const& defaults = prototype->defaults;
	List const& arguments = call.arguments;

	int64_t const pDotIndex = prototype->dotIndex;
	int64_t const end = std::min(arguments.length(), pDotIndex);

    Context* context = new Context();
    context->parent = env;
    context->call = call.call;
    context->function = func;
    context->nargs = arguments.length();
    context->named = false; 
    context->onexit = Null::Singleton();
    context->missing = Logical(parameters.length());

    Environment* fenv = new Environment(
        (int64_t)call.arguments.length(),
        func.environment(),
        context);

    // set extra args (they must be named)
    for(size_t i = 0; i < call.extraArgs.length(); ++i) {
        assignArgument(thread, env, fenv, 
            call.extraNames[i], call.extraArgs[i]);
    }

	// set parameters from arguments & defaults
	for(int64_t i = 0; i < parameters.length(); i++) {
		if(i < end && !arguments[i].isNil()) {
			assignArgument(thread, env, fenv, parameters[i], arguments[i]);
            context->missing[i] = Logical::FalseElement;
        }
		else {
			assignArgument(thread, fenv, fenv, parameters[i], defaults[i]);
            context->missing[i] = Logical::TrueElement;
        }
	}

	// handle unused arguments
    if(arguments.length() > end) {
	    if(pDotIndex >= parameters.length()) {
		    // called function doesn't take dots, unused args is an error 
			_error(std::string("Unused args in call: ") + thread.state.deparse(call.call));
	    }
        // called function has dots, all unused args go into ...
        context->missing[pDotIndex] = Logical::FalseElement;
        context->dots.reserve(arguments.length()-end);
        for(int64_t i = end; i < (int64_t)arguments.length(); i++) {
            assignDot(thread, env, fenv, Strings::empty, arguments[i]);
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
		List call = List::c(
                        CreateSymbol(op),
                        Quote(thread, a));
        CompiledCall cc = Compiler::makeCall(thread, call, Character(0));
		Environment* fenv = FastMatchArgs(thread, thread.frame.environment, ((Closure const&)f), cc);
		return buildStackFrame(thread, fenv, ((Closure const&)f).prototype()->code, out, &inst+1);
	}
	_error(std::string("Failed to find generic for builtin op: ") + op);
}

Instruction const* GenericDispatch(Thread& thread, Instruction const& inst, String op, Value const& a, Value const& b, int64_t out) {
	Environment* penv;
	Value const& f = thread.frame.environment->getRecursive(op, penv);
	if(f.isClosure()) { 
		List call = List::c(
                        CreateSymbol(op),
                        Quote(thread, a),
                        Quote(thread, b));
        CompiledCall cc = Compiler::makeCall(thread, call, Character(0));
		Environment* fenv = FastMatchArgs(thread, thread.frame.environment, ((Closure const&)f), cc);
		return buildStackFrame(thread, fenv, ((Closure const&)f).prototype()->code, out, &inst+1);
	}
	_error(std::string("Failed to find generic for builtin op: ") + op + " type: " + Type::toString(a.type()) + " " + Type::toString(b.type()));
}

Instruction const* GenericDispatch(Thread& thread, Instruction const& inst, String op, Value const& a, Value const& b, Value const& c, int64_t out) {
	Environment* penv;
	Value const& f = thread.frame.environment->getRecursive(op, penv);
	if(f.isClosure()) { 
		List call = List::c(
                        CreateSymbol(op),
                        Quote(thread, a),
                        Quote(thread, b),
                        Quote(thread, c));
        Character names = Character::c(
                        Strings::empty,
                        Strings::empty,
                        Strings::empty,
                        Strings::value);
        CompiledCall cc = Compiler::makeCall(thread, call, names);
        Environment* fenv = MatchArgs(thread, thread.frame.environment, ((Closure const&)f), cc);
		return buildStackFrame(thread, fenv, ((Closure const&)f).prototype()->code, out, &inst+1);
	}
	_error(std::string("Failed to find generic for builtin op: ") + op);
}


Instruction const* StopDispatch(Thread& thread, Instruction const& inst, String msg, int64_t out) {
	Environment* penv;
	Value const& f = thread.frame.environment->getRecursive(thread.internStr("__stop__"), penv);
	if(f.isClosure()) {
		List call = List::c(
                        f, 
                        Character::c(msg));
        CompiledCall cc = Compiler::makeCall(thread, call, Character(0));
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
                    return force(thread, ((Promise const&)v), 
                        ((REnvironment&)a).environment(), b,
                        inst.c, &inst+1); 
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

