
#include "call.h"
#include "frontend.h"
#include "compiler.h"
#include "inst.h"

void dumpStack(State& state) {
   
    for(int64_t i = state.stack.size()-1; i >= 0; --i) {
        StackFrame const& s = state.stack[i];

        std::cout << i << ": ";
        if(s.isPromise) {
            std::cout << "Forcing " << state.deparse(s.registers[1]) << std::endl;
        }
        else {
            Value const* v = s.environment->get(Strings::__call__);
            if(v)
                std::cout << state.deparse(*v) << std::endl;
            else
                std::cout << "<no call in environment>" << std::endl;
        }
    }

    /*std::cout << "\t(Executing in " << 
        "0x" << intToHexStr((int64_t)state.frame.environment) << ")" << std::endl;
	
    state.frame.code->printByteCode(state.global);
    
    std::cout << "Returning to: " << 
        "0x" << intToHexStr((int64_t)state.frame.returnpc) << std::endl;
    */
}

Instruction const* buildStackFrame(State& state, 
    Environment* environment, Code const* code, 
    int64_t outRegister, Instruction const* returnpc) {

	// make new stack frame
	StackFrame& s = state.push();
	s.environment = environment;
	s.code = code;
	s.returnpc = returnpc;
	s.registers += outRegister;
    s.isPromise = false;
	
	if(s.registers+code->registers > state.registers+DEFAULT_NUM_REGISTERS) {
        dumpStack(state);
		_internalError("Register overflow");
    }

    // avoid confusing the GC with bogus stuff in registers...
    // can we avoid this somehow?
    for(int64_t i = 0; i < code->registers; ++i) {
        s.registers[i] = Value::Nil();
    }

	return (Instruction const*)&(code->bc[0]);
}

inline void assignArgument(State& state, Environment* evalEnv, Environment* assignEnv, String n, Value const& v) {
	assert(!v.isFuture());
	Value& w = assignEnv->insert(n);
	w = v;
	if(v.isPromise()) {
        ((Promise&)w).environment(evalEnv);
	}
}

inline void assignDot(State& state, Value const& v, Environment* evalEnv, Value& out) {
	out = v;

	if(v.isPromise()) {
        ((Promise&)out).environment(evalEnv);
	}
	assert(!v.isFuture());
}

Value argument(int64_t index, List const* dots, CompiledCall const& call, Environment* env) {
	if(index < call.dotIndex()) {
		return call.arguments()[index];
	} else {
		index -= call.dotIndex();
        int64_t ndots = (dots && dots->isList()) ? dots->length() : 0;
		if(index < ndots) {
			// Promises in the dots can't be passed down 
			//     (general rule is that promises only
			//	occur once anywhere in the program). 
			// But everything else can be passed down.
			if((*dots)[index].isPromise()) {
				Value p;
				Promise::Init(p, env, index, false);
				return p;
			} 
			else {
				return (*dots)[index];
			}
		}
		else {
			index -= ndots;
			return call.arguments()[call.dotIndex()+index+1];
		}
	}
}

String name(int64_t index, List const* dots, 
                Character const* dotnames, CompiledCall const& call) {
	if(index < call.dotIndex()) {
		return index < call.names().length() 
            ? call.names()[index] 
            : Strings::empty;
	} else {
		index -= call.dotIndex();
        int64_t ndots = (dots && dots->isList()) ? dots->length() : 0;
		if(index < ndots) {
		    return (dotnames && dotnames->isCharacter()) ? (*dotnames)[index] : Strings::empty;
		}
		else {
			index -= ndots;
			return call.dotIndex()+index+1 < call.names().length()
                ? call.names()[call.dotIndex()+index+1]
                : Strings::empty;
		}
	}
}

int64_t numArguments(List const* dots, CompiledCall const& call) {
	if(call.dotIndex() < (int64_t)call.arguments().length()) {
		// subtract 1 to not count the dots
        int64_t ndots = (dots && dots->isList()) ? dots->length() : 0;
		return call.arguments().length() - 1 + ndots;
	} else {
		return call.arguments().length();
	}
}

bool namedArguments(Character const* dotnames, CompiledCall const& call) {
	if(call.dotIndex() < (int64_t)call.arguments().length()) {
        bool dotsNamed = (dotnames && dotnames->isCharacter()) ? dotnames->length() : false;
		return call.names().length()>0 || dotsNamed;
	} else {
		return call.names().length()>0;
	}
}


// Generic argument matching
Environment* MatchArgs(State& state, Environment* env, Closure const& func, CompiledCall const& call) {
    Character const& parameters = func.prototype()->parameters;
    List const& defaults = func.prototype()->defaults;
	int64_t pDotIndex = func.prototype()->dotIndex;

    List const* dots = (List const*)env->get(Strings::__dots__);
    Character const* dotnames = (Character const*)env->get(Strings::__names__);

	int64_t numArgs = numArguments(dots, call);
	bool named = namedArguments(dotnames, call);

    Environment* fenv = new Environment(
        (int64_t)std::min(numArgs, (int64_t)parameters.length()) + 5,
        func.environment());

    // set extra args (they must be named)
    for(size_t i = 0; i < call.extraArgs().length(); ++i) {
        assignArgument(state, env, fenv, call.extraNames()[i], call.extraArgs()[i]);
    }

	// set defaults
	for(int64_t i = 0; i < (int64_t)parameters.length(); ++i) {
		assignArgument(state, fenv, fenv, parameters[i], defaults[i]);
    }

	if(!named) {
		// call arguments are not named, 
        // do posititional matching up to the prototype's dots
		int64_t end = std::min(numArgs, pDotIndex);
		for(int64_t i = 0; i < end; ++i) {
			Value arg = argument(i, dots, call, env);
			if(!arg.isNil()) {
				assignArgument(state, env, fenv, parameters[i], arg);
            }
		}

        // if we have left over arguments:
        if(numArgs > end) {
            // if we have a ... stick them there
            if(pDotIndex < (int64_t)parameters.length()) {
                List newdots(numArgs-end);
                for(int64_t i = end; i < numArgs; i++) {
	                Value arg = argument(i, dots, call, env);
                    assignDot(state, arg, env, newdots[i-end]);
                }
                fenv->insert(Strings::__dots__) = newdots;
            }
            else {
                _error(std::string("Unused args in call: ")
                        + state.global.deparse(call.call()));
            }
        }
    }
    // function only has dots, can just stick everything there
    else if(parameters.length() == 1 && pDotIndex == 0) {
        if(numArgs > 0) {
            List newdots(numArgs);
            Character names(numArgs);
		    for(int64_t i = 0; i < numArgs; i++) {
                Value arg = argument(i, dots, call, env);
                String n = name(i, dots, dotnames, call);
                assignDot(state, arg, env, newdots[i]);
                names[i] = n;
            }
            fenv->insert(Strings::__dots__) = newdots;
            fenv->insert(Strings::__names__) = names;
        }
    }
	else {
		// call arguments are named, do matching by name
		// we should be able to cache and reuse this assignment 
        // for pairs of functions and call sites.

        if(numArgs > 256) {
            _error("Too many arguments for fixed size assignment arrays");
        }
	
		int64_t *assignment = state.assignment, *set = state.set;
		for(int64_t i = 0; i < numArgs; i++)
            assignment[i] = -1;
		for(int64_t i = 0; i < (int64_t)parameters.length(); i++)
            set[i] = -(i+1);

		// named args, search for complete matches
		for(int64_t i = 0; i < numArgs; ++i) {
			String n = name(i, dots, dotnames, call);
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
			String n = name(i, dots, dotnames, call);
			if(n != Strings::empty && assignment[i] < 0) {
				for(int64_t j = 0; j < pDotIndex; ++j) {
					if(set[j] < 0 && 
                        strncmp(n->s, parameters[j]->s, strlen(n->s)) == 0) {
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
			String n = name(i, dots, dotnames, call);
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
        int64_t numDots = numArgs;

		// assign all the arguments
		for(int64_t j = 0; j < (int64_t)parameters.length(); ++j) {
			if(j != pDotIndex && set[j] >= 0) {
				Value arg = argument(set[j], dots, call, env);
				if(!arg.isNil()) {
					assignArgument(state, env, fenv, parameters[j], arg);
			    }
                numDots--;
            }
		}

		// put unused args into the dots
        if( numDots > 0 ) {
            if(pDotIndex < (int64_t)parameters.length()) {
                named = false;
                List newdots(numDots);
                Character names(numDots);
                int64_t j = 0;
		        for(int64_t i = 0; i < numArgs; i++) {
                    if(assignment[i] < 0) {
                        Value arg = argument(i, dots, call, env);
                        String n = name(i, dots, dotnames, call);
                        if(n != Strings::empty)
                            named = true;
                        assignDot(state, arg, env, newdots[j]);
                        names[j] = n;
                        j++;
			        }
		        }

                fenv->insert(Strings::__dots__) = newdots;
                if(named)
                    fenv->insert(Strings::__names__) = names;
            }
            else {
                _error(std::string("Unused args in call: ")
                        + state.global.deparse(call.call()));
	        }
        }
    }

    REnvironment::Init(fenv->insert(Strings::__parent__), env);
    fenv->insert(Strings::__call__) = call.call();
    fenv->insert(Strings::__function__) = func;
    fenv->insert(Strings::__nargs__) = Integer::c(numArgs);

    return fenv;
}

// Assumes no names and no ... in the argument list.
// Supports ... in the parameter list.
Environment* FastMatchArgs(State& state, Environment* env, Closure const& func, CompiledCall const& call) {
    Prototype const* prototype = func.prototype();
	Character const& parameters = prototype->parameters;
    List const& defaults = prototype->defaults;
	List const& arguments = call.arguments();

	int64_t const pDotIndex = prototype->dotIndex;
	int64_t const end = std::min(arguments.length(), pDotIndex);

    Environment* fenv = new Environment(
        (int64_t)call.arguments().length() + 5,
        func.environment());

    // set extra args (they must be named)
    for(size_t i = 0; i < call.extraArgs().length(); ++i) {
        assignArgument(state, env, fenv, 
            call.extraNames()[i], call.extraArgs()[i]);
    }

	// set parameters from arguments & defaults
	for(int64_t i = 0; i < parameters.length(); i++) {
		if(i < end && !arguments[i].isNil()) {
			assignArgument(state, env, fenv, parameters[i], arguments[i]);
        }
		else {
			assignArgument(state, fenv, fenv, parameters[i], defaults[i]);
        }
	}

	// handle unused arguments
    if( arguments.length() > end ) {
        // if we have a ..., put them there
	    if(pDotIndex < parameters.length()) {
            List dots(arguments.length()-end);
            for(int64_t i = end; i < (int64_t)arguments.length(); i++) {
                assignDot(state, arguments[i], env, dots[i-end]);
            }
            fenv->insert(Strings::__dots__) = dots;
        }
        else if(arguments.length() > end) {
            _error(std::string("Unused args in call: ")
                    + state.global.deparse(call.call()));
	    }
    }

    REnvironment::Init(fenv->insert(Strings::__parent__), env);
    fenv->insert(Strings::__call__) = call.call();
    fenv->insert(Strings::__function__) = func;
    fenv->insert(Strings::__nargs__) = Integer::c(arguments.length());
    
    return fenv;
}

Value Quote(State& state, Value const& v) {
    if(isSymbol(v) || isCall(v) || isExpression(v)) {
        List call(2);
        call[0] = CreateSymbol(state.global, Strings::quote);
        call[1] = v;
        return CreateCall(state.global, call);
    } else {
        return v;
    }
}

Instruction const* GenericDispatch(State& state, Instruction const& inst, String op, Value const& a, int64_t out) {
	Environment* penv;
    op = state.global.strings.intern(op->s);
	Value const* f = state.frame.environment->getRecursive(op, penv);
	if(f && f->isClosure()) {
		List call = List::c(
                        CreateSymbol(state.global, op),
                        Quote(state, a));
        CompiledCall cc = Compiler::makeCall(state, call, Character(0));
		Environment* fenv = FastMatchArgs(state, state.frame.environment, static_cast<Closure const&>(*f), cc);
		return buildStackFrame(state, fenv, static_cast<Closure const&>(*f).prototype()->code, out, &inst+1);
	}
	_error(std::string("Failed to find generic for builtin op: ") + op->s);
}

Instruction const* GenericDispatch(State& state, Instruction const& inst, String op, Value const& a, Value const& b, int64_t out) {
	Environment* penv;
    op = state.global.strings.intern(op->s);
	Value const* f = state.frame.environment->getRecursive(op, penv);
	if(f && f->isClosure()) { 
		List call = List::c(
                        CreateSymbol(state.global, op),
                        Quote(state, a),
                        Quote(state, b));
        CompiledCall cc = Compiler::makeCall(state, call, Character(0));
		Environment* fenv = FastMatchArgs(state, state.frame.environment, static_cast<Closure const&>(*f), cc);
		return buildStackFrame(state, fenv, static_cast<Closure const&>(*f).prototype()->code, out, &inst+1);
	}
	_error(std::string("Failed to find generic for builtin op: ") + op->s + " type: " + Type::toString(a.type()) + " " + Type::toString(b.type()));
}

Instruction const* GenericDispatch(State& state, Instruction const& inst, String op, Value const& a, Value const& b, Value const& c, int64_t out) {
	Environment* penv;
    op = state.global.strings.intern(op->s);
	Value const* f = state.frame.environment->getRecursive(op, penv);
	if(f && f->isClosure()) { 
		List call = List::c(
                        CreateSymbol(state.global, op),
                        Quote(state, a),
                        Quote(state, b),
                        Quote(state, c));
        Character names = Character::c(
                        Strings::empty,
                        Strings::empty,
                        Strings::empty,
                        Strings::value);
        CompiledCall cc = Compiler::makeCall(state, call, names);
        Environment* fenv = MatchArgs(state, state.frame.environment, static_cast<Closure const&>(*f), cc);
		return buildStackFrame(state, fenv, static_cast<Closure const&>(*f).prototype()->code, out, &inst+1);
	}
	_error(std::string("Failed to find generic for builtin op: ") + op->s);
}


Instruction const* StopDispatch(State& state, Instruction const& inst, String msg, int64_t out) {
	Environment* penv;
	Value const* f = state.frame.environment->getRecursive(Strings::__stop__, penv);
	if(f && f->isClosure()) {
		List call = List::c(
                        *f, 
                        Character::c(msg));
        CompiledCall cc = Compiler::makeCall(state, call, Character(0));
		Environment* fenv = FastMatchArgs(state, state.frame.environment, static_cast<Closure const&>(*f), cc);
		return buildStackFrame(state, fenv, static_cast<Closure const&>(*f).prototype()->code, out, &inst+1);
	}
	_error(std::string("Failed to find stop handler (__stop__)"));
}


template<>
bool EnvironmentBinaryDispatch< struct eqVOp<REnvironment, REnvironment> >
(State& state, void* args, Value const& a, Value const& b, Value& c) {
    Logical::InitScalar(c,
        ((REnvironment const&)a).environment() == ((REnvironment const&)b).environment() ?
            Logical::TrueElement : Logical::FalseElement );
    return true;
}

template<>
bool EnvironmentBinaryDispatch< struct neqVOp<REnvironment, REnvironment> >
(State& state, void* args, Value const& a, Value const& b, Value& c) {
    Logical::InitScalar(c,
        ((REnvironment const&)a).environment() != ((REnvironment const&)b).environment() ?
            Logical::TrueElement : Logical::FalseElement );
    return true;
}

template<>
bool ClosureBinaryDispatch< struct eqVOp<Closure, Closure> >
(State& state, void* args, Value const& a, Value const& b, Value& c) {
    Logical::InitScalar(c,
        static_cast<Closure const&>(a).environment() == static_cast<Closure const&>(b).environment() &&
        static_cast<Closure const&>(a).prototype() == static_cast<Closure const&>(b).prototype() ?
            Logical::TrueElement : Logical::FalseElement );
    return true;
}

template<>
bool ClosureBinaryDispatch< struct neqVOp<Closure, Closure> >
(State& state, void* args, Value const& a, Value const& b, Value& c) {
    Logical::InitScalar(c,
        static_cast<Closure const&>(a).environment() != static_cast<Closure const&>(b).environment() ||
        static_cast<Closure const&>(a).prototype() != static_cast<Closure const&>(b).prototype() ?
            Logical::TrueElement : Logical::FalseElement );
    return true;
}

void IfElseDispatch(State& state, void* args, Value const& a, Value const& b, Value const& cond, Value& c) {
	if(a.isVector() && b.isVector())
	{
        if(a.isCharacter() || b.isCharacter())
		    Zip3< IfElseVOp<Character> >::eval(args, As<Character>(a), As<Character>(b), As<Logical>(cond), c);
	    else if(a.isDouble() || b.isDouble())
		    Zip3< IfElseVOp<Double> >::eval(args, As<Double>(a), As<Double>(b), As<Logical>(cond), c);
	    else if(a.isInteger() || b.isInteger())	
		    Zip3< IfElseVOp<Integer> >::eval(args, As<Integer>(a), As<Integer>(b), As<Logical>(cond), c);
	    else if(a.isLogical() || b.isLogical())	
		    Zip3< IfElseVOp<Logical> >::eval(args, As<Logical>(a), As<Logical>(b), As<Logical>(cond), c);
	    else if(a.isNull() || b.isNull() || cond.isNull())
		    c = Null::Singleton();
    }
    else {
	    _error("non-zippable argument to ifelse operator");
    }
}

#ifdef EPEE
template< template<class X> class Group, IROpCode::Enum Op>
bool RecordUnary(State& state, Value const& a, Value& c) {
    // If we can record the instruction, we can delay execution
    if(state.traces.isTraceable<Group>(a)) {
        c = state.traces.EmitUnary<Group>(state.frame.environment, Op, a, 0);
        state.traces.OptBind(state, c);
        return true;
    }
    // If we couldn't delay, and the argument is a future, then we need to evaluate it.
    if(a.isFuture()) {
        state.traces.Bind(state, a);
    }
    return false;
}

template< template<class X, class Y> class Group, IROpCode::Enum Op>
bool RecordBinary(State& state, Value const& a, Value const& b, Value& c) {
    // If we can record the instruction, we can delay execution
    if(state.traces.isTraceable<Group>(a, b)) {
        c = state.traces.EmitBinary<Group>(state.frame.environment, Op, a, b, 0);
        state.traces.OptBind(state, c);
        return true;
    }
    // If we couldn't delay, and the arguments are futures, then we need to evaluate them.
    if(a.isFuture()) {
        state.traces.Bind(state, a);
    }
    if(b.isFuture()) {
        state.traces.Bind(state, b);
    }
    return false;
}
#endif

