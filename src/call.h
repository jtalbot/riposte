
#ifndef RIPOSTE_CALL_H

// code for making function calls

#ifdef USE_THREADED_INTERPRETER
static const void** glabels = 0;
#endif

static void printCode(Thread const& thread, Prototype const* prototype) {
	std::string r = "block:\nconstants: " + intToStr(prototype->constants.size()) + "\n";
	for(int64_t i = 0; i < (int64_t)prototype->constants.size(); i++)
		r = r + intToStr(i) + "=\t" + thread.stringify(prototype->constants[i]) + "\n";

	r = r + "code: " + intToStr(prototype->bc.size()) + "\n";
	for(int64_t i = 0; i < (int64_t)prototype->bc.size(); i++)
		r = r + intToHexStr((uint64_t)&(prototype->bc[i])) + "--: " + intToStr(i) + ":\t" + prototype->bc[i].toString() + "\n";

	std::cout << r << std::endl;
}

static Instruction const* buildStackFrame(Thread& thread, Environment* environment, bool ownEnvironment, Prototype const* prototype, Instruction const* returnpc) {
	//printCode(thread, prototype);
	StackFrame& s = thread.push();
	s.environment = environment;
	s.ownEnvironment = ownEnvironment;
	s.returnpc = returnpc;
	s.returnbase = thread.base;
	s.prototype = prototype;
	thread.base -= prototype->registers;
	if(prototype->constants.size() > 0)
		memcpy(thread.base, &prototype->constants[0], sizeof(Value)*prototype->constants.size());
	if(thread.base < thread.registers)
		throw RiposteError("Register overflow");

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

static Instruction const* buildStackFrame(Thread& thread, Environment* environment, bool ownEnvironment, Prototype const* prototype, int64_t resultSlot, Instruction const* returnpc) {
	Environment* env = thread.frame.environment;
	Instruction const* i = buildStackFrame(thread, environment, ownEnvironment, prototype, returnpc);
	thread.frame.i = resultSlot;
	thread.frame.env = env;
	return i;
}

static Instruction const* buildStackFrame(Thread& thread, Environment* environment, bool ownEnvironment, Prototype const* prototype, Environment* env, String s, Instruction const* returnpc) {
	Instruction const* i = buildStackFrame(thread, environment, ownEnvironment, prototype, returnpc);
	thread.frame.env = env;
	thread.frame.s = s;
	return i;
}

static void ExpandDots(Thread& thread, List& arguments, Character& names, int64_t dots) {
	Environment* environment = thread.frame.environment;
	uint64_t dotslength = environment->dots.size();
	// Expand dots into the parameter list...
	if(dots < arguments.length) {
		List a(arguments.length + dotslength - 1);
		for(int64_t i = 0; i < dots; i++) a[i] = arguments[i];
		for(uint64_t i = dots; i < dots+dotslength; i++) { a[i] = Function(Compiler::compilePromise(thread.state, CreateSymbol((String)-(i-dots+1))), NULL).AsPromise(); } // TODO: should cache these.
		for(uint64_t i = dots+dotslength; i < arguments.length+dotslength-1; i++) a[i] = arguments[i-dotslength];

		arguments = a;
		
		uint64_t named = 0;
		for(uint64_t i = 0; i < dotslength; i++) if(environment->dots[i] != Strings::empty) named++;

		if(names.length > 0 || named > 0) {
			Character n(arguments.length + dotslength - 1);
			for(int64_t i = 0; i < n.length; i++) n[i] = Strings::empty;
			if(names.length > 0) {
				for(int64_t i = 0; i < dots; i++) n[i] = names[i];
				for(uint64_t i = dots+dotslength; i < arguments.length+dotslength-1; i++) n[i] = names[i-dotslength];
			}
			if(named > 0) {
				for(uint64_t i = dots; i < dots+dotslength; i++) n[i] = environment->dots[i]; 
			}
			names = n;
		}
	}
}

inline void argAssign(Environment* env, String n, Value const& v, Environment* execution) {
	Value w = v;
	if(!w.isNil()) {
		if(w.isPromise() || w.isDefault()) {
			assert(w.p == 0);
			w.p = execution;
		}
		env->insert(n) = w;
	}
}

static void MatchArgs(Thread& thread, Environment* env, Environment* fenv, Function const& func, List const& arguments, Character const& anames) {
	List const& defaults = func.prototype()->defaults;
	Character const& parameters = func.prototype()->parameters;
	int64_t fdots = func.prototype()->dots;

	// set defaults
	for(int64_t i = 0; i < defaults.length; ++i) {
		argAssign(fenv, parameters[i], defaults[i], fenv);
	}

	// call arguments are not named, do posititional matching
	if(anames.length == 0) {
		int64_t end = std::min(arguments.length, fdots);
		for(int64_t i = 0; i < end; ++i) {
			if(!arguments[i].isNil()) argAssign(fenv, parameters[i], arguments[i], fenv);
		}

		// set dots if necessary
		if(fdots < parameters.length) {
			int64_t idx = 1;
			for(int64_t i = fdots; i < arguments.length; i++) {
				argAssign(fenv, (String)-idx, arguments[i], fenv);
				fenv->dots.push_back(Strings::empty);
				idx++;
			}
			end++;
		}
	}
	// call arguments are named, do matching by name
	else {
		// we should be able to cache and reuse this assignment for pairs of functions and call sites.
		int64_t *assignment = thread.assignment, *set = thread.set;
		for(int64_t i = 0; i < arguments.length; i++) assignment[i] = -1;
		for(int64_t i = 0; i < parameters.length; i++) set[i] = -(i+1);

		// named args, search for complete matches
		for(int64_t i = 0; i < arguments.length; ++i) {
			if(anames[i] != Strings::empty) {
				for(int64_t j = 0; j < parameters.length; ++j) {
					if(j != fdots && anames[i] == parameters[j]) {
						assignment[i] = j;
						set[j] = i;
						break;
					}
				}
			}
		}
		// named args, search for incomplete matches
		for(int64_t i = 0; i < arguments.length; ++i) {
			if(anames[i] != Strings::empty && assignment[i] < 0) {
				std::string a = thread.externStr(anames[i]);
				for(int64_t j = 0; j < parameters.length; ++j) {
					if(set[j] < 0 && j != fdots &&
							thread.externStr(parameters[j]).compare( 0, a.size(), a ) == 0 ) {	
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
			if(anames[i] == Strings::empty) {
				for(; firstEmpty < fdots; ++firstEmpty) {
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
		for(int64_t j = 0; j < parameters.length; ++j) {
			if(j != fdots && set[j] >= 0 && !arguments[set[j]].isNil()) {
				argAssign(fenv, parameters[j], arguments[set[j]], fenv);
			}
		}

		// put unused args into the dots
		if(fdots < parameters.length) {
			int64_t idx = 1;
			for(int64_t i = 0; i < arguments.length; i++) {
				if(assignment[i] < 0) {
					argAssign(fenv, (String)-idx, arguments[i], fenv);
					fenv->dots.push_back(anames[i]);
					idx++;
				}
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

// Get a Value by Symbol from the current environment,
//  TODO: UseMethod also should search in some cached library locations.
static Value GenericGet(Thread& thread, String s) {
	Value const& value = thread.frame.environment->getRecursive(s);
	if(value.isPromise()) {
		//value = force(thread, value);
		//environment->assign(s, value);
		_error("UseMethod does not yet support evaluating promises");
	}
	return value;
}

static Value GenericSearch(Thread& thread, Character klass, String generic, String& method) {
		
	// first search for type specific method
	Value func = Value::Nil();
	for(int64_t i = 0; i < klass.length && func.isNil(); i++) {
		method = thread.internStr(thread.externStr(generic) + "." + thread.externStr(klass[i]));
		func = GenericGet(thread, method);	
	}

	// TODO: look for group generics

	// look for default if necessary
	if(func.isNil()) {
		method = thread.internStr(thread.externStr(generic) + ".default");
		func = GenericGet(thread, method);
	}

	return func;
}

static Instruction const* GenericDispatch(Thread& thread, Instruction const& inst, String func, Value const& a, int64_t out) {
	String method;
	Value f = GenericSearch(thread, klass(thread, a), func, method);
	if(f.isFunction()) {
		Function func(f);
		Environment* fenv = CreateEnvironment(thread, func.environment(), thread.frame.environment, Null::Singleton());
		MatchArgs(thread, thread.frame.environment, fenv, func, List::c(a), Character(0));
		return buildStackFrame(thread, fenv, true, func.prototype(), out, &inst+1);
	}
	_error("Failed to find generic for builtin function");
}

static Instruction const* GenericDispatch(Thread& thread, Instruction const& inst, String func, Value const& a, Value const& b, int64_t out) {
	String method;
	Value f = a.isObject() ?  
		GenericSearch(thread, klass(thread, a), func, method) :  
		GenericSearch(thread, klass(thread, b), func, method);
	// TODO: R checks if these match for some ops (not all)
	if(f.isFunction()) { 
		Function func(f);
		Environment* fenv = CreateEnvironment(thread, func.environment(), thread.frame.environment, Null::Singleton());
		MatchArgs(thread, thread.frame.environment, fenv, func, List::c(a, b), Character(0));
		return buildStackFrame(thread, fenv, true, func.prototype(), out, &inst+1);
	}
	_error("Failed to find generic for builtin function");
}


#endif
