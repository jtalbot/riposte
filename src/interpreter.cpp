#include <string>
#include <sstream>
#include <stdexcept>
#include <string>
#include <dlfcn.h>

#include "value.h"
#include "type.h"
#include "bc.h"
#include "ops.h"
#include "runtime.h"
#include "interpreter.h"
#include "compiler.h"
#include "sse.h"
#include "call.h"

State* globalState;

static inline Instruction const* mov_op(Thread& thread, Instruction const& inst) ALWAYS_INLINE;
static inline Instruction const* store_op(Thread& thread, Instruction const& inst) ALWAYS_INLINE;
static inline Instruction const* forend_op(Thread& thread, Instruction const& inst) ALWAYS_INLINE;
static inline Instruction const* add_op(Thread& thread, Instruction const& inst) ALWAYS_INLINE;
static inline Instruction const* get_op(Thread& thread, Instruction const& inst) ALWAYS_INLINE;
static inline Instruction const* getsub_op(Thread& thread, Instruction const& inst) ALWAYS_INLINE;
static inline Instruction const* jc_op(Thread& thread, Instruction const& inst) ALWAYS_INLINE;
static inline Instruction const* lt_op(Thread& thread, Instruction const& inst) ALWAYS_INLINE;
static inline Instruction const* ret_op(Thread& thread, Instruction const& inst) ALWAYS_INLINE;
static inline Instruction const* retp_op(Thread& thread, Instruction const& inst) ALWAYS_INLINE;
static inline Instruction const* strip_op(Thread& thread, Instruction const& inst) ALWAYS_INLINE;


// CONTROL_FLOW_BYTECODES 

static inline Instruction const* call_op(Thread& thread, Instruction const& inst) {
    thread.visible = true;
	Heap::Global.collect(thread.state);
	DECODE(a); BIND(a);
	if(!a.isClosure())
        return StopDispatch(thread, inst, thread.internStr(
		    (std::string("Non-function (") + Type::toString(a.type()) + ") as first parameter to call\n").c_str()),
            inst.c);
	
    Closure const& func = (Closure const&)a;
	CompiledCall const& call = thread.frame.code->calls[inst.b];

	Environment* fenv = MatchArgs(thread, thread.frame.environment, func, call);
    return buildStackFrame(thread, fenv, func.prototype()->code, inst.c, &inst+1);
}

static inline Instruction const* fastcall_op(Thread& thread, Instruction const& inst) {
    thread.visible = true;
	Heap::Global.collect(thread.state);
	DECODE(a); BIND(a);
	if(!a.isClosure())
        return StopDispatch(thread, inst, thread.internStr(
		    (std::string("Non-function (") + Type::toString(a.type()) + ") as first parameter to call\n").c_str()),
            inst.c);
	
    Closure const& func = (Closure const&)a;
	CompiledCall const& call = thread.frame.code->calls[inst.b];
	
	Environment* fenv = FastMatchArgs(thread, thread.frame.environment, func, call);
    return buildStackFrame(thread, fenv, func.prototype()->code, inst.c, &inst+1);
}

static inline Instruction const* ret_op(Thread& thread, Instruction const& inst) {
	// we can return futures from functions, so don't BIND
	DECODE(a);
	
    if(thread.stack.size() == 1)
        _error("no function to return from, jumping to top level");

    if(thread.frame.isPromise) {
        Environment* env = thread.frame.environment;
        do {
            if(thread.stack.size() <= 1)
                _error("no function to return from, jumping to top level");
            thread.pop();
        } while( !(thread.frame.environment == env 
                 && thread.frame.isPromise == false) );
    }

	REGISTER(0) = a;
	Instruction const* returnpc = thread.frame.returnpc;
    
    Value& onexit = thread.frame.environment->insert(Strings::__onexit__);
    if(onexit.isObject()) {
        Promise::Init(onexit,
            thread.frame.environment,
            Compiler::deferPromiseCompilation(thread, onexit), false);
		return force(thread, (Promise const&)onexit,
            thread.frame.environment, Value::Nil(),
            1, &inst);
	}
    
	// We can free this environment for reuse
	// as long as we don't return a closure...
	// but also can't if an assignment to an 
    // out of scope variable occurs (<<-, assign) with a value of a closure!
#ifdef EPEE
	if(!(a.isClosure() || a.isEnvironment() || a.isList())) {
		thread.traces.KillEnvironment(thread.frame.environment);
	}
#endif

    thread.pop();

#ifdef EPEE
	thread.traces.LiveEnvironment(thread.frame.environment, a);
#endif
	return returnpc;
}

static inline Instruction const* retp_op(Thread& thread, Instruction const& inst) {
	// we can return futures from promises, so don't BIND
	DECODE(a);
	
    if(REGISTER(1).isCharacter()) {
        assert(REGISTER(0).isEnvironment());
	    Environment* env = ((REnvironment const&)REGISTER(0)).environment();
		
        env->insert(REGISTER(1).s) = a;
#ifdef EPEE
	    thread.traces.LiveEnvironment(env, a);
#endif
	} else if(REGISTER(1).isInteger()) {
        assert(REGISTER(0).isEnvironment());
	    Environment* env = ((REnvironment const&)REGISTER(0)).environment();
        
        assert(env->get(Strings::__dots__).isList());
        ((List&)env->insert(Strings::__dots__))[REGISTER(1).i] = a;
#ifdef EPEE
	    thread.traces.LiveEnvironment(env, a);
#endif
	} // otherwise, don't store anywhere...
	
	REGISTER(0) = a;
	
    Instruction const* returnpc = thread.frame.returnpc;
	thread.pop();
	
	return returnpc;
}

static inline Instruction const* jmp_op(Thread& thread, Instruction const& inst) {
	return &inst+inst.a;
}

static inline Instruction const* jc_op(Thread& thread, Instruction const& inst) {
    thread.visible = true;
	DECODE(c); BIND(c);
    Logical::Element cond;
    if(c.isLogical1())
        cond = c.c;
    else if(c.isInteger1())
        cond = Cast<Integer, Logical>(thread, c.i);
    else if(c.isDouble1())
        cond = Cast<Double, Logical>(thread, c.i);
    // It breaks my heart to allow non length-1 vectors,
    // but this seems to be somewhat widely used, even
    // in some of the recommended R packages.
    else if(c.isLogical()) {
        if(((Logical const&)c).length() > 0)
            cond = ((Logical const&)c)[0];
        else
            return StopDispatch(thread, inst, thread.internStr(
            "conditional is of length zero"), 
            inst.c);
    }
    else if(c.isInteger()) {
        if(((Integer const&)c).length() > 0)
            cond = Cast<Integer, Logical>(thread, ((Integer const&)c)[0]);
        else
            return StopDispatch(thread, inst, thread.internStr(
            "conditional is of length zero"), 
            inst.c);
    }
    else if(c.isDouble()) {
        if(((Double const&)c).length() > 0)
            cond = Cast<Double, Logical>(thread, ((Double const&)c)[0]);
        else
            return StopDispatch(thread, inst, thread.internStr(
            "conditional is of length zero"), 
            inst.c);
    }
    else if(c.isRaw()) {
        if(((Raw const&)c).length() > 0)
            cond = Cast<Raw, Logical>(thread, ((Raw const&)c)[0]);
        else
            return StopDispatch(thread, inst, thread.internStr(
            "conditional is of length zero"), 
            inst.c);
    }
    else {
        return StopDispatch(thread, inst, thread.internStr(
                    "conditional argument is not interpretable as logical"), 
                    inst.c);
    }
    
    if(Logical::isTrue(cond)) return &inst+inst.a;
    else if(Logical::isFalse(cond)) return &inst+inst.b;
    else if((&inst+1)->bc != ByteCode::stop) return &inst+1;
    else return StopDispatch(thread, inst, thread.internStr(
            "NA where TRUE/FALSE needed"), 
            inst.c);
}

static inline Instruction const* branch_op(Thread& thread, Instruction const& inst) {
    thread.visible = true;
	DECODE(a);
	int64_t index = -1;
	if(a.isDouble1()) index = (int64_t)a.d;
	else if(a.isInteger1()) index = a.i;
	else if(a.isLogical1()) index = a.i;
	else if(a.isCharacter1()) {
		for(int64_t i = 1; i <= inst.b; i++) {
			String s = CONSTANT((&inst+i)->a).s;
			if(s == a.s) {
				index = i;
				break;
			}
			if(index < 0 && s == Strings::empty) {
				index = i;
			}
		}
	}
    else
        _error("EXPR must be a length 1 vector");

	if(index >= 1 && index <= inst.b) {
		return &inst + ((&inst+index)->c);
	} 
	BIND(a);
	return &inst+1+inst.b;
}

static inline Instruction const* forbegin_op(Thread& thread, Instruction const& inst) {
    thread.visible = true;
	// a = loop variable (e.g. i), b = loop vector(e.g. 1:100), c = counter register
	// following instruction is a jmp that contains offset
    Value const& b = REGISTER(inst.b);
	BIND(b);
    if(!b.isVector())
		_error("Invalid for() loop sequence");
	Vector const& v = (Vector const&)b;
	if((int64_t)v.length() <= 0) {
		return &inst+(&inst+1)->a;	// offset is in following JMP, dispatch together
	} else {
        String i = ((Character const&)CONSTANT(inst.a)).s;
		Element2(v, 0, thread.frame.environment->insert(i));
		Integer::InitScalar(REGISTER(inst.c), 1);
		Integer::InitScalar(REGISTER(inst.c+1), v.length());
		return &inst+2;			// skip over following JMP
	}
}

static inline Instruction const* forend_op(Thread& thread, Instruction const& inst) {
    thread.visible = true;
	Value& counter = REGISTER(inst.c);
	Value& limit = REGISTER(inst.c+1);
	if(__builtin_expect(counter.i < limit.i, true)) {
        String i = ((Character const&)CONSTANT(inst.a)).s;
		Value const& b = REGISTER(inst.b);
		Element2(b, counter.i, thread.frame.environment->insert(i));
		counter.i++;
		return &inst+(&inst+1)->a;
	} else {
		return &inst+2;			// skip over following JMP
	}
}

static inline Instruction const* mov_op(Thread& thread, Instruction const& inst) {
	DECODE(a);
	OUT(c) = a;
	return &inst+1;
}

static inline Instruction const* invisible_op(Thread& thread, Instruction const& inst) {
    thread.visible = false;
    DECODE(a);
    OUT(c) = a;
    return &inst+1;
}

static inline Instruction const* visible_op(Thread& thread, Instruction const& inst) {
    thread.visible = true;
    DECODE(a);
    OUT(c) = a;
    return &inst+1;
}

static inline Instruction const* external_op(Thread& thread, Instruction const& inst) {
    thread.visible = true;
	DECODE(a);

    void* func = NULL;
    if(a.isCharacter1()) {
        String name = a.s;
        for(std::map<std::string,void*>::iterator i = thread.state.handles.begin();
            i != thread.state.handles.end(); ++i) {
            func = dlsym(i->second, name->s);
            if(func != NULL)
                break;
        }
        if(func == NULL)
            _error(std::string("Can't find external function: ") + name->s);
    }
    else if(a.isExternalptr()) {
        func = ((Externalptr const&)a).ptr();
    }

    if(func == NULL) {
        return StopDispatch(thread, inst, thread.internStr(
            ".External needs a Character(1) or Externalptr as its first argument"), 
            inst.c);
    }

    uint64_t nargs = inst.b;
	for(int64_t i = 0; i < nargs; i++) {
		BIND(REGISTER(inst.a+i+1));
	}
    try {
        typedef Value (*Func)(Thread&, Value const*);
        Func f = (Func)func;
        OUT(c) = f(thread, &REGISTER(inst.a+1));
    }
    catch( char const* e ) {
        _error(std::string("External function call failed: ") + e);
    }
    catch( ... ) {
        _error(std::string("External function call failed with unknown error"));
    }

	return &inst+1;
}

static inline Instruction const* map_op(Thread& thread, Instruction const& inst) {
    thread.visible = true;
    DECODE(a); BIND(a);
    DECODE(b); BIND(b);
    DECODE(c); BIND(c);

    if(!b.isList())
        _error("External map args must be a list");

    if(c.isCharacter1()) {
        if(!a.isCharacter())
            _error("External map return types must be a character vector");
        OUT(c) = Map(thread, c.s, (List const&)b, (Character const&)a);
    }
    else if(c.isClosure()) {
        if(a.isCharacter())
            OUT(c) = MapR(thread, (Closure const&)c, (List const&)b, (Character const&)a);
        else
            OUT(c) = MapI(thread, (Closure const&)c, (List const&)b);
    }
    else
        _error(".Map function name must be a string or a closure");
    
    return &inst+1;
}

static inline Instruction const* scan_op(Thread& thread, Instruction const& inst) {
    thread.visible = true;
    DECODE(a); BIND(a);
    DECODE(b); BIND(b);
    DECODE(c); BIND(c);

    if(!c.isCharacter1())
        _error("External scan function name must be a string");
    if(!b.isList())
        _error("External scan args must be a list");
    if(!a.isCharacter())
        _error("External scan return types must be a character vector");

    OUT(c) = Scan(thread, c.s, (List const&)b, (Character const&)a);
    
    return &inst+1;
}

static inline Instruction const* fold_op(Thread& thread, Instruction const& inst) {
    thread.visible = true;
    DECODE(a); BIND(a);
    DECODE(b); BIND(b);
    DECODE(c); BIND(c);

    if(!c.isCharacter1())
        _error("External fold function name must be a string");
    if(!b.isList())
        _error("External fold args must be a list");
    if(!a.isCharacter())
        _error("External fold return types must be a character vector");

    OUT(c) = Fold(thread, c.s, (List const&)b, (Character const&)a);
    
    return &inst+1;
}

// LOAD_STORE_BYTECODES

static inline Instruction const* load_op(Thread& thread, Instruction const& inst) {
    thread.visible = true;
	String s = ((Character const&)CONSTANT(inst.a)).s;
	Environment* env;
	Value const& v = thread.frame.environment->getRecursive(s, env);
	if(v.isObject()) {
		OUT(c) = v;
		return &inst+1;
    }
    else if(v.isPromise()) {
        return force(thread, (Promise const&)v, 
            env, ((Character const&)CONSTANT(inst.a)),
            inst.c, &inst+1);
    }
    else {
        return StopDispatch(thread, inst, thread.internStr(
            (std::string("Object '") + s->s + "' not found").c_str()), 
            inst.c);
    }
}

static inline Instruction const* loadfn_op(Thread& thread, Instruction const& inst) {
    thread.visible = true;
	String s = ((Character const&)CONSTANT(inst.a)).s;
	Environment* env = thread.frame.environment;

    // Iterate until we find a function
    do {
	    Value const& v = env->getRecursive(s, env);

        if(v.isClosure()) {
            OUT(c) = v;
            return &inst+1;
        }
	    else if(v.isPromise()) {
            // Must return to this instruction to check if it's a function.
            return force(thread, (Promise const&)v, 
                env, ((Character const&)CONSTANT(inst.a)),
                inst.c, &inst);
        }
        else if(v.isNil()) {
            return StopDispatch(thread, inst, thread.internStr(
                (std::string("Object '") + s->s + "' not found").c_str()), 
                inst.c);
        }
        env = env->getEnclosure();
	} while(env != 0);

    _internalError("loadfn failed to find value");
}

static inline Instruction const* store_op(Thread& thread, Instruction const& inst) {
    thread.visible = false;
    String s = ((Character const&)CONSTANT(inst.a)).s; 
	DECODE(c); // don't BIND
	thread.frame.environment->insert(s) = c;
	return &inst+1;
}

static inline Instruction const* storeup_op(Thread& thread, Instruction const& inst) {
    thread.visible = false;
	// assign2 is always used to assign up at least one scope level...
	// so start off looking up one level...
	assert(thread.frame.environment->getEnclosure() != 0);

	DECODE(c); BIND(c);
	
	String s = ((Character const&)CONSTANT(inst.a)).s;
	Environment* penv;
	Value& dest = thread.frame.environment->getEnclosure()->insertRecursive(s, penv);
    if(!dest.isNil())
        dest = c;
    else
        thread.state.global->insert(s) = c;

	return &inst+1;
}

static inline Instruction const* rm_op(Thread& thread, Instruction const& inst) {
    String s = ((Character const&)CONSTANT(inst.a)).s;
    thread.frame.environment->remove( s );
    OUT(c) = Null::Singleton();
    return &inst+1;
}

static inline Instruction const* force_op(Thread& thread, Instruction const& inst) {
    Value const& a = REGISTER(inst.a);

    assert(thread.frame.environment->get(Strings::__dots__).isList());
    Value const& t = ((List const&)thread.frame.environment->get(Strings::__dots__))[a.i];

    if(t.isObject()) {
        OUT(c) = t;
        return &inst+1;
    }
    else if(t.isPromise()) {
        return force(thread, (Promise const&)t,
                thread.frame.environment, a,
                inst.c, &inst+1);
    }
    else {
        _internalError("Unexpected Nil operand in force_op");
    }
}

static inline Instruction const* dotsv_op(Thread& thread, Instruction const& inst) {
    thread.visible = true;
    DECODE(a); BIND(a);

    int64_t idx = 0;
    if(a.isInteger1())
        idx = ((Integer const&)a)[0] - 1;
    else if(a.isDouble1())
        idx = (int64_t)((Double const&)a)[0] - 1;
    else
        return StopDispatch(thread, inst, thread.internStr("Invalid type in dotsv"), inst.c);

    Value const& t = thread.frame.environment->get(Strings::__dots__);

	if(!t.isList() ||
       idx >= (int64_t)((List const&)t).length() ||
       idx < (int64_t)0)
        return StopDispatch(thread, inst, thread.internStr((std::string("The '...' list does not contain ") + intToStr(idx+1) + " elements").c_str()), inst.c);
	
    Value const& v = ((List const&)t)[idx];
   
    if(v.isObject()) {
        OUT(c) = v;
        return &inst+1;
    }
    else if(v.isPromise()) {
        return force(thread, (Promise const&)v,
            thread.frame.environment, Integer::c(idx),
            inst.c, &inst+1);
    }
    else {
        return StopDispatch(thread, inst, thread.internStr(
            (std::string("Object '..") + intToStr(idx+1) + 
                "' not found, missing argument?").c_str()), 
            inst.c);
    }
}

static inline Instruction const* dotsc_op(Thread& thread, Instruction const& inst) {
    thread.visible = true;
    if(!thread.frame.environment->get(Strings::__dots__).isList())
        OUT(c) = Integer::c(0);
    else
        OUT(c) = Integer::c((int64_t)((List const&)thread.frame.environment->get(Strings::__dots__)).length());
    return &inst+1;
}

static inline Instruction const* dots_op(Thread& thread, Instruction const& inst) {
    thread.visible = true;
	static const List empty(0);
    List const& dots = 
        thread.frame.environment->has(Strings::__dots__)
            ? (List const&)thread.frame.environment->get(Strings::__dots__)
            : empty;
	
	Value& iter = REGISTER(inst.a);
	Value& out = OUT(c);

	// First time through, make a result vector...
	if(iter.i == -1) {
		Heap::Global.collect(thread.state);
		out = List(dots.length());
		memset(((List&)out).v(), 0, dots.length()*sizeof(List::Element));
	    iter.i++;
    }
	
	while(iter.i < (int64_t)dots.length()) {
		Value const& v = dots[iter.i];

        if(v.isObject()) {
		    BIND(v); // BIND since we don't yet support futures in lists
		    ((List&)out)[iter.i] = v;
		    iter.i++;
        }
        else if(v.isPromise()) {
            return force(thread, (Promise const&)v,
                thread.frame.environment, Integer::c(iter.i),
                inst.b, &inst);
        }
        else if(v.isNil()) {
            // We're allowing Nils to escape now
		    ((List&)out)[iter.i] = v;
		    iter.i++;
            /*return StopDispatch(thread, inst, thread.internStr(
                "argument is missing, with no default"),
                inst.c);*/
        }
	}
	
	// check to see if we need to add names
    if(thread.frame.environment->get(Strings::__names__).isCharacter() 
        && dots.length() > 0) {
        Dictionary* d = new Dictionary(1);
        d->insert(Strings::names) = thread.frame.environment->get(Strings::__names__);
        ((Object&)out).attributes(d);
	}
    return &inst+1;
}

static inline Instruction const* missing_op(Thread& thread, Instruction const& inst) {
    thread.visible = true;
    DECODE(a);
    bool missing = false;

    Environment* e = thread.frame.environment;
    Value x = a;

    do {

    if( x.isCharacter() && ((Character const&)x).length() == 1 ) {
        Environment* foundEnv;
        Value const& v = e->getRecursive(x.s, foundEnv);
        missing = (    v.isPromise()
                    && ((Promise const&)v).isDefault() 
                    && foundEnv == thread.frame.environment
                  ) || v.isNil();

        if(v.isPromise() && !((Promise const&)v).isDefault() && foundEnv == e) {
            // see if missing is passed down
            // missing is only passed down if
            // the referenced symbol is an argument.
            // this whole feature is a disaster.
            if(((Promise const&)v).isExpression()) {
                Value const& expr = ((Promise const&)v).code()->expression;
                Environment* env = ((Promise const&)v).environment();
                Value const& func = env->get(Strings::__function__);
                if(isSymbol(expr) && func.isClosure()) {
                    // see if the expr is an argument
                    Character const& parameters = ((Closure const&)func).prototype()->parameters;
                    bool matched = false;
                    for(size_t i = 0; i < parameters.length() && !matched; ++i) {
                        if(((Character const&)expr).s == parameters[i])
                            matched = true;
                    }
                    
                    if(!matched)
                        break;

                    e = env;
                    x = expr;
                    continue;
                }
            }
        }
        break;
    }
    else {
        int64_t index = -1;
        if( x.isInteger() && ((Integer const&)x).length() == 1 )
            index = x.i-1;
        else if( x.isDouble() && ((Double const&)x).length() == 1 )
            index = x.d-1;
        else
            _error("Invalid argument to missing");

        List const& dots = (List const&)
            e->get(Strings::__dots__);

        missing = !dots.isList() ||
                  index < 0 ||
                  index >= dots.length() ||
                  dots[index].isNil();

        break;
    }

    } while(true);

    Logical::InitScalar(OUT(c),
        missing ? Logical::TrueElement : Logical::FalseElement);

    return &inst+1;
}

// STACK_FRAME_BYTECODES
static inline Instruction const* frame_op(Thread& thread, Instruction const& inst) {
    thread.visible = true;
    DECODE(a); BIND(a);
    int64_t index = a.i;

    Environment* env = thread.frame.environment;

    while(index > 0) {
        Value const& v = env->get(Strings::__parent__);
        if(v.isEnvironment())
            env = ((REnvironment const&)v).environment();
        else
            break;
        index--;
    }

    if(index == 0) {
        REnvironment::Init(OUT(c), env);
        return &inst+1;
    }
    else {
        return StopDispatch(thread, inst, thread.internStr(
            "not that many frames on the stack"), 
            inst.c);
    }
}

// PROMISE_BYTECODES

static inline Instruction const* pr_new_op(Thread& thread, Instruction const& inst) {
    thread.visible = true;
    DECODE(a); BIND(a);
    DECODE(b); BIND(b);

    REnvironment& eval = (REnvironment&)REGISTER((&inst+1)->a);
    REnvironment& assign = (REnvironment&)REGISTER((&inst+1)->b);

    Value& v = assign.environment()->insert(a.s);

    try {
        Promise::Init(v,
            eval.environment(),
            Compiler::deferPromiseCompilation(thread, b), false);
    }
    catch(RuntimeError const& e) {
        return StopDispatch(thread, inst, thread.internStr(
            e.what().c_str()), 
            inst.c);
    }

    OUT(c) = Null::Singleton();

    return &inst+2;
}

static inline Instruction const* pr_expr_op(Thread& thread, Instruction const& inst) {
    thread.visible = true;
    DECODE(a); BIND(a);
    DECODE(b); BIND(b);

    if(a.isEnvironment() && b.isCharacter() && ((Character const&)b).length() == 1) {
        REnvironment const& env = ((REnvironment const&)a);
	    String s = ((Character const&)b).s;

	    Value v = env.environment()->get(s);
        if(v.isPromise())
            v = ((Promise const&)v).code()->expression;
        OUT(c) = v;
        return &inst+1;
    }
    else if(a.isList() && b.isInteger1()) {
        List const& l = ((List const&)a);
        int64_t i = ((Integer const&)b).i - 1;
        if(i >= 0 && i < l.length()) {
            Value v = l[i];
            if(v.isPromise())
                v = ((Promise const&)v).code()->expression;
            OUT(c) = v;
            return &inst+1;
        }
    }
    printf("%d %d\n", a.type(), b.type());
    return StopDispatch(thread, inst, thread.internStr(
                "invalid pr_expr expression"), 
                inst.c);
}

static inline Instruction const* pr_env_op(Thread& thread, Instruction const& inst) {
    thread.visible = true;
    DECODE(a); BIND(a);
    DECODE(b); BIND(b);

    // TODO: check types

    REnvironment const& env = ((REnvironment const&)a);
	String s = ((Character const&)b).s;
	Value v = env.environment()->get(s);
    if(v.isPromise())
        REnvironment::Init(v, ((Promise const&)v).environment());
    else
        v = Null::Singleton();
	OUT(c) = v;
    return &inst+1;
}

// OBJECT_BYTECODES

static inline Instruction const* isnil_op(Thread& thread, Instruction const& inst) {
    thread.visible = true;
	DECODE(a);
	OUT(c) = a.isNil() ? Logical::True() : Logical::False();
    return &inst+1;
}

static inline Instruction const* type_op(Thread& thread, Instruction const& inst) {
    thread.visible = true;
	DECODE(a);
#ifdef EPEE
	switch(thread.traces.futureType(a)) {
#else
    switch(a.type()) {
#endif
        #define CASE(name, str, ...) case Type::name: OUT(c) = Character::c(Strings::name); break;
        TYPES(CASE)
        #undef CASE
        default:
            // This can happen in stop dispatch, so don't re-dispatch.
            /*return StopDispatch(thread, inst, thread.internStr(
                (std::string("Unknown type (") + intToStr(a.type()) + ") in type to string, that's bad!").c_str()), 
                inst.c);*/
            printf("Unknown type in type_op %d\n", a.type());
            OUT(c) = Character::c(thread.internStr(std::string("Unknown type (") + intToStr(a.type()) + ")"));
            break;
    }
	return &inst+1;
}

static inline Instruction const* length_op(Thread& thread, Instruction const& inst) {
    thread.visible = true;
	DECODE(a);
	if(a.isVector())
		Integer::InitScalar(OUT(c), ((Vector const&)a).length());
#ifdef EPEE
	else if(a.isFuture()) {
		IRNode::Shape shape = thread.traces.futureShape(a);
		if(shape.split < 0 && shape.filter < 0) {
			Integer::InitScalar(OUT(c), shape.length);
		} else {
			OUT(c) = thread.traces.EmitUnary<CountFold>(thread.frame.environment, IROpCode::length, a, 0);
			thread.traces.OptBind(thread, OUT(c));
		}
	}
#endif
	else if(((Object const&)a).hasAttributes()) { 
		return GenericDispatch(thread, inst, Strings::length, a, inst.c); 
	} else {
		Integer::InitScalar(OUT(c), 1);
	}
	return &inst+1;
}

static inline Instruction const* get_op(Thread& thread, Instruction const& inst) {
    thread.visible = true;
	DECODE(a); DECODE(b);
    if(GetFast(thread, a, b, OUT(c)))
        return &inst+1;
    else {
        try {
            return GetSlow( thread, inst, a, b, OUT(c) );
        }
        catch(RuntimeError const& e) {
            return StopDispatch(thread, inst, thread.internStr(
                e.what().c_str()), 
                inst.c);
        }
    }
}

static inline Instruction const* set_op(Thread& thread, Instruction const& inst) {
    thread.visible = true;
	// a = value, b = index, c = dest
	DECODE(a); DECODE(b); DECODE(c);
	BIND(b);

#ifdef EPEE
	if(a.isFuture() && (c.isVector() || c.isFuture())) {
		if(b.isInteger() && ((Integer const&)b).length() == 1) {
			OUT(c) = thread.traces.EmitSStore(thread.frame.environment, c, ((Integer&)b)[0], a);
			return &inst+1;
		}
		else if(b.isDouble() && ((Double const&)b).length() == 1) {
			OUT(c) = thread.traces.EmitSStore(thread.frame.environment, c, ((Double&)b)[0], a);
			return &inst+1;
		}
	}
#endif

	BIND(a);
	BIND(c);

    if( !((Object const&)c).hasAttributes() ) {
        if(c.isVector() && ( b.isDouble1() || b.isInteger1() )) {
            Subset2Assign(thread, c, true, b, a, OUT(c));
            return &inst+1;
        }
        else if(c.isEnvironment() && b.isCharacter1()) {
	        String s = ((Character const&)b).s;
            if(a.isNil())
                ((REnvironment&)c).environment()->remove(s);
            else
                ((REnvironment&)c).environment()->insert(s) = a;
            OUT(c) = c;
            return &inst+1;
        }
        else if(c.isClosure() && b.isCharacter1()) {
            //Closure const& f = (Closure const&)c;
	        //String s = ((Character const&)b).s;
            _error("Assignment to function members is not yet implemented");
        }
    }
    return GenericDispatch(thread, inst, Strings::bbAssign, c, b, a, inst.c); 
}

static inline Instruction const* getsub_op(Thread& thread, Instruction const& inst) {
    thread.visible = true;
	DECODE(a); DECODE(b);

	if(a.isVector() && !((Object const&)a).hasAttributes()) {
		if(b.isDouble1()
            && (int64_t)(b.d-1) >= 0
            && (int64_t)(b.d-1) < ((Vector const&)a).length()) { 
            Element(a, b.d-1, OUT(c)); return &inst+1; }
		else if(b.isInteger1()
            && (b.i-1) >= 0
            && (b.i-1) < ((Vector const&)a).length()) { 
            Element(a, b.i-1, OUT(c)); return &inst+1; }
	}

#ifdef EPEE
	if( thread.traces.isTraceable(a, b) 
		&& thread.traces.futureType(b) == Type::Logical 
		&& thread.traces.futureShape(a) == thread.traces.futureShape(b)) {
		OUT(c) = thread.traces.EmitFilter(thread.frame.environment, a, b);
		thread.traces.OptBind(thread, OUT(c));
		return &inst+1;
	}
#endif

	BIND(a);

#ifdef EPEE
	if(thread.traces.isTraceable(a, b) 
		&& (thread.traces.futureType(b) == Type::Integer 
			|| thread.traces.futureType(b) == Type::Double)) {
		OUT(c) = thread.traces.EmitGather(thread.frame.environment, a, b);
		thread.traces.OptBind(thread, OUT(c));
		return &inst+1;
	}
#endif	

    BIND(b);

    if(a.isVector() && !((Object const&)a).hasAttributes() &&
        (b.isInteger() || b.isDouble() || b.isLogical()) ) {
        SubsetSlow(thread, a, b, OUT(c)); 
        return &inst+1;
    }
    // TODO: this should force promises...
    else if(a.isEnvironment() && b.isCharacter()) {
        REnvironment const& env = ((REnvironment const&)a);
        Character const& i = ((Character const&)b);
        List r(i.length());
        for(int64_t k = 0; k < i.length(); ++k) {
            if(env.environment()->has(i[k]))
                r[k] = env.environment()->get(i[k]);
            else
                r[k] = Null::Singleton();
        }
        OUT(c) = r; 
        return &inst+1;
    }

    return GenericDispatch(thread, inst, Strings::bracket, a, b, inst.c); 
}

static inline Instruction const* setsub_op(Thread& thread, Instruction const& inst) {
    thread.visible = true;
	// a = value, b = index, c = dest 
	DECODE(a); DECODE(b); DECODE(c); 
    BIND(b); BIND(c);

#ifdef EPEE	
	if(a.isFuture() && (c.isVector() || c.isFuture())) {
		if(b.isInteger() && ((Integer const&)b).length() == 1) {
			OUT(c) = thread.traces.EmitSStore(thread.frame.environment, c, ((Integer&)b)[0], a);
			return &inst+1;
		}
		else if(b.isDouble() && ((Double const&)b).length() == 1) {
			OUT(c) = thread.traces.EmitSStore(thread.frame.environment, c, ((Double&)b)[0], a);
			return &inst+1;
		}
	}
#endif

	BIND(a);

    if( !((Object const&)c).hasAttributes() ) {
        if(c.isVector() && ( b.isDouble() || b.isInteger() || b.isLogical() )) {
            SubsetAssign(thread, c, true, b, a, OUT(c));
            return &inst+1;
        }
	}
    if(c.isEnvironment() && b.isCharacter()) {
        REnvironment const& env = ((REnvironment const&)c);
        Character const& i = ((Character const&)b);
        if(a.isVector()) {
            List const& l = As<List>(thread, a);
            int64_t len = std::max(l.length(), i.length());
            if(l.length() == 0 || i.length() == 0) len = 0;
            for(int64_t k = 0; k < len; ++k) {
                int64_t ix = k % i.length();
                int64_t lx = k % l.length();
                if(i[ix] != Strings::empty 
                    && !Character::isNA(i[ix])) {
                    if(l[lx].isNil())
                        env.environment()->remove(i[ix]);
                    else
                        env.environment()->insert(i[ix]) = l[lx];
                }
            }
        }
        else {
            int64_t len = i.length();
            if(a.isNil()) {
                for(int64_t k = 0; k < len; ++k) {
                    if(i[k] != Strings::empty && !Character::isNA(i[k]))
                        env.environment()->remove(i[k]);
                }
            }
            else {
                for(int64_t k = 0; k < len; ++k) {
                    if(i[k] != Strings::empty && !Character::isNA(i[k]))
                        env.environment()->insert(i[k]) = a;
                }
            }
        }
        OUT(c) = env; 
        return &inst+1;
    }

	return GenericDispatch(thread, inst, Strings::bracketAssign, c, b, a, inst.c); 
}

static inline Instruction const* getenv_op(Thread& thread, Instruction const& inst) {
    thread.visible = true;
    DECODE(a); BIND(a);

    if(a.isEnvironment()) {
        Environment* enc = ((REnvironment const&)a).environment()->getEnclosure();
        if(enc == 0)
            _error("environment does not have an enclosing environment");
        REnvironment::Init(OUT(c), enc);
    }
    else if(a.isClosure()) {
        REnvironment::Init(OUT(c), ((Closure const&)a).environment());
    }
    else if(a.isNull()) {
        REnvironment::Init(OUT(c), thread.frame.environment);
    }
    else {
        OUT(c) = Null::Singleton();
    }
    return &inst+1;
}

static inline Instruction const* setenv_op(Thread& thread, Instruction const& inst) {
    thread.visible = true;
    DECODE(a); BIND(a);
    DECODE(b); BIND(b);

    if(!b.isEnvironment())
        _error("replacement object is not an environment");

    Environment* value = ((REnvironment const&)b).environment();

    if(a.isEnvironment()) {
        Environment* target = ((REnvironment const&)a).environment();
        
        // Riposte allows enclosing environment replacement,
        // but requires that no loops be introduced in the environment chain.
        Environment* p = value;
        while(p) {
            if(p == target) 
                _error("an environment cannot be its own ancestor");
            p = p->getEnclosure();
        }
        
        ((REnvironment const&)a).environment()->setEnclosure(
            ((REnvironment const&)b).environment());
        OUT(c) = a;
    }
    else if(a.isClosure()) {
        Closure::Init(OUT(c), ((Closure const&)a).prototype(), value);
    }
    else {
        return StopDispatch(thread, inst, thread.internStr(
            "target of assignment does not have an enclosing environment"), 
            inst.c);
    }
    return &inst+1;
}

static inline Instruction const* getattr_op(Thread& thread, Instruction const& inst) {
    thread.visible = true;
	DECODE(a);
	DECODE(b); BIND(b);
	if(a.isObject() && b.isCharacter1()) {
		String name = ((Character const&)b)[0];
		Object const& o = (Object const&)a;
        if(o.isEnvironment()) {
            if(((REnvironment&)o).environment()->hasAttributes() &&
               ((REnvironment&)o).environment()->getAttributes()->has(name))
                OUT(c) = ((REnvironment&)o).environment()->getAttributes()->get(name);
            else
                OUT(c) = Null::Singleton();
        }
        else {
            if(o.hasAttributes() && o.attributes()->has(name))
                OUT(c) = o.attributes()->get(name);
            else
                OUT(c) = Null::Singleton();
        }
		return &inst+1;
	}
    printf("%d %d\n", a.type(), b.type());
	_error("Invalid attrget operation");
}

static inline Instruction const* setattr_op(Thread& thread, Instruction const& inst) {
    thread.visible = true;
	DECODE(c);
	DECODE(b); BIND(b);
	DECODE(a); BIND(a);
	if(c.isObject() && b.isCharacter1()) {
		String name = ((Character const&)b)[0];
		Object o = (Object const&)c;
        if(a.isNil() || a.isNull()) {
            if(o.isEnvironment() &&
                ((REnvironment&)o).environment()->hasAttributes() &&
                ((REnvironment&)o).environment()->getAttributes()->has(name)) {
                if(((REnvironment&)o).environment()->getAttributes()->Size() > 1) {
                    Dictionary* d = ((REnvironment&)o).environment()->getAttributes()->clone(0);
                    d->remove(name);
		            ((REnvironment&)o).environment()->setAttributes(d);
                }
                else {
		            ((REnvironment&)o).environment()->setAttributes(NULL);
                }
            }
            else if(o.hasAttributes()
               && o.attributes()->has(name)) {
                if(o.attributes()->Size() > 1) {
                    Dictionary* d = o.attributes()->clone(0);
                    d->remove(name);
		            o.attributes(d);
                }
                else {
                    o.attributes(NULL);
                }
            }
        }
        else {
            Value v = a;
            if(name == Strings::rownames && v.isInteger()) {
                Integer const& i = (Integer const&)v;
                if(i.length() == 2 && Integer::isNA(i[0])) {    
                    v = Sequence((int64_t)1,1,abs(i[1]));
                }
            }
            if(o.isEnvironment()) {
                Dictionary* d = ((REnvironment&)o).environment()->getAttributes();
                d = d ? d->clone(1) : new Dictionary(1);
                d->insert(name) = v;
                ((REnvironment&)o).environment()->setAttributes(d);
            }
            else {
                Dictionary* d = o.hasAttributes()
                            ? o.attributes()->clone(1)
                            : new Dictionary(1);
                d->insert(name) = v;
                o.attributes(d);
            }
        }
		OUT(c) = o;
		return &inst+1;
	}
	_error("Invalid attrset operation");
}

static inline Instruction const* attributes_op(Thread& thread, Instruction const& inst) {
    thread.visible = true;
    DECODE(a);
    if(a.isObject()) {
        Object o = (Object const&)a;

        Dictionary* d = o.isEnvironment()
            ? ((REnvironment&)o).environment()->getAttributes()
            : o.attributes();

        if(d == NULL || d->Size() == 0) {
            OUT(c) = Null::Singleton();
        }
        else {
            Character n(d->Size());
            List v(d->Size());
            int64_t j = 0;
            for(Dictionary::const_iterator i = d->begin();
                    i != d->end(); 
                    ++i, ++j) {
                n[j] = i.string();
                v[j] = i.value();
            }
			Dictionary* r = new Dictionary(1);
			r->insert(Strings::names) = n;
			v.attributes(r);
            OUT(c) = v;
        }
    }
    else {
        OUT(c) = Null::Singleton();
    }
    return &inst+1;
}

static inline Instruction const* strip_op(Thread& thread, Instruction const& inst) {
    thread.visible = true;
	DECODE(a);
	Value& c = OUT(c);
	c = a;
	((Object&)c).attributes(0);
	return &inst+1;
}

static inline Instruction const* as_op(Thread& thread, Instruction const& inst) {
    thread.visible = true;
	DECODE(a); BIND(a);
    DECODE(b); BIND(b);
    if(!b.isCharacter1()) {
        return StopDispatch(thread, inst, thread.internStr(
            "invalid type argument to 'as'"), 
            inst.c);
    }
	String type = b.s;
    try {
        if(type == Strings::Null)
            OUT(c) = As<Null>(thread, a);
        else if(type == Strings::Logical)
            OUT(c) = As<Logical>(thread, a);
        else if(type == Strings::Integer)
            OUT(c) = As<Integer>(thread, a);
        else if(type == Strings::Double)
            OUT(c) = As<Double>(thread, a);
        else if(type == Strings::Character)
            OUT(c) = As<Character>(thread, a);
        else if(type == Strings::List)
            OUT(c) = As<List>(thread, a);
        else if(type == Strings::Raw)
            OUT(c) = As<Raw>(thread, a);
        else
            return StopDispatch(thread, inst, thread.internStr(
                "'as' not yet defined for this type"), 
                inst.c);
    }
    catch(RuntimeError const& e) {
        return StopDispatch(thread, inst, thread.internStr(
                e.what()), 
                inst.c);
    }

    // Add support for futures
	return &inst+1; 
}

// ENVIRONMENT_BYTECODES

static inline Instruction const* env_new_op(Thread& thread, Instruction const& inst) {
    thread.visible = true;
    DECODE(a); BIND(a);

    if(!a.isEnvironment())
        _error("'enclos' must be an environment");

    Environment* env = new Environment(4,((REnvironment const&)a).environment());
    Value p;
    REnvironment::Init(p, thread.frame.environment);
    env->insert(Strings::__parent__) = p;
    REnvironment::Init(OUT(c), env);
    return &inst+1;
}

static inline Instruction const* env_names_op(Thread& thread, Instruction const& inst) {
    thread.visible = true;
    DECODE(a); BIND(a);

    if(!a.isEnvironment())
        _error("'enclos' must be an environment");

    Environment* env = ((REnvironment const&)a).environment();

    Character r(env->Size());
    int64_t j = 0;
    for(Dictionary::const_iterator i = env->begin(); i != env->end(); ++i, ++j)
    {
        r[j] = i.string();
    }

    OUT(c) = r;
    return &inst+1;
}

static inline Instruction const* env_get_op(Thread& thread, Instruction const& inst) {
    thread.visible = true;
    DECODE(a); BIND(a);
    DECODE(b); BIND(b);

    if(!a.isEnvironment()) {
        return StopDispatch(thread, inst, thread.internStr(
            "invalid 'envir' argument to .getenv"), 
            inst.c);
    }
    if(!b.isCharacter() || b.pac != 1) {
        return StopDispatch(thread, inst, thread.internStr(
            "invalid 'x' argument to .getenv"), 
            inst.c);
    }

    OUT(c) = ((REnvironment const&)a).environment()->
                get(((Character const&)b).s);
    return &inst+1;
}

static inline Instruction const* env_global_op(Thread& thread, Instruction const& inst) {
    thread.visible = true;
    REnvironment::Init(OUT(c), thread.state.global);
    return &inst+1;
}

// FUNCTION_BYTECODES

static inline Instruction const* fn_new_op(Thread& thread, Instruction const& inst) {
    thread.visible = true;
	Value const& function = CONSTANT(inst.a);
	Value& out = OUT(c);
	Closure::Init(out, ((Closure const&)function).prototype(), thread.frame.environment);
	return &inst+1;
}


// VECTOR BYTECODES

#define OP(Name, string, Group, Func) \
static inline Instruction const* Name##_op(Thread& thread, Instruction const& inst) { \
    thread.visible = true; \
	DECODE(a);	\
    if( Group##Fast<Name##VOp>( thread, NULL, a, OUT(c) ) ) \
        return &inst+1; \
    else { \
        try { \
            return Name##Slow( thread, inst, NULL, a, OUT(c) ); \
        } \
        catch(RuntimeError const& e) {\
            return StopDispatch(thread, inst, thread.internStr( \
                e.what().c_str()), \
                inst.c); \
        } \
    } \
}
UNARY_FOLD_SCAN_BYTECODES(OP)
#undef OP

#define OP(Name, string, Group, Func) \
static inline Instruction const* Name##_op(Thread& thread, Instruction const& inst) { \
    thread.visible = true; \
	DECODE(a);	\
	DECODE(b);	\
    if( Group##Fast<Name##VOp>( thread, NULL, a, b, OUT(c) ) ) \
        return &inst+1; \
    else { \
        try { \
            return Name##Slow( thread, inst, NULL, a, b, OUT(c) ); \
        } \
        catch(RuntimeError const& e) { \
            return StopDispatch(thread, inst, thread.internStr( \
                e.what().c_str()), \
                inst.c); \
        } \
    } \
}
BINARY_BYTECODES(OP)
#undef OP

static inline Instruction const* ifelse_op(Thread& thread, Instruction const& inst) {
    thread.visible = true;
	DECODE(a);
	DECODE(b);
	DECODE(c);
	if(c.isLogical1()) {
		OUT(c) = Logical::isTrue(c.c) ? b : a;
		return &inst+1; 
	}
	else if(c.isInteger1()) {
		OUT(c) = c.i ? b : a;
		return &inst+1; 
	}
	else if(c.isDouble1()) {
		OUT(c) = c.d ? b : a;
		return &inst+1; 
	}
#ifdef EPEE
	if(thread.traces.isTraceable<IfElse>(a,b,c)) {
		OUT(c) = thread.traces.EmitIfElse(thread.frame.environment, a, b, c);
		thread.traces.OptBind(thread, OUT(c));
		return &inst+1;
	}
#endif
	BIND(a); BIND(b); BIND(c);

    IfElseDispatch(thread, NULL, b, a, c, OUT(c));
	return &inst+1; 
}

static inline Instruction const* split_op(Thread& thread, Instruction const& inst) {
    thread.visible = true;
#ifdef EPEE
	DECODE(a); BIND(a);
	DECODE(b);
	DECODE(c);
	int64_t levels = As<Integer>(thread, a)[0];
	if(thread.traces.isTraceable<Split>(b,c)) {
		OUT(c) = thread.traces.EmitSplit(thread.frame.environment, c, b, levels);
		thread.traces.OptBind(thread, OUT(c));
		return &inst+1;
	}
	BIND(a); BIND(b); BIND(c);
#endif

	_error("split not defined in scalar yet");
	return &inst+1; 
}

static inline Instruction const* vector_op(Thread& thread, Instruction const& inst) {
    thread.visible = true;
	DECODE(a); BIND(a);
	DECODE(b); BIND(b);
	String stype = As<Character>(thread, a)[0];
	Type::Enum type = string2Type( stype );
	int64_t l = As<Integer>(thread, b)[0];

#ifdef EPEE	
	if(thread.state.epeeEnabled 
		&& (type == Type::Double || type == Type::Integer || type == Type::Logical)
		&& l >= TRACE_VECTOR_WIDTH) {
		OUT(c) = thread.traces.EmitConstant(thread.frame.environment, type, l, 0);
		thread.traces.OptBind(thread, OUT(c));
		return &inst+1;
	}
#endif

	if(type == Type::Logical) {
		Logical v(l);
		for(int64_t i = 0; i < l; i++) v[i] = Logical::FalseElement;
		OUT(c) = v;
	} else if(type == Type::Integer) {
		Integer v(l);
		for(int64_t i = 0; i < l; i++) v[i] = 0;
		OUT(c) = v;
	} else if(type == Type::Double) {
		Double v(l);
		for(int64_t i = 0; i < l; i++) v[i] = 0;
		OUT(c) = v;
	} else if(type == Type::Character) {
		Character v(l);
		for(int64_t i = 0; i < l; i++) v[i] = Strings::empty;
		OUT(c) = v;
	} else if(type == Type::Raw) {
		Raw v(l);
		for(int64_t i = 0; i < l; i++) v[i] = 0;
		OUT(c) = v;
	} else if(type == Type::List) {
        List v(l);
        for(int64_t i = 0; i < l; i++) v[i] = Null::Singleton();
        OUT(c) = v;
    } else {
		_error(std::string("Invalid type in vector: ") + stype->s);
	} 
	return &inst+1;
}

static inline Instruction const* seq_op(Thread& thread, Instruction const& inst) {
    thread.visible = true;
    DECODE(a); BIND(a);

    int64_t len = As<Integer>(thread, a)[0];

#ifdef EPEE
    if(len >= TRACE_VECTOR_WIDTH) {
        OUT(c) = thread.traces.EmitSequence(thread.frame.environment, len, 1LL, 1LL);
        thread.traces.OptBind(thread, OUT(c));
        return &inst+1;
    }
#endif

    OUT(c) = Sequence(1LL, 1LL, len);
    return &inst+1;
}

static inline Instruction const* index_op(Thread& thread, Instruction const& inst) {
    thread.visible = true;
	// c = n, b = each, a = length
	DECODE(a); BIND(a);
	DECODE(b); BIND(b);
	DECODE(c); BIND(c);

	int64_t n = As<Integer>(thread, c)[0];
	int64_t each = As<Integer>(thread, b)[0];
	int64_t len = As<Integer>(thread, a)[0];

#ifdef EPEE	
	if(len >= TRACE_VECTOR_WIDTH) {
		OUT(c) = thread.traces.EmitIndex(thread.frame.environment, len, (int64_t)n, (int64_t)each);
		thread.traces.OptBind(thread, OUT(c));
		return &inst+1;
	}
#endif

	OUT(c) = Repeat((int64_t)n, (int64_t)each, len);
	return &inst+1;
}

static inline Instruction const* random_op(Thread& thread, Instruction const& inst) {
    thread.visible = true;
	DECODE(a); BIND(a);

	int64_t len = As<Integer>(thread, a)[0];
	
	/*if(len >= TRACE_VECTOR_WIDTH) {
		OUT(c) = thread.EmitRandom(thread.frame.environment, len);
		thread.OptBind(OUT(c));
		return &inst+1;
	}*/

	OUT(c) = RandomVector(thread, len);
	return &inst+1;
}

static inline Instruction const* semijoin_op(Thread& thread, Instruction const& inst) {
    thread.visible = true;
    DECODE(a); BIND(a);
    DECODE(b); BIND(b);

    // assumes that the two arguments are the same type...
    if(a.type() != b.type())
        return StopDispatch(thread, inst, thread.internStr(
		    std::string("Arguments to semijoin must have the same type\n").c_str()),
            inst.c);
    assert(a.type() == b.type());
    OUT(c) = Semijoin(a, b);

    return &inst+1;
}

//
//    Main interpreter loop 
//
//__attribute__((__noinline__,__noclone__)) 
bool interpret(Thread& thread, Instruction const* pc) {

#ifdef USE_THREADED_INTERPRETER
    #define LABELS_THREADED(name,type,...) (void*)&&name##_label,
    static const void* labels[] = {BYTECODES(LABELS_THREADED)};

    goto *(void*)(labels[pc->bc]);
    #define LABELED_OP(name,type,...) \
        name##_label: \
            { pc = name##_op(thread, *pc); goto *(void*)(labels[pc->bc]); } 
    STANDARD_BYTECODES(LABELED_OP)
    stop_label: { 
        return false; 
    }
    done_label: {
        return true;
    }
#else
    while(pc->bc != ByteCode::done) {
        switch(pc->bc) {
            #define SWITCH_OP(name,type,...) \
            case ByteCode::name: { pc = name##_op(thread, *pc); } break;
            STANDARD_BYTECODES(SWITCH_OP)
            case ByteCode::stop: { 
                return false; 
            } break;
	        case ByteCode::done: { 
                return true;
            }
        };
    }
#endif
}

Value Thread::eval(Code const* code) {
	return eval(code, frame.environment, frame.code->registers);
}

Value Thread::eval(Code const* code, Environment* environment, int64_t resultSlot) {
	uint64_t stackSize = stack.size();
    StackFrame oldFrame = frame;

    Instruction done(ByteCode::done, 0, 0, 0);

	// make room for the result
	Instruction const* run = buildStackFrame(*this, environment, code, resultSlot, &done);
	try {
		bool success = interpret(*this, run);
        if(success) {
            pop();
            if(stackSize != stack.size())
		        _error("Stack was the wrong size at the end of eval");
		    return frame.registers[resultSlot];
        }
        else {
            stack.resize(stackSize);
            frame = oldFrame;
            return Value::Nil();
        }
    } catch(...) {
        std::cout << "Unknown error: " << std::endl;
/*        if(!frame.isPromise && frame.environment->getContext()) {
            std::cout << stack.size() << ": " << stringify(frame.environment->getContext()->call);
        }
        for(int64_t i = stack.size()-1; i > std::max(stackSize, 1ULL); --i) {
            if(!stack[i].isPromise && stack[i].environment->getContext())
                std::cout << i << ": " << stringify(stack[i].environment->getContext()->call);
        }*/
        
        stack.resize(stackSize);
        frame = oldFrame;
	    throw;
    }
}

Value Thread::eval(Promise const& p, int64_t resultSlot) {
    
	uint64_t stackSize = stack.size();
    StackFrame oldFrame = frame;
    
    Instruction done(ByteCode::done, 0, 0, 0);

    Instruction const* run = force(*this, p, NULL, Value::Nil(), resultSlot, &done);
   
    try {
		bool success = interpret(*this, run);
        if(success) {
            if(stackSize != stack.size())
		        _error("Stack was the wrong size at the end of eval");
		    return frame.registers[resultSlot];
        }
        else {
            stack.resize(stackSize);
            frame = oldFrame;
            return Value::Nil();
        }
    } catch(...) {
        std::cout << "Unknown error: " << std::endl;
        
        stack.resize(stackSize);
        frame = oldFrame;
	    throw;
    } 
}


const int64_t Random::primes[100] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43,
47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131,
137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223,
227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311,
313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409,
419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503,
509, 521, 523, 541};

Thread::Thread(State& state, TaskQueue* queue) 
    : state(state)
    , visible(true)
#ifdef EPEE
    , traces(state.epeeEnabled)
#endif
    , random(0)
    , queue(queue)
{
	registers = new Value[DEFAULT_NUM_REGISTERS];
	frame.registers = registers;
}

State::State(uint64_t threads, int64_t argc, char** argv) 
	: verbose(false)
    , epeeEnabled(true)
    , format(State::RiposteFormat)
    , queues(threads) {

    // initialize string table
	#define ENUM_STRING_TABLE(name, str) \
        Strings::name = strings.in(std::string(str));
    STRINGS(ENUM_STRING_TABLE);
   
    // initialize basic environments 
	this->empty = new Environment(1,(Environment*)0);
    this->global = new Environment(1,empty);

    this->apiStack = NULL;

    // intialize arguments list
	arguments = Character(argc);
	for(int64_t i = 0; i < argc; i++) {
		arguments[i] = internStr(std::string(argv[i]));
	}

    promiseCode = new (Code::Finalize) Code();
    promiseCode->bc.push_back(Instruction(ByteCode::force, 2, 0, 2));
    promiseCode->bc.push_back(Instruction(ByteCode::retp, 2, 0, 0));
    promiseCode->registers = 3;
    promiseCode->expression = Value::Nil();
}

Thread* State::getThread() {
    // TODO: assign threads to different task queues
    Thread* r = new Thread(*this, queues.queues[0]);
    threads.push_back(r);
    return r;
}

void State::deleteThread(Thread* s) {
    for(std::list<Thread*>::reverse_iterator i = threads.rbegin();
        i != threads.rend(); ++i) {
        if(*i == s) {
            threads.erase((++i).base());
            break;
        }
    }
    delete s;
}

void Code::printByteCode(State const& state) const {
	std::cout << "Code: " << intToHexStr((int64_t)this) << std::endl;
	std::cout << "\tRegisters: " << registers << std::endl;
	if(constants.size() > 0) {
		std::cout << "\tConstants: " << std::endl;
		for(int64_t i = 0; i < (int64_t)constants.size(); i++)
			std::cout << "\t\t" << i << ":\t" << state.stringify(constants[i]) << std::endl;
	}
	if(bc.size() > 0) {
		std::cout << "\tCode: " << std::endl;
		for(int64_t i = 0; i < (int64_t)bc.size(); i++) {
			std::cout << std::hex << &bc[i] << std::dec << "\t" << i << ":\t" << bc[i].toString();
			if(bc[i].bc == ByteCode::call || bc[i].bc == ByteCode::fastcall) {
				std::cout << "\t\t(arguments: " << calls[bc[i].b].arguments.length() << ")";
			}
			std::cout << std::endl;
		}
	}
	std::cout << std::endl;
}

