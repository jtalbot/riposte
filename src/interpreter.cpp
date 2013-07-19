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

static inline Instruction const* mov_op(Thread& thread, Instruction const& inst) ALWAYS_INLINE;
static inline Instruction const* fastmov_op(Thread& thread, Instruction const& inst) ALWAYS_INLINE;
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
	Heap::Global.collect(thread.state);
	DECODE(a); FORCE(a); BIND(a);
	if(!a.isClosure())
		_error(std::string("Non-function (") + Type::toString(a.type()) + ") as first parameter to call\n");
	
    Closure const& func = (Closure const&)a;
	CompiledCall const& call = thread.frame.prototype->calls[inst.b];

	Environment* fenv = MatchArgs(thread, thread.frame.environment, func, call);
    return buildStackFrame(thread, fenv, func.prototype(), inst.c, &inst+1);
}

static inline Instruction const* fastcall_op(Thread& thread, Instruction const& inst) {
	Heap::Global.collect(thread.state);
	DECODE(a); FORCE(a); BIND(a);
	if(!a.isClosure())
		_error(std::string("Non-function (") + Type::toString(a.type()) + ") as first parameter to call\n");
	
    Closure const& func = (Closure const&)a;
	CompiledCall const& call = thread.frame.prototype->calls[inst.b];
	
	Environment* fenv = FastMatchArgs(thread, thread.frame.environment, func, call);
    return buildStackFrame(thread, fenv, func.prototype(), inst.c, &inst+1);
}

static inline Instruction const* ret_op(Thread& thread, Instruction const& inst) {
	// we can return futures from functions, so don't BIND
	DECODE(a); FORCE(a);
	
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
    
    Value const& onexit = thread.frame.environment->get(Strings::__onexit__);
    if(onexit.isObject()) {
        // TODO: yuck, get rid of the onexittmp variable
        Value& v = thread.frame.environment->insert(Strings::__onexittmp__);
        Promise::Init(v,
            thread.frame.environment,
            Compiler::compilePromise(thread, onexit),
            false);
        // remove it, so we don't call it recursively
        thread.frame.environment->remove(Strings::__onexit__);
		return force(thread, inst, v, 
            thread.frame.environment, Strings::__onexittmp__);
	}
    
	// We can free this environment for reuse
	// as long as we don't return a closure...
	// TODO: but also can't if an assignment to an out of scope variable occurs (<<-, assign) with a value of a closure!
	if(!(a.isClosure() || a.isEnvironment() || a.isList())) {
		thread.traces.KillEnvironment(thread.frame.environment);
	}

    thread.pop();

	thread.traces.LiveEnvironment(thread.frame.environment, a);

	return returnpc;
}

static inline Instruction const* rets_op(Thread& thread, Instruction const& inst) {
	// top-level statements can't return futures, so bind 
	DECODE(a); FORCE(a); BIND(a);	
	
	REGISTER(0) = a;
	thread.pop();
	
	// there should always be a done_op after a rets
	return &inst+1;
}

static inline Instruction const* retp_op(Thread& thread, Instruction const& inst) {
	// we can return futures from promises, so don't BIND
	DECODE(a); FORCE(a);
	
	if(thread.frame.dest > 0) {
		thread.frame.env->insert((String)thread.frame.dest) = a;
	} else {
        assert(thread.frame.env->getContext());
        ((Context*)thread.frame.env->getContext())->dots[-thread.frame.dest].v = a;
	}
	thread.traces.LiveEnvironment(thread.frame.env, a);
	
	Instruction const* returnpc = thread.frame.returnpc;
	thread.pop();
	
	return returnpc;
}

static inline Instruction const* done_op(Thread& thread, Instruction const& inst) {
	return 0;
}

static inline Instruction const* jmp_op(Thread& thread, Instruction const& inst) {
	return &inst+inst.a;
}

static inline Instruction const* jc_op(Thread& thread, Instruction const& inst) {
	DECODE(c);
	if(c.isLogical1()) {
		if(Logical::isTrue(c.c)) return &inst+inst.a;
		else if(Logical::isFalse(c.c)) return &inst+inst.b;
		else _error("NA where TRUE/FALSE needed"); 
	} else if(c.isInteger1()) {
		if(Integer::isNA(c.i)) _error("NA where TRUE/FALSE needed");
		else if(c.i != 0) return &inst + inst.a;
		else return & inst+inst.b;
	} else if(c.isDouble1()) {
		if(Double::isNA(c.d)) _error("NA where TRUE/FALSE needed");
		else if(c.d != 0) return &inst + inst.a;
		else return & inst+inst.b;
	}
	FORCE(c); BIND(c);
	_error("Need single element logical in conditional jump");
}

static inline Instruction const* branch_op(Thread& thread, Instruction const& inst) {
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
	if(index >= 1 && index <= inst.b) {
		return &inst + ((&inst+index)->c);
	} 
	FORCE(a); BIND(a);
	return &inst+1+inst.b;
}

static inline Instruction const* forbegin_op(Thread& thread, Instruction const& inst) {
	// a = loop variable (e.g. i), b = loop vector(e.g. 1:100), c = counter register
	// following instruction is a jmp that contains offset
	Value& b = REGISTER(inst.b);
	if(!b.isVector())
		_error("Invalid for() loop sequence");
	Vector const& v = (Vector const&)b;
	if((int64_t)v.length() <= 0) {
		return &inst+(&inst+1)->a;	// offset is in following JMP, dispatch together
	} else {
		Element2(v, 0, thread.frame.environment->insert((String)inst.a));
		Integer::InitScalar(REGISTER(inst.c), 1);
		Integer::InitScalar(REGISTER(inst.c-1), v.length());
		return &inst+2;			// skip over following JMP
	}
}

static inline Instruction const* forend_op(Thread& thread, Instruction const& inst) {
	Value& counter = REGISTER(inst.c);
	Value& limit = REGISTER(inst.c-1);
	if(__builtin_expect(counter.i < limit.i, true)) {
		Value& b = REGISTER(inst.b);
		Element2(b, counter.i, thread.frame.environment->insert((String)inst.a));
		counter.i++;
		return &inst+(&inst+1)->a;
	} else {
		return &inst+2;			// skip over following JMP
	}
}

static inline Instruction const* mov_op(Thread& thread, Instruction const& inst) {
	DECODE(a); FORCE(a); BIND(a);
	OUT(c) = a;
	return &inst+1;
}

static inline Instruction const* fastmov_op(Thread& thread, Instruction const& inst) {
	DECODE(a); FORCE(a); // fastmov assumes we don't need to bind. So next op better be able to handle a future 
	OUT(c) = a;
	return &inst+1;
}

static inline Instruction const* external_op(Thread& thread, Instruction const& inst) {
	String name = (String)inst.a;
    void* func = NULL;
    for(std::map<std::string,void*>::iterator i = thread.state.handles.begin();
        i != thread.state.handles.end(); ++i) {
        func = dlsym(i->second, name);
        if(func != NULL)
            break;
    }
    if(func == NULL)
        _error(std::string("Can't find external function: ") + name);

    uint64_t nargs = inst.b;
	for(int64_t i = 0; i < nargs; i++) {
		BIND(REGISTER(inst.c-i));
	}
    {
        typedef Value (*Func)(Thread&, Value const*);
        Func f = (Func)func;
        OUT(c) = f(thread, &REGISTER(inst.c));
    }
	return &inst+1;
}

static void* find_function(Thread& thread, String name) {
    static String lastname = Strings::empty;
    static void* lastfunc = NULL;

    void* func = NULL;
    /*if(std::string(name) == std::string(lastname)) {
        func = lastfunc;
    }
    else {*/
        for(std::map<std::string,void*>::iterator i = thread.state.handles.begin();
            i != thread.state.handles.end(); ++i) {
            func = dlsym(i->second, name);
            if(func != NULL)
                break;
        }
        lastfunc = func;
        lastname = name;
    //}

    if(func == NULL)
        _error("Can't find external function");
    
    return func;
}

static inline Instruction const* map_op(Thread& thread, Instruction const& inst) {
    DECODE(a); BIND(a);
    DECODE(b); BIND(b);
    DECODE(c); BIND(c);

    if(!c.isCharacter1())
        _error("External map function name must be a string");
    if(!b.isList())
        _error("External map args must be a list");
    if(!a.isCharacter())
        _error("External map return types must be a character vector");

    OUT(c) = Map(thread, c.s, (List const&)b, (Character const&)a);
    
    return &inst+1;
}

static inline Instruction const* map1_d_op(Thread& thread, Instruction const& inst) {
    DECODE(a); BIND(a);
    DECODE(b); BIND(b);

    if(!a.isCharacter1())
        _error("External map function name must be a string");

    if(!Map1Dispatch< Zip1, UnaryFuncOp, Double >(thread, find_function(thread, a.s), b, OUT(c)))
        _error("Invalid external map function type");
    
    return &inst+1;
}

static inline Instruction const* map1_i_op(Thread& thread, Instruction const& inst) {
    DECODE(a); BIND(a);
    DECODE(b); BIND(b);

    if(!a.isCharacter1())
        _error("External map function name must be a string");

    if(!Map1Dispatch< Zip1, UnaryFuncOp, Integer >(thread, find_function(thread, a.s), b, OUT(c)))
        _error("Invalid external map function type");
    
    return &inst+1;
}

static inline Instruction const* map1_l_op(Thread& thread, Instruction const& inst) {
    DECODE(a); BIND(a);
    DECODE(b); BIND(b);

    if(!a.isCharacter1())
        _error("External map function name must be a string");

    if(!Map1Dispatch< Zip1, UnaryFuncOp, Logical >(thread, find_function(thread, a.s), b, OUT(c)))
        _error("Invalid external map function type");
    
    return &inst+1;
}

static inline Instruction const* map1_c_op(Thread& thread, Instruction const& inst) {
    DECODE(a); BIND(a);
    DECODE(b); BIND(b);

    if(!a.isCharacter1())
        _error("External map function name must be a string");

    if(!Map1Dispatch< Zip1, UnaryFuncOp, Character >(thread, find_function(thread, a.s), b, OUT(c)))
        _error("Invalid external map function type");
    
    return &inst+1;
}

static inline Instruction const* map1_r_op(Thread& thread, Instruction const& inst) {
    DECODE(a); BIND(a);
    DECODE(b); BIND(b);

    if(!a.isCharacter1())
        _error("External map function name must be a string");

    if(!Map1Dispatch< Zip1, UnaryFuncOp, Raw >(thread, find_function(thread, a.s), b, OUT(c)))
        _error("Invalid external map function type");
    
    return &inst+1;
}

static inline Instruction const* map1_g_op(Thread& thread, Instruction const& inst) {
    DECODE(a); BIND(a);
    DECODE(b); BIND(b);

    if(!a.isCharacter1())
        _error("External map function name must be a string");

    if(!Map1Dispatch< Zip1, UnaryFuncOp, List >(thread, find_function(thread, a.s), b, OUT(c)))
        _error("Invalid external map function type");
    
    return &inst+1;
}

static inline Instruction const* map2_d_op(Thread& thread, Instruction const& inst) {
    DECODE(a); BIND(a);
    DECODE(b); BIND(b);
    DECODE(c); BIND(c);

    if(!c.isCharacter1())
        _error("External map function name must be a string");

    if(!Map2Dispatch< BinaryFuncOp, Double >(thread, find_function(thread, c.s), a, b, OUT(c)))
        _error("Invalid external map function type");
    
    return &inst+1;
}

static inline Instruction const* map2_i_op(Thread& thread, Instruction const& inst) {
    DECODE(a); BIND(a);
    DECODE(b); BIND(b);
    DECODE(c); BIND(c);

    if(!c.isCharacter1())
        _error("External map function name must be a string");

    if(!Map2Dispatch< BinaryFuncOp, Integer >(thread, find_function(thread, c.s), a, b, OUT(c)))
        _error("Invalid external map function type");
    
    return &inst+1;
}

static inline Instruction const* map2_l_op(Thread& thread, Instruction const& inst) {
    DECODE(a); BIND(a);
    DECODE(b); BIND(b);
    DECODE(c); BIND(c);

    if(!c.isCharacter1())
        _error("External map function name must be a string");

    if(!Map2Dispatch< BinaryFuncOp, Logical >(thread, find_function(thread, c.s), a, b, OUT(c)))
        _error("Invalid external map function type");
    
    return &inst+1;
}

static inline Instruction const* map2_c_op(Thread& thread, Instruction const& inst) {
    DECODE(a); BIND(a);
    DECODE(b); BIND(b);
    DECODE(c); BIND(c);

    if(!c.isCharacter1())
        _error("External map function name must be a string");

    if(!Map2Dispatch< BinaryFuncOp, Character >(thread, find_function(thread, c.s), a, b, OUT(c)))
        _error("Invalid external map function type");
    
    return &inst+1;
}

static inline Instruction const* map2_r_op(Thread& thread, Instruction const& inst) {
    DECODE(a); BIND(a);
    DECODE(b); BIND(b);
    DECODE(c); BIND(c);

    if(!c.isCharacter1())
        _error("External map function name must be a string");

    if(!Map2Dispatch< BinaryFuncOp, Raw >(thread, find_function(thread, c.s), a, b, OUT(c)))
        _error("Invalid external map function type");
    
    return &inst+1;
}

static inline Instruction const* map2_g_op(Thread& thread, Instruction const& inst) {
    DECODE(a); BIND(a);
    DECODE(b); BIND(b);
    DECODE(c); BIND(c);

    if(!c.isCharacter1())
        _error("External map function name must be a string");

    if(!Map2Dispatch< BinaryFuncOp, List >(thread, find_function(thread, c.s), a, b, OUT(c)))
        _error("Invalid external map function type");
    
    return &inst+1;
}

FoldFuncArgs find_fold_function(Thread& thread, String name) {
    void* init_func = find_function(thread, (std::string(name) + "_init").c_str());
    void* op_func = find_function(thread, (std::string(name) + "_op").c_str());
    void* fini_func = find_function(thread, (std::string(name) + "_fini").c_str());

    FoldFuncArgs funcs;
    funcs.base = init_func;
    funcs.func = op_func;
    funcs.fini = fini_func;

   return funcs; 
}

static inline Instruction const* fold_d_op(Thread& thread, Instruction const& inst) {
    DECODE(a); BIND(a);
    DECODE(b); BIND(b);

    if(!a.isCharacter1())
        _error("External fold function name must be a string");

    FoldFuncArgs funcs = find_fold_function(thread, a.s);

    if(!Map1Dispatch< FoldLeft2, FoldFuncOp, Double >(thread, &funcs, b, OUT(c)))
        _error("Invalid external fold function type");

    return &inst+1;
}

static inline Instruction const* fold_i_op(Thread& thread, Instruction const& inst) {
    DECODE(a); BIND(a);
    DECODE(b); BIND(b);

    if(!a.isCharacter1())
        _error("External fold function name must be a string");

    FoldFuncArgs funcs = find_fold_function(thread, a.s);

    if(!Map1Dispatch< FoldLeft2, FoldFuncOp, Integer >(thread, &funcs, b, OUT(c)))
        _error("Invalid external fold function type");

    return &inst+1;
}

static inline Instruction const* fold_l_op(Thread& thread, Instruction const& inst) {
    DECODE(a); BIND(a);
    DECODE(b); BIND(b);

    if(!a.isCharacter1())
        _error("External fold function name must be a string");

    FoldFuncArgs funcs = find_fold_function(thread, a.s);

    if(!Map1Dispatch< FoldLeft2, FoldFuncOp, Logical >(thread, &funcs, b, OUT(c)))
        _error("Invalid external fold function type");

    return &inst+1;
}

static inline Instruction const* fold_c_op(Thread& thread, Instruction const& inst) {
    DECODE(a); BIND(a);
    DECODE(b); BIND(b);

    if(!a.isCharacter1())
        _error("External fold function name must be a string");

    FoldFuncArgs funcs = find_fold_function(thread, a.s);

    if(!Map1Dispatch< FoldLeft2, FoldFuncOp, Character >(thread, &funcs, b, OUT(c)))
        _error("Invalid external fold function type");

    return &inst+1;
}

static inline Instruction const* fold_r_op(Thread& thread, Instruction const& inst) {
    DECODE(a); BIND(a);
    DECODE(b); BIND(b);

    if(!a.isCharacter1())
        _error("External fold function name must be a string");

    FoldFuncArgs funcs = find_fold_function(thread, a.s);

    if(!Map1Dispatch< FoldLeft2, FoldFuncOp, Raw >(thread, &funcs, b, OUT(c)))
        _error("Invalid external fold function type");

    return &inst+1;
}

static inline Instruction const* fold_g_op(Thread& thread, Instruction const& inst) {
    DECODE(a); BIND(a);
    DECODE(b); BIND(b);

    if(!a.isCharacter1())
        _error("External fold function name must be a string");

    FoldFuncArgs funcs = find_fold_function(thread, a.s);

    if(!Map1Dispatch< FoldLeft2, FoldFuncOp, List >(thread, &funcs, b, OUT(c)))
        _error("Invalid external fold function type");

    return &inst+1;
}

static inline Instruction const* scan_d_op(Thread& thread, Instruction const& inst) {
    DECODE(a); BIND(a);
    DECODE(b); BIND(b);

    if(!a.isCharacter1())
        _error("External scan function name must be a string");

    FoldFuncArgs funcs = find_fold_function(thread, a.s);

    if(!Map1Dispatch< ScanLeft2, FoldFuncOp, Double >(thread, &funcs, b, OUT(c)))
        _error("Invalid external scan function type");

    return &inst+1;
}

static inline Instruction const* scan_i_op(Thread& thread, Instruction const& inst) {
    DECODE(a); BIND(a);
    DECODE(b); BIND(b);

    if(!a.isCharacter1())
        _error("External scan function name must be a string");

    FoldFuncArgs funcs = find_fold_function(thread, a.s);

    if(!Map1Dispatch< ScanLeft2, FoldFuncOp, Integer >(thread, &funcs, b, OUT(c)))
        _error("Invalid external scan function type");

    return &inst+1;
}

static inline Instruction const* scan_l_op(Thread& thread, Instruction const& inst) {
    DECODE(a); BIND(a);
    DECODE(b); BIND(b);

    if(!a.isCharacter1())
        _error("External scan function name must be a string");

    FoldFuncArgs funcs = find_fold_function(thread, a.s);

    if(!Map1Dispatch< ScanLeft2, FoldFuncOp, Logical >(thread, &funcs, b, OUT(c)))
        _error("Invalid external scan function type");

    return &inst+1;
}

static inline Instruction const* scan_c_op(Thread& thread, Instruction const& inst) {
    DECODE(a); BIND(a);
    DECODE(b); BIND(b);

    if(!a.isCharacter1())
        _error("External scan function name must be a string");

    FoldFuncArgs funcs = find_fold_function(thread, a.s);

    if(!Map1Dispatch< ScanLeft2, FoldFuncOp, Character >(thread, &funcs, b, OUT(c)))
        _error("Invalid external scan function type");

    return &inst+1;
}

static inline Instruction const* scan_r_op(Thread& thread, Instruction const& inst) {
    DECODE(a); BIND(a);
    DECODE(b); BIND(b);

    if(!a.isCharacter1())
        _error("External scan function name must be a string");

    FoldFuncArgs funcs = find_fold_function(thread, a.s);

    if(!Map1Dispatch< ScanLeft2, FoldFuncOp, Raw >(thread, &funcs, b, OUT(c)))
        _error("Invalid external scan function type");

    return &inst+1;
}

static inline Instruction const* scan_g_op(Thread& thread, Instruction const& inst) {
    DECODE(a); BIND(a);
    DECODE(b); BIND(b);

    if(!a.isCharacter1())
        _error("External scan function name must be a string");

    FoldFuncArgs funcs = find_fold_function(thread, a.s);

    if(!Map1Dispatch< ScanLeft2, FoldFuncOp, List >(thread, &funcs, b, OUT(c)))
        _error("Invalid external scan function type");

    return &inst+1;
}

// LOAD_STORE_BYTECODES

static inline Instruction const* load_op(Thread& thread, Instruction const& inst) {
	String s = ((Character const&)CONSTANT(inst.a)).s;
	Environment* env;
	Value const& v = thread.frame.environment->getRecursive(s, env);
	if(!v.isObject()) {
		return force(thread, inst, v, env, s);
	}
	else {
		OUT(c) = v;
		return &inst+1;
	}
}

static inline Instruction const* loadfn_op(Thread& thread, Instruction const& inst) {
	String s = ((Character const&)CONSTANT(inst.a)).s;
	Environment* env = thread.frame.environment;

    // Iterate until we find a function
    do {
	    Value const& v = env->getRecursive(s, env);

	    if(!v.isObject()) {
		    return force(thread, inst, v, env, s);
	    }
        else if(v.isClosure()) {
            OUT(c) = v;
            return &inst+1;
        }
        env = env->getParent();
	} while(env != 0);

    _error("loadfn failed to find value");
}

static inline Instruction const* store_op(Thread& thread, Instruction const& inst) {
    String s = ((Character const&)CONSTANT(inst.a)).s; 
	DECODE(c); // don't BIND or FORCE
	thread.frame.environment->insert(s) = c;
	return &inst+1;
}

static inline Instruction const* storeup_op(Thread& thread, Instruction const& inst) {
	// assign2 is always used to assign up at least one scope level...
	// so start off looking up one level...
	assert(thread.frame.environment->getParent() != 0);

    // TODO: just BIND in this scenario instead of tracking liveness?
	
	DECODE(c); /* FORCE(c); BIND(value);*/
	
	String s = ((Character const&)CONSTANT(inst.a)).s;
	Environment* penv;
	Value& dest = thread.frame.environment->getParent()->insertRecursive(s, penv);

	if(!dest.isNil()) {
		dest = c;
		thread.traces.LiveEnvironment(penv, dest);
	}
	else {
		Value& global = thread.state.global->insert(s);
		global = c;
		thread.traces.LiveEnvironment(thread.state.global, global);
	}
	return &inst+1;
}

static inline Instruction const* rm_op(Thread& thread, Instruction const& inst) {
	String s = ((Character const&)CONSTANT(inst.a)).s;
	thread.frame.environment->remove( s );
	OUT(c) = Null::Singleton();
    return &inst+1;
}

static inline Instruction const* dotsv_op(Thread& thread, Instruction const& inst) {
    DECODE(a); FORCE(a); BIND(a);

    int64_t idx = 0;
    if(a.isInteger1())
        idx = ((Integer const&)a)[0] - 1;
    else if(a.isDouble1())
        idx = (int64_t)((Double const&)a)[0] - 1;
    else
        _error(std::string("Invalid type in dotsv: "));

	if(!thread.frame.environment->getContext() ||
       idx >= (int64_t)thread.frame.environment->getContext()->dots.size() ||
       idx < (int64_t)0)
        _error(std::string("The '...' list does not contain ") + intToStr(idx+1) + " elements");
	
    DOTDOT(v, idx); FORCE_DOTDOT(v, idx); // no need to bind since value is in a register
	OUT(c) = v;
	return &inst+1;
}

static inline Instruction const* dotsc_op(Thread& thread, Instruction const& inst) {
    if(!thread.frame.environment->getContext())
        OUT(c) = Integer::c(0);
    else
        OUT(c) = Integer::c((int64_t)thread.frame.environment->getContext()->dots.size());
    return &inst+1;
}

static inline Instruction const* dots_op(Thread& thread, Instruction const& inst) {
	static const PairList empty;
    PairList const& dots = 
        thread.frame.environment->getContext()
            ? thread.frame.environment->getContext()->dots
            : empty;
	
	Value& iter = REGISTER(inst.a);
	Value& out = OUT(c);

	// First time through, make a result vector...
	if(iter.i == 0) {
		Heap::Global.collect(thread.state);
		out = List(dots.size());
		memset(((List&)out).v(), 0, dots.size()*sizeof(List::Element));
	}
	
	if(iter.i < (int64_t)dots.size()) {
		DOTDOT(a, iter.i); FORCE_DOTDOT(a, iter.i); 
		BIND(a); // BIND since we don't yet support futures in lists
		((List&)out)[iter.i] = a;
		iter.i++;
	}
	
	// If we're all done, check to see if we need to add names and then exit
	if(iter.i >= (int64_t)dots.size() ) {
		if(thread.frame.environment->getContext()->named && dots.size() > 0) {
			Character names(dots.size());
			for(int64_t i = 0; i < (int64_t)dots.size(); i++)
				names[i] = dots[i].n;
			Dictionary* d = new Dictionary(1);
			d->insert(Strings::names) = names;
			((Object&)out).attributes(d);
		}
		return &inst+1;
	}
	
	// Loop on this instruction until done.
	return &inst;
}

static inline Instruction const* missing_op(Thread& thread, Instruction const& inst) {
	Value const& c = CONSTANT(inst.a);
    bool missing = true;
    if( c.isCharacter() ) {
        Value const& v = thread.frame.environment->get(c.s);
	    missing = v.isNil() || (v.isPromise() && ((Promise const&)v).isDefault());
    }
    else {
        if(thread.frame.environment->getContext()
            && (c.i-1) < thread.frame.environment->getContext()->dots.size()) {
            Value const& v = thread.frame.environment->getContext()->dots[c.i-1].v;
	        missing = v.isNil() || (v.isPromise() && ((Promise const&)v).isDefault());
        }
    }
	Logical::InitScalar(OUT(c), 
        missing ? Logical::TrueElement : Logical::FalseElement);
	return &inst+1;
}

// STACK_FRAME_BYTECODES
static inline Instruction const* frame_op(Thread& thread, Instruction const& inst) {
    DECODE(a); FORCE(a); BIND(a);
    int64_t index = a.i;

    Environment* env = thread.frame.environment;

    while(index > 0
        && env->getContext()
        && env->getContext()->parent) {
        env = env->getContext()->parent;
        index--;
    }

    if(index == 0) {
        List r(4);
        REnvironment::Init(r[0], env);
        if(env->getContext()) {
            r[1] = env->getContext()->call;
            r[2] = env->getContext()->function;
            r[3] = Integer::c(env->getContext()->nargs);
            r[4] = env->getContext()->onexit;
        }
        else {
            r[1] = Null::Singleton();
            r[2] = Null::Singleton();
            r[3] = Integer::NA();
            r[4] = Null::Singleton();
        }
        OUT(c) = r;
    }
    else {
        _error("not that many frames on the stack");
    }
    return &inst+1;
}

static inline Instruction const* setexit_op(Thread& thread, Instruction const& inst) {
    DECODE(a); BIND(a);

    //if(thread.frame.environment->getContext())
    //    thread.frame.environment->getContext()->onexit = a;

    return &inst+1;
}

// PROMISE_BYTECODES

static inline Instruction const* pr_new_op(Thread& thread, Instruction const& inst) {
    DECODE(a); FORCE(a); BIND(a);
    DECODE(b); FORCE(b); BIND(b);

    REnvironment& eval = (REnvironment&)REGISTER((&inst+1)->a);
    REnvironment& assign = (REnvironment&)REGISTER((&inst+1)->b);

    Value& v = assign.environment()->insert(a.s);
    Promise::Init(v,
        eval.environment(),
        Compiler::compilePromise(thread, b),
        false);

    OUT(c) = Null::Singleton();

    return &inst+2;
}

static inline Instruction const* pr_expr_op(Thread& thread, Instruction const& inst) {
    DECODE(a); FORCE(a); BIND(a);
    DECODE(b); FORCE(b); BIND(b);

    // TODO: check types

    REnvironment const& env = ((REnvironment const&)a);
	String s = ((Character const&)b).s;
	Value v = env.environment()->get(s);
    if(v.isPromise())
        v = ((Promise const&)v).prototype()->expression;
    else if(v.isNil())
        v = Null::Singleton();
	OUT(c) = v;
    return &inst+1;
}

static inline Instruction const* pr_env_op(Thread& thread, Instruction const& inst) {
    DECODE(a); FORCE(a); BIND(a);
    DECODE(b); FORCE(b); BIND(b);

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

static inline Instruction const* type_op(Thread& thread, Instruction const& inst) {
	DECODE(a); FORCE(a);
	switch(thread.traces.futureType(a)) {
        #define CASE(name, str, ...) case Type::name: OUT(c) = Character::c(Strings::name); break;
        TYPES(CASE)
        #undef CASE
        default: _error("Unknown type in type to string, that's bad!"); break;
    }
	return &inst+1;
}

static inline Instruction const* length_op(Thread& thread, Instruction const& inst) {
	DECODE(a); FORCE(a); 
	if(a.isVector())
		Integer::InitScalar(OUT(c), ((Vector const&)a).length());
	else if(a.isFuture()) {
		IRNode::Shape shape = thread.traces.futureShape(a);
		if(shape.split < 0 && shape.filter < 0) {
			Integer::InitScalar(OUT(c), shape.length);
		} else {
			OUT(c) = thread.traces.EmitUnary<CountFold>(thread.frame.environment, IROpCode::length, a, 0);
			thread.traces.OptBind(thread, OUT(c));
		}
	}
	else if(((Object const&)a).hasAttributes()) { 
		return GenericDispatch(thread, inst, Strings::length, a, inst.c); 
	} else {
		Integer::InitScalar(OUT(c), 1);
	}
	return &inst+1;
}

static inline Instruction const* get_op(Thread& thread, Instruction const& inst) {
	DECODE(a); DECODE(b);

    if(GetFast(thread, a, b, OUT(c)))
        return &inst+1;
    else
        return GetSlow( thread, inst, a, b, OUT(c) );
}

static inline Instruction const* set_op(Thread& thread, Instruction const& inst) {
	// a = value, b = index, c = dest
	DECODE(a); DECODE(b); DECODE(c);
	BIND(b);

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

	BIND(a);
	BIND(c);

    if( !((Object const&)c).hasAttributes() ) {
        if(c.isVector() && ( b.isDouble1() || b.isInteger1() )) {
            Subset2Assign(thread, c, true, b, a, OUT(c));
            return &inst+1;
        }
        else if(c.isEnvironment() && b.isCharacter1()) {
	        String s = ((Character const&)b).s;
            ((REnvironment&)c).environment()->insert(s) = a;
            OUT(c) = c;
            return &inst+1;
        }
        else if(c.isClosure() && b.isCharacter1()) {
            //Closure const& f = (Closure const&)c;
	        //String s = ((Character const&)b).s;
            // TODO: implement assignment to function members
            _error("Assignment to function members is not yet implemented");
        }
    }
    return GenericDispatch(thread, inst, Strings::bbAssign, c, b, a, inst.c); 
}

static inline Instruction const* getsub_op(Thread& thread, Instruction const& inst) {
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

	if( thread.traces.isTraceable(a, b) 
		&& thread.traces.futureType(b) == Type::Logical 
		&& thread.traces.futureShape(a) == thread.traces.futureShape(b)) {
		OUT(c) = thread.traces.EmitFilter(thread.frame.environment, a, b);
		thread.traces.OptBind(thread, OUT(c));
		return &inst+1;
	}

	BIND(a);

	if(thread.traces.isTraceable(a, b) 
		&& (thread.traces.futureType(b) == Type::Integer 
			|| thread.traces.futureType(b) == Type::Double)) {
		OUT(c) = thread.traces.EmitGather(thread.frame.environment, a, b);
		thread.traces.OptBind(thread, OUT(c));
		return &inst+1;
	}
	
    BIND(b);

    if(a.isVector() && !((Object const&)a).hasAttributes() &&
        (b.isInteger() || b.isDouble() || b.isLogical()) ) {
        SubsetSlow(thread, a, b, OUT(c)); 
        return &inst+1;
    }

    return GenericDispatch(thread, inst, Strings::bracket, a, b, inst.c); 
}

static inline Instruction const* setsub_op(Thread& thread, Instruction const& inst) {
	// a = value, b = index, c = dest 
	DECODE(a); DECODE(b); DECODE(c); 
    BIND(b); BIND(c);
	
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

	BIND(a);

    if( !((Object const&)c).hasAttributes() ) {
        if(c.isVector() && ( b.isDouble() || b.isInteger() || b.isLogical() )) {
            SubsetAssign(thread, c, true, b, a, OUT(c));
            return &inst+1;
        }
	}

	return GenericDispatch(thread, inst, Strings::bracketAssign, c, b, a, inst.c); 
}

static inline Instruction const* getenv_op(Thread& thread, Instruction const& inst) {
    DECODE(a); FORCE(a); BIND(a);

    if(a.isEnvironment()) {
        Environment* parent = ((REnvironment const&)a).environment()->getParent();
        if(parent == 0)
            _error("environment does not have a parent");
        REnvironment::Init(OUT(c), parent);
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
    DECODE(a); FORCE(a); BIND(a);
    DECODE(b); FORCE(b); BIND(b);

    if(!b.isEnvironment())
        _error("replacement object is not an environment");

    Environment* value = ((REnvironment const&)b).environment();

    if(a.isEnvironment()) {
        Environment* target = ((REnvironment const&)a).environment();
        
        // Riposte allows parent environment replacement,
        // but requires that no loops be introduced in the environment chain.
        Environment* p = value;
        while(p) {
            if(p == target) 
                _error("an environment cannot be its own ancestor");
            p = p->getParent();
        }
        
        ((REnvironment const&)a).environment()->setParent(((REnvironment const&)b).environment());
        OUT(c) = a;
    }
    else if(a.isClosure()) {
        Closure::Init(OUT(c), ((Closure const&)a).prototype(), value);
    }
    else {
        _error("target of assignment does not have an enclosing environment");
    }
    return &inst+1;
}

static inline Instruction const* getattr_op(Thread& thread, Instruction const& inst) {
	DECODE(a); FORCE(a);
	DECODE(b); FORCE(b); BIND(b);
	if(a.isObject() && b.isCharacter1()) {
		String name = ((Character const&)b)[0];
		Object const& o = (Object const&)a;
		if(o.hasAttributes() && o.attributes()->has(name))
			OUT(c) = o.attributes()->get(name);
		else
			OUT(c) = Null::Singleton();
		return &inst+1;
	}
	_error("Invalid attrget operation");
}

static inline Instruction const* setattr_op(Thread& thread, Instruction const& inst) {
	DECODE(c); FORCE(c);
	DECODE(b); FORCE(b); BIND(b);
	DECODE(a); FORCE(a); BIND(a);
	if(c.isObject() && b.isCharacter1()) {
		String name = ((Character const&)b)[0];
		Object o = (Object const&)c;
        if(a.isNull()) {
            if(o.hasAttributes()
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
		    Dictionary* d = o.hasAttributes()
                    	? o.attributes()->clone(1)
                    	: new Dictionary(1);
		    d->insert(name) = a;
		    o.attributes(d);
        }
		OUT(c) = o;
		return &inst+1;
	}
	_error("Invalid attrset operation");
}

static inline Instruction const* attributes_op(Thread& thread, Instruction const& inst) {
    DECODE(a); FORCE(a);
    if(a.isObject()) {
        Object o = (Object const&)a;
        if(!o.hasAttributes() || o.attributes()->Size() == 0) {
            OUT(c) = Null::Singleton();
        }
        else {
            Character n(o.attributes()->Size());
            List v(o.attributes()->Size());
            int64_t j = 0;
            for(Dictionary::const_iterator i = o.attributes()->begin();
                    i != o.attributes()->end(); 
                    ++i, ++j) {
                n[j] = i.string();
                v[j] = i.value();
            }
			Dictionary* d = new Dictionary(1);
			d->insert(Strings::names) = n;
			((Object&)v).attributes(d);
            OUT(c) = v;
        }
    }
    else {
        OUT(c) = Null::Singleton();
    }
    return &inst+1;
}

static inline Instruction const* strip_op(Thread& thread, Instruction const& inst) {
	DECODE(a); FORCE(a);
	Value& c = OUT(c);
	c = a;
	((Object&)c).attributes(0);
	return &inst+1;
}

static inline Instruction const* as_op(Thread& thread, Instruction const& inst) {
	DECODE(a); FORCE(a); BIND(a);
	String type = ((Character const&)CONSTANT(inst.b)).s;
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
        _error("as not yet defined for this type");

    // TODO: extend to work on e.g. environment->list and list->environment, etc.
    // Add support for futures
	return &inst+1; 
}

/*static inline Instruction const* env_ls_op(Thread& thread, Instruction const& inst) {
    DECODE(a); FORCE(a); BIND(a);

    if(!a.isEnvironment())
        _error("invalid 'envir' argument");

    Environment const* env = ((REnvironment&)a).environment();
	
    Character result(env->Size());

    Environment::const_iterator i = env->begin();
    uint64_t j = 0;
    for(; i != env->end(); ++i, ++j) {
        result[j] = i.string();
    }

    OUT(c) = result;

    return &inst+1;
}*/

// ENVIRONMENT_BYTECODES

static inline Instruction const* env_new_op(Thread& thread, Instruction const& inst) {
    DECODE(a); FORCE(a); BIND(a);
    DECODE(b); FORCE(b); BIND(b);

    if(!a.isEnvironment())
        _error("'enclos' must be an environment");

    REnvironment::Init(OUT(c), new Environment(4,((REnvironment const&)a).environment()));
    return &inst+1;
}

static inline Instruction const* env_names_op(Thread& thread, Instruction const& inst) {
    DECODE(a); FORCE(a); BIND(a);

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

static inline Instruction const* env_exists_op(Thread& thread, Instruction const& inst) {
    DECODE(a); FORCE(a); BIND(a);
    DECODE(b); FORCE(b); BIND(b);

    if(!a.isEnvironment())
        _error("invalid 'envir' argument");

    if(!b.isCharacter() || b.pac != 1)
        _error("invalid exists argument");

    OUT(c) = ((REnvironment const&)a).environment()->has(((Character const&)b).s)
                ? Logical::True() : Logical::False();
    return &inst+1;
}

// TODO: appropriately generalize
static inline Instruction const* env_remove_op(Thread& thread, Instruction const& inst) {
    DECODE(a); FORCE(a); BIND(a);
    DECODE(b); FORCE(b); BIND(b);

    if(!a.isEnvironment())
        _error("invalid 'envir' argument");

    if(!b.isCharacter1())
        _error("invalid remove argument");

	((REnvironment const&)a).environment()->remove( ((Character const&)b).s );
	OUT(c) = Null::Singleton();
    return &inst+1;
}

static inline Instruction const* env_global_op(Thread& thread, Instruction const& inst) {
    REnvironment::Init(OUT(c), thread.state.global);
    return &inst+1;
}

// FUNCTION_BYTECODES

static inline Instruction const* fn_new_op(Thread& thread, Instruction const& inst) {
	Value const& function = CONSTANT(inst.a);
	Value& out = OUT(c);
	Closure::Init(out, ((Closure const&)function).prototype(), thread.frame.environment);
	return &inst+1;
}


// VECTOR BYTECODES

#define OP(Name, string, Group, Func) \
static inline Instruction const* Name##_op(Thread& thread, Instruction const& inst) { \
	DECODE(a);	\
    if( Group##Fast<Name##VOp>( thread, NULL, a, OUT(c) ) ) \
        return &inst+1; \
    else \
        return Name##Slow( thread, inst, NULL, a, OUT(c) ); \
}
UNARY_FOLD_SCAN_BYTECODES(OP)
#undef OP

#define OP(Name, string, Group, Func) \
static inline Instruction const* Name##_op(Thread& thread, Instruction const& inst) { \
	DECODE(a);	\
	DECODE(b);	\
    if( Group##Fast<Name##VOp>( thread, NULL, a, b, OUT(c) ) ) \
        return &inst+1; \
    else \
        return Name##Slow( thread, inst, NULL, a, b, OUT(c) ); \
}
BINARY_BYTECODES(OP)
#undef OP

static inline Instruction const* ifelse_op(Thread& thread, Instruction const& inst) {
	DECODE(a); FORCE(a);
	DECODE(b); FORCE(b);
	DECODE(c); FORCE(c);
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
	if(thread.traces.isTraceable<IfElse>(a,b,c)) {
		OUT(c) = thread.traces.EmitIfElse(thread.frame.environment, a, b, c);
		thread.traces.OptBind(thread, OUT(c));
		return &inst+1;
	}
	BIND(a); BIND(b); BIND(c);

    IfElseDispatch(thread, NULL, b, a, c, OUT(c));
	return &inst+1; 
}

static inline Instruction const* split_op(Thread& thread, Instruction const& inst) {
	DECODE(a); FORCE(a); BIND(a);
	DECODE(b); FORCE(b);
	DECODE(c); FORCE(c);
	int64_t levels = As<Integer>(thread, a)[0];
	if(thread.traces.isTraceable<Split>(b,c)) {
		OUT(c) = thread.traces.EmitSplit(thread.frame.environment, c, b, levels);
		thread.traces.OptBind(thread, OUT(c));
		return &inst+1;
	}
	BIND(a); BIND(b); BIND(c);

	_error("split not defined in scalar yet");
	return &inst+1; 
}

static inline Instruction const* vector_op(Thread& thread, Instruction const& inst) {
	DECODE(a); FORCE(a); BIND(a);
	DECODE(b); FORCE(b); BIND(b);
	Type::Enum type = string2Type( As<Character>(thread, a)[0] );
	int64_t l = As<Integer>(thread, b)[0];
	
	// TODO: replace with isTraceable...
	if(thread.state.epeeEnabled 
		&& (type == Type::Double || type == Type::Integer || type == Type::Logical)
		&& l >= TRACE_VECTOR_WIDTH) {
		OUT(c) = thread.traces.EmitConstant(thread.frame.environment, type, l, 0);
		thread.traces.OptBind(thread, OUT(c));
		return &inst+1;
	}

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
		_error("Invalid type in vector");
	} 
	return &inst+1;
}

static inline Instruction const* seq_op(Thread& thread, Instruction const& inst) {
    DECODE(a); BIND(a);

    int64_t len = As<Integer>(thread, a)[0];

    if(len >= TRACE_VECTOR_WIDTH) {
        OUT(c) = thread.traces.EmitSequence(thread.frame.environment, len, 1LL, 1LL);
        thread.traces.OptBind(thread, OUT(c));
        return &inst+1;
    }

    OUT(c) = Sequence(1LL, 1LL, len);
    return &inst+1;
}

static inline Instruction const* index_op(Thread& thread, Instruction const& inst) {
	// c = n, b = each, a = length
	DECODE(a); FORCE(a); BIND(a);
	DECODE(b); FORCE(b); BIND(b);
	DECODE(c); FORCE(c); BIND(c);

	int64_t n = As<Integer>(thread, c)[0];
	int64_t each = As<Integer>(thread, b)[0];
	int64_t len = As<Integer>(thread, a)[0];
	
	if(len >= TRACE_VECTOR_WIDTH) {
		OUT(c) = thread.traces.EmitIndex(thread.frame.environment, len, (int64_t)n, (int64_t)each);
		thread.traces.OptBind(thread, OUT(c));
		return &inst+1;
	}

	OUT(c) = Repeat((int64_t)n, (int64_t)each, len);
	return &inst+1;
}

static inline Instruction const* random_op(Thread& thread, Instruction const& inst) {
	DECODE(a); FORCE(a); BIND(a);

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
    DECODE(a); FORCE(a); BIND(a);
    DECODE(b); FORCE(b); BIND(b);

    // assumes that the two arguments are the same type...
    assert(a.type() == b.type());
    OUT(c) = Semijoin(a, b);

    return &inst+1;
}

//
//    Main interpreter loop 
//
//__attribute__((__noinline__,__noclone__)) 
void interpret(Thread& thread, Instruction const* pc) {

#ifdef USE_THREADED_INTERPRETER
    	#define LABELS_THREADED(name,type,...) (void*)&&name##_label,
	static const void* labels[] = {BYTECODES(LABELS_THREADED)};

	goto *(void*)(labels[pc->bc]);
	#define LABELED_OP(name,type,...) \
		name##_label: \
			{ pc = name##_op(thread, *pc); goto *(void*)(labels[pc->bc]); } 
	STANDARD_BYTECODES(LABELED_OP)
	done_label: {}
#else
	while(pc->bc != ByteCode::done) {
		switch(pc->bc) {
			#define SWITCH_OP(name,type,...) \
				case ByteCode::name: { pc = name##_op(thread, *pc); } break;
			BYTECODES(SWITCH_OP)
		};
	}
#endif

}

void State::interpreter_init(Thread& thread) {
	// nothing for now
}

Value Thread::eval(Prototype const* prototype) {
	return eval(prototype, frame.environment, frame.prototype->registers);
}

Value Thread::eval(Prototype const* prototype, Environment* environment, int64_t resultSlot) {
	uint64_t stackSize = stack.size();
    StackFrame oldFrame = frame;

	// make room for the result
	Instruction const* run = buildStackFrame(*this, environment, prototype, -resultSlot, (Instruction const*)0);
	try {
		interpret(*this, run);
        if(stackSize != stack.size())
		    _error("Stack was the wrong size at the end of eval");
		return frame.registers[resultSlot];
	} catch(...) {
        if(!frame.isPromise && stack.size() > 1) {
            std::cout << stack.size() << ": " << stringify(frame.environment->getContext()->call);
        }
        for(int64_t i = stack.size()-1; i > std::max(stackSize, 1ULL); --i) {
            if(!stack[i].isPromise)
                std::cout << i << ": " << stringify(stack[i].environment->getContext()->call);
        }
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

Thread::Thread(State& state, uint64_t index) 
    : state(state)
    , index(index)
#ifdef EPEE
    , traces(state.epeeEnabled)
#endif
    , random(index) 
    , steals(1)
{
	registers = new Value[DEFAULT_NUM_REGISTERS];
	frame.registers = registers;
}

void Prototype::printByteCode(Prototype const* prototype, State const& state) {
	std::cout << "Prototype: " << intToHexStr((int64_t)prototype) << std::endl;
	std::cout << "\tRegisters: " << prototype->registers << std::endl;
	if(prototype->constants.size() > 0) {
		std::cout << "\tConstants: " << std::endl;
		for(int64_t i = 0; i < (int64_t)prototype->constants.size(); i++)
			std::cout << "\t\t" << i << ":\t" << state.stringify(prototype->constants[i]) << std::endl;
	}
	if(prototype->bc.size() > 0) {
		std::cout << "\tCode: " << std::endl;
		for(int64_t i = 0; i < (int64_t)prototype->bc.size(); i++) {
			std::cout << std::hex << &prototype->bc[i] << std::dec << "\t" << i << ":\t" << prototype->bc[i].toString();
			if(prototype->bc[i].bc == ByteCode::call || prototype->bc[i].bc == ByteCode::fastcall) {
				std::cout << "\t\t(arguments: " << prototype->calls[prototype->bc[i].b].arguments.size() << ")";
			}
			std::cout << std::endl;
		}
	}
	std::cout << std::endl;
}


