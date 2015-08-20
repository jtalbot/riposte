/*
    Riposte

    Instruction handlers

    To help out the compiler, instruction handlers are split
    into a fast-path that should be inlined into the interpreter
    (defined in this file) and a slow-path impl function that
    will not be inlined.
*/

#pragma once


#include "bc.h"
#include "call.h"
#include "interpreter.h"
#include "type.h"
#include "value.h"


// Forward define impls for all instructions
#define DEFINE_IMPL(name,...) \
    Instruction const* name##_impl(State& state, Instruction const& inst);
STANDARD_BYTECODES(DEFINE_IMPL);


// Common instruction decoding code
#define REGISTER(i) (*(state.frame.registers+(i)))
#define CONSTANT(i) (state.frame.code->constants[-1-(i)])
#define OUT(X)      (*(state.frame.registers+(inst.X)))

// Most instructions can take either registers or constants
// as arguments. They are distinguished by the sign of the argument.
#define DECODE(X) \
Value const& X = \
	__builtin_expect((inst.X) >= 0, true) \
		? *(state.frame.registers+(inst.X)) \
	    : state.frame.code->constants[-1-(inst.X)];

// Epee's deferred evaluation approach requires us to "bind" futures
// when we can no longer defer evaluation. This causes them to be evaluated.
#ifdef EPEE
#define BIND(X) \
if(__builtin_expect(X.isFuture(), false)) { \
	state.traces.Bind(state,X); \
	return &inst; \
}
#else
#define BIND(X)
#endif


// Some utility functions for dealing with interned strings
ALWAYS_INLINE
Value const* internAndGet(State const& state, Dictionary const* d, String s)
{
    // All environment names are already interned.
    // So if we can't find an interned version of s,
    // it doesn't exist in the environment and there
    // is nothing to get.
    String i = state.global.strings.get(s->s);
    return i ? d->get(i) : nullptr;
}

ALWAYS_INLINE
Value* internAndGetRec(State const& state, Environment* e, String s, Environment*& out)
{
    // All environment names are already interned.
    // So if we can't find an interned version of s,
    // it doesn't exist in the environment and there
    // is nothing to get.
    String i = state.global.strings.get(s->s);
    out = nullptr;
    return i ? e->getRecursive(i, out) : nullptr;
}

ALWAYS_INLINE
void internAndRemove(State const& state, Dictionary* d, String s)
{
    // All environment names are already interned.
    // So if we can't find an interned version of s,
    // it doesn't exist in the environment and there
    // is nothing to remove.
    String name = state.global.strings.get(s->s);
    if(name)
        d->remove(name);
}

// Helper function to trigger forcing of promises
// without recursion on the interpreter.
Instruction const* force(
    State& state, Promise const& p,
    Environment* targetEnv, Value targetIndex,
    int64_t outRegister, Instruction const* returnpc);


// ---Instruction handlers---


// CONTROL_FLOW_BYTECODES 

ALWAYS_INLINE
Instruction const* call_inst(State& state, Instruction const& inst)
{
    return call_impl(state, inst);
}


ALWAYS_INLINE
Instruction const* ret_inst(State& state, Instruction const& inst)
{
    return ret_impl(state, inst);
}


ALWAYS_INLINE
Instruction const* jmp_inst(State& state, Instruction const& inst)
{
    return &inst+inst.a;
}


ALWAYS_INLINE
Instruction const* jc_inst(State& state, Instruction const& inst)
{
    DECODE(c);

    if(c.isLogical1()) {
        if(Logical::isTrue(c.c))
            return &inst+inst.a;
        else if(Logical::isFalse(c.c))
            return &inst+inst.b;
        else if((&inst+1)->bc != ByteCode::stop)
            return &inst+1;
    }

    return jc_impl(state, inst);
}


ALWAYS_INLINE
Instruction const* forbegin_inst(State& state, Instruction const& inst)
{
    return forbegin_impl(state, inst);
}


ALWAYS_INLINE
Instruction const* forend_inst(State& state, Instruction const& inst)
{
    // The forend instruction takes up two instruction slots.
    // The second slot has the jump offset.
    Value& counter = REGISTER(inst.c);
    Value& limit = REGISTER(inst.c+1);

    if(__builtin_expect(counter.i < limit.i, true))
    {
        String i = static_cast<Character const&>(CONSTANT(inst.a)).s;
        Value const& b = REGISTER(inst.b);
        Element2(b, counter.i, state.frame.environment->insert(i));
        counter.i++;
        return &inst+(&inst+1)->a;
    }
    else
    {
        return &inst+2;            // skip over following JMP
    }
}


ALWAYS_INLINE
Instruction const* mov_inst(State& state, Instruction const& inst)
{
    DECODE(a);
    OUT(c) = a;
    return &inst+1;
}


ALWAYS_INLINE
Instruction const* invisible_inst(State& state, Instruction const& inst)
{
    state.visible = false;
    DECODE(a);
    OUT(c) = a;
    return &inst+1;
}


ALWAYS_INLINE
Instruction const* visible_inst(State& state, Instruction const& inst)
{
    state.visible = true;
    DECODE(a);
    OUT(c) = a;
    return &inst+1;
}


ALWAYS_INLINE
Instruction const* withVisible_inst(State& state, Instruction const& inst)
{
    DECODE(a);
    List result(2);
    result[0] = a;
    result[1] = state.visible ? Logical::True() : Logical::False();
    OUT(c) = result;
    return &inst+1;
}


ALWAYS_INLINE
Instruction const* external_inst(State& state, Instruction const& inst)
{
    return external_impl(state, inst);
}


ALWAYS_INLINE
Instruction const* map_inst(State& state, Instruction const& inst)
{
    return map_impl(state, inst);
}


ALWAYS_INLINE
Instruction const* scan_inst(State& state, Instruction const& inst)
{
    return scan_impl(state, inst);
}


ALWAYS_INLINE
Instruction const* fold_inst(State& state, Instruction const& inst)
{
    return fold_impl(state, inst);
}

// LOAD_STORE_BYTECODES

ALWAYS_INLINE
Instruction const* load_inst(State& state, Instruction const& inst)
{
    state.visible = true;
    String s = static_cast<Character const&>(CONSTANT(inst.a)).s;

    Environment* env;
    Value const* v = state.frame.environment->getRecursive(s, env);

    if(v)
    {
        if(!v->isPromise())
        {
            OUT(c) = *v;
            return &inst+1;
        }
        else
        {
            return force(state, static_cast<Promise const&>(*v),
                env, static_cast<Character const&>(CONSTANT(inst.a)),
                inst.c, &inst+1);
        }
    }

    return load_impl(state, inst);
}


ALWAYS_INLINE
Instruction const* loadfn_inst(State& state, Instruction const& inst)
{
    state.visible = true;
    String s = static_cast<Character const&>(CONSTANT(inst.a)).s;

    Environment* env = state.frame.environment;

    // Iterate until we find a function
    do
    {
        Value* v = env->getRecursive(s, env);

        if(v)
        {
            if(v->isClosure())
            {
                OUT(c) = *v;
                return &inst+1;
            }
            else if(v->isPromise())
            {
                // Must return to this instruction to check if it's a function.
                return force(state, static_cast<Promise const&>(*v), 
                    env, static_cast<Character const&>(CONSTANT(inst.a)),
                    inst.c, &inst);
            }
            else
            {
                env = env->getEnclosure();
            }
        }

    } while(env != 0);

    return loadfn_impl(state, inst);
}


ALWAYS_INLINE
Instruction const* store_inst(State& state, Instruction const& inst)
{
    state.visible = false;
    String s = static_cast<Character const&>(CONSTANT(inst.a)).s; 
    DECODE(c); // don't BIND

    state.frame.environment->insert(s) = c;

    return &inst+1;
}


ALWAYS_INLINE
Instruction const* storeup_inst(State& state, Instruction const& inst)
{
    state.visible = false;
    String s = static_cast<Character const&>(CONSTANT(inst.a)).s;
    DECODE(c); BIND(c);

    // assign2 is always used to assign up at least one scope level...
    // so start off looking up one level...
    Environment* up = state.frame.environment->getEnclosure();
    assert(up != 0);

    Environment* penv;
    Value* dest = up->getRecursive(s, penv);

    if(dest)
        *dest = c;
    else
        state.global.global->insert(s) = c;

    return &inst+1;
}


ALWAYS_INLINE
Instruction const* force_inst(State& state, Instruction const& inst)
{
    Value const* dots = state.frame.environment->get(Strings::__dots__);

    if(dots && dots->isList())
    {
        Value const& a = REGISTER(inst.a);
        Value const& t = static_cast<List const&>(*dots)[a.i];

        if(t.isObject())
        {
            OUT(c) = t;
            return &inst+1;
        }
        else if(t.isPromise())
        {
            return force(state, static_cast<Promise const&>(t),
                    state.frame.environment, a,
                    inst.c, &inst+1);
        }
    }

    _internalError("Unexpected operand type in force_inst");
}


ALWAYS_INLINE
Instruction const* dotsv_inst(State& state, Instruction const& inst)
{
    return dotsv_impl(state, inst);
}


ALWAYS_INLINE
Instruction const* dotsc_inst(State& state, Instruction const& inst)
{
    state.visible = true;
    Value const* dots = state.frame.environment->get(Strings::__dots__);

    OUT(c) = Integer::c(dots && dots->isList()
        ? static_cast<List const&>(*dots).length()
        : 0);

    return &inst+1;
}


ALWAYS_INLINE
Instruction const* dots_inst(State& state, Instruction const& inst)
{
    return dots_impl(state, inst);
}


// STACK_FRAME_BYTECODES

ALWAYS_INLINE
Instruction const* frame_inst(State& state, Instruction const& inst)
{
    return frame_impl(state, inst);
}


// PROMISE_BYTECODES

ALWAYS_INLINE
Instruction const* pr_new_inst(State& state, Instruction const& inst)
{
    return pr_new_impl(state, inst);
}


ALWAYS_INLINE
Instruction const* pr_expr_inst(State& state, Instruction const& inst)
{
    return pr_expr_impl(state, inst);
}


ALWAYS_INLINE
Instruction const* pr_env_inst(State& state, Instruction const& inst)
{
    return pr_env_inst(state, inst);
}


// OBJECT_BYTECODES

ALWAYS_INLINE
Instruction const* id_inst(State& state, Instruction const& inst)
{
    state.visible = true;
    DECODE(a);
    DECODE(b);

    OUT(c) = Id(a,b) ? Logical::True() : Logical::False();
    return &inst+1;
}


ALWAYS_INLINE
Instruction const* nid_inst(State& state, Instruction const& inst)
{
    state.visible = true;
    DECODE(a);
    DECODE(b);

    OUT(c) = Id(a,b) ? Logical::False() : Logical::True();
    return &inst+1;
}


ALWAYS_INLINE
Instruction const* isnil_inst(State& state, Instruction const& inst)
{
    state.visible = true;
    DECODE(a);
    OUT(c) = a.isNil() ? Logical::True() : Logical::False();
    return &inst+1;
}


ALWAYS_INLINE
Instruction const* type_inst(State& state, Instruction const& inst)
{
    return type_impl(state, inst);
}


ALWAYS_INLINE
Instruction const* length_inst(State& state, Instruction const& inst)
{
    state.visible = true;
    DECODE(a);

    if(a.isVector()) {
        Integer::InitScalar(OUT(c), static_cast<Vector const&>(a).length());
        return &inst+1;
    }

    return length_impl(state, inst);
}


ALWAYS_INLINE
Instruction const* get_inst(State& state, Instruction const& inst)
{
    state.visible = true;
    DECODE(a); DECODE(b);

    if(a.isVector() && !static_cast<Object const&>(a).hasAttributes())
    {
        auto v = static_cast<Vector const&>(a);
		if(    b.isInteger() 
            && static_cast<Integer const&>(b).length() == 1
            && !Integer::isNA(b.i)
            && (b.i-1) >= 0 
            && (b.i-1) < v.length() ) {
            Element2(v, b.i-1, OUT(c));
            return &inst+1;
        }
		else if(b.isDouble() 
            && static_cast<Double const&>(b).length() == 1
            && !Double::isNA(b.d)
            && (b.d-1) >= 0
            && (b.d-1) < v.length()) {
            Element2(a, (int64_t)b.d-1, OUT(c));
            return &inst+1;
        }
	}
    else if(a.isEnvironment())
    {
        auto env = static_cast<REnvironment const&>(a);

        if( b.isCharacter() && b.pac == 1 )
        {
	        String s = static_cast<Character const&>(b).s;

            Value const* v = internAndGet(state, env.environment(), s);
            if(v && v->isObject()) {
                OUT(c) = *v;
                return &inst+1;
            }
        }
    }

    return get_impl(state, inst);
}


ALWAYS_INLINE
Instruction const* set_inst(State& state, Instruction const& inst)
{
    return set_impl(state, inst);
}


ALWAYS_INLINE
Instruction const* getsub_inst(State& state, Instruction const& inst)
{
    return getsub_impl(state, inst);
}


ALWAYS_INLINE
Instruction const* setsub_inst(State& state, Instruction const& inst)
{
    return setsub_impl(state, inst);
}


ALWAYS_INLINE
Instruction const* getenv_inst(State& state, Instruction const& inst)
{
    state.visible = true;
    DECODE(a); BIND(a);

    if(a.isEnvironment()) {
        Environment* enc = static_cast<REnvironment const&>(a).environment()->getEnclosure();
        if(enc == 0)
            _error("environment does not have an enclosing environment");
        REnvironment::Init(OUT(c), enc);
    }
    else if(a.isClosure()) {
        REnvironment::Init(OUT(c), static_cast<Closure const&>(a).environment());
    }
    else if(a.isNull()) {
        REnvironment::Init(OUT(c), state.frame.environment);
    }
    else {
        OUT(c) = Null::Singleton();
    }
    return &inst+1;
}


ALWAYS_INLINE
Instruction const* setenv_inst(State& state, Instruction const& inst)
{
    return setenv_impl(state, inst);
}


ALWAYS_INLINE
Instruction const* getattr_inst(State& state, Instruction const& inst)
{
    return getattr_impl(state, inst);
}


ALWAYS_INLINE
Instruction const* setattr_inst(State& state, Instruction const& inst)
{
    return setattr_impl(state, inst);
}


ALWAYS_INLINE
Instruction const* attributes_inst(State& state, Instruction const& inst)
{
    return attributes_impl(state, inst);
}


ALWAYS_INLINE
Instruction const* strip_inst(State& state, Instruction const& inst)
{
    state.visible = true;
    DECODE(a);

    Value& c = OUT(c);
    c = a;
    ((Object&)c).attributes(0);

    return &inst+1;
}


ALWAYS_INLINE
Instruction const* as_inst(State& state, Instruction const& inst)
{
    return as_impl(state, inst);
}

// ENVIRONMENT_BYTECODES

ALWAYS_INLINE
Instruction const* env_new_inst(State& state, Instruction const& inst)
{
    return env_new_impl(state, inst);
}


ALWAYS_INLINE
Instruction const* env_names_inst(State& state, Instruction const& inst)
{
    return env_names_impl(state, inst);
}


ALWAYS_INLINE
Instruction const* env_has_inst(State& state, Instruction const& inst)
{
    state.visible = true;
    DECODE(a);
    DECODE(b);

    if(a.isEnvironment() && b.isCharacter() && b.pac == 1)
    {
        auto env = static_cast<REnvironment const&>(a);
        auto str = static_cast<Character const&>(b);

        Value const* v = internAndGet(state, env.environment(), str.s);
        OUT(c) = v ? Logical::True() : Logical::False();

        return &inst+1;
    }

    return env_has_impl(state, inst);
}


ALWAYS_INLINE
Instruction const* env_rm_inst(State& state, Instruction const& inst)
{
    DECODE(a);
    DECODE(b);

    if(a.isEnvironment() && b.isCharacter() && b.pac == 1)
    {
        internAndRemove(state, static_cast<REnvironment const&>(a).environment(), b.s);
        return &inst+1;
    }
   
    return env_rm_impl(state, inst);
}


ALWAYS_INLINE
Instruction const* env_missing_inst(State& state, Instruction const& inst)
{
    return env_missing_impl(state, inst);
}


ALWAYS_INLINE
Instruction const* env_global_inst(State& state, Instruction const& inst)
{
    state.visible = true;
    REnvironment::Init(OUT(c), state.global.global);
    return &inst+1;
}

// FUNCTION_BYTECODES

ALWAYS_INLINE
Instruction const* fn_new_inst(State& state, Instruction const& inst)
{
    state.visible = true;

    Value const& function = CONSTANT(inst.a);
    Closure::Init(OUT(c),
        static_cast<Closure const&>(function).prototype(),
        state.frame.environment);

    return &inst+1;
}


// VECTOR BYTECODES

#define OP(Name, string, Group, Func) \
ALWAYS_INLINE \
Instruction const* Name##_inst(State& state, Instruction const& inst) \
{ \
    state.visible = true; \
    DECODE(a);    \
    if( Group##Fast<Name##VOp>( state, NULL, a, OUT(c) ) ) \
        return &inst+1; \
    \
    return Name##_impl(state, inst); \
}
UNARY_FOLD_SCAN_BYTECODES(OP)
#undef OP

#define OP(Name, string, Group, Func) \
ALWAYS_INLINE \
Instruction const* Name##_inst(State& state, Instruction const& inst) \
{ \
    state.visible = true; \
    DECODE(a);    \
    DECODE(b);    \
    if( Group##Fast<Name##VOp>( state, NULL, a, b, OUT(c) ) ) \
        return &inst+1; \
    \
    return Name##_impl(state, inst); \
}
BINARY_BYTECODES(OP)
#undef OP

ALWAYS_INLINE
Instruction const* ifelse_inst(State& state, Instruction const& inst)
{
    return ifelse_impl(state, inst);
}


ALWAYS_INLINE
Instruction const* split_inst(State& state, Instruction const& inst)
{
    return split_impl(state, inst);
}


ALWAYS_INLINE
Instruction const* vector_inst(State& state, Instruction const& inst)
{
    return vector_impl(state, inst);
}


ALWAYS_INLINE
Instruction const* seq_inst(State& state, Instruction const& inst)
{
    return seq_impl(state, inst);
}


ALWAYS_INLINE
Instruction const* index_inst(State& state, Instruction const& inst)
{
    return index_impl(state, inst);
}


ALWAYS_INLINE
Instruction const* random_inst(State& state, Instruction const& inst)
{
    return random_impl(state, inst);
}


ALWAYS_INLINE
Instruction const* semijoin_inst(State& state, Instruction const& inst)
{
    return semijoin_impl(state, inst);
}


ALWAYS_INLINE
Instruction const* done_inst(State& state, Instruction const& inst)
{ 
    DECODE(a);

    // Finish execution of the block and store the result in
    // the requested location.

    REnvironment& env = static_cast<REnvironment&>(REGISTER(0));
    Value const& index = REGISTER(1);

    if(index.isCharacter1())
    {
        assert(env.isEnvironment());
        // caller guarantees that index is interned already.
        env.environment()->insert(index.s) = a;
    }
    else if(index.isInteger1())
    {
        assert(env.isEnvironment());
        // caller guarantees that dots exists and is an ok length
        Value* dots = env.environment()->get(Strings::__dots__);
        assert(dots && dots->isList());
        static_cast<List&>(*dots)[index.i] = a;
    }
    
    REGISTER(0) = a;

    Instruction const* pc = state.frame.returnpc; 
    state.pop();
    return pc;
}

