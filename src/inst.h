
#include "call.h"

// To help out the compiler, bytecode handlers are split
// into a fast-path that should be inlined into the interpreter
// (defined in this file) and a slow-path impl function that should
// be called from the interpreter.


// Forward define impls for all bytecodes
#define DEFINE_IMPL(name,...) \
    Instruction const* name##_impl(State& state, Instruction const& inst);
STANDARD_BYTECODES(DEFINE_IMPL);




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
    Value const* v = state.frame.environment->getRecursive2(s, env);

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
        Value* v = env->getRecursive2(s, env);

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
    Value& dest = up->insertRecursive(s, penv);

    if(!dest.isNil())
        dest = c;
    else
        state.global.global->insert(s) = c;

    return &inst+1;
}


ALWAYS_INLINE
Instruction const* force_inst(State& state, Instruction const& inst)
{
    Value const& a = REGISTER(inst.a);

    Value const& dots = state.frame.environment->get(Strings::__dots__);
    assert(dots.isList());

    Value const& t = static_cast<List const&>(dots)[a.i];

    if(t.isObject()) {
        OUT(c) = t;
        return &inst+1;
    }
    else if(t.isPromise()) {
        return force(state, static_cast<Promise const&>(t),
                state.frame.environment, a,
                inst.c, &inst+1);
    }
    else {
        _internalError("Unexpected Nil operand in force_I");
    }
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
    Value const& dots = state.frame.environment->get(Strings::__dots__);

    if(!dots.isList())
        OUT(c) = Integer::c(0);
    else
        OUT(c) = Integer::c(static_cast<List const&>(dots).length());

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
    if(GetFast(state, a, b, OUT(c)))
        return &inst+1;

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
        OUT(c) = (static_cast<REnvironment const&>(a).environment()->
                get(static_cast<Character const&>(b).s)).isNil()
                ? Logical::False()
                : Logical::True();
        return &inst+1;
    }

    return env_has_impl(state, inst);
}


ALWAYS_INLINE
Instruction const* env_get_inst(State& state, Instruction const& inst)
{
    state.visible = true;
    DECODE(a);
    DECODE(b);

    if(a.isEnvironment() && b.isCharacter() && b.pac == 1)
    {
        OUT(c) = static_cast<REnvironment const&>(a).environment()->
                get(static_cast<Character const&>(b).s);
        return &inst+1;
    }

    return env_get_impl(state, inst);
}

// TODO: This isn't implemented correctly yet
ALWAYS_INLINE
Instruction const* env_set_inst(State& state, Instruction const& inst)
{
    // a = index, b = environment, c = value
    DECODE(a);
    DECODE(b);
    DECODE(c);

    if(a.isCharacter() && a.pac == 1 && b.isEnvironment() && c.isObject())
    {
        static_cast<REnvironment const&>(b).environment()->insert(a.s) = c;
        return &inst+1;
    }

    return env_set_impl(state, inst);
}


ALWAYS_INLINE
Instruction const* env_rm_inst(State& state, Instruction const& inst)
{
    DECODE(a);
    DECODE(b);

    if(a.isEnvironment() && b.isCharacter() && b.pac == 1)
    {
        static_cast<REnvironment const&>(a).environment()->remove(b.s);
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
    REGISTER(0) = a;

    Instruction const* pc = state.frame.returnpc; 
    state.pop();
    return pc;
}

