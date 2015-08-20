
#include <string>
#include <dlfcn.h>

#include "inst.h"

#include "call.h"
#include "compiler.h"
#include "ops.h"
#include "runtime.h"


Instruction const* force(
    State& state, Promise const& p,
    Environment* targetEnv, Value targetIndex,
    int64_t outRegister, Instruction const* returnpc) {

    Code* code = p.isExpression() ? p.code() : state.global.promiseCode;
    Compiler::doPromiseCompilation(state, code);
    Instruction const* r = buildStackFrame(
        state, p.environment(), code, outRegister, returnpc);
    state.frame.isPromise = true;
            
    REnvironment::Init(REGISTER(0), targetEnv);
    REGISTER(1) = targetIndex;
	
    if(p.isDotdot())		
        Integer::InitScalar(REGISTER(2), p.dotIndex());

    return r;
}


Instruction const* call_impl(State& state, Instruction const& inst)
{
    Heap::GlobalHeap.collect(state.global);

    DECODE(a); BIND(a);
    state.visible = true;

    if(a.isClosure())
    {
        auto func = static_cast<Closure const&>(a);

        CompiledCall const& call = state.frame.code->calls[inst.b];

        Environment* fenv =
            (call.names.length() == 0 &&
             call.dotIndex >= (int64_t)call.arguments.length())
            ? FastMatchArgs(state, state.frame.environment, func, call)
            : MatchArgs(state, state.frame.environment, func, call);

        return buildStackFrame(state, fenv, func.prototype()->code, inst.c, &inst+1);
    }

    return StopDispatch(state, inst, state.internStr(
        (std::string("Non-function (") + Type::toString(a.type()) + ") as first parameter to call\n").c_str()), inst.c);
}


Instruction const* ret_impl(State& state, Instruction const& inst)
{
    // we can return futures from functions, so don't BIND
    DECODE(a);

    if(state.stack.size() == 1)
        _error("no function to return from, jumping to top level");

    if(state.frame.isPromise) {
        Environment* env = state.frame.environment;
        do {
            if(state.stack.size() <= 1)
                _error("no function to return from, jumping to top level");
            state.pop();
        } while( !(state.frame.environment == env 
                 && state.frame.isPromise == false) );
    }

    REGISTER(0) = a;
    Instruction const* returnpc = state.frame.returnpc;
    
    Value& onexit = state.frame.environment->insert(Strings::__onexit__);
    if(onexit.isObject() && !onexit.isNull()) {
        Promise::Init(onexit,
            state.frame.environment,
            Compiler::deferPromiseCompilation(state, onexit), false);
        return force(state, static_cast<Promise const&>(onexit),
            state.frame.environment, Value::Nil(),
            1, &inst);
    }
    
    // We can free this environment for reuse
    // as long as we don't return a closure...
    // but also can't if an assignment to an 
    // out of scope variable occurs (<<-, assign) with a value of a closure!
#ifdef EPEE
    if(!(a.isClosure() || a.isEnvironment() || a.isList())) {
        state.traces.KillEnvironment(state.frame.environment);
    }
#endif

    state.pop();

#ifdef EPEE
    state.traces.LiveEnvironment(state.frame.environment, a);
#endif
    return returnpc;
}


Instruction const* jc_impl(State& state, Instruction const& inst)
{
    state.visible = true;
    DECODE(c); BIND(c);
    Logical::Element cond;
    if(c.isLogical1())
        cond = c.c;
    else if(c.isInteger1())
        cond = Cast<Integer, Logical>(state, c.i);
    else if(c.isDouble1())
        cond = Cast<Double, Logical>(state, c.i);
    // It breaks my heart to allow non length-1 vectors,
    // but this seems to be somewhat widely used, even
    // in some of the recommended R packages.
    else if(c.isLogical()) {
        if(static_cast<Logical const&>(c).length() > 0)
            cond = static_cast<Logical const&>(c)[0];
        else
            return StopDispatch(state, inst, state.internStr(
            "conditional is of length zero"), 
            inst.c);
    }
    else if(c.isInteger()) {
        if(static_cast<Integer const&>(c).length() > 0)
            cond = Cast<Integer, Logical>(state, static_cast<Integer const&>(c)[0]);
        else
            return StopDispatch(state, inst, state.internStr(
            "conditional is of length zero"), 
            inst.c);
    }
    else if(c.isDouble()) {
        if(static_cast<Double const&>(c).length() > 0)
            cond = Cast<Double, Logical>(state, static_cast<Double const&>(c)[0]);
        else
            return StopDispatch(state, inst, state.internStr(
            "conditional is of length zero"), 
            inst.c);
    }
    else if(c.isRaw()) {
        if(static_cast<Raw const&>(c).length() > 0)
            cond = Cast<Raw, Logical>(state, static_cast<Raw const&>(c)[0]);
        else
            return StopDispatch(state, inst, state.internStr(
            "conditional is of length zero"), 
            inst.c);
    }
    else {
        return StopDispatch(state, inst, state.internStr(
                    "conditional argument is not interpretable as logical"), 
                    inst.c);
    }
    
    if(Logical::isTrue(cond)) return &inst+inst.a;
    else if(Logical::isFalse(cond)) return &inst+inst.b;
    else if((&inst+1)->bc != ByteCode::stop) return &inst+1;
    else return StopDispatch(state, inst, state.internStr(
            "NA where TRUE/FALSE needed"), 
            inst.c);
}


Instruction const* forbegin_impl(State& state, Instruction const& inst)
{
    //state.visible = true;
    // a = loop variable (e.g. i)
    // b = loop vector(e.g. 1:100)
    // c = counter register
    // following instruction is a jmp that contains offset
    Value const& b = REGISTER(inst.b);
    BIND(b);
    if(!b.isVector())
        _error("Invalid for() loop sequence");
    Vector const& v = (Vector const&)b;
    if((int64_t)v.length() <= 0) {
        return &inst+(&inst+1)->a;    // offset is in following JMP, dispatch together
    } else {
        String i = static_cast<Character const&>(CONSTANT(inst.a)).s;
        Element2(v, 0, state.frame.environment->insert(i));
        Integer::InitScalar(REGISTER(inst.c), 1);
        Integer::InitScalar(REGISTER(inst.c+1), v.length());
        return &inst+2;            // skip over following JMP
    }
}


Instruction const* external_impl(State& state, Instruction const& inst)
{
    state.visible = true;
    DECODE(a);

    void* func = NULL;
    if(a.isCharacter1()) {
        String name = a.s;
        for(std::map<std::string,void*>::iterator i = state.global.handles.begin();
            i != state.global.handles.end(); ++i) {
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
        return StopDispatch(state, inst, state.internStr(
            ".External needs a Character(1) or Externalptr as its first argument"), 
            inst.c);
    }

    uint64_t nargs = inst.b;
    for(int64_t i = 0; i < nargs; i++) {
        BIND(REGISTER(inst.a+i+1));
    }
    try {
        typedef Value (*Func)(State&, Value const*);
        Func f = (Func)func;
        OUT(c) = f(state, &REGISTER(inst.a+1));
    }
    catch( char const* e ) {
        _error(std::string("External function call failed: ") + e);
    }
    catch( ... ) {
        _error(std::string("External function call failed with unknown error"));
    }

    return &inst+1;
}


Instruction const* map_impl(State& state, Instruction const& inst)
{
    state.visible = true;
    DECODE(a); BIND(a);
    DECODE(b); BIND(b);
    DECODE(c); BIND(c);

    if(!b.isList())
        _error("External map args must be a list");

    if(c.isCharacter1()) {
        if(!a.isCharacter())
            _error("External map return types must be a character vector");
        OUT(c) = Map(state, c.s, static_cast<List const&>(b), static_cast<Character const&>(a));
    }
    else if(c.isClosure()) {
        if(a.isCharacter())
            OUT(c) = MapR(state, static_cast<Closure const&>(c), static_cast<List const&>(b), static_cast<Character const&>(a));
        else
            OUT(c) = MapI(state, static_cast<Closure const&>(c), static_cast<List const&>(b));
    }
    else
        _error(".Map function name must be a string or a closure");
    
    return &inst+1;
}


Instruction const* scan_impl(State& state, Instruction const& inst)
{
    state.visible = true;
    DECODE(a); BIND(a);
    DECODE(b); BIND(b);
    DECODE(c); BIND(c);

    if(!c.isCharacter1())
        _error("External scan function name must be a string");
    if(!b.isList())
        _error("External scan args must be a list");
    if(!a.isCharacter())
        _error("External scan return types must be a character vector");

    OUT(c) = Scan(state, c.s, static_cast<List const&>(b), static_cast<Character const&>(a));
    
    return &inst+1;
}


Instruction const* fold_impl(State& state, Instruction const& inst)
{
    state.visible = true;
    DECODE(a); BIND(a);
    DECODE(b); BIND(b);
    DECODE(c); BIND(c);

    if(!c.isCharacter1())
        _error("External fold function name must be a string");
    if(!b.isList())
        _error("External fold args must be a list");
    if(!a.isCharacter())
        _error("External fold return types must be a character vector");

    OUT(c) = Fold(state, c.s, static_cast<List const&>(b), static_cast<Character const&>(a));
    
    return &inst+1;
}


Instruction const* load_impl(State& state, Instruction const& inst)
{
    String s = static_cast<Character const&>(CONSTANT(inst.a)).s;
    
    return StopDispatch(state, inst, state.internStr(
            (std::string("Object '") + s->s + "' not found").c_str()), 
            inst.c);
}


Instruction const* loadfn_impl(State& state, Instruction const& inst)
{
    String s = static_cast<Character const&>(CONSTANT(inst.a)).s;
    
    return StopDispatch(state, inst, state.internStr(
            (std::string("Object '") + s->s + "' not found").c_str()), 
            inst.c);
}


Instruction const* dotsv_impl(State& state, Instruction const& inst)
{
    state.visible = true;
    DECODE(a); BIND(a);

    int64_t idx = 0;
    if(a.isInteger1())
        idx = static_cast<Integer const&>(a)[0] - 1;
    else if(a.isDouble1())
        idx = (int64_t)static_cast<Double const&>(a)[0] - 1;
    else
        return StopDispatch(state, inst, state.internStr("Invalid type in dotsv"), inst.c);

    Value const* t = state.frame.environment->get(Strings::__dots__);

    if(!t || !t->isList() ||
       idx >= (int64_t)static_cast<List const&>(*t).length() ||
       idx < (int64_t)0)
        return StopDispatch(state, inst, state.internStr((std::string("The '...' list does not contain ") + intToStr(idx+1) + " elements").c_str()), inst.c);

    Value const& v = static_cast<List const&>(*t)[idx];

    if(v.isObject()) {
        OUT(c) = v;
        return &inst+1;
    }
    else if(v.isPromise()) {
        return force(state, static_cast<Promise const&>(v),
            state.frame.environment, Integer::c(idx),
            inst.c, &inst+1);
    }
    else {
        return StopDispatch(state, inst, state.internStr(
            (std::string("Object '..") + intToStr(idx+1) + 
                "' not found, missing argument?").c_str()), 
            inst.c);
    }
}


Instruction const* dots_impl(State& state, Instruction const& inst)
{
    state.visible = true;

    Value const* t = state.frame.environment->get(Strings::__dots__);
    if(!t)
    {
        OUT(c) =  List(0);
        return &inst+1;
    }
    else if(!t->isList())
    {
        OUT(c) = *t;
        return &inst+1;
    }

    List const& dots = static_cast<List const&>(*t); 

    Value& iter = REGISTER(inst.a);
    Value& out = OUT(c);

    // First time through, make a result vector...
    if(iter.i == -1)
    {
        Heap::GlobalHeap.collect(state.global);
        out = List(dots.length());
        memset(((List&)out).v(), 0, dots.length()*sizeof(List::Element));

        iter.i++;
    }

    while(iter.i < (int64_t)dots.length())
    {
        Value const& v = dots[iter.i];

        if(!v.isPromise())
        {
            BIND(v); // BIND since we don't yet support futures in lists
            ((List&)out)[iter.i] = v;
            iter.i++;
        }
        else
        {
            return force(state, static_cast<Promise const&>(v),
                state.frame.environment, Integer::c(iter.i),
                inst.b, &inst);
        }
    }

    // check to see if we need to add names
    Value const* names = state.frame.environment->get(Strings::__names__);

    if(names && names->isCharacter())
    {
        Dictionary* d = new Dictionary(1);
        d->insert(Strings::names) = *names;
        ((Object&)out).attributes(d);
    }

    return &inst+1;
}


Instruction const* frame_impl(State& state, Instruction const& inst)
{
    state.visible = true;
    DECODE(a); BIND(a);
    int64_t index = a.i;

    Environment* env = state.frame.environment;

    while(index > 0)
    {
        Value const* v = env->get(Strings::__parent__);
        if(v && v->isEnvironment())
            env = static_cast<REnvironment const&>(*v).environment();
        else
            break;
        index--;
    }

    if(index == 0) {
        REnvironment::Init(OUT(c), env);
        return &inst+1;
    }
    else {
        return StopDispatch(state, inst, state.internStr(
            "not that many frames on the stack"), 
            inst.c);
    }
}


Instruction const* pr_new_impl(State& state, Instruction const& inst)
{
    state.visible = true;
    DECODE(a); BIND(a);
    DECODE(b); BIND(b);

    if( !a.isCharacter1() ||
        !REGISTER((&inst+1)->a).isEnvironment() ||
        !REGISTER((&inst+1)->b).isEnvironment())
        return StopDispatch(state, inst, state.internStr("wrong types in pr_new"), inst.c);

    REnvironment& eval = (REnvironment&)REGISTER((&inst+1)->a);
    REnvironment& assign = (REnvironment&)REGISTER((&inst+1)->b);

    String in = state.global.strings.in(a.s->s);
    Value& v = assign.environment()->insert(in);

    try {
        Promise::Init(v,
            eval.environment(),
            Compiler::deferPromiseCompilation(state, b), false);
    }
    catch(RuntimeError const& e) {
        return StopDispatch(state, inst, state.internStr(
            e.what().c_str()), 
            inst.c);
    }

    OUT(c) = Null::Singleton();

    return &inst+2;
}


Instruction const* pr_expr_impl(State& state, Instruction const& inst)
{
    state.visible = true;
    DECODE(a); BIND(a);
    DECODE(b); BIND(b);

    if(a.isEnvironment() && b.isCharacter() && static_cast<Character const&>(b).length() == 1)
    {
        REnvironment const& env = static_cast<REnvironment const&>(a);
        String s = static_cast<Character const&>(b).s;

        Value const* v = internAndGet(state, env.environment(), s);
        if(v)
        {
            OUT(c) = (v->isPromise())
                ? static_cast<Promise const&>(*v).code()->expression
                : *v;
            return &inst+1;
        }
    }
    else if(a.isList() && b.isInteger1())
    {
        List const& l = static_cast<List const&>(a);
        int64_t i = static_cast<Integer const&>(b).i - 1;
        if(i >= 0 && i < l.length()) {
            Value v = l[i];
            if(v.isPromise())
                v = static_cast<Promise const&>(v).code()->expression;
            OUT(c) = v;
            return &inst+1;
        }
    }

    printf("%d %d\n", a.type(), b.type());
    return StopDispatch(state, inst, state.internStr(
                "invalid pr_expr expression"), 
                inst.c);
}


Instruction const* pr_env_impl(State& state, Instruction const& inst)
{
    state.visible = true;
    DECODE(a); BIND(a);
    DECODE(b); BIND(b);

    // TODO: check types

    REnvironment const& env = static_cast<REnvironment const&>(a);
    String s = static_cast<Character const&>(b).s;

    Value const* v = internAndGet(state, env.environment(), s);
    if(v && v->isPromise())
        REnvironment::Init(OUT(c), static_cast<Promise const&>(*v).environment());
    else
        OUT(c) = Null::Singleton();
    
    return &inst+1;
}


Instruction const* type_impl(State& state, Instruction const& inst)
{
    state.visible = true;
    DECODE(a);

#ifdef EPEE
    switch(state.traces.futureType(a)) {
#else
    switch(a.type()) {
#endif
        #define CASE(name, str, ...) case Type::name: \
            OUT(c) = Character::c(Strings::name); \
        break;
        TYPES(CASE)
        #undef CASE
        default:
            _internalError("Unhandled type in type_impl");
            break;
    }

    return &inst+1;
}


Instruction const* length_impl(State& state, Instruction const& inst)
{
    state.visible = true;
    DECODE(a);
    if(a.isVector())
        Integer::InitScalar(OUT(c), static_cast<Vector const&>(a).length());
#ifdef EPEE
    else if(a.isFuture()) {
        IRNode::Shape shape = state.traces.futureShape(a);
        if(shape.split < 0 && shape.filter < 0) {
            Integer::InitScalar(OUT(c), shape.length);
        } else {
            OUT(c) = state.traces.EmitUnary<CountFold>(state.frame.environment, IROpCode::length, a, 0);
            state.traces.OptBind(state, OUT(c));
        }
    }
#endif
    else if(static_cast<Object const&>(a).hasAttributes()) { 
        return GenericDispatch(state, inst, Strings::length, a, inst.c); 
    } else {
        Integer::InitScalar(OUT(c), 1);
    }
    return &inst+1;
}


Instruction const* get_impl(State& state, Instruction const& inst)
{
    state.visible = true;
    DECODE(a); BIND(a);
    DECODE(b); BIND(b);
    try
    { 
        // This case seems wrong, but some base code relies on this behavior. 
        if(a.isNull()) {
            OUT(c) = Null::Singleton();
            return &inst+1;
        }
        else if(!static_cast<Object const&>(a).hasAttributes())
        {
	        if(a.isVector()) {
                auto v = static_cast<Vector const&>(a);
		        if(b.isInteger()) {
                    if(  ((Integer const&)b).length() != 1
                    || (b.i-1) < 0 )
                        _error("attempt to select more or less than one element");
                    if( (b.i-1) >= v.length() )
                        _error("subscript out of bounds");

                    Element2(v, b.i-1, OUT(c));
                    return &inst+1;
                }
		        else if(b.isDouble()) {
                    if(  ((Double const&)b).length() != 1
                    || ((int64_t)b.d-1) < 0 )
                        _error("attempt to select more or less than one element");
                    if( ((int64_t)b.d-1) >= v.length())
                        _error("subscript out of bounds");
                
                    Element2(a, (int64_t)b.d-1, OUT(c));
                    return &inst+1;
                }
	        }
            else if(a.isEnvironment()) {
                if( b.isCharacter()
                    && static_cast<Character const&>(b).length() == 1) {
	                String s = static_cast<Character const&>(b).s;
                    auto env = static_cast<REnvironment const&>(a);
                    Value const* v = internAndGet(state, env.environment(), s);
                    
                    if(!v)
                    {
                        OUT(c) = Null::Singleton();
                        return &inst+1;
                    }
                    else if(v->isObject() || v->isNil()) {
                        OUT(c) = *v;
                        return &inst+1;
                    }
                    else {
                        return force(state, static_cast<Promise const&>(*v), 
                            static_cast<REnvironment const&>(a).environment(), b,
                            inst.c, &inst+1); 
                    }
                }
            }
            else if(a.isClosure()) {
                if( b.isCharacter()
                    && static_cast<Character const&>(b).length() == 1 ) {
                    auto f = static_cast<Closure const&>(a);
	                String s = static_cast<Character const&>(b).s;
                    if(s == Strings::body) {
                        OUT(c) = f.prototype()->code->expression;
                        return &inst+1;
                    }
                    else if(s == Strings::formals) {
                        OUT(c) = f.prototype()->formals;
                        return &inst+1;
                    }
                }
            }
        }
 
        return GenericDispatch(state, inst, Strings::bb, a, b, inst.c); 
    }
    catch(RuntimeError const& e)
    {
        return StopDispatch(state, inst, state.internStr(
            e.what().c_str()), 
            inst.c);
    }
}


Instruction const* set_impl(State& state, Instruction const& inst)
{
    state.visible = true;
    // a = value, b = index, c = dest
    DECODE(a); DECODE(b); DECODE(c);
    BIND(b);

#ifdef EPEE
    if(a.isFuture() && (c.isVector() || c.isFuture())) {
        if(b.isInteger() && static_cast<Integer const&>(b).length() == 1) {
            OUT(c) = state.traces.EmitSStore(state.frame.environment, c, ((Integer&)b)[0], a);
            return &inst+1;
        }
        else if(b.isDouble() && static_cast<Double const&>(b).length() == 1) {
            OUT(c) = state.traces.EmitSStore(state.frame.environment, c, ((Double&)b)[0], a);
            return &inst+1;
        }
    }
#endif

    BIND(a);
    BIND(c);

    if( !static_cast<Object const&>(c).hasAttributes() ) {
        if(c.isVector() && ( b.isDouble1() || b.isInteger1() )) {
            Subset2Assign(state, c, true, b, a, OUT(c));
            return &inst+1;
        }
        else if(c.isEnvironment() && b.isCharacter1()) {
            auto env = static_cast<REnvironment const&>(c);
            String s = static_cast<Character const&>(b).s;
            String in = state.global.strings.in(s->s);
            
            env.environment()->insert(in) = a;
            OUT(c) = c;
            return &inst+1;
        }
        else if(c.isClosure() && b.isCharacter1()) {
            //Closure const& f = static_cast<Closure const&>(c);
            //String s = static_cast<Character const&>(b).s;
            _error("Assignment to function members is not yet implemented");
        }
    }
    return GenericDispatch(state, inst, Strings::bbAssign, c, b, a, inst.c); 
}


Instruction const* getsub_impl(State& state, Instruction const& inst)
{
    state.visible = true;
    DECODE(a); DECODE(b);

    if(a.isVector() && !static_cast<Object const&>(a).hasAttributes()) {
        if(b.isDouble1()
            && (int64_t)(b.d-1) >= 0
            && (int64_t)(b.d-1) < static_cast<Vector const&>(a).length()) { 
            Element(a, b.d-1, OUT(c)); return &inst+1; }
        else if(b.isInteger1()
            && (b.i-1) >= 0
            && (b.i-1) < static_cast<Vector const&>(a).length()) { 
            Element(a, b.i-1, OUT(c)); return &inst+1; }
    }

#ifdef EPEE
    if( state.traces.isTraceable(a, b) 
        && state.traces.futureType(b) == Type::Logical 
        && state.traces.futureShape(a) == state.traces.futureShape(b)) {
        OUT(c) = state.traces.EmitFilter(state.frame.environment, a, b);
        state.traces.OptBind(state, OUT(c));
        return &inst+1;
    }
#endif

    BIND(a);

#ifdef EPEE
    if(state.traces.isTraceable(a, b) 
        && (state.traces.futureType(b) == Type::Integer 
            || state.traces.futureType(b) == Type::Double)) {
        OUT(c) = state.traces.EmitGather(state.frame.environment, a, b);
        state.traces.OptBind(state, OUT(c));
        return &inst+1;
    }
#endif

    BIND(b);

    if(a.isVector() && !static_cast<Object const&>(a).hasAttributes() &&
        (b.isInteger() || b.isDouble() || b.isLogical()) ) {
        SubsetSlow(state, a, b, OUT(c)); 
        return &inst+1;
    }
    // TODO: this should force promises...
    else if(a.isEnvironment() && b.isCharacter()) {
        REnvironment const& env = static_cast<REnvironment const&>(a);
        Character const& i = static_cast<Character const&>(b);
        List r(i.length());
        for(int64_t k = 0; k < i.length(); ++k)
        {
            Value const* v = internAndGet(state, env.environment(), i[k]);
            r[k] = v ? *v : Null::Singleton();
        }
        OUT(c) = r; 
        return &inst+1;
    }

    return GenericDispatch(state, inst, Strings::bracket, a, b, inst.c); 
}


Instruction const* setsub_impl(State& state, Instruction const& inst)
{
    state.visible = true;
    // a = value, b = index, c = dest 
    DECODE(a); DECODE(b); DECODE(c); 
    BIND(b); BIND(c);

#ifdef EPEE
    if(a.isFuture() && (c.isVector() || c.isFuture())) {
        if(b.isInteger() && static_cast<Integer const&>(b).length() == 1) {
            OUT(c) = state.traces.EmitSStore(state.frame.environment, c, ((Integer&)b)[0], a);
            return &inst+1;
        }
        else if(b.isDouble() && static_cast<Double const&>(b).length() == 1) {
            OUT(c) = state.traces.EmitSStore(state.frame.environment, c, ((Double&)b)[0], a);
            return &inst+1;
        }
    }
#endif

    BIND(a);

    if( !static_cast<Object const&>(c).hasAttributes() ) {
        if(c.isVector() && ( b.isDouble() || b.isInteger() || b.isLogical() )) {
            SubsetAssign(state, c, true, b, a, OUT(c));
            return &inst+1;
        }
    }
    if(c.isEnvironment() && b.isCharacter())
    {
        auto env = static_cast<REnvironment const&>(c);
        auto i   = static_cast<Character const&>(b);

        if(a.isVector())
        {
            List const& l = As<List>(state, a);
            int64_t len = std::max(l.length(), i.length());
            if(l.length() == 0 || i.length() == 0) len = 0;
            for(int64_t k = 0; k < len; ++k)
            {
                int64_t ix = k % i.length();
                int64_t lx = k % l.length();
                if(!Character::isNA(i[ix]))
                {
                    String in = state.global.strings.in(i[ix]->s);
                    env.environment()->insert(in) = l[lx];
                }
            }
        }
        else
        {
            int64_t len = i.length();
                
            for(int64_t k = 0; k < len; ++k)
            {
                if(!Character::isNA(i[k]))
                {
                    String in = state.global.strings.in(i[k]->s);
                    env.environment()->insert(in) = a;
                }
            }
        }
        OUT(c) = env; 
        return &inst+1;
    }

    return GenericDispatch(state, inst, Strings::bracketAssign, c, b, a, inst.c); 
}


Instruction const* setenv_impl(State& state, Instruction const& inst)
{
    state.visible = true;
    DECODE(a); BIND(a);
    DECODE(b); BIND(b);

    if(!b.isEnvironment()) {
        return StopDispatch(state, inst, state.internStr(
            "setenv replacement object is not an environment"), 
            inst.c);
    }

    Environment* value = static_cast<REnvironment const&>(b).environment();

    if(a.isEnvironment()) {
        Environment* target = static_cast<REnvironment const&>(a).environment();
        
        // Riposte allows enclosing environment replacement,
        // but requires that no loops be introduced in the environment chain.
        Environment* p = value;
        while(p) {
            if(p == target) 
                _error("an environment cannot be its own ancestor");
            p = p->getEnclosure();
        }
        
        static_cast<REnvironment const&>(a).environment()->setEnclosure(
            static_cast<REnvironment const&>(b).environment());
        OUT(c) = a;
    }
    else if(a.isClosure()) {
        Closure::Init(OUT(c), static_cast<Closure const&>(a).prototype(), value);
    }
    else {
        return StopDispatch(state, inst, state.internStr(
            "target of assignment does not have an enclosing environment"), 
            inst.c);
    }
    return &inst+1;
}


Instruction const* getattr_impl(State& state, Instruction const& inst)
{
    state.visible = true;
    DECODE(a);
    DECODE(b); BIND(b);
    if(a.isObject() && b.isCharacter1())
    {
        String name = static_cast<Character const&>(b)[0];
        Object const& o = (Object const&)a;
        REnvironment const& env = (REnvironment const&)a;

        Value const* v = nullptr;
        if(a.isEnvironment())
        {
            v = internAndGet(state, env.environment()->getAttributes(), name);
        }
        else if(a.isObject())
        {
            v = internAndGet(state, o.attributes(), name);
        }

        OUT(c) = v ? *v : Null::Singleton();
        return &inst+1;
    }

    printf("%d %d\n", a.type(), b.type());
    _error("Invalid attrget operation");
}


Instruction const* setattr_impl(State& state, Instruction const& inst)
{
    state.visible = true;
    DECODE(c);
    DECODE(b); BIND(b);
    DECODE(a); BIND(a);

    if(b.isCharacter1())
    {
        String str = static_cast<Character const&>(b)[0];
        String name = state.global.strings.in(str->s);

        // Special casing for R's compressed rownames representation.
        // Can we move this to the R compatibility layer?
        Value v = a;
        if(name == Strings::rownames && v.isInteger())
        {
            auto i = static_cast<Integer const&>(v);
            if(i.length() == 2 && Integer::isNA(i[0]))
            { 
                v = Sequence((int64_t)1,1,abs(i[1]));
            }
        }

        if(c.isEnvironment())
        {
            REnvironment env = (REnvironment const&)c;
            Dictionary const* d = env.environment()->getAttributes();

            // Delete the attribute if the user assign NULL
            env.environment()->setAttributes(
                a.isNull() ? d->cloneWithout(name) : d->cloneWith(name,v));

            OUT(c) = env;
            return &inst+1;
        }
        else if(c.isObject())
        {
            Object o = (Object const&)c;
            Dictionary const* d = o.attributes();

            // Delete the attribute if the user assigns NULL
            o.attributes(
                a.isNull() ? d->cloneWithout(name) : d->cloneWith(name, v));

            OUT(c) = o;
            return &inst+1;
        }
    }
    
    return StopDispatch(state, inst, state.internStr(
            "Invalid setattr operation"), 
            inst.c);
}


Instruction const* attributes_impl(State& state, Instruction const& inst)
{
    state.visible = true;
    DECODE(a);

    if(a.isObject())
    {
        Object const& obj = (Object const&)a;
        REnvironment const& env = (REnvironment const&)a;

        Dictionary const* d = a.isEnvironment()
            ? env.environment()->getAttributes()
            : obj.attributes();

        if(d->Size() > 0)
        {
            Character n(d->Size());
            List v(d->Size());

            int64_t j = 0;
            for(auto i = d->begin(); i != d->end(); ++i, ++j)
            {
                n[j] = i.string();
                v[j] = i.value();
            }

            v.attributes(new Dictionary(Strings::names, n)); 
            OUT(c) = v;
            return &inst+1;
        }
    }
    
    OUT(c) = Null::Singleton();
    return &inst+1;
}


Instruction const* as_impl(State& state, Instruction const& inst)
{
    state.visible = true;
    DECODE(a); BIND(a);
    DECODE(b); BIND(b);
    if(!b.isCharacter1()) {
        return StopDispatch(state, inst, state.internStr(
            "invalid type argument to 'as'"), 
            inst.c);
    }
    String type = b.s;
    try {
        if(type == Strings::Null)
            OUT(c) = As<Null>(state, a);
        else if(type == Strings::Logical)
            OUT(c) = As<Logical>(state, a);
        else if(type == Strings::Integer)
            OUT(c) = As<Integer>(state, a);
        else if(type == Strings::Double)
            OUT(c) = As<Double>(state, a);
        else if(type == Strings::Character)
            OUT(c) = As<Character>(state, a);
        else if(type == Strings::List)
            OUT(c) = As<List>(state, a);
        else if(type == Strings::Raw)
            OUT(c) = As<Raw>(state, a);
        else
            return StopDispatch(state, inst, state.internStr(
                "'as' not yet defined for this type"), 
                inst.c);
    }
    catch(RuntimeError const& e) {
        return StopDispatch(state, inst, state.internStr(
                e.what()), 
                inst.c);
    }

    // Add support for futures
    return &inst+1;
}


Instruction const* env_new_impl(State& state, Instruction const& inst)
{
    state.visible = true;
    DECODE(a); BIND(a);

    if(!a.isEnvironment())
        _error("'enclos' must be an environment");

    Environment* env = new Environment(4,static_cast<REnvironment const&>(a).environment());
    Value p;
    REnvironment::Init(p, state.frame.environment);
    env->insert(Strings::__parent__) = p;
    REnvironment::Init(OUT(c), env);
    return &inst+1;
}


Instruction const* env_names_impl(State& state, Instruction const& inst)
{
    state.visible = true;
    DECODE(a); BIND(a);

    if(!a.isEnvironment())
        _error("'enclos' must be an environment");

    Environment* env = static_cast<REnvironment const&>(a).environment();

    Character r(env->Size());
    int64_t j = 0;
    for(Dictionary::const_iterator i = env->begin(); i != env->end(); ++i, ++j)
    {
        r[j] = i.string();
    }

    OUT(c) = r;
    return &inst+1;
}


Instruction const* env_has_impl(State& state, Instruction const& inst)
{
    state.visible = true;
    DECODE(a); BIND(a);
    DECODE(b); BIND(b);

    if(!a.isEnvironment()) {
        return StopDispatch(state, inst, state.internStr(
            "invalid 'envir' argument to .getenv"), 
            inst.c);
    }
    if(!b.isCharacter() || b.pac != 1) {
        return StopDispatch(state, inst, state.internStr(
            "invalid 'x' argument to .getenv"), 
            inst.c);
    }

    auto env = static_cast<REnvironment const&>(a);
    auto str = static_cast<Character const&>(b);

    Value const* v = internAndGet(state, env.environment(), str.s);
    OUT(c) = v ? Logical::True() : Logical::False();

    return &inst+1;
}


Instruction const* env_rm_impl(State& state, Instruction const& inst)
{
    state.visible = true;
    DECODE(a); BIND(a);
    DECODE(b); BIND(b);

    if(!a.isEnvironment() && !a.isNull()) {
        return StopDispatch(state, inst, state.internStr(
            "invalid 'envir' argument to .env_rm"), 
            inst.c);
    }
    if(!b.isCharacter()) {
        return StopDispatch(state, inst, state.internStr(
            "invalid 'x' argument to .env_rm"), 
            inst.c);
    }

    Environment* env = a.isEnvironment()
        ? static_cast<REnvironment const&>(a).environment()
        : state.frame.environment;

    Character const& names = static_cast<Character const&>(b);

    // TODO: could have bulk removal code that would be a lot more efficient
    for(int64_t i = 0; i < names.length(); ++i)
    {
        internAndRemove(state, env, names[i]);
    }

    OUT(c) = Null::Singleton();
    return &inst+1;
}


Instruction const* env_missing_impl(State& state, Instruction const& inst)
{
    state.visible = true;
    DECODE(a); BIND(a);
    DECODE(b); BIND(b);

    if(!a.isEnvironment() && !a.isNull()) {
        return StopDispatch(state, inst, state.internStr(
            "invalid 'envir' argument to .env_missing"), 
            inst.c);
    }

    Environment* original_env = a.isEnvironment()
        ? static_cast<REnvironment const&>(a).environment()
        : state.frame.environment;
    Environment* e = original_env;

    Value x = b;

    bool missing = false;
    do {

    if( x.isCharacter() && static_cast<Character const&>(x).length() == 1 ) {
        Environment* foundEnv;
        Value const* v = internAndGetRec(state, e, x.s, foundEnv);
        
        missing = !v || v->isNil() ||
                  ( v->isPromise()
                    && static_cast<Promise const&>(*v).isDefault() 
                    && foundEnv == original_env 
                  );

        if(v->isPromise() && !static_cast<Promise const&>(*v).isDefault() && foundEnv == e) {
            // see if missing is passed down
            // missing is only passed down if
            // the referenced symbol is an argument.
            // this whole feature is a disaster.
            if(static_cast<Promise const&>(*v).isExpression()) {
                Value const& expr = static_cast<Promise const&>(*v).code()->expression;
                Environment* env = static_cast<Promise const&>(*v).environment();
                Value const* func = env->get(Strings::__function__);
                if(isSymbol(expr) && func && func->isClosure()) {
                    // see if the expr is an argument
                    Character const& parameters = static_cast<Closure const&>(*func).prototype()->parameters;
                    bool matched = false;
                    for(size_t i = 0; i < parameters.length() && !matched; ++i) {
                        if(static_cast<Character const&>(expr).s == parameters[i])
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
        if( x.isInteger() && static_cast<Integer const&>(x).length() == 1 )
            index = x.i-1;
        else if( x.isDouble() && static_cast<Double const&>(x).length() == 1 )
            index = x.d-1;
        else
            _error("Invalid argument to missing");

        Value const* dots = e->get(Strings::__dots__);

        missing = !dots ||
                  !dots->isList() ||
                  index < 0 ||
                  index >= static_cast<List const&>(*dots).length() ||
                  static_cast<List const&>(*dots)[index].isNil();

        break;
    }

    } while(true);

    Logical::InitScalar(OUT(c),
        missing ? Logical::TrueElement : Logical::FalseElement);

    return &inst+1;
}

#ifdef EPEE
 
#define OP(Name, string, Group, Func) \
Instruction const* Name##_impl(State& state, Instruction const& inst) \
{ \
    state.visible = true; \
    DECODE(a);    \
    try { \
        if(RecordUnary<Group, IROpCode::Name>(state, a, OUT(c))) \
            return &inst+1; \
        else if(!static_cast<Object const&>(a).hasAttributes() \
                && Group##Dispatch<Name##VOp>(state, nullptr, a, OUT(c))) \
            return &inst+1; \
        else \
            return GenericDispatch(state, inst, Strings::Name, a, inst.c); \
    } \
    catch(RuntimeError const& e) {\
        return StopDispatch(state, inst, state.internStr( \
            e.what().c_str()), \
            inst.c); \
    } \
}
UNARY_FOLD_SCAN_BYTECODES(OP)
#undef OP

#define OP(Name, string, Group, Func) \
Instruction const* Name##_impl(State& state, Instruction const& inst) \
{ \
    state.visible = true; \
    DECODE(a);    \
    DECODE(b);    \
    try { \
        return Name##Slow( state, inst, NULL, a, b, OUT(c) ); \
        if(RecordBinary<Group, IROpCode::Name>(state, a, b, OUT(c))) \
            return &inst+1; \
        else if(   !static_cast<Object const&>(a).hasAttributes() \
                && !static_cast<Object const&>(b).hasAttributes() \
                && Group##Dispatch<Name##VOp>(state, nullptr, a, b, OUT(c))) \
            return &inst+1; \
        else \
            return GenericDispatch(state, inst, Strings::Name, a, b, inst.c); \
    } \
    catch(RuntimeError const& e) { \
        return StopDispatch(state, inst, state.internStr( \
            e.what().c_str()), \
            inst.c); \
    } \
}
BINARY_BYTECODES(OP)
#undef OP

#else

#define OP(Name, string, Group, Func) \
Instruction const* Name##_impl(State& state, Instruction const& inst) \
{ \
    state.visible = true; \
    DECODE(a);    \
    try { \
        if(!static_cast<Object const&>(a).hasAttributes() \
            && Group##Dispatch<Name##VOp>(state, nullptr, a, OUT(c))) \
            return &inst+1; \
        else \
            return GenericDispatch(state, inst, Strings::Name, a, inst.c); \
    } \
    catch(RuntimeError const& e) {\
        return StopDispatch(state, inst, state.internStr( \
            e.what().c_str()), \
            inst.c); \
    } \
}
UNARY_FOLD_SCAN_BYTECODES(OP)
#undef OP

#define OP(Name, string, Group, Func) \
Instruction const* Name##_impl(State& state, Instruction const& inst) \
{ \
    state.visible = true; \
    DECODE(a);    \
    DECODE(b);    \
    try { \
        if(   !static_cast<Object const&>(a).hasAttributes() \
                && !static_cast<Object const&>(b).hasAttributes() \
                && Group##Dispatch<Name##VOp>(state, nullptr, a, b, OUT(c))) \
            return &inst+1; \
        else \
            return GenericDispatch(state, inst, Strings::Name, a, b, inst.c); \
    } \
    catch(RuntimeError const& e) { \
        return StopDispatch(state, inst, state.internStr( \
            e.what().c_str()), \
            inst.c); \
    } \
}
BINARY_BYTECODES(OP)
#undef OP

#endif


Instruction const* ifelse_impl(State& state, Instruction const& inst)
{
    state.visible = true;
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
    if(state.traces.isTraceable<IfElse>(a,b,c)) {
        OUT(c) = state.traces.EmitIfElse(state.frame.environment, a, b, c);
        state.traces.OptBind(state, OUT(c));
        return &inst+1;
    }
#endif
    BIND(a); BIND(b); BIND(c);

    IfElseDispatch(state, NULL, b, a, c, OUT(c));
    return &inst+1; 
}


Instruction const* split_impl(State& state, Instruction const& inst)
{
    state.visible = true;
#ifdef EPEE
    DECODE(a); BIND(a);
    DECODE(b);
    DECODE(c);
    int64_t levels = As<Integer>(state, a)[0];
    if(state.traces.isTraceable<Split>(b,c)) {
        OUT(c) = state.traces.EmitSplit(state.frame.environment, c, b, levels);
        state.traces.OptBind(state, OUT(c));
        return &inst+1;
    }
    BIND(a); BIND(b); BIND(c);
#endif

    _error("split not defined in scalar yet");
    return &inst+1; 
}


Instruction const* vector_impl(State& state, Instruction const& inst)
{
    state.visible = true;
    DECODE(a); BIND(a);
    DECODE(b); BIND(b);
    String stype = As<Character>(state, a)[0];
    Type::Enum type = string2Type( stype );
    int64_t l = As<Integer>(state, b)[0];

#ifdef EPEE
    if(state.global.epeeEnabled
        && (type == Type::Double || type == Type::Integer || type == Type::Logical)
        && l >= TRACE_VECTOR_WIDTH) {
        OUT(c) = state.traces.EmitConstant(state.frame.environment, type, l, 0);
        state.traces.OptBind(state, OUT(c));
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


Instruction const* seq_impl(State& state, Instruction const& inst)
{
    state.visible = true;
    DECODE(a); BIND(a);

    int64_t len = As<Integer>(state, a)[0];

#ifdef EPEE
    if(len >= TRACE_VECTOR_WIDTH) {
        OUT(c) = state.traces.EmitSequence(state.frame.environment, len, 1LL, 1LL);
        state.traces.OptBind(state, OUT(c));
        return &inst+1;
    }
#endif

    OUT(c) = Sequence(1LL, 1LL, len);
    return &inst+1;
}


Instruction const* index_impl(State& state, Instruction const& inst)
{
    state.visible = true;
    // c = n, b = each, a = length
    DECODE(a); BIND(a);
    DECODE(b); BIND(b);
    DECODE(c); BIND(c);

    int64_t n = As<Integer>(state, c)[0];
    int64_t each = As<Integer>(state, b)[0];
    int64_t len = As<Integer>(state, a)[0];

#ifdef EPEE
    if(len >= TRACE_VECTOR_WIDTH) {
        OUT(c) = state.traces.EmitIndex(state.frame.environment, len, (int64_t)n, (int64_t)each);
        state.traces.OptBind(state, OUT(c));
        return &inst+1;
    }
#endif

    OUT(c) = Repeat((int64_t)n, (int64_t)each, len);
    return &inst+1;
}


Instruction const* random_impl(State& state, Instruction const& inst)
{
    state.visible = true;
    DECODE(a); BIND(a);

    int64_t len = As<Integer>(state, a)[0];

    /*if(len >= TRACE_VECTOR_WIDTH) {
        OUT(c) = state.EmitRandom(state.frame.environment, len);
        state.OptBind(OUT(c));
        return &inst+1;
    }*/

    OUT(c) = RandomVector(state, len);
    return &inst+1;
}


Instruction const* semijoin_impl(State& state, Instruction const& inst)
{
    state.visible = true;
    DECODE(a); BIND(a);
    DECODE(b); BIND(b);

    // assumes that the two arguments are the same type...
    if(a.type() != b.type())
        return StopDispatch(state, inst, state.internStr(
            std::string("Arguments to semijoin must have the same type\n").c_str()),
            inst.c);
    assert(a.type() == b.type());
    OUT(c) = Semijoin(a, b);

    return &inst+1;
}
