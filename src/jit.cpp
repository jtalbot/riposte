
#include "jit.h"
#include "bc.h"
#include "interpreter.h"
#include "call.h"
#include "ops.h"

DEFINE_ENUM_TO_STRING(TraceOpCode, TRACE_ENUM)

const JIT::Shape JIT::Shape::Empty( 0, true, 0 );
const JIT::Shape JIT::Shape::Scalar( 1, true, 1 );

const JIT::IRRef JIT::FalseRef = 2;

size_t JIT::Trace::traceCount = 0;

JIT::IRRef JIT::Emit(IR const& ir) {
    trace.push_back(ir);
    return (IRRef) trace.size()-1;
}

JIT::IRRef JIT::Emit(IR const& ir, Instruction const* reenter, bool inScope) {
    IRRef i = Emit(ir);
    trace[ trace.size()-1 ].reenter.reenter = reenter;
    trace[ trace.size()-1 ].reenter.inScope = inScope;
    return i;
}

JIT::Shape JIT::SpecializeLength(size_t length, IRRef irlength) {
    // if short enough, guard length and insert a EmitConstant length instead
    if(length <= SPECIALIZE_LENGTH) {
        IRRef s = EmitConstant(Integer::c(length)).v;
        return Shape(s, true, length);
    }
    else {
        return Shape(irlength, false, length);
    }
}

JIT::Shape JIT::SpecializeValue(Value const& v, IRRef r) {
    if(v.isNull()) {
        return Shape::Empty;
    }
    else if(v.isVector()) {
        size_t len = (size_t)v.length;
        if(shapes.find(len) != shapes.end())
            return shapes.find(len)->second;
        else {
            IR l = IR(TraceOpCode::length, r, Type::Integer, Shape::Empty, Shape::Scalar);
            Shape r( Emit(l), false, len );
            shapes.insert(std::make_pair(len, r));
            return r;
        }
    }
    else {
        return Shape::Scalar;
    }
}

JIT::IRRef JIT::SpecializeNA(Value const& v, IRRef r) {
    bool mayHaveNA = false;
    switch(v.type) {
        #define CASE(Name) case Type::Name: mayHaveNA = ((Name const&)v).getMayHaveNA(); break;
        VECTOR_TYPES(CASE)
        #undef CASE
        default: break;
    }
    if(!mayHaveNA) {
        return FalseRef;
    }
    else { 
        return Emit( IR(TraceOpCode::decodena, r, Type::Logical, trace[r].out, trace[r].out) );
    }
}

JIT::Var JIT::load(Thread& thread, int64_t a, Instruction const* reenter) {

    // registers
    OPERAND(operand, a);

    IRRef r;
    
    if(a <= 0) {
        Variable v = { -1, (thread.base+a)-(thread.registers+DEFAULT_NUM_REGISTERS)};
        r = Emit( IR( TraceOpCode::sload, -1, v.i, Type::Any, Shape::Empty, Shape::Scalar ) );
    }
    else {
        IRRef aa = EmitConstant(Character::c((String)a)).v;
        
        Environment const* env = thread.frame.environment;
        r = Emit( IR( TraceOpCode::curenv, Type::Environment, Shape::Empty, Shape::Scalar ) );
        while(!env->has((String)a)) {
            env = env->LexicalScope();
            Emit( IR( TraceOpCode::load, r, aa, Type::Any, Shape::Scalar, Shape::Scalar ) );
            r = Emit( IR( TraceOpCode::lenv, r, Type::Environment, Shape::Scalar, Shape::Scalar ) );
        }
        
        r = Emit( IR( TraceOpCode::load, r, aa, Type::Any, Shape::Empty, Shape::Scalar ) );
    }

    Shape s = SpecializeValue(operand, r); 
    IRRef v = Emit( IR( TraceOpCode::unbox, r, operand.type, Shape::Scalar, s ), reenter, true );
          r = Emit( IR( TraceOpCode::decodevl, v, operand.type, s, s ) );
    IRRef na = SpecializeNA(operand, v);
    return Var(trace, r, na);
}

void JIT::store(Thread& thread, Var a, int64_t c) {
    a.v = Emit( IR( TraceOpCode::encode, a.v, a.na, trace[a.v].type, trace[a.v].out, trace[a.v].out ) );
    IRRef r = Emit( IR( TraceOpCode::box, a.v, Type::Any, trace[a.v].out, Shape::Scalar ) );
    if(c <= 0) {
        int64_t slot = (thread.base+c)-(thread.registers+DEFAULT_NUM_REGISTERS);
        Emit( IR( TraceOpCode::sstore, -1, slot, r, Type::Nil, Shape::Scalar, Shape::Empty ) );
    }
    else {
        IRRef cc = EmitConstant(Character::c((String)c)).v;
        IRRef e = Emit( IR( TraceOpCode::curenv, Type::Environment, Shape::Empty, Shape::Scalar) );
        Emit( IR( TraceOpCode::store, e, cc, r, Type::Nil, Shape::Scalar, Shape::Empty ) );
    }
}

void JIT::emitPush(Thread const& thread) {
    StackFrame frame;
   
    frame.prototype = thread.frame.prototype;
    frame.returnpc = thread.frame.returnpc;
    frame.returnbase = thread.frame.returnbase;
    frame.dest = thread.frame.dest;
    
    Emit( IR( TraceOpCode::push, getEnv(thread.frame.environment), getEnv(thread.frame.env), frames.size(), 
        Type::Nil, Shape::Scalar, Shape::Empty ) );
    
    frames.push_back(frame);
}

JIT::Var JIT::EmitUnary(TraceOpCode::Enum op, Var a, Type::Enum rty) {
    return Var(trace, Emit( IR(op, a.v, rty, a.s, a.s ) ), a.na); 
}

JIT::Var JIT::EmitCast(Var a, Type::Enum type) {
    if(a.type == type) 
        return a;

    TraceOpCode::Enum op;
    if(type == Type::Double)         op = TraceOpCode::asdouble;
    else if(type == Type::Integer)   op = TraceOpCode::asinteger;
    else if(type == Type::Logical)   op = TraceOpCode::aslogical;
    else if(type == Type::Character) op = TraceOpCode::ascharacter;
    else _error("Unexpected EmitCast type");
    
    return EmitUnary(op, a, type);
}

JIT::Var JIT::EmitBroadcast( Var a, Shape target ) {
    if( a.s == target )
        return a;

    return Var(trace, 
        Emit( IR( TraceOpCode::brcast, a.v, a.type, target, target ) ),
        Emit( IR( TraceOpCode::brcast, a.na, Type::Logical, target, target ) ) );
}

JIT::Var JIT::EmitRep( Var l, Var e, Shape target ) {
    // TODO: need to guard length 1 && not NA
    Var li = EmitCast(l, Type::Integer);
    Var ei = EmitCast(e, Type::Integer);
    Var m =  EmitBinary( TraceOpCode::mul, li, ei, Type::Integer, 0 );
     
    Var mb = EmitBroadcast( m, target );
    Var eb = EmitBroadcast( ei, target );

    Var s( trace, Emit( IR( TraceOpCode::seq, 0, 1, Type::Integer, target, target ) ), FalseRef );
    s = EmitBinary( TraceOpCode::mod, s, mb, Type::Integer, 0 );
    s = EmitBinary( TraceOpCode::idiv, s, eb, Type::Integer, 0 );
    return s;
}

JIT::Var JIT::EmitFold(TraceOpCode::Enum op, Var a, Type::Enum rty) {
    return Var(trace, 
        Emit( IR(op, a.v, rty, a.s, Shape::Scalar) ),
        Emit( IR(TraceOpCode::any, a.na, Type::Logical, a.s, Shape::Scalar ) ) );
}

JIT::Shape JIT::MergeShapes(Shape a, Shape b, Instruction const* inst) {
    Shape shape = Shape::Empty;
    if(a == b) {
        shape = a;
    }
    else if(a == Shape::Empty || b == Shape::Empty) {
        shape = Shape::Empty;
    }
    else if(a == Shape::Scalar) {
        shape = b;
    }
    else if(b == Shape::Scalar) {
        shape = a;
    }
    else if(a.traceLength < b.traceLength) {
        Var al( trace, a.length, FalseRef );
        Var bl( trace, b.length, FalseRef );
        Var x = EmitBinary( TraceOpCode::le, al, bl, Type::Logical, 0 );
        Var y = EmitBinary( TraceOpCode::gt, al, Var(trace,0,FalseRef), Type::Logical, 0 );
        Var z = EmitBinary( TraceOpCode::land, x, y, Type::Logical, 0 );
        Emit( IR( TraceOpCode::gtrue, z.v, Type::Nil, Shape::Scalar, Shape::Empty), inst, true );
        shape = b;
    }
    else if(a.traceLength > b.traceLength) {
        Var al( trace, a.length, FalseRef );
        Var bl( trace, b.length, FalseRef );
        Var x = EmitBinary( TraceOpCode::le, bl, al, Type::Logical, 0 );
        Var y = EmitBinary( TraceOpCode::gt, bl, Var(trace,0,FalseRef), Type::Logical, 0 );
        Var z = EmitBinary( TraceOpCode::land, x, y, Type::Logical, 0 );
        Emit( IR( TraceOpCode::gtrue, z.v, Type::Nil, Shape::Scalar, Shape::Empty), inst, true );
        shape = a;
    }
    return shape;
}

JIT::Var JIT::EmitBinary(TraceOpCode::Enum op, Var a, Var b, Type::Enum rty, Instruction const* inst) {
    Shape shape = MergeShapes(a.s, b.s, inst);
   
    a = EmitBroadcast(a, shape);
    b = EmitBroadcast(b, shape);
 
    if(op == TraceOpCode::lor) {
        IRRef r = Emit( IR( op, a.v, b.v, rty, shape, shape ) ); 
        IRRef x = Emit( IR( TraceOpCode::land, a.na, b.na, Type::Logical, shape, shape ) );
        IRRef y = Emit( IR( TraceOpCode::lnot, b.v, Type::Logical, shape, shape ) );
              y = Emit( IR( TraceOpCode::land, a.na, y, Type::Logical, shape, shape ) );
        IRRef z = Emit( IR( TraceOpCode::lnot, a.v, Type::Logical, shape, shape ) );
              z = Emit( IR( TraceOpCode::land, b.na, y, Type::Logical, shape, shape ) );
        IRRef n = Emit( IR( TraceOpCode::lor, x, y, Type::Logical, shape, shape ) );
              n = Emit( IR( TraceOpCode::lor, n, z, Type::Logical, shape, shape ) );
        return Var( trace, r, n );
    }
    else if(op == TraceOpCode::land) {
        IRRef r = Emit( IR( op, a.v, b.v, rty, shape, shape ) ); 
        IRRef x = Emit( IR( TraceOpCode::land, a.na, b.na, Type::Logical, shape, shape ) );
        IRRef y = Emit( IR( TraceOpCode::land, a.na, b.v, Type::Logical, shape, shape ) );
        IRRef z = Emit( IR( TraceOpCode::land, b.na, a.v, Type::Logical, shape, shape ) );
        IRRef n = Emit( IR( TraceOpCode::lor, x, y, Type::Logical, shape, shape ) );
              n = Emit( IR( TraceOpCode::lor, n, z, Type::Logical, shape, shape ) );
        return Var( trace, r, n );
    }
    else {
        return Var( trace,
            Emit( IR( op, a.v, b.v, rty, shape, shape ) ), 
            Emit( IR( TraceOpCode::lor, a.na, b.na, Type::Logical, shape, shape ) ) );
    }
}

JIT::Var JIT::EmitTernary(TraceOpCode::Enum op, Var a, Var b, Var c, Type::Enum rty, Instruction const* inst) {
    Shape shape = MergeShapes(a.s, MergeShapes(b.s, c.s, inst), inst);
    
    a = EmitBroadcast(a, shape);
    b = EmitBroadcast(b, shape);
    c = EmitBroadcast(c, shape);

    if(op == TraceOpCode::ifelse) {
        IRRef n = Emit( IR( TraceOpCode::ifelse, a.v, b.na, c.na, Type::Logical, shape, shape ) );
              n = Emit( IR( TraceOpCode::lor, a.na, n, Type::Logical, shape, shape ) );
        return Var( trace, Emit( IR( op, a.v, b.v, c.v, rty, shape, shape ) ), n);
    }
    else { 
        IRRef n = Emit( IR( TraceOpCode::lor, a.na, b.na, Type::Logical, shape, shape ) );
              n = Emit( IR( TraceOpCode::lor, n, c.na, Type::Logical, shape, shape ) );
        return Var( trace, Emit( IR( op, a.v, b.v, c.v, rty, shape, shape ) ), n);
    }
}

JIT::Var JIT::EmitConstant(Value const& value) {
    IRRef r;
    IR a = makeConstant(value);
    if(uniqueConstants.find(a.a) != uniqueConstants.end())
        r = uniqueConstants[a.a];
    else {
        r = Emit(a);
        uniqueConstants[a.a] = r;
    }

          r = Emit( IR( TraceOpCode::decodevl, r, trace[r].type, trace[r].out, trace[r].out ) );
    IRRef n = Emit( IR( TraceOpCode::decodena, r, Type::Logical, trace[r].out, trace[r].out ) );    
    return Var( trace, r, n );    
}

bool JIT::EmitNest(Thread& thread, Trace* t) {
    Emit( IR( TraceOpCode::nest, (IRRef)t, Type::Nil, Shape::Empty, Shape::Empty ) );
    return true;
}

bool JIT::EmitIR(Thread& thread, Instruction const& inst, bool branch) {
    switch(inst.bc) {

        case ByteCode::loop: {
        } break;
        case ByteCode::jc: {
            Var p = EmitCast( load(thread, inst.c, &inst), Type::Logical );
           
            // TODO: guard length==1 and that condition is not an NA
            /*IRRef len = Emit( IR( TraceOpCode::eq, p.s.length, 1, Type::Logical, Shape::Scalar, Shape::Scalar) );
            Emit( IR( TraceOpCode::gtrue, len, Type::Nil, trace[len].out, Shape::Empty ) ); 
            
            IRRef notna = Emit( IR( TraceOpCode::eq, trace[p].na, Type::Logical, Shape::Scalar, Shape::Scalar) );
            Emit( IR( TraceOpCode::gtrue, notna, Type::Nil, trace[notna].out, Shape::Empty ) ); 
            */
            if(inst.c <= 0) {
                Variable v = { -1, (thread.base+inst.c)-(thread.registers+DEFAULT_NUM_REGISTERS)};
                Emit( IR( TraceOpCode::kill, v.i, Type::Nil, Shape::Empty, Shape::Empty ) );
            }

            Emit( IR ( branch ? TraceOpCode::gtrue : TraceOpCode::gfalse, 
                    p.v, Type::Nil, p.s, Shape::Empty ),
                &inst + (branch ? inst.b : inst.a), (inst.a>=0&&inst.b>0) );
        }   break;
    
        case ByteCode::constant: {
            Value const& c = thread.frame.prototype->constants[inst.a];
            store(thread, EmitConstant(c), inst.c);
        }   break;

        case ByteCode::mov:
        case ByteCode::fastmov: {
            store(thread, load(thread, inst.a, &inst), inst.c);
        }   break;

        case ByteCode::assign: {
            store(thread, load(thread, inst.c, &inst), inst.a);
        }   break;

        case ByteCode::get: {
            Var cc = EmitConstant(Character::c((String)inst.a));
            Var e = load(thread, inst.b, &inst);
   
            // TODO: guard NAs

            OPERAND(env, inst.b);     
            Value const& operand = ((REnvironment&)env).environment()->get((String)inst.a);
            
            IRRef r = Emit( IR( TraceOpCode::load, e.v, cc.v, operand.type, Shape::Empty, Shape::Scalar ) );
            Shape s = SpecializeValue(operand, r); 
            IRRef v = Emit( IR( TraceOpCode::unbox, r, operand.type, Shape::Scalar, s ), &inst, true );
            IRRef na = SpecializeNA(operand, v);
            store(thread, Var(trace, r, na), inst.c);
        }   break;

        case ByteCode::gassign: {
            Var cc = EmitConstant(Character::c((String)inst.a));
            Var e = load(thread, inst.b, &inst);
            Var v = load(thread, inst.c, &inst);

            IRRef r = Emit( IR( TraceOpCode::encode, v.v, v.na, v.type, v.s, v.s ) );
                  r = Emit( IR( TraceOpCode::box, r, Type::Any, v.s, Shape::Scalar ) );

            Emit( IR( TraceOpCode::store, e.v, cc.v, r, Type::Nil, v.s, Shape::Empty ) );
            store(thread, e, inst.c);
        }   break;

        case ByteCode::gather1: {
        case ByteCode::gather:
            // TODO: need to check for negative numbers, out of range accesses, logical gather vectors, etc.
            Var a = load(thread, inst.a, &inst);
            Var b = EmitCast(load(thread, inst.b, &inst), Type::Integer);
            b = EmitBinary( TraceOpCode::sub, b, Var(trace, 1, FalseRef), Type::Integer, 0 );

            IRRef n = Emit( IR( TraceOpCode::gather, a.na, b.v, Type::Logical, b.s, b.s ) );
                  n = Emit( IR( TraceOpCode::lor, b.na, n, Type::Logical, b.s, b.s ) );
            IRRef r = Emit( IR( TraceOpCode::gather, a.v, b.v, a.type, b.s, b.s ) ); 
            store(thread, Var( trace, r, n ), inst.c); 
        }   break;

        case ByteCode::scatter1: {
        case ByteCode::scatter:
            // TODO: Need to check for NAs in the index vector
            Var a = load(thread, inst.a, &inst);
            Var b = EmitCast(load(thread, inst.b, &inst), Type::Integer);
            b = EmitBinary( TraceOpCode::sub, b, Var(trace, 1, FalseRef), Type::Integer, 0 );
            Var c = load(thread, inst.c, &inst);
            
            IRRef len = Emit( IR( TraceOpCode::reshape, b.v, c.s.length, Type::Integer, b.s, Shape::Scalar ) );

            // TODO: needs to know the recorded reshaped length for matching with other same 
            // sized shapes
            Shape reshaped( len, false, 0 );
            
            Shape s = MergeShapes(a.s, b.s, &inst);

            a = EmitBroadcast( a, s );
            b = EmitBroadcast( b, s );

            IRRef n = Emit( IR( TraceOpCode::scatter, c.na, b.v, a.na, Type::Logical, s, reshaped ) );
            IRRef r = Emit( IR( TraceOpCode::scatter, c.v, b.v, a.v, c.type, s, reshaped ) );
            store(thread, Var( trace, r, n ), inst.c);
        }   break;

        case ByteCode::ifelse: {
            Var a = load(thread, inst.a, &inst);
            Var b = load(thread, inst.b, &inst);
            Var c = load(thread, inst.c, &inst);
            store(thread, EmitTernary<IfElse>(TraceOpCode::ifelse, c, b, a, &inst), inst.c);
        }   break;

        #define EMIT(Name, string, Group, ...)                      \
        case ByteCode::Name: {                              \
            Var a = load(thread, inst.a, &inst);          \
            Var r = EmitUnary<Group>(TraceOpCode::Name, a);  \
            store(thread, r, inst.c);  \
        }   break;
        UNARY_BYTECODES(EMIT)
        #undef EMIT

        #define EMIT(Name, string, Group, ...)                      \
        case ByteCode::Name: {                              \
            Var a = load(thread, inst.a, &inst);          \
            Var b = load(thread, inst.b, &inst);          \
            Var r = EmitBinary<Group>(TraceOpCode::Name, a, b, &inst); \
            store(thread, r, inst.c);  \
        }   break;
        BINARY_BYTECODES(EMIT)
        #undef EMIT

        #define EMIT(Name, string, Group, ...)                      \
        case ByteCode::Name: {                              \
            Var a = load(thread, inst.a, &inst);          \
            Var r = EmitFold<Group>(TraceOpCode::Name, a);  \
            store(thread, r, inst.c);  \
        }   break;
        FOLD_BYTECODES(EMIT)
        #undef EMIT
        
        case ByteCode::length:
        {
            Var a = load(thread, inst.a, &inst);
            store(thread, Var( trace, a.s.length, FalseRef ), inst.c);
        }   break;

        case ByteCode::forbegin:
        {
            IRRef counter = 0;
            Var vec = load(thread, inst.b, &inst);
            IRRef a = vec.s.length;
            
            IRRef b = Emit( IR( TraceOpCode::lt, counter, a, Type::Logical, Shape::Scalar, Shape::Scalar ) );
            Emit( IR( TraceOpCode::gtrue, b, Type::Nil, Shape::Scalar, Shape::Empty), &inst+(&inst+1)->a, false );

            IRRef r = Emit( IR( TraceOpCode::gather, vec.v, counter, vec.type, Shape::Scalar, Shape::Scalar ) );
            IRRef n = Emit( IR( TraceOpCode::gather, vec.na, counter, Type::Logical, Shape::Scalar, Shape::Scalar ) );

            store(thread, Var( trace, r, n ), inst.a);
            store(thread, Var( trace, 1, FalseRef ), inst.c); 
        }   break;

        case ByteCode::forend:
        {
            IRRef counter = load(thread, inst.c, &inst).v;
            Var vec = load(thread, inst.b, &inst);
            IRRef a = vec.s.length;

            IRRef b = Emit( IR( TraceOpCode::lt, counter, a, Type::Logical, Shape::Scalar, Shape::Scalar ) );
            Emit( IR( TraceOpCode::gtrue, b, Type::Nil, Shape::Scalar, Shape::Empty), &inst+2, false );
            
            IRRef r = Emit( IR( TraceOpCode::gather, vec.v, counter, vec.type, Shape::Scalar, Shape::Scalar ) );
            IRRef n = Emit( IR( TraceOpCode::gather, vec.na, counter, Type::Logical, Shape::Scalar, Shape::Scalar ) );

            store(thread, Var( trace, r, n ), inst.a);

            counter = Emit( IR( TraceOpCode::add, counter, 1, Type::Integer, Shape::Scalar, Shape::Scalar ) );
            store(thread, Var( trace, counter, FalseRef ), inst.c); 
        }   break;

        case ByteCode::strip:
        {
            /*OPERAND(val, inst.a);
            if(val.isObject()) {
                IRRef a = load(thread, inst.a, &inst);

                Shape s = SpecializeValue(((Object const&)val).base(), 
                    IR(TraceOpCode::olength, a, Type::Integer, Shape::Empty, Shape::Scalar));
                IRRef g = Emit( IR( TraceOpCode::load, a, ((Object const&)val).base().type, Shape::Scalar, s ), &inst, true );
                store(thread, g, inst.c);
            }
            else {
                store(thread, load(thread, inst.a, &inst), inst.c);
            }*/
            _error("strip NYI in trace");
        }   break;

        case ByteCode::nargs:
        {
            store(thread, EmitConstant(Integer::c(thread.frame.environment->call.length-1)), inst.c);
        }   break;

        case ByteCode::attrget:
        {
            _error("attrget NYI in trace");
            /*OPERAND(object, inst.a);
            OPERAND(whichTmp, inst.b);
            
            if(object.isObject()) {
                Value r;
                Character which = As<Character>(thread, whichTmp);
                r = ((Object const&)object).get(which[0]);
            
                IRRef name = EmitCast(load(thread, inst.b, &inst), Type::Character);

                Shape s = SpecializeValue(r, IR(TraceOpCode::alength, load(thread, inst.a, &inst), name, Type::Integer, Shape::Empty, Shape::Scalar));
                
                IRRef g = insert(trace, TraceOpCode::load, load(thread, inst.a, &inst), name, 0, r.type, Shape::Empty, s);
                trace[g].reenter = (Reenter) { &inst, true };
                store(thread, g, inst.c);
            }
            else {
                store(thread, EmitConstant(Null::Singleton()), inst.c);
            }*/
        }   break;

        case ByteCode::attrset:
        {
            _error("attrset NYI in trace");
            // need to make this an object if it's not already
            /*store(thread, insert(trace, TraceOpCode::store,
                load(thread, inst.c, &inst),
                load(thread, inst.b, &inst),
                load(thread, inst.a, &inst),
                Type::Object, Shape::Scalar, Shape::Empty), inst.c);
            */ 
        }   break;

        case ByteCode::missing:
        {
            String s = (String)inst.a;
            Value const& v = thread.frame.environment->get(s);
            bool missing = v.isNil() || v.isDefault();
            store(thread, EmitConstant(Logical::c(missing ? Logical::TrueElement : Logical::FalseElement)), inst.c);
        }   break;

        case ByteCode::rep:
        {
            OPERAND(len, inst.a);
            Var l = EmitCast(load(thread, inst.a, &inst), Type::Integer);
            Shape s = SpecializeLength(As<Integer>(thread, len)[0], l.v);
           
            Var b = load(thread, inst.b, &inst); 
            Var c = load(thread, inst.c, &inst); 
 
            store(thread, EmitRep(c, b, s), inst.c);
        }   break;
        case ByteCode::seq:
        {
            OPERAND(len, inst.a);
            Var l = EmitCast(load(thread, inst.a, &inst), Type::Integer);
            Shape s = SpecializeLength(As<Integer>(thread, len)[0], l.v);
            // requires a dependent type
            Var c = load(thread, inst.c, &inst);
            Var b = load(thread, inst.b, &inst);
            Type::Enum type = c.type == Type::Double || b.type == Type::Double
                ? Type::Double : Type::Integer; 
                
            IRRef r = Emit( IR( TraceOpCode::seq, EmitCast(c, type).v, EmitCast(b, type).v, type, s, s) );
            store(thread, Var( trace, r, FalseRef ), inst.c); 
        }   break;

        case ByteCode::call:
        case ByteCode::ncall:
            // nothing since it's currently
            break;

        case ByteCode::newenv: 
            {
            IRRef r = Emit( IR( TraceOpCode::newenv, 
                    load(thread, inst.a, &inst).v,
                    load(thread, inst.a, &inst).v,
                    EmitConstant(Null::Singleton()).v, Type::Environment, Shape::Scalar, Shape::Scalar) );
            store(thread, Var( trace, r, FalseRef ), inst.c);
            } break;
        case ByteCode::parentframe:
            {
                IRRef e = Emit( IR( TraceOpCode::curenv, Type::Environment, Shape::Empty, Shape::Scalar ) );
                IRRef r = Emit( IR( TraceOpCode::denv, e, Type::Environment, Shape::Scalar, Shape::Scalar ) );
                store(thread, Var(trace, r, FalseRef ), inst.c);
            } break;

        default: {
            if(thread.state.verbose)
                printf("Trace halted by %s\n", ByteCode::toString(inst.bc));
            return false;
        }   break;
    }
    return true;
}

void JIT::Replay(Thread& thread) {
   
    code.clear();
    exits.clear();
 
    size_t n = trace.size();
    
    std::vector<IRRef> forward(n, 0);
    std::tr1::unordered_map<IR, IRRef> cse;
    Snapshot snapshot;

    // after each guard reemit the entire body of the code
    // up to that point, omitting all guards.
    // conceptually this gives us all possible sinking locations
    // how do we keep the size from getting out of control?
        // eliminate as many guards as possible
    // avoid the order N^2?
    /*
        replay just the sunk operations.
        means I'd need to figure out what to sink after emiting guard
        not all stores can be sunk.
            
        I can compute the CSE cost when emiting the first set.
        If forwarding is not profitable, put in 

        what can be sunk?
            not any operations needed to evaluate the guard condition
            not any loads or constants
            

        want to do cost-driven CSE, is this all I need?
        

          load a
        0 seq i 100         // if I use this one it means that all branches share this instance.
          store a
        1 lt i 50
        2 gtrue   ->
        
          load a
        3 seq i 100         // if I use this one it means the original is either dead or only computed on the side exit
          store a
        4 lt i 50     (xx CSEd or DCE)
        5 sum 0
          store b
        6 lt 100
        7 gtrue   ->
        
        What if I just replayed the stores?
        All stores before guard get marked as sunk
        Do DSE on reemitted stores.

        Want to eliminate stores on the fast path
        >store a
        guard => store is needed here
        store a -> this store makes the previous one dead and is not needed?
        phi
        jmp

        What stores can be sunk?
        
        store global "a"   => this can because it aliases itself in all previous iterations
                           => DSE applied to loop carried global
        guard 

        compute key
        newenv blah
        store blah computed key
        >store global "a" blah
        guard              => this can because last store dominates, now there's no use of blah
                              in main path.
        
        What stores can't be sunk?
        loop
        compute key
        store global computedkey  => this can't because it only may alias
                                  => previous stores aren't dead.
        guard
        jmp

        store global a foo   => this can be sunk since it is dominated by the following store
        guard
        store global a bar

        DSE:
            if we haven't crossed a guard, the store becomes a NOP
            if we have crossed a guard, the store becomes a SUNK store

            >store global a foo
            guard
            loop
            ...
            >store global a bar
            guard
            jmp

            
            loop
            >newenv blah
            >store blah bar
            guard
            jmp
    */

    Loop = 0; 
 
    forward[0] = 0;
    forward[1] = 1;
    forward[2] = 2;
 
    // Emit constants
    for(size_t i = 0; i < n; i++) {
        if(trace[i].op == TraceOpCode::constant) {
            forward[i] = EmitOptIR(thread, trace[i], code, forward, cse, snapshot);
        }
    }
 
    // Emit loop header...
    for(size_t i = 0; i < n; i++) {
        forward[i] = EmitOptIR(thread, trace[i], code, forward, cse, snapshot);
    }

    if(rootTrace == 0) 
    {
        Loop = Insert(thread, code, cse, snapshot, IR(TraceOpCode::loop, Type::Nil, Shape::Empty, Shape::Empty));

        std::map<IRRef, IRRef> phis;

        // Emit loop
        for(size_t i = 0; i < n; i++) {
            IRRef fwd = EmitOptIR(thread, trace[i], code, forward, cse, snapshot);
           
            if(code[fwd].op != TraceOpCode::constant) {
                if(fwd < Loop && phis.find(fwd) == phis.end())
                    phis[fwd] = fwd;

                if(phis.find(forward[i]) != phis.end())
                    phis[forward[i]] = fwd;
            }
            forward[i] = fwd;
        }

        size_t actualSize = code.size();

        code.resize(actualSize);

        for(std::map<IRRef, IRRef>::const_iterator i = phis.begin(); i != phis.end(); ++i) {
            IR const& ir = code[i->first];
            Insert(thread, code, cse, snapshot, IR(TraceOpCode::phi, i->first, i->second, ir.type, ir.out, Shape::Empty));
        }

        // Emit the JMP
        Insert(thread, code, cse, snapshot, IR(TraceOpCode::jmp, Type::Nil, Shape::Empty, Shape::Empty));
    }
    else {
        IR exit( TraceOpCode::exit, Type::Nil, Shape::Empty, Shape::Empty);
        exit.reenter = Reenter( startPC, true );
        Insert(thread, code, cse, snapshot, exit);
    }
}

void JIT::end_recording(Thread& thread) {

    assert(state == RECORDING);
    state = OFF;

    // mark trace live so it'll print out
    for(size_t i = 0; i < trace.size(); i++) {
        trace[i].live = true;
        trace[i].reg = 0;
    }

    //dump(thread, trace);
    Replay(thread);
    
    for(std::map<size_t, Exit>::const_iterator i = exits.begin(); i != exits.end(); ++i) {
        Trace tr;
        tr.traceIndex = Trace::traceCount++;
        tr.Reenter = i->second.reenter.reenter;
        tr.InScope = i->second.reenter.inScope;
        tr.counter = 0;
        tr.ptr = 0;
        tr.function = 0;
        tr.root = dest->root;
        assert(i->second.index == dest->exits.size());
        dest->exits.push_back(tr);
    }

    Liveness();
    SINK();
    //Schedule();
    schedule();
    RegisterAssignment();

    // add the tail exit for side traces
    if(rootTrace) {
        dest->exits.back().function = rootTrace->function;
    }

    if(thread.state.verbose) {
        printf("---------------- Trace %li ------------------\n", dest->traceIndex);
        dump(thread, code);
    } 

    compile(thread);
}

void JIT::specialize() {
    // basically, we want to score how valuable a particular specialization
    // (replacing a load with a EmitConstant) might be.
    // Only worth doing on loads in the loop header.
    // Valuable things:
    //  1) Eliminating a guard to enable fusion.
    //  2) Turn unvectorized op into a vectorized op
    //      a) Lowering gather to shuffle
    //      b) Lowering pow to vectorized mul or sqrt
    //  3) Making a size EmitConstant (e.g. out of a filter)
    // 
    //  Might be target specific
    //
    // Valuable is a tradeoff between reuse and benefit.
    //  How to judge?
    //  Not valuable for very long vectors or scalars.
    //  Valuable for small multiples of HW vector length,
    //      where we can unroll the loop completely.
    //  Unless the entire vector is a EmitConstant
/*
    size_t n = code.size();
    std::vector<IR> out;
    std::vector<IRRef> forward(n, -1);
    std::vector<size_t> ngroup;
    std::map<size_t, Exit> nexits;
    std::map<Variable, IRRef> loads;
    std::map<Variable, IRRef> stores;
    std::tr1::unordered_map<IR, IRRef> cse;

    for(int g = maxGroup; g >= 0; g--) {
        for(size_t i = 0; i < n; i++) {
            if(group[i] == g) {
                EmitOptIR(i, code[i], out, forward, loads, stores, cse);
                ngroup.push_back(group[i]);
                if(exits.find(i) != exits.end()) {
                    // add compensation code
                    Exit e = exits[i];
                    e.compensation.clear();
                    printf("Exit initial %d\n", i);
                    for(int k = 0; k < forward[i]; k++)
                        e.compensation.push_back(IR(TraceOpCode::nop, Type::Promise, Shape::Empty, Shape::Empty));
                    std::vector<IRRef> eforward = forward;
                    std::map<Variable, IRRef> loads = loads;
                    std::map<Variable, IRRef> estores = stores;
                    std::tr1::unordered_map<IR, IRRef> ecse = cse;
                    for(size_t k = 0; k < i; k++) {
                        if(forward[k] == -1) {
                            printf("Compensation: %d\n", k);
                            EmitOptIR(k, code[k], e.compensation, eforward, loads, estores, ecse);
                        }
                    }
                    for(std::map<Variable, IRRef>::iterator k = e.o.begin(); k != e.o.end(); ++k) {
                        k->second = eforward[k->second];
                    }
                    nexits[out.size()-1] = e;  
                }           
            }
        }
    }

    // iterate through the exits. If the code was before the exit before and is now after,
    // add to the compensation list.
    for(std::map<size_t, Exit>::const_iterator j = exits.begin(); j != exits.end(); ++j) {
        for(size_t i = 0; i < n; i++) {
            if(i < j->first && forward[i] > forward[j->first]) {
                
            }
        }
    }

 
    code = out;
    group = ngroup;
    exits = nexits;
*/
}

bool JIT::Ready(JIT::IR ir, std::vector<bool>& done) {
        switch(ir.op) {
            #define CASE(Name, ...) case TraceOpCode::Name:
            case TraceOpCode::sload:
            case TraceOpCode::curenv:
            case TraceOpCode::phi:
            case TraceOpCode::loop: 
            case TraceOpCode::constant: 
                return true;
                break;
            UNARY_BYTECODES(CASE) 
            FOLD_BYTECODES(CASE) 
            case TraceOpCode::gproto: 
            case TraceOpCode::gtrue: 
            case TraceOpCode::gfalse:
            case TraceOpCode::load: {
                return done[ir.a];
            } break; 
            BINARY_BYTECODES(CASE)
            case TraceOpCode::gather:
            case TraceOpCode::rep: {
                return done[ir.a] && done[ir.b];
            } break;
            TERNARY_BYTECODES(CASE)
            case TraceOpCode::scatter:
                return done[ir.a] && done[ir.b] && done[ir.c];
            break;
            case TraceOpCode::jmp:
                return false;
            break;
            default:
                printf("Unknown op is %s\n", TraceOpCode::toString(ir.op));
                _error("Unknown op in Ready");
                break;
            #undef CASE
        }
}

size_t Score(JIT::IR ir) {
    return ir.in.length;
}

void JIT::Schedule() {
   
    // Scheduling...want to move move unused ops down into side traces...
    // Linear scheduling doesn't do this aggressively enough.
/* 
    size_t n = code.size();
    std::vector<IR> out;
    std::vector<IRRef> forward(n, -1);
    std::map<size_t, Exit> nexits;
    std::map<Variable, IRRef> loads;
    std::map<Variable, IRRef> stores;
    std::tr1::unordered_map<IR, IRRef> cse;

    // find everything that has no dependencies...
    std::vector<bool> ready(n, false);
    std::vector<bool> done(n, false);

    IRRef best = 0;
    size_t score = 1000000000000;
    do {
        // Update the ready list
        for(IRRef i = best; i != Loop; i++) {
            if(!ready[i]) {
                ready[i] = Ready(code[i], done);
            }
        }
        // Select the best instruction and put it in the done list
        score = 1000000000000;
        for(IRRef i = 0; i != Loop; i++) {
            if(ready[i] && !done[i]) {
                size_t s = Score(code[i]);
                if(s < score) {
                    score = s;
                    best = i;
                }
            }
        }

        if(score != 1000000000000) {
            EmitOptIR(best, code[best], out, forward, loads, stores, cse);
            done[best] = true;
        }
    } while(score != 1000000000000);

    // emit the loop instruction and phis
    Insert(out, cse, IR(TraceOpCode::loop, Type::Promise, Shape::Empty, Shape::Empty));

    best = Loop+1;
    //for(; code[best].op == TraceOpCode::phi; best++) {
    //    EmitOptIR(best, code[best], out, forward, loads, stores, cse);
    //    done[best] = ready[best] = true;
    //}

    do {
        // Update the ready list
        for(IRRef i = best; i != n; i++) {
            if(!ready[i]) {
                ready[i] = Ready(code[i], done);
            }
        }
        // Select the best instruction and put it in the done list
        score = 1000000000000;
        for(IRRef i = Loop+1; i != n; i++) {
            if(ready[i] && !done[i]) {
                size_t s = Score(code[i]);
                if(s < score) {
                    score = s;
                    best = i;
                }
            }
        }

        if(score != 1000000000000) {
            EmitOptIR(best, code[best], out, forward, loads, stores, cse);
            done[best] = true;
        }
    } while(score != 1000000000000);

    // update the exits with the new instruction locations...
    for(std::map<size_t, Exit>::const_iterator i = exits.begin(); i != exits.end(); ++i) {
        Exit e = i->second;
        for(std::map<Variable, IRRef>::iterator j = e.o.begin(); j != e.o.end(); ++j) {
            j->second = forward[j->second];
        }
        nexits[forward[i->first]] = e;
    }

    // Emit the JMP
    Insert(out, cse, IR(TraceOpCode::jmp, Type::Promise, Shape::Empty, Shape::Empty));
    code = out; 
    exits = nexits;
    */
}


void JIT::schedule() {

    // do a forwards pass identifying fusion groups.
    Shape gSize(-1, false, -1);
    std::set<IRRef> gMembers;

    fusable = std::vector<bool>(code.size(), true);

    for(IRRef i = 0; i < code.size(); i++) {
         if( code[i].in != gSize
          || (code[i].op == TraceOpCode::scatter && gMembers.find(code[i].c) != gMembers.end())
          || (code[i].op == TraceOpCode::gather  && gMembers.find(code[i].c) != gMembers.end())
          || code[i].op == TraceOpCode::gtrue 
          || code[i].op == TraceOpCode::gfalse 
          || code[i].op == TraceOpCode::load 
          || code[i].op == TraceOpCode::sload ) {
            fusable[i] = false;
            gSize = code[i].in;
            gMembers.clear();
        }
        gMembers.insert(i);
        if(code[i].op == TraceOpCode::gather)
            gMembers.insert(code[i].b);
        if(code[i].op == TraceOpCode::scatter)
            gMembers.insert(code[i].c);
    }

    // do a backwards pass, assigning instructions to a fusion group.
    // this happens after all optimization and specialization decisions
    //  have been made.

    // Problem: Gathers and scatters to same vector can't be fused. How to assert?
    // Unless in different registers and a whole copy occurs.

    /*

        fusion and register assignment interact
        
        True dependency. Write has to complete before read starts.
            0: SCATTER a
            1: GATHER a
        Can't fuse generally

        Anti-dependency: Read comes before write.
            0: GATHER a
            1: SCATTER a
        Can fuse if scatter writes to distinct register, requires copy of a on loop backedge.
        Alternatively: not fuse, intermediate (size) must be written out and read back in.

        Store-store dependency: Write after write
            0: SCATTER a
            1: SCATTER b
        Can't fuse generally, unless scatter kills entire thing

        GATHER-GATHER is fine.
    */

    /* replace with forward reordering */
    
}

void JIT::IR::dump() const {
    if(type != Type::Nil)
        printf("  %.3s  ", Type::toString(type));
    else
        printf("       ");
   
    printf("%2li->%2li", in.length, out.length); 
    std::cout << "\t" << TraceOpCode::toString(op);

    switch(op) {
        #define CASE(Name, ...) case TraceOpCode::Name:
        case TraceOpCode::loop: {
            std::cout << " --------------------------------";
        } break;
        case TraceOpCode::sload: {
            std::cout << "\t " << (int64_t)b << "\t \t";
        } break;
        case TraceOpCode::sstore: {
            std::cout << "\t " << (int64_t)b << "\t " << c;
        } break;    
        case TraceOpCode::gproto:
        {
            std::cout << "\t " << a << "\t [" << b << "]";
        } break;
        case TraceOpCode::kill:
            std::cout << "\t " << (int64_t)a << "\t \t";
            break;
        case TraceOpCode::box:
        case TraceOpCode::unbox:
        case TraceOpCode::brcast:
        case TraceOpCode::length:
        case TraceOpCode::gtrue:
        case TraceOpCode::gfalse: 
        case TraceOpCode::olength: 
        case TraceOpCode::lenv: 
        case TraceOpCode::denv: 
        case TraceOpCode::cenv: 
        case TraceOpCode::decodena:
        case TraceOpCode::decodevl:
        UNARY_FOLD_SCAN_BYTECODES(CASE)
        {
            std::cout << "\t " << a << "\t \t";
        } break;
        case TraceOpCode::encode:
        case TraceOpCode::reshape:
        case TraceOpCode::phi: 
        case TraceOpCode::load:
        case TraceOpCode::push:
        case TraceOpCode::rep:
        case TraceOpCode::seq:
        case TraceOpCode::gather:
        case TraceOpCode::alength:
        BINARY_BYTECODES(CASE)
        {
            std::cout << "\t " << a << "\t " << b << "\t";
        } break;
        case TraceOpCode::newenv:
        case TraceOpCode::store:
        case TraceOpCode::scatter:
        TERNARY_BYTECODES(CASE)
        {
            std::cout << "\t " << a << "\t " << b << "\t " << c;
        } break;
        default: {} break;

        #undef CASE
    };
}

void JIT::dump(Thread& thread, std::vector<IR> const& t) {

    for(size_t i = 0; i < t.size(); i++) {
        IR const& ir = t[i];
        if(ir.live) {
            printf("%4li ", i);
            if(ir.sunk)
                printf("}");
            else
                printf(" ");
            if(exits.find(i) != exits.end())
                printf(">");
            else if(fusable.size() == t.size() && !fusable[i]) 
                printf("-");
            else
                printf(" ");
            if(ir.reg > 0) 
                printf(" %2d ", ir.reg);
            else if(ir.reg < 0)
                printf(" !! ");
            else
                printf("    ");
            ir.dump();
    
            if(ir.op == TraceOpCode::constant) {
                std::cout <<  "    " << thread.deparse(constants[ir.a]);
            }
            if(exits.find(i) != exits.end()) {
                std::cout << "\t\t-> " << dest->exits[exits[i].index].traceIndex;
            }

            if(ir.op == TraceOpCode::nest) {
                std::cout << "\t\t\t\t\t-> " << ((Trace*)ir.a)->traceIndex;
            }
            
            std::cout << std::endl;
        }
    }
    std::cout << std::endl;
}

