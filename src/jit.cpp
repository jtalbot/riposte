
#include "jit.h"
#include "bc.h"
#include "interpreter.h"
#include "call.h"
#include "ops.h"

DEFINE_ENUM_TO_STRING(TraceOpCode, TRACE_ENUM)

const JIT::Shape JIT::Shape::Empty = { 0, 0 };
const JIT::Shape JIT::Shape::Scalar = { 1, 1 };

JIT::IRRef JIT::insert(
        std::vector<IR>& t,
        TraceOpCode::Enum op, 
        IRRef a, 
        IRRef b, 
        IRRef c,
        Type::Enum type, 
        Shape in,
        Shape out) {
    IR ir = (IR) { op, a, b, c, type, in, out };
    t.push_back(ir);
    return (IRRef) { t.size()-1 };
}

JIT::Shape JIT::SpecializeLength(size_t length, IRRef irlength) {
    // if short enough, guard length and insert a constant length instead
    if(length <= SPECIALIZE_LENGTH) {
        IRRef s = constant(Integer::c(length));
        insert(trace, TraceOpCode::glenEQ, std::min(irlength,s), std::max(irlength,s), 0, Type::Promise, Shape::Scalar, Shape::Empty);
        return Shape(s, length);
    }
    else {
        return Shape(irlength, length);
    }
}

JIT::Shape JIT::SpecializeValue(Value const& v, IR ir) {
    if(v.isNull())
        return Shape::Empty;
    else if(v.isVector()) {
        trace.push_back(ir);
        return SpecializeLength((size_t)v.length, trace.size()-1);
    }
    else
        return Shape::Scalar;
}

/*

    LICM lifts load out of loop. Load does recursive look up beyond the loop scope.
    Inside loop we track scopes and can statically determine where look up succeeds.
    Thus, for each load, we know either it hits in a known locally created scope
    or it requires a load from outside the loop scope.

    Thus, in optimized IR, loads are either
        (1) from the loop-level scope
        (2) or are eliminated by load-store forwarding for local scopes.

*/

JIT::IRRef JIT::load(Thread& thread, int64_t a, Instruction const* reenter) {

    // registers
    OPERAND(operand, a);

    IRRef r;
    
    if(a <= 0) {
        Variable v = { -1, (thread.base+a)-(thread.registers+DEFAULT_NUM_REGISTERS)};
    
        Shape s = SpecializeValue(operand, IR(TraceOpCode::slength, (IRRef)-1, v.i, Type::Integer, Shape::Empty, Shape::Scalar));
        r = insert(trace, TraceOpCode::sload, (IRRef)-1, v.i, 0, operand.type, Shape::Empty, s);
    }
    else {
        Variable v = { getEnv(thread.frame.environment->findRecursive((String)a)), a };
        Shape s = SpecializeValue(operand, IR(TraceOpCode::elength, v.env, v.i, Type::Integer, Shape::Empty, Shape::Scalar));
        r = insert(trace, TraceOpCode::eload, v.env, v.i, 0, operand.type, Shape::Empty, s);
    }
    reenters[r] = reenter;
    return r;
}

JIT::IRRef JIT::store(Thread& thread, IRRef a, int64_t c) {
    if(c <= 0) {
        Variable v = { -1, (thread.base+c)-(thread.registers+DEFAULT_NUM_REGISTERS)};
        insert(trace, TraceOpCode::sstore, a, 0, v.i, trace[a].type, trace[a].out, Shape::Empty);
    }
    else {
        Variable v = { getEnv(thread.frame.environment), (int64_t)c };
        insert(trace, TraceOpCode::estore, a, v.env, v.i, trace[a].type, trace[a].out, Shape::Empty);
    }
    return a;
}

void JIT::emitPush(Thread const& thread) {
    StackFrame frame;
    frame.environment = getEnv(thread.frame.environment);
    frame.prototype = thread.frame.prototype;
    frame.returnpc = thread.frame.returnpc;
    frame.returnbase = thread.frame.returnbase;
    frame.dest = thread.frame.dest;
    frame.env = getEnv(thread.frame.env);
    IRRef a = insert(trace, TraceOpCode::PUSH, getEnv(thread.frame.environment), 0, 0, Type::Promise, Shape::Scalar, Shape::Empty);
    frames[a] = frame;
}

JIT::IRRef JIT::cast(IRRef a, Type::Enum type) {
    if(trace[a].type != type) {
        Shape s = trace[a].out;
        if(type == Type::Double)
            return insert(trace, TraceOpCode::asdouble, a, 0, 0, type, s, s);
        else if(type == Type::Integer)
            return insert(trace, TraceOpCode::asinteger, a, 0, 0, type, s, s);
        else if(type == Type::Logical)
            return insert(trace, TraceOpCode::aslogical, a, 0, 0, type, s, s);
        else if(type == Type::Character)
            return insert(trace, TraceOpCode::ascharacter, a, 0, 0, type, s, s);
        else
            _error("Unexpected cast");
    }
    else {
        return a;
    }
}

JIT::IRRef JIT::rep(IRRef a, Shape target) {
    if(trace[a].out != target) {
        IRRef l = trace[a].out.length;
        IRRef e = constant(Integer::c(1));
        IRRef r = insert(trace, TraceOpCode::rep, l, e, 0, trace[e].type, target, target);
        return insert(trace, TraceOpCode::gather, a, r, 0, trace[a].type, target, target);
    }
    else {
        return a;
    }
}

JIT::IRRef JIT::EmitUnary(TraceOpCode::Enum op, IRRef a, Type::Enum rty, Type::Enum mty) {
   return insert(trace, op, cast(a, mty), 0, 0, rty, trace[a].out, trace[a].out);
}

JIT::IRRef JIT::EmitFold(TraceOpCode::Enum op, IRRef a, Type::Enum rty, Type::Enum mty) {
   return insert(trace, op, cast(a, mty), 0, 0, rty, trace[a].out, Shape::Scalar);
}

JIT::Shape JIT::MergeShapes(Shape a, Shape b) {
    Shape shape = Shape::Empty;
    if(a == b) {
        shape = a;
    }
    else if(a == Shape::Empty || b == Shape::Empty) {
        shape = Shape::Empty;
    }
    else if(a.traceLength == b.traceLength) {
        insert(trace, TraceOpCode::glenEQ, std::min(a.length, b.length), std::max(a.length,b.length), 0,
                Type::Promise, Shape::Scalar, Shape::Empty);
        shape = a.length < b.length ? a : b;
    }
    else if(a.traceLength < b.traceLength) {
        insert(trace, TraceOpCode::glenLT, a.length, b.length, 0,
                Type::Promise, Shape::Scalar, Shape::Empty);
        shape = b;
    }
    else if(a.traceLength > b.traceLength) {
        insert(trace, TraceOpCode::glenLT, b.length, a.length, 0,
                Type::Promise, Shape::Scalar, Shape::Empty);
        shape = a;
    }
    return shape;
}

JIT::IRRef JIT::EmitBinary(TraceOpCode::Enum op, IRRef a, IRRef b, Type::Enum rty, Type::Enum maty, Type::Enum mbty) {
    // specialization depends on observed lengths. 
    //  If depedent length is the same, no need for a guard. We've already proved the lengths are equal
    //  If one of the lengths is zero, result length is also known, no need for guard.
    //  If equal, guard equality and continue.
    //  If unequal, guard less than
    Shape shape = MergeShapes(trace[a].out,trace[b].out);
    return insert(trace, op, rep(cast(a,maty),shape), rep(cast(b,mbty),shape), 0, rty, shape, shape);
}

JIT::IRRef JIT::EmitTernary(TraceOpCode::Enum op, IRRef a, IRRef b, IRRef c, Type::Enum rty, Type::Enum maty, Type::Enum mbty, Type::Enum mcty) {
    Shape s = MergeShapes(trace[a].out, MergeShapes(trace[b].out, trace[c].out));
    return insert(trace, op, rep(cast(a,maty),s), rep(cast(b,mbty),s), rep(cast(c,mcty),s), rty, s, s);
}

JIT::IRRef JIT::constant(Value const& value) {
    IRRef a;
    if(constantsMap.find(value) != constantsMap.end())
        a = constantsMap.find(value)->second;
    else {
        size_t ci = constants.size();
        constants.push_back(value);
        size_t len = value.isVector() ? value.length : 1;
        IRRef s = constant(Integer::c(len));
        a = insert(trace, TraceOpCode::constant, ci, 0, 0, value.type, Shape::Empty, Shape(s, len));
        constantsMap[value] = a;
    }
    return a;
}

bool JIT::EmitIR(Thread& thread, Instruction const& inst, bool branch) {
    switch(inst.bc) {

        case ByteCode::loop: {
        } break;
        case ByteCode::jc: {
            IRRef p = load(thread, inst.c, &inst);
            
            IRRef r = insert(trace, branch ? TraceOpCode::guardT : TraceOpCode::guardF, 
                p, 0, 0, Type::Promise, trace[p].out, Shape::Empty );
            reenters[r] = &inst;
            if(inst.a <= 0)
                insert(trace, TraceOpCode::kill, trace[p].a, 0, 0, Type::Promise, Shape::Empty, Shape::Empty); 
        }   break;
    
        case ByteCode::call:
        case ByteCode::ncall:
        {
            emitPush(thread);
        }   break;

        case ByteCode::constant: {
            Value const& c = thread.frame.prototype->constants[inst.a];
            store(thread, constant(c), inst.c);
        }   break;

        case ByteCode::mov:
        case ByteCode::fastmov: {
            store(thread, load(thread, inst.a, &inst), inst.c);
        }   break;

        case ByteCode::assign: {
            store(thread, load(thread, inst.c, &inst), inst.a);
        }   break;

        case ByteCode::gather1: {
        case ByteCode::gather:
            IRRef a = load(thread, inst.a, &inst);
            IRRef b = cast(load(thread, inst.b, &inst), Type::Integer);
            b = insert(trace, TraceOpCode::sub, b, rep(constant(Integer::c(1)), trace[b].out), 0, trace[b].type, trace[b].out, trace[b].out);
            store(thread, insert(trace, TraceOpCode::gather, a, b, 0, trace[a].type, trace[b].out, trace[b].out), inst.c);
        }   break;

        case ByteCode::scatter1: {
        case ByteCode::scatter:
            IRRef a = load(thread, inst.a, &inst);
            IRRef b = cast(load(thread, inst.b, &inst), Type::Integer);
            b = insert(trace, TraceOpCode::sub, b, rep(constant(Integer::c(1)), trace[b].out), 0, trace[b].type, trace[b].out, trace[b].out);
            IRRef c = load(thread, inst.c, &inst);
            Shape s = MergeShapes(trace[a].out, trace[b].out);
            store(thread, insert(trace, TraceOpCode::scatter, rep(a, s), rep(b, s), c, trace[c].type, s, trace[c].out), inst.c);
        }   break;

        case ByteCode::ifelse: {
            IRRef a = load(thread, inst.a, &inst);
            IRRef b = load(thread, inst.b, &inst);
            IRRef c = load(thread, inst.c, &inst);
            Shape s = MergeShapes(trace[a].out, MergeShapes(trace[b].out, trace[c].out));
            store(thread, EmitTernary<IfElse>(TraceOpCode::ifelse, rep(c,s), rep(b,s), rep(a,s)), inst.c);
        }   break;

        #define EMIT(Name, string, Group, ...)                      \
        case ByteCode::Name: {                              \
            IRRef a = load(thread, inst.a, &inst);          \
            store(thread, EmitUnary<Group>(TraceOpCode::Name, a), inst.c);  \
        }   break;
        UNARY_BYTECODES(EMIT)
        #undef EMIT

        #define EMIT(Name, string, Group, ...)                      \
        case ByteCode::Name: {                              \
            IRRef a = load(thread, inst.a, &inst);          \
            IRRef b = load(thread, inst.b, &inst);          \
            IRRef r = EmitBinary<Group>(TraceOpCode::Name, a, b); \
            store(thread, r, inst.c);  \
        }   break;
        BINARY_BYTECODES(EMIT)
        #undef EMIT

        #define EMIT(Name, string, Group, ...)                      \
        case ByteCode::Name: {                              \
            IRRef a = load(thread, inst.a, &inst);          \
            store(thread, EmitFold<Group>(TraceOpCode::Name, a), inst.c);  \
        }   break;
        FOLD_BYTECODES(EMIT)
        #undef EMIT
        
        case ByteCode::length:
        {
            IRRef a = load(thread, inst.a, &inst); 
            store(thread, insert(trace, TraceOpCode::length, a, 0, 0, Type::Integer, Shape::Scalar, Shape::Scalar), inst.c);
        }   break;

        case ByteCode::forend:
        {
            IRRef counter = load(thread, inst.c, &inst);
            IRRef vec = load(thread, inst.b, &inst);

            IRRef a = insert(trace, TraceOpCode::length, vec, 0, 0, Type::Integer, Shape::Scalar, Shape::Scalar);
            IRRef b = insert(trace, TraceOpCode::lt, counter, a, 0, Type::Logical, Shape::Scalar, Shape::Scalar);
            IRRef c = insert(trace, TraceOpCode::guardT, b, 0, 0, Type::Promise, Shape::Scalar, Shape::Empty);
            reenters[c] = &inst+2;
            store(thread, insert(trace, TraceOpCode::gather, vec, counter, 0, trace[vec].type, Shape::Scalar, Shape::Scalar), inst.a);
            store(thread, insert(trace, TraceOpCode::add, counter, constant(Integer::c(1)), 0, Type::Integer, Shape::Scalar, Shape::Scalar), inst.c); 
        }   break;

        case ByteCode::strip:
        {
            OPERAND(a, inst.a);
            if(a.isObject()) {
                Shape s = SpecializeValue(((Object const&)a).base(), IR(TraceOpCode::olength, load(thread, inst.a, &inst), Type::Integer, Shape::Empty, Shape::Scalar));
                store(thread, insert(trace, TraceOpCode::strip, load(thread, inst.a, &inst), 0, 0, ((Object const&)a).base().type, Shape::Scalar, s), inst.c);
            }
            else {
                store(thread, load(thread, inst.a, &inst), inst.c);
            }
        }   break;

        case ByteCode::nargs:
        {
            store(thread, constant(Integer::c(thread.frame.environment->call.length-1)), inst.c);
        }   break;

        case ByteCode::attrget:
        {
            OPERAND(object, inst.a);
            OPERAND(whichTmp, inst.b);
            Value r;
            if(object.isObject()) {
                Character which = As<Character>(thread, whichTmp);
                r = ((Object const&)object).get(which[0]);
            }
            else {
                r = Null::Singleton();
            }
            IRRef name = cast(load(thread, inst.b, &inst), Type::Character);
            Shape s = SpecializeValue(r, IR(TraceOpCode::alength, load(thread, inst.a, &inst), name, Type::Integer, Shape::Empty, Shape::Scalar));
            store(thread, insert(trace, TraceOpCode::attrget, load(thread, inst.a, &inst), name, 0, r.type, Shape::Empty, s), inst.c);
        }   break;

        case ByteCode::attrset:
        {
            store(thread, insert(trace, TraceOpCode::attrset,
                load(thread, inst.c, &inst),
                load(thread, inst.b, &inst),
                load(thread, inst.a, &inst),
                Type::Object, Shape::Scalar, Shape::Empty), inst.c); 
        }   break;

        case ByteCode::missing:
        {
            String s = (String)inst.a;
            Value const& v = thread.frame.environment->get(s);
            bool missing = v.isNil() || v.isDefault();
            store(thread, constant(Logical::c(missing ? Logical::TrueElement : Logical::FalseElement)), inst.c);
        }   break;

        case ByteCode::rep:
        {
            OPERAND(len, inst.a);
            IRRef l = load(thread, inst.a, &inst);
            Shape s = SpecializeLength(As<Integer>(thread, len)[0], l);
            // requires a dependent type
            store(thread, insert(trace, TraceOpCode::rep,
                cast(load(thread, inst.a, &inst), Type::Integer), 
                cast(load(thread, inst.b, &inst), Type::Integer), 0,
                Type::Integer, s, s), inst.c);
        }   break;
        case ByteCode::seq:
        {
            OPERAND(len, inst.a);
            IRRef l = load(thread, inst.a, &inst);
            Shape s = SpecializeLength(As<Integer>(thread, len)[0], l);
            // requires a dependent type
            IRRef c = load(thread, inst.c, &inst);
            IRRef b = load(thread, inst.b, &inst);
            Type::Enum type = trace[c].type == Type::Double || trace[b].type == Type::Double
                ? Type::Double : Type::Integer; 
            store(thread, insert(trace, TraceOpCode::seq,
                cast(c, type), cast(b, type), 0,
                type, s, s), inst.c);
        }   break;

        default: {
            //dump(thread,trace);
            if(thread.state.verbose)
                printf("Trace halted by %s\n", ByteCode::toString(inst.bc));
            return false;
        }   break;
    }
    return true;
}

JIT::IRRef JIT::duplicate(IR const& ir, std::vector<IRRef> const& forward) {
    return insert(code, ir.op, forward[ir.a], forward[ir.b], forward[ir.c], ir.type, ir.in, ir.out);
}

template<class T>
void Swap(T& a, T& b) {
    T t = a;
    a = b;
    b = t;
}

JIT::IR JIT::Normalize(IR ir) {
    switch(ir.op) {
        case TraceOpCode::add:
        case TraceOpCode::mul:
        case TraceOpCode::eq:
        case TraceOpCode::neq:
            if(ir.a > ir.b)
                Swap(ir.a, ir.b);
            break;
        case TraceOpCode::lt:
            if(ir.a > ir.b) {
                Swap(ir.a, ir.b);
                ir.op = TraceOpCode::gt;
            }
            break;
        case TraceOpCode::le:
            if(ir.a > ir.b) {
                Swap(ir.a, ir.b);
                ir.op = TraceOpCode::ge;
            }
            break;
        case TraceOpCode::gt:
            if(ir.a > ir.b) {
                Swap(ir.a, ir.b);
                ir.op = TraceOpCode::lt;
            }
            break;
        case TraceOpCode::ge:
            if(ir.a > ir.b) {
                Swap(ir.a, ir.b);
                ir.op = TraceOpCode::le;
            }
            break;
        default:
            // nothing
            break;
    }
    return ir;
}

double memcost(size_t size) {
    return (size <= (2<<10) ? 0.075 : (size <= (2<<20) ? 0.3 : 1));
}

/*

    Instruction scheduling
    Careful CSE to forwarding to minimize cost

    Lift filters and gathers above other instructions.

*/

JIT::IRRef JIT::Insert(std::vector<IR>& code, std::tr1::unordered_map<IR, IRRef>& cse, IR ir) {
    ir = Normalize(ir);

    // CSE cost:
    //  length * load cost
    // No CSE cost:
    //  length * op cost + cost of inputs (if CSEd)
    size_t mysize = ir.out.traceLength * (ir.type == Type::Logical ? 1 : 8);
    double csecost = mysize * memcost(mysize);
    double nocsecost = Opcost(code, ir);

    ir.cost = std::min(csecost, nocsecost);

    if(csecost <= nocsecost && cse.find(ir) != cse.end()) {
        printf("For %s => %f <= %f\n", TraceOpCode::toString(ir.op), csecost, nocsecost);
        // for mutating operations, have to check if there is a possible intervening mutation...
        if(ir.op == TraceOpCode::attrget) {
            for(IRRef i = code.size()-1; i != cse.find(ir)->second; i--) {
                if(code[i].op == TraceOpCode::attrset && ir.a == code[i].a) {
                    return code[i].c;   // forward get to previous set.
                }
            }
        }
        if(ir.op == TraceOpCode::attrset) {
            for(IRRef i = code.size()-1; i != cse.find(ir)->second; i--) {
                if(code[i].op == TraceOpCode::attrset && ir.a == code[i].a) {
                    code.push_back(ir);
                    cse[ir] = code.size()-1;
                    return code.size()-1;
                }
            }
        }

        return cse.find(ir)->second;
    }
    else {

        // (1) some type of guard strengthening and guard lifting to permit fusion
        // (2) determine fuseable sequences and uses between sequences.
        // (3) from the back to the front, replace uses in the loop with uses outside of the loop,
        //          IF PROFITABLE
        // (4) DCE

        // Do LICM as DCE rather than CSE.

        // want to do selective, cost driven cse...
        // replacing instruction with instruction outside the loop results in potentially high cost.
        // at each point I can choose to use CSE or recompute the instruction we are on.
        
        // But if the entire thing can be CSE'd that's great e.g. seq of maps followed by sum.
        // So want to do a backwards pass?
        // For a given instruction we know all uses. If dead, no need to compute.
        // If we're going to materialize it in the loop anyway, we should lift it out.
        // If we're not going to materialize it in the loop, we should selectively lift it out.

        // Decision has to be made while/after fusion decisions.
        // What would make us materialize in the loop?
        //      Gather from the vector
        //      
        code.push_back(ir);
        cse[ir] = code.size()-1;
        return code.size()-1;
    }
}

double JIT::Opcost(std::vector<IR>& code, IR ir) {
        switch(ir.op) {
            case TraceOpCode::LOADENV: 
            case TraceOpCode::NEWENV:
            case TraceOpCode::sload:
            case TraceOpCode::eload:
            case TraceOpCode::estore:
            case TraceOpCode::sstore:
            case TraceOpCode::kill:
            case TraceOpCode::PUSH:
            case TraceOpCode::POP:
            case TraceOpCode::strip:
            case TraceOpCode::GPROTO:
            case TraceOpCode::guardF: 
            case TraceOpCode::guardT:
            case TraceOpCode::attrget:
            case TraceOpCode::scatter:
            case TraceOpCode::jmp:
            case TraceOpCode::loop:
            case TraceOpCode::mov:
            case TraceOpCode::repscalar:
            case TraceOpCode::phi:
            case TraceOpCode::length:
            case TraceOpCode::constant:
            case TraceOpCode::rep:
            case TraceOpCode::elength:
            case TraceOpCode::slength:
            case TraceOpCode::olength:
            case TraceOpCode::alength:
            case TraceOpCode::glenEQ:
            case TraceOpCode::glenLT:
                return 10000000000000;
                break;
            
            #define CASE1(Name, str, group, func, Cost) \
                case TraceOpCode::Name: \
                    return (ir.in.traceLength * Cost) + code[ir.a].cost; break;
            UNARY_FOLD_SCAN_BYTECODES(CASE1)

            #define CASE2(Name, str, group, func, Cost) \
                case TraceOpCode::Name: \
                    return (ir.in.traceLength * Cost) + code[ir.a].cost + code[ir.b].cost; break;
            BINARY_BYTECODES(CASE2)

            #define CASE3(Name, str, group, func, Cost) \
                case TraceOpCode::Name: \
                    return (ir.in.traceLength * Cost) + code[ir.a].cost + code[ir.b].cost + code[ir.c].cost; break;
            TERNARY_BYTECODES(CASE3)

            case TraceOpCode::seq:
                return ir.out.traceLength * 1 + code[ir.a].cost + code[ir.b].cost;
                break;

            case TraceOpCode::gather:
                return ir.out.traceLength * memcost(code[ir.a].out.traceLength) + code[ir.b].cost;
                break;

            default:
            {
                printf("Unknown op: %s\n", TraceOpCode::toString(ir.op));
                _error("Unknown op in Opcost");
            }
    }
            #undef CASE1
            #undef CASE2
            #undef CASE3
}

JIT::Exit JIT::BuildExit( std::vector<IRRef>& environments, std::vector<StackFrame>& frames,
        std::map<Variable, IRRef>& stores, Instruction const* reenter) {

    // OK, attempt to do tracing something here...
    
    // get live environments
    std::vector<IRRef> live;
    for(size_t i = 0; i < frames.size(); i++) {
        live.push_back(frames[i].environment);
        live.push_back(frames[i].env);
    }
    //for(size_t i = 0; i < environments.size(); i++) {
    //    if(code[environments[i]].op == TraceOpCode::LOADENV)
    //        live.push_back(environments[i]);
    //}
    
    // get live stores (those that are into a live environment)
    // this is very inefficient, replace
    std::map<Variable, IRRef> livestores;
    for(std::map<Variable, IRRef>::const_iterator i = stores.begin(); i != stores.end(); ++i) {
        if(i->first.env == -1 || code[i->first.env].op == TraceOpCode::LOADENV)
            livestores.insert(*i);
        else {
            for(size_t j = 0; j < live.size(); j++) {
                if(live[j] == i->first.env) {
                    livestores.insert(*i);
                    break;
                }
            }
        }
    }

    std::vector<int64_t> tmp;
    Exit e = { live, frames, livestores, tmp, reenter };
    return e;
}


void JIT::EmitOptIR(
            IRRef i,
            IR ir,
            std::vector<IR>& code, 
            std::vector<IRRef>& forward, 
            std::map<Variable, IRRef>& loads,
            std::map<Variable, IRRef>& stores,
            std::tr1::unordered_map<IR, IRRef>& cse,
            std::vector<IRRef>& environments,
            std::vector<StackFrame>& frames) {

    if(i >= 2) {
        ir.in.length = forward[ir.in.length];
        ir.out.length = forward[ir.out.length];
    }
        switch(ir.op) {
            #define CASE(Name, ...) case TraceOpCode::Name:
            
            case TraceOpCode::LOADENV: 
            {
                forward[i] = Insert(code, cse, ir);
            } break;

            case TraceOpCode::NEWENV: {
                std::tr1::unordered_map<IR, IRRef> tcse;
                ir.a = forward[ir.a]; ir.b = forward[ir.b]; ir.c = forward[ir.c];
                forward[i] = Insert(code, tcse, ir);
                environments.push_back(forward[i]);
            } break;

            case TraceOpCode::sload:
            case TraceOpCode::eload: {
                Variable v = { ir.op == TraceOpCode::sload ? -1 : forward[ir.a], (int64_t)ir.b };
                // forward to previous store, if there is one. 
                if(stores.find(v) != stores.end()) {
                    forward[i] = stores[v];
                    if(loads.find(v) == loads.end())
                        loads[v] = stores[v];
                }
                else if(loads.find(v) != loads.end())
                    forward[i] = loads[v];
                else {
                    forward[i] = loads[v] = 
                        Insert(code, cse, ir);
                    if(forward[i] == code.size()-1)
                        exits[code.size()-1] = BuildExit( environments, frames, stores, reenters[i] );
                }
            } break; 
            case TraceOpCode::slength:
            case TraceOpCode::elength: {
                Variable v = { ir.op == TraceOpCode::slength ? -1 : forward[ir.a], (int64_t)ir.b };
                // forward to previous store, if there is one. 
                if(stores.find(v) != stores.end()) {
                    forward[i] = code[stores[v]].out.length;
                }
                else if(loads.find(v) != loads.end()) {
                    forward[i] = loads[v]-2;
                }
                else {
                    forward[i] = Insert(code, cse, ir);
                }
            } break; 
            case TraceOpCode::estore: {
                Variable v = { forward[ir.b], (int64_t)ir.c };
                stores[v] = forward[i] = forward[ir.a];
            } break;
            case TraceOpCode::sstore: {
                Variable v = { (IRRef)-1, (int64_t)ir.c };
                stores[v] = forward[i] = forward[ir.a];

                // compiler approach guarantees that an sstore kills all higher indexed registers.
                for(std::map<Variable, IRRef>::iterator k = stores.begin(); k != stores.end(); ) {
                    if(k->first.i < v.i) {
                        stores.erase(k++);
                    } else {
                        ++k;
                    }
                }
            } break;

            case TraceOpCode::kill: {
                // an explicit register kill for certain instructions that don't store
                for(std::map<Variable, IRRef>::iterator k = stores.begin(); k != stores.end(); ) {
                    if(k->first.i <= (int64_t)ir.a) {
                        stores.erase(k++);
                    } else {
                        ++k;
                    }
                }
            } break;

            case TraceOpCode::PUSH: {
                frames.push_back(this->frames[i]);
                frames.back().environment = forward[frames.back().environment];
                frames.back().env = forward[frames.back().env];
            } break;
            case TraceOpCode::POP: {
                frames.pop_back();
            } break;

            case TraceOpCode::strip:
            case TraceOpCode::GPROTO:
            case TraceOpCode::guardF: 
            case TraceOpCode::guardT: {
                ir.a = forward[ir.a];
                forward[i] = Insert(code, cse, ir);
                if(forward[i] == code.size()-1)
                    exits[code.size()-1] = BuildExit( environments, frames, stores, reenters[i] );
            } break;
            case TraceOpCode::glenEQ: {
                ir.a = forward[ir.a]; ir.b = forward[ir.b];
                if(ir.a != ir.b) {
                    forward[i] = Insert(code, cse, ir);
                    if(forward[i] == code.size()-1)
                        exits[code.size()-1] = BuildExit( environments, frames, stores, reenters[i] );
                }
            } break;
            case TraceOpCode::glenLT:
            case TraceOpCode::attrget: {
                ir.a = forward[ir.a]; ir.b = forward[ir.b];
                forward[i] = Insert(code, cse, ir);
                if(forward[i] == code.size()-1)
                    exits[code.size()-1] = BuildExit( environments, frames, stores, reenters[i] );
            } break;
            case TraceOpCode::scatter: {
                ir.a = forward[ir.a]; ir.b = forward[ir.b]; ir.c = forward[ir.c];
                forward[i] = Insert(code, cse, ir);
            } break;

            case TraceOpCode::repscalar:
            case TraceOpCode::olength:
            case TraceOpCode::phi:
            UNARY_FOLD_SCAN_BYTECODES(CASE) {
                ir.a = forward[ir.a];
                forward[i] = Insert(code, cse, ir);
            } break;

            case TraceOpCode::gather:
            case TraceOpCode::mov:
            case TraceOpCode::rep:
            case TraceOpCode::alength:
            BINARY_BYTECODES(CASE)
            {
                ir.a = forward[ir.a]; ir.b = forward[ir.b];
                forward[i] = Insert(code, cse, ir);
            } break;

            case TraceOpCode::seq:
            TERNARY_BYTECODES(CASE)
            {
                ir.a = forward[ir.a]; ir.b = forward[ir.b]; ir.c = forward[ir.c];
                forward[i] = Insert(code, cse, ir);
            } break;

            case TraceOpCode::jmp:
            case TraceOpCode::loop:
            CASE(constant)
            {
                forward[i] = Insert(code, cse, ir);
            } break;

            case TraceOpCode::length:
            {
                forward[i] = forward[ir.out.length];
            } break;

            default:
            {
                printf("Unknown op: %s\n", TraceOpCode::toString(ir.op));
                _error("Unknown op in EmitOptIR");
            }

            #undef CASE
        }
}

void JIT::Replay(Thread& thread) {
   
    code.clear();
    exits.clear();
 
    size_t n = trace.size();
    
    std::vector<IRRef> forward(n, 0);
    std::map<Variable, IRRef> loads;
    std::map<Variable, IRRef> stores, stores_old;
    std::tr1::unordered_map<IR, IRRef> cse;
    std::vector<IRRef> environments;
    std::vector<StackFrame> frames;

    // Emit loop header...
    for(size_t i = 0; i < n; i++) {
        EmitOptIR(i, trace[i], code, forward, loads, stores, cse, environments, frames);
    }

    Loop = Insert(code, cse, IR(TraceOpCode::loop, Type::Promise, Shape::Empty, Shape::Empty));
    
    // Emit PHIs 
    std::map<Variable, IRRef>::iterator s,t;
    for(s = stores.begin(); s != stores.end(); ++s) {
        IR const& ir = code[s->second];
        IRRef a = Insert(code, cse, 
                IR(TraceOpCode::phi, s->second, ir.type, ir.out, ir.out));
        s->second = a;
        forward[s->second] = a;
        stores[s->first] = a;
    }
    stores_old = stores;
    loads.clear();
 
    // Emit loop
    for(size_t i = 0; i < n; i++) {
        EmitOptIR(i, trace[i], code, forward, loads, stores, cse, environments, frames);
    }

    // Emit MOVs
    Exit e;
    for(s = stores.begin(); s != stores.end(); ++s) {
        t = stores_old.find(s->first);
        if(t != stores_old.end() && s->second > t->second) {
            IR const& ir = code[s->second];
            IRRef a = Insert(code, cse, 
                    IR(TraceOpCode::mov, t->second, s->second, ir.type, ir.out, ir.out));
            s->second = a;
            e.o[s->first] = a;
        }
    }
    
   
    // Emit the JMP
    IRRef jmp = Insert(code, cse, IR(TraceOpCode::jmp, Type::Promise, Shape::Empty, Shape::Empty));
    exits[jmp] = e;
}

void JIT::markLiveOut(Exit const& exit) {
    /*std::map<int64_t, JIT::IRRef>::const_iterator i;
    for(i = exit.o.begin(); i != exit.o.end(); i++) {
        code[i->second].liveout = true;
    }*/
}

JIT::Ptr JIT::end_recording(Thread& thread) {

    // general optimization strategy
    //  do constant propogation & instruction simplification during recording
    //  while duplicating body, do licm and phi elimination

    // 1) Figure out loop carried dependencies
    //      Only variables will be loop carried.
    //      We do a store at the end of a loop that the next iteration
    //          should load.
    // 2) Figure out live out

    assert(state == RECORDING);
    state = OFF;

    //dump(thread, trace);
    Replay(thread);
    //dump(thread, code);
    //Schedule();
    schedule();
    RegisterAssignment(exits[code.size()-1]);
    if(thread.state.verbose)
        dump(thread, code);

    
    return compile(thread);
}

void JIT::specialize() {
    // basically, we want to score how valuable a particular specialization
    // (replacing a load with a constant) might be.
    // Only worth doing on loads in the loop header.
    // Valuable things:
    //  1) Eliminating a guard to enable fusion.
    //  2) Turn unvectorized op into a vectorized op
    //      a) Lowering gather to shuffle
    //      b) Lowering pow to vectorized mul or sqrt
    //  3) Making a size constant (e.g. out of a filter)
    // 
    //  Might be target specific
    //
    // Valuable is a tradeoff between reuse and benefit.
    //  How to judge?
    //  Not valuable for very long vectors or scalars.
    //  Valuable for small multiples of HW vector length,
    //      where we can unroll the loop completely.
    //  Unless the entire vector is a constant
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
                    std::map<Variable, IRRef> eloads = loads;
                    std::map<Variable, IRRef> estores = stores;
                    std::tr1::unordered_map<IR, IRRef> ecse = cse;
                    for(size_t k = 0; k < i; k++) {
                        if(forward[k] == -1) {
                            printf("Compensation: %d\n", k);
                            EmitOptIR(k, code[k], e.compensation, eforward, eloads, estores, ecse);
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

/*

    Do fusion scheduling via lazy evaluation of trace

    Output should be a DAG of fused operations,
        with the exception of phi nodes.

    Along edges, def-use dependencies are recorded.

    Question: leave stuff in SSA form??
        Argument: Yes. Simplicity, only one form. No need to translate indices, etc.
        Argument: No. SSA not good for multiple simultaneous dependencies, need to insert loads anyway

    Other form??
        Groups of instructions. Isn't that what scheduling was doing anyway??
        Group with a length and a list of inputs and outputs...
        But we can already get that from groups...

        Groups have a scheduling DAG, where to represent that?
        That has to be linearized. 

    Rules:

        Phis can be executed together in a fused loop (e.g. if they get lowered to a mov)
        Phis can be fused with what comes after them, but not before them.
        
        Guards must be executed before everything after them, but things before them can be moved
            after if there are no dependencies. 

        Can't fuse across loop boundary. PHIs can mark that they're 1st operand cannot be fused.

        Ignore STOREs and LOADs, already forwarded in SSA construction.
        If you reference a LOAD, just load.
    
        Live variables are only introduced at guards.

        Delay as long as possible (lower numbers better).

    Plan:
        Move PHIs to bottom of trace, followed by guard and JMP
            Reasoning: allows fusion of ops in loop body with PHIs at the end.
        Switch compiler to use everything in pointers. Small vectors are Alloca'd and an optimization
            pass lowers them to registers. Big ones are in memory or on the stack. No attempt to lower.
            Reasoning: allows uniform treatment of small and large vectors
                        PHIs at end is not the LLVM style so use alloca instead.

*/

bool JIT::Ready(JIT::IR ir, std::vector<bool>& done) {
        switch(ir.op) {
            #define CASE(Name, ...) case TraceOpCode::Name:
            case TraceOpCode::sload:
            case TraceOpCode::LOADENV:
            case TraceOpCode::phi:
            case TraceOpCode::loop: 
            case TraceOpCode::constant: 
                return true;
                break;
            UNARY_BYTECODES(CASE) 
            FOLD_BYTECODES(CASE) 
            case TraceOpCode::GPROTO: 
            case TraceOpCode::guardF: 
            case TraceOpCode::guardT:
            case TraceOpCode::eload: {
                return done[ir.a];
            } break; 
            BINARY_BYTECODES(CASE)
            case TraceOpCode::gather:
            case TraceOpCode::mov:
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
    Shape gSize(-1, -1);
    std::set<IRRef> gMembers;

    fusable = std::vector<bool>(code.size(), true);

    for(IRRef i = 0; i < code.size(); i++) {
         if( code[i].in != gSize
          || (code[i].op == TraceOpCode::scatter && gMembers.find(code[i].c) != gMembers.end())
          || (code[i].op == TraceOpCode::gather  && gMembers.find(code[i].c) != gMembers.end())
          || code[i].op == TraceOpCode::guardT 
          || code[i].op == TraceOpCode::guardF 
          || code[i].op == TraceOpCode::eload 
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
    
/*
    size_t g = 1;
    group = std::vector<size_t>(code.size(), 0);

    for(size_t i = code.size()-1; i < code.size(); --i) {
        IR& ir = code[i];
        switch(ir.op) {
            #define CASE(Name, ...) case TraceOpCode::Name:
            case TraceOpCode::estore:
            case TraceOpCode::eload: {
                group[ir.a] = std::max(group[ir.a], group[i]);
            } break; 
            case TraceOpCode::loop: {
                group[i] = g+=2;
            } break;
            case TraceOpCode::phi: {
                group[ir.a] = std::max(group[ir.a], g+2);
            } break;
            case TraceOpCode::mov: {
                group[i] = g;
                group[ir.a] = std::max(group[ir.a], g);
                group[ir.b] = std::max(group[ir.b], g);
            } break;
            case TraceOpCode::GPROTO: 
            //case TraceOpCode::GTYPE: 
            case TraceOpCode::guardF: 
            case TraceOpCode::guardT: {
                // Do I also need to update any values that
                // are live out at this exit? Yes.
                group[i] = (g += 2);
                group[ir.a] = g+1;
                //std::map<Variable, IRRef>::const_iterator j;
                //for(j = exits[i].o.begin(); j != exits[i].o.end(); ++j) {
                //    group[j->second] = g+1;
                //}
            } break;
            case TraceOpCode::scatter: {
                group[i]++;
                group[ir.a] = 
                    std::max(group[ir.a],
                        group[i]);
                group[ir.b] = 
                    std::max(group[ir.b],
                        group[i]);
                group[ir.c] = 
                    std::max(group[ir.c],
                        group[i]+1);
            } break;
            case TraceOpCode::gather: {
                group[ir.a] = 
                    std::max(group[ir.a],
                        group[i]+1);
                group[ir.b] = 
                    std::max(group[ir.b],
                        group[i]);
            } break;
            TERNARY_BYTECODES(CASE)
            {
                group[ir.a] = 
                    std::max(group[ir.a],
                        group[i]);
                group[ir.b] = 
                    std::max(group[ir.b],
                        group[i]);
                group[ir.c] = 
                    std::max(group[ir.c],
                        group[i]);
            } break;
            BINARY_BYTECODES(CASE)
            {
                group[ir.a] = 
                    std::max(group[ir.a],
                        group[i]);
                group[ir.b] = 
                    std::max(group[ir.b],
                        group[i]);
            } break;
            case TraceOpCode::castd: 
            case TraceOpCode::casti: 
            case TraceOpCode::castl: 
            UNARY_BYTECODES(CASE) 
            {
                group[ir.a] = 
                    std::max(group[ir.a],
                        group[i]);
            } break;
            case TraceOpCode::length:
            FOLD_BYTECODES(CASE) 
            {
                group[i]++;
                group[ir.a] = 
                    std::max(group[ir.a],
                        group[i]);
            } break;
            case TraceOpCode::rep: {
                group[ir.a] = 
                    std::max(group[ir.a],
                        group[i]+1);
                group[ir.b] = 
                    std::max(group[ir.b],
                        group[i]+1);
            } break;
            default: {
            } break;

            #undef CASE
        }
    }
    maxGroup = g+2;
*/
}

void JIT::AssignRegister(size_t src, std::vector<int64_t>& assignment, size_t index) {
    if(assignment[index] <= 0) {
        IR const& ir = code[index];
        if(ir.op == TraceOpCode::sload ||
            ir.op == TraceOpCode::eload ||
            ir.op == TraceOpCode::LOADENV ||
            ir.op == TraceOpCode::constant) {
            assignment[index] = 0;
            return;
        }
 
        Register r = { ir.type, ir.out }; 

        // if usage crosses loop boundary, it requires a unique live register
        if(src > Loop && index < Loop && code[src].op != TraceOpCode::phi) {
            assignment[index] = registers.size();
            registers.push_back(r);
            return;
        }

        // if there's a preferred register look for that first.
        if(assignment[index] < 0) {
            std::pair<std::multimap<Register, size_t>::iterator,std::multimap<Register, size_t>::iterator> ret;
            ret = freeRegisters.equal_range(r);
            for (std::multimap<Register, size_t>::iterator it = ret.first; it != ret.second; ++it) {
                if(it->second == -assignment[index]) {
                    assignment[index] = it->second;
                    freeRegisters.erase(it); 
                    return;
                }
            }
        }

        // if no preferred or preferred wasn't available fall back to any available or create new.
        std::map<Register, size_t>::iterator i = freeRegisters.find(r);
        if(i != freeRegisters.end()) {
            assignment[index] = i->second;
            freeRegisters.erase(i);
            return;
        }
        else {
            assignment[index] = registers.size();
            registers.push_back(r);
            return;
        }
    }
}

void JIT::PreferRegister(std::vector<int64_t>& assignment, size_t index, size_t share) {
    if(assignment[index] == 0) {
        assignment[index] = assignment[share] > 0 ? -assignment[share] : assignment[share];
    }
}

void JIT::ReleaseRegister(std::vector<int64_t>& assignment, size_t index) {
    if(assignment[index] > 0) {
        freeRegisters.insert( std::make_pair(registers[assignment[index]], assignment[index]) );
    }
    else if(assignment[index] < 0) {
        printf("Preferred at %d\n", index);
        _error("Preferred register never assigned");
    }
}

void JIT::RegisterAssignment(Exit& e) {
    // backwards pass to do register assignment on a node.
    // its register assignment becomes dead, its operands get assigned to registers if not already.
    // have to maintain same memory space on register assignments.

    // lift this out somewhere
    registers.clear();
    Register invalid = {Type::Promise, Shape::Empty};
    registers.push_back(invalid);
    freeRegisters.clear();
 
    std::vector<int64_t>& a = e.assignment;

    a.clear();
    a.resize(code.size(), 0);
   
    // add all register to the freeRegisters list
    for(size_t i = 0; i < registers.size(); i++) {
        freeRegisters.insert( std::make_pair(registers[i], i) );
    }
 
    // first mark all outputs as live...
    for(std::map<Variable, IRRef>::const_iterator i = e.o.begin(); i != e.o.end(); ++i) {
        AssignRegister(code.size(), a, i->second);
    }

    for(size_t i = code.size()-1; i < code.size(); --i) {
        IR& ir = code[i];

        ReleaseRegister(a, i);
        switch(ir.op) {
            #define CASE(Name, ...) case TraceOpCode::Name:
            case TraceOpCode::loop: {
            } break;
            case TraceOpCode::phi: {
                PreferRegister(a, ir.a, i);
                AssignRegister(i, a, ir.a);
            } break;
            case TraceOpCode::mov: {
                PreferRegister(a, ir.b, i);
                AssignRegister(i, a, ir.b);
                PreferRegister(a, ir.a, ir.b);
            } break;
            case TraceOpCode::strip: 
            case TraceOpCode::GPROTO: 
            case TraceOpCode::guardF: 
            case TraceOpCode::guardT: {
                AssignRegister(i, a, ir.a);
                std::map<Variable, IRRef>::const_iterator j;
                for(j = exits[i].o.begin(); j != exits[i].o.end(); ++j) {
                    AssignRegister(i, a, j->second);
                }
            } break;
            case TraceOpCode::glenEQ: 
            case TraceOpCode::glenLT: {
                AssignRegister(i, a, ir.a);
                AssignRegister(i, a, ir.b);
                std::map<Variable, IRRef>::const_iterator j;
                for(j = exits[i].o.begin(); j != exits[i].o.end(); ++j) {
                    AssignRegister(i, a, j->second);
                }
            } break;
            case TraceOpCode::scatter: {
                AssignRegister(i, a, ir.c);
                AssignRegister(i, a, ir.a);
                AssignRegister(i, a, ir.b);
            } break;
            case TraceOpCode::attrset:
            TERNARY_BYTECODES(CASE)
            {
                AssignRegister(i, a, ir.c);
                AssignRegister(i, a, ir.b);
                AssignRegister(i, a, ir.a);
            } break;
            case TraceOpCode::attrget:
            case TraceOpCode::alength:
            case TraceOpCode::rep:
            case TraceOpCode::seq:
            case TraceOpCode::gather:
            BINARY_BYTECODES(CASE)
            {
                AssignRegister(i, a, std::max(ir.a, ir.b));
                AssignRegister(i, a, std::min(ir.a, ir.b));
            } break;
            case TraceOpCode::repscalar:
            case TraceOpCode::olength:
            UNARY_FOLD_SCAN_BYTECODES(CASE)
            {
                AssignRegister(i, a, ir.a);
            } break;
            case TraceOpCode::jmp:
            case TraceOpCode::sload:
            case TraceOpCode::slength:
            case TraceOpCode::eload:
            case TraceOpCode::elength:
            case TraceOpCode::constant:
            case TraceOpCode::NEWENV:
            case TraceOpCode::LOADENV:
                // do nothing
                break;
            default: {
                printf("Unknown op is %s\n", TraceOpCode::toString(ir.op));
                _error("Unknown op in RegisterAssignment");
            } break;
            #undef CASE
        }
    }
}

void JIT::IR::dump() const {
    std::cout << in.length << "->" << out.length;
    printf(" %.3s\t\t", Type::toString(type));

    std::cout << TraceOpCode::toString(op);

    switch(op) {
        #define CASE(Name, ...) case TraceOpCode::Name:
        case TraceOpCode::loop: {
            std::cout << " --------------------";
        } break;
        case TraceOpCode::sload:
        case TraceOpCode::slength: {
            std::cout << "\t " << (int64_t)b;
        } break;
        case TraceOpCode::eload:
        case TraceOpCode::elength: {
            std::cout << "\t " << a << "\t\"" << (String)b << "\"";
        } break;
        case TraceOpCode::sstore: {
            std::cout << "\t " << a << "\t " << (int64_t)c;
        } break;    
        case TraceOpCode::estore: {
            std::cout << "\t " << a << "\t " << b << "\t\"" << (String)c << "\"";
        } break;
        case TraceOpCode::GPROTO:
        {
            std::cout << "\t " << a << "\t [" << b << "]";
        } break;
        case TraceOpCode::LOADENV:
        {
            std::cout << "\t [" << a << "]";
        } break;
        case TraceOpCode::kill:
            std::cout << "\t " << (int64_t)a;
            break;
        case TraceOpCode::repscalar:
        case TraceOpCode::strip:
        case TraceOpCode::phi: 
        case TraceOpCode::PUSH:
        case TraceOpCode::length:
        case TraceOpCode::guardF:
        case TraceOpCode::guardT: 
        case TraceOpCode::olength: 
        UNARY_FOLD_SCAN_BYTECODES(CASE)
        {
            std::cout << "\t " << a;
        } break;
        case TraceOpCode::mov:
        case TraceOpCode::rep:
        case TraceOpCode::seq:
        case TraceOpCode::gather:
        case TraceOpCode::attrget:
        case TraceOpCode::alength:
        case TraceOpCode::glenEQ:
        case TraceOpCode::glenLT:
        BINARY_BYTECODES(CASE)
        {
            std::cout << "\t " << a << "\t " << b;
        } break;
        case TraceOpCode::NEWENV:
        case TraceOpCode::attrset:
        case TraceOpCode::scatter:
        TERNARY_BYTECODES(CASE)
        {
            std::cout << "\t " << a << "\t " << b << "\t " << c;
        } break;
        default: {} break;

        #undef CASE
    };

    //if(liveout)
    //    std::cout << "\t=>";
}

void JIT::dump(Thread& thread, std::vector<IR> const& t) {

    printf("There at %d exits\n", exits.size());

    std::map<size_t, Exit>::const_iterator e = exits.find(t.size()-1);

    for(size_t i = 0; i < t.size(); i++) {
        IR const& ir = t[i];
        if(ir.op != TraceOpCode::nop) {
            printf("%4li:", i);
            if(fusable.size() == t.size() && !fusable[i]) 
                printf("-");
            else
                printf(" ");
            if(e != exits.end()
                && e->second.assignment.size() == t.size()) printf(" (%2d) ", e->second.assignment[i]);
            ir.dump();
    
   
            if(exits.find(i) != exits.end()) { 
                std::cout << "\n\t\t=> ";
                Exit const& e = exits[i];
                std::cout << "[" << e.frames.size() << " frames, " << e.environments.size() << " envs, " << e.reenter << "] ";
                for(std::map<Variable, IRRef>::const_iterator i = e.o.begin(); i != e.o.end(); ++i) {
                    std::cout << i->second << "->";
                    if(i->first.i >= 0) 
                        std::cout << i->first.env << ":" << (String)(i->first.i) << " ";
                    else std::cout << (int64_t)i->first.i << " ";
                }
            }
            if(ir.op == TraceOpCode::constant) {
                std::cout <<  "\t\t\t; " << thread.stringify(constants[ir.a]);
            }
            std::cout << std::endl;
        }
    }
    std::cout << std::endl;
}

#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Analysis/Passes.h"
#include "llvm/Analysis/Verifier.h"
#include "llvm/Bitcode/BitstreamWriter.h"
#include "llvm/Bitcode/ReaderWriter.h"
#include "llvm/DerivedTypes.h"
#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/ExecutionEngine/JIT.h"
#include "llvm/LLVMContext.h"
#include "llvm/Linker.h"
#include "llvm/Module.h"
#include "llvm/Pass.h"
#include "llvm/PassManager.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/IRBuilder.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Type.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Value.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/ValueHandle.h"
#include "llvm/Module.h"
#include "llvm/LLVMContext.h"
#include "llvm/Support/system_error.h"
#include "llvm/Intrinsics.h"

struct LLVMState {
    llvm::Module * M;
    llvm::LLVMContext * C;
    llvm::ExecutionEngine * EE;
    llvm::FunctionPassManager * FPM;

    LLVMState() {
        llvm::InitializeNativeTarget();

        C = &llvm::getGlobalContext();

        llvm::OwningPtr<llvm::MemoryBuffer> buffer;
        llvm::MemoryBuffer::getFile("bin/ops.bc", buffer);
        M = ParseBitcodeFile(buffer.get(), *C);

        std::string err;
        EE = llvm::EngineBuilder(M).setErrorStr(&err).setEngineKind(llvm::EngineKind::JIT).create();
        if (!EE) {
            _error(err);
        }

        FPM = new llvm::FunctionPassManager(M);

        //TODO: add optimization passes here, these are just from llvm tutorial and are probably not good
        //look here: http://lists.cs.uiuc.edu/pipermail/llvmdev/2011-December/045867.html
        FPM->add(new llvm::TargetData(*EE->getTargetData()));
        // do this first so the verifier doesn't freak out about our empty blocks
        FPM->add(llvm::createCFGSimplificationPass());
        
        FPM->add(llvm::createVerifierPass());
        // Provide basic AliasAnalysis support for GVN.
        FPM->add(llvm::createBasicAliasAnalysisPass());
        // Promote allocas to registers.
        FPM->add(llvm::createPromoteMemoryToRegisterPass());
        // Also promote aggregates like structs....
        FPM->add(llvm::createScalarReplAggregatesPass());
        // Do simple "peephole" optimizations and bit-twiddling optzns.
        // TODO: This causes an invalid optimization somewhere that results in LLVM eliminating all
        // my code and replacing it with a trap. ????
        //FPM->add(llvm::createInstructionCombiningPass());
        
        // Reassociate expressions.
        FPM->add(llvm::createReassociatePass());
        // Eliminate Common SubExpressions.
        //FPM->add(llvm::createGVNPass());
        // Simplify the control flow graph (deleting unreachable blocks, etc).
        FPM->add(llvm::createCFGSimplificationPass());
        // Promote allocas to registers.
        FPM->add(llvm::createPromoteMemoryToRegisterPass());
        // Simplify the control flow graph (deleting unreachable blocks, etc).
        FPM->add(llvm::createAggressiveDCEPass());
        
        FPM->doInitialization();
    }
};

static LLVMState llvmState;

struct Fusion {
    JIT& jit;
    LLVMState* S;
    llvm::Function* function;
    std::vector<llvm::Value*> const& values;
    std::vector<llvm::Value*> const& registers;
    std::vector<int64_t> const& assignment;
    
    llvm::BasicBlock* header;
    llvm::BasicBlock* condition;
    llvm::BasicBlock* body;
    llvm::BasicBlock* after;

    llvm::Value* iterator;
    llvm::Value* length;

    llvm::Constant *zerosD, *zerosI, *onesD, *onesI, *seqD, *seqI;

    size_t width;
    llvm::IRBuilder<> builder;

    std::map<size_t, llvm::Value*> outs;
    std::map<size_t, llvm::Value*> reductions;

    size_t instructions;

    Fusion(JIT& jit, LLVMState* S, llvm::Function* function, std::vector<llvm::Value*> const& values, std::vector<llvm::Value*> const& registers, std::vector<int64_t> const& assignment, llvm::Value* length, size_t width)
        : jit(jit)
          , S(S)
          , function(function)
          , values(values)
          , registers(registers)
          , assignment(assignment)
          , length(length)
          , width(width)
          , builder(*S->C) {
       
        if(llvm::isa<llvm::ConstantInt>(length) && ((llvm::ConstantInt*)length)->getSExtValue() <= SPECIALIZE_LENGTH) {
            // short vector, don't emit while loop
            this->width = (size_t)((llvm::ConstantInt*)length)->getZExtValue();
            this->length = 0;
        }

        if(this->width > 0) {
            std::vector<llvm::Constant*> zeros;
            for(size_t i = 0; i < this->width; i++) 
                zeros.push_back(llvm::ConstantFP::get(builder.getDoubleTy(), 0));
            zerosD = llvm::ConstantVector::get(zeros);
       
            zeros.clear(); 
            for(size_t i = 0; i < this->width; i++) 
                zeros.push_back(builder.getInt64(0));
            zerosI = llvm::ConstantVector::get(zeros);

            std::vector<llvm::Constant*> ones;
            for(size_t i = 0; i < this->width; i++) 
                ones.push_back(llvm::ConstantFP::get(builder.getDoubleTy(), 1));
            onesD = llvm::ConstantVector::get(ones);
            
            ones.clear(); 
            for(size_t i = 0; i < this->width; i++) 
                ones.push_back(builder.getInt64(1));
            onesI = llvm::ConstantVector::get(ones);

            std::vector<llvm::Constant*> sD;
            for(size_t i = 0; i < this->width; i++) 
                sD.push_back(llvm::ConstantFP::get(builder.getDoubleTy(), i));
            seqD = llvm::ConstantVector::get(sD);
            
            std::vector<llvm::Constant*> sI;
            for(size_t i = 0; i < this->width; i++) 
                sI.push_back(builder.getInt64(i));
            seqI = llvm::ConstantVector::get(sI);
        }

        instructions = 0;
    }

    llvm::Type* llvmType(Type::Enum type) {
        llvm::Type* t;
        switch(type) {
            case Type::Double: t = builder.getDoubleTy(); break;
            case Type::Integer: t = builder.getInt64Ty(); break;
            case Type::Logical: t = builder.getInt1Ty(); break;
            case Type::Promise: t = builder.getInt1Ty(); break;
            default: _error("Bad type in trace");
        }
        return t;
    }

    llvm::Type* llvmType(Type::Enum type, size_t width) {
        return llvm::VectorType::get(llvmType(type), width);
    }

    llvm::AllocaInst* CreateEntryBlockAlloca(llvm::Type* type) {
        llvm::IRBuilder<> TmpB(&function->getEntryBlock(),
                function->getEntryBlock().begin());
        llvm::AllocaInst* r = TmpB.CreateAlloca(type);
        r->setAlignment(16);
        return r;
    }

    void Open(llvm::BasicBlock* before) {
        header = llvm::BasicBlock::Create(*S->C, "fusedHeader", function, before);
        condition = llvm::BasicBlock::Create(*S->C, "fusedCondition", function, before);
        body = llvm::BasicBlock::Create(*S->C, "fusedBody", function, before);
        after = llvm::BasicBlock::Create(*S->C, "fusedAfter", function, before);

        builder.SetInsertPoint(header);
        llvm::Value* initial = builder.getInt64(0);

        if(length != 0) {
            builder.SetInsertPoint(condition);
            iterator = builder.CreatePHI(builder.getInt64Ty(), 2);
            ((llvm::PHINode*)iterator)->addIncoming(initial, header);
        }
        else {
            iterator = initial;
        }

        builder.SetInsertPoint(body);
    }

    llvm::Value* RawLoad(size_t ir) {
        return values[ir];
    }

    llvm::Value* Load(size_t ir) {

        if(outs.find(assignment[ir]) != outs.end())
            return outs[assignment[ir]];

        llvm::Value* a = (assignment[ir] == 0) ? values[ir] : registers[assignment[ir]];
        llvm::Type* t = llvm::VectorType::get(
                ((llvm::SequentialType*)a->getType())->getElementType(),
                width)->getPointerTo();
        a = builder.CreateInBoundsGEP(a, iterator);
        a = builder.CreatePointerCast(a, t);
        a = builder.CreateLoad(a);

        if(jit.code[ir].type == Type::Logical)
            a = builder.CreateICmpEQ(a, llvm::ConstantVector::get(builder.getInt8(255)));

        return a;
    }

    void Store(llvm::Value* a, size_t reg, llvm::Value* iterator) {
        size_t width = ((llvm::VectorType*)a->getType())->getNumElements();

        if(jit.registers[reg].type == Type::Logical)
            a = builder.CreateSExt(a, llvm::VectorType::get(builder.getInt8Ty(), width));

        llvm::Value* out = builder.CreateInBoundsGEP(registers[reg], iterator);

        llvm::Type* t = llvm::VectorType::get(
                ((llvm::SequentialType*)a->getType())->getElementType(),
                width)->getPointerTo();

        out = builder.CreatePointerCast(out, t);

        builder.CreateStore(a, out);
    }

    llvm::Value* SSEIntrinsic(llvm::Intrinsic::ID Op1, llvm::Intrinsic::ID Op2, JIT::IR const& ir) {
        llvm::Value* in = Load(ir.a);
        llvm::Value* out = llvm::UndefValue::get(llvmType(ir.type, width)); 
        uint32_t i = 0;                                                                 
        for(; i < (width-1); i+=2) {                                                    
            llvm::Function* f = llvm::Intrinsic::getDeclaration(S->M, Op2);
            llvm::Value* v2 = llvm::UndefValue::get(llvmType(jit.code[ir.a].type, 2)); 
            llvm::Value* j0 = builder.CreateExtractElement(in, builder.getInt32(i));    
            llvm::Value* j1 = builder.CreateExtractElement(in, builder.getInt32(i+1));  
            v2 = builder.CreateInsertElement(v2, j0, builder.getInt32(i));              
            v2 = builder.CreateInsertElement(v2, j1, builder.getInt32(i+1));            
            v2 = builder.CreateCall(f, v2);                                     
            out = builder.CreateInsertElement(out, 
                builder.CreateExtractElement(v2, builder.getInt32(0)), builder.getInt32(i));
            out = builder.CreateInsertElement(out, 
                builder.CreateExtractElement(v2, builder.getInt32(1)), builder.getInt32(i+1));
        }
        for(; i < width; i++) {
            llvm::Function* f = llvm::Intrinsic::getDeclaration(S->M, Op1);
            llvm::Value* v1 = llvm::UndefValue::get(llvmType(jit.code[ir.a].type, 2)); 
            llvm::Value* j0 = builder.CreateExtractElement(in, builder.getInt32(i));
            v1 = builder.CreateInsertElement(v1, j0, builder.getInt32(i)); 
            v1 = builder.CreateCall(f, v1);
            out = builder.CreateInsertElement(out, 
                builder.CreateExtractElement(v1, builder.getInt32(0)), builder.getInt32(i));
        }
        return out;
    }

    llvm::Value* SSEIntrinsic2(llvm::Intrinsic::ID Op1, llvm::Intrinsic::ID Op2, JIT::IR const& ir) {
        llvm::Value* ina = Load(ir.a);
        llvm::Value* inb = Load(ir.b);
        llvm::Value* out = llvm::UndefValue::get(llvmType(ir.type, width)); 
        uint32_t i = 0;                                                                 
        for(; i < (width-1); i+=2) {                                                    
            llvm::Function* f = llvm::Intrinsic::getDeclaration(S->M, Op2);
            llvm::Value* v2 = llvm::UndefValue::get(llvmType(jit.code[ir.a].type, 2)); 
            llvm::Value* w2 = llvm::UndefValue::get(llvmType(jit.code[ir.b].type, 2)); 
            llvm::Value* j0 = builder.CreateExtractElement(ina, builder.getInt32(i));    
            llvm::Value* j1 = builder.CreateExtractElement(ina, builder.getInt32(i+1));  
            llvm::Value* k0 = builder.CreateExtractElement(inb, builder.getInt32(i));    
            llvm::Value* k1 = builder.CreateExtractElement(inb, builder.getInt32(i+1));  
            v2 = builder.CreateInsertElement(v2, j0, builder.getInt32(i));              
            v2 = builder.CreateInsertElement(v2, j1, builder.getInt32(i+1));            
            w2 = builder.CreateInsertElement(v2, k0, builder.getInt32(i));              
            w2 = builder.CreateInsertElement(v2, k1, builder.getInt32(i+1));            
            v2 = builder.CreateCall2(f, v2, w2);                                     
            out = builder.CreateInsertElement(out, 
                builder.CreateExtractElement(v2, builder.getInt32(0)), builder.getInt32(i));
            out = builder.CreateInsertElement(out, 
                builder.CreateExtractElement(v2, builder.getInt32(1)), builder.getInt32(i+1));
        }
        for(; i < width; i++) {
            llvm::Function* f = llvm::Intrinsic::getDeclaration(S->M, Op1);
            llvm::Value* v1 = llvm::UndefValue::get(llvmType(jit.code[ir.a].type, 2)); 
            llvm::Value* w1 = llvm::UndefValue::get(llvmType(jit.code[ir.b].type, 2)); 
            llvm::Value* j0 = builder.CreateExtractElement(ina, builder.getInt32(i));
            llvm::Value* k0 = builder.CreateExtractElement(inb, builder.getInt32(i));
            v1 = builder.CreateInsertElement(v1, j0, builder.getInt32(i)); 
            w1 = builder.CreateInsertElement(w1, k0, builder.getInt32(i)); 
            v1 = builder.CreateCall2(f, v1, w1);
            out = builder.CreateInsertElement(out, 
                builder.CreateExtractElement(v1, builder.getInt32(0)), builder.getInt32(i));
        }
        return out;
    }

    llvm::Value* SSERound(llvm::Value* in, uint32_t k) {
        llvm::Value* out = llvm::UndefValue::get(llvmType(Type::Double, width)); 
        uint32_t i = 0;
        // Why does llvm think that round takes two vector arguments?                  
        for(; i < (width-1); i+=2) {                                                    
            llvm::Function* f = llvm::Intrinsic::getDeclaration(S->M, llvm::Intrinsic::x86_sse41_round_pd);
            llvm::Value* v2 = llvm::UndefValue::get(llvmType(Type::Double, 2)); 
            llvm::Value* j0 = builder.CreateExtractElement(in, builder.getInt32(i));    
            llvm::Value* j1 = builder.CreateExtractElement(in, builder.getInt32(i+1));  
            v2 = builder.CreateInsertElement(v2, j0, builder.getInt32(i));              
            v2 = builder.CreateInsertElement(v2, j1, builder.getInt32(i+1));            
            v2 = builder.CreateCall3(f, v2, v2, builder.getInt32(k));                                     
            out = builder.CreateInsertElement(out, 
                builder.CreateExtractElement(v2, builder.getInt32(0)), builder.getInt32(i));
            out = builder.CreateInsertElement(out, 
                builder.CreateExtractElement(v2, builder.getInt32(1)), builder.getInt32(i+1));
        }
        for(; i < width; i++) {
            llvm::Function* f = llvm::Intrinsic::getDeclaration(S->M, llvm::Intrinsic::x86_sse41_round_sd);
            llvm::Value* v1 = llvm::UndefValue::get(llvmType(Type::Double, 2)); 
            llvm::Value* j0 = builder.CreateExtractElement(in, builder.getInt32(i));
            v1 = builder.CreateInsertElement(v1, j0, builder.getInt32(i)); 
            v1 = builder.CreateCall3(f, v1, v1, builder.getInt32(k));
            out = builder.CreateInsertElement(out, 
                builder.CreateExtractElement(v1, builder.getInt32(0)), builder.getInt32(i));
        }
        return out;
    }

    llvm::Value* UnaryCall(std::string func, JIT::IR const& ir) {
        std::vector<llvm::Type*> args;
        args.push_back(llvmType(jit.code[ir.a].type));
        llvm::Type* outTy = llvmType(ir.type);
        llvm::FunctionType* ft = llvm::FunctionType::get(outTy, args, false);
        llvm::Function* f = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, func, S->M);

        llvm::Value* in = Load(ir.a);
        llvm::Value* out = llvm::UndefValue::get(llvmType(ir.type, width));
        for(uint32_t i = 0; i < width; i++) {
            llvm::Value* v = builder.CreateExtractElement(in, builder.getInt32(i));
            v = builder.CreateCall(f, v);
            out = builder.CreateInsertElement(out, v, builder.getInt32(i));
        }
        return out;
    }

    llvm::Value* BinaryCall(std::string func, JIT::IR const& ir) {
        llvm::Function* f = S->M->getFunction(func);
        if(f == 0) {
            std::vector<llvm::Type*> args;
            args.push_back(llvmType(jit.code[ir.a].type));
            args.push_back(llvmType(jit.code[ir.b].type));
            llvm::Type* outTy = llvmType(ir.type);
            llvm::FunctionType* ft = llvm::FunctionType::get(outTy, args, false);
            f = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, func, S->M);
        }
        llvm::Value* ina = Load(ir.a);
        llvm::Value* inb = Load(ir.b);
        llvm::Value* out = llvm::UndefValue::get(llvmType(ir.type, width));
        for(uint32_t i = 0; i < width; i++) {
            llvm::Value* v = builder.CreateExtractElement(ina, builder.getInt32(i));
            llvm::Value* w = builder.CreateExtractElement(inb, builder.getInt32(i));
            v = builder.CreateCall2(f, v, w);
            out = builder.CreateInsertElement(out, v, builder.getInt32(i));
        }
        return out;
    }

#define MARKER(str) \
    builder.CreateCall(S->M->getFunction("MARKER"), builder.CreateConstGEP2_64(builder.CreateGlobalString(str), 0, 0))

    void Emit(size_t index) {

        // DCE
        if(assignment[index] == 0)
            return;

        instructions++;
        JIT::IR ir = jit.code[index];
        size_t reg = assignment[index];

#define CASE_UNARY(Op, FName, IName) \
    case TraceOpCode::Op: { \
        outs[reg] = (jit.code[ir.a].type == Type::Double)   \
        ? builder.Create##FName(Load(ir.a)) \
        : builder.Create##IName(Load(ir.a));\
    } break

#define CASE_BINARY(Op, FName, IName) \
    case TraceOpCode::Op: { \
        outs[reg] = (jit.code[ir.a].type == Type::Double)   \
        ? builder.Create##FName(Load(ir.a), Load(ir.b)) \
        : builder.Create##IName(Load(ir.a), Load(ir.b));\
    } break

#define CASE_UNARY_LOGICAL(Op, Name) \
    case TraceOpCode::Op: { \
        outs[reg] = builder.Create##Name(Load(ir.a)); \
    } break

#define CASE_BINARY_LOGICAL(Op, Name) \
    case TraceOpCode::Op: { \
        outs[reg] = builder.Create##Name(Load(ir.a), Load(ir.b)); \
    } break

#define IDENTITY \
    outs[reg] = Load(ir.a);

#define SCALARIZE1(Name, A) { \
    llvm::Value* in = Load(ir.a);                        \
    outs[reg] = llvm::UndefValue::get(llvmType(ir.type, width)); \
    for(uint32_t i = 0; i < width; i++) {                                       \
        llvm::Value* ii = builder.getInt32(i);                                  \
        llvm::Value* j = builder.CreateExtractElement(in, ii);  \
        j = builder.Create##Name(j, A);                                         \
        outs[reg] = builder.CreateInsertElement(outs[reg], j, ii);              \
    } \
}

        switch(ir.op) {
            case TraceOpCode::pos: IDENTITY; break;

            CASE_UNARY(neg, FNeg, Neg);
            
            case TraceOpCode::sqrt: 
                outs[reg] = SSEIntrinsic(llvm::Intrinsic::x86_sse2_sqrt_sd, llvm::Intrinsic::x86_sse2_sqrt_pd, ir); 
                break;           
            CASE_BINARY(add, FAdd, Add);
            CASE_BINARY(sub, FSub, Sub);
            CASE_BINARY(mul, FMul, Mul);
            CASE_BINARY(div, FDiv, SDiv);
          
            CASE_BINARY(eq, FCmpOEQ, ICmpEQ);  
            CASE_BINARY(neq, FCmpONE, ICmpNE);  
            CASE_BINARY(lt, FCmpOLT, ICmpSLT);  
            CASE_BINARY(le, FCmpOLE, ICmpSLE);  
            CASE_BINARY(gt, FCmpOGT, ICmpSGT);  
            CASE_BINARY(ge, FCmpOGE, ICmpSGE);  
           
            CASE_UNARY_LOGICAL(lnot, Not); 
            CASE_BINARY_LOGICAL(lor, Or); 
            CASE_BINARY_LOGICAL(land, And); 
           
            case TraceOpCode::floor: 
                outs[reg] = SSERound(Load(ir.a), 0x1 /* round down */); 
                break;
            case TraceOpCode::ceiling: 
                outs[reg] = SSERound(Load(ir.a), 0x2 /* round up */); 
                break;
            case TraceOpCode::trunc: 
                outs[reg] = SSERound(Load(ir.a), 0x3 /* round to zero */); 
                break;
            case TraceOpCode::abs:
                // TODO: this could be faster with some bit twidling
                if(ir.type == Type::Double) {
                    llvm::Value* o = builder.CreateFNeg(Load(ir.a));
                    outs[reg] = builder.CreateSelect(
                        builder.CreateFCmpOLT(Load(ir.a), o),
                        o, Load(ir.a));
                }
                else {
                    llvm::Value* o = builder.CreateNeg(Load(ir.a));
                    outs[reg] = builder.CreateSelect(
                        builder.CreateICmpSLT(Load(ir.a), o),
                        o, Load(ir.a));
                } 
            break;
            case TraceOpCode::sign:
                // TODO: make this faster
                outs[reg] = builder.CreateSelect(
                    builder.CreateFCmpOLT(Load(ir.a), zerosD),
                        builder.CreateFNeg(onesD),
                        builder.CreateSelect(
                            builder.CreateFCmpOGT(Load(ir.a), zerosD),
                            onesD,
                            zerosD));
            break;

            case TraceOpCode::exp: outs[reg] = UnaryCall("exp", ir); break;
            case TraceOpCode::log: outs[reg] = UnaryCall("log", ir); break;
            case TraceOpCode::cos: outs[reg] = UnaryCall("cos", ir); break;
            case TraceOpCode::sin: outs[reg] = UnaryCall("sin", ir); break;
            case TraceOpCode::tan: outs[reg] = UnaryCall("tan", ir); break;
            case TraceOpCode::acos: outs[reg] = UnaryCall("acos", ir); break;
            case TraceOpCode::asin: outs[reg] = UnaryCall("asin", ir); break;
            case TraceOpCode::atan: outs[reg] = UnaryCall("atan", ir); break;

            case TraceOpCode::pow: outs[reg] = BinaryCall("pow", ir); break;
            case TraceOpCode::atan2: outs[reg] = BinaryCall("atan2", ir); break;
            case TraceOpCode::hypot: outs[reg] = BinaryCall("hypot", ir); break;

            case TraceOpCode::idiv:
                if(ir.type == Type::Double) {
                    outs[reg] = SSERound(builder.CreateFDiv(Load(ir.a), Load(ir.b)), 0x1); 
                }
                else {
                    outs[reg] = builder.CreateSDiv(Load(ir.a), Load(ir.b));
                }
            break;

            case TraceOpCode::mod:
                if(ir.type == Type::Double) {
                    outs[reg] = SSERound(builder.CreateFDiv(Load(ir.a), Load(ir.b)), 0x1);
                    outs[reg] = builder.CreateFSub(Load(ir.a), builder.CreateFMul(outs[reg], Load(ir.b)));
                } else {
                    outs[reg] = builder.CreateSDiv(Load(ir.a), Load(ir.b));
                    outs[reg] = builder.CreateSub(Load(ir.a), builder.CreateMul(outs[reg], Load(ir.b)));
                }
            break;

            case TraceOpCode::pmin:
                if(ir.type == Type::Double) {
                    outs[reg] = SSEIntrinsic2(llvm::Intrinsic::x86_sse2_min_sd,
                                                llvm::Intrinsic::x86_sse2_min_pd, ir);
                }
                else {
                    outs[reg] = builder.CreateSelect(
                        builder.CreateICmpSLT(Load(ir.a), Load(ir.b)), Load(ir.a), Load(ir.b));
                }
            break;

            case TraceOpCode::pmax:
                if(ir.type == Type::Double) {
                    outs[reg] = SSEIntrinsic2(llvm::Intrinsic::x86_sse2_max_sd,
                                                llvm::Intrinsic::x86_sse2_max_pd, ir);
                }
                else {
                    outs[reg] = builder.CreateSelect(
                        builder.CreateICmpSGT(Load(ir.a), Load(ir.b)), Load(ir.a), Load(ir.b));
                }
            break;

            case TraceOpCode::ifelse:
                outs[reg] = builder.CreateSelect(Load(ir.a), Load(ir.b), Load(ir.c));
            break;

            case TraceOpCode::asdouble:
                switch(jit.code[ir.a].type) {
                    case Type::Integer: SCALARIZE1(SIToFP, builder.getDoubleTy()); break;
                    case Type::Logical: SCALARIZE1(SIToFP, builder.getDoubleTy()); break;
                    case Type::Double: IDENTITY; break;
                    default: _error("Unexpected cast"); break;
                }
                break;
            
            case TraceOpCode::asinteger:
                switch(jit.code[ir.a].type) {
                    case Type::Double: SCALARIZE1(FPToSI, builder.getInt64Ty()); break;
                    case Type::Logical: SCALARIZE1(ZExt, builder.getInt64Ty()); break;
                    case Type::Integer: IDENTITY; break;
                    default: _error("Unexpected cast"); break;
                }
                break;
            
            case TraceOpCode::aslogical:
                switch(jit.code[ir.a].type) {
                    case Type::Double: SCALARIZE1(FCmpONE, llvm::ConstantFP::get(builder.getDoubleTy(), 0)); break;
                    case Type::Integer: SCALARIZE1(ICmpEQ, builder.getInt64(0)); break;
                    case Type::Logical: IDENTITY; break;
                    default: _error("Unexpected cast"); break;
                }
                break;
            
            case TraceOpCode::mov:
                if(assignment[ir.a] != assignment[ir.b]) 
                    outs[assignment[ir.a]] = Load(ir.b);
                break;
            
            case TraceOpCode::rep: 
            {
                // there's all sorts of fast variants if lengths are known.
                //if(llvm::isa<llvm::Constant>(a) && llvm::isa<llvm::Constant>(b)) {
                outs[reg] = zerosI;
                //}
                //else {
                //    _error("Unsupported rep");
                // }
            } break;
            case TraceOpCode::gather: 
            {
                llvm::Value* v = RawLoad(ir.a);
                llvm::Value* idx = Load(ir.b);
                // scalarize the gather...
                llvm::Value* r = llvm::UndefValue::get(llvmType(ir.type, width));

                for(uint32_t i = 0; i < width; i++) {
                    llvm::Value* ii = builder.getInt32(i);
                    llvm::Value* j = builder.CreateExtractElement(idx, ii);
                    j = builder.CreateLoad(builder.CreateGEP(v, j));
                    if(ir.type == Type::Logical)
                        j = builder.CreateICmpEQ(j, builder.getInt8(255));
                    r = builder.CreateInsertElement(r, j, ii);
                }
                outs[assignment[index]] = r;
            } break;
            case TraceOpCode::scatter:
            {
                llvm::Value* v = Load(ir.a);
                llvm::Value* idx = Load(ir.b);
                
                if(assignment[ir.c] != assignment[index]) {
                    // must duplicate (copy from the in register to the out). 
                    // Do this in the fusion header.
                    llvm::IRBuilder<> TmpB(header,
                        header->begin());
                    TmpB.CreateMemCpy(RawLoad(index), RawLoad(ir.c),
                        builder.CreateMul(
                            builder.CreateLoad(RawLoad(ir.out.length)),
                            builder.getInt64(ir.type == Type::Logical ? 1 : 8)), 
                        16);
                }
                /*if(jit.assignment[ir.c] != jit.assignment[index]) {
                    r = Load(values[ir.c]);
                    llvm::Type* trunc = llvm::VectorType::get(builder.getInt32Ty(), jit.code[ir.b].width);
                    idx = builder.CreateTrunc(idx, trunc);  
                    // constant version could be a shuffle. No idea if that will generate better code.
                    for(size_t i = 0; i < width; i++) {
                        llvm::Value* ii = builder.getInt32((uint32_t)i);
                        llvm::Value* j = builder.CreateExtractElement(v, ii);
                        ii = builder.CreateExtractElement(idx, ii);
                        r = builder.CreateInsertElement(r, j, ii);
                    }
                }
                else {*/
                    // reusing register, just assign in place.
                    llvm::Type* mt = llvmType(ir.type)->getPointerTo();
                    llvm::Value* x = builder.CreatePointerCast(RawLoad(index), mt);
                    for(uint32_t i = 0; i < width; i++) {
                        llvm::Value* ii = builder.getInt32(i);
                        llvm::Value* j = builder.CreateExtractElement(v, ii);
                        ii = builder.CreateExtractElement(idx, ii);
                        builder.CreateStore(j,
                            builder.CreateGEP(x, ii));
                    }
                //} 
            } break;

            case TraceOpCode::repscalar:
            {
                llvm::Value* r = llvm::UndefValue::get(llvmType(ir.type, width));
                llvm::Value* v = builder.CreateLoad(RawLoad(ir.a));
                for(size_t i = 0; i < this->width; i++) {
                    r = builder.CreateInsertElement(r, v, builder.getInt32(i));
                }
                outs[reg] = r;
                
            } break;

            // Generators
            case TraceOpCode::seq:
            {
                llvm::AllocaInst* r = CreateEntryBlockAlloca(llvmType(ir.type, width));
            
                llvm::IRBuilder<> TmpB(header, header->end());

                // now initialize
                llvm::Value* start = TmpB.CreateLoad(RawLoad(ir.a));
                llvm::Value* step = TmpB.CreateLoad(RawLoad(ir.b));

                llvm::Value* starts = llvm::UndefValue::get(llvmType(ir.type, width)); 
                llvm::Value* steps = llvm::UndefValue::get(llvmType(ir.type, width)); 
                llvm::Value* bigstep = llvm::UndefValue::get(llvmType(ir.type, width));
                for(size_t i = 0; i < this->width; i++) {
                    starts = TmpB.CreateInsertElement(starts, start, builder.getInt32(i));
                    steps = TmpB.CreateInsertElement(steps, step, builder.getInt32(i));
                    if(ir.type == Type::Integer)
                        bigstep = TmpB.CreateInsertElement(bigstep, TmpB.CreateMul(step,builder.getInt64(width)), builder.getInt32(i));
                    else if(ir.type == Type::Double)
                        bigstep = TmpB.CreateInsertElement(bigstep, TmpB.CreateFMul(step,llvm::ConstantFP::get(builder.getDoubleTy(), width)), builder.getInt32(i));
                    else
                        _error("Unexpected seq type");
                } 
               
                llvm::Value* added;
                if(ir.type == Type::Integer) { 
                    TmpB.CreateStore(TmpB.CreateSub(TmpB.CreateAdd(TmpB.CreateMul(seqI, steps), starts), bigstep), r);
                    added = builder.CreateAdd(builder.CreateLoad(r), bigstep);
                }
                else if(ir.type == Type::Double) {
                    TmpB.CreateStore(TmpB.CreateFSub(TmpB.CreateFAdd(TmpB.CreateFMul(seqD, steps), starts), bigstep), r);
                    added = builder.CreateFAdd(builder.CreateLoad(r), bigstep);
                }
                else
                    _error("Unexpected seq type");
                builder.CreateStore(added, r);
                outs[reg] = added;
            } break;

            // Reductions
            case TraceOpCode::sum:
            {
                llvm::Value* agg;
                if(ir.type == Type::Double) {
                    agg = CreateEntryBlockAlloca(llvmType(ir.type, width));
                    llvm::IRBuilder<> TmpB(header, header->end());
                    TmpB.CreateStore(zerosD, agg);
                    builder.CreateStore(builder.CreateFAdd(builder.CreateLoad(agg), Load(ir.a)), agg);
                }
                else {
                    agg = CreateEntryBlockAlloca(llvmType(ir.type, width));
                    llvm::IRBuilder<> TmpB(header, header->end());
                    TmpB.CreateStore(zerosI, agg);
                    builder.CreateStore(builder.CreateAdd(builder.CreateLoad(agg), Load(ir.a)), agg);
                } 
                reductions[index] = agg;
            } break;
            default:
                printf("Unsupported op is %s\n", TraceOpCode::toString(ir.op));
                _error("Unsupported op in Fusion::Emit");
                break;
        }
#undef SCALARIZE_SSE
#undef SCALARIZE1
#undef IDENTITY
#undef CASE_UNARY
#undef CASE_BINARY
#undef CASE_UNARY_LOGICAL
#undef CASE_BINARY_LOGICAL
    }

    void Reduce(llvm::Value* a, size_t i) {
        JIT::IR const& ir = jit.code[i];
        llvm::Value* r = llvm::UndefValue::get(llvmType(ir.type, 1));
        llvm::Value* t;
        switch(ir.op) {
            case TraceOpCode::sum:
                a = builder.CreateLoad(a);
                if(ir.type == Type::Double) {
                    t = llvm::ConstantFP::get(builder.getDoubleTy(), 0);
                    for(size_t i = 0; i < width; i++) {
                        t = builder.CreateFAdd(t, builder.CreateExtractElement(a, builder.getInt32(i)));
                    }
                }
                else {
                    t = builder.getInt64(0);
                    for(size_t i = 0; i < width; i++) {
                        t = builder.CreateAdd(r, builder.CreateExtractElement(a, builder.getInt32(i)));
                    }
                }
                break;
            default:
                _error("Unsupported reduction");
                break;
        }
        r = builder.CreateInsertElement(r, t, builder.getInt32(0));
        Store(r, assignment[i], builder.getInt64(0)); 
    }

    llvm::BasicBlock* Close() {
       
        if(instructions == 0) {
            header = after;
            return after;
        }

        std::map<size_t, llvm::Value*>::const_iterator i;
        for(i = outs.begin(); i != outs.end(); i++) {
            Store(i->second, i->first, iterator);
        }

        builder.SetInsertPoint(header);
        builder.CreateBr(condition);

        if(length == 0) {
            builder.SetInsertPoint(body);
            builder.CreateBr(after);

            builder.SetInsertPoint(condition);
            builder.CreateBr(body);
        }
        else {
            builder.SetInsertPoint(body);
            llvm::Value* increment = builder.CreateAdd(iterator, builder.getInt64(width));
            ((llvm::PHINode*)iterator)->addIncoming(increment, body);
            builder.CreateBr(condition);

            builder.SetInsertPoint(condition);
            llvm::Value* endCond = builder.CreateICmpULT(iterator, length);
            builder.CreateCondBr(endCond, body, after);
        }

        builder.SetInsertPoint(after);
        for(i = reductions.begin(); i != reductions.end(); i++) {
            Reduce(i->second, i->first);
        }

        return after;
    }
};

struct LLVMCompiler {
    Thread& thread;
    JIT& jit;
    LLVMState* S;
    llvm::Function * function;
    llvm::BasicBlock * EntryBlock;
    llvm::BasicBlock * HeaderBlock;
    llvm::BasicBlock * PhiBlock;
    llvm::BasicBlock * LoopStart;
    llvm::BasicBlock * InnerBlock;
    llvm::BasicBlock * EndBlock;
    llvm::IRBuilder<> builder;

    llvm::Type* thread_type;
    llvm::Type* instruction_type;
    llvm::Type* value_type;
    llvm::Type* actual_value_type;

    llvm::Value* thread_var;
    llvm::Value* result_var;

    std::vector<llvm::Value*> values;
    std::vector<llvm::Value*> registers;
    std::vector<llvm::CallInst*> calls;
    Fusion* fusion;   
 
    LLVMCompiler(Thread& thread, JIT& jit) 
        : thread(thread), jit(jit), S(&llvmState), builder(*S->C) 
    {
        fusion = 0;
    }

    llvm::CallInst* Save(llvm::CallInst* ci) {
        calls.push_back(ci);
        return ci;
    }

#define CALL0(F) \
    Save(builder.CreateCall(S->M->getFunction(F), thread_var))

#define CALL1(F, A) \
    Save(builder.CreateCall2(S->M->getFunction(F), thread_var, A))

#define CALL2(F, A, B) \
    Save(builder.CreateCall3(S->M->getFunction(F), thread_var, A, B))

#define CALL3(F, A, B, C) \
    Save(builder.CreateCall4(S->M->getFunction(F), thread_var, A, B, C))

#define CALL4(F, A, B, C, D) \
    Save(builder.CreateCall5(S->M->getFunction(F), thread_var, A, B, C, D))

#define CALL6(F, A, B, C, D, E, G) \
    llvm::Value* args[] = { thread_var, A, B, C, D, E, G }; \
    Save(builder.CreateCall(S->M->getFunction(F), args))

    void* Compile() {
        thread_type = S->M->getTypeByName("class.Thread")->getPointerTo();
        instruction_type = S->M->getTypeByName("struct.Instruction")->getPointerTo();
        value_type = llvm::StructType::get(builder.getInt64Ty(), builder.getInt64Ty(), NULL);
        actual_value_type = S->M->getTypeByName("struct.Value");

        std::vector<llvm::Type*> argTys;
        argTys.push_back(thread_type);

        llvm::FunctionType* functionTy = llvm::FunctionType::get(
                instruction_type,
                argTys, /*isVarArg=*/false);

        function = llvm::Function::Create(functionTy,
                llvm::Function::ExternalLinkage,
                "trace", S->M);

        EntryBlock = llvm::BasicBlock::Create(
                *S->C, "entry", function, 0);
        HeaderBlock = llvm::BasicBlock::Create(
                *S->C, "header", function, 0);
        InnerBlock = llvm::BasicBlock::Create(
                *S->C, "inner", function, 0);
        EndBlock = llvm::BasicBlock::Create(
                *S->C, "end", function, 0);

        llvm::Function::arg_iterator ai = function->arg_begin();
        ai->setName("thread");
        thread_var = ai++;

        builder.SetInsertPoint(EntryBlock);

        result_var = CreateEntryBlockAlloca(instruction_type, builder.getInt64(1));

        // create registers...
        registers.clear();
        registers.push_back(0);
        for(size_t i = 1; i < jit.registers.size(); i++) {
            registers.push_back(
                jit.registers[i].shape.length <= SPECIALIZE_LENGTH 
                  ?  CreateEntryBlockAlloca(
                        llvmMemoryType(jit.registers[i].type), builder.getInt64(jit.registers[i].shape.length))
                  : Malloc(jit.registers[i].type, jit.registers[i].shape.length));
        }
       
        // create values for each ssa node
        std::vector<int64_t> const& assignment = jit.exits[jit.code.size()-1].assignment; 
        for(size_t i = 0; i < jit.code.size(); i++) {
            if(assignment[i] != 0)
                values.push_back(registers[assignment[i]]);
            else {
                // this case will be filled in as we emit instructions
                values.push_back(0);
            }
        }

        builder.SetInsertPoint(HeaderBlock);

        for(size_t i = 0; i < jit.code.size(); i++) {
            if(jit.code[i].op != TraceOpCode::nop)
                Emit(jit.code[i], i);
        }
            if(fusion) {
                llvm::BasicBlock* after = fusion->Close();
                builder.CreateBr(fusion->header);
                builder.SetInsertPoint(after);
            }
        builder.CreateBr(PhiBlock);

        builder.SetInsertPoint(PhiBlock);
        builder.CreateBr(LoopStart);

        builder.SetInsertPoint(EntryBlock);
        builder.CreateBr(HeaderBlock);

        builder.SetInsertPoint(EndBlock);
        builder.CreateRet(builder.CreateLoad(result_var));

        S->FPM->run(*function);
        //function->dump();

        return S->EE->getPointerToFunction(function);   
    }

    llvm::AllocaInst* CreateEntryBlockAlloca(llvm::Type* type, llvm::Value* size) {
        llvm::IRBuilder<> TmpB(&function->getEntryBlock(),
                function->getEntryBlock().end());
        llvm::AllocaInst* r = TmpB.CreateAlloca(type, size);
        r->setAlignment(16);
        return r;
    }

    llvm::Value* Malloc(Type::Enum type, size_t size) {
        void* v;
        switch(type) {
            case Type::Double:
            {
                Double d(size);
                v = d.v();
            } break;
            case Type::Integer:
            {
                Integer d(size);
                v = d.v();
            } break;
            case Type::Logical:
            {
                Logical d(size);
                v = d.v();
            } break;
            case Type::Character:
            {
                Character d(size);
                v = d.v();
            } break;
            default: _error("Unknown alloc");
        }
        llvm::IRBuilder<> TmpB(&function->getEntryBlock(),
                function->getEntryBlock().end());
        return builder.CreateIntToPtr(
            builder.getInt64((int64_t)v),
            llvmMemoryType(type)->getPointerTo());
    } 

    llvm::Type* llvmMemoryType(Type::Enum type) {
        llvm::Type* t;
        switch(type) {
            case Type::Double: t = builder.getDoubleTy(); break;
            case Type::Integer: t = builder.getInt64Ty(); break;
            case Type::Logical: t = builder.getInt8Ty(); break;
            case Type::Character: t = builder.getInt8Ty()->getPointerTo(); break;
            default: t = value_type; break;
        }
        return t;
    }

    llvm::Type* llvmMemoryType(Type::Enum type, size_t width) {
        return llvm::ArrayType::get(llvmMemoryType(type), width);
    }

    llvm::Type* llvmType(Type::Enum type) {
        llvm::Type* t;
        switch(type) {
            case Type::Double: t = builder.getDoubleTy(); break;
            case Type::Integer: t = builder.getInt64Ty(); break;
            case Type::Logical: t = builder.getInt1Ty(); break;
            case Type::Character: t = builder.getInt8Ty()->getPointerTo(); break;
            default: t = value_type; break;
        }
        return t;
    }

    llvm::Type* llvmType(Type::Enum type, size_t width) {
        return llvm::VectorType::get(llvmType(type), width);
    }

    llvm::Value* Load(llvm::Value* v) {
        if(v->getType()->isPointerTy()) {
            return builder.CreateLoad(v);
        }
        else {
            return v;
        }
    }

    bool Unboxed(Type::Enum type) {
        return type == Type::Double 
            || type == Type::Integer
            || type == Type::Logical
            || type == Type::Character;
    }

    llvm::Value* Unbox(JIT::IRRef index, llvm::Value* v) {
        Type::Enum type = jit.code[index].type;
        if(Unboxed(type)) {
            llvm::Value* guard = CALL3("GTYPE",
                    builder.CreateExtractValue(v, 0), 
                    builder.CreateExtractValue(v, 1), 
                    builder.getInt64(type));

            if(jit.exits.find(index) == jit.exits.end())
                _error("Missing exit on unboxing operation");

            EmitExit(guard, jit.exits[index]);
            
            llvm::Value* tmp = CreateEntryBlockAlloca(value_type, builder.getInt64(0));
            builder.CreateStore(v, tmp);
            return CALL1(std::string("UNBOX_")+Type::toString(type), 
                builder.CreatePointerCast(tmp, actual_value_type->getPointerTo()));
        }
        else {
            return v;
        }
    }

    void Emit(JIT::IR ir, size_t index) { 

        std::vector<int64_t> const& assignment = jit.exits[jit.code.size()-1].assignment; 
        /*if(     ir.op == TraceOpCode::GTYPE
            ||  ir.op == TraceOpCode::guardT
            ||  ir.op == TraceOpCode::guardF
            ||  ir.op == TraceOpCode::jmp
            ||  ir.op == TraceOpCode::loop) {
            for(int i = 99; i > jit.group[index]; i--) {
                std::map<JIT::Shape, Fusion*>::iterator j;
                for(j = fusions[i].begin(); j != fusions[i].end(); ++j) {
                    Fusion* f = j->second;
                    builder.CreateBr(f->header);
                    builder.SetInsertPoint(f->Close());
                }
                fusions[i].clear();
            }
        }*/
        //if(!jit.fusable[index]) {
            if(fusion) {
                llvm::BasicBlock* after = fusion->Close();
                builder.CreateBr(fusion->header);
                builder.SetInsertPoint(after);
            }
            fusion = new Fusion(jit, S, function, values, registers, assignment, builder.CreateLoad(values[ir.out.length]), 2);
            fusion->Open(InnerBlock);
            
        //}

        if(ir.op == TraceOpCode::mov &&
            assignment[ir.a] == assignment[ir.b]) {
            ir.op = TraceOpCode::nop;
        }

        switch(ir.op) {
            #define CASE(Name, ...) case TraceOpCode::Name:

            case TraceOpCode::loop:
            {
                PhiBlock = llvm::BasicBlock::Create(
                    *S->C, "phis", function, InnerBlock);
                builder.CreateBr(PhiBlock);
                
                LoopStart = llvm::BasicBlock::Create(
                    *S->C, "loop", function, InnerBlock);
                builder.SetInsertPoint(LoopStart);
            }   break;

            case TraceOpCode::constant:
            {
                std::vector<llvm::Constant*> c;
                if(Unboxed(ir.type)) {
                    // types that are unboxed in the JITed code
                    if(ir.type == Type::Double) {
                        Double const& v = (Double const&)jit.constants[ir.a];
                        for(size_t i = 0; i < ir.out.traceLength; i++)
                            c.push_back(llvm::ConstantFP::get(builder.getDoubleTy(), v[i]));
                    } else if(ir.type == Type::Integer) {
                        Integer const& v = (Integer const&)jit.constants[ir.a];
                        for(size_t i = 0; i < ir.out.traceLength; i++)
                            c.push_back(builder.getInt64(v[i]));
                    } else if(ir.type == Type::Logical) {
                        Logical const& v = (Logical const&)jit.constants[ir.a];
                        for(size_t i = 0; i < ir.out.traceLength; i++)
                            c.push_back(builder.getInt8(v[i] != 0 ? 255 : 0));
                    } else if(ir.type == Type::Character) {
                        Character const& v = (Character const&)jit.constants[ir.a];
                        for(size_t i = 0; i < ir.out.traceLength; i++)
                            c.push_back((llvm::Constant*)builder.CreateIntToPtr(builder.getInt64((int64_t)v[i]), builder.getInt8Ty()->getPointerTo()));
                    }
                    values[index] = CreateEntryBlockAlloca(llvmMemoryType(ir.type), builder.getInt64(ir.out.traceLength));
                    for(size_t i = 0; i < ir.out.traceLength; i++) {
                        builder.CreateStore(c[i], builder.CreateConstGEP1_64(values[index], i));
                    }
                }
                else {
                    // a boxed type
                    values[index] = CreateEntryBlockAlloca(value_type, builder.getInt64(0));
                    llvm::Value* p = builder.CreatePointerCast(values[index], builder.getInt64Ty()->getPointerTo());
                    builder.CreateStore(builder.getInt64(jit.constants[ir.a].header), builder.CreateConstGEP1_64(p, 0));
                    builder.CreateStore(builder.getInt64(jit.constants[ir.a].i), builder.CreateConstGEP1_64(p, 1));
                    values[index] = builder.CreateLoad(values[index]);
                }
            } break;

            case TraceOpCode::LOADENV: {
                values[index] = CALL1(std::string("LOAD_environment"), builder.getInt64(ir.a));
            } break;
            case TraceOpCode::sload: 
            {
                values[index] = Unbox(index, CALL1("SLOAD", builder.getInt64(ir.b)));
            } break;
            case TraceOpCode::eload: 
            {
                values[index] = Unbox(index, CALL2("ELOAD", values[ir.a], builder.getInt64(ir.b)));
            } break;
            
            case TraceOpCode::GPROTO: {
                if(ir.in != JIT::Shape::Scalar) {
                    _error("Emitting guard on non-scalar");
                }
                llvm::Value* r = builder.CreateICmpEQ(
                    builder.CreatePtrToInt(CALL2("GET_prototype", 
                            builder.CreateExtractValue(values[ir.a], 0), 
                            builder.CreateExtractValue(values[ir.a], 1))
                        , builder.getInt64Ty()),
                    builder.getInt64(ir.b));
                EmitExit(r, jit.exits[index]);
            } break;

            case TraceOpCode::strip: {
                if(jit.code[ir.a].type == Type::Object) {
                    values[index] = Unbox(index, CALL2("GET_strip", 
                            builder.CreateExtractValue(values[ir.a], 0), 
                            builder.CreateExtractValue(values[ir.a], 1)));
                }
                else {
                    // if not an object, this is a no op.
                    values[index] = values[ir.a];
                }
            } break;

            case TraceOpCode::attrget: {
                if(jit.code[ir.b].out != JIT::Shape::Scalar)
                    _error("Emitting attrget with non-scalar attribute name");
                values[index] = Unbox(index, CALL3("GET_attr", 
                            builder.CreateExtractValue(values[ir.a], 0), 
                            builder.CreateExtractValue(values[ir.a], 1),
                            values[ir.b]));
            } break;

            case TraceOpCode::guardT:
            case TraceOpCode::guardF: {
                if(ir.in != JIT::Shape::Scalar) {
                    _error("Emitting guard on non-scalar");
                }
                // TODO: check the NA mask
                llvm::Value* r = builder.CreateTrunc(Load(values[ir.a]), builder.getInt1Ty());
                if(ir.op == TraceOpCode::guardF)
                    r = builder.CreateNot(r);
                EmitExit(r, jit.exits[index]);
            } break;

            case TraceOpCode::repscalar:
            case TraceOpCode::gather:
            case TraceOpCode::scatter:
            case TraceOpCode::mov: 
            TERNARY_BYTECODES(CASE)
            BINARY_BYTECODES(CASE)
            UNARY_FOLD_SCAN_BYTECODES(CASE)
            GENERATOR_BYTECODES(CASE)
            {
                if(assignment[index] > 0)
                    fusion->Emit(index);
            } break;
            case TraceOpCode::phi:
            {
                values[index] = values[ir.a];
            } break;
            case TraceOpCode::NEWENV:
            case TraceOpCode::PUSH:
            case TraceOpCode::POP:
            case TraceOpCode::jmp:
            case TraceOpCode::sstore:
            case TraceOpCode::estore:
            case TraceOpCode::nop:
            {
                // do nothing
            } break;
            
            default: 
            {
                printf("Unknown op: %s\n", TraceOpCode::toString(ir.op));
                _error("Unknown op in LLVMCompiler::Emit");
            } break;
        };
    }

    void EmitExit(llvm::Value* cond, JIT::Exit const& e) 
    {
        llvm::BasicBlock* next = llvm::BasicBlock::Create(*S->C, "next", function, InnerBlock);
        llvm::BasicBlock* exit = llvm::BasicBlock::Create(*S->C, "exit", function, EndBlock);
        builder.CreateCondBr(cond, next, exit);
        builder.SetInsertPoint(exit);

        // First create all live new environments
        for(size_t i = 0; i < e.environments.size(); i++) {
            if(jit.code[e.environments[i]].op == TraceOpCode::NEWENV) {
                values[e.environments[i]] = CALL4(
                    std::string("NEW_environment"),
                    values[jit.code[e.environments[i]].a],
                    values[jit.code[e.environments[i]].b],
                    builder.CreateExtractValue(values[jit.code[e.environments[i]].c], 0),
                    builder.CreateExtractValue(values[jit.code[e.environments[i]].c], 1));
            }
        }
        
        // Store out all variables
        std::map<JIT::Variable, JIT::IRRef>::const_iterator i;
        for(i = e.o.begin(); i != e.o.end(); i++) {
            
            JIT::Variable v = i->first;
            JIT::IR const& ir = jit.code[i->second];
            llvm::Value* r = values[i->second];
            
            // if unboxed type, box
            if(Unboxed(ir.type))
                r = CALL2(std::string("BOX_")+Type::toString(ir.type),
                    r, builder.CreateLoad(values[ir.out.length]));

            // Silly LLVM means I have to manually pass Values
            if(v.i >= 0) {
                CALL4("ESTORE", values[v.env], builder.getInt64(v.i), 
                    builder.CreateExtractValue(r, 0), 
                    builder.CreateExtractValue(r, 1));
            }
            else {
                CALL3("SSTORE", builder.getInt64(v.i),
                    builder.CreateExtractValue(r, 0), 
                    builder.CreateExtractValue(r, 1));
            }
        }

        // Create stack frames
        for(size_t i = 0; i < e.frames.size(); i++) {
            JIT::StackFrame frame = e.frames[i];
            CALL6("NEW_frame", 
                values[frame.environment], 
                builder.getInt64((int64_t)frame.prototype),
                builder.getInt64((int64_t)frame.returnpc),
                builder.getInt64((int64_t)frame.returnbase),
                builder.getInt64(frame.dest),
                values[frame.env]);
        } 

        builder.CreateStore(
            builder.CreateIntToPtr(builder.getInt64((int64_t)e.reenter), instruction_type), 
            result_var);
        
        builder.CreateBr(EndBlock);
        builder.SetInsertPoint(next); 
    }

};

JIT::Ptr JIT::compile(Thread& thread) {
    timespec a = get_time();
    LLVMCompiler compiler(thread, *this);
    Ptr result = (Ptr)compiler.Compile();
    printf("Compile time: %f\n", time_elapsed(a));
    return result;
}

