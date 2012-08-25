
#include "jit.h"
#include "bc.h"
#include "interpreter.h"
#include "call.h"
#include "ops.h"

DEFINE_ENUM_TO_STRING(TraceOpCode, TRACE_ENUM)

const JIT::Shape JIT::Shape::Empty = { 0 };
const JIT::Shape JIT::Shape::Scalar = { 1 };

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

JIT::Variable JIT::intern(Thread& thread, int64_t a) {
    if(a <= 0) {
        return (Variable) {0, (thread.base+a)-(thread.registers+DEFAULT_NUM_REGISTERS)};
    }
    else {
        return getVar(getEnv(thread.frame.environment), (String)a); 
    }
}

JIT::Shape shape(Value const& a) {
    if(a.isVector())
        return (JIT::Shape) { (size_t)a.length };
    else
        return JIT::Shape::Scalar;
}

JIT::IRRef JIT::load(Thread& thread, int64_t a, Instruction const* reenter) {

    // registers
    OPERAND(operand, a);

    Variable v = intern(thread, a);
    Shape s = shape(operand);
    
    IRRef r;
    if(v.i < 0) {
        r = insert(trace, TraceOpCode::sload, (IRRef)-1, v.i, 0, operand.type, s, s);
    }
    else {
        r = insert(trace, TraceOpCode::eload, v.env, v.i, 0, operand.type, s, s);
    }
    reenters[r] = reenter;
    return r;
}

JIT::IRRef JIT::cast(IRRef a, Type::Enum type) {
    if(trace[a].type != type) {
        Shape s = trace[a].out;
        if(trace[a].type == Type::Double)
            return insert(trace, TraceOpCode::castd, a, 0, 0, type, s, s);
        else if(trace[a].type == Type::Integer)
            return insert(trace, TraceOpCode::casti, a, 0, 0, type, s, s);
        else
            return insert(trace, TraceOpCode::castl, a, 0, 0, type, s, s);
    }
    else {
        return a;
    }
}

JIT::IRRef JIT::rep(IRRef a, size_t length) {
    if(trace[a].out.length != length) {
        Shape s = (Shape) { length };
        IRRef l = insert(trace, TraceOpCode::length, a, 0, 0, Type::Integer, Shape::Scalar, Shape::Scalar);
        IRRef e = constant(Integer::c(1));
        IRRef r = insert(trace, TraceOpCode::rep, l, e, 0, trace[a].type, s, s);
        return insert(trace, TraceOpCode::gather, a, r, 0, trace[a].type, s, s);
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

JIT::IRRef JIT::EmitBinary(TraceOpCode::Enum op, IRRef a, IRRef b, Type::Enum rty, Type::Enum maty, Type::Enum mbty) {
    size_t len = 0;
    if(trace[a].out.length > 0 && trace[b].out.length > 0)
        len = std::max(trace[a].out.length, trace[b].out.length);
    Shape s = (Shape) { len };

    return insert(trace, op, rep(cast(a,maty),len), rep(cast(b,mbty),len), 0, rty, s, s);
}

JIT::IRRef JIT::EmitTernary(TraceOpCode::Enum op, IRRef a, IRRef b, IRRef c, Type::Enum rty, Type::Enum maty, Type::Enum mbty, Type::Enum mcty) {
    size_t len = 0;
    if(trace[a].out.length > 0 && trace[b].out.length > 0 && trace[c].out.length > 0)
        len = std::max(trace[a].out.length, std::max(trace[b].out.length, trace[c].out.length));
    Shape s = (Shape) { len };

    return insert(trace, op, rep(cast(a,maty),len), rep(cast(b,mbty),len), rep(cast(c,mcty),len), rty, s, s);
}

JIT::IRRef JIT::store(Thread& thread, IRRef a, int64_t c) {
    Variable v = intern(thread,c);

    if(v.i < 0) {
        insert(trace, TraceOpCode::sstore, a, 0, v.i, trace[a].type, trace[a].out, Shape::Empty);
    }
    else {
        insert(trace, TraceOpCode::estore, a, v.env, v.i, trace[a].type, trace[a].out, Shape::Empty);
    }
    return a;
}

JIT::IRRef JIT::constant(Value const& value) {
    IRRef a;
    if(constantsMap.find(value) != constantsMap.end())
        a = constantsMap.find(value)->second;
    else {
        size_t ci = constants.size();
        constants.push_back(value);
        Shape s = shape(value);
        a = insert(trace, TraceOpCode::constant, ci, 0, 0, value.type, s, s);
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
        }   break;
    
        case ByteCode::call:
        {
            insert(trace, TraceOpCode::PUSH, envs[thread.frame.environment], 0, 0, Type::Promise, Shape::Scalar, Shape::Empty);
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
            b = insert(trace, TraceOpCode::sub, b, rep(constant(Integer::c(1)), trace[b].out.length), 0, trace[b].type, trace[b].out, trace[b].out);
            store(thread, insert(trace, TraceOpCode::gather, a, b, 0, trace[a].type, trace[b].out, trace[b].out), inst.c);
        }   break;

        case ByteCode::scatter1: {
        case ByteCode::scatter:
            IRRef a = load(thread, inst.a, &inst);
            IRRef b = cast(load(thread, inst.b, &inst), Type::Integer);
            b = insert(trace, TraceOpCode::sub, b, rep(constant(Integer::c(1)), trace[b].out.length), 0, trace[b].type, trace[b].out, trace[b].out);
            IRRef c = load(thread, inst.c, &inst);
            size_t len = std::max(trace[a].out.length, trace[b].out.length);
            Shape s = { len };
            //c = insert(trace, TraceOpCode::dup, c, 0, 0, trace[c].type, trace[c].out, trace[c].out);
            store(thread, insert(trace, TraceOpCode::scatter, rep(a, len), rep(b, len), c, trace[c].type, s, trace[c].out), inst.c);
        }   break;

        case ByteCode::ifelse: {
            IRRef a = load(thread, inst.a, &inst);
            IRRef b = load(thread, inst.b, &inst);
            IRRef c = load(thread, inst.c, &inst);
            store(thread, EmitTernary<IfElse>(TraceOpCode::ifelse, c, b, a), inst.c);
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
            store(thread, insert(trace, TraceOpCode::length, a, 0, 0, Type::Integer, trace[a].out, Shape::Scalar), inst.c);
        }   break;

        case ByteCode::forend:
        {
            IRRef counter = load(thread, inst.c, &inst);
            IRRef vec = load(thread, inst.b, &inst);

            IRRef a = insert(trace, TraceOpCode::length, vec, 0, 0, Type::Integer, trace[vec].out, Shape::Scalar);
            IRRef b = insert(trace, TraceOpCode::lt, counter, a, 0, Type::Logical, Shape::Scalar, Shape::Scalar);
            IRRef c = insert(trace, TraceOpCode::guardT, b, 0, 0, Type::Promise, Shape::Scalar, Shape::Empty);
            reenters[c] = &inst+2;
            store(thread, insert(trace, TraceOpCode::gather, vec, counter, 0, trace[vec].type, Shape::Scalar, Shape::Scalar), inst.a);
            store(thread, insert(trace, TraceOpCode::add, counter, constant(Integer::c(1)), 0, Type::Integer, Shape::Scalar, Shape::Scalar), inst.c); 
        }   break;

        default: {
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

/*

    Instruction scheduling
    Careful CSE to forwarding to minimize cost

    Lift filters and gathers above other instructions.

*/

JIT::IRRef JIT::Insert(std::vector<IR>& code, std::tr1::unordered_map<IR, IRRef>& cse, IR ir) {
    ir = Normalize(ir);
    if(cse.find(ir) != cse.end()) {
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

void JIT::EmitOptIR(
            IRRef i, 
            std::vector<IRRef>& forward, 
            std::map<Variable, IRRef>& loads,
            std::map<Variable, IRRef>& stores,
            std::tr1::unordered_map<IR, IRRef>& cse) {
        IR ir = trace[i];
        switch(ir.op) {
            #define CASE(Name, ...) case TraceOpCode::Name:
            
            case TraceOpCode::LOADENV: 
            {
                forward[i] = Insert(code, cse, IR(ir.op, ir.a, ir.type, ir.in, ir.out));
            } break;

            case TraceOpCode::NEWENV: {
                forward[i] = Insert(code, cse, IR(ir.op, ir.type, ir.in, ir.out));
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
                    Exit e = { stores, reenters[i] };
                    exits[code.size()] = e;
                    forward[i] = loads[v] = 
                        Insert(code, cse, IR(ir.op, v.env, v.i, ir.type, ir.in, ir.out));
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

            case TraceOpCode::PUSH: {
                // do nothing now. Eventually record stack reconstruction info
            } break;
            case TraceOpCode::POP: {
                // rely on store to 0 to kill registers in frame.
                // POP also *might* kill environment. Need a lifetime analysis pass to determine
                //  if environment escapes.
            } break;

            case TraceOpCode::GEQ:
            case TraceOpCode::guardF: 
            case TraceOpCode::guardT: {
                Exit e = { stores, reenters[i] };
                exits[code.size()] = e;
                forward[i] = Insert(code, cse, IR(ir.op, forward[ir.a], ir.b, ir.c, ir.type, ir.in, ir.out));
            } break;
            case TraceOpCode::scatter: {
                forward[i] = Insert(code, cse, IR(ir.op, forward[ir.a], forward[ir.b], forward[ir.c], ir.type, ir.in, ir.out));
            } break;

            case TraceOpCode::length:
            case TraceOpCode::castd: 
            case TraceOpCode::casti: 
            case TraceOpCode::castl: 
            UNARY_FOLD_SCAN_BYTECODES(CASE) {
                forward[i] = Insert(code, cse, IR(ir.op, forward[ir.a], ir.type, ir.in, ir.out));
            } break;

            case TraceOpCode::rep:
            BINARY_BYTECODES(CASE)
            {
                forward[i] = Insert(code, cse, IR(ir.op, forward[ir.a], forward[ir.b], ir.type, ir.in, ir.out));
            } break;

            TERNARY_BYTECODES(CASE)
            {
                forward[i] = Insert(code, cse, IR(ir.op, forward[ir.a], forward[ir.b], forward[ir.c], ir.type, ir.in, ir.out));
            } break;

            CASE(constant)
            {
                forward[i] = Insert(code, cse, IR(ir.op, ir.a, ir.type, ir.in, ir.out));
            } break;

            case TraceOpCode::gather:
            {
                forward[i] = Insert(code, cse, IR(ir.op, forward[ir.a], forward[ir.b], ir.type, ir.in, ir.out));
            } break;

            default:
            {
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
    std::map<Variable, IRRef> stores;
    std::tr1::unordered_map<IR, IRRef> cse;

    // Emit loop header...
    for(size_t i = 0; i < n; i++) {
        EmitOptIR(i, forward, loads, stores, cse);
    }

    // Emit PHIs 
    std::map<Variable, IRRef>::iterator s,l;
    for(s = stores.begin(); s != stores.end(); ++s) {
        //if(s->first.env != (IRRef)-1) {
            l = loads.find(s->first);
            if(l != loads.end() && l->second < s->second) {
                IR const& ir = code[s->second];
                IRRef a = Insert(code, cse, 
                        IR(TraceOpCode::phi, s->second, ir.type, ir.out, ir.out));
                s->second = a;
            }
        //}
    }
    loads.clear();
 
    Insert(code, cse, IR(TraceOpCode::loop, Type::Promise, Shape::Empty, Shape::Empty));
    
    // Emit loop
    for(size_t i = 0; i < n; i++) {
        EmitOptIR(i, forward, loads, stores, cse);
    }

    // Emit MOVs
    for(s = stores.begin(); s != stores.end(); ++s) {
        //if(s->first.env != (IRRef)-1) {
            l = loads.find(s->first);
            if(l != loads.end() && l->second < s->second) {
                IR const& ir = code[s->second];
                IRRef a = Insert(code, cse, 
                        IR(TraceOpCode::mov, l->second, s->second, ir.type, ir.out, ir.out));
                s->second = a;
            }
        //}
    }
    
   
    // Emit the JMP
    Insert(code, cse, IR(TraceOpCode::jmp, Type::Promise, Shape::Empty, Shape::Empty));
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

    Replay(thread);
    schedule();
    RegisterAssignment();
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

void JIT::schedule() {
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
                group[i] = ++g;
            } break;
            case TraceOpCode::phi: {
                group[i] = g;
                group[ir.a] = std::max(group[ir.a], g+1);
            } break;
            case TraceOpCode::mov: {
                group[i] = g;
                group[ir.a] = std::max(group[ir.a], g);
                group[ir.b] = std::max(group[ir.b], g);
            } break;
            case TraceOpCode::GEQ: 
            //case TraceOpCode::GTYPE: 
            case TraceOpCode::guardF: 
            case TraceOpCode::guardT: {
                // Do I also need to update any values that
                // are live out at this exit? Yes.
                group[i] = ++g;
                group[ir.a] = g+1;
                std::map<Variable, IRRef>::const_iterator j;
                for(j = exits[i].o.begin(); j != exits[i].o.end(); ++j) {
                    group[j->second] = g+1;
                }
            } break;
            case TraceOpCode::scatter: {
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
}

void JIT::AssignRegister(size_t index) {
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

void JIT::PreferRegister(size_t index, size_t share) {
    if(assignment[index] == 0) {
        assignment[index] = assignment[share] > 0 ? -assignment[share] : assignment[share];
    }
}

void JIT::ReleaseRegister(size_t index) {
    if(assignment[index] > 0) {
        freeRegisters.insert( std::make_pair(registers[assignment[index]], assignment[index]) );
    }
    else if(assignment[index] < 0) {
        printf("Missing index is %d %d\n", index, assignment[index]);
        _error("Preferred register never assigned");
    }
}

void JIT::RegisterAssignment() {
    // fused operators without a live out don't need a register!
    // is this already taken care of?

    // backwards pass to do register assignment
    // on a node.
    // its register assignment becomes dead, its operands get assigned to registers if not already.
    // have to maintain same memory space on register assignments.
    // try really, really hard to avoid a copy on scatters. handle in a first pass.

    assignment.clear();
    assignment.resize(code.size(), 0);
    
    registers.clear();
    Register invalid = {Type::Promise, Shape::Empty};
    registers.push_back(invalid);
    
    freeRegisters.clear();

    for(size_t i = code.size()-1; i < code.size(); --i) {
        IR& ir = code[i];

        ReleaseRegister(i);
        switch(ir.op) {
            #define CASE(Name, ...) case TraceOpCode::Name:
            case TraceOpCode::loop: {
            } break;
            case TraceOpCode::phi: {
                PreferRegister(ir.a, i);
                AssignRegister(ir.a);
            } break;
            case TraceOpCode::mov: {
                // shouldn't have been assigned a register in the first place.
                // create a register for the second element
                AssignRegister(ir.b);
                PreferRegister(ir.a, ir.b);
                assignment[i] = assignment[ir.b];
            } break;
            case TraceOpCode::GEQ: 
            //case TraceOpCode::GTYPE: 
            case TraceOpCode::guardF: 
            case TraceOpCode::guardT: {
                AssignRegister(ir.a);
                std::map<Variable, IRRef>::const_iterator j;
                for(j = exits[i].o.begin(); j != exits[i].o.end(); ++j) {
                    AssignRegister(j->second);
                }
            } break;
            case TraceOpCode::scatter: {
                AssignRegister(ir.c);
                AssignRegister(ir.a);
                AssignRegister(ir.b);
            } break;
            TERNARY_BYTECODES(CASE)
            {
                AssignRegister(ir.c);
                AssignRegister(ir.b);
                AssignRegister(ir.a);
            } break;
            case TraceOpCode::rep:
            case TraceOpCode::gather:
            BINARY_BYTECODES(CASE)
            {
                AssignRegister(std::max(ir.a, ir.b));
                AssignRegister(std::min(ir.a, ir.b));
            } break;
            case TraceOpCode::length:
            case TraceOpCode::castd: 
            case TraceOpCode::casti: 
            case TraceOpCode::castl: 
            UNARY_FOLD_SCAN_BYTECODES(CASE)
            {
                AssignRegister(ir.a);
            } break;
            default: {
            } break;
            #undef CASE
        }
    }
}

void JIT::IR::dump() const {
    if(type == Type::Double)
        std::cout << "num";
    else if(type == Type::Integer)
        std::cout << "int";
    else if(type == Type::Logical)
        std::cout << "log";
    else if(type == Type::Function)
        std::cout << "fun";
    else if(type == Type::Environment)
        std::cout << "env";
    else
        std::cout << "   ";
    std::cout << in.length << "->" << out.length << "\t ";

    std::cout << TraceOpCode::toString(op);

    switch(op) {
        #define CASE(Name, ...) case TraceOpCode::Name:
        case TraceOpCode::loop: {
            std::cout << " --------------------";
        } break;
        //case TraceOpCode::GTYPE: {
        //} break;
        case TraceOpCode::sload: {
            std::cout << "\t " << (int64_t)b;
        } break;
        case TraceOpCode::eload: {
            std::cout << "\t " << a << "\t\"" << (String)b << "\"";
        } break;
        case TraceOpCode::sstore: {
            std::cout << "\t " << a << "\t " << (int64_t)c;
        } break;    
        case TraceOpCode::estore: {
            std::cout << "\t " << a << "\t " << b << "\t\"" << (String)c << "\"";
        } break;
        case TraceOpCode::mov: {
            std::cout << "\t " << a << "\t " << b;
        } break;
        case TraceOpCode::phi: {
            std::cout << "\t " << a;
        } break;
        case TraceOpCode::PUSH:
        case TraceOpCode::length:
        case TraceOpCode::castd:
        case TraceOpCode::casti:
        case TraceOpCode::castl:
        case TraceOpCode::guardF:
        case TraceOpCode::guardT: 
        UNARY_FOLD_SCAN_BYTECODES(CASE)
        {
            std::cout << "\t " << a;
        } break;
        case TraceOpCode::GEQ:
        {
            std::cout << "\t " << a << "\t [" << b << "]";
        } break;
        case TraceOpCode::LOADENV:
        {
            std::cout << "\t [" << a << "]";
        } break;
        case TraceOpCode::scatter: {
            std::cout << "\t " << a << "\t " << b << "\t " << c;
        } break;
        case TraceOpCode::rep:
        case TraceOpCode::gather:
        BINARY_BYTECODES(CASE)
        {
            std::cout << "\t " << a << "\t " << b;
        } break;
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
    for(size_t i = 0; i < t.size(); i++) {
        IR const& ir = t[i];
        if(ir.op != TraceOpCode::nop) {
            printf("%4d: ", i);
            if(assignment.size() == t.size()) printf(" (%2d) ", assignment[i]);
            if(group.size() == t.size()) printf("%2d ", group[i]);
            ir.dump();
    
            if( exits.size() > 0 && (
                    ir.op == TraceOpCode::GEQ
                ||  ir.op == TraceOpCode::guardF
                ||  ir.op == TraceOpCode::guardT
                ||  ir.op == TraceOpCode::sload
                ||  ir.op == TraceOpCode::eload ) ) {
    
                std::cout << "\t\t=> ";
                Exit const& e = exits[i];
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
        FPM->add(llvm::createCFGSimplificationPass());
        // Provide basic AliasAnalysis support for GVN.
        FPM->add(llvm::createBasicAliasAnalysisPass());
        // Promote allocas to registers.
        FPM->add(llvm::createPromoteMemoryToRegisterPass());
        // Also promote aggregates like structs....
        FPM->add(llvm::createScalarReplAggregatesPass());
        // Do simple "peephole" optimizations and bit-twiddling optzns.
        FPM->add(llvm::createInstructionCombiningPass());
        // Reassociate expressions.
        FPM->add(llvm::createReassociatePass());
        // Eliminate Common SubExpressions.
        FPM->add(llvm::createGVNPass());
        // Simplify the control flow graph (deleting unreachable blocks, etc).
        FPM->add(llvm::createCFGSimplificationPass());
        // Promote allocas to registers.
        FPM->add(llvm::createPromoteMemoryToRegisterPass());
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
    
    llvm::BasicBlock* header;
    llvm::BasicBlock* condition;
    llvm::BasicBlock* body;

    llvm::Value* iterator;
    llvm::Value* length;

    llvm::Constant *zerosD, *onesD;

    size_t width;
    llvm::IRBuilder<> builder;

    std::map<size_t, llvm::Value*> outs;
    std::map<size_t, llvm::Value*> reductions;

    Fusion(JIT& jit, LLVMState* S, llvm::Function* function, std::vector<llvm::Value*> const& values, std::vector<llvm::Value*> const& registers, llvm::Value* length, size_t width)
        : jit(jit)
          , S(S)
          , length(length)
          , function(function)
          , values(values)
          , registers(registers)
          , width(width)
          , builder(*S->C) {
        
        if(llvm::isa<llvm::ConstantInt>(length) && ((llvm::ConstantInt*)length)->getSExtValue() < 16) {
            // short vector, don't emit while loop
            this->width = (size_t)((llvm::ConstantInt*)length)->getZExtValue();
            this->length = 0;
        }

        if(this->width > 0) {
            std::vector<llvm::Constant*> zeros;
            for(size_t i = 0; i < this->width; i++) 
                zeros.push_back(llvm::ConstantFP::get(builder.getDoubleTy(), 0));
            zerosD = llvm::ConstantVector::get(zeros);
        
            std::vector<llvm::Constant*> ones;
            for(size_t i = 0; i < this->width; i++) 
                ones.push_back(llvm::ConstantFP::get(builder.getDoubleTy(), 1));
            onesD = llvm::ConstantVector::get(ones);
        }
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

    llvm::AllocaInst* CreateEntryBlockAlloca(llvm::Type* type, llvm::Value* init) {
        llvm::IRBuilder<> TmpB(&function->getEntryBlock(),
                function->getEntryBlock().begin());
        llvm::AllocaInst* r = TmpB.CreateAlloca(type);
        r->setAlignment(16);
        if(init != 0)
            TmpB.CreateStore(init, r);
        return r;
    }

    void Open(llvm::BasicBlock* before) {
        header = llvm::BasicBlock::Create(*S->C, "fusedHeader", function, before);
        condition = llvm::BasicBlock::Create(*S->C, "fusedCondition", function, before);
        body = llvm::BasicBlock::Create(*S->C, "fusedBody", function, before);

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

        if(outs.find(jit.assignment[ir]) != outs.end())
            return outs[jit.assignment[ir]];

        llvm::Value* a = (jit.assignment[ir] == 0) ? values[ir] : registers[jit.assignment[ir]];
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

    void Store(llvm::Value* a, size_t reg) {

        if(reg != 0) {
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
        std::vector<llvm::Type*> args;
        args.push_back(llvmType(jit.code[ir.a].type));
        args.push_back(llvmType(jit.code[ir.b].type));
        llvm::Type* outTy = llvmType(ir.type);
        llvm::FunctionType* ft = llvm::FunctionType::get(outTy, args, false);
        llvm::Function* f = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, func, S->M);

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

    void Emit(size_t index) {
        JIT::IR ir = jit.code[index];
        size_t reg = jit.assignment[index];

        if(width == 0) return;

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

            case TraceOpCode::castd:
                switch(ir.type) {
                    case Type::Integer: SCALARIZE1(FPToSI, builder.getInt64Ty()); break;
                    case Type::Logical: SCALARIZE1(FCmpONE, llvm::ConstantFP::get(builder.getDoubleTy(), 0)); break;
                    default: _error("Unexpected cast"); break;
                }
                break;
            
            case TraceOpCode::casti:
                switch(ir.type) {
                    case Type::Double: SCALARIZE1(SIToFP, builder.getDoubleTy()); break;
                    case Type::Logical: SCALARIZE1(ICmpEQ, builder.getInt64(0)); break;
                    default: _error("Unexpected cast"); break;
                }
                break;
            
            case TraceOpCode::castl:
                switch(ir.type) {
                    case Type::Double: SCALARIZE1(SIToFP, builder.getDoubleTy()); break;
                    case Type::Integer: SCALARIZE1(ZExt, builder.getInt64Ty()); break;
                    default: _error("Unexpected cast"); break;
                }
                break;
            
            case TraceOpCode::mov:
                if(jit.assignment[ir.a] != jit.assignment[ir.b]) 
                    outs[jit.assignment[ir.a]] = Load(ir.b);
                break;
            
            case TraceOpCode::rep: 
            {
                // there's all sorts of fast variants if lengths are known.
                //if(llvm::isa<llvm::Constant>(a) && llvm::isa<llvm::Constant>(b)) {
                std::vector<llvm::Constant*> c;
                for(size_t i = 0; i < width; i++)
                    c.push_back(builder.getInt64(0));
                outs[jit.assignment[index]] = llvm::ConstantVector::get(c);
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
                outs[jit.assignment[index]] = r;
            } break;
            case TraceOpCode::scatter:
            {
                llvm::Value* v = Load(ir.a);
                llvm::Value* idx = Load(ir.b);
                llvm::Value* r;
                
                if(jit.assignment[ir.c] != jit.assignment[index]) {
                    // must duplicate (copy from the in register to the out). 
                    // Do this in the fusion header.
                    llvm::IRBuilder<> TmpB(header,
                        header->begin());
                    TmpB.CreateMemCpy(RawLoad(index), RawLoad(ir.c), 
                        builder.getInt64(ir.out.length*(ir.type == Type::Logical ? 1 : 8)), 16);
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

            // Reductions
            case TraceOpCode::sum:
            {
                llvm::Value* agg = CreateEntryBlockAlloca(llvmType(ir.type, width), zerosD);
                if(ir.type == Type::Double)
                    builder.CreateStore(builder.CreateFAdd(builder.CreateLoad(agg), Load(ir.a)), agg);
                else
                    builder.CreateStore(builder.CreateAdd(builder.CreateLoad(agg), Load(ir.a)), agg);
                
                reductions[index] = agg;
            } break;
            case TraceOpCode::length:
                reductions[index] = 0;
            break;
            default:
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
            case TraceOpCode::length:
                t = builder.getInt64(jit.code[i].in.length); 
                break;
            default:
                _error("Unsupported reduction");
                break;
        }
        r = builder.CreateInsertElement(r, t, builder.getInt32(0));
        Store(r, jit.assignment[i]); 
    }

    llvm::BasicBlock* Close() {
        std::map<size_t, llvm::Value*>::const_iterator i;
        for(i = outs.begin(); i != outs.end(); i++) {
            Store(i->second, i->first);
        }
        llvm::BasicBlock* after = llvm::BasicBlock::Create(*S->C, "fusedAfter", function, 0);

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
    llvm::BasicBlock * PhiBlock;
    llvm::BasicBlock * LoopStart;
    llvm::BasicBlock * InnerBlock;
    llvm::BasicBlock * EndBlock;
    llvm::IRBuilder<> builder;

    llvm::Type* thread_type;
    llvm::Type* instruction_type;
    llvm::Type* function_type;

    llvm::Value* thread_var;
    llvm::Value* result_var;

    std::vector<llvm::Value*> values;
    std::vector<llvm::Value*> registers;
    std::vector<llvm::CallInst*> calls;
    std::map<JIT::Shape, Fusion*> fusions[100];
    
    LLVMCompiler(Thread& thread, JIT& jit) 
        : thread(thread), jit(jit), S(&llvmState), builder(*S->C) 
    {
        for(int i = 0; i < 100; i++)
            fusions[i].clear();
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

    void* Compile() {
        thread_type = S->M->getTypeByName("class.Thread")->getPointerTo();
        instruction_type = S->M->getTypeByName("struct.Instruction")->getPointerTo();
        function_type = S->M->getTypeByName("struct.Prototype")->getPointerTo();

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
        InnerBlock = llvm::BasicBlock::Create(
                *S->C, "inner", function, 0);
        EndBlock = llvm::BasicBlock::Create(
                *S->C, "end", function, 0);

        llvm::Function::arg_iterator ai = function->arg_begin();
        ai->setName("thread");
        thread_var = ai++;

        result_var = CreateEntryBlockAlloca(instruction_type, builder.getInt64(1));

        builder.SetInsertPoint(EntryBlock);

        // create registers...
        registers.clear();
        registers.push_back(0);
        for(size_t i = 1; i < jit.registers.size(); i++) {
            registers.push_back(
                CreateEntryBlockAlloca(
                    llvmMemoryType(jit.registers[i].type), builder.getInt64(jit.registers[i].shape.length)));
        }
       
        // create values for each ssa node 
        for(size_t i = 0; i < jit.code.size(); i++) {
            if(jit.assignment[i] != 0)
                values.push_back(registers[jit.assignment[i]]);
            else {
                // this case will be filled in as we emit instructions
                values.push_back(0);
            }
        }

 
        for(size_t i = 0; i < jit.code.size(); i++) {
            if(jit.code[i].op != TraceOpCode::nop)
                Emit(jit.code[i], i);
        }
        builder.CreateBr(PhiBlock);
        builder.SetInsertPoint(PhiBlock);
        builder.CreateBr(LoopStart);
        builder.SetInsertPoint(EndBlock);
        builder.CreateRet(builder.CreateLoad(result_var));

        S->FPM->run(*function);
        function->dump();

        return S->EE->getPointerToFunction(function);   
    }

    llvm::AllocaInst* CreateEntryBlockAlloca(llvm::Type* type, llvm::Value* size) {
        llvm::IRBuilder<> TmpB(&function->getEntryBlock(),
                function->getEntryBlock().begin());
        llvm::AllocaInst* r = TmpB.CreateAlloca(type, size);
        r->setAlignment(16);
        return r;
    }

    llvm::Type* llvmMemoryType(Type::Enum type) {
        llvm::Type* t;
        switch(type) {
            case Type::Double: t = builder.getDoubleTy(); break;
            case Type::Integer: t = builder.getInt64Ty(); break;
            case Type::Logical: t = builder.getInt8Ty(); break;
            case Type::Function: t = function_type; break;
            case Type::Promise: t = function_type; break;
            default: _error("Bad type in trace");
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
            case Type::Function: t = function_type; break;
            case Type::Promise: t = function_type; break;
            default: _error("Bad type in trace");
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

    void Emit(JIT::IR ir, size_t index) { 

        if(     ir.op == TraceOpCode::GTYPE
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
        }

        if(ir.op == TraceOpCode::mov &&
            jit.assignment[ir.a] == jit.assignment[ir.b]) {
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
                if(ir.type == Type::Double || ir.type == Type::Integer || ir.type == Type::Logical) {
                    if(ir.type == Type::Double) {
                        Double const& v = (Double const&)jit.constants[ir.a];
                        for(size_t i = 0; i < ir.in.length; i++)
                            c.push_back(llvm::ConstantFP::get(builder.getDoubleTy(), v[i]));
                    } else if(ir.type == Type::Integer) {
                        Integer const& v = (Integer const&)jit.constants[ir.a];
                        for(size_t i = 0; i < ir.in.length; i++)
                            c.push_back(builder.getInt64(v[i]));
                    } else if(ir.type == Type::Logical) {
                        Logical const& v = (Logical const&)jit.constants[ir.a];
                        for(size_t i = 0; i < ir.in.length; i++)
                            c.push_back(builder.getInt8(v[i] != 0 ? 255 : 0));
                    }
                    values[index] = CreateEntryBlockAlloca(llvmMemoryType(ir.type), builder.getInt64(ir.out.length));
                    for(size_t i = 0; i < ir.in.length; i++) {
                        builder.CreateStore(c[i], builder.CreateConstGEP1_64(values[index], i));
                    }
                }
                else if(ir.type == Type::Function || ir.type == Type::Promise || ir.type == Type::Default) {
                        Function const& v = (Function const&)jit.constants[ir.a];
                        values[index] = builder.CreateIntToPtr(builder.getInt64((int64_t)v.prototype()), function_type);
                }
                else if(ir.type == Type::Null) {
                    values[index] = builder.CreateIntToPtr(builder.getInt64(0), builder.getInt8Ty()->getPointerTo());
                }
            } break;

            case TraceOpCode::LOADENV: {
                values[index] = CALL1(std::string("LOAD_environment"), builder.getInt64(ir.a));
            } break;
            case TraceOpCode::sload: 
            {
                values[index] = CALL1(std::string("SLOAD_")+Type::toString(ir.type), builder.getInt64(ir.b));
            } break;
            case TraceOpCode::eload: 
            {
                values[index] = CALL2(std::string("ELOAD_")+Type::toString(ir.type), values[ir.a], builder.getInt64(ir.b));

                llvm::Value* guard = builder.CreateIsNotNull(values[index]);
                EmitExit(guard, jit.exits[index]);
            } break;
            
            case TraceOpCode::GEQ: {
                if(ir.in.length != 1) {
                    _error("Emitting guard on non-scalar");
                }
                llvm::Value* r = builder.CreateICmpEQ(
                    builder.CreatePtrToInt(values[ir.a], builder.getInt64Ty()),
                    builder.getInt64(ir.b));
                EmitExit(r, jit.exits[index]);
            } break;

            case TraceOpCode::guardT:
            case TraceOpCode::guardF: {
                if(ir.in.length != 1) {
                    _error("Emitting guard on non-scalar");
                }
                // TODO: check the NA mask
                llvm::Value* r = builder.CreateTrunc(Load(values[ir.a]), builder.getInt1Ty());
                if(ir.op == TraceOpCode::guardF)
                    r = builder.CreateNot(r);
                EmitExit(r, jit.exits[index]);
            } break;

            case TraceOpCode::castd:
            case TraceOpCode::casti:
            case TraceOpCode::castl:
            case TraceOpCode::rep:
            case TraceOpCode::gather:
            case TraceOpCode::scatter:
            case TraceOpCode::mov: 
            case TraceOpCode::length: 
            TERNARY_BYTECODES(CASE)
            BINARY_BYTECODES(CASE)
            UNARY_FOLD_SCAN_BYTECODES(CASE)
            {
                if(fusions[jit.group[index]].find(ir.in) == fusions[jit.group[index]].end()) {
                    Fusion* f = new Fusion(jit, S, function, values, registers, builder.getInt64(ir.in.length), 4);
                    f->Open(InnerBlock);
                    fusions[jit.group[index]][ir.in] = f;
                }
                Fusion* f = fusions[jit.group[index]][ir.in];

                f->Emit(index);
            } break;
            case TraceOpCode::phi:
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
        
        std::map<JIT::Variable, JIT::IRRef>::const_iterator i;
        for(i = e.o.begin(); i != e.o.end(); i++) {
            
            JIT::Variable v = i->first;
            JIT::IR const& ir = jit.code[i->second];
            llvm::Value* r = values[i->second];
            if(v.i >= 0) {
                if(jit.code[v.env].op == TraceOpCode::LOADENV) {
                    CALL4(std::string("ESTORE_")+Type::toString(ir.type),
                        values[v.env], 
                        builder.getInt64(v.i), 
                        builder.getInt64(ir.out.length), 
                        r);
                }
                else {
                    // TODO: allocate escaping environment and assign to that.
                }
            }
            else {
                CALL3(std::string("SSTORE_")+Type::toString(ir.type),
                        builder.getInt64(v.i), 
                        builder.getInt64(ir.out.length), 
                        r);
            }
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

