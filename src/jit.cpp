
#include "jit.h"
#include "bc.h"
#include "interpreter.h"
#include "call.h"
#include "ops.h"

DEFINE_ENUM_TO_STRING(TraceOpCode, TRACE_ENUM)

JIT::IRRef JIT::insert(
        std::vector<IR>& t,
        TraceOpCode::Enum op, 
        IRRef a, 
        IRRef b, 
        IRRef c,
        Type::Enum type, 
        size_t width) {
    IR ir = (IR) { op, a, b, c, type, width, 0 , 0};
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

JIT::IRRef JIT::load(Thread& thread, int64_t a, Instruction const* reenter) {

    // registers
    OPERAND(operand, a);

    Variable v = intern(thread, a);

    IRRef r;
    if(v.i < 0) {
        r = insert(trace, TraceOpCode::sload, 0, v.i, 0, operand.type, operand.isVector() ? operand.length : 1);
    }
    else {
        r = insert(trace, TraceOpCode::eload, v.env, v.i, 0, operand.type, operand.isVector() ? operand.length : 1);
    }
    trace[r].reenter = reenter;
    return r;
}

JIT::IRRef JIT::cast(IRRef a, Type::Enum type) {
    if(trace[a].type != type) {
        return insert(trace, TraceOpCode::cast, a, 0, 0, type, trace[a].width);
    }
    else {
        return a;
    }
}

JIT::IRRef JIT::rep(IRRef a, size_t width) {
    if(trace[a].width != width) {
        IRRef l = insert(trace, TraceOpCode::length, a, 0, 0, Type::Integer, 1);
        IRRef e = constant(Integer::c(1));
        IRRef r = insert(trace, TraceOpCode::rep, l, e, 0, trace[a].type, width);
        return insert(trace, TraceOpCode::gather, a, r, 0, trace[a].type, width);
    }
    else {
        return a;
    }
}

JIT::IRRef JIT::EmitUnary(TraceOpCode::Enum op, IRRef a, Type::Enum rty, Type::Enum mty) {
   return insert(trace, op, cast(a, mty), 0, 0, rty, trace[a].width);
}

JIT::IRRef JIT::EmitBinary(TraceOpCode::Enum op, IRRef a, IRRef b, Type::Enum rty, Type::Enum maty, Type::Enum mbty) {
    size_t len = 0;
    if(trace[a].width > 0 && trace[b].width > 0)
        len = std::max(trace[a].width, trace[b].width);

    return insert(trace, op, rep(cast(a,maty),len), rep(cast(b,mbty),len), 0, rty, len);
}

JIT::IRRef JIT::EmitTernary(TraceOpCode::Enum op, IRRef a, IRRef b, IRRef c, Type::Enum rty, Type::Enum maty, Type::Enum mbty, Type::Enum mcty) {
    size_t len = 0;
    if(trace[a].width > 0 && trace[b].width > 0 && trace[c].width > 0)
        len = std::max(trace[a].width, std::max(trace[b].width, trace[c].width));

    return insert(trace, op, rep(cast(a,maty),len), rep(cast(b,mbty),len), rep(cast(c,mcty),len), rty, len);
}

JIT::IRRef JIT::store(Thread& thread, IRRef a, int64_t c) {
    Variable v = intern(thread,c);

    if(v.i < 0) {
        insert(trace, TraceOpCode::sstore, a, 0, v.i, trace[a].type, trace[a].width);
    }
    else {
        insert(trace, TraceOpCode::estore, a, v.env, v.i, trace[a].type, trace[a].width);
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
        a = insert(trace, TraceOpCode::constant, ci, 0, 0, value.type, value.length);
        constantsMap[value] = a;
    }
    return a;
}


void JIT::EmitIR(Thread& thread, Instruction const& inst, bool branch) {
    switch(inst.bc) {

        case ByteCode::jc: {
            IRRef p = load(thread, inst.c, &inst);
            
            IRRef r = insert(trace, branch ? TraceOpCode::guardT : TraceOpCode::guardF, 
                p, 0, 0, Type::Promise, trace[p].width );
            trace[r].reenter = &inst;
        }   break;
    
        case ByteCode::call:
        {
            insert(trace, TraceOpCode::PUSH, envs[thread.frame.environment], 0, 0, Type::Promise, 1);
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
            IRRef a = load(thread, inst.a, &inst);
            IRRef b = load(thread, inst.b, &inst);
            store(thread, insert(trace, TraceOpCode::gather, a, cast(b, Type::Integer), 0, trace[a].type, trace[b].width), inst.c);
        }   break;

        case ByteCode::scatter1: {
            IRRef a = load(thread, inst.a, &inst);
            IRRef b = load(thread, inst.b, &inst);
            IRRef c = load(thread, inst.c, &inst);
            store(thread, insert(trace, TraceOpCode::scatter, a, cast(b, Type::Integer), c, trace[c].type, trace[c].width), inst.c);
        }   break;

        #define UNARY_EMIT(Name, string, Group, ...)                      \
        case ByteCode::Name: {                              \
            IRRef a = load(thread, inst.a, &inst);          \
            store(thread, EmitUnary<Group>(TraceOpCode::Name, a), inst.c);  \
        }   break;
        UNARY_BYTECODES(UNARY_EMIT)
        #undef UNARY_EMIT

        #define BINARY_EMIT(Name, string, Group, ...)                      \
        case ByteCode::Name: {                              \
            IRRef a = load(thread, inst.a, &inst);          \
            IRRef b = load(thread, inst.b, &inst);          \
            IRRef r = EmitBinary<Group>(TraceOpCode::Name, a, b); \
            store(thread, r, inst.c);  \
        }   break;
        BINARY_BYTECODES(BINARY_EMIT)
        #undef BINARY_EMIT

        default: {
            _error("Not supported in emit ir");
        }   break;
    }
}

JIT::IRRef JIT::duplicate(IR const& ir, std::vector<IRRef> const& forward) {
    return insert(code, ir.op, forward[ir.a], forward[ir.b], forward[ir.c], ir.type, ir.width);
}

void JIT::EmitOptIR(
            IRRef i, 
            std::vector<IRRef>& forward, 
            std::map<Variable, IRRef>& map,
            std::map<Variable, IRRef>& stores,
            bool LICM) {
        IR ir = trace[i];
        switch(ir.op) {
            #define CASE(Name, ...) case TraceOpCode::Name:
            
            case TraceOpCode::LOADENV: 
            {
                if(!LICM) {
                    forward[i] = insert(code, ir.op, ir.a, 0, 0, ir.type, ir.width);
                }
            } break;

            case TraceOpCode::NEWENV: {
                forward[i] = duplicate(ir, forward);
            } break;

            case TraceOpCode::eload: {
                Variable v = { forward[ir.a], ir.b }; 
                if(map.find(v) == map.end()) {
                    Exit e = { stores, ir.reenter };
                    exits[code.size()] = e;
                    map[v] = insert(code, ir.op, v.env, v.i, 0, ir.type, ir.width);
                }
                // does load-load and load-store forwarding
                forward[i] = map[v];
            } break; 
            case TraceOpCode::sload: {
                Variable v = { 0, ir.b }; 
                if(map.find(v) == map.end()) {
                    Exit e = { stores, ir.reenter };
                    exits[code.size()] = e;
                    map[v] = insert(code, ir.op, 0, v.i, 0, ir.type, ir.width);
                }
                forward[i] = map[v];
            } break;

            case TraceOpCode::estore: {
                Variable v = { forward[ir.b], ir.c };
                stores[v] = map[v] = forward[i] = forward[ir.a];
            } break;
            case TraceOpCode::sstore: {
                Variable v = { 0, ir.c };
                stores[v] = map[v] = forward[i] = forward[ir.a];

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
                if(forward[ir.a] != ir.a) {
                    Exit e = { stores, ir.reenter };
                    exits[code.size()] = e;
                    forward[i] = insert(code, ir.op, forward[ir.a], ir.b, ir.c, ir.type, ir.width);
                }
            } break;
            case TraceOpCode::scatter: {
                if(forward[ir.a] == ir.a &&
                    forward[ir.b] == ir.b &&
                    forward[ir.c] == ir.c) {
                } else {
                    forward[i] = insert(code, ir.op, forward[ir.a], forward[ir.b], forward[ir.c], ir.type, ir.width);
                }
            } break;

            case TraceOpCode::length:
            case TraceOpCode::cast: 
            UNARY_BYTECODES(CASE) {
                if(forward[ir.a] != ir.a) {
                    forward[i] = insert(code, ir.op, forward[ir.a], 0, 0, ir.type, ir.width);
                }
            } break;

            case TraceOpCode::rep:
            BINARY_BYTECODES(CASE)
            {
                if(forward[ir.a] != ir.a || forward[ir.b] != ir.b) {
                    forward[i] = insert(code, ir.op, forward[ir.a], forward[ir.b], 0, ir.type, ir.width);
                }
            } break;

            CASE(constant)
            {
                if(!LICM) {
                    forward[i] = insert(code, ir.op, ir.a, 0, 0, ir.type, ir.width);
                }
                // Do nothing
            } break;

            default:
            {
                _error("Unknown op");
            }

            #undef CASE
        }
}

void JIT::Replay(Thread& thread) {
   
    code.clear();
 
    size_t n = trace.size();
    
    std::vector<IRRef> forward(n, 0);
    std::map<Variable, IRRef> map;
    std::map<Variable, IRRef> stores;

    // Emit loop header...
    for(size_t i = 0; i < n; i++) {
        EmitOptIR(i, forward, map, stores, false);
    }

    std::vector<IRRef> old_forward = forward;

    insert(code, TraceOpCode::loop, 0, 0, 0, Type::Promise, 1);
    
    // Emit loop
    for(size_t i = 0; i < n; i++) {
        EmitOptIR(i, forward, map, stores, true);
    }
   
    // Emit PHIs 
    for(size_t i = 0; i < n; i++) {
        if(trace[i].op == TraceOpCode::estore && forward[i] != old_forward[i]) {
            insert(code, TraceOpCode::phi, old_forward[i], forward[i], 0, trace[i].type, trace[i].width);
        } 
    } 

    // Emit the JMP
    insert(code, TraceOpCode::jmp, 0, 0, 0, Type::Promise, 1);
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

    dump(thread, trace);

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
    size_t group = 1;
    for(size_t i = code.size()-1; i < code.size(); --i) {
        IR& ir = code[i];
        switch(ir.op) {
            #define CASE(Name, ...) case TraceOpCode::Name:
            
            case TraceOpCode::loop: {
                ir.group = ++group;
            } break;
            case TraceOpCode::phi: {
                ir.group = group;
                code[ir.a].group = std::max(code[ir.a].group, group+1);
                code[ir.b].group = std::max(code[ir.b].group, group);
            } break;
            case TraceOpCode::GEQ: 
            //case TraceOpCode::GTYPE: 
            case TraceOpCode::guardF: 
            case TraceOpCode::guardT: {
                // Do I also need to update any values that
                // are live out at this exit? Yes.
                ir.group = ++group;
                code[ir.a].group = group+1;
                std::map<Variable, IRRef>::const_iterator j;
                for(j = exits[i].o.begin(); j != exits[i].o.end(); ++j) {
                    code[j->second].group = group+1;
                }
            } break;
            case TraceOpCode::scatter: {
                code[ir.a].group = 
                    std::max(code[ir.a].group,
                        ir.group);
                code[ir.b].group = 
                    std::max(code[ir.b].group,
                        ir.group);
                code[ir.c].group = 
                    std::max(code[ir.c].group,
                        ir.group+1);
            } break;
            case TraceOpCode::gather: {
                code[ir.a].group = 
                    std::max(code[ir.a].group,
                        ir.group+1);
                code[ir.b].group = 
                    std::max(code[ir.b].group,
                        ir.group);
            } break;
            BINARY_BYTECODES(CASE)
            {
                code[ir.a].group = 
                    std::max(code[ir.a].group,
                        ir.group);
                code[ir.b].group = 
                    std::max(code[ir.b].group,
                        ir.group);
            } break;
            case TraceOpCode::cast: 
            UNARY_BYTECODES(CASE) 
            {
                code[ir.a].group = 
                    std::max(code[ir.a].group,
                        ir.group);
            } break;
            case TraceOpCode::rep: {
                code[ir.a].group =
                    std::max(code[ir.a].group, ir.group+1); 
                code[ir.b].group =
                    std::max(code[ir.b].group, ir.group+1); 
            } break;
            case TraceOpCode::length: {
                code[ir.a].group =
                    std::max(code[ir.a].group, ir.group+1); 
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
 
        Register r = { ir.type, ir.width }; 

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
    Register invalid = {Type::Promise,0};
    registers.push_back(invalid);
    
    freeRegisters.clear();

    size_t group = 1;
    for(size_t i = code.size()-1; i < code.size(); --i) {
        IR& ir = code[i];

        ReleaseRegister(i);
        switch(ir.op) {
            #define CASE(Name, ...) case TraceOpCode::Name:
            case TraceOpCode::loop: {
            } break;
            case TraceOpCode::phi: {
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
            case TraceOpCode::rep:
            case TraceOpCode::gather:
            BINARY_BYTECODES(CASE)
            {
                AssignRegister(ir.a);
                AssignRegister(ir.b);
            } break;
            case TraceOpCode::length:
            case TraceOpCode::cast: 
            UNARY_BYTECODES(CASE)
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
    printf("%2d ", group);
    if(type == Type::Double)
        std::cout << "num" << width << "\t";
    else if(type == Type::Integer)
        std::cout << "int" << width << "\t";
    else if(type == Type::Logical)
        std::cout << "log" << width << "\t";
    else if(type == Type::Function)
        std::cout << "fun" << width << "\t";
    else if(type == Type::Environment)
        std::cout << "env" << width << "\t";
    else
        std::cout << "\t\t";

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
        case TraceOpCode::phi: {
            std::cout << "\t " << a << "\t " << b;
        } break;
        case TraceOpCode::PUSH:
        case TraceOpCode::length:
        case TraceOpCode::cast:
        case TraceOpCode::guardF:
        case TraceOpCode::guardT: 
        UNARY_BYTECODES(CASE)
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
            ir.dump();
    
            if( exits.size() > 0 && (
                    ir.op == TraceOpCode::GEQ
                ||  ir.op == TraceOpCode::guardF
                ||  ir.op == TraceOpCode::guardT ) ) {
    
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
    
    llvm::BasicBlock* header;
    llvm::BasicBlock* condition;
    llvm::BasicBlock* body;

    llvm::Value* iterator;
    llvm::Value* length;

    size_t width;
    llvm::IRBuilder<> builder;

    std::map<llvm::Value*, llvm::Value*> outs;
    std::map<llvm::Value*, size_t> outs_ir;


    Fusion(JIT& jit, LLVMState* S, llvm::Function* function, std::vector<llvm::Value*> const& values, llvm::Value* length, size_t width)
        : jit(jit)
          , S(S)
          , length(length)
          , function(function)
          , values(values)
          , width(width)
          , builder(*S->C) {
        
        if(llvm::isa<llvm::ConstantInt>(length) && ((llvm::ConstantInt*)length)->getSExtValue() < 16) {
            // short vector, don't emit while loop
            this->width = (size_t)((llvm::ConstantInt*)length)->getZExtValue();
            this->length = 0;
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
        llvm::Value* a = RawLoad(ir);

        if(outs.find(a) != outs.end())
            return outs[a];

        llvm::Type* t = llvm::VectorType::get(
            ((llvm::SequentialType*)a->getType())->getElementType(),
            width)->getPointerTo();
        a = builder.CreateInBoundsGEP(a, iterator);
        a = builder.CreatePointerCast(a, t);
        a = builder.CreateLoad(a);

        if(jit.code[ir].type == Type::Logical)
            a = builder.CreateTrunc(a, llvm::VectorType::get(builder.getInt1Ty(), width));
    
        return a;
    }

    void Store(llvm::Value* a, size_t ir, llvm::Value* out) {

        if(jit.code[ir].type == Type::Logical)
            a = builder.CreateSExt(a, llvm::VectorType::get(builder.getInt8Ty(), width));

        out = builder.CreateInBoundsGEP(out, iterator);
        
        llvm::Type* t = llvm::VectorType::get(
            ((llvm::SequentialType*)a->getType())->getElementType(),
            width)->getPointerTo();
        
        out = builder.CreatePointerCast(out, t);

        builder.CreateStore(a, out);
    }

    void Emit(size_t index) {
        JIT::IR ir = jit.code[index];
        llvm::Type* t = llvmType(ir.type, width);

        llvm::Value* out = RawLoad(index);
        // Create an output vector...
        // 
        switch(ir.op) {
            case TraceOpCode::pos: 
                {
                    outs[out] = Load(ir.a);
                    outs_ir[out] = index;
                }   break;
            case TraceOpCode::neg: 
                {
                    outs[out] = builder.CreateFNeg(Load(ir.a));
                    outs_ir[out] = index;
                }   break;
            case TraceOpCode::add: 
                {
                    outs[out] = builder.CreateFAdd(Load(ir.a), Load(ir.b));
                    outs_ir[out] = index;
                }   break;
            case TraceOpCode::sub: 
                {
                    outs[out] = builder.CreateFSub(Load(ir.a), Load(ir.b));
                    outs_ir[out] = index;
                }   break;
            case TraceOpCode::mul: 
                {
                    outs[out] = builder.CreateFMul(Load(ir.a), Load(ir.b));
                    outs_ir[out] = index;
                }   break;
            case TraceOpCode::div: 
                {
                    outs[out] = builder.CreateFDiv(Load(ir.a), Load(ir.b));
                    outs_ir[out] = index;
                }   break;
            case TraceOpCode::lt: 
                {
                    outs[out] = builder.CreateFCmpOLT(Load(ir.a), Load(ir.b));
                    outs_ir[out] = index;
                } break;
            case TraceOpCode::lor: 
                {
                    outs[out] = builder.CreateOr(Load(ir.a), Load(ir.b));
                    outs_ir[out] = index;
                } break;
            case TraceOpCode::land: 
                {
                    outs[out] = builder.CreateAnd(Load(ir.a), Load(ir.b));
                    outs_ir[out] = index;
                } break;
            case TraceOpCode::lnot: 
                {
                    outs[out] = builder.CreateNot(Load(ir.a));
                    outs_ir[out] = index;
                } break;
            case TraceOpCode::phi: 
                {
                    if(jit.assignment[ir.a] != jit.assignment[ir.b]) {
                        outs[RawLoad(ir.a)] = Load(ir.b);
                        outs_ir[out] = index;
                    }
                } break;
            case TraceOpCode::rep: 
            {
                // there's all sorts of fast variants if lengths are known.
                //if(llvm::isa<llvm::Constant>(a) && llvm::isa<llvm::Constant>(b)) {
                std::vector<llvm::Constant*> c;
                for(size_t i = 0; i < width; i++)
                    c.push_back(builder.getInt64(0));
                outs[RawLoad(ir.a)] = llvm::ConstantVector::get(c);
                //}
                //else {
                //    _error("Unsupported rep");
                // }
            } break;
            case TraceOpCode::gather: 
            {
                llvm::Value* v = RawLoad(ir.a);
                llvm::Value* idx = Load(ir.b);
                llvm::Type* trunc = llvm::VectorType::get(builder.getInt32Ty(), width);
                idx = builder.CreateTrunc(idx, trunc);  
                // scalarize the gather...
                llvm::Value* r = llvm::UndefValue::get(t);

                for(size_t i = 0; i < width; i++) {
                    llvm::Value* ii = builder.getInt32((uint32_t)i);
                    llvm::Value* j = builder.CreateExtractElement(idx, ii);
                    j = builder.CreateGEP(v, j);
                    r = builder.CreateInsertElement(r, j, ii);
                }
                outs[RawLoad(ir.a)] = r;
            };
        }
    }

    llvm::BasicBlock* Close() {
        std::map<llvm::Value*, llvm::Value*>::const_iterator i;
        for(i = outs.begin(); i != outs.end(); i++) {
            Store(i->second, outs_ir[i->first], i->first);
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

    llvm::Value* thread_var;
    llvm::Value* result_var;

    std::vector<llvm::Value*> values;
    std::vector<llvm::CallInst*> calls;
    std::map<size_t, Fusion*> fusions[100];
    
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
        std::vector<llvm::Value*> registers;
        registers.push_back(0);
        for(size_t i = 1; i < jit.registers.size(); i++) {
            registers.push_back(
                CreateEntryBlockAlloca(
                    llvmMemoryType(jit.registers[i].type), builder.getInt64(jit.registers[i].length)));
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
                Emit(jit.code[i], jit.code[i].width, i);
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
        return TmpB.CreateAlloca(type, size);
    }

    llvm::Type* llvmMemoryType(Type::Enum type) {
        llvm::Type* t;
        switch(type) {
            case Type::Double: t = builder.getDoubleTy(); break;
            case Type::Integer: t = builder.getInt64Ty(); break;
            case Type::Logical: t = builder.getInt8Ty(); break;
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

    void Emit(JIT::IR ir, size_t width, size_t index) {
        

        if(     ir.op == TraceOpCode::GTYPE
            ||  ir.op == TraceOpCode::guardT
            ||  ir.op == TraceOpCode::guardF
            ||  ir.op == TraceOpCode::jmp
            ||  ir.op == TraceOpCode::loop) {
            for(int i = 99; i > ir.group; i--) {
                std::map<size_t, Fusion*>::iterator j;
                for(j = fusions[i].begin(); j != fusions[i].end(); ++j) {
                    Fusion* f = j->second;
                    builder.CreateBr(f->header);
                    builder.SetInsertPoint(f->Close());
                }
                fusions[i].clear();
            }
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
                if(ir.type == Type::Double) {
                    Double const& v = (Double const&)jit.constants[ir.a];
                    for(size_t i = 0; i < width; i++)
                        c.push_back(llvm::ConstantFP::get(builder.getDoubleTy(), v[i]));
                } else if(ir.type == Type::Integer) {
                    Integer const& v = (Integer const&)jit.constants[ir.a];
                    for(size_t i = 0; i < width; i++)
                        c.push_back(builder.getInt64(v[i]));
                } else if(ir.type == Type::Logical) {
                    Logical const& v = (Logical const&)jit.constants[ir.a];
                    for(size_t i = 0; i < width; i++)
                        c.push_back(builder.getInt1(v[i] != 0));
                } else {
                    _error("Unexpected constant type");
                }
                values[index] = CreateEntryBlockAlloca(llvmMemoryType(ir.type), builder.getInt64(ir.width));
                for(size_t i = 0; i < width; i++) {
                    builder.CreateStore(c[i], builder.CreateConstGEP1_64(values[index], i));
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
                if(jit.code[ir.a].width != 1) {
                    _error("Emitting guard on non-scalar");
                }
                llvm::Value* r = builder.CreateICmpEQ(
                    builder.CreatePtrToInt(values[ir.a], builder.getInt64Ty()),
                    builder.getInt64(ir.b));
                EmitExit(r, jit.exits[index]);
            } break;

            case TraceOpCode::guardT:
            case TraceOpCode::guardF: {
                if(jit.code[ir.a].width != 1) {
                    _error("Emitting guard on non-scalar");
                }
                // TODO: check the NA mask
                llvm::Value* r = builder.CreateTrunc(Load(values[ir.a]), builder.getInt1Ty());
                if(ir.op == TraceOpCode::guardF)
                    r = builder.CreateNot(r);
                EmitExit(r, jit.exits[index]);
            } break;

            case TraceOpCode::rep:
            case TraceOpCode::gather:
            case TraceOpCode::phi: 
            BINARY_BYTECODES(CASE)
            UNARY_BYTECODES(CASE)
            {
                if(fusions[ir.group].find(width) == fusions[ir.group].end()) {
                    Fusion* f = new Fusion(jit, S, function, values, builder.getInt64(width), 4);
                    f->Open(InnerBlock);
                    fusions[ir.group][width] = f;
                }
                Fusion* f = fusions[ir.group][width];

                f->Emit(index);
            } break;
            
            case TraceOpCode::scatter:
            {
                llvm::Value* v = Load(values[ir.a]);
                llvm::Value* idx = Load(values[ir.b]);
                llvm::Value* r;
                if(jit.assignment[ir.c] != jit.assignment[index]) {
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
                else {
                    // reusing register, just assign in place.
                    llvm::Type* mt = llvmType(ir.type)->getPointerTo();
                    llvm::Value* x = builder.CreatePointerCast(Load(values[ir.c]), mt);
                    for(size_t i = 0; i < jit.code[ir.b].width; i++) {
                        llvm::Value* ii = builder.getInt32((uint32_t)i);
                        llvm::Value* j = builder.CreateExtractElement(v, ii);
                        ii = builder.CreateExtractElement(idx, ii);
                        builder.CreateStore(j,
                            builder.CreateGEP(x, ii));
                    }
                } 
            } break;
            case TraceOpCode::cast:
            {
                llvm::Type* t = llvmType(ir.type, width);
                llvm::Value* r = llvm::UndefValue::get(t);
                switch(ir.type) {
                    case Type::Double:
                    break;
                    case Type::Integer: {
                        switch(jit.code[ir.a].type) {
                            case Type::Double: {
                                for(size_t i = 0; i < width; i++) {
                                    llvm::Value* ii = builder.getInt32((uint32_t)i);
                                    llvm::Value* j = builder.CreateExtractElement(Load(values[ir.a]), ii);
                                    j = builder.CreateFPToSI(j, builder.getInt64Ty());
                                    r = builder.CreateInsertElement(r, j, ii);
                                }
                            } break;
                        }
                    } break;
                    case Type::Logical: {
                    } break;
                    default: {
                    } break;
                }
            } break;
            
            case TraceOpCode::length:
            {
                llvm::Type* t = llvmType(ir.type, width);
                llvm::Value* r = llvm::UndefValue::get(t);
                r = builder.CreateInsertElement(r, builder.getInt64(jit.code[ir.a].width), builder.getInt32(0));
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
                _error("Unknown op");
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
            if(v.i >= 0) {
                if(jit.code[v.env].op == TraceOpCode::LOADENV) {
                    CALL4(std::string("ESTORE_")+Type::toString(ir.type),
                        values[v.env], 
                        builder.getInt64(v.i), 
                        builder.getInt64(ir.width), 
                        values[i->second]);
                }
                else {
                    // TODO: allocate escaping environment and assign to that.
                }
            }
            else {
                CALL3(std::string("SSTORE_")+Type::toString(ir.type),
                        builder.getInt64(v.i), 
                        builder.getInt64(ir.width), 
                        values[i->second]);
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

