
#include "jit.h"
#include "bc.h"
#include "interpreter.h"
#include "call.h"

DEFINE_ENUM_TO_STRING(TraceOpCode, TRACE_ENUM)

JIT::IRRef JIT::insert(
        TraceOpCode::Enum op, 
        IRRef a, 
        IRRef b, 
        IRRef c,
        int64_t target, 
        Type::Enum type, 
        size_t width) {
    IR ir = (IR) { op, a, b, c, target, type, width, 0 };
    code.push_back(ir);
    return (IRRef) { code.size()-1 };
}

int64_t intern(Thread& thread, int64_t a) {
    if(a <= 0) {
        return (thread.base+a)-(thread.registers+DEFAULT_NUM_REGISTERS);
    }
    else {
        return a;
    }
}

JIT::IRRef JIT::load(Thread& thread, int64_t a, Instruction const* reenter) {
    
    std::map<int64_t, IRRef>::const_iterator i;
    i = map.find(intern(thread,a));
    if(i != map.end()) {
        return i->second;
    }
    else {
        OPERAND(operand, a);
        
        Exit e = { map, reenter };
        exits[code.size()] = e;
        insert(TraceOpCode::GTYPE, 0, 0, 0, a, operand.type, operand.length);
        
        loads[intern(thread,a)] = (IRRef){code.size()};
        map[intern(thread,a)] = (IRRef){code.size()};
        return insert(TraceOpCode::load, 0, 0, 0, a, operand.type, operand.length);
    }
}

JIT::IRRef JIT::emit(Thread& thread, TraceOpCode::Enum op, IRRef a, IRRef b, int64_t c) {
    Value const& v = OUT(thread, c);
    return insert(op, a, b, 0, 0, v.type, v.length);
}

JIT::IRRef JIT::emit(Thread& thread, TraceOpCode::Enum op, IRRef a, IRRef b, IRRef c, int64_t d) {
    Value const& v = OUT(thread, d);
    return insert(op, a, b, c, 0, v.type, v.length);
}

JIT::IRRef JIT::store(Thread& thread, IRRef a, int64_t c) {
    insert(TraceOpCode::store, a, 0, 0, intern(thread,c), code[a].type, code[a].width);
    if(loads.find(intern(thread,c)) == loads.end()) {
        loads[intern(thread,c)] = a;
    }
    map[intern(thread,c)] = a;
    return a;
}

void JIT::EmitIR(Thread& thread, Instruction const& inst, bool branch) {
    switch(inst.bc) {

        case ByteCode::jc: {
            IRRef p = load(thread, inst.c, &inst);
            
            Exit e = { map, branch ? &inst+inst.b : &inst+inst.a };
            exits[code.size()] = e;
            markLiveOut(e);
            
            insert(branch ? TraceOpCode::guardT : TraceOpCode::guardF, 
                p, 0, 0, 0, Type::Promise, code[p].width );
        }   break;

        case ByteCode::constant: {
            Value const& c = thread.frame.prototype->constants[inst.a];
            store(thread, insert(TraceOpCode::constant, 0, 0, 0, (int64_t)&c, c.type, c.length), inst.c);
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
            if(code[b].type != Type::Integer)
                b = insert(TraceOpCode::cast, b, 0, 0, 0, Type::Integer, code[b].width);
            store(thread, emit(thread, TraceOpCode::gather, a, b, inst.c), inst.c);
        }   break;

        case ByteCode::scatter1: {
            IRRef a = load(thread, inst.a, &inst);
            IRRef b = load(thread, inst.b, &inst);
            IRRef c = load(thread, inst.c, &inst);
            if(code[b].type != Type::Integer)
                b = insert(TraceOpCode::cast, b, 0, 0, 0, Type::Integer, code[b].width);
            store(thread, emit(thread, TraceOpCode::scatter, a, b, c, inst.c), inst.c);
        }   break;

        #define BINARY_EMIT(Name, ...)                      \
        case ByteCode::Name: {                              \
            IRRef a = load(thread, inst.a, &inst);          \
            IRRef b = load(thread, inst.b, &inst);          \
            store(thread, emit(thread, TraceOpCode::Name, a, b, inst.c), inst.c);  \
        }   break;
        BINARY_BYTECODES(BINARY_EMIT)
        #undef BINARY_EMIT

        default: {
            _error("Not supported in emit ir");
        }   break;
    }
}

void JIT::Replay(Thread& thread) {
    
    IRRef forward[1024];
   
    for(size_t i = 0; i < 1024; i++)
        forward[i] = i;

    size_t n = code.size()-2;   // don't replay the loop marker or the last guard.
    for(size_t i = 0; i < n; i++) {
        IR& ir = code[i];
        IRRef a, b, c;
        switch(ir.op) {

            case TraceOpCode::loop: {
            } break;

            case TraceOpCode::phi: {
                ir.b = forward[ir.a];
            } break;

            case TraceOpCode::load: {
                // forwards to last store
                forward[i] = map[ir.target];
            } break;

            case TraceOpCode::constant: {
            } break;

            case TraceOpCode::store: {
                map[ir.target] = ir.a;
            } break;

            case TraceOpCode::GTYPE: {
            } break;

            case TraceOpCode::guardF: 
            case TraceOpCode::guardT: {
                if(forward[ir.a] == ir.a) {
                } else {
                    Exit e = { map, exits[i].reenter };
                    exits[code.size()] = e;
                    forward[i] = insert(ir.op, forward[ir.a], 0, 0, 0, ir.type, ir.width);
                }
            } break;
            case TraceOpCode::scatter: {
                if(forward[ir.a] == ir.a &&
                    forward[ir.b] == ir.b &&
                    forward[ir.c] == ir.c) {
                } else {
                    forward[i] = insert(ir.op, forward[ir.a], forward[ir.b], forward[ir.c], 0, ir.type, ir.width);
                }
            } break;
            case TraceOpCode::cast: {
                if(forward[ir.a] == ir.a) {
                } else {
                    forward[i] = insert(ir.op, forward[ir.a], 0, 0, 0, ir.type, ir.width);
                }
            } break;
            default: {
                if(forward[ir.a] == ir.a &&
                    forward[ir.b] == ir.b) {
                } else {
                    forward[i] = insert(ir.op, forward[ir.a], forward[ir.b], 0, 0, ir.type, ir.width);
                }
            } break;
        }
    }
    
    // Emit the PHIs 
    for(std::map<int64_t, IRRef>::const_iterator i = loads.begin(); i != loads.end(); ++i) {
        std::map<int64_t, IRRef>::const_iterator j = map.find(i->first);
        IR const& in = code[i->second];
        if(j->second != forward[j->second])
            insert(TraceOpCode::phi, j->second, forward[j->second], 0, i->first, in.type, in.width);
    }

    // Emit the JMP
    insert(TraceOpCode::jmp, 0, 0, 0, 0, Type::Promise, 1);
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

    //dump();
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
            case TraceOpCode::loop: {
                ir.group = group;
            } break;
            case TraceOpCode::phi: {
                ir.group = group;
                code[ir.a].group = std::max(code[ir.a].group, group+1);
                code[ir.b].group = std::max(code[ir.b].group, group);
            } break;
            case TraceOpCode::GTYPE: 
            case TraceOpCode::guardF: 
            case TraceOpCode::guardT: {
                // Do I also need to update any values that
                // are live out at this exit? Yes.
                ir.group = ++group;
                code[ir.a].group = group+1;
                std::map<int64_t, IRRef>::const_iterator j;
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
                        ir.group);
            } break;
            case TraceOpCode::gather:
            case TraceOpCode::lt:
            case TraceOpCode::add: {
                code[ir.a].group = 
                    std::max(code[ir.a].group,
                        ir.group);
                code[ir.b].group = 
                    std::max(code[ir.b].group,
                        ir.group);
            } break;
            case TraceOpCode::cast: {
                code[ir.a].group = 
                    std::max(code[ir.a].group,
                        ir.group);
            } break;
            default: {
            } break;
        }
    }
}

void JIT::AssignRegister(size_t index) {
    if(assignment[index] <= 0) {
        IR const& ir = code[index];
        if(ir.op == TraceOpCode::load || ir.op == TraceOpCode::constant) {
            assignment[index] = 0;
            return;
        }
        size_t size = ir.width * (ir.type == Type::Logical ? 1 : 8);

        // if there's a preferred register look for that first.
        if(assignment[index] < 0) {
            std::pair<std::multimap<size_t, size_t>::iterator,std::multimap<size_t, size_t>::iterator> ret;
            ret = freeRegisters.equal_range(-assignment[index]);
            for (std::multimap<size_t, size_t>::iterator it = ret.first; it != ret.second; ++it) {
                if(it->second == -assignment[index]) {
                    assignment[index] = it->second;
                    freeRegisters.erase(it); 
                    return;
                }
            }
        }

        // if no preferred or preferred wasn't available fall back to any available or create new.
        std::map<size_t, size_t>::iterator i = freeRegisters.find(size);
        if(i != freeRegisters.end()) {
            assignment[index] = i->second;
            freeRegisters.erase(i);
            return;
        }
        else {
            assignment[index] = registers.size();
            registers.push_back(size);
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

    // backwards pass to do register assignment
    // on a node.
    // its register assignment becomes dead, its operands get assigned to registers if not already.
    // have to maintain same memory space on register assignments.
    // try really, really hard to avoid a copy on scatters. handle in a first pass.

    assignment.clear();
    assignment.resize(code.size(), 0);
    
    registers.clear();
    registers.push_back(0);
    
    freeRegisters.clear();

    size_t group = 1;
    for(size_t i = code.size()-1; i < code.size(); --i) {
        IR& ir = code[i];

        ReleaseRegister(i);
        switch(ir.op) {
            case TraceOpCode::loop: {
            } break;
            case TraceOpCode::phi: {
                // shouldn't have been assigned a register in the first place.
                // create a register for the second element
                AssignRegister(ir.b);
                PreferRegister(ir.a, ir.b);
            } break;
            case TraceOpCode::GTYPE: 
            case TraceOpCode::guardF: 
            case TraceOpCode::guardT: {
                std::map<int64_t, IRRef>::const_iterator j;
                for(j = exits[i].o.begin(); j != exits[i].o.end(); ++j) {
                    AssignRegister(j->second);
                }
            } break;
            case TraceOpCode::scatter: {
                AssignRegister(ir.a);
                AssignRegister(ir.b);
                AssignRegister(ir.c);
            } break;
            case TraceOpCode::gather:
            case TraceOpCode::lt:
            case TraceOpCode::add: {
                AssignRegister(ir.a);
                AssignRegister(ir.b);
            } break;
            case TraceOpCode::cast: {
                AssignRegister(ir.a);
            } break;
            default: {
            } break;
        } 
    }
}

void JIT::IR::dump() {
    printf("%2d ", group);
    if(type == Type::Double)
        std::cout << "num" << width << "\t";
    else if(type == Type::Integer)
        std::cout << "int" << width << "\t";
    else if(type == Type::Logical)
        std::cout << "log" << width << "\t";
    else
        std::cout << "\t";

    std::cout << TraceOpCode::toString(op);

    switch(op) {
        case TraceOpCode::loop: {
            std::cout << " --------------------";
        } break;
        case TraceOpCode::GTYPE:
        case TraceOpCode::load: {
            if(target <= 0)
                std::cout << "\t " << (int64_t)target;
            else
                std::cout << "\t " << (String)target;
        } break;
        case TraceOpCode::store: {
            if(target <= 0)
                std::cout << "\t " << a << "\t " << (int64_t)target;
            else
                std::cout << "\t " << a << "\t " << (String)target;
        } break;
        case TraceOpCode::phi: {
            if(target <= 0)
                std::cout << "\t " << a << "\t " << b << "\t " << (int64_t)target;
            else
                std::cout << "\t " << a << "\t " << b << "\t " << (String)target;
        } break;
        case TraceOpCode::cast:
        case TraceOpCode::guardF:
        case TraceOpCode::guardT: {
            std::cout << "\t " << a;
        } break;
        case TraceOpCode::scatter: {
            std::cout << "\t " << a << "\t " << b << "\t " << c;
        } break;
        case TraceOpCode::gather:
        case TraceOpCode::add:
        case TraceOpCode::lt: {
            std::cout << "\t " << a << "\t " << b;
        } break;
        default: {} break;
    };

    //if(liveout)
    //    std::cout << "\t=>";
}

void JIT::dump() {
    for(size_t i = 0; i < code.size(); i++) {
        if(code[i].op != TraceOpCode::nop && code[i].op != TraceOpCode::store) {
            printf("%4d: ", i);
            printf(" (%2d) ", assignment[i]);
            code[i].dump();
    
            if(     code[i].op == TraceOpCode::GTYPE
                ||  code[i].op == TraceOpCode::guardF
                ||  code[i].op == TraceOpCode::guardT ) {
    
                std::cout << "\t\t=> ";
                Exit const& e = exits[i];
                for(std::map<int64_t, IRRef>::const_iterator i = e.o.begin(); i != e.o.end(); ++i) {
                    std::cout << i->second << ":";
                    if(i->first > 0) std::cout << (char*)i->first << " ";
                    else std::cout << i->first << " ";
                }
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

struct LLVMCompiler {
    Thread& thread;
    JIT& jit;
    LLVMState* S;
    llvm::Function * function;
    llvm::BasicBlock * EntryBlock;
    llvm::BasicBlock * PhiBlock;
    llvm::BasicBlock * LoopStart;
    llvm::BasicBlock * EndBlock;
    llvm::BasicBlock * PrePhiBlock;
    llvm::IRBuilder<> builder;

    llvm::Type* thread_type;
    llvm::Type* instruction_type;

    llvm::Value* thread_var;
    llvm::Value* result_var;

    llvm::Value* values[1024];

    //std::vector<llvm::Value*> registers;
    llvm::Value* registers[100];

    std::vector<llvm::CallInst*> calls;

    LLVMCompiler(Thread& thread, JIT& jit) 
        : thread(thread), jit(jit), S(&llvmState), builder(*S->C) 
    {
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
        PhiBlock = llvm::BasicBlock::Create(
                *S->C, "phis", function, 0);
        LoopStart = llvm::BasicBlock::Create(
                *S->C, "loop", function, 0);
        EndBlock = llvm::BasicBlock::Create(
                *S->C, "end", function, 0);

        llvm::Function::arg_iterator ai = function->arg_begin();
        ai->setName("thread");
        thread_var = ai++;

        result_var = CreateEntryBlockAlloca(instruction_type);

        builder.SetInsertPoint(EntryBlock);

        // create registers...
        /*registers.push_back(0);
        for(size_t i = 1; i < jit.registers.size(); i++) {
            registers.push_back(CALL1("Alloc", builder.getInt64(jit.registers[i])));
        }*/

        //registers.resize(jit.registers.size(), 0);
        for(int i = 0; i < 100; i++) registers[i] = 0;
        for(size_t i = 0; i < jit.code.size(); i++) {
            if(jit.assignment[i] != 0) {
                size_t n = jit.assignment[i];
                if(registers[n] == 0) {
                    llvm::Value* r = CreateEntryBlockAlloca(llvmType(jit.code[i].type, jit.code[i].width));
                    registers[n] = r;
                }
            }
        }
        // split up code by execution order and size.
        // emit each into own block
        //

        // Control flow unites at group changes...
        //  

        /* Conceptually:

                Split trace into semi-scalar basic blocks ("strand")
                References to ops in another strand are replaced with loads.
                No control flow inside strand
                    All type checks and length checks are emitted in a header and all exit to the
                    same instruction.
                HARD: once I start reordering instructions, which instruction should I
                    return to if a guard fails.
                    Limit reordering to within a particular execution group.
                    Entire execution group runs or doesn't atomically.
                    Downside to bigger execution groups? Not really.
                    Think of it like delayed execution still.
                    Still need to see through control flow, move evaluation to side exits.
                Record like it's delayed evaluation???
                Record scalar code directly, put vector ops in futures, when forced to evaluate,
                    record future trace into buffer.
                For side exits record forcing of the futures in the side exit.
                Should futures be stored in the trace when forced or separately?
                Can we do the deferred evaluation reordering with a static pass over a trace?? 
        */
        for(size_t i = 0; i < 1024; i++) {
            values[i] = 0;
        }        

 
        for(size_t i = 0; i < jit.code.size(); i++) {
            if(jit.code[i].op != TraceOpCode::nop)
                values[i] = Emit(jit.code[i], jit.code[i].width, i);
        }
        builder.CreateBr(PhiBlock);
        builder.SetInsertPoint(PhiBlock);
        builder.CreateBr(LoopStart);
        builder.SetInsertPoint(EndBlock);
        builder.CreateRet(builder.CreateLoad(result_var));

        for(size_t i = 0; i < calls.size(); i++) {
            if (calls[i]->getCalledFunction()
                ->hasFnAttr(llvm::Attribute::AlwaysInline)) {
                    llvm::InlineFunctionInfo ifi;
                    llvm::InlineFunction(calls[i], ifi);
            }
        }
    
        S->FPM->run(*function);
        //function->dump();

        return S->EE->getPointerToFunction(function);   
    }

    llvm::AllocaInst* CreateEntryBlockAlloca(llvm::Type* type) {
        llvm::IRBuilder<> TmpB(&function->getEntryBlock(),
                function->getEntryBlock().begin());
        return TmpB.CreateAlloca(type);
    }

    std::string postfix(Type::Enum type) {
        switch(type) {
            case Type::Double: return "d";
            case Type::Integer: return "i";
            case Type::Logical: return "l";
            default: _error("Bad type in trace");
        }
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
        return llvm::VectorType::get(llvmMemoryType(type), width);
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

    std::string postfix(Type::Enum type1, Type::Enum type2) {
        return postfix(type1) + postfix(type2);
    }

    std::string postfix(Type::Enum type1, Type::Enum type2, Type::Enum type3) {
        return postfix(type1) + postfix(type2) + postfix(type3);
    }

    llvm::Value* Load(llvm::Value* v) {
        if(v->getType()->isPointerTy()) {
            return builder.CreateLoad(v);
        }
        else {
            return v;
        }
    }

    llvm::Value* Store(llvm::Value* v) {
        if(v->getType()->isPointerTy()) {
            return v;
        }
        else {
            llvm::Value* t = CreateEntryBlockAlloca(v->getType());
            builder.CreateStore(v, t);
            return t;
        }
    }


    struct Fusion {
        LLVMState* S;
        llvm::Function* function;
        llvm::BasicBlock* header;
        llvm::BasicBlock* condition;
        llvm::BasicBlock* body;
        
        llvm::PHINode* iterator;
        llvm::Value* length;

        size_t width;
        llvm::IRBuilder<> builder;

        Fusion(LLVMState* S, llvm::Function* function, llvm::Value* length, size_t width)
            : S(S)
            , length(length)
            , function(function)
            , width(width)
            , builder(*S->C) {
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

        void Open() {

            header = llvm::BasicBlock::Create(*S->C, "fusedHeader", function, 0);
            condition = llvm::BasicBlock::Create(*S->C, "fusedCondition", function, 0);
            body = llvm::BasicBlock::Create(*S->C, "fusedBody", function, 0);

            builder.SetInsertPoint(header);
            llvm::Value* initial = builder.getInt64(0);
            builder.CreateBr(condition);
            
            builder.SetInsertPoint(condition);
            iterator = builder.CreatePHI(builder.getInt64Ty(), 2);
            iterator->addIncoming(initial, header);

            builder.SetInsertPoint(body);
        }

        llvm::Value* Load(llvm::Value* a) {
            llvm::Type* t = llvmType(Type::Double, width)->getPointerTo();
            llvm::Type* mt = llvmType(Type::Double)->getPointerTo();
            a = builder.CreatePointerCast(a, mt);
            a = builder.CreateInBoundsGEP(a, iterator);
            a = builder.CreatePointerCast(a, t);
            
            return builder.CreateLoad(a);
            /*llvm::Value* r = llvm::UndefValue::get(llvmType(Type::Double, width));
            for(int i = 0; i < width; i++) {
                r = builder.CreateInsertElement(
                    r,
                    builder.CreateLoad(builder.CreateConstGEP1_32(a, i)),
                    builder.getInt32(i));
            }
            return r;*/
        }

        void Store(llvm::Value* a, llvm::Value* out) {
            llvm::Type* t = llvmType(Type::Double, width)->getPointerTo();
            llvm::Type* mt = llvmType(Type::Double)->getPointerTo();
            out = builder.CreatePointerCast(out, mt);
            out = builder.CreateInBoundsGEP(out, iterator);
            out = builder.CreatePointerCast(out, t);

            builder.CreateStore(a, out);
       
            /*for(int i = 0; i < width; i++) {
                builder.CreateStore(
                    builder.CreateExtractElement(a, builder.getInt32(i)),
                    builder.CreateConstGEP1_32(out, i));
            } */
        }

        void Emit(JIT::IR ir, llvm::Value* a, llvm::Value* b, llvm::Value* out) {
            // Create an output vector...
            // 
            switch(ir.op) {
                case TraceOpCode::add: {
                    Store(builder.CreateFAdd(Load(a), Load(b)), out);
                }   break;
            }
        }

        llvm::BasicBlock* Close() {
            llvm::BasicBlock* after = llvm::BasicBlock::Create(*S->C, "fusedAfter", function, 0);
            builder.SetInsertPoint(body);
            llvm::Value* increment = builder.CreateAdd(iterator, builder.getInt64(width));
            iterator->addIncoming(increment, body);
            builder.CreateBr(condition);
            
            builder.SetInsertPoint(condition);
            llvm::Value* endCond = builder.CreateICmpULT(iterator, length);
            builder.CreateCondBr(endCond, body, after);

            return after;
        }
    };

    llvm::Value* Operand(size_t index) {
        if(jit.assignment[index] != 0) {
            return Load(registers[jit.assignment[index]]);
        }
        else {
            return Load(values[index]);
        }
    }

    llvm::Value* RawOperand(size_t index) {
        if(jit.assignment[index] != 0) {
            return registers[jit.assignment[index]];
        }
        else {
            return values[index];
        }
    }



    llvm::Value* Emit(JIT::IR ir, size_t width, size_t index) {
        // need to pass in:
        //  1) load/store offset
        //  2) need to store outside of exit, but where?
        //  3) on mov??
        //  4) but need to kill overwriting movs
        //  5) they're not live out at any exit point
        llvm::Type* t = llvmType(ir.type, width);

        llvm::Value* r;
        switch(ir.op) {
            case TraceOpCode::loop:
            {
                PrePhiBlock = builder.GetInsertBlock();
                builder.CreateBr(PhiBlock);
                builder.SetInsertPoint(LoopStart);
            }   break;
            case TraceOpCode::constant:
            {
                std::vector<llvm::Constant*> c;
                if(ir.type == Type::Double) {
                    Double* v = (Double*)ir.target;
                    for(size_t i = 0; i < width; i++)
                        c.push_back(llvm::ConstantFP::get(builder.getDoubleTy(), (*v)[i]));
                } else if(ir.type == Type::Integer) {
                    Integer* v = (Integer*)ir.target;
                    for(size_t i = 0; i < width; i++)
                        c.push_back(builder.getInt64((*v)[i]));
                } else if(ir.type == Type::Logical) {
                    Logical* v = (Logical*)ir.target;
                    for(size_t i = 0; i < width; i++)
                        c.push_back(builder.getInt1((*v)[i] != 0));
                } else {
                    _error("Unexpected constant type");
                }
                r = llvm::ConstantVector::get(c);
            } break;
            case TraceOpCode::load: 
            {
                r = CALL1(std::string("Load_")+Type::toString(ir.type), builder.getInt64(ir.target));
                llvm::Type* mt = llvmMemoryType(ir.type, width)->getPointerTo();
                r = builder.CreatePointerCast(r, mt);
                
                //if(ir.type == Type::Logical)
                  //  r = builder.CreateTrunc(r, t); 
            } break;
            case TraceOpCode::phi: 
            {
                if(jit.assignment[ir.a] != jit.assignment[ir.b]) {
                    builder.CreateStore(
                        Operand(ir.b),
                        registers[jit.assignment[ir.a]]);
                }
            } break;

            case TraceOpCode::GTYPE: {
                llvm::Value* guard = 
                    CALL2("Guard_Type", builder.getInt64(ir.target), builder.getInt32(ir.type));
                EmitExit(guard, jit.exits[index]);
            } break;

            case TraceOpCode::guardT:
            case TraceOpCode::guardF: {
                if(jit.code[ir.a].width != 1) {
                    _error("Emitting guard on non-scalar");
                }
                // TODO: check the NA mask
                r = builder.CreateExtractElement(Operand(ir.a), builder.getInt32(0));
                r = builder.CreateTrunc(r, builder.getInt1Ty());
                if(ir.op == TraceOpCode::guardF)
                    r = builder.CreateNot(r);
                EmitExit(r, jit.exits[index]);
            } break;

            case TraceOpCode::add:
            {
                if(ir.width < 16)
                    r = builder.CreateFAdd(Operand(ir.a), Operand(ir.b));
                else {
                    Fusion f(S, function, builder.getInt64(width), 8);
                    f.Open();
                    f.Emit(ir, RawOperand(ir.a), RawOperand(ir.b), RawOperand(index));
                    builder.CreateBr(f.header);
                    builder.SetInsertPoint(f.Close());
                    r = 0;
                }
            } break;
            case TraceOpCode::lt:
            {
                r = builder.CreateFCmpOLT(Operand(ir.a), Operand(ir.b));
            } break;
            case TraceOpCode::gather:
            {
                llvm::Value* v = Operand(ir.a);
                llvm::Value* idx = Operand(ir.b);
                llvm::Type* trunc = llvm::VectorType::get(builder.getInt32Ty(), jit.code[ir.b].width);
                idx = builder.CreateTrunc(idx, trunc);  
                if(llvm::isa<llvm::Constant>(idx)) {
                    r = builder.CreateShuffleVector(v, llvm::UndefValue::get(v->getType()), idx);
                } else {
                    // scalarize the gather...
                    r = llvm::UndefValue::get(t);

                    for(size_t i = 0; i < width; i++) {
                        llvm::Value* ii = builder.getInt32((uint32_t)i);
                        llvm::Value* j = builder.CreateExtractElement(idx, ii);
                        j = builder.CreateExtractElement(v, j);
                        r = builder.CreateInsertElement(r, j, ii);
                    }
                }
            } break;
            case TraceOpCode::scatter:
            {
                llvm::Value* v = Operand(ir.a);
                llvm::Value* idx = Operand(ir.b);
                r = Operand(ir.c);
                llvm::Type* trunc = llvm::VectorType::get(builder.getInt32Ty(), jit.code[ir.b].width);
                idx = builder.CreateTrunc(idx, trunc);  
                // constant version could be a shuffle. No idea if that will generate better code.
                for(size_t i = 0; i < width; i++) {
                        llvm::Value* ii = builder.getInt32((uint32_t)i);
                        llvm::Value* j = builder.CreateExtractElement(v, ii);
                        ii = builder.CreateExtractElement(idx, ii);
                        r = builder.CreateInsertElement(r, j, ii);
                } 
            } break;
            case TraceOpCode::cast:
            {
                r = llvm::UndefValue::get(t);
                switch(ir.type) {
                    case Type::Double:
                    break;
                    case Type::Integer: {
                        switch(jit.code[ir.a].type) {
                            case Type::Double: {
                                for(size_t i = 0; i < width; i++) {
                                    llvm::Value* ii = builder.getInt32((uint32_t)i);
                                    llvm::Value* j = builder.CreateExtractElement(Operand(ir.a), ii);
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
            case TraceOpCode::store:
            case TraceOpCode::nop:
            {
                // do nothing
            } break;
            case TraceOpCode::jmp:
            {
                // close loop.
            } break;
            default: 
            {
                _error("Unknown op");
            } break;
        };
        if(jit.assignment[index] != 0 && r != 0) {
            builder.CreateStore(r, registers[jit.assignment[index]]);
            return registers[jit.assignment[index]];
        }
        else {
            return r;
        }
    }

    void EmitExit(llvm::Value* cond, JIT::Exit const& e) {
        llvm::BasicBlock* next = llvm::BasicBlock::Create(*S->C, "next", function, 0);
        llvm::BasicBlock* exit = llvm::BasicBlock::Create(*S->C, "exit", function, 0);
        builder.CreateCondBr(cond, next, exit);
        builder.SetInsertPoint(exit);
        
        std::map<int64_t, JIT::IRRef>::const_iterator i;
        for(i = e.o.begin(); i != e.o.end(); i++) {
            JIT::IR& ir = jit.code[i->second];
    
            llvm::Type* mt = llvmMemoryType(ir.type, ir.width);
            
            llvm::Value* r = Operand(i->second);
            if(ir.type == Type::Logical)
                r = builder.CreateSExt(r, mt);

            mt = llvmMemoryType(ir.type)->getPointerTo();

            if(i->first <= 0)
                CALL3(std::string("storer_")+postfix(ir.type),
                        builder.getInt64(i->first), 
                        builder.getInt64(ir.width), 
                        builder.CreatePointerCast(
                            Store(r),mt));
            else
                CALL3(std::string("storem_")+postfix(ir.type),
                        builder.getInt64(i->first), 
                        builder.getInt64(ir.width), 
                        builder.CreatePointerCast(
                            Store(r),mt));
        }
        builder.CreateStore(builder.CreateIntToPtr(builder.getInt64((int64_t)e.reenter), instruction_type), result_var);
        
        builder.CreateBr(EndBlock);
        builder.SetInsertPoint(next); 
    }

    void Execute() {
    }
};

JIT::Ptr JIT::compile(Thread& thread) {
    timespec a = get_time();
    LLVMCompiler compiler(thread, *this);
    Ptr result = (Ptr)compiler.Compile();
    printf("Compile time: %f\n", time_elapsed(a));
    return result;
}

