
#include "jit.h"
#include "bc.h"
#include "interpreter.h"
#include "call.h"

DEFINE_ENUM_TO_STRING(TraceOpCode, TRACE_ENUM)

JIT::IRRef JIT::insert(TraceOpCode::Enum op, int64_t i, Type::Enum type, size_t width) {
    IR ir = (IR) { op, 0, 0, 0, i, type, width, 0, false }; 
    code.push_back(ir);
    return (IRRef) { code.size()-1 };
}

JIT::IRRef JIT::insert(TraceOpCode::Enum op, IRRef a, int64_t i, Type::Enum type, size_t width) {
    IR ir = (IR) { op, a, 0, 0, i, type, width, 0, false }; 
    code.push_back(ir);
    return (IRRef) { code.size()-1 };
}

JIT::IRRef JIT::insert(TraceOpCode::Enum op, IRRef a, Type::Enum type, size_t width) {
    IR ir = (IR) { op, a, 0, 0, 0, type, width, 0, false };
    code.push_back(ir);
    return (IRRef) { code.size()-1 };
}

JIT::IRRef JIT::insert(TraceOpCode::Enum op, IRRef a, IRRef b, Type::Enum type, size_t width) {
    IR ir = (IR) { op, a, b, 0, 0, type, width, 0, false };
    code.push_back(ir);
    return (IRRef) { code.size()-1 };
}

JIT::IRRef JIT::insert(TraceOpCode::Enum op, IRRef a, IRRef b, IRRef c, Type::Enum type, size_t width) {
    IR ir = (IR) { op, a, b, c, 0, type, width, 0, false };
    code.push_back(ir);
    return (IRRef) { code.size()-1 };
}

JIT::IRRef JIT::load(Thread& thread, int64_t a, Instruction const* reenter) {
    
    std::map<int64_t, IRRef>::const_iterator i;
    i = map.find(a);
    if(i != map.end()) {
        return i->second;
    }
    else {
        OPERAND(operand, a);
        if(a <= 0 && -a < thread.frame.prototype->constants.size()) {
            return insert(TraceOpCode::constant, (int64_t)&operand, operand.type, operand.length);
        }
        else {
            Exit e = { map, reenter };
            exits[code.size()] = e;
            map[a] = (IRRef){code.size()};
            return insert(TraceOpCode::load, a, operand.type, operand.length);
        }
    }
}

JIT::IRRef JIT::emit(Thread& thread, TraceOpCode::Enum op, IRRef a, IRRef b, int64_t c) {
    Value const& v = OUT(thread, c);
    printf("Looking up value at %d: %s\n", c, Type::toString(v.type));
    map[c] = (IRRef) {code.size()};
    return insert(op, a, b, v.type, v.length);
}

JIT::IRRef JIT::emit(Thread& thread, TraceOpCode::Enum op, IRRef a, IRRef b, IRRef c, int64_t d) {
    Value const& v = OUT(thread, d);
    map[d] = (IRRef) {code.size()};
    return insert(op, a, b, c, v.type, v.length);
}

JIT::IRRef JIT::store(Thread& thread, IRRef a, int64_t c) {
    map[c] = a;
    return a;
}

void JIT::guardF(Thread& thread, Instruction const* reenter) {
    IRRef p = (IRRef) { code.size()-1 }; 
    Exit e = { map, reenter };
    exits[code.size()] = e;
    markLiveOut(e);
    insert(TraceOpCode::guardF, p, Type::Promise, code[p].width );
}

void JIT::guardT(Thread& thread, Instruction const* reenter) {
    IRRef p = (IRRef) { code.size()-1 }; 
    Exit e = { map, reenter };
    exits[code.size()] = e;
    markLiveOut(e);
    insert(TraceOpCode::guardT, p, Type::Promise, code[p].width );
}

void JIT::EmitIR(Thread& thread, Instruction const& inst, bool branch) {
    switch(inst.bc) {

        case ByteCode::jc: {
            if(branch) 
                guardT(thread, &inst+inst.b);
            else
                guardF(thread, &inst+inst.a);
        }   break;

        case ByteCode::mov:
        case ByteCode::fastmov: {
            load(thread, inst.a, &inst);
        }   break;

        case ByteCode::assign: {
            IRRef c = load(thread, inst.c, &inst);
            if(map.find(inst.a) != map.end()) {
                phi[map.find(inst.a)->second] = c;
            }
            store(thread, c, inst.a);
        }   break;

        #define BINARY_EMIT(Name, ...)                      \
        case ByteCode::Name: {                              \
            IRRef a = load(thread, inst.a, &inst);          \
            IRRef b = load(thread, inst.b, &inst);          \
            emit(thread, TraceOpCode::Name, a, b, inst.c);  \
        }   break;
        BINARY_BYTECODES(BINARY_EMIT)
        #undef BINARY_EMIT

        default: {
            _error("Not supported in emit ir");
        }   break;
    }
}

void JIT::Replay(Thread& thread) {
    // emit phis
    //  phis are mapped values whose entry has changed from loop begin
    //  to loop end.
    // TODO: check that all PHI nodes match on type...if not, fail
    for(std::map<IRRef, IRRef>::const_iterator i = phi.begin(); i != phi.end(); ++i) {
        insert(TraceOpCode::phi, i->first, i->second, code[i->first].type, code[i->first].width);
    }
}

void JIT::markLiveOut(Exit const& exit) {
    std::map<int64_t, JIT::IRRef>::const_iterator i;
    for(i = exit.o.begin(); i != exit.o.end(); i++) {
        code[i->second].liveout = true;
    }
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

    assert(state == RECORDING_BODY);
    state = OFF;

    /*loopStart = pc;
    size_t n = pc.i;
    size_t forward[1024];

    for(size_t i = 0; i < n; i++) {
        IR& ir = code[i];
        IRRef a, b, c;
        switch(ir.op) {
            case TraceOpCode::constant: {
                insert(TraceOpCode::nop, a, b, code[i].type, code[i].width);
                forward[i] = i;
            } break;
            case TraceOpCode::load: {
            } break;
                if(i == map[ir.i].i) {
                    insert(TraceOpCode::nop, a, b, code[i].type, code[i].width);
                    forward[i] = i;
                } else {
                    a.i = map[ir.i].i; b.i = n+map[ir.i].i;
                    Exit e = { map, exits[i].reenter };
                    exits[pc.i] = e;
                    markLiveOut(e);
                    map[ir.i] = pc;
                    insert(TraceOpCode::phi, a, b, code[i].type, code[i].width);
                    code[pc.i-1].c.i = i;
                    forward[i] = n+i;
                }
            } break;
            case TraceOpCode::phi:
            case TraceOpCode::jmp: {
                _error("Invalid node");
            } break;
            case TraceOpCode::guardF: 
            case TraceOpCode::guardT: {
                if(forward[ir.a.i] == ir.a.i) {
                    insert(TraceOpCode::nop, a, b, code[i].type, code[i].width);
                    forward[i] = i;
                } else {
                    a.i = n+ir.a.i;
                    Exit e = { map, exits[i].reenter };
                    exits[pc.i] = e;
                    markLiveOut(e);
                    insert(ir.op, a, code[i].type, code[i].width);
                    forward[i] = n+i;
                }
            } break;
            case TraceOpCode::store2: {
                _error("No store2");
                //a.i = n+ir.a.i; b.i = n+ir.b.i; c.i = n+ir.c.i;
                //insert(ir.op, a, b, c, code[i].type, code[i].width);
            } break;
            default: {
                if(forward[ir.a.i] == ir.a.i &&
                    forward[ir.b.i] == ir.b.i) {
                    insert(TraceOpCode::nop, a, b, code[i].type, code[i].width);
                    forward[i] = i;
                } else {
                    printf("Forwarding %d to %d\n", ir.a.i, forward[ir.a.i]);
                    a.i = forward[ir.a.i]; b.i = forward[ir.b.i];
                    insert(ir.op, a, b, code[i].type, code[i].width);
                    forward[i] = n+i;
                }
            } break;
        }
    }
    insert(TraceOpCode::jmp, loopStart, Type::Promise, 1);
     
    // check that all PHI nodes match on type...if not, fail
    for(size_t i = 0; i < pc.i; i++) {
        IR& ir = code[i];
        if(ir.op == TraceOpCode::phi) {
            if(code[ir.a.i].type != code[ir.b.i].type)
                return (JIT::Ptr)0;
        }
    }
    */

    Replay(thread);

    schedule();
    dump();
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

void JIT::schedule() {
    // do a backwards pass, assigning instructions to a fusion group.
    // this happens after all optimization and specialization decisions
    //  have been made.
    /*size_t group = 0;
    for(size_t i = pc.i-1; i < pc.i; --i) {
        IR& ir = code[i];
        switch(ir.op) {
            case TraceOpCode::constant: {
                ir.group = group;
            } break;
            case TraceOpCode::nop: {
                ir.group = group;
            } break;
            case TraceOpCode::phi: {
                group++;
                ir.group = group;
                group++;
                code[ir.a.i].group = group;
            } break;
            case TraceOpCode::jmp: {
            } break;
            case TraceOpCode::load:
            case TraceOpCode::guardF: 
            case TraceOpCode::guardT: {
                // Do I also need to update any values that
                // are live out at this exit? Yes.
                group++;
                ir.group = group;
                group++;
                code[ir.a.i].group = group;
                std::map<int64_t, IRRef>::const_iterator j;
                for(j = exits[i].o.begin(); j != exits[i].o.end(); ++j) {
                    code[j->second.i].group = group;
                }
            } break;
            case TraceOpCode::store2: {
                _error("No store2");
            } break;
            default: {
                code[ir.a.i].group = 
                    std::max(code[ir.a.i].group,
                        ir.group);
                code[ir.b.i].group = 
                    std::max(code[ir.b.i].group,
                        ir.group);
            } break;
        }
    }
*/
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
        case TraceOpCode::load: {
            if(i <= 0)
                std::cout << "\t" << i;
            else
                std::cout << "\t " << (String)i;
        } break;
        case TraceOpCode::phi: {
            std::cout << "\t " << a << "\t " << b;
        } break;
        case TraceOpCode::guardF:
        case TraceOpCode::guardT: {
            std::cout << "\t " << a;
        } break;
        case TraceOpCode::store2: {
            std::cout << "\t " << a << "\t " << b << "\t " << c;
        } break;
        case TraceOpCode::add:
        case TraceOpCode::lt: {
            std::cout << "\t " << a << "\t " << b;
        } break;
        default: {} break;
    };

    if(liveout)
        std::cout << "\t=>";
}

void JIT::dump() {
    for(size_t i = 0; i < code.size(); i++) {
        printf("%4d: ", i);
        code[i].dump();
        std::cout << std::endl;
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
    llvm::BasicBlock * LoopStart;
    llvm::BasicBlock * EndBlock;
    llvm::IRBuilder<> builder;

    llvm::Type* thread_type;
    llvm::Type* instruction_type;

    llvm::Value* thread_var;
    llvm::Value* result_var;

    llvm::Value* values[1024];

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
        LoopStart = llvm::BasicBlock::Create(
                *S->C, "loop", function, 0);
        EndBlock = llvm::BasicBlock::Create(
                *S->C, "end", function, 0);

        llvm::Function::arg_iterator ai = function->arg_begin();
        ai->setName("thread");
        thread_var = ai++;

        result_var = CreateEntryBlockAlloca(instruction_type);


        builder.SetInsertPoint(EntryBlock);

        // split up code by execution order and size.
        // emit each into own block
        // 
        for(size_t i = 0; i < jit.code.size(); i++) {
            values[i] = Emit(jit.code[i], jit.code[i].width, i);
        }
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
    
        function->dump();
        S->FPM->run(*function);

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

    llvm::Type* llvmScalarType(Type::Enum type) {
        llvm::Type* t;
        switch(type) {
            case Type::Double: t = builder.getDoubleTy(); break;
            case Type::Integer: t = builder.getInt64Ty(); break;
            case Type::Logical: t = builder.getInt8Ty(); break;
            case Type::Promise: t = builder.getInt1Ty(); break;
            default: _error("Bad type in trace");
        }
        return t;
    }

    llvm::Type* llvmType(Type::Enum type, size_t width) {
        return llvm::VectorType::get(llvmScalarType(type), width);
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
                builder.CreateBr(LoopStart);
                builder.SetInsertPoint(LoopStart);
            }   break;
            case TraceOpCode::constant:
            {
                std::vector<llvm::Constant*> c;
                if(ir.type == Type::Double) {
                    Double* v = (Double*)ir.i;
                    for(size_t i = 0; i < width; i++)
                        c.push_back(llvm::ConstantFP::get(builder.getDoubleTy(), (*v)[i]));
                } else if(ir.type == Type::Integer) {
                    Integer* v = (Integer*)ir.i;
                    for(size_t i = 0; i < width; i++)
                        c.push_back(builder.getInt64((*v)[i]));
                } else if(ir.type == Type::Logical) {
                    Logical* v = (Logical*)ir.i;
                    for(size_t i = 0; i < width; i++)
                        c.push_back(builder.getInt8((*v)[i]));
                } else {
                    _error("Unexpected constant type");
                }
                r = llvm::ConstantVector::get(c);
            } break;
            case TraceOpCode::load: 
            {
                llvm::Type* pt = llvmScalarType(ir.type)->getPointerTo();
                r = CreateEntryBlockAlloca(t);
                llvm::Value* guard;
                if(ir.i <= 0) {
                  guard = CALL3(std::string("loadr_")+postfix(ir.type),
                    builder.getInt64(ir.i),
                    builder.getInt64(width),
                    builder.CreatePointerCast(r,pt));
                }
                else {
                  guard = CALL3(std::string("loadm_")+postfix(ir.type),
                    builder.getInt64(ir.i),
                    builder.getInt64(width),
                    builder.CreatePointerCast(r,pt));
                }
                EmitExit(guard, jit.exits[index]);
            } break;
            case TraceOpCode::phi: 
            {
                r = CreateEntryBlockAlloca(Load(values[ir.a])->getType());
                builder.CreateStore(Load(values[ir.b]), r);
                llvm::IRBuilder<> TmpB(&function->getEntryBlock(),
                    function->getEntryBlock().end());
                TmpB.CreateStore(Load(values[ir.a]), r);
            } break;

            case TraceOpCode::guardT:
            case TraceOpCode::guardF: {
                if(jit.code[ir.a].width != 1) {
                    _error("Emitting guard on non-scalar");
                }
                // TODO: check the NA mask
                r = builder.CreateExtractElement(Load(values[ir.a]), builder.getInt32(0));
                r = builder.CreateTrunc(r, builder.getInt1Ty());
                if(ir.op == TraceOpCode::guardF)
                    r = builder.CreateNot(r);
                EmitExit(r, jit.exits[index]);
            } break;

            case TraceOpCode::store2: 
            {
                r = CALL3(std::string(TraceOpCode::toString(ir.op))+"_"+postfix(jit.code[ir.a].type, jit.code[ir.b].type, jit.code[ir.c].type), Load(values[ir.a]), Load(values[ir.b]), Load(values[ir.c])); 
            } break;
            case TraceOpCode::add:
            {
                r = builder.CreateFAdd(Load(values[ir.a]), Load(values[ir.b]));
            } break;
            case TraceOpCode::lt:
            {
                r = builder.CreateFCmpOLT(Load(values[ir.a]), Load(values[ir.b]));
                r = builder.CreateSExt(r, t);
            } break;
            case TraceOpCode::gather:
            {
                llvm::Value* v = Load(values[ir.a]);
                llvm::Value* idx = Load(values[ir.b]);
                if(llvm::isa<llvm::Constant>(idx)) {
                    llvm::Type* t = llvm::VectorType::get(builder.getInt32Ty(), jit.code[ir.b].width);
                    idx = builder.CreateTrunc(idx, t);  
                    r = builder.CreateShuffleVector(v, llvm::UndefValue::get(v->getType()), idx);
                } else {
                    _error("No non-const gather");
                }
            } break;
            case TraceOpCode::nop:
            {
                // do nothing
            } break;
            default: 
            {
                r = CALL2(std::string(TraceOpCode::toString(ir.op))+"_"+postfix(jit.code[ir.a].type, jit.code[ir.b].type), Load(values[ir.a]), Load(values[ir.b])); 
            } break;
        };
        return r;
    }

    void EmitExit(llvm::Value* cond, JIT::Exit const& e) {
        llvm::BasicBlock* next = llvm::BasicBlock::Create(*S->C, "next", function, 0);
        llvm::BasicBlock* exit = llvm::BasicBlock::Create(*S->C, "exit", function, 0);
        builder.CreateCondBr(cond, next, exit);
        builder.SetInsertPoint(exit);
        
        std::map<int64_t, JIT::IRRef>::const_iterator i;
        for(i = e.o.begin(); i != e.o.end(); i++) {
            JIT::IR& ir = jit.code[i->second];
            llvm::Type* t = llvmScalarType(ir.type)->getPointerTo();
            if(i->first <= 0)
                CALL3(std::string("storer_")+postfix(ir.type),
                        builder.getInt64(i->first), 
                        builder.getInt64(ir.width), 
                        builder.CreatePointerCast(
                            Store(values[i->second]),t));
            else
                CALL3(std::string("storem_")+postfix(ir.type),
                        builder.getInt64(i->first), 
                        builder.getInt64(ir.width), 
                        builder.CreatePointerCast(
                            Store(values[i->second]),t));
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

