
#include "interpreter.h"

#ifdef USE_LLVM_COMPILER

#include "llvm/Analysis/Passes.h"
#include "llvm/Analysis/Verifier.h"
#include "llvm/DerivedTypes.h"
#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/ExecutionEngine/JIT.h"
#include "llvm/LLVMContext.h"
#include "llvm/Linker.h"
#include "llvm/Module.h"
#include "llvm/PassManager.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/IRBuilder.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/Program.h"


struct LLVMState {
    llvm::Module * M;
    llvm::LLVMContext * C;
    llvm::ExecutionEngine * EE;
    llvm::FunctionPassManager * FPM;
};


void TraceLLVMCompiler_init(State & state) {
    LLVMState * L = state.llvmState = new (GC) LLVMState;
    llvm::InitializeNativeTarget();
    
    L->C = &llvm::getGlobalContext();
    L->M = new llvm::Module("riposte",*L->C);
    std::string err;
    L->EE = llvm::EngineBuilder(L->M).setErrorStr(&err).setEngineKind(llvm::EngineKind::JIT).create();
    if (!L->EE) {
        _error(err);
    }
    
    L->FPM = new llvm::FunctionPassManager(L->M);
    
    //TODO: add optimization passes here, these are just from llvm tutorial and are probably not good
    //look here: http://lists.cs.uiuc.edu/pipermail/llvmdev/2011-December/045867.html
    L->FPM->add(new llvm::TargetData(*L->EE->getTargetData()));
    // Provide basic AliasAnalysis support for GVN.
    L->FPM->add(llvm::createBasicAliasAnalysisPass());
    // Promote allocas to registers.
    L->FPM->add(llvm::createPromoteMemoryToRegisterPass());
    // Also promote aggregates like structs....
    L->FPM->add(llvm::createScalarReplAggregatesPass());
    // Do simple "peephole" optimizations and bit-twiddling optzns.
    L->FPM->add(llvm::createInstructionCombiningPass());
    // Reassociate expressions.
    L->FPM->add(llvm::createReassociatePass());
    // Eliminate Common SubExpressions.
    L->FPM->add(llvm::createGVNPass());
    // Simplify the control flow graph (deleting unreachable blocks, etc).
    L->FPM->add(llvm::createCFGSimplificationPass());
    
    L->FPM->doInitialization();
}

struct TraceLLVMCompiler {
    LLVMState * L;
    Thread * thread;
    Trace * trace;
    llvm::Function * function;
    llvm::BasicBlock * entry;
    llvm::Value * loopIndexAddr;
    llvm::Type * doubleType;
    llvm::Type * intType;
    llvm::IRBuilder<> * B;
    std::vector<llvm::Value *> values;
    
    TraceLLVMCompiler(Thread * th, Trace * tr) 
    : L(th->state.llvmState), thread(th), trace(tr), values(tr->nodes.size(),NULL) {}
    
    void Compile() {
        intType = llvm::Type::getInt64Ty(*L->C);
        doubleType = llvm::Type::getDoubleTy(*L->C);
        
        llvm::FunctionType * ftype = llvm::FunctionType::get(llvm::Type::getVoidTy(*L->C),false);
        
        function = llvm::Function::Create(ftype, llvm::Function::ExternalLinkage,"",L->M);
        
        entry = llvm::BasicBlock::Create(*L->C,"entry",function);
        B = new llvm::IRBuilder<>(*L->C);
        B->SetInsertPoint(entry);
        
        
        //create the enclosing loop:
        /*
        for(i = 0; i < trace->Size; i++) {
           <CompileLoopBody>
        }
        */
        
        llvm::Constant * Size = ConstantInt(trace->Size);
        loopIndexAddr = B->CreateAlloca(intType);
        B->CreateStore(ConstantInt(0LL),loopIndexAddr);
        
        
        llvm::BasicBlock * cond = createAndInsertBB("cond");
        llvm::BasicBlock * body = createAndInsertBB("body");
        llvm::BasicBlock * end = createAndInsertBB("end");
        
        B->CreateBr(cond);
        B->SetInsertPoint(cond);
        llvm::Value * c = B->CreateICmpULT(loopIndex(), Size);
        B->CreateCondBr(c,body,end);
        
        B->SetInsertPoint(body);
        CompileLoopBody();
        B->CreateStore(B->CreateAdd(loopIndex(), ConstantInt(1LL)),loopIndexAddr);
        B->CreateBr(cond);
        
        B->SetInsertPoint(end);
        B->CreateRetVoid();
        
        L->M->dump();
        llvm::verifyFunction(*function);
        
        L->FPM->run(*function);
        L->M->dump();
        
        delete B;
    }
    llvm::Constant * ConstantInt(int64_t i) {
        return llvm::ConstantInt::get(intType,i);
    }
    llvm::Constant * ConstantDouble(double d) {
        return llvm::ConstantFP::get(doubleType, d);
    }
    llvm::Value * ConstantPointer(void * ptr, llvm::Type * typ) {
        llvm::Constant* ci = llvm::ConstantInt::get(intType, (uint64_t)ptr); 
        llvm::Value* cp = llvm::ConstantExpr::getIntToPtr(ci, llvm::PointerType::getUnqual(typ));
        return cp; 
    }
    void CompileLoopBody() {
        llvm::Value * loopIndexValue = loopIndex();
        
        std::vector<llvm::Value *> loopIndexArray;
        loopIndexArray.push_back(loopIndexValue);
                    
        for(size_t i = 0; i < trace->nodes.size(); i++) {
            IRNode & n = trace->nodes[i];
            switch(n.op) {
                case IROpCode::load: {
                    void * p;
                    if(n.in.isLogical())
                        p = ((Logical&)n.in).v();
                    else if(n.in.isInteger())
                        p = ((Integer&)n.in).v();
                    else if(n.in.isDouble())
                        p = ((Double&)n.in).v();
                    else
                        _error("unsupported type");
                    llvm::Type * t = getType(n.type);
                    
                    
                    llvm::Value * vector = ConstantPointer(p, t);
                    llvm::Value * elementAddr = B->CreateGEP(vector, loopIndexArray);
                     
                    values[i] = B->CreateLoad(elementAddr);
                } break;
                case IROpCode::add:
                    switch(n.type) {
                        case Type::Double:
                            values[i] = B->CreateFAdd(values[n.binary.a],values[n.binary.b]);
                            break;
                        case Type::Integer:
                            values[i] =  B->CreateAdd(values[n.binary.a],values[n.binary.b]);
                            break;
                        default:
                            _error("unsupported type");
                    }
                    break;
                case IROpCode::sub:
                    switch(n.type) {
                        case Type::Double:
                            values[i] = B->CreateFSub(values[n.binary.a],values[n.binary.b]);
                            break;
                        case Type::Integer:
                            values[i] =  B->CreateSub(values[n.binary.a],values[n.binary.b]);
                            break;
                        default:
                            _error("unsupported type");
                    }
                    break;
                case IROpCode::constant:
                    switch(n.type) {
                        case Type::Double:
                            values[i] = ConstantDouble(n.constant.d);
                            break;
                        case Type::Integer:
                            values[i] = ConstantInt(n.constant.i);
                            break;
                        default:
                            _error("unsupported type");
                            break;
                    }
                    break;
                default:
                    _error("unsupported op");
                    break;
            }
            
            if(n.liveOut) {
				int64_t length = n.outShape.length;
				void * p;
                if(n.type == Type::Double) {
                    n.out = Double(length);
                    p = ((Double&)n.out).v();
                } else if(n.type == Type::Integer) {
					n.out = Integer(length);
                    p = ((Integer&)n.out).v();
                } else if(n.type == Type::Logical) {
					n.out = Logical(length);
                    p = ((Logical&)n.out).v();
				} else {
					_error("Unknown type in initialize outputs");
				}
                llvm::Type * t = getType(n.type);
                llvm::Value * vector = ConstantPointer(p,t);
                llvm::Value * elementAddr = B->CreateGEP(vector, loopIndexArray); 
                B->CreateStore(values[i], elementAddr);
            }
        }
    
    }
    
    void Execute() {
        void (*fptr)(void) = (void(*)(void)) L->EE->getPointerToFunction(function);
        fptr();
        
    }
    llvm::BasicBlock * createAndInsertBB(const char * name) {
        llvm::BasicBlock * bb = llvm::BasicBlock::Create(*L->C, name);
        function->getBasicBlockList().push_back(bb);
        return bb;
    }
    llvm::Value * loopIndex() {
        return B->CreateLoad(loopIndexAddr);
    }
    llvm::Type * getType(Type::Enum ty) {
        switch(ty) {
            case Type::Double:
                return doubleType;
            case Type::Integer:
                return intType;
            default:
                _error("type not understood");
                return NULL;
        }
    }
    
};

void Trace::JIT(Thread & thread) {
    std::cout << toString(thread) << "\n";
    TraceLLVMCompiler c(&thread,this);
    c.Compile();
    c.Execute();
}

#endif
