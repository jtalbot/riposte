
#include "interpreter.h"
#include "nvvm.h"

#include <cassert>
#include <climits>
#include <vector>
#include <map>
#include <string>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <builtin_types.h>


#ifdef USE_LLVM_COMPILER

#if defined(_MSC_VER)
// llvm/Instructions.h runs into this
#pragma warning (disable : 4355)
#endif

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
#include "llvm/InlineAsm.h"
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
#include "llvm/Value.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/ValueHandle.h"
#include "llvm/Module.h"
#include "llvm/LLVMContext.h"
#include "llvm/GlobalVariable.h"

#define checkCudaErrors(Err)  checkCudaErrors_internal (Err, __FILE__, __LINE__)
//#define USE_TEXT_NVVM_INTERFACE 1 

#define __NVVM_SAFE_CALL(X) do { \
nvvmResult ResCode = (X); \
if (ResCode != NVVM_SUCCESS) { \
    std::cout << "NVVM call (" << #X << ") failed. Error Code : " << ResCode << std::endl;\
} \
} while (0)

struct LLVMState {
    llvm::Module * M;
    llvm::LLVMContext * C;
    llvm::ExecutionEngine * EE;
    llvm::FunctionPassManager * FPM;
};
//inside pieces of the compiler


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
    
    int numBlock;
    int numThreads;
    int outputReductionSize;
    
    bool reduction;
    bool reductionBlocksPresent;
    
    llvm::Constant * Size;
    llvm::Function * function;
    llvm::Function * Sync;
    llvm::Function *KernelFunc;
    llvm::BasicBlock * entry;
    llvm::Value * loopIndexAddr;
    llvm::Instruction *firstEntry;
    
    llvm::Value * nThreads;
    llvm::Type * doubleType;
    llvm::Type * intType;
	llvm::Type * logicalType1;
	llvm::Type * logicalType8;
    llvm::IRBuilder<> * B;
    llvm::Module * mainModule;
    
    llvm::LLVMContext * C;
    llvm::ExecutionEngine * EE;
    llvm::FunctionPassManager * FPM;
    
    llvm::BasicBlock * cond;
    llvm::BasicBlock * body;
    llvm::BasicBlock * end;
    llvm::BasicBlock * returnBlock;
    
    llvm::BasicBlock * condBlocks[4];
    
    llvm::BasicBlock * bodyBlocks[4];
    
    llvm::BasicBlock * endBlocks[4];
    

    llvm::BasicBlock * condFinal;
    llvm::BasicBlock * bodyFinal;
    llvm::BasicBlock * endFinal;    
    
    llvm::BasicBlock * origEnd; 
    
    
    std::vector<llvm::Value *> values;
    std::vector<void *> outputGPU;
    
    std::vector<void *> inputGPU;
    std::vector<llvm::Value *> inputGPUAddr;
    
    std::vector<void *> thingsToFree;
    
    TraceLLVMCompiler(Thread * th, Trace * tr) 
    : L(th->state.llvmState), thread(th), trace(tr), values(tr->nodes.size(),NULL) {

        llvm::InitializeNativeTarget();
        C = &llvm::getGlobalContext();
        mainModule = new llvm::Module("riposte",*C);
        
        
        numBlock = 24;
        numThreads = 512;
        
        
        std::string lTargDescrStr;
        if (sizeof(void *) == 8) { // pointer size, alignment == 8
            lTargDescrStr= "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-"
            "i64:64:64-f32:32:32-f64:64:64-v16:16:16-"
            "v32:32:32-v64:64:64-v128:128:128-n16:32:64";
        } else {
            lTargDescrStr= "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-"
            "i64:64:64-f32:32:32-f64:64:64-v16:16:16-"
            "v32:32:32-v64:64:64-v128:128:128-n16:32:64";
        }
        
        mainModule->setDataLayout(lTargDescrStr);
        
        
        std::string err;
        EE = llvm::EngineBuilder(mainModule).setErrorStr(&err).setEngineKind(llvm::EngineKind::JIT).create();
        if (!EE) {
            _error(err);
        }
        
        FPM = new llvm::FunctionPassManager(mainModule);
        
        //TODO: add optimization passes here, these are just from llvm tutorial and are probably not good
        //look here: http://lists.cs.uiuc.edu/pipermail/llvmdev/2011-December/045867.html
        FPM->add(new llvm::TargetData(*L->EE->getTargetData()));
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
    
    


    
    void GenerateIndexFunction() {
        intType = llvm::Type::getInt64Ty(*C);
        doubleType = llvm::Type::getDoubleTy(*C);
		logicalType1 = llvm::Type::getInt1Ty(*C);
		logicalType8 = llvm::Type::getInt8Ty(*C);
        
        llvm::Constant *cons = mainModule->getOrInsertFunction("indexFunc", llvm::Type::getVoidTy(*C), intType, intType, intType, NULL);
        function = llvm::cast<llvm::Function>(cons);
        function->setCallingConv(llvm::CallingConv::C);
        
        llvm::Function::arg_iterator args = function->arg_begin();
        llvm::Value* index = args++;
        llvm::Value* tid = args++;
        tid->setName("ThreadID");
        llvm::Value* blockID = args++;
        blockID->setName("blockID");
        
        int sizeOfArray = 64;
        if (numThreads > 32)
            sizeOfArray = numThreads;
        
        
        
        reduction = false;
        reductionBlocksPresent = false;
        
        entry = llvm::BasicBlock::Create(*C,"entry",function);
        B = new llvm::IRBuilder<>(*C);
        B->SetInsertPoint(entry);
        
        Sync = llvm::Function::Create(llvm::FunctionType::get(llvm::Type::getVoidTy(llvm::getGlobalContext()), false),
                                      llvm::Function::ExternalLinkage,
                                      "llvm.nvvm.barrier0", mainModule);
        Size = ConstantInt(trace->Size);
        int vectorLength = trace->Size;
        int numOutput = vectorLength/numThreads;
        
        if (numOutput < numBlock) {
            if (vectorLength%numThreads == 0)
                outputReductionSize = numOutput;
            else
                outputReductionSize = numOutput + 1;
        }
        else {
            outputReductionSize = numBlock;
        }
        loopIndexAddr = B->CreateAlloca(intType);
        B->CreateStore(index, loopIndexAddr);
        
        /*
         * We need to loop this.
         * Loop it so that a index touches all that it needs to touch, refer to RG code how to loop.
         */
        
        cond = createAndInsertBB("cond");
        body = createAndInsertBB("body");
        end = createAndInsertBB("end");
        origEnd = end;
        
        
        B->CreateBr(cond);
        B->SetInsertPoint(cond);
        llvm::Value * c = B->CreateICmpULT(loopIndex(), Size);
        B->CreateCondBr(c,body,end);
        
        B->SetInsertPoint(body);
        CompileBody(blockID, tid, sizeOfArray);
        B->CreateStore(B->CreateAdd(loopIndex(), ConstantInt(numBlock * numThreads)),loopIndexAddr);
        B->CreateBr(cond);
        
        B->SetInsertPoint(end);
        B->CreateRetVoid();
                
        if (reduction == true) {
            ReductionHelper(tid);
        }
        
        llvm::verifyFunction(*function);
        

        
        if (thread->state.verbose)
           mainModule->dump();
        
        
        
        
        FPM->run(*function);
    }
    
    void ReductionHelper(llvm::Value * tid) {
        B->SetInsertPoint(origEnd);
        B->CreateBr(condBlocks[0]);
        int current = 512;
        for (int i = 0; i < 3; i++) {
            if (numThreads >= current) {
                B->SetInsertPoint(condBlocks[i]);
                llvm::Value *Condition = B->CreateICmpSLT(tid, ConstantInt(current/2));
                B->CreateCondBr(Condition, bodyBlocks[i], endBlocks[i]);
                B->SetInsertPoint(bodyBlocks[i]);
                B->CreateBr(endBlocks[i]);
                B->SetInsertPoint(endBlocks[i]);
                B->CreateBr(condBlocks[i+1]);
            }
            else {
                B->SetInsertPoint(condBlocks[i]);
                B->CreateBr(bodyBlocks[i]);
                B->SetInsertPoint(bodyBlocks[i]);
                B->CreateBr(endBlocks[i]);
                B->SetInsertPoint(endBlocks[i]);
                B->CreateBr(condBlocks[i+1]);
            }
            current /= 2;
        }
        B->SetInsertPoint(condBlocks[3]);
        llvm::Value *Cond32 = B->CreateICmpSLT(tid, ConstantInt(32));
        B->CreateCondBr(Cond32, bodyBlocks[3], endBlocks[3]);
        B->SetInsertPoint(bodyBlocks[3]);
        B->CreateBr(endBlocks[3]);
        B->SetInsertPoint(endBlocks[3]);
        B->CreateBr(condFinal);
        B->SetInsertPoint(condFinal);
        llvm::Value *CondFinal = B->CreateICmpEQ(tid, ConstantInt(0));
        B->CreateCondBr(CondFinal, bodyFinal, endFinal);
        B->SetInsertPoint(endFinal);
        B->CreateBr(returnBlock);
        B->SetInsertPoint(bodyFinal);
        B->CreateBr(endFinal);
    }


    
      
    void GenerateKernelFunction() {
        ///IMPORTANT: TEST RUN AT MOVING KERNEL FUNCTION INTO COMPILE PART OF PROGRAM
        //WE have a function and we want a new function
        
        
        llvm::Function *TidXFunc, *BlockDimXFunc, *BlockIdxXFunc;
        
        llvm::LLVMContext &VMContext = llvm::getGlobalContext();
        llvm::Type *Int32Ty = llvm::Type::getInt32Ty(VMContext);
        
        TidXFunc = llvm::Function::Create(llvm::FunctionType::get(Int32Ty, false),
                                          llvm::Function::ExternalLinkage,
                                          "llvm.nvvm.read.ptx.sreg.tid.x", mainModule);
        BlockDimXFunc = llvm::Function::Create(llvm::FunctionType::get(Int32Ty, false),
                                               llvm::Function::ExternalLinkage,
                                               "llvm.nvvm.read.ptx.sreg.ntid.x", mainModule);
        BlockIdxXFunc = llvm::Function::Create(llvm::FunctionType::get(Int32Ty, false),
                                               llvm::Function::ExternalLinkage,
                                               "llvm.nvvm.read.ptx.sreg.ctaid.x", mainModule);
        
        std::vector<llvm::Type *>ArgTys;
        
        
        
        llvm::FunctionType *KernelFuncTy = llvm::FunctionType::get(
                                                                   llvm::Type::getVoidTy(VMContext),
                                                                   ArgTys, /*isVarArg=*/false);
        
        
        KernelFunc = llvm::Function::Create(KernelFuncTy,
                                            llvm::Function::ExternalLinkage,
                                            "ker", mainModule);
        
        /*
         *This code is used to annotate which function we will use as an entry point
         */
        
        std::vector<llvm::Value *> Vals;
        Int32Ty = llvm::Type::getInt32Ty(VMContext); 
        llvm::NamedMDNode *Annot = mainModule->getOrInsertNamedMetadata("nvvm.annotations");
        llvm::MDString *Str = llvm::MDString::get(VMContext, "kernel");
        Vals.push_back(KernelFunc);
        Vals.push_back(Str);
        Vals.push_back(llvm::ConstantInt::get(Int32Ty, 1));
        llvm::MDNode *MdNode = llvm::MDNode::get(VMContext, Vals);
        Annot->addOperand(MdNode);
        
     
        llvm::BasicBlock *EntryBB = llvm::BasicBlock::Create(VMContext, "entry", KernelFunc, /*InserBefore*/0);
        
        
        B = new llvm::IRBuilder<>(VMContext);
        B->SetInsertPoint(EntryBB);
        
        // id = blockDim.x * blockIdx.x + threadIdx.x
        llvm::Value *TidXRead = B->CreateSExt(B->CreateCall(TidXFunc, std::vector<llvm::Value *>(),"calltmp"), intType);
        llvm::Value *BlockDimXRead = B->CreateSExt(B->CreateCall(BlockDimXFunc, std::vector<llvm::Value *>(), "calltmp"), intType);
        llvm::Value *BlockIdxXRead = B->CreateSExt(B->CreateCall(BlockIdxXFunc, std::vector<llvm::Value *>(), "calltmp"), intType);
        llvm::Value *Id = B->CreateMul(BlockDimXRead, BlockIdxXRead);
        Id = B->CreateAdd(Id, TidXRead);
        
        
        std::vector<llvm::Value *> IndexArgs;
        Id->setName("index");
        IndexArgs.push_back(Id);
        TidXRead->setName("tid");
        IndexArgs.push_back(TidXRead);
        IndexArgs.push_back(BlockIdxXRead);
        
        //call index function
        B->CreateCall(function, IndexArgs);
        
        
        //complete the function
        
        B->CreateRetVoid();
        
        
        //END ATTEMPT AT KERNEL FUNCTION
        
        
        delete B;

        
    }
    
    void Compile() {
        
        
        GenerateIndexFunction();
        
        GenerateKernelFunction();
        
        
    }
    
    llvm::Constant * ConstantInt(int64_t i) {
        return llvm::ConstantInt::get(intType,i);
    }
    llvm::Constant * ConstantDouble(double d) {
        return llvm::ConstantFP::get(doubleType, d);
    }
    llvm::Constant * ConstantFloat(float d) {
        return llvm::ConstantFP::get(llvm::Type::getFloatTy(*C), d);
    }
	llvm::Constant * ConstantLogical(char l) {
		if (l == 0xff)		
			return llvm::ConstantInt::get(logicalType1,true);
		else
			return llvm::ConstantInt::get(logicalType1,false);
	}
    llvm::Value * ConstantPointer(void * ptr, llvm::Type * typ) {
        llvm::Constant* ci = llvm::ConstantInt::get(intType, (uint64_t)ptr); 
        llvm::Value* cp = llvm::ConstantExpr::getIntToPtr(ci, llvm::PointerType::getUnqual(typ));
        return cp; 
    }
    void Output() {
        //We loaded the addresses sequentially, so we can access them in order via this hack.
        int output = 0;
        int sizeOfResult;
        for(size_t i = 0; i < trace->nodes.size(); i++) {
            IRNode & n = trace->nodes[i];
            if (n.liveOut) {
                if (n.group == IRNode::MAP || n.group == IRNode::GENERATOR) {
                    sizeOfResult = n.out.length;
                    if(n.type == Type::Double) {
                        cudaMemcpy(((Double&)n.out).v(), outputGPU[output], sizeOfResult * sizeof(Double::Element), cudaMemcpyDeviceToHost);
                    } else if(n.type == Type::Integer) {
                        cudaMemcpy(((Integer&)n.out).v(), outputGPU[output], sizeOfResult * sizeof(Integer::Element), cudaMemcpyDeviceToHost);
                    } else if(n.type == Type::Logical) {
                        cudaMemcpy(((Logical&)n.out).v(), outputGPU[output], sizeOfResult * sizeof(Logical::Element), cudaMemcpyDeviceToHost);
                    } else {
                        _error("Unknown type in initialize outputs");
                    }
                }
                if (n.group == IRNode::FOLD) {
                    sizeOfResult = n.out.length;
                    if(n.type == Type::Double) {
                        double * reductionVector = new double[outputReductionSize];
                        cudaMemcpy(reductionVector, outputGPU[output], outputReductionSize * sizeof(Double::Element), cudaMemcpyDeviceToHost);
                        
                        switch(n.op) {

                            case IROpCode::sum: {
                                double sum = 0.0;
                                for (int i = 0; i < outputReductionSize; i++) {
                                    sum += reductionVector[i];
                                }
                                n.out.d = sum;
                                break;
                            }
                            default:
                                break;
                        }
                    }
                    else if(n.type == Type::Integer) {
                        sizeOfResult = n.out.length;
                        int64_t * reductionVector = new int64_t[outputReductionSize];
                        cudaMemcpy(reductionVector, outputGPU[output], outputReductionSize * sizeof(Integer::Element), cudaMemcpyDeviceToHost);
                        switch(n.op) {
                            case IROpCode::sum: {
                                int64_t sum = 0;
                                for (int i = 0; i < outputReductionSize; i++) {
                                    sum += reductionVector[i];
                                }
                                n.out.i = sum;
                                break;
                            }
                            default:
                                break;
                        }
                    } else if(n.type == Type::Logical) {
                        cudaMemcpy(((Logical&)n.out).v(), outputGPU[output], sizeOfResult * sizeof(Logical::Element), cudaMemcpyDeviceToHost);
                    } else {
                        _error("Unknown type in initialize outputs");
                    }
                }
                output++;

            }
        }
        for (int i = 0; i < thingsToFree.size(); i++) {
            cudaFree(thingsToFree[i]);
        }
    }
    void CompileBody(llvm::Value *blockID, llvm::Value *tid, int sizeOfArray) {
        llvm::Value * loopIndexValue = loopIndex();
        
        std::vector<llvm::Value *> loopIndexArray;
        loopIndexArray.push_back(loopIndexValue);
        for(size_t i = 0; i < trace->nodes.size(); i++) {
            IRNode & n = trace->nodes[i];
            switch(n.op) {
                case IROpCode::load: {
                    void * p;
                    if(n.in.isLogical()) {
                        //p = ((Logical&)n.in).v();
                        int size = ((Logical&)n.in).length*sizeof(Logical::Element);
                        cudaMalloc((void**)&p, size);
                        cudaMemcpy(p, ((Logical&)n.in).v(), size, cudaMemcpyHostToDevice);

                    }
                    else if(n.in.isInteger()) {
                        //p = ((Integer&)n.in).v();
                        int size = ((Integer&)n.in).length*sizeof(Integer::Element);
                        cudaMalloc((void**)&p, size);
                        cudaMemcpy(p, ((Integer&)n.in).v(), size, cudaMemcpyHostToDevice);
                    }
                    else if(n.in.isDouble()) {
                        //p = ((Double&)n.in).v();
                        int size = ((Double&)n.in).length*sizeof(Double::Element);
                        cudaMalloc((void**)&p, size);
                        cudaMemcpy(p, ((Double&)n.in).v(), size, cudaMemcpyHostToDevice);
                    }
                    else
                        _error("unsupported type");
                    
                    thingsToFree.push_back(p);
                    llvm::Type * t = getType(n.type);
                    
                    inputGPU.push_back(p);
                    llvm::Value * vector = ConstantPointer(p, t);
                    llvm::Value * elementAddr = B->CreateGEP(vector, loopIndexArray);
                    inputGPUAddr.push_back(elementAddr);
                    values[i] = B->CreateLoad(elementAddr);
                } break;
				case IROpCode::nop:
                    break;
                case IROpCode::add:
                    switch(n.type) {
                        case Type::Double:
                            values[i] = B->CreateFAdd(values[n.binary.a],values[n.binary.b]);
                            break;
                        case Type::Integer:
                            values[i] = B->CreateAdd(values[n.binary.a],values[n.binary.b]);
                            break;
                        default:
                            _error("unsupported type");
                    }
                    break;
				case IROpCode::addc:
                    switch(n.type) {
                        case Type::Double:{
							llvm::Value * temp = ConstantDouble(n.constant.d);
                            values[i] = B->CreateFAdd(values[n.unary.a],temp);
                            break;
						}
                        case Type::Integer:{
							llvm::Value * temp = ConstantInt(n.constant.i);
                            values[i] = B->CreateAdd(values[n.unary.a],temp);
                            break;
						}
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
                            values[i] = B->CreateSub(values[n.binary.a],values[n.binary.b]);
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
						case Type::Logical:
							values[i] = ConstantLogical(n.constant.l);
							break;
                        default:
                            _error("unsupported type");
                            break;
					}
                    break;
                case IROpCode::exp:
                    switch(n.type) {
                        case Type::Double: {
                            
                            llvm::Value * xlog2e = ConstantDouble(1.442695040888963407359924681001);
                            llvm::Value * val = values[n.unary.a];
                            val->setName("val");
                            
                            llvm::Value * expComp = B->CreateFPTrunc(B->CreateFMul(val, xlog2e, "xloge"), llvm::Type::getFloatTy(*C));
                           
                            
                            std::vector<llvm::Type *> indices;
                            indices.push_back(llvm::Type::getFloatTy(*C));
                            llvm::ArrayRef<llvm::Type *>IndexArgs= llvm::ArrayRef<llvm::Type*>(indices);
                            llvm::FunctionType *placeHolder = llvm::FunctionType::get(llvm::Type::getFloatTy(*C), IndexArgs, false);
                            llvm::StringRef Constraints = llvm::StringRef("=f,f");
                            llvm::InlineAsm * base2 = llvm::InlineAsm::get(placeHolder, "ex2.approx.f32 $0, $1;", Constraints, false, false);
                            
                            llvm::Value * callee = base2;
                            
                            std::vector<llvm::Value *> asmArgs;
                            expComp->setName("expComp");
                            asmArgs.push_back(expComp);

                            
                            values[i] = B->CreateFPExt(B->CreateCall(callee, asmArgs, "2^xloge"), llvm::Type::getDoubleTy(*C));
                            break;
                        }
                        case Type::Integer: {
                            _error("unsupported type");
                            break;
                        }
                        default:
                            _error("unsupported type");
                            break;
					}
                    break;
                case IROpCode::log:
                    switch(n.type) {
                        case Type::Double: {
                            
                            llvm::Value * xlog2e = ConstantFloat(0.693147180559945309417232121458176);
                            
                            llvm::Value * x = B->CreateFPTrunc(values[n.unary.a], llvm::Type::getFloatTy(*C));
                            x->setName("x");
                            
                            std::vector<llvm::Type *> indices;
                            indices.push_back(llvm::Type::getFloatTy(*C));
                            llvm::ArrayRef<llvm::Type *>IndexArgs= llvm::ArrayRef<llvm::Type*>(indices);
                            llvm::FunctionType *placeHolder = llvm::FunctionType::get(llvm::Type::getFloatTy(*C), IndexArgs, false);
                            llvm::StringRef Constraints = llvm::StringRef("=f,f");
                            llvm::InlineAsm * base2 = llvm::InlineAsm::get(placeHolder, "lg2.approx.f32 $0, $1;", Constraints, false, false);
                            
                            llvm::Value * callee = base2;
                            
                            std::vector<llvm::Value *> asmArgs;
                            asmArgs.push_back(x);
                            
                            
                            llvm::Value* logVal = B->CreateCall(callee, asmArgs, "");
                            values[i] = B->CreateFPExt(B->CreateFMul(xlog2e, logVal), llvm::Type::getDoubleTy(*C));
                            break;
                        }
                        case Type::Integer: {
                            _error("unsupported type");
                            break;
                        }
                        default:
                            _error("unsupported type");
                            break;
					}
                    break;
                case IROpCode::sqrt:
                    switch(n.type) {
                        case Type::Double: {
                            
                            
                            llvm::Value * x = values[n.unary.a];
                            x->setName("x");
                            
                            std::vector<llvm::Type *> indices;
                            indices.push_back(llvm::Type::getDoubleTy(*C));
                            llvm::ArrayRef<llvm::Type *>IndexArgs= llvm::ArrayRef<llvm::Type*>(indices);
                            llvm::FunctionType *placeHolder = llvm::FunctionType::get(llvm::Type::getDoubleTy(*C), IndexArgs, false);
                            llvm::StringRef Constraints = llvm::StringRef("=d,d");
                            llvm::InlineAsm * sqrt = llvm::InlineAsm::get(placeHolder, "sqrt.f64 $0, $1;", Constraints, false, false);
                            
                            llvm::Value * callee = sqrt;
                            
                            std::vector<llvm::Value *> asmArgs;
                            asmArgs.push_back(x);
                            
                            
                            values[i] = B->CreateCall(callee, asmArgs, "");
                            break;
                        }
                        case Type::Integer: {
                            _error("unsupported type");
                            break;
                        }
                        default:
                            _error("unsupported type");
                            break;
					}
                    break;
                case IROpCode::abs:
                    switch(n.type) {
                        case Type::Double: {
                            llvm::Value * Result = B->CreateBitCast(values[n.unary.a], llvm::Type::getInt64Ty(*C));
                            values[i] = B->CreateBitCast(B->CreateAnd(Result, ConstantInt(0x7fffffffffffffffULL)), llvm::Type::getDoubleTy(*C));
                            break;
                        }
                        case Type::Integer: {
                            std::vector<llvm::Type *> indices;
                            indices.push_back(llvm::Type::getInt64Ty(*C));
                            llvm::ArrayRef<llvm::Type *>IndexArgs= llvm::ArrayRef<llvm::Type*>(indices);
                            llvm::FunctionType *placeHolder = llvm::FunctionType::get(llvm::Type::getInt64Ty(*C), IndexArgs, false);
                            llvm::StringRef Constraints = llvm::StringRef("=l,l");
                            llvm::InlineAsm * absInt = llvm::InlineAsm::get(placeHolder, "abs.s64 $0, $1;", Constraints, false, false);

                            llvm::InlineAsm::ConstraintInfoVector constraintInfoV = absInt->ParseConstraints();
                            
                            llvm::Value * callee = absInt;
                            llvm::Value * a = values[n.unary.a];

                            
                            std::vector<llvm::Value *> asmArgs;
                            a->setName("a");
                            asmArgs.push_back(a);
                            values[i] = B->CreateCall(callee, asmArgs, "absInt");
                           
                            break;
                        }
                        default:
                            _error("unsupported type");
                            break;
					}
                    break;
                case IROpCode::mul:
                    switch(n.type) {
                        case Type::Double:
                            values[i] = B->CreateFMul(values[n.binary.a],values[n.binary.b]);
                            break;
                        case Type::Integer:
                            values[i] = B->CreateMul(values[n.binary.a],values[n.binary.b]);
                            break;
                        default:
                            _error("unsupported type");
                    }					
                    break;
				case IROpCode::mulc:
                    switch(n.type) {
                        case Type::Double:{
							llvm::Value * temp = ConstantDouble(n.constant.d);
                            values[i] = B->CreateFMul(values[n.unary.a],temp);
                            break;
						}
                        case Type::Integer:{
							llvm::Value * temp = ConstantInt(n.constant.i);
                            values[i] = B->CreateMul(values[n.unary.a],temp);
                            break;
						}
                        default:
                            _error("unsupported type");
                    }
                    break;
				case IROpCode::div:
                    switch(n.type) {
                        case Type::Double:
                            values[i] = B->CreateFDiv(values[n.binary.a],values[n.binary.b]);
                            break;
                        case Type::Integer:
                            values[i] = B->CreateSDiv(values[n.binary.a],values[n.binary.b]);
                            break;
                        default:
                            _error("unsupported type");
                    }					
                    break;
				case IROpCode::idiv:
                    switch(n.type) {
                        case Type::Double: {
							llvm::Value * temp;
                            temp = B->CreateFDiv(values[n.binary.a],values[n.binary.b]);
							temp = B->CreateFPToSI(temp, intType);
							values[i] = B->CreateSIToFP(temp, doubleType);
                            break;
						}
                        case Type::Integer:
                            values[i] = B->CreateSDiv(values[n.binary.a],values[n.binary.b]);
                            break;
                        default:
                            _error("unsupported type");
                    }					
                    break;
					
				case IROpCode::mod:
                    switch(n.type) {
                        case Type::Double: {
							llvm::Value * temp;
                            temp = B->CreateFDiv(values[n.binary.a],values[n.binary.b]);
							temp = B->CreateFPToSI(temp, intType);
							temp = B->CreateSIToFP(temp, doubleType);
							temp = B->CreateFMul(temp,values[n.binary.b]);
							values[i] = B->CreateFSub(values[n.binary.a],temp);
                            break;
						}
                        case Type::Integer: {
							llvm::Value * temp;
                            temp = B->CreateSDiv(values[n.binary.a],values[n.binary.b]);
                            temp = B->CreateMul(temp,values[n.binary.b]);
                            values[i] = B->CreateSub(values[n.binary.a],temp);
                            break;
						}
                        default:
                            _error("unsupported type");
                    }					
                    break;
				case IROpCode::neg:
                    switch(n.type) {
                        case Type::Double:
                            values[i] = B->CreateFNeg(values[n.unary.a]);
                            break;
                        case Type::Integer:
                            values[i] = B->CreateNeg(values[n.unary.a]);
                            break;
                        default:
                            _error("unsupported type");
                    }					
                    break;
                case IROpCode::ifelse:
                    {

                        llvm::Value * control = values[n.trinary.c];
                        llvm::BasicBlock * condBlock = createAndInsertBB("condBlock");
                        llvm::BasicBlock * ifBlock = createAndInsertBB("ifBlock");
                        llvm::BasicBlock * elseBlock = createAndInsertBB("elseBlock");
                        
                        llvm::BasicBlock * nextBlock = createAndInsertBB("nextBlock");
                            
                        B->CreateBr(condBlock);
                        B->SetInsertPoint(condBlock);
                        B->CreateCondBr(control, ifBlock, elseBlock);
                        B->SetInsertPoint(ifBlock);
                        
                        B->CreateBr(nextBlock);
                        B->SetInsertPoint(elseBlock);
                            
                        B->CreateBr(nextBlock);
                        B->SetInsertPoint(nextBlock);
                            
                        llvm::PHINode * result = B->CreatePHI(values[n.trinary.a]->getType(), 2);
                        result->addIncoming(values[n.trinary.b], ifBlock);
                        result->addIncoming(values[n.trinary.a], elseBlock);
                        values[i] = result;
                        body = nextBlock;
                        break;
                    }					
                case IROpCode::sum:
                    reduction = true;
                    switch(n.type) {
                        case Type::Double: {
                            
                            llvm::GlobalVariable *shared = new llvm::GlobalVariable((*mainModule), llvm::ArrayType::get(llvm::Type::getDoubleTy(*C), sizeOfArray), false, llvm::GlobalValue::ExternalLinkage, 0, "sharedMemory", 0, 0, 3);
                            //set to the top of the function
                            if (!reductionBlocksPresent) {
                                returnBlock = createAndInsertBB("return");
                                condBlocks[0] = createAndInsertBB("cond512");
                                condBlocks[1] = createAndInsertBB("cond256");
                                condBlocks[2] = createAndInsertBB("cond128");
                                condBlocks[3] = createAndInsertBB("cond32");
                                
                                endBlocks[0] = createAndInsertBB("end512");
                                endBlocks[1] = createAndInsertBB("end256");
                                endBlocks[2] = createAndInsertBB("end128");
                                endBlocks[3] = createAndInsertBB("end32");
                                
                                bodyBlocks[0] = createAndInsertBB("body512");
                                bodyBlocks[1] = createAndInsertBB("body256");
                                bodyBlocks[2] = createAndInsertBB("body128");
                                bodyBlocks[3] = createAndInsertBB("body32");
                                
                                condFinal = createAndInsertBB("condFinal");
                                bodyFinal = createAndInsertBB("bodyFinal");
                                endFinal = createAndInsertBB("endFinal");
                                reductionBlocksPresent = true;
                            }
                                
                            Synchronize(function, tid);

                            InitializeSharedDouble(function, shared, tid);
                            
                            llvm::AllocaInst *Alloca = CreateEntryBlockAllocaDouble(function, "mySum");
                            
                            
                            
                            B->SetInsertPoint(body);
                            
                            
                            B->CreateStore((B->CreateFAdd(values[n.unary.a], B->CreateLoad(Alloca))), Alloca);
                            
                           
                            B->SetInsertPoint(origEnd);
                            std::vector<llvm::Value *> indices;
                            indices.push_back(llvm::ConstantInt::get(llvm::Type::getInt64Ty(*C), 0));
                            indices.push_back(tid);
                            
                            llvm::ArrayRef<llvm::Value *> gepIndices = llvm::ArrayRef<llvm::Value*>(indices);
                            llvm::Value *sharedMemIndex = B->CreateGEP(shared, gepIndices, "index");
                            B->CreateStore(B->CreateLoad(Alloca), sharedMemIndex);
                            
                            
                            B->CreateCall(Sync);
                            

                            
                            int current = 512;
                            for(int ind = 0 ; ind < 3; ind++) {
                                if(numThreads >= current) {
                                    B->SetInsertPoint(bodyBlocks[ind]);
                                
                                    std::vector<llvm::Value *> indices;
                                    indices.push_back(llvm::ConstantInt::get(llvm::Type::getInt64Ty(*C), 0));
                                    indices.push_back(B->CreateAdd(tid, ConstantInt(current/2)));
                                
                                    llvm::Value *sharedMemIndexT = B->CreateGEP(shared, indices, "T");

                                    B->CreateStore(B->CreateFAdd(B->CreateLoad(sharedMemIndex), B->CreateLoad(sharedMemIndexT)), sharedMemIndex);
                                
                                    B->SetInsertPoint(endBlocks[ind]);
                                
                                    B->CreateCall(Sync);
                                }
                                current /= 2;
                            }
                            
                            
                            B->SetInsertPoint(bodyBlocks[3]);
                            //declare something volatile
                            for (current = 64; current > 1; current/=2) {
                                if (numThreads >= current) {
                                    std::vector<llvm::Value *> indices;
                                    indices.push_back(ConstantInt(0));
                                    indices.push_back(B->CreateAdd(tid, ConstantInt(current/2)));
                                    
                                    llvm::Value *sharedMemIndexT = B->CreateGEP(shared, indices, "");
                                    
                                    
                                    B->CreateStore(B->CreateFAdd(B->CreateLoad(sharedMemIndex), B->CreateLoad(sharedMemIndexT)), sharedMemIndex, true);
                                    B->CreateCall(Sync);
                                }

                            }
                            
                            B->SetInsertPoint(endBlocks[3]);
                             
                            
                            B->CreateCall(Sync);
                            
                            
                            B->SetInsertPoint(bodyFinal);
                            
                            
                            std::vector<llvm::Value *> outputIndices;
                            outputIndices.push_back(ConstantInt(0));
                            outputIndices.push_back(ConstantInt(0)); //should be 0
                            
                            llvm::ArrayRef<llvm::Value *> outputIndixes = llvm::ArrayRef<llvm::Value*>(outputIndices);
                            llvm::Value *output = B->CreateGEP(shared, outputIndixes, "finalOutput");
                            
                            values[i] = B->CreateLoad(output); 
                            
                            end = returnBlock;
                            
                            B->SetInsertPoint(bodyFinal);
                            break;
                        }
                        case Type::Integer: {
                            //set to the top of the function
                            
                            llvm::GlobalVariable *shared = new llvm::GlobalVariable((*mainModule), llvm::ArrayType::get(llvm::Type::getInt64Ty(*C), sizeOfArray), false, llvm::GlobalValue::ExternalLinkage, 0, "sharedMemory", 0, 0, 3);
                            //set to the top of the function
                             if (!reductionBlocksPresent) {
                                returnBlock = createAndInsertBB("return");
                                condBlocks[0] = createAndInsertBB("cond512");
                                condBlocks[1] = createAndInsertBB("cond256");
                                condBlocks[2] = createAndInsertBB("cond128");
                                condBlocks[3] = createAndInsertBB("cond32");
                             
                                endBlocks[0] = createAndInsertBB("end512");
                                endBlocks[1] = createAndInsertBB("end256");
                                endBlocks[2] = createAndInsertBB("end128");
                                endBlocks[3] = createAndInsertBB("end32");
                             
                                bodyBlocks[0] = createAndInsertBB("body512");
                                bodyBlocks[1] = createAndInsertBB("body256");
                                bodyBlocks[2] = createAndInsertBB("body128");
                                bodyBlocks[3] = createAndInsertBB("body32");
                             
                                condFinal = createAndInsertBB("condFinal");
                                bodyFinal = createAndInsertBB("bodyFinal");
                                endFinal = createAndInsertBB("endFinal");
                                reductionBlocksPresent = true;
                             }
                            
                            Synchronize(function, tid);
                            
                            InitializeSharedInteger(function, shared, tid);
                            
                            llvm::AllocaInst *Alloca = CreateEntryBlockAllocaInteger(function, "mySum");
                            
                            
                            
                            B->SetInsertPoint(body);
                            
                            
                            B->CreateStore((B->CreateAdd(values[n.unary.a], B->CreateLoad(Alloca))), Alloca);
                            
                            
                            B->SetInsertPoint(origEnd);
                            std::vector<llvm::Value *> indices;
                            indices.push_back(llvm::ConstantInt::get(llvm::Type::getInt64Ty(*C), 0));
                            indices.push_back(tid);
                            
                            llvm::ArrayRef<llvm::Value *> gepIndices = llvm::ArrayRef<llvm::Value*>(indices);
                            llvm::Value *sharedMemIndex = B->CreateGEP(shared, gepIndices, "index");
                            B->CreateStore(B->CreateLoad(Alloca), sharedMemIndex);
                            
                            
                            B->CreateCall(Sync);
                            
                            int current = 512;
                            for(int ind = 0 ; ind < 3; ind++) {
                                if(numThreads >= current) {
                                    B->SetInsertPoint(bodyBlocks[ind]);
                                    
                                    std::vector<llvm::Value *> indices;
                                    indices.push_back(llvm::ConstantInt::get(llvm::Type::getInt64Ty(*C), 0));
                                    indices.push_back(B->CreateAdd(tid, ConstantInt(current/2)));
                                    
                                    llvm::Value *sharedMemIndexT = B->CreateGEP(shared, indices, "T");
                                    
                                    B->CreateStore(B->CreateAdd(B->CreateLoad(sharedMemIndex), B->CreateLoad(sharedMemIndexT)), sharedMemIndex);
                                    
                                    B->SetInsertPoint(endBlocks[ind]);
                                    
                                    B->CreateCall(Sync);
                                }
                                current /= 2;
                            }
                            
                            
                            B->SetInsertPoint(bodyBlocks[3]);
                            //declare something volatile
                            for (current = 64; current > 1; current/=2) {
                                if (numThreads >= current) {
                                    std::vector<llvm::Value *> indices;
                                    indices.push_back(ConstantInt(0));
                                    indices.push_back(B->CreateAdd(tid, ConstantInt(current/2)));
                                    
                                    llvm::Value *sharedMemIndexT = B->CreateGEP(shared, indices, "");
                                    
                                    
                                    B->CreateStore(B->CreateAdd(B->CreateLoad(sharedMemIndex), B->CreateLoad(sharedMemIndexT)), sharedMemIndex, true);
                                    B->CreateCall(Sync);
                                }
                                
                            }

                            
                            B->SetInsertPoint(endBlocks[3]);
                            
                            
                            B->CreateCall(Sync);
                            
                            
                            B->SetInsertPoint(bodyFinal);
                            
                            
                            std::vector<llvm::Value *> outputIndices;
                            outputIndices.push_back(ConstantInt(0));
                            outputIndices.push_back(ConstantInt(0)); //should be 0
                            
                            llvm::ArrayRef<llvm::Value *> outputIndixes = llvm::ArrayRef<llvm::Value*>(outputIndices);
                            llvm::Value *output = B->CreateGEP(shared, outputIndixes, "finalOutput");
                            
                            values[i] = B->CreateLoad(output); 
                            
                            end = returnBlock;
                            
                            B->SetInsertPoint(bodyFinal);
                            break;
                    }
                        default:
                            _error("unsupported type");
                    }					
                    break;
				case IROpCode::cast:
				{
					Type::Enum output = n.type;
					Type::Enum input = trace->nodes[n.unary.a].type;
					switch(output) {
						case Type::Double:
							switch (input) {
								case Type::Integer:
									values[i] = B->CreateSIToFP(values[n.unary.a],doubleType);
									break;
								case Type::Logical: {
									llvm::Value * temp;
									temp = B->CreateZExt(values[n.unary.a],intType);
									values[i] = B->CreateSIToFP(temp,doubleType);
									break;
								}
								default:
									_error("unsupported type");
									break;
							}
							break;
						case Type::Integer:
							switch (input) {
								case Type::Double:
									values[i] = B->CreateFPToSI(values[n.unary.a],intType);
									break;
								case Type::Logical:
									values[i] = B->CreateZExt(values[n.unary.a],intType);
									break;
								default:
									_error("unsupported type");
									break;
							}
							break;
						case Type::Logical:
							switch(input) {
								case Type::Double:{
									llvm::Value * zero = ConstantDouble(0.0);
									values[i] = B->CreateFCmpONE(values[n.unary.a], zero);
								}
									break;
								case Type::Integer:{
									llvm::Value * zero = ConstantInt(0L);
									values[i] = B->CreateICmpNE(values[n.unary.a], zero);
								}
									break;
								default:
									_error("unsupported type");
									break;
							}
							break;	
						default:
							_error("unsupported type");
					}
					break;
				}
				case IROpCode::lt:
                    switch(trace->nodes[n.unary.a].type) {
                        case Type::Double:
                            values[i] = B->CreateFCmpOLT(values[n.binary.a],values[n.binary.b]);
                            break;
                        case Type::Integer:
							values[i] = B->CreateICmpSLT(values[n.binary.a],values[n.binary.b]);
                            break;
                        default:
                            _error("unsupported type");
                    }
                    break;
				case IROpCode::le:
                    switch(trace->nodes[n.unary.a].type) {
                        case Type::Double:
                            values[i] = B->CreateFCmpOLE(values[n.binary.a],values[n.binary.b]);
                            break;
                        case Type::Integer:
							values[i] = B->CreateICmpSLE(values[n.binary.a],values[n.binary.b]);
                            break;
                        default:
                            _error("unsupported type");
                    }
                    break;
				case IROpCode::eq:
                    switch(trace->nodes[n.unary.a].type) {
                        case Type::Double:
                            values[i] = B->CreateFCmpOEQ(values[n.binary.a],values[n.binary.b]);
                            break;
                        case Type::Integer:
							values[i] = B->CreateICmpEQ(values[n.binary.a],values[n.binary.b]);
                            break;
                        default:
                            _error("unsupported type");
                    }
                    break;
				case IROpCode::neq:
                    switch(trace->nodes[n.unary.a].type) {
                        case Type::Double:
                            values[i] = B->CreateFCmpONE(values[n.binary.a],values[n.binary.b]);
                            break;
                        case Type::Integer:
							values[i] = B->CreateICmpNE(values[n.binary.a],values[n.binary.b]);
                            break;
                        default:
                            _error("unsupported type");
                    }
                    break;
				case IROpCode::land:
					values[i] = B->CreateAnd(values[n.binary.a],values[n.binary.b]);
                    break;
				case IROpCode::lor:
					values[i] = B->CreateOr(values[n.binary.a],values[n.binary.b]);
                    break;
				case IROpCode::lnot:
					values[i] = B->CreateNot(values[n.unary.a]);
                    break;
				case IROpCode::rep:{

					llvm::Value * repIdx = ConstantInt(n.sequence.ia);
					llvm::Value * each = ConstantInt(n.sequence.ib);

					llvm::Value * prod = B->CreateMul(repIdx,each);
					
					llvm::Value * temp = B->CreateSDiv(loopIndexValue,prod);
					temp = B->CreateMul(temp,prod);
					temp = B->CreateSub(loopIndexValue,temp);
					
					values[i] = B->CreateSDiv(temp,each);

                    break;
				}
                case IROpCode::gather:{
                    void * p;
                    if(n.in.isLogical()) {
                        int size = ((Logical&)n.in).length*sizeof(Logical::Element);
                        cudaMalloc((void**)&p, size);
                        cudaMemcpy(p, ((Logical&)n.in).v(), size, cudaMemcpyHostToDevice);
                    }
                    else if(n.in.isInteger()) {
                        int size = ((Integer&)n.in).length*sizeof(Integer::Element);
                        cudaMalloc((void**)&p, size);
                        cudaMemcpy(p, ((Integer&)n.in).v(), size, cudaMemcpyHostToDevice);
                    }
                    else if(n.in.isDouble()) {
                        int size = ((Double&)n.in).length*sizeof(Double::Element);
                        cudaMalloc((void**)&p, size);
                        cudaMemcpy(p, ((Double&)n.in).v(), size, cudaMemcpyHostToDevice);
                    }
                    else
                        _error("unsupported type");
                    
                    thingsToFree.push_back(p);
                    std::vector<llvm::Value *> indices;
                    indices.push_back(values[n.unary.a]);
                    
                    llvm::Type * t = getType(n.type);
                    
                    llvm::Value * vector = ConstantPointer(p, t);
                    
                    
                    llvm::Value *value = B->CreateGEP(vector, indices);
                    values[i] = B->CreateLoad(value);
                    
                    
                    
                    
                    break;
                }                
                case IROpCode::seq:{
                    llvm::Value *value;
                    switch(n.type) {
                        case Type::Integer: {
                            llvm::Value* from = ConstantInt(n.sequence.ia);
                            llvm::Value* by = ConstantInt(n.sequence.ib);
                            value = B->CreateMul(by, loopIndexValue);
                            value = B->CreateAdd(value, from);
                            values[i] = value;
                            break;
                        }
                        case Type::Double: {
                            llvm::Value* from = ConstantDouble(n.sequence.da);
                            llvm::Value* by = ConstantDouble(n.sequence.db);
                            value = B->CreateFMul(by, B->CreateSIToFP(loopIndexValue,doubleType));
                            value = B->CreateFAdd(value, from);
                            values[i] = value;
                            break;
                        }
                        default:
                            _error("unsupported type");
                    }
					
					
					
					
                    break;
				}
                default:
                    _error("unsupported op");
                    break;
                
            }
            
            if(n.liveOut) {
                
                /*
                 *Changes need to be made here regarding the intermediate size
                 */
                
                
                
                if (n.group == IRNode::MAP || n.group == IRNode::GENERATOR) {
                    int64_t length = n.outShape.length;
                    void * p;
               
                    if(n.type == Type::Double) {
                        n.out = Double(length);

                        int size = length*sizeof(Double::Element);
                        cudaMalloc((void**)&p, size);
						
						//Grab the addresses and save them because we'll access them to put the output into
						llvm::Type * t = getType(n.type);
						llvm::Value * vector = ConstantPointer(p,t);
						B->CreateStore(values[i], B->CreateGEP(vector, loopIndexArray));
						
                    } else if(n.type == Type::Integer) {
                        n.out = Integer(length);
                        int size = length*sizeof(Integer::Element);
                        cudaMalloc((void**)&p, size);
                    
						//Grab the addresses and save them because we'll access them to put the output into
						llvm::Type * t = getType(n.type);
						llvm::Value * vector = ConstantPointer(p,t);
						B->CreateStore(values[i], B->CreateGEP(vector, loopIndexArray));
						
                    } else if(n.type == Type::Logical) {
                        n.out = Logical(length);

                        int size = length*sizeof(Logical::Element);
                        cudaMalloc((void**)&p, size);
						// Convert to 8 bit logical
						llvm::Value * temp8 = B->CreateSExt(values[i],logicalType8);
						
						//Grab the addresses and save them because we'll access them to put the output into
						llvm::Type * t = logicalType8;
						llvm::Value * vector = ConstantPointer(p,t);
						B->CreateStore(temp8, B->CreateGEP(vector, loopIndexArray));
						
                    } else {
                        _error("Unknown type in initialize outputs");
                    }
                    thingsToFree.push_back(p);
                    outputGPU.push_back(p);
                }
                else if (n.group == IRNode::FOLD) {
                    int64_t length = 1;
                    void * p;
                    
                    if(n.type == Type::Double) {
                        n.out = Double(length);
                        
                        int size = ((Double&)n.in).length*sizeof(Double::Element);
                        cudaMalloc((void**)&p, size);
						
                        //Grab the addresses and save them because we'll access them to put the output into
						llvm::Type * t = getType(n.type);
						llvm::Value * vector = ConstantPointer(p,t);
						B->CreateStore(values[i], B->CreateGEP(vector, blockID));
						
                    } else if(n.type == Type::Integer) {
                        n.out = Integer(length);
                        int size = ((Integer&)n.in).length*sizeof(Integer::Element);
                        cudaMalloc((void**)&p, size);
						
						//Grab the addresses and save them because we'll access them to put the output into
						llvm::Type * t = getType(n.type);
						llvm::Value * vector = ConstantPointer(p,t);
						B->CreateStore(values[i], B->CreateGEP(vector, blockID));
                        
                    } else if(n.type == Type::Logical) {
                        n.out = Logical(length);
                        
                        int size = ((Logical&)n.in).length*sizeof(Logical::Element);
                        cudaMalloc((void**)&p, size);
                        // Convert to 8 bit logical
						llvm::Value * temp8 = B->CreateSExt(values[i],logicalType8);
						
						//Grab the addresses and save them because we'll access them to put the output into
						llvm::Type * t = logicalType8;
						llvm::Value * vector = ConstantPointer(p,t);
						B->CreateStore(temp8, B->CreateGEP(vector, blockID));
						
                    } else {
                        _error("Unknown type in initialize outputs");
                    }
                    thingsToFree.push_back(p);
                    //Grab the addresses and save them because we'll access them to put the output into
                    outputGPU.push_back(p);
                    B->SetInsertPoint(body);
                }

            }
        }
    
    }
    
    static llvm::AllocaInst * CreateEntryBlockAllocaDouble(llvm::Function *TheFunction,
                                              const std::string &VarName) {
        llvm::IRBuilder<> TmpB(&TheFunction->getEntryBlock(),
                         TheFunction->getEntryBlock().begin());
        llvm::AllocaInst * Alloca = TmpB.CreateAlloca(llvm::Type::getDoubleTy(llvm::getGlobalContext()), 0,
                                 VarName.c_str());
        llvm::Value * temp = llvm::ConstantFP::get(llvm::Type::getDoubleTy(llvm::getGlobalContext()), 0.0);
        TmpB.CreateStore(temp, Alloca);
        return Alloca;
    }
    
    static llvm::AllocaInst * CreateEntryBlockAllocaInteger(llvm::Function *TheFunction,
                                                           const std::string &VarName) {
        llvm::IRBuilder<> TmpB(&TheFunction->getEntryBlock(),
                               TheFunction->getEntryBlock().begin());
        llvm::AllocaInst * Alloca = TmpB.CreateAlloca(llvm::Type::getInt64Ty(llvm::getGlobalContext()), 0,
                                                      VarName.c_str());
        llvm::Value * temp = llvm::ConstantInt::get(llvm::Type::getInt64Ty(llvm::getGlobalContext()), 0);
        TmpB.CreateStore(temp, Alloca);
        return Alloca;
    }
    
    void InitializeSharedDouble(llvm::Function *TheFunction, llvm::GlobalVariable *shared, llvm::Value *Tid) {
        llvm::IRBuilder<> TmpB(&TheFunction->getEntryBlock(),
                               TheFunction->getEntryBlock().begin());
        std::vector<llvm::Value *> indicesZero;
        indicesZero.push_back(ConstantInt(0));
        indicesZero.push_back(Tid);
        
        
        llvm::ArrayRef<llvm::Value *> gepIndicesZero = llvm::ArrayRef<llvm::Value*>(indicesZero);
        llvm::Value *sharedMemIndexZero = TmpB.CreateGEP(shared, gepIndicesZero, "zero");
        TmpB.CreateStore(ConstantDouble(0.0), sharedMemIndexZero);
        return;
    }
    
    void InitializeSharedInteger(llvm::Function *TheFunction, llvm::GlobalVariable *shared, llvm::Value *Tid) {
        llvm::IRBuilder<> TmpB(&TheFunction->getEntryBlock(),
                               TheFunction->getEntryBlock().begin());
        std::vector<llvm::Value *> indicesZero;
        indicesZero.push_back(ConstantInt(0));
        indicesZero.push_back(Tid);
        
        
        llvm::ArrayRef<llvm::Value *> gepIndicesZero = llvm::ArrayRef<llvm::Value*>(indicesZero);
        llvm::Value *sharedMemIndexZero = TmpB.CreateGEP(shared, gepIndicesZero, "zero");
        TmpB.CreateStore(ConstantInt(0), sharedMemIndexZero);
        return;
    }
    
    void Synchronize(llvm::Function *TheFunction, llvm::Value *Tid) {
        llvm::IRBuilder<> TmpB(&TheFunction->getEntryBlock(),
                               TheFunction->getEntryBlock().begin());
        TmpB.CreateCall(Sync);
        return;
    }
    
    
    void Execute() {
        

      //Get the bitcode from the module
        
#if defined(USE_TEXT_NVVM_INTERFACE)
        std::string BitCodeBuf;
        raw_string_ostream BitCodeBufStream(BitCodeBuf);
        BitCodeBufStream << *mainModule;
        BitCodeBufStream.str();
#else /* USE_TEXT_NVVM_INTERFACE */
        std::vector<unsigned char> BitCodeBuf;
        llvm::BitstreamWriter Stream(BitCodeBuf);
        
        BitCodeBuf.reserve(256*1024);
        llvm::WriteBitcodeToStream(mainModule, Stream);
#endif /* USE_TEXT_NVVM_INTERFACE */
        
        // generate PTX
        nvvmCU CU;
        size_t Size;
        char *PtxBuf;
        __NVVM_SAFE_CALL(nvvmCreateCU(&CU));
        
#if defined(USE_TEXT_NVVM_INTERFACE)
        __NVVM_SAFE_CALL(nvvmCUAddModule(CU, BitCodeBuf.c_str(), BitCodeBuf.size()));
#else /* USE_TEXT_NVVM_INTERFACE */
        __NVVM_SAFE_CALL(nvvmCUAddModule(CU, (char *)&BitCodeBuf.front(), BitCodeBuf.size()));
#endif /* USE_TEXT_NVVM_INTERFACE */
        
        
        //Turn LLVM IR to PTX
        __NVVM_SAFE_CALL(nvvmCompileCU(CU, /*numOptions = */0, /*options = */NULL));
        __NVVM_SAFE_CALL(nvvmGetCompiledResultSize(CU, &Size));
        PtxBuf = new char[Size+1];
        __NVVM_SAFE_CALL(nvvmGetCompiledResult(CU, PtxBuf));
        PtxBuf[Size] = 0;
        __NVVM_SAFE_CALL(nvvmDestroyCU(&CU));
                

        //Create the threads based on the size
        
        const char *ptxstr = PtxBuf;
        const char *kname = "ker";
        unsigned len = trace->Size;
        //need to get the maxlength
        
        unsigned nthreads = numThreads;
        unsigned nblocks;
        if (nthreads >= len) {
            nthreads = len;
            nblocks = 1;
        }
        else {
            nblocks = 1 + (len - 1)/nthreads;
        }
        
        //Create the CUthings you need and make sure that none of them are zero
        
        CUmodule module;
        CUfunction kernel;
        

        
        CUresult error = cuModuleLoadDataEx(&module, ptxstr, 0, 0, 0);
        assert(error == 0);
        error = cuModuleGetFunction(&kernel, module, kname);
        assert(error == 0);
        error = cuFuncSetBlockShape(kernel, nthreads, 1, 1);
        assert(error == 0);
        

        
        error = cuLaunchKernel(kernel,
                       nblocks, 1, 1,
                       nthreads, 1, 1,
                       0, 0,
                       0,
                       0);
        assert(error == 0);
        //Print out the output
        Output();

        
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
			case Type::Logical:
				return logicalType1;
            default:
                _error("type not understood");
                return NULL;
        }
    }
    
};

void Trace::JIT(Thread & thread) {
    std::cout << toString(thread) << "\n";
    TraceLLVMCompiler c(&thread,this);
    
    cuInit(0);
    nvvmInit();
    
    
    timespec start = get_time();
    c.Compile();
    c.Execute();
    double end = time_elapsed(start);
    std::cout << "Kernel end" << std::endl;
}

#endif
