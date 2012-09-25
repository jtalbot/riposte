
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

struct CompiledTrace {
	llvm::Function *F;
    int parameters;
    int outputCount;
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
    CompiledTrace cT;
    Thread * thread;
    Trace * trace;
    
    int numBlock;
    int numThreads;
    int outputReductionSize;
    
    bool reduction;
    bool reductionBlocksPresent;
    
    bool scan;
    bool scanBlocksPresent;
    
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
    
    //scan blocls
    llvm::BasicBlock * laneInitializer;
    llvm::BasicBlock * laneCondBlocks[5];
  	llvm::BasicBlock * laneBodyBlocks[5];
  	llvm::BasicBlock * laneEndBlocks[5];
    
    llvm::BasicBlock * laneInitializer2;
    llvm::BasicBlock * laneCondBlocks2[5];
  	llvm::BasicBlock * laneBodyBlocks2[5];
  	llvm::BasicBlock * laneEndBlocks2[5];
    
    llvm::BasicBlock * scanBlockLevelBody;
    
    llvm::BasicBlock * condLane31;
    llvm::BasicBlock * bodyLane31;
    llvm::BasicBlock * endLane31;
    
    llvm::BasicBlock * condWarp0;
    llvm::BasicBlock * bodyWarp0;
    llvm::BasicBlock * endWarp0;
    
    llvm::BasicBlock * condWarpNot0;
    llvm::BasicBlock * bodyWarpNot0;
    llvm::BasicBlock * endWarpNot0;
    
    llvm::BasicBlock * condBlockResult;
    llvm::BasicBlock * bodyBlockResult;
    llvm::BasicBlock * endBlockResult;
    
    //reductionBlocks
    llvm::BasicBlock * condBlocks[4];
    llvm::BasicBlock * bodyBlocks[4];
    llvm::BasicBlock * endBlocks[4];

    llvm::BasicBlock * condFinal;
    llvm::BasicBlock * bodyFinal;
    llvm::BasicBlock * endFinal;    
    
    llvm::BasicBlock * origEnd; 
    llvm::BasicBlock * origBody;
    
    std::vector<llvm::Value *> values;
    std::vector<llvm::Value *> outputGPU;
    
    std::vector<void *> inputGPU;
    std::vector<llvm::Value *> inputGPUAddr;
    
    
    TraceLLVMCompiler(Thread * th, Trace * tr) 
    : L(th->state.llvmState), thread(th), trace(tr), values(tr->nodes.size(),NULL) {

        llvm::InitializeNativeTarget();
        C = &llvm::getGlobalContext();
        mainModule = new llvm::Module("riposte",*C);
        
        
        numBlock = 180;
        numThreads = 128;
        
        
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

    void GeneratePTXIndexFunction() {

        intType = llvm::Type::getInt64Ty(*C);
        doubleType = llvm::Type::getDoubleTy(*C);
        logicalType1 = llvm::Type::getInt1Ty(*C);
        logicalType8 = llvm::Type::getInt8Ty(*C);
        
        int maxLengthOfArrays = trace->nodes.size();
        for(size_t i = 0; i < trace->nodes.size(); i++) {
            IRNode & n = trace->nodes[i];
        }
        llvm::PointerType * ptrInt = llvm::PointerType::getUnqual(llvm::Type::getInt64PtrTy(*C));
        llvm::PointerType * ptrDouble = llvm::PointerType::getUnqual(llvm::Type::getDoublePtrTy(*C));
        llvm::PointerType * ptrLogical = llvm::PointerType::getUnqual(llvm::Type::getInt1PtrTy(*C));

        llvm::Constant *cons = mainModule->getOrInsertFunction("indexFunc", llvm::Type::getVoidTy(*C), intType, intType, intType, 
            ptrInt, ptrDouble, ptrLogical, ptrInt, ptrDouble, ptrLogical, NULL);

        function = llvm::cast<llvm::Function>(cons);
        function->setCallingConv(llvm::CallingConv::C);
        
        llvm::Function::arg_iterator args = function->arg_begin();
        llvm::Value* index = args++;
        llvm::Value* tid = args++;
        tid->setName("ThreadID");
        llvm::Value* blockID = args++;
        blockID->setName("blockID");

        llvm::Value * outputAddrInt = args++;
        blockID->setName("outputAddrInt");
        llvm::Value * outputAddrDouble = args++;
        blockID->setName("outputAddrDouble");
        llvm::Value * outputAddrLogical = args++;
        blockID->setName("outputAddrLogical");

        llvm::Value * inputAddrInt = args++;
        blockID->setName("inputAddrInt");
        llvm::Value * inputAddrDouble = args++;
        blockID->setName("inputAddrDouble");
        llvm::Value * inputAddrLogical = args++;
        blockID->setName("inputAddrLogical");

        int sizeOfArray = 64;
        if (numThreads > 32)
            sizeOfArray = numThreads;
        
        
        
        reduction = false;
        reductionBlocksPresent = false;
        
        scan = false;
        scanBlocksPresent = false;
        
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
        origBody = body;
        
        B->CreateBr(cond);
        B->SetInsertPoint(cond);
        llvm::Value * c = B->CreateICmpULT(loopIndex(), Size);
        B->CreateCondBr(c,body,end);
        
        B->SetInsertPoint(body);

        CompilePTXBody(blockID, tid, sizeOfArray, outputAddrInt, outputAddrDouble, outputAddrLogical, inputAddrInt, inputAddrDouble, inputAddrLogical);
        if (scan == true) {
            ScanBlocksHelper(tid, loopIndex());
        }
        B->CreateStore(B->CreateAdd(loopIndex(), ConstantInt(numBlock * numThreads)),loopIndexAddr);
        B->CreateBr(cond);
        
        B->SetInsertPoint(end);
        B->CreateRetVoid();
                
        if (reduction == true) {
            ReductionHelper(tid);
        }
        
        mainModule->dump();
        
        llvm::verifyFunction(*function);
        
        if (thread->state.verbose)
           mainModule->dump();
        
        
        
        
        FPM->run(*function);
    }

    void ScanBlocksHelper(llvm::Value *tid, llvm::Value * loopIndexValue) {
        B->SetInsertPoint(origEnd);
        B->CreateBr(laneInitializer);
        B->SetInsertPoint(laneInitializer);
        int lane = 1;
        llvm::Value * laneNum = B->CreateAnd(tid, ConstantInt(31));
        llvm::Value * warpID = B->CreateAShr(tid, ConstantInt(5));
        B->CreateBr(laneCondBlocks[0]);
        for (int i = 0; i < 5; i++) {
            B->SetInsertPoint(laneCondBlocks[i]);
            llvm::Value * c = B->CreateICmpUGE(laneNum, ConstantInt(lane));
            B->CreateCondBr(c,laneBodyBlocks[i], laneEndBlocks[i]);
            B->SetInsertPoint(laneBodyBlocks[i]);
            B->CreateBr(laneEndBlocks[i]);
            B->SetInsertPoint(laneEndBlocks[i]);
            if (i == 4) {
                B->CreateBr(scanBlockLevelBody);
            }
            else {
                B->CreateBr(laneCondBlocks[i+1]);
            }
            lane *= 2;
        }
        B->SetInsertPoint(origBody);
        B->CreateBr(laneInitializer);
        
        B->SetInsertPoint(scanBlockLevelBody);
        B->CreateBr(condLane31);
        
        B->SetInsertPoint(condLane31);
        llvm::Value *lastLane = B->CreateICmpEQ(laneNum, ConstantInt(31));
        B->CreateCondBr(lastLane, bodyLane31, endLane31);
        B->SetInsertPoint(bodyLane31);
        B->CreateBr(endLane31);
        
        B->SetInsertPoint(endLane31);
        B->CreateBr(laneInitializer2);
        
        B->SetInsertPoint(laneInitializer2);
        B->CreateBr(laneCondBlocks2[0]);
        lane = 1;
        for (int i = 0; i < 5; i++) {
            B->SetInsertPoint(laneCondBlocks2[i]);
            llvm::Value * c = B->CreateICmpUGE(laneNum, ConstantInt(lane));
            B->CreateCondBr(c,laneBodyBlocks2[i], laneEndBlocks2[i]);
            B->SetInsertPoint(laneBodyBlocks2[i]);
            B->CreateBr(laneEndBlocks2[i]);
            B->SetInsertPoint(laneEndBlocks2[i]);
            if (i == 4) {
            }
            else {
                B->CreateBr(laneCondBlocks2[i+1]);
            }
            lane *= 2;
        }
        
        B->CreateBr(condWarp0);
        
        B->SetInsertPoint(condWarp0);
        llvm::Value *warp0 = B->CreateICmpEQ(warpID, ConstantInt(0));
        B->CreateCondBr(warp0, bodyWarp0, endWarp0);
        
        B->SetInsertPoint(bodyWarp0);
        B->CreateBr(endWarp0);
        
        B->SetInsertPoint(endWarp0);
        B->CreateBr(condWarpNot0);
        
        B->SetInsertPoint(condWarpNot0);
        llvm::Value *warpNot0 = B->CreateICmpUGT(warpID, ConstantInt(0));
        B->CreateCondBr(warpNot0, bodyWarpNot0, endWarpNot0);
        
        B->SetInsertPoint(bodyWarpNot0);
        B->CreateBr(endWarpNot0);
        B->SetInsertPoint(endWarpNot0);
        B->CreateBr(condBlockResult);
        B->SetInsertPoint(condBlockResult);
        llvm::Value * lastBlock = B->CreateICmpEQ(B->CreateSub(ConstantInt(numThreads), ConstantInt(1)), tid);
        B->CreateCondBr(lastBlock, bodyBlockResult, endBlockResult);
        
        
        B->SetInsertPoint(bodyBlockResult);
        B->CreateBr(endBlockResult);
        B->SetInsertPoint(endBlockResult);
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

    void initializeScanBlocks() {
        if (!scanBlocksPresent) {
            
            laneInitializer = createAndInsertBB("laneInitializer");
            
            laneCondBlocks[0] = createAndInsertBB("condLane1");
            laneCondBlocks[1] = createAndInsertBB("condLane2");
            laneCondBlocks[2] = createAndInsertBB("condLane4");
            laneCondBlocks[3] = createAndInsertBB("condLane8");
            laneCondBlocks[4] = createAndInsertBB("condLane16");
            
            
            laneBodyBlocks[0] = createAndInsertBB("bodyLane1");
            laneBodyBlocks[1] = createAndInsertBB("bodyLane2");
            laneBodyBlocks[2] = createAndInsertBB("bodyLane4");
            laneBodyBlocks[3] = createAndInsertBB("bodyLane8");
            laneBodyBlocks[4] = createAndInsertBB("bodyLane16");
            
            
            laneEndBlocks[0] = createAndInsertBB("endLane1");
            laneEndBlocks[1] = createAndInsertBB("endLane2");
            laneEndBlocks[2] = createAndInsertBB("endLane4");
            laneEndBlocks[3] = createAndInsertBB("endLane8");
            laneEndBlocks[4] = createAndInsertBB("endLane16");
            
            laneInitializer2 = createAndInsertBB("laneInitializer");
            
            laneCondBlocks2[0] = createAndInsertBB("condLane1");
            laneCondBlocks2[1] = createAndInsertBB("condLane2");
            laneCondBlocks2[2] = createAndInsertBB("condLane4");
            laneCondBlocks2[3] = createAndInsertBB("condLane8");
            laneCondBlocks2[4] = createAndInsertBB("condLane16");
            
            
            laneBodyBlocks2[0] = createAndInsertBB("bodyLane1");
            laneBodyBlocks2[1] = createAndInsertBB("bodyLane2");
            laneBodyBlocks2[2] = createAndInsertBB("bodyLane4");
            laneBodyBlocks2[3] = createAndInsertBB("bodyLane8");
            laneBodyBlocks2[4] = createAndInsertBB("bodyLane16");
            
            
            laneEndBlocks2[0] = createAndInsertBB("endLane1");
            laneEndBlocks2[1] = createAndInsertBB("endLane2");
            laneEndBlocks2[2] = createAndInsertBB("endLane4");
            laneEndBlocks2[3] = createAndInsertBB("endLane8");
            laneEndBlocks2[4] = createAndInsertBB("endLane16");
            
            scanBlockLevelBody = createAndInsertBB("scanBlockLevelBody");
            
            condLane31 = createAndInsertBB("condLane31");
            bodyLane31 = createAndInsertBB("bodyLane31");
            endLane31 = createAndInsertBB("endLane31");
            
            condWarp0 = createAndInsertBB("condWarp0");
            bodyWarp0 = createAndInsertBB("bodyWarp0");
            endWarp0 = createAndInsertBB("endWarp0");;
            
            condWarpNot0 = createAndInsertBB("condWarpNot0");
            bodyWarpNot0 = createAndInsertBB("bodyWarpNot0");
            endWarpNot0 = createAndInsertBB("endWarpNot0");
            
            condBlockResult = createAndInsertBB("condBlockResult");
            bodyBlockResult = createAndInsertBB("bodyBlockResult");
            endBlockResult = createAndInsertBB("endBlockResult");
            
            initializeReturn();
            
            scanBlocksPresent = true;
        }
    }
    
    void reductionBlocksInitialize() {
        if (!reductionBlocksPresent) {
            
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
            
            initializeReturn();
            
            reductionBlocksPresent = true;
        }
    }
    
    void initializeReturn() {
        if (!reductionBlocksPresent && !scanBlocksPresent) {
            returnBlock = createAndInsertBB("return");
        }
    }
      
    llvm::Value * GEPGetter(llvm::GlobalVariable * mem, llvm::Value *index) {
        std::vector<llvm::Value *> indices;
        indices.push_back(llvm::ConstantInt::get(llvm::Type::getInt64Ty(*C), 0));
        indices.push_back(index);
        
        llvm::ArrayRef<llvm::Value *> gepIndices = llvm::ArrayRef<llvm::Value*>(indices);
        return B->CreateGEP(mem, gepIndices, "T");
    }

    llvm::Value * Loader(llvm::Value * mem, int index) {
        std::vector<llvm::Value *> indices;
        indices.push_back(llvm::ConstantInt::get(llvm::Type::getInt64Ty(*C), 0));
        indices.push_back(ConstantInt(index));
        
        llvm::ArrayRef<llvm::Value *> gepIndices = llvm::ArrayRef<llvm::Value*>(indices);
        return B->CreateGEP(mem, gepIndices, "T");
    }
    
    void GeneratePTXKernelFunction() {
        ///IMPORTANT: TEST RUN AT MOVING KERNEL FUNCTION INTO COMPILE PART OF PROGRAM
        //WE have a function and we want a new function
        
        int inSize = cT.parameters*sizeof(llvm::Value *);
        int outSize = cT.outputCount*sizeof(llvm::Value *);

        void ** inputAddrInt;
        void ** inputAddrDouble;
        void ** inputAddrLogical;

        cudaMalloc((void**)inputAddrLogical, inSize);
        cudaMalloc((void**)inputAddrInt, inSize);
        cudaMalloc((void**)inputAddrDouble, inSize);

        
        void ** outputAddrInt;
        void ** outputAddrDouble;
        void ** outputAddrLogical;

        cudaMalloc((void**)outputAddrLogical, outSize);
        cudaMalloc((void**)outputAddrInt, outSize);
        cudaMalloc((void**)outputAddrDouble, outSize);

        int inPos = 0;
        for(size_t i = 0; i < trace->nodes.size(); i++) {
            IRNode & n = trace->nodes[i];
            switch(n.op) {
                case IROpCode::load: {
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
                        cudaError_t error = cudaMalloc((void**)&p, size);
                        cudaMemcpy(p, ((Double&)n.in).v(), size, cudaMemcpyHostToDevice);
                    }
                    else
                        _error("unsupported type");
                    llvm::Type * t = getType(n.type);
                    
                    inputGPU.push_back(p);
                    llvm::Value * vector = ConstantPointer(p, t);
                    inputAddrInt[inPos] = vector;
                    inputAddrDouble[inPos] = vector;
                    inputAddrLogical[inPos] = vector;

                    //inputGPUAddr.push_back(elementAddr);

                } break;
                default:
                    break;
            }
            inPos++;
        }
        int outPos = 0;
        for(size_t i = 0; i < trace->nodes.size(); i++) {
            IRNode & n = trace->nodes[i];
            if(n.liveOut) {
                
                /*
                 *Changes need to be made here regarding the intermediate size
                 */
                
                
                void *p;
                if (n.group == IRNode::MAP || n.group == IRNode::GENERATOR) {
                    int64_t length = n.outShape.length;
                    llvm::Value * vector;
               
                    if(n.type == Type::Double) {
                        n.out = Double(length);

                        int size = length*sizeof(Double::Element);
                        cudaError_t error = cudaMalloc((void**)&p, size);
                        //Grab the addresses and save them because we'll access them to put the output into
                        llvm::Type * t = getType(n.type);
                        llvm::Value * vector = ConstantPointer(p,t);
                        
                    } else if(n.type == Type::Integer) {
                        n.out = Integer(length);
                        int size = length*sizeof(Integer::Element);
                        cudaMalloc((void**)&p, size);
                    
                        //Grab the addresses and save them because we'll access them to put the output into
                        llvm::Type * t = getType(n.type);
                        llvm::Value * vector = ConstantPointer(p,t);
                        
                    } else if(n.type == Type::Logical) {
                        n.out = Logical(length);

                        int size = length*sizeof(Logical::Element);
                        cudaMalloc((void**)&p, size);
                        // Convert to 8 bit logical
                        llvm::Value * temp8 = B->CreateSExt(values[i],logicalType8);
                        
                        //Grab the addresses and save them because we'll access them to put the output into
                        llvm::Type * t = logicalType8;
                        llvm::Value * vector = ConstantPointer(p,t);
                        
                    } else {
                        _error("Unknown type in initialize outputs");
                    }
                    
                    outputGPU.push_back(vector);
                    outputAddrDouble[outPos] = vector;
                    outputAddrInt[outPos] = vector;
                    outputAddrLogical[outPos] = vector;

                }
                else if (n.group == IRNode::FOLD) {
                    int64_t length = 1;
                    void * p;
                    
                    if(n.type == Type::Double) {
                        n.out = Double(length);
                        
                        int size = ((Double&)n.in).length*sizeof(Double::Element);
                        cudaError_t error = cudaMalloc((void**)&p, size);
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
                    
                    //Grab the addresses and save them because we'll access them to put the output into
                    outputGPU.push_back(vector);
                    outputAddrDouble[outPos] = vector;
                    outputAddrInt[outPos] = vector;
                    outputAddrLogical[outPos] = vector;
                    B->SetInsertPoint(body);
                }
                else if (n.group == IRNode::SCAN) {
                    int64_t length = n.outShape.length;
                    void * p;
                    
                    if(n.type == Type::Double) {
                        n.out = Double(length);
                        
                        int size = length*sizeof(Double::Element);
                        cudaError_t error = cudaMalloc((void**)&p, size);
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
                    outputGPU.push_back(p);

                    outputAddrDouble[outPos] = vector;
                    outputAddrInt[outPos] = vector);
                    outputAddrLogical[outPos] = vector;
                }
            }
            outPos++;
        }

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
        IndexArgs.push_back(outputAddrInt);
        IndexArgs.push_back(outputAddrDouble);
        IndexArgs.push_back(outputAddrLogical);
        IndexArgs.push_back(inputAddrInt);
        IndexArgs.push_back(inputAddrDouble);
        IndexArgs.push_back(inputAddrLogical);
        
        
        //call index function
        B->CreateCall(function, IndexArgs);
        
        
        //complete the function
        
        B->CreateRetVoid();
        
        
        //END ATTEMPT AT KERNEL FUNCTION
        
        
        delete B;

        
    }
    
    llvm::Value *scanWarp(llvm::GlobalVariable * shared, llvm::Value *tid) {
        B->SetInsertPoint(laneInitializer);
        //warp scan
        int laneNum = 1;
        
        llvm::Value *value = GEPGetter(shared, tid);
        for (int i = 0; i < 5; i++) {
            B->SetInsertPoint(laneBodyBlocks[i]);
            
            llvm::Value *Back = GEPGetter(shared, B->CreateSub(tid, ConstantInt(laneNum)));
            
            B->CreateStore(B->CreateFAdd(B->CreateLoad(Back), B->CreateLoad(value)), value);
            
            B->SetInsertPoint(laneEndBlocks[i]);
            B->CreateCall(Sync);
            laneNum *= 2;
        }
        B->SetInsertPoint(scanBlockLevelBody);
        return value;
    }
    
    llvm::Value *scanWarp2(llvm::GlobalVariable * shared, llvm::Value *tid) {
        B->SetInsertPoint(laneInitializer2);
        //warp scan
        int laneNum = 1;
        
        llvm::Value *value = GEPGetter(shared, tid);
        for (int i = 0; i < 5; i++) {
            B->SetInsertPoint(laneBodyBlocks2[i]);
            
            llvm::Value *Back = GEPGetter(shared, B->CreateSub(tid, ConstantInt(laneNum)));
            
            B->CreateStore(B->CreateFAdd(B->CreateLoad(Back), B->CreateLoad(value)), value);
            
            B->SetInsertPoint(laneEndBlocks2[i]);
            B->CreateCall(Sync);
            laneNum *= 2;
        }
        B->SetInsertPoint(scanBlockLevelBody);
        return value;
    }
    
    llvm::Value *scanBlock(llvm::GlobalVariable * shared, llvm::Value * tid) {
        B->SetInsertPoint(scanBlockLevelBody);
        llvm::Value *lane = B->CreateAnd(loopIndex(), ConstantInt(31));
        llvm::Value *warpID = B->CreateAShr(loopIndex(), 5);
        llvm::Value *val = B->CreateAlloca(llvm::Type::getDoubleTy(llvm::getGlobalContext()), 0,
                                        "value");
        B->CreateStore(B->CreateLoad(scanWarp(shared, tid)), val);
        B->CreateCall(Sync);
        
        B->SetInsertPoint(bodyLane31);
        
        llvm::Value *loopPoint = GEPGetter(shared, tid);
        B->CreateCall(Sync);
        
        llvm::Value *warpIDPoint = GEPGetter(shared, warpID);

        B->CreateStore(B->CreateLoad(loopPoint), warpIDPoint);
        B->CreateCall(Sync);
        B->SetInsertPoint(endLane31);
        B->CreateCall(Sync);
        
        B->SetInsertPoint(bodyWarp0);
        scanWarp2(shared, tid);
        B->SetInsertPoint(endWarp0);
        B->CreateCall(Sync);
        
        B->SetInsertPoint(bodyWarpNot0);
        
        llvm::Value *backWarpIDPoint = GEPGetter(shared, B->CreateSub(warpID, ConstantInt(1)));
        llvm::Value *result = B->CreateFAdd(B->CreateLoad(backWarpIDPoint), B->CreateLoad(val));
        B->CreateStore(result, val);
        B->SetInsertPoint(endWarpNot0);
        B->CreateCall(Sync);
        
        llvm::Value * insertPoint = GEPGetter(shared, tid);
        
        B->CreateStore(B->CreateLoad(val), insertPoint);
        B->CreateCall(Sync);
        
        return B->CreateLoad(val);
        
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
                if (n.group == IRNode::MAP || n.group == IRNode::GENERATOR || n.group == IRNode::SCAN) {
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
    }
    void CompilePTXBody(llvm::Value *blockID, llvm::Value *tid, int sizeOfArray, llvm::Value * outputAddrInt, 
            llvm::Value * outputAddrDouble, llvm::Value * outputAddrLogical, 
            llvm::Value *inputAddrInt, llvm::Value * inputAddrDouble, llvm::Value *inputAddrLogical) {
        llvm::Value * loopIndexValue = loopIndex();
        
        std::vector<llvm::Value *> loopIndexArray;
        loopIndexArray.push_back(loopIndexValue);
        for(size_t i = 0; i < trace->nodes.size(); i++) {
            IRNode & n = trace->nodes[i];
            switch(n.op) {
                case IROpCode::load: {
                    llvm::Value * p;
                    if(n.in.isLogical()) {
                        /*
                        int size = ((Logical&)n.in).length*sizeof(Logical::Element);
                        cudaMalloc((void**)&p, size);
                        cudaMemcpy(p, ((Logical&)n.in).v(), size, cudaMemcpyHostToDevice);
                        */
                        
                        p = Loader(inputAddrLogical, cT.parameters);

                    }
                    else if(n.in.isInteger()) {
                        //p = ((Integer&)n.in).v();
                        /*
                        int size = ((Integer&)n.in).length*sizeof(Integer::Element);
                        cudaMalloc((void**)&p, size);
                        cudaMemcpy(p, ((Integer&)n.in).v(), size, cudaMemcpyHostToDevice);
                        */
                        p = Loader(inputAddrInt, cT.parameters);
                    }
                    else if(n.in.isDouble()) {
                        //p = ((Double&)n.in).v();
                        /*
                        int size = ((Double&)n.in).length*sizeof(Double::Element);
                        cudaError_t error = cudaMalloc((void**)&p, size);
                        cudaMemcpy(p, ((Double&)n.in).v(), size, cudaMemcpyHostToDevice);
                        */
                        p = Loader(inputAddrDouble, cT.parameters);
                    }
                    else
                        _error("unsupported type");
                    
                    llvm::Type * t = getType(n.type);
                    
                    llvm::Value * elementAddr = B->CreateGEP(p, loopIndexArray);
                    inputGPUAddr.push_back(elementAddr);
                    values[i] = B->CreateLoad(elementAddr);
                    cT.parameters++;
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
                case IROpCode::cumsum:
                    cT.parameters++;
                    scan = true;
                    void *global;
                    n.group = IRNode::SCAN;
                    switch (n.type) {
                        case Type::Double: {
                            initializeScanBlocks();
                            llvm::GlobalVariable *shared = new llvm::GlobalVariable((*mainModule), llvm::ArrayType::get(llvm::Type::getDoubleTy(*C), numThreads), false, llvm::GlobalValue::ExternalLinkage, 0, "sharedMemory", 0, 0, 3);
                            
                            InitializeSharedDouble(function, shared, tid);
                            cudaMalloc((void**)&global, trace->Size);
                            cudaMemcpy(global, ((Double&)n.in).v(), trace->Size, cudaMemcpyHostToDevice);
                            
                            
                            std::vector<llvm::Value *> indices;
                            llvm::AllocaInst * alloca = B->CreateAlloca(intType);
                            B->CreateStore(loopIndex(), alloca);
                            indices.push_back(B->CreateLoad(alloca));
                            
                            llvm::Type * t = getType(n.type);
                            
                            llvm::Value * vector = ConstantPointer(global, t);
                            
                            
                            llvm::Value *value = B->CreateGEP(vector, indices);
                            llvm::Value *sharedMemoryIndex = GEPGetter(shared, tid);

                            B->CreateStore(values[n.unary.a], sharedMemoryIndex);
                            
                            B->CreateCall(Sync);
                            
                            
                            end = returnBlock;
                            
                            llvm::AllocaInst *returnVal = B->CreateAlloca(doubleType);
                            B->CreateStore(scanBlock(shared, tid), value);
                            B->CreateCall(Sync);
                            
                            //store the partial result from each block i to block_results[i]
                            
                            
                            values[i] = B->CreateLoad(returnVal);
                            /*
                            
                            B->SetInsertPoint(bodyBlockResult);
                            llvm::Value *blockNum = B->CreateUDiv(loopIndexValue, ConstantInt(numThreads));
                            llvm::Value *blockResultsIndex = GEPGetter(blockResults, blockNum);
                            llvm::Value *partialResult = GEPGetter(global, loopIndexValue);
                            B->CreateStore(B->CreateLoad(partialResult), blockResultsIndex);
                            B->CreateCall(Sync);
                            
                            
                            B->SetInsertPoint(origEnd);
                            
                            B->SetInsertPoint(scanBlockBody);
                            scanBlocksFunc(blockResults, loopIndexValue);
                            B->SetInsertPoint(scanBlockEnd);
                            B->CreateCall(Sync);
                            
                            B->SetInsertPoint(scanBlockApplyBody);
                            llvm::Value * blockPartial = GEPGetter(blockResults, B->CreateSub(blockNum, ConstantInt(1)));
                            val = B->CreateFAdd(blockPartial, val);
                            B->SetInsertPoint(scanBlockApplyEnd);
                            B->CreateCall(Sync);
                            values[i] = B->CreateLoad(val);
                            
                            body = scanBodyReplacement;
                            B->SetInsertPoint(scanBodyReplacement);
                            */
                            break;
                        }
                        case Type::Integer: {
                            
                        }
                        default:
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
                           
                            
                            llvm::FunctionType *placeHolder = llvm::FunctionType::get(llvm::Type::getFloatTy(*C), llvm::Type::getFloatTy(*C), false);
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
                            
                            llvm::FunctionType *placeHolder = llvm::FunctionType::get(llvm::Type::getFloatTy(*C), llvm::Type::getFloatTy(*C), false);
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
                            
                            llvm::FunctionType *placeHolder = llvm::FunctionType::get(llvm::Type::getDoubleTy(*C), llvm::Type::getDoubleTy(*C), false);
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
                            llvm::FunctionType *placeHolder = llvm::FunctionType::get(llvm::Type::getInt64Ty(*C), llvm::Type::getInt64Ty(*C), false);
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
                            reductionBlocksInitialize();
                                
                            Synchronize(function, tid);

                            InitializeSharedDouble(function, shared, tid);
                            
                            llvm::AllocaInst *Alloca = CreateEntryBlockAllocaDouble(function, "mySum");
                            
                            
                            
                            B->SetInsertPoint(body);
                            
                            
                            B->CreateStore((B->CreateFAdd(values[n.unary.a], B->CreateLoad(Alloca))), Alloca);
                            
                           
                            B->SetInsertPoint(origEnd);
                            
                            
                            llvm::Value *sharedMemIndex = GEPGetter(shared, tid);
                            B->CreateStore(B->CreateLoad(Alloca), sharedMemIndex);
                            
                            
                            B->CreateCall(Sync);
                            

                            
                            int current = 512;
                            for(int ind = 0 ; ind < 3; ind++) {
                                if(numThreads >= current) {
                                    B->SetInsertPoint(bodyBlocks[ind]);
                                
                                    llvm::Value *sharedMemIndexT = GEPGetter(shared, B->CreateAdd(tid, ConstantInt(current/2)));

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
                                    
                                    llvm::Value *sharedMemIndexT = GEPGetter(shared, B->CreateAdd(tid, ConstantInt(current/2)));
                                    
                                    
                                    B->CreateStore(B->CreateFAdd(B->CreateLoad(sharedMemIndex), B->CreateLoad(sharedMemIndexT)), sharedMemIndex, true);
                                    B->CreateCall(Sync);
                                }

                            }
                            
                            B->SetInsertPoint(endBlocks[3]);
                             
                            
                            B->CreateCall(Sync);
                            
                            
                            B->SetInsertPoint(bodyFinal);
                            
                            llvm::Value *output = GEPGetter(shared, ConstantInt(0));
                            
                            values[i] = B->CreateLoad(output); 
                            
                            end = returnBlock;
                            
                            B->SetInsertPoint(bodyFinal);
                            break;
                        }
                        case Type::Integer: {
                            //set to the top of the function
                            
                            llvm::GlobalVariable *shared = new llvm::GlobalVariable((*mainModule), llvm::ArrayType::get(llvm::Type::getInt64Ty(*C), sizeOfArray), false, llvm::GlobalValue::ExternalLinkage, 0, "sharedMemory", 0, 0, 3);
                            //set to the top of the function
                            reductionBlocksInitialize();
                            
                            Synchronize(function, tid);
                            
                            InitializeSharedInteger(function, shared, tid);
                            
                            llvm::AllocaInst *Alloca = CreateEntryBlockAllocaInteger(function, "mySum");
                            
                            
                            
                            B->SetInsertPoint(body);
                            
                            
                            B->CreateStore((B->CreateAdd(values[n.unary.a], B->CreateLoad(Alloca))), Alloca);
                            
                            
                            B->SetInsertPoint(origEnd);
                            
                            
                            llvm::Value *sharedMemIndex = GEPGetter(shared, tid);
                            B->CreateStore(B->CreateLoad(Alloca), sharedMemIndex);
                            
                            
                            B->CreateCall(Sync);
                            
                            
                            
                            int current = 512;
                            for(int ind = 0 ; ind < 3; ind++) {
                                if(numThreads >= current) {
                                    B->SetInsertPoint(bodyBlocks[ind]);
                                    
                                    llvm::Value *sharedMemIndexT = GEPGetter(shared, B->CreateAdd(tid, ConstantInt(current/2)));
                                    
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
                                    
                                    llvm::Value *sharedMemIndexT = GEPGetter(shared, B->CreateAdd(tid, ConstantInt(current/2)));
                                    
                                    
                                    B->CreateStore(B->CreateAdd(B->CreateLoad(sharedMemIndex), B->CreateLoad(sharedMemIndexT)), sharedMemIndex, true);
                                    B->CreateCall(Sync);
                                }
                                
                            }
                            
                            B->SetInsertPoint(endBlocks[3]);
                            
                            
                            B->CreateCall(Sync);
                            
                            
                            B->SetInsertPoint(bodyFinal);
                            
                            llvm::Value *output = GEPGetter(shared, ConstantInt(0));
                            
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
                        /*
                        int size = ((Logical&)n.in).length*sizeof(Logical::Element);
                        cudaMalloc((void**)&p, size);
                        cudaMemcpy(p, ((Logical&)n.in).v(), size, cudaMemcpyHostToDevice);
                        */
                        p = Loader(inputAddrLogical, cT.parameters);
                    }
                    else if(n.in.isInteger()) {
                        /*
                        int size = ((Integer&)n.in).length*sizeof(Integer::Element);
                        cudaMalloc((void**)&p, size);
                        cudaMemcpy(p, ((Integer&)n.in).v(), size, cudaMemcpyHostToDevice);
                        */
                        p = Loader(inputAddrInt, cT.parameters);
                    }
                    else if(n.in.isDouble()) {
                        /*
                        int size = ((Double&)n.in).length*sizeof(Double::Element);
                        cudaMalloc((void**)&p, size);
                        cudaMemcpy(p, ((Double&)n.in).v(), size, cudaMemcpyHostToDevice);
                        */
                        p = Loader(inputAddrDouble, cT.parameters);
                    }
                    else
                        _error("unsupported type");
                    
                    
                    std::vector<llvm::Value *> indices;
                    indices.push_back(values[n.unary.a]);
                    
                    llvm::Type * t = getType(n.type);
                    
                   // llvm::Value * vector = ConstantPointer(p, t);
                    
                    
                    llvm::Value *value = B->CreateGEP(p, indices);
                    values[i] = B->CreateLoad(value);
                    
                    cT.parameters++;
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
                    llvm::Value * vector;
               
                    if(n.type == Type::Double) {
                        n.out = Double(length);

                        //int size = length*sizeof(Double::Element);
                        //cudaError_t error = cudaMalloc((void**)&p, size);
						//Grab the addresses and save them because we'll access them to put the output into
						//llvm::Type * t = getType(n.type);
						//llvm::Value * vector = ConstantPointer(p,t);
                        vector = Loader(outputAddrLogical, ct.outputCount);

						B->CreateStore(values[i], B->CreateGEP(vector, loopIndexArray));
						
                    } else if(n.type == Type::Integer) {
                        n.out = Integer(length);
                        //int size = length*sizeof(Integer::Element);
                        //cudaMalloc((void**)&p, size);
                    
						//Grab the addresses and save them because we'll access them to put the output into
						//llvm::Type * t = getType(n.type);
						//llvm::Value * vector = ConstantPointer(p,t);
                        vector = Loader(outputAddrInt, ct.outputCount);
						B->CreateStore(values[i], B->CreateGEP(vector, loopIndexArray));
						
                    } else if(n.type == Type::Logical) {
                        n.out = Logical(length);

                        int size = length*sizeof(Logical::Element);
                        //cudaMalloc((void**)&p, size);
						// Convert to 8 bit logical
						llvm::Value * temp8 = B->CreateSExt(values[i],logicalType8);
						
						//Grab the addresses and save them because we'll access them to put the output into
						//llvm::Type * t = logicalType8;
						//llvm::Value * vector = ConstantPointer(p,t);
						vector = Loader(outputAddrLogical, ct.outputCount);
                        B->CreateStore(temp8, B->CreateGEP(vector, loopIndexArray));
						
                    } else {
                        _error("Unknown type in initialize outputs");
                    }
                    
                    outputGPU.push_back(vector);
                    ct.outputCount++;
                }
                else if (n.group == IRNode::FOLD) {
                    int64_t length = 1;
                    llvm::Value * vector;
                    
                    if(n.type == Type::Double) {
                        n.out = Double(length);
                        
                        int size = ((Double&)n.in).length*sizeof(Double::Element);

                        vector = Loader(outputAddrDouble, ct.outputCount);

                        B->CreateStore(values[i], B->CreateGEP(vector, blockID));
						
                    } else if(n.type == Type::Integer) {
                        n.out = Integer(length);
                        int size = ((Integer&)n.in).length*sizeof(Integer::Element);
						
						//Grab the addresses and save them because we'll access them to put the output into
						vector = Loader(outputAddrInt, ct.outputCount);

                        B->CreateStore(values[i], B->CreateGEP(vector, blockID));
                        
                    } else if(n.type == Type::Logical) {
                        n.out = Logical(length);
                        
                        int size = ((Logical&)n.in).length*sizeof(Logical::Element);
                        // Convert to 8 bit logical
						llvm::Value * temp8 = B->CreateSExt(values[i],logicalType8);
						
						//Grab the addresses and save them because we'll access them to put the output into
						vector = Loader(outputAddrInt, ct.outputCount);

                        B->CreateStore(temp8, B->CreateGEP(vector, blockID));
                    } else {
                        _error("Unknown type in initialize outputs");
                    }
                    
                    //Grab the addresses and save them because we'll access them to put the output into
                    ct.outputCount++;
                    outputGPU.push_back(vector);
                    B->SetInsertPoint(body);
                }
                else if (n.group == IRNode::SCAN) {
                    int64_t length = n.outShape.length;
                    llvm::Value * vector;
                    
                    if(n.type == Type::Double) {
                        n.out = Double(length);
                        
                        int size = length*sizeof(Double::Element);
                        /*
                        cudaError_t error = cudaMalloc((void**)&p, size);
						//Grab the addresses and save them because we'll access them to put the output into
						llvm::Type * t = getType(n.type);
						llvm::Value * vector = ConstantPointer(p,t);
						B->CreateStore(values[i], B->CreateGEP(vector, loopIndexArray));
                        */
						vector = Loader(outputAddrDouble, ct.outputCount);

                        B->CreateStore(values[i], B->CreateGEP(vector, loopIndexArray));
                    } else if(n.type == Type::Integer) {
                        n.out = Integer(length);
                        int size = length*sizeof(Integer::Element);
						
                        vector = Loader(outputAddrInt, ct.outputCount);

                        B->CreateStore(values[i], B->CreateGEP(vector, loopIndexArray));
                    } else if(n.type == Type::Logical) {
                        n.out = Logical(length);
                        
                        int size = length*sizeof(Logical::Element);
						// Convert to 8 bit logical
						llvm::Value * temp8 = B->CreateSExt(values[i],logicalType8);


                        vector = Loader(outputAddrLogical, ct.outputCount);

                        B->CreateStore(temp8, B->CreateGEP(vector, loopIndexArray));
						
                    } else {
                        _error("Unknown type in initialize outputs");
                    }
                    ct.outputCount++;
                    outputGPU.push_back(vector);
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
    
    
    void PTXExecute() {

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
        unsigned nblocks = outputReductionSize;
        
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
        cudaThreadSynchronize();
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

CompiledTrace PTXCompile(Thread & thread, Trace * tr) {
        TraceLLVMCompiler c(&thread, tr);
        c.cT.parameters = 0;
        c.cT.outputCount = 0;
        c.GeneratePTXIndexFunction(); 
        c.cT.F = c.function; 
        return c.cT;
}

void PTXRun(Thread & thread, Trace * tr, CompiledTrace result) {
        TraceLLVMCompiler c(&thread, tr);
        c.function = result.function;
        c.cT = result;
        c.GeneratePTXKernelFunction(); 
        return;
}

void Trace::JIT(Thread & thread) {
    std::cout << toString(thread) << "\n";
    cuInit(0);
    nvvmInit();
    
    CompiledTrace result = PTXCompile(thread,this);
    PTXRun(thread, this, result);
    //c.Compile();
    //c.Execute();
}

#endif
