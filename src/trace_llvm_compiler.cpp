
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
    
    llvm::Constant * Size;
    llvm::Function * function;
    llvm::Function *KernelFunc;
    llvm::BasicBlock * entry;
    llvm::Value * loopIndexAddr;
    llvm::Value * nThreads;
    llvm::Type * doubleType;
    llvm::Type * intType;
    llvm::IRBuilder<> * B;
    llvm::Module * mainModule;
    
    llvm::LLVMContext * C;
    llvm::ExecutionEngine * EE;
    llvm::FunctionPassManager * FPM;
    
    
    std::vector<llvm::Value *> values;
    std::vector<void *> outputGPU;
    std::vector<llvm::Value *> outputGPUAddr;
    std::vector<void *> inputGPU;
    std::vector<llvm::Value *> inputGPUAddr;
    
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
    
    
    
    int isPowerOfTwo (unsigned int x)
    {
        return ((x != 0) && ((x & (~x + 1)) == x));
    }

    
    void GenerateIndexFunction() {
        intType = llvm::Type::getInt64Ty(*C);
        doubleType = llvm::Type::getDoubleTy(*C);
        
        
        llvm::Constant *cons = mainModule->getOrInsertFunction("indexFunc", llvm::Type::getVoidTy(*C), intType, NULL);
        function = llvm::cast<llvm::Function>(cons);
        function->setCallingConv(llvm::CallingConv::C);
        
        llvm::Function::arg_iterator args = function->arg_begin();
        llvm::Value* index = args++;
        index->setName("index");
        llvm::Value* tid = args++;
        //tid->setName("tid");
        llvm::Value* i = args++;
        i->setName("i");
        llvm::Value* gridSize = args++;
       // gridSize->setName("gridSize");
        llvm::PointerType* shared_Pointer = llvm::PointerType::get(llvm::Type::getDoubleTy(*C), 3);
        
        int sizeOfArray = 64;
        if (numThreads > 32)
            sizeOfArray = numThreads;
        
        llvm::GlobalVariable *sharedMem = new llvm::GlobalVariable((*mainModule), llvm::ArrayType::get(llvm::Type::getDoubleTy(*C), sizeOfArray), false, llvm::GlobalValue::InternalLinkage, 0, "sharedMemory", 0, 0, 3);
        
        
        
        
        
        
        entry = llvm::BasicBlock::Create(*C,"entry",function);
        B = new llvm::IRBuilder<>(*C);
        B->SetInsertPoint(entry);
        
        
        Size = ConstantInt(trace->Size);
        bool pow2 = false;
        
        
        if (isPowerOfTwo(trace->Size))
            pow2 = true;
        
        loopIndexAddr = B->CreateAlloca(intType);
        B->CreateStore(index, loopIndexAddr);
        
        /*
         * We need to loop this.
         * Loop it so that a index touches all that it needs to touch, refer to RG code how to loop.
         */
        
        llvm::BasicBlock * cond = createAndInsertBB("cond");
        llvm::BasicBlock * body = createAndInsertBB("body");
        llvm::BasicBlock * end = createAndInsertBB("end");
        
        B->CreateBr(cond);
        B->SetInsertPoint(cond);
        llvm::Value * c = B->CreateICmpULT(loopIndex(), Size);
        B->CreateCondBr(c,body,end);
        
        B->SetInsertPoint(body);
        CompileBody(pow2, tid, sharedMem, i, gridSize);
        B->CreateStore(B->CreateAdd(loopIndex(), ConstantInt(numBlock * numThreads)),loopIndexAddr);
        B->CreateBr(cond);
        
        B->SetInsertPoint(end);
        B->CreateRetVoid();
        
        
        llvm::verifyFunction(*function);
        
        //Gavin changes
        //end Gavin changes
        
        //if (thread->state.verbose)
            mainModule->dump();
        
        
        
        
        
        FPM->run(*function);
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
        
     
        /* create the following body for the kernel function:
         {
         unsigned id = blockDim.x * blockIdx.x + threadIdx.x;
         if (id < len) {
                IndexFunc(id);
            }
         }
         */
        llvm::BasicBlock *EntryBB = llvm::BasicBlock::Create(VMContext, "entry", KernelFunc, /*InserBefore*/0);
        
        
        B = new llvm::IRBuilder<>(VMContext);
        B->SetInsertPoint(EntryBB);
        
        // id = blockDim.x * blockIdx.x + threadIdx.x
        llvm::Value *TidXRead = B->CreateSExt(B->CreateCall(TidXFunc, std::vector<llvm::Value *>(),"calltmp"), intType);
        llvm::Value *BlockDimXRead = B->CreateSExt(B->CreateCall(BlockDimXFunc, std::vector<llvm::Value *>(), "calltmp"), intType);
        llvm::Value *BlockIdxXRead = B->CreateSExt(B->CreateCall(BlockIdxXFunc, std::vector<llvm::Value *>(), "calltmp"), intType);
        llvm::Value *Id = B->CreateMul(BlockDimXRead, BlockIdxXRead);
        Id = B->CreateAdd(Id, TidXRead);
        
        
        llvm::Value *Length = Size;
        
        llvm::Value *Tid = TidXRead;
        
        llvm::Value *TempA = B->CreateMul(BlockIdxXRead, ConstantInt(numThreads));
        llvm::Value *Two = ConstantInt(2);
        llvm::Value *TempB = B->CreateMul(TempA, Two);
        llvm::Value *I = B->CreateAdd(TempB, Tid);
        
        llvm::Value *TempC = B->CreateMul(Two, BlockDimXRead);
        llvm::Value *GridSize = B->CreateMul(TempC, ConstantInt(numThreads));
        
        
        
        // if (id < len)
        
        llvm::BasicBlock *EndBlock = llvm::BasicBlock::Create(VMContext, "if.end", 0, 0);
        llvm::BasicBlock *IfBlock = llvm::BasicBlock::Create(VMContext, "if.body", 0, 0);
        
        
        llvm::Value *Cond = B->CreateICmpSLT(Id, Length);
        B->CreateCondBr(Cond, IfBlock, EndBlock);
        KernelFunc->getBasicBlockList().push_back(IfBlock);
        B->SetInsertPoint(IfBlock);
        
        
        std::vector<llvm::Value *> IndexArgs;
        IndexArgs.push_back(Id);
        IndexArgs.push_back(Tid);
        IndexArgs.push_back(I);
        IndexArgs.push_back(GridSize);
        
        //call index function
        B->CreateCall(function, IndexArgs);
        
        
        //complete the function
        
        B->CreateBr(EndBlock);
        KernelFunc->getBasicBlockList().push_back(EndBlock);
        
        
        B->SetInsertPoint(EndBlock);
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
    llvm::Value * ConstantPointer(void * ptr, llvm::Type * typ) {
        llvm::Constant* ci = llvm::ConstantInt::get(intType, (uint64_t)ptr); 
        llvm::Value* cp = llvm::ConstantExpr::getIntToPtr(ci, llvm::PointerType::getUnqual(typ));
        return cp; 
    }
    void Output() {
        //We loaded the addresses sequentially, so we can access them in order via this hack.
        int output = 0;
        for(size_t i = 0; i < trace->nodes.size(); i++) {
            IRNode & n = trace->nodes[i];
            if (n.liveOut) {
                if(n.type == Type::Double) {
                    cudaMemcpy(n.out.p, outputGPU[output], n.out.length * sizeof(Double::Element), cudaMemcpyDeviceToHost);
                } else if(n.type == Type::Integer) {
					cudaMemcpy(n.out.p, outputGPU[output], n.out.length * sizeof(Integer::Element), cudaMemcpyDeviceToHost);
                } else if(n.type == Type::Logical) {
					cudaMemcpy(n.out.p, outputGPU[output], n.out.length * sizeof(Logical::Element), cudaMemcpyDeviceToHost);
				} else {
					_error("Unknown type in initialize outputs");
				}
                output++;
            }
        }
    }
    void CompileBody(bool pow2, llvm::Value *tid, llvm::GlobalVariable *shared, llvm::Value *i, llvm::Value * gridSize) {
        llvm::Value * loopIndexValue = loopIndex();
        
        std::vector<llvm::Value *> loopIndexArray;
        loopIndexArray.push_back(loopIndexValue);
        std::cout << "Trace nodes " << trace->nodes.size() << std::endl;
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
                    llvm::Type * t = getType(n.type);
                    
                    inputGPU.push_back(p);
                    llvm::Value * vector = ConstantPointer(p, t);
                    llvm::Value * elementAddr = B->CreateGEP(vector, loopIndexArray);
                    inputGPUAddr.push_back(elementAddr);
                    values[i] = B->CreateLoad(elementAddr);
                } break;
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
				case IROpCode::mod:
                    switch(n.type) {
                        case Type::Double:
                            values[i] = B->CreateFRem(values[n.binary.a],values[n.binary.b]);
                            break;
                        case Type::Integer:
                            values[i] = B->CreateSRem(values[n.binary.a],values[n.binary.b]);
                            break;
                        default:
                            _error("unsupported type");
                    }					
                    break;
                case IROpCode::sum:
                    switch(n.type) {
                        case Type::Double:
                            values[i] = B->CreateFRem(values[n.binary.a],values[n.binary.b]);
                            break;
                        case Type::Integer:
                            values[i] = B->CreateSRem(values[n.binary.a],values[n.binary.b]);
                            break;
                        default:
                            _error("unsupported type");
                    }					
                    break;
		    case IROpCode::cast:
			{
					_error("cast not yet implemented");
					break;
			/*		Type::Enum output = n.type;
					Type::Enum input = trace->nodes[n.unary.a].type;

					
					 switch(n.type) {
						 case Type::Double:
							 // convert input to double
							 break;
						 case Type::Integer:
							 // convert input to integer
							 break;
						 case Type::Logical:
							 switch(input) {
								 case Type::Double:
									 // check if equal to 0
									 B->CreateFCmpONE(values[trace->nodes[n.unary.a]], 0.0);
									 break;
								 case Type::Integer:
									 // convert input to integer
									 break;
								 case Type::Logical:
									 // convert input to logical (pass thru?)
									 break;
									 
								 default:
									 _error("unsupported type");
							 }
							 break;
							 
					 default:
							 _error("unsupported type");
					 }	
					
			*/		
					/*
                    switch(n.type) {
                        case Type::Double:
                            values[i] = B->CreateFRem(values[n.binary.a],values[n.binary.b]);
                            break;
                        case Type::Integer:
                            values[i] =  B->CreateSRem(values[n.binary.a],values[n.binary.b]);
                            break;
                        default:
                            _error("unsupported type");
                    }*/					
                    } break;
                default:
                    _error("unsupported op");
                    break;
            }
            
            if(n.liveOut) {
                
                /*
                 *Changes need to be made here regarding the intermediate size
                 */
                
                
                
                
				int64_t length = n.outShape.length;
				void * p;
               
                if(n.type == Type::Double) {
                    n.out = Double(length);

                    int size = ((Double&)n.in).length*sizeof(Double::Element);
                    cudaMalloc((void**)&p, size);
                    
                } else if(n.type == Type::Integer) {
					n.out = Integer(length);
                    int size = ((Integer&)n.in).length*sizeof(Integer::Element);
                    cudaMalloc((void**)&p, size);
                    
                } else if(n.type == Type::Logical) {
					n.out = Logical(length);

                    int size = ((Logical&)n.in).length*sizeof(Logical::Element);
                    cudaMalloc((void**)&p, size);
                    
				} else {
					_error("Unknown type in initialize outputs");
				}
                
                //Grab the addresses and save them because we'll access them to put the output into
                llvm::Type * t = getType(n.type);
                llvm::Value * vector = ConstantPointer(p,t);
                llvm::Value * elementAddr = B->CreateGEP(vector, loopIndexArray); 
                B->CreateStore(values[i], elementAddr);
                outputGPU.push_back(p);

                outputGPUAddr.push_back(elementAddr);
            }
        }
    
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
                

        std::cout << "Ptx code: " << PtxBuf << std::endl;
        //Create the threads based on the size
        
        const char *ptxstr = PtxBuf;
        const char *kname = "ker";
        unsigned len = trace->Size;
        //need to get the maxlength
        
        unsigned nthreads = 512;
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
    c.Compile();
    c.Execute();
  //  c.Output();
}

#endif
