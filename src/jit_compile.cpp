#include "jit.h"

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
        // Promote newenvas to registers.
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
        // Promote newenvas to registers.
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
    
    llvm::BasicBlock* header;
    llvm::BasicBlock* condition;
    llvm::BasicBlock* body;
    llvm::BasicBlock* after;

    llvm::Value* iterator;
    llvm::Value* length;
    llvm::Value* sequenceI;
    llvm::Value* sequenceD;

    llvm::Constant *zerosD, *zerosI, *onesD, *onesI, *seqD, *seqI, *widthD, *widthI, *trueL, *falseL;

    size_t width;
    llvm::IRBuilder<> builder;

    std::map<size_t, llvm::Value*> outs;
    std::map<size_t, llvm::Value*> reductions;

    size_t instructions;

    Fusion(JIT& jit, LLVMState* S, llvm::Function* function, std::vector<llvm::Value*> const& values, std::vector<llvm::Value*> const& registers, llvm::Value* length, size_t width)
        : jit(jit)
          , S(S)
          , function(function)
          , values(values)
          , registers(registers)
          , length(length)
          , width(width)
          , builder(*S->C) {
       
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
            
            std::vector<llvm::Constant*> wD;
            for(size_t i = 0; i < this->width; i++) 
                wD.push_back(llvm::ConstantFP::get(builder.getDoubleTy(), this->width));
            widthD = llvm::ConstantVector::get(wD);
            
            std::vector<llvm::Constant*> wI;
            for(size_t i = 0; i < this->width; i++) 
                wI.push_back(builder.getInt64(this->width));
            widthI = llvm::ConstantVector::get(wI);
            
            std::vector<llvm::Constant*> tL;
            for(size_t i = 0; i < this->width; i++) 
                tL.push_back(builder.getInt8(255));
            trueL = llvm::ConstantVector::get(tL);
            
            std::vector<llvm::Constant*> fL;
            for(size_t i = 0; i < this->width; i++) 
                fL.push_back(builder.getInt8(0));
            falseL = llvm::ConstantVector::get(fL);
        }

        instructions = 0;
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
            sequenceI = builder.CreatePHI(llvmType(Type::Integer, width), 2);
            ((llvm::PHINode*)sequenceI)->addIncoming(seqI, header);
            sequenceD = builder.CreatePHI(llvmType(Type::Double, width), 2);
            ((llvm::PHINode*)sequenceD)->addIncoming(seqD, header);
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
        size_t reg = jit.code[ir].reg;
        if(outs.find(reg) != outs.end())
            return outs[reg];

        llvm::Value* a = (reg == 0) ? values[ir] : registers[reg];
        llvm::Type* t = llvm::VectorType::get(
                ((llvm::SequentialType*)a->getType())->getElementType(),
                width)->getPointerTo();
        a = builder.CreateInBoundsGEP(a, iterator);
        a = builder.CreatePointerCast(a, t);
        a = builder.CreateLoad(a);

        if(jit.code[ir].type == Type::Logical) {
            a = builder.CreateICmpEQ(a, trueL);
        }
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
            v2 = builder.CreateCall2(f, v2, builder.getInt32(k)); 
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
            v1 = builder.CreateCall2(f, v1, builder.getInt32(k));
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

#define DUMP(str, i) \
    builder.CreateCall2(S->M->getFunction("DUMP"), builder.CreateConstGEP2_64(builder.CreateGlobalString(str), 0, 0), i)

    void Emit(size_t index) {

        instructions++;
        JIT::IR ir = jit.code[index];
        size_t reg = ir.reg;

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
            {
                llvm::Value* a = Load(ir.a);
                llvm::Value* b = Load(ir.b);
                if(ir.type == Type::Double) {
                    outs[reg] = SSERound(builder.CreateFDiv(a, b), 0x1);
                    outs[reg] = builder.CreateFSub(a, builder.CreateFMul(outs[reg], b));
                } else {
                    outs[reg] = builder.CreateSDiv(a, b);
                    outs[reg] = builder.CreateSub(a, builder.CreateMul(outs[reg], b));
                }
            } break;

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
            
            case TraceOpCode::phi:
                if(jit.code[ir.a].reg != jit.code[ir.b].reg) 
                    outs[jit.code[ir.a].reg] = Load(ir.b);
                break;
            
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
                outs[reg] = r;
            } break;
            case TraceOpCode::scatter:
            {
                llvm::Value* v = Load(ir.c);
                llvm::Value* idx = Load(ir.b);
              
                //DUMP("scatter to ", builder.CreateExtractElement(idx, builder.getInt32(0)));
 
                if(jit.code[ir.a].reg != reg) {
                    // must duplicate (copy from the in register to the out). 
                    // Do this in the fusion header.
                    llvm::IRBuilder<> TmpB(header,
                        header->begin());
                    TmpB.CreateMemCpy(RawLoad(index), RawLoad(ir.a),
                        TmpB.CreateMul(
                            TmpB.CreateLoad(RawLoad(ir.out.length)),
                            TmpB.getInt64(ir.type == Type::Logical ? 1 : 8)), 
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
            case TraceOpCode::brcast:
            {
                if(jit.code[ir.a].out.length == 1) {
                    llvm::Value* e = builder.CreateLoad( RawLoad(ir.a) );
                    llvm::Value* r = llvm::UndefValue::get(llvmType(ir.type, width));
                    for(int32_t i = 0; i < width; i++) {
                        r = builder.CreateInsertElement(r, e, builder.getInt32(i)); 
                    }
                    outs[reg] = r;
                }
                else {
                    llvm::Value* v = RawLoad(ir.a);
                    // scalarize the broadcast...
                    llvm::Value* r = llvm::UndefValue::get(llvmType(ir.type, width));
                    for(uint32_t i = 0; i < width; i++) {
                        llvm::Value* ii = builder.getInt32(i);
                        llvm::Value* j = builder.CreateExtractElement(sequenceI, ii);
                        llvm::Value* idx = builder.CreateSDiv(j, Load(jit.code[ir.a].out.length));
                        j = builder.CreateLoad(builder.CreateGEP(v, j));
                        if(ir.type == Type::Logical)
                            j = builder.CreateICmpEQ(j, builder.getInt8(255));
                        r = builder.CreateInsertElement(r, j, ii);
                    }
                    outs[reg] = r;
                }
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
        Store(r, ir.reg, builder.getInt64(0)); 
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
            llvm::Value* sI = builder.CreateAdd(sequenceI, widthI);
            ((llvm::PHINode*)sequenceI)->addIncoming(sI, body);
            llvm::Value* sD = builder.CreateFAdd(sequenceD, widthD);
            ((llvm::PHINode*)sequenceD)->addIncoming(sD, body);
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

#define BOXED_ARG(val) \
    builder.CreateExtractValue(val, 0), \
    builder.CreateExtractValue(val, 1)

struct TraceCompiler {
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

    llvm::FunctionType* functionTy;
    llvm::Type* thread_type;
    llvm::Type* value_type;
    llvm::Type* actual_value_type;

    llvm::Value* thread_var;
    llvm::Value* result_var;

    std::vector<llvm::Value*> values;
    std::vector<llvm::Value*> registers;
    std::vector<llvm::CallInst*> calls;
 
    TraceCompiler(Thread& thread, JIT& jit) 
        : thread(thread), jit(jit), S(&llvmState), builder(*S->C) 
    {
        registers = std::vector<llvm::Value*>(jit.registers.size(), 0);
        values = std::vector<llvm::Value*>(jit.code.size(), 0);

        thread_type = S->M->getTypeByName("class.Thread")->getPointerTo();
        value_type = llvm::StructType::get(builder.getInt64Ty(), builder.getInt64Ty(), NULL);
        actual_value_type = S->M->getTypeByName("struct.Value");

        std::vector<llvm::Type*> argTys;
        argTys.push_back(thread_type);

        functionTy = llvm::FunctionType::get(
                builder.getInt64Ty(),
                argTys, /*isVarArg=*/false);

    }

    llvm::Function* Compile(llvm::Function* func) {
        
        function = func == 0 ? llvm::Function::Create(functionTy,
                                    llvm::Function::PrivateLinkage,
                                    "trace", S->M) : func;

        function->deleteBody();
        function->setLinkage(llvm::Function::PrivateLinkage);
        function->setCallingConv(llvm::CallingConv::Fast);

        llvm::Function::arg_iterator ai = function->arg_begin();
        ai->setName("thread");
        thread_var = ai++;

        EntryBlock  = llvm::BasicBlock::Create(*S->C, "entry", function, 0);
        HeaderBlock = llvm::BasicBlock::Create(*S->C, "header", function, 0);
        InnerBlock  = llvm::BasicBlock::Create(*S->C, "inner", function, 0);
        EndBlock    = llvm::BasicBlock::Create(*S->C, "end", function, 0);

        builder.SetInsertPoint(EntryBlock);

        result_var = CreateEntryBlockAlloca(builder.getInt64Ty(), builder.getInt64(1));

        builder.SetInsertPoint(HeaderBlock);

        for(size_t i = 0; i < jit.code.size(); i++) {
            if(jit.code[i].live) {
                values[i] = CreateRegister(jit.code[i]);
                Emit(jit.code[i], i, values[i]);
            }
        }
        
        builder.SetInsertPoint(EntryBlock);
        builder.CreateBr(HeaderBlock);

        builder.SetInsertPoint(EndBlock);
        builder.CreateRet(builder.CreateLoad(result_var));

        // inline functions
        /*for(size_t i = 0; i < calls.size(); ++i) {
            llvm::InlineFunctionInfo ifi;
            llvm::InlineFunction(calls[i], ifi, true);
        }*/
        
        S->FPM->run(*function);
        //function->dump();

        return function;
    }

    Fusion* StartFusion(JIT::IR ir) {
        llvm::Value* length = 0;
        size_t width = 2; 

        JIT::IRRef len = ir.in.length; 
        if(jit.code[len].op == TraceOpCode::constant &&
                ((Integer const&)jit.constants[jit.code[len].a])[0] <= SPECIALIZE_LENGTH) {
            length = 0;
            Integer const& v = (Integer const&)jit.constants[jit.code[len].a];
            width = v[0];
        } 
        else {
            length = builder.CreateLoad(values[ir.in.length]);
            width = 2;
        }
        Fusion* fusion = new Fusion(jit, S, function, values, registers, length, width);
        fusion->Open(InnerBlock);
        return fusion;
    }

    void EndFusion(Fusion* fusion) {
        if(fusion) {
            llvm::BasicBlock* after = fusion->Close();
            builder.CreateBr(fusion->header);
            builder.SetInsertPoint(after);
        }
    }

    llvm::AllocaInst* CreateEntryBlockAlloca(llvm::Type* type, llvm::Value* size) {
        llvm::IRBuilder<> TmpB(&function->getEntryBlock(),
                function->getEntryBlock().end());
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
            case Type::Character: t = builder.getInt8Ty()->getPointerTo(); break;
            default: t = value_type; break;
        }
        return t;
    }

    llvm::Type* llvmMemoryType(Type::Enum type, size_t width) {
        return llvm::ArrayType::get(llvmMemoryType(type), width);
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
            
            llvm::Value* tmp = CreateEntryBlockAlloca(value_type, builder.getInt64(0));
            builder.CreateStore(v, tmp);
            
            llvm::Value* length =
                builder.CreateLoad(values[jit.code[index].out.length]);

            llvm::Value* r = Call(std::string("UNBOX_")+Type::toString(type),
                builder.CreatePointerCast(tmp, actual_value_type->getPointerTo()),
                length);
                     
            llvm::Value* guard = builder.CreateICmpNE(
                builder.CreatePtrToInt(r, builder.getInt64Ty()),
                builder.getInt64(0));

            if(jit.exits.find(index) == jit.exits.end())
                _error("Missing exit on unboxing operation");

            EmitExit(guard, jit.exits[index], index);

            builder.CreateMemCpy(values[index], r, 
                builder.CreateMul(length,  builder.getInt64(type == Type::Logical ? 1 : 8)), 8);

            return values[index];
        }
        else {
            return v;
        }
    }

    llvm::Value* Box(JIT::IRRef index) {
        llvm::Value* r = values[index];
        
        // if unboxed type, box
        Type::Enum type = jit.code[index].type;
        if(Unboxed(type)) {
            r = Call(std::string("BOX_")+Type::toString(type),
                    r, builder.CreateLoad(values[jit.code[index].out.length]));
        }

        return r;
    }

    llvm::Value* CreateRegister(JIT::IR const& ir) {
        if(ir.reg != 0 && registers[ir.reg] == 0) {
            JIT::IRRef len = jit.registers[ir.reg].shape.length;
            llvm::Value* length = builder.CreateLoad(values[len]);
            if(jit.code[len].op == TraceOpCode::constant &&
                    ((Integer const&)jit.constants[jit.code[len].a])[0] <= SPECIALIZE_LENGTH) {
                Integer const& v = (Integer const&)jit.constants[jit.code[len].a];
                registers[ir.reg] =
                    CreateEntryBlockAlloca(llvmMemoryType(jit.registers[ir.reg].type), builder.getInt64(v[0]));
            }
            else {
                registers[ir.reg] =
                    Call(std::string("MALLOC_")+Type::toString(jit.registers[ir.reg].type), length);
            }
        }
        return registers[ir.reg]; 
    }

    void Emit(JIT::IR ir, size_t index, llvm::Value*& reg) { 

        switch(ir.op) {

            // Control flow op codes

            case TraceOpCode::loop:
            {
                PhiBlock = llvm::BasicBlock::Create(*S->C, "phis", function, InnerBlock);
                builder.CreateBr(PhiBlock);
                
                LoopStart = llvm::BasicBlock::Create(*S->C, "loop", function, InnerBlock);
                builder.SetInsertPoint(LoopStart);
            }   break;
            
            case TraceOpCode::jmp:
            {
                builder.CreateBr(PhiBlock);

                builder.SetInsertPoint(PhiBlock);
                builder.CreateBr(LoopStart);

            } break;

            case TraceOpCode::exit:
            {
                EmitExit(builder.getInt1(0), jit.exits[index], index);
            } break;

            case TraceOpCode::nest:
            {
                llvm::Value* r = 
                    builder.CreateCall((llvm::Function*)((JIT::Trace*)ir.a)->function, thread_var);

                llvm::Value* cond = 
                    builder.CreateICmpEQ(r, builder.getInt64(0));

                llvm::BasicBlock* next = llvm::BasicBlock::Create(*S->C, "A", function, InnerBlock);
                llvm::BasicBlock* exit = llvm::BasicBlock::Create(*S->C, "B", function, EndBlock);
                
                builder.CreateCondBr(cond, next, exit);

                builder.SetInsertPoint(exit);
                builder.CreateStore(r, result_var);
                builder.CreateBr(EndBlock);
        
                builder.SetInsertPoint(next); 
            } break;

            case TraceOpCode::push:
            {
                JIT::StackFrame frame = jit.frames[ir.c];
                Call("PUSH",
                        BOXED_ARG(values[ir.a]), 
                        builder.getInt64((int64_t)frame.prototype),
                        builder.getInt64((int64_t)frame.returnpc),
                        builder.getInt64((int64_t)frame.returnbase),
                        BOXED_ARG(values[ir.b]),
                        builder.getInt64(frame.dest));
            } break;

            case TraceOpCode::pop:
            {
                Call("POP");
            } break;

            case TraceOpCode::phi:
            {
                reg = values[ir.a];
            } break;

            // Load/Store op codes

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
                    reg = CreateEntryBlockAlloca(llvmMemoryType(ir.type), builder.getInt64(ir.out.traceLength));
                    for(size_t i = 0; i < ir.out.traceLength; i++) {
                        builder.CreateStore(c[i], builder.CreateConstGEP1_64(reg, i));
                    }
                }
                else {
                    // a boxed type
                    reg = CreateEntryBlockAlloca(value_type, builder.getInt64(0));
                    llvm::Value* p = builder.CreatePointerCast(reg, builder.getInt64Ty()->getPointerTo());
                    builder.CreateStore(builder.getInt64(jit.constants[ir.a].header), builder.CreateConstGEP1_64(p, 0));
                    builder.CreateStore(builder.getInt64(jit.constants[ir.a].i), builder.CreateConstGEP1_64(p, 1));
                    reg = builder.CreateLoad(reg);
                }
            } break;

            case TraceOpCode::sload: 
            {
                reg = Unbox(index, Call("SLOAD", builder.getInt64(ir.b)));
            } break;

            case TraceOpCode::load: 
            {
                if(jit.code[ir.a].type == Type::Environment) {
                    reg = Unbox(index, Call("ELOAD",
                                BOXED_ARG(values[ir.a]), values[ir.b]));
                }
                else if(jit.code[ir.a].type == Type::Object) {
                    if(ir.b == 0) {         // strip
                        reg = Unbox(index, Call("GET_strip", BOXED_ARG(values[ir.a]))); 
                    }
                    else {                  // attribute
                        reg = Unbox(index, Call("GET_attr",
                            BOXED_ARG(values[ir.a]), values[ir.b]));
                    }
                }
                else if(jit.code[ir.a].type == Type::Function) {
                    reg = Call("GET_environment", BOXED_ARG(values[ir.a]));
                }
                else {
                    _error("Unknown load target");
                }
            } break;
           
            case TraceOpCode::store:
            {
                Call("ESTORE", BOXED_ARG(values[ir.a]), values[ir.b], BOXED_ARG(Box(ir.c))); 
            } break;

            case TraceOpCode::sstore:
            {
                Call("SSTORE", builder.getInt64(ir.b), BOXED_ARG(Box(ir.c)));
            } break;
            

            // Environment op codes
             
            case TraceOpCode::curenv: {
                reg = Call(std::string("curenv"));
            } break;

            case TraceOpCode::newenv:
            {
                reg = Call("NEW_environment");
            } break;

            case TraceOpCode::lenv:
            {
                reg = Call("GET_lenv", BOXED_ARG(values[ir.a])); 
            } break;
 
            case TraceOpCode::denv:
            {
                reg = Call("GET_denv", BOXED_ARG(values[ir.a])); 
            } break;

            case TraceOpCode::cenv:
            {
                reg = Call("GET_call", BOXED_ARG(values[ir.a])); 
            } break;

            // Length op codes

            case TraceOpCode::slength: 
            {
                builder.CreateStore(Call("SLENGTH", builder.getInt64(ir.b)), reg);
            } break;

            case TraceOpCode::elength: 
            {
                builder.CreateStore(Call("ELENGTH",
                        BOXED_ARG(values[ir.a]), values[ir.b]), reg);
            } break;

            case TraceOpCode::alength: 
            {
                builder.CreateStore(Call("ALENGTH", 
                            BOXED_ARG(values[ir.a]), values[ir.b]), reg);
            } break;

            case TraceOpCode::olength: 
            {
                builder.CreateStore(Call("OLENGTH", BOXED_ARG(values[ir.a])), reg);
            } break;
           
            // Guards
             
            case TraceOpCode::gtrue:
            case TraceOpCode::gfalse: {
                // TODO: check the NA mask
                llvm::Value* r = builder.CreateTrunc(builder.CreateLoad(values[ir.a]), builder.getInt1Ty());
                if(ir.op == TraceOpCode::gfalse)
                    r = builder.CreateNot(r);
                EmitExit(r, jit.exits[index], index);
            } break;

            case TraceOpCode::gproto: {
                llvm::Value* r = builder.CreateICmpEQ(
                    builder.CreatePtrToInt(
                        Call("GET_prototype", BOXED_ARG(values[ir.a])) 
                        , builder.getInt64Ty())
                    , builder.getInt64(ir.b));
                EmitExit(r, jit.exits[index], index);
            } break;

            case TraceOpCode::brcast:
            case TraceOpCode::gather:
            case TraceOpCode::scatter:
            #define CASE(_,...) case TraceOpCode::_:
            TERNARY_BYTECODES(CASE)
            BINARY_BYTECODES(CASE)
            UNARY_FOLD_SCAN_BYTECODES(CASE)
            GENERATOR_BYTECODES(CASE)
            {
                Fusion* fusion = StartFusion(ir);
                fusion->Emit(index);
                EndFusion(fusion);
            } break;
            #undef CASE

            case TraceOpCode::nop:
            {
                // do nothing
            } break;

            default: 
            {
                printf("Unknown op: %s\n", TraceOpCode::toString(ir.op));
                _error("Unknown op in TraceCompiler::Emit");
            } break;
        };
    }

    void EmitExit(llvm::Value* cond, JIT::Exit const& e, size_t index) 
    {
        llvm::BasicBlock* next = llvm::BasicBlock::Create(*S->C, "next", function, InnerBlock);
        llvm::BasicBlock* exit = llvm::BasicBlock::Create(*S->C, "exit", function, EndBlock);
        builder.CreateCondBr(cond, next, exit);
        builder.SetInsertPoint(exit);

        //MARKER(std::string("Taking exit at ") + intToStr(index));
        /*
            else if(jit.code[var.env].type == Type::Object) {
                // dim(x)
                // attrset(x, 'dim', v)
                // can objects be aliased? Yes if they're environments?
                // environments are passed by reference. Everything else is passed by value.
                // pass everything by reference. Copy on write if we modify.
                // environments can be aliased, everything else can't
                if(var.i == 0) {
                    Call("SET_strip",
                            builder.CreateExtractValue(values[var.env], 0), 
                            builder.CreateExtractValue(values[var.env], 1),
                            builder.CreateExtractValue(r, 0), 
                            builder.CreateExtractValue(r, 1));
                }
                else {
                    Call("SET_attr", 
                        builder.CreateExtractValue(values[var.env], 0), 
                        builder.CreateExtractValue(values[var.env], 1),
                        values[var.i], 
                        builder.CreateExtractValue(r, 0), 
                        builder.CreateExtractValue(r, 1));
                }
            }
            else {
                _error("Unknown store target");
            } 
        }
        */
        if(e.reenter.reenter == 0)
            _error("Null reenter");
       
        if(jit.dest->exits[e.index].InScope) {

            if(jit.dest->exits[e.index].function == 0) {
                // create exit stub.
                llvm::Function* stubfn = 
                    llvm::Function::Create(functionTy,
                            llvm::Function::PrivateLinkage,
                            "side", S->M);
                stubfn->setCallingConv(llvm::CallingConv::Fast);

                llvm::BasicBlock* stub = 
                    llvm::BasicBlock::Create(*S->C, "stub", stubfn, 0);

                llvm::IRBuilder<> TmpB(&stubfn->getEntryBlock(),
                        stubfn->getEntryBlock().end());
                TmpB.SetInsertPoint(stub);

                TmpB.CreateRet(TmpB.getInt64((int64_t)&(jit.dest->exits[e.index])));
                jit.dest->exits[e.index].function = stubfn;
            }

            llvm::CallInst* r = builder.CreateCall((llvm::Function*)jit.dest->exits[e.index].function, thread_var);
            r->setTailCall(true);
            builder.CreateStore(r, result_var);
        }
        else {
            builder.CreateStore(builder.getInt64(0), result_var);
        }        
        builder.CreateBr(EndBlock);
        builder.SetInsertPoint(next); 
    }

    llvm::CallInst* Save(llvm::CallInst* ci) {
        calls.push_back(ci);
        return ci;
    }

    llvm::Value* Call(std::string F) {
        return Save(builder.CreateCall(S->M->getFunction(F), thread_var));
    }

    llvm::Value* Call(std::string F, llvm::Value* A) {
        return Save(builder.CreateCall2(S->M->getFunction(F), thread_var, A));
    }

    llvm::Value* Call(std::string F, llvm::Value* A, llvm::Value* B) {
        return Save(builder.CreateCall3(S->M->getFunction(F), thread_var, A, B));
    }

    llvm::Value* Call(std::string F, llvm::Value* A, llvm::Value* B, llvm::Value* C) {
        return Save(builder.CreateCall4(S->M->getFunction(F), thread_var, A, B, C));
    }

    llvm::Value* Call(std::string F, llvm::Value* A, llvm::Value* B, llvm::Value* C, llvm::Value* D) {
        return Save(builder.CreateCall5(S->M->getFunction(F), thread_var, A, B, C, D));
    }

    llvm::Value* Call(std::string F, llvm::Value* A, llvm::Value* B, llvm::Value* C, llvm::Value* D, llvm::Value* E) {
        llvm::Value* args[] = { thread_var, A, B, C, D, E };
        return Save(builder.CreateCall(S->M->getFunction(F), args));
    }

    llvm::Value* Call(std::string F, llvm::Value* A, llvm::Value* B, llvm::Value* C, llvm::Value* D, llvm::Value* E, llvm::Value* G, llvm::Value* H, llvm::Value* I) {
        llvm::Value* args[] = { thread_var, A, B, C, D, E, G, H, I };
        return Save(builder.CreateCall(S->M->getFunction(F), args));
    }

};

void JIT::compile(Thread& thread) {
    timespec a = get_time();
    TraceCompiler compiler(thread, *this);
    dest->function = compiler.Compile((llvm::Function*)dest->function);
    dest->ptr = (Ptr)llvmState.EE->recompileAndRelinkFunction((llvm::Function*)dest->function);
}

