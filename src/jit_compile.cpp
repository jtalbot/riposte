#include "jit.h"
#include "interpreter.h"

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

void DebugMarker(char* string) {
    printf("%s\n", string);
}

struct LLVMState {
    llvm::Module * M;
    llvm::LLVMContext * C;
    llvm::ExecutionEngine * EE;
    llvm::FunctionPassManager * FPM;


    llvm::StructType* value_type;
    llvm::StructType* actual_value_type;
    llvm::Type* thread_type;

    llvm::PassManager * PM;

    LLVMState() {
        llvm::InitializeNativeTarget();

        C = &llvm::getGlobalContext();

        llvm::OwningPtr<llvm::MemoryBuffer> buffer;
        llvm::MemoryBuffer::getFile("bin/ops.bc", buffer);
        M = ParseBitcodeFile(buffer.get(), *C);

        std::string err;
        EE = llvm::EngineBuilder(M)
            .setErrorStr(&err)
            .setEngineKind(llvm::EngineKind::JIT)
            .create();
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
        
        // Promote newenvas to registers.
        FPM->add(llvm::createPromoteMemoryToRegisterPass());
        // Also promote aggregates like structs....
        FPM->add(llvm::createScalarReplAggregatesPass());
        
        // Reassociate expressions.
        FPM->add(llvm::createReassociatePass());
        // Provide basic AliasAnalysis support for GVN.
        FPM->add(llvm::createBasicAliasAnalysisPass());
        FPM->add(llvm::createScalarEvolutionAliasAnalysisPass());
        // Eliminate Common SubExpressions.
        FPM->add(llvm::createGVNPass());
        // Simplify the control flow graph (deleting unreachable blocks, etc).
        FPM->add(llvm::createCFGSimplificationPass());
        // Promote newenvas to registers.
        FPM->add(llvm::createPromoteMemoryToRegisterPass());
        // Simplify the control flow graph (deleting unreachable blocks, etc).
        FPM->add(llvm::createAggressiveDCEPass());
        
        // Do simple "peephole" optimizations and bit-twiddling optzns.
        FPM->add(llvm::createInstructionCombiningPass());
        // Simplify the control flow graph (deleting unreachable blocks, etc).
        FPM->add(llvm::createCFGSimplificationPass());
        FPM->add(llvm::createPromoteMemoryToRegisterPass());
      
        FPM->doInitialization();

        PM = new llvm::PassManager();
       
        PM->add(llvm::createLoopIdiomPass()); 
        PM->add(llvm::createLoopInstSimplifyPass()); 
        PM->add(llvm::createLoopDeletionPass());
       
        value_type = llvm::StructType::get( 
            llvm::Type::getInt64Ty(*C), llvm::Type::getInt64Ty(*C), NULL ); 
        actual_value_type = M->getTypeByName("struct.Value");
        thread_type = M->getTypeByName("class.Thread")->getPointerTo();
    }
};

static LLVMState llvmState;

class BuilderBase : public llvm::IRBuilder<>
{
public:
    BuilderBase(llvm::LLVMContext* C) : llvm::IRBuilder<>(*C) {}

    BuilderBase(llvm::BasicBlock* bb, llvm::BasicBlock::iterator i)
        : llvm::IRBuilder<>(bb, i) {}
    
    llvm::Constant* getConstantVector(llvm::Constant* k, size_t width) const {
        if(width > 0) {
            std::vector<llvm::Constant*> v(width, k);
            return llvm::ConstantVector::get(v);
        }
        else {
            return 0;
        }
    }

    llvm::Value* getVector(llvm::Value* k, size_t width) {
        if(width > 0) {
            llvm::VectorType* vt = llvm::VectorType::get(k->getType(), width);
            llvm::Value* out = llvm::UndefValue::get(vt);
            for(size_t i = 0; i < width; i++)
                out = CreateInsertElement(out, k, getInt32(i)); 
            return out;
        }
        else {
            return 0;
        }
    }

    llvm::Constant* getDouble( double d ) {
        return llvm::ConstantFP::get( getDoubleTy(), d );
    }
    
    llvm::Constant* getFloat( float f ) {
        return llvm::ConstantFP::get( getFloatTy(), f );
    }

    static llvm::FunctionType* getFunctionTy( llvm::Type* Result, llvm::Type* Param1 ) {
        llvm::Type* args[] = { Param1 };
        return llvm::FunctionType::get( Result, args, false );
    }
};

class ModuleBuilder : public BuilderBase 
{
public:
    ModuleBuilder(llvm::Module* module) 
        : BuilderBase( &module->getContext() )
        , Module( module ) {}

    ModuleBuilder(llvm::BasicBlock* bb, llvm::BasicBlock::iterator i)
        : BuilderBase(bb, i) 
        , Module( bb->getParent()->getParent() ) {}
   
    llvm::Function* getFunction( std::string const& function ) {
        llvm::Function* f = Module->getFunction(function);
        if(f == 0)
            _error(std::string("Function not found: ") + function);
        return f;
    }

    llvm::Function* getFunction( std::string const& function,
        llvm::Type* arg1, llvm::Type* out ) { 
        
        llvm::Type* args[] = { arg1 };
        llvm::FunctionType* ft = llvm::FunctionType::get( out, args, false );
        return llvm::Function::Create( ft, llvm::Function::ExternalLinkage, function, Module );
    }

    llvm::Function* getFunction( std::string const& function,
        llvm::Type* arg1, llvm::Type* arg2, llvm::Type* out ) { 
        
        llvm::Type* args[] = { arg1, arg2 };
        llvm::FunctionType* ft = llvm::FunctionType::get( out, args, false );
        return llvm::Function::Create( ft, llvm::Function::ExternalLinkage, function, Module );
    }

    llvm::Function* getIntrinsic( llvm::Intrinsic::ID id ) {
        return llvm::Intrinsic::getDeclaration(Module, id);
    }

    llvm::Function* getIntrinsic( llvm::Intrinsic::ID id, llvm::Type*& types ) {
        return llvm::Intrinsic::getDeclaration(Module, id, types);
    }

protected:
    llvm::Module* Module;

};

class FunctionBuilder : public ModuleBuilder
{
public:

    FunctionBuilder( llvm::Function* function )
        : ModuleBuilder( function->getParent() )
        , m_function( function )
        , m_state( function->arg_begin() )
    {
    }

    FunctionBuilder(
        llvm::BasicBlock* bb,
        llvm::BasicBlock::iterator i)
        : ModuleBuilder( bb, i )
        , m_function( bb->getParent() )
        , m_state( m_function->arg_begin() )
    {
    }

    llvm::Function* function() {
        return m_function;
    } 
 
    static FunctionBuilder Appender(llvm::BasicBlock* bb) {    
        return FunctionBuilder( bb, bb->end() );
    }

    FunctionBuilder EntryBlock() {
        return Appender( &m_function->getEntryBlock() );
    }
    
    llvm::AllocaInst* CreateUninitializedAlloca( llvm::Type* type, llvm::Value* length = 0 ) {
        llvm::AllocaInst* r = EntryBlock().CreateAlloca( type, length );
        r->setAlignment(16);
        return r;
    }

    llvm::AllocaInst* CreateInitializedAlloca( llvm::Value* init ) {
        llvm::AllocaInst* r = EntryBlock().CreateAlloca( init->getType() );
        r->setAlignment(16);
        if( !llvm::isa<llvm::UndefValue>(init) ) {
            CreateStore(init, r);
        }
        return r;
    }

    llvm::BasicBlock* CreateBlock( char const* name, llvm::BasicBlock* before = 0 ) {
        return llvm::BasicBlock::Create(Context, name, m_function, before);
    }

    llvm::CallInst* Call(std::string const& F) {
        return Save(CreateCall(getFunction(F), m_state));
    }

    llvm::CallInst* Call(std::string const& F, llvm::Value* A) {
        return Save(CreateCall2(getFunction(F), m_state, A));
    }

    llvm::CallInst* Call(std::string const& F, llvm::Value* A, llvm::Value* B) {
        return Save(CreateCall3(getFunction(F), m_state, A, B));
    }

    llvm::CallInst* Call(std::string const& F, llvm::Value* A, llvm::Value* B, llvm::Value* C) {
        return Save(CreateCall4(getFunction(F), m_state, A, B, C));
    }

    llvm::CallInst* Call(std::string const& F, llvm::Value* A, llvm::Value* B, llvm::Value* C, llvm::Value* D) {
        return Save(CreateCall5(getFunction(F), m_state, A, B, C, D));
    }

    llvm::CallInst* Call(std::string const& F, llvm::Value* A, llvm::Value* B, llvm::Value* C, llvm::Value* D, llvm::Value* E) {
        llvm::Value* args[] = { m_state, A, B, C, D, E };
        return Save(CreateCall(getFunction(F), args));
    }

    llvm::CallInst* Call(std::string const& F, llvm::Value* A, llvm::Value* B, llvm::Value* C, llvm::Value* D, llvm::Value* E, llvm::Value* G, llvm::Value* H, llvm::Value* I) {
        llvm::Value* args[] = { m_state, A, B, C, D, E, G, H, I };
        return Save(CreateCall(getFunction(F), args));
    }

    void InlineCalls() {
        // inline functions
        for(size_t i = 0; i < calls.size(); ++i) {
            llvm::InlineFunctionInfo ifi;
            if( calls[i]->getParent() ) {
                llvm::Function* f = calls[i]->getCalledFunction();
                if(f->hasFnAttr(llvm::Attribute::AlwaysInline))
                    llvm::InlineFunction(calls[i], ifi, true);
            }
        }
    }

    std::vector<llvm::CallInst*> calls;

private:
    
    llvm::CallInst* Save(llvm::CallInst* call) {
        calls.push_back(call);
        return call;
    }

    llvm::Function* m_function;
    llvm::Value* m_state;
    
};

#define MARKER(str) \
    builder.CreateCall(builder.getFunction("MARKER"), builder.CreateConstGEP2_64(builder.CreateGlobalString(str), 0, 0))

#define DUMP(str, i) \
    builder.CreateCall2(builder.getFunction("DUMP"), builder.CreateConstGEP2_64(builder.CreateGlobalString(str), 0, 0), i)

#define DUMPD(str, i) \
    builder.CreateCall2(builder.getFunction("DUMPD"), builder.CreateConstGEP2_64(builder.CreateGlobalString(str), 0, 0), i)

#define DUMPL(str, i) \
    builder.CreateCall2(builder.getFunction("DUMPL"), builder.CreateConstGEP2_64(builder.CreateGlobalString(str), 0, 0), i)

/*

    Not all registers are mutable...
    ...Constants are not mutable
    ...Loads are not mutable

    ...Don't want to make copies in.

    Registers used as loop-carried values, but change their contents

*/


class Register {

private:
    LLVMState* S;
    Type::Enum type;
    llvm::Value* v;
    llvm::Value* l;
    bool initialized;
    std::string name;

    llvm::Type* getType(FunctionBuilder& builder, Type::Enum type) {
        llvm::Type* t;
        switch(type) {
            case Type::Double: t = builder.getDoubleTy()->getPointerTo(); break;
            case Type::Integer: t = builder.getInt64Ty()->getPointerTo(); break;
            case Type::Logical: t = builder.getInt8Ty()->getPointerTo(); break;
            case Type::Character: t = builder.getInt8Ty()->getPointerTo()->getPointerTo(); break;
            case Type::List: t = S->actual_value_type->getPointerTo(); break;
            default: 
                t = S->actual_value_type;
                break;
        }
        return t;
    }

    llvm::Type* getAllocType(FunctionBuilder& builder, Type::Enum type) {
        llvm::Type* t;
        switch(type) {
            case Type::Double: t = builder.getDoubleTy(); break;
            case Type::Integer: t = builder.getInt64Ty(); break;
            case Type::Logical: t = builder.getInt8Ty(); break;
            case Type::Character: t = builder.getInt8Ty()->getPointerTo(); break;
            case Type::List: t = S->actual_value_type; break;
            default: 
                _error("Invalid alloc type");
                break;
        }
        return t;
    }

public:

    size_t index;
    bool onheap;

    Register( LLVMState* S, FunctionBuilder& builder, Type::Enum type, JIT::Shape shape, std::string const& name ) 
        : S(S)
        , type(type)
        , initialized(false)
        , name(name)
    {
        v = builder.CreateInitializedAlloca( llvm::UndefValue::get( getType(builder, type) ) );
        l = 0; 
    }

    bool Initialized() const {
        return initialized;
    }

    void Deinitialize() {
        initialized = false;
    }

    void Initialize(FunctionBuilder& builder, size_t l) {
        if(!Initialized()) {
            llvm::AllocaInst* a = builder.CreateUninitializedAlloca(
                                        getAllocType(builder, type), builder.getInt64(l));
            builder.CreateStore(a, v);
            this->l = builder.EntryBlock().CreateInitializedAlloca(builder.getInt64(l));
            this->initialized = true;
            onheap = false;
        }
    }

    void Initialize(FunctionBuilder& builder, Register& len) {
        if(!Initialized()) {
            l = builder.EntryBlock().CreateInitializedAlloca(builder.getInt64(0));
            llvm::Value* newlen = builder.CreateLoad(len.value(builder)); 
            llvm::Value* newv = 
                builder.Call(std::string("REALLOC_")+Type::toString(type), value(builder), l, newlen);
            builder.CreateStore(newv, v);

            this->initialized = true;
            onheap = true;
        }
    }

    void Duplicate(FunctionBuilder& builder, Register& val, Register& len) {
        if(!Initialized()) {
            Initialize(builder, len);

            int64_t size = 8;
            if(type == Type::Logical)            
                size = 1;
            else if(type == Type::List)
                size = 16;

            llvm::Value* l = builder.CreateLoad(len.value(builder));
            llvm::Value* bytes = builder.CreateMul(l, 
                builder.getInt64(size)); 
            builder.CreateMemCpy(
                value(builder),
                val.value(builder),
                bytes, 16); 
        }
    }

    llvm::Value* value(FunctionBuilder& builder) const {
        return builder.CreateLoad(v);
    }

    llvm::Value* length(FunctionBuilder& builder) const {
        return builder.CreateLoad(l);
    }
    
    void Store(FunctionBuilder& builder, llvm::Value* a) {
        builder.CreateStore(a, v);
        l = builder.EntryBlock().CreateInitializedAlloca(builder.getInt64(0));
    }

    void Set(FunctionBuilder& builder, llvm::Value* a) {
        builder.CreateStore(a, builder.CreateLoad(v));
    }

    void Resize(FunctionBuilder& builder, llvm::Value* newlen) {
        llvm::BasicBlock* ifbb = builder.CreateBlock("resize");
        llvm::BasicBlock* elsebb = builder.CreateBlock("afterResize");
       
        llvm::Value* cond = builder.CreateICmpSGT(newlen, length(builder));
        builder.CreateCondBr(cond, ifbb, elsebb);
        
        builder.SetInsertPoint(ifbb);
        llvm::CallInst* newv = 
            (llvm::CallInst*)
                builder.Call(std::string("REALLOC_")+Type::toString(type), value(builder), l, newlen);
        builder.CreateStore(newv, v);

        builder.CreateBr(elsebb);
        builder.SetInsertPoint(elsebb);
    }
    
    llvm::Value* ExtractAlignedVector(FunctionBuilder& builder, llvm::Value* index, size_t width) 
    {
        llvm::Value* a = value(builder);
        if( type == Type::Double 
         || type == Type::Integer 
         || type == Type::Logical 
         || type == Type::Character) {   
            a = builder.CreateInBoundsGEP(a, index);

            llvm::Type* t = llvm::VectorType::get(
                    ((llvm::SequentialType*)a->getType())->getElementType(),
                    width)->getPointerTo();

            a = builder.CreatePointerCast(a, t);
            a = builder.CreateLoad(a);
        }
        else if( type == Type::List ) {
            a = builder.CreateInBoundsGEP(a, index);
        }
        return a;
    }

    void InsertAlignedVector(FunctionBuilder& builder, llvm::Value* a, llvm::Value* index, size_t width) 
    {
        if( type == Type::Double 
         || type == Type::Integer 
         || type == Type::Logical 
         || type == Type::Character ) {      
            llvm::Value* out = value(builder);
            out = builder.CreateInBoundsGEP(out, index);

            llvm::Type* t = llvm::VectorType::get(
                    ((llvm::SequentialType*)a->getType())->getElementType(),
                    width)->getPointerTo();

            out = builder.CreatePointerCast(out, t);

            builder.CreateStore(a, out);
        }
        else if( type == Type::List ) {
            llvm::Value* out = value(builder);
            out = builder.CreateInBoundsGEP(out, index);
            builder.CreateMemCpy(out, a, width*sizeof(List::Element), 16);
        }
        else {
            builder.CreateStore(a, v);
        }
    }
};


struct Fusion {
    LLVMState* S;
    JIT& jit;
    FunctionBuilder builder, headerBuilder;
    std::vector<Register>& registers;

    llvm::BasicBlock* header;
    llvm::BasicBlock* body;
    llvm::BasicBlock* condition;
    llvm::BasicBlock* after;

    llvm::Value* iterator;
    llvm::Value* length;
    llvm::Value* sequenceI;
    llvm::Value* sequenceD;

    llvm::Constant *zerosD, *zerosI, *onesD, *onesI, 
                   *seqD, *seqI, *widthD, *widthI, 
                   *trueL, *falseL;

    size_t width;

    std::map<size_t, llvm::Value*> outs;
    std::map<size_t, JIT::IRRef> outIRs;
    std::map<size_t, llvm::Value*> reductions;


    size_t instructions;

    JIT::IRRef lengthIR;

    Fusion(LLVMState* S, JIT& jit, FunctionBuilder builder, 
            std::vector<Register>& registers, llvm::Value* length, 
            size_t width, JIT::IRRef lengthIR)
        : S(S)
          , jit(jit)
          , builder(builder)
          , headerBuilder(builder)
          , registers(registers)
          , length(length)
          , width(width)
          , lengthIR(lengthIR) {

        zerosD = builder.getConstantVector( builder.getDouble(0), width );       
        zerosI = builder.getConstantVector( builder.getInt64(0), width );       
        
        onesD  = builder.getConstantVector( builder.getDouble(1), width );       
        onesI  = builder.getConstantVector( builder.getInt64(1), width );       
         
        widthD = builder.getConstantVector( builder.getDouble(width), width );       
        widthI = builder.getConstantVector( builder.getInt64(width), width );       

        trueL  = builder.getConstantVector( builder.getInt8(255), width);
        falseL = builder.getConstantVector( builder.getInt8(0), width);

        if(width > 0) {        
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
            case Type::Logical: t = builder.getInt8Ty(); break;
            case Type::List: t = S->actual_value_type; break;
            default: _error("Bad type in trace");
        }
        return t;
    }

    llvm::Type* llvmType(Type::Enum type, size_t width) {
        return llvm::VectorType::get(llvmType(type), width);
    }

    void Open(llvm::BasicBlock* before) {
        header      = builder.CreateBlock( "fusedHeader", before);
        condition   = builder.CreateBlock( "fusedCondition", before);
        body        = builder.CreateBlock( "fusedBody", before);
        after       = builder.CreateBlock( "fusedAfter", before);

        builder.SetInsertPoint(header);
        headerBuilder.SetInsertPoint(header);
        llvm::Value* initial = builder.getInt64(0);

        if(length != 0) {
            builder.SetInsertPoint(condition);
            iterator = builder.CreatePHI(builder.getInt64Ty(), 2);
            sequenceI = builder.CreatePHI(llvmType(Type::Integer, width), 2);
            sequenceD = builder.CreatePHI(llvmType(Type::Double, width), 2);
        }
        else {
            iterator = initial;
            sequenceI = seqI;
            sequenceD = seqD;
        }

        builder.SetInsertPoint(body);
    }

    llvm::Value* RawLoad(FunctionBuilder& builder, size_t ir) 
    {
        size_t reg = jit.code[ir].reg;
        return registers[reg].value(builder);
    }

    llvm::Value* Load(FunctionBuilder& builder, size_t ir) 
    {
        size_t reg = jit.code[ir].reg;
        if(outs.find(reg) != outs.end())
            return outs[reg];

        llvm::Value* a = registers[reg].ExtractAlignedVector(builder, iterator, width);
        return a;
    }

    void Store(FunctionBuilder& builder, llvm::Value* a, size_t reg, llvm::Value* iterator, size_t width) 
    {
        registers[reg].InsertAlignedVector(builder, a, iterator, width);
        outs[reg] = a;
    }

    llvm::Value* SSEIntrinsic(llvm::Intrinsic::ID Op1, llvm::Intrinsic::ID Op2, JIT::IR const& ir) {
        llvm::Value* in = Load(builder, ir.a);
        llvm::Value* out = llvm::UndefValue::get(llvmType(ir.type, width)); 
        uint32_t i = 0;                                                                 
        for(; i < (width-1); i+=2) {                                                    
            llvm::Function* f = builder.getIntrinsic(Op2);
            llvm::Value* v2 = llvm::UndefValue::get(llvmType(jit.code[ir.a].type, 2)); 
            llvm::Value* j0 = builder.CreateExtractElement(in, builder.getInt32(i));    
            llvm::Value* j1 = builder.CreateExtractElement(in, builder.getInt32(i+1));  
            v2 = builder.CreateInsertElement(v2, j0, builder.getInt32(0));              
            v2 = builder.CreateInsertElement(v2, j1, builder.getInt32(1));            
            v2 = builder.CreateCall(f, v2);                                     
            out = builder.CreateInsertElement(out, 
                builder.CreateExtractElement(v2, builder.getInt32(0)), builder.getInt32(i));
            out = builder.CreateInsertElement(out, 
                builder.CreateExtractElement(v2, builder.getInt32(1)), builder.getInt32(i+1));
        }
        for(; i < width; i++) {
            llvm::Function* f = builder.getIntrinsic(Op1);
            llvm::Value* v1 = llvm::UndefValue::get(llvmType(jit.code[ir.a].type, 2)); 
            llvm::Value* j0 = builder.CreateExtractElement(in, builder.getInt32(i));
            v1 = builder.CreateInsertElement(v1, j0, builder.getInt32(0)); 
            v1 = builder.CreateCall(f, v1);
            out = builder.CreateInsertElement(out, 
                builder.CreateExtractElement(v1, builder.getInt32(0)), builder.getInt32(i));
        }
        return out;
    }

    llvm::Value* SSEIntrinsic2(llvm::Intrinsic::ID Op1, llvm::Intrinsic::ID Op2, JIT::IR const& ir) {
        llvm::Value* ina = Load(builder, ir.a);
        llvm::Value* inb = Load(builder, ir.b);
        llvm::Value* out = llvm::UndefValue::get(llvmType(ir.type, width)); 
        uint32_t i = 0;                                                                 
        for(; i < (width-1); i+=2) {                                                    
            llvm::Function* f = builder.getIntrinsic(Op2);
            llvm::Value* v2 = llvm::UndefValue::get(llvmType(jit.code[ir.a].type, 2)); 
            llvm::Value* w2 = llvm::UndefValue::get(llvmType(jit.code[ir.b].type, 2)); 
            llvm::Value* j0 = builder.CreateExtractElement(ina, builder.getInt32(i));    
            llvm::Value* j1 = builder.CreateExtractElement(ina, builder.getInt32(i+1));  
            llvm::Value* k0 = builder.CreateExtractElement(inb, builder.getInt32(i));    
            llvm::Value* k1 = builder.CreateExtractElement(inb, builder.getInt32(i+1));  
            v2 = builder.CreateInsertElement(v2, j0, builder.getInt32(0));              
            v2 = builder.CreateInsertElement(v2, j1, builder.getInt32(1));            
            w2 = builder.CreateInsertElement(w2, k0, builder.getInt32(0));              
            w2 = builder.CreateInsertElement(w2, k1, builder.getInt32(1));            
            v2 = builder.CreateCall2(f, v2, w2);                                     
            out = builder.CreateInsertElement(out, 
                builder.CreateExtractElement(v2, builder.getInt32(0)), builder.getInt32(i));
            out = builder.CreateInsertElement(out, 
                builder.CreateExtractElement(v2, builder.getInt32(1)), builder.getInt32(i+1));
        }
        for(; i < width; i++) {
            llvm::Function* f = builder.getIntrinsic(Op1);
            llvm::Value* v1 = llvm::UndefValue::get(llvmType(jit.code[ir.a].type, 2)); 
            llvm::Value* w1 = llvm::UndefValue::get(llvmType(jit.code[ir.b].type, 2)); 
            llvm::Value* j0 = builder.CreateExtractElement(ina, builder.getInt32(i));
            llvm::Value* k0 = builder.CreateExtractElement(inb, builder.getInt32(i));
            v1 = builder.CreateInsertElement(v1, j0, builder.getInt32(0)); 
            w1 = builder.CreateInsertElement(w1, k0, builder.getInt32(0)); 
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
            llvm::Function* f = builder.getIntrinsic(llvm::Intrinsic::x86_sse41_round_pd);
            llvm::Value* v2 = llvm::UndefValue::get(llvmType(Type::Double, 2)); 
            llvm::Value* j0 = builder.CreateExtractElement(in, builder.getInt32(i));    
            llvm::Value* j1 = builder.CreateExtractElement(in, builder.getInt32(i+1));  
            v2 = builder.CreateInsertElement(v2, j0, builder.getInt32(0));              
            v2 = builder.CreateInsertElement(v2, j1, builder.getInt32(1));            
            v2 = builder.CreateCall3(f, v2, v2, builder.getInt32(k)); 
            out = builder.CreateInsertElement(out, 
                builder.CreateExtractElement(v2, builder.getInt32(0)), builder.getInt32(i));
            out = builder.CreateInsertElement(out, 
                builder.CreateExtractElement(v2, builder.getInt32(1)), builder.getInt32(i+1));
        }
        for(; i < width; i++) {
            llvm::Function* f = builder.getIntrinsic(llvm::Intrinsic::x86_sse41_round_sd);
            llvm::Value* v1 = llvm::UndefValue::get(llvmType(Type::Double, 2)); 
            llvm::Value* j0 = builder.CreateExtractElement(in, builder.getInt32(i));
            v1 = builder.CreateInsertElement(v1, j0, builder.getInt32(0)); 
            v1 = builder.CreateCall3(f, v1, v1, builder.getInt32(k));
            out = builder.CreateInsertElement(out, 
                builder.CreateExtractElement(v1, builder.getInt32(0)), builder.getInt32(i));
        }
        return out;
    }

    llvm::Value* UnaryCall(std::string func, JIT::IR const& ir) {
        llvm::Function* f = builder.getFunction(func);
        if(f == 0) {
            f = builder.getFunction(func, llvmType(jit.code[ir.a].type), llvmType(ir.type));
        }
        llvm::Value* in = Load(builder, ir.a);
        llvm::Value* out = llvm::UndefValue::get(llvmType(ir.type, width));
        for(uint32_t i = 0; i < width; i++) {
            llvm::Value* v = builder.CreateExtractElement(in, builder.getInt32(i));
            v = builder.CreateCall(f, v);
            out = builder.CreateInsertElement(out, v, builder.getInt32(i));
        }
        return out;
    }

    llvm::Value* BinaryCall(std::string func, JIT::IR const& ir) {
        llvm::Function* f = builder.getFunction(func);
        if(f == 0) {
            f = builder.getFunction(func, llvmType(jit.code[ir.a].type), llvmType(jit.code[ir.a].type), llvmType(ir.type));
        }
        llvm::Value* ina = Load(builder, ir.a);
        llvm::Value* inb = Load(builder, ir.b);
        llvm::Value* out = llvm::UndefValue::get(llvmType(ir.type, width));
        for(uint32_t i = 0; i < width; i++) {
            llvm::Value* v = builder.CreateExtractElement(ina, builder.getInt32(i));
            llvm::Value* w = builder.CreateExtractElement(inb, builder.getInt32(i));
            v = builder.CreateCall2(f, v, w);
            out = builder.CreateInsertElement(out, v, builder.getInt32(i));
        }
        return out;
    }

    llvm::Value* ToInt1(llvm::Value* v) {
        return builder.CreateTrunc(v, llvm::VectorType::get(builder.getInt1Ty(), width));
    }

    llvm::Value* ToInt8(llvm::Value* v) {
        return builder.CreateSExt(v, llvm::VectorType::get(builder.getInt8Ty(), width));
    }

    void Emit(size_t index) {

        instructions++;
        JIT::IR ir = jit.code[index];
        size_t reg = ir.reg;

        outIRs[reg] = index;

#define CASE_UNARY(Op, FName, IName) \
    case TraceOpCode::Op: { \
        outs[reg] = (jit.code[ir.a].type == Type::Double)   \
        ? builder.Create##FName(Load(builder, ir.a)) \
        : builder.Create##IName(Load(builder, ir.a));\
    } break

#define CASE_BINARY(Op, FName, IName) \
    case TraceOpCode::Op: { \
        outs[reg] = (jit.code[ir.a].type == Type::Double)   \
        ? builder.Create##FName(Load(builder, ir.a), Load(builder, ir.b)) \
        : builder.Create##IName(Load(builder, ir.a), Load(builder, ir.b));\
    } break

#define CASE_BINARY_ORDINAL(Op, FName, IName) \
    case TraceOpCode::Op: { \
        outs[reg] = ToInt8((jit.code[ir.a].type == Type::Double)   \
        ? builder.Create##FName(Load(builder, ir.a), Load(builder, ir.b)) \
        : builder.Create##IName(Load(builder, ir.a), Load(builder, ir.b)));\
    } break

#define CASE_UNARY_LOGICAL(Op, Name) \
    case TraceOpCode::Op: { \
        outs[reg] = builder.Create##Name(Load(builder, ir.a)); \
    } break

#define CASE_BINARY_LOGICAL(Op, Name) \
    case TraceOpCode::Op: { \
        outs[reg] = builder.Create##Name(Load(builder, ir.a), Load(builder, ir.b)); \
    } break

#define IDENTITY \
    outs[reg] = Load(builder, ir.a);

#define SCALARIZE1(in, Name, A) { \
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
          
            CASE_BINARY_ORDINAL(eq, FCmpOEQ, ICmpEQ);  
            CASE_BINARY_ORDINAL(neq, FCmpONE, ICmpNE);  
            CASE_BINARY_ORDINAL(lt, FCmpOLT, ICmpSLT);  
            CASE_BINARY_ORDINAL(le, FCmpOLE, ICmpSLE);  
            CASE_BINARY_ORDINAL(gt, FCmpOGT, ICmpSGT);  
            CASE_BINARY_ORDINAL(ge, FCmpOGE, ICmpSGE);  
           
            CASE_UNARY_LOGICAL(lnot, Not); 
            CASE_BINARY_LOGICAL(lor, Or); 
            CASE_BINARY_LOGICAL(land, And); 
           
            case TraceOpCode::floor: 
                outs[reg] = SSERound(Load(builder, ir.a), 0x1 /* round down */); 
                break;
            case TraceOpCode::ceiling: 
                outs[reg] = SSERound(Load(builder, ir.a), 0x2 /* round up */); 
                break;
            case TraceOpCode::trunc: 
                outs[reg] = SSERound(Load(builder, ir.a), 0x3 /* round to zero */); 
                break;
            case TraceOpCode::abs:
                // TODO: this could be faster with some bit twidling
                if(ir.type == Type::Double) {
                    llvm::Value* o = builder.CreateFNeg(Load(builder, ir.a));
                    outs[reg] = builder.CreateSelect(
                        builder.CreateFCmpOLT(Load(builder, ir.a), o),
                        o, Load(builder, ir.a));
                }
                else {
                    llvm::Value* o = builder.CreateNeg(Load(builder, ir.a));
                    outs[reg] = builder.CreateSelect(
                        builder.CreateICmpSLT(Load(builder, ir.a), o),
                        o, Load(builder, ir.a));
                } 
            break;
            case TraceOpCode::sign:
                // TODO: make this faster
                outs[reg] = builder.CreateSelect(
                    builder.CreateFCmpOLT(Load(builder, ir.a), zerosD),
                        builder.CreateFNeg(onesD),
                        builder.CreateSelect(
                            builder.CreateFCmpOGT(Load(builder, ir.a), zerosD),
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
                    outs[reg] = SSERound(builder.CreateFDiv(Load(builder, ir.a), Load(builder, ir.b)), 0x1); 
                }
                else {
                    outs[reg] = builder.CreateSDiv(Load(builder, ir.a), Load(builder, ir.b));
                }
            break;

            case TraceOpCode::mod:
            {
                llvm::Value* a = Load(builder, ir.a);
                llvm::Value* b = Load(builder, ir.b);
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
                        builder.CreateICmpSLT(Load(builder, ir.a), Load(builder, ir.b)), Load(builder, ir.a), Load(builder, ir.b));
                }
            break;

            case TraceOpCode::pmax:
                if(ir.type == Type::Double) {
                    outs[reg] = SSEIntrinsic2(llvm::Intrinsic::x86_sse2_max_sd,
                                                llvm::Intrinsic::x86_sse2_max_pd, ir);
                }
                else {
                    outs[reg] = builder.CreateSelect(
                        builder.CreateICmpSGT(Load(builder, ir.a), Load(builder, ir.b)), Load(builder, ir.a), Load(builder, ir.b));
                }
            break;

            case TraceOpCode::ifelse:
                outs[reg] = builder.CreateSelect(ToInt1(Load(builder, ir.a)), Load(builder, ir.b), Load(builder, ir.c));
            break;

            case TraceOpCode::asdouble:
                switch(jit.code[ir.a].type) {
                    case Type::Integer: {
                        llvm::Value* in = Load(builder, ir.a);
                        SCALARIZE1(in, SIToFP, builder.getDoubleTy()); 
                    } break;
                    case Type::Logical: {
                        llvm::Value* in = ToInt1(Load(builder, ir.a));
                        outs[reg] = builder.CreateSelect(in, onesD, zerosD);
                        //SCALARIZE1(in, SIToFP, builder.getDoubleTy()); 
                    } break;
                    case Type::Double: IDENTITY; break;
                    default: _error("Unexpected cast"); break;
                }
                break;
            
            case TraceOpCode::asinteger:
                switch(jit.code[ir.a].type) {
                    case Type::Double: {
                        llvm::Value* in = Load(builder, ir.a);
                        SCALARIZE1(in, FPToSI, builder.getInt64Ty());
                    } break;
                    case Type::Logical: outs[reg] = ToInt1(Load(builder, ir.a));
                    case Type::Integer: IDENTITY; break;
                    default: _error("Unexpected cast"); break;
                }
                break;
            
            case TraceOpCode::aslogical:
                switch(jit.code[ir.a].type) {
                    case Type::Double: outs[reg] = ToInt8(builder.CreateFCmpONE(Load(builder, ir.a), zerosD)); break;
                    case Type::Integer: outs[reg] = ToInt8(builder.CreateICmpNE(Load(builder, ir.a), zerosI)); break;
                    case Type::Logical: IDENTITY; break;
                    default: _error("Unexpected cast"); break;
                }
                break;
            
            case TraceOpCode::phi:
                if(jit.code[ir.a].reg != jit.code[ir.b].reg) { 
                    // if size has changed, resize the destination register
                    if(ir.in.length != ir.out.length) {
                        FunctionBuilder& b = headerBuilder;
                        llvm::Value* newlen = b.CreateLoad(RawLoad(b, ir.in.length));
                        registers[jit.code[ir.a].reg].Resize(b, newlen);
                    }
                    outs[jit.code[ir.a].reg] = Load(builder, ir.b);
                }
                else {
                    // register is the same, do nothing.
                    // if the size changed, another instruction already resized this register
                    instructions--;
                }
                break;
            
            case TraceOpCode::gather1: 
            {
                llvm::Value* v = RawLoad(builder, ir.a);
                llvm::Value* idx = Load(builder, ir.b);

                llvm::Value* j = builder.CreateExtractElement(idx, builder.getInt32(0));
                v = builder.CreateLoad(builder.CreateGEP(v, j));
                
                if(jit.code[ir.a].type == Type::List) {
                    outs[reg] = v;
                }
                else {
                    llvm::Value* r = llvm::UndefValue::get(llvmType(ir.type, width));
                    outs[reg] = builder.CreateInsertElement(r,v,builder.getInt32(0));
                }
            } break;
            case TraceOpCode::gather: 
            {
                llvm::Value* v = RawLoad(builder, ir.a);
                llvm::Value* idx = Load(builder, ir.b);
                // scalarize the gather...
                llvm::Value* r = llvm::UndefValue::get(llvmType(ir.type, width));

                for(uint32_t i = 0; i < width; i++) {
                    llvm::Value* ii = builder.getInt32(i);
                    llvm::Value* j = builder.CreateExtractElement(idx, ii);
                    j = builder.CreateLoad(builder.CreateGEP(v, j));
                    r = builder.CreateInsertElement(r, j, ii);
                }
                outs[reg] = r;
            } break;
            case TraceOpCode::scatter1:
            {
                llvm::Value* v = Load(builder, ir.c);
                llvm::Value* idx = Load(builder, ir.b);
              
                // we've computed the new shape, realloc
                llvm::Value* newlen = builder.CreateLoad(RawLoad(builder, ir.out.length));
                registers[ir.reg].Resize(builder, newlen);
                
                llvm::Type* mt = llvmType(ir.type)->getPointerTo();
                llvm::Value* x = builder.CreatePointerCast(RawLoad(builder, index), mt);
                llvm::Value* ii = builder.CreateExtractElement(idx, builder.getInt32(0));
                
                if(ir.type != Type::List)
                    v = builder.CreateExtractElement(v, builder.getInt32(0));

                builder.CreateStore(v, builder.CreateGEP(x, ii));
            } break;
            case TraceOpCode::scatter:
            {
                llvm::Value* v = Load(builder, ir.c);
                llvm::Value* idx = Load(builder, ir.b);
              
                // we've computed the new shape, realloc
                llvm::Value* newlen = builder.CreateLoad(RawLoad(builder, ir.out.length));
                registers[ir.reg].Resize(builder, newlen);
                
                llvm::Type* mt = llvmType(ir.type)->getPointerTo();
                llvm::Value* x = builder.CreatePointerCast(RawLoad(builder, index), mt);
                for(uint32_t i = 0; i < width; i++) {
                    llvm::Value* ii = builder.getInt32(i);
                    llvm::Value* j = builder.CreateExtractElement(v, ii);
                    ii = builder.CreateExtractElement(idx, ii);
                    builder.CreateStore(j,
                            builder.CreateGEP(x, ii));
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
                */
            } break;

            case TraceOpCode::resize:
            {
                // Initialize to current length from input
                FunctionBuilder& b = headerBuilder;
                llvm::Value* len = b.CreateInitializedAlloca( b.CreateLoad( RawLoad(b, ir.a) ) );
                
                // compute max of indices
                llvm::Value* v = Load(builder, ir.b);
                llvm::Value* nlen = builder.CreateLoad(len);
                for(uint32_t i = 0; i < width; i++) {
                    llvm::Value* q = builder.CreateExtractElement(v, builder.getInt32(i));
                    q = builder.CreateAdd(q, builder.getInt64(1));
                    nlen = builder.CreateSelect(builder.CreateICmpSGT(q, nlen), q, nlen);
                }
                builder.CreateStore(nlen, len);
                reductions[index] = len;
                registers[reg].Set(builder, nlen);
            } break;

            case TraceOpCode::rep:
            {
                _error("rep NYI");
            } break;

            case TraceOpCode::recycle:
            {
                FunctionBuilder& b = headerBuilder;
                if(ir.b == 1) {
                    llvm::Value* e = b.CreateLoad( RawLoad(b, ir.a) );
                    llvm::Value* r = llvm::UndefValue::get(llvmType(ir.type, width));
                    for(int32_t i = 0; i < width; i++) {
                        r = b.CreateInsertElement(r, e, b.getInt32(i)); 
                    }
                    outs[reg] = r;
                }
                else {
                    llvm::Value* v = RawLoad(b, ir.a);
                    llvm::Value* len = b.CreateLoad(RawLoad(b, ir.b));
                    llvm::Value* idx = builder.CreateURem(sequenceI, b.getVector(len, width));
                    // scalarize the broadcast...
                    llvm::Value* r = llvm::UndefValue::get(llvmType(ir.type, width));
                    for(uint32_t i = 0; i < width; i++) {
                        llvm::Value* ii = builder.getInt32(i);
                        llvm::Value* j = builder.CreateExtractElement(idx, ii);
                        j = builder.CreateLoad(builder.CreateGEP(v, j));
                        r = builder.CreateInsertElement(r, j, ii);
                    }
                    outs[reg] = r;
                }
            } break;

            case TraceOpCode::decodena:
            {
                llvm::Value* c = Load(builder, ir.a);
                llvm::Value* na;
     
                switch(jit.code[ir.a].type) {
                    case Type::Double:
                        c = builder.CreateBitCast(Load(builder, ir.a), llvm::VectorType::get(builder.getInt64Ty(), width));
                        na = builder.getConstantVector( builder.getInt64(Double::NAelementInt), width );
                        break; 
                    case Type::Integer: 
                        na = builder.getConstantVector( builder.getInt64(Integer::NAelement), width );
                        break; 
                    case Type::Logical:
                        na = builder.getConstantVector( builder.getInt8(Logical::NAelement), width );
                        break;
                    default: _error("Unexpected decodena type"); break;
                }
                outs[reg] = ToInt8(builder.CreateICmpEQ(c, na));
            } break;

            case TraceOpCode::decodevl:
            {
                outs[reg] = Load(builder, ir.a);
            } break;

            case TraceOpCode::encode:
            {
                llvm::Value* isna = Load(builder, ir.b);
                llvm::Value* c = Load(builder, ir.a);
                llvm::Value* na;
     
                switch(jit.code[ir.a].type) {
                    case Type::Double:
                        na = builder.getConstantVector( builder.getDouble(Double::NAelement), width );
                        break; 
                    case Type::Integer: 
                        na = builder.getConstantVector( builder.getInt64(Integer::NAelement), width );
                        break; 
                    case Type::Logical:
                        na = builder.getConstantVector( builder.getInt8(Logical::NAelement), width );
                        break;
                    default: _error("Unexpected decodena type"); break;
                }
                outs[reg] = builder.CreateSelect(ToInt1(isna), na, c);
                
            } break;

            // Generators
            case TraceOpCode::seq:
            {
                FunctionBuilder& b = headerBuilder;

                // now initialize
                llvm::Value* start = b.CreateLoad(RawLoad(b, ir.a));
                llvm::Value* step = b.CreateLoad(RawLoad(b, ir.b));

                llvm::Value* starts = llvm::UndefValue::get(llvmType(ir.type, width)); 
                llvm::Value* steps = llvm::UndefValue::get(llvmType(ir.type, width)); 
                llvm::Value* bigstep = llvm::UndefValue::get(llvmType(ir.type, width));
                for(size_t i = 0; i < this->width; i++) {
                    starts = b.CreateInsertElement(starts, start, builder.getInt32(i));
                    steps = b.CreateInsertElement(steps, step, builder.getInt32(i));
                    if(ir.type == Type::Integer)
                        bigstep = b.CreateInsertElement(bigstep, b.CreateMul(step,builder.getInt64(width)), builder.getInt32(i));
                    else if(ir.type == Type::Double)
                        bigstep = b.CreateInsertElement(bigstep, b.CreateFMul(step,llvm::ConstantFP::get(builder.getDoubleTy(), width)), builder.getInt32(i));
                    else
                        _error("Unexpected seq type");
                } 
               
                llvm::Value* r, *added;
                if(ir.type == Type::Integer) { 
                    r = b.CreateInitializedAlloca(b.CreateSub(b.CreateAdd(b.CreateMul(seqI, steps), starts), bigstep));
                    added = builder.CreateAdd(builder.CreateLoad(r), bigstep);
                }
                else if(ir.type == Type::Double) {
                    r = b.CreateInitializedAlloca(b.CreateFSub(b.CreateFAdd(b.CreateFMul(seqD, steps), starts), bigstep));
                    added = builder.CreateFAdd(builder.CreateLoad(r), bigstep);
                }
                else
                    _error("Unexpected seq type");
                builder.CreateStore(added, r);
                outs[reg] = added;
            } break;

            case TraceOpCode::random:
            {
                llvm::Value* out = llvm::UndefValue::get(llvmType(ir.type, width));
                for(uint32_t i = 0; i < width; i++) {
                    llvm::Value* v = builder.Call("Riposte_random");
                    out = builder.CreateInsertElement(out, v, builder.getInt32(i));
                }
                outs[reg] = out;
            } break;

            // Reductions
            case TraceOpCode::sum:
            {
                llvm::Value* agg;
                if(ir.type == Type::Double) {
                    agg = headerBuilder.CreateInitializedAlloca(zerosD);
                    builder.CreateStore(builder.CreateFAdd(builder.CreateLoad(agg), Load(builder, ir.a)), agg);
                }
                else {
                    agg = headerBuilder.CreateInitializedAlloca(zerosI);
                    builder.CreateStore(builder.CreateAdd(builder.CreateLoad(agg), Load(builder, ir.a)), agg);
                } 
                reductions[index] = agg;
            } break;
            case TraceOpCode::any:
            {
                llvm::Value* agg;
                agg = headerBuilder.CreateInitializedAlloca(falseL);
                builder.CreateStore(builder.CreateOr(builder.CreateLoad(agg), Load(builder, ir.a)), agg);
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
                        t = builder.CreateAdd(t, builder.CreateExtractElement(a, builder.getInt32(i)));
                    }
                }
                break;
            case TraceOpCode::any:
                a = builder.CreateLoad(a);
                t = builder.getInt8(0);
                for(size_t i = 0; i < width; i++) {
                    t = builder.CreateOr(t, builder.CreateExtractElement(a, builder.getInt32(i)));
                }
                break;
            case TraceOpCode::resize:
                t = builder.CreateLoad(a);
                break;
            default:
                _error("Unsupported reduction");
                break;
        }
        r = builder.CreateInsertElement(r, t, builder.getInt32(0));
        Store(builder, r, ir.reg, builder.getInt64(0), 1); 
    }

    llvm::BasicBlock* Close(JIT::IRRef postIR) {
       
        if(instructions == 0) {
            header = after;
            return after;
        }

        std::map<size_t, llvm::Value*>::const_iterator i;
        for(i = outs.begin(); i != outs.end(); i++) {
            JIT::IRRef ir = outIRs[i->first];
            if(jit.code[ir].use >= postIR)
                Store(builder, i->second, i->first, iterator, width);
        }

        if(length == 0) {
            builder.CreateBr(after);

            builder.SetInsertPoint(condition);
            builder.CreateBr(body);
        }
        else {
            llvm::Value* initial = builder.getInt64(0);
            
            llvm::BasicBlock* bottom = builder.GetInsertBlock();
            llvm::Value* increment = builder.CreateAdd(iterator, builder.getInt64(width));
            
            ((llvm::PHINode*)iterator)->addIncoming(initial, headerBuilder.GetInsertBlock());
            ((llvm::PHINode*)iterator)->addIncoming(increment, bottom);
            llvm::Value* sI = builder.CreateAdd(sequenceI, widthI);
            ((llvm::PHINode*)sequenceI)->addIncoming(seqI, headerBuilder.GetInsertBlock());
            ((llvm::PHINode*)sequenceI)->addIncoming(sI, bottom);
            llvm::Value* sD = builder.CreateFAdd(sequenceD, widthD);
            ((llvm::PHINode*)sequenceD)->addIncoming(seqD, headerBuilder.GetInsertBlock());
            ((llvm::PHINode*)sequenceD)->addIncoming(sD, bottom);
            builder.CreateBr(condition);

            builder.SetInsertPoint(condition);
            llvm::Value* endCond = builder.CreateICmpULT(iterator, length);
            builder.CreateCondBr(endCond, body, after);
        }

        headerBuilder.CreateBr(condition);

        builder.SetInsertPoint(after);
        for(i = reductions.begin(); i != reductions.end(); i++) {
            Reduce(i->second, i->first);
        }

        return after;
    }
};

#define BOXED_ARG(val) \
    builder.CreateExtractValue( builder.CreateExtractValue(val, 0), 0 ), \
    builder.CreateExtractValue( builder.CreateExtractValue(val, 1), 0 )

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

    llvm::FunctionType* function_type;

    llvm::Value* thread_var;
    llvm::Value* result_var;

    FunctionBuilder builder;
    std::vector<Register> registers;
    std::vector<llvm::CallInst*> calls;
 
    TraceCompiler(Thread& thread, JIT& jit, llvm::Function* func) 
        : thread(thread)
        , jit(jit)
        , S(&llvmState)
        , function_type( 
            BuilderBase::getFunctionTy( llvm::Type::getInt64Ty(*S->C), S->thread_type ) )
        , builder( 
            func != 0 
                ? func
                : llvm::Function::Create( function_type, llvm::Function::PrivateLinkage, (std::string("trace_")+intToStr(jit.dest->traceIndex)).c_str(), S->M)
            )
    {
        function = builder.function();

        function->deleteBody();
        function->setLinkage(llvm::Function::PrivateLinkage);
        function->setCallingConv(llvm::CallingConv::Fast);

        llvm::Function::arg_iterator ai = function->arg_begin();
        ai->setName("thread");
        thread_var = ai++;
    }

    llvm::Function* Compile() {
        
        EntryBlock   = builder.CreateBlock("entry");
        HeaderBlock  = builder.CreateBlock("header");
        InnerBlock   = builder.CreateBlock("inner");
        EndBlock     = builder.CreateBlock("end");

        builder.SetInsertPoint(EntryBlock);

        result_var = CreateEntryBlockAlloca(builder.getInt64Ty(), builder.getInt64(1));

        for(size_t i = 0; i < jit.registers.size(); i++) {
            std::string name = std::string("r") + intToStr(i);
            registers.push_back( Register( S, builder, jit.registers[i].type, jit.registers[i].shape, name ) );
        }

        builder.SetInsertPoint(HeaderBlock);

        //MARKER(std::string("Entering trace ") + intToStr(jit.dest->traceIndex));
        Fusion* fusion = 0;
        for(size_t i = 0; i < jit.code.size(); i++) {
            JIT::IR const& ir = jit.code[i];
            if(ir.live && !ir.sunk) {
                if(Fuseable(ir)) {
                    if(!CanFuse(fusion, ir) || !thread.state.doFusion) {
                        EndFusion(fusion, i);
                        fusion = StartFusion(ir);
                    }
                    InitializeRegister(ir);
                    fusion->Emit(i);
                }
                else {
                    EndFusion(fusion, i);
                    InitializeRegister(ir);
                    Emit(ir, i, registers[ir.reg] );
                }
            }
        }
        
        builder.SetInsertPoint(EntryBlock);
        builder.CreateBr(HeaderBlock);

        InnerBlock->eraseFromParent();

        builder.SetInsertPoint(EndBlock);
        builder.CreateRet(builder.CreateLoad(result_var));

        //function->dump();
        S->FPM->run(*function);
        //builder.InlineCalls();
        //S->FPM->run(*function);
        //function->dump();

        return function;
    }

    bool Fuseable(JIT::IR ir) {
        switch(ir.op) {
            case TraceOpCode::phi:
            case TraceOpCode::resize:
            case TraceOpCode::recycle:
            case TraceOpCode::gather:
            case TraceOpCode::gather1:
            case TraceOpCode::scatter:
            case TraceOpCode::scatter1:
            case TraceOpCode::decodena:
            case TraceOpCode::decodevl:
            case TraceOpCode::encode:
            #define CASE(_,...) case TraceOpCode::_:
            TERNARY_BYTECODES(CASE)
            BINARY_BYTECODES(CASE)
            UNARY_FOLD_SCAN_BYTECODES(CASE)
            GENERATOR_BYTECODES(CASE)
                return true; 
                break;
            #undef CASE
            default:
                return false;
                break;
        }
    }

    llvm::Value* value( JIT::IRRef a ) {
        
        return registers[ jit.code[a].reg ].value( builder );
    }

    Fusion* StartFusion(JIT::IR ir) {
        llvm::Value* length = 0;
        int64_t width = std::min((int64_t)4, std::max((int64_t)1, (int64_t)thread.state.specializationLength)); 

        JIT::IRRef len = ir.in.length; 
        if(jit.code[len].op == TraceOpCode::constant &&
            ((Integer const&)jit.constants[jit.code[len].a])[0] <= thread.state.specializationLength) {
            Integer const& v = (Integer const&)jit.constants[jit.code[len].a];
            length = 0;
            width = v[0];
        } 
        else {
            length = builder.CreateLoad(value(ir.in.length));
        }
        Fusion* fusion = new Fusion(S, jit, FunctionBuilder( function ), registers, length, width, len);
        fusion->Open(InnerBlock);
        return fusion;
    }

    void EndFusion(Fusion*& fusion, JIT::IRRef postIR) {
        if(fusion) {
            llvm::BasicBlock* after = fusion->Close(postIR);
            builder.CreateBr(fusion->header);
            builder.SetInsertPoint(after);
            delete fusion;
            fusion = 0;
        }
    }

    bool CanFuse(Fusion*& fusion, JIT::IR ir) {
        return fusion != 0 && fusion->lengthIR == ir.in.length;
    }

    llvm::AllocaInst* CreateEntryBlockAlloca(llvm::Type* type, llvm::Value* size = 0) {
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
            case Type::List: t = S->actual_value_type; break;
            default: t = S->actual_value_type; break;
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
            || type == Type::Character
            || type == Type::List;
    }

    llvm::Value* Unbox(JIT::IRRef index, llvm::Value* v) {

        Type::Enum type = jit.code[index].type;

        if(Unboxed(type)) {
            
            llvm::Value* tmp = CreateEntryBlockAlloca(S->actual_value_type);
            builder.CreateStore(v, tmp);
            
            llvm::Value* length =
                builder.CreateLoad(value(jit.code[index].out.length));

            llvm::Value* r = builder.Call(std::string("UNBOX_")+Type::toString(type), tmp, length);

            //if(type == Type::List) {
                //r = builder.CreatePointerCast(r, S->value_type->getPointerTo());
            //}
            
            // If exit is missing, no need to check guard         
            if(jit.code[index].exit >= 0) {
                llvm::Value* guard = builder.CreateICmpNE(
                    r,
                    llvm::ConstantPointerNull::get((llvm::PointerType*)r->getType()));

                EmitExit(guard, index);
            }

            return r;
        }
        else {
            return v;
        }
    }

    llvm::Value* Box(JIT::IRRef index, bool isSunk) {
        llvm::Value* r = value(index);
        
        // if unboxed type, box
        Type::Enum type = jit.code[index].type;
        if(Unboxed(type)) {
            r = ValueCoerce( 
                    builder.Call(std::string("BOX_")+Type::toString(type),
                        r, builder.CreateLoad(value(jit.code[index].out.length))) );
        }

        return r;
    }

    Register& InitializeRegister(JIT::IR const& ir) {
        Type::Enum type = jit.registers[ir.reg].type;
        JIT::Shape shape = jit.registers[ir.reg].shape;

        Register& r = registers[ir.reg];
        r.index = ir.reg;

        // scatter is special. It initially needs a register that duplicates its input.
        if( ir.op == TraceOpCode::scatter || ir.op == TraceOpCode::scatter1 )
        {
            r.Duplicate(
                builder,
                registers[jit.code[ir.a].reg],
                registers[jit.code[jit.code[ir.a].out.length].reg]);
        }
        else if(Unboxed(type) &&
            !(ir.op == TraceOpCode::constant || ir.op == TraceOpCode::unbox)) 
        {
            if( jit.code[shape.length].op == TraceOpCode::constant ) {
                r.Initialize( builder, ((Integer const&)jit.constants[jit.code[shape.length].a])[0] );
            }
            else {
                r.Initialize( builder, registers[jit.code[shape.length].reg] ); 
            }
        }
        return r;
    }

    llvm::Value* ValueCoerce( llvm::Value* a ) {
        llvm::Value* a0 = builder.CreateExtractValue( a, 0 );
        llvm::Value* a1 = builder.CreateExtractValue( a, 1 );

        llvm::Value* r = llvm::UndefValue::get( S->actual_value_type );
      
        llvm::Type* t0 = S->actual_value_type->getElementType(0);
        llvm::Type* t1 = S->actual_value_type->getElementType(1);

        unsigned int i0[] = { 0, 0 };
        unsigned int i1[] = { 1, 0 };
 
        r = builder.CreateInsertValue( r, a0, i0 ); 
        r = builder.CreateInsertValue( r, a1, i1 ); 

        return r;
    }

    void Emit(JIT::IR ir, size_t index, Register& reg) { 

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
                EmitExit(0, index);
                //builder.CreateUnreachable();
            } break;

            case TraceOpCode::nest:
            {
                ReconstructExit(jit.dest->exits[ir.exit].snapshot, index);
                llvm::CallInst* r = 
                    builder.CreateCall((llvm::Function*)((JIT::Trace*)ir.a)->function, thread_var);
                r->setCallingConv(llvm::CallingConv::Fast);
                DeconstructExit(jit.dest->exits[ir.exit].snapshot, index);
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
                llvm::Value* a = value(ir.a);
                llvm::Value* b = value(ir.b);
                builder.Call("PUSH",
                        BOXED_ARG(a), 
                        builder.getInt64((int64_t)frame.prototype),
                        builder.getInt64((int64_t)frame.returnpc),
                        builder.getInt64((int64_t)frame.returnbase),
                        BOXED_ARG(b),
                        builder.getInt64(frame.dest));
            } break;

            case TraceOpCode::pop:
            {
                builder.Call("POP");
            } break;

            // Load/Store op codes

            case TraceOpCode::constant:
            {
                if(Unboxed(ir.type)) {
                    llvm::Value* r = CreateEntryBlockAlloca(
                        llvmMemoryType(ir.type), builder.getInt64(ir.out.traceLength));
                    llvm::Value* t;
                    // types that are unboxed in the JITed code
                    if(ir.type == Type::Double) {
                        Double const& v = (Double const&)jit.constants[ir.a];
                        for(size_t i = 0; i < ir.out.traceLength; i++) {
                            t = llvm::ConstantFP::get(builder.getDoubleTy(), v[i]);
                            builder.CreateStore(t, builder.CreateConstGEP1_64(r, i));
                        }
                    } else if(ir.type == Type::Integer) {
                        Integer const& v = (Integer const&)jit.constants[ir.a];
                        for(size_t i = 0; i < ir.out.traceLength; i++) {
                            t = builder.getInt64(v[i]);
                            builder.CreateStore(t, builder.CreateConstGEP1_64(r, i));
                        }
                    } else if(ir.type == Type::Logical) {
                        Logical const& v = (Logical const&)jit.constants[ir.a];
                        for(size_t i = 0; i < ir.out.traceLength; i++) {
                            t = builder.getInt8(v[i]);
                            builder.CreateStore(t, builder.CreateConstGEP1_64(r, i));
                        }
                    } else if(ir.type == Type::Character) {
                        Character const& v = (Character const&)jit.constants[ir.a];
                        for(size_t i = 0; i < ir.out.traceLength; i++) {
                            t = builder.CreateIntToPtr(builder.getInt64((int64_t)v[i]), builder.getInt8Ty()->getPointerTo());
                            builder.CreateStore(t, builder.CreateConstGEP1_64(r, i));
                        }
                    } else if(ir.type == Type::List) {
                        List const& v = (List const&)jit.constants[ir.a];
                        for(size_t i = 0; i < ir.out.traceLength; i++) {
                            llvm::Value* t = llvm::ConstantStruct::get(
                                S->value_type, 
                                builder.getInt64(v[i].header),
                                builder.getInt64(v[i].i),
                                NULL);
                            t = ValueCoerce(t);
                            builder.CreateStore(t, builder.CreateConstGEP1_64(r, i));
                        }
                    }
                    reg.Store( builder, r );
                }
                else {
                    // a boxed type
                    
                    llvm::Value* c = llvm::ConstantStruct::get(
                        S->value_type, 
                        builder.getInt64(jit.constants[ir.a].header),
                        builder.getInt64(jit.constants[ir.a].i),
                        NULL);
                    c = ValueCoerce(c);
                    reg.Store( builder, c );
                }
            } break;

            case TraceOpCode::sload: 
            {
                reg.Store( builder, ValueCoerce( builder.Call("SLOAD", builder.getInt64(ir.b)) ) );
            } break;

            case TraceOpCode::load: 
            {
                llvm::Value* a = value(ir.a);
                if(jit.code[ir.a].type == Type::Environment) {
                    reg.Store( builder, ValueCoerce( builder.Call("ELOAD", BOXED_ARG(a), value(ir.b)) ) );
                }
                else if(jit.code[ir.a].type == Type::Function) {
                    reg.Store( builder, ValueCoerce( builder.Call("GET_environment", BOXED_ARG(a)) ) );
                }
                else {
                    _error("Unknown load target");
                }
            } break;

            case TraceOpCode::strip:
            {
                llvm::Value* a = value(ir.a);
                reg.Store( builder, ValueCoerce( builder.Call("GET_strip", BOXED_ARG(a)) ) );
            } break;

            case TraceOpCode::attrget:
            {
                llvm::Value* a = value(ir.a);
                llvm::Value* b = value(ir.b);
                reg.Store( builder, ValueCoerce( builder.Call("GET_attr", BOXED_ARG(a), b) ) );
            } break;

            case TraceOpCode::unbox:
            {
               reg.Store( builder, Unbox( index, value(ir.a) ) ); 
            } break;

            case TraceOpCode::box:
            {
                reg.Store( builder, Box( ir.a, ir.sunk ) );
            } break;
           
            case TraceOpCode::store:
            {
                llvm::Value* a = value(ir.a);
                llvm::Value* c = value(ir.c);
                builder.Call("ESTORE", BOXED_ARG(a), value(ir.b), BOXED_ARG(c)); 
            } break;

            case TraceOpCode::sstore:
            {
                llvm::Value* c = value(ir.c);
                builder.Call("SSTORE", builder.getInt64(ir.b), BOXED_ARG(c));
            } break;
            

            // Environment op codes
             
            case TraceOpCode::curenv: {
                reg.Store( builder, ValueCoerce( builder.Call(std::string("curenv")) ) );
            } break;

            case TraceOpCode::newenv:
            {
                reg.Store( builder, ValueCoerce( builder.Call("NEW_environment") ) );
            } break;

            case TraceOpCode::lenv:
            {
                llvm::Value* a = value(ir.a);
                reg.Store( builder, ValueCoerce( builder.Call("GET_lenv", BOXED_ARG(a)) ) ); 
            } break;
 
            case TraceOpCode::denv:
            {
                llvm::Value* a = value(ir.a);
                reg.Store( builder, ValueCoerce( builder.Call("GET_denv", BOXED_ARG(a)) ) ); 
            } break;

            case TraceOpCode::cenv:
            {
                llvm::Value* a = value(ir.a);
                reg.Store( builder, ValueCoerce( builder.Call("GET_call", BOXED_ARG(a)) ) ); 
            } break;

            // Length op codes

            case TraceOpCode::length: 
            {
                llvm::Value* a = value(ir.a);

                llvm::Value* r;
                if(jit.code[ir.a].type == Type::Any)
                    r = builder.Call( "LENGTH", BOXED_ARG(a) );
                else if(jit.code[ir.a].type == Type::Integer)
                    r = builder.CreateLoad(a);
                else
                    _error("Unknown length usage");
                
                if(jit.code[index].exit >= 0) {
                    llvm::Value* guard = builder.CreateICmpEQ(r, builder.CreateLoad(value(ir.c)));
                    EmitExit(guard, index);
                }

                if(ir.reg > 0)
                    reg.Set( builder, r );
            
            } break;

            case TraceOpCode::rlength: 
            {
                llvm::Value* a = builder.CreateLoad(value(ir.a));
                llvm::Value* b = builder.CreateLoad(value(ir.b));
                llvm::Value* z = builder.getInt64(0);

                //llvm::Value* r = builder.Call( "RLENGTH", a, b );
                
                llvm::Value* r = 
                    builder.CreateSelect(
                        builder.CreateICmpSGT(a, b), a, b);
                r = builder.CreateSelect(
                        builder.CreateOr(builder.CreateICmpEQ(a,z), builder.CreateICmpEQ(b,z)),
                        z, r);


                if(jit.code[index].exit >= 0) {
                    llvm::Value* guard = builder.CreateICmpEQ(r, builder.CreateLoad(value(ir.c)));
                    EmitExit(guard, index);
                }

                if(ir.reg > 0)
                    reg.Set( builder, r );
            
            } break;

            // Guards
             
            case TraceOpCode::geq: {
                // TODO: check the NA mask
                llvm::Value* a = builder.CreateLoad(value(ir.a));
                llvm::Value* b = builder.CreateLoad(value(ir.b));

                llvm::Value* r;
           
                Type::Enum type = jit.code[ir.a].type;
                if( type == Type::Logical
                 || type == Type::Integer )
                    r = builder.CreateICmpEQ(a, b);
                else
                    _error("Undefined geq"); 
                
                EmitExit(r, index);
            } break;

            case TraceOpCode::gproto: {
                llvm::Value* a = value(ir.a);
                llvm::Value* r = builder.CreateICmpEQ(
                    builder.CreatePtrToInt(
                        builder.Call("GET_prototype", BOXED_ARG(a)) 
                        , builder.getInt64Ty())
                    , builder.getInt64(ir.b));
                EmitExit(r, index);
            } break;

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

    void DeconstructExit(JIT::Snapshot const& snapshot, size_t index) {
    
        // emit stack deconstruction code
        for( size_t i = 0; i < snapshot.stack.size(); ++i) {
                builder.Call("POP");
        } 
    }

    void ReconstructExit(JIT::Snapshot const& snapshot, size_t index) {
    
        // emit sunk code
        Fusion* fusion = 0;
        for(size_t i = 0; i < index; i++) {
            JIT::IR const& ir = jit.code[i];
            if(ir.live && ir.sunk) {
                registers[ir.reg].Deinitialize(); // something of a hack to get each exit its own
                                              // initialized register
                InitializeRegister(ir);
                if(Fuseable(ir)) {
                    if(!CanFuse(fusion, ir)) {
                        EndFusion(fusion, i);
                        fusion = StartFusion(ir);
                    }
                    fusion->Emit(i);
                }
                else {
                    EndFusion(fusion, i);
                    Emit(ir, i, registers[ir.reg] );
                }
            }
        }

        // emit sstores
        for(std::map<int64_t, JIT::IRRef>::const_iterator i = snapshot.slots.begin();
                i != snapshot.slots.end(); ++i) {
            llvm::Value* c = value(i->second);
            builder.Call("SSTORE", builder.getInt64(i->first), BOXED_ARG(c));
        }

        // emit stack reconstruction code
        for( size_t i = 0; i < snapshot.stack.size(); ++i) {
                JIT::StackFrame const& frame = snapshot.stack[i];
                llvm::Value* environment = value(frame.environment);
                llvm::Value* env = value(frame.env);
                builder.Call("PUSH",
                        BOXED_ARG(environment), 
                        builder.getInt64((int64_t)frame.prototype),
                        builder.getInt64((int64_t)frame.returnpc),
                        builder.getInt64((int64_t)frame.returnbase),
                        BOXED_ARG(env),
                        builder.getInt64(frame.dest));
        } 

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
    }

    std::map<int64_t, llvm::BasicBlock*> exits;

    void EmitExit(llvm::Value* cond, size_t index) 
    {
        JIT::Trace& exit = jit.dest->exits[jit.code[index].exit];
   
        llvm::BasicBlock* exitBB = 0;

        std::map<int64_t, llvm::BasicBlock*>::const_iterator e = exits.find(jit.code[index].exit);
        if(e != exits.end()) {
            exitBB = e->second;
        }
        else {
            llvm::BasicBlock* currentBB = builder.GetInsertBlock();
            std::string exitStr = std::string("exit") + intToStr(exit.traceIndex) + "_";
            exitBB = llvm::BasicBlock::Create(*S->C, exitStr.c_str(), function, EndBlock);
            builder.SetInsertPoint(exitBB);

            //MARKER(std::string("Taking exit ") + intToStr(exit.traceIndex));

            ReconstructExit(exit.snapshot, index);

            //MARKER(std::string("Done with exit ") + intToStr(index));

            if(exit.Reenter == 0)
                _error("Null reenter");

            if(exit.InScope) {

                if(exit.function == 0) {
                    // create exit stub.
                    llvm::Function* stubfn = 
                        llvm::Function::Create(function_type,
                                llvm::Function::PrivateLinkage,
                                "side", S->M);
                    stubfn->setCallingConv(llvm::CallingConv::Fast);

                    llvm::BasicBlock* stub = 
                        llvm::BasicBlock::Create(*S->C, "stub", stubfn, 0);

                    llvm::IRBuilder<> TmpB(&stubfn->getEntryBlock(),
                            stubfn->getEntryBlock().end());
                    TmpB.SetInsertPoint(stub);

                    TmpB.CreateRet(TmpB.getInt64((int64_t)&exit));
                    exit.function = stubfn;
                }

                llvm::CallInst* r = builder.CreateCall((llvm::Function*)exit.function, thread_var);
                r->setTailCall(true);
                r->setCallingConv(llvm::CallingConv::Fast);
                builder.CreateStore(r, result_var);
            }
            else {
                builder.CreateStore(builder.getInt64(0), result_var);
            }
            builder.CreateBr(EndBlock);
            builder.SetInsertPoint(currentBB);
        }
   
        if(cond) {
            llvm::BasicBlock* nextBB = llvm::BasicBlock::Create(*S->C, "next", function, InnerBlock);
            builder.CreateCondBr(cond, nextBB, exitBB);
            builder.SetInsertPoint(nextBB); 
        }
        else {
            builder.CreateBr(exitBB);
            builder.SetInsertPoint(exitBB);
        }
    }

};

void JIT::compile(Thread& thread) {
    timespec ts = get_time();
    TraceCompiler compiler(thread, *this, (llvm::Function*)dest->function);
    dest->function = compiler.Compile();
    dest->ptr = (Ptr)llvmState.EE->recompileAndRelinkFunction((llvm::Function*)dest->function);
    //print_time_elapsed("Compile time", ts);
}

