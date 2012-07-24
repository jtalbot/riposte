
#include "jit.h"
#include "bc.h"
#include "interpreter.h"
#include "call.h"

DEFINE_ENUM_TO_STRING(TraceOpCode, TRACE_ENUM)

JIT::IRRef JIT::insert(TraceOpCode::Enum op, int64_t i, Type::Enum type) {
	code[pc.i] = 	(IR) { op, {0}, {0}, i, type }; 
	pc.i++;
	return (IRRef) { pc.i-1 };
}

JIT::IRRef JIT::insert(TraceOpCode::Enum op, IRRef a, int64_t i, Type::Enum type) {
	code[pc.i] = 	(IR) { op, a, {0}, i, type }; 
	pc.i++;
	return (IRRef) { pc.i-1 };
}

JIT::IRRef JIT::insert(TraceOpCode::Enum op, IRRef a, Type::Enum type) {
	code[pc.i] = 	(IR) { op, a, {0}, 0, type };
	pc.i++;
	return (IRRef) { pc.i-1 };
}

JIT::IRRef JIT::insert(TraceOpCode::Enum op, IRRef a, IRRef b, Type::Enum type) {
	code[pc.i] = 	(IR) { op, a, b, 0, type };
	pc.i++;
	return (IRRef) { pc.i-1 };
}

JIT::IRRef JIT::read(Thread& thread, int64_t a) {
	
	std::map<int64_t, IRRef>::const_iterator i;
	i = map.find(a);
	if(i != map.end()) {
		return i->second;
	}
	else {
		map[a] = pc;
		OPERAND(operand, a);
		return insert(TraceOpCode::read, a, operand.type);
	}
}

JIT::IRRef JIT::emit(Thread& thread, TraceOpCode::Enum op, IRRef a, IRRef b, int64_t c) {
	Value const& v = OUT(thread, c);
	map[c] = pc;
	return insert(op, a, b, v.type);
}

JIT::IRRef JIT::write(Thread& thread, IRRef a, int64_t c) {
	map[c] = pc;
	return insert(TraceOpCode::mov, a, c, code[a.i].type);
}

void JIT::guardF(Thread& thread, Instruction const* reenter) {
	IRRef p = (IRRef) { pc.i-1 }; 
	Exit e = { map, reenter };
	exits[pc.i] = e;
	insert(TraceOpCode::guardF, p, Type::Promise );
}

void JIT::guardT(Thread& thread, Instruction const* reenter) {
	IRRef p = (IRRef) { pc.i-1 }; 
	Exit e = { map, reenter };
	exits[pc.i] = e;
	insert(TraceOpCode::guardT, p, Type::Promise );
}

JIT::Ptr JIT::end_recording(Thread& thread) {
	assert(recording);
	recording = false;

	// duplicate body, replacing read nodes with phi nodes...
	loopStart = pc;
	size_t n = pc.i;
	for(size_t i = 0; i < n; i++) {
		IR& ir = code[i];
		IRRef a, b;
		switch(ir.op) {
			case TraceOpCode::read: {
				a.i = i; b.i = n+map[ir.i].i;
				map[ir.i] = pc;
				insert(TraceOpCode::phi, a, b, code[i].type);
			} break;
			case TraceOpCode::mov: {
				a.i = n+ir.a.i;
				map[ir.i] = pc;
				insert(TraceOpCode::mov, a, ir.i, code[i].type);
			} break;
			case TraceOpCode::phi:
			case TraceOpCode::jmp: {
				_error("Invalid node");
			} break;
			case TraceOpCode::guardF: 
			case TraceOpCode::guardT: {
				a.i = n+ir.a.i;
				Exit e = { map, exits[i].reenter };
				exits[pc.i] = e;
				insert(ir.op, a, code[i].type);
			} break;
			default: {
				a.i = n+ir.a.i; b.i = n+ir.b.i;
				insert(ir.op, a, b, code[i].type);
			} break;
		}
	}
	insert(TraceOpCode::jmp, loopStart, Type::Promise);
	//dump();
	return compile(thread);
}

void JIT::IR::dump() {
	if(type != Type::Promise)
		std::cout << "(" << Type::toString(type) << ")\t";
	else
		std::cout << "\t\t";

	switch(op) {
		case TraceOpCode::read: {
			if(i <= 0)
				std::cout << "read\tr" << i;
			else
				std::cout << "read\t " << (String)i;
		} break;
		case TraceOpCode::mov: {
			if(i <= 0)
				std::cout << "mov\t " << a.i << "\tr" << i;
			else
				std::cout << "mov\t " << a.i << "\t " << (String)i;
		} break;
		case TraceOpCode::phi: {
			std::cout << "phi\t " << a.i << "\t " << b.i;
		} break;
		case TraceOpCode::jmp:
		case TraceOpCode::guardF:
		case TraceOpCode::guardT: {
			std::cout << TraceOpCode::toString(op) << "\t " << a.i;
		} break;
		default: {
			std::cout << TraceOpCode::toString(op) << "\t " << a.i << "\t " << b.i;
		} break;
	};
}

void JIT::dump() {
	for(size_t i = 0; i < pc.i; i++) {
		std::cout << i << ": ";
		code[i].dump();
		if(i == loopStart.i)
			std::cout << "\t\t<== LOOP";
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

		for(size_t i = 0; i < jit.pc.i; i++) {
			if(i == jit.loopStart.i) {
				builder.CreateBr(LoopStart);
				builder.SetInsertPoint(LoopStart);
			}
			Emit(i, jit.code[i]);
		}
		// insert forware edge on phi nodes
		for(size_t i = 0; i < jit.pc.i; i++) {
			if(jit.code[i].op == TraceOpCode::phi) {
				builder.CreateStore(Load(values[jit.code[i].b.i]), values[jit.code[i].a.i]);
			}
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

	llvm::Type* llvmType(Type::Enum type) {
		switch(type) {
			case Type::Double: return builder.getDoubleTy();
			case Type::Integer: return builder.getInt64Ty();
			case Type::Logical: return builder.getInt8Ty();
			default: _error("Bad type in trace");
		}
	}

	std::string postfix(Type::Enum type1, Type::Enum type2) {
		return postfix(type1) + postfix(type2);
	}

	llvm::Value* Load(llvm::Value* v) {
		if(v->getType()->isPointerTy()) {
			return builder.CreateLoad(v);
		}
		else {
			return v;
		}
	}

	void Emit(size_t i, JIT::IR ir) {

		switch(ir.op) {
			case TraceOpCode::read: 
			{
				llvm::Value* tmp = (ir.i <= 0)
				? CALL1(std::string("loadr_")+postfix(ir.type),
					builder.getInt64(ir.i))
				: CALL1(std::string("loadm_")+postfix(ir.type),
					builder.getInt64(ir.i));
				values[i] = CreateEntryBlockAlloca(tmp->getType());
				builder.CreateStore(tmp, values[i]);
			} break;
			case TraceOpCode::mov:
			{
				values[i] = values[ir.a.i];
			} break;
			case TraceOpCode::phi: 
			{
				values[i] = values[ir.a.i];
			} break;
			case TraceOpCode::jmp: {
				// jmp inserted above
			} break;
			case TraceOpCode::guardT:
			case TraceOpCode::guardF:
			{
				llvm::Value* cond = 
				builder.CreateICmpEQ(
					CALL1(std::string("guard_")+postfix(jit.code[ir.a.i].type), Load(values[ir.a.i])),
					builder.getInt8(ir.op == TraceOpCode::guardT ? -1 : 0));
			       llvm::BasicBlock* next = llvm::BasicBlock::Create(
				       *S->C, "next", function, 0);
			       llvm::BasicBlock* exit = llvm::BasicBlock::Create(
				       *S->C, "exit", function, 0);
			       builder.CreateCondBr(cond, next, exit);
			       builder.SetInsertPoint(exit);
				EmitExit(jit.exits[i]);
			       builder.CreateBr(EndBlock);
			       builder.SetInsertPoint(next); 

			} break;
			default: 
			{
				values[i] = CALL2(std::string(TraceOpCode::toString(ir.op))+"_"+postfix(jit.code[ir.a.i].type, jit.code[ir.b.i].type), Load(values[ir.a.i]), Load(values[ir.b.i])); 
			} break;
		};

	}

	void EmitExit(JIT::Exit const& e) {
		std::map<int64_t, JIT::IRRef>::const_iterator i;
		for(i = e.m.begin(); i != e.m.end(); i++) {
			JIT::IR const& ir = jit.code[i->second.i];
			if(i->first <= 0)
				CALL2(std::string("storer_")+postfix(ir.type),
					builder.getInt64(i->first), Load(values[i->second.i]));
			else
				CALL2(std::string("storem_")+postfix(ir.type),
					builder.getInt64(i->first), Load(values[i->second.i]));
		}
		builder.CreateStore(builder.CreateIntToPtr(builder.getInt64((int64_t)e.reenter), instruction_type), result_var);
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

