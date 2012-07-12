
#include <string>
#include <sstream>
#include <stdexcept>
#include <string>
#include <map>

#ifndef __STDC_LIMIT_MACROS
#define __STDC_LIMIT_MACROS
#endif

#include "llvm/DerivedTypes.h"
#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/ExecutionEngine/JIT.h"
#include "llvm/LLVMContext.h"
#include "llvm/Module.h"
#include "llvm/Intrinsics.h"
#include "llvm/PassManager.h"
#include "llvm/Analysis/Verifier.h"
#include "llvm/Analysis/Passes.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Support/IRBuilder.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/Assembly/PrintModulePass.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Bitcode/ReaderWriter.h"
#include "llvm/Support/system_error.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Transforms/Utils/BasicInliner.h"

#include "jit.h"
#include "internal.h"
#include "value.h"
#include "type.h"
#include "bc.h"
#include "ops.h"
#include "internal.h"
#include "interpreter.h"
#include "compiler.h"
#include "sse.h"
#include "call.h"

static ByteCode::Enum op1(String const& func) {
	if(func == Strings::add) return ByteCode::pos; 
	if(func == Strings::sub) return ByteCode::neg; 
	if(func == Strings::abs) return ByteCode::abs; 
	if(func == Strings::sign) return ByteCode::sign; 
	if(func == Strings::sqrt) return ByteCode::sqrt; 
	if(func == Strings::floor) return ByteCode::floor; 
	if(func == Strings::ceiling) return ByteCode::ceiling; 
	if(func == Strings::trunc) return ByteCode::trunc; 
	if(func == Strings::exp) return ByteCode::exp; 
	if(func == Strings::log) return ByteCode::log; 
	if(func == Strings::cos) return ByteCode::cos; 
	if(func == Strings::sin) return ByteCode::sin; 
	if(func == Strings::tan) return ByteCode::tan; 
	if(func == Strings::acos) return ByteCode::acos; 
	if(func == Strings::asin) return ByteCode::asin; 
	if(func == Strings::atan) return ByteCode::atan; 
	if(func == Strings::isna) return ByteCode::isna; 
	if(func == Strings::isnan) return ByteCode::isnan; 
	if(func == Strings::isfinite) return ByteCode::isfinite; 
	if(func == Strings::isinfinite) return ByteCode::isinfinite; 
	
	if(func == Strings::lnot) return ByteCode::lnot; 
	
	if(func == Strings::sum) return ByteCode::sum; 
	if(func == Strings::prod) return ByteCode::prod; 
	if(func == Strings::mean) return ByteCode::mean; 
	if(func == Strings::min) return ByteCode::min; 
	if(func == Strings::max) return ByteCode::max; 
	if(func == Strings::any) return ByteCode::any; 
	if(func == Strings::all) return ByteCode::all; 
	if(func == Strings::cumsum) return ByteCode::cumsum; 
	if(func == Strings::cumprod) return ByteCode::cumprod; 
	if(func == Strings::cummin) return ByteCode::cummin; 
	if(func == Strings::cummax) return ByteCode::cummax; 
	
	if(func == Strings::type) return ByteCode::type; 
	if(func == Strings::length) return ByteCode::length; 
	if(func == Strings::strip) return ByteCode::strip; 
	
	if(func == Strings::random) return ByteCode::random; 
	
	throw RuntimeError(std::string("unexpected symbol '") + func + "' used as a unary operator"); 
}

static ByteCode::Enum op2(String const& func) {
	if(func == Strings::add) return ByteCode::add; 
	if(func == Strings::sub) return ByteCode::sub; 
	if(func == Strings::mul) return ByteCode::mul;
	if(func == Strings::div) return ByteCode::div; 
	if(func == Strings::idiv) return ByteCode::idiv; 
	if(func == Strings::mod) return ByteCode::mod; 
	if(func == Strings::pow) return ByteCode::pow; 
	if(func == Strings::atan2) return ByteCode::atan2; 
	if(func == Strings::hypot) return ByteCode::hypot; 
	if(func == Strings::lt) return ByteCode::lt; 
	if(func == Strings::gt) return ByteCode::gt; 
	if(func == Strings::eq) return ByteCode::eq; 
	if(func == Strings::neq) return ByteCode::neq; 
	if(func == Strings::ge) return ByteCode::ge; 
	if(func == Strings::le) return ByteCode::le; 
	if(func == Strings::lor) return ByteCode::lor; 
	if(func == Strings::land) return ByteCode::land; 

	if(func == Strings::pmin) return ByteCode::pmin; 
	if(func == Strings::pmax) return ByteCode::pmax; 

	if(func == Strings::cm2) return ByteCode::cm2; 
	
	if(func == Strings::bracket) return ByteCode::subset;
	if(func == Strings::bb) return ByteCode::subset2;

	if(func == Strings::vector) return ByteCode::vector;

	if(func == Strings::round) return ByteCode::round; 
	if(func == Strings::signif) return ByteCode::signif; 

	if(func == Strings::attrget) return ByteCode::attrget;
	
	throw RuntimeError(std::string("unexpected symbol '") + func + "' used as a binary operator"); 
}

static ByteCode::Enum op3(String const& func) {
	if(func == Strings::bracketAssign) return ByteCode::iassign;
	if(func == Strings::bbAssign) return ByteCode::eassign;
	if(func == Strings::split) return ByteCode::split;
	if(func == Strings::ifelse) return ByteCode::ifelse;
	if(func == Strings::seq) return ByteCode::seq;
	if(func == Strings::rep) return ByteCode::rep;
	if(func == Strings::attrset) return ByteCode::attrset;
	throw RuntimeError(std::string("unexpected symbol '") + func + "' used as a trinary operator"); 
}

struct JITContext {
	llvm::ExecutionEngine * execution_engine;
	llvm::Module * mod;

	llvm::PassManager* pm;
	llvm::FunctionPassManager* fpm;

	llvm::Type* void_type;
	llvm::Type* i64;
	llvm::PointerType* i64p;
	llvm::Type* i8;
	llvm::PointerType* i8p;
	llvm::PointerType* i8pp;
	llvm::StructType* value_type;
	llvm::PointerType* thread_type;
	
	llvm::Function*	gcroot;

	JITContext() {
		llvm::InitializeNativeTarget();
		llvm::OwningPtr<llvm::MemoryBuffer> buffer;
		llvm::MemoryBuffer::getFile("bin/ops.bc",buffer);
		mod = llvm::ParseBitcodeFile(buffer.get(),llvm::getGlobalContext());
		std::string ErrStr;
		llvm::ExecutionEngine * ee = llvm::EngineBuilder(mod).setEngineKind(llvm::EngineKind::JIT)
			.setOptLevel(llvm::CodeGenOpt::Aggressive)
			.setErrorStr(&ErrStr).create();
		if(!ee) {
			fprintf(stderr,"%s",ErrStr.c_str());
			exit(1);
		}
		execution_engine = ee;

		i64 = llvm::IntegerType::getInt64Ty(llvm::getGlobalContext());
		i64p = i64->getPointerTo();
		i8 = llvm::IntegerType::getInt8Ty(llvm::getGlobalContext());
		i8p = i8->getPointerTo();
		i8pp = i8p->getPointerTo();
		void_type = llvm::Type::getVoidTy(llvm::getGlobalContext());
		value_type = mod->getTypeByName("struct.Value");
		thread_type = mod->getTypeByName("class.Thread")->getPointerTo();
		gcroot = mod->getFunction("gcroot");
		
		pm = new llvm::PassManager();
		pm->add(new llvm::TargetData(mod));

		fpm = new llvm::FunctionPassManager(mod);	
		fpm->add(new llvm::TargetData(mod));

		// Opts from open shading language
		//llvm::PassManagerBuilder builder;
		//builder.OptLevel = 3;
		//builder.Inliner = llvm::createFunctionInliningPass();
		//// builder.DisableUnrollLoops = true;
		//builder.populateFunctionPassManager(*fpm);
		//builder.populateModulePassManager(*pm);
		//// Skip this for now, investigate later.  FIXME.
		////    builder.populateLTOPassManager (passes, true /* internalize */,
		////                                    true /* inline once again */);
		//builder.populateModulePassManager(*pm);

		pm->add(llvm::createVerifierPass());
		pm->add(llvm::createAlwaysInlinerPass());
		//p.add(llvm::createLowerSetJmpPass());
		//p.add(llvm::createCFGSimplificationPass());
		//p.add(llvm::createPromoteMemoryToRegisterPass());
		//p.add(llvm::createGlobalOptimizerPass());
		//p.add(llvm::createGlobalDCEPass());
		//p.add(llvm::createConstantPropagationPass());
		//p.add(llvm::createBasicAliasAnalysisPass());
		//p.add(llvm::createLICMPass());

		fpm->add(llvm::createVerifierPass());
		  fpm->add(llvm::createEarlyCSEPass());
		  fpm->add(llvm::createJumpThreadingPass());
		  fpm->add(llvm::createCFGSimplificationPass());
		  fpm->add(llvm::createBasicAliasAnalysisPass());
		  fpm->add(llvm::createPromoteMemoryToRegisterPass());
		  fpm->add(llvm::createScalarReplAggregatesPass());
		  fpm->add(llvm::createInstructionCombiningPass());
		  fpm->add(llvm::createReassociatePass());
		  fpm->add(llvm::createLICMPass());
		  fpm->add(llvm::createBasicAliasAnalysisPass());
		  fpm->add(llvm::createGVNPass());
		  fpm->add(llvm::createCFGSimplificationPass());
		  fpm->doInitialization();
	/*	
		// list of passes from vmkit (& julia)
		fpm->add(llvm::createCFGSimplificationPass()); // Clean up disgusting code
		fpm->add(llvm::createPromoteMemoryToRegisterPass());// Kill useless allocas
		fpm->add(llvm::createInstructionCombiningPass()); // Cleanup for scalarrepl.
		fpm->add(llvm::createScalarReplAggregatesPass()); // Break up aggregate allocas
		fpm->add(llvm::createInstructionCombiningPass()); // Cleanup for scalarrepl.
		fpm->add(llvm::createJumpThreadingPass());        // Thread jumps.
		fpm->add(llvm::createCFGSimplificationPass());    // Merge & remove BBs
		//fpm->add(llvm::createInstructionCombiningPass()); // Combine silly seq's
		//fpm->add(llvm::createCFGSimplificationPass());    // Merge & remove BBs
		fpm->add(llvm::createReassociatePass());          // Reassociate expressions
#if defined(LLVM_VERSION_MAJOR) && LLVM_VERSION_MAJOR == 3 && LLVM_VERSION_MINOR >= 1

		//fpm->add(llvm::createBBVectorizePass());

#endif
		fpm->add(llvm::createEarlyCSEPass()); //// ****
		//fpm->add(llvm::createLoopIdiomPass()); //// ****
		fpm->add(llvm::createLoopRotatePass());           // Rotate loops.
		fpm->add(llvm::createLICMPass());                 // Hoist loop invariants
		fpm->add(llvm::createLoopUnswitchPass());         // Unswitch loops.
		fpm->add(llvm::createInstructionCombiningPass()); 
		fpm->add(llvm::createIndVarSimplifyPass());       // Canonicalize indvars
		//fpm->add(llvm::createLoopDeletionPass());         // Delete dead loops
		fpm->add(llvm::createLoopUnrollPass());           // Unroll small loops
		//fpm->add(llvm::createLoopStrengthReducePass());   // (jwb added)
		fpm->add(llvm::createInstructionCombiningPass()); // Clean up after the unroller
		fpm->add(llvm::createGVNPass());                  // Remove redundancies
		//fpm->add(llvm::createMemCpyOptPass());            // Remove memcpy / form memset  
		fpm->add(llvm::createSCCPPass());                 // Constant prop with SCCP
		// Run instcombine after redundancy elimination to exploit opportunities
		// opened up by them.
		//fpm->add(llvm::createSinkingPass()); ////////////// ****
		//fpm->add(llvm::createInstructionSimplifierPass());///////// ****
		fpm->add(llvm::createInstructionCombiningPass());
		fpm->add(llvm::createJumpThreadingPass());         // Thread jumps
		fpm->add(llvm::createDeadStoreEliminationPass());  // Delete dead stores
		fpm->add(llvm::createAggressiveDCEPass());         // Delete dead instructions
		fpm->add(llvm::createCFGSimplificationPass());     // Merge & remove BBs
	*/	
		  fpm->add(llvm::createCFGSimplificationPass());
		  fpm->add(llvm::createScalarReplAggregatesPass());
		  fpm->add(llvm::createGVNPass());
		  
		  fpm->add(llvm::createCFGSimplificationPass());
		  fpm->add(llvm::createScalarReplAggregatesPass());
		  fpm->add(llvm::createGVNPass());
		  
		  fpm->add(llvm::createCFGSimplificationPass());
		  fpm->add(llvm::createScalarReplAggregatesPass());
		  fpm->add(llvm::createGVNPass());
		fpm->doInitialization();
	}

};

JITContext jitContext;

class JIT {
public:
	enum Scope {
		TOPLEVEL,
		FUNCTION,
		PROMISE
	};

private:

	enum Loc {
		INVALID,
		STACK,
		SLOT,
		CONSTANT
	};

	struct Operand {
		Loc loc;
		llvm::Value* v;
		uint64_t i;

		Operand()
			: loc(INVALID), v(0), i(1000000000) {}
		Operand(Loc loc, llvm::Value* v)
			: loc(loc), v(v) {}
	};

	Thread& thread;
	State& state;
	Environment* env;

	Scope scope;
	uint64_t loopDepth;
	uint64_t stackDepth;

        struct ValueComp {
                bool operator()(Value const& a, Value const& b) {
                        return a.header < b.header ||
                                (a.header == b.header && a.i < b.i);
                }
        };

        std::map<Value, Operand, ValueComp> constants;

	llvm::IRBuilder<> builder;
	llvm::Function* function;
	llvm::Value* slots_var;
	llvm::Value* stack_var;
	llvm::Value* thread_var;

	llvm::Value* createEntryBlockAlloca(llvm::Type* type);
	llvm::Constant* createInt64(int64_t i);
	llvm::Function* getCPPFunction(std::string s);

	Operand ref(Operand dest);
	Operand kill(Operand dest);
	Operand store(Operand value, Operand dest);

	// Language-level constructs	
	Operand compile(Value const& expr, Operand dest=Operand());
	Operand compileConstant(Value const& expr, Operand dest=Operand()); 
	Operand compileSymbol(Character const& symbol, Operand dest);
	Operand compileCall(List const& call, Character const& names, Operand dest);
	Operand compileExpression(List const& call, Operand dest);

	// Control-flow
	Operand compileWhile(List const& call, Operand dest);
	//Operand compileBrace(List const& call, Operand dest);

	// Memory ops
	Operand compileAssign(List const& call, Operand dest);
	
	// Math
	Operand compileUnary(char const* fn, List const& call, Operand dest);
	Operand compileBinary(char const* fn, List const& call, Operand dest);
	

public:
	JIT(Thread& thread, Environment* env, Scope scope)
		: thread(thread)
		, state(thread.state)
		, env(env)
		, scope(scope)
		, loopDepth(0)
		, stackDepth(0)
		, builder(llvm::getGlobalContext()) 
	{
	}
	Prototype* jit(Value const& expr);
};

llvm::Value* JIT::createEntryBlockAlloca(llvm::Type* type) {
	llvm::IRBuilder<> tmp(&function->getEntryBlock(), 
			function->getEntryBlock().begin());
	return tmp.CreateAlloca(type);
}

llvm::Constant* JIT::createInt64(int64_t i) {
	return llvm::ConstantInt::get(jitContext.i64, i, true);
}

llvm::Function* JIT::getCPPFunction(std::string name) {
	for(llvm::Module::FunctionListType::iterator i =
		jitContext.mod->begin(); i != jitContext.mod->end(); i++) {
		if(i->getName().str().find(name) != std::string::npos) {
			return i;
		}
	}
	_error(std::string("Couldn't find ") + name);
}

JIT::Operand JIT::ref(Operand dest) {
	if(dest.loc == INVALID) {
		dest.i = stackDepth++;
		dest.v = builder.CreateConstGEP1_64(stack_var, stackDepth);
		dest.loc = STACK;
	}
	return dest;
}

JIT::Operand JIT::kill(Operand dest) {
	stackDepth = std::min(stackDepth, dest.i);
	return dest;
}

JIT::Operand JIT::store(Operand value, Operand dest) {
	if(dest.loc == STACK || dest.loc == SLOT) {
		builder.CreateStore(builder.CreateLoad(value.v), dest.v);
		kill(value);
		return dest;
	}
	else {
		return value;
	}
}

JIT::Operand JIT::compileConstant(Value const& expr, Operand dest) {
	if(dest.loc == STACK || dest.loc == SLOT) {
		builder.CreateStore(
                                createInt64(expr.header),
                                builder.CreateStructGEP(builder.CreateStructGEP(dest.v,0),0));
                builder.CreateStore(
                                createInt64(expr.i),
                                builder.CreateStructGEP(builder.CreateStructGEP(dest.v,1),0));
		return dest;
	} else {
		std::map<Value, Operand>::const_iterator i = constants.find(expr);
		if(i == constants.end()) {
			llvm::Value* v = createEntryBlockAlloca(jitContext.value_type);
			builder.CreateStore(
                                createInt64(expr.header),
                                builder.CreateStructGEP(builder.CreateStructGEP(v,0),0));
        	        builder.CreateStore(
                                createInt64(expr.i),
                                builder.CreateStructGEP(builder.CreateStructGEP(v,1),0));
			dest = Operand(CONSTANT, v);
			constants[expr] = dest;
			return dest;
		} else {
			return i->second;
		}
	}
}

static int64_t isDotDot(String s) {
	if(s != 0 && s[0] == '.' && s[1] == '.') {
		int64_t v = 0;
		int64_t i = 2;
		// maximum 64-bit integer has 19 digits, but really who's going to pass
		// that many var args?
		while(i < (19+2) && s[i] >= '0' && s[i] <= '9') {
			v = v*10 + (s[i] - '0');
			i++;
		}
		if(i < (19+2) && s[i] == 0) return v;
	}
	return -1;	
}

#define EMIT_A(fn, A)	\
	builder.CreateCall2( jitContext.mod->getFunction(std::string(fn)+"_op"), thread_var, kill(A).v );

#define EMIT_A_C(fn, A, C)	\
	builder.CreateCall3( jitContext.mod->getFunction(std::string(fn)+"_op"), thread_var, kill(A).v, C.v);

#define EMIT_ABC(fn, A, B, C)	\
	builder.CreateCall4( jitContext.mod->getFunction(std::string(fn)+"_op"), thread_var, kill(A).v, kill(B).v, C.v);

JIT::Operand JIT::compileSymbol(Character const& symbol, Operand dest) {
	String s = symbol.s;
	int64_t dd = isDotDot(s);

	llvm::Value* c = 0;	
	if(dd > 0) {
		_error("dotdot not yet implemented");
		//emit(ByteCode::dotdot, dd-1, 0, result);
	}
	else {
		std::map<String, int64_t>::iterator i = env->m.find(s);
		if(i != env->m.end()) {
			c = builder.CreateConstGEP1_64(slots_var, i->second);
			return store(Operand(SLOT, c), dest);
		}
		else {
			dest = ref(dest);
			//c = EMIT_A("get", compileConstant(Character::c(s)));
			return dest;
		}
	}
}

/*
Compiler::Operand Compiler::placeInRegister(Operand r) {
	if(r.loc != REGISTER && r.loc != INVALID) {
		kill(r);
		Operand t = allocRegister();
		emit(ByteCode::fastmov, r, 0, t);
		return t;
	}
	return r;
}

Compiler::Operand Compiler::forceInRegister(Operand r) {
	if(r.loc != INVALID) {
		kill(r);
		Operand t = allocRegister();
		emit(ByteCode::mov, r, 0, t);
		return t;
	}
	return r;
}

CompiledCall Compiler::makeCall(List const& call, Character const& names) {
	// compute compiled call...precompiles promise code and some necessary values
	int64_t dotIndex = call.length()-1;
	PairList arguments;
	for(int64_t i = 1; i < call.length(); i++) {
		Pair p;
		if(names.length() > 0) p.n = names[i]; else p.n = Strings::empty;

		if(isSymbol(call[i]) && SymbolStr(call[i]) == Strings::dots) {
			p.v = call[i];
			dotIndex = i-1;
		} else if(isCall(call[i]) || isSymbol(call[i])) {
			Promise::Init(p.v, NULL, Compiler::compilePromise(thread, env, call[i]), false);
		} else {
			p.v = call[i];
		}
		arguments.push_back(p);
	}
	return CompiledCall(call, arguments, dotIndex, names.length() > 0);
}

// a standard call, not an op
Compiler::Operand Compiler::compileFunctionCall(List const& call, Character const& names, Prototype* code, Operand result) {
	Operand function = compile(call[0], code, Operand());
	CompiledCall a = makeCall(call, names);
	code->calls.push_back(a);
	kill(function);
	Operand out = allocRegister();
	if(!a.named && a.dotIndex >= (int64_t)a.arguments.size())
		emit(ByteCode::fastcall, function, code->calls.size()-1, out);
	else
		emit(ByteCode::call, function, code->calls.size()-1, out);
	kill(out);
	result = allocResult(result);
	if(out != result)
		emit(ByteCode::fastmov, out, 0, result);
	return result;
}

Compiler::Operand Compiler::compileInternalFunctionCall(List const& call, Prototype* code, Operand result) {
	String func = SymbolStr(call[0]);
	std::map<String, int64_t>::const_iterator itr = state.internalFunctionIndex.find(func);
	
	int64_t function = -1;
	
	if(itr == state.internalFunctionIndex.end()) {
		// this should eventually be an error, but so many are unimplemented right now that we'll just make it a warning
		_warning(thread, std::string("Unimplemented internal function ") + state.externStr(func));
	}
	else {
		function = itr->second;
		// check parameter count
		if(state.internalFunctions[function].params != call.length()-1)
			_error(std::string("Incorrect number of arguments to internal function ") + state.externStr(func));
	}
	// compile parameters directly...reserve registers for them.
	Operand liveIn = top();
	int64_t reg = liveIn.i-1;
	for(int64_t i = 1; i < call.length(); i++) {
		Operand r = placeInRegister(compile(call[i], code, Operand()));
		assert(r.i == reg+1);
		reg = r.i; 
	}
	// kill all parameters
	kill(reg); 	// only necessary to silence unused warning
	kill(liveIn); 
	result = allocResult(result);
	emit(ByteCode::internal, function, result, result);
	return result;
}
*/

JIT::Operand JIT::compileUnary(char const* fn, List const& call, Operand dest) {
	dest = ref(dest);
	//llvm::Function * op_fn = jitContext.mod->getFunction(std::string(fn)+"_op");
	//op_fn->addFnAttr(llvm::Attribute::AlwaysInline);
	EMIT_A_C(fn, compile(call[1]), dest)
	return dest;
}

JIT::Operand JIT::compileBinary(char const* fn, List const& call, Operand dest) {
	dest = ref(dest);
	//llvm::Function * op_fn = jitContext.mod->getFunction(std::string(fn)+"_op");
	//op_fn->addFnAttr(llvm::Attribute::AlwaysInline);
	EMIT_ABC(fn, compile(call[1]), compile(call[2]), dest)
	return dest;
}

JIT::Operand JIT::compileWhile(List const& call, Operand dest) {
	llvm::BasicBlock* body = llvm::BasicBlock::Create(llvm::getGlobalContext(), "while_body", function);
	llvm::BasicBlock* end =  llvm::BasicBlock::Create(llvm::getGlobalContext(), "while_end", function);
	
	llvm::Value* condition = EMIT_A("jc", compile(call[1]));
	llvm::Value* br = builder.CreateICmpEQ(condition, createInt64(1));
	builder.CreateCondBr(br, body, end);

	builder.SetInsertPoint(body);
	compile(call[2]);
	condition = EMIT_A("jc", compile(call[1]));
	br = builder.CreateICmpEQ(condition, createInt64(1));
	builder.CreateCondBr(br, body, end);
	
	builder.SetInsertPoint(end);
	return compileConstant(Null::Singleton(), dest);
}

JIT::Operand JIT::compileAssign(List const& call, Operand dest) {
	Value target = call[1];

	// handle complex LHS assignment instructions...
	// semantics here are kind of tricky. Consider compiling:
	//	 dim(a)[1] <- x
	// This is progressively turned `inside out`:
	//	1) dim(a) <- `[<-`(dim(a), 1, x)
	//	2) a <- `dim<-`(a, `[<-`(dim(a), 1, x))
	// TODO: One complication is that the result value of a complex assignment is the RHS
	// of the original expression, not the RHS of the inside out expression.
	Value value = call[2];
	while(isCall(target)) {
		List const& c = (List const&)target;
		List n(c.length()+1);

		for(int64_t i = 0; i < c.length(); i++) { n[i] = c[i]; }
		String as = state.internStr(state.externStr(SymbolStr(c[0])) + "<-");
		n[0] = CreateSymbol(as);
		n[c.length()] = value;

		Character nnames(c.length()+1);

		// Hacky, hacky, hacky...	
		if(!hasNames(c) 
				&& (((as == Strings::bracketAssign || as == Strings::bbAssign) && c.length() == 3) || 
					(as == Strings::attrset && c.length() == 3))) {
			for(int64_t i = 0; i < c.length()+1; i++) { nnames[i] = Strings::empty; }
		}
		else {
			if(hasNames(c)) {
				Value names = getNames(c);
				for(int64_t i = 0; i < c.length(); i++) { nnames[i] = ((Character const&)names)[i]; }
			} else {
				for(int64_t i = 0; i < c.length(); i++) { nnames[i] = Strings::empty; }
			}
			nnames[c.length()] = Strings::value;
		}
		value = CreateCall(n, nnames);
		target = c[1];
	}

	env->insert(SymbolStr(target));
	Operand assignDest(SLOT,
		builder.CreateConstGEP1_64(slots_var, env->m[SymbolStr(target)]));
	assignDest = compile(value, assignDest);
	return store(assignDest, dest);
}

/*llvm::Value* JIT::compileBrace(List const& call, llvm::Value* dest) {
	int64_t length = call.length();
	if(length <= 1) {
		return compileConstant(Null::Singleton(), dest);
	} else {
		for(int64_t i = 1; i < length-1; i++) {
			// memory results need to be forced to handle things like:
			// 	function(x,y) { x; y }
			// if x is a promise, it must be forced
			if(isSymbol(call[i]))
				kill(placeInRegister(compile(call[i], code, Operand())));
			else
				kill(compile(call[i], code, Operand()));
		}
		if(isSymbol(call[length-1]))
			return compile(call[length-1], code, result);
		else
			return placeInRegister(compile(call[length-1], code, result));
	}
}*/

JIT::Operand JIT::compileCall(List const& call, Character const& names, Operand dest) {
	int64_t length = call.length();
	if(length <= 0) throw CompileError("invalid empty call");

	//if(!isSymbol(call[0]) && !call[0].isCharacter1())
	//	return compileFunctionCall(call, names, code, result);

	String func = SymbolStr(call[0]);
	
	
	//if(func == Strings::list && call.length() == 2 
	//	&& isSymbol(call[1]) && SymbolStr(call[1]) == Strings::dots)
	//	return compileDotslist(call, dest);
	
	// These functions can't be called directly if the arguments are named or if
	// there is a ... in the args list

	bool complicated = false;
	for(int64_t i = 0; i < length; i++) {
		if(isSymbol(call[i]) && SymbolStr(call[i]) == Strings::dots) 
			complicated = true;
	}

	if(complicated)
		_error("Function call NYI");
		//return compileFunctionCall(call, names, code, result);

	if(func == Strings::switchSym)
		_error("Switch NYI");
		//return compileSwitch(call, names, dest);

	for(int64_t i = 0; i < length; i++) {
		if(names.length() > i && names[i] != Strings::empty) 
			complicated = true;
	}
	
	if(complicated)
		_error("Function call NYI");
		//return compileFunctionCall(call, names, code, result);

	if(func == Strings::assign || func == Strings::eqassign)
		return compileAssign(call, dest);
	else if(func == Strings::whileSym)
		return compileWhile(call, dest);
	//else if(func == Strings::brace)
	//	return compileBrace(call, dest);
	else if(func == Strings::paren) 
		return compile(call[1], dest);
	// Binary operators
	else if((func == Strings::add ||
		func == Strings::sub ||
		func == Strings::mul ||
		func == Strings::div || 
		func == Strings::idiv || 
		func == Strings::pow || 
		func == Strings::mod ||
		func == Strings::atan2 ||
		func == Strings::hypot ||
		func == Strings::eq ||
		func == Strings::neq ||
		func == Strings::lt ||
		func == Strings::gt ||
		func == Strings::ge ||
		func == Strings::le ||
		func == Strings::pmin ||
		func == Strings::pmax ||
		func == Strings::cm2 ||
		func == Strings::lor ||
		func == Strings::land ||
		func == Strings::bracket ||
		func == Strings::bb ||
		func == Strings::vector ||
		func == Strings::round ||
		func == Strings::signif ||
		func == Strings::attrget) &&
		call.length() == 3) 
		return compileBinary(ByteCode::toString(op2(func)), call, dest); 
	// Unary operators
	else if((func == Strings::add ||
		func == Strings::sub ||
		func == Strings::lnot || 
		func == Strings::abs || 
		func == Strings::sign ||
		func == Strings::sqrt ||
		func == Strings::floor ||
		func == Strings::ceiling || 
		func == Strings::trunc ||
		func == Strings::exp ||
		func == Strings::log ||
		func == Strings::cos ||
		func == Strings::sin ||
		func == Strings::tan ||
		func == Strings::acos ||
		func == Strings::asin ||
		func == Strings::atan ||
		func == Strings::isna ||
		func == Strings::isnan ||
		func == Strings::isfinite ||
		func == Strings::isinfinite ||
		func == Strings::sum ||
		func == Strings::prod ||
		func == Strings::mean ||
		func == Strings::min ||
		func == Strings::max ||
		func == Strings::any ||
		func == Strings::all ||
		func == Strings::cumsum ||
		func == Strings::cumprod ||
		func == Strings::cummin ||
		func == Strings::cummax ||
		func == Strings::type ||
		func == Strings::length ||
		func == Strings::strip ||
		func == Strings::random) &&
		call.length() == 2)
		return compileUnary(ByteCode::toString(op1(func)), call, dest);
	else
		_error("Not yet implemented bytecode");
		//return compileFunctionCall(call, names, code, result);
}

JIT::Operand JIT::compileExpression(List const& expr, Operand dest) {
	if(expr.length() == 0) {
		return compileConstant(Null::Singleton(), dest);
	}
	else {
		for(int64_t i = 0; i < expr.length()-1; i++) {
			if(isSymbol(expr[i]))
				compile(expr[i]);
			else
				compile(expr[i]);
		}
		if(isSymbol(expr[expr.length()-1]))
			return compile(expr[expr.length()-1], dest);
		else
			return compile(expr[expr.length()-1], dest);
	}
}

JIT::Operand JIT::compile(Value const& expr, Operand dest) {
	if(isSymbol(expr)) {
		assert(expr.isCharacter());
		return compileSymbol((Character const&)expr, dest);
	}
	else if(isExpression(expr)) {
		assert(expr.isList());
		return compileExpression((List const&)expr, dest);
	}
	else if(isCall(expr)) {
		assert(expr.isList());
		return compileCall((List const&)expr, 
			hasNames((List const&)expr) ? 
				(Character const&)getNames((List const&)expr) : 
				Character(0), 
			dest);
	}
	else {
		return compileConstant(expr, dest);
	}
}

Prototype* JIT::jit(Value const& expr) {

	timespec time_a = get_time();

	char fn_name[128];
	static int count = 0;
	sprintf(fn_name,"function_%d",count++);
	function = llvm::cast<llvm::Function>(jitContext.mod->getOrInsertFunction(fn_name, jitContext.value_type->getPointerTo(), jitContext.thread_type,NULL));
	llvm::Function::arg_iterator arguments = function->arg_begin();
	thread_var = arguments++;
	thread_var->setName("thread");

	llvm::BasicBlock* entry = llvm::BasicBlock::Create(llvm::getGlobalContext(),"entry",function);
	builder.SetInsertPoint(entry);
	
	slots_var = builder.CreateCall(
		jitContext.mod->getFunction("getSlots"), thread_var);
	
	stack_var = builder.CreateCall(
		jitContext.mod->getFunction("getStack"), thread_var);
	
	Operand result = compile(expr);
	
	// insert appropriate termination statement at end of code
	if(scope == FUNCTION) {
		//emit(ByteCode::ret, result, 0, 0);
	}
	else if(scope == PROMISE) {
		//emit(ByteCode::retp, result, 0, 0);
	} 
	else { // TOPLEVEL
		EMIT_A("rets", result);
		builder.CreateRet(result.v);
	}

	function->dump();

	jitContext.pm->run(*jitContext.mod);
	jitContext.fpm->run(*function);

	//function->dump();

	Prototype* prototype = new Prototype();
	void * impl = jitContext.execution_engine->getPointerToFunction(function);
	prototype->func = impl;

	printf("Compile time elapsed: %f\n", time_elapsed(time_a));
	
	return prototype;	
}


Prototype* JITCompiler::compileTopLevel(Thread& thread, Environment* env, Value const& expr) {                
	JIT jit(thread, env, JIT::TOPLEVEL);
	return jit.jit(expr);
}
        
Prototype* JITCompiler::compileFunctionBody(Thread& thread, Environment* env, Value const& expr) {                
	JIT jit(thread, env, JIT::FUNCTION);
	return jit.jit(expr);
}

Prototype* JITCompiler::compilePromise(Thread& thread, Environment* env, Value const& expr) {                
	JIT jit(thread, env, JIT::PROMISE);
	return jit.jit(expr);
}

