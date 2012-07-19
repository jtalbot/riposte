
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
#include "llvm/Transforms/Utils/Cloning.h"

#include "jit.h"
#include "internal.h"
#include "value.h"
#include "type.h"
#include "bc.h"
#include "ops.h"
#include "internal.h"
#include "interpreter.h"
#include "sse.h"
#include "call.h"
#include "frontend.h"

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
	llvm::FunctionType* internalfunction_type;
	
	llvm::Function*	gcroot;

	JITContext() {
		llvm::InitializeNativeTarget();
		llvm::OwningPtr<llvm::MemoryBuffer> buffer;
		llvm::MemoryBuffer::getFile("bin/ops.bc",buffer);
		mod = llvm::ParseBitcodeFile(buffer.get(),llvm::getGlobalContext());

		std::string ErrStr;
		llvm::TargetOptions to;
		to.EnableFastISel = true;
		to.JITExceptionHandling = true;
		llvm::ExecutionEngine * ee = llvm::EngineBuilder(mod)
			.setEngineKind(llvm::EngineKind::JIT)
			.setOptLevel(llvm::CodeGenOpt::Aggressive)
			.setTargetOptions(to)
			.setUseMCJIT(true)
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
		llvm::Type* internalfunction_args[] = 
			{ thread_type, value_type->getPointerTo(), value_type->getPointerTo() };
		internalfunction_type = llvm::FunctionType::get(void_type, internalfunction_args, false);
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
			: loc(loc), v(v), i(1000000000) {}
	};

	struct LoopEscape {
		llvm::BasicBlock* nxt;
		llvm::BasicBlock* brk;
	};

	Thread& thread;
	State& state;
	Shape* shape;

	Scope scope;
	uint64_t loopDepth;
	uint64_t stackDepth;
	uint64_t stackHighWater;

        struct ValueComp {
                bool operator()(Value const& a, Value const& b) {
                        return a.header < b.header ||
                                (a.header == b.header && a.i < b.i);
                }
        };

        std::map<Value, Operand, ValueComp> constants;
	std::vector<CallSite> calls;
	std::map<String, size_t> ic_map;
	std::vector<IC> ics;
	std::vector<LoopEscape> loops;

	llvm::IRBuilder<> builder;
	llvm::Function* function;
	llvm::Value* dots_var;
	llvm::Value* slots_var;
	llvm::Value* register_var;
	llvm::Value* thread_var;
	Operand returnDest;
	llvm::BasicBlock* returnBlock;
	std::vector<llvm::CallInst*> callInsts;

	llvm::Value* createEntryBlockAlloca(llvm::Type* type);
	llvm::ConstantInt* createInt64(int64_t i);
	llvm::ConstantExpr* createString(String s);
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
	Operand compileIf(List const& call, Operand dest);
	Operand compileSwitch(List const& call, Character const& names, Operand dest);
	Operand compileWhile(List const& call, Operand dest);
	Operand compileFor(List const& call, Operand dest);
	Operand compileNext(List const& call, Operand dest);
	Operand compileBreak(List const& call, Operand dest);
	Operand compileBrace(List const& call, Operand dest);
	Operand compileFunctionCall(List const& call, Character const& names, Operand dest);
	Operand compileInternalFunctionCall(List const& call, Operand dest);
	Operand compileReturn(List const& call, Operand dest);

	// Type constructors
	Operand compileFunction(List const& call, Operand dest);
	
	// Memory ops
	Operand compileAssign(List const& call, Operand dest);
	Operand compileAssign2(List const& call, Operand dest);
	
	// Common functions 
	Operand compileUnary(char const* fn, List const& call, Operand dest);
	Operand compileBinary(char const* fn, List const& call, Operand dest);
	Operand compileTernary(char const* fn, List const& call, Operand dest);
	Operand compileDotslist(List const& call, Operand dest);
	

	Code* compileFunctionBody(Value const& expr, Shape* shape) {
		JIT jit(thread, shape, JIT::FUNCTION);
		return jit.jit(expr);
	}

	Code* compilePromise(Value const& expr) { 
		JIT jit(thread, shape, JIT::PROMISE);
		return jit.jit(expr);
	}

	llvm::CallInst* record(llvm::CallInst* ci) {
		callInsts.push_back(ci);
		return ci;
	}

public:
	JIT(Thread& thread, Shape* shape, Scope scope)
		: thread(thread)
		, state(thread.state)
		, shape(shape)
		, scope(scope)
		, loopDepth(0)
		, stackDepth(0)
		, stackHighWater(0)
		, builder(llvm::getGlobalContext()) 
	{
	}

	Code* jit(Value const& expr);
};

#define CALL0(fn) record(\
	builder.CreateCall( jitContext.mod->getFunction(std::string(fn)+"_op"), \
		thread_var ));

#define CALL1(fn, A) record(\
	builder.CreateCall2( jitContext.mod->getFunction(std::string(fn)+"_op"), \
		thread_var, A ));

#define CALL2(fn, A, B) record(\
	builder.CreateCall3( jitContext.mod->getFunction(std::string(fn)+"_op"), \
		thread_var, A, B ));

#define CALL3(fn, A, B, C) record(\
	builder.CreateCall4( jitContext.mod->getFunction(std::string(fn)+"_op"), \
		thread_var, A, B, C ));

#define CALL4(fn, A, B, C, D) record(\
	builder.CreateCall5( jitContext.mod->getFunction(std::string(fn)+"_op"), \
		thread_var, A, B, C, D ));



llvm::Value* JIT::createEntryBlockAlloca(llvm::Type* type) {
	llvm::IRBuilder<> tmp(&function->getEntryBlock(), 
			function->getEntryBlock().begin());
	return tmp.CreateAlloca(type);
}

llvm::ConstantInt* JIT::createInt64(int64_t i) {
	return (llvm::ConstantInt*)llvm::ConstantInt::get(jitContext.i64, i, true);
}

llvm::ConstantExpr* JIT::createString(String s) {
	llvm::Constant* c = llvm::ConstantInt::get(jitContext.i64, (int64_t)s, true);
	return (llvm::ConstantExpr*)llvm::ConstantExpr::getIntToPtr(c, jitContext.i8p);
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
		dest.v = builder.CreateConstGEP1_64(register_var, stackDepth);
		dest.loc = STACK;
		stackHighWater = std::max(stackHighWater, stackDepth);
	}
	else if(dest.loc == CONSTANT) {
		_error("Invalid destination ref");
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
	else if(dest.loc == INVALID) {
		return value;
	}
	else {
		_error("Unexpected dest");
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
			llvm::IRBuilder<> tmp(&function->getEntryBlock(), 
				function->getEntryBlock().begin());
			llvm::Value* v = tmp.CreateAlloca(jitContext.value_type);
			tmp.CreateStore(
                                createInt64(expr.header),
                                tmp.CreateStructGEP(tmp.CreateStructGEP(v,0),0));
        	        tmp.CreateStore(
                                createInt64(expr.i),
                                tmp.CreateStructGEP(tmp.CreateStructGEP(v,1),0));
			/*llvm::Value* cv = tmp.CreatePointerCast(v, jitContext.i8p);
			llvm::Value* start = tmp.CreateCall2(
				llvm::Intrinsic::getDeclaration(jitContext.mod, llvm::Intrinsic::invariant_start),
				createInt64(sizeof(Value)), cv);

			
			llvm::IRBuilder<> tmp2(returnBlock, 
				returnBlock->end());
			tmp2.CreateCall3(
				llvm::Intrinsic::getDeclaration(jitContext.mod, llvm::Intrinsic::invariant_end),
				start, createInt64(sizeof(Value)), cv);
			*/
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

JIT::Operand JIT::compileSymbol(Character const& symbol, Operand dest) {
	String s = symbol.s;
	int64_t dd = isDotDot(s);

	dest = ref(dest);

	if(dd > 0) {
		CALL2("getDot", createInt64(dd-1), dest.v);
	}
	else {
		std::map<String, size_t>::const_iterator i = shape->m.find(s);
		if(i != shape->m.end()) {
			CALL3("getSlot", createInt64(i->second), createString(s), dest.v);
		}
		else {
			CALL2("get", createString(s), dest.v);
		}
	}
	return dest;
}

/*
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
	Operand a = compile(call[1]);
	kill(a);
	dest = ref(dest);
	CALL2(fn, a.v, dest.v)
	return dest;
}

JIT::Operand JIT::compileBinary(char const* fn, List const& call, Operand dest) {
	Operand a = compile(call[1]);
	Operand b = compile(call[2]);
	kill(b); kill(a);
	dest = ref(dest);
	CALL3(fn, a.v, b.v, dest.v)
	return dest;
}

JIT::Operand JIT::compileTernary(char const* fn, List const& call, Operand dest) {
	Operand a = compile(call[1]);
	Operand b = compile(call[2]);
	Operand c = compile(call[3]);
	kill(c); kill(b); kill(a);
	dest = ref(dest);
	CALL4(fn, a.v, b.v, c.v, dest.v)
	return dest;
}

JIT::Operand JIT::compileWhile(List const& call, Operand dest) {
	llvm::BasicBlock* cond = llvm::BasicBlock::Create(llvm::getGlobalContext(), "while_body", function);
	llvm::BasicBlock* body = llvm::BasicBlock::Create(llvm::getGlobalContext(), "while_body", function);
	llvm::BasicBlock* end =  llvm::BasicBlock::Create(llvm::getGlobalContext(), "while_end", function);
	builder.CreateBr(cond);

	builder.SetInsertPoint(cond);
	Operand a = kill(compile(call[1]));
	llvm::Value* condition = CALL1("jc", a.v);
	llvm::Value* br = builder.CreateICmpEQ(condition, createInt64(1));
	builder.CreateCondBr(br, body, end);

	LoopEscape le = { cond, end };
	loops.push_back(le);
	builder.SetInsertPoint(body);
	kill(compile(call[2]));
	a = kill(compile(call[1]));
	builder.CreateBr(cond);
	
	builder.SetInsertPoint(end);
	loops.pop_back();
	return compileConstant(Null::Singleton(), dest);
}

JIT::Operand JIT::compileFor(List const& call, Operand dest) {
	llvm::BasicBlock* cond = llvm::BasicBlock::Create(llvm::getGlobalContext(), "while_body", function);
	llvm::BasicBlock* body = llvm::BasicBlock::Create(llvm::getGlobalContext(), "while_body", function);
	llvm::BasicBlock* end =  llvm::BasicBlock::Create(llvm::getGlobalContext(), "while_end", function);
	
	size_t i = shape->add(SymbolStr(call[1]));
	Operand index(SLOT,
		builder.CreateConstGEP1_64(slots_var, (int64_t)i));
	index = compileConstant(Null::Singleton(), index);

	llvm::Value* internalIndex = createEntryBlockAlloca(jitContext.i64);
	builder.CreateStore(createInt64(0), internalIndex);
	
	Operand vector = compile(call[2]);
	llvm::Value* length = CALL1("forbegin", vector.v);
	builder.CreateBr(cond);

	builder.SetInsertPoint(cond);
	llvm::Value* br = builder.CreateICmpSLT(builder.CreateLoad(internalIndex), length);
	builder.CreateCondBr(br, body, end);

	LoopEscape le = { cond, end };
	loops.push_back(le);
	builder.SetInsertPoint(body);
	CALL3("element2", vector.v, builder.CreateLoad(internalIndex), index.v);
	kill(compile(call[3]));
	builder.CreateStore(builder.CreateAdd(builder.CreateLoad(internalIndex), createInt64(1)), internalIndex);
	builder.CreateBr(cond);
	
	builder.SetInsertPoint(end);
	loops.pop_back();
	return compileConstant(Null::Singleton(), dest);
}

JIT::Operand JIT::compileNext(List const& call, Operand dest) {
	if(loops.size() == 0)
		_error("no loop for break/next");
	builder.CreateBr(loops.back().nxt);
	llvm::BasicBlock* dead = llvm::BasicBlock::Create(llvm::getGlobalContext(), "", function);
	builder.SetInsertPoint(dead);
	return dest;
}

JIT::Operand JIT::compileBreak(List const& call, Operand dest) {
	if(loops.size() == 0)
		_error("no loop for break/next");
	builder.CreateBr(loops.back().brk);
	llvm::BasicBlock* dead = llvm::BasicBlock::Create(llvm::getGlobalContext(), "", function);
	builder.SetInsertPoint(dead);
	return dest;
}

JIT::Operand JIT::compileIf(List const& call, Operand dest) {
	llvm::BasicBlock* ifB = llvm::BasicBlock::Create(llvm::getGlobalContext(), "if", function);
	llvm::BasicBlock* elseB = llvm::BasicBlock::Create(llvm::getGlobalContext(), "else", function);
	llvm::BasicBlock* end =  llvm::BasicBlock::Create(llvm::getGlobalContext(), "if_end", function);

	dest = ref(dest);
	Operand a = kill(compile(call[1]));
	llvm::Value* condition = CALL1("jc", a.v);
	llvm::Value* br = builder.CreateICmpEQ(condition, createInt64(1));

	builder.CreateCondBr(br, ifB, elseB);
	
	builder.SetInsertPoint(ifB);
	dest = compile(call[2], dest);
	builder.CreateBr(end);
	builder.SetInsertPoint(elseB);
	dest = call.length() == 4 	? compile(call[3], dest)
					: compileConstant(Null::Singleton(), dest);
	builder.CreateBr(end);
	builder.SetInsertPoint(end);
	return dest;
}

JIT::Operand JIT::compileFunctionCall(List const& call, Character const& names, Operand dest) {
	// compute compiled call...precompiles promise code and some necessary values
	size_t dotIndex = call.length()-1;
	PairList arguments;
	for(int64_t i = 1; i < call.length(); i++) {
		Pair p;
		if(names.length() > 0) p.n = names[i]; else p.n = Strings::empty;

		if(isSymbol(call[i]) && SymbolStr(call[i]) == Strings::dots) {
			p.v = call[i];
			dotIndex = i-1;
		} else if(isCall(call[i]) || isSymbol(call[i])) {
			Promise::Init(p.v, NULL, compilePromise(call[i]), false);
		} else {
			p.v = call[i];
		}
		arguments.push_back(p);
	}

	CallSite a(
		call,
		arguments, 
		dotIndex, 
		names.length() > 0, 
		dotIndex < arguments.size());
	calls.push_back(a);

	Operand function = kill(compile(call[0], Operand()));
	kill(function);
	
	dest = ref(dest);
	CALL4("call", function.v, 
		createInt64(calls.size()-1), createInt64(stackDepth), dest.v);
	return dest;
}

JIT::Operand JIT::compileInternalFunctionCall(List const& outerCall, Operand dest) {
	if(outerCall.length() != 2 || !outerCall[1].isList() || !isCall(outerCall[1]))
		throw CompileError(std::string(".Internal has invalid arguments (") + Type::toString(outerCall[1].type()) + ")");

	List const& call = (List const&)outerCall[1];	
	
	String func = SymbolStr(call[0]);
	std::map<String, int64_t>::const_iterator i = 
		state.internalFunctionIndex.find(func);
	int64_t f = -1;
	if(i == state.internalFunctionIndex.end())
		// TODO: make this an error when more internal functions are implemented
		_warning(thread, std::string("Unimplemented internal function ")
			+ state.externStr(func));
	else if(state.internalFunctions[i->second].params != call.length()-1)
		_error(std::string("Incorrect number of parameters to internal function ")
			+ state.externStr(func));	
	else
		f = i->second;

	llvm::Value* registerPtr = llvm::ConstantPointerNull::get(jitContext.value_type->getPointerTo());
	for(int64_t i = 1; i < call.length(); i++) {
		Operand a = ref(Operand());
		compile(call[i], a);
		if(i == 1)
			registerPtr = a.v;
	}

	dest = ref(dest);

	llvm::Value* intfunc = builder.CreateIntToPtr(
		createInt64((int64_t)state.internalFunctions[i->second].ptr),
		jitContext.internalfunction_type->getPointerTo());
	builder.CreateCall3(intfunc, thread_var, registerPtr, dest.v);
	return dest;
}

JIT::Operand JIT::compileReturn(List const& call, Operand dest) {
	if(scope != FUNCTION)
		_error("Attempting to return from top-level expression or from non-function. Riposte doesn't support return inside promises currently, and may never do so");

	if(call.length() == 1) {
		compileConstant(Null::Singleton(), returnDest);
	}
	else if(call.length() == 2) {
		compile(call[1], returnDest);
	}
	else
		_error("Too many return value");

	builder.CreateBr(returnBlock);
	llvm::BasicBlock* dead = llvm::BasicBlock::Create(llvm::getGlobalContext(), "", function);
	builder.SetInsertPoint(dead);
	return returnDest;
}

JIT::Operand JIT::compileSwitch(List const& call, Character const& names, Operand dest) {
	if(call.length() < 2) {
		_error("'EXPR' is missing");
	}
	
	dest = ref(dest);
	
	llvm::BasicBlock* end = llvm::BasicBlock::Create(llvm::getGlobalContext(), "switch_end", function);
	
	// Last block handles the case: switch("d", a=1, b=2, c=3, d=)
	// This should return NULL.
	llvm::BasicBlock* last = llvm::BasicBlock::Create(llvm::getGlobalContext(), "switch_end", function);

	llvm::ArrayType* at = llvm::ArrayType::get(jitContext.i8p, call.length()-2);
	std::vector<llvm::Constant*> strs;
	for(size_t i = 2; i < (size_t)call.length(); i++) {
		strs.push_back(createString(
			i < (size_t)names.length()
				? names[i]
				: Strings::empty
			));
	}

	llvm::Constant* strArray = llvm::ConstantArray::get(at, strs);
	llvm::Value* gg = new llvm::GlobalVariable(
		*jitContext.mod, 
		strArray->getType(),
		true, 
		llvm::GlobalValue::PrivateLinkage, 
		strArray, 
		"");
	llvm::Value* strPtr = builder.CreatePointerCast(
		builder.CreateConstGEP1_64(gg, 0), jitContext.i8pp);

	Operand c = compile(call[1], Operand());
	llvm::Value* switchCase = CALL3("switch", c.v, strPtr, createInt64(call.length()-2));
	llvm::SwitchInst* s = builder.CreateSwitch(switchCase, last, call.length()-2);

	
	std::vector<llvm::BasicBlock*> bbs;

	builder.SetInsertPoint(last);
	compileConstant(Null::Singleton(), dest);
	builder.CreateBr(end);
	bbs.push_back(last);

	for(size_t i = call.length()-1; i >= 2; i--) {
		if(!call[i].isNil()) {
			llvm::BasicBlock* bb = 
				llvm::BasicBlock::Create(llvm::getGlobalContext(), "switch", function);
			builder.SetInsertPoint(bb);
			compile(call[i], dest);
			builder.CreateBr(end);
			bbs.push_back(bb);
		}
		else {
			bbs.push_back(bbs.back());
		}
	}
	std::reverse(bbs.begin(), bbs.end());

	for(size_t i = 2; i < call.length(); i++) {
		s->addCase(createInt64(i-2), bbs[i-2]);
	}
	
	builder.SetInsertPoint(end);
	return dest;
}

JIT::Operand JIT::compileFunction(List const& call, Operand dest) {
	//compile the default parameters
	assert(call[1].isList());
	List const& c = (List const&)call[1];
	Character names = hasNames(c) ?
		(Character const&)getNames(c) : Character(0);

	PairList parameters;
	for(int64_t i = 0; i < c.length(); i++) {
		Pair p;
		if(names.length() > 0) p.n = names[i]; else p.n = Strings::empty;
		if(!c[i].isNil()) {
			Promise::Init(p.v, NULL, compilePromise(c[i]), true);
		}
		else {
			p.v = c[i];
		}
		parameters.push_back(p);
	}

	Shape* functionLayout = new Shape();
	for(int64_t i = 0; i < names.length(); i++) {
		functionLayout->add(names[i]);
	}
	//compile the source for the body
	Code* functionCode = compileFunctionBody(call[2], functionLayout);

	// Populate prototype
	Prototype* p 	= new Prototype();
	p->expression 	= call[2];
	p->string 	= SymbolStr(call[3]);
	p->parameters	= parameters;
	p->parametersSize = parameters.size();
	p->dotIndex	= parameters.size();
	p->hasDots	= false;
	p->code		= functionCode;

	for(int64_t i = 0; i < (int64_t)parameters.size(); i++) {
		if(parameters[i].n == Strings::dots) {
			p->dotIndex = i;
			p->hasDots = true;
		}
	}

	Value function;
	Function::Init(function, p, 0);
	Operand funcOp = kill(compileConstant(function, Operand()));
	dest = ref(dest);
	CALL2("function", funcOp.v, dest.v);
	return dest;
}

void assignLHS(State& state, Value& target, Value& value) {
	// handle complex LHS assignment instructions...
	// semantics here are kind of tricky. Consider compiling:
	//	 dim(a)[1] <- x
	// This is progressively turned `inside out`:
	//	1) dim(a) <- `[<-`(dim(a), 1, x)
	//	2) a <- `dim<-`(a, `[<-`(dim(a), 1, x))
	// TODO: One complication is that the result value of a complex assignment is the RHS
	// of the original expression, not the RHS of the inside out expression.
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
} 

JIT::Operand JIT::compileAssign(List const& call, Operand dest) {
	Value target = call[1];
	Value value = call[2];
	assignLHS(state, target, value);

	
	size_t i = shape->add(SymbolStr(target));
	Operand assignDest(SLOT,
		builder.CreateConstGEP1_64(slots_var, (int64_t)i));
	assignDest = compile(value, assignDest);
	return store(assignDest, dest);
}

JIT::Operand JIT::compileAssign2(List const& call, Operand dest) {
	Value target = call[1];
	Value value = call[2];
	assignLHS(state, target, value);

	dest = compile(value, dest);
	CALL2("assign2", dest.v, createInt64((int64_t)SymbolStr(target)));
	return dest;
}

JIT::Operand JIT::compileBrace(List const& call, Operand dest) {
	int64_t length = call.length();
	if(length <= 1) {
		return compileConstant(Null::Singleton(), dest);
	} else {
		for(int64_t i = 1; i < length-1; i++) {
			kill(compile(call[i], Operand()));
		}
		return compile(call[length-1], dest);
	}
}

JIT::Operand JIT::compileDotslist(List const& call, Operand dest) {
	dest = ref(dest);
	CALL1("dotslist", dest.v)
	return dest;
}

JIT::Operand JIT::compileCall(List const& call, Character const& names, Operand dest) {
	int64_t length = call.length();
	if(length <= 0) throw CompileError("invalid empty call");

	if(!isSymbol(call[0]) && !call[0].isCharacter1())
		return compileFunctionCall(call, names, dest);

	String func = SymbolStr(call[0]);

	if(func == Strings::list && call.length() == 2 
		&& isSymbol(call[1]) && SymbolStr(call[1]) == Strings::dots)
		return compileDotslist(call, dest);
	
	// These functions can't be called directly if the arguments are named or if
	// there is a ... in the args list

	bool complicated = false;
	for(int64_t i = 0; i < length; i++) {
		if(isSymbol(call[i]) && SymbolStr(call[i]) == Strings::dots) 
			complicated = true;
	}

	if(complicated)
		return compileFunctionCall(call, names, dest);

	if(func == Strings::switchSym)
		return compileSwitch(call, names, dest);

	for(int64_t i = 0; i < length; i++) {
		if(names.length() > i && names[i] != Strings::empty) 
			complicated = true;
	}
	
	if(complicated)
		return compileFunctionCall(call, names, dest);

	if(func == Strings::assign || func == Strings::eqassign)
		return compileAssign(call, dest);
	else if(func == Strings::assign2)
		return compileAssign2(call, dest);
	else if(func == Strings::whileSym)
		return compileWhile(call, dest);
	else if(func == Strings::forSym)
		return compileFor(call, dest);
	else if(func == Strings::nextSym)
		return compileNext(call, dest);
	else if(func == Strings::breakSym)
		return compileBreak(call, dest);
	else if(func == Strings::ifSym)
		return compileIf(call, dest);
	else if(func == Strings::returnSym)
		return compileReturn(call, dest);
	else if(func == Strings::brace)
		return compileBrace(call, dest);
	else if(func == Strings::function)
		return compileFunction(call, dest);
	else if(func == Strings::paren) 
		return compile(call[1], dest);
	else if(func == Strings::internal)
                return compileInternalFunctionCall(call, dest);
        else if((func == Strings::bracketAssign ||
                func == Strings::bbAssign ||
                func == Strings::split ||
                func == Strings::ifelse ||
                func == Strings::seq ||
                func == Strings::rep ||
                func == Strings::attrset) &&
                call.length() == 4) {
		return compileTernary(ByteCode::toString(op3(func)), call, dest);
        }
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
		return compileFunctionCall(call, names, dest);
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

Code* JIT::jit(Value const& expr) {

	timespec time_a = get_time();

	char fn_name[128];
	static int count = 0;
	sprintf(fn_name,"function_%d",count++);
	function = llvm::cast<llvm::Function>(jitContext.mod->getOrInsertFunction(fn_name, jitContext.value_type, jitContext.thread_type,NULL));
	function->addFnAttr(llvm::Attribute::UWTable);
	llvm::Function::arg_iterator arguments = function->arg_begin();
	thread_var = arguments++;
	thread_var->setName("thread");

	llvm::BasicBlock* entry = llvm::BasicBlock::Create(llvm::getGlobalContext(),"entry",function);
	builder.SetInsertPoint(entry);
	
	returnDest = Operand(STACK, createEntryBlockAlloca(jitContext.value_type));
	returnBlock = llvm::BasicBlock::Create(llvm::getGlobalContext(),"return",function);
	
	dots_var = CALL0("dotsAddress");
	slots_var = CALL0("slotsAddress");
	register_var = CALL0("registersAddress");

	returnDest = kill(compile(expr, returnDest));
	
	if(scope == TOPLEVEL) {
		CALL1("bind", returnDest.v);
	} 

	builder.CreateBr(returnBlock);
	builder.SetInsertPoint(returnBlock);
	builder.CreateRet(builder.CreateLoad(returnDest.v));
	

	// inline all always inline call sites
	// note, relying on modular arithmetic in loop.
	for(size_t i = callInsts.size()-1; i < callInsts.size(); i--) {
		if(callInsts[i]->getCalledFunction()->hasFnAttr(
			llvm::Attribute::AlwaysInline)) {
			llvm::InlineFunctionInfo ifi;
			llvm::InlineFunction(callInsts[i], ifi);
		}
	}
	//function->dump();
	// optimize
	//jitContext.pm->run(*jitContext.mod);
	jitContext.fpm->run(*function);
	//function->dump();
	
	Code* code = new Code();
	code->ptr = (Code::Ptr)jitContext.execution_engine->getPointerToFunction(function);
	code->registers = stackHighWater;
	code->shape = shape;
	code->calls.swap(calls);
	code->ics.swap(ics);
	
	//printf("Compile time elapsed: %f\n", time_elapsed(time_a));
	
	return code;
}

// Template strategy
// Compile new template for all scopes
// For top level, do a "restructuring" pass where
//  we restructure the environment to match the new template.

// Have separate addressing spaces for stack and slots means
// that they can be separated on the stack OR separated between
// the stack and the heap. So we can still allocate environment slots
// on the stack to start with.

Code* JITCompiler::compile(Thread& thread, Value const& expr) { 
	return compile(thread, expr, thread.frame.environment);
}

Code* JITCompiler::compile(Thread& thread, Value const& expr, Environment* env) { 
	Shape* s = new Shape();
	JIT jit(thread, s, JIT::TOPLEVEL);
	return jit.jit(expr);
}
        

