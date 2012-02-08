
#ifndef COMPILER_H
#define COMPILER_H

#include <string>
#include <sstream>
#include <stdexcept>
#include <string>

#include "common.h"
#include "exceptions.h"
#include "value.h"
#include "type.h"
#include "bc.h"
#include "interpreter.h"

class Compiler {
private:
	State& state;

	enum Scope {
		TOPLEVEL,
		FUNCTION,
		PROMISE
	};
	
	Scope scope;
	Character parameters;	// only valid if in FUNCTION scope 
	uint64_t loopDepth;
	std::vector<int64_t> constRegisters;

	int64_t n;
	int64_t allocRegister() { return --n; }

	Compiler(State& state, Scope scope) : state(state), scope(scope), loopDepth(0), n(0) {}
	
	Prototype* compile(Value const& expr);			// compile function block, code ends with return
	int64_t compile(Value const& expr, Prototype* code);		// compile into existing code block

	int64_t compileConstant(Value const& expr, Prototype* code);
	int64_t compileSymbol(Value const& symbol, Prototype* code); 
	int64_t compileCall(List const& call, Character const& names, Prototype* code); 
	int64_t compileFunctionCall(List const& call, Character const& names, Prototype* code); 
	int64_t compileInternalFunctionCall(Object const& o, Prototype* code); 
	int64_t compileExpression(List const& values, Prototype* code);
	
	CompiledCall makeCall(List const& call, Character const& names);

	int64_t emit(Prototype* code, ByteCode::Enum bc, int64_t a, int64_t b, int64_t c);
	int64_t emit(Prototype* code, ByteCode::Enum bc, String s, int64_t b, int64_t c);

public:
	static Prototype* compileTopLevel(State& state, Value const& expr) {
		Compiler compiler(state, TOPLEVEL);
		return compiler.compile(expr);
	}
	
	static Prototype* compileFunctionBody(State& state, Value const& expr, Character& parameters) {
		Compiler compiler(state, FUNCTION);
		compiler.parameters = parameters;
		return compiler.compile(expr);
	}
	
	static Prototype* compilePromise(State& state, Value const& expr) {
		Compiler compiler(state, PROMISE);
		return compiler.compile(expr);
	}
};

// compilation routines
#endif
