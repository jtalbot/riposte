
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

class Compiler {
private:
	State& state;

	struct Scope {
		std::vector<Symbol> symbols;
		Value parameters;
		int64_t registers;
	}; 

	std::vector<Scope> scope;

	int64_t registerDepth;
	int64_t loopDepth;	

	Compiler(State& state) : state(state) {
		loopDepth = 0;
	}
	
	Code* compile(Value const& expr);			// compile function block, code ends with return
	int64_t compile(Value const& expr, Code* code);		// compile into existing code block

	int64_t compileConstant(Value const& expr, Code* code);
	int64_t compileSymbol(Symbol const& symbol, Code* code); 
	int64_t compileOp(Call const& call, Code* code); 
	int64_t compileCall(Call const& call, Code* code); 
	int64_t compileFunctionCall(Call const& call, Code* code); 
	int64_t compileExpression(Expression const& values, Code* code);
public:
	static Code* compile(State& state, Value const& expr) {
		Compiler compiler(state);
		return compiler.compile(expr);
	}
	
	static Code* compile(State& state, Value const& expr, Environment* env) {
		Compiler compiler(state);
		return compiler.compile(expr);
	}
};

// compilation routines
#endif
