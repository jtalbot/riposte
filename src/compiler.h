
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
		int64_t registers;
	}; 

	std::vector<Scope> scope;

	int64_t registerDepth;
	int64_t loopDepth;	

	Compiler(State& state) : state(state) {
		loopDepth = 0;
	}
	
	Closure compile(Value const& expr); 				// compile into new closure
	int64_t compile(Value const& expr, Closure& closure);		// compile into existing closure

	int64_t compileConstant(Value const& expr, Closure& closure);
	int64_t compileSymbol(Symbol const& symbol, Closure& closure); 
	int64_t compileOp(Call const& call, Closure& closure); 
	int64_t compileCall(Call const& call, Closure& closure); 
	int64_t compileFunctionCall(Call const& call, Closure& closure); 
	int64_t compileExpression(Expression const& values, Closure& closure);
public:
	static Closure compile(State& state, Value const& expr) {
		Compiler compiler(state);
		return compiler.compile(expr);
	}
};

// compilation routines
#endif
