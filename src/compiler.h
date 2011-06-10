
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
	bool inFunction;
	std::vector<uint64_t> slots;

	uint64_t registerDepth;
	uint64_t loopDepth;	

	Compiler(State& state) : state(state) {
		loopDepth = 0;
	}
	
	Closure compile(Value const& expr); 				// compile into new closure
	uint64_t compile(Value const& expr, Closure& closure);		// compile into existing closure

	uint64_t compileConstant(Value const& expr, Closure& closure);
	uint64_t compileSymbol(Symbol const& symbol, Closure& closure); 
	uint64_t compileOp(Call const& call, Closure& closure); 
	uint64_t compileCall(Call const& call, Closure& closure); 
	uint64_t compileFunctionCall(Call const& call, Closure& closure); 
	uint64_t compileExpression(Expression const& values, Closure& closure);
public:
	static Closure compile(State& state, Value const& expr) {
		Compiler compiler(state);
		return compiler.compile(expr);
	}
};

// compilation routines
#endif
