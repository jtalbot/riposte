
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

	struct Register {
		enum Type { CONSTANT, VARIABLE, TEMP };
		Type type;
		Register() { type = VARIABLE; }
		Register(Type type) : type(type) {}
	};

	struct Scope {
		bool topLevel;
		std::vector<Register> registers;
		int64_t maxRegister;
		std::vector<Symbol> symbols;
		Value parameters;

		Scope() : topLevel(false), maxRegister(-1) {}

		int64_t live() const { return registers.size()-1; }
		int64_t allocRegister(Register::Type type) { int64_t r = registers.size(); registers.push_back(Register(type)); maxRegister = maxRegister > r ? maxRegister : r; return r; }
		void deadAfter(int64_t i) { registers.resize(i+1); }
	}; 

	std::vector<Scope> scopes;

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
	
	CompiledCall makeCall(Call const& call);

	void emit(Code* code, ByteCode bc, int64_t a, int64_t b, int64_t c);
	int64_t getSlot(Symbol s);
public:
	static Code* compile(State& state, Value const& expr) {
		Compiler compiler(state);
		Scope scope;
		scope.topLevel = true;
		compiler.scopes.push_back(scope);
		return compiler.compile(expr);
	}
	
	static Code* compile(State& state, Value const& expr, Environment* env) {
		Compiler compiler(state);
		Scope scope;
		scope.topLevel = true;
		compiler.scopes.push_back(scope);
		return compiler.compile(expr);
	}
};

// compilation routines
#endif
