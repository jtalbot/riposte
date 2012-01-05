
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

	struct Register {
		enum Type { CONSTANT, VARIABLE, TEMP };
		Type type;
		Register() { type = VARIABLE; }
		Register(Type type) : type(type) {}
	};

	struct Scope {
		Environment* env;
		bool topLevel;
		std::vector<Register> registers;
		int64_t maxRegister;
		Character parameters;

		Scope() : env(0), topLevel(false), maxRegister(-1) {}

		int64_t live() const { return registers.size()-1; }
		int64_t allocRegister(Register::Type type) { int64_t r = registers.size(); registers.push_back(Register(type)); maxRegister = maxRegister > r ? maxRegister : r; return r; }
		void deadAfter(int64_t i) { registers.resize(i+1); }
	}; 

	std::vector<Scope> scopes;

	int64_t loopDepth;	

	Compiler(State& state) : state(state) {
		loopDepth = 0;
	}
	
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
public:
	static Prototype* compile(State& state, Value const& expr) {
		Compiler compiler(state);
		Scope scope;
		scope.topLevel = true;
		compiler.scopes.push_back(scope);
		return compiler.compile(expr);
	}
};

// compilation routines
#endif
