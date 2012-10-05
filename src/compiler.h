
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
#include "frontend.h"

class Compiler {
private:
	Thread& thread;
	State& state;

	enum Scope {
		TOPLEVEL,
		FUNCTION,
		PROMISE
	};
	
	Scope scope;
	uint64_t loopDepth;
	std::map<Value, int64_t> constants;

	enum Loc {
		INVALID,
		REGISTER,
		VARIABLE,
		CONSTANT,
		INTEGER
	};

	struct Operand {
		Loc loc;
		union {
			int64_t i;
			String s;
		};
		Operand() : loc(INVALID), i(0) {}
		Operand(Loc loc, int64_t i) : loc(loc), i(i) {}
		Operand(Loc loc, String s) : loc(loc), s(s) {}
		Operand(int64_t i) : loc(INTEGER), i(i) {}
		Operand(int i) : loc(INTEGER), i(i) {}
		Operand(size_t i) : loc(INTEGER), i(i) {}

		bool operator==(Operand const& o) const { return loc == o.loc && i == o.i; }
		bool operator!=(Operand const& o) const { return loc != o.loc || i != o.i; }
		std::string toString() const {
			if(loc == INVALID) return "I";
			else if(loc == REGISTER) return intToStr(i) + "R";
			else if(loc == CONSTANT) return intToStr(i) + "C";
			else if(loc == VARIABLE) return std::string(s);
			else return intToStr(i) + "L";
		}
	};

	struct IRNode {
		ByteCode::Enum bc;
		Operand a, b, c;
		IRNode(ByteCode::Enum bc, Operand a, Operand b, Operand c) :
			bc(bc), a(a), b(b), c(c) {}
	};

	std::vector<IRNode> ir;

	int64_t n, max_n;
	Operand allocRegister() { max_n = std::max(max_n, n+1); return Operand(REGISTER, n++); }
	Operand kill(Operand i) { if(i.loc == REGISTER) { n = std::min(n, i.i); } return i; }
	Operand top() { return Operand(REGISTER, n); }

	Compiler(Thread& thread, Scope scope) : thread(thread), state(thread.state), scope(scope), loopDepth(0), n(0), max_n(0) {}
	
	Prototype* compile(Value const& expr);			// compile function block, code ends with return
	Operand compile(Value const& expr, Prototype* code);		// compile into existing code block

	Operand compileConstant(Value const& expr, Prototype* code);
	Operand compileSymbol(Value const& symbol, Prototype* code); 
	Operand compileCall(List const& call, Character const& names, Prototype* code); 
	Operand compileFunctionCall(List const& call, Character const& names, Prototype* code); 
	Operand compileInternalFunctionCall(Object const& o, Prototype* code); 
	Operand compileExpression(List const& values, Prototype* code);
	
	CompiledCall makeCall(List const& call, Character const& names);

	Operand placeInRegister(Operand r);
	Operand forceInRegister(Operand r);
	int64_t emit(ByteCode::Enum bc, Operand a, Operand b, Operand c);
	void resolveLoopExits(int64_t start, int64_t end, int64_t nextTarget, int64_t breakTarget);
	int64_t encodeOperand(Operand op, int64_t n) const;
	void dumpCode() const;

public:
	static Prototype* compileTopLevel(Thread& thread, Value const& expr) {
		Compiler compiler(thread, TOPLEVEL);
		return compiler.compile(expr);
	}
	
	static Prototype* compileFunctionBody(Thread& thread, Value const& expr) {
		Compiler compiler(thread, FUNCTION);
		return compiler.compile(expr);
	}
	
	static Prototype* compilePromise(Thread& thread, Value const& expr) {
		Compiler compiler(thread, PROMISE);
		return compiler.compile(expr);
	}
};

// compilation routines
#endif
