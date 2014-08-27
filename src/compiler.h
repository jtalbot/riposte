
#ifndef COMPILER_H
#define COMPILER_H

#include <string>
#include <sstream>
#include <stdexcept>
#include <string>
#include <map>

#include "common.h"
#include "exceptions.h"
#include "value.h"
#include "type.h"
#include "bc.h"
#include "frontend.h"
#include "interpreter.h"

class Compiler {
private:
	State& state;
	Global& global;

	enum Scope {
		TOPLEVEL,
		CLOSURE,
		PROMISE
	};
	
	Scope scope;
	uint64_t loopDepth;

	struct ValueComp {
		bool operator()(Value const& a, Value const& b) const {
			return a.header < b.header || 
				(a.header == b.header && a.i < b.i);
		}
	};

	std::map<Value, int64_t, ValueComp> constants;

	enum Loc {
		INVALID,
		REGISTER,
		CONSTANT,
		INTEGER
	};

    struct Operand {
        Loc loc;
        int64_t i;
		Operand() : loc(INVALID), i(0) {}
		Operand(Loc loc, int64_t i) : loc(loc), i(i) {}
		Operand(int64_t i) : loc(INTEGER), i(i) {}
		Operand(int i) : loc(INTEGER), i(i) {}
		Operand(size_t i) : loc(INTEGER), i(i) {}

		bool operator==(Operand const& o) const { return loc == o.loc && i == o.i; }
		bool operator!=(Operand const& o) const { return loc != o.loc || i != o.i; }
		std::string toString() const {
			if(loc == INVALID) return "I";
			else if(loc == REGISTER) return intToStr(i) + "R";
			else if(loc == CONSTANT) return intToStr(i) + "C";
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

	Compiler(State& state, Scope scope) : state(state), global(state.global), scope(scope), loopDepth(0), n(0), max_n(0) {}
	
	Code* compile(Value const& expr);			// compile function block, code ends with return
	Operand compile(Value const& expr, Code* code);		// compile into existing code block

	Operand compileConstant(Value const& expr, Code* code);
	Operand compileSymbol(Value const& symbol, Code* code, bool isClosure); 
	Operand compileCall(List const& call, Character const& names, Code* code); 
	Operand compileFunctionCall(Operand function, List const& call, Character const& names, Code* code); 
	Operand compileInternalFunctionCall(List const& call, Code* code); 
	Operand compileExternalFunctionCall(List const& call, Code* code); 
	Operand compileExpression(List const& values, Code* code);
    Operand visible(Operand op);
    Operand invisible(Operand op);

	
	Operand placeInRegister(Operand r);
	int64_t emit(ByteCode::Enum bc, Operand a, Operand b, Operand c);
	void resolveLoopExits(int64_t start, int64_t end, int64_t nextTarget, int64_t breakTarget);
	int64_t encodeOperand(Operand op) const;
	void dumpCode() const;

public:
	static CompiledCall makeCall(State& state, List const& call, Character const& names);

	static Code* compileTopLevel(State& state, Value const& expr) {
		Compiler compiler(state, TOPLEVEL);
		return compiler.compile(expr);
	}
	
	static Code* compilePromise(State& state, Value const& expr) {
		Compiler compiler(state, PROMISE);
		return compiler.compile(expr);
	}
	
    static Prototype* compileClosureBody(State& state, Value const& expr) {
		Compiler compiler(state, CLOSURE);
        Code* code = compiler.compile(expr);
        Prototype* p = new Prototype();
        p->code = code;
        return p;
	}

    static Code* deferPromiseCompilation(State& state, Value const& expr) {
        Code* code = new (Code::Finalize) Code();
        assert(((int64_t)code) % 16 == 0); // do we still need this assumption?
        code->expression = expr;
        return code;
    }

    static void doPromiseCompilation(State& state, Code* code) {
        if(!code->bc.empty())
            return;
 
        Compiler compiler(state, PROMISE);

        // promises use first two registers to pass environment info
        // for replacing promise with evaluated value
        compiler.allocRegister();
        compiler.allocRegister();

        Operand result = compiler.compile(code->expression, code);

        compiler.emit(ByteCode::retp, result, 0, 0);

        for(std::vector<IRNode>::const_iterator i = compiler.ir.begin();
                i != compiler.ir.end(); ++i) {
            code->bc.push_back(Instruction(i->bc,
                compiler.encodeOperand(i->a),
                compiler.encodeOperand(i->b),
                compiler.encodeOperand(i->c)));
        }

        code->registers = compiler.max_n;
    }
};

// compilation routines
#endif
