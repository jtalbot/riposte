
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

	uint64_t loopDepth;

	struct ValueComp {
		bool operator()(Value const& a, Value const& b) const {
			return a.header < b.header || 
				(a.header == b.header && a.i < b.i);
		}
	};

	std::map<Value, int64_t, ValueComp> constants;
    std::vector<Value> constantList;
    std::vector<Value> compiledCalls;

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

	std::vector<Instruction> ir;

	int64_t n, max_n;
	Operand allocRegister() { max_n = std::max(max_n, n+1); return Operand(REGISTER, n++); }
	Operand kill(Operand i) { if(i.loc == REGISTER) { n = std::min(n, i.i); } return i; }
	Operand top() { return Operand(REGISTER, n); }

	Compiler(State& state) : state(state), global(state.global), loopDepth(0), n(0), max_n(0) {}
	
	Operand compile(Value const& expr, Code* code);

	Operand compileConstant(Value expr);
	Operand compileSymbol(Value const& symbol, bool isClosure); 
	Operand compileCall(List const& call, Character const& names, Code* code); 
	Operand compileFunctionCall(Operand function, List const& call, Character const& names, Code* code); 
	Operand compileExpression(List const& values, Code* code);
    Operand visible(Operand op);
    Operand invisible(Operand op);
	Operand placeInRegister(Operand r);

	int64_t emit(ByteCode::Enum bc, Operand a, Operand b, Operand c);
	void resolveLoopExits(int64_t start, int64_t end, int64_t nextTarget, int64_t breakTarget);
	int64_t encodeOperand(Operand op) const;

    Operand emitMissing(ByteCode::Enum bc, List const& call, Code* code);
	Operand emitExternal(ByteCode::Enum bc, List const& call, Code* code); 
    Operand emitAssign(ByteCode::Enum bc, List const& call, Code* code);
    Operand emitPromise(ByteCode::Enum bc, List const& call, Code* code);
    Operand emitFunction(ByteCode::Enum bc, List const& call, Code* code);
    Operand emitReturn(ByteCode::Enum bc, List const& call, Code* code);
    Operand emitFor(ByteCode::Enum bc, List const& call, Code* code);
    Operand emitWhile(ByteCode::Enum bc, List const& call, Code* code);
    Operand emitRepeat(ByteCode::Enum bc, List const& call, Code* code);
    Operand emitNext(ByteCode::Enum bc, List const& call, Code* code);
    Operand emitBreak(ByteCode::Enum bc, List const& call, Code* code);
    Operand emitIf(ByteCode::Enum bc, List const& call, Code* code);
    Operand emitBrace(ByteCode::Enum bc, List const& call, Code* code);
    Operand emitParen(ByteCode::Enum bc, List const& call, Code* code);
    Operand emitTernary(ByteCode::Enum bc, List const& call, Code* code);
    Operand emitBinaryMap(ByteCode::Enum bc, List const& call, Code* code);
    Operand emitBinary(ByteCode::Enum bc, List const& call, Code* code);
    Operand emitUnary(ByteCode::Enum bc, List const& call, Code* code);
    Operand emitNullary(ByteCode::Enum bc, List const& call, Code* code);

    typedef Operand (Compiler::*EmitFn)(ByteCode::Enum bc, List const& call, Code* code);
    
    class EmitTable {
        struct Emit {
            ByteCode::Enum bc;
            EmitFn fn;
        };

        std::map<std::pair<std::string, int>, Emit> emits;

        void add(String name, int args,
                EmitFn fn, ByteCode::Enum bc=ByteCode::done) {
            Emit emit;
            emit.bc = bc;
            emit.fn = fn;
            emits[std::make_pair(std::string(name->s), args)] = emit;
        }

        public:
        Operand operator()(Compiler& compiler, String fn, List const& call, Code* code) const {
            // first try to find an exact match on number of args
            auto i = emits.find(std::make_pair(std::string(fn->s), call.length()-1));
            
            // then try to find a version that can take any number of args
            if(i == emits.end())
                i = emits.find(std::make_pair(std::string(fn->s), -1));
            
            if(i != emits.end()) {
                return (compiler.*(i->second.fn))(
                    i->second.bc, call, code);
            }
            else {
                return compiler.compileFunctionCall(
                    compiler.compileSymbol(call[0], true),
                    call, Character(0), code); 
            }
        }

        EmitTable(); 
    };

    static EmitTable const& GetEmitTable() {
        static EmitTable* emitTable = 0;
        if(!emitTable) emitTable = new EmitTable();
        return *emitTable;
    }

    friend class EmitTable;

public:
	static CompiledCall makeCall(State& state, List const& call, Character const& names);

	static Code* compileExpression(State& state, Value const& expr);
    static Prototype* compileClosureBody(State& state, Value const& expr);

    static Code* deferPromiseCompilation(State& state, Value const& expr);
    static void doPromiseCompilation(State& state, Code* code);
};

// compilation routines
#endif
