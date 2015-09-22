
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

    struct Tail {
        enum Type {
            None,
            Return,
            Done
        };

        enum Visibility {
            Default = 0,
            Visible = 1,
            Invisible = 2
        };

        Type type;
        Visibility visibility;

        Tail(Type type, Visibility visibility)
            : type(type), visibility(visibility) {}

        Tail()
            : type(Type::None), visibility(Visibility::Default) {}
    };

    Tail visible(Tail t) {
        return Tail { t.type, t.visibility == Tail::Default ? Tail::Visible : t.visibility };
    }

    Tail invisible(Tail t) {
        return Tail { t.type, t.visibility == Tail::Default ? Tail::Invisible : t.visibility };
    }

	std::vector<Instruction> ir;

	int64_t n, max_n;
	Operand allocRegister() { max_n = std::max(max_n, n+1); return Operand(REGISTER, n++); }
	Operand kill(Operand i) { if(i.loc == REGISTER) { n = std::min(n, i.i); } return i; }
	Operand top() { return Operand(REGISTER, n); }

	Compiler(State& state) : state(state), global(state.global), loopDepth(0), n(0), max_n(0) {}
	
	Operand compile(Value const& expr, Tail tail, Code* code);

	Operand compileConstant(Value expr, Tail tail);
	Operand compileSymbol(Value const& symbol, bool isClosure, Tail tail); 
	Operand compileCall(List const& call, Character const& names, Tail tail, Code* code); 
	Operand compileFunctionCall(Operand function, List const& call, Character const& names, Tail tail, Code* code); 
	Operand compileExpression(List const& values, Tail tail, Code* code);
	Operand placeInRegister(Operand r);

	int64_t emit(ByteCode::Enum bc, Operand a, Operand b, Operand c);
	void resolveLoopExits(int64_t start, int64_t end, int64_t nextTarget, int64_t breakTarget);
	int64_t encodeOperand(Operand op) const;

    Operand emitTail(Operand op, Tail tail);

    Operand emitMissing(ByteCode::Enum bc, List const& call, Tail tail, Code* code);
	Operand emitExternal(ByteCode::Enum bc, List const& call, Tail tail, Code* code); 
    Operand emitAssign(ByteCode::Enum bc, List const& call, Tail tail, Code* code);
    Operand emitPromise(ByteCode::Enum bc, List const& call, Tail tail, Code* code);
    Operand emitFunction(ByteCode::Enum bc, List const& call, Tail tail, Code* code);
    Operand emitReturn(ByteCode::Enum bc, List const& call, Tail tail, Code* code);
    Operand emitFor(ByteCode::Enum bc, List const& call, Tail tail, Code* code);
    Operand emitWhile(ByteCode::Enum bc, List const& call, Tail tail, Code* code);
    Operand emitRepeat(ByteCode::Enum bc, List const& call, Tail tail, Code* code);
    Operand emitNext(ByteCode::Enum bc, List const& call, Tail tail, Code* code);
    Operand emitBreak(ByteCode::Enum bc, List const& call, Tail tail, Code* code);
    Operand emitIf(ByteCode::Enum bc, List const& call, Tail tail, Code* code);
    Operand emitBrace(ByteCode::Enum bc, List const& call, Tail tail, Code* code);
    Operand emitParen(ByteCode::Enum bc, List const& call, Tail tail, Code* code);
    Operand emitTernary(ByteCode::Enum bc, List const& call, Tail tail, Code* code);
    Operand emitBinaryMap(ByteCode::Enum bc, List const& call, Tail tail, Code* code);
    Operand emitBinary(ByteCode::Enum bc, List const& call, Tail tail, Code* code);
    Operand emitUnary(ByteCode::Enum bc, List const& call, Tail tail, Code* code);
    Operand emitNullary(ByteCode::Enum bc, List const& call, Tail tail, Code* code);
    Operand emitVisible(ByteCode::Enum bc, List const& call, Tail tail, Code* code);
    Operand emitInvisible(ByteCode::Enum bc, List const& call, Tail tail, Code* code);

    typedef Operand (Compiler::*EmitFn)(ByteCode::Enum bc, List const& call, Tail tail, Code* code);
    
    class EmitTable
    {
        struct Emit {
            ByteCode::Enum bc;
            EmitFn fn;
        };

        struct PairHash {
            size_t operator()(std::pair<String, int> s) const {
                return Hash(s.first) ^ (size_t)s.second;
            }
        };

        struct PairEq {
            bool operator()(std::pair<String, int> s,
                            std::pair<String, int> t) const {
                return s.second == t.second &&
                       s.first->length == t.first->length &&
                       strncmp(s.first->s, t.first->s, s.first->length) == 0;
            }
        };

        std::unordered_map<
            std::pair<String, int>, Emit, PairHash, PairEq> emits;

        void add(String name, int args,
                EmitFn fn, ByteCode::Enum bc=ByteCode::done) {
            Emit emit;
            emit.bc = bc;
            emit.fn = fn;
            emits[std::make_pair(name, args)] = emit;
        }

        public:
        Operand operator()(Compiler& compiler, String fn, List const& call, Tail tail, Code* code) const {
            // first try to find an exact match on number of args
            auto i = emits.find(std::make_pair(fn, call.length()-1));
            
            // then try to find a version that can take any number of args
            if(i == emits.end())
                i = emits.find(std::make_pair(fn, -1));
            
            if(i != emits.end()) {
                return (compiler.*(i->second.fn))(
                    i->second.bc, call, tail, code);
            }
            else {
                return compiler.compileFunctionCall(
                    compiler.compileSymbol(call[0], true, Tail()),
                    call, Character(0), tail, code); 
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

    static void compilePromise(State& state, Code* code);

public:
	static CompiledCall makeCall(State& state, List const& call, Character const& names);

	static Code* compileExpression(State& state, Value const& expr);
    static Prototype* compileClosureBody(State& state, Value const& expr);

    static Code* deferPromiseCompilation(State& state, Value const& expr);
    static void doPromiseCompilation(State& state, Code* code)
    {
        if(code->bc.length() == 0)
            compilePromise(state, code);
    }

};

// compilation routines
#endif
