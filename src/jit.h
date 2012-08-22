
#ifndef JIT_H
#define JIT_H

#include <map>
#include <vector>
#include <tr1/unordered_map>
#include <assert.h>

#include "enum.h"
#include "bc.h"
#include "type.h"
#include "value.h"

#define TRACE_ENUM(_) \
		MAP_BYTECODES(_) \
        _(loop, "LOOP", ___) \
        _(jmp, "JMP", ___) \
        _(constant, "CONS", ___) \
		_(eload, "ELOAD", ___) /* loads are type guarded */ \
		_(sload, "SLOAD", ___) /* loads are type guarded */ \
        _(estore, "ESTORE", ___) \
        _(sstore, "SSTORE", ___) \
        _(GTYPE, "GTYPE", ___) \
        _(GEQ,   "GEQ", ___) \
		_(guardT, "GTRUE", ___) \
		_(guardF, "GFALSE", ___) \
		_(gather, "GATH", ___) \
		_(scatter, "SCAT", ___) \
		_(phi, "PHI", ___) \
		_(castd, "CASTD", ___) \
		_(casti, "CASTI", ___) \
		_(castl, "CASTL", ___) \
        _(length, "LENGTH", ___) \
        _(rep, "REP", ___) \
        _(LOADENV, "LOADENV", ___) \
        _(NEWENV, "NEWENV", ___) \
        _(PUSH, "PUSH", ___) \
        _(POP, "POP", ___) \
        _(dup, "DUP", ___) \
		_(nop, "NOP", ___)

DECLARE_ENUM(TraceOpCode,TRACE_ENUM)

class Thread;

class JIT {

public:

    enum Arity {
        NULLARY,
        UNARY,
        BINARY,
        TERNARY
    };

    enum Group {
        NOP,
        GENERATOR,
        MAP,
        FILTER,
        FOLD,
        SPLIT
    };

    enum State {
        OFF,
        RECORDING
    };

	typedef Instruction const* (*Ptr)(Thread& thread);

	typedef uint64_t IRRef;

    struct Variable {
        IRRef env;
        int64_t i;

        bool operator<(Variable const& o) const {
            return env < o.env || (env == o.env && i < o.i);
        }
    };

	struct IR {
		TraceOpCode::Enum op;
		IRRef a, b, c;

        Type::Enum type;
        size_t width;

        void dump() const;

        bool operator==(IR const& o) const {
            return  op == o.op &&
                    a == o.a &&
                    b == o.b &&
                    c == o.c &&
                    type == o.type &&
                    width == o.width;
        }

        IR(TraceOpCode::Enum op, Type::Enum type, size_t width)
            : op(op), a(0), b(0), c(0), type(type), width(width) {}
        
        IR(TraceOpCode::Enum op, IRRef a, Type::Enum type, size_t width)
            : op(op), a(a), b(0), c(0), type(type), width(width) {}
        
        IR(TraceOpCode::Enum op, IRRef a, IRRef b, Type::Enum type, size_t width)
            : op(op), a(a), b(b), c(0), type(type), width(width) {}
        
        IR(TraceOpCode::Enum op, IRRef a, IRRef b, IRRef c, Type::Enum type, size_t width)
            : op(op), a(a), b(b), c(c), type(type), width(width) {}
	};

	unsigned short counters[1024];

	static const unsigned short RECORD_TRIGGER = 4;

    State state;
	Instruction const* startPC;

    std::map<Environment*, IRRef> envs;
    std::vector<Value> constants;
    std::map<Value, IRRef> constantsMap;

    std::vector<IR> trace;
    std::map<IRRef, Instruction const*> reenters;
    std::vector<IR> code;

    struct Register {
        Type::Enum type;
        size_t length;

        bool operator<(Register const& o) const {
            return type < o.type ||
                   (type == o.type && length < o.length); 
        }
    };

    std::vector<Register> registers;
    std::multimap<Register, size_t> freeRegisters;
    std::vector<int64_t> assignment;
    std::vector<size_t> group;

	struct Exit {
		std::map<Variable, IRRef> o;
		Instruction const* reenter;
	};
	std::map<size_t, Exit > exits;

	JIT() 
		: state(OFF)
	{
	}

	void start_recording(Instruction const* startPC) {
		assert(state == OFF);
		state = RECORDING;
		this->startPC = startPC;
        trace.clear();
        envs.clear();
        constants.clear();
        constantsMap.clear();
        reenters.clear();
	}

    void record(Thread& thread, Instruction const* pc, bool branch=false) {
        EmitIR(thread, *pc, branch);
    }

	bool loop(Thread& thread, Instruction const* pc) {
		if(pc == startPC) {
            EmitIR(thread, *pc, false);
            return true;
        }
        else {
            EmitIR(thread, *pc, false);
            return false;
        }
	}

	Ptr end_recording(Thread& thread);
	
	void fail_recording() {
		assert(state != OFF);
		state = OFF;
	}

	IRRef insert(
        std::vector<IR>& t,
        TraceOpCode::Enum op, 
        IRRef a, 
        IRRef b, 
        IRRef c,
        Type::Enum type, 
        size_t width);
    
    IRRef duplicate(IR const& ir, std::vector<IRRef> const& forward);
	IRRef load(Thread& thread, int64_t a, Instruction const* reenter);
    IRRef constant(Value const& v);
	Variable intern(Thread& thread, int64_t a);
	IRRef rep(IRRef a, size_t width);
	IRRef cast(IRRef a, Type::Enum type);
    IRRef store(Thread& thread, IRRef a, int64_t c);
	IRRef EmitUnary(TraceOpCode::Enum op, IRRef a, Type::Enum rty, Type::Enum mty);
	IRRef EmitBinary(TraceOpCode::Enum op, IRRef a, IRRef b, Type::Enum rty, Type::Enum maty, Type::Enum mbty);
	IRRef EmitTernary(TraceOpCode::Enum op, IRRef a, IRRef b, IRRef c, Type::Enum rty, Type::Enum maty, Type::Enum mbty, Type::Enum mcty);


    void emitCall(IRRef a, Function const& func, Environment* env, Instruction const* inst) {
        IRRef guard = insert(trace, TraceOpCode::GEQ, a, (IRRef)func.prototype(), 0, Type::Function, 1);
        reenters[guard] = inst;
        IRRef ne = insert(trace, TraceOpCode::NEWENV, 0, 0, 0, Type::Environment, 1);
        envs[env] = ne;
    }

    void emitPush(Environment* env) {
        insert(trace, TraceOpCode::PUSH, envs[env], 0, 0, Type::Promise, 1);
    }

    void storeArg(Environment* env, String name, Value const& v) {
        estore(constant(v), env, name);
    }

    IRRef getEnv(Environment* env) {
        std::map<Environment*, IRRef>::const_iterator i;
        i = envs.find(env);
        if(i != envs.end())
            return i->second;
        else {
            IRRef e = insert(trace, TraceOpCode::LOADENV, (IRRef)env, 0, 0, Type::Environment, 1);
            envs[env] = e;
            return e;
        }
    }

    Variable getVar(IRRef env, String name) {
        Variable v = {env, (int64_t)name};
        return v;
    }
    
    void estore(IRRef a, Environment* env, String name) {
        IRRef e = getEnv(env);
        Variable v = getVar(e, name);
        insert(trace, TraceOpCode::estore, a, v.env, v.i, trace[a].type, trace[a].width);
    }

    void markLiveOut(Exit const& exit);

	void dump(Thread& thread, std::vector<IR> const&);

	Ptr compile(Thread& thread);

	void specialize();
	void schedule();

    void EmitIR(Thread& thread, Instruction const& inst, bool branch);
    void EmitOptIR(IRRef i, std::vector<IRRef>& forward, std::map<Variable, IRRef>& map, std::map<Variable, IRRef>& stores, std::tr1::unordered_map<IR, IRRef>& cse);
    void Replay(Thread& thread);

    void AssignRegister(size_t index);
    void PreferRegister(size_t index, size_t share);
    void ReleaseRegister(size_t index);
    void RegisterAssignment();

    IRRef Insert(std::vector<IR>& code, std::tr1::unordered_map<IR, IRRef>& cse, IR ir);
    IR Normalize(IR ir);

	template< template<class X> class Group >
	IRRef EmitUnary(TraceOpCode::Enum op, IRRef a) {
		Type::Enum aty = trace[a].type;

        #define EMIT(TA)                  \
        if(aty == Type::TA)                     \
            return EmitUnary(op, a,             \
                Group<TA>::R::VectorType,    \
                Group<TA>::MA::VectorType);
        UNARY_TYPES(EMIT)
        #undef EMIT
        _error("Unknown type in EmitUnary");
    }

	template< template<class X, class Y> class Group >
	IRRef EmitBinary(TraceOpCode::Enum op, IRRef a, IRRef b) {
		Type::Enum aty = trace[a].type;
		Type::Enum bty = trace[b].type;

        #define EMIT(TA, TB)                 \
        if(aty == Type::TA && bty == Type::TB)      \
            return EmitBinary(op, a, b,             \
                Group<TA,TB>::R::VectorType,        \
                Group<TA,TB>::MA::VectorType,       \
                Group<TA,TB>::MB::VectorType);
        BINARY_TYPES(EMIT)
        #undef EMIT
        _error("Unknown type pair in EmitBinary");
    }
    
    template< template<class X, class Y, class Z> class Group >
    IRRef EmitTernary(TraceOpCode::Enum op, IRRef a, IRRef b, IRRef c) {
		Type::Enum aty = trace[a].type;
		Type::Enum bty = trace[b].type;
		Type::Enum cty = trace[c].type;

        #define EMIT(TA, TB, TC)                            \
        if(aty == Type::TA && bty == Type::TB && cty == Type::TC)   \
            return EmitTernary(op, a, b, c,                         \
                Group<TA,TB,TC>::R::VectorType,                     \
                Group<TA,TB,TC>::MA::VectorType,                    \
                Group<TA,TB,TC>::MB::VectorType,                    \
                Group<TA,TB,TC>::MC::VectorType);
        TERNARY_TYPES(EMIT)
        #undef EMIT
        _error("Unknown type pair in EmitTernary");
    }
#undef TYPES_TMP

};

namespace std {
    namespace tr1 {
    template<>
        struct hash<JIT::IR> {
            size_t operator()(JIT::IR const& key) const {
                return key.op ^ key.a ^ key.b ^ key.c ^ key.width ^ key.type;
            }
        };
    }
}

#endif
