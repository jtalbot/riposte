
#ifndef JIT_H
#define JIT_H

#include <map>
#include <vector>
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
		_(load, "LOAD", ___) /* loads are type guarded */ \
        _(store, "STORE", ___) \
        _(GTYPE, "GTYPE", ___) \
		_(guardT, "GTRUE", ___) \
		_(guardF, "GFALSE", ___) \
		_(gather, "GATH", ___) \
		_(scatter, "SCAT", ___) \
		_(phi, "PHI", ___) \
		_(cast, "CAST", ___) \
        _(length, "LENGTH", ___) \
        _(rep, "REP", ___) \
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

	typedef size_t IRRef;

	struct IR {
		TraceOpCode::Enum op;
		IRRef a, b, c;
        int64_t target;

        Type::Enum type;
        size_t width;

		size_t group;
        void dump();
	};

	unsigned short counters[1024];

	static const unsigned short RECORD_TRIGGER = 4;

    State state;
	Instruction const* startPC;

    std::map<int64_t, IRRef> loads;
	std::map<int64_t, IRRef> map;
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

	struct Exit {
		std::map<int64_t, IRRef> o;
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
        code.clear();
        map.clear();
        loads.clear();
	}

    void record(Thread& thread, Instruction const* pc) {
        EmitIR(thread, *pc, false);
    }

	bool loop(Thread& thread, Instruction const* pc, bool branch=false) {
		if(pc == startPC) {
            insert(TraceOpCode::loop, 0, 0, 0, 0, Type::Promise, 1);
            EmitIR(thread, *pc, branch);
            return true;
        }
        else {
            EmitIR(thread, *pc, branch);
            return false;
        }
	}

	Ptr end_recording(Thread& thread);
	
	void fail_recording() {
		assert(state != OFF);
		state = OFF;
	}

	IRRef insert(
        TraceOpCode::Enum op, 
        IRRef a, 
        IRRef b, 
        IRRef c,
        int64_t target, 
        Type::Enum type, 
        size_t width);

	IRRef load(Thread& thread, int64_t a, Instruction const* reenter);
	IRRef rep(IRRef a, size_t width);
	IRRef cast(IRRef a, Type::Enum type);
	IRRef store(Thread& thread, IRRef a, int64_t c);
	IRRef EmitUnary(TraceOpCode::Enum op, IRRef a, Type::Enum rty, Type::Enum mty);
	IRRef EmitBinary(TraceOpCode::Enum op, IRRef a, IRRef b, Type::Enum rty, Type::Enum maty, Type::Enum mbty);
	IRRef EmitTernary(TraceOpCode::Enum op, IRRef a, IRRef b, IRRef c, Type::Enum rty, Type::Enum maty, Type::Enum mbty, Type::Enum mcty);

    void markLiveOut(Exit const& exit);

	void dump();

	Ptr compile(Thread& thread);

	void specialize();
	void schedule();

    void EmitIR(Thread& thread, Instruction const& inst, bool branch);
    void Replay(Thread& thread);

    void AssignRegister(size_t index);
    void PreferRegister(size_t index, size_t share);
    void ReleaseRegister(size_t index);
    void RegisterAssignment();

#define TYPES_TMP(_) \
    _(Double) \
    _(Integer) \
    _(Logical) 

	template< template<class X> class Group >
	IRRef EmitUnary(TraceOpCode::Enum op, IRRef a) {
		Type::Enum aty = code[a].type;

        #define EMIT_UNARY(TA)                  \
        if(aty == Type::TA)                     \
            return EmitUnary(op, a,             \
                Group<TA>::R::VectorType,    \
                Group<TA>::MA::VectorType);
        TYPES_TMP(EMIT_UNARY)
        #undef EMIT_BINARY
    }
#undef TYPES_TMP

#define TYPES_TMP(_) \
    _(Double, Double) \
    _(Integer, Integer) \
    _(Logical, Logical) \
    _(Double, Integer) \
    _(Integer, Double) \
    _(Double, Logical) \
    _(Logical, Double) \
    _(Integer, Logical) \
    _(Logical, Integer)

	template< template<class X, class Y> class Group >
	IRRef EmitBinary(TraceOpCode::Enum op, IRRef a, IRRef b) {
		Type::Enum aty = code[a].type;
		Type::Enum bty = code[b].type;

        #define EMIT_BINARY(TA, TB)                 \
        if(aty == Type::TA && bty == Type::TB)      \
            return EmitBinary(op, a, b,             \
                Group<TA,TB>::R::VectorType,        \
                Group<TA,TB>::MA::VectorType,       \
                Group<TA,TB>::MB::VectorType);
        TYPES_TMP(EMIT_BINARY)
        #undef EMIT_BINARY
    }
#undef TYPES_TMP
};

#endif
