
#ifndef JIT_H
#define JIT_H

#include <map>
#include <vector>
#include <assert.h>

#include "enum.h"
#include "bc.h"
#include "type.h"

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
		_(nop, "NOP", ___)

DECLARE_ENUM(TraceOpCode,TRACE_ENUM)

class Thread;

class JIT {

public:

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

    std::vector<int64_t> assignment;
    std::vector<size_t> registers;
    std::multimap<size_t, size_t> freeRegisters;

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
	IRRef store(Thread& thread, IRRef a, int64_t c);
	IRRef emit(Thread& thread, TraceOpCode::Enum op, IRRef a, IRRef b, int64_t c);
	IRRef emit(Thread& thread, TraceOpCode::Enum op, IRRef a, IRRef b, IRRef c, int64_t d);

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
};

#endif
