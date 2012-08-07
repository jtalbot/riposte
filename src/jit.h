
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
        _(constant, "CONS", ___) \
		_(load, "LOAD", ___) /* loads are type guarded */ \
        /* no store instruction since all stores are implicitly sunk in exits */ \
		_(guardT, "GRD_T", ___) \
		_(guardF, "GRD_F", ___) \
		_(gather, "GATHER", ___) \
		_(phi, "PHI", ___) \
		_(store2, "STORE2", ___) \
		_(nop, "NOP", ___)

DECLARE_ENUM(TraceOpCode,TRACE_ENUM)

class Thread;

class JIT {

public:

    enum State {
        OFF,
        RECORDING_HEADER,
        RECORDING_BODY
    };

	typedef Instruction const* (*Ptr)(Thread& thread);

	typedef size_t IRRef;

	struct IR {
		TraceOpCode::Enum op;
		IRRef a, b, c;
		int64_t i;
		Type::Enum type;
		size_t width;
		size_t group;
        bool liveout;
		void dump();
	};

	unsigned short counters[1024];

	static const unsigned short RECORD_TRIGGER = 4;

    State state;
	Instruction const* startPC;

	std::map<int64_t, IRRef> map;
    std::map<IRRef, IRRef> phi;
    std::vector<IR> code;

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
		state = RECORDING_HEADER;
		this->startPC = startPC;
        code.clear();
        map.clear();
        phi.clear();
	}

    bool record(Thread& thread, Instruction const* pc, bool branch=false) {
        EmitIR(thread, *pc, branch);
        if(pc != startPC) {
            return false;
        } else if(state == RECORDING_HEADER) {
            phi.clear();
            insert(TraceOpCode::loop, (int64_t)0, Type::Promise, 1);
            state = RECORDING_BODY;
            return false;
        }
        else {
            return true;
        }
    }

	bool loop(Instruction const* pc) {
		return pc == startPC;
	}

	Ptr end_recording(Thread& thread);
	
	void fail_recording() {
		assert(state != OFF);
		state = OFF;
	}

	IRRef insert(TraceOpCode::Enum op, IRRef a, Type::Enum type, size_t width);
	IRRef insert(TraceOpCode::Enum op, IRRef a, int64_t i, Type::Enum type, size_t width);
	IRRef insert(TraceOpCode::Enum op, IRRef a, IRRef b, Type::Enum type, size_t width);
	IRRef insert(TraceOpCode::Enum op, IRRef a, IRRef b, IRRef c, Type::Enum type, size_t width);
	IRRef insert(TraceOpCode::Enum op, int64_t i, Type::Enum type, size_t width);

	IRRef load(Thread& thread, int64_t a, Instruction const* reenter);
	IRRef store(Thread& thread, IRRef a, int64_t c);
	IRRef emit(Thread& thread, TraceOpCode::Enum op, IRRef a, IRRef b, int64_t c);
	IRRef emit(Thread& thread, TraceOpCode::Enum op, IRRef a, IRRef b, IRRef c, int64_t d);
	void guardT(Thread& thread, Instruction const* reenter);
	void guardF(Thread& thread, Instruction const* reenter);

    void markLiveOut(Exit const& exit);

	void dump();

	Ptr compile(Thread& thread);

	void specialize();
	void schedule();

    void EmitIR(Thread& thread, Instruction const& inst, bool branch);
    void Replay(Thread& thread);
};

#endif
