
#ifndef JIT_H
#define JIT_H

#include <map>
#include <assert.h>

#include "enum.h"
#include "bc.h"
#include "type.h"

#define TRACE_ENUM(_) \
		MAP_BYTECODES(_) \
		_(mov, "mov", ___) \
		_(guardT, "guardT", ___) \
		_(guardF, "guardF", ___) \
		_(read, "read",___) \
		_(phi, "phi", ___) \
		_(jmp, "jmp", ___)

DECLARE_ENUM(TraceOpCode,TRACE_ENUM)

class Thread;

class JIT {

public:

	typedef Instruction const* (*Ptr)(Thread& thread);

	struct IRRef {
		size_t i;
	};

	struct IR {
		TraceOpCode::Enum op;	
		IRRef a, b;
		int64_t i;
		Type::Enum type;
		void dump();
	};

	IR code[1024];
	IRRef pc;
	IRRef loopStart;
	
	unsigned short counters[1024];

	static const unsigned short RECORD_TRIGGER = 4;

	bool recording;
	Instruction const* startPC;

	std::map<int64_t, IRRef> map;

	struct Exit {
		std::map<int64_t, IRRef> m;
		Instruction const* reenter;
	};
	std::map<size_t, Exit > exits;

	JIT() 
		: recording(false)
	{
	}

	void start_recording(Instruction const* startPC) {
		assert(!recording);
		recording = true;
		this->startPC = startPC;
		pc.i = 0;
		map.clear();
	}

	bool loop(Instruction const* pc) {
		return pc == startPC;
	}

	Ptr end_recording(Thread& thread);
	
	void fail_recording() {
		assert(recording);
		recording = false;
	}

	IRRef insert(TraceOpCode::Enum op, IRRef a, Type::Enum type);
	IRRef insert(TraceOpCode::Enum op, IRRef a, int64_t i, Type::Enum type);
	IRRef insert(TraceOpCode::Enum op, IRRef a, IRRef b, Type::Enum type);
	IRRef insert(TraceOpCode::Enum op, int64_t i, Type::Enum type);

	IRRef read(Thread& thread, int64_t a);
	IRRef write(Thread& thread, IRRef a, int64_t c);
	IRRef emit(Thread& thread, TraceOpCode::Enum op, IRRef a, IRRef b, int64_t c);
	void guardT(Thread& thread, Instruction const* reenter);
	void guardF(Thread& thread, Instruction const* reenter);

	void dump();

	Ptr compile(Thread& thread);
};

#endif
