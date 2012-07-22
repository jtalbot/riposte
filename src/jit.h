
#ifndef JIT_H
#define JIT_H

#include <map>
#include <assert.h>

#include "enum.h"
#include "bc.h"
#include "type.h"

#define TRACE_ENUM(_) \
		MAP_BYTECODES(_) \
		_(guardT, "guardT", ___) \
		_(guardF, "guardF", ___) \
		_(read, "read",___) \
		_(phi, "phi", ___) \

DECLARE_ENUM(TraceOpCode,TRACE_ENUM)

class Thread;

class JIT {

public:

	struct IR {

		TraceOpCode::Enum op;	
		IR *a, *b;
		int64_t i;
		IR *firstUse;
		IR *lastUse;
		Type::Enum type;
		void dump(IR* header, IR* body, IR* base);
	};

	IR header[1024];
	IR body[1024];
	size_t pc;
	
	struct IRRef {
		size_t i;
	};

	unsigned short counters[1024];

	static const unsigned short RECORD_TRIGGER = 8;

	bool recording;
	Instruction const* startPC;

	std::map<int64_t, IRRef> map;


	JIT() 
		: recording(false)
	{
	}

	void start_recording(Instruction const* startPC) {
		assert(!recording);
		recording = true;
		this->startPC = startPC;
		pc = 0;
		map.clear();
	}

	bool loop(Instruction const* pc) {
		return pc == startPC;
	}

	void end_recording();
	
	void fail_recording() {
		assert(recording);
		recording = false;
	}

	IRRef insert(TraceOpCode::Enum op, IRRef a, Type::Enum type);
	IRRef insert(TraceOpCode::Enum op, IRRef a, IRRef b, Type::Enum type);
	IRRef insert(TraceOpCode::Enum op, int64_t i, Type::Enum type);

	IRRef read(Thread& thread, int64_t a);
	IRRef write(Thread& thread, IRRef a, int64_t c);
	IRRef emit(Thread& thread, TraceOpCode::Enum op, IRRef a, IRRef b, int64_t c);
	void guardT(Thread& thread);
	void guardF(Thread& thread);

	void dump();
};

#endif
