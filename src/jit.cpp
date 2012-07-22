
#include "jit.h"
#include "bc.h"
#include "interpreter.h"
#include "call.h"

DEFINE_ENUM_TO_STRING(TraceOpCode, TRACE_ENUM)

JIT::IRRef JIT::insert(TraceOpCode::Enum op, int64_t i, Type::Enum type) {
	header[pc] = 	(IR) { op, 0, 0, i, 0, header + pc, type }; 
	body[pc] = 	(IR) { op, 0, 0, i, 0, body + pc, type }; 
	return (IRRef) { pc++ };
}

JIT::IRRef JIT::insert(TraceOpCode::Enum op, IRRef a, Type::Enum type) {
	header[pc] = 	(IR) { op, header+a.i, 0, 0, 0, header + pc, type };
	body[pc] = 	(IR) { op, body+a.i,    0, 0, 0, body + pc, type }; 
	return (IRRef) { pc++ };
}

JIT::IRRef JIT::insert(TraceOpCode::Enum op, IRRef a, IRRef b, Type::Enum type) {
	header[pc] = 	(IR) { op, header+a.i, header+b.i, 0, 0, header + pc, type };
	body[pc] = 	(IR) { op, body+a.i,    body+b.i,   0, 0, body + pc, type }; 
	return (IRRef) { pc++ };
}

JIT::IRRef JIT::read(Thread& thread, int64_t a) {
	
	std::map<int64_t, IRRef>::const_iterator i;
	i = map.find(a);
	if(i != map.end()) {
		return i->second;
	}
	else {
		OPERAND(operand, a);
		Type::Enum type = operand.type;
		header[pc] = 	(IR) { TraceOpCode::read, 0, 0, a, 0, header + pc, type}; 
		body[pc] = 	(IR) { TraceOpCode::phi,  0, 0, a, 0, body + pc, type }; 
		map[a] = (IRRef) { pc };
		return (IRRef) { pc++ };
	}
}

JIT::IRRef JIT::emit(Thread& thread, TraceOpCode::Enum op, IRRef a, IRRef b, int64_t c) {
	Value const& v = OUT(thread, c);
	IRRef ir = insert(op, a, b, v.type);
	map[c] = ir;
	return ir;
}

JIT::IRRef JIT::write(Thread& thread, IRRef a, int64_t c) {
	map[c] = a;
	return a;
}

void JIT::guardF(Thread& thread) {
	IRRef p = (IRRef) { pc-1 }; 
	insert(TraceOpCode::guardF, p, Type::Promise );
}

void JIT::guardT(Thread& thread) {
	IRRef p = (IRRef) { pc-1 }; 
	insert(TraceOpCode::guardT, p, Type::Promise );
}

void JIT::end_recording() {
	assert(recording);
	recording = false;

	// fix up phi nodes, do this along with another pass
	for(size_t i = 0; i < pc; i++) {
		if(body[i].op == TraceOpCode::phi) {
			body[i].a = header + map.find(body[i].i)->second.i;
			body[i].b = body + map.find(body[i].i)->second.i;
		}
	}
}

void JIT::IR::dump(IR* header, IR* body, IR* base) {
	printf("(%s) ", Type::toString(type));
	switch(op) {
		case TraceOpCode::read: {
			if(i <= 0)
				printf("read\t %d", i);
			else
				printf("read\t %s", (String)i);
		} break;
		case TraceOpCode::phi: {
			printf("phi\t^%d\t_%d", a-header, b-body);
		} break;
		case TraceOpCode::guardF: {
			printf("guardF\t %d", a-base);
		} break;
		case TraceOpCode::guardT: {
			printf("guardT\t %d", a-base);
		} break;
		default: {
			printf("%s\t %d\t %d", TraceOpCode::toString(op), a-base, b-base);
		} break;
	};
}

void JIT::dump() {
	for(size_t i = 0; i < pc; i++) {
		printf("h %d: ", i);
		header[i].dump(header, body, header);
		printf("\n");
	}
	for(size_t i = 0; i < pc; i++) {
		printf("b %d: ", i);
		body[i].dump(header, body, body);
		printf("\n");
	}
	printf("\n");
}
