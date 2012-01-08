#include "recording.h"
#include "interpreter.h"
#include "ir.h"
#include "ops.h"

#define ENUM_RECORDING_STATUS(_) \
	_(NO_ERROR,"NO_ERROR") \
	_(FALLBACK, "trace falling back to normal interpreter but not exiting") \
	_(RECORD_LIMIT, "maximum record limit reached without executing any traces") \
	_(UNSUPPORTED_OP,"trace encountered unsupported op") \
	_(UNSUPPORTED_TYPE,"trace encountered an unsupported type") \
	_(NO_LIVE_TRACES, "all traces are empty")

DECLARE_ENUM(RecordingStatus,ENUM_RECORDING_STATUS)
DEFINE_ENUM_TO_STRING(RecordingStatus,ENUM_RECORDING_STATUS)

//for brevity
#define REG(thread, i) (*(thread.base+i))

inline static Trace & traceForFuture(Thread & thread, const Value& v) {
	return thread.tracing.traces[v.future.trace_id];
}


#define OP_NOT_IMPLEMENTED(op,...) \
RecordingStatus::Enum op##_record(Thread & thread, Instruction const & inst, Instruction const ** pc) { \
	return RecordingStatus::UNSUPPORTED_OP; \
} \


RecordingStatus::Enum get_record(Thread & thread, Instruction const & inst, Instruction const ** pc) {
	*pc = get_op(thread,inst);
	Value & r = REG(thread,inst.c);

	thread.tracing.UnionWithMaxLiveRegister(thread.base,inst.c);

	if(r.isFuture()) {
		Trace & trace = traceForFuture(thread,r);
		trace.EmitRegOutput(thread.base,inst.c);
		thread.tracing.Commit(thread,trace);
	}
	return RecordingStatus::NO_ERROR;
}

RecordingStatus::Enum kget_record(Thread & thread, Instruction const & inst, Instruction const ** pc) {
	*pc = kget_op(thread,inst);
	return RecordingStatus::NO_ERROR;
}


RecordingStatus::Enum assign_record(Thread & thread, Instruction const & inst, Instruction const ** pc) {
	*pc = assign_op(thread,inst);
	Value& r = REG(thread, inst.c);
	if(r.isFuture()) {
		//Note: this call to makePointer is redundant:
		//if the variable is cached then we could construct the Pointer from the cache
		//otherwise the inline cache is updated, which involves creating a pointer

		//Inline this logic here would make the recorder more fragile, so for now we simply construct the pointer again:
		if(r.isFuture()) {
			Trace & trace = traceForFuture(thread,r);
			trace.EmitVarOutput(thread,thread.frame.environment->makePointer((String)inst.a));
			thread.tracing.Commit(thread,trace);
		}
	}
	thread.tracing.SetMaxLiveRegister(thread.base,inst.c);
	return RecordingStatus::NO_ERROR;
}

OP_NOT_IMPLEMENTED(assign2)

#define CHECK_REG(r) do { \
	Value & v = REG(thread,r);\
	if(v.isFuture()) \
		thread.tracing.Flush(thread,traceForFuture(thread,v)); \
} while(0)
//temporary defines to generate code for checked interpret
#define A CHECK_REG(inst.a);
#define B CHECK_REG(inst.b);
#define C CHECK_REG(inst.c);
#define B_1 CHECK_REG(inst.b - 1);
#define C_1 CHECK_REG(inst.c - 1);

//all operations that first verify their inputs are not futures, eliminating futures by flush the their trace. It then calls into the scalar interpreter to fulfill them
#define CHECKED_INTERPRET(op, checks) \
		RecordingStatus::Enum op##_record(Thread & thread, Instruction const & inst, Instruction const ** pc) { \
			checks \
			*pc = op##_op(thread,inst); \
			return RecordingStatus::NO_ERROR; \
		} \

CHECKED_INTERPRET(eassign, A B C)
CHECKED_INTERPRET(iassign, A B C)
CHECKED_INTERPRET(jt, B)
CHECKED_INTERPRET(jf, B)
CHECKED_INTERPRET(branch, A)
//CHECKED_INTERPRET(subset, A B C)
CHECKED_INTERPRET(subset2, A B C)
CHECKED_INTERPRET(colon, A B C)
CHECKED_INTERPRET(forbegin, B_1)
CHECKED_INTERPRET(forend, B_1)
CHECKED_INTERPRET(UseMethod, A C)
CHECKED_INTERPRET(call, A)
#undef A
#undef B
#undef B_1
#undef C
#undef C_1
#undef CHECKED_INTERPRET

OP_NOT_IMPLEMENTED(internal)

struct LoadCache {
	IRef get(Trace & trace, const Value& v) {
		uint64_t idx = (int64_t) v.p;
		idx += idx >> 32;
		idx += idx >> 16;
		idx += idx >> 8;
		idx &= 0xFF;
		IRef cached = cache[idx];
		if(cached < trace.n_pending_nodes &&
		   trace.nodes[cached].op == IROpCode::loadv &&
		   trace.nodes[cached].loadv.p == v.p) {
			return cached;
		} else {
			return (cache[idx] = trace.EmitLoadV(v.type,v.p));
		}
	}
	IRef cache[256];
};

bool isRecordableType(Value & v) {
	return (v.isFuture() && v.length != 1) || v.isDouble() || v.isInteger() || v.isLogical();
}


bool isRecordableShape(int64_t l) {
	return l >= TRACE_VECTOR_WIDTH && l % TRACE_VECTOR_WIDTH == 0;
}
bool isRecordableShape(Value & v) {
	return isRecordableShape(v.length);
}

//invariants:
//isRecordableType must be true
//v.length == 1 || v.length == trace.length
IRef getRef(Trace & trace, Value & v) {
	static LoadCache load_cache;
	if(v.isFuture()) {
		return v.future.ref;
	} else {
		if(v.length == 1) {
			return trace.EmitLoadC(v.type,v.i);
		} else {
			assert(v.length == trace.length);
			return load_cache.get(trace,v);
		}
	}
}

void coerce_scalar(Type::Enum to, IRNode & n) {
	switch(n.type) {
	case Type::Integer: switch(to) {
		case Type::Double: n.loadc.d = n.loadc.i; break;
		case Type::Logical: n.loadc.l = n.loadc.i; break;
		default: _error("unknown cast"); break;
	} break;
	case Type::Double: switch(to) {
		case Type::Integer: n.loadc.i = (int64_t) n.loadc.d; break;
		case Type::Logical: n.loadc.l = (char) n.loadc.d; break;
		default: _error("unknown cast"); break;
	} break;
	case Type::Logical: switch(to) {
		case Type::Double: n.loadc.d = n.loadc.l; break;
		case Type::Integer: n.loadc.i = n.loadc.l; break;
		default: _error("unknown cast"); break;
	} break;
	default: _error("unknown cast"); break;
	}
	n.type = to;
}

IRef coerce(Trace & trace, Type::Enum dst_type, IRef v) {
	IRNode & n = trace.nodes[v];
	if(dst_type == n.type)
		return v;
	else if(n.op == IROpCode::loadc) {
		coerce_scalar(dst_type,n);
		return v;
	} else {
		return trace.EmitUnary(IROpCode::cast,dst_type,v);
	}
}

RecordingStatus::Enum fallback(Thread & thread, Value & a, Value & b) {
	if(a.isFuture()) {
		thread.tracing.Flush(thread,traceForFuture(thread,a));
	}
	if(b.isFuture()) {
		thread.tracing.Flush(thread,traceForFuture(thread,b));
	}
	return RecordingStatus::FALLBACK;
}

RecordingStatus::Enum subset_record(Thread & thread, Instruction const & inst, Instruction const** pc) {
	CHECK_REG(inst.a);
	Value & b = REG(thread,inst.b);
	if(b.isInteger() || (b.isFuture() && b.future.typ == Type::Integer) ||
	   b.isDouble() || (b.isFuture() && b.future.typ == Type::Double)) {
		Value & a = REG(thread,inst.a);

		//get current trace
		uint64_t trace_shape = b.length;
		Trace & trace = thread.tracing.GetOrAllocateTrace(thread,trace_shape);

		IRef bref = getRef(trace,b);
		Type::Enum rtyp,atyp,btyp;
		rtyp = atyp = a.type;
		btyp = Type::Integer;
		trace.EmitRegOutput(thread.base,inst.c);
		thread.tracing.SetMaxLiveRegister(thread.base,inst.c);
		Future::Init(REG(thread,inst.c),
				 rtyp,
				 trace.length,
				 thread.tracing.TraceID(trace),
				 trace.EmitUnary(IROpCode::gather,rtyp,coerce(trace,btyp,bref),((int64_t)a.p)-8));
		thread.tracing.Commit(thread,trace);
		(*pc)++; 
		return RecordingStatus::NO_ERROR;
	}
	else if(b.isLogical() || (b.isFuture() && b.future.typ == Type::Logical)) {
		Value& a = REG(thread, inst.a);
		uint64_t trace_shape = b.length;
		Trace & trace = thread.tracing.GetOrAllocateTrace(thread,trace_shape);

		IRef bref = getRef(trace,b);
		Type::Enum rtyp,atyp,btyp;
		rtyp = atyp = a.type;
		btyp = Type::Logical;
		trace.EmitRegOutput(thread.base,inst.c);
		thread.tracing.SetMaxLiveRegister(thread.base,inst.c);
		Future::Init(REG(thread,inst.c),
				 rtyp,
				 trace.uniqueShapes--,
				 thread.tracing.TraceID(trace),
				 trace.EmitUnary(IROpCode::filter,rtyp,coerce(trace,btyp,bref),((int64_t)a.p)));
		thread.tracing.Commit(thread,trace);
		(*pc)++; 
		return RecordingStatus::NO_ERROR;
	} 
	else {
		CHECK_REG(inst.b);
		CHECK_REG(inst.c);
		*pc = subset_op(thread,inst);
		return RecordingStatus::NO_ERROR;
	}
}

RecordingStatus::Enum binary_record(ByteCode::Enum bc, IROpCode::Enum op, Thread & thread, Instruction const & inst) {
	Value & a = REG(thread,inst.a);
	Value & b = REG(thread,inst.b);
	//check for valid types
	if(!isRecordableType(a) || !isRecordableType(b))
		return fallback(thread,a,b);

	//check for valid shapes, find common shape if possible
	uint64_t shapes = ((a.length == 1) << 1) | (b.length == 1);
	uint64_t trace_shape;
	switch(shapes) {
	case 0:
		if(a.length != b.length)
			return fallback(thread,a,b);
		/*fallthrough*/
	case 1:
		if(!isRecordableShape(a))
			return fallback(thread,a,b);
		trace_shape = a.length;
		break;
	case 2:
		if(!isRecordableShape(b))
			return fallback(thread,a,b);
		trace_shape = b.length;
		break;
	case 3:
		return fallback(thread,a,b);
		break;
	}
	//get current trace
	Trace & trace = thread.tracing.GetOrAllocateTrace(thread,trace_shape);

	IRef aref = getRef(trace,a);
	IRef bref = getRef(trace,b);
	Type::Enum rtyp,atyp,btyp;
	selectType(bc,trace.nodes[aref].type,trace.nodes[bref].type,&atyp,&btyp,&rtyp);
	trace.EmitRegOutput(thread.base,inst.c);
	thread.tracing.SetMaxLiveRegister(thread.base,inst.c);
	Future::Init(REG(thread,inst.c),
				 rtyp,
				 trace.length,
				 thread.tracing.TraceID(trace),
				 trace.EmitBinary(op,rtyp,coerce(trace,atyp,aref),coerce(trace,btyp,bref)));
	thread.tracing.Commit(thread,trace);
	return RecordingStatus::NO_ERROR;
}

RecordingStatus::Enum unary_record(ByteCode::Enum bc, IROpCode::Enum op, Thread & thread, bool isReduction, Instruction const & inst) {
	Value & a = REG(thread,inst.a);

	if(!isRecordableType(a) || a.length == 1 || !isRecordableShape(a)) {
		if(a.isFuture())
			thread.tracing.Flush(thread,traceForFuture(thread,a));
		return RecordingStatus::FALLBACK;
	}

	Trace & trace = thread.tracing.GetOrAllocateTrace(thread,a.length);

    IRef aref = getRef(trace,a);
    Type::Enum rtyp,atyp;
    selectType(bc,trace.nodes[aref].type,&atyp,&rtyp);
	Future::Init(REG(thread,inst.c),
				 rtyp,
				 isReduction ? 1 : trace.length,
				 thread.tracing.TraceID(trace),
				 trace.EmitUnary(op,rtyp,coerce(trace,atyp,aref)));
	trace.EmitRegOutput(thread.base,inst.c);
	thread.tracing.SetMaxLiveRegister(thread.base,inst.c);
	thread.tracing.Commit(thread,trace);
	return RecordingStatus::NO_ERROR;
}

//all arithmetic binary ops share the same recording implementation
#define BINARY_OP(op,...) RecordingStatus::Enum op##_record(Thread & thread, Instruction const & inst, Instruction const ** pc) { \
	RecordingStatus::Enum status = binary_record(ByteCode :: op,IROpCode :: op, thread, inst);\
	if(RecordingStatus::FALLBACK == status) { \
		*pc = op##_op(thread,inst); \
		return RecordingStatus::NO_ERROR; \
	} \
	if(RecordingStatus::NO_ERROR == status) \
		(*pc)++; \
	return status; \
}
//all unary arithmetic ops share the same implementation as well
#define UNARY_OP(op,...) RecordingStatus::Enum op##_record(Thread & thread, Instruction const & inst, Instruction const ** pc) { \
	RecordingStatus::Enum status = unary_record(ByteCode :: op , IROpCode :: op, thread, false, inst);\
	if(RecordingStatus::FALLBACK == status) { \
		*pc = op##_op(thread,inst); \
		return RecordingStatus::NO_ERROR; \
	} \
	if(RecordingStatus::NO_ERROR == status) \
		(*pc)++; \
	return status; \
}
#define FOLD_OP(op,...) RecordingStatus::Enum op##_record(Thread & thread, Instruction const & inst, Instruction const ** pc) { \
	RecordingStatus::Enum status = unary_record(ByteCode :: op, IROpCode :: op, thread, true, inst);\
	if(RecordingStatus::FALLBACK == status) { \
		*pc = op##_op(thread,inst); \
		return RecordingStatus::NO_ERROR; \
	} \
	if(RecordingStatus::NO_ERROR == status) \
		(*pc)++; \
	return status; \
}


BINARY_ARITH_MAP_BYTECODES(BINARY_OP)
UNARY_ARITH_MAP_BYTECODES(UNARY_OP)
ARITH_FOLD_BYTECODES(FOLD_OP)

BINARY_ORDINAL_MAP_BYTECODES(BINARY_OP)
UNARY_LOGICAL_MAP_BYTECODES(UNARY_OP)
BINARY_LOGICAL_MAP_BYTECODES(BINARY_OP)

OP_NOT_IMPLEMENTED(sland)
OP_NOT_IMPLEMENTED(slor)

OP_NOT_IMPLEMENTED(list)
OP_NOT_IMPLEMENTED(logical1)
OP_NOT_IMPLEMENTED(integer1)
OP_NOT_IMPLEMENTED(double1)
OP_NOT_IMPLEMENTED(complex1)
OP_NOT_IMPLEMENTED(character1)
OP_NOT_IMPLEMENTED(raw1)

OP_NOT_IMPLEMENTED(min)
OP_NOT_IMPLEMENTED(max)
OP_NOT_IMPLEMENTED(any)
OP_NOT_IMPLEMENTED(all)

UNARY_OP(cumsum)
UNARY_OP(cumprod)

OP_NOT_IMPLEMENTED(cummin)
OP_NOT_IMPLEMENTED(cummax)
OP_NOT_IMPLEMENTED(cumany)
OP_NOT_IMPLEMENTED(cumall)

OP_NOT_IMPLEMENTED(mmul)

OP_NOT_IMPLEMENTED(apply)

RecordingStatus::Enum jmp_record(Thread & thread, Instruction const & inst, Instruction const ** pc) {
	//this is just a constant jump, nothing to record
	*pc = jmp_op(thread,inst);
	return RecordingStatus::NO_ERROR;
}

RecordingStatus::Enum function_record(Thread & thread, Instruction const & inst, Instruction const ** pc) {
	*pc = function_op(thread,inst);
	return RecordingStatus::NO_ERROR;
}


//we can extract the type from the future so we can continue a trace beyond a type-check
RecordingStatus::Enum type_record(Thread & thread, Instruction const & inst, Instruction const ** pc) {
	Character c(1);
	// Should have a direct mapping from type to symbol.
	Value & a = REG(thread, inst.a);
	Type::Enum atyp = (a.isFuture()) ? a.future.typ : a.type;
	c[0] = thread.internStr(Type::toString(atyp));
	REG(thread, inst.c) = c;
	(*pc)++;
	return RecordingStatus::NO_ERROR;
}
RecordingStatus::Enum length_record(Thread & thread, Instruction const & inst, Instruction const ** pc) {
	*pc = length_op(thread, inst);
	return RecordingStatus::NO_ERROR;
}
RecordingStatus::Enum missing_record(Thread & thread, Instruction const & inst, Instruction const ** pc) {
	*pc = length_op(thread, inst);
	return RecordingStatus::NO_ERROR;
}
RecordingStatus::Enum ret_record(Thread & thread, Instruction const & inst, Instruction const ** pc) {
	//ret writes a value into a register of the caller's frame. If this value is a future we need to
	//record it as a potential output location
	Value * result = thread.frame.result;
	int64_t offset = result - thread.frame.returnbase;
	int64_t max_live = thread.frame.returnbase - thread.base;
	*pc = ret_op(thread,inst); //warning: ret_op will change 'frame'
	thread.tracing.SetMaxLiveRegister(thread.base,max_live);
	if(result->isFuture()) {
		Trace & trace = traceForFuture(thread,*result);
		trace.EmitRegOutput(thread.base,offset);
		thread.tracing.Commit(thread,trace);
	}
	return RecordingStatus::NO_ERROR;
}

//done forces the trace to flush to ensure that there are no futures when the interpreter exits
RecordingStatus::Enum done_record(Thread & thread, Instruction const & inst, Instruction const ** pc) {
	return RecordingStatus::UNSUPPORTED_OP;
}

RecordingStatus::Enum seq_record(Thread & thread, Instruction const & inst, Instruction const ** pc) {
	Value & a = REG(thread,inst.a);
	Value & b = REG(thread,inst.b);

	if(a.isFuture()) {
		thread.tracing.Flush(thread,traceForFuture(thread,a));
	}
	if(b.isFuture()) {
		thread.tracing.Flush(thread,traceForFuture(thread,b));
	}
	int64_t len = As<Integer>(thread, REG(thread, inst.a))[0];
	int64_t step = As<Integer>(thread, REG(thread, inst.b))[0];

	if(!isRecordableShape(len)) {
		*pc = seq_op(thread,inst); //this isn't ideal, as this will redo the As operators above
	} else {
		Trace & trace = thread.tracing.GetOrAllocateTrace(thread,len);
		Future::Init(REG(thread,inst.c),
				     Type::Integer,
				     len,
				     thread.tracing.TraceID(trace),
				     trace.EmitSpecial(IROpCode::seq,Type::Integer,len,step));
		thread.tracing.SetMaxLiveRegister(thread.base,inst.c);
		(*pc)++;
		thread.tracing.Commit(thread,trace);
	}
	return RecordingStatus::NO_ERROR;
}

//check trace exit condition
// -- do we need to abort due to conditions applying to all opcodes? if so, abort the trace, then exit the recorder
// -- is the trace complete? if so, install the trace, and exit the recorder
// -- otherwise, the recorder continues normally
//returns true if we should continue recording
static RecordingStatus::Enum recording_check_conditions(Thread& thread, Instruction const * inst) {
	if(++thread.tracing.n_recorded_since_last_exec > TRACE_MAX_RECORDED) {
		return RecordingStatus::RECORD_LIMIT;
	} else if(thread.tracing.live_traces.empty()) {
		return RecordingStatus::NO_LIVE_TRACES;
	}
	return RecordingStatus::NO_ERROR;
}

Instruction const * recording_interpret(Thread& thread, Instruction const* pc) {
	RecordingStatus::Enum status = RecordingStatus::NO_ERROR;
	while( true ) {
#define RUN_RECORD(name,str,...) case ByteCode::name: {  /*printf("rec " #name "\n");*/ status = name##_record(thread, *pc,&pc); } break;
		switch(pc->bc) {
			BYTECODES(RUN_RECORD)
		}
#undef RUN_RECORD
		if(   RecordingStatus::NO_ERROR != status
		   || RecordingStatus::NO_ERROR != (status = recording_check_conditions(thread,pc))) {
			if(thread.state.verbose)
				printf("%s op caused trace vm to exit: %s\n",ByteCode::toString(pc->bc),RecordingStatus::toString(status));
			thread.tracing.EndTracing(thread);
			return pc;
		}
		//printf(" .\n");
	}
	return pc;
}
