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

// Figure out the type of the operation given an input type

template<class X>
void unaryResultType(Type::Enum& mtype, Type::Enum& rtype) {
	mtype = X::MA::VectorType;
	rtype = X::R::VectorType;
}


void selectType(ByteCode::Enum bc, Type::Enum input, Type::Enum& mtype, Type::Enum& rtype) {
	switch(bc) {
#define UNARY_OP(Name, String, Op, Group, ...) \
		case ByteCode::Name: \
		switch(input) { \
			case Type::Logical: unaryResultType< Name##VOp<Logical> >(mtype, rtype); \
			case Type::Integer: unaryResultType< Name##VOp<Integer> >(mtype, rtype); \
			case Type::Double:  unaryResultType< Name##VOp<Double> >(mtype, rtype); \
			default: _error("Unsupported type in JIT"); \
		} break;
UNARY_FOLD_SCAN_BYTECODES(UNARY_OP)
#undef UNARY_OP
		default: _error("Called unaryResultType with non-unary bytecode");
	}
}

template< template<class X, class Y> class Group, class X, class Y>
void binaryResultType(Type::Enum& matype, Type::Enum& mbtype, Type::Enum& rtype) {
	matype = Group<X, Y>::MA::VectorType;
	mbtype = Group<X, Y>::MB::VectorType;
	rtype = Group<X, Y>::R::VectorType;
}


void selectType(ByteCode::Enum bc, Type::Enum inputa, Type::Enum inputb, Type::Enum& matype, Type::Enum& mbtype, Type::Enum& rtype) {
	switch(bc) {
#define BINARY_OP(Name, String, Op, Group, ...) \
		case ByteCode::Name: \
		if(inputa == Type::Logical) { \
			if(inputb == Type::Logical) \
				binaryResultType<Group, Logical, Logical>(matype, mbtype, rtype); \
			else if(inputb == Type::Integer) \
				binaryResultType<Group, Logical, Integer>(matype, mbtype, rtype); \
			else if(inputb == Type::Double) \
				binaryResultType<Group, Logical, Double>(matype, mbtype, rtype); \
		} else if(inputa == Type::Integer) { \
			if(inputb == Type::Logical) \
				binaryResultType<Group, Integer, Logical>(matype, mbtype, rtype); \
			else if(inputb == Type::Integer) \
				binaryResultType<Group, Integer, Integer>(matype, mbtype, rtype); \
			else if(inputb == Type::Double) \
				binaryResultType<Group, Integer, Double>(matype, mbtype, rtype); \
		} else if(inputa == Type::Double) { \
			if(inputb == Type::Logical) \
				binaryResultType<Group, Double, Logical>(matype, mbtype, rtype); \
			else if(inputb == Type::Integer) \
				binaryResultType<Group, Double, Integer>(matype, mbtype, rtype); \
			else if(inputb == Type::Double) \
				binaryResultType<Group, Double, Double>(matype, mbtype, rtype); \
		} else _error("Unsupported type in JIT"); 
BINARY_BYTECODES(BINARY_OP)
#undef BINARY_OP
		default: _error("Called binaryResultType with non-binary bytecode");
	}
}

//for brevity
#define REG(thread, i) (*(thread.base+i))

#define OP_NOT_IMPLEMENTED(op,...) \
RecordingStatus::Enum op##_record(Thread & thread, Instruction const & inst, Instruction const ** pc) { \
	return RecordingStatus::UNSUPPORTED_OP; \
} \

Type::Enum getType(Value& v) {
	if(v.isFuture()) return v.future.typ;
	else return v.type;
}



/*
RecordingStatus::Enum get_record(Thread & thread, Instruction const & inst, Instruction const ** pc) {
	*pc = get_op(thread,inst);
	Value & r = REG(thread,inst.c);

	thread.trace.UnionWithMaxLiveRegister(thread.base,inst.c);

	if(r.isFuture()) {
		thread.trace.RegOutput(r.future.ref, thread.base, inst.c);
		thread.trace.Commit(thread);
	}
	return RecordingStatus::NO_ERROR;
}
*/
RecordingStatus::Enum assign_record(Thread & thread, Instruction const & inst, Instruction const ** pc) {
	*pc = assign_op(thread,inst);
	Value& r = REG(thread, inst.c);
	//Inline this logic here would make the recorder more fragile, so for now we simply construct the pointer again:
	if(r.isFuture()) {
		thread.trace.VarOutput(r.future.ref, thread.frame.environment->makePointer((String)inst.a));
		thread.trace.Commit(thread);
	}
	thread.trace.SetMaxLiveRegister(thread.base,inst.c);
	return RecordingStatus::NO_ERROR;
}

OP_NOT_IMPLEMENTED(assign2)

#define CHECK_REG(r) do { \
	thread.trace.Force(thread, REG(thread, r)); \
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
CHECKED_INTERPRET(jc, B)
CHECKED_INTERPRET(branch, A)
//CHECKED_INTERPRET(subset, A B C)
CHECKED_INTERPRET(subset2, A B C)
CHECKED_INTERPRET(colon, A B C)
CHECKED_INTERPRET(forbegin, B_1)
CHECKED_INTERPRET(forend, B_1)
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
		if(cached < trace.nodes.size() &&
		   trace.nodes[cached].op == IROpCode::load &&
		   trace.nodes[cached].out.p == v.p) {
			return cached;
		} else {
			return (cache[idx] = trace.EmitLoad(v));
		}
	}
	IRef cache[256];
};

bool isRecordableType(Value & v) {
	return (v.isFuture() && v.length != 1) || v.isDouble() || v.isInteger() || v.isLogical();
}


bool isRecordableShape(int64_t l) {
	return l >= TRACE_VECTOR_WIDTH;
}
bool isRecordableShape(Value & v) {
	return isRecordableShape(v.length) || (v.isFuture() && v.length < 0);
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
			return trace.EmitConstant(v.type,v.i);
		} else {
			return load_cache.get(trace,v);
		}
	}
}

RecordingStatus::Enum fallback(Thread & thread, Value & a, Value & b) {
	thread.trace.Force(thread, a);
	thread.trace.Force(thread, b);
	return RecordingStatus::FALLBACK;
}

RecordingStatus::Enum subset_record(Thread & thread, Instruction const & inst, Instruction const** pc) {
	Trace& trace = thread.trace;
	
	Value & b = REG(thread,inst.b);
	if(getType(b) == Type::Integer || getType(b) == Type::Double) {
		CHECK_REG(inst.a);
		Value & a = REG(thread,inst.a);

		IRef bref = getRef(trace,b);
		Type::Enum rtype = a.type;

		IRef r = trace.EmitUnary(IROpCode::gather, rtype, trace.EmitCoerce(bref, Type::Integer), ((int64_t)a.p)-8);
		Future::Init(REG(thread,inst.c), trace.nodes[r].type, trace.nodes[r].shape.length, r);
		
		trace.RegOutput(r, thread.base,inst.c);
		trace.SetMaxLiveRegister(thread.base,inst.c);
		trace.Commit(thread);
		(*pc)++; 
		return RecordingStatus::NO_ERROR;
	}
	else if(getType(b) == Type::Logical) {
		Value& a = REG(thread, inst.a);

		IRef aref = getRef(trace,a);
		IRef bref = getRef(trace,b);

		if(trace.nodes[aref].shape != trace.nodes[bref].shape)
			return fallback(thread, a, b);

		IRef r = trace.EmitFilter(aref, bref);
		Future::Init(REG(thread,inst.c), trace.nodes[r].type, trace.nodes[r].shape.length, r);
		
		trace.RegOutput(r, thread.base, inst.c);
		trace.SetMaxLiveRegister(thread.base,inst.c);
		trace.Commit(thread);
		(*pc)++; 
		return RecordingStatus::NO_ERROR;
	} 
	else {
		CHECK_REG(inst.a);
		CHECK_REG(inst.b);
		CHECK_REG(inst.c);
		*pc = subset_op(thread,inst);
		return RecordingStatus::NO_ERROR;
	}
}

RecordingStatus::Enum ifelse_record(Thread & thread, Instruction const & inst, Instruction const** pc) {
	Trace& trace = thread.trace;
	
	Value & cond = REG(thread,inst.c);
	Value & yes = REG(thread,inst.b);
	Value & no = REG(thread,inst.a);

	Type::Enum ytype = getType(yes);	
	Type::Enum ntype = getType(no);

	Type::Enum rtype = std::max(ytype, ntype);

	IRef cref = getRef(trace, cond);	
	IRef yref = getRef(trace, yes);	
	IRef nref = getRef(trace, no);	

	IRef r = trace.EmitTrinary(IROpCode::ifelse, rtype, trace.EmitCoerce(nref, rtype), trace.EmitCoerce(yref, rtype), trace.EmitCoerce(cref, Type::Logical));
	Future::Init(REG(thread,inst.c), trace.nodes[r].type, trace.nodes[r].shape.length, r);
	
	trace.RegOutput(r, thread.base, inst.c);
	trace.SetMaxLiveRegister(thread.base, inst.c);
	trace.Commit(thread);
	(*pc)++;
	return RecordingStatus::NO_ERROR;
}

RecordingStatus::Enum split_record(Thread & thread, Instruction const & inst, Instruction const** pc) {
	Trace& trace = thread.trace;
	
	Value & x = REG(thread,inst.c);
	Value & f = REG(thread,inst.b);
	CHECK_REG(inst.c);
	int64_t levels = As<Integer>(thread, REG(thread,inst.a))[0];

	IRef xref = getRef(trace, x);
	IRef fref = getRef(trace, f);	

	IRef r = trace.EmitSplit(xref, trace.EmitCoerce(fref, Type::Integer), levels);
	Future::Init(REG(thread,inst.c), trace.nodes[r].type, trace.nodes[r].shape.length, r);
	
	trace.RegOutput(r, thread.base, inst.c);
	trace.SetMaxLiveRegister(thread.base, inst.c);
	trace.Commit(thread);
	(*pc)++;
	return RecordingStatus::NO_ERROR;
}

RecordingStatus::Enum binary_record(ByteCode::Enum bc, IROpCode::Enum op, Thread & thread, Instruction const & inst) {
	Trace& trace = thread.trace;
	
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
	
	IRef aref = getRef(trace,a);
	IRef bref = getRef(trace,b);
	
	Type::Enum rtype, atype, btype;
	selectType(bc,trace.nodes[aref].type,trace.nodes[bref].type,atype,btype,rtype);

	IRef r = trace.EmitBinary(op, rtype, trace.EmitCoerce(aref,atype), trace.EmitCoerce(bref,btype), 0);
	Future::Init(REG(thread,inst.c), trace.nodes[r].type, trace.nodes[r].shape.length, r);
	
	trace.RegOutput(r, thread.base, inst.c);
	trace.SetMaxLiveRegister(thread.base, inst.c);
	trace.Commit(thread);
	return RecordingStatus::NO_ERROR;
}

RecordingStatus::Enum unary_record(ByteCode::Enum bc, IROpCode::Enum op, Thread & thread, Instruction const & inst) {
	Trace& trace = thread.trace;

	Value & a = REG(thread,inst.a);

	if(!isRecordableType(a) || a.length == 1 || !isRecordableShape(a)) {
		trace.Force(thread, a);
		return RecordingStatus::FALLBACK;
	}

	IRef aref = getRef(trace,a);

	Type::Enum rtype,atype;
	selectType(bc,trace.nodes[aref].type,atype,rtype);

	IRef r = trace.EmitUnary(op,rtype,trace.EmitCoerce(aref,atype), 0);
	Future::Init(REG(thread,inst.c), trace.nodes[r].type, trace.nodes[r].shape.length, r);

	trace.RegOutput(r, thread.base,inst.c);
	trace.SetMaxLiveRegister(thread.base,inst.c);
	trace.Commit(thread);
	return RecordingStatus::NO_ERROR;
}

RecordingStatus::Enum fold_record(ByteCode::Enum bc, IROpCode::Enum op, Thread & thread, Instruction const & inst) {
	Trace& trace = thread.trace;
	
	Value & a = REG(thread,inst.a);

	if(!isRecordableType(a) || a.length == 1 || !isRecordableShape(a)) {
		trace.Force(thread, a);
		return RecordingStatus::FALLBACK;
	}

	IRef aref = getRef(trace,a);

	Type::Enum rtype,atype;
	selectType(bc,trace.nodes[aref].type,atype,rtype);

	IRef r = trace.EmitFold(op,trace.EmitCoerce(aref,rtype));
	Future::Init(REG(thread,inst.c), trace.nodes[r].type, trace.nodes[r].shape.length, r);

	trace.RegOutput(r, thread.base,inst.c);
	trace.SetMaxLiveRegister(thread.base,inst.c);
	trace.Commit(thread);
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
	RecordingStatus::Enum status = unary_record(ByteCode :: op , IROpCode :: op, thread, inst);\
	if(RecordingStatus::FALLBACK == status) { \
		*pc = op##_op(thread,inst); \
		return RecordingStatus::NO_ERROR; \
	} \
	if(RecordingStatus::NO_ERROR == status) \
		(*pc)++; \
	return status; \
}
/*#define FOLD_OP(op,...) RecordingStatus::Enum op##_record(Thread & thread, Instruction const & inst, Instruction const ** pc) { \
	RecordingStatus::Enum status = fold_record(ByteCode :: op, IROpCode :: op, thread, inst);\
	if(RecordingStatus::FALLBACK == status) { \
		*pc = op##_op(thread,inst); \
		return RecordingStatus::NO_ERROR; \
	} \
	if(RecordingStatus::NO_ERROR == status) \
		(*pc)++; \
	return status; \
}*/

UNARY_BYTECODES(UNARY_OP)
FOLD_BYTECODES(UNARY_OP)
BINARY_BYTECODES(BINARY_OP)

#undef UNARY_OP
#undef BINARY_OP

OP_NOT_IMPLEMENTED(sland)
OP_NOT_IMPLEMENTED(slor)

OP_NOT_IMPLEMENTED(list)
OP_NOT_IMPLEMENTED(logical1)
OP_NOT_IMPLEMENTED(integer1)
OP_NOT_IMPLEMENTED(double1)
OP_NOT_IMPLEMENTED(complex1)
OP_NOT_IMPLEMENTED(character1)
OP_NOT_IMPLEMENTED(raw1)

/*OP_NOT_IMPLEMENTED(min)
OP_NOT_IMPLEMENTED(max)
OP_NOT_IMPLEMENTED(any)
OP_NOT_IMPLEMENTED(all)
*/

OP_NOT_IMPLEMENTED(cumsum)
OP_NOT_IMPLEMENTED(cumprod)
OP_NOT_IMPLEMENTED(cummin)
OP_NOT_IMPLEMENTED(cummax)
OP_NOT_IMPLEMENTED(cumany)
OP_NOT_IMPLEMENTED(cumall)


OP_NOT_IMPLEMENTED(apply)

OP_NOT_IMPLEMENTED(dotdot)
OP_NOT_IMPLEMENTED(mov)

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
RecordingStatus::Enum strip_record(Thread & thread, Instruction const & inst, Instruction const ** pc) {
	*pc = strip_op(thread, inst);
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
	thread.trace.SetMaxLiveRegister(thread.base,max_live);
	if(result->isFuture()) {
		thread.trace.RegOutput(result->future.ref, thread.base,offset);
		thread.trace.Commit(thread);
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

	thread.trace.Force(thread, a);
	thread.trace.Force(thread, b);
	int64_t len = As<Integer>(thread, REG(thread, inst.a))[0];
	int64_t step = As<Integer>(thread, REG(thread, inst.b))[0];

	if(!isRecordableShape(len)) {
		*pc = seq_op(thread,inst); //this isn't ideal, as this will redo the As operators above
	} else {
		Future::Init(REG(thread,inst.c),
				     Type::Integer,
				     len,
				     thread.trace.EmitSpecial(IROpCode::seq,Type::Integer,len,len,step));
		thread.trace.SetMaxLiveRegister(thread.base,inst.c);
		(*pc)++;
		thread.trace.Commit(thread);
	}
	return RecordingStatus::NO_ERROR;
}

//check trace exit condition
// -- do we need to abort due to conditions applying to all opcodes? if so, abort the trace, then exit the recorder
// -- is the trace complete? if so, install the trace, and exit the recorder
// -- otherwise, the recorder continues normally
//returns true if we should continue recording
static RecordingStatus::Enum recording_check_conditions(Thread& thread, Instruction const * inst) {
	if(++thread.trace.n_recorded_since_last_exec > TRACE_MAX_RECORDED) {
		return RecordingStatus::RECORD_LIMIT;
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
			thread.trace.EndTracing(thread);
			return pc;
		}
		//printf(" .\n");
	}
	return pc;
}
