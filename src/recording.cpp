#include "recording.h"
#include "interpreter.h"
#include "ir.h"

#define ENUM_RECORDING_STATUS(_) \
	_(NO_ERROR,"NO_ERROR") \
	_(FALLBACK, "trace falling back to normal interpreter but not exiting") \
	_(RESOURCE, "trace ran out of resources") \
	_(UNSUPPORTED_OP,"trace encountered unsupported op") \
	_(UNSUPPORTED_TYPE,"trace encountered an unsupported type") \

DECLARE_ENUM(RecordingStatus,ENUM_RECORDING_STATUS)
DEFINE_ENUM_TO_STRING(RecordingStatus,ENUM_RECORDING_STATUS)

//for brevity
#define TRACE (state.tracing.current_trace)


static RecordingStatus::Enum reserve(State & state, size_t num_nodes, size_t num_outputs) {
	if(TRACE.n_nodes + num_nodes >= TRACE_MAX_NODES)
		return RecordingStatus::RESOURCE;
	else if(TRACE.n_outputs + num_outputs >= TRACE_MAX_NODES)
		return RecordingStatus::RESOURCE;
	else
		return RecordingStatus::NO_ERROR;
}

static void add_output(State & state, Value & v) {
	Trace::Output & out = TRACE.outputs[TRACE.n_outputs++];
	out.is_variable = false;
	out.location = &v;
	out.ref = v.i;
}
static void add_voutput(State & state, Symbol s, IRef i) {
	Trace::Output & out = TRACE.outputs[TRACE.n_outputs++];
	out.is_variable = true;
	out.variable = s.i;
	out.ref = i;
}


struct InputValue {
	IROp::Encoding encoding;
	bool is_external;
	void * data;
};

IRef emitir(State & state, IROpCode::Enum opcode,
		                   const InputValue & a,
		                   const InputValue & b) {
	IRNode & n = TRACE.nodes[TRACE.n_nodes];

	n.op.a_enc = a.encoding;
	n.a_external = a.is_external;
	n.a.p = (double *) a.data;

	n.op.b_enc = b.encoding;
	n.b_external = b.is_external;
	n.b.p = (double *) b.data;

	n.op.typ = IROp::T_DOUBLE;


	n.r_external = false;
	n.r.p = NULL;

	n.op.code = opcode;


	return TRACE.n_nodes++;
}

//attempt to execute fn, otherwise return error code
#define RECORDING_DO(fn) \
	do { \
		RecordingStatus::Enum s = fn; \
		if(RecordingStatus::NO_ERROR != s) { \
			return s; \
		} \
	} while(0)

#define RESERVE(nodes,outputs) RECORDING_DO(reserve(state,nodes,outputs))
#define REG(state, i) (*(state.base+i))

#define OP_NOT_IMPLEMENTED(op) \
RecordingStatus::Enum op##_record(State & state, Instruction const & inst, Instruction const ** pc) { \
	return RecordingStatus::UNSUPPORTED_OP; \
} \


OP_NOT_IMPLEMENTED(call)

RecordingStatus::Enum get_record(State & state, Instruction const & inst, Instruction const ** pc) {
	RESERVE(0,1);
	*pc = get_op(state,inst);
	Value & r = REG(state,inst.c);
	if(r.header == Type::Future) {
		add_output(state,r);	
	}
	return RecordingStatus::NO_ERROR;
}

OP_NOT_IMPLEMENTED(sget);

RecordingStatus::Enum kget_record(State & state, Instruction const & inst, Instruction const ** pc) {
	*pc = kget_op(state,inst);
	return RecordingStatus::NO_ERROR;
}


OP_NOT_IMPLEMENTED(iget)

RecordingStatus::Enum assign_record(State & state, Instruction const & inst, Instruction const ** pc) {
	RESERVE(0,1);
	Symbol s(inst.a);
	Value & r = state.frame.environment->hassign(s, REG(state, inst.c));
	if(r.header == Type::Future) {
		add_voutput(state,s,r.i);
	}
	(*pc)++;
	return RecordingStatus::NO_ERROR;
}

OP_NOT_IMPLEMENTED(sassign)
OP_NOT_IMPLEMENTED(eassign)
OP_NOT_IMPLEMENTED(iassign)
OP_NOT_IMPLEMENTED(subset)
OP_NOT_IMPLEMENTED(subset2)

OP_NOT_IMPLEMENTED(forbegin)

OP_NOT_IMPLEMENTED(forend)

OP_NOT_IMPLEMENTED(iforbegin)

OP_NOT_IMPLEMENTED(iforend)

OP_NOT_IMPLEMENTED(jt)
OP_NOT_IMPLEMENTED(jf)

OP_NOT_IMPLEMENTED(colon)


void assign(State & state, Value & r, IRef ref) {
	Future::Init(r,ref);
	add_output(state,r);
}

RecordingStatus::Enum get_input(State & state, Value & v, InputValue * ret) {
	ret->data = v.p;
	if(v.isDouble1()) {
		ret->encoding = IROp::E_SCALAR;
		ret->is_external = false;
	} else if(v.isFuture()) {
		ret->encoding = IROp::E_VECTOR;
		ret->is_external = false;
	} else if(v.isDouble() && v.length == TRACE.length) {
		ret->encoding = IROp::E_VECTOR;
		ret->is_external = true;
	} else return RecordingStatus::UNSUPPORTED_TYPE;
	return RecordingStatus::NO_ERROR;
}

RecordingStatus::Enum binary_record(IROpCode::Enum opcode, State & state, Instruction const & inst) {
	Value & r = REG(state,inst.c);
	Value & a = REG(state,inst.a);
	Value & b = REG(state,inst.b);
	if(a.isFuture()) {
		InputValue aenc = { IROp::E_VECTOR, false, a.p };
		InputValue benc;
		RECORDING_DO(get_input(state,b,&benc));
		RESERVE(1,1);
		assign(state,r,emitir(state,opcode,aenc,benc));
	} else if(b.isFuture()) {
		InputValue aenc;
		RECORDING_DO(get_input(state,a,&aenc));
		InputValue benc = { IROp::E_VECTOR, false, b.p };
		RESERVE(1,1);
		assign(state,r,emitir(state,opcode,aenc,benc));
	} else if(b.length == TRACE.length || a.length == TRACE.length) {
		InputValue aenc;
		InputValue benc;
		RecordingStatus::Enum ar = get_input(state,a,&aenc);
		RecordingStatus::Enum br = get_input(state,b,&benc);
	    if(RecordingStatus::NO_ERROR == ar && RecordingStatus::NO_ERROR == br) {
	    	RESERVE(1,1);
	    	assign(state,r,emitir(state,opcode,aenc,benc));
		} else
	    	return RecordingStatus::FALLBACK;
	} else {
		return RecordingStatus::FALLBACK;
	}
	return RecordingStatus::NO_ERROR;
}

RecordingStatus::Enum unary_record(IROpCode::Enum opcode, State & state, Instruction const & inst) {
	Value & r = REG(state,inst.c);
	Value & a = REG(state,inst.a);
	if(a.header == Type::Future) {
		RESERVE(1,1);
		InputValue aenc = {IROp::E_VECTOR, false, a.p};
		InputValue benc = {IROp::E_VECTOR, false, NULL};
		assign(state,r,emitir(state,opcode,aenc,benc));
	} else if(a.isDouble() && a.length == TRACE.length) {
		RESERVE(1,1);
		InputValue aenc = {IROp::E_VECTOR, true, a.p};
		InputValue benc = {IROp::E_VECTOR, false, NULL};
		assign(state,r,emitir(state,opcode,aenc,benc));
	} else {
		return RecordingStatus::FALLBACK;
	}
	return RecordingStatus::NO_ERROR;
}

//all arithmetic binary ops share the same recording implementation
#define BINARY_OP(op) RecordingStatus::Enum op##_record(State & state, Instruction const & inst, Instruction const ** pc) { \
	RecordingStatus::Enum status = binary_record(IROpCode :: op, state, inst);\
	if(RecordingStatus::FALLBACK == status) { \
		*pc = op##_op(state,inst); \
		return RecordingStatus::NO_ERROR; \
	} \
	(*pc)++; \
	return status; \
}
//all unary arithmetic ops share the same implementation as well
#define UNARY_OP(op) RecordingStatus::Enum op##_record(State & state, Instruction const & inst, Instruction const ** pc) { \
	RecordingStatus::Enum status = unary_record(IROpCode :: op, state, inst);\
	if(RecordingStatus::FALLBACK == status) { \
		*pc = op##_op(state,inst); \
		return RecordingStatus::NO_ERROR; \
	} \
	(*pc)++; \
	return status; \
}


OP_NOT_IMPLEMENTED(pos)

BINARY_OP(add)
BINARY_OP(sub)

UNARY_OP(neg)

BINARY_OP(mul)
BINARY_OP(div)
OP_NOT_IMPLEMENTED(idiv)
OP_NOT_IMPLEMENTED(mod)
BINARY_OP(pow)
OP_NOT_IMPLEMENTED(lt)
OP_NOT_IMPLEMENTED(gt)
OP_NOT_IMPLEMENTED(eq)
OP_NOT_IMPLEMENTED(neq)
OP_NOT_IMPLEMENTED(le)
OP_NOT_IMPLEMENTED(ge)
OP_NOT_IMPLEMENTED(lnot)
OP_NOT_IMPLEMENTED(land)
OP_NOT_IMPLEMENTED(lor)
OP_NOT_IMPLEMENTED(sland)
OP_NOT_IMPLEMENTED(slor)

UNARY_OP(abs)
OP_NOT_IMPLEMENTED(sign)
UNARY_OP(sqrt)
UNARY_OP(floor)
UNARY_OP(ceiling)
OP_NOT_IMPLEMENTED(trunc)
UNARY_OP(round)
OP_NOT_IMPLEMENTED(signif)
UNARY_OP(exp)
UNARY_OP(log)
UNARY_OP(cos)
UNARY_OP(sin)
UNARY_OP(tan)
UNARY_OP(acos)
UNARY_OP(asin)
UNARY_OP(atan)
OP_NOT_IMPLEMENTED(logical1)
OP_NOT_IMPLEMENTED(integer1)
OP_NOT_IMPLEMENTED(double1)
OP_NOT_IMPLEMENTED(complex1)
OP_NOT_IMPLEMENTED(character1)


RecordingStatus::Enum jmp_record(State & state, Instruction const & inst, Instruction const ** pc) {
	//this is just a constant jump, nothing to record
	*pc = jmp_op(state,inst);
	return RecordingStatus::NO_ERROR;
}

OP_NOT_IMPLEMENTED(function)
OP_NOT_IMPLEMENTED(raw1)
OP_NOT_IMPLEMENTED(UseMethod)

OP_NOT_IMPLEMENTED(seq)
OP_NOT_IMPLEMENTED(type)
OP_NOT_IMPLEMENTED(ret)
OP_NOT_IMPLEMENTED(done)


//check trace exit conditions
// -- do we need to abort due to conditions applying to all opcodes? if so, abort the trace, then exit the recorder
// -- is the trace complete? if so, install the trace, and exit the recorder
// -- otherwise, the recorder continues normally
//returns true if we should continue recording
static RecordingStatus::Enum recording_check_conditions(State& state, Instruction const * inst) {
	TRACE.n_recorded++;
	if(++TRACE.n_recorded > TRACE_MAX_RECORDED) {
		return RecordingStatus::RESOURCE;
	}
	return RecordingStatus::NO_ERROR;
}
void recording_end(State & state, RecordingStatus::Enum status) {
	state.tracing.end_tracing(state);
}

Instruction const * recording_interpret(State& state, Instruction const* pc, size_t length) {
	RecordingStatus::Enum status = RecordingStatus::NO_ERROR;
	state.tracing.begin_tracing();
	TRACE.length = length;
	while(true) {
#define RUN_RECORD(name,str) case ByteCode::name: { printf("rec " #name "\n"); status = name##_record(state, *pc,&pc); } break;
		switch(pc->bc) {
			BYTECODES(RUN_RECORD)
		}
#undef RUN_RECORD
		if(   RecordingStatus::NO_ERROR != status
		   || RecordingStatus::NO_ERROR != (status = recording_check_conditions(state,pc))) {
			recording_end(state,status);
			return pc;
		}
	}
	return NULL;
}
