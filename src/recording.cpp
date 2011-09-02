#include "recording.h"
#include "interpreter.h"
#include "ir.h"

#define ENUM_RECORDING_STATUS(_) \
	_(NO_ERROR,"NO_ERROR") \
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

//a function must call reserve before making any calls to emitir/emitconst
static IRef emitir(State & state, IROpCode::Enum op, int64_t a, int64_t b) {
	IRNode & n = TRACE.nodes[TRACE.n_nodes];
	n.opcode = op;
	n.a = a;
	n.b = b;
	return TRACE.n_nodes++;
}
static IRef broadcast(State & state, double s) {
	IRNode & n = TRACE.nodes[TRACE.n_nodes];
	n.opcode = IROpCode::broadcast;
	n.const_a = s;
	n.b = 0;
	return TRACE.n_nodes++;
}
static IRef vload(State & state, double * s) {
	IRNode & n = TRACE.nodes[TRACE.n_nodes];
	n.opcode = IROpCode::vload;
	n.reg_a = s;
	n.b = 0;
	return TRACE.n_nodes++;
}

static void add_output(State & state, Value & v) {
	Trace::Output & out = TRACE.outputs[TRACE.n_outputs++];
	out.location = &v;
	out.ref = v.i;
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

#define OP_NOT_IMPLEMENTED(bc) \
	return RecordingStatus::UNSUPPORTED_OP


//all arithmetic binary ops share the same recording implementation
#define BINARY_OP(op) RecordingStatus::Enum op##_record(State & state, Instruction const & inst, Instruction const ** pc) { \
	OP_NOT_IMPLEMENTED(op); \
}
//all unary arithmetic ops share the same implementation as well
#define UNARY_OP(op) RecordingStatus::Enum op##_record(State & state, Instruction const & inst, Instruction const ** pc) { \
		OP_NOT_IMPLEMENTED(op); \
	}

RecordingStatus::Enum call_record(State & state, Instruction const & inst, Instruction const ** pc) {
	OP_NOT_IMPLEMENTED(call);
}
RecordingStatus::Enum get_record(State & state, Instruction const & inst, Instruction const ** pc) {
	RESERVE(0,1);
	*pc = get_op(state,inst);
	Value & r = REG(state,inst.c);
	if(r.header == Type::Future) {
		add_output(state,r);	
	}
	return RecordingStatus::NO_ERROR;
}
RecordingStatus::Enum sget_record(State & state, Instruction const & inst, Instruction const ** pc) {
	OP_NOT_IMPLEMENTED(sget);
}

RecordingStatus::Enum kget_record(State & state, Instruction const & inst, Instruction const ** pc) {
	*pc = kget_op(state,inst);
	return RecordingStatus::NO_ERROR;
}

RecordingStatus::Enum iget_record(State & state, Instruction const & inst, Instruction const ** pc) {
	OP_NOT_IMPLEMENTED(iget);
}

RecordingStatus::Enum assign_record(State & state, Instruction const & inst, Instruction const ** pc) {
	RESERVE(0,1);
	Value & r = state.frame.environment->hassign(Symbol(inst.a), REG(state, inst.c));
	if(r.header == Type::Future) {
		add_output(state,r);
	}
	(*pc)++;
	return RecordingStatus::NO_ERROR;
}

RecordingStatus::Enum sassign_record(State & state, Instruction const & inst, Instruction const ** pc) {
	OP_NOT_IMPLEMENTED(sassign);
}

RecordingStatus::Enum iassign_record(State & state, Instruction const & inst, Instruction const ** pc) {
	OP_NOT_IMPLEMENTED(iassign);
}
RecordingStatus::Enum eassign_record(State & state, Instruction const & inst, Instruction const ** pc) {
	OP_NOT_IMPLEMENTED(eassign);
}
RecordingStatus::Enum subset_record(State & state, Instruction const & inst, Instruction const ** pc) {
	OP_NOT_IMPLEMENTED(iassign);
}
RecordingStatus::Enum subset2_record(State & state, Instruction const & inst, Instruction const ** pc) {
	OP_NOT_IMPLEMENTED(iassign);
}
RecordingStatus::Enum forbegin_record(State & state, Instruction const & inst, Instruction const ** pc) {
	OP_NOT_IMPLEMENTED(forbegin);
}
RecordingStatus::Enum forend_record(State & state, Instruction const & inst, Instruction const ** pc) {
	OP_NOT_IMPLEMENTED(forend);
}
RecordingStatus::Enum iforbegin_record(State & state, Instruction const & inst, Instruction const ** pc) {
	OP_NOT_IMPLEMENTED(iforbegin);
}
RecordingStatus::Enum iforend_record(State & state, Instruction const & inst, Instruction const ** pc) {
	OP_NOT_IMPLEMENTED(iforend);
}

RecordingStatus::Enum jt_record(State & state, Instruction const & inst, Instruction const ** pc) {
	OP_NOT_IMPLEMENTED(jt);
}
RecordingStatus::Enum jf_record(State & state, Instruction const & inst, Instruction const ** pc) {
	OP_NOT_IMPLEMENTED(jf);
}
RecordingStatus::Enum colon_record(State & state, Instruction const & inst, Instruction const ** pc) {
	OP_NOT_IMPLEMENTED(colon);
}

void assign(State & state, Value & r, IRef ref) {
	Future::Init(r,ref);
	add_output(state,r);
}

RecordingStatus::Enum trace_input(State & state, Value & i, IRef * r) {
	if(i.isDouble1()) {
		*r = broadcast(state,i.d);
		return RecordingStatus::NO_ERROR;
	} else if(i.isDouble() && TRACE.length == (uint64_t) i.length) {
		*r = vload(state,(double *) i.p);
		//assign(state,i,*r);
		return RecordingStatus::NO_ERROR;
	} else return RecordingStatus::UNSUPPORTED_TYPE;
}

RecordingStatus::Enum add_record(State & state, Instruction const & inst, Instruction const ** pc) {
	Value & r = REG(state,inst.c);
	Value & a = REG(state,inst.a);
	Value & b = REG(state,inst.b);
	if(a.header == Type::Future) {
		if(b.header == Type::Future) {
			RESERVE(1,1);
			assign(state,r,emitir(state,IROpCode::add,a.i,b.i));
			(*pc)++;
		} else {
			RESERVE(2,2);
			IRef bref;
			RECORDING_DO(trace_input(state,b,&bref));
			assign(state,r,emitir(state,IROpCode::add,a.i,bref));
			(*pc)++;
		}
	} else if(b.header == Type::Future) {
		RESERVE(2,2);
		IRef aref;
		RECORDING_DO(trace_input(state,a,&aref));
		assign(state,r,emitir(state,IROpCode::add,aref,b.i));
		(*pc)++;
	} else if( (uint64_t) a.length == TRACE.length || (uint64_t) b.length == TRACE.length) {
		RESERVE(3,3);
		IRef aref,bref;
		RECORDING_DO(trace_input(state,a,&aref));
		RECORDING_DO(trace_input(state,b,&bref));
		assign(state,r,emitir(state,IROpCode::add,aref,bref));
		(*pc)++;
	} else {
		*pc = add_op(state,inst);
	}
	return RecordingStatus::NO_ERROR;
}

RecordingStatus::Enum sqrt_record(State & state, Instruction const & inst, Instruction const ** pc) {
	Value & r = REG(state,inst.c);
	Value & a = REG(state,inst.a);
	if(a.header == Type::Future) {
		RESERVE(1,1);
		assign(state,r,emitir(state,IROpCode::sqrt,a.i,0));
		(*pc)++;
	} else if((uint64_t)a.length == TRACE.length) {
		RESERVE(2,2);
		IRef aref;
		RECORDING_DO(trace_input(state,a,&aref));
		assign(state,r,emitir(state,IROpCode::sqrt,aref,0));
		(*pc)++;
	} else {
		*pc = sqrt_op(state,inst);
	}
	return RecordingStatus::NO_ERROR;
}


UNARY_OP(pos)

//BINARY_OP(add)
BINARY_OP(sub)

UNARY_OP(neg)

BINARY_OP(mul)
BINARY_OP(div)
BINARY_OP(idiv)
BINARY_OP(mod)
BINARY_OP(pow)
BINARY_OP(lt)
BINARY_OP(gt)
BINARY_OP(eq)
BINARY_OP(neq)
BINARY_OP(le)
BINARY_OP(ge)
BINARY_OP(lnot)
BINARY_OP(land)
BINARY_OP(lor)
BINARY_OP(sland)
BINARY_OP(slor)

UNARY_OP(abs)
UNARY_OP(sign)
//UNARY_OP(sqrt)
UNARY_OP(floor)
UNARY_OP(ceiling)
UNARY_OP(trunc)
UNARY_OP(round)
UNARY_OP(signif)
UNARY_OP(exp)
UNARY_OP(log)
UNARY_OP(cos)
UNARY_OP(sin)
UNARY_OP(tan)
UNARY_OP(acos)
UNARY_OP(asin)
UNARY_OP(atan)
UNARY_OP(logical1)
UNARY_OP(integer1)
UNARY_OP(double1)
UNARY_OP(complex1)
UNARY_OP(character1)


RecordingStatus::Enum jmp_record(State & state, Instruction const & inst, Instruction const ** pc) {
	//this is just a constant jump, nothing to record
	*pc = jmp_op(state,inst);
	return RecordingStatus::NO_ERROR;
}
RecordingStatus::Enum function_record(State & state, Instruction const & inst, Instruction const ** pc) {
	OP_NOT_IMPLEMENTED(function);
}

RecordingStatus::Enum raw1_record(State & state, Instruction const & inst, Instruction const ** pc) {
	OP_NOT_IMPLEMENTED(raw1);
}
RecordingStatus::Enum UseMethod_record(State & state, Instruction const & inst, Instruction const ** pc) {
	OP_NOT_IMPLEMENTED(UseMethod);
}
RecordingStatus::Enum seq_record(State & state, Instruction const & inst, Instruction const ** pc) {
	OP_NOT_IMPLEMENTED(seq);
}
RecordingStatus::Enum type_record(State & state, Instruction const & inst, Instruction const ** pc) {
	OP_NOT_IMPLEMENTED(type);
}

RecordingStatus::Enum ret_record(State & state, Instruction const & inst, Instruction const ** pc) {
	OP_NOT_IMPLEMENTED(ret);
}
RecordingStatus::Enum done_record(State & state, Instruction const & inst, Instruction const ** pc) {
	OP_NOT_IMPLEMENTED(done);
}


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
