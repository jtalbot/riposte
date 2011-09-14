#include "recording.h"
#include "interpreter.h"
#include "ir.h"
#include "ops.h"

#define ENUM_RECORDING_STATUS(_) \
	_(NO_ERROR,"NO_ERROR") \
	_(FALLBACK, "trace falling back to normal interpreter but not exiting") \
	_(RESOURCE, "trace ran out of resources") \
	_(UNSUPPORTED_OP,"trace encountered unsupported op") \
	_(UNSUPPORTED_TYPE,"trace encountered an unsupported type") \
	_(INTERPRETER_DEPENDENCY, "non-traceable op requires the value of a future")

DECLARE_ENUM(RecordingStatus,ENUM_RECORDING_STATUS)
DEFINE_ENUM_TO_STRING(RecordingStatus,ENUM_RECORDING_STATUS)

//for brevity
#define TRACE (state.tracing.current_trace)


static RecordingStatus::Enum reserve(State & state, size_t num_nodes, size_t num_outputs) {
	if(TRACE.n_nodes + num_nodes >= TRACE_MAX_NODES)
		return RecordingStatus::RESOURCE;
	else if(TRACE.n_outputs + num_outputs >= TRACE_MAX_OUTPUTS)
		return RecordingStatus::RESOURCE;
	else
		return RecordingStatus::NO_ERROR;
}

static void add_output(State & state, Trace::Location::Type location_type, int64_t id, Value & v) {
	Trace::Output & out = TRACE.outputs[TRACE.n_outputs++];
	out.location.type = location_type;
	out.location.id = id;
	if(location_type == Trace::Location::REG)
		out.location.base = state.base;
	else
		out.location.environment = state.frame.environment;
}

static void set_max_live_register(State & state, int64_t r) {
	TRACE.max_live_register_base = state.base;
	TRACE.max_live_register = r;
}

struct InputValue {
	IROp::Encoding encoding;
	Type::Enum typ;
	bool is_external;
	IRNode::InputReg data;
};

static const InputValue InputValue_unused = {IROp::E_VECTOR, Type::Integer, false, { NULL } };

void coerce(State & state, Type::Enum result_type, InputValue & a) {
	IRNode & n = TRACE.nodes[TRACE.n_nodes];

	if(a.encoding == IROp::E_VECTOR) {
		n.op.a_enc = a.encoding;
		n.op.a_typ = IROp::T_INT;
		n.a_external = a.is_external;
		n.a.p = a.data.p;

		n.op.b_enc = InputValue_unused.encoding;
		n.op.b_typ = IROp::T_DOUBLE; //coerse op holds the result type in n.op.b_typ
		n.b_external = InputValue_unused.is_external;
		n.b.p = InputValue_unused.data.p;

		n.r_external = false;
		n.r.p =  NULL;
		n.op.code = IROpCode::coerce;
		IRef result = TRACE.n_nodes++;

		a.data.i = result;

	} else {
		//no need to issues a scalar conversion, just promote the integer data to a double
		a.data.d = a.data.i;
	}
	a.typ = result_type;
	a.is_external = false;
}

#define REG(state, i) (*(state.base+i))
//this emits an opcode assuming that a and b are the same type
void emitir(State & state, IROpCode::Enum opcode,
						   Type::Enum ret_type,
		                   const InputValue & a,
		                   const InputValue & b,
		                   int64_t r) {
	IRNode & n = TRACE.nodes[TRACE.n_nodes];
	n.op.a_enc = a.encoding;
	n.op.a_typ = (a.typ == Type::Integer) ? IROp::T_INT : IROp::T_DOUBLE;
	n.a_external = a.is_external;
	n.a.p = a.data.p;

	n.op.b_enc = b.encoding;
	n.op.b_typ = (b.typ == Type::Integer) ? IROp::T_INT : IROp::T_DOUBLE;
	n.b_external = b.is_external;
	n.b.p = b.data.p;

	n.r_external = false;
	n.r.p = NULL;

	n.op.code = opcode;

	//TODO: replace binary OR with a function that calculates output types from input types
	Value & v = REG(state,r);
	Future::Init(v, ret_type,TRACE.n_nodes++);
	add_output(state,Trace::Location::REG,r,v);
	set_max_live_register(state,r);
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


#define OP_NOT_IMPLEMENTED(op) \
RecordingStatus::Enum op##_record(State & state, Instruction const & inst, Instruction const ** pc) { \
	return RecordingStatus::UNSUPPORTED_OP; \
} \


RecordingStatus::Enum get_record(State & state, Instruction const & inst, Instruction const ** pc) {
	RESERVE(0,1);
	*pc = get_op(state,inst); // danger! can recursively invoke the recorder
	                          // if the trace is no longer active we bail out now
	if(!state.tracing.is_tracing())
		return RecordingStatus::UNSUPPORTED_OP;

	Value & r = REG(state,inst.c);
	if(r.isFuture()) {
		add_output(state,Trace::Location::REG,inst.c,r);
	}
	set_max_live_register(state,inst.c);
	return RecordingStatus::NO_ERROR;
}

/*RecordingStatus::Enum sget_record(State & state, Instruction const & inst, Instruction const ** pc) {
	RESERVE(0,1);
	*pc = sget_op(state,inst); // danger! can recursively invoke the recorder
							   // if the trace is no longer active we bail out now
	if(!state.tracing.is_tracing())
		return RecordingStatus::UNSUPPORTED_OP;

	Value & r = REG(state,inst.c);
	if(r.isFuture()) {
		add_output(state,Trace::Location::REG,inst.c,r);
	}
	set_max_live_register(state,inst.c);
	return RecordingStatus::NO_ERROR;
}*/

RecordingStatus::Enum kget_record(State & state, Instruction const & inst, Instruction const ** pc) {
	*pc = kget_op(state,inst);
	return RecordingStatus::NO_ERROR;
}


OP_NOT_IMPLEMENTED(iget)

RecordingStatus::Enum assign_record(State & state, Instruction const & inst, Instruction const ** pc) {
	RESERVE(0,1);
	state.frame.environment->assign(Symbol(inst.a), REG(state, inst.c));
	Value& r = REG(state, inst.c); // This is wrong!
	if(r.isFuture()) {
		add_output(state,Trace::Location::VAR,inst.a,r);
	}
	set_max_live_register(state,inst.c);
	(*pc)++;
	return RecordingStatus::NO_ERROR;
}
/*
RecordingStatus::Enum sassign_record(State & state, Instruction const & inst, Instruction const ** pc) {
	*pc = sassign_op(state,inst);
	Value & v = state.frame.environment->get(inst.a);
	if(v.isFuture()) {
		RESERVE(0,1);
		add_output(state,Trace::Location::SLOT,inst.a,v);
	}
	set_max_live_register(state,inst.c);
	return RecordingStatus::NO_ERROR;
}
*/

#define CHECK_REG(r) (REG(state,r).isFuture())
//temporary defines to generate code for checked interpret
#define A CHECK_REG(inst.a) ||
#define B CHECK_REG(inst.b) ||
#define C CHECK_REG(inst.c) ||
#define B_1 CHECK_REG(inst.b - 1) ||
#define C_1 CHECK_REG(inst.c - 1) ||

//all operations that first verify their inputs are not futures, and then call into the scalar interpreter to fulfill them
#define CHECKED_INTERPRET(op, checks) \
		RecordingStatus::Enum op##_record(State & state, Instruction const & inst, Instruction const ** pc) { \
			if(checks false) \
				return RecordingStatus::INTERPRETER_DEPENDENCY; \
			*pc = op##_op(state,inst); \
			return RecordingStatus::NO_ERROR; \
		} \

CHECKED_INTERPRET(eassign, A B C)
CHECKED_INTERPRET(iassign, A B C)
CHECKED_INTERPRET(jt, B)
CHECKED_INTERPRET(jf, B)
CHECKED_INTERPRET(subset, A B C)
CHECKED_INTERPRET(subset2, A B C)
CHECKED_INTERPRET(colon, A B C)
CHECKED_INTERPRET(forbegin, B_1)
CHECKED_INTERPRET(forend, B_1)
//CHECKED_INTERPRET(iforbegin, C C_1)
//CHECKED_INTERPRET(iforend, C C_1)
CHECKED_INTERPRET(seq, A)
CHECKED_INTERPRET(UseMethod, A C)
CHECKED_INTERPRET(call, A)
#undef A
#undef B
#undef B_1
#undef C
#undef C_1
#undef CHECKED_INTERPRET

RecordingStatus::Enum get_input(State & state, Value & v, InputValue * ret, bool * can_fallback, bool * should_record) {

	//encode the type;
	switch(v.type) {
	case Type::Integer: /* fallthrough */
	case Type::Double:
		ret->typ = v.type;
		//encode the data (for vectors this copies the pointer, for scalars this copies the value from the Value object)
		ret->data.p = (double *) v.p;
		//scalar or vector?
		if(v.length == 1) {
			ret->is_external = false;
			ret->encoding = IROp::E_SCALAR;
		} else if(v.length == TRACE.length) {
			ret->encoding = IROp::E_VECTOR;
			ret->is_external = true;
			*should_record = true;
		} else {
			return RecordingStatus::UNSUPPORTED_TYPE;
		}
		return RecordingStatus::NO_ERROR;
		break;
	case Type::Future:
		ret->encoding = IROp::E_VECTOR;
		ret->typ = v.future.typ;
		ret->data.i = v.future.ref;
		ret->is_external = false;
		*can_fallback = false;
		*should_record = true;
		return RecordingStatus::NO_ERROR;
		break;
	default:
		return RecordingStatus::UNSUPPORTED_TYPE;
		break;
	}
}

RecordingStatus::Enum binary_record(ByteCode::Enum bc, IROpCode::Enum opcode, State & state, Instruction const & inst) {
	Value & a = REG(state,inst.a);
	Value & b = REG(state,inst.b);

	bool can_fallback = true;
	bool should_record = false;
	InputValue aenc;
	InputValue benc;
	RecordingStatus::Enum ar = get_input(state,a,&aenc,&can_fallback,&should_record);
	RecordingStatus::Enum br = get_input(state,b,&benc,&can_fallback,&should_record);
	if(should_record && RecordingStatus::NO_ERROR == ar && RecordingStatus::NO_ERROR == br) {
		RESERVE(2,1);
		Type::Enum arg_type = std::max(aenc.typ,benc.typ);
		if(aenc.typ != arg_type)
			coerce(state,arg_type,aenc);
		if(benc.typ != arg_type)
			coerce(state,arg_type,benc);
		emitir(state,opcode,resultType(bc,aenc.typ,benc.typ),aenc,benc,inst.c);
		return RecordingStatus::NO_ERROR;
	} else {
		return (can_fallback) ? RecordingStatus::FALLBACK : RecordingStatus::UNSUPPORTED_TYPE;
	}

}

RecordingStatus::Enum unary_record(ByteCode::Enum bc, IROpCode::Enum opcode, State & state, Instruction const & inst) {
	Value & a = REG(state,inst.a);
	bool can_fallback = true;
	bool should_record = false;
	InputValue aenc;
	RecordingStatus::Enum ar = get_input(state,a,&aenc,&can_fallback,&should_record);

	if(should_record && RecordingStatus::NO_ERROR == ar) {
		RESERVE(1,1);
		emitir(state,opcode,resultType(bc,aenc.typ),aenc,InputValue_unused,inst.c);
		return RecordingStatus::NO_ERROR;
	} else {
		return (can_fallback) ? RecordingStatus::FALLBACK : RecordingStatus::UNSUPPORTED_TYPE;
	}

}

//all arithmetic binary ops share the same recording implementation
#define BINARY_OP(op) RecordingStatus::Enum op##_record(State & state, Instruction const & inst, Instruction const ** pc) { \
	RecordingStatus::Enum status = binary_record(ByteCode :: op, IROpCode :: op, state, inst);\
	if(RecordingStatus::FALLBACK == status) { \
		*pc = op##_op(state,inst); \
		return RecordingStatus::NO_ERROR; \
	} \
	(*pc)++; \
	return status; \
}
//all unary arithmetic ops share the same implementation as well
#define UNARY_OP(op) RecordingStatus::Enum op##_record(State & state, Instruction const & inst, Instruction const ** pc) { \
	RecordingStatus::Enum status = unary_record(ByteCode :: op , IROpCode :: op, state, inst);\
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

OP_NOT_IMPLEMENTED(sum)
OP_NOT_IMPLEMENTED(prod)
OP_NOT_IMPLEMENTED(min)
OP_NOT_IMPLEMENTED(max)
OP_NOT_IMPLEMENTED(any)
OP_NOT_IMPLEMENTED(all)

OP_NOT_IMPLEMENTED(cumsum)
OP_NOT_IMPLEMENTED(cumprod)
OP_NOT_IMPLEMENTED(cummin)
OP_NOT_IMPLEMENTED(cummax)
OP_NOT_IMPLEMENTED(cumany)
OP_NOT_IMPLEMENTED(cumall)
OP_NOT_IMPLEMENTED(raw1)


RecordingStatus::Enum jmp_record(State & state, Instruction const & inst, Instruction const ** pc) {
	//this is just a constant jump, nothing to record
	*pc = jmp_op(state,inst);
	return RecordingStatus::NO_ERROR;
}

RecordingStatus::Enum function_record(State & state, Instruction const & inst, Instruction const ** pc) {
	*pc = function_op(state,inst);
	return RecordingStatus::NO_ERROR;
}


//we can extract the type from the future so we can continue a trace beyond a type-check
RecordingStatus::Enum type_record(State & state, Instruction const & inst, Instruction const ** pc) {
	Character c(1);
	// Should have a direct mapping from type to symbol.
	Value & a = REG(state, inst.a);
	Type::Enum atyp = (a.isFuture()) ? a.future.typ : a.type;
	c[0] = state.StrToSym(Type::toString(atyp));
	REG(state, inst.c) = c;
	(*pc)++;
	return RecordingStatus::NO_ERROR;
}
RecordingStatus::Enum ret_record(State & state, Instruction const & inst, Instruction const ** pc) {
	//danger: ret op writes a (potentially) future output somewhere
	//but it doesn't go directly to a reg/slot/or variable so we can't record it here
	//we need to either change the behavior of ret op so it can only writes to slots/reg/variables
	//or ensure that we record the output somewhere else

	//currently we simply do not support returning from a function
	//*pc = ret_op(state,inst);
	return RecordingStatus::UNSUPPORTED_OP;

}
//not called, interpreter will exit beforehand
RecordingStatus::Enum done_record(State & state, Instruction const & inst, Instruction const ** pc) {
	return RecordingStatus::NO_ERROR;
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

Instruction const * recording_interpret(State& state, Instruction const* pc) {
	RecordingStatus::Enum status = RecordingStatus::NO_ERROR;
	while( pc->bc != ByteCode::done) {
#define RUN_RECORD(name,str,...) case ByteCode::name: { /*printf("rec " #name "\n");*/ status = name##_record(state, *pc,&pc); } break;
		switch(pc->bc) {
			BYTECODES(RUN_RECORD)
		}
#undef RUN_RECORD
		if(   RecordingStatus::NO_ERROR != status
		   || RecordingStatus::NO_ERROR != (status = recording_check_conditions(state,pc))) {
			state.tracing.end_tracing(state);
			return pc;
		}
	}
	return pc;
}
