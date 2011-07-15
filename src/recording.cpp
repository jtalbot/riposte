#include "recording.h"
#include "interpreter.h"
#include "ir.h"

#define ENUM_RECORDING_STATUS(_,p) \
	_(NO_ERROR,"NO_ERROR",p) \
	_(FOUND_LOOP,"trace has found the loop, and can be compiled",p) \
	_(RETURN,"trace escaped calling scope",p) \
	_(LENGTH,"trace too big",p) \
	_(RECURSION,"trace encountered an invoke trace node",p) \
	_(UNSUPPORTED_OP,"trace encountered unsupported op",p) \
	_(UNSUPPORTED_TYPE,"trace encountered an unsupported type",p) \

DECLARE_ENUM(RecordingStatus,ENUM_RECORDING_STATUS)
DEFINE_ENUM(RecordingStatus,ENUM_RECORDING_STATUS)
DEFINE_ENUM_TO_STRING(RecordingStatus,ENUM_RECORDING_STATUS)

//for brevity
#define TRACE (state.tracing.current_trace)

static RecordingStatus emitir(State & state, IROpCode op, IRType const & typ, int64_t a, int64_t b, IRef * r) {
	if(typ.base_type == IRScalarType::T_unsupported)
		return RecordingStatus::UNSUPPORTED_TYPE;
	TRACE->recorded.push_back(IRNode(op,typ,a,b));
	if(r)
		*r = TRACE->recorded.size() - 1;
	return RecordingStatus::NO_ERROR;
}

static RecordingStatus emitir(State & state, IROpCode op, Value const & v, int64_t a, int64_t b, IRef * r) {
	IRType typ(v);
	return emitir(state,op,typ,a,b,r);
}

#define EMITIR(op,v,b,c,aptr) RECORDING_DO(emitir(state,IROpCode :: op,v,b,c,aptr))

static void recording_end(State & state, RecordingStatus reason) {
	switch(reason.Enum()) {
	case RecordingStatus::E_FOUND_LOOP:
		trace_compile_and_install(state,TRACE);
		TRACE = NULL;
		break;
	default:
		printf("trace aborted: %s\n",reason.toString());
		delete TRACE;
		TRACE = NULL;
		break;
	}
}

//attempt to execute fn, otherwise return error code
#define RECORDING_DO(fn) \
	do { \
		RecordingStatus s = fn; \
		if(RecordingStatus::NO_ERROR != s) { \
			return s; \
		} \
	} while(0)



#define OP_NOT_IMPLEMENTED(bc) \
	printf("NYI " #bc "\n"); \
	return RecordingStatus::UNSUPPORTED_OP;


static RecordingStatus get_reg(State & state, int64_t slot_id, IRef * node) {
	if(!TRACE->renaming_table.get(RenamingTable::REG,slot_id,node)) {
		Value & value = interpreter_reg(state,slot_id);
		EMITIR(rload,value,slot_id,-1,node);
		TRACE->renaming_table.input(RenamingTable::REG,slot_id,*node);
	}
	return RecordingStatus::NO_ERROR;
}

static RecordingStatus get_var(State & state, Value & value, int64_t var_id, IRef * node) {
	if(!TRACE->renaming_table.get(RenamingTable::VARIABLE,var_id,node)) {
		EMITIR(vload,value,var_id,-1,node);
		TRACE->renaming_table.input(RenamingTable::VARIABLE,var_id,*node);
	}
	return RecordingStatus::NO_ERROR;
}
static RecordingStatus get_slot(State & state, Value & value, int64_t slot_id, IRef * node) {
	if(!TRACE->renaming_table.get(RenamingTable::SLOT,slot_id,node)) {
		EMITIR(sload,value,slot_id,-1,node);
		TRACE->renaming_table.input(RenamingTable::SLOT,slot_id,*node);
	}
	return RecordingStatus::NO_ERROR;
}

//convert slot_id into a scalar boolean, or if it already is one, just return it.
static RecordingStatus get_predicate(State & state, int64_t slot_id, bool invert, IRef * pnode) {
	IRef node;
	RECORDING_DO(get_reg(state,slot_id,&node));
	IRType & typ = TRACE->recorded[node].typ;

	*pnode = node;
	if(typ.base_type != IRScalarType::T_logical || typ.isVector) {
		EMITIR(istrue,IRType::Bool(),*pnode,0,pnode);
	}
	if(invert) {
		EMITIR(lnot,IRType::Bool(),*pnode,0,pnode);
	}
	return RecordingStatus::NO_ERROR;
}

static RecordingStatus insert_guard(State & state, int64_t slot_id, bool invert, const Instruction * other_branch, int64_t n_live_registers) {
	IRef node;
	RECORDING_DO(get_predicate(state,slot_id,invert,&node));
	TraceExit e = { TRACE->renaming_table.create_snapshot(), n_live_registers, other_branch - TRACE->trace_start };
	TRACE->exits.push_back(e);
	EMITIR(guard,IRType::Void(),node,TRACE->exits.size() - 1,NULL);
	return RecordingStatus::NO_ERROR;
}

//many opcodes turn into constant loads
static RecordingStatus load_constant(State & state, int64_t dest_slot, const Value & value) {
	TRACE->constants.push_back(value);
	IRef node;
	EMITIR(kload,value,TRACE->constants.size() - 1,0,&node);
	TRACE->renaming_table.assign(RenamingTable::REG,dest_slot,node);
	return RecordingStatus::NO_ERROR;
}

static RecordingStatus promote_types(State & state, IRef a, IRef b, IRef * ao, IRef * bo) {
	IRType const & at = TRACE->recorded[a].typ;
	IRType const & bt = TRACE->recorded[b].typ;
	if(at.base_type.Enum() == bt.base_type.Enum()) {
		*ao = a;
		*bo = b;
	} else if(at.base_type.Enum() < bt.base_type.Enum()){
		EMITIR(cast,bt,a,0,ao);
		*bo = b;
	} else {
		EMITIR(cast,at,b,0,bo);
		*ao = a;
	}
	return RecordingStatus::NO_ERROR;
}

//all arithmetic binary ops share the same recording implementation
#define BINARY_OP(op) \
RecordingStatus op##_record(State & state, Instruction const & inst, Instruction const ** pc) { \
		IRef node_a; \
		IRef node_b; \
		RECORDING_DO(get_reg(state,inst.a,&node_a)); \
		RECORDING_DO(get_reg(state,inst.b,&node_b)); \
		*pc = op##_op(state,inst); \
		Value & c = interpreter_reg(state,inst.c); \
		IRef output; \
		IRef node_a1,node_b1;\
		RECORDING_DO(promote_types(state,node_a,node_b,&node_a1,&node_b1)); \
		EMITIR(op,c,node_a1,node_b1,&output); \
		TRACE->renaming_table.assign(RenamingTable::REG,inst.c,output); \
		return RecordingStatus::NO_ERROR; \
}

//all unary arithmetic ops share the same implementation as well
#define UNARY_OP(op) \
RecordingStatus op##_record(State & state, Instruction const & inst, Instruction const ** pc) { \
		IRef node_a; \
		RECORDING_DO(get_reg(state,inst.a,&node_a)); \
		*pc = op##_op(state,inst); \
		Value & c = interpreter_reg(state,inst.c); \
		IRef output; \
		/* NYI - correct casting behavior between node_a's type and the type of output */ \
		EMITIR(op,c,node_a,0,&output); \
		TRACE->renaming_table.assign(RenamingTable::REG,inst.c,output); \
		return RecordingStatus::NO_ERROR; \
}

RecordingStatus call_record(State & state, Instruction const & inst, Instruction const ** pc) {
	state.tracing.current_trace->depth++;
	OP_NOT_IMPLEMENTED(call);
}
RecordingStatus get_record(State & state, Instruction const & inst, Instruction const ** pc) {
	*pc = get_op(state,inst);
	IRef node;
	RECORDING_DO(get_var(state,interpreter_reg(state,inst.c),inst.a,&node));
	TRACE->renaming_table.assign(RenamingTable::REG,inst.c,node);
	return RecordingStatus::NO_ERROR;
}
RecordingStatus sget_record(State & state, Instruction const & inst, Instruction const ** pc) {
	*pc = sget_op(state,inst);
	IRef node;
	RECORDING_DO(get_slot(state,interpreter_reg(state,inst.c),inst.a,&node));
	TRACE->renaming_table.assign(RenamingTable::REG,inst.c,node);
	return RecordingStatus::NO_ERROR;
}

RecordingStatus kget_record(State & state, Instruction const & inst, Instruction const ** pc) {
	*pc = kget_op(state,inst);
	RECORDING_DO(load_constant(state,inst.c,interpreter_reg(state,inst.c)));
	return RecordingStatus::NO_ERROR;
}

RecordingStatus iget_record(State & state, Instruction const & inst, Instruction const ** pc) {
	OP_NOT_IMPLEMENTED(iget);
}

RecordingStatus assign_record(State & state, Instruction const & inst, Instruction const ** pc) {
	*pc = assign_op(state,inst);
	IRef node;
	RECORDING_DO(get_reg(state,inst.c,&node));
	TRACE->renaming_table.assign(RenamingTable::VARIABLE,inst.a,node);
	return RecordingStatus::NO_ERROR;
}
RecordingStatus sassign_record(State & state, Instruction const & inst, Instruction const ** pc) {
	*pc = sassign_op(state,inst);
	IRef node;
	RECORDING_DO(get_reg(state,inst.c,&node));
	TRACE->renaming_table.assign(RenamingTable::SLOT,inst.a,node);
	return RecordingStatus::NO_ERROR;
}

RecordingStatus iassign_record(State & state, Instruction const & inst, Instruction const ** pc) {
	OP_NOT_IMPLEMENTED(iassign);
}
RecordingStatus eassign_record(State & state, Instruction const & inst, Instruction const ** pc) {
	OP_NOT_IMPLEMENTED(eassign);
}
RecordingStatus forbegin_record(State & state, Instruction const & inst, Instruction const ** pc) {
	OP_NOT_IMPLEMENTED(forbegin);
}
RecordingStatus forend_record(State & state, Instruction const & inst, Instruction const ** pc) {
	OP_NOT_IMPLEMENTED(forend);
}
RecordingStatus iforbegin_record(State & state, Instruction const & inst, Instruction const ** pc) {
	OP_NOT_IMPLEMENTED(iforbegin);
}
RecordingStatus iforend_record(State & state, Instruction const & inst, Instruction const ** pc) {
	OP_NOT_IMPLEMENTED(iforend);
}
RecordingStatus whilebegin_record(State & state, Instruction const & inst, Instruction const ** pc) {
	*pc = whilebegin_op(state,inst);
	int64_t offset = *pc - &inst;
	RECORDING_DO(load_constant(state,inst.c,interpreter_reg(state,inst.c)));
	const Instruction * other_branch = &inst + ((offset == 1) ? inst.a : 1);
	RECORDING_DO(insert_guard(state,inst.b,(offset == 1),other_branch,std::max(inst.b,inst.c)));
	return RecordingStatus::NO_ERROR;
}
RecordingStatus whileend_record(State & state, Instruction const & inst, Instruction const ** pc) {
	*pc = whileend_op(state,inst);
	int64_t offset = *pc - &inst;
	const Instruction * other_branch = &inst + ( (offset == 1) ? inst.a : 1 );
	RECORDING_DO(insert_guard(state,inst.b,offset != 1,other_branch,inst.b));
	return RecordingStatus::NO_ERROR;
}
RecordingStatus repeatbegin_record(State & state, Instruction const & inst, Instruction const ** pc) {
	*pc = repeatbegin_op(state,inst);
	RECORDING_DO(load_constant(state,inst.c,interpreter_reg(state,inst.c)));
	return RecordingStatus::NO_ERROR;
}
RecordingStatus repeatend_record(State & state, Instruction const & inst, Instruction const ** pc) {
	//this is just a constant jump, nothing to record
	*pc = repeatend_op(state,inst);
	return RecordingStatus::NO_ERROR;
}
RecordingStatus next_record(State & state, Instruction const & inst, Instruction const ** pc) {
	//this is just a constant jump, nothing to record
	*pc = next_op(state,inst);
	return RecordingStatus::NO_ERROR;
}
RecordingStatus break1_record(State & state, Instruction const & inst, Instruction const ** pc) {
	//this is just a constant jump, nothing to record
	*pc = break1_op(state,inst);
	return RecordingStatus::NO_ERROR;
}
RecordingStatus if1_record(State & state, Instruction const & inst, Instruction const ** pc) {
	*pc = if1_op(state,inst);
	int64_t offset = *pc - &inst;
	const Instruction * other_branch = &inst + ((offset == 1) ? inst.a : 1);
	RECORDING_DO(insert_guard(state,inst.b,(offset == 1),other_branch,inst.b));
	return RecordingStatus::NO_ERROR;
}
RecordingStatus if0_record(State & state, Instruction const & inst, Instruction const ** pc) {
	*pc = if0_op(state,inst);
	int64_t offset = *pc - &inst;
	const Instruction * other_branch = &inst + ((offset == 1) ? inst.a : 1);
	RECORDING_DO(insert_guard(state,inst.b,(offset != 1),other_branch,inst.b));
	return RecordingStatus::NO_ERROR;
}
RecordingStatus colon_record(State & state, Instruction const & inst, Instruction const ** pc) {
	OP_NOT_IMPLEMENTED(colon);
}

BINARY_OP(add)

UNARY_OP(pos)

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

UNARY_OP(abs)
UNARY_OP(sign)
UNARY_OP(sqrt)
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
UNARY_OP(istrue)
UNARY_OP(logical1)
UNARY_OP(integer1)
UNARY_OP(double1)
UNARY_OP(complex1)
UNARY_OP(character1)


RecordingStatus jmp_record(State & state, Instruction const & inst, Instruction const ** pc) {
	//this is just a constant jump, nothing to record
	*pc = next_op(state,inst);
	return RecordingStatus::NO_ERROR;
}
RecordingStatus function_record(State & state, Instruction const & inst, Instruction const ** pc) {
	OP_NOT_IMPLEMENTED(function);
}

RecordingStatus raw1_record(State & state, Instruction const & inst, Instruction const ** pc) {
	*pc = raw1_op(state,inst);
	RECORDING_DO(load_constant(state,inst.c,interpreter_reg(state,inst.c)));
	return RecordingStatus::NO_ERROR;
}
RecordingStatus UseMethod_record(State & state, Instruction const & inst, Instruction const ** pc) {
	OP_NOT_IMPLEMENTED(UseMethod);
}
RecordingStatus seq_record(State & state, Instruction const & inst, Instruction const ** pc) {
	OP_NOT_IMPLEMENTED(seq);
}
RecordingStatus type_record(State & state, Instruction const & inst, Instruction const ** pc) {
	//we type specialize, so this value is a constant
	*pc = type_op(state,inst);
	RECORDING_DO(load_constant(state,inst.c,interpreter_reg(state,inst.c)));
	return RecordingStatus::NO_ERROR;
}
RecordingStatus invoketrace_record(State & state, Instruction const & inst, Instruction const ** pc) {
	return RecordingStatus::RECURSION;
}
RecordingStatus ret_record(State & state, Instruction const & inst, Instruction const ** pc) {
	*pc = ret_op(state,inst);
	if(TRACE->depth == 0)
		return RecordingStatus::RETURN;
	else
		TRACE->depth--;
	return RecordingStatus::NO_ERROR;
}
RecordingStatus done_record(State & state, Instruction const & inst, Instruction const ** pc) {
	return RecordingStatus::RETURN;
}


//check trace exit conditions
// -- do we need to abort due to conditions applying to all opcodes? if so, abort the trace, then exit the recorder
// -- is the trace complete? if so, install the trace, and exit the recorder
// -- otherwise, the recorder continues normally
//returns true if we should continue recording
static RecordingStatus recording_check_conditions(State& state, Instruction const * inst) {
	if((int64_t)TRACE->recorded.size() > state.tracing.max_length) {
		return RecordingStatus::LENGTH;
	}
	if(TRACE->depth == 0 && TRACE->recorded.size() > 0 && TRACE->trace_start == inst) {
		return RecordingStatus::FOUND_LOOP;
	}
	return RecordingStatus::NO_ERROR;
}

Instruction const * recording_interpret(State& state, Instruction const* pc) {
	RecordingStatus status;
	state.tracing.current_trace = new Trace(const_cast<Code*>(state.frame().code),const_cast<Instruction*>(pc));
	while(true) {
#define RUN_RECORD(name,str,p) case ByteCode::E_##name: { status = name##_record(state, *pc,&pc); } break;
		switch(pc->bc.Enum()) {
			BC_ENUM(RUN_RECORD,0)
		}
#undef RUN_RECORD
		if(   RecordingStatus::NO_ERROR != status
		   || RecordingStatus::NO_ERROR != (status = recording_check_conditions(state,pc))) {
			recording_end(state,status);
			return pc;
		}
	}
}
