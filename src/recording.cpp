#include "recording.h"
#include "interpreter.h"
#include "ir.h"

#define ENUM_RECORDING_STATUS(_) \
	_(NO_ERROR,"NO_ERROR") \
	_(FOUND_LOOP,"trace has found the loop, and can be compiled") \
	_(RETURN,"trace escaped calling scope") \
	_(LENGTH,"trace too big") \
	_(RECURSION,"trace encountered an invoke trace node") \
	_(UNSUPPORTED_OP,"trace encountered unsupported op") \
	_(UNSUPPORTED_TYPE,"trace encountered an unsupported type") \

DECLARE_ENUM(RecordingStatus,ENUM_RECORDING_STATUS)
DEFINE_ENUM_TO_STRING(RecordingStatus,ENUM_RECORDING_STATUS)

//for brevity
#define TRACE (state.tracing.current_trace)

static RecordingStatus::Enum emitir(State & state, IROpCode::Enum op, IRType const & typ, int64_t a, int64_t b, IRef * r) {
	if(typ.base_type == IRScalarType::T_unsupported)
		return RecordingStatus::UNSUPPORTED_TYPE;
	TRACE->recorded.push_back(IRNode(op,typ,a,b));
	if(r)
		*r = TRACE->recorded.size() - 1;
	return RecordingStatus::NO_ERROR;
}

static RecordingStatus::Enum emitir(State & state, IROpCode::Enum op, Value const & v, int64_t a, int64_t b, IRef * r) {
	IRType typ(v);
	return emitir(state,op,typ,a,b,r);
}

#define EMITIR(op,v,b,c,aptr) RECORDING_DO(emitir(state,IROpCode::op,v,b,c,aptr))

static void recording_end(State & state, RecordingStatus::Enum reason) {
	switch(reason) {
	case RecordingStatus::FOUND_LOOP:
		trace_compile_and_install(state,TRACE);
		TRACE = NULL;
		break;
	default:
		printf("trace aborted: %s\n",RecordingStatus::toString(reason));
		delete TRACE;
		TRACE = NULL;
		break;
	}
}

//attempt to execute fn, otherwise return error code
#define RECORDING_DO(fn) \
	do { \
		RecordingStatus::Enum s = fn; \
		if(RecordingStatus::NO_ERROR != s) { \
			return s; \
		} \
	} while(0)



#define OP_NOT_IMPLEMENTED(bc) \
	printf("NYI " #bc "\n"); \
	return RecordingStatus::UNSUPPORTED_OP;


static RecordingStatus::Enum get_reg(State & state, int64_t slot_id, IRef * node) {
	if(!TRACE->renaming_table.get(RenamingTable::REG,slot_id,node)) {
		Value & value = interpreter_reg(state,slot_id);
		EMITIR(rload,value,slot_id,-1,node);
		TRACE->renaming_table.input(RenamingTable::REG,slot_id,*node);
	}
	return RecordingStatus::NO_ERROR;
}

static RecordingStatus::Enum get_var(State & state, Value & value, int64_t var_id, IRef * node) {
	if(!TRACE->renaming_table.get(RenamingTable::VARIABLE,var_id,node)) {
		EMITIR(vload,value,var_id,-1,node);
		TRACE->renaming_table.input(RenamingTable::VARIABLE,var_id,*node);
	}
	return RecordingStatus::NO_ERROR;
}
static RecordingStatus::Enum get_slot(State & state, Value & value, int64_t slot_id, IRef * node) {
	if(!TRACE->renaming_table.get(RenamingTable::SLOT,slot_id,node)) {
		EMITIR(sload,value,slot_id,-1,node);
		TRACE->renaming_table.input(RenamingTable::SLOT,slot_id,*node);
	}
	return RecordingStatus::NO_ERROR;
}

//convert slot_id into a scalar boolean, or if it already is one, just return it.
static RecordingStatus::Enum get_predicate(State & state, int64_t slot_id, bool invert, IRef * pnode) {
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

static RecordingStatus::Enum insert_guard(State & state, int64_t slot_id, bool invert, const Instruction * other_branch, int64_t n_live_registers) {
	IRef node;
	RECORDING_DO(get_predicate(state,slot_id,invert,&node));
	TraceExit e = { TRACE->renaming_table.create_snapshot(), n_live_registers, other_branch - TRACE->trace_start };
	TRACE->exits.push_back(e);
	EMITIR(guard,IRType::Void(),node,TRACE->exits.size() - 1,NULL);
	return RecordingStatus::NO_ERROR;
}

//many opcodes turn into constant loads
static RecordingStatus::Enum load_constant(State & state, int64_t dest_slot, const Value & value) {
	TRACE->constants.push_back(value);
	IRef node;
	EMITIR(kload,value,TRACE->constants.size() - 1,0,&node);
	TRACE->renaming_table.assign(RenamingTable::REG,dest_slot,node);
	return RecordingStatus::NO_ERROR;
}

static RecordingStatus::Enum promote_types(State & state, IRef a, IRef b, IRef * ao, IRef * bo) {
	IRType const & at = TRACE->recorded[a].typ;
	IRType const & bt = TRACE->recorded[b].typ;
	if(at.base_type == bt.base_type) {
		*ao = a;
		*bo = b;
	} else if(at.base_type < bt.base_type){
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
RecordingStatus::Enum op##_record(State & state, Instruction const & inst, Instruction const ** pc) { \
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
RecordingStatus::Enum op##_record(State & state, Instruction const & inst, Instruction const ** pc) { \
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

RecordingStatus::Enum call_record(State & state, Instruction const & inst, Instruction const ** pc) {
	state.tracing.current_trace->depth++;
	OP_NOT_IMPLEMENTED(call);
}
RecordingStatus::Enum get_record(State & state, Instruction const & inst, Instruction const ** pc) {
	*pc = get_op(state,inst);
	IRef node;
	RECORDING_DO(get_var(state,interpreter_reg(state,inst.c),inst.a,&node));
	TRACE->renaming_table.assign(RenamingTable::REG,inst.c,node);
	return RecordingStatus::NO_ERROR;
}
RecordingStatus::Enum sget_record(State & state, Instruction const & inst, Instruction const ** pc) {
	*pc = sget_op(state,inst);
	IRef node;
	RECORDING_DO(get_slot(state,interpreter_reg(state,inst.c),inst.a,&node));
	TRACE->renaming_table.assign(RenamingTable::REG,inst.c,node);
	return RecordingStatus::NO_ERROR;
}

RecordingStatus::Enum kget_record(State & state, Instruction const & inst, Instruction const ** pc) {
	*pc = kget_op(state,inst);
	RECORDING_DO(load_constant(state,inst.c,interpreter_reg(state,inst.c)));
	return RecordingStatus::NO_ERROR;
}

RecordingStatus::Enum iget_record(State & state, Instruction const & inst, Instruction const ** pc) {
	OP_NOT_IMPLEMENTED(iget);
}

RecordingStatus::Enum assign_record(State & state, Instruction const & inst, Instruction const ** pc) {
	*pc = assign_op(state,inst);
	IRef node;
	RECORDING_DO(get_reg(state,inst.c,&node));
	TRACE->renaming_table.assign(RenamingTable::VARIABLE,inst.a,node);
	return RecordingStatus::NO_ERROR;
}
RecordingStatus::Enum sassign_record(State & state, Instruction const & inst, Instruction const ** pc) {
	*pc = sassign_op(state,inst);
	IRef node;
	RECORDING_DO(get_reg(state,inst.c,&node));
	TRACE->renaming_table.assign(RenamingTable::SLOT,inst.a,node);
	return RecordingStatus::NO_ERROR;
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
RecordingStatus::Enum whilebegin_record(State & state, Instruction const & inst, Instruction const ** pc) {
	*pc = whilebegin_op(state,inst);
	int64_t offset = *pc - &inst;
	RECORDING_DO(load_constant(state,inst.c,interpreter_reg(state,inst.c)));
	const Instruction * other_branch = &inst + ((offset == 1) ? inst.a : 1);
	RECORDING_DO(insert_guard(state,inst.b,(offset == 1),other_branch,std::max(inst.b,inst.c)));
	return RecordingStatus::NO_ERROR;
}
RecordingStatus::Enum whileend_record(State & state, Instruction const & inst, Instruction const ** pc) {
	*pc = whileend_op(state,inst);
	int64_t offset = *pc - &inst;
	const Instruction * other_branch = &inst + ( (offset == 1) ? inst.a : 1 );
	RECORDING_DO(insert_guard(state,inst.b,offset != 1,other_branch,inst.b));
	return RecordingStatus::NO_ERROR;
}
RecordingStatus::Enum repeatbegin_record(State & state, Instruction const & inst, Instruction const ** pc) {
	*pc = repeatbegin_op(state,inst);
	RECORDING_DO(load_constant(state,inst.c,interpreter_reg(state,inst.c)));
	return RecordingStatus::NO_ERROR;
}
RecordingStatus::Enum repeatend_record(State & state, Instruction const & inst, Instruction const ** pc) {
	//this is just a constant jump, nothing to record
	*pc = repeatend_op(state,inst);
	return RecordingStatus::NO_ERROR;
}
RecordingStatus::Enum next_record(State & state, Instruction const & inst, Instruction const ** pc) {
	//this is just a constant jump, nothing to record
	*pc = next_op(state,inst);
	return RecordingStatus::NO_ERROR;
}
RecordingStatus::Enum break1_record(State & state, Instruction const & inst, Instruction const ** pc) {
	//this is just a constant jump, nothing to record
	*pc = break1_op(state,inst);
	return RecordingStatus::NO_ERROR;
}
RecordingStatus::Enum if1_record(State & state, Instruction const & inst, Instruction const ** pc) {
	*pc = if1_op(state,inst);
	int64_t offset = *pc - &inst;
	const Instruction * other_branch = &inst + ((offset == 1) ? inst.a : 1);
	RECORDING_DO(insert_guard(state,inst.b,(offset == 1),other_branch,inst.b));
	return RecordingStatus::NO_ERROR;
}
RecordingStatus::Enum if0_record(State & state, Instruction const & inst, Instruction const ** pc) {
	*pc = if0_op(state,inst);
	int64_t offset = *pc - &inst;
	const Instruction * other_branch = &inst + ((offset == 1) ? inst.a : 1);
	RECORDING_DO(insert_guard(state,inst.b,(offset != 1),other_branch,inst.b));
	return RecordingStatus::NO_ERROR;
}
RecordingStatus::Enum colon_record(State & state, Instruction const & inst, Instruction const ** pc) {
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
BINARY_OP(sland)
BINARY_OP(slor)

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
UNARY_OP(logical1)
UNARY_OP(integer1)
UNARY_OP(double1)
UNARY_OP(complex1)
UNARY_OP(character1)


RecordingStatus::Enum jmp_record(State & state, Instruction const & inst, Instruction const ** pc) {
	//this is just a constant jump, nothing to record
	*pc = next_op(state,inst);
	return RecordingStatus::NO_ERROR;
}
RecordingStatus::Enum function_record(State & state, Instruction const & inst, Instruction const ** pc) {
	OP_NOT_IMPLEMENTED(function);
}

RecordingStatus::Enum raw1_record(State & state, Instruction const & inst, Instruction const ** pc) {
	*pc = raw1_op(state,inst);
	RECORDING_DO(load_constant(state,inst.c,interpreter_reg(state,inst.c)));
	return RecordingStatus::NO_ERROR;
}
RecordingStatus::Enum UseMethod_record(State & state, Instruction const & inst, Instruction const ** pc) {
	OP_NOT_IMPLEMENTED(UseMethod);
}
RecordingStatus::Enum seq_record(State & state, Instruction const & inst, Instruction const ** pc) {
	OP_NOT_IMPLEMENTED(seq);
}
RecordingStatus::Enum type_record(State & state, Instruction const & inst, Instruction const ** pc) {
	//we type specialize, so this value is a constant
	*pc = type_op(state,inst);
	RECORDING_DO(load_constant(state,inst.c,interpreter_reg(state,inst.c)));
	return RecordingStatus::NO_ERROR;
}
RecordingStatus::Enum invoketrace_record(State & state, Instruction const & inst, Instruction const ** pc) {
	return RecordingStatus::RECURSION;
}
RecordingStatus::Enum ret_record(State & state, Instruction const & inst, Instruction const ** pc) {
	*pc = ret_op(state,inst);
	if(TRACE->depth == 0)
		return RecordingStatus::RETURN;
	else
		TRACE->depth--;
	return RecordingStatus::NO_ERROR;
}
RecordingStatus::Enum done_record(State & state, Instruction const & inst, Instruction const ** pc) {
	return RecordingStatus::RETURN;
}


//check trace exit conditions
// -- do we need to abort due to conditions applying to all opcodes? if so, abort the trace, then exit the recorder
// -- is the trace complete? if so, install the trace, and exit the recorder
// -- otherwise, the recorder continues normally
//returns true if we should continue recording
static RecordingStatus::Enum recording_check_conditions(State& state, Instruction const * inst) {
	if((int64_t)TRACE->recorded.size() > state.tracing.max_length) {
		return RecordingStatus::LENGTH;
	}
	if(TRACE->depth == 0 && TRACE->recorded.size() > 0 && TRACE->trace_start == inst) {
		return RecordingStatus::FOUND_LOOP;
	}
	return RecordingStatus::NO_ERROR;
}

Instruction const * recording_interpret(State& state, Instruction const* pc) {
	RecordingStatus::Enum status = RecordingStatus::NO_ERROR;
	state.tracing.current_trace = new Trace(const_cast<Prototype*>(state.frame.prototype),const_cast<Instruction*>(pc));
	while(true) {
#define RUN_RECORD(name,str) case ByteCode::name: { status = name##_record(state, *pc,&pc); } break;
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
}
