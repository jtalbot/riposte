#include "recording.h"
#include "interpreter.h"
#include "ir.h"



void recording_unpatch_inst(Code const * code, Instruction const & inst) {
	const_cast<Instruction&>(inst).ibc = interpreter_label_for(bytecode_for_threaded_inst(code,&inst),false);
}

int64_t recording_patch_inst(State& state,Code const * code, Instruction const & inst, int64_t offset) {
	Instruction * next = const_cast<Instruction*>(&inst + offset);
	next->ibc = interpreter_label_for(bytecode_for_threaded_inst(code,next),true);
	return offset;
}

#define ENUM_RECORDING_STATUS(_,p) \
	_(NO_ERROR,"NO_ERROR",p) \
	_(FOUND_LOOP,"trace has found the loop, and can be compiled",p) \
	_(RETURN,"trace escaped calling scope",p) \
	_(LENGTH,"trace too big",p) \
	_(RECURSION,"trace encountered an invoke trace node",p) \
	_(UNSUPPORTED_OP,"trace encountered unsupported op",p) \
	_(UNSUPPORTED_TYPE,"trace encountered an unsupported type",p) \
	_(NOT_RECORDING,"tracing is not enabled",p) /*this last error is temporary, but needed to exit the recorder when a patched ByteCode enters the recorder when tracing is disabled */


DECLARE_ENUM(RecordingStatus,ENUM_RECORDING_STATUS)
DEFINE_ENUM(RecordingStatus,ENUM_RECORDING_STATUS)
DEFINE_ENUM_TO_STRING(RecordingStatus,ENUM_RECORDING_STATUS)

//for brevity
#define TRACE (state.tracing.current_trace)

static RecordingStatus emitir(State & state, IROpCode op, IRType const & typ, int64_t b, int64_t c, int32_t * a) {
	if(typ.base_type == IRScalarType::T_unsupported)
		return RecordingStatus::UNSUPPORTED_TYPE;
	int64_t next_op = TRACE->recorded.size();
	TRACE->recorded.push_back(IRNode(op,typ,next_op,b,c));
	if(a)
		*a = next_op;
	return RecordingStatus::NO_ERROR;
}

static RecordingStatus emitir(State & state, IROpCode op, Value const & v, int64_t b, int64_t c, int32_t * a) {
	IRType typ(v);
	return emitir(state,op,typ,b,c,a);
}

#define EMITIR(op,v,b,c,aptr) RECORDING_DO(emitir(state,IROpCode :: op,v,b,c,aptr))

static void recording_end(State & state, RecordingStatus reason) {
	switch(reason.Enum()) {
	case RecordingStatus::E_FOUND_LOOP:
		trace_compile_and_install(state,TRACE);
		TRACE = NULL;
		break;
	case RecordingStatus::E_NOT_RECORDING: //TODO: fix the way recording happens so we don't enter a recording function when we are not recording
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
	*offset = 0; \
	return RecordingStatus::UNSUPPORTED_OP;


static RecordingStatus get_slot(State & state, int64_t slot_id, int32_t * node) {
	if(!TRACE->renaming_table.get(RenamingTable::SLOT,slot_id,node)) {
		Value & value = state.Registers[slot_id];
		EMITIR(sload,value,slot_id,-1,node);
		TRACE->renaming_table.input(RenamingTable::SLOT,slot_id,*node);
	}
	return RecordingStatus::NO_ERROR;
}

static RecordingStatus get_var(State & state, int64_t var_id, int32_t * node) {
	if(!TRACE->renaming_table.get(RenamingTable::VARIABLE,var_id,node)) {
		Value value;
		state.global->get(state,Symbol(var_id),value);
		EMITIR(vload,value,var_id,-1,node);
		TRACE->renaming_table.input(RenamingTable::VARIABLE,var_id,*node);
	}
	return RecordingStatus::NO_ERROR;
}

//convert slot_id into a scalar boolean, or if it already is one, just return it.
static RecordingStatus get_predicate(State & state, int64_t slot_id, bool invert, int32_t * pnode) {
	int32_t node;
	RECORDING_DO(get_slot(state,slot_id,&node));
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

//all arithmetic binary ops share the same recording implementation
#define BINARY_OP(op) \
RecordingStatus op##_record_impl(State & state, Code const * code, Instruction const & inst, int64_t * offset) { \
		*offset = op##_op(state,code,inst); \
		Value & c = state.registers[inst.c]; \
		int32_t node_a; \
		int32_t node_b; \
		int32_t output; \
		RECORDING_DO(get_slot(state,inst.a,&node_a)); \
		RECORDING_DO(get_slot(state,inst.b,&node_b)); \
		EMITIR(op,c,node_a,node_b,&output); \
		TRACE->renaming_table.assign(RenamingTable::SLOT,inst.c,output); \
		return RecordingStatus::NO_ERROR; \
}

RecordingStatus call_record_impl(State & state, Code const * code, Instruction const & inst, int64_t * offset) {
	state.tracing.current_trace->depth++;
	OP_NOT_IMPLEMENTED(call);
}
RecordingStatus get_record_impl(State & state, Code const * code, Instruction const & inst, int64_t * offset) {
	*offset = get_op(state,code,inst);
	int32_t node;
	RECORDING_DO(get_var(state,inst.a,&node));
	TRACE->renaming_table.assign(RenamingTable::SLOT,inst.c,node);
	return RecordingStatus::NO_ERROR;
}

RecordingStatus kget_record_impl(State & state, Code const * code, Instruction const & inst, int64_t * offset) {
	*offset = kget_op(state,code,inst);
	Value & value = state.registers[inst.c];
	TRACE->constants.push_back(value);
	int32_t node;
	EMITIR(kload,value,TRACE->constants.size() - 1,0,&node);
	TRACE->renaming_table.assign(RenamingTable::SLOT,inst.c,node);
	return RecordingStatus::NO_ERROR;
}

RecordingStatus iget_record_impl(State & state, Code const * code, Instruction const & inst, int64_t * offset) {
	OP_NOT_IMPLEMENTED(iget);
}

RecordingStatus assign_record_impl(State & state, Code const * code, Instruction const & inst, int64_t * offset) {
	*offset = assign_op(state,code,inst);
	int32_t node;
	RECORDING_DO(get_slot(state,inst.a,&node));
	TRACE->renaming_table.assign(RenamingTable::VARIABLE,inst.c,node);
	return RecordingStatus::NO_ERROR;
}

RecordingStatus iassign_record_impl(State & state, Code const * code, Instruction const & inst, int64_t * offset) {
	OP_NOT_IMPLEMENTED(iassign);
}
RecordingStatus eassign_record_impl(State & state, Code const * code, Instruction const & inst, int64_t * offset) {
	OP_NOT_IMPLEMENTED(eassign);
}
RecordingStatus forbegin_record_impl(State & state, Code const * code, Instruction const & inst, int64_t * offset) {
	OP_NOT_IMPLEMENTED(forbegin);
}
RecordingStatus forend_record_impl(State & state, Code const * code, Instruction const & inst, int64_t * offset) {
	OP_NOT_IMPLEMENTED(forend);
}
RecordingStatus iforbegin_record_impl(State & state, Code const * code, Instruction const & inst, int64_t * offset) {
	OP_NOT_IMPLEMENTED(iforbegin);
}
RecordingStatus iforend_record_impl(State & state, Code const * code, Instruction const & inst, int64_t * offset) {
	OP_NOT_IMPLEMENTED(iforend);
}
RecordingStatus whilebegin_record_impl(State & state, Code const * code, Instruction const & inst, int64_t * offset) {
	OP_NOT_IMPLEMENTED(whilebegin);
}
RecordingStatus whileend_record_impl(State & state, Code const * code, Instruction const & inst, int64_t * offset) {
	*offset = whileend_op(state,code,inst);
	int32_t node;
	RECORDING_DO(get_predicate(state,inst.b,(*offset == 1),&node));
	TraceExit e = { TRACE->renaming_table.create_snapshot() };
	TRACE->exits.push_back(e);
	EMITIR(guard,IRType::Void(),node,TRACE->exits.size() - 1,NULL);
	return RecordingStatus::NO_ERROR;
}
RecordingStatus repeatbegin_record_impl(State & state, Code const * code, Instruction const & inst, int64_t * offset) {
	OP_NOT_IMPLEMENTED(repeatbegin);
}
RecordingStatus repeatend_record_impl(State & state, Code const * code, Instruction const & inst, int64_t * offset) {
	OP_NOT_IMPLEMENTED(repeatend);
}
RecordingStatus next_record_impl(State & state, Code const * code, Instruction const & inst, int64_t * offset) {
	OP_NOT_IMPLEMENTED(next);
}
RecordingStatus break1_record_impl(State & state, Code const * code, Instruction const & inst, int64_t * offset) {
	OP_NOT_IMPLEMENTED(break1);
}
RecordingStatus if1_record_impl(State & state, Code const * code, Instruction const & inst, int64_t * offset) {
	*offset = if1_op(state,code,inst);
	int32_t node;
	RECORDING_DO(get_predicate(state,inst.b,(*offset != 1),&node));
	TraceExit e = { TRACE->renaming_table.create_snapshot() };
	TRACE->exits.push_back(e);
	EMITIR(guard,IRType::Void(),node,TRACE->exits.size() - 1,NULL);
	return RecordingStatus::NO_ERROR;
}
RecordingStatus if0_record_impl(State & state, Code const * code, Instruction const & inst, int64_t * offset) {
	*offset = if0_op(state,code,inst);
	int32_t node;
	RECORDING_DO(get_predicate(state,inst.b,(*offset == 1),&node));
	TraceExit e = { TRACE->renaming_table.create_snapshot() };
	TRACE->exits.push_back(e);
	EMITIR(guard,IRType::Void(),node,TRACE->exits.size() - 1,NULL);
	return RecordingStatus::NO_ERROR;
}
RecordingStatus colon_record_impl(State & state, Code const * code, Instruction const & inst, int64_t * offset) {
	OP_NOT_IMPLEMENTED(colon);
}

BINARY_OP(add)

RecordingStatus pos_record_impl(State & state, Code const * code, Instruction const & inst, int64_t * offset) {
	OP_NOT_IMPLEMENTED(pos);
}

BINARY_OP(sub)

RecordingStatus neg_record_impl(State & state, Code const * code, Instruction const & inst, int64_t * offset) {
	OP_NOT_IMPLEMENTED(neg);
}

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

RecordingStatus abs_record_impl(State & state, Code const * code, Instruction const & inst, int64_t * offset) {
	OP_NOT_IMPLEMENTED(abs);
}
RecordingStatus sign_record_impl(State & state, Code const * code, Instruction const & inst, int64_t * offset) {
	OP_NOT_IMPLEMENTED(sign);
}
RecordingStatus sqrt_record_impl(State & state, Code const * code, Instruction const & inst, int64_t * offset) {
	OP_NOT_IMPLEMENTED(sqrt);
}
RecordingStatus floor_record_impl(State & state, Code const * code, Instruction const & inst, int64_t * offset) {
	OP_NOT_IMPLEMENTED(floor);
}
RecordingStatus ceiling_record_impl(State & state, Code const * code, Instruction const & inst, int64_t * offset) {
	OP_NOT_IMPLEMENTED(ceiling);
}
RecordingStatus trunc_record_impl(State & state, Code const * code, Instruction const & inst, int64_t * offset) {
	OP_NOT_IMPLEMENTED(trunc);
}
RecordingStatus round_record_impl(State & state, Code const * code, Instruction const & inst, int64_t * offset) {
	OP_NOT_IMPLEMENTED(round);
}
RecordingStatus signif_record_impl(State & state, Code const * code, Instruction const & inst, int64_t * offset) {
	OP_NOT_IMPLEMENTED(signif);
}
RecordingStatus exp_record_impl(State & state, Code const * code, Instruction const & inst, int64_t * offset) {
	OP_NOT_IMPLEMENTED(exp);
}
RecordingStatus log_record_impl(State & state, Code const * code, Instruction const & inst, int64_t * offset) {
	OP_NOT_IMPLEMENTED(log);
}
RecordingStatus cos_record_impl(State & state, Code const * code, Instruction const & inst, int64_t * offset) {
	OP_NOT_IMPLEMENTED(cos);
}
RecordingStatus sin_record_impl(State & state, Code const * code, Instruction const & inst, int64_t * offset) {
	OP_NOT_IMPLEMENTED(sin);
}
RecordingStatus tan_record_impl(State & state, Code const * code, Instruction const & inst, int64_t * offset) {
	OP_NOT_IMPLEMENTED(tan);
}
RecordingStatus acos_record_impl(State & state, Code const * code, Instruction const & inst, int64_t * offset) {
	OP_NOT_IMPLEMENTED(acos);
}
RecordingStatus asin_record_impl(State & state, Code const * code, Instruction const & inst, int64_t * offset) {
	OP_NOT_IMPLEMENTED(asin);
}
RecordingStatus atan_record_impl(State & state, Code const * code, Instruction const & inst, int64_t * offset) {
	OP_NOT_IMPLEMENTED(atan);
}
RecordingStatus jmp_record_impl(State & state, Code const * code, Instruction const & inst, int64_t * offset) {
	OP_NOT_IMPLEMENTED(jmp);
}
RecordingStatus null_record_impl(State & state, Code const * code, Instruction const & inst, int64_t * offset) {
	*offset = null_op(state,code,inst);
	Value & v = state.registers[inst.c];
	int32_t node;
	EMITIR(null,v,0,0,&node);
	TRACE->renaming_table.assign(RenamingTable::SLOT,inst.c,node);
	return RecordingStatus::NO_ERROR;
}
RecordingStatus true1_record_impl(State & state, Code const * code, Instruction const & inst, int64_t * offset) {
	OP_NOT_IMPLEMENTED(true1);
}
RecordingStatus false1_record_impl(State & state, Code const * code, Instruction const & inst, int64_t * offset) {
	OP_NOT_IMPLEMENTED(false1);
}
RecordingStatus NA_record_impl(State & state, Code const * code, Instruction const & inst, int64_t * offset) {
	OP_NOT_IMPLEMENTED(NA);
}
RecordingStatus istrue_record_impl(State & state, Code const * code, Instruction const & inst, int64_t * offset) {
	OP_NOT_IMPLEMENTED(istrue);
}
RecordingStatus function_record_impl(State & state, Code const * code, Instruction const & inst, int64_t * offset) {
	OP_NOT_IMPLEMENTED(function);
}
RecordingStatus logical1_record_impl(State & state, Code const * code, Instruction const & inst, int64_t * offset) {
	OP_NOT_IMPLEMENTED(logical1);
}
RecordingStatus integer1_record_impl(State & state, Code const * code, Instruction const & inst, int64_t * offset) {
	OP_NOT_IMPLEMENTED(integer1);
}
RecordingStatus double1_record_impl(State & state, Code const * code, Instruction const & inst, int64_t * offset) {
	OP_NOT_IMPLEMENTED(double1);
}
RecordingStatus complex1_record_impl(State & state, Code const * code, Instruction const & inst, int64_t * offset) {
	OP_NOT_IMPLEMENTED(complex1);
}
RecordingStatus character1_record_impl(State & state, Code const * code, Instruction const & inst, int64_t * offset) {
	OP_NOT_IMPLEMENTED(character1);
}
RecordingStatus raw1_record_impl(State & state, Code const * code, Instruction const & inst, int64_t * offset) {
	OP_NOT_IMPLEMENTED(raw1);
}
RecordingStatus UseMethod_record_impl(State & state, Code const * code, Instruction const & inst, int64_t * offset) {
	OP_NOT_IMPLEMENTED(UseMethod);
}
RecordingStatus seq_record_impl(State & state, Code const * code, Instruction const & inst, int64_t * offset) {
	OP_NOT_IMPLEMENTED(seq);
}
RecordingStatus type_record_impl(State & state, Code const * code, Instruction const & inst, int64_t * offset) {
	OP_NOT_IMPLEMENTED(type);
}
RecordingStatus invoketrace_record_impl(State & state, Code const * code, Instruction const & inst, int64_t * offset) {
	return RecordingStatus::RECURSION;
}
RecordingStatus ret_record_impl(State & state, Code const * code, Instruction const & inst, int64_t * offset) {
	if(TRACE->depth == 0)
		return RecordingStatus::RETURN;
	else
		TRACE->depth--;
	return RecordingStatus::NO_ERROR;
}

//do setup code that is common to all ops
//in particular, we need to check:
// -- have we aborted the trace on the previous instruction? if so, exit the recorder
// -- do we need to abort due to conditions applying to all opcodes? if so, abort the trace, then exit the recorder
// -- is the trace complete? if so, install the trace, and exit the recorder
// -- otherwise, the recorder continues normally
//returns true if we should continue recording
static RecordingStatus recording_enter_op(State& state,Code const * code, Instruction const & inst) {
	recording_unpatch_inst(code,inst);
	if(TRACE == NULL)
		return RecordingStatus::NOT_RECORDING;
	if((int64_t)TRACE->recorded.size() > state.tracing.max_length) {
		return RecordingStatus::LENGTH;
	}
	if(TRACE->depth == 0 && TRACE->recorded.size() > 0 && TRACE->trace_start == &inst) {
		return RecordingStatus::FOUND_LOOP;
	}
	return RecordingStatus::NO_ERROR;
}

//once we can seperate the recording loop from the interpreter, we make this code part of the recording loop
#define CREATE_RECORDING_OP(bc,name,p) \
int64_t bc##_record(State & state, Code const * code, Instruction const & inst) { \
	RecordingStatus status; \
	status = recording_enter_op(state,code,inst); \
	if(RecordingStatus::NO_ERROR != status) { \
		recording_end(state,status); \
		return 0; \
	} \
	int64_t offset = 0; \
	status = bc##_record_impl(state,code,inst,&offset); \
	if(RecordingStatus::NO_ERROR != status) { \
		recording_end(state,status); \
		return offset; \
	} \
	if(ByteCode::ret == ByteCode :: bc) { \
		/* special case: ret needs to cause the recursive interpreter to actually return */ \
		return 0; \
	} \
	recording_patch_inst(state,code,inst,offset); \
	return offset; \
}

BC_ENUM(CREATE_RECORDING_OP,0)
