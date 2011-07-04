#include "recording.h"
#include "interpreter.h"

static void unpatch_inst(Code const * code, Instruction const & inst) {
	const_cast<Instruction&>(inst).ibc = interpreter_label_for(bytecode_for_threaded_inst(code,&inst),false);
}

static int64_t patch_inst(State& state,Code const * code, Instruction const & inst, int64_t offset) {
	if(state.recording_trace) {
		Instruction * next = const_cast<Instruction*>(&inst + offset);
		next->ibc = interpreter_label_for(bytecode_for_threaded_inst(code,next),true);
	}
	return offset;
}

#define ENTER_RECORDING_OP unpatch_inst(code,inst)
#define LEAVE_RECORDING_OP(offset) return patch_inst(state,code,inst,offset)

#define OP_NOT_IMPLEMENTED(bc) \
	ENTER_RECORDING_OP; \
	printf("recording " #bc "\n"); \
int64_t offset = bc##_op(state,code,inst); \
	LEAVE_RECORDING_OP(offset); \


int64_t call_record(State& state, Code const* code, Instruction const& inst) {
	OP_NOT_IMPLEMENTED(call);
}
int64_t get_record(State& state, Code const* code, Instruction const& inst) {
	OP_NOT_IMPLEMENTED(get);
}
int64_t kget_record(State& state, Code const* code, Instruction const& inst) {
	OP_NOT_IMPLEMENTED(kget);
}
int64_t iget_record(State& state, Code const* code, Instruction const& inst) {
	OP_NOT_IMPLEMENTED(iget);
}
int64_t assign_record(State& state, Code const* code, Instruction const& inst) {
	OP_NOT_IMPLEMENTED(assign);
}
int64_t iassign_record(State& state, Code const* code, Instruction const& inst) {
	OP_NOT_IMPLEMENTED(iassign);
}
int64_t eassign_record(State& state, Code const* code, Instruction const& inst) {
	OP_NOT_IMPLEMENTED(eassign);
}
int64_t forbegin_record(State& state, Code const* code, Instruction const& inst) {
	OP_NOT_IMPLEMENTED(forbegin);
}
int64_t forend_record(State& state, Code const* code, Instruction const& inst) {
	OP_NOT_IMPLEMENTED(forend);
}
int64_t iforbegin_record(State& state, Code const* code, Instruction const& inst) {
	OP_NOT_IMPLEMENTED(iforbegin);
}
int64_t iforend_record(State& state, Code const* code, Instruction const& inst) {
	OP_NOT_IMPLEMENTED(iforend);
}
int64_t whilebegin_record(State& state, Code const* code, Instruction const& inst) {
	OP_NOT_IMPLEMENTED(whilebegin);
}
int64_t whileend_record(State& state, Code const* code, Instruction const& inst) {
	OP_NOT_IMPLEMENTED(whileend);
}
int64_t repeatbegin_record(State& state, Code const* code, Instruction const& inst) {
	OP_NOT_IMPLEMENTED(repeatbegin);
}
int64_t repeatend_record(State& state, Code const* code, Instruction const& inst) {
	OP_NOT_IMPLEMENTED(repeatend);
}
int64_t next_record(State& state, Code const* code, Instruction const& inst) {
	OP_NOT_IMPLEMENTED(next);
}
int64_t break1_record(State& state, Code const* code, Instruction const& inst) {
	OP_NOT_IMPLEMENTED(break1);
}
int64_t if1_record(State& state, Code const* code, Instruction const& inst) {
	OP_NOT_IMPLEMENTED(if1);
}
int64_t if0_record(State& state, Code const* code, Instruction const& inst) {
	OP_NOT_IMPLEMENTED(if0);
}
int64_t colon_record(State& state, Code const* code, Instruction const& inst) {
	OP_NOT_IMPLEMENTED(colon);
}
int64_t add_record(State& state, Code const* code, Instruction const& inst) {
	OP_NOT_IMPLEMENTED(add);
}
int64_t pos_record(State& state, Code const* code, Instruction const& inst) {
	OP_NOT_IMPLEMENTED(pos);
}
int64_t sub_record(State& state, Code const* code, Instruction const& inst) {
	OP_NOT_IMPLEMENTED(sub);
}
int64_t neg_record(State& state, Code const* code, Instruction const& inst) {
	OP_NOT_IMPLEMENTED(neg);
}
int64_t mul_record(State& state, Code const* code, Instruction const& inst) {
	OP_NOT_IMPLEMENTED(mul);
}
int64_t div_record(State& state, Code const* code, Instruction const& inst) {
	OP_NOT_IMPLEMENTED(div);
}
int64_t idiv_record(State& state, Code const* code, Instruction const& inst) {
	OP_NOT_IMPLEMENTED(idiv);
}
int64_t mod_record(State& state, Code const* code, Instruction const& inst) {
	OP_NOT_IMPLEMENTED(mod);
}
int64_t pow_record(State& state, Code const* code, Instruction const& inst) {
	OP_NOT_IMPLEMENTED(pow);
}
int64_t lt_record(State& state, Code const* code, Instruction const& inst) {
	OP_NOT_IMPLEMENTED(lt);
}
int64_t gt_record(State& state, Code const* code, Instruction const& inst) {
	OP_NOT_IMPLEMENTED(gt);
}
int64_t eq_record(State& state, Code const* code, Instruction const& inst) {
	OP_NOT_IMPLEMENTED(eq);
}
int64_t neq_record(State& state, Code const* code, Instruction const& inst) {
	OP_NOT_IMPLEMENTED(neq);
}
int64_t ge_record(State& state, Code const* code, Instruction const& inst) {
	OP_NOT_IMPLEMENTED(ge);
}
int64_t le_record(State& state, Code const* code, Instruction const& inst) {
	OP_NOT_IMPLEMENTED(le);
}
int64_t lnot_record(State& state, Code const* code, Instruction const& inst) {
	OP_NOT_IMPLEMENTED(lnot);
}
int64_t land_record(State& state, Code const* code, Instruction const& inst) {
	OP_NOT_IMPLEMENTED(land);
}
int64_t lor_record(State& state, Code const* code, Instruction const& inst) {
	OP_NOT_IMPLEMENTED(lor);
}
int64_t abs_record(State& state, Code const* code, Instruction const& inst) {
	OP_NOT_IMPLEMENTED(abs);
}
int64_t sign_record(State& state, Code const* code, Instruction const& inst) {
	OP_NOT_IMPLEMENTED(sign);
}
int64_t sqrt_record(State& state, Code const* code, Instruction const& inst) {
	OP_NOT_IMPLEMENTED(sqrt);
}
int64_t floor_record(State& state, Code const* code, Instruction const& inst) {
	OP_NOT_IMPLEMENTED(floor);
}
int64_t ceiling_record(State& state, Code const* code, Instruction const& inst) {
	OP_NOT_IMPLEMENTED(ceiling);
}
int64_t trunc_record(State& state, Code const* code, Instruction const& inst) {
	OP_NOT_IMPLEMENTED(trunc);
}
int64_t round_record(State& state, Code const* code, Instruction const& inst) {
	OP_NOT_IMPLEMENTED(round);
}
int64_t signif_record(State& state, Code const* code, Instruction const& inst) {
	OP_NOT_IMPLEMENTED(signif);
}
int64_t exp_record(State& state, Code const* code, Instruction const& inst) {
	OP_NOT_IMPLEMENTED(exp);
}
int64_t log_record(State& state, Code const* code, Instruction const& inst) {
	OP_NOT_IMPLEMENTED(log);
}
int64_t cos_record(State& state, Code const* code, Instruction const& inst) {
	OP_NOT_IMPLEMENTED(cos);
}
int64_t sin_record(State& state, Code const* code, Instruction const& inst) {
	OP_NOT_IMPLEMENTED(sin);
}
int64_t tan_record(State& state, Code const* code, Instruction const& inst) {
	OP_NOT_IMPLEMENTED(tan);
}
int64_t acos_record(State& state, Code const* code, Instruction const& inst) {
	OP_NOT_IMPLEMENTED(acos);
}
int64_t asin_record(State& state, Code const* code, Instruction const& inst) {
	OP_NOT_IMPLEMENTED(asin);
}
int64_t atan_record(State& state, Code const* code, Instruction const& inst) {
	OP_NOT_IMPLEMENTED(atan);
}
int64_t jmp_record(State& state, Code const* code, Instruction const& inst) {
	OP_NOT_IMPLEMENTED(jmp);
}
int64_t null_record(State& state, Code const* code, Instruction const& inst) {
	OP_NOT_IMPLEMENTED(null);
}
int64_t true1_record(State& state, Code const* code, Instruction const& inst) {
	OP_NOT_IMPLEMENTED(true1);
}
int64_t false1_record(State& state, Code const* code, Instruction const& inst) {
	OP_NOT_IMPLEMENTED(false1);
}
int64_t NA_record(State& state, Code const* code, Instruction const& inst) {
	OP_NOT_IMPLEMENTED(NA);
}
int64_t istrue_record(State& state, Code const* code, Instruction const& inst) {
	OP_NOT_IMPLEMENTED(istrue);
}
int64_t function_record(State& state, Code const* code, Instruction const& inst) {
	OP_NOT_IMPLEMENTED(function);
}
int64_t logical1_record(State& state, Code const* code, Instruction const& inst) {
	OP_NOT_IMPLEMENTED(logical1);
}
int64_t integer1_record(State& state, Code const* code, Instruction const& inst) {
	OP_NOT_IMPLEMENTED(integer1);
}
int64_t double1_record(State& state, Code const* code, Instruction const& inst) {
	OP_NOT_IMPLEMENTED(double1);
}
int64_t complex1_record(State& state, Code const* code, Instruction const& inst) {
	OP_NOT_IMPLEMENTED(complex1);
}
int64_t character1_record(State& state, Code const* code, Instruction const& inst) {
	OP_NOT_IMPLEMENTED(character1);
}
int64_t raw1_record(State& state, Code const* code, Instruction const& inst) {
	OP_NOT_IMPLEMENTED(raw1);
}
int64_t UseMethod_record(State& state, Code const* code, Instruction const& inst) {
	OP_NOT_IMPLEMENTED(UseMethod);
}
int64_t seq_record(State& state, Code const* code, Instruction const& inst) {
	OP_NOT_IMPLEMENTED(seq);
}
int64_t type_record(State& state, Code const* code, Instruction const& inst) {
	OP_NOT_IMPLEMENTED(type);
}
int64_t ret_record(State& state, Code const* code, Instruction const& inst) {
	ENTER_RECORDING_OP;
	printf("ret!\n");
	return 0; //execute the ret operation again, but do not record it, this will cause eval to return.
}
