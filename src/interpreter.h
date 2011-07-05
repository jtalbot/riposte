#ifndef _RIPOSTE_INTERPRETER_H
#define _RIPOSTE_INTERPRETER_H

#include "bc.h"
#include "value.h"

#define DECLARE_INTERPRETER_FNS(bc,name,p) \
		int64_t bc##_op(State& state, Code const* code, Instruction const& inst);

BC_ENUM(DECLARE_INTERPRETER_FNS,0)

const void * interpreter_label_for(ByteCode bc, bool recording);
ByteCode bytecode_for_threaded_inst(Code const * code, Instruction const * inst);

#endif
