#ifndef _RIPOSTE_RECORDING_H
#define _RIPOSTE_RECORDING_H
#include "bc.h"
#include "value.h"

void recording_unpatch_inst(Code const * code, Instruction const & inst);
int64_t recording_patch_inst(State& state,Code const * code, Instruction const & inst, int64_t offset);


#define DECLARE_RECORD_FNS(bc,name,p) \
		int64_t bc##_record(State& state, Code const* code, Instruction const& inst);

BC_ENUM(DECLARE_RECORD_FNS,0)

#endif
