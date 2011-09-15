#ifndef _RIPOSTE_INTERPRETER_H
#define _RIPOSTE_INTERPRETER_H

#include "bc.h"
#include "value.h"

#define DECLARE_INTERPRETER_FNS(bc,name,...) \
		Instruction const * bc##_op(State& state, Instruction const& inst);

BYTECODES(DECLARE_INTERPRETER_FNS)

Instruction const* get_with_environment(State& state, Instruction const& inst, Environment ** env_p);

#endif
