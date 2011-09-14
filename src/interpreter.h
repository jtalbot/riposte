#ifndef _RIPOSTE_INTERPRETER_H
#define _RIPOSTE_INTERPRETER_H

#include "bc.h"
#include "value.h"

#define DECLARE_INTERPRETER_FNS(bc,name,...) \
		Instruction const * bc##_op(State& state, Instruction const& inst);

BYTECODES(DECLARE_INTERPRETER_FNS)

/*Value & interpreter_reg(State & state, int64_t i);
Value interpreter_get(State & state, Symbol s);
Value interpreter_sget(State & state, int64_t i);
void interpreter_assign(State & state, Symbol s, Value v);
void interpreter_sassign(State & state, int64_t s, Value v);
*/
#endif
