#ifndef _RIPOSTE_RECORDING_H
#define _RIPOSTE_RECORDING_H
#include "bc.h"
#include "value.h"


Instruction const * recording_interpret(State& state, Instruction const* pc);

#endif
