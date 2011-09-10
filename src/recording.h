#ifndef _RIPOSTE_RECORDING_H
#define _RIPOSTE_RECORDING_H

struct State;
struct Instruction;
Instruction const * recording_interpret(State& state, Instruction const* pc);

#endif
