#ifndef _RIPOSTE_RECORDING_H
#define _RIPOSTE_RECORDING_H

struct Thread;
struct Instruction;
Instruction const * recording_interpret(Thread& thread, Instruction const* pc);

#endif
