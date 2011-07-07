#include "trace.h"
#include "interpreter.h"



Trace::Trace(Code * code, Instruction * trace_start)
: trace_inst(*trace_start) {
	this->code = code;
	this->trace_start = trace_start;
	trace_inst.bc = bytecode_for_threaded_inst(code,trace_start);
}



void trace_compile_and_install(State & state, Trace * trace) {
	//NYI: compile trace

	//patch trace into bytecode
	Code * code = trace->code;
	Instruction * inst = trace->trace_start;

	printf("trace: patching into bytecode: %s\n\n",trace->trace_inst.toString().c_str());
	printf("%s\n",state.stringify(*trace).c_str());
	code->traces.push_back(trace);

	int64_t offset = trace->trace_start - &code->tbc[0];

	//we need to patch both tbc and bc to be consistent
	Instruction * bc = const_cast<Instruction*>(&code->bc[offset]);
	inst->ibc = interpreter_label_for(ByteCode::invoketrace,false);
	inst->a = code->traces.size() - 1;
	bc->bc = ByteCode::invoketrace;
	bc->a = inst->a;
}
