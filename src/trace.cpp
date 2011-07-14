#include "trace.h"
#include "interpreter.h"



Trace::Trace(Code * code, Instruction * trace_start)
: trace_inst(*trace_start) {
	this->code = code;
	this->trace_start = trace_start;
	trace_inst.bc = bytecode_for_threaded_inst(code,trace_start);
}



void trace_compile_and_install(State & state, Trace * trace) {

	trace->optimize();
	printf("trace:\n%s\n",state.stringify(*trace).c_str());
	trace->compiled.reset( TraceCompiler::create(trace) );

	TCStatus status = trace->compiled->compile();
	if(TCStatus::SUCCESS != status) {
		printf("trace: compiler error: %s\n",status.toString());
		return;
	}

	//patch trace into bytecode
	Code * code = trace->code;
	Instruction * inst = trace->trace_start;

	printf("trace: patching into bytecode: %s\n\n",trace->trace_inst.toString().c_str());
	code->traces.push_back(trace);

	int64_t offset = trace->trace_start - &code->tbc[0];

	//we need to patch both tbc and bc to be consistent
	Instruction * bc = const_cast<Instruction*>(&code->bc[offset]);
	inst->ibc = interpreter_label_for(ByteCode::invoketrace,false);
	inst->a = code->traces.size() - 1;
	bc->bc = ByteCode::invoketrace;
	bc->a = inst->a;
}


void Trace::optimize() {
	optimized.clear();
	optimized.insert(optimized.end(),recorded.begin(),recorded.end());
	std::vector<bool> loop_invariant(optimized.size(),false);

	for(IRef i = 0; i < optimized.size(); i++) {
		IRNode & node = optimized[i];
		if(node.opcode == IROpCode::sload || node.opcode == IROpCode::vload) {
			uint32_t location = node.opcode == IROpCode::sload ? RenamingTable::SLOT : RenamingTable::VARIABLE;
			IRef var;
			bool read;
			bool write = false;
			renaming_table.get(location,node.a,renaming_table.current_view(),false,&var,&read,&write);

			loop_invariant[i] = !write;
			if(write) { //create phi node
				node.b = var;
				phis.push_back(i);
			} else {
				loads.push_back(i);
			}

		} else {
			loop_invariant[i] =    ( (node.flags() & IRNode::REF_A) == 0 || loop_invariant[node.a] )
					            && ( (node.flags() & IRNode::REF_B) == 0 || loop_invariant[node.b] );
			if(loop_invariant[i]) {
				if(node.opcode == IROpCode::guard) { //if we move a guard, we have to change the exit point
					node.b = -1; //exit offset is the trace instruction itself
				}
				loop_header.push_back(i);
			} else {
				loop_body.push_back(i);
			}
		}
	}
}
