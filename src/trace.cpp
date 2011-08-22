#include "trace.h"
#include "interpreter.h"



Trace::Trace(Prototype * code, Instruction * trace_start)
: trace_inst(*trace_start) {
	this->code = code;
	this->trace_start = trace_start;
#ifdef THREAED_INTERPRETER
#error  "Need to get bytecode for trace_inst"
#endif
}



void trace_compile_and_install(State & state, Trace * trace) {

	trace->optimize();
	printf("trace:\n%s\n",state.stringify(*trace).c_str());
	trace->compiled.reset( TraceCompiler::create(trace) );

	TCStatus::Enum status = trace->compiled->compile();
	if(TCStatus::SUCCESS != status) {
		printf("trace: compiler error: %s\n",TCStatus::toString(status));
		return;
	}

	//patch trace into bytecode
	Prototype * code = trace->code;
	Instruction * inst = trace->trace_start;

	printf("trace: patching into bytecode: %s\n\n",trace->trace_inst.toString().c_str());
	code->traces.push_back(trace);

	//currently we assume inst is not a threaded instruction
	inst->bc = ByteCode::invoketrace;
	inst->a = code->traces.size() - 1;
#ifdef THREAED_INTERPRETER
#error  "Need to get patch the threaded bytecodes as well as the non-thread ones"
#endif
}


void Trace::optimize() {
	optimized.clear();
	optimized.insert(optimized.end(),recorded.begin(),recorded.end());
	std::vector<bool> loop_invariant(optimized.size(),false);

	for(IRef i = 0; i < optimized.size(); i++) {
		IRNode & node = optimized[i];
		uint32_t location;
		if(renaming_table.locationFor(node.opcode,&location)) {
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
