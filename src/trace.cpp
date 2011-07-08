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
	printf("compiling\n");
	printf("%s\n",state.stringify(*trace).c_str());
	trace->compile();

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


void Trace::compile() {

	std::vector<IRNode> loads;
	std::vector<IRNode> phis;
	std::vector<IRNode> loop_header;
	std::vector<IRNode> loop_body;
	std::vector<bool> loop_invariant(recorded.size(),false);

	for(size_t i = 0; i < recorded.size(); i++) {
		IRNode node = recorded[i];
		if(node.opcode == IROpCode::sload || node.opcode == IROpCode::vload) {
			uint32_t location = node.opcode == IROpCode::sload ? RenamingTable::SLOT : RenamingTable::VARIABLE;
			IRef var;
			bool read,write;
			renaming_table.get(location,node.a,renaming_table.current_view(),&var,&read,&write);

			loop_invariant[i] = !write;
			if(write) { //create phi node
				node.b = var;
				phis.push_back(node);
			} else {
				loads.push_back(node);
			}

		} else {
			loop_invariant[i] =    ( (node.flags() & IRNode::REF_A) == 0 || loop_invariant[node.a] )
					            && ( (node.flags() & IRNode::REF_B) == 0 || loop_invariant[node.b] );
			if(loop_invariant[i]) {
				if(node.opcode == IROpCode::guard) { //if we move a guard, we have to change the exit point
					IRNode n = node;
					n.a = -1; //no exit
					n.b = 0; //exit offset is the trace instruction itself
				}
				loop_header.push_back(node);
			} else {
				loop_body.push_back(node);
			}
		}
	}
	recorded.clear();
	recorded.insert(recorded.end(),loads.begin(),loads.end());
	IRNode foo(IROpCode::null,IRType::Void(),0,0);
	recorded.push_back(foo);
	recorded.insert(recorded.end(),phis.begin(),phis.end());
	recorded.push_back(foo);
	recorded.insert(recorded.end(),loop_header.begin(),loop_header.end());
	recorded.push_back(foo);
	recorded.insert(recorded.end(),loop_body.begin(),loop_body.end());
}
