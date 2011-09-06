#include "trace.h"
#include "interpreter.h"

#include "math.h"
#include <stdlib.h>

void Trace::reset() {
	n_nodes = n_recorded = length = n_outputs = 0;
}


static void print_regs(int foo) {
	for(int i = 0; i < 32; i++)
		if( foo & (1 << i))
			printf("1");
		else
			printf("0");
	printf("\n");
}
void Trace::execute(State & state) {
	//remove outputs that have been killed, allocate space for valid onces
	for(size_t i = 0; i < n_outputs; ) {
		Output & o = outputs[i];

		const Value & loc = (o.is_variable) ? state.frame.environment->hget(Symbol(o.variable)) : *o.location;
		if(loc.header != Type::Future || (uint64_t) loc.i != o.ref) {
			o = outputs[--n_outputs];
		} else {
			nodes[o.ref].is_output = true;
			if(nodes[o.ref].reg_r == NULL) //if this is the first VM value that refers to this output, allocate space for it in the VM
				nodes[o.ref].reg_r = new (GC) double[length];
			Value v;
			Value::Init(v,Type::Double,length);
			v.p = nodes[o.ref].reg_r;
			if(o.is_variable)
				state.frame.environment->hassign(Symbol(o.variable),v);
			else
				*(o.location) = v;
			i++;
		}
	}

	printf("executing trace:\n%s\n",toString().c_str());

	//register allocate
	//we got backward through the trace so we see all uses before the def
	//when we encounter the first use we allocate a register
	//when we encounter the def, we free that register

	uint32_t free_regs = ~0;
	for(size_t i = n_nodes; i > 0; i--) {
		IRNode & n = nodes[i - 1];
		if(!n.is_output) { //outputs do not get assigned registers, we use the memory previous allocated for them to hold intermediates
			//handle the def of node n by freeing allocated register
			int reg = (n.reg_r - registers[0]) / TRACE_VECTOR_WIDTH;
			//printf("freeing register %d\n",reg);
			free_regs |= (1 << reg);
			//print_regs(free_regs);
		}
		if(n.atyp == IRNode::I_REG) { //a is a register, handle the use of a
			IRNode & def = nodes[n.a.i];
			if(def.is_output) { //since 'a' refers to an output, the interpreter will need to advance the memory reference on each iteration
				                //we set a's type to I_VECTOR so it knows to do this
				n.atyp = IRNode::I_VECTOR;
			} else if(def.reg_r == NULL) { //no register has be assigned to the def. This is the first encountered use so we allocate a register for it
				//allocate a register
				//TODO: we need to make sure we can't run out of registers
				int reg = ffs(free_regs) - 1;
				//printf("allocated register %d\n",reg);
				free_regs &= ~(1 << reg);
				//print_regs(free_regs);
				def.reg_r = registers[reg];
			}
			//replace the reference to the node with the pointer to where the values will be stored, this is either a register, part of an input array, or part of an output array
			n.a.p = def.reg_r;
		}
		//we handle the use of b similar to the use of a
		if(n.btyp == IRNode::I_REG) {
			IRNode & def = nodes[n.b.i];
			if(def.is_output) {
				n.btyp = IRNode::I_VECTOR;
			} else {
				//allocate a register
				//TODO: we need to make sure we can't run out of register
				int reg = ffs(free_regs) - 1;
				//printf("allocated register %d\n",reg);
				free_regs &= ~(1 << reg);
				//print_regs(free_regs);
				def.reg_r = registers[reg];
			}
			n.b.p = def.reg_r;
		}
	}
	
	for(int64_t i = 0; i < length; i += TRACE_VECTOR_WIDTH) {
		for(size_t j = 0; j < n_nodes; j++) {
			IRNode & node = nodes[j];		
#define BINARY_IMPL(op,nm,body) \
case IROpCode :: op : { \
	if(node.atyp == IRNode::I_CONST) { \
		double a = node.a.d; \
		double * bv = node.b.p; \
		double * cv = node.reg_r; \
		for(size_t z = 0; z < TRACE_VECTOR_WIDTH; z++) { \
			double b = bv[z]; \
			cv[z] = body; \
		} \
		if(node.is_output) \
			node.reg_r += TRACE_VECTOR_WIDTH; \
		if(node.btyp == IRNode::I_VECTOR) \
			node.b.p += TRACE_VECTOR_WIDTH; \
	} else if(node.btyp == IRNode::I_CONST) { \
		double * av = node.a.p; \
		double b = node.b.d; \
		double * cv = node.reg_r; \
		for(size_t z = 0; z < TRACE_VECTOR_WIDTH; z++) { \
			double a = av[z]; \
			cv[z] = body; \
		} \
		if(node.is_output) \
			node.reg_r += TRACE_VECTOR_WIDTH; \
		if(node.atyp == IRNode::I_VECTOR) \
			node.a.p += TRACE_VECTOR_WIDTH; \
	} else { \
		double * av = node.a.p; \
		double * bv = node.b.p; \
		double * cv = node.reg_r; \
		for(size_t z = 0; z < TRACE_VECTOR_WIDTH; z++) { \
			double a = av[z]; \
			double b = bv[z]; \
			cv[z] = body; \
		} \
		if(node.is_output) \
			node.reg_r += TRACE_VECTOR_WIDTH; \
		if(node.atyp == IRNode::I_VECTOR) \
			node.a.p += TRACE_VECTOR_WIDTH; \
		if(node.btyp == IRNode::I_VECTOR) \
			node.b.p += TRACE_VECTOR_WIDTH; \
	} \
} break;
#define UNARY_IMPL(op,nm,body) \
case IROpCode :: op : { \
	double * av = node.a.p; \
	double * cv = node.reg_r; \
	for(size_t z = 0; z < TRACE_VECTOR_WIDTH; z++) { \
		double a = av[z]; \
		cv[z] = body; \
	} \
} break;
	
			switch(node.op.code) {
				IR_BINARY(BINARY_IMPL)
				IR_UNARY(UNARY_IMPL)
			}
		}
	}
}

std::string Trace::toString() {
	std::ostringstream out;
	out << "recorded: \n";
	for(size_t j = 0; j < n_nodes; j++) {
		IRNode & node = nodes[j];
		out << "r" << j << ": " << node.toString() << "\n";
	}
	out << "outputs: \n";
	for(size_t i = 0; i < n_outputs; i++) {
		Output & o = outputs[i];
		out << o.ref << "\n";
	}
	return out.str();
}
