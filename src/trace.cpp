#include "trace.h"
#include "interpreter.h"

#include "math.h"
#include <stdlib.h>

void Trace::reset() {
	n_nodes = n_recorded = length = n_outputs = 0;
}

void Trace::execute(State & state) {
	//remove outputs that have been killed
	for(size_t i = 0; i < n_outputs; ) {
		Output & o = outputs[i];
		if(o.location->header != Type::Future || (uint64_t) o.location->i != o.ref) {
			o = outputs[--n_outputs];
		} else {
			i++;
		}
	}
	printf("executing trace:\n%s\n",toString().c_str());
	//allocate outputs
	for(size_t i = 0; i < n_outputs; i++) {
		Output & o = outputs[i];
		*o.location = Double(length);
	}
	
	for(size_t i = 0; i < length; i += TRACE_VECTOR_WIDTH) {
		for(size_t j = 0; j < n_nodes; j++) {
			IRNode & node = nodes[j];		
#define BINARY_IMPL(op,nm,body) \
case IROpCode :: op : { \
	double * av = registers[node.a]; \
	double * bv = registers[node.b]; \
	double * cv = registers[j]; \
	for(size_t z = 0; z < TRACE_VECTOR_WIDTH; z++) { \
		double a = av[z]; \
		double b = bv[z]; \
		cv[z] = body; \
	} \
} break;
#define UNARY_IMPL(op,nm,body) \
case IROpCode :: op : { \
	double * av = registers[node.a]; \
	double * cv = registers[j]; \
	for(size_t z = 0; z < TRACE_VECTOR_WIDTH; z++) { \
		double a = av[z]; \
		cv[z] = body; \
	} \
} break;
	
			switch(node.opcode) {
				IR_BINARY(BINARY_IMPL)
				IR_UNARY(UNARY_IMPL)
				case IROpCode::vload: {
					double * cv = registers[j];
					for(size_t z = 0; z < TRACE_VECTOR_WIDTH; z++)
						cv[z] = node.reg_a[i + z];
				} break;
				case IROpCode::broadcast: {
					double * cv = registers[j];
					for(size_t z = 0; z < TRACE_VECTOR_WIDTH; z++)
						cv[z] = node.const_a;
				} break;
			}
		}
		//write outputs
		for(size_t j = 0; j < n_outputs; j++) {
			Output & o = outputs[j];
			double * ov = (double *) o.location->p;
			double * av = registers[o.ref];
			for(size_t z = 0; z < TRACE_VECTOR_WIDTH; z++) {
				ov[i + z] = av[z];
			}
		}
	}
}

std::string Trace::toString() {
	std::ostringstream out;
	out << "recorded: \n";
	for(size_t j = 0; j < n_nodes; j++) {
		IRNode & node = nodes[j];
		out << node.toString() << "\n";
	}
	out << "outputs: \n";
	for(size_t i = 0; i < n_outputs; i++) {
		Output & o = outputs[i];
		out << o.ref << "\n";
	}
	return out.str();
}
