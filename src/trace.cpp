#include "trace.h"
#include "interpreter.h"
#include "vector.h"
#include "ops.h"
#include "sse.h"

#include <stdlib.h>

void Trace::reset() {
	n_nodes = n_recorded = length = n_outputs = max_live_register = 0;
}


static void print_regs(int foo) {
	for(int i = 0; i < 32; i++)
		if( foo & (1 << i))
			printf("1");
		else
			printf("0");
	printf("\n");
}

#define REG(state, i) (*(state.base+i))

static const Value & get_output_value(State & state, const Trace::Output & o) {
	switch(o.location_type) {
	case Trace::Output::E_REG:
		return REG(state,o.location);
	case Trace::Output::E_SLOT:
		return state.frame.environment->get(o.location);
	default:
	case Trace::Output::E_VAR:
		return state.frame.environment->hget(Symbol(o.location));
	}
}
static void set_output_value(State & state, const Trace::Output & o, const Value & v) {
	switch(o.location_type) {
	case Trace::Output::E_REG:
		REG(state,o.location) = v;
		return;
	case Trace::Output::E_SLOT:
		state.frame.environment->get(o.location) = v;
		return;
	case Trace::Output::E_VAR:
		state.frame.environment->hassign(Symbol(o.location),v);
		return;
	}
}

void Trace::execute(State & state) {
	//remove outputs that have been killed, allocate space for valid onces
	for(size_t i = 0; i < n_outputs; ) {
		Output & o = outputs[i];

		const Value & loc = get_output_value(state,o);
		if(loc.header != Type::Future ||
		   loc.future.ref != o.ref ||
		   (o.location_type == Trace::Output::E_REG && o.location > max_live_register)) {
			o = outputs[--n_outputs];
		} else {
			nodes[o.ref].r_external = true;
			if(nodes[o.ref].r.p == NULL) //if this is the first VM value that refers to this output, allocate space for it in the VM
				nodes[o.ref].r.p = new (PointerFreeGC) double[length];
			Value v;
			Value::Init(v,o.typ,length);
			v.p = nodes[o.ref].r.p;
			set_output_value(state,o,v);
			i++;
		}
	}

	//printf("executing trace:\n%s\n",toString(state).c_str());

	//register allocate
	//we got backward through the trace so we see all uses before the def
	//when we encounter the first use we allocate a register
	//when we encounter the def, we free that register

	uint32_t free_regs = ~0;
	for(size_t i = n_nodes; i > 0; i--) {
		IRNode & n = nodes[i - 1];
		if(!n.r_external) { //outputs do not get assigned registers, we use the memory previous allocated for them to hold intermediates
			//handle the def of node n by freeing allocated register
			int reg = (n.r.p - registers[0]) / TRACE_VECTOR_WIDTH;
			//printf("freeing register %d\n",reg);
			//printf("n%d: reg%d\n",i,reg);
			free_regs |= (1 << reg);
			//print_regs(free_regs);
		}
		if( n.usesRegA() ){ //a is a register, handle the use of a
			IRNode & def = nodes[n.a.i];
			if(def.r_external) { //since 'a' refers to an output, the interpreter will need to advance the memory reference on each iteration
				                //we set a's type to I_VECTOR so it knows to do this
				n.a_external = true;
			} else if(def.r.p == NULL) { //no register has be assigned to the def. This is the first encountered use so we allocate a register for it
				//allocate a register
				//TODO: we need to make sure we can't run out of registers
				int reg = ffs(free_regs) - 1;
				//printf("n%d = register %d\n",n.a.i,reg);
				free_regs &= ~(1 << reg);
				//print_regs(free_regs);
				def.r.p = registers[reg];
			}
			//replace the reference to the node with the pointer to where the values will be stored, this is either a register, part of an input array, or part of an output array
			n.a.p = def.r.p;
		}
		//we handle the use of b similar to the use of a
		if(n.usesRegB()) {
			IRNode & def = nodes[n.b.i];
			if(def.r_external) {
				n.b_external = true;
			} else if (def.r.p == NULL) {
				//allocate a register
				//TODO: we need to make sure we can't run out of register
				int reg = ffs(free_regs) - 1;
				//printf("n%d = register %d\n",n.b.i,reg);
				free_regs &= ~(1 << reg);
				//print_regs(free_regs);
				def.r.p = registers[reg];
			}
			n.b.p = def.r.p;
		}
	}
	
	for(int64_t i = 0; i < length; i += TRACE_VECTOR_WIDTH) {
		for(size_t j = 0; j < n_nodes; j++) {
			IRNode & node = nodes[j];
#define BINARY_CASE(opcode, typea, typeb, sva, svb) \
	((IROpCode::opcode << 4) + (typeb << 3) + (typea << 2) + (svb << 1) + sva)

#define BINARY_IMPL(opcode,nm,OP) \
	case BINARY_CASE(opcode, IROp::T_INT, IROp::T_INT, IROp::E_SCALAR, IROp::E_VECTOR): \
		Map2SV< OP<TInteger>, TRACE_VECTOR_WIDTH >::eval(state, node.a.i, (int64_t*)node.b.p, (OP<TInteger>::R*)node.r.p); break; \
	case BINARY_CASE(opcode, IROp::T_INT, IROp::T_INT, IROp::E_VECTOR, IROp::E_SCALAR): \
		Map2VS< OP<TInteger>, TRACE_VECTOR_WIDTH >::eval(state, (int64_t*)node.a.p, node.b.i, (OP<TInteger>::R*)node.r.p); break; \
	case BINARY_CASE(opcode, IROp::T_INT, IROp::T_INT, IROp::E_VECTOR, IROp::E_VECTOR): \
		Map2VV< OP<TInteger>, TRACE_VECTOR_WIDTH >::eval(state, (int64_t*)node.a.p, (int64_t*)node.b.p, (OP<TInteger>::R*)node.r.p); break; \
	case BINARY_CASE(opcode, IROp::T_DOUBLE, IROp::T_DOUBLE, IROp::E_SCALAR, IROp::E_VECTOR): \
		Map2SV< OP<TDouble>, TRACE_VECTOR_WIDTH >::eval(state, node.a.d, node.b.p, node.r.p); break; \
	case BINARY_CASE(opcode, IROp::T_DOUBLE, IROp::T_DOUBLE, IROp::E_VECTOR, IROp::E_SCALAR): \
		Map2VS< OP<TDouble>, TRACE_VECTOR_WIDTH >::eval(state, node.a.p, node.b.d, node.r.p); break; \
	case BINARY_CASE(opcode, IROp::T_DOUBLE, IROp::T_DOUBLE, IROp::E_VECTOR, IROp::E_VECTOR): \
		Map2VV< OP<TDouble>, TRACE_VECTOR_WIDTH >::eval(state, node.a.p, node.b.p, node.r.p); break; \

#define UNARY_CASE(opcode, typea) \
	((IROpCode::opcode << 4) + (typea << 2) + 3)

#define UNARY_IMPL(opcode,nm,OP) \
	case UNARY_CASE(opcode, IROp::T_INT): \
		Map1< OP<TInteger>, TRACE_VECTOR_WIDTH >::eval(state, (int64_t*)node.a.p, (OP<TInteger>::R*)node.r.p); break; \
	case UNARY_CASE(opcode, IROp::T_DOUBLE): \
		Map1< OP<TDouble>, TRACE_VECTOR_WIDTH >::eval(state, node.a.p, node.r.p); break; 
			switch(node.op.op) {
				IR_BINARY(BINARY_IMPL)
				IR_UNARY(UNARY_IMPL)
				case (IROpCode::coerce << 4) + (IROp::T_DOUBLE << 3) + (IROp::T_INT << 2) + 3:
					Map1< CastOp<Integer, Double> , TRACE_VECTOR_WIDTH>::eval(state, (int64_t*)node.a.p, node.r.p);
				break;
				default:
					printf("%d (%s)(%d)\n",(int)node.op.op, IROpCode::toString(node.op.code), (int) node.op.a_typ);
					_error("Invalid op code short vector machine");
			}
			//if(i == 0) printf("n%d: %f\n", (int)j, node.r.p[0]);
			if(node.r_external) \
				node.r.p += TRACE_VECTOR_WIDTH; \
			if(node.op.a_enc == IROp::E_VECTOR && node.a_external) \
				node.a.p += TRACE_VECTOR_WIDTH; \
			if(node.op.b_enc == IROp::E_VECTOR && node.b_external) \
				node.b.p += TRACE_VECTOR_WIDTH; \
		}
	}
}

std::string Trace::toString(State & state) {
	std::ostringstream out;
	out << "recorded: \n";
	for(size_t j = 0; j < n_nodes; j++) {
		IRNode & node = nodes[j];
		out << "n" << j << ": " << node.toString() << "\n";
	}
	out << "outputs: \n";
	for(size_t i = 0; i < n_outputs; i++) {

		Output & o = outputs[i];
		switch(o.location_type) {
		case Trace::Output::E_REG:
			out << "r" << o.location; break;
		case Trace::Output::E_SLOT:
			out << "s" << o.location; break;
		case Trace::Output::E_VAR:
			out << "v" << o.location; break;
		}
		out << " = n" << o.ref << "\n";
	}
	return out.str();
}
