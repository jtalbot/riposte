#include "interpreter.h"
#include "vector.h"
#include "ops.h"
#include "sse.h"

#include <stdlib.h>

void Trace::Reset() {
	active = false;
	n_recorded_since_last_exec = 0;
	n_nodes = n_outputs = n_pending_nodes = n_pending_outputs = 0;
}

std::string Trace::toString(Thread & thread) {
	std::ostringstream out;
	for(size_t j = 0; j < n_nodes; j++) {
		IRNode & node = nodes[j];
		if(node.length >= 0)
			out << "n" << j << " : " << Type::toString(node.type) << "[" << node.length << "]" << " = " << IROpCode::toString(node.op) << "\t";
		else
			out << "n" << j << " : " << Type::toString(node.type) << "[n" << -node.length << "]" << " = " << IROpCode::toString(node.op) << "\t";
		switch(node.op) {
#define BINARY(op,...) case IROpCode::op: out << "n" << node.binary.a << "\tn" << node.binary.b; break;
#define UNARY(op,...) case IROpCode::op: out << "n" << node.unary.a; break;
		BINARY_ARITH_MAP_BYTECODES(BINARY)
		BINARY_LOGICAL_MAP_BYTECODES(BINARY)
		BINARY_ORDINAL_MAP_BYTECODES(BINARY)
		UNARY_ARITH_MAP_BYTECODES(UNARY)
		UNARY_LOGICAL_MAP_BYTECODES(UNARY)
		ARITH_FOLD_BYTECODES(UNARY)
		ARITH_SCAN_BYTECODES(UNARY)
		UNARY(cast)
		BINARY(seq)
		BINARY(filter)
		case IROpCode::gather: out << "n" << node.unary.a << "\t" << "$" << (void*)node.unary.data; break;
		case IROpCode::loadc: out << ( (node.type == Type::Integer) ? node.loadc.i : node.loadc.d); break;
		case IROpCode::loadv: out << "$" << node.loadv.src.p; break;
		case IROpCode::storec: /*fallthrough*/
		case IROpCode::storev: out << "n" << node.store.a; break;
		case IROpCode::nop: break;
		}
		out << "\n";
	}
	out << "outputs: \n";
	for(size_t i = 0; i < n_outputs; i++) {

		Output & o = outputs[i];
		switch(o.location.type) {
		case Trace::Location::REG:
			out << "r[" << o.location.reg.offset << "]"; break;
		case Trace::Location::VAR:
			out << thread.externStr(o.location.pointer.name); break;
		}
		out << " = n" << o.ref << "\n";
	}
	/*out << "output_values: \n";
	for(size_t i = 0; i < n_output_values; i++) {
		out << "o" << i << " : " << Type::toString(output_values[i].type) << "[" << output_values[i].length << "]\n";
	}*/
	return out.str();
}

#define REG(thread, i) (*(thread.base+i))

static const Value & get_location_value(Thread & thread, const Trace::Location & l) {
	switch(l.type) {
	case Trace::Location::REG:
		return l.reg.base[l.reg.offset];
	default:
	case Trace::Location::VAR:
		return Dictionary::get(l.pointer);
	}
}
static void set_location_value(Thread & thread, const Trace::Location & l, const Value & v) {
	switch(l.type) {
	case Trace::Location::REG:
		l.reg.base[l.reg.offset] = v;
		return;
	case Trace::Location::VAR:
		Dictionary::assign(l.pointer,v);
		return;
	}
}

static bool isOutputAlive(Thread& thread, Trace& trace, Trace::Output const& o) {
	if(thread.trace.LocationIsDead(o.location)) return false;
	Value const& v = get_location_value(thread, o.location);
	return v.isFuture() && (v.future.ref == o.ref);
}

void Trace::InitializeOutputs(Thread& thread) {
	IRef store[TRACE_MAX_NODES];
	bzero(store, sizeof(IRef)*TRACE_MAX_NODES);

	for(size_t i = 0; i < n_outputs; ) {
		Output & o = outputs[i];
		if(isOutputAlive(thread,*this,o)) {
			if(!store[o.ref]) {
				store[o.ref] = EmitStore(o.ref);
				n_nodes = n_pending_nodes;
			}
			o.ref = store[o.ref];
			i++;
		}
		else  {
			o = outputs[--n_outputs];
		}	
	}	
}

void Trace::WriteOutputs(Thread & thread) {
	if(thread.state.verbose) {
		for(size_t i = 0; i < n_outputs; i++) {
			Output& o = outputs[i];
			std::string v = thread.stringify(nodes[o.ref].store.dst);
			printf("n%lld = %s\n", o.ref, v.c_str());
		}
	}
	for(size_t i = 0; i < n_outputs; i++) {
		Output & o = outputs[i];
		set_location_value(thread,o.location,nodes[o.ref].store.dst);
	}
}

void Trace::SimplifyOps(Thread& thread) {
	for(IRef ref = 0; ref < n_nodes; ref++) {
		IRNode & node = nodes[ref];
		switch(node.op) {
			case IROpCode::gt: node.op = IROpCode::lt; std::swap(node.binary.a,node.binary.b); break;
			case IROpCode::ge: node.op = IROpCode::le; std::swap(node.binary.a,node.binary.b); break;
			case IROpCode::add: /* fallthrough */ 
			case IROpCode::mul:
			case IROpCode::land:
			case IROpCode::lor: 
					   if(nodes[node.binary.a].op == IROpCode::loadc) std::swap(node.binary.a,node.binary.b); break;
			default: /*pass*/ break;
		}
	}
}

void Trace::AlgebraicSimplification(Thread& thread) {
	for(IRef ref = 0; ref < n_nodes; ref++) {
		IRNode & node = nodes[ref];

		if(node.enc == IRNode::UNARY && nodes[node.unary.a].op == IROpCode::pos)
			node.unary.a = nodes[node.unary.a].unary.a;
		if(node.enc == IRNode::STORE && nodes[node.store.a].op == IROpCode::pos)
			node.store.a = nodes[node.store.a].unary.a;
		if(node.enc == IRNode::BINARY && nodes[node.binary.a].op == IROpCode::pos)
			node.binary.a = nodes[node.binary.a].unary.a;
		if(node.enc == IRNode::BINARY && nodes[node.binary.b].op == IROpCode::pos)
			node.binary.b = nodes[node.binary.b].unary.a;

		if(node.op == IROpCode::pow &&
			nodes[node.binary.b].op == IROpCode::loadc) {
			if(nodes[node.binary.b].loadc.d == 0) {
				// x^0 => 1
				node.op = IROpCode::loadc;
				node.enc = IRNode::LOADC;
				node.length = 1;
				node.loadc.d = 1;
			} else if(nodes[node.binary.b].loadc.d == 1) {
				// x^1 => x
				node.op = IROpCode::pos;
				node.enc = IRNode::UNARY;
				node.unary.a = node.binary.a;
			} else if(nodes[node.binary.b].loadc.d == 2) {
				// x^2 => x*x
				node.op = IROpCode::mul;
				node.binary.b = node.binary.a;
			} else if(nodes[node.binary.b].loadc.d == 0.5) {
				// x^0.5 => sqrt(x)
				node.op = IROpCode::sqrt;
				node.enc = IRNode::UNARY;
				node.unary.a = node.binary.a;
			} else if(nodes[node.binary.b].loadc.d == -1) {
				// x^-1 => 1/x
				node.op = IROpCode::div;
				nodes[node.binary.b].loadc.d = 1;
				std::swap(node.binary.a, node.binary.b);
			}
			// could also do x^-2 => 1/x * 1/x and x^-0.5 => sqrt(1/x) ?
		}

		if(	node.op == IROpCode::neg &&
				nodes[node.unary.a].op == IROpCode::neg) {
			node.op = IROpCode::pos;
			node.unary.a = nodes[node.unary.a].unary.a;
		}
		if(	node.op == IROpCode::lnot &&
				nodes[node.unary.a].op == IROpCode::lnot) {
			node.op = IROpCode::pos;
			node.unary.a = nodes[node.unary.a].unary.a;
		}
		if(	node.op == IROpCode::pos &&
				nodes[node.unary.a].op == IROpCode::pos) {
			node.unary.a = nodes[node.unary.a].unary.a;
		}
		if(node.op == IROpCode::add &&
				nodes[node.binary.b].op == IROpCode::loadc &&
				nodes[node.binary.a].op == IROpCode::add &&
				nodes[nodes[node.binary.a].binary.b].op == IROpCode::loadc) {
			if(node.isInteger())
				nodes[node.binary.b].loadc.i += nodes[nodes[node.binary.a].binary.b].loadc.i;
			else
				nodes[node.binary.b].loadc.d += nodes[nodes[node.binary.a].binary.b].loadc.d;
			node.binary.a = nodes[node.binary.a].binary.a;
		}
		if(node.op == IROpCode::mul &&
				nodes[node.binary.b].op == IROpCode::loadc &&
				nodes[node.binary.a].op == IROpCode::mul &&
				nodes[nodes[node.binary.a].binary.b].op == IROpCode::loadc) {
			if(node.isInteger())
				nodes[node.binary.b].loadc.i *= nodes[nodes[node.binary.a].binary.b].loadc.i;
			else
				nodes[node.binary.b].loadc.d *= nodes[nodes[node.binary.a].binary.b].loadc.d;
			node.binary.a = nodes[node.binary.a].binary.a;
		}

		if(node.op == IROpCode::add &&
				nodes[node.binary.b].op == IROpCode::loadc &&
				nodes[node.binary.b].loadc.i == 0) {
			node.op = IROpCode::pos;
			node.unary.a = node.binary.a;
		}

		if(node.op == IROpCode::mul &&
				nodes[node.binary.b].op == IROpCode::loadc &&
				((node.isDouble() && nodes[node.binary.b].loadc.d == 1) || 
				 (node.isInteger() && nodes[node.binary.b].loadc.i == 1))) {
			node.op = IROpCode::pos;
			node.unary.a = node.binary.a;
		}
		
		if(node.op == IROpCode::land &&
				nodes[node.binary.b].op == IROpCode::loadc &&
				Logical::isTrue(nodes[node.binary.b].loadc.l)) {
			node.op = IROpCode::pos;
			node.unary.a = node.binary.a;
		}

		if(node.op == IROpCode::land &&
				nodes[node.binary.b].op == IROpCode::loadc &&
				Logical::isFalse(nodes[node.binary.b].loadc.l)) {
			node.op = IROpCode::loadc;
			node.enc = IRNode::LOADC;
			node.length = 1;
			node.loadc.l = 0;
		}

		if(node.op == IROpCode::lor &&
				nodes[node.binary.b].op == IROpCode::loadc &&
				Logical::isFalse(nodes[node.binary.b].loadc.l)) {
			node.op = IROpCode::pos;
			node.unary.a = node.binary.a;
		}

		if(node.op == IROpCode::lor &&
				nodes[node.binary.b].op == IROpCode::loadc &&
				Logical::isTrue(nodes[node.binary.b].loadc.l)) {
			node.op = IROpCode::loadc;
			node.enc = IRNode::LOADC;
			node.length = 1;
			node.loadc.l = 1;
		}
	}
}

void Trace::DeadCodeElimination(Thread& thread) {
	// kill any defs whose use I haven't seen!
	for(IRef ref = 0; ref < n_nodes; ref++) {
		IRNode & node = nodes[ref];
		node.used = false;
	}
	for(IRef ref = n_nodes; ref > 0; ref--) {
		IRNode & node = nodes[ref-1];
		if(node.enc == IRNode::STORE) {
			nodes[node.store.a].used = true;
		} else if(node.used) {
			switch(node.enc) {
				case IRNode::BINARY: 
					nodes[node.binary.a].used = true;
					nodes[node.binary.b].used = true;
					break;
				case IRNode::UNARY:
					nodes[node.unary.a].used = true;
					break;
				case IRNode::LOADC: /*fallthrough*/
				case IRNode::LOADV: /*fallthrough*/
				case IRNode::SPECIAL: /*fallthrough*/
				case IRNode::STORE: /*fallthrough*/
				case IRNode::NOP: 
					/* nothing */
					break;
			}
		} else {
			node.op = IROpCode::nop;
			node.enc = IRNode::NOP;
		}
	}
}


// ref must be evaluated. Other stuff doesn't need to be executed unless it improves performance.
void Trace::Execute(Thread & thread, IRef ref) {
	Execute(thread);
}

// everything must be evaluated in the end...
void Trace::Execute(Thread & thread) {
	
	if(thread.state.verbose)
		printf("executing trace:\n%s\n",toString(thread).c_str());

	InitializeOutputs(thread);
	DeadCodeElimination(thread);	// avoid optimizing code that's dead anyway
	
	SimplifyOps(thread);
	AlgebraicSimplification(thread);
	
	// Initialize Output...
	for(IRef ref = 0; ref < n_nodes; ref++) {
		IRNode& node = nodes[ref];
		if(node.op == IROpCode::storev) {
			if(nodes[node.store.a].op == IROpCode::loadv) {
				node.store.dst = nodes[node.store.a].loadv.src;
				node.op = IROpCode::nop;
				node.enc = IRNode::NOP;
			} else {
				if(node.type == Type::Double) {
					node.store.dst = Double(node.length);
				} else if(node.type == Type::Integer) {
					node.store.dst = Integer(node.length);
				} else if(node.type == Type::Logical) {
					node.store.dst = Logical(node.length);
				} else {
					_error("Unknown type in initialize outputs");
				}
			}
		} else if(node.op == IROpCode::storec) {
			uint64_t n = std::max((int64_t)128, thread.state.nThreads*8);
			if(node.type == Type::Double) {
				node.store.dst = Double(n);
			} else if(node.type == Type::Integer) {
				node.store.dst = Integer(n);
			} else if(node.type == Type::Logical) {
				node.store.dst = Logical(n);
			} else {
				_error("Unknown type in initialize outputs");
			}
		}
	}

	DeadCodeElimination(thread);

	if(thread.state.verbose)
		printf("optimized:\n%s\n",toString(thread).c_str());

	
	JIT(thread);
	
	WriteOutputs(thread);
}
