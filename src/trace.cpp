#include "interpreter.h"
#include "vector.h"
#include "ops.h"
#include "sse.h"

#include <stdlib.h>

void Trace::Reset() {
	active = false;
	n_recorded_since_last_exec = 0;
	nodes.clear();
	outputs.clear();
}

std::string Trace::toString(Thread & thread) {
	std::ostringstream out;
	for(size_t j = 0; j < nodes.size(); j++) {
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
		case IROpCode::ifelse: out << "n" << node.ifelse.cond << "\t" << "n" << node.ifelse.yes << "\t" << "n" << node.ifelse.no; break;
		case IROpCode::gather: out << "n" << node.unary.a << "\t" << "$" << (void*)node.unary.data; break;
		case IROpCode::loadc: out << ( (node.type == Type::Integer) ? node.loadc.i : (node.type == Type::Logical) ? node.loadc.l : node.loadc.d); break;
		case IROpCode::loadv: out << "$" << node.loadv.src.p; break;
		case IROpCode::storec: /*fallthrough*/
		case IROpCode::storev: out << "n" << node.store.a; break;
		case IROpCode::nop: break;
		}
		out << "\n";
	}
	out << "outputs: \n";
	for(size_t i = 0; i < outputs.size(); i++) {

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

void coerce_scalar(IRNode & n, Type::Enum to) {
	switch(n.type) {
	case Type::Integer: switch(to) {
		case Type::Double: n.loadc.d = n.loadc.i; break;
		case Type::Logical: n.loadc.l = n.loadc.i; break;
		default: _error("unknown cast"); break;
	} break;
	case Type::Double: switch(to) {
		case Type::Integer: n.loadc.i = (int64_t) n.loadc.d; break;
		case Type::Logical: n.loadc.l = (char) n.loadc.d; break;
		default: _error("unknown cast"); break;
	} break;
	case Type::Logical: switch(to) {
		case Type::Double: n.loadc.d = n.loadc.l; break;
		case Type::Integer: n.loadc.i = n.loadc.l; break;
		default: _error("unknown cast"); break;
	} break;
	default: _error("unknown cast"); break;
	}
	n.type = to;
}

IRef Trace::EmitCoerce(IRef a, Type::Enum dst_type) {
	IRNode& n = nodes[a];
	if(dst_type == n.type) {
		return a;
	} else if(n.op == IROpCode::loadc) {
		coerce_scalar(n, dst_type);
		return a;
	} else {
		return EmitUnary(IROpCode::cast,dst_type,a,0);
	} 
}
IRef Trace::EmitBinary(IROpCode::Enum op, Type::Enum type, IRef a, IRef b) {
	IRNode n;
	n.enc = IRNode::BINARY;
	n.op = op;
	n.type = type;
	n.length = nodes[a].length == 0 || nodes[b].length == 0 ? 0 
		: nodes[a].length < 0 ? nodes[a].length
		: nodes[b].length < 0 ? nodes[b].length
		: std::max(nodes[a].length, nodes[b].length);
	n.binary.a = a;
	n.binary.b = b;
	nodes.push_back(n);
	return nodes.size()-1;
}
IRef Trace::EmitSpecial(IROpCode::Enum op, Type::Enum type, int64_t length, int64_t a, int64_t b) {
	IRNode n;
	n.enc = IRNode::SPECIAL;
	n.op = op;
	n.type = type;
	n.length = length;
	n.special.a = a;
	n.special.b = b;
	nodes.push_back(n);
	return nodes.size()-1;
}
IRef Trace::EmitUnary(IROpCode::Enum op, Type::Enum type, IRef a, int64_t data) {
	IRNode n;
	n.enc = IRNode::UNARY;
	n.op = op;
	n.type = type;
	n.length = nodes[a].length;
	n.unary.a = a;
	n.unary.data = data;
	nodes.push_back(n);
	return nodes.size()-1;
}
IRef Trace::EmitFold(IROpCode::Enum op, Type::Enum type, IRef a) {
	IRNode n;
	n.enc = IRNode::FOLD;
	n.op = op;
	n.type = type;
	n.length = 1;
	n.fold.mask = nodes[a].length;
	n.fold.a = a;
	nodes.push_back(n);
	return nodes.size()-1;
}
IRef Trace::EmitFilter(IROpCode::Enum op, IRef a, IRef b) {
	IRNode n;
	n.enc = IRNode::BINARY;
	n.op = op;
	n.type = nodes[a].type;
	n.length = -b;
	n.binary.a = a;
	n.binary.b = b;
	nodes.push_back(n);
	return nodes.size()-1;
}
IRef Trace::EmitBlend(IRef cond, IRef yes, IRef no) {
	IRNode n;
	n.enc = IRNode::IFELSE;
	n.op = IROpCode::ifelse;
	assert(nodes[yes].type == nodes[no].type);
	n.type = nodes[yes].type;
	n.length = nodes[cond].length;
	n.ifelse.cond = cond;
	n.ifelse.yes = yes;
	n.ifelse.no = no;
	nodes.push_back(n);
	return nodes.size()-1;
}
IRef Trace::EmitLoadC(Type::Enum type, int64_t length, int64_t c) {
	IRNode n;
	n.enc = IRNode::LOADC;
	n.op = IROpCode::loadc;
	n.type = type;
	n.length = length;
	n.loadc.i = c;
	nodes.push_back(n);
	return nodes.size()-1;
}
IRef Trace::EmitLoadV(Value const& v) {
	IRNode n;
	n.enc = IRNode::LOADV;
	n.op = IROpCode::loadv;
	n.type = v.type;
	n.length = v.length;
	n.loadv.src = v;
	nodes.push_back(n);
	return nodes.size()-1;
}
IRef Trace::EmitStore(IRef a) {
	IRNode n;
	n.enc = IRNode::STORE;
	n.op = nodes[a].enc == IRNode::FOLD ? IROpCode::storec : IROpCode::storev;
	n.type = nodes[a].type;
	n.length = nodes[a].length;
	n.store.a = a;
	nodes.push_back(n);
	return nodes.size()-1;
}
void Trace::RegOutput(IRef ref, Value * base, int64_t id) {
	Output out;
	out.ref = ref;
	out.location.type = Location::REG;
	out.location.reg.base = base;
	out.location.reg.offset = id;
	outputs.push_back(out);
}
void Trace::VarOutput(IRef ref, const Environment::Pointer & p) {
	Output out;
	out.ref = ref;
	out.location.type = Trace::Location::VAR;
	out.location.pointer = p;
	outputs.push_back(out);
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
	IRef* store = new (PointerFreeGC) IRef[nodes.size()];
	bzero(store, sizeof(IRef)*nodes.size());

	for(size_t i = 0; i < outputs.size(); ) {
		Output & o = outputs[i];
		if(isOutputAlive(thread,*this,o)) {
			if(!store[o.ref]) {
				store[o.ref] = EmitStore(o.ref);
			}
			o.ref = store[o.ref];
			i++;
		}
		else  {
			o = outputs.back();
			outputs.pop_back();
		}	
	}	
}

void Trace::WriteOutputs(Thread & thread) {
	if(thread.state.verbose) {
		for(size_t i = 0; i < outputs.size(); i++) {
			Output& o = outputs[i];
			std::string v = thread.stringify(nodes[o.ref].store.dst);
			printf("n%lld = %s\n", o.ref, v.c_str());
		}
	}
	for(size_t i = 0; i < outputs.size(); i++) {
		Output & o = outputs[i];
		set_location_value(thread,o.location,nodes[o.ref].store.dst);
	}
}

void Trace::SimplifyOps(Thread& thread) {
	for(IRef ref = 0; ref < nodes.size(); ref++) {
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
	for(IRef ref = 0; ref < nodes.size(); ref++) {
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
			node.loadc.l = Logical::FalseElement;
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
			node.loadc.l = Logical::TrueElement;
		}
	}
}

void Trace::DeadCodeElimination(Thread& thread) {
	// kill any defs whose use I haven't seen!
	for(IRef ref = 0; ref < nodes.size(); ref++) {
		IRNode & node = nodes[ref];
		node.used = false;
	}
	for(IRef ref = nodes.size(); ref > 0; ref--) {
		IRNode & node = nodes[ref-1];
		if(node.enc == IRNode::STORE) {
			nodes[node.store.a].used = true;
			if(node.length < 0) nodes[-node.length].used = true;
		} else if(node.used) {
			switch(node.enc) {
				case IRNode::BINARY: 
					nodes[node.binary.a].used = true;
					nodes[node.binary.b].used = true;
					break;
				case IRNode::UNARY:
				case IRNode::FOLD:
					nodes[node.unary.a].used = true;
					break;
				case IRNode::IFELSE: 
					nodes[node.ifelse.cond].used = true;
					nodes[node.ifelse.yes].used = true;
					nodes[node.ifelse.no].used = true;
					break;
				case IRNode::LOADC: /*fallthrough*/
				case IRNode::LOADV: /*fallthrough*/
				case IRNode::SPECIAL: /*fallthrough*/
				case IRNode::STORE: /*fallthrough*/
				case IRNode::NOP: 
					/* nothing */
					break;
			}
			if(node.length < 0) nodes[-node.length].used = true;
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
	for(IRef ref = 0; ref < nodes.size(); ref++) {
		IRNode& node = nodes[ref];
		if(node.op == IROpCode::storev) {
			if(nodes[node.store.a].op == IROpCode::loadv) {
				node.store.dst = nodes[node.store.a].loadv.src;
				node.op = IROpCode::nop;
				node.enc = IRNode::NOP;
			} else {
				// compute length
				IRNode* n = &node;
				if(n->length < 0) {
					n = &nodes[-n->length];
				}
				int64_t length = n->length;
				if(node.type == Type::Double) {
					node.store.dst = Double(length);
				} else if(node.type == Type::Integer) {
					node.store.dst = Integer(length);
				} else if(node.type == Type::Logical) {
					node.store.dst = Logical(length);
				} else {
					_error("Unknown type in initialize outputs");
				}
				if(node.length < 0)
					node.store.dst.length = 0;
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

	Reset();
}
