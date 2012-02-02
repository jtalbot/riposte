#include "interpreter.h"
#include "vector.h"
#include "ops.h"
#include "sse.h"

#include <stdlib.h>

void Trace::Reset() {
	n_recorded_since_last_exec = 0;
	nodes.clear();
	outputs.clear();
}

std::string Trace::toString(Thread & thread) {
	std::ostringstream out;
	for(size_t j = 0; j < nodes.size(); j++) {
		IRNode & node = nodes[j];
		out << "n" << j << " : " << Type::toString(node.type) << "[" << node.shape.length;
		if(node.shape.filter > 0) out << "[n" << node.shape.filter << "]";
		if(node.shape.split > 0) out << "/n" << node.shape.split << "(" << node.shape.levels << ")";
		out << "]" << "\t";
		out << " = " << IROpCode::toString(node.op) << "\t\t"; 
		switch(node.op) {
#define TRINARY(op,...) case IROpCode::op: out << "n" << node.trinary.a << "\tn" << node.trinary.b << "\tn" << node.trinary.c; break;
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
		BINARY(split)
		TRINARY(ifelse)
		case IROpCode::gather: out << "n" << node.unary.a << "\t" << "$" << (void*)node.unary.data; break;
		case IROpCode::constant: out << ( (node.type == Type::Integer) ? node.constant.i : (node.type == Type::Logical) ? node.constant.l : node.constant.d); break;
		case IROpCode::load: out << "$" << node.out.p; break;
		case IROpCode::nop: break;
		}
		if(node.liveOut) {
			out << "\t -> ";
			for(size_t i = 0; i < outputs.size(); i++) {
				Output & o = outputs[i];
				if(o.ref == j) {
					switch(o.location.type) {
					case Trace::Location::REG:
						out << "r[" << o.location.reg.offset << "] "; break;
					case Trace::Location::VAR:
						out << thread.externStr(o.location.pointer.name) << " "; break;
					}
				}
			}
		}
		out << "\n";
	}
	return out.str();
}

void coerce_scalar(IRNode & n, Type::Enum to) {
	switch(n.type) {
	case Type::Integer: switch(to) {
		case Type::Double: n.constant.d = n.constant.i; break;
		case Type::Logical: n.constant.l = n.constant.i; break;
		default: _error("unknown cast"); break;
	} break;
	case Type::Double: switch(to) {
		case Type::Integer: n.constant.i = (int64_t) n.constant.d; break;
		case Type::Logical: n.constant.l = (char) n.constant.d; break;
		default: _error("unknown cast"); break;
	} break;
	case Type::Logical: switch(to) {
		case Type::Double: n.constant.d = n.constant.l; break;
		case Type::Integer: n.constant.i = n.constant.l; break;
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
	} else if(n.op == IROpCode::constant) {
		coerce_scalar(n, dst_type);
		return a;
	} else {
		return EmitUnary(IROpCode::cast,dst_type,a,0);
	} 
}
IRef Trace::EmitUnary(IROpCode::Enum op, Type::Enum type, IRef a, int64_t data) {
	IRNode n;
	n.enc = IRNode::UNARY;
	n.op = op;
	n.type = type;
	n.shape = nodes[a].shape;
	n.unary.a = a;
	n.unary.data = data;
	nodes.push_back(n);
	return nodes.size()-1;
}
IRef Trace::EmitBinary(IROpCode::Enum op, Type::Enum type, IRef a, IRef b, int64_t data) {
	IRNode n;
	n.enc = IRNode::BINARY;
	n.op = op;
	n.type = type;
	if(nodes[a].enc == IRNode::CONSTANT)
		n.shape = nodes[b].shape;
	else if(nodes[b].enc == IRNode::CONSTANT)
		n.shape = nodes[a].shape;
	else {
		assert(nodes[a].shape == nodes[b].shape);
		n.shape = nodes[a].shape;
	}
	n.binary.a = a;
	n.binary.b = b;
	n.binary.data = data;
	nodes.push_back(n);
	return nodes.size()-1;
}
IRef Trace::EmitTrinary(IROpCode::Enum op, Type::Enum type, IRef a, IRef b, IRef c) {
	IRNode n;
	n.enc = IRNode::TRINARY;
	n.op = op;
	n.type = type;
	assert(nodes[a].shape == nodes[b].shape && nodes[b].shape == nodes[c].shape);
	n.shape = nodes[a].shape;
	n.trinary.a = a;
	n.trinary.b = b;
	n.trinary.c = c;
	nodes.push_back(n);
	return nodes.size()-1;
}
IRef Trace::EmitFold(IROpCode::Enum op, IRef a) {
	IRNode n;
	n.enc = IRNode::FOLD;
	n.op = op;
	n.type = nodes[a].type;
	n.shape = (IRNode::Shape) { nodes[a].shape.levels, 0, 1, 0 };
	n.unary.a = a;
	nodes.push_back(n);
	return nodes.size()-1;
}

IRef Trace::EmitFilter(IRef a, IRef b) {
	// and the filters together if necessary
	if(nodes[a].shape.filter > 0) {
		b = EmitBinary(IROpCode::land, Type::Logical, nodes[a].shape.filter, b, 0);
	}
	IRNode n;
	n.enc = IRNode::BINARY;
	n.op = IROpCode::filter;
	n.type = nodes[a].type;
	n.shape = nodes[a].shape;
	n.shape.filter = b;
	n.binary.a = a;
	n.binary.b = b;
	nodes.push_back(n);
	return nodes.size()-1;
}
IRef Trace::EmitSplit(IRef x, IRef f, int64_t levels) {
	// subsplit if necessary...
	if(nodes[x].shape.split > 0) {
		levels *= nodes[x].shape.levels;
		IRef e = EmitBinary(IROpCode::mul, Type::Integer, f, EmitConstant(Type::Integer, levels), 0);
		f = EmitBinary(IROpCode::add, Type::Integer, e, f, 0);
	}
	IRNode n;
	n.enc = IRNode::BINARY;
	n.op = IROpCode::split;
	n.type = nodes[x].type;
	n.shape = nodes[x].shape;
	n.shape.split = f;
	n.shape.levels = levels;
	n.binary.a = x;
	n.binary.b = f;
	nodes.push_back(n);
	return nodes.size()-1;
}

IRef Trace::EmitSpecial(IROpCode::Enum op, Type::Enum type, int64_t length, int64_t a, int64_t b) {
	IRNode n;
	n.enc = IRNode::SPECIAL;
	n.op = op;
	n.type = type;
	n.shape = (IRNode::Shape) { length, 0, 1, 0 };
	n.special.a = a;
	n.special.b = b;
	nodes.push_back(n);
	return nodes.size()-1;
}
IRef Trace::EmitConstant(Type::Enum type, int64_t c) {
	IRNode n;
	n.enc = IRNode::CONSTANT;
	n.op = IROpCode::constant;
	n.type = type;
	n.shape = (IRNode::Shape) { 1, 0, 1, 0 };
	n.constant.i = c;
	nodes.push_back(n);
	return nodes.size()-1;
}
IRef Trace::EmitLoad(Value const& v) {
	IRNode n;
	n.enc = IRNode::LOAD;
	n.op = IROpCode::load;
	n.type = v.type;
	n.shape = (IRNode::Shape) { v.length, 0, 1, 0 };
	n.out = v;
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

void Trace::MarkLiveOutputs(Thread& thread) {
	for(size_t i = 0; i < nodes.size(); i++) {
		nodes[i].liveOut = false;
	}
	for(size_t i = 0; i < outputs.size(); i++) {
		nodes[outputs[i].ref].liveOut = true;
	}
}

void Trace::DiscardDeadOutputs(Thread& thread) {
	for(size_t i = 0; i < outputs.size(); ) {
		Output & o = outputs[i];
		if(isOutputAlive(thread,*this,o)) {
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
		for(size_t i = 0; i < nodes.size(); i++) {
			if(nodes[i].liveOut) {
				std::string v = thread.stringify(nodes[i].out);
				printf("n%d = %s\n", i, v.c_str());
			}
		}
	}
	for(size_t i = 0; i < outputs.size(); i++) {
		Output & o = outputs[i];
		set_location_value(thread,o.location,nodes[o.ref].out);
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
					   if(nodes[node.binary.a].op == IROpCode::constant) std::swap(node.binary.a,node.binary.b); break;
			default: /*pass*/ break;
		}
	}
}

void Trace::AlgebraicSimplification(Thread& thread) {
	for(IRef ref = 0; ref < nodes.size(); ref++) {
		IRNode & node = nodes[ref];

		if(node.enc == IRNode::UNARY && nodes[node.unary.a].op == IROpCode::pos)
			node.unary.a = nodes[node.unary.a].unary.a;
		if(node.enc == IRNode::FOLD && nodes[node.fold.a].op == IROpCode::pos)
			node.fold.a = nodes[node.fold.a].unary.a;
		if(node.enc == IRNode::BINARY && nodes[node.binary.a].op == IROpCode::pos)
			node.binary.a = nodes[node.binary.a].unary.a;
		if(node.enc == IRNode::BINARY && nodes[node.binary.b].op == IROpCode::pos)
			node.binary.b = nodes[node.binary.b].unary.a;

		if(node.op == IROpCode::pow &&
			nodes[node.binary.b].op == IROpCode::constant) {
			if(nodes[node.binary.b].constant.d == 0) {
				// x^0 => 1
				node.op = IROpCode::constant;
				node.enc = IRNode::CONSTANT;
				node.shape = (IRNode::Shape) { 1, 0, 1, 0 };
				node.constant.d = 1;
			} else if(nodes[node.binary.b].constant.d == 1) {
				// x^1 => x
				node.op = IROpCode::pos;
				node.enc = IRNode::UNARY;
				node.unary.a = node.binary.a;
			} else if(nodes[node.binary.b].constant.d == 2) {
				// x^2 => x*x
				node.op = IROpCode::mul;
				node.binary.b = node.binary.a;
			} else if(nodes[node.binary.b].constant.d == 0.5) {
				// x^0.5 => sqrt(x)
				node.op = IROpCode::sqrt;
				node.enc = IRNode::UNARY;
				node.unary.a = node.binary.a;
			} else if(nodes[node.binary.b].constant.d == -1) {
				// x^-1 => 1/x
				node.op = IROpCode::div;
				nodes[node.binary.b].constant.d = 1;
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
				nodes[node.binary.b].op == IROpCode::constant &&
				nodes[node.binary.a].op == IROpCode::add &&
				nodes[nodes[node.binary.a].binary.b].op == IROpCode::constant) {
			if(node.isInteger())
				nodes[node.binary.b].constant.i += nodes[nodes[node.binary.a].binary.b].constant.i;
			else
				nodes[node.binary.b].constant.d += nodes[nodes[node.binary.a].binary.b].constant.d;
			node.binary.a = nodes[node.binary.a].binary.a;
		}
		if(node.op == IROpCode::mul &&
				nodes[node.binary.b].op == IROpCode::constant &&
				nodes[node.binary.a].op == IROpCode::mul &&
				nodes[nodes[node.binary.a].binary.b].op == IROpCode::constant) {
			if(node.isInteger())
				nodes[node.binary.b].constant.i *= nodes[nodes[node.binary.a].binary.b].constant.i;
			else
				nodes[node.binary.b].constant.d *= nodes[nodes[node.binary.a].binary.b].constant.d;
			node.binary.a = nodes[node.binary.a].binary.a;
		}

		if(node.op == IROpCode::add &&
				nodes[node.binary.b].op == IROpCode::constant &&
				nodes[node.binary.b].constant.i == 0) {
			node.op = IROpCode::pos;
			node.unary.a = node.binary.a;
		}

		if(node.op == IROpCode::mul &&
				nodes[node.binary.b].op == IROpCode::constant &&
				((node.isDouble() && nodes[node.binary.b].constant.d == 1) || 
				 (node.isInteger() && nodes[node.binary.b].constant.i == 1))) {
			node.op = IROpCode::pos;
			node.unary.a = node.binary.a;
		}
		
		if(node.op == IROpCode::land &&
				nodes[node.binary.b].op == IROpCode::constant &&
				Logical::isTrue(nodes[node.binary.b].constant.l)) {
			node.op = IROpCode::pos;
			node.unary.a = node.binary.a;
		}

		if(node.op == IROpCode::land &&
				nodes[node.binary.b].op == IROpCode::constant &&
				Logical::isFalse(nodes[node.binary.b].constant.l)) {
			node.op = IROpCode::constant;
			node.enc = IRNode::CONSTANT;
			node.shape = (IRNode::Shape) { 1, 0, 1, 0 };
			node.constant.l = Logical::FalseElement;
		}

		if(node.op == IROpCode::lor &&
				nodes[node.binary.b].op == IROpCode::constant &&
				Logical::isFalse(nodes[node.binary.b].constant.l)) {
			node.op = IROpCode::pos;
			node.unary.a = node.binary.a;
		}

		if(node.op == IROpCode::lor &&
				nodes[node.binary.b].op == IROpCode::constant &&
				Logical::isTrue(nodes[node.binary.b].constant.l)) {
			node.op = IROpCode::constant;
			node.enc = IRNode::CONSTANT;
			node.shape = (IRNode::Shape) { 1, 0, 1, 0 };
			node.constant.l = Logical::TrueElement;
		}
	}
}

void Trace::DeadCodeElimination(Thread& thread) {
	for(size_t i = 0; i < nodes.size(); i++) {
		nodes[i].live = false | nodes[i].liveOut;
	}
	for(IRef ref = nodes.size(); ref > 0; ref--) {
		IRNode & node = nodes[ref-1];
		if(node.live) {
			switch(node.enc) {
				case IRNode::TRINARY: 
					nodes[node.trinary.a].live = true;
					nodes[node.trinary.b].live = true;
					nodes[node.trinary.c].live = true;
					break;
				case IRNode::BINARY: 
					nodes[node.binary.a].live = true;
					nodes[node.binary.b].live = true;
					break;
				case IRNode::UNARY:
					nodes[node.unary.a].live = true;
					break;
				case IRNode::FOLD: 
					nodes[node.fold.a].live = true;
					break;
				case IRNode::CONSTANT: /*fallthrough*/
				case IRNode::LOAD: /*fallthrough*/
				case IRNode::SPECIAL: /*fallthrough*/
				case IRNode::NOP: 
					/* nothing */
					break;
			}
			// mark used shape nodes
			if(node.shape.filter > 0) 
				nodes[node.shape.filter].live = true;
			if(node.shape.split > 0) 
				nodes[node.shape.split].live = true;
		} else {
			node.op = IROpCode::nop;
			node.enc = IRNode::NOP;
			node.shape = (IRNode::Shape) { 0, 0, 0, 0 };
		}
	}
}


// ref must be evaluated. Other stuff doesn't need to be executed unless it improves performance.
void Trace::Execute(Thread & thread, IRef ref) {
	Execute(thread);
}

// everything must be evaluated in the end...
void Trace::Execute(Thread & thread) {
	
	DiscardDeadOutputs(thread);
	MarkLiveOutputs(thread);

	if(thread.state.verbose)
		printf("executing trace:\n%s\n",toString(thread).c_str());

	DeadCodeElimination(thread);	// avoid optimizing code that's dead anyway
	
	SimplifyOps(thread);
	AlgebraicSimplification(thread);

	// move outputs up...
	for(size_t i = 0; i < outputs.size(); i++) {
		IRef& r = outputs[i].ref;
		while(nodes[r].op == IROpCode::pos) {
			nodes[r].liveOut = false;
			r = nodes[r].unary.a;
		}
		nodes[r].liveOut = true;
	}
	
	DeadCodeElimination(thread);

	// Initialize Outputs...
	for(IRef ref = 0; ref < nodes.size(); ref++) {
		IRNode& node = nodes[ref];
		if((node.liveOut && node.enc != IRNode::LOAD) || 
		   (node.live && node.enc == IRNode::FOLD)) {
			int64_t length = node.shape.length;
			if(node.enc == IRNode::FOLD) {
				length = std::max(nodes[node.fold.a].shape.levels*2, 8LL) * thread.state.nThreads; 
				// * 8 fills cache line (assuming aggregates are all stored in 8-byte fields)
			} else {
				if(node.shape.levels != 1)
					_error("Group by without aggregate not yet supported");
			}
			if(node.type == Type::Double) {
				node.out = Double(length);
			} else if(node.type == Type::Integer) {
				node.out = Integer(length);
			} else if(node.type == Type::Logical) {
				node.out = Logical(length);
			} else {
				_error("Unknown type in initialize outputs");
			}
		}
	}

	if(thread.state.verbose)
		printf("optimized:\n%s\n",toString(thread).c_str());

	
	JIT(thread);
	
	WriteOutputs(thread);

	Reset();
}
