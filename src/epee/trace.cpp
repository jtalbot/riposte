#include "../interpreter.h"
#include "../vector.h"
#include "../ops.h"
#include "../sse.h"
//#include <unordered_set>
#include <stdlib.h>

void Trace::Reset() {
	n_recorded_since_last_exec = 0;
	nodes.clear();
	outputs.clear();
	liveEnvironments.clear();
}

Trace::Trace() { 
	Reset(); 
	code_buffer = NULL;
}

std::string shape2string(IRNode::Shape const& shape) {
	std::ostringstream out;
	out << "[" << shape.length;
	if(shape.filter > 0) out << "[n" << shape.filter << "]";
	if(shape.split > 0) out << "/n" << shape.split << "(" << shape.levels << ")";
	out << "]";
	return out.str();
}

std::string Trace::toString(Thread & thread) {
	std::ostringstream out;
	for(size_t j = 0; j < nodes.size(); j++) {
		IRNode & node = nodes[j];
		out << "n" << j << " : " << Type::toString(node.type) << "\t";
		out << shape2string(node.shape) << "\t=>\t" << shape2string(node.outShape);
		out << "\t = " << IROpCode::toString(node.op) << "\t\t"; 
		switch(node.op) {
#define TRINARY(op,...) case IROpCode::op: out << "n" << node.trinary.a << "\tn" << node.trinary.b << "\tn" << node.trinary.c; break;
#define BINARY(op,...) case IROpCode::op: out << "n" << node.binary.a << "\tn" << node.binary.b; break;
#define UNARY(op,...) case IROpCode::op: out << "n" << node.unary.a; break;
#define NULLARY(op,...) case IROpCode::op: break;
		UNARY_FOLD_SCAN_BYTECODES(UNARY)
		BINARY_BYTECODES(BINARY)
		NULLARY(length)
		NULLARY(random)
		BINARY(mean)
		TRINARY(cm2)
		UNARY(cast)
		UNARY(filter)
		BINARY(split)
		TRINARY(ifelse)
		case IROpCode::seq:
		case IROpCode::rep:
			if(node.type == Type::Double)
				out << node.sequence.da << "\t" << node.sequence.db;
			else
				out << node.sequence.ia << "\t" << node.sequence.ib;
		break;
		case IROpCode::constant: out << ( (node.type == Type::Integer) ? node.constant.i : (node.type == Type::Logical) ? node.constant.l : node.constant.d); break;
		case IROpCode::addc: 
		case IROpCode::mulc: 
			out << "n" << node.unary.a << "\t" << ( (node.type == Type::Integer) ? node.constant.i : (node.type == Type::Logical) ? node.constant.l : node.constant.d); break;
		case IROpCode::load: out << "$" << node.in.p << "[" << node.constant.i << "]\t"; break;
		case IROpCode::gather: out << "$" << node.in.p << "\tn" << node.unary.a; break;
		case IROpCode::sload: out << "$" << node.in.p; break;
		case IROpCode::sstore: out << "n" << node.binary.a << "[" << node.binary.data << "]\tn" << node.binary.b; break;
		case IROpCode::nop: break;
		}
		if(node.liveOut) {
			out << "\t -> ";
			for(size_t i = 0; i < outputs.size(); i++) {
				Output & o = outputs[i];
				if(o.ref == (int64_t)j) {
					switch(o.type) {
					case Trace::Output::REG:
						out << "r[" << o.reg << "] "; break;
					case Trace::Output::MEMORY:
						out << thread.externStr(o.pointer.name) << " "; break;
					}
				}
			}
		}
		out << "\n";
	}
	return out.str();
}

IRef Trace::EmitCoerce(IRef a, Type::Enum dst_type) {
	IRNode& n = nodes[a];
	if(dst_type == n.type) {
		return a;
	} else {
		return EmitUnary(IROpCode::cast,dst_type,a,0);
	} 
}
IRef Trace::EmitUnary(IROpCode::Enum op, Type::Enum type, IRef a, int64_t data) {
	IRNode n;
	n.op = op;
	n.type = type;
	n.shape = nodes[a].outShape;
	n.unary.a = a;
	n.unary.data = data;
	if(	(op == IROpCode::sum || op == IROpCode::prod || 
		 op == IROpCode::min || op == IROpCode::max ||
		 op == IROpCode::any || op == IROpCode::all ||
		 op == IROpCode::length)) {
		n.group = IRNode::FOLD;
		n.arity = op == IROpCode::length ? IRNode::NULLARY : IRNode::UNARY;
		n.outShape = (IRNode::Shape) { nodes[a].outShape.levels, -1, 1, -1, true };
	} else if(op == IROpCode::mean) {
		union {
			double d;
			int64_t i;
		};
		d = 1.0;
		n.binary.b = EmitBinary(IROpCode::div, Type::Double, EmitConstant(Type::Double, 1, i), EmitUnary(IROpCode::length, Type::Double, a, 0), 0);
		n.group = IRNode::FOLD;
		n.arity = IRNode::BINARY;
		n.outShape = (IRNode::Shape) { nodes[a].outShape.levels, -1, 1, -1, true };
	} else {
		n.group = IRNode::MAP;
		n.arity = IRNode::UNARY;
		n.outShape = n.shape;
	}
	nodes.push_back(n);
	return nodes.size()-1;
}
IRef Trace::EmitBinary(IROpCode::Enum op, Type::Enum type, IRef a, IRef b, int64_t data) {
	IRNode n;
	n.op = op;
	n.type = type;
	n.binary.a = a;
	n.binary.b = b;
	n.binary.data = data;
	// TODO: this should really check to see if 
	// the operands are the same length or are constants that we can insert
	// a rep and gather for. This length == 1 check will break on sum(a) + 1 which
	// we can't fuse.
	if(nodes[a].shape.length == 1)
		n.shape = nodes[b].shape;
	else if(nodes[b].shape.length == 1)
		n.shape = nodes[a].shape;
	else {
		//assert(nodes[a].outShape == nodes[b].outShape);
		n.shape = nodes[a].outShape;
	}
	if(op == IROpCode::cm2) {
		n.binary.a = EmitUnary(IROpCode::mean, Type::Double, a, 0); 
		n.binary.b = EmitUnary(IROpCode::mean, Type::Double, b, 0);
		union {
			double d;
			int64_t i;
		};
		d = 1.0;
		n.trinary.c = EmitBinary(IROpCode::div, Type::Double, EmitConstant(Type::Double, 1, i), EmitUnary(IROpCode::length, Type::Double, a, 0), 0);
		n.arity = IRNode::TRINARY;
		n.group = IRNode::FOLD;
		n.outShape = (IRNode::Shape) { nodes[a].outShape.levels, -1, 1, -1, true };
	} else {
		n.arity = IRNode::BINARY;
		n.group = IRNode::MAP;
		n.outShape = n.shape;
	}
	nodes.push_back(n);
	return nodes.size()-1;
}
IRef Trace::EmitTrinary(IROpCode::Enum op, Type::Enum type, IRef a, IRef b, IRef c) {
	IRNode n;
	n.arity = IRNode::TRINARY;
	n.group = IRNode::MAP;
	n.op = op;
	n.type = type;
	// TODO: get rid of this assert and replace with recycle rule
	assert(nodes[a].outShape == nodes[b].outShape && nodes[b].outShape == nodes[c].outShape);
	n.shape = nodes[a].outShape;
	n.outShape = n.shape;
	n.trinary.a = a;
	n.trinary.b = b;
	n.trinary.c = c;
	nodes.push_back(n);
	return nodes.size()-1;
}

IRef Trace::EmitFilter(IRef a, IRef b) {
	// TODO: get rid of this assert and replace with recycle rule
	assert(nodes[a].outShape == nodes[b].outShape);
	IRNode n;
	n.arity = IRNode::UNARY;
	n.group = IRNode::FILTER;
	n.op = IROpCode::filter;
	n.type = nodes[b].type;
	n.shape = nodes[b].outShape;
	n.outShape = n.shape;
	n.outShape.filter = nodes.size();
	n.unary.a = b;
	nodes.push_back(n);
	IRef f = nodes.size()-1;
	IRef r = EmitUnary(IROpCode::pos, nodes[a].type, a, 0);
	nodes[r].shape.filter = f;
	nodes[r].outShape.filter = f;
	return r;
}

IRef Trace::EmitSplit(IRef x, IRef f, int64_t levels) {
	IRNode n;
	n.arity = IRNode::UNARY;
	n.group = IRNode::SPLIT;
	n.op = IROpCode::split;
	n.type = nodes[f].type;
	n.shape = nodes[f].outShape;
	n.outShape = n.shape;
	n.outShape.split = nodes.size();
	n.outShape.levels = levels;
	n.unary.a = f;
	nodes.push_back(n);
	IRef s = nodes.size()-1;
	IRef r = EmitUnary(IROpCode::pos, nodes[x].type, x, 0);
	nodes[r].shape.split = s;
	nodes[r].shape.levels = levels;
	nodes[r].outShape.split = s;
	nodes[r].outShape.levels = levels;
	return nodes.size()-1;
}

IRef Trace::EmitGenerator(IROpCode::Enum op, Type::Enum type, int64_t length, int64_t a, int64_t b) {
	IRNode n;
	n.arity = IRNode::NULLARY;
	n.group = IRNode::GENERATOR;
	n.op = op;
	n.type = type;
	n.shape = (IRNode::Shape) { length, -1, 1, -1, false };
	n.outShape = n.shape;
	n.sequence.ia = a;
	n.sequence.ib = b;
	nodes.push_back(n);
	return nodes.size()-1;
}

IRef Trace::EmitRandom(int64_t length) {
	return EmitGenerator(IROpCode::random, Type::Double, length, 0, 0);
}

IRef Trace::EmitRepeat(int64_t length, int64_t a, int64_t b) {
	return EmitGenerator(IROpCode::rep, Type::Integer, length, a, b);
}

IRef Trace::EmitSequence(int64_t length, int64_t a, int64_t b) {
	return EmitGenerator(IROpCode::seq, Type::Integer, length, a, b);
}

IRef Trace::EmitSequence(int64_t length, double a, double b) {
	union {
		double d1;
		int64_t i1;
	};
	union {
		double d2;
		int64_t i2;
	};
	d1 = a;
	d2 = b;
	return EmitGenerator(IROpCode::seq, Type::Double, length, i1, i2);
}

IRef Trace::EmitConstant(Type::Enum type, int64_t length, int64_t c) {
	IRNode n;
	n.arity = IRNode::NULLARY;
	n.group = IRNode::GENERATOR;
	n.op = IROpCode::constant;
	n.type = type;
	n.shape = (IRNode::Shape) { length, -1, 1, -1, false };
	n.outShape = n.shape;
	n.constant.i = c;
	nodes.push_back(n);
	return nodes.size()-1;
}
IRef Trace::EmitGather(Value const& v, IRef i) {
	IRNode n;
	n.arity = IRNode::UNARY;
	n.group = IRNode::GENERATOR;
	n.op = IROpCode::gather;
	n.type = v.type;
	n.shape = nodes[i].outShape;
	n.outShape = n.shape;
	n.unary.a = i;
	n.in = v;
	nodes.push_back(n);
	return nodes.size()-1;
}

IRef Trace::EmitLoad(Value const& v, int64_t length, int64_t offset) {
	IRNode n;
	n.arity = IRNode::NULLARY;
	n.group = IRNode::GENERATOR;
	n.op = IROpCode::load;
	n.type = v.type;
	n.shape = (IRNode::Shape) { length, -1, 1, -1, false };
	n.outShape = n.shape;
	n.constant.i = offset;
	n.in = v;
	nodes.push_back(n);
	return nodes.size()-1;
}

Type::Enum UnifyTypes(Type::Enum a, Type::Enum b) {
	#define UNIFY(X, Y, Z) if(a == Type::X && b == Type::Y) return Type::Z;
        DEFAULT_TYPE_MEET(UNIFY)
        #undef UNIFY
	else _error("Unknown unify types");
}

IRef Trace::EmitIfElse(IRef a, IRef b, IRef cond) {
	IRNode n;
	n.arity = IRNode::TRINARY;
	n.group = IRNode::MAP;
	n.op = IROpCode::ifelse;
	n.type = UnifyTypes(nodes[a].type, nodes[b].type);
	n.shape = nodes[cond].outShape;
	n.outShape = n.shape;
	n.trinary.a = EmitCoerce(a, n.type);
	n.trinary.b = EmitCoerce(b, n.type);
	n.trinary.c = EmitCoerce(cond, Type::Logical);
	nodes.push_back(n);
	return nodes.size()-1;
}

IRef Trace::EmitSLoad(Value const& v) {
	IRNode n;
	n.arity = IRNode::NULLARY;
	n.group = IRNode::SCALAR;
	n.op = IROpCode::sload;
	n.type = v.type;
	n.shape = (IRNode::Shape) { Size, -1, 1, -1, false };
	n.outShape = (IRNode::Shape) { v.length, -1, 1, -1, true };
	n.in = v;
	nodes.push_back(n);
	return nodes.size()-1;
}

IRef Trace::EmitSStore(IRef ref, int64_t index, IRef v) {
	IRNode n;
	n.arity = IRNode::BINARY;
	n.group = IRNode::SCALAR;
	n.op = IROpCode::sstore;
	n.type = nodes[ref].type;
	n.shape = (IRNode::Shape) { Size, -1, 1, -1, false };
	n.outShape = (IRNode::Shape) { std::max(nodes[ref].outShape.length, index), -1, 1, -1, true };
	n.binary.a = ref;
	n.binary.b = v;
	n.binary.data = index;
	nodes.push_back(n);
	return nodes.size()-1;
}

void Trace::MarkLiveOutputs(Thread& thread) {
	
	// Can find live outputs in the stack or in the recorded set of environments
	
	for(size_t i = 0; i < nodes.size(); i++) {
		nodes[i].liveOut = false;
	}
	
	for(Value* v = thread.base-thread.frame.prototype->registers; 
		v < thread.registers+DEFAULT_NUM_REGISTERS; v++) {
		if(v->isFuture() && v->length == Size) {
			nodes[v->future.ref].liveOut = true;
			Output o;
			o.type = Output::REG;
			o.reg = v;
			o.ref = v->future.ref;
			outputs.push_back(o);
		}
	}
	
	for(std::set<Environment*>::const_iterator i = liveEnvironments.begin(); i != liveEnvironments.end(); ++i) {
		for(Environment::const_iterator j = (*i)->begin(); j != (*i)->end(); ++j) {
			Value const& v = j.value();
			if(v.isFuture() && v.length == Size) {
				nodes[v.future.ref].liveOut = true;
				Output o;
				o.type = Output::MEMORY;
				o.pointer = (*i)->makePointer(j.string());
				o.ref = v.future.ref;
				outputs.push_back(o);
			}
		}

		for(int64_t j = 0; j < (*i)->dots.size(); j++) {
			Value const& v = (*i)->dots[j].v;
			if(v.isFuture() && v.length == Size) {
				nodes[v.future.ref].liveOut = true;
				Output o;
				o.type = Output::REG;
				o.reg = (Value*)&v;
				o.ref = v.future.ref;
				outputs.push_back(o);
			}
		}
	}
}

void Trace::WriteOutputs(Thread & thread) {
	if(thread.state.verbose) {
		for(size_t i = 0; i < nodes.size(); i++) {
			if(nodes[i].liveOut) {
				std::string v = thread.stringify(nodes[i].out);
				printf("n%llu = %s\n", i, v.c_str());
			}
		}
	}

	for(size_t i = 0; i < outputs.size(); i++) {
		Output & o = outputs[i];
		Value const& v = nodes[o.ref].out;

		switch(o.type) {
		case Output::REG:
			*o.reg = v;
			break;
		case Output::MEMORY:
			Environment::assignPointer(o.pointer,v);
			break;
		}
	}
}

void Trace::SimplifyOps(Thread& thread) {
	for(IRef ref = 0; ref < (int64_t)nodes.size(); ref++) {
		IRNode & node = nodes[ref];
		switch(node.op) {
			case IROpCode::gt: node.op = IROpCode::lt; std::swap(node.binary.a,node.binary.b); break;
			case IROpCode::ge: node.op = IROpCode::le; std::swap(node.binary.a,node.binary.b); break;
			case IROpCode::add: /* fallthrough */ 
			case IROpCode::mul:
			case IROpCode::land:
			case IROpCode::lor: 
					   if(nodes[node.binary.a].shape.length == 1) std::swap(node.binary.a,node.binary.b); break;
					   if(nodes[node.binary.a].op == IROpCode::constant && nodes[node.binary.b].op != IROpCode::constant) std::swap(node.binary.a,node.binary.b); break;
			default: /*pass*/ break;
		}
	}
}

void Trace::AlgebraicSimplification(Thread& thread) {
	for(IRef ref = 0; ref < (int64_t)nodes.size(); ref++) {
		IRNode & node = nodes[ref];

		// simplify casts
		// cast to same type is a NOP, go to pos
		if(node.op == IROpCode::cast && node.type == nodes[node.unary.a].type) {
			node.op = IROpCode::pos;
			node.arity = IRNode::UNARY;
			node.group = IRNode::MAP;
		}
		if(node.op == IROpCode::cast && nodes[node.unary.a].op == IROpCode::constant) {
			Type::Enum t = node.type;
			Type::Enum s = nodes[node.unary.a].type;
			node = nodes[node.unary.a];
			node.type = t;
			switch(s) {
				case Type::Integer: switch(t) {
					case Type::Double: node.constant.d = Cast<Integer,Double>(thread, node.constant.i); break;
					case Type::Logical: node.constant.l = Cast<Integer,Logical>(thread, node.constant.i); break;
					default: _error("Invalid cast");
				} break;
				case Type::Double: switch(t) {
					case Type::Integer: node.constant.i = Cast<Double,Integer>(thread, node.constant.d); break;
					case Type::Logical: node.constant.l = Cast<Double,Logical>(thread, node.constant.d); break;
					default: _error("Invalid cast");
				} break;
				case Type::Logical: switch(t) {
					case Type::Double: node.constant.d = Cast<Logical,Double>(thread, node.constant.l); break;
					case Type::Integer: node.constant.i = Cast<Logical,Integer>(thread, node.constant.l); break;
					default: _error("Invalid cast");
				} break;
				default: _error("Invalid cast");
			} 
		}
		if(node.op == IROpCode::cast && nodes[node.unary.a].op == IROpCode::seq) {
			Type::Enum t = node.type;
			Type::Enum s = nodes[node.unary.a].type;
			if(t == Type::Double && s == Type::Integer) {
				node = nodes[node.unary.a];
				node.type = Type::Double;
				node.sequence.da = Cast<Integer,Double>(thread, node.sequence.ia);
				node.sequence.db = Cast<Integer,Double>(thread, node.sequence.ib);
			}	
		}

		if(node.arity == IRNode::UNARY && nodes[node.unary.a].op == IROpCode::pos)
			node.unary.a = nodes[node.unary.a].unary.a;
		if(node.arity == IRNode::BINARY && nodes[node.binary.a].op == IROpCode::pos)
			node.binary.a = nodes[node.binary.a].unary.a;
		if(node.arity == IRNode::BINARY && nodes[node.binary.b].op == IROpCode::pos)
			node.binary.b = nodes[node.binary.b].unary.a;

		if(node.op == IROpCode::pow &&
			nodes[node.binary.b].op == IROpCode::constant) {
			if(nodes[node.binary.b].constant.d == 0) {
				// x^0 => 1
				node.op = IROpCode::constant;
				node.arity = IRNode::NULLARY;
				node.group = IRNode::GENERATOR;
				node.constant.d = 1;
			} else if(nodes[node.binary.b].constant.d == 1) {
				// x^1 => x
				node.op = IROpCode::pos;
				node.arity = IRNode::UNARY;
				node.group = IRNode::MAP;
				node.unary.a = node.binary.a;
			} else if(nodes[node.binary.b].constant.d == 2) {
				// x^2 => x*x
				node.op = IROpCode::mul;
				node.binary.b = node.binary.a;
			} else if(nodes[node.binary.b].constant.d == 0.5) {
				// x^0.5 => sqrt(x)
				node.op = IROpCode::sqrt;
				node.arity = IRNode::UNARY;
				node.group = IRNode::MAP;
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
				nodes[node.binary.a].op == IROpCode::constant &&
				nodes[node.binary.b].op == IROpCode::constant) {
			if(node.isInteger()) {
				node.op = IROpCode::constant;
				node.arity = IRNode::NULLARY;
				node.group = IRNode::GENERATOR;
				node.constant.i = addVOp<Integer,Integer>::eval(thread, nodes[node.binary.a].constant.i, nodes[node.binary.b].constant.i);
			} else {
				node.op = IROpCode::constant;
				node.arity = IRNode::NULLARY;
				node.group = IRNode::GENERATOR;
				node.constant.d = addVOp<Double, Double>::eval(thread, nodes[node.binary.a].constant.d, nodes[node.binary.b].constant.d);
			}
		}
		if(node.op == IROpCode::mul &&
				nodes[node.binary.a].op == IROpCode::constant &&
				nodes[node.binary.b].op == IROpCode::constant) {
			if(node.isInteger()) {
				node.op = IROpCode::constant;
				node.arity = IRNode::NULLARY;
				node.group = IRNode::GENERATOR;
				node.constant.i = mulVOp<Integer,Integer>::eval(thread, nodes[node.binary.a].constant.i, nodes[node.binary.b].constant.i);
			} else {
				node.op = IROpCode::constant;
				node.arity = IRNode::NULLARY;
				node.group = IRNode::GENERATOR;
				node.constant.d = mulVOp<Double, Double>::eval(thread, nodes[node.binary.a].constant.d, nodes[node.binary.b].constant.d);
			}
		}

		if(node.op == IROpCode::add && nodes[node.binary.b].op == IROpCode::constant) {
			if(node.isInteger())
				node.constant.i = nodes[node.binary.b].constant.i;
			else
				node.constant.d = nodes[node.binary.b].constant.d;
			node.op = IROpCode::addc;
			node.arity = IRNode::UNARY;
		}
		if(node.op == IROpCode::mul && nodes[node.binary.b].op == IROpCode::constant) {
			if(node.isInteger())
				node.constant.i = nodes[node.binary.b].constant.i;
			else
				node.constant.d = nodes[node.binary.b].constant.d;
			node.op = IROpCode::mulc;
			node.arity = IRNode::UNARY;
		}

		if(node.op == IROpCode::addc && nodes[node.binary.a].op == IROpCode::addc) {
			if(node.isInteger())
				node.constant.i += nodes[node.binary.b].constant.i;
			else
				node.constant.d += nodes[node.binary.b].constant.d;
			node.binary.a = nodes[node.binary.a].binary.a;
		}
		if(node.op == IROpCode::mulc && nodes[node.binary.a].op == IROpCode::mulc) {
			if(node.isInteger())
				nodes[node.binary.b].constant.i *= nodes[nodes[node.binary.a].binary.b].constant.i;
			else
				nodes[node.binary.b].constant.d *= nodes[nodes[node.binary.a].binary.b].constant.d;
			node.binary.a = nodes[node.binary.a].binary.a;
		}

		if(node.op == IROpCode::addc &&
				nodes[node.binary.a].op == IROpCode::seq) {
			int64_t a = node.binary.a;
			if(node.isInteger()) {
				node.op = IROpCode::seq;
				node.arity = IRNode::NULLARY;
				node.group = IRNode::GENERATOR;
				node.sequence.ia = addVOp<Integer,Integer>::eval(thread, nodes[a].sequence.ia, node.constant.i);
				node.sequence.ib = nodes[a].sequence.ib;
			} else {
				node.op = IROpCode::seq;
				node.arity = IRNode::NULLARY;
				node.group = IRNode::GENERATOR;
				node.sequence.da = addVOp<Double,Double>::eval(thread, nodes[a].sequence.da, node.constant.d);
				node.sequence.db = nodes[a].sequence.db;
			}
		}
		
		if(node.op == IROpCode::sub &&
				nodes[node.binary.a].op == IROpCode::seq &&
				nodes[node.binary.b].op == IROpCode::constant) {
			int64_t a = node.binary.a;
			int64_t b = node.binary.b;
			if(node.isInteger()) {
				node.op = IROpCode::seq;
				node.arity = IRNode::NULLARY;
				node.group = IRNode::GENERATOR;
				node.sequence.ia = subVOp<Integer,Integer>::eval(thread, nodes[a].sequence.ia, nodes[b].constant.i);
				node.sequence.ib = nodes[a].sequence.ib;
			} else {
				node.op = IROpCode::seq;
				node.arity = IRNode::NULLARY;
				node.group = IRNode::GENERATOR;
				node.sequence.da = subVOp<Double,Double>::eval(thread, nodes[a].sequence.da, nodes[b].constant.d);
				node.sequence.db = nodes[a].sequence.db;
			}
		}

		if(node.op == IROpCode::mulc && nodes[node.binary.a].op == IROpCode::seq) {
			int64_t a = node.binary.a;
			if(node.isInteger()) {
				node.op = IROpCode::seq;
				node.arity = IRNode::NULLARY;
				node.group = IRNode::GENERATOR;
				int64_t i = node.constant.i;
				node.sequence.ia = mulVOp<Integer,Integer>::eval(thread, nodes[a].sequence.ia, i);
				node.sequence.ib = mulVOp<Integer,Integer>::eval(thread, nodes[a].sequence.ib, i);
			} else {
				node.op = IROpCode::seq;
				node.arity = IRNode::NULLARY;
				node.group = IRNode::GENERATOR;
				double d = node.constant.d;
				node.sequence.da = mulVOp<Double,Double>::eval(thread, nodes[a].sequence.da, d);
				node.sequence.db = mulVOp<Double,Double>::eval(thread, nodes[a].sequence.db, d);
			}
		}

		if(node.op == IROpCode::addc && node.constant.i == 0) {
			node.op = IROpCode::pos;
			node.unary.a = node.binary.a;
		}

		if(node.op == IROpCode::mulc && 
			((node.isDouble() && node.constant.d == 1) || (node.isInteger() && node.constant.i == 1))) {
			node.op = IROpCode::pos;
			node.arity = IRNode::UNARY;
			node.group = IRNode::MAP;
			node.unary.a = node.binary.a;
		}
		
		if(node.op == IROpCode::land &&
				nodes[node.binary.b].op == IROpCode::constant &&
				Logical::isTrue(nodes[node.binary.b].constant.l)) {
			node.op = IROpCode::pos;
			node.arity = IRNode::UNARY;
			node.group = IRNode::MAP;
			node.unary.a = node.binary.a;
		}

		if(node.op == IROpCode::land &&
				nodes[node.binary.b].op == IROpCode::constant &&
				Logical::isFalse(nodes[node.binary.b].constant.l)) {
			node.op = IROpCode::constant;
			node.arity = IRNode::NULLARY;
			node.group = IRNode::GENERATOR;
			node.constant.l = Logical::FalseElement;
		}

		if(node.op == IROpCode::lor &&
				nodes[node.binary.b].op == IROpCode::constant &&
				Logical::isFalse(nodes[node.binary.b].constant.l)) {
			node.op = IROpCode::pos;
			node.arity = IRNode::UNARY;
			node.group = IRNode::MAP;
			node.unary.a = node.binary.a;
		}

		if(node.op == IROpCode::lor &&
				nodes[node.binary.b].op == IROpCode::constant &&
				Logical::isTrue(nodes[node.binary.b].constant.l)) {
			node.op = IROpCode::constant;
			node.arity = IRNode::NULLARY;
			node.group = IRNode::GENERATOR;
			node.constant.l = Logical::TrueElement;
		}
		
		if(node.op == IROpCode::sub &&
				nodes[node.binary.b].op == IROpCode::constant &&
				nodes[node.binary.b].constant.i == 1 &&
				nodes[node.binary.a].op == IROpCode::addc &&
				nodes[node.binary.a].constant.i == 1) {
			node.op = IROpCode::pos;
			node.arity = IRNode::UNARY;
			node.group = IRNode::MAP;
			node.unary.a = nodes[node.binary.a].binary.a;
		}

		if(node.op == IROpCode::gather &&
			nodes[node.unary.a].op == IROpCode::seq &&
			nodes[node.unary.a].sequence.ib == 1) {
			node.op = IROpCode::load;
			node.arity = IRNode::NULLARY;
			node.group = IRNode::GENERATOR;
			node.constant.i = nodes[node.unary.a].sequence.ia;
		}
	}
}

void Trace::CSEElimination(Thread& thread) {
	// look for exact same op somewhere above, replace myself with pos of that one...
	//std::tr1::unordered_set<IRNode, IRHash> hash;
	for(IRef i = 0; i < (IRef)nodes.size(); i++) {
		IRNode& node = nodes[i];
		switch(node.arity) {
			case IRNode::TRINARY:
				if(nodes[node.trinary.c].op == IROpCode::pos)
					node.trinary.c = nodes[node.trinary.c].unary.a;
			case IRNode::BINARY:
				if(nodes[node.binary.b].op == IROpCode::pos)
					node.binary.b = nodes[node.binary.b].unary.a;
			case IRNode::UNARY:
				if(nodes[node.unary.a].op == IROpCode::pos)
					node.unary.a = nodes[node.unary.a].unary.a;
			case IRNode::NULLARY:
			default:
				if(node.shape.filter >= 0 && nodes[node.shape.filter].op == IROpCode::pos)
					node.shape.filter = nodes[node.shape.filter].unary.a;
				if(node.shape.split >= 0 && nodes[node.shape.split].op == IROpCode::pos)
					node.shape.split = nodes[node.shape.split].unary.a;
		}
		/*std::ir1::unordered_set<IRNode, IRHash>::const_iterator j =
			hash.find(node);
		if(j != */
		for(IRef j = 0; j < i; j++) {
			if(node == nodes[j]) {
				node.op = IROpCode::pos;
				node.arity = IRNode::UNARY;
				node.group = IRNode::MAP;
				node.unary.a = j;
				break;
			}
		}
	}
}

// Propogate liveOut to live
void Trace::UsePropogation(Thread& thread) {
	for(size_t i = 0; i < nodes.size(); i++) {
		nodes[i].live = false | nodes[i].liveOut;
	}
	for(IRef ref = nodes.size(); ref > 0; ref--) {
		IRNode & node = nodes[ref-1];
		if(node.live) {
			switch(node.arity) {
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
				case IRNode::NULLARY:
					break;
			}
			// mark used shape nodes
			if(node.shape.filter >= 0) 
				nodes[node.shape.filter].live = true;
			if(node.shape.split >= 0) 
				nodes[node.shape.split].live = true;
		}
	}
}

// Propogate live def information down to all uses
void Trace::DefPropogation(Thread& thread) {
	for(IRef ref = 0; ref < (IRef)nodes.size(); ref++) {
		IRNode & node = nodes[ref];
		if(!node.live) {
			switch(node.arity) {
				case IRNode::TRINARY:
					node.live = 	nodes[node.trinary.a].live &&
							nodes[node.trinary.b].live &&
							nodes[node.trinary.c].live;
					break;
				case IRNode::BINARY: 
					node.live = 	nodes[node.binary.a].live &&
							nodes[node.binary.b].live;
					break;
				case IRNode::UNARY:
					node.live = 	nodes[node.unary.a].live;
					break;
				case IRNode::NULLARY: 
					/* nothing */
					break;
			}
			// mark used shape nodes
			if(node.shape.filter >= 0) 
				node.live = node.live && nodes[node.shape.filter].live;
			if(node.shape.split >= 0) 
				node.live = node.live && nodes[node.shape.split].live;
		}
	}
}

void Trace::DeadCodeElimination(Thread& thread) {
	for(IRef ref = 0; ref < (IRef)nodes.size(); ref++) {
		IRNode& node = nodes[ref];
		if(!node.live) {
			node.op = IROpCode::nop;
			node.arity = IRNode::NULLARY;
			node.group = IRNode::NOP;
		}
	}
}

void Trace::PropogateShape(IRNode::Shape shape, IRNode& node) {
	if(	node.group != IRNode::FOLD &&
		node.group != IRNode::FILTER &&
		node.group != IRNode::SPLIT) {
		if(node.shape.length == -1) {
			node.shape = shape;
		} else {
			// unify shapes
			assert(node.shape.length == shape.length);
			// look for meet of filters
			while(node.shape.filter != shape.filter) {
				if(node.shape.filter > shape.filter) {
					node.shape.filter = nodes[node.shape.filter].shape.filter;
				} else {
					shape.filter = nodes[shape.filter].shape.filter;
				}
			}
		}
	}
}

void Trace::ShapePropogation(Thread& thread) {
	
	// wipe out original shapes on MAPs
	for(IRef ref = (IRef)nodes.size()-1; ref >= 0; ref--) {
		IRNode& node = nodes[ref];
		if(	node.group != IRNode::FOLD &&
			node.group != IRNode::FILTER &&
			node.group != IRNode::SPLIT) {
			node.shape.length = -1;
		}
	}

	for(IRef ref = (IRef)nodes.size()-1; ref >= 0; ref--) {
		IRNode& node = nodes[ref];
		if(node.liveOut)
			PropogateShape(node.outShape, node);
		switch(node.arity) {
			case IRNode::TRINARY:
				PropogateShape(node.shape, nodes[node.trinary.a]);
				PropogateShape(node.shape, nodes[node.trinary.b]);
				PropogateShape(node.shape, nodes[node.trinary.c]);
				break;
			case IRNode::BINARY: 
				PropogateShape(node.shape, nodes[node.binary.a]);
				PropogateShape(node.shape, nodes[node.binary.b]);
				break;
			case IRNode::UNARY:
				PropogateShape(node.shape, nodes[node.unary.a]);
				break;
			case IRNode::NULLARY: 
				/* nothing */
				break;
		}
	}
}

void Trace::Optimize(Thread& thread) {

	if(thread.state.verbose)
		printf("executing trace:\n%s\n",toString(thread).c_str());

	MarkLiveOutputs(thread);
	UsePropogation(thread);
	DeadCodeElimination(thread);	// avoid optimizing code that's dead anyway

	//ShapePropogation(thread);	
	SimplifyOps(thread);
	
	// Turn these back on when I'm sure about optimizing across shape changes...
	// E.g. a <- 1:64; a[a < 32]
	AlgebraicSimplification(thread);
	//CSEElimination(thread);

	// move outputs up...
	for(size_t i = 0; i < outputs.size(); i++) {
		IRef& r = outputs[i].ref;
		while(nodes[r].op == IROpCode::pos) {
			nodes[r].liveOut = false;
			r = nodes[r].unary.a;
		}
		nodes[r].liveOut = true;
	}
	
	UsePropogation(thread);
	DeadCodeElimination(thread);

	if(thread.state.verbose)
		printf("optimized:\n%s\n",toString(thread).c_str());
}


// ref must be evaluated. Other stuff doesn't need to be executed unless it improves performance.
void Trace::Execute(Thread & thread, IRef ref) {
	// partition into stuff that we will execute and stuff that we won't
	/*std::vector<IRNode, traceable_allocator<IRNode> > left;

	for(IRef i = 0; i < nodes.size(); i++) {
		nodes[i].liveOut = false;
	}
	nodes[ref].liveOut = true;
	
	// walk backwards marking everything we'll need
	UsePropogation(thread);
	// walk forwards propogating use information down
	DefPropogation(thread);	
	
	for(IRef i = 0; i < nodes.size(); i++) {
		Node& node = nodes[i];
		if(!node.live) {
			// pull out unmarked items and put NOPs in their place
			// update uses
			// uses of live variables need to be replaced with a load of the output
			// but outputs aren't defined yet.
			// it makes a liveOut
		}
	}

	Optimize(thread);
	JIT(thread);
	WriteOutputs(thread);
	
	n_recorded_since_last_exec = 0;
	nodes.swap(left);
	outputs.clear();*/

	Execute(thread);
}

// everything must be evaluated in the end...
void Trace::Execute(Thread & thread) {
	Optimize(thread);
	// if there were any live outputs
	if(outputs.size() > 0) {
		JIT(thread);
		WriteOutputs(thread);
	}
	Reset();
}
