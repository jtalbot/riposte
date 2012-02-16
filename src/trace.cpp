#include "interpreter.h"
#include "vector.h"
#include "ops.h"
#include "sse.h"

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

void Trace::Bind(Thread& thread, Value const& v) {
	if(!v.isFuture()) return;
	Execute(thread, v.future.ref);
}

void Trace::Flush(Thread & thread) {
	n_recorded_since_last_exec = 0;
	if(nodes.size() > 0) {
		Execute(thread);
	}
	Reset();
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
		UNARY_FOLD_SCAN_BYTECODES(UNARY)
		BINARY_BYTECODES(BINARY)
		UNARY(cast)
		BINARY(filter)
		BINARY(split)
		TRINARY(ifelse)
		case IROpCode::seq:
			if(node.type == Type::Double)
				out << node.sequence.da << "\t" << node.sequence.db;
			else
				out << node.sequence.ia << "\t" << node.sequence.ib;
		break;
		case IROpCode::constant: out << ( (node.type == Type::Integer) ? node.constant.i : (node.type == Type::Logical) ? node.constant.l : node.constant.d); break;
		case IROpCode::load: out << "$" << node.out.p << "\tn" << node.unary.a; break;
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
	if(	(op == IROpCode::sum || op == IROpCode::prod || 
		 op == IROpCode::min || op == IROpCode::max ||
		 op == IROpCode::any || op == IROpCode::all)) {
		n.enc = IRNode::FOLD;
		n.shape = (IRNode::Shape) { nodes[a].shape.levels, 0, 1, 0 };
	} else {
		n.enc = IRNode::UNARY;
		n.shape = nodes[a].shape;
	}
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
	if(nodes[a].shape.length == 1)
		n.shape = nodes[b].shape;
	else if(nodes[b].shape.length == 1)
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

IRef Trace::EmitFilter(IRef a, IRef b) {
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
		IRef e = EmitBinary(IROpCode::mul, Type::Integer, f, EmitConstant(Type::Integer, 1, levels), 0);
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

IRef Trace::EmitSequence(int64_t length, int64_t a, int64_t b) {
	IRNode n;
	n.enc = IRNode::SEQUENCE;
	n.op = IROpCode::seq;
	n.type = Type::Integer;
	n.shape = (IRNode::Shape) { length, -1, 1, -1 };
	n.sequence.ia = a;
	n.sequence.ib = b;
	nodes.push_back(n);
	return nodes.size()-1;
}

IRef Trace::EmitSequence(int64_t length, double a, double b) {
	IRNode n;
	n.enc = IRNode::SEQUENCE;
	n.op = IROpCode::seq;
	n.type = Type::Double;
	n.shape = (IRNode::Shape) { length, -1, 1, -1 };
	n.sequence.da = a;
	n.sequence.db = b;
	nodes.push_back(n);
	return nodes.size()-1;
}

IRef Trace::EmitConstant(Type::Enum type, int64_t length, int64_t c) {
	IRNode n;
	n.enc = IRNode::CONSTANT;
	n.op = IROpCode::constant;
	n.type = type;
	n.shape = (IRNode::Shape) { length, -1, 1, -1 };
	n.constant.i = c;
	nodes.push_back(n);
	return nodes.size()-1;
}
IRef Trace::EmitLoad(Value const& v, IRef i) {
	IRNode n;
	n.enc = IRNode::LOAD;
	n.op = IROpCode::load;
	n.type = v.type;
	n.shape = (IRNode::Shape) { v.length, -1, 1, -1 };
	n.unary.a = i;
	n.out = v;
	nodes.push_back(n);
	return nodes.size()-1;
}

void Trace::MarkLiveOutputs(Thread& thread) {
	
	// Can find live outputs in the stack or in the recorded set of environments
	
	for(size_t i = 0; i < nodes.size(); i++) {
		nodes[i].liveOut = false;
	}
	
	for(Value* v = thread.base; v < thread.registers+DEFAULT_NUM_REGISTERS; v++) {
		if(v->isFuture()) {
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
			if(v.isFuture()) {
				nodes[v.future.ref].liveOut = true;
				Output o;
				o.type = Output::MEMORY;
				o.pointer = (*i)->makePointer(j.string());
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
					   if(nodes[node.binary.a].enc == IRNode::CONSTANT && nodes[node.binary.b].enc != IRNode::CONSTANT) std::swap(node.binary.a,node.binary.b); break;
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
			node.enc = IRNode::UNARY;
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
				nodes[node.binary.a].op == IROpCode::constant &&
				nodes[node.binary.b].op == IROpCode::constant) {
			if(node.isInteger()) {
				node.op = IROpCode::constant;
				node.enc = IRNode::CONSTANT;
				node.constant.i = addVOp<Integer,Integer>::eval(thread, nodes[node.binary.a].constant.i, nodes[node.binary.b].constant.i);
			} else {
				node.op = IROpCode::constant;
				node.enc = IRNode::CONSTANT;
				node.constant.d = addVOp<Double, Double>::eval(thread, nodes[node.binary.a].constant.d, nodes[node.binary.b].constant.d);
			}
		}
		if(node.op == IROpCode::mul &&
				nodes[node.binary.a].op == IROpCode::constant &&
				nodes[node.binary.b].op == IROpCode::constant) {
			if(node.isInteger()) {
				node.op = IROpCode::constant;
				node.enc = IRNode::CONSTANT;
				node.constant.i = mulVOp<Integer,Integer>::eval(thread, nodes[node.binary.a].constant.i, nodes[node.binary.b].constant.i);
			} else {
				node.op = IROpCode::constant;
				node.enc = IRNode::CONSTANT;
				node.constant.d = mulVOp<Double, Double>::eval(thread, nodes[node.binary.a].constant.d, nodes[node.binary.b].constant.d);
			}
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
				nodes[node.binary.a].op == IROpCode::seq &&
				nodes[node.binary.b].op == IROpCode::constant) {
			int64_t a = node.binary.a;
			int64_t b = node.binary.b;
			if(node.isInteger()) {
				node.op = IROpCode::seq;
				node.enc = IRNode::SEQUENCE;
				node.sequence.ia = addVOp<Integer,Integer>::eval(thread, nodes[a].sequence.ia, nodes[b].constant.i);
				node.sequence.ib = nodes[a].sequence.ib;
			} else {
				node.op = IROpCode::seq;
				node.enc = IRNode::SEQUENCE;
				node.sequence.da = addVOp<Double,Double>::eval(thread, nodes[a].sequence.da, nodes[b].constant.d);
				node.sequence.db = nodes[a].sequence.db;
			}
		}
		
		if(node.op == IROpCode::mul &&
				nodes[node.binary.a].op == IROpCode::seq &&
				nodes[node.binary.b].op == IROpCode::constant) {
			int64_t a = node.binary.a;
			int64_t b = node.binary.b;
			if(node.isInteger()) {
				node.op = IROpCode::seq;
				node.enc = IRNode::SEQUENCE;
				node.sequence.ia = mulVOp<Integer,Integer>::eval(thread, nodes[a].sequence.ia, nodes[b].constant.i);
				node.sequence.ib = mulVOp<Integer,Integer>::eval(thread, nodes[a].sequence.ib, nodes[b].constant.i);
			} else {
				node.op = IROpCode::seq;
				node.enc = IRNode::SEQUENCE;
				node.sequence.da = mulVOp<Double,Double>::eval(thread, nodes[a].sequence.da, nodes[b].constant.d);
				node.sequence.db = mulVOp<Double,Double>::eval(thread, nodes[a].sequence.db, nodes[b].constant.d);
			}
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
				case IRNode::LOAD: /*fallthrough*/
				case IRNode::UNARY:
					nodes[node.unary.a].live = true;
					break;
				case IRNode::FOLD: 
					nodes[node.fold.a].live = true;
					break;
				case IRNode::CONSTANT: /*fallthrough*/
				case IRNode::SEQUENCE: /*fallthrough*/
				case IRNode::NOP: 
					/* nothing */
					break;
			}
			// mark used shape nodes
			if(node.shape.filter >= 0) 
				nodes[node.shape.filter].live = true;
			if(node.shape.split >= 0) 
				nodes[node.shape.split].live = true;
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
	for(IRef ref = 0; ref < (int64_t)nodes.size(); ref++) {
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
