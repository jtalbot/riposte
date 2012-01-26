#include "interpreter.h"
#include "vector.h"
#include "ops.h"
#include "sse.h"

#include <stdlib.h>

void Trace::Reset() {
	n_nodes = length = n_outputs = n_output_values = n_pending_nodes = n_pending_outputs = 0;
	uniqueShapes = -1;
}

std::string Trace::toString(Thread & thread) {
	std::ostringstream out;
	out << "recorded: \n";
	for(size_t j = 0; j < n_nodes; j++) {
		IRNode & node = nodes[j];
		out << "n" << j << " : " << Type::toString(node.type) << "[" << node.length << "]" << " = " << IROpCode::toString(node.op) << "\t";
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
		case IROpCode::gather: out << "n" << node.unary.a << "\t" << "$" << (void*)node.unary.data; break;
		case IROpCode::loadc: out << ( (node.type == Type::Integer) ? node.loadc.i : node.loadc.d); break;
		case IROpCode::loadv: out << "$" << node.loadv.p; break;
		case IROpCode::storec: /*fallthrough*/
		case IROpCode::storev: out << "o" << node.store.dst - output_values << "\tn" << node.store.a; break;
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
		out << " = o" << o.value - output_values << "\n";
	}
	out << "output_values: \n";
	for(size_t i = 0; i < n_output_values; i++) {
		out << "o" << i << " : " << Type::toString(output_values[i].type) << "[" << output_values[i].length << "]\n";
	}
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

//attempts to find a future at location l, returns true if the location is live and contains a future
static bool get_location_value_if_live(Thread & thread, Trace & trace, const Trace::Location & l, Value & v) {
	if(thread.tracing.LocationIsDead(l))
		return false;
	v = get_location_value(thread,l);
	return v.isFuture() && v.future.trace_id == thread.tracing.TraceID(trace);
}

void Trace::InitializeOutputs(Thread & thread) {
	//check list of recorded output locations for live futures
	//for each live future found, replace the future with a concrete object
	//the data for the object will be filled in by the trace interpreter
	Value * values[TRACE_MAX_NODES];
	bzero(values,sizeof(Value *) * n_nodes);
	for(size_t i = 0; i < n_outputs; ) {
		Output & o = outputs[i];
		Value loc;
		if(!get_location_value_if_live(thread,*this,o.location,loc)) {
			o = outputs[--n_outputs];
		} else {
			IRef ref = loc.future.ref;
			Type::Enum typ = loc.future.typ;
			assert(ref < n_nodes);
			if(values[ref] == NULL) { //if this is the first time we see this node as an output we create a value for it
				Value & v = output_values[n_output_values++];
				Value::Init(v,typ,loc.length>=0?loc.length:0); //initialize the type of the output value, the actual value (i.e. v.p) will be set after it is calculated in the trace
				if(loc.length == length) {

					if(typ == Type::Logical) {
						v.p = new (PointerFreeGC) char[length];
					} else if(length < 128) {
						double * dp = new (PointerFreeGC) double[length + 1];
						if( ( (int64_t)dp & 0xF) != 0)
							v.p = dp + 1;
						else
							v.p = dp;
						assert( ((int64_t)v.p & 0xF) == 0);
					} else {
						v.p = new (PointerFreeGC) double[length];
						assert( ((int64_t)v.p & 0xF) == 0);
					}

					EmitStoreV(typ,loc.length,&v,ref);
				} else if(loc.length >= 0) {
					// conservative allocation for reductions
					// assume all threads might write. Allocate
					// 64-byte wide region for each to avoid false
					// sharing. Allocate at least 128 so it's aligned.
					// Clean this up...
					uint64_t n = std::max((int64_t)128, thread.state.nThreads*8);
					v.length = n;
					v.p = new (PointerFreeGC) double[n];
					assert( ((int64_t)v.p & 0xF) == 0);
					EmitStoreC(typ,loc.length,&v,ref);
				} else {
					EmitStoreV(typ,loc.length,&v,ref);
				}
				n_nodes = n_pending_nodes;
				values[ref] = &v;
			}
			set_location_value(thread,o.location,Value::Nil()); //mark this location in the interpreter as already seen
			o.value = values[ref];
			i++;
		}
	}
}

void Trace::WriteOutputs(Thread & thread) {
	if(thread.state.verbose) {
		for(size_t i = 0; i < n_output_values; i++) {
			std::string v = thread.stringify(output_values[i]);
			printf("o%d = %s\n", (int) i, v.c_str());
		}
	}
	for(size_t i = 0; i < n_outputs; i++) {
		set_location_value(thread,outputs[i].location,*outputs[i].value);
	}
}

void Trace::Execute(Thread & thread) {
	JIT(thread);
}
