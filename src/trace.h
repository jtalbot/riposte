#ifndef _RIPOSTE_TRACE_H
#define _RIPOSTE_TRACE_H
#include <gc/gc_cpp.h>
#include<vector>
#include<map>
#include<memory>
#include "bc.h"
#include "ir.h"
#include "type.h"
#include "recording.h"

struct Trace;
struct Prototype;
//member of State, manages information for all traces
//and the currently recording trace (if any)

#define TRACE_MAX_NODES (128)
#define TRACE_MAX_OUTPUTS (128)
#define TRACE_MAX_VECTOR_REGISTERS (32)
#define TRACE_VECTOR_WIDTH (64)
//maximum number of instructions to record before dropping out of the
//recording interpreter
#define TRACE_MAX_RECORDED (1024)

struct State;
struct Environment;
struct Trace {

	double registers[TRACE_MAX_VECTOR_REGISTERS][TRACE_VECTOR_WIDTH];

	IRNode nodes[TRACE_MAX_NODES];
	size_t n_nodes;
	size_t n_recorded;

	int64_t length;

	struct Location {
		enum Type { SLOT, REG, VAR};
		Type type;
		union {
			Environment * environment; //pointer to environment for slots and variables
			Value * base; //pointer to base register for registers
		};
		int64_t id;
	};
	struct Output {
		Location location; //location where an output might exist
		                   //if that location is live and contains a future then that is a live output
		IRef ref; //(used only to enable pretty printing) value of the output in the trace code,
	};

	Output outputs[TRACE_MAX_OUTPUTS];
	size_t n_outputs;
	Value * max_live_register_base;
	int64_t max_live_register;

	void reset();
	void execute(State & state);
	std::string toString(State & state);
};

struct TraceState {
	TraceState() {
		active = false;
	}
	Trace current_trace;
	bool active;

	bool is_tracing() const { return active; }

	Instruction const * begin_tracing(State & state, Instruction const * inst, size_t length) {
		current_trace.reset();
		current_trace.length = length;
		active = true;
		return recording_interpret(state,inst);

	}

	void end_tracing(State & state) {
		if(active) {
			active = false;
			current_trace.execute(state);
		}
	}
};

#endif
