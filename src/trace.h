#ifndef _RIPOSTE_TRACE_H
#define _RIPOSTE_TRACE_H
#include <gc/gc_cpp.h>
#include<vector>
#include<map>
#include<memory>
#include "bc.h"
#include "ir.h"
#include "type.h"

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
struct Trace {

	double registers[TRACE_MAX_VECTOR_REGISTERS][TRACE_VECTOR_WIDTH];

	IRNode nodes[TRACE_MAX_NODES];
	size_t n_nodes;
	size_t n_recorded;

	int64_t length;

	struct Output {
		enum Location { E_SLOT, E_REG, E_VAR };
		Location location_type;
		IRef ref;
		Type::Enum typ;
		int64_t location;
	};

	Output outputs[TRACE_MAX_OUTPUTS];
	size_t n_outputs;
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

	void begin_tracing() {
		current_trace.reset();
		active = true;
	}

	void end_tracing(State & state) {
		active = false;
		current_trace.execute(state);
	}
};

#endif
