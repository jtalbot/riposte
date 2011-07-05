#ifndef _RIPOSTE_TRACE_H
#define _RIPOSTE_TRACE_H
#include <gc/gc_cpp.h>
#include<vector>
#include "bc.h"

struct Trace;
struct Code;
//member of State, manages information for all traces
//and the currently recording trace (if any)
struct TraceState {
	TraceState() {
		//numbers are currently just for debugging
		start_count = 2;
		max_attempts = 3;
		max_length = 10;
		current_trace = NULL;
	}

	Trace * current_trace;

	bool is_tracing() const { return current_trace != NULL; }

	//configurable constants
	int64_t start_count; //invocations of back-edge before starting a trace
	int64_t max_attempts; //how many times should we attempt to create  a trace before giving up
	int64_t max_length; //how many IRNodes should we attempt to record before giving up
};

struct Trace : public gc {
	Trace(Code * code, Instruction * trace_start);
	Code * code; //code segment in which trace begins
	Instruction * trace_start; //pointer to first instruction recorded by trace
	Instruction trace_inst; //copy of first instruction run by trace
	                        //trace_start will be overwritten with a trace bytecode when it is installed
	                        //trace_inst holds the old instruction to execution when we need to fallback to the interpreter

	int64_t depth; //how many additional frames have been pushed since recording began
	std::vector<Instruction> recorded_bcs;
};

void trace_compile_and_install(Trace * trace);

#endif
