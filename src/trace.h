#ifndef _RIPOSTE_TRACE_H
#define _RIPOSTE_TRACE_H
#include <gc/gc_cpp.h>
#include<vector>
#include<map>
#include "bc.h"
#include "ir.h"

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


struct RenamingTable {

	struct Status {
		Status() { flags = UNDEF; }
		enum {
			UNDEF = 0,
			READ = 1, //do we read this into the trace externally?
			WRITE = 2, //have we written to the variable during the trace?
		};
		int32_t flags;
		int32_t node; //what IRNode defines its value (undefined if flags == 0)
	};

	Status slots[1024]; //this will be updated to work with slots when slots are implemented, for now it refers to registers
	std::map<int64_t,Status> variables;

	void assign_slot(int64_t slot_id,int32_t node) {
		slots[slot_id].flags |= Status::WRITE;
		slots[slot_id].node = node;
	}
	void assign_var(int64_t var_id, int32_t node) {
		Status & v = variables[var_id];
		v.flags |= Status::WRITE;
		v.node = node;
	}
	//returns true if slot already exists, node points to definitions
	//returns false if it did not exist, sets READ status indicating that it must be read from outside the trace, and sets it initial value to load_inst, the offset where the load instruction for the value should be inserted
	bool get_slot(int64_t slot_id, int32_t load_inst, int32_t * node) {
		Status & s = slots[slot_id];
		return define_or_get(&s,load_inst,node);
	}
	bool get_var(int64_t var_id, int32_t load_inst, int32_t * node) {
		Status & s= variables[var_id];
		return define_or_get(&s,load_inst,node);
	}

	bool define_or_get(Status * s, int32_t load_inst, int32_t * node) {
		if(s->flags == Status::UNDEF) {
			s->flags |= Status::READ;
			s->node = load_inst;
			*node = load_inst;
			return false;
		} else {
			*node = s->node;
			return true;
		}
	}

	bool var_is_loop_invariant(int64_t var_id) {
		return (variables[var_id].flags != (Status::WRITE | Status::READ));
	}
	bool slot_is_loop_invariant(int64_t slot_id) {
		return (slots[slot_id].flags != (Status::WRITE | Status::READ));
	}
	typedef int Snapshot;
	Snapshot create_snapshot() {
		//NYI: save a view of the renaming table, so we know how to store state when trace exits
		//this can be implemented with a snapshot being an offset, and the renaming table being implemented as
		//a journal
		return 0;
	}
};

struct Trace : public gc {
	Trace(Code * code, Instruction * trace_start);
	Code * code; //code segment in which trace begins
	Instruction * trace_start; //pointer to first instruction recorded by trace
	Instruction trace_inst; //copy of first instruction run by trace
	                        //trace_start will be overwritten with a trace bytecode when it is installed
	                        //trace_inst holds the old instruction to execution when we need to fallback to the interpreter

	int64_t depth; //how many additional frames have been pushed since recording began

	RenamingTable renaming_table;

	std::vector<Value>  constants;
	std::vector<IRNode> recorded;
};

class State;
void trace_compile_and_install(State & state, Trace * trace);

#endif
