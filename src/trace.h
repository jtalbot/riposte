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
		max_length = 100;
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

	enum {SLOT = 0, VARIABLE = 1};

	struct Status {
		int64_t id;
		int32_t ir_node;

		uint32_t location : 1; //SLOT or VARIABLE

		uint32_t read : 1; //do we read this into the trace externally?
		uint32_t write : 1; //have we written to the variable during the trace?
	};

	struct Entry {
		int64_t id;
		uint32_t location : 1;
	};

	int32_t last_snapshot;
	std::vector<Status> journal;
	std::vector<Entry> outputs; //list of values that will be written when the trace completes

	RenamingTable() {
		last_snapshot = 0;
	}


	bool lookup(uint32_t location, int64_t id, int32_t snapshot, int32_t * idx) const {
		for(int32_t i = snapshot; i > 0; i--) {
			const Status & s = journal[i - 1];
			if(s.id == id && s.location == location) {
				*idx = i - 1;
				return true;
			}
		}
		return false;
	}

	bool get(uint32_t location, int64_t id, int32_t * node) const { return get(location,id,current_view(),node); }
	bool get(uint32_t location, int64_t id, int32_t snapshot, int32_t * node, bool * read = NULL, bool * write = NULL) const {
		int32_t idx;
		if(lookup(location,id,snapshot,&idx)) {
			*node = journal[idx].ir_node;
			if(read)
				*read = journal[idx].read != 0;
			if(write)
				*write = journal[idx].write != 0;
			return true;
		} else
			return false;
	}
	void assign(uint32_t location, int64_t id, int32_t node) {
		int32_t idx;
		if(lookup(location,id,current_view(),&idx)) {
			if(journal[idx].write == 0) { //only add to outputs the first time we set the write bit
				Entry w = {id, location};
				outputs.push_back(w);
			}
			Status * s;
			if(idx >= last_snapshot) {
				//no snapshots for this range have been requested, we can
				//just make the update in place
				s = &journal[idx];
			} else {
				//otherwise, make a copy
				journal.push_back(journal[idx]);
				s = &journal.back();
			}
			s->write = 1;
			s->ir_node = node;
		} else {
			Status s = { id, node, location, 0, 1};
			Entry w = { id, location };
			journal.push_back(s);
			outputs.push_back(w);
		}
	}

	void input(uint32_t location, int64_t id, int32_t node) {
		Status s  = { id, node, location, 1, 0};
		journal.push_back(s);
	}

	int32_t current_view() const {
		return journal.size();
	}
	int32_t create_snapshot() {
		//save a view of the renaming table, so we know how to store state when trace exits
		last_snapshot = current_view();
		return last_snapshot;
	}
};

struct TraceExit {
	int32_t snapshot;
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
	std::vector<TraceExit> exits;

	std::vector<IRNode> recorded;

	std::vector<IRNode> compiled;

	void compile();
};

class State;
void trace_compile_and_install(State & state, Trace * trace);

#endif
