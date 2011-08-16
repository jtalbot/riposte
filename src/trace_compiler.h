#ifndef _RIPOSTE_TRACE_COMPILER_H
#define _RIPOSTE_TRACE_COMPILER_H
#include "common.h"
#include "enum.h"
class State;
class Trace;


#define ENUM_TC_STATUS(_) \
	_(SUCCESS,"success") \
	_(UNSUPPORTED_IR,"trace compiler encountered an unsupported ir node") \
	_(RUNTIME_ERROR,"trace execution encountered a runtime error") \
	_(DISABLED,"trace is disabled")

DECLARE_ENUM(TCStatus,ENUM_TC_STATUS)

//implementation details are hidden in trace_compiler.cpp so we
struct TraceCompiler {

	virtual TCStatus::Enum compile() = 0;
	virtual TCStatus::Enum execute(State & state, int64_t * offset) = 0;
	virtual ~TraceCompiler() {}
	static TraceCompiler * create(Trace * trace);
protected:
	TraceCompiler() {}
};


#endif
