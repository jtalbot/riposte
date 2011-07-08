#ifndef _RIPOSTE_TRACE_COMPILER_H
#define _RIPOSTE_TRACE_COMPILER_H
#include "common.h"
#include "enum.h"
class State;
class Trace;


#define ENUM_TC_STATUS(_,p) \
	_(SUCCESS,"success",p) \
	_(UNSUPPORTED_IR,"trace compiler encountered an unsupported ir node",p) \
	_(RUNTIME_ERROR,"trace execution encountered a runtime error",p) \
	_(DISABLED,"trace is disabled",p)

DECLARE_ENUM(TCStatus,ENUM_TC_STATUS)

//implementation details are hidden in trace_compiler.cpp so we
struct TraceCompiler {

	virtual TCStatus compile() = 0;
	virtual TCStatus execute(State & state, int64_t * offset) = 0;
	virtual ~TraceCompiler() {}
	static TraceCompiler * create(Trace * trace);
protected:
	TraceCompiler() {}
};


#endif
