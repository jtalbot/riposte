#include "trace_compiler.h"
#include "trace.h"

#ifndef RIPOSTE_DISABLE_TRACING


struct TraceCompilerImpl : public TraceCompiler {
	TraceCompilerImpl(Trace * trace) {
		this->trace = trace;
	}
	TCStatus compile() {
		return TCStatus::SUCCESS;
	}
	TCStatus execute(State & s, int64_t * offset) {
		*offset = 0;
		return TCStatus::SUCCESS;
	}

	Trace * trace;
};


#else

struct TraceCompilerImpl : public TraceCompiler {
	TraceCompilerImpl(Trace * trace) {}
	TCStatus compile() {
		return TCStatus::DISABLED;
	}
	TCStatus execute(State & s, int64_t * offset) {
		*offset = 0;
		return TCStatus::DISABLED;
	}
};

#endif

DEFINE_ENUM(TCStatus,ENUM_TC_STATUS)
DEFINE_ENUM_TO_STRING(TCStatus,ENUM_TC_STATUS)
TraceCompiler * TraceCompiler::create(Trace * t) {
	return new TraceCompilerImpl(t);
}
