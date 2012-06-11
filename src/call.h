
#ifndef RIPOSTE_CALL_H

// code for making function calls

#include "interpreter.h"
#include "ops.h"

#ifdef USE_THREADED_INTERPRETER
extern const void** glabels;
#endif

void printCode(Thread const& thread, Prototype const* prototype, Environment* env);

void threadByteCode(Prototype* prototype);

Instruction const* buildStackFrame(Thread& thread, Environment* environment, Prototype const* prototype, Instruction const* returnpc, int64_t stackOffset);

Instruction const* buildStackFrame(Thread& thread, Environment* environment, Prototype const* prototype, int64_t resultSlot, Instruction const* returnpc);

Instruction const* buildStackFrame(Thread& thread, Environment* environment, Prototype const* prototype, Environment* env, String s, Instruction const* returnpc);

Instruction const* buildStackFrame(Thread& thread, Environment* environment, Prototype const* prototype, Environment* env, int64_t resultSlot, Instruction const* returnpc);

void MatchArgs(Thread& thread, Environment const* env, Environment* fenv, Function const& func, CompiledCall const& call);

void MatchNamedArgs(Thread& thread, Environment* env, Environment* fenv, Function const& func, CompiledCall const& call);

inline Environment* CreateEnvironment(Thread& thread, Environment* l, Environment* d, Value const& call) {
	Environment* env = new Environment(l, d, call);
	return env;
}

Instruction const* GenericDispatch(Thread& thread, Instruction const& inst, String op, Value const& a, int64_t out);

Instruction const* GenericDispatch(Thread& thread, Instruction const& inst, String op, Value const& a, Value const& b, int64_t out);

Instruction const* forceDot(Thread& thread, Instruction const& inst, Value const* a, Environment* env, int64_t index);

Instruction const* forceReg(Thread& thread, Instruction const& inst, Value const* a, String name);

#define REGISTER(i) (*(thread.base+(i)))

// Out register is currently always a register, not memory
#define OUT(thread, i) (*(thread.base+(i)))

#define OPERAND(a, i) \
Value const& a = __builtin_expect((i) <= 0, true) ? \
		*(thread.base+(i)) : \
		thread.frame.environment->getRecursive((String)(i)); 
	
#define FORCE(a, i) \
if(__builtin_expect((i) > 0 && !a.isConcrete(), false)) { \
	if(a.isDotdot()) { \
		Value const& t = ((Environment*)a.p)->dots[a.length].v; \
		if(t.isConcrete()) { \
			thread.frame.environment->insert((String)(i)) = t; \
			thread.LiveEnvironment(thread.frame.environment, t); \
			return &inst; \
		} \
		else return forceDot(thread, inst, &t, (Environment*)a.p, a.length); \
	} \
	else return forceReg(thread, inst, &a, (String)(i)); \
} \

#define DOTDOT(a, i) \
Value const& a = thread.frame.environment->dots[(i)].v;

#define FORCE_DOTDOT(a, i) \
if(!a.isConcrete()) { \
	if(a.isDotdot()) { \
		Value const& t = ((Environment*)a.p)->dots[a.length].v; \
		if(t.isConcrete()) { \
			thread.frame.environment->dots[(i)].v = t; \
			thread.LiveEnvironment(thread.frame.environment, t); \
			return &inst; \
		} \
		else return forceDot(thread, inst, &t, (Environment*)a.p, a.length); \
	} \
	else return forceDot(thread, inst, &a, thread.frame.environment, (i)); \
}

#define BIND(a) \
if(__builtin_expect(a.isFuture(), false)) { \
	thread.Bind(a); \
	return &inst; \
}

inline bool isTraceableType(Thread const& thread, Value const& a) {
	Type::Enum type = thread.futureType(a);
        return type == Type::Double || type == Type::Integer || type == Type::Logical;
}

inline bool isTraceableShape(Thread const& thread, Value const& a) {
	IRNode::Shape const& shape = thread.futureShape(a);
	return !shape.blocking && shape.length >= TRACE_VECTOR_WIDTH;
}

inline bool isTraceableShape(Thread const& thread, Value const& a, Value const& b) {
	IRNode::Shape const& shapea = thread.futureShape(a);
	IRNode::Shape const& shapeb = thread.futureShape(b);
	return 	!shapea.blocking &&
		!shapeb.blocking &&
		(shapea.length >= TRACE_VECTOR_WIDTH || shapeb.length >=TRACE_VECTOR_WIDTH) &&
		!(a.isFuture() && b.isFuture() && shapea.length != shapeb.length);
}

inline bool isTraceable(Thread const& thread, Value const& a) {
	return 	thread.state.jitEnabled && 
		isTraceableType(thread, a) &&
		isTraceableShape(thread, a);
}

inline bool isTraceable(Thread const& thread, Value const& a, Value const& b) {
	return  thread.state.jitEnabled &&
		isTraceableType(thread, a) && 
		isTraceableType(thread, b) && 
		isTraceableShape(thread, a, b);
}

template< template<class X> class Group>
bool isTraceable(Thread const& thread, Value const& a) { 
	return 	isTraceable(thread, a);
}

template<>
inline bool isTraceable<ArithScan>(Thread const& thread, Value const& a) { return false; }

template<>
inline bool isTraceable<UnifyScan>(Thread const& thread, Value const& a) { return false; }

template< template<class X, class Y> class Group>
bool isTraceable(Thread const& thread, Value const& a, Value const& b) {
	return  isTraceable(thread, a, b);
}

template< template<class X, class Y, class Z> class Group>
bool isTraceable(Thread const& thread, Value const& a, Value const& b, Value const& c) {
	return false;
}

template<>
inline bool isTraceable<IfElse>(Thread const& thread, Value const& a, Value const& b, Value const& c) { 
	return	thread.state.jitEnabled &&
		isTraceableType(thread, a) &&
		isTraceableType(thread, b) &&
		isTraceableType(thread, c) &&
		isTraceableShape(thread, a, c) &&
		isTraceableShape(thread, b, c);
}

#endif
