
#ifndef RIPOSTE_CALL_H

// code for making function calls

#include "interpreter.h"
#include "ops.h"
#include "vector.h"
#include "exceptions.h"

void printCode(Thread const& thread, Prototype const* prototype, Environment* env);

Instruction const* buildStackFrame(Thread& thread, Environment* environment, Prototype const* prototype, Instruction const* returnpc, int64_t stackOffset);

Instruction const* buildStackFrame(Thread& thread, Environment* environment, Prototype const* prototype, int64_t resultSlot, Instruction const* returnpc);

Instruction const* buildStackFrame(Thread& thread, Environment* environment, Prototype const* prototype, Environment* env, String s, Instruction const* returnpc);

Instruction const* buildStackFrame(Thread& thread, Environment* environment, Prototype const* prototype, Environment* env, int64_t resultSlot, Instruction const* returnpc);

void MatchArgs(Thread& thread, Environment* env, Environment* fenv, Function const& func, CompiledCall const& call);

void FastMatchArgs(Thread& thread, Environment* env, Environment* fenv, Function const& func, CompiledCall const& call);

Instruction const* GenericDispatch(Thread& thread, Instruction const& inst, String op, Value const& a, int64_t out);

Instruction const* GenericDispatch(Thread& thread, Instruction const& inst, String op, Value const& a, Value const& b, int64_t out);

#define REGISTER(i) (*(thread.frame.registers+(-(i))))
#define CONSTANT(i) (thread.frame.prototype->constants[(i)-1])

// Out register is currently always a register, not memory
#define OUT(X) \
	(inst.X <= 0 \
		? *(thread.frame.registers+(-inst.X)) \
		: thread.frame.environment->insert((String)(inst.X)))

#define DECODE(X) \
Environment* X##Env = 0; \
Value const& X = \
	__builtin_expect((inst.X) <= 0, true) \
		? *(thread.frame.registers+(-inst.X)) \
		: ((inst.X) < 256) \
			? thread.frame.prototype->constants[(inst.X)-1] \
			: thread.frame.environment->getRecursive((String)(inst.X), X##Env); 

#define FORCE(X) \
if(__builtin_expect((inst.X) > 0 && !X.isObject(), false)) { \
	return force(thread, inst, X, X##Env, (String)(inst.X)); \
}


#define DOTDOT(X, i) \
Environment* X##Env = thread.frame.environment; \
Value const& X = \
	(thread.frame.environment->dots[(i)].v);
	
#define FORCE_DOTDOT(X, i) \
if(__builtin_expect(!X.isObject(), false)) { \
	return forceDot(thread, inst, X, X##Env, (i)); \
}


#define BIND(X) \
if(__builtin_expect(X.isFuture(), false)) { \
	thread.traces.Bind(thread,X); \
	return &inst; \
}

template< template<typename T> class Op > 
void ArithUnary1Dispatch(Thread& thread, Value a, Value& c) {
	if(a.isDouble())	Zip1< Op<Double> >::eval(thread, (Double const&)a, c);
	else if(a.isInteger())	Zip1< Op<Integer> >::eval(thread, (Integer const&)a, c);
	else if(a.isLogical())	Zip1< Op<Logical> >::eval(thread, (Logical const&)a, c);
	else if(a.isNull())	c = Null::Singleton();
	else _error("non-numeric argument to unary numeric operator");
}

template< template<typename T> class Op > 
void ArithUnary2Dispatch(Thread& thread, Value a, Value& c) {
	ArithUnary1Dispatch<Op>(thread, a, c);
}

template< template<typename T> class Op > 
void LogicalUnaryDispatch(Thread& thread, Value a, Value& c) {
	if(a.isLogical())	Zip1< Op<Logical> >::eval(thread, (Logical const&)a, c);
	else if(a.isDouble())	Zip1< Op<Double> >::eval(thread, (Double const&)a, c);
	else if(a.isInteger())	Zip1< Op<Integer> >::eval(thread, (Integer const&)a, c);
	else if(a.isNull())	c = Logical(0);
	else _error("non-logical argument to unary logical operator");
};

template< template<typename T> class Op > 
void OrdinalUnaryDispatch(Thread& thread, Value a, Value& c) {
	if(a.isDouble())	Zip1< Op<Double> >::eval(thread, (Double const&)a, c);
	else if(a.isInteger())	Zip1< Op<Integer> >::eval(thread, (Integer const&)a, c);
	else if(a.isLogical())	Zip1< Op<Logical> >::eval(thread, (Logical const&)a, c);
	else if(a.isCharacter())Zip1< Op<Character> >::eval(thread, (Character const&)a, c);
	else c = Logical::False();
}

template< template<typename S, typename T> class Op > 
void ArithBinary1Dispatch(Thread& thread, Value a, Value b, Value& c) {
	if(a.isDouble()) {
		if(b.isDouble()) 	Zip2< Op<Double,Double> >::eval(thread, (Double const&)a, (Double const&)b, c);
		else if(b.isInteger()) 	Zip2< Op<Double,Integer> >::eval(thread, (Double const&)a, (Integer const&)b, c);
		else if(b.isLogical()) 	Zip2< Op<Double,Logical> >::eval(thread, (Double const&)a, (Logical const&)b, c);
		else if(b.isNull())	c = Double(0);
		else _error("non-numeric argument to binary numeric operator");
	} else if(a.isInteger()) {
		if(b.isDouble()) 	Zip2< Op<Integer,Double> >::eval(thread, (Integer const&)a, (Double const&)b, c);
		else if(b.isInteger()) 	Zip2< Op<Integer,Integer> >::eval(thread, (Integer const&)a, (Integer const&)b, c);
		else if(b.isLogical()) 	Zip2< Op<Integer,Logical> >::eval(thread, (Integer const&)a, (Logical const&)b, c);
		else if(b.isNull())	c = Double(0);
		else _error("non-numeric argument to binary numeric operator");
	} else if(a.isLogical()) {
		if(b.isDouble()) 	Zip2< Op<Logical,Double> >::eval(thread, (Logical const&)a, (Double const&)b, c);
		else if(b.isInteger()) 	Zip2< Op<Logical,Integer> >::eval(thread, (Logical const&)a, (Integer const&)b, c);
		else if(b.isLogical()) 	Zip2< Op<Logical,Logical> >::eval(thread, (Logical const&)a, (Logical const&)b, c);
		else if(b.isNull())	c = Double(0);
		else _error("non-numeric argument to binary numeric operator");
	} else if(a.isNull()) {
		if(b.isDouble() || b.isInteger() || b.isLogical()) c = Double(0);
		else _error("non-numeric argument to binary numeric operator");
	} 
	else	_error("non-numeric argument to binary numeric operator");
}

template< template<typename S, typename T> class Op > 
void ArithBinary2Dispatch(Thread& thread, Value a, Value b, Value& c) {
	ArithBinary1Dispatch<Op>(thread, a, b, c);
}

template< template<typename S, typename T> class Op >
void LogicalBinaryDispatch(Thread& thread, Value a, Value b, Value& c) {
	if(a.isLogical()) {
		if(b.isLogical()) 	Zip2< Op<Logical,Logical> >::eval(thread, (Logical const&)a, (Logical const&)b, c);
		else if(b.isDouble()) 	Zip2< Op<Logical,Double> >::eval(thread, (Logical const&)a, (Double const&)b, c);
		else if(b.isInteger()) 	Zip2< Op<Logical,Integer> >::eval(thread, (Logical const&)a, (Integer const&)b, c);
		else if(b.isNull())	c = Logical(0);
		else _error("non-logical argument to binary logical operator");
	} else if(a.isDouble()) {
		if(b.isLogical()) 	Zip2< Op<Double,Logical> >::eval(thread, (Double const&)a, (Logical const&)b, c);
		else if(b.isDouble()) 	Zip2< Op<Double,Double> >::eval(thread, (Double const&)a, (Double const&)b, c);
		else if(b.isInteger()) 	Zip2< Op<Double,Integer> >::eval(thread, (Double const&)a, (Integer const&)b, c);
		else if(b.isNull())	c = Logical(0);
		else _error("non-logical argument to binary logical operator");
	} else if(a.isInteger()) {
		if(b.isLogical()) 	Zip2< Op<Integer,Logical> >::eval(thread, (Integer const&)a, (Logical const&)b, c);
		else if(b.isDouble()) 	Zip2< Op<Integer,Double> >::eval(thread, (Integer const&)a, (Double const&)b, c);
		else if(b.isInteger()) 	Zip2< Op<Integer,Integer> >::eval(thread, (Integer const&)a, (Integer const&)b, c);
		else if(b.isNull())	c = Logical(0);
		else _error("non-logical argument to binary logical operator");
	} else if(a.isNull()) {
		if(b.isDouble() || b.isInteger() || b.isLogical() || b.isNull()) c = Logical(0);
		else _error("non-logical argument to binary logical operator");
	} 
	else _error("non-logical argument to binary logical operator");
}

template< template<typename S, typename T> class Op > 
void UnifyBinaryDispatch(Thread& thread, Value const& a, Value const& b, Value& c) {
	if(!a.isVector() || !b.isVector())
		_error("comparison is possible only for atomic and list types");
	if(a.isCharacter() || b.isCharacter())
		Zip2< Op<Character, Character> >::eval(thread, As<Character>(thread, a), As<Character>(thread, b), c);
	else if(a.isDouble() || b.isDouble())
		Zip2< Op<Double, Double> >::eval(thread, As<Double>(thread, a), As<Double>(thread, b), c);
	else if(a.isInteger() || b.isInteger())	
		Zip2< Op<Integer, Integer> >::eval(thread, As<Integer>(thread, a), As<Integer>(thread, b), c);
	else if(a.isLogical() || b.isLogical())	
		Zip2< Op<Logical, Logical> >::eval(thread, As<Logical>(thread, a), As<Logical>(thread, b), c);
	else if(a.isNull() || b.isNull())
		c = Null::Singleton();
	else	_error("non-ordinal argument to ordinal operator");
}

template< template<typename S, typename T> class Op > 
void OrdinalBinaryDispatch(Thread& thread, Value const& a, Value const& b, Value& c) {
	UnifyBinaryDispatch<Op>(thread, a, b, c);
}

template< template<typename S, typename T> class Op > 
void RoundBinaryDispatch(Thread& thread, Value a, Value b, Value& c) {
	ArithBinary1Dispatch<Op>(thread, a, b, c);
}

template< template<typename T> class Op >
void ArithFoldDispatch(Thread& thread, Value const& a, Value& c) {
	if(a.isDouble())	FoldLeft< Op<Double> >::eval(thread, (Double const&)a, c);
	else if(a.isInteger())	FoldLeft< Op<Integer> >::eval(thread, (Integer const&)a, c);
	else if(a.isLogical())	FoldLeft< Op<Logical> >::eval(thread, (Logical const&)a, c);
	else if(a.isNull())	Op<Double>::Scalar(thread, Op<Double>::base(), c);
	else _error("non-numeric argument to numeric fold operator");
}

template< template<typename T> class Op >
void LogicalFoldDispatch(Thread& thread, Value const& a, Value& c) {
	if(a.isLogical())	FoldLeft< Op<Logical> >::eval(thread, (Logical const&)a, c);
	else if(a.isDouble())	FoldLeft< Op<Double> >::eval(thread, (Double const&)a, c);
	else if(a.isInteger())	FoldLeft< Op<Integer> >::eval(thread, (Integer const&)a, c);
	else if(a.isNull())	Op<Logical>::Scalar(thread, Op<Logical>::base(), c);
	else _error("non-logical argument to logical fold operator");
}

template< template<typename T> class Op >
void UnifyFoldDispatch(Thread& thread, Value const& a, Value& c) {
	if(a.isCharacter())	FoldLeft< Op<Character> >::eval(thread, (Character const&)a, c);
	else if(a.isDouble())	FoldLeft< Op<Double> >::eval(thread, (Double const&)a, c);
	else if(a.isInteger())	FoldLeft< Op<Integer> >::eval(thread, (Integer const&)a, c);
	else if(a.isLogical())	FoldLeft< Op<Logical> >::eval(thread, (Logical const&)a, c);
	else if(a.isNull())	c = Null::Singleton();
	else _error("non-numeric argument to numeric fold operator");
}

template< template<typename T> class Op >
void ArithScanDispatch(Thread& thread, Value const& a, Value& c) {
	if(a.isDouble())	ScanLeft< Op<Double> >::eval(thread, (Double const&)a, c);
	else if(a.isInteger())	ScanLeft< Op<Integer> >::eval(thread, (Integer const&)a, c);
	else if(a.isLogical())	ScanLeft< Op<Logical> >::eval(thread, (Logical const&)a, c);
	else if(a.isNull())	Op<Double>::Scalar(thread, Op<Double>::base(), c);
	else _error("non-numeric argument to numeric scan operator");
}

template< template<typename T> class Op >
void UnifyScanDispatch(Thread& thread, Value const& a, Value& c) {
	if(a.isCharacter())	ScanLeft< Op<Character> >::eval(thread, (Character const&)a, c);
	else if(a.isDouble())	ScanLeft< Op<Double> >::eval(thread, (Double const&)a, c);
	else if(a.isInteger())	ScanLeft< Op<Integer> >::eval(thread, (Integer const&)a, c);
	else if(a.isLogical())	ScanLeft< Op<Logical> >::eval(thread, (Logical const&)a, c);
	else if(a.isNull())	Op<Double>::Scalar(thread, Op<Double>::base(), c);
	else _error("non-numeric argument to numeric scan operator");
}


#endif
