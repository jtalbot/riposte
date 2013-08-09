
#ifndef RIPOSTE_CALL_H

// code for making function calls

#include "interpreter.h"
#include "ops.h"
#include "vector.h"
#include "exceptions.h"
#include "runtime.h"


#define REGISTER(i) (*(thread.frame.registers+(i)))
#define CONSTANT(i) (thread.frame.code->constants[-1-(i)])
#define DOTS(i) (((List const&)thread.frame.environment->get(Strings::__dots__))[(i)])

// Out register is currently always a register, not memory
#define OUT(X) (*(thread.frame.registers+(inst.X)))

#define DECODE(X) \
Value const& X = \
	__builtin_expect((inst.X) >= 0, true) \
		? *(thread.frame.registers+(inst.X)) \
	    : thread.frame.code->constants[-1-(inst.X)];

#define BIND(X) \
if(__builtin_expect(X.isFuture(), false)) { \
	thread.traces.Bind(thread,X); \
	return &inst; \
}

Instruction const* force(
    Thread& thread, Promise const& p,
    Environment* targetEnv, Value targetIndex,
    int64_t outRegister, Instruction const* returnpc);

Instruction const* buildStackFrame(Thread& thread, 
    Environment* environment, Code const* code, 
    int64_t outRegister, Instruction const* returnpc);

Environment* MatchArgs(Thread& thread, Environment* env, Closure const& func, CompiledCall const& call);

Environment* FastMatchArgs(Thread& thread, Environment* env, Closure const& func, CompiledCall const& call);

Instruction const* GenericDispatch(Thread& thread, Instruction const& inst, String op, Value const& a, int64_t out);

Instruction const* GenericDispatch(Thread& thread, Instruction const& inst, String op, Value const& a, Value const& b, int64_t out);

Instruction const* GenericDispatch(Thread& thread, Instruction const& inst, String op, Value const& a, Value const& b, Value const& c, int64_t out);

Instruction const* StopDispatch(Thread& thread, Instruction const& inst, String msg, int64_t out);


template< template<typename T> class Op >
bool ArithUnary1Fast(Thread& thread, void* args, Value const& a, Value& c) {
    if     (a.isDouble1())  { Op<Double>::Scalar(thread, args, a.d, c);   return true; }
    else if(a.isInteger1()) { Op<Integer>::Scalar(thread, args, a.i, c);  return true; }
    else return false;
}

template< template<typename T> class Op > 
bool ArithUnary1Dispatch(Thread& thread, void* args, Value const& a, Value& c) {
	if(a.isDouble()) { Zip1< Op<Double> >::eval(thread, args, (Double const&)a, c); return true; }
	else if(a.isInteger()) { Zip1< Op<Integer> >::eval(thread, args, (Integer const&)a, c); return true; }
	else if(a.isLogical()) { Zip1< Op<Logical> >::eval(thread, args, (Logical const&)a, c); return true; }
	else if(a.isNull())	{ c = Null::Singleton(); return true; }
	else return false;
}

template< template<typename T> class Op >
bool ArithUnary2Fast(Thread& thread, void* args, Value const& a, Value& c) {
    if (a.isDouble1())  { Op<Double>::Scalar(thread, args, a.d, c);   return true; }
    else return false;
}

template< template<typename T> class Op > 
bool ArithUnary2Dispatch(Thread& thread, void* args, Value a, Value& c) {
	return ArithUnary1Dispatch<Op>(thread, args, a, c);
}

template< template<typename T> class Op >
bool LogicalUnaryFast(Thread& thread, void* args, Value a, Value& c) {
    if     (a.isLogical1()) { Op<Logical>::Scalar(thread, args, a.c, c);  return true; }
    else if(a.isDouble1())  { Op<Double>::Scalar(thread, args, a.d, c);   return true; }
    else if(a.isInteger1()) { Op<Integer>::Scalar(thread, args, a.i, c);  return true; }
    else return false;
}

template< template<typename T> class Op > 
bool LogicalUnaryDispatch(Thread& thread, void* args, Value a, Value& c) {
	if(a.isLogical())	{ Zip1< Op<Logical> >::eval(thread, args, (Logical const&)a, c); return true; }
	else if(a.isDouble()) { Zip1< Op<Double> >::eval(thread, args, (Double const&)a, c); return true; }
	else if(a.isInteger()) { Zip1< Op<Integer> >::eval(thread, args, (Integer const&)a, c); return true; }
	else if(a.isNull())	{ c = Logical(0); return true; }
	else return false;
};

template< template<typename T> class Op >
bool OrdinalUnaryFast(Thread& thread, void* args, Value a, Value& c) {
    if(a.isDouble1())         { Op<Double>::Scalar(thread, args, a.d, c);   return true; }
    else if(a.isInteger1())   { Op<Integer>::Scalar(thread, args, a.i, c);  return true; }
    else if(a.isLogical1())   { Op<Logical>::Scalar(thread, args, a.c, c);  return true; }
    else if(a.isCharacter1()) { Op<Character>::Scalar(thread, args, a.s, c);return true; }
    else return false;
}

template< template<typename T> class Op > 
bool OrdinalUnaryDispatch(Thread& thread, void* args, Value a, Value& c) {
	if(a.isDouble())	{ Zip1< Op<Double> >::eval(thread, args, (Double const&)a, c); return true; }
	else if(a.isInteger())	{ Zip1< Op<Integer> >::eval(thread, args, (Integer const&)a, c); return true; }
	else if(a.isLogical())	{ Zip1< Op<Logical> >::eval(thread, args, (Logical const&)a, c); return true; }
	else if(a.isCharacter()) { Zip1< Op<Character> >::eval(thread, args, (Character const&)a, c); return true; }
	else { c = Logical::False(); return true; }
}

template< template<typename S, typename T> class Op > 
bool ArithBinary1Fast(Thread& thread, void* args, Value a, Value b, Value& c) {
    if(a.isDouble1() && b.isDouble1())
        { Op<Double, Double>::Scalar(thread, args, a.d, b.d, c); return true; }
    if(a.isInteger1() && b.isInteger1())
        { Op<Integer, Integer>::Scalar(thread, args, a.i, b.i, c); return true; }
	else return false;
}

template< template<typename S, typename T> class Op > 
bool ArithBinary1Dispatch(Thread& thread, void* args, Value a, Value b, Value& c) {
	if(a.isDouble()) {
		if(b.isDouble()) {	Zip2< Op<Double,Double> >::eval(thread, args, (Double const&)a, (Double const&)b, c); return true; }
		else if(b.isInteger()) {	Zip2< Op<Double,Integer> >::eval(thread, args, (Double const&)a, (Integer const&)b, c); return true; }
		else if(b.isLogical()) {	Zip2< Op<Double,Logical> >::eval(thread, args, (Double const&)a, (Logical const&)b, c); return true; }
		else if(b.isNull())	{ c = Double(0); return true; }
		else return false;
	} else if(a.isInteger()) {
		if(b.isDouble()) {	Zip2< Op<Integer,Double> >::eval(thread, args, (Integer const&)a, (Double const&)b, c); return true; }
		else if(b.isInteger()) {	Zip2< Op<Integer,Integer> >::eval(thread, args, (Integer const&)a, (Integer const&)b, c); return true; }
		else if(b.isLogical()) {	Zip2< Op<Integer,Logical> >::eval(thread, args, (Integer const&)a, (Logical const&)b, c); return true; }
		else if(b.isNull())	{ c = Double(0); return true; }
		else return false;
	} else if(a.isLogical()) {
		if(b.isDouble()) {	Zip2< Op<Logical,Double> >::eval(thread, args, (Logical const&)a, (Double const&)b, c); return true; }
		else if(b.isInteger()) { 	Zip2< Op<Logical,Integer> >::eval(thread, args, (Logical const&)a, (Integer const&)b, c); return true; }
		else if(b.isLogical()) {	Zip2< Op<Logical,Logical> >::eval(thread, args, (Logical const&)a, (Logical const&)b, c); return true; }
		else if(b.isNull())	{ c = Double(0); return true; }
		else return false;
	} else if(a.isNull()) {
		if(b.isDouble() || b.isInteger() || b.isLogical()) { c = Double(0); return true; }
		else return false;
	} 
	else return false;
}

template< template<typename S, typename T> class Op > 
bool ArithBinary2Fast(Thread& thread, void* args, Value a, Value b, Value& c) {
    if(a.isDouble1() && b.isDouble1())
        { Op<Double, Double>::Scalar(thread, args, a.d, b.d, c); return true; }
	else return false;
}

template< template<typename S, typename T> class Op > 
bool ArithBinary2Dispatch(Thread& thread, void* args, Value a, Value b, Value& c) {
	return ArithBinary1Dispatch<Op>(thread, args, a, b, c);
}

template< template<typename S, typename T> class Op > 
bool LogicalBinaryFast(Thread& thread, void* args, Value a, Value b, Value& c) {
    if(a.isLogical1() && b.isLogical1())
        { Op<Logical, Logical>::Scalar(thread, args, a.c, b.c, c); return true; }
	else return false;
}

template< template<typename S, typename T> class Op >
bool LogicalBinaryDispatch(Thread& thread, void* args, Value a, Value b, Value& c) {
	if(a.isLogical()) {
		if(b.isLogical()) { 	Zip2< Op<Logical,Logical> >::eval(thread, args, (Logical const&)a, (Logical const&)b, c); return true; }
		else if(b.isDouble()) { 	Zip2< Op<Logical,Double> >::eval(thread, args, (Logical const&)a, (Double const&)b, c); return true; }
		else if(b.isInteger()) { 	Zip2< Op<Logical,Integer> >::eval(thread, args, (Logical const&)a, (Integer const&)b, c); return true; }
		else if(b.isNull())	{ c = Logical(0); return true; }
		else return false;
	} else if(a.isDouble()) {
		if(b.isLogical()) {	Zip2< Op<Double,Logical> >::eval(thread, args, (Double const&)a, (Logical const&)b, c); return true; }
		else if(b.isDouble()) {	Zip2< Op<Double,Double> >::eval(thread, args, (Double const&)a, (Double const&)b, c); return true; }
		else if(b.isInteger()) {	Zip2< Op<Double,Integer> >::eval(thread, args, (Double const&)a, (Integer const&)b, c); return true; }
		else if(b.isNull())	{ c = Logical(0); return true; }
		else return false;
	} else if(a.isInteger()) {
		if(b.isLogical()) {	Zip2< Op<Integer,Logical> >::eval(thread, args, (Integer const&)a, (Logical const&)b, c); return true; }
		else if(b.isDouble()) {	Zip2< Op<Integer,Double> >::eval(thread, args, (Integer const&)a, (Double const&)b, c); return true; }
		else if(b.isInteger()) { Zip2< Op<Integer,Integer> >::eval(thread, args, (Integer const&)a, (Integer const&)b, c); return true; }
		else if(b.isNull())	{ c = Logical(0); return true; }
		else return false; 
	} else if(a.isNull()) {
		if(b.isDouble() || b.isInteger() || b.isLogical() || b.isNull()) { c = Logical(0); return true; }
		else return false;
	} 
	else return false;
}

template< class Op >
bool EnvironmentBinaryFast(Thread& thread, void* args, Value const& a, Value const& b, Value& c) {
    return false;
}

template< class Op >
bool EnvironmentBinaryDispatch(Thread& thread, void* args, Value const& a, Value const& b, Value& c) {
	return false;
}

template<>
bool EnvironmentBinaryDispatch< struct eqVOp<REnvironment, REnvironment> >
(Thread& thread, void* args, Value const& a, Value const& b, Value& c);

template<>
bool EnvironmentBinaryDispatch< struct neqVOp<REnvironment, REnvironment> >
(Thread& thread, void* args, Value const& a, Value const& b, Value& c);

template< template<typename S, typename T> class Op > 
bool UnifyBinaryFast(Thread& thread, void* args, Value a, Value b, Value& c) {
    if(a.isDouble1() && b.isDouble1())
        { Op<Double, Double>::Scalar(thread, args, a.d, b.d, c); return true; }
    if(a.isInteger1() && b.isInteger1())
        { Op<Integer, Integer>::Scalar(thread, args, a.i, b.i, c); return true; }
    if(a.isLogical1() && b.isLogical1())
        { Op<Logical, Logical>::Scalar(thread, args, a.c, b.c, c); return true; }
    if(a.isCharacter1() && b.isCharacter1())
        { Op<Character, Character>::Scalar(thread, args, a.s, b.s, c); return true; }
	else return false;
}

template< template<typename S, typename T> class Op > 
bool UnifyBinaryDispatch(Thread& thread, void* args, Value const& a, Value const& b, Value& c) {
	if(a.isVector() && b.isVector())
	{
        if(a.isCharacter() || b.isCharacter()) {
		    Zip2< Op<Character, Character> >::eval(thread, args, As<Character>(thread, a), As<Character>(thread, b), c);
            return true;
        }
	    else if(a.isDouble() || b.isDouble()) {
		    Zip2< Op<Double, Double> >::eval(thread, args, As<Double>(thread, a), As<Double>(thread, b), c);
            return true;
        }
	    else if(a.isInteger() || b.isInteger()) {	
		    Zip2< Op<Integer, Integer> >::eval(thread, args, As<Integer>(thread, a), As<Integer>(thread, b), c);
            return true;
        }
	    else if(a.isLogical() || b.isLogical())	{
		    Zip2< Op<Logical, Logical> >::eval(thread, args, As<Logical>(thread, a), As<Logical>(thread, b), c);
            return true;
        }
	    else if(a.isNull() || b.isNull()) {
		    c = Null::Singleton();
            return true;
        }
        else
            return false;
    }
    else if(a.isEnvironment() && b.isEnvironment()) {
        return EnvironmentBinaryDispatch< Op<REnvironment, REnvironment> >(thread, args, a, b, c);
    }
    else {
	    return false; 
    }
}

void IfElseDispatch(Thread& thread, void* args, Value const& a, Value const& b, Value const& cond, Value& c);

template< template<typename S, typename T> class Op > 
bool OrdinalBinaryFast(Thread& thread, void* args, Value const& a, Value const& b, Value& c) {
	return UnifyBinaryFast<Op>(thread, args, a, b, c);
}

template< template<typename S, typename T> class Op > 
bool OrdinalBinaryDispatch(Thread& thread, void* args, Value const& a, Value const& b, Value& c) {
	return UnifyBinaryDispatch<Op>(thread, args, a, b, c);
}

template< template<typename S, typename T> class Op > 
bool RoundBinaryFast(Thread& thread, void* args, Value a, Value b, Value& c) {
	return ArithBinary2Fast<Op>(thread, args, a, b, c);
}

template< template<typename S, typename T> class Op > 
bool RoundBinaryDispatch(Thread& thread, void* args, Value a, Value b, Value& c) {
	return ArithBinary2Dispatch<Op>(thread, args, a, b, c);
}

template< template<typename T> class Op >
bool ArithFold1Fast(Thread& thread, void* args, Value a, Value& c) {
    if(a.isDouble1())         { Op<Double>::Scalar(thread, args, a.d, c);   return true; }
    else if(a.isInteger1())   { Op<Integer>::Scalar(thread, args, a.i, c);  return true; }
    else if(a.isLogical1())   { Op<Logical>::Scalar(thread, args, a.c, c);  return true; }
    else return false;
}

template< template<typename T> class Op >
bool ArithFold1Dispatch(Thread& thread, void* args, Value const& a, Value& c) {
	if(a.isDouble())	{ FoldLeft< Op<Double> >::eval(thread, args, (Double const&)a, c); return true; }
	else if(a.isInteger())	{ FoldLeft< Op<Integer> >::eval(thread, args, (Integer const&)a, c); return true; }
	else if(a.isLogical())	{ FoldLeft< Op<Logical> >::eval(thread, args, (Logical const&)a, c); return true; }
	else if(a.isNull())	{ Op<Double>::Scalar(thread, args, Op<Double>::base(), c); return true; }
	else return false; 
}

template< template<typename T> class Op >
bool ArithFold2Fast(Thread& thread, void* args, Value a, Value& c) {
    if(a.isDouble1())         { Op<Double>::Scalar(thread, args, a.d, c);   return true; }
    else if(a.isInteger1())   { Op<Integer>::Scalar(thread, args, a.i, c);  return true; }
    else if(a.isLogical1())   { Op<Logical>::Scalar(thread, args, a.c, c);  return true; }
    else return false;
}

template< template<typename T> class Op >
bool ArithFold2Dispatch(Thread& thread, void* args, Value const& a, Value& c) {
	if(a.isDouble())	{ FoldLeft< Op<Double> >::eval(thread, args, (Double const&)a, c); return true; }
	else if(a.isInteger())	{ FoldLeft< Op<Integer> >::eval(thread, args, (Integer const&)a, c); return true; }
	else if(a.isLogical())	{ FoldLeft< Op<Logical> >::eval(thread, args, (Logical const&)a, c); return true; }
	else if(a.isNull())	{ Op<Double>::Scalar(thread, args, Op<Double>::base(), c); return true; }
	else return false;
}

template< template<typename T> class Op >
bool LogicalFoldFast(Thread& thread, void* args, Value a, Value& c) {
    if(a.isLogical1())        { Op<Logical>::Scalar(thread, args, a.c, c);  return true; }
    else if(a.isDouble1())    { Op<Double>::Scalar(thread, args, a.d, c);   return true; }
    else if(a.isInteger1())   { Op<Integer>::Scalar(thread, args, a.i, c);  return true; }
    else return false;
}

template< template<typename T> class Op >
bool LogicalFoldDispatch(Thread& thread, void* args, Value const& a, Value& c) {
	if(a.isLogical())	{ FoldLeft< Op<Logical> >::eval(thread, args, (Logical const&)a, c); return true; }
	else if(a.isDouble())	{ FoldLeft< Op<Double> >::eval(thread, args, (Double const&)a, c); return true; }
	else if(a.isInteger())	{ FoldLeft< Op<Integer> >::eval(thread, args, (Integer const&)a, c); return true; }
	else if(a.isNull())	{ Op<Logical>::Scalar(thread, args, Op<Logical>::base(), c); return true; }
	else return false;
}

template< template<typename T> class Op >
bool UnifyFoldFast(Thread& thread, void* args, Value a, Value& c) {
    if(a.isCharacter1())      { Op<Character>::Scalar(thread, args, a.s, c);   return true; }
    else if(a.isDouble1())    { Op<Double>::Scalar(thread, args, a.d, c);   return true; }
    else if(a.isLogical1())   { Op<Logical>::Scalar(thread, args, a.c, c);  return true; }
    else if(a.isInteger1())   { Op<Integer>::Scalar(thread, args, a.i, c);  return true; }
    else return false;
}

template< template<typename T> class Op >
bool UnifyFoldDispatch(Thread& thread, void* args, Value const& a, Value& c) {
	if(a.isCharacter())	{ FoldLeft< Op<Character> >::eval(thread, args, (Character const&)a, c); return true; }
	else if(a.isDouble())	{ FoldLeft< Op<Double> >::eval(thread, args, (Double const&)a, c); return true; }
	else if(a.isInteger())	{ FoldLeft< Op<Integer> >::eval(thread, args, (Integer const&)a, c); return true; }
	else if(a.isLogical())	{ FoldLeft< Op<Logical> >::eval(thread, args, (Logical const&)a, c); return true; }
	else if(a.isNull())	{ c = Null::Singleton(); return true; }
	else return false; 
}

template< template<typename T> class Op >
bool ArithScanFast(Thread& thread, void* args, Value a, Value& c) {
    if     (a.isDouble1())  { Op<Double>::Scalar(thread, args, a.d, c);   return true; }
    else if(a.isInteger1()) { Op<Integer>::Scalar(thread, args, a.i, c);  return true; }
    else if(a.isLogical1()) { Op<Logical>::Scalar(thread, args, a.c, c);  return true; }
    else return false;
}

template< template<typename T> class Op >
bool ArithScanDispatch(Thread& thread, void* args, Value const& a, Value& c) {
	if(a.isDouble())	{ ScanLeft< Op<Double> >::eval(thread, args, (Double const&)a, c); return true; }
	else if(a.isInteger())	{ ScanLeft< Op<Integer> >::eval(thread, args, (Integer const&)a, c); return true; }
	else if(a.isLogical())	{ ScanLeft< Op<Logical> >::eval(thread, args, (Logical const&)a, c); return true; }
	else if(a.isNull())	{ Op<Double>::Scalar(thread, args, Op<Double>::base(), c); return true; }
	else return false;
}

template< template<typename T> class Op >
bool UnifyScanFast(Thread& thread, void* args, Value a, Value& c) {
    if(a.isCharacter1())      { Op<Character>::Scalar(thread, args, a.s, c);   return true; }
    else if(a.isDouble1())    { Op<Double>::Scalar(thread, args, a.d, c);   return true; }
    else if(a.isLogical1())   { Op<Logical>::Scalar(thread, args, a.c, c);  return true; }
    else if(a.isInteger1())   { Op<Integer>::Scalar(thread, args, a.i, c);  return true; }
    else return false;
}

template< template<typename T> class Op >
bool UnifyScanDispatch(Thread& thread, void* args, Value const& a, Value& c) {
	if(a.isCharacter())	{ ScanLeft< Op<Character> >::eval(thread, args, (Character const&)a, c); return true; }
	else if(a.isDouble())	{ ScanLeft< Op<Double> >::eval(thread, args, (Double const&)a, c); return true; }
	else if(a.isInteger())	{ ScanLeft< Op<Integer> >::eval(thread, args, (Integer const&)a, c); return true; }
	else if(a.isLogical())	{ ScanLeft< Op<Logical> >::eval(thread, args, (Logical const&)a, c); return true; }
	else if(a.isNull())	{ Op<Double>::Scalar(thread, args, Op<Double>::base(), c); return true; }
	else return false;
}

#define SLOW_DISPATCH_DEFN(Name, String, Group, Func) \
Instruction const* Name##Slow(Thread& thread, Instruction const& inst, void* args, Value const& a, Value& c);
UNARY_FOLD_SCAN_BYTECODES(SLOW_DISPATCH_DEFN)
#undef SLOW_DISPATCH_DEFN

#define SLOW_DISPATCH_DEFN(Name, String, Group, Func) \
Instruction const* Name##Slow(Thread& thread, Instruction const& inst, void* args, Value const& a, Value const& b, Value& c);
BINARY_BYTECODES(SLOW_DISPATCH_DEFN)
#undef SLOW_DISPATCH_DEFN

inline bool GetFast(Thread& thread, Value a, Value b, Value& c) {
	if(((Object const&)a).hasAttributes())
        return false;

	else if(a.isVector()) {
        Vector const& v = (Vector const&)a;
		if(    b.isInteger() 
            && ((Integer const&)b).length() == 1
            && !Integer::isNA(b.i)
            && (b.i-1) >= 0 
            && (b.i-1) < v.length() ) {
            Element2(v, b.i-1, c);
            return true;
        }
		else if(b.isDouble() 
            && ((Double const&)b).length() == 1
            && !Double::isNA(b.d)
            && (b.d-1) >= 0
            && (b.d-1) < v.length()) {
            Element2(a, b.d-1, c);
            return true;
        }
	}

    else if(a.isEnvironment() ) {
         if( b.isCharacter()
         && ((Character const&)b).length() == 1) {
	        String s = ((Character const&)b).s;
            Value const& v = ((REnvironment&)a).environment()->get(s);
            if(v.isObject()) {
                c = v;
                return true;
            }
        }
    }

    return false;
}

Instruction const* GetSlow(Thread& thread, Instruction const& inst, Value const& a, Value const& b, Value& c);

#endif
