
#ifndef RIPOSTE_CALL_H

// code for making function calls

#include "interpreter.h"
#include "ops.h"
#include "vector.h"
#include "exceptions.h"
#include "runtime.h"


#define REGISTER(i) (*(state.frame.registers+(i)))
#define CONSTANT(i) (state.frame.code->constants[-1-(i)])

// Out register is currently always a register, not memory
#define OUT(X) (*(state.frame.registers+(inst.X)))

#define DECODE(X) \
Value const& X = \
	__builtin_expect((inst.X) >= 0, true) \
		? *(state.frame.registers+(inst.X)) \
	    : state.frame.code->constants[-1-(inst.X)];

#ifdef EPEE
#define BIND(X) \
if(__builtin_expect(X.isFuture(), false)) { \
	state.traces.Bind(state,X); \
	return &inst; \
}
#else
#define BIND(X)
#endif

Instruction const* force(
    State& state, Promise const& p,
    Environment* targetEnv, Value targetIndex,
    int64_t outRegister, Instruction const* returnpc);

Instruction const* buildStackFrame(State& state, 
    Environment* environment, Code const* code, 
    int64_t outRegister, Instruction const* returnpc);

Environment* MatchArgs(State& state, Environment* env, Closure const& func, CompiledCall const& call);

Environment* FastMatchArgs(State& state, Environment* env, Closure const& func, CompiledCall const& call);

Instruction const* GenericDispatch(State& state, Instruction const& inst, String op, Value const& a, int64_t out);

Instruction const* GenericDispatch(State& state, Instruction const& inst, String op, Value const& a, Value const& b, int64_t out);

Instruction const* GenericDispatch(State& state, Instruction const& inst, String op, Value const& a, Value const& b, Value const& c, int64_t out);

Instruction const* StopDispatch(State& state, Instruction const& inst, String msg, int64_t out);


template< template<typename T> class Op >
bool ArithUnary1Fast(State& state, void* args, Value const& a, Value& c) {
    if     (a.isDouble1())  { Op<Double>::Scalar(state, args, a.d, c);   return true; }
    else if(a.isInteger1()) { Op<Integer>::Scalar(state, args, a.i, c);  return true; }
    else return false;
}

template< template<typename T> class Op > 
bool ArithUnary1Dispatch(State& state, void* args, Value const& a, Value& c) {
	if(a.isDouble()) { Zip1< Op<Double> >::eval(state, args, (Double const&)a, c); return true; }
	else if(a.isInteger()) { Zip1< Op<Integer> >::eval(state, args, (Integer const&)a, c); return true; }
	else if(a.isLogical()) { Zip1< Op<Logical> >::eval(state, args, (Logical const&)a, c); return true; }
	else if(a.isNull())	{ c = Null::Singleton(); return true; }
	else return false;
}

template< template<typename T> class Op >
bool ArithUnary2Fast(State& state, void* args, Value const& a, Value& c) {
    if (a.isDouble1())  { Op<Double>::Scalar(state, args, a.d, c);   return true; }
    else return false;
}

template< template<typename T> class Op > 
bool ArithUnary2Dispatch(State& state, void* args, Value a, Value& c) {
	return ArithUnary1Dispatch<Op>(state, args, a, c);
}

template< template<typename T> class Op >
bool LogicalUnaryFast(State& state, void* args, Value a, Value& c) {
    if     (a.isLogical1()) { Op<Logical>::Scalar(state, args, a.c, c);  return true; }
    else if(a.isDouble1())  { Op<Double>::Scalar(state, args, a.d, c);   return true; }
    else if(a.isInteger1()) { Op<Integer>::Scalar(state, args, a.i, c);  return true; }
    else if(a.isRaw1()) { Op<Raw>::Scalar(state, args, a.u, c);  return true; }
    else return false;
}

template< template<typename T> class Op > 
bool LogicalUnaryDispatch(State& state, void* args, Value a, Value& c) {
	if(a.isLogical())	{ Zip1< Op<Logical> >::eval(state, args, (Logical const&)a, c); return true; }
	else if(a.isDouble()) { Zip1< Op<Double> >::eval(state, args, (Double const&)a, c); return true; }
	else if(a.isInteger()) { Zip1< Op<Integer> >::eval(state, args, (Integer const&)a, c); return true; }
    else if(a.isRaw()) { Zip1< Op<Raw> >::eval(state, args, (Raw const&)a, c); return true; }
	else if(a.isNull())	{ c = Logical(0); return true; }
	else return false;
};

template< template<typename T> class Op >
bool OrdinalUnaryFast(State& state, void* args, Value a, Value& c) {
    if(a.isDouble1())         { Op<Double>::Scalar(state, args, a.d, c);   return true; }
    else if(a.isInteger1())   { Op<Integer>::Scalar(state, args, a.i, c);  return true; }
    else if(a.isLogical1())   { Op<Logical>::Scalar(state, args, a.c, c);  return true; }
    else if(a.isCharacter1()) { Op<Character>::Scalar(state, args, a.s, c);return true; }
    else return false;
}

template< template<typename T> class Op > 
bool OrdinalUnaryDispatch(State& state, void* args, Value a, Value& c) {
	if(a.isDouble())	{ Zip1< Op<Double> >::eval(state, args, (Double const&)a, c); return true; }
	else if(a.isInteger())	{ Zip1< Op<Integer> >::eval(state, args, (Integer const&)a, c); return true; }
	else if(a.isLogical())	{ Zip1< Op<Logical> >::eval(state, args, (Logical const&)a, c); return true; }
	else if(a.isCharacter()) { Zip1< Op<Character> >::eval(state, args, (Character const&)a, c); return true; }
    else if(a.isList() && ((List const&)a).length() == 0) {
        c = Logical::c(); return true;
    }
	else { c = Logical::False(); return true; }
}

template< template<typename S, typename T> class Op > 
bool ArithBinary1Fast(State& state, void* args, Value a, Value b, Value& c) {
    if(a.isDouble1() && b.isDouble1())
        { Op<Double, Double>::Scalar(state, args, a.d, b.d, c); return true; }
    if(a.isInteger1() && b.isInteger1())
        { Op<Integer, Integer>::Scalar(state, args, a.i, b.i, c); return true; }
	else return false;
}

template< template<typename S, typename T> class Op > 
bool ArithBinary1Dispatch(State& state, void* args, Value a, Value b, Value& c) {
	if(a.isDouble()) {
		if(b.isDouble()) {	Zip2< Op<Double,Double> >::eval(state, args, (Double const&)a, (Double const&)b, c); return true; }
		else if(b.isInteger()) {	Zip2< Op<Double,Integer> >::eval(state, args, (Double const&)a, (Integer const&)b, c); return true; }
		else if(b.isLogical()) {	Zip2< Op<Double,Logical> >::eval(state, args, (Double const&)a, (Logical const&)b, c); return true; }
		else if(b.isNull())	{ c = Double(0); return true; }
		else return false;
	} else if(a.isInteger()) {
		if(b.isDouble()) {	Zip2< Op<Integer,Double> >::eval(state, args, (Integer const&)a, (Double const&)b, c); return true; }
		else if(b.isInteger()) {	Zip2< Op<Integer,Integer> >::eval(state, args, (Integer const&)a, (Integer const&)b, c); return true; }
		else if(b.isLogical()) {	Zip2< Op<Integer,Logical> >::eval(state, args, (Integer const&)a, (Logical const&)b, c); return true; }
		else if(b.isNull())	{ c = Double(0); return true; }
		else return false;
	} else if(a.isLogical()) {
		if(b.isDouble()) {	Zip2< Op<Logical,Double> >::eval(state, args, (Logical const&)a, (Double const&)b, c); return true; }
		else if(b.isInteger()) { 	Zip2< Op<Logical,Integer> >::eval(state, args, (Logical const&)a, (Integer const&)b, c); return true; }
		else if(b.isLogical()) {	Zip2< Op<Logical,Logical> >::eval(state, args, (Logical const&)a, (Logical const&)b, c); return true; }
		else if(b.isNull())	{ c = Double(0); return true; }
		else return false;
	} else if(a.isNull()) {
		if(b.isDouble() || b.isInteger() || b.isLogical()) { c = Double(0); return true; }
		else return false;
	} 
	else return false;
}

template< template<typename S, typename T> class Op > 
bool ArithBinary2Fast(State& state, void* args, Value a, Value b, Value& c) {
    if(a.isDouble1() && b.isDouble1())
        { Op<Double, Double>::Scalar(state, args, a.d, b.d, c); return true; }
	else return false;
}

template< template<typename S, typename T> class Op > 
bool ArithBinary2Dispatch(State& state, void* args, Value a, Value b, Value& c) {
	return ArithBinary1Dispatch<Op>(state, args, a, b, c);
}

template< template<typename S, typename T> class Op > 
bool LogicalBinaryFast(State& state, void* args, Value a, Value b, Value& c) {
    if(a.isLogical1() && b.isLogical1()) { 
        Op<Logical, Logical>::Scalar(state, args, a.c, b.c, c);
        return true;
    }
	else if(a.isRaw() && b.isRaw()) {
		Op<Raw,Raw>::Scalar(state, args, a.u, b.u, c); 
        return true;
    } 
	else return false;
}

template< template<typename S, typename T> class Op >
bool LogicalBinaryDispatch(State& state, void* args, Value a, Value b, Value& c) {
	if(a.isLogical()) {
		if(b.isLogical()) { 	Zip2< Op<Logical,Logical> >::eval(state, args, (Logical const&)a, (Logical const&)b, c); return true; }
		else if(b.isDouble()) { 	Zip2< Op<Logical,Double> >::eval(state, args, (Logical const&)a, (Double const&)b, c); return true; }
		else if(b.isInteger()) { 	Zip2< Op<Logical,Integer> >::eval(state, args, (Logical const&)a, (Integer const&)b, c); return true; }
		else if(b.isNull())	{ c = Logical(0); return true; }
		else return false;
	} else if(a.isDouble()) {
		if(b.isLogical()) {	Zip2< Op<Double,Logical> >::eval(state, args, (Double const&)a, (Logical const&)b, c); return true; }
		else if(b.isDouble()) {	Zip2< Op<Double,Double> >::eval(state, args, (Double const&)a, (Double const&)b, c); return true; }
		else if(b.isInteger()) {	Zip2< Op<Double,Integer> >::eval(state, args, (Double const&)a, (Integer const&)b, c); return true; }
		else if(b.isNull())	{ c = Logical(0); return true; }
		else return false;
	} else if(a.isInteger()) {
		if(b.isLogical()) {	Zip2< Op<Integer,Logical> >::eval(state, args, (Integer const&)a, (Logical const&)b, c); return true; }
		else if(b.isDouble()) {	Zip2< Op<Integer,Double> >::eval(state, args, (Integer const&)a, (Double const&)b, c); return true; }
		else if(b.isInteger()) { Zip2< Op<Integer,Integer> >::eval(state, args, (Integer const&)a, (Integer const&)b, c); return true; }
		else if(b.isNull())	{ c = Logical(0); return true; }
		else return false; 
	} else if(a.isRaw() && b.isRaw()) {
		Zip2< Op<Raw,Raw> >::eval(state, args, (Raw const&)a, (Raw const&)b, c); 
        return true;
	} else if(a.isNull()) {
		if(b.isDouble() || b.isInteger() || b.isLogical() || b.isNull()) { c = Logical(0); return true; }
		else return false;
	}
    else return false;
}

template< class Op >
bool EnvironmentBinaryFast(State& state, void* args, Value const& a, Value const& b, Value& c) {
    return false;
}

template< class Op >
bool EnvironmentBinaryDispatch(State& state, void* args, Value const& a, Value const& b, Value& c) {
	return false;
}

template<>
bool EnvironmentBinaryDispatch< struct eqVOp<REnvironment, REnvironment> >
(State& state, void* args, Value const& a, Value const& b, Value& c);

template<>
bool EnvironmentBinaryDispatch< struct neqVOp<REnvironment, REnvironment> >
(State& state, void* args, Value const& a, Value const& b, Value& c);

template< template<typename S, typename T> class Op > 
bool UnifyBinaryFast(State& state, void* args, Value a, Value b, Value& c) {
    if(a.isDouble1() && b.isDouble1())
        { Op<Double, Double>::Scalar(state, args, a.d, b.d, c); return true; }
    if(a.isInteger1() && b.isInteger1())
        { Op<Integer, Integer>::Scalar(state, args, a.i, b.i, c); return true; }
    if(a.isLogical1() && b.isLogical1())
        { Op<Logical, Logical>::Scalar(state, args, a.c, b.c, c); return true; }
    if(a.isCharacter1() && b.isCharacter1())
        { Op<Character, Character>::Scalar(state, args, a.s, b.s, c); return true; }
	else return false;
}

template< template<typename S, typename T> class Op > 
bool UnifyBinaryDispatch(State& state, void* args, Value const& a, Value const& b, Value& c) {
	if(a.isVector() && b.isVector())
	{
        if(a.isCharacter() || b.isCharacter()) {
		    Zip2< Op<Character, Character> >::eval(state, args, As<Character>(state, a), As<Character>(state, b), c);
            return true;
        }
	    else if(a.isDouble() || b.isDouble()) {
		    Zip2< Op<Double, Double> >::eval(state, args, As<Double>(state, a), As<Double>(state, b), c);
            return true;
        }
	    else if(a.isInteger() || b.isInteger()) {	
		    Zip2< Op<Integer, Integer> >::eval(state, args, As<Integer>(state, a), As<Integer>(state, b), c);
            return true;
        }
	    else if(a.isLogical() || b.isLogical())	{
		    Zip2< Op<Logical, Logical> >::eval(state, args, As<Logical>(state, a), As<Logical>(state, b), c);
            return true;
        }
	    else if(a.isNull() || b.isNull()) {
		    c = Null::Singleton();
            return true;
        }
        else if(a.isList() && ((List const&)a).length() == 0 &&
                b.isList() && ((List const&)b).length() == 0) {
            c = List(0);
            return true;
        }
        else
            return false;
    }
    else if(a.isEnvironment() && b.isEnvironment()) {
        return EnvironmentBinaryDispatch< Op<REnvironment, REnvironment> >(state, args, a, b, c);
    }
    else {
	    return false; 
    }
}

void IfElseDispatch(State& state, void* args, Value const& a, Value const& b, Value const& cond, Value& c);

template< template<typename S, typename T> class Op > 
bool OrdinalBinaryFast(State& state, void* args, Value const& a, Value const& b, Value& c) {
	return UnifyBinaryFast<Op>(state, args, a, b, c);
}

template< template<typename S, typename T> class Op > 
bool OrdinalBinaryDispatch(State& state, void* args, Value const& a, Value const& b, Value& c) {
    if(a.isList() && ((List const&)a).length() == 0 &&
       b.isList() && ((List const&)b).length() == 0) {
        c = Logical(0);
        return true;
    }
    else {
	    return UnifyBinaryDispatch<Op>(state, args, a, b, c);
    }
}

template< template<typename S, typename T> class Op > 
bool RoundBinaryFast(State& state, void* args, Value a, Value b, Value& c) {
	return ArithBinary2Fast<Op>(state, args, a, b, c);
}

template< template<typename S, typename T> class Op > 
bool RoundBinaryDispatch(State& state, void* args, Value a, Value b, Value& c) {
	return ArithBinary2Dispatch<Op>(state, args, a, b, c);
}

template< template<typename T> class Op >
bool ArithFold1Fast(State& state, void* args, Value a, Value& c) {
    if(a.isDouble1())         { Op<Double>::Scalar(state, args, a.d, c);   return true; }
    else if(a.isInteger1())   { Op<Integer>::Scalar(state, args, a.i, c);  return true; }
    else if(a.isLogical1())   { Op<Logical>::Scalar(state, args, a.c, c);  return true; }
    else return false;
}

template< template<typename T> class Op >
bool ArithFold1Dispatch(State& state, void* args, Value const& a, Value& c) {
	if(a.isDouble())	{ FoldLeft< Op<Double> >::eval(state, args, (Double const&)a, c); return true; }
	else if(a.isInteger())	{ FoldLeft< Op<Integer> >::eval(state, args, (Integer const&)a, c); return true; }
	else if(a.isLogical())	{ FoldLeft< Op<Logical> >::eval(state, args, (Logical const&)a, c); return true; }
	else if(a.isNull())	{ Op<Double>::Scalar(state, args, Op<Double>::base(), c); return true; }
	else return false; 
}

template< template<typename T> class Op >
bool ArithFold2Fast(State& state, void* args, Value a, Value& c) {
    if(a.isDouble1())         { Op<Double>::Scalar(state, args, a.d, c);   return true; }
    else if(a.isInteger1())   { Op<Integer>::Scalar(state, args, a.i, c);  return true; }
    else if(a.isLogical1())   { Op<Logical>::Scalar(state, args, a.c, c);  return true; }
    else return false;
}

template< template<typename T> class Op >
bool ArithFold2Dispatch(State& state, void* args, Value const& a, Value& c) {
	if(a.isDouble())	{ FoldLeft< Op<Double> >::eval(state, args, (Double const&)a, c); return true; }
	else if(a.isInteger())	{ FoldLeft< Op<Integer> >::eval(state, args, (Integer const&)a, c); return true; }
	else if(a.isLogical())	{ FoldLeft< Op<Logical> >::eval(state, args, (Logical const&)a, c); return true; }
	else if(a.isNull())	{ Op<Double>::Scalar(state, args, Op<Double>::base(), c); return true; }
	else return false;
}

template< template<typename T> class Op >
bool LogicalFoldFast(State& state, void* args, Value a, Value& c) {
    if(a.isLogical1())        { Op<Logical>::Scalar(state, args, a.c, c);  return true; }
    else if(a.isDouble1())    { Op<Double>::Scalar(state, args, a.d, c);   return true; }
    else if(a.isInteger1())   { Op<Integer>::Scalar(state, args, a.i, c);  return true; }
    else return false;
}

template< template<typename T> class Op >
bool LogicalFoldDispatch(State& state, void* args, Value const& a, Value& c) {
	if(a.isLogical())	{ FoldLeft< Op<Logical> >::eval(state, args, (Logical const&)a, c); return true; }
	else if(a.isDouble())	{ FoldLeft< Op<Double> >::eval(state, args, (Double const&)a, c); return true; }
	else if(a.isInteger())	{ FoldLeft< Op<Integer> >::eval(state, args, (Integer const&)a, c); return true; }
	else if(a.isNull())	{ Op<Logical>::Scalar(state, args, Op<Logical>::base(), c); return true; }
	else return false;
}

template< template<typename T> class Op >
bool UnifyFoldFast(State& state, void* args, Value a, Value& c) {
    if(a.isCharacter1())      { Op<Character>::Scalar(state, args, a.s, c);   return true; }
    else if(a.isDouble1())    { Op<Double>::Scalar(state, args, a.d, c);   return true; }
    else if(a.isLogical1())   { Op<Logical>::Scalar(state, args, a.c, c);  return true; }
    else if(a.isInteger1())   { Op<Integer>::Scalar(state, args, a.i, c);  return true; }
    else return false;
}

template< template<typename T> class Op >
bool UnifyFoldDispatch(State& state, void* args, Value const& a, Value& c) {
	if(a.isCharacter())	{ FoldLeft< Op<Character> >::eval(state, args, (Character const&)a, c); return true; }
	else if(a.isDouble())	{ FoldLeft< Op<Double> >::eval(state, args, (Double const&)a, c); return true; }
	else if(a.isInteger())	{ FoldLeft< Op<Integer> >::eval(state, args, (Integer const&)a, c); return true; }
	else if(a.isLogical())	{ FoldLeft< Op<Logical> >::eval(state, args, (Logical const&)a, c); return true; }
	else if(a.isNull())	{ c = Null::Singleton(); return true; }
	else return false; 
}

template< template<typename T> class Op >
bool ArithScanFast(State& state, void* args, Value a, Value& c) {
    if     (a.isDouble1())  { Op<Double>::Scalar(state, args, a.d, c);   return true; }
    else if(a.isInteger1()) { Op<Integer>::Scalar(state, args, a.i, c);  return true; }
    else if(a.isLogical1()) { Op<Logical>::Scalar(state, args, a.c, c);  return true; }
    else return false;
}

template< template<typename T> class Op >
bool ArithScanDispatch(State& state, void* args, Value const& a, Value& c) {
	if(a.isDouble())	{ ScanLeft< Op<Double> >::eval(state, args, (Double const&)a, c); return true; }
	else if(a.isInteger())	{ ScanLeft< Op<Integer> >::eval(state, args, (Integer const&)a, c); return true; }
	else if(a.isLogical())	{ ScanLeft< Op<Logical> >::eval(state, args, (Logical const&)a, c); return true; }
	else if(a.isNull())	{ Op<Double>::Scalar(state, args, Op<Double>::base(), c); return true; }
	else return false;
}

template< template<typename T> class Op >
bool UnifyScanFast(State& state, void* args, Value a, Value& c) {
    if(a.isCharacter1())      { Op<Character>::Scalar(state, args, a.s, c);   return true; }
    else if(a.isDouble1())    { Op<Double>::Scalar(state, args, a.d, c);   return true; }
    else if(a.isLogical1())   { Op<Logical>::Scalar(state, args, a.c, c);  return true; }
    else if(a.isInteger1())   { Op<Integer>::Scalar(state, args, a.i, c);  return true; }
    else return false;
}

template< template<typename T> class Op >
bool UnifyScanDispatch(State& state, void* args, Value const& a, Value& c) {
	if(a.isCharacter())	{ ScanLeft< Op<Character> >::eval(state, args, (Character const&)a, c); return true; }
	else if(a.isDouble())	{ ScanLeft< Op<Double> >::eval(state, args, (Double const&)a, c); return true; }
	else if(a.isInteger())	{ ScanLeft< Op<Integer> >::eval(state, args, (Integer const&)a, c); return true; }
	else if(a.isLogical())	{ ScanLeft< Op<Logical> >::eval(state, args, (Logical const&)a, c); return true; }
	else if(a.isNull())	{ Op<Double>::Scalar(state, args, Op<Double>::base(), c); return true; }
	else return false;
}

#define SLOW_DISPATCH_DEFN(Name, String, Group, Func) \
Instruction const* Name##Slow(State& state, Instruction const& inst, void* args, Value const& a, Value& c);
UNARY_FOLD_SCAN_BYTECODES(SLOW_DISPATCH_DEFN)
#undef SLOW_DISPATCH_DEFN

#define SLOW_DISPATCH_DEFN(Name, String, Group, Func) \
Instruction const* Name##Slow(State& state, Instruction const& inst, void* args, Value const& a, Value const& b, Value& c);
BINARY_BYTECODES(SLOW_DISPATCH_DEFN)
#undef SLOW_DISPATCH_DEFN

inline bool GetFast(State& state, Value a, Value b, Value& c) {
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

Instruction const* GetSlow(State& state, Instruction const& inst, Value const& a, Value const& b, Value& c);

#endif
