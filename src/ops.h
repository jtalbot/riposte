
#ifndef _RIPOSTE_OPS_H
#define _RIPOSTE_OPS_H

#include "vector.h"
#include "coerce.h"
#include "exceptions.h"
#include <cmath>
#include <stdlib.h>

template<class X> struct CastLogical { typedef X A; typedef Logical MA; typedef Logical R; };
template<class X> struct CastInteger { typedef X A; typedef Integer MA; typedef Integer R; };
template<class X> struct CastDouble { typedef X A; typedef Double MA; typedef Double R; };
template<class X> struct CastCharacter { typedef X A; typedef Character MA; typedef Character R; };
template<class X> struct CastList { typedef X A; typedef List MA; typedef List R; };

template<class X> struct ArithUnary1   { typedef X A; typedef Double MA; typedef Double R; };
template<> struct ArithUnary1<Logical> { typedef Logical A; typedef Integer MA; typedef Integer R; };
template<> struct ArithUnary1<Integer> { typedef Integer A; typedef Integer MA; typedef Integer R; };

template<class X> struct ArithUnary2   { typedef X A; typedef Double MA; typedef Double R; };

template<class X> struct LogicalUnary  { typedef X A; typedef Logical MA; typedef Logical R; };

template<class X> struct OrdinalUnary  { typedef X A; typedef X MA; typedef Logical R; };

// Unary operators
#define UNARY_OP(Name, String, Group, Func) \
template<typename T> \
struct Name##VOp {\
	typedef typename Group<T>::A A; \
	typedef typename Group<T>::MA MA; \
	typedef typename Group<T>::R R; \
	static typename R::Element PassNA(typename MA::Element const a, typename R::Element const fa) { \
		return !MA::isCheckedNA(a) ? fa : R::NAelement; \
	} \
	static typename R::Element eval(Thread& thread, typename A::Element const v) {\
		typename MA::Element a = Cast<A, MA>(thread, v); \
		return (Func); \
	} \
	static void Scalar(Thread& thread, typename A::Element const a, Value& c) { \
		R::InitScalar(c, eval(thread, a)); \
	} \
};

inline double Abs(double a) { return fabs(a); }
inline int64_t Abs(int64_t a) { return llabs(a); }

CAST_UNARY_BYTECODES(UNARY_OP)
ARITH_UNARY_BYTECODES(UNARY_OP)
LOGICAL_UNARY_BYTECODES(UNARY_OP)
ORDINAL_UNARY_BYTECODES(UNARY_OP)
#undef UNARY_OP 
/*
template<typename T>
struct NcharOp : UnaryOp<Character, Integer> {
	static typename NcharOp::R eval(Thread& thread, typename NcharOp::A const a) { return (a == Strings::NA) ? 2 : thread.externStr(a).length(); }
};

template<typename T>
struct NzcharOp : UnaryOp<Character, Logical> {
	static typename NzcharOp::R eval(Thread& thread, typename NzcharOp::A const a) { return a != Strings::empty; }
};
*/

template<class X, class Y> struct ArithBinary1 
	{ typedef X A; typedef Y B; typedef Double MA; typedef Double MB; typedef Double R; };
template<> struct ArithBinary1<Logical,Logical> 
	{ typedef Logical A; typedef Logical B; typedef Integer MA; typedef Integer MB; typedef Integer R; };
template<> struct ArithBinary1<Logical,Integer> 
	{ typedef Logical A; typedef Integer B; typedef Integer MA; typedef Integer MB; typedef Integer R; };
template<> struct ArithBinary1<Integer,Logical> 
	{ typedef Integer A; typedef Logical B; typedef Integer MA; typedef Integer MB; typedef Integer R; };
template<> struct ArithBinary1<Integer,Integer> 
	{ typedef Integer A; typedef Integer B; typedef Integer MA; typedef Integer MB; typedef Integer R; };

template<class X, class Y> struct ArithBinary2 
	{ typedef X A; typedef Y B; typedef Double MA; typedef Double MB; typedef Double R; };

template<class X, class Y> struct LogicalBinary
	{ typedef X A; typedef Y B; typedef Logical MA; typedef Logical MB; typedef Logical R; };

template<class X, class Y> struct RoundBinary
	{ typedef X A; typedef Y B; typedef Double MA; typedef Integer MB; typedef Double R; };


template<class X, class Y> struct OrdinalBinary {};
#define ORDINAL_BINARY(X, Y, Z) \
	template<> struct OrdinalBinary<X, Y> \
	{ typedef X A; typedef Y B; typedef Z MA; typedef Z MB; typedef Logical R; };
ORDINAL_TYPE_MEET(ORDINAL_BINARY)
#undef ORDINAL_BINARY

template<class X, class Y> struct UnifyBinary {};
#define UNIFY_BINARY(X, Y, Z) \
	template<> struct UnifyBinary<X, Y> \
	{ typedef X A; typedef Y B; typedef Z MA; typedef Z MB; typedef Z R; };
ORDINAL_TYPE_MEET(UNIFY_BINARY)
#undef UNIFY_BINARY


#define BINARY_OP(Name, String, Group, Func) \
template<typename S, typename T> \
struct Name##VOp {\
	typedef typename Group<S,T>::A A; \
	typedef typename Group<S,T>::B B; \
	typedef typename Group<S,T>::MA MA; \
	typedef typename Group<S,T>::MB MB; \
	typedef typename Group<S,T>::R R; \
	static typename R::Element PassNA(typename MA::Element const a, typename MB::Element const b, typename R::Element const f) { \
		return (!MA::isCheckedNA(a) && !MB::isCheckedNA(b)) ? f : R::NAelement; \
	} \
	static typename R::Element eval(Thread& thread, typename A::Element const v, typename B::Element const w) {\
		typename MA::Element const a = Cast<A, MA>(thread, v); \
		typename MB::Element const b = Cast<B, MB>(thread, w); \
		return (Func); \
	} \
	static void Scalar(Thread& thread, typename A::Element const a, typename B::Element const b, Value& c) { \
		R::InitScalar(c, eval(thread, a, b)); \
	} \
};

inline double IDiv(double a, double b) { return floor(a/b); /* TODO: Replace with ugly R version */ }
inline int64_t IDiv(int64_t a, int64_t b) { return a/b; }

inline double Mod(double a, double b) { return a - IDiv(a,b) * b; /* TODO: Replace with ugly R version */ }
inline int64_t Mod(int64_t a, int64_t b) { return a % b; }

inline double riposte_max(Thread& thread, double a, double b) { return a > b ? a : b; }
inline int64_t riposte_max(Thread& thread, int64_t a, int64_t b) { return a > b ? a : b; }
inline int64_t riposte_max(Thread& thread, char a, char b) { return a | b; }
inline String riposte_max(Thread& thread, String a, String b) { return strcmp(a,b) > 0 ? a : b; } 

inline double riposte_min(Thread& thread, double a, double b) { return a < b ? a : b; }
inline int64_t riposte_min(Thread& thread, int64_t a, int64_t b) { return a < b ? a : b; }
inline int64_t riposte_min(Thread& thread, char a, char b) { return a & b; }
inline String riposte_min(Thread& thread, String a, String b) { return strcmp(a,b) < 0 ? a : b; }

inline double riposte_round(Thread& thread, double a, int64_t b) { double s = pow(10, b); return round(a*s)/s; }
inline double riposte_signif(Thread& thread, double a, int64_t b) {
	double d = ceil(log10(a < 0 ? -a : a));
	return riposte_round(thread, a,b-(int64_t)d);
}

inline bool gt(Thread& thread, double a, double b) { return a > b; }
inline bool gt(Thread& thread, int64_t a, int64_t b) { return a > b; }
inline bool gt(Thread& thread, char a, char b) { return (unsigned char)a > (unsigned char)b; }
inline bool gt(Thread& thread, String a, String b) { return strcmp(a,b) > 0; }

inline bool ge(Thread& thread, double a, double b) { return a >= b; }
inline bool ge(Thread& thread, int64_t a, int64_t b) { return a >= b; }
inline bool ge(Thread& thread, char a, char b) { return (unsigned char)a >= (unsigned char)b; }
inline bool ge(Thread& thread, String a, String b) { return strcmp(a,b) >= 0; }

inline bool lt(Thread& thread, double a, double b) { return a < b; }
inline bool lt(Thread& thread, int64_t a, int64_t b) { return a < b; }
inline bool lt(Thread& thread, char a, char b) { return (unsigned char)a < (unsigned char)b; }
inline bool lt(Thread& thread, String a, String b) { return strcmp(a,b) < 0; }

inline bool le(Thread& thread, double a, double b) { return a <= b; }
inline bool le(Thread& thread, int64_t a, int64_t b) { return a <= b; }
inline bool le(Thread& thread, char a, char b) { return (unsigned char)a <= (unsigned char)b; }
inline bool le(Thread& thread, String a, String b) { return strcmp(a,b) <= 0; }

ARITH_BINARY_BYTECODES(BINARY_OP)
ORDINAL_BINARY_BYTECODES(BINARY_OP)
LOGICAL_BINARY_BYTECODES(BINARY_OP)
UNIFY_BINARY_BYTECODES(BINARY_OP)
ROUND_BINARY_BYTECODES(BINARY_OP)
#undef BINARY_OP

template<class X> struct ArithFold   { typedef X A; typedef Double MA; typedef Double R; };
template<> struct ArithFold<Logical> { typedef Logical A; typedef Integer MA; typedef Integer R; };
template<> struct ArithFold<Integer> { typedef Integer A; typedef Integer MA; typedef Integer R; };

template<class X> struct UnifyFold { typedef X A; typedef X MA; typedef X R; };

template<class X> struct LogicalFold { typedef X A; typedef Logical MA; typedef Logical R; };

template<class X> struct ArithScan   { typedef X A; typedef Double MA; typedef Double R; };
template<> struct ArithScan<Logical> { typedef Logical A; typedef Integer MA; typedef Integer R; };
template<> struct ArithScan<Integer> { typedef Integer A; typedef Integer MA; typedef Integer R; };

template<class X> struct UnifyScan { typedef X A; typedef X MA; typedef X R; };

template<class X> struct addBase {};
template<> struct addBase<Double> { static Double::Element base() { return 0; } };
template<> struct addBase<Integer> { static Integer::Element base() { return 0; } };
template<> struct addBase<Logical> { static Integer::Element base() { return 0; } };

template<class X> struct mulBase {};
template<> struct mulBase<Double> { static Double::Element base() { return 1; } };
template<> struct mulBase<Integer> { static Integer::Element base() { return 1; } };
template<> struct mulBase<Logical> { static Integer::Element base() { return 1; } };

template<class X> struct lorBase { static Logical::Element base() { return Logical::FalseElement; } };

template<class X> struct landBase { static Logical::Element base() { return Logical::TrueElement; } };

template<class X> struct pminBase {};
template<> struct pminBase<Double> { static Double::Element base() { return std::numeric_limits<double>::infinity(); } };
template<> struct pminBase<Integer> { static Integer::Element base() { return std::numeric_limits<int64_t>::max(); } };
template<> struct pminBase<Logical> { static Logical::Element base() { return Logical::TrueElement; } };
template<> struct pminBase<Character> { static Character::Element base() { return Strings::Maximal; } };

template<class X> struct pmaxBase {};
template<> struct pmaxBase<Double> { static Double::Element base() { return -std::numeric_limits<double>::infinity(); } };
template<> struct pmaxBase<Integer> { static Integer::Element base() { return std::numeric_limits<int64_t>::min()+1; } };
template<> struct pmaxBase<Logical> { static Logical::Element base() { return Logical::FalseElement; } };
template<> struct pmaxBase<Character> { static Character::Element base() { return Strings::empty; } };

// Fold and scan ops
#define FOLD_OP(Name, String, Group, Func) \
template<typename T> \
struct Name##VOp : public Func##VOp<typename Func##VOp<T, T>::R, T> {\
	static typename Name##VOp::A::Element base() { return Func##Base<T>::base(); } \
	static void Scalar(Thread& thread, typename Name##VOp::B::Element const b, Value& c) { \
		Name##VOp::R::InitScalar(c, b); \
	} \
};

ARITH_FOLD_BYTECODES(FOLD_OP)
LOGICAL_FOLD_BYTECODES(FOLD_OP)
UNIFY_FOLD_BYTECODES(FOLD_OP)
ARITH_SCAN_BYTECODES(FOLD_OP)
UNIFY_SCAN_BYTECODES(FOLD_OP)
#undef FOLD_OP

// Some special folds that we need to formalize
template<class X> struct CountFold   { typedef X A; typedef X MA; typedef Integer R; };
template<class X> struct MomentFold { typedef X A; typedef Double MA; typedef Double R; };
template<class X, class Y> struct Moment2Fold { 
	typedef X A; typedef Y B; typedef Double MA; typedef Double MB; typedef Double R; };


// More complicated ops
template<class X, class Y, class Z> struct IfElse  { 
	typedef X A; typedef Y B; typedef Z C; 
	typedef Logical MA; typedef typename UnifyBinary<Y, Z>::MA MB; typedef typename UnifyBinary<Y, Z>::MB MC;
	typedef typename UnifyBinary<Y, Z>::R R; 
};

template<class X, class Y> struct Split
	{ typedef X A; typedef Y B; typedef Integer MA; typedef Y MB; typedef Y R; };

/*
template<int Len>
inline void Sequence(int64_t start, int64_t step, int64_t* dest) {
	for(int64_t i = 0, j = start; i < Len; i++, j += step) {
		dest[i] = j;
	}
}
*/


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

template< template<typename T> class Op >
void CastLogicalDispatch(Thread& thread, Value const& a, Value& c) {
    c = As<Logical>(thread, a);
}

template< template<typename T> class Op >
void CastIntegerDispatch(Thread& thread, Value const& a, Value& c) {
    c = As<Integer>(thread, a);
}

template< template<typename T> class Op >
void CastDoubleDispatch(Thread& thread, Value const& a, Value& c) {
    c = As<Double>(thread, a);
}

template< template<typename T> class Op >
void CastCharacterDispatch(Thread& thread, Value const& a, Value& c) {
    c = As<Character>(thread, a);
}

template< template<typename T> class Op >
void CastListDispatch(Thread& thread, Value const& a, Value& c) {
    c = As<List>(thread, a);
}

#endif

