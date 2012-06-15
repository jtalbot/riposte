
#ifndef _RIPOSTE_OPS_H
#define _RIPOSTE_OPS_H

#include "opgroups.h"
#include "coerce.h"
#include <cmath>
#include <stdlib.h>

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


#endif

