
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
	static typename R::Element eval(State& state, void* args, typename A::Element const v) {\
		typename MA::Element a = Cast<A, MA>(state, v); \
		return (Func); \
	} \
	static void Scalar(State& state, void* args, typename A::Element const a, Value& c) { \
		R::InitScalar(c, eval(state, args, a)); \
	} \
};

inline double Abs(double a) { return fabs(a); }
inline int64_t Abs(int64_t a) { return llabs(a); }

ARITH_UNARY_BYTECODES(UNARY_OP)
LOGICAL_UNARY_BYTECODES(UNARY_OP)
ORDINAL_UNARY_BYTECODES(UNARY_OP)
#undef UNARY_OP 

template<>
struct lnotVOp<Raw> {
    typedef Raw A;
    typedef Raw MA;
    typedef Raw R;
    static Raw::Element eval(State& state, void* args, Raw::Element const v) {
        return ~v;
    }
    static void Scalar(State& state, void* args, Raw::Element const a, Value& c) {
        Raw::InitScalar(c, eval(state, args, a));
    }
};


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
	static typename R::Element PassCheckedNA(typename MA::Element const a, typename MB::Element const b, typename R::Element const f) { \
        return (!MA::isNA(a) && !MB::isNA(b)) ? f : R::NAelement; \
	} \
	static typename R::Element eval(State& state, void* args, typename A::Element const v, typename B::Element const w) {\
		typename MA::Element const a = Cast<A, MA>(state, v); \
		typename MB::Element const b = Cast<B, MB>(state, w); \
		return (Func); \
	} \
	static void Scalar(State& state, void* args, typename A::Element const a, typename B::Element const b, Value& c) { \
		R::InitScalar(c, eval(state, args, a, b)); \
	} \
};

inline double IDiv(double a, double b) { return floor(a/b); }
inline int64_t IDiv(int64_t a, int64_t b) { return b == 0 ? Integer::NAelement : a/b; }

inline double Mod(double a, double b) { return a - IDiv(a,b) * b; }
inline int64_t Mod(int64_t a, int64_t b) { return b == 0 ? Integer::NAelement : a % b; }

inline double riposte_max(State& state, double a, double b) { return a > b ? a : b; }
inline int64_t riposte_max(State& state, int64_t a, int64_t b) { return a > b ? a : b; }
inline int64_t riposte_max(State& state, char a, char b) { return a | b; }
inline String riposte_max(State& state, String a, String b) { return strcmp(a->s,b->s) > 0 ? a : b; } 

inline double riposte_min(State& state, double a, double b) { return a < b ? a : b; }
inline int64_t riposte_min(State& state, int64_t a, int64_t b) { return a < b ? a : b; }
inline int64_t riposte_min(State& state, char a, char b) { return a & b; }
inline String riposte_min(State& state, String a, String b) { return strcmp(a->s,b->s) < 0 ? a : b; }

inline bool gt(State& state, double a, double b) { return a > b; }
inline bool gt(State& state, int64_t a, int64_t b) { return a > b; }
inline bool gt(State& state, char a, char b) { return (unsigned char)a > (unsigned char)b; }
inline bool gt(State& state, String a, String b) { return strcmp(a->s,b->s) > 0; }

inline bool ge(State& state, double a, double b) { return a >= b; }
inline bool ge(State& state, int64_t a, int64_t b) { return a >= b; }
inline bool ge(State& state, char a, char b) { return (unsigned char)a >= (unsigned char)b; }
inline bool ge(State& state, String a, String b) { return strcmp(a->s,b->s) >= 0; }

inline bool lt(State& state, double a, double b) { return a < b; }
inline bool lt(State& state, int64_t a, int64_t b) { return a < b; }
inline bool lt(State& state, char a, char b) { return (unsigned char)a < (unsigned char)b; }
inline bool lt(State& state, String a, String b) { return strcmp(a->s,b->s) < 0; }

inline bool le(State& state, double a, double b) { return a <= b; }
inline bool le(State& state, int64_t a, int64_t b) { return a <= b; }
inline bool le(State& state, char a, char b) { return (unsigned char)a <= (unsigned char)b; }
inline bool le(State& state, String a, String b) { return strcmp(a->s,b->s) <= 0; }

ARITH_BINARY_BYTECODES(BINARY_OP)
ORDINAL_BINARY_BYTECODES(BINARY_OP)
LOGICAL_BINARY_BYTECODES(BINARY_OP)
UNIFY_BINARY_BYTECODES(BINARY_OP)
#undef BINARY_OP

template<>
struct lorVOp<Raw,Raw> {
    typedef Raw A;
    typedef Raw B;
    typedef Raw MA;
    typedef Raw MB;
    typedef Raw R;

    static Raw::Element eval(State& state, void* args, Raw::Element const v, Raw::Element const w) {
        return v | w;
    }
    static void Scalar(State& state, void* args, Raw::Element const a, Raw::Element const b, Value& c) {
        Raw::InitScalar(c, eval(state, args, a, b));
    }
};

template<>
struct landVOp<Raw,Raw> {
    typedef Raw A;
    typedef Raw B;
    typedef Raw MA;
    typedef Raw MB;
    typedef Raw R;

    static Raw::Element eval(State& state, void* args, Raw::Element const v, Raw::Element const w) {
        return v & w;
    }
    static void Scalar(State& state, void* args, Raw::Element const a, Raw::Element const b, Value& c) {
        Raw::InitScalar(c, eval(state, args, a, b));
    }
};

template<class X> struct addBase {};
template<> struct addBase<Double> { static Double::Element base() { return 0; } };
template<> struct addBase<Integer> { static Integer::Element base() { return 0; } };
template<> struct addBase<Logical> { static Integer::Element base() { return 0; } };

template<class X> struct mulBase { static Double::Element base() { return 1; } };

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
	static void Scalar(State& state, void* args, typename Name##VOp::B::Element const b, Value& c) { \
		Name##VOp::R::InitScalar(c, Cast<typename Name##VOp::B, typename Name##VOp::MB>(state, b)); \
	} \
};

ARITH_FOLD_BYTECODES(FOLD_OP)
LOGICAL_FOLD_BYTECODES(FOLD_OP)
UNIFY_FOLD_BYTECODES(FOLD_OP)
ARITH_SCAN_BYTECODES(FOLD_OP)
UNIFY_SCAN_BYTECODES(FOLD_OP)
#undef FOLD_OP

template<typename S>
struct IfElseVOp {
    typedef S A;
    typedef S B;
    typedef Logical C;
    typedef S R;
    static typename R::Element eval(State& state, void* args, typename A::Element const a, typename B::Element const b, typename C::Element const c) {
        if(Logical::isTrue(c))
            return a;
        else if(Logical::isFalse(c))
            return b;
        else
            return R::NAelement;
    }
    static void Scalar(State& state, void* args, typename A::Element const a, typename B::Element const b, typename C::Element const c, Value& out) {
        R::InitScalar(out, eval(state, args, a, b, c));
    }
};

struct FoldFuncArgs {
    void* base;
    void* func;
    void* fini;
};

template<typename S, typename T>
struct FoldFuncOp {
    typedef S A;
    typedef T R;
    typedef void* I;

    typedef I (*Base)(State&);
    typedef I (*Func)(State&, I, typename A::Element);
    typedef typename R::Element (*Fini)(State&, I);
    
    static I base(State& state, void* funcs) {
        return ((Base)((FoldFuncArgs*)funcs)->base)(state);
    }

    static I eval(State& state, void* funcs, I const a, typename A::Element b) {
		return ((Func)((FoldFuncArgs*)funcs)->func)(state, a, b);
    }

    static typename R::Element finalize(State& state, void* funcs, I const a) {
        return ((Fini)((FoldFuncArgs*)funcs)->fini)(state, a);
    }
};

#endif

