
#ifndef _RIPOSTE_OPS_H
#define _RIPOSTE_OPS_H

#include "vector.h"
#include "coerce.h"
#include "exceptions.h"
#include <cmath>
#include <stdlib.h>

struct TLogical {
	typedef Logical Self;
	typedef Logical Up;
};
struct TInteger {
	typedef Integer Self;
	typedef Double Up;
};
struct TDouble {
	typedef Double Self;
	typedef Double Up;
};
struct TCharacter {
	typedef Character Self;
	typedef Character Up;
};
struct TList {
	typedef List Self;
	typedef List Up;
};

// Unary operators
#define UNARY_OP(Name, T1, T2, Func) \
template<typename T> \
struct Name : public UnaryOp<typename T::T1, typename T::T2> {\
	static typename Name::R eval(Thread& thread, typename Name::A const a) {\
		if(!Name::AV::isCheckedNA(a)) return (Func);\
		else return Name::RV::NAelement;\
	}\
};

#define NYI_UNARY_OP(Name, T1, T2) \
template<typename T> \
struct Name : public UnaryOp<typename T::T1, typename T::T2> {\
	static typename Name::R eval(Thread& thread, typename Name::A const a) { _error("NYI: "#Name); } \
};

UNARY_OP(PosOp, Self, Self, a)
UNARY_OP(NegOp, Self, Self, -a)
inline double Abs(double a) { return fabs(a); }
inline int64_t Abs(int64_t a) { return llabs(a); }
UNARY_OP(AbsOp, Self, Self, Abs(a))
UNARY_OP(SignOp, Self, Up, a > 0 ? 1 : (a < 0 ? -1 : 0))
UNARY_OP(SqrtOp, Self, Up, sqrt(a))
UNARY_OP(FloorOp, Self, Up, floor(a))
UNARY_OP(CeilingOp, Self, Up, ceil(a))
UNARY_OP(TruncOp, Self, Up, a >= 0 ? floor(a) : ceil(a))
UNARY_OP(RoundOp, Self, Up, round(a))
NYI_UNARY_OP(SignifOp, Self, Up)
UNARY_OP(ExpOp, Self, Up, exp(a))
UNARY_OP(LogOp, Self, Up, log(a))
UNARY_OP(CosOp, Self, Up, cos(a))
UNARY_OP(SinOp, Self, Up, sin(a))
UNARY_OP(TanOp, Self, Up, tan(a))
UNARY_OP(ACosOp, Self, Up, acos(a))
UNARY_OP(ASinOp, Self, Up, asin(a))
UNARY_OP(ATanOp, Self, Up, atan(a))

#define UNARY_FILTER_OP(Name, Func) \
template<typename T> \
struct Name : public UnaryOp<typename T::Self, Logical> {\
	static typename Name::R eval(Thread& thread, typename Name::A const a) {\
		return (Func);\
	}\
};

UNARY_FILTER_OP(IsNAOp, IsNAOp::AV::isNA(a))
UNARY_FILTER_OP(IsNaNOp, IsNaNOp::AV::isNaN(a))
UNARY_FILTER_OP(IsFiniteOp, IsFiniteOp::AV::isFinite(a))
UNARY_FILTER_OP(IsInfiniteOp, IsInfiniteOp::AV::isInfinite(a))

template<typename T> 
struct LNotOp : UnaryOp<Logical, Logical> {
	static typename LNotOp::R eval(Thread& thread, typename LNotOp::A const a) { return !a; }
};

template<typename T>
struct NcharOp : UnaryOp<Character, Integer> {
	static typename NcharOp::R eval(Thread& thread, typename NcharOp::A const a) { return (a == Strings::NA) ? 2 : thread.externStr(a).length(); }
};

template<typename T>
struct NzcharOp : UnaryOp<Character, Logical> {
	static typename NzcharOp::R eval(Thread& thread, typename NzcharOp::A const a) { return a != Strings::empty; }
};

#undef UNARY_OP 
#undef NYI_UNARY_OP

// Binary operators
#define BINARY_OP(Name, T1, T2, T3, Func) \
template<typename T> \
struct Name : public BinaryOp<typename T::T1, typename T::T2, typename T::T3> {\
	static typename Name::R eval(Thread& thread, typename Name::A const a, typename Name::B const b) { \
		if(!Name::AV::isCheckedNA(a) && !Name::BV::isCheckedNA(b)) return (Func); \
		else return Name::RV::NAelement; \
	} \
};

inline double riposte_max(Thread& thread, double a, double b) { 
	if(a!=a) return a; if(b!=b) return b; return a > b ? a : b; 
}
inline int64_t riposte_max(Thread& thread, int64_t a, int64_t b) { 
	return a > b ? a : b;
}
// TODO: Fix string min and max
inline String riposte_max(Thread& thread, String a, String b) {
	return thread.externStr(a) > thread.externStr(b) ? a : b;
} 

inline double riposte_min(Thread& thread, double a, double b) { 
	if(a!=a) return a; if(b!=b) return b; return a < b ? a : b; 
}
inline int64_t riposte_min(Thread& thread, int64_t a, int64_t b) { 
	return a < b ? a : b;
}
inline String riposte_min(Thread& thread, String a, String b) {
	return thread.externStr(a) < thread.externStr(b) ? a : b;
}

BINARY_OP(AddOp, Self, Self, Self, a+b)
BINARY_OP(SubOp, Self, Self, Self, a-b)
BINARY_OP(MulOp, Self, Self, Self, a*b)
BINARY_OP(DivOp, Self, Self, Up, ((typename DivOp::R)a)/b)
inline double IDiv(double a, double b) { return floor(a/b); /* TODO: Replace with ugly R version */ }
inline int64_t IDiv(int64_t a, int64_t b) { return a/b; }
BINARY_OP(IDivOp, Self, Self, Self, IDiv(a,b))
BINARY_OP(PowOp, Self, Self, Up, pow(a,b))
inline double Mod(double a, double b) { return a - IDiv(a,b) * b; /* TODO: Replace with ugly R version */ }
inline int64_t Mod(int64_t a, int64_t b) { return a % b; }
BINARY_OP(ModOp, Self, Self, Self, Mod(a,b))
BINARY_OP(ATan2Op, Self, Self, Self, atan2(a,b))
BINARY_OP(HypotOp, Self, Self, Self, hypot(a,b))
BINARY_OP(PMinOp, Self, Self, Self, riposte_min(thread, a,b))
BINARY_OP(PMaxOp, Self, Self, Self, riposte_max(thread, a,b))

#undef BINARY_OP

// Ordinal binary ops
#define ORDINAL_OP(Name, Func) \
template<typename T> \
struct Name : public BinaryOp<typename T::Self, typename T::Self, Logical> {\
	static typename Name::R eval(Thread& thread, typename Name::A const a, typename Name::B const b) { \
		if(!Name::AV::isNA(a) && !Name::BV::isNA(b)) return (Func); \
		else return Name::RV::NAelement; \
	} \
};\

#define CHARACTER_ORDINAL_OP(Name, Func) \
template<> \
struct Name<TCharacter> : public BinaryOp<Character, Character, Logical> {\
	static Name::R eval(Thread& thread, Name::A const a, Name::B const b) { \
		if(!Name::AV::isNA(a) && !Name::BV::isNA(b)) return (Func); \
		else return Name::RV::NAelement; \
	} \
};\

ORDINAL_OP(LTOp, a<b)	CHARACTER_ORDINAL_OP(LTOp, thread.externStr(a).compare(thread.externStr(b)) < 0)
ORDINAL_OP(GTOp, a>b)	CHARACTER_ORDINAL_OP(GTOp, thread.externStr(a).compare(thread.externStr(b)) > 0)
ORDINAL_OP(LEOp, a<=b)	CHARACTER_ORDINAL_OP(LEOp, thread.externStr(a).compare(thread.externStr(b)) <= 0)
ORDINAL_OP(GEOp, a>=b)	CHARACTER_ORDINAL_OP(GEOp, thread.externStr(a).compare(thread.externStr(b)) >= 0)
ORDINAL_OP(EqOp, a==b)	/* Character equality can just compare Strings */
ORDINAL_OP(NeqOp, a!=b) /* Character inequality can just compare Strings */

#undef ORDINAL_OP
#undef CHARACTER_ORDINAL_OP

// Logical binary ops
template<typename T>
struct AndOp : BinaryOp<Logical, Logical, Logical> {
	static typename AndOp::R eval(Thread& thread, typename AndOp::A const a, typename AndOp::B const b) {
		if(AV::isNA(a)) return b ? RV::NAelement : 0;
		else if(BV::isNA(b)) return a ? RV::NAelement : 0;
		else return a && b ? 1 : 0;
	}
};

template<typename T>
struct OrOp : BinaryOp<Logical, Logical, Logical> {
	static typename OrOp::R eval(Thread& thread, typename OrOp::A const a, typename OrOp::B const b) {
		if(AV::isNA(a)) return b ? 1 : RV::NAelement;
		else if(BV::isNA(b)) return a ? 1 : RV::NAelement;
		return (a || b) ? 1 : 0;
	}
};

// Fold ops

#define FOLD_OP(Name, Func, Initial) \
template<typename T> \
struct Name : FoldOp<typename T::Self> { \
	static typename Name::A base() { return Initial; } \
	static typename Name::R eval(Thread& thread, typename Name::R const a, typename Name::A const b) { \
		return (Func); \
	} \
}; \

#define CHARACTER_FOLD_OP(Name, Func, Initial) \
template<> \
struct Name<TCharacter> : FoldOp<Character> { \
	static Name::A base() { return Initial; } \
	static Name::R eval(Thread& thread, Name::R const a, Name::A const b) { \
		return (Func); \
	} \
}; \

FOLD_OP(MaxOp, PMaxOp<T>::eval(thread, a, b), -std::numeric_limits<typename MaxOp<T>::A>::infinity())
CHARACTER_FOLD_OP(MaxOp, PMaxOp<TCharacter>::eval(thread, a, b), Strings::empty) 
FOLD_OP(MinOp, PMinOp<T>::eval(thread, a, b), std::numeric_limits<typename MaxOp<T>::A>::infinity()) 
CHARACTER_FOLD_OP(MinOp, PMinOp<TCharacter>::eval(thread, a, b), Strings::empty) 
FOLD_OP(SumOp, AddOp<T>::eval(thread, a, b), 0) 
FOLD_OP(ProdOp, MulOp<T>::eval(thread, a, b), 1) 
FOLD_OP(AnyOp, OrOp<TLogical>::eval(thread, a, b), 0)
FOLD_OP(AllOp, AndOp<TLogical>::eval(thread, a, b), 1)

#undef FOLD_OP

template< template<class Op> class Lift, template<typename T> class Op > 
void unaryArith(Thread& thread, Value a, Value& c) {
	if(a.isDouble())
		Lift< Op<TDouble> >::eval(thread, (Double const&)a, c);
	else if(a.isInteger())
		Lift< Op<TInteger> >::eval(thread, (Integer const&)a, c);
	else if(a.isLogical())
		Lift< Op<TInteger> >::eval(thread, As<Integer>(thread, a), c);
	else if(a.isNull())
		c = Null::Singleton();
	else if(a.isObject()) {
		unaryArith<Lift, Op>(thread, ((Object const&)a).base(), c);
		if(((Object const&)a).hasNames()) {
			Object::Init(c, c, ((Object const&)a).getNames());
		}
	}
	else 
		_error("non-numeric argument to unary numeric operator");
};

template< template<class Op> class Lift, template<typename T> class Op > 
void unaryLogical(Thread& thread, Value a, Value& c) {
	if(a.isLogicalCoerce())
		Lift< Op<TLogical> >::eval(thread, As<Logical>(thread, a), c);
	else if(a.isNull())
		c = Logical(0);
	else if(a.isObject()) {
		unaryLogical<Lift, Op>(thread, ((Object const&)a).base(), c);
		if(((Object const&)a).hasNames()) {
			Object::Init(c, c, ((Object const&)a).getNames());
		}
	}
	else
		_error("non-logical argument to unary logical operator");
};

template< template<class Op> class Lift, template<typename T> class Op > 
void unaryOrdinal(Thread& thread, Value a, Value& c) {
	if(a.isDouble())
		Lift< Op<TDouble> >::eval(thread, (Double const&)a, c);
	else if(a.isInteger())
		Lift< Op<TInteger> >::eval(thread, (Integer const&)a, c);
	else if(a.isLogical())
		Lift< Op<TInteger> >::eval(thread, As<Integer>(thread, a), c);
	else if(a.isCharacter())
		Lift< Op<TCharacter> >::eval(thread, (Character const&)a, c);
	else if(a.isNull()) {
		Double::InitScalar(c, Op<TDouble>::base());
		_warning(thread, "no non-missing arguments to min; returning Inf");
	}
	else if(a.isObject()) {
		unaryOrdinal<Lift, Op>(thread, ((Object const&)a).base(), c);
		if(((Object const&)a).hasNames()) {
			Object::Init(c, c, ((Object const&)a).getNames());
		}
	}
	else {
		_error("non-ordinal argument to ordinal operator");
	}
}

template< template<class Op> class Lift, template<typename T> class Op > 
void unaryCharacter(Thread& thread, Value a, Value& c) {
	if(a.isVector())
		Lift< Op<TCharacter> >::eval(thread, As<Character>(thread, a), c);
	else if(a.isObject()) {
		unaryCharacter<Lift, Op>(thread, ((Object const&)a).base(), c);
		if(((Object const&)a).hasNames()) {
			Object::Init(c, c, ((Object const&)a).getNames());
		}
	}
};

template< template<class Op> class Lift, template<typename T> class Op > 
void unaryFilter(Thread& thread, Value a, Value& c) {
	if(a.isDouble())
		Lift< Op<TDouble> >::eval(thread, (Double const&)a, c);
	else if(a.isInteger())
		Lift< Op<TInteger> >::eval(thread, (Integer const&)a, c);
	else if(a.isLogical())
		Lift< Op<TLogical> >::eval(thread, (Logical const&)a, c);
	else if(a.isCharacter())
		Lift< Op<TCharacter> >::eval(thread, (Character const&)a, c);
	else if(a.isNull())
		Logical(0);
	else if(a.isList())
		Lift< Op<TList> >::eval(thread, (List const&)a, c);
	else if(a.isObject()) {
		unaryFilter<Lift, Op>(thread, ((Object const&)a).base(), c);
		if(((Object const&)a).hasNames()) {
			Object::Init(c, c, ((Object const&)a).getNames());
		}
	}
	else {
		_error("unexpected type in unaryFilter");
	}
};

template< template<class Op> class Lift, template<typename T> class Op > 
void binaryArithSlow(Thread& thread, Value a, Value b, Value& c) {
	if(a.isDouble() && b.isDouble())
		Lift< Op<TDouble> >::eval(thread, (Double const&)a, (Double const&)b, c);
	else if((a.isDouble() && b.isMathCoerce()) || (b.isDouble() && a.isMathCoerce()))
		Lift< Op<TDouble> >::eval(thread, As<Double>(thread, a), As<Double>(thread, b), c);
	else if(a.isMathCoerce() && b.isMathCoerce()) 
		Lift< Op<TInteger> >::eval(thread, As<Integer>(thread, a), As<Integer>(thread, b), c);
	else if(a.isObject())
		binaryArithSlow<Lift, Op>(thread, ((Object const&)a).base(), b, c);
	else if(b.isObject())
		binaryArithSlow<Lift, Op>(thread, a, ((Object const&)b).base(), c);
	else
		_error("non-numeric argument to binary numeric operator");
}

template< template<class Op> class Lift, template<typename T> class Op > 
void binaryArith(Thread& thread, Value const& a, Value const& b, Value& c) ALWAYS_INLINE; 

template< template<class Op> class Lift, template<typename T> class Op > 
void binaryArith(Thread& thread, Value const& a, Value const& b, Value& c) {
	if(a.isDouble1()) {
		if(b.isDouble1()) 
			{ Double::InitScalar(c, Op<TDouble>::eval(thread, a.d, b.d)); return; }
		else if(b.isInteger1())	
			{ Double::InitScalar(c, Op<TDouble>::eval(thread, a.d, (double)b.i));return; }
	}
	else if(a.isInteger1()) {
		if(b.isDouble1()) 
			{ Double::InitScalar(c, Op<TDouble>::eval(thread, (double)a.i, b.d)); return; }
		else if(b.isInteger1())	
			{ Integer::InitScalar(c, Op<TInteger>::eval(thread, a.i, b.i)); return;}
	}
	binaryArithSlow<Lift, Op>(thread, a, b, c);
}
template< template<class Op> class Lift, template<typename T> class Op > 
void binaryLogical(Thread& thread, Value const& a, Value const& b, Value& c) {
	if(a.isLogical() && b.isLogical())
		Lift< Op<TLogical> >::eval(thread, (Logical const&)a, (Logical const&)b, c);
	else if(a.isLogicalCoerce() && b.isLogicalCoerce())
		Lift< Op<TLogical> >::eval(thread, As<Logical>(thread, a), As<Logical>(thread, b), c);
	else if(a.isObject())
		binaryLogical<Lift, Op>(thread, ((Object const&)a).base(), b, c);
	else if(b.isObject())
		binaryLogical<Lift, Op>(thread, a, ((Object const&)b).base(), c);
	else 
		_error("non-logical argument to binary logical operator");
}

template< template<class Op> class Lift, template<typename T> class Op > 
void binaryOrdinal(Thread& thread, Value const& a, Value const& b, Value& c) {
	if(a.isDouble() && b.isDouble())
		Lift< Op<TDouble> >::eval(thread, (Double const&)a, (Double const&)b, c);
	else if((a.isDouble() && b.isMathCoerce()) || (b.isDouble() && a.isMathCoerce()))
		Lift< Op<TDouble> >::eval(thread, As<Double>(thread, a), As<Double>(thread, b), c);
	else if(a.isMathCoerce() && b.isMathCoerce()) 
		Lift< Op<TInteger> >::eval(thread, As<Integer>(thread, a), As<Integer>(thread, b),c );
	else if(a.isCharacter() && b.isCharacter())
		Lift< Op<TCharacter> >::eval(thread, Character(a), Character(b), c);
	else if(a.isObject())
		binaryOrdinal<Lift, Op>(thread, ((Object const&)a).base(), b, c);
	else if(b.isObject())
		binaryOrdinal<Lift, Op>(thread, a, ((Object const&)b).base(), c);
	else {
		_error("non-ordinal argument to ordinal operator");
	}
}

// Figure out the type of the operation given an input type
inline void selectType(ByteCode::Enum bc, Type::Enum input, Type::Enum * atyp, Type::Enum * rtyp) {
	switch(bc) {
#define ARITH_CASE(name,str,Op,...) \
	case ByteCode::name: \
		if(input == Type::Integer) { *atyp = Op<TInteger>::RV::VectorType; *rtyp = Op<TInteger>::RV::VectorType; } \
		else if(input == Type::Double || input == Type::Logical) { *atyp = Op<TDouble>::RV::VectorType; *rtyp = Op<TDouble>::RV::VectorType; } \
		else _error("Unknown type"); \
		break;
#define LOGICAL_CASE(name,str,Op,...) \
	case ByteCode::name: \
		*atyp = Type::Logical; *rtyp = Type::Logical; \
		break;
#define BINARY_ORDINAL_CASE(name,str,Op,...) \
	case ByteCode::name: /*logical inputs get promoted to double so that we can use sse ops to implement the ordinals*/\
		*atyp = (input == Type::Logical) ? Type::Double : input; *rtyp = Type::Logical; \
		break;
#define ORDINAL_CASE(name,str,Op,...) \
	case ByteCode::name: \
		*atyp = input; *rtyp = input; \
		break;
ARITH_BYTECODES(ARITH_CASE)
LOGICAL_BYTECODES(LOGICAL_CASE)
BINARY_ORDINAL_MAP_BYTECODES(BINARY_ORDINAL_CASE)
ORDINAL_FOLD_BYTECODES(ORDINAL_CASE)
ORDINAL_SCAN_BYTECODES(ORDINAL_CASE)
#undef ARITH_CASE
#undef LOGICAL_RESULT_CASE
#undef ORDINAL_CASE
	default:
		_error("Not a known op in selectType");
	}

}

inline Type::Enum meetType(Type::Enum a, Type::Enum b) {
	return std::max(a,b);
}

inline void selectType(ByteCode::Enum bc, Type::Enum a, Type::Enum b, Type::Enum * atyp, Type::Enum * btyp, Type::Enum * rtyp) {
	selectType(bc,meetType(a,b),atyp,rtyp);
	*btyp = *atyp;
}



template<int Len>
inline void Sequence(int64_t start, int64_t step, int64_t* dest) {
	for(int64_t i = 0, j = start; i < Len; i++, j += step) {
		dest[i] = j;
	}
}


#endif

