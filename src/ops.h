
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
struct TComplex {
	typedef Complex Self;
	typedef Complex Up;
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
	static typename Name::R eval(State& state, typename Name::A const a) {\
		if(!Name::AV::isCheckedNA(a)) return (Func);\
		else return Name::RV::NAelement;\
	}\
};

#define NYI_UNARY_OP(Name, T1, T2) \
template<typename T> \
struct Name : public UnaryOp<typename T::T1, typename T::T2> {\
	static typename Name::R eval(State& state, typename Name::A const a) { _error("NYI: "#Name); } \
};

#define NYI_COMPLEX_OP(T) template<> \
struct T<TComplex> : public UnaryOp<TComplex::Self, TComplex::Self> { \
	static T::R eval(State& state, T::A const a) { _error("unimplemented complex function"); } \
};

UNARY_OP(PosOp, Self, Self, a)
UNARY_OP(NegOp, Self, Self, -a)
inline double Abs(double a) { return fabs(a); }
inline int64_t Abs(int64_t a) { return llabs(a); }
inline std::complex<double> Abs(std::complex<double> const& a) { return std::complex<double>(sqrt(a.real()*a.real() + a.imag()*a.imag()), 0); }
UNARY_OP(AbsOp, Self, Self, Abs(a))
UNARY_OP(SignOp, Self, Up, a > 0 ? 1 : (a < 0 ? -1 : 0))	NYI_COMPLEX_OP(SignOp)
UNARY_OP(SqrtOp, Self, Up, sqrt(a))
UNARY_OP(FloorOp, Self, Up, floor(a))				NYI_COMPLEX_OP(FloorOp)
UNARY_OP(CeilingOp, Self, Up, ceil(a))   			NYI_COMPLEX_OP(CeilingOp)
UNARY_OP(TruncOp, Self, Up, a >= 0 ? floor(a) : ceil(a))	NYI_COMPLEX_OP(TruncOp)
UNARY_OP(RoundOp, Self, Up, round(a))	   			NYI_COMPLEX_OP(RoundOp)
NYI_UNARY_OP(SignifOp, Self, Up)				NYI_COMPLEX_OP(SignifOp)
UNARY_OP(ExpOp, Self, Up, exp(a))
UNARY_OP(LogOp, Self, Up, log(a))
UNARY_OP(CosOp, Self, Up, cos(a))
UNARY_OP(SinOp, Self, Up, sin(a))
UNARY_OP(TanOp, Self, Up, tan(a))
UNARY_OP(ACosOp, Self, Up, acos(a))				NYI_COMPLEX_OP(ACosOp)
UNARY_OP(ASinOp, Self, Up, asin(a))				NYI_COMPLEX_OP(ASinOp)
UNARY_OP(ATanOp, Self, Up, atan(a))				NYI_COMPLEX_OP(ATanOp)

#define UNARY_FILTER_OP(Name, Func) \
template<typename T> \
struct Name : public UnaryOp<typename T::Self, Logical> {\
	static typename Name::R eval(State& state, typename Name::A const a) {\
		return (Func);\
	}\
};

UNARY_FILTER_OP(IsNAOp, IsNAOp::AV::isNA(a))
UNARY_FILTER_OP(IsNaNOp, IsNaNOp::AV::isNaN(a))
UNARY_FILTER_OP(IsFiniteOp, IsFiniteOp::AV::isFinite(a))
UNARY_FILTER_OP(IsInfiniteOp, IsInfiniteOp::AV::isInfinite(a))

template<typename T> 
struct LNotOp : UnaryOp<Logical, Logical> {
	static typename LNotOp::R eval(State& state, typename LNotOp::A const a) { return !a; }
};

template<typename T>
struct NcharOp : UnaryOp<Character, Integer> {
	static typename NcharOp::R eval(State& state, typename NcharOp::A const a) { return (a == Symbols::NA) ? 2 : state.SymToStr(a).length(); }
};

template<typename T>
struct NzcharOp : UnaryOp<Character, Logical> {
	static typename NzcharOp::R eval(State& state, typename NzcharOp::A const a) { return a != Symbols::empty; }
};

#undef UNARY_OP 
#undef NYI_UNARY_OP
#undef NYI_COMPLEX_OP

// Binary operators
#define BINARY_OP(Name, T1, T2, T3, Func) \
template<typename T> \
struct Name : public BinaryOp<typename T::T1, typename T::T2, typename T::T3> {\
	static typename Name::R eval(State& state, typename Name::A const a, typename Name::B const b) { \
		if(!Name::AV::isCheckedNA(a) && !Name::BV::isCheckedNA(b)) return (Func); \
		else return Name::RV::NAelement; \
	} \
};

#define NYI_COMPLEX_OP(T) template<> \
struct T<TComplex> : public BinaryOp<Complex, Complex, Complex> { \
	static T::R eval(State& state, T::A const a, T::B const b) { _error("unimplemented complex function"); } \
};

inline double riposte_max(State& state, double a, double b) { 
	if(a!=a) return a; if(b!=b) return b; return a > b ? a : b; 
}
inline int64_t riposte_max(State& state, int64_t a, int64_t b) { 
	return a > b ? a : b;
}
inline Symbol riposte_max(State& state, Symbol a, Symbol b) {
	return state.SymToStr(a) > state.SymToStr(b) ? a : b;
} 

inline double riposte_min(State& state, double a, double b) { 
	if(a!=a) return a; if(b!=b) return b; return a < b ? a : b; 
}
inline int64_t riposte_min(State& state, int64_t a, int64_t b) { 
	return a < b ? a : b;
}
inline Symbol riposte_min(State& state, Symbol a, Symbol b) {
	return state.SymToStr(a) < state.SymToStr(b) ? a : b;
} 

BINARY_OP(AddOp, Self, Self, Self, a+b)
BINARY_OP(SubOp, Self, Self, Self, a-b)
BINARY_OP(MulOp, Self, Self, Self, a*b)
BINARY_OP(DivOp, Self, Self, Up, ((typename DivOp::R)a)/b)
inline double IDiv(double a, double b) { return floor(a/b); /* TODO: Replace with ugly R version */ }
inline int64_t IDiv(int64_t a, int64_t b) { return a/b; }
BINARY_OP(IDivOp, Self, Self, Self, IDiv(a,b))			NYI_COMPLEX_OP(IDivOp)
BINARY_OP(PowOp, Self, Self, Up, pow(a,b))
inline double Mod(double a, double b) { return a - IDiv(a,b) * b; /* TODO: Replace with ugly R version */ }
inline int64_t Mod(int64_t a, int64_t b) { return a % b; }
BINARY_OP(ModOp, Self, Self, Self, Mod(a,b))			NYI_COMPLEX_OP(ModOp)
BINARY_OP(PMinOp, Self, Self, Self, riposte_min(state, a,b))		NYI_COMPLEX_OP(PMinOp)
BINARY_OP(PMaxOp, Self, Self, Self, riposte_max(state, a,b))		NYI_COMPLEX_OP(PMaxOp)

#undef BINARY_OP
#undef NYI_COMPLEX_OP

// Ordinal binary ops
#define ORDINAL_OP(Name, Func) \
template<typename T> \
struct Name : public BinaryOp<typename T::Self, typename T::Self, Logical> {\
	static typename Name::R eval(State& state, typename Name::A const a, typename Name::B const b) { \
		if(!Name::AV::isNA(a) && !Name::BV::isNA(b)) return (Func); \
		else return Name::RV::NAelement; \
	} \
};\

#define CHARACTER_ORDINAL_OP(Name, Func) \
template<> \
struct Name<TCharacter> : public BinaryOp<Character, Character, Logical> {\
	static Name::R eval(State& state, Name::A const a, Name::B const b) { \
		if(!Name::AV::isNA(a) && !Name::BV::isNA(b)) return (Func); \
		else return Name::RV::NAelement; \
	} \
};\

#define INVALID_COMPLEX_OP(Name) \
template<> \
struct Name<TComplex> : public BinaryOp<Complex, Complex, Logical> { \
	static Name::R eval(State& state, Name::A const a, Name::B const b) { _error("invalid complex function"); } \
};

ORDINAL_OP(LTOp, a<b)	CHARACTER_ORDINAL_OP(LTOp, state.SymToStr(a).compare(state.SymToStr(b)) < 0)		INVALID_COMPLEX_OP(LTOp)
ORDINAL_OP(GTOp, a>b)	CHARACTER_ORDINAL_OP(GTOp, state.SymToStr(a).compare(state.SymToStr(b)) > 0)		INVALID_COMPLEX_OP(GTOp)
ORDINAL_OP(LEOp, a<=b)	CHARACTER_ORDINAL_OP(LEOp, state.SymToStr(a).compare(state.SymToStr(b)) <= 0)	INVALID_COMPLEX_OP(LEOp)
ORDINAL_OP(GEOp, a>=b)	CHARACTER_ORDINAL_OP(GEOp, state.SymToStr(a).compare(state.SymToStr(b)) >= 0)	INVALID_COMPLEX_OP(GEOp)
ORDINAL_OP(EqOp, a==b)	/* Character equality can just compare Symbols */
ORDINAL_OP(NeqOp, a!=b) /* Character inequality can just compare Symbols */

#undef ORDINAL_OP
#undef CHARACTER_ORDINAL_OP
#undef INVALID_COMPLEX_OP

// Logical binary ops
template<typename T>
struct AndOp : BinaryOp<Logical, Logical, Logical> {
	static typename AndOp::R eval(State& state, typename AndOp::A const a, typename AndOp::B const b) {
		if(AV::isNA(a)) return b ? RV::NAelement : 0;
		else if(BV::isNA(b)) return a ? RV::NAelement : 0;
		else return a && b ? 1 : 0;
	}
};

template<typename T>
struct OrOp : BinaryOp<Logical, Logical, Logical> {
	static typename OrOp::R eval(State& state, typename OrOp::A const a, typename OrOp::B const b) {
		if(AV::isNA(a)) return b ? 1 : RV::NAelement;
		else if(BV::isNA(b)) return a ? 1 : RV::NAelement;
		return (a || b) ? 1 : 0;
	}
};

#undef INVALID_COMPLEX_FUNCTION

// Fold ops

#define FOLD_OP(Name, Func, Initial) \
template<typename T> \
struct Name : FoldOp<typename T::Self> { \
	static typename Name::A base() { return Initial; } \
	static typename Name::R eval(State& state, typename Name::R const a, typename Name::A const b) { \
		return (Func); \
	} \
}; \

#define CHARACTER_FOLD_OP(Name, Func, Initial) \
template<> \
struct Name<TCharacter> : FoldOp<Character> { \
	static Name::A base() { return Initial; } \
	static Name::R eval(State& state, Name::R const a, Name::A const b) { \
		return (Func); \
	} \
}; \

#define INVALID_COMPLEX_FUNCTION(T) template<> \
struct T<TComplex> : public BinaryOp<Complex, Complex, Complex> { \
	static T::A base() { _error("invalid complex function"); } \
	static T::R eval(State& state, T::A const a, T::B const b) { _error("invalid complex function"); } \
}; 

FOLD_OP(MaxOp, PMaxOp<T>::eval(state, a, b), -std::numeric_limits<typename MaxOp<T>::A>::infinity())
CHARACTER_FOLD_OP(MaxOp, PMaxOp<TCharacter>::eval(state, a, b), Symbols::empty) 
INVALID_COMPLEX_FUNCTION(MaxOp);
FOLD_OP(MinOp, PMinOp<T>::eval(state, a, b), std::numeric_limits<typename MaxOp<T>::A>::infinity()) 
CHARACTER_FOLD_OP(MinOp, PMinOp<TCharacter>::eval(state, a, b), Symbols::empty) 
INVALID_COMPLEX_FUNCTION(MinOp);
FOLD_OP(SumOp, AddOp<T>::eval(state, a, b), 0) 
FOLD_OP(ProdOp, MulOp<T>::eval(state, a, b), 1) 
FOLD_OP(AnyOp, OrOp<TLogical>::eval(state, a, b), 0)
FOLD_OP(AllOp, AndOp<TLogical>::eval(state, a, b), 1)

#undef FOLD_OP
#undef INVALID_COMPLEX_FUNCTION

template< template<class Op> class Lift, template<typename T> class Op > 
void unaryArith(State& state, Value const& a, Value& c) {
	if(a.isDouble())
		Lift< Op<TDouble> >::eval(state, (Double const&)a, c);
	else if(a.isComplex())
		Lift< Op<TComplex> >::eval(state, (Complex const&)a, c);
	else if(a.isMathCoerce())
		Lift< Op<TInteger> >::eval(state, As<Integer>(state, a), c);
	else if(a.isNull())
		c = Null::Singleton();
	else 
		_error("non-numeric argument to unary numeric operator");
};

template< template<class Op> class Lift, template<typename T> class Op > 
void unaryLogical(State& state, Value const& a, Value& c) {
	if(a.isLogicalCoerce())
		Lift< Op<TLogical> >::eval(state, As<Logical>(state, a), c);
	else if(a.isNull())
		c = Logical(0);
	else
		_error("non-logical argument to unary logical operator");
};

template< template<class Op> class Lift, template<typename T> class Op > 
void unaryOrdinal(State& state, Value const& a, Value& c) {
	if(a.isDouble())
		Lift< Op<TDouble> >::eval(state, (Double const&)a, c);
	else if(a.isComplex())
		Lift< Op<TComplex> >::eval(state, (Complex const&)a, c);
	else if(a.isInteger())
		Lift< Op<TInteger> >::eval(state, (Integer const&)a, c);
	else if(a.isLogical())
		Lift< Op<TInteger> >::eval(state, As<Integer>(state, a), c);
	else if(a.isCharacter())
		Lift< Op<TCharacter> >::eval(state, (Character const&)a, c);
	else if(a.isNull()) {
		Double::InitScalar(c, Op<TDouble>::base());
		_warning(state, "no non-missing arguments to min; returning Inf");
	}
	else {
		printf("1: %s\n", Type::toString(a.type));
		_error("non-ordinal argument to ordinal operator");
	}
}

template< template<class Op> class Lift, template<typename T> class Op > 
void unaryCharacter(State& state, Value const& a, Value& c) {
	Lift< Op<TCharacter> >::eval(state, As<Character>(state, a), c);
};

template< template<class Op> class Lift, template<typename T> class Op > 
void unaryFilter(State& state, Value const& a, Value& c) {
	if(a.isDouble())
		Lift< Op<TDouble> >::eval(state, (Double const&)a, c);
	else if(a.isComplex())
		Lift< Op<TComplex> >::eval(state, (Complex const&)a, c);
	else if(a.isInteger())
		Lift< Op<TInteger> >::eval(state, (Integer const&)a, c);
	else if(a.isLogical())
		Lift< Op<TLogical> >::eval(state, (Logical const&)a, c);
	else if(a.isCharacter())
		Lift< Op<TCharacter> >::eval(state, (Character const&)a, c);
	else if(a.isNull())
		Logical(0);
	else if(a.isList())
		Lift< Op<TList> >::eval(state, (List const&)a, c);
};

template< template<class Op> class Lift, template<typename T> class Op > 
void binaryArithSlow(State& state, Value const& a, Value const& b, Value& c) {
	if(a.isDouble() && b.isDouble()) {
		Lift< Op<TDouble> >::eval(state, (Double const&)a, (Double const&)b, c);
	}
	else if((a.isComplex() && b.isMathCoerce()) || (b.isComplex() && a.isMathCoerce()))
		Lift< Op<TComplex> >::eval(state, As<Complex>(state, a), As<Complex>(state, b), c);
	else if((a.isDouble() && b.isMathCoerce()) || (b.isDouble() && a.isMathCoerce()))
		Lift< Op<TDouble> >::eval(state, As<Double>(state, a), As<Double>(state, b), c);
	else if(a.isMathCoerce() && b.isMathCoerce()) 
		Lift< Op<TInteger> >::eval(state, As<Integer>(state, a), As<Integer>(state, b), c);
	else 
		_error("non-numeric argument to binary numeric operator");
}

template< template<class Op> class Lift, template<typename T> class Op > 
void binaryArith(State& state, Value const& a, Value const& b, Value& c) __attribute__((always_inline));

template< template<class Op> class Lift, template<typename T> class Op > 
void binaryArith(State& state, Value const& a, Value const& b, Value& c) {
	if(a.isDouble1()) {
		if(b.isDouble1()) 
			{ Double::InitScalar(c, Op<TDouble>::eval(state, a.d, b.d)); return; }
		else if(b.isInteger1())	
			{ Double::InitScalar(c, Op<TDouble>::eval(state, a.d, (double)b.i));return; }
	}
	else if(a.isInteger1()) {
		if(b.isDouble1()) 
			{ Double::InitScalar(c, Op<TDouble>::eval(state, (double)a.i, b.d)); return; }
		else if(b.isInteger1())	
			{ Integer::InitScalar(c, Op<TInteger>::eval(state, a.i, b.i)); return;}
	}
	binaryArithSlow<Lift, Op>(state, a, b, c);
}
template< template<class Op> class Lift, template<typename T> class Op > 
void binaryLogical(State& state, Value const& a, Value const& b, Value& c) {
	if(a.isLogicalCoerce() && b.isLogicalCoerce()) 
		Lift< Op<TLogical> >::eval(state, As<Logical>(state, a), As<Logical>(state, b), c);
	else 
		_error("non-logical argument to binary logical operator");
}

template< template<class Op> class Lift, template<typename T> class Op > 
void binaryOrdinal(State& state, Value const& a, Value const& b, Value& c) {
	if((a.isComplex() && b.isMathCoerce()) || (b.isComplex() && a.isMathCoerce()))
		Lift< Op<TComplex> >::eval(state, As<Complex>(state, a), As<Complex>(state, b), c);
	else if((a.isDouble() && b.isMathCoerce()) || (b.isDouble() && a.isMathCoerce()))
		Lift< Op<TDouble> >::eval(state, As<Double>(state, a), As<Double>(state, b), c);
	else if(a.isMathCoerce() && b.isMathCoerce()) 
		Lift< Op<TInteger> >::eval(state, As<Integer>(state, a), As<Integer>(state, b),c );
	else if(a.isCharacter() && b.isCharacter())
		Lift< Op<TCharacter> >::eval(state, Character(a), Character(b), c);
	else {
		printf("2: %s\n", Type::toString(a.type));
		_error("non-ordinal argument to ordinal operator");
	}
}

// Figure out the output...
inline Type::Enum resultType(ByteCode::Enum bc, Type::Enum input) {
	switch(bc) {
#define CASE(name, str, Op) \
	case ByteCode::name: \
		if(input == Type::Integer) return Op<TInteger>::RV::VectorType; \
		else if(input == Type::Double) return Op<TDouble>::RV::VectorType; \
		else _error("Unknown type"); \
		break;
MAP_BYTECODES(CASE)
FOLD_BYTECODES(CASE)
SCAN_BYTECODES(CASE)
#undef CASE
	default:
		_error("Not a known op in resultType");
	};
} 
inline Type::Enum resultType(ByteCode::Enum bc, Type::Enum a, Type::Enum b) {
	return resultType(bc,std::max(a,b));
}

template<int Len>
inline void Sequence(int64_t start, int64_t step, int64_t* dest) {
	for(int64_t i = 0, j = start; i < Len; i++, j += step) {
		dest[i] = j;
	}
}


#endif

