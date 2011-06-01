
#ifndef _RIPOSTE_INTERNAL_H
#define _RIPOSTE_INTERNAL_H

#include "vector.h"
#include "coerce.h"
#include "exceptions.h"
#include <cmath>
#include <algorithm>
#include <set>

void addMathOps(State& state);

inline Value force(State& state, Value v) { 
	while(v.type == Type::I_promise) {
		eval(state, Closure(v)); 
		v = state.registers[0]; 
	} 
	return v; 
}
inline Value expression(Value const& v) { 
	if(v.type == Type::I_promise)
		return Closure(v).expression();
	else return v; 
}

inline double asReal1(Value const& v) { if(v.type != Type::R_double && v.type != Type::R_integer) _error("Can't cast argument to number"); if(v.type == Type::R_integer) return Integer(v)[0]; else return Double(v)[0]; }

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

// Unary operators
#define UNARY_OP(Name, T1, T2, Func) \
template<typename T> \
struct Name : public UnaryOp<typename T::T1, typename T::T2> {\
	static typename Name::R eval(State& state, typename Name::A const& a) {\
		if(!Name::AV::CheckNA || !Name::AV::isNA(a)) return (Func);\
		else return Name::RV::NAelement;\
	}\
};

#define NYI_UNARY_OP(Name, T1, T2) \
template<typename T> \
struct Name : public UnaryOp<typename T::T1, typename T::T2> {\
	static typename Name::R eval(State& state, typename Name::A const& a) { _error("NYI: "#Name); } \
};

#define NYI_COMPLEX_OP(T) template<> \
struct T<TComplex> : public UnaryOp<TComplex::Self, TComplex::Self> { \
	static T::R eval(State& state, T::A const& a) { _error("unimplemented complex function"); } \
};

UNARY_OP(PosOp, Self, Self, a)
UNARY_OP(NegOp, Self, Self, -a)
UNARY_OP(AbsOp, Self, Self, abs(a))
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

struct LNotOp : UnaryOp<Logical, Logical> {
	static LNotOp::R eval(State& state, LNotOp::A const& a) { return !a; }
};

#undef UNARY_OP 
#undef NYI_UNARY_OP
#undef NYI_COMPLEX_OP

// Binary operators
#define BINARY_OP(Name, T1, T2, T3, Func) \
template<typename T> \
struct Name : public BinaryOp<typename T::T1, typename T::T2, typename T::T3> {\
	static typename Name::R eval(State& state, typename Name::A const& a, typename Name::B const& b) { \
		if((!Name::AV::CheckNA || !Name::AV::isNA(a)) && (!Name::BV::CheckNA || !Name::BV::isNA(b))) return (Func); \
		else return Name::RV::NAelement; \
	} \
};

#define NYI_COMPLEX_OP(T) template<> \
struct T<TComplex> : public BinaryOp<Complex, Complex, Complex> { \
	static T::R eval(State& state, T::A const& a, T::B const& b) { _error("unimplemented complex function"); } \
};

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

#undef BINARY_OP
#undef NYI_COMPLEX_OP

// Ordinal binary ops
#define ORDINAL_OP(Name, Func) \
template<typename T> \
struct Name : public BinaryOp<typename T::Self, typename T::Self, Logical> {\
	static typename Name::R eval(State& state, typename Name::A const& a, typename Name::B const& b) { \
		if(!Name::AV::isNA(a) && !Name::BV::isNA(b)) return (Func); \
		else return Name::RV::NAelement; \
	} \
};\

#define INVALID_COMPLEX_OP(Name) \
template<> \
struct Name<TComplex> : public BinaryOp<Complex, Complex, Logical> { \
	static Name::R eval(State& state, Name::A const& a, Name::B const& b) { _error("invalid complex function"); } \
};

ORDINAL_OP(LTOp, a<b)						INVALID_COMPLEX_OP(LTOp)
ORDINAL_OP(GTOp, a>b)						INVALID_COMPLEX_OP(GTOp)
ORDINAL_OP(LEOp, a<=b)						INVALID_COMPLEX_OP(LEOp)
ORDINAL_OP(GEOp, a>=b)						INVALID_COMPLEX_OP(GEOp)
ORDINAL_OP(EqOp, a==b)
ORDINAL_OP(NeqOp, a!=b)

#undef ORDINAL_OP
#undef INVALID_COMPLEX_OP

// Logical binary ops
struct AndOp : BinaryOp<Logical, Logical, Logical> {
	static AndOp::R eval(State& state, AndOp::A const& a, AndOp::B const& b) {
		if(AV::isNA(a)) return b ? RV::NAelement : 0;
		else if(BV::isNA(b)) return a ? RV::NAelement : 0;
		else return a && b ? 1 : 0;
	}
};

struct OrOp : BinaryOp<Logical, Logical, Logical> {
	static OrOp::R eval(State& state, OrOp::A const& a, OrOp::B const& b) {
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
	static const typename Name::A Base; \
	static typename Name::R eval(State& state, typename Name::R const& a, typename Name::A const& b) { \
		if(!Name::AV::CheckNA || !Name::AV::isNA(b)) return (Func); \
		else return Name::RV::NAelement; \
	} \
}; \
template<typename T> \
const typename Name<T>::A Name<T>::Base = Initial; 

#define INVALID_COMPLEX_FUNCTION(T) template<> \
struct T<TComplex> : public BinaryOp<Complex, Complex, Complex> { \
	static const T::A Base; \
	static T::R eval(State& state, T::A const& a, T::B const& b) { _error("invalid complex function"); } \
}; 

FOLD_OP(MaxOp, std::max(a,b), std::numeric_limits<typename MaxOp<T>::A>::min()) 
INVALID_COMPLEX_FUNCTION(MaxOp);
FOLD_OP(MinOp, std::min(a,b), std::numeric_limits<typename MaxOp<T>::A>::max()) 
INVALID_COMPLEX_FUNCTION(MinOp);
FOLD_OP(SumOp, a+b, 0) 
FOLD_OP(ProdOp, a*b, 1) 

#undef FOLD_OP
#undef INVALID_COMPLEX_FUNCTION

template< class A >
struct SubsetInclude {
	static A eval(State& state, A const& a, Integer const& d, uint64_t nonzero)
	{
		A r(nonzero);
		uint64_t j = 0;
		for(uint64_t i = 0; i < d.length; i++) {
			if(Integer::isNA(d[i])) r[j++] = A::NAelement;	
			else if(d[i] != 0) r[j++] = a[d[i]-1];
		}
		return r;
	}
};

template< class A >
struct SubsetExclude {
	static A eval(State& state, A const& a, Integer const& d, uint64_t nonzero)
	{
		std::set<Integer::Element> index; 
		for(uint64_t i = 0; i < d.length; i++) if(-d[i] > 0 && -d[i] < (int64_t)a.length) index.insert(-d[i]);
		// iterate through excluded elements copying intervening ranges.
		A r(a.length-index.size());
		uint64_t start = 1;
		uint64_t k = 0;
		for(std::set<Integer::Element>::const_iterator i = index.begin(); i != index.end(); ++i) {
			uint64_t end = *i;
			for(uint64_t j = start; j < end; j++) r[k++] = a[j-1];
			start = end+1;
		}
		for(uint64_t j = start; j <= a.length; j++) r[k++] = a[j-1];
		return r;
	}
};

template< class A >
struct SubsetLogical {
	static A eval(State& state, A const& a, Logical const& d)
	{
		// determine length
		uint64_t length = 0;
		if(d.length > 0) {
			uint64_t j = 0;
			for(uint64_t i = 0; i < std::max(a.length, d.length); i++) {
				if(!Logical::isFalse(d[j])) length++;
				if(++j >= d.length) j = 0;
			}
		}
		A r(length);
		uint64_t j = 0, k = 0;
		for(uint64_t i = 0; i < std::max(a.length, d.length) && k < length; i++) {
			if(i >= a.length || Logical::isNA(d[j])) r[k++] = A::NAelement;
			else if(Logical::isTrue(d[j])) r[k++] = a[i];
			if(++j >= d.length) j = 0;
		}
		return r;
	}
};


template< class A  >
struct SubsetAssign {
	static A eval(State& state, A const& a, Integer const& d, A const& b)
	{
		// compute max index 
		int64_t outlength = 0;
		for(uint64_t i = 0; i < d.length; i++) {
			outlength = std::max((int64_t)outlength, d[i]);
		}

		// should use max index here to extend vector if necessary	
		A r = a;//Clone(a);	
		for(uint64_t i = 0; i < d.length; i++) {	
			int64_t idx = d[i];
			if(idx != 0)
				r[idx-1] = b[i];
		}
		return r;
	}
};

template< template<class Op> class Lift, template<typename T> class Op > 
void unaryArith(State& state, Value const& a, Value& c) {
	if(a.isDouble())
		c = Lift< Op<TDouble> >::eval(state, Double(a));
	else if(a.isComplex())
		c = Lift< Op<TComplex> >::eval(state, Complex(a));
	else if(a.isMathCoerce())
		c = Lift< Op<TInteger> >::eval(state, As<Integer>(state, a));
	else 
		_error("non-numeric argument to unary numeric operator");
};

template< template<class Op> class Lift, class Op > 
void unaryLogical(State& state, Value const& a, Value& c) {
	if(a.isLogicalCoerce())
		c = Lift<Op>::eval(state, As<Logical>(state, a));
	else
		_error("non-logical argument to unary logical operator");
};

template< template<class Op> class Lift, template<typename T> class Op > 
void binaryArith(State& state, Value const& a, Value const& b, Value& c) {
	if((a.isComplex() && b.isMathCoerce()) || (b.isComplex() && a.isMathCoerce()))
		c = Lift< Op<TComplex> >::eval(state, As<Complex>(state, a), As<Complex>(state, b));
	else if((a.isDouble() && b.isMathCoerce()) || (b.isDouble() && a.isMathCoerce()))
		c = Lift< Op<TDouble> >::eval(state, As<Double>(state, a), As<Double>(state, b));
	else if(a.isMathCoerce() && b.isMathCoerce()) 
		c = Lift< Op<TInteger> >::eval(state, As<Integer>(state, a), As<Integer>(state, b));
	else 
		_error("non-numeric argument to binary numeric operator");
}

template< template<class Op> class Lift, class Op > 
void binaryLogical(State& state, Value const& a, Value const& b, Value& c) {
	if(a.isLogicalCoerce() && b.isLogicalCoerce()) 
		c = Lift<Op>::eval(state, As<Logical>(state, a), As<Logical>(state, b));
	else 
		_error("non-logical argument to binary logical operator");
}

template< template<class Op> class Lift, template<typename T> class Op > 
void binaryOrdinal(State& state, Value const& a, Value const& b, Value& c) {
	if((a.isComplex() && b.isMathCoerce()) || (b.isComplex() && a.isMathCoerce()))
		c = Lift< Op<TComplex> >::eval(state, As<Complex>(state, a), As<Complex>(state, b));
	else if((a.isDouble() && b.isMathCoerce()) || (b.isDouble() && a.isMathCoerce()))
		c = Lift< Op<TDouble> >::eval(state, As<Double>(state, a), As<Double>(state, b));
	else if(a.isMathCoerce() && b.isMathCoerce()) 
		c = Lift< Op<TInteger> >::eval(state, a, b);
	else
		_error("non-ordinal argument to ordinal operator");
}

inline void subAssign(State& state, Value const& a, Value const& i, Value const& b, Value& c) {
	Integer idx = As<Integer>(state, i);
	if(a.isDouble()) c = SubsetAssign<Double>::eval(state, a, idx, As<Double>(state, b));
	else if(a.isInteger()) c = SubsetAssign<Integer>::eval(state, a, idx, As<Integer>(state, b));
	else if(a.isLogical()) c = SubsetAssign<Logical>::eval(state, a, idx, As<Logical>(state, b));
	else if(a.isCharacter()) c = SubsetAssign<Character>::eval(state, a, idx, As<Character>(state, b));
	else _error("NYI: subset assign type");
}

inline void Insert(State& state, Vector const& src, uint64_t srcIndex, Vector& dst, uint64_t dstIndex, uint64_t length) {
	if(srcIndex+length > src.length || dstIndex+length > dst.length)
		_error("insert index out of bounds");
	Vector as = As(state, dst.type, src);
	memcpy(dst.data(dstIndex), as.data(srcIndex), length*as.width);
}

inline Vector Subset(Vector const& src, uint64_t start, uint64_t length) {
	if(start+length > src.length)
		_error("subset index out of bounds");
	Vector v(src.type, length);
	memcpy(v.data(0), src.data(start), length*src.width);
	return v;
}

inline Double Sequence(double from, double by, double len) {
	Double r(len);
	double j = 0;
	for(uint64_t i = 0; i < len; i++) {
		r[i] = from+j;
		j = j + by;
	}
	return r;
}

inline Vector Element(Vector const& src, uint64_t index)
{
	return Subset(src, index, 1);
}

inline Value Element2(Vector const& src, uint64_t index)
{
	if(src.type == Type::R_list) return List(src)[index];
	else return Subset(src, index, 1);
}

#endif

