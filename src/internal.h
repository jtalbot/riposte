
#ifndef _RIPOSTE_INTERNAL_H
#define _RIPOSTE_INTERNAL_H

#include "vector.h"
#include "coerce.h"
#include "exceptions.h"
#include <math.h>
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

// Unary operators

template<typename A>
struct PosOp : public UnaryOp<A, A> {
	static typename PosOp::R::Element eval(State& state, typename PosOp::A::Element const& a) { return a; }
};

template<typename A>
struct NegOp : UnaryOp<A, A> {
	static typename NegOp::R::Element eval(State& state, typename NegOp::A::Element const& a) { return -a; }
};

struct LNotOp : UnaryOp<Logical, Logical> {
	static LNotOp::R::Element eval(State& state, LNotOp::A::Element const& a) { return !a; }
};

template<typename A>
struct AbsOp : UnaryOp<A, A> {
	static typename AbsOp::R::Element eval(State& state, typename AbsOp::A::Element const& a) { return fabs(a); }
};

template<typename A>
struct SignOp : UnaryOp<A, Double> {
	static typename SignOp::R::Element eval(State& state, typename SignOp::A::Element const& a) { return a > 0 ? 1 : (a < 0 ? -1 : 0); }
};

template<typename A>
struct SqrtOp : UnaryOp<A, Double> {
	static typename SqrtOp::R::Element eval(State& state, typename SqrtOp::A::Element const& a) { return sqrt(a); }
};

template<typename A>
struct FloorOp : UnaryOp<A, Double> {
	static typename FloorOp::R::Element eval(State& state, typename FloorOp::A::Element const& a) { return floor(a); }
};

template<typename A>
struct CeilingOp : UnaryOp<A, Double> {
	static typename CeilingOp::R::Element eval(State& state, typename CeilingOp::A::Element const& a) { return ceil(a); }
};

template<typename A>
struct TruncOp : UnaryOp<A, Double> {
	static typename TruncOp::R::Element eval(State& state, typename TruncOp::A::Element const& a) { return a >= 0 ? floor(a) : ceil(a); }
};

template<typename A>
struct RoundOp : UnaryOp<A, Double> {
	static typename RoundOp::R::Element eval(State& state, typename RoundOp::A::Element const& a) { return round(a); }
};

template<typename A>
struct SignifOp : UnaryOp<A, Double> {
	static typename SignifOp::R::Element eval(State& state, typename SignifOp::A::Element const& a) { _error("NYI: signif"); }
};

template<typename A>
struct ExpOp : UnaryOp<A, Double> {
	static typename ExpOp::R::Element eval(State& state, typename ExpOp::A::Element const& a) { return exp(a); }
};

template<typename A>
struct LogOp : UnaryOp<A, Double> {
	static typename LogOp::R::Element eval(State& state, typename LogOp::A::Element const& a) { return log(a); }
};

template<typename A>
struct CosOp : UnaryOp<A, Double> {
	static typename CosOp::R::Element eval(State& state, typename CosOp::A::Element const& a) { return cos(a); }
};

template<typename A>
struct SinOp : UnaryOp<A, Double> {
	static typename SinOp::R::Element eval(State& state, typename SinOp::A::Element const& a) { return sin(a); }
};

template<typename A>
struct TanOp : UnaryOp<A, Double> {
	static typename TanOp::R::Element eval(State& state, typename TanOp::A::Element const& a) { return tan(a); }
};

template<typename A>
struct ACosOp : UnaryOp<A, Double> {
	static typename ACosOp::R::Element eval(State& state, typename ACosOp::A::Element const& a) { return acos(a); }
};

template<typename A>
struct ASinOp : UnaryOp<A, Double> {
	static typename ASinOp::R::Element eval(State& state, typename ASinOp::A::Element const& a) { return asin(a); }
};

template<typename A>
struct ATanOp : UnaryOp<A, Double> {
	static typename ATanOp::R::Element eval(State& state, typename ATanOp::A::Element const& a) { return atan(a); }
};


// Binary operators

template<typename A>
struct AddOp : BinaryOp<A, A, A> {
	static typename AddOp::R::Element eval(State& state, typename AddOp::A::Element const& a, typename AddOp::B::Element const& b) {
		return a+b;
	}
};

template<typename A>
struct SubOp : BinaryOp<A, A, A> {
	static typename SubOp::R::Element eval(State& state, typename SubOp::A::Element const& a, typename SubOp::B::Element const& b) {
		return a-b;
	}
};

template<typename A>
struct MulOp : BinaryOp<A, A, A> {
	static typename MulOp::R::Element eval(State& state, typename MulOp::A::Element const& a, typename MulOp::B::Element const& b) {
		return a*b;
	}
};

template<typename A>
struct DivOp : BinaryOp<A, A, Double> {
	static typename DivOp::R::Element eval(State& state, typename DivOp::A::Element const& a, typename DivOp::B::Element const& b) {
		return ((Double::Element)a)/b; // don't use to integer division
	}
};

inline double IDiv(double a, double b) { return floor(a/b); /* TODO: Replace with ugly R version */ }
inline int64_t IDiv(int64_t a, int64_t b) { return a/b; }

template<typename A>
struct IDivOp : BinaryOp<A, A, A> {
	static typename IDivOp::R::Element eval(State& state, typename IDivOp::A::Element const& a, typename IDivOp::B::Element const& b) {
		return IDiv(a, b);
	}
};

template<typename A>
struct PowOp : BinaryOp<A, A, Double> {
	static typename PowOp::R::Element eval(State& state, typename PowOp::A::Element const& a, typename PowOp::B::Element const& b) {
		return pow(a, b);
	}
};

inline double Mod(double a, double b) { return a - IDiv(a,b) * b; /* TODO: Replace with ugly R version */ }
inline int64_t Mod(int64_t a, int64_t b) { return a % b; }

template<typename A>
struct ModOp : BinaryOp<A, A, A> {
	static typename ModOp::R::Element eval(State& state, typename ModOp::A::Element const& a, typename ModOp::B::Element const& b) {
		return Mod(a, b);
	}
};

template<typename A>
struct LTOp : BinaryOp<A, A, Logical> {
	static typename LTOp::R::Element eval(State& state, typename LTOp::A::Element const& a, typename LTOp::B::Element const& b) {
		return a<b;
	}
};

template<typename A>
struct GTOp : BinaryOp<A, A, Logical> {
	static typename GTOp::R::Element eval(State& state, typename GTOp::A::Element const& a, typename GTOp::B::Element const& b) {
		return a>b;
	}
};

template<typename A>
struct EqOp : BinaryOp<A, A, Logical> {
	static typename EqOp::R::Element eval(State& state, typename EqOp::A::Element const& a, typename EqOp::B::Element const& b) {
		return a==b;
	}
};

template<typename A>
struct NeqOp : BinaryOp<A, A, Logical> {
	static typename NeqOp::R::Element eval(State& state, typename NeqOp::A::Element const& a, typename NeqOp::B::Element const& b) {
		return a!=b;
	}
};

template<typename A>
struct GEOp : BinaryOp<A, A, Logical> {
	static typename GEOp::R::Element eval(State& state, typename GEOp::A::Element const& a, typename GEOp::B::Element const& b) {
		return a>=b;
	}
};

template<typename A>
struct LEOp : BinaryOp<A, A, Logical> {
	static typename LEOp::R::Element eval(State& state, typename LEOp::A::Element const& a, typename LEOp::B::Element const& b) {
		return a<=b;
	}
};

struct AndOp : BinaryOp<Logical, Logical, Logical> {
	static AndOp::R::Element eval(State& state, AndOp::A::Element const& a, AndOp::B::Element const& b) {
		if(A::isNA(a)) return b ? R::NAelement : 0;
		else if(B::isNA(b)) return a ? R::NAelement : 0;
		else return a && b ? 1 : 0;
	}
};

struct OrOp : BinaryOp<Logical, Logical, Logical> {
	static OrOp::R::Element eval(State& state, OrOp::A::Element const& a, OrOp::B::Element const& b) {
		if(A::isNA(a)) return b ? 1 : R::NAelement;
		else if(B::isNA(b)) return a ? 1 : R::NAelement;
		return (a || b) ? 1 : 0;
	}
};

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

template< template<class Op> class Lift, template<typename A> class Op > 
void unaryArith(State& state, Value const& a, Value& c) {
	if(a.isDouble())
		c = Lift< NA1< Op<Double> > >::eval(state, Double(a));
	else if(a.isMathCoerce())
		c = Lift< NA1< Op<Integer> > >::eval(state, As<Integer>(state, a));
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

template< template<class Op> class Lift, template<typename A> class Op > 
void binaryArith(State& state, Value const& a, Value const& b, Value& c) {
	if((a.isDouble() && b.isMathCoerce()) || (b.isDouble() && a.isMathCoerce()))
		c = Lift< NA2< Op<Double> > >::eval(state, As<Double>(state, a), As<Double>(state, b));
	else if(a.isMathCoerce() && b.isMathCoerce()) 
		c = Lift< NA2< Op<Integer> > >::eval(state, As<Integer>(state, a), As<Integer>(state, b));
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

template< template<class Op> class Lift, template<typename A> class Op > 
void binaryOrdinal(State& state, Value const& a, Value const& b, Value& c) {
	if((a.isDouble() && b.isMathCoerce()) || (b.isDouble() && a.isMathCoerce()))
		c = Lift< NA2< Op<Double> > >::eval(state, As<Double>(state, a), As<Double>(state, b));
	else if(a.isMathCoerce() && b.isMathCoerce()) 
		c = Lift< NA2< Op<Integer> > >::eval(state, a, b);
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

