
#ifndef _RIPOSTE_INTERNAL_H
#define _RIPOSTE_INTERNAL_H

#include "value.h"
#include <assert.h>
#include <math.h>

void addMathOps(State& state);


inline Value force(State& state, Value const& v) { 
	if(v.type == Type::I_promise) {
		eval(state, Block(v), state.env); 
		return state.stack.pop();
	} else if(v.type == Type::I_sympromise) {
		Value value;
		state.env->get(state, v.i, value);
		return value;
	} else return v; 
}
inline Value quoted(Value const& v) { 
	if(v.type == Type::I_promise || v.type == Type::I_sympromise) 
		return Block(v).expression();
	 else return v; 
}
inline Value code(Value const& v) {
	return v; 
}

// Casting functions (default is to attempt a C coercion)
template<class I, class O> struct Cast {
	static typename O::Element eval(typename I::Element const& i) { return (typename O::Element)i; }
};

// More involved casting functions
template<> struct Cast<Logical, Double> {
	static Double::Element eval(Logical::Element const& i) { return i ? 1.0 : 0.0; }
};
template<> struct Cast<Logical, Integer> {
	static Integer::Element eval(Logical::Element const& i) { return i ? 1 : 0; }
};
template<> struct Cast<Double, Logical> {
	static Logical::Element eval(Double::Element const& i) { return i != 0.0 ? 1 : 0; }
};
template<> struct Cast<Integer, Logical> {
	static Logical::Element eval(Integer::Element const& i) { return i != 0 ? 1 : 0; }
};


template<class X, class Y, class Z>
struct UnaryOp {
	typedef X A;
	typedef Y TR;
	typedef Z R;
	
};

template<typename A, typename TA, typename R>
struct PosOp : UnaryOp<A, TA, R> {
	static typename R::Element eval(typename A::Element const& a) {
		return Cast<A, TA>::eval(a);
	}
};

template<typename A, typename TA, typename R>
struct NegOp : UnaryOp<A, TA, R> {
	static typename R::Element eval(typename A::Element const& a) {
		return -Cast<A, TA>::eval(a);
	}
};

template<typename A, typename TA, typename R>
struct LNegOp : UnaryOp<A, TA, R> {
	static typename R::Element eval(typename A::Element const& a) {
		return !Cast<A, TA>::eval(a);
	}
};

template<typename A, typename TA, typename R>
struct AbsOp : UnaryOp<A, TA, R> {
	static typename R::Element eval(typename A::Element const& a) {
		return fabs(Cast<A, TA>::eval(a));
	}
};

template<typename A, typename TA, typename R>
struct SignOp : UnaryOp<A, TA, R> {
	static typename R::Element eval(typename A::Element const& a) {
		typename TA::Element ta = Cast<A, TA>::eval(a);
		return ta > 0 ? 1 : (ta < 0 ? -1 : 0);
	}
};

template<typename A, typename TA, typename R>
struct SqrtOp : UnaryOp<A, TA, R> {
	static typename R::Element eval(typename A::Element const& a) {
		return sqrt(Cast<A, TA>::eval(a));
	}
};

template<typename A, typename TA, typename R>
struct FloorOp : UnaryOp<A, TA, R> {
	static typename R::Element eval(typename A::Element const& a) {
		return floor(Cast<A, TA>::eval(a));
	}
};

template<typename A, typename TA, typename R>
struct CeilingOp : UnaryOp<A, TA, R> {
	static typename R::Element eval(typename A::Element const& a) {
		return ceil(Cast<A, TA>::eval(a));
	}
};

template<typename A, typename TA, typename R>
struct TruncOp : UnaryOp<A, TA, R> {
	static typename R::Element eval(typename A::Element const& a) {
		typename TA::Element ta = Cast<A, TA>::eval(a);
		return ta >= 0 ? floor(ta) : ceil(ta);
	}
};

template<typename A, typename TA, typename R>
struct RoundOp : UnaryOp<A, TA, R> {
	static typename R::Element eval(typename A::Element const& a) {
		return round(Cast<A, TA>::eval(a));
	}
};

template<typename A, typename TA, typename R>
struct SignifOp : UnaryOp<A, TA, R> {
	static typename R::Element eval(typename A::Element const& a) {
		// NYI
		return round(Cast<A, TA>::eval(a));
	}
};

template<typename A, typename TA, typename R>
struct ExpOp : UnaryOp<A, TA, R> {
	static typename R::Element eval(typename A::Element const& a) {
		return exp(Cast<A, TA>::eval(a));
	}
};

template<typename A, typename TA, typename R>
struct LogOp : UnaryOp<A, TA, R> {
	static typename R::Element eval(typename A::Element const& a) {
		return log(Cast<A, TA>::eval(a));
	}
};

template<typename A, typename TA, typename R>
struct CosOp : UnaryOp<A, TA, R> {
	static typename R::Element eval(typename A::Element const& a) {
		return cos(Cast<A, TA>::eval(a));
	}
};

template<typename A, typename TA, typename R>
struct SinOp : UnaryOp<A, TA, R> {
	static typename R::Element eval(typename A::Element const& a) {
		return sin(Cast<A, TA>::eval(a));
	}
};

template<typename A, typename TA, typename R>
struct TanOp : UnaryOp<A, TA, R> {
	static typename R::Element eval(typename A::Element const& a) {
		return tan(Cast<A, TA>::eval(a));
	}
};

template<typename A, typename TA, typename R>
struct ACosOp : UnaryOp<A, TA, R> {
	static typename R::Element eval(typename A::Element const& a) {
		return acos(Cast<A, TA>::eval(a));
	}
};

template<typename A, typename TA, typename R>
struct ASinOp : UnaryOp<A, TA, R> {
	static typename R::Element eval(typename A::Element const& a) {
		return asin(Cast<A, TA>::eval(a));
	}
};

template<typename A, typename TA, typename R>
struct ATanOp : UnaryOp<A, TA, R> {
	static typename R::Element eval(typename A::Element const& a) {
		return atan(Cast<A, TA>::eval(a));
	}
};



// Binary operators

template<typename X, typename TX, typename Y, typename TY, typename Z> struct BinaryOp {
	typedef X A;
	typedef Y B;
	typedef Z R;
};

template<typename A, typename TA, typename B, typename TB, typename R>
struct AddOp : BinaryOp<A, TA, B, TB, R> {
	static typename R::Element eval(typename A::Element const& a, typename B::Element const& b) {
		return Cast<A, TA>::eval(a) + Cast<B, TB>::eval(b);
	}
};

template<typename A, typename TA, typename B, typename TB, typename R>
struct SubOp : BinaryOp<A, TA, B, TB, R> {
	static typename R::Element eval(typename A::Element const& a, typename B::Element const& b) {
		return Cast<A, TA>::eval(a) - Cast<B, TB>::eval(b);
	}
};

template<typename A, typename TA, typename B, typename TB, typename R>
struct MulOp : BinaryOp<A, TA, B, TB, R> {
	static typename R::Element eval(typename A::Element const& a, typename B::Element const& b) {
		return Cast<A, TA>::eval(a) * Cast<B, TB>::eval(b);
	}
};

template<typename A, typename TA, typename B, typename TB, typename R>
struct DivOp : BinaryOp<A, TA, B, TB, R> {
	static typename R::Element eval(typename A::Element const& a, typename B::Element const& b) {
		return Cast<A, TA>::eval(a) / Cast<B, TB>::eval(b);
	}
};

inline double IDiv(double a, double b) { return floor(a/b); /* TODO: Replace with ugly R version */ }
inline int64_t IDiv(int64_t a, int64_t b) { return a/b; }

template<typename A, typename TA, typename B, typename TB, typename R>
struct IDivOp : BinaryOp<A, TA, B, TB, R> {
	static typename R::Element eval(typename A::Element const& a, typename B::Element const& b) {
		return IDiv( Cast<A, TA>::eval(a), Cast<B, TB>::eval(b));
	}
};

template<typename A, typename TA, typename B, typename TB, typename R>
struct PowOp : BinaryOp<A, TA, B, TB, R> {
	static typename R::Element eval(typename A::Element const& a, typename B::Element const& b) {
		return pow( Cast<A, TA>::eval(a), Cast<B, TB>::eval(b));
	}
};

inline double Mod(double a, double b) { return a - IDiv(a,b) * b; /* TODO: Replace with ugly R version */ }
inline int64_t Mod(int64_t a, int64_t b) { return a % b; }

template<typename A, typename TA, typename B, typename TB, typename R>
struct ModOp : BinaryOp<A, TA, B, TB, R> {
	static typename R::Element eval(typename A::Element const& a, typename B::Element const& b) {
		return Mod( Cast<A, TA>::eval(a), Cast<B, TB>::eval(b));
	}
};

template<typename A, typename TA, typename B, typename TB, typename R>
struct LTOp : BinaryOp<A, TA, B, TB, R> {
	static typename R::Element eval(typename A::Element const& a, typename B::Element const& b) {
		return Cast<A, TA>::eval(a) < Cast<B, TB>::eval(b);
	}
};

template<typename A, typename TA, typename B, typename TB, typename R>
struct GTOp : BinaryOp<A, TA, B, TB, R> {
	static typename R::Element eval(typename A::Element const& a, typename B::Element const& b) {
		return Cast<A, TA>::eval(a) > Cast<B, TB>::eval(b);
	}
};

template<typename A, typename TA, typename B, typename TB, typename R>
struct EqOp : BinaryOp<A, TA, B, TB, R> {
	static typename R::Element eval(typename A::Element const& a, typename B::Element const& b) {
		return Cast<A, TA>::eval(a) == Cast<B, TB>::eval(b);
	}
};

template<typename A, typename TA, typename B, typename TB, typename R>
struct NeqOp : BinaryOp<A, TA, B, TB, R> {
	static typename R::Element eval(typename A::Element const& a, typename B::Element const& b) {
		return Cast<A, TA>::eval(a) != Cast<B, TB>::eval(b);
	}
};

template<typename A, typename TA, typename B, typename TB, typename R>
struct GEOp : BinaryOp<A, TA, B, TB, R> {
	static typename R::Element eval(typename A::Element const& a, typename B::Element const& b) {
		return Cast<A, TA>::eval(a) >= Cast<B, TB>::eval(b);
	}
};

template<typename A, typename TA, typename B, typename TB, typename R>
struct LEOp : BinaryOp<A, TA, B, TB, R> {
	static typename R::Element eval(typename A::Element const& a, typename B::Element const& b) {
		return Cast<A, TA>::eval(a) <= Cast<B, TB>::eval(b);
	}
};

template<typename A, typename TA, typename B, typename TB, typename R>
struct AndOp : BinaryOp<A, TA, B, TB, R> {
	static typename R::Element eval(typename A::Element const& a, typename B::Element const& b) {
		return Cast<A, TA>::eval(a) && Cast<B, TB>::eval(b);
	}
};

template<typename A, typename TA, typename B, typename TB, typename R>
struct OrOp : BinaryOp<A, TA, B, TB, R> {
	static typename R::Element eval(typename A::Element const& a, typename B::Element const& b) {
		return Cast<A, TA>::eval(a) || Cast<B, TB>::eval(b);
	}
};


template< class Op >
struct Zip1 {
	static typename Op::R eval(typename Op::A const& a)
	{
		typename Op::R r = typename Op::R(a.length());
		for(uint64_t i = 0; i < a.length(); ++i) {
			r[i] = Op::eval(a[i]);
		}
		return r;
	}
};

/*
template<UnaryOp func>
struct ZipInto {
	static void eval(void* in, uint64_t in_index, void* out, uint64_t out_index, uint64_t length)
	{
		for(uint64_t i = 0; i < length; ++i) {
			func(in, in_index+i, out, out_index+i);
		}
	}
};
*/

template< class Op >
struct Zip2 {
	static typename Op::R eval(typename Op::A const& a, typename Op::B const& b)
	{
		if(a.length() == b.length()) {
			typename Op::R r(a.length());
			for(uint64_t i = 0; i < a.length(); ++i) {
				r[i] = Op::eval(a[i], b[i]);
			}
			return r;
		}
		else if(a.length() == 0 || b.length() == 0) {
			return typename Op::R(0);
		}
		else if(b.length() == 1) {
			typename Op::R r(a.length());
			for(uint64_t i = 0; i < a.length(); ++i) {
				r[i] = Op::eval(a[i], b[0]);
			}
			return r;
		}
		else if(a.length() == 1) {
			typename Op::R r(b.length());
			for(uint64_t i = 0; i < b.length(); ++i) {
				r[i] = Op::eval(a[0], b[i]);
			}
			return r;
		}
		else if(a.length() > b.length()) {
			typename Op::R r(a.length());
			uint64_t j = 0;
			for(uint64_t i = 0; i < a.length(); ++i) {
				r[i] = Op::eval(a[i], b[j]);
				++j;
				if(j >= b.length()) j = 0;
			}
			return r;
		}
		else {
			typename Op::R r(b.length());
			uint64_t j = 0;
			for(uint64_t i = 0; i < b.length(); ++i) {
				r[i] = Op::eval(a[j], b[i]);
				++j;
				if(j >= a.length()) j = 0;
			}
			return r;
		}
	}
};

template< 
	template<class Op> class Lift,
	template<typename A, typename TA, typename R> class Op > 
uint64_t unaryArith(State& state, uint64_t nargs) {

	assert(nargs == 1);
	
	Stack& stack = state.stack;
	
	Value a = force(state, stack.pop());	

	Vector r;
	if(a.type == Type::R_double) {
		Lift< Op<Double, Double, Double> >::eval(a).toVector(r);
	}
	else if(a.type == Type::R_integer) {
		Lift< Op<Integer, Integer, Integer> >::eval(a).toVector(r);
	}
	else if(a.type == Type::R_logical) {
		Lift< Op<Logical, Integer, Integer> >::eval(a).toVector(r);
	}
	else {
		printf("non-numeric argument to unary numeric operator\n");
		assert(false);
	}
	Value& v = stack.reserve();
	r.toValue(v);
	return 1;
};

template< 
	template<class Op> class Lift,
	template<typename A, typename TA, typename R> class Op > 
uint64_t unaryLogical(State& state, uint64_t nargs) {

	assert(nargs == 1);
	
	Stack& stack = state.stack;
	
	Value a = force(state, stack.pop());	

	Vector r;
	if(a.type == Type::R_double) {
		Lift< Op<Double, Logical, Logical> >::eval(a).toVector(r);
	}
	else if(a.type == Type::R_integer) {
		Lift< Op<Integer, Logical, Logical> >::eval(a).toVector(r);
	}
	else if(a.type == Type::R_logical) {
		Lift< Op<Logical, Logical, Logical> >::eval(a).toVector(r);
	}
	else {
		printf("non-numeric argument to unary logical operator\n");
		assert(false);
	}
	Value& v = stack.reserve();
	r.toValue(v);
	return 1;
};


template< 
	template<class Op> class Lift,
	template<typename A, typename TA, typename B, typename TB, typename R> class Op > 
uint64_t binaryArith(State& state, uint64_t nargs) {

	assert(nargs == 2);
	
	Stack& stack = state.stack;
	
	Value a = force(state, stack.pop());	
	Value b = force(state, stack.pop());	

	Vector r;
	if(a.type == Type::R_double && b.type == Type::R_double) {
		Lift< Op<Double, Double, Double, Double, Double> >::eval(a, b).toVector(r);
	}
	else if(a.type == Type::R_integer && b.type == Type::R_double) {
		Lift< Op<Integer, Double, Double, Double, Double> >::eval(a, b).toVector(r);
	}
	else if(a.type == Type::R_double && b.type == Type::R_integer) {
		Lift< Op<Double, Double, Integer, Double, Double> >::eval(a, b).toVector(r);
	}
	else if(a.type == Type::R_integer && b.type == Type::R_integer) {
		Lift< Op<Integer, Integer, Integer, Integer, Integer> >::eval(a, b).toVector(r);
	}
	else if(a.type == Type::R_double && b.type == Type::R_logical) {
		Lift< Op<Double, Double, Logical, Double, Double> >::eval(a, b).toVector(r);
	}
	else if(a.type == Type::R_logical && b.type == Type::R_double) {
		Lift< Op<Logical, Double, Double, Double, Double> >::eval(a, b).toVector(r);
	}
	else if(a.type == Type::R_integer && b.type == Type::R_logical) {
		Lift< Op<Integer, Integer, Logical, Integer, Integer> >::eval(a, b).toVector(r);
	}
	else if(a.type == Type::R_logical && b.type == Type::R_integer) {
		Lift< Op<Logical, Integer, Integer, Integer, Integer> >::eval(a, b).toVector(r);
	}
	else if(a.type == Type::R_logical && b.type == Type::R_logical) {
		Lift< Op<Logical, Integer, Logical, Integer, Integer> >::eval(a, b).toVector(r);
	}
	else {
		printf("non-numeric argument to binary numeric operator\n");
		assert(false);
	}
	Value& v = stack.reserve();
	r.toValue(v);
	return 1;
}

template< 
	template<class Op> class Lift,
	template<typename A, typename TA, typename B, typename TB, typename R> class Op > 
uint64_t binaryDoubleArith(State& state, uint64_t nargs) {

	assert(nargs == 2);
	
	Stack& stack = state.stack;
	
	Value a = force(state, stack.pop());	
	Value b = force(state, stack.pop());	

	Vector r;
	if(a.type == Type::R_double && b.type == Type::R_double) {
		Lift< Op<Double, Double, Double, Double, Double> >::eval(a, b).toVector(r);
	}
	else if(a.type == Type::R_integer && b.type == Type::R_double) {
		Lift< Op<Integer, Double, Double, Double, Double> >::eval(a, b).toVector(r);
	}
	else if(a.type == Type::R_double && b.type == Type::R_integer) {
		Lift< Op<Double, Double, Integer, Double, Double> >::eval(a, b).toVector(r);
	}
	else if(a.type == Type::R_integer && b.type == Type::R_integer) {
		Lift< Op<Integer, Double, Integer, Double, Double> >::eval(a, b).toVector(r);
	}
	else if(a.type == Type::R_double && b.type == Type::R_logical) {
		Lift< Op<Double, Double, Logical, Double, Double> >::eval(a, b).toVector(r);
	}
	else if(a.type == Type::R_logical && b.type == Type::R_double) {
		Lift< Op<Logical, Double, Double, Double, Double> >::eval(a, b).toVector(r);
	}
	else if(a.type == Type::R_integer && b.type == Type::R_logical) {
		Lift< Op<Integer, Double, Logical, Double, Double> >::eval(a, b).toVector(r);
	}
	else if(a.type == Type::R_logical && b.type == Type::R_integer) {
		Lift< Op<Logical, Double, Integer, Double, Double> >::eval(a, b).toVector(r);
	}
	else if(a.type == Type::R_logical && b.type == Type::R_logical) {
		Lift< Op<Logical, Double, Logical, Double, Double> >::eval(a, b).toVector(r);
	}
	else {
		printf("non-numeric argument to numeric operator\n");
		assert(false);
	}
	Value& v = stack.reserve();
	r.toValue(v);
	return 1;
}

template< 
	template<class Op> class Lift,
	template<typename A, typename TA, typename B, typename TB, typename R> class Op > 
uint64_t binaryLogical(State& state, uint64_t nargs) {

	assert(nargs == 2);
	
	Stack& stack = state.stack;
	
	Value a = force(state, stack.pop());	
	Value b = force(state, stack.pop());	

	Vector r;
	if(a.type == Type::R_double && b.type == Type::R_double) {
		Lift< Op<Double, Logical, Double, Logical, Logical> >::eval(a, b).toVector(r);
	}
	else if(a.type == Type::R_integer && b.type == Type::R_double) {
		Lift< Op<Integer, Logical, Double, Logical, Logical> >::eval(a, b).toVector(r);
	}
	else if(a.type == Type::R_double && b.type == Type::R_integer) {
		Lift< Op<Double, Logical, Integer, Logical, Logical> >::eval(a, b).toVector(r);
	}
	else if(a.type == Type::R_integer && b.type == Type::R_integer) {
		Lift< Op<Integer, Logical, Integer, Logical, Logical> >::eval(a, b).toVector(r);
	}
	else if(a.type == Type::R_double && b.type == Type::R_logical) {
		Lift< Op<Double, Logical, Logical, Logical, Logical> >::eval(a, b).toVector(r);
	}
	else if(a.type == Type::R_logical && b.type == Type::R_double) {
		Lift< Op<Logical, Logical, Double, Logical, Logical> >::eval(a, b).toVector(r);
	}
	else if(a.type == Type::R_integer && b.type == Type::R_logical) {
		Lift< Op<Integer, Logical, Logical, Logical, Logical> >::eval(a, b).toVector(r);
	}
	else if(a.type == Type::R_logical && b.type == Type::R_integer) {
		Lift< Op<Logical, Logical, Integer, Logical, Logical> >::eval(a, b).toVector(r);
	}
	else if(a.type == Type::R_logical && b.type == Type::R_logical) {
		Lift< Op<Logical, Logical, Logical, Logical, Logical> >::eval(a, b).toVector(r);
	}
	else {
		printf("non-logical argument to logical operator\n");
		assert(false);
	}
	Value& v = stack.reserve();
	r.toValue(v);
	return 1;
}

template< 
	template<class Op> class Lift,
	template<typename A, typename TA, typename B, typename TB, typename R> class Op > 
uint64_t binaryOrdinal(State& state, uint64_t nargs) {

	assert(nargs == 2);
	
	Stack& stack = state.stack;
	
	Value a = force(state, stack.pop());	
	Value b = force(state, stack.pop());	

	Vector r;
	if(a.type == Type::R_double && b.type == Type::R_double) {
		Lift< Op<Double, Double, Double, Double, Logical> >::eval(a, b).toVector(r);
	}
	else if(a.type == Type::R_integer && b.type == Type::R_double) {
		Lift< Op<Integer, Double, Double, Double, Logical> >::eval(a, b).toVector(r);
	}
	else if(a.type == Type::R_double && b.type == Type::R_integer) {
		Lift< Op<Double, Double, Integer, Double, Logical> >::eval(a, b).toVector(r);
	}
	else if(a.type == Type::R_integer && b.type == Type::R_integer) {
		Lift< Op<Integer, Integer, Integer, Integer, Logical> >::eval(a, b).toVector(r);
	}
	else if(a.type == Type::R_double && b.type == Type::R_logical) {
		Lift< Op<Double, Double, Logical, Double, Logical> >::eval(a, b).toVector(r);
	}
	else if(a.type == Type::R_logical && b.type == Type::R_double) {
		Lift< Op<Logical, Double, Double, Double, Logical> >::eval(a, b).toVector(r);
	}
	else if(a.type == Type::R_integer && b.type == Type::R_logical) {
		Lift< Op<Integer, Integer, Logical, Integer, Logical> >::eval(a, b).toVector(r);
	}
	else if(a.type == Type::R_logical && b.type == Type::R_integer) {
		Lift< Op<Logical, Integer, Integer, Integer, Logical> >::eval(a, b).toVector(r);
	}
	else if(a.type == Type::R_logical && b.type == Type::R_logical) {
		Lift< Op<Logical, Logical, Logical, Logical, Logical> >::eval(a, b).toVector(r);
	}
	else {
		printf("non-ordinal argument to ordinal operator\n");
		assert(false);
	}
	Value& v = stack.reserve();
	r.toValue(v);
	return 1;
}

/*
void Clone(Vector const& in, Vector& out) {
	out = Vector(in.type, in.length());
	memcpy
}

void SubAssign(Vector const& in, uint64_t in_index, Vector& out, uint64_t out_index, uint64_t length) {
}
*/

/*void CastInto(Vector const& in, uint64_t in_index, Vector& out, uint64_t out_index, uint64_t length) {
	if(out.type == Type::R_logical) {
		
	}
	else if(out.type == Type::R_double) {
	
	}

	if(a.type == Type::R_double) {
		VectorOp<Op<double, boolean, boolean>::eval>
			::eval(Vector(a), Type::R_logical, r);
	}
	else if(a.type == Type::R_integer) {
		VectorOp<Op<int64_t, boolean, boolean>::eval>
			::eval(Vector(a), Type::R_logical, r);
	}
	else if(a.type == Type::R_logical) {
		VectorOp<Op<boolean, boolean, boolean>::eval>
			::eval(Vector(a), Type::R_logical, r);
	}
	else {
		printf("non-numeric argument to unary logical operator\n");
		assert(false);
	}
	Value& v = stack.reserve();
	r.toValue(v);
	return 1;
}*/


#endif

