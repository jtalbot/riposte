
#ifndef _RIPOSTE_VECTOR_H
#define _RIPOSTE_VECTOR_H

#include "value.h"

template<class X, class Y>
struct UnaryOp {
	typedef X A;
	typedef Y R;
};

template<class Op>
struct NA1 : UnaryOp<typename Op::A, typename Op::R> {
	static typename Op::R::Element eval(State& state, typename Op::A::Element const& a) {
		if(!Op::A::CheckNA || !Op::A::isNA(a))
			return Op::eval(state, a);
		else
			return Op::R::NAelement;
	}
};

template<typename X, typename Y, typename Z> struct BinaryOp {
	typedef X A;
	typedef Y B;
	typedef Z R;
};

template<class Op>
struct NA2 : BinaryOp<typename Op::A, typename Op::B, typename Op::R> {
	static typename Op::R::Element eval(State& state, typename Op::A::Element const& a, typename Op::B::Element const& b) {
		if((!Op::A::CheckNA || !Op::A::isNA(a)) && (!Op::B::CheckNA || !Op::B::isNA(b)))
			return Op::eval(state, a, b);
		else
			return Op::R::NAelement;
	}
};

template< class Op >
struct Zip1 {
	static typename Op::R eval(State& state, typename Op::A const& a)
	{
		typename Op::R r = typename Op::R(a.length);
		for(uint64_t i = 0; i < a.length; ++i) {
			r[i] = Op::eval(state, a[i]);
		}
		return r;
	}
};

template< class Op >
struct Zip2 {
	static typename Op::R eval(State& state, typename Op::A const& a, typename Op::B const& b)
	{
		if(a.length == b.length) {
			typename Op::R r(a.length);
			for(uint64_t i = 0; i < a.length; ++i) {
				r[i] = Op::eval(state, a[i], b[i]);
			}
			return r;
		}
		else if(a.length == 1) {
			typename Op::R r(b.length);
			for(uint64_t i = 0; i < b.length; ++i) {
				r[i] = Op::eval(state, a[0], b[i]);
			}
			return r;
		}
		else if(b.length == 1) {
			typename Op::R r(a.length);
			for(uint64_t i = 0; i < a.length; ++i) {
				r[i] = Op::eval(state, a[i], b[0]);
			}
			return r;
		}
		else if(a.length == 0 || b.length == 0) {
			return typename Op::R(0);
		}
		else if(a.length > b.length) {
			typename Op::R r(a.length);
			uint64_t j = 0;
			for(uint64_t i = 0; i < a.length; ++i) {
				r[i] = Op::eval(state, a[i], b[j]);
				++j;
				if(j >= b.length) j = 0;
			}
			return r;
		}
		else {
			typename Op::R r(b.length);
			uint64_t j = 0;
			for(uint64_t i = 0; i < b.length; ++i) {
				r[i] = Op::eval(state, a[j], b[i]);
				++j;
				if(j >= a.length) j = 0;
			}
			return r;
		}
	}
};

template<class T>
T Clone(T const& in) {
	T out(in.length());
	memcpy(out.data(), in.data(), in.length*in.width);
	out.attributes = in.attributes;
	return out;	
}

inline Vector Clone(Vector const& in) {
	Vector out(in.type, in.length);
	memcpy(out.data(), in.data(), in.length*in.width);
	out.attributes = in.attributes;
	return out;
}

#endif
