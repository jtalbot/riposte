
#ifndef _RIPOSTE_VECTOR_H
#define _RIPOSTE_VECTOR_H

#include "value.h"

template<class X, class Y>
struct UnaryOp {
	typedef X AV;
	typedef Y RV;
	typedef typename X::Element A;
	typedef typename Y::Element R;
};

template<typename X, typename Y, typename Z> struct BinaryOp {
	typedef X AV;
	typedef Y BV;
	typedef Z RV;
	typedef typename X::Element A;
	typedef typename Y::Element B;
	typedef typename Z::Element R;
};

template<class X>
struct FoldOp {
	typedef X AV;
	typedef X RV;
	typedef typename X::Element A;
	typedef typename X::Element R;
};

template<class Op>
struct NAFold : public Op {
	static typename Op::R eval(State& state, typename Op::R const& r, typename Op::A const& a) {
		if(!Op::AV::CheckNA || !Op::AV::isNA(a)) return Op::eval(state, r, a);
		else return Op::RV::NAelement;
	}
};

template< class Op >
struct Zip1 {
	static typename Op::RV eval(State& state, typename Op::AV const& a)
	{
		typename Op::RV r = typename Op::RV(a.length);
		for(int64_t i = 0; i < a.length; ++i) {
			r[i] = Op::eval(state, a[i]);
		}
		return r;
	}
};

template< class Op >
struct Zip2 {
	static typename Op::RV eval(State& state, typename Op::AV const& a, typename Op::BV const& b)
	{
		if(a.length == 1 && b.length == 1) {
			typename Op::RV r(1);
			r[0] = Op::eval(state, a[0], b[0]);
			return r;
		}
		if(a.length == b.length) {
			typename Op::RV r(a.length);
			for(int64_t i = 0; i < a.length; ++i) {
				r[i] = Op::eval(state, a[i], b[i]);
			}
			return r;
		}
		else if(a.length == 1) {
			typename Op::RV r(b.length);
			for(int64_t i = 0; i < b.length; ++i) {
				r[i] = Op::eval(state, a[0], b[i]);
			}
			return r;
		}
		else if(b.length == 1) {
			typename Op::RV r(a.length);
			for(int64_t i = 0; i < a.length; ++i) {
				r[i] = Op::eval(state, a[i], b[0]);
			}
			return r;
		}
		else if(a.length == 0 || b.length == 0) {
			return typename Op::RV(0);
		}
		else if(a.length > b.length) {
			typename Op::RV r(a.length);
			int64_t j = 0;
			for(int64_t i = 0; i < a.length; ++i) {
				r[i] = Op::eval(state, a[i], b[j]);
				++j;
				if(j >= b.length) j = 0;
			}
			return r;
		}
		else {
			typename Op::RV r(b.length);
			int64_t j = 0;
			for(int64_t i = 0; i < b.length; ++i) {
				r[i] = Op::eval(state, a[j], b[i]);
				++j;
				if(j >= a.length) j = 0;
			}
			return r;
		}
	}
};

template< class Op >
struct FoldLeft {
	static typename Op::RV eval(State& state, typename Op::AV const& a)
	{
		typename Op::R b = Op::Base;
		for(int64_t i = 0; i < a.length; ++i) {
			b = Op::eval(state, b, a[i]);
		}
		return Op::RV::c(b);
	}
};

template< class Op >
struct ScanLeft {
	static typename Op::RV eval(State& state, typename Op::AV const& a)
	{
		typename Op::R b = Op::Base;
		typename Op::RV result(a.length);
		for(int64_t i = 0; i < a.length; ++i) {
			result[i] = b = Op::eval(state, b, a[i]);
		}
		return result;
	}
};

template<class T>
T Clone(T const& in) {
	T out(in.length);
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
