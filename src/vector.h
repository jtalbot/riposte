
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
		if(!Op::AV::isCheckedNA(a)) return Op::eval(state, r, a);
		else return Op::RV::NAelement;
	}
};

template< class Op, int64_t N >
struct Map1 {
	static void eval(State& state, typename Op::A const* a, typename Op::R* r) {
		for(int64_t i = 0; i < N; ++i) r[i] = Op::eval(state, a[i]);
	}
};

template< class Op, int64_t N >
struct Map2VV {
	static void eval(State& state, typename Op::A const* a, typename Op::B const* b, typename Op::R* r) {
		for(int64_t i = 0; i < N; ++i) r[i] = Op::eval(state, a[i], b[i]);
	}
};

template< class Op, int64_t N >
struct Map2SV {
	static void eval(State& state, typename Op::A const a, typename Op::B const* b, typename Op::R* r) {
		for(int64_t i = 0; i < N; ++i) r[i] = Op::eval(state, a, b[i]);
	}
};

template< class Op, int64_t N >
struct Map2VS {
	static void eval(State& state, typename Op::A const* a, typename Op::B const b, typename Op::R* r) {
		for(int64_t i = 0; i < N; ++i) r[i] = Op::eval(state, a[i], b);
	}
};

template< class Op, int64_t N >
struct FoldLeftT {
	static typename Op::R eval(State& state, typename Op::A const* a, typename Op::R r) {
		for(int64_t i = 0; i < N; ++i) r = Op::eval(state, r, a[i]);
		return r;
	}
};

template< class Op, int64_t N >
struct ScanLeftT {
	static typename Op::R eval(State& state, typename Op::A const* a, typename Op::R b, typename Op::R* r) {
		for(int64_t i = 0; i < N; ++i) r[i] = b = Op::eval(state, b, a[i]);
		return b;
	}
};

template< class Op >
struct Zip1 {
	static void eval(State& state, typename Op::AV const& a, Value& out)
	{
		if(a.isScalar()) {
			Op::RV::InitScalar(out, Op::eval(state, a.s()));
		}
		else {
			typename Op::RV r(a.length);
			typename Op::R* re = r.v();
			typename Op::A const* ae = a.v();
			int64_t length = a.length;
			for(int64_t i = 0; i < length; ++i) {
				re[i] = Op::eval(state, ae[i]);
			}
			out = (Value&)r;
		}
	}
};

template< class Op >
struct Zip2 {
	static void eval(State& state, typename Op::AV const& a, typename Op::BV const& b, Value& out)
	{
		if(a.isScalar() && b.isScalar()) {
			Op::RV::InitScalar(out, Op::eval(state, a.s(), b.s()));
		}
		else if(b.isScalar()) {
			typename Op::RV r(a.length);
			typename Op::R* re = r.v();
			typename Op::A const* ae = a.v();
			typename Op::B be = b.s();
			int64_t length = a.length;
			for(int64_t i = 0; i < length; ++i) {
				re[i] = Op::eval(state, ae[i], be);
			}
			out = (Value&)r;
		}
		else if(a.isScalar()) {
			typename Op::RV r(b.length);
			typename Op::R* re = r.v();
			typename Op::A ae = a.s();
			typename Op::B const* be = b.v();
			int64_t length = b.length;
			for(int64_t i = 0; i < length; ++i) {
				re[i] = Op::eval(state, ae, be[i]);
			}
			out = (Value&)r;
		}
		else if(a.length == b.length) {
			typename Op::RV r(a.length);
			typename Op::R* re = r.v();
			typename Op::A const* ae = a.v();
			typename Op::B const* be = b.v();
			int64_t length = a.length;
			for(int64_t i = 0; i < length; ++i) {
				re[i] = Op::eval(state, ae[i], be[i]);
			}
			out = (Value&)r;
		}
		else if(a.length == 0 || b.length == 0) {
			Op::RV::Init(out, 0);
		}
		else if(a.length > b.length) {
			typename Op::RV r(a.length);
			typename Op::R* re = r.v();
			typename Op::A const* ae = a.v();
			typename Op::B const* be = b.v();
			int64_t alength = a.length;
			int64_t blength = b.length;
			int64_t j = 0;
			for(int64_t i = 0; i < alength; ++i) {
				re[i] = Op::eval(state, ae[i], be[j]);
				++j;
				if(j >= blength) j = 0;
			}
			out = (Value&)r;
		}
		else {
			typename Op::RV r(b.length);
			typename Op::R* re = r.v();
			typename Op::A const* ae = a.v();
			typename Op::B const* be = b.v();
			int64_t alength = a.length;
			int64_t blength = b.length;
			int64_t j = 0;
			for(int64_t i = 0; i < blength; ++i) {
				re[i] = Op::eval(state, ae[j], be[i]);
				++j;
				if(j >= alength) j = 0;
			}
			out = (Value&)r;
		}
	}
};

template< class Op >
struct Zip2N {
	static void eval(State& state, int64_t N, typename Op::AV const& a, typename Op::BV const& b, Value& out)
	{
		typename Op::A const* ae = a.v();
		typename Op::B const* be = b.v();
		typename Op::RV r(N);
		typename Op::R* re = r.v();
		int64_t j = 0, k = 0;
		for(int64_t i = 0; i < N; i++) {
			re[i] = Op::eval(state, ae[j++], be[k++]);
			if(j >= a.length) j = 0;
			if(k >= b.length) k = 0;
		}
		out = (Value&)r;
	}
};

template< class Op >
struct FoldLeft {
	static void eval(State& state, typename Op::AV const& a, Value& out)
	{
		typename Op::A const* ae = a.v();
		typename Op::R b = Op::base();
		int64_t length = a.length;
		for(int64_t i = 0; i < length; ++i) {
			b = Op::eval(state, b, ae[i]);
		}
		Op::RV::InitScalar(out, b);
	}
};

template< class Op >
struct ScanLeft {
	static void eval(State& state, typename Op::AV const& a, Value& out)
	{
		typename Op::A const* ae = a.v();
		typename Op::R b = Op::base();
		typename Op::RV r(a.length);
		typename Op::R* re = r.v();
		int64_t length = a.length;
		for(int64_t i = 0; i < length; ++i) {
			re[i] = b = Op::eval(state, b, ae[i]);
		}
		out = (Value&)r;
	}
};

template<class T>
T Clone(T const& in) {
	T out(in.length);
	memcpy(out.v(), in.v(), in.length*in.width);
	return out;	
}

#endif
