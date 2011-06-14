
#ifndef _RIPOSTE_INTERNAL_H
#define _RIPOSTE_INTERNAL_H

#include "exceptions.h"
#include "vector.h"
#include "coerce.h"
#include "ops.h"
#include <cmath>
#include <set>
#include <algorithm>

void importCoreLibrary(State& state);

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

template< class A >
struct SubsetInclude {
	static A eval(State& state, A const& a, Integer const& d, int64_t nonzero)
	{
		A r(nonzero);
		int64_t j = 0;
		for(int64_t i = 0; i < d.length; i++) {
			if(Integer::isNA(d[i])) r[j++] = A::NAelement;	
			else if(d[i] != 0) r[j++] = a[d[i]-1];
		}
		return r;
	}
};

template< class A >
struct SubsetExclude {
	static A eval(State& state, A const& a, Integer const& d, int64_t nonzero)
	{
		std::set<Integer::Element> index; 
		for(int64_t i = 0; i < d.length; i++) if(-d[i] > 0 && -d[i] < (int64_t)a.length) index.insert(-d[i]);
		// iterate through excluded elements copying intervening ranges.
		A r(a.length-index.size());
		int64_t start = 1;
		int64_t k = 0;
		for(std::set<Integer::Element>::const_iterator i = index.begin(); i != index.end(); ++i) {
			int64_t end = *i;
			for(int64_t j = start; j < end; j++) r[k++] = a[j-1];
			start = end+1;
		}
		for(int64_t j = start; j <= a.length; j++) r[k++] = a[j-1];
		return r;
	}
};

template< class A >
struct SubsetLogical {
	static A eval(State& state, A const& a, Logical const& d)
	{
		// determine length
		int64_t length = 0;
		if(d.length > 0) {
			int64_t j = 0;
			for(int64_t i = 0; i < std::max(a.length, d.length); i++) {
				if(!Logical::isFalse(d[j])) length++;
				if(++j >= d.length) j = 0;
			}
		}
		A r(length);
		int64_t j = 0, k = 0;
		for(int64_t i = 0; i < std::max(a.length, d.length) && k < length; i++) {
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
		for(int64_t i = 0; i < d.length; i++) {
			outlength = std::max((int64_t)outlength, d[i]);
		}

		// should use max index here to extend vector if necessary	
		A r = Clone(a);	
		for(int64_t i = 0; i < d.length; i++) {	
			int64_t idx = d[i];
			if(idx != 0)
				r[idx-1] = b[i];
		}
		return r;
	}
};

inline void subAssign(State& state, Value const& a, Value const& i, Value const& b, Value& c) {
	Integer idx = As<Integer>(state, i);
	if(a.isDouble()) c = SubsetAssign<Double>::eval(state, a, idx, As<Double>(state, b));
	else if(a.isInteger()) c = SubsetAssign<Integer>::eval(state, a, idx, As<Integer>(state, b));
	else if(a.isLogical()) c = SubsetAssign<Logical>::eval(state, a, idx, As<Logical>(state, b));
	else if(a.isCharacter()) c = SubsetAssign<Character>::eval(state, a, idx, As<Character>(state, b));
	else if(a.isComplex()) c = SubsetAssign<Complex>::eval(state, a, idx, As<Complex>(state, b));
	else if(a.isList()) c = SubsetAssign<List>::eval(state, a, idx, As<List>(state, b));
	else _error("NYI: subset assign type");
}

template<class S, class D>
inline void Insert(State& state, S const& src, int64_t srcIndex, D& dst, int64_t dstIndex, int64_t length) {
	if(length > 0 && srcIndex+length > src.length || dstIndex+length > dst.length)
		_error("insert index out of bounds");
	D as = As<D>(state, src);
	memcpy(dst.data(dstIndex), as.data(srcIndex), length*as.width);
}

inline void Insert(State& state, Vector const& src, int64_t srcIndex, Vector& dst, int64_t dstIndex, int64_t length) {
	if(length > 0 && srcIndex+length > src.length || dstIndex+length > dst.length)
		_error("insert index out of bounds");
	Vector as = As(state, dst.type, src);
	memcpy(dst.data(dstIndex), as.data(srcIndex), length*as.width);
}

template<class T>
inline T Subset(T const& src, int64_t start, int64_t length) {
	if(length > 0 && start+length > src.length)
		_error("subset index out of bounds");
	T v(length);
	memcpy(v.data(0), src.data(start), length*src.width);
	return v;
}

inline Vector Subset(Vector const& src, int64_t start, int64_t length) {
	if(length > 0 && start+length > src.length)
		_error("subset index out of bounds");
	Vector v(src.type, length);
	memcpy(v.data(0), src.data(start), length*src.width);
	return v;
}

inline Integer Sequence(int64_t length) {
	Integer r(length);
	for(int64_t i = 0; i < length; i++) {
		r[i] = i+1;
	}
	return r;
}

inline Double Sequence(double from, double by, double len) {
	Double r(len);
	double j = 0;
	for(int64_t i = 0; i < len; i++) {
		r[i] = from+j;
		j = j + by;
	}
	return r;
}

inline Vector Element(Vector const& src, int64_t index)
{
	return Subset(src, index, 1);
}

inline Value Element2(Vector const& src, int64_t index)
{
	if(src.type == Type::R_list) return List(src)[index];
	else return Subset(src, index, 1);
}

inline Character klass(State& state, Value const& v)
{
	if(!hasClass(v)) {
		Character c(1);
		if(v.type == Type::R_integer || v.type == Type::R_double)
			c[0] = Symbol(state, "numeric");
		else if(v.type == Type::R_symbol)
			c[0] = Symbol(state, "name");
		else c[0] = Symbol(state, (v).type.toString());
		return c;
	}
	else {
		return getClass(v);
	}
	return 1;
}


#endif

