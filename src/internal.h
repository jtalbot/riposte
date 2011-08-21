
#ifndef _RIPOSTE_INTERNAL_H
#define _RIPOSTE_INTERNAL_H

#include "exceptions.h"
#include "vector.h"
#include "coerce.h"
#include "ops.h"
#include <cmath>
#include <set>
#include <algorithm>

void importCoreLibrary(State& state, Environment* env);

inline Value force(State& state, Value v) { 
	while(v.isPromise()) {
		Environment* env = Function(v).environment();
		v = eval(state, Function(v).code(), 
			env != 0 ? env : state.frame.environment); 
	} 
	return v;
}
inline Value expression(Value const& v) { 
	if(v.isPromise())
		return Function(v).code()->expression;
	else return v; 
}

inline double asReal1(Value const& v) { 
	if(v.isInteger()) return ((Integer const&)v)[0]; 
	else if(v.isDouble()) return ((Double const&)v)[0]; 
	else _error("Can't cast argument to number"); 
}

template< class A >
struct SubsetInclude {
	static A eval(State& state, A const& a, Integer const& d, int64_t nonzero)
	{
		A r(nonzero);
		int64_t j = 0;
		typename A::Element const* ae = a.v();
		typename Integer::Element const* de = d.v();
		typename A::Element* re = r.v();
		int64_t length = d.length;
		for(int64_t i = 0; i < length; i++) {
			if(Integer::isNA(de[i])) re[j++] = A::NAelement;	
			else if(de[i] != 0) re[j++] = ae[de[i]-1];
		}
		return r;
	}
};

template< class A >
struct SubsetExclude {
	static A eval(State& state, A const& a, Integer const& d, int64_t nonzero)
	{
		std::set<Integer::Element> index; 
		typename A::Element const* ae = a.v();
		typename Integer::Element const* de = d.v();
		int64_t length = d.length;
		for(int64_t i = 0; i < length; i++) if(-de[i] > 0 && -de[i] < (int64_t)a.length) index.insert(-de[i]);
		// iterate through excluded elements copying intervening ranges.
		A r(a.length-index.size());
		typename A::Element* re = r.v();
		int64_t start = 1;
		int64_t k = 0;
		for(std::set<Integer::Element>::const_iterator i = index.begin(); i != index.end(); ++i) {
			int64_t end = *i;
			for(int64_t j = start; j < end; j++) re[k++] = ae[j-1];
			start = end+1;
		}
		for(int64_t j = start; j <= a.length; j++) re[k++] = ae[j-1];
		return r;
	}
};

template< class A >
struct SubsetLogical {
	static A eval(State& state, A const& a, Logical const& d)
	{
		typename A::Element const* ae = a.v();
		typename Logical::Element const* de = d.v();
		// determine length
		int64_t length = 0;
		if(d.length > 0) {
			int64_t j = 0;
			int64_t maxlength = std::max(a.length, d.length);
			for(int64_t i = 0; i < maxlength; i++) {
				if(!Logical::isFalse(de[j])) length++;
				if(++j >= d.length) j = 0;
			}
		}
		A r(length);
		typename A::Element* re = r.v();
		int64_t j = 0, k = 0;
		for(int64_t i = 0; i < std::max(a.length, d.length) && k < length; i++) {
			if(i >= a.length || Logical::isNA(de[j])) re[k++] = A::NAelement;
			else if(Logical::isTrue(de[j])) re[k++] = ae[i];
			if(++j >= d.length) j = 0;
		}
		return r;
	}
};


template< class A  >
struct SubsetAssign {
	static A eval(State& state, A const& a, Integer const& d, A const& b)
	{
		typename A::Element const* be = b.v();
		typename Integer::Element const* de = d.v();

		// compute max index 
		int64_t outlength = 0;
		int64_t length = d.length;
		for(int64_t i = 0; i < length; i++) {
			outlength = std::max((int64_t)outlength, de[i]);
		}

		// should use max index here to extend vector if necessary	
		A r = Clone(a);	
		typename A::Element* re = r.v();
		for(int64_t i = 0; i < length; i++) {	
			int64_t idx = de[i];
			if(idx != 0)
				re[idx-1] = be[i];
		}
		return r;
	}
};

inline void subAssign(State& state, Value const& a, Value const& i, Value const& b, Value& c) {
	Integer idx = As<Integer>(state, i);
	switch(a.type) {
		#define CASE(Name) case Type::Name: c = SubsetAssign<Name>::eval(state, (Name const&)a, idx, As<Name>(state, b)); break;
		VECTOR_TYPES(CASE)
		#undef CASE
		default: _error("NYI: subset assign type"); break;
	};
}

template<class D>
inline void Insert(State& state, D const& src, int64_t srcIndex, D& dst, int64_t dstIndex, int64_t length) {
	if((length > 0 && srcIndex+length > src.length) || dstIndex+length > dst.length)
		_error("insert index out of bounds");
	memcpy(dst.v()+dstIndex, src.v()+srcIndex, length*src.width);
}

template<class S, class D>
inline void Insert(State& state, S const& src, int64_t srcIndex, D& dst, int64_t dstIndex, int64_t length) {
	D as = As<D>(state, src);
	Insert(state, as, srcIndex, dst, dstIndex, length);
}

template<class D>
inline void Insert(State& state, Value const& src, int64_t srcIndex, D& dst, int64_t dstIndex, int64_t length) {
	switch(src.type) {
		#define CASE(Name) case Type::Name: Insert(state, (Name const&)src, srcIndex, dst, dstIndex, length); break;
		VECTOR_TYPES(CASE)
		#undef CASE
		default: _error("NYI: Insert into this type"); break;
	};
}

inline void Insert(State& state, Value const& src, int64_t srcIndex, Value& dst, int64_t dstIndex, int64_t length) {
	switch(dst.type) {
		#define CASE(Name) case Type::Name: { Insert(state, src, srcIndex, (Name&) dst, dstIndex, length); } break;
		VECTOR_TYPES(CASE)
		#undef CASE
		default: _error("NYI: Insert into this type"); break;
	};
}

template<class T>
inline T Subset(T const& src, int64_t start, int64_t length) {
	if(length > 0 && start+length > src.length)
		_error("subset index out of bounds");
	T v(length);
	memcpy(v.v(), src.v()+start, length*src.width);
	return v;
}

void Element(Value const& v, int64_t index, Value& out) __attribute__((always_inline));
inline void Element(Value const& v, int64_t index, Value& out) {
	switch(v.type) {
		#define CASE(Name) case Type::Name: Name::InitScalar(out, ((Name const&)v)[index]); break;
		VECTOR_TYPES(CASE)
		#undef CASE
		default: _error("NYI: Element of this type"); break;
	};
}

void Element2(Value const& v, int64_t index, Value& out) __attribute__((always_inline));
inline void Element2(Value const& v, int64_t index, Value& out) {
	switch(v.type) {
		#define CASE(Name) case Type::Name: Name::InitScalar(out, ((Name const&)v)[index]); break;
		ATOMIC_VECTOR_TYPES(CASE)
		#undef CASE
		#define CASE(Name) case Type::Name: out = ((Name const&)v)[index]; break;
		LISTLIKE_VECTOR_TYPES(CASE)
		#undef CASE
		default: _error("NYI: Element of this type"); break;
	};
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

inline Character klass(State& state, Value const& v)
{
	Type::Enum type;
	if(v.isObject()) {
		if(((Object const&)v).hasClass())
			return Character(((Object const&)v).getClass());
		else
			type = ((Object const&)v).base().type;
	}
	else {
		type = v.type;
	}
			
	Character c(1);
	if(type == Type::Integer || type == Type::Double)
		c[0] = Symbols::Numeric;
	else if(type == Type::Symbol)
		c[0] = Symbols::Name;
	else c[0] = state.StrToSym(Type::toString(type));
	return c;
}


#endif

