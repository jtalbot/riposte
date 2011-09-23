
#ifndef _RIPOSTE_INTERNAL_H
#define _RIPOSTE_INTERNAL_H

#include "exceptions.h"
#include "vector.h"
#include "coerce.h"
#include "ops.h"
#include <cmath>
#include <set>
#include <algorithm>

inline Value expression(Value const& v) { 
	if(v.isPromise())
		return Function(v).prototype()->expression;
	else return v; 
}

inline double asReal1(Value const& v) { 
	if(v.isInteger()) return ((Integer const&)v)[0]; 
	else if(v.isDouble()) return ((Double const&)v)[0]; 
	else _error("Can't cast argument to number"); 
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

inline void Element2Slow(Object const& v, int64_t index, Value& out) {
	Element2(v.base(), index, out); 
}

inline void Element2(Value const& v, int64_t index, Value& out) {
	switch(v.type) {
		#define CASE(Name) case Type::Name: \
			if(index < 0 || index >= v.length) \
				_error("Out-of-range index"); \
			Name::InitScalar(out, ((Name const&)v)[index]); \
			break;
		ATOMIC_VECTOR_TYPES(CASE)
		#undef CASE
		case Type::List: 
			if(index < 0 || index >= v.length) 
				_error("Out-of-range index");
			else if(List::isNA(((List const&)v)[index]))
				_error("Extracting missing element");
			out = ((List const&)v)[index]; 
			break;
		case Type::Object: 
			Element2Slow(((Object const&)v), index, out); 
			break;
		default: _error("NYI: Element of this type"); break;
	};
}

void SubsetSlow(State& state, Value const& a, Value const& i, Value& out); 

inline void Subset(State& state, Value const& a, Value const& i, Value& out) {
	if(i.isDouble1() && i.d >= 1) {
		Element(a, (int64_t)i.d-1, out);
	}
	else if(i.isInteger1() && i.i >= 1) {
		Element(a, i.i-1, out);
	}
	else {
		SubsetSlow(state, a, i, out);
	}
}

inline void Subset2(State& state, Value const& a, Value const& i, Value& out) {
	if(i.isDouble1() && i.d >= 1) {
		Element2(a, (int64_t)i.d-1, out);
		return;
	}
	else if(i.isInteger1() && i.i >= 1) {
		Element2(a, i.i-1, out);
		return;
	}
	else if(i.isCharacter1() && a.isObject() && ((Object const&)a).hasNames()) {
		Character c = Character(((Object const&)a).getNames());
		String const* data = c.v();
		int64_t length = c.length;
		for(int64_t j = 0; j < length; j++) {
			if(data[j].i == i.i) {
				Element2(a, j, out);
				return;
			}
		}
	}
	_error("Invalid subset index");
}

void ElementAssign(Value const& v, int64_t index, Value& out) __attribute__((always_inline));
inline void ElementAssign(Value const& v, int64_t index, Value& out) {
	switch(v.type) {
		#define CASE(Name) case Type::Name: ((Name&)out)[index] = ((Name const&)v)[0]; break;
		VECTOR_TYPES(CASE)
		#undef CASE
		default: _error("NYI: Element of this type"); break;
	};
}

void Element2Assign(Value const& v, int64_t index, Value& out) __attribute__((always_inline));
inline void Element2Assign(Value const& v, int64_t index, Value& out) {
	if(index < 0 || index > v.length) _error("Out-of-range index");
	switch(v.type) {
		#define CASE(Name) case Type::Name: ((Name&)out)[index] = ((Name const&)v)[0]; break;
		ATOMIC_VECTOR_TYPES(CASE)
		#undef CASE
		#define CASE(Name) case Type::Name: ((Name&)out)[index] = ((Name const&)v); break;
		LISTLIKE_VECTOR_TYPES(CASE)
		#undef CASE
		default: _error("NYI: Element of this type"); break;
	};
}

void SubsetAssignSlow(State& state, Value const& a, bool clone, Value const& i, Value const& b, Value& c);
void Subset2AssignSlow(State& state, Value const& a, bool clone, Value const& i, Value const& b, Value& c);
 
inline void SubsetAssign(State& state, Value const& a, bool clone, Value const& i, Value const& b, Value& c) {
	if(!clone && a.type == b.type) {
		if(i.isDouble1()) {
			int64_t index = (int64_t)i.d-1;
			if(index >= 0 && index < a.length) {
				c = a;
				ElementAssign(b, index, c);
				return;
			}
		}
		else if(i.isInteger1()) {
			int64_t index = i.i-1;
			if(index >= 0 && index < a.length) {
				c = a;
				ElementAssign(b, index, c);
				return;
			}
		}
	}
	SubsetAssignSlow(state, a, clone, i, b, c);
}

inline void Subset2Assign(State& state, Value const& a, bool clone, Value const& i, Value const& b, Value& c) {
	if(!clone && a.type == b.type) {
		if(i.isDouble1()) {
			int64_t index = (int64_t)i.d-1;
			if(index >= 0 && index < a.length) {
				c = a;
				Element2Assign(b, index, c);
				return;
			}
		}
		else if(i.isInteger1()) {
			int64_t index = i.i-1;
			if(index >= 0 && index < a.length) {
				c = a;
				Element2Assign(b, index, c);
				return;
			}
		}
	}
	Subset2AssignSlow(state, a, clone, i, b, c);
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


inline Integer Sequence(int64_t length, int64_t start, int64_t step) {
	Integer r(length);
	for(int64_t i = 0, j = start; i < length; i++, j+=step) {
		r[i] = j;
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
		c[0] = Strings::Numeric;
	else if(type == Type::Symbol)
		c[0] = Strings::Name;
	else c[0] = state.internStr(Type::toString(type));
	return c;
}


#endif

