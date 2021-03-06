
#ifndef _RIPOSTE_INTERNAL_H
#define _RIPOSTE_INTERNAL_H

#include "value.h"
#include "exceptions.h"

#include <cmath>
#include <set>
#include <algorithm>

inline double asReal1(Value const& v) { 
	if(v.isInteger()) return ((Integer const&)v)[0]; 
	else if(v.isDouble()) return ((Double const&)v)[0]; 
	else _error("Can't cast argument to number"); 
}

String type2String(Type::Enum type);
Type::Enum string2Type(String str);

void Element(Value const& v, int64_t index, Value& out) ALWAYS_INLINE;
inline void Element(Value const& v, int64_t index, Value& out) {
	switch(v.type()) {
		#define CASE(Name) case Type::Name: Name::InitScalar(out, ((Name const&)v)[index]); break;
		VECTOR_TYPES(CASE)
		#undef CASE
		default: _error("NYI: Element of this type"); break;
	};
}

void Element2(Value const& v, int64_t index, Value& out) ALWAYS_INLINE;
inline void Element2(Value const& v, int64_t index, Value& out) {
	if(index < 0 || index >= ((Vector const&)v).length()) _error("Out-of-range index");
	switch(v.type()) {
		#define CASE(Name) case Type::Name: \
			Name::InitScalar(out, ((Name const&)v)[index]); \
			break;
		ATOMIC_VECTOR_TYPES(CASE)
		#undef CASE
		case Type::List: 
			if(List::isNA(((List const&)v)[index]))
				_error("Extracting missing element");
			out = ((List const&)v)[index]; 
			break;
		default: _error("NYI: Element of this type"); break;
	};
}

void SubsetSlow(Thread& thread, Value const& a, Value const& i, Value& out); 

inline void Subset(Thread& thread, Value const& a, Value const& i, Value& out) {
	if(i.isDouble1() && i.d >= 1) {
		Element(a, (int64_t)i.d-1, out);
	}
	else if(i.isInteger1() && i.i >= 1) {
		Element(a, i.i-1, out);
	}
	else {
		SubsetSlow(thread, a, i, out);
	}
}

inline void Subset2(Thread& thread, Value const& a, Value const& i, Value& out) {
	if(i.isDouble1() && i.d >= 1) {
		Element2(a, (int64_t)i.d-1, out);
		return;
	}
	else if(i.isInteger1() && i.i >= 1) {
		Element2(a, i.i-1, out);
		return;
	}
	/*else if(i.isCharacter1() && a.isObject() && ((Object const&)a).hasNames()) {
		Character c = Character(((Object const&)a).getNames());
		String const* data = c.v();
		int64_t length = c.length;
		for(int64_t j = 0; j < length; j++) {
			if(data[j] == i.s) {
				Element2(a, j, out);
				return;
			}
		}
	}*/
	_error("Invalid subset index");
}

void ElementAssign(Value const& v, int64_t index, Value& out) ALWAYS_INLINE;
inline void ElementAssign(Value const& v, int64_t index, Value& out) {
	switch(v.type()) {
		#define CASE(Name) case Type::Name: ((Name&)out)[index] = ((Name const&)v)[0]; break;
		VECTOR_TYPES(CASE)
		#undef CASE
		default: _error("NYI: Element of this type"); break;
	};
}

void Element2Assign(Value const& v, int64_t index, Value& out) ALWAYS_INLINE;
inline void Element2Assign(Value const& v, int64_t index, Value& out) {
	if(index < 0 || index > ((Vector const&)v).length()) _error("Out-of-range index");
	switch(v.type()) {
		#define CASE(Name) case Type::Name: ((Name&)out)[index] = ((Name const&)v)[0]; break;
		ATOMIC_VECTOR_TYPES(CASE)
		#undef CASE
		#define CASE(Name) case Type::Name: ((Name&)out)[index] = v; break;
		LISTLIKE_VECTOR_TYPES(CASE)
		#undef CASE
		default: _error("NYI: Element of this type"); break;
	};
}

void SubsetAssignSlow(Thread& thread, Value const& a, bool clone, Value const& i, Value const& b, Value& c);
void Subset2AssignSlow(Thread& thread, Value const& a, bool clone, Value const& i, Value const& b, Value& c);
 
inline void SubsetAssign(Thread& thread, Value const& a, bool clone, Value const& i, Value const& b, Value& c) {
	if(!clone && a.type() == b.type()) {
		if(i.isDouble1()) {
			int64_t index = (int64_t)i.d-1;
			if(index >= 0 && index < ((Vector const&)a).length()) {
				c = a;
				ElementAssign(b, index, c);
				return;
			}
		}
		else if(i.isInteger1()) {
			int64_t index = i.i-1;
			if(index >= 0 && index < ((Vector const&)a).length()) {
				c = a;
				ElementAssign(b, index, c);
				return;
			}
		}
	}
	SubsetAssignSlow(thread, a, clone, i, b, c);
}

inline void Subset2Assign(Thread& thread, Value const& a, bool clone, Value const& i, Value const& b, Value& c) {
	if(!clone && (a.type() == b.type() || a.isList())) {
		if(i.isDouble1()) {
			int64_t index = (int64_t)i.d-1;
			if(index >= 0 && index < ((Vector const&)a).length()) {
				c = a;
				Element2Assign(b, index, c);
				return;
			}
		}
		else if(i.isInteger1()) {
			int64_t index = i.i-1;
			if(index >= 0 && index < ((Vector const&)a).length()) {
				c = a;
				Element2Assign(b, index, c);
				return;
			}
		}
	}
	Subset2AssignSlow(thread, a, clone, i, b, c);
}


void Insert(Thread& thread, Value const& src, int64_t srcIndex, Value& dst, int64_t dstIndex, int64_t length);

void Resize(Thread& thread, bool clone, Value& src, int64_t newLength);

template<class T>
inline T Subset(T const& src, int64_t start, int64_t length) {
	if(length > 0 && start+length > src.length)
		_error("subset index out of bounds");
	T v(length);
	memcpy(v.v(), src.v()+start, length*src.width);
	return v;
}


inline Integer Sequence(int64_t start, int64_t step, int64_t length) {
	Integer r(length);
	for(int64_t i = 0, j = start; i < length; i++, j+=step) {
		r[i] = j;
	}
	return r;
}

inline Double Sequence(double start, double step, int64_t length) {
	Double r(length);
	double j = 0;
	for(int64_t i = 0; i < length; i++, j+=step) {
		r[i] = start+j;
	}
	return r;
}

inline Integer Repeat(int64_t const n, int64_t const each, int64_t const length) {
	Integer r(length);
	for(int64_t i = 0, j = 1, e = 1; i < length; i++) {
		r[i] = j;
		e++; 
		if(e > each) { e = 1; j++; }
		if(j > n) j = 1;
	}
	return r;
}

inline Integer Repeat(Integer const& each, int64_t const length) {
	Integer r(length);
	for(int64_t i = 0, j = 0, e = 1; i < length; i++) {
		r[i] = j+1;
		e++; 
		if(e > each[j]) { e = 1; j++; }
		if(j >= each.length()) j = 0;
	}
	return r;
}

Double RandomVector(Thread& thread, int64_t const length);

#endif

