
// A number of helper functions for manipulating objects

#ifndef OBJECT_H
#define OBJECT_H

#include "value.h"
#include "vector.h"

}

inline void At(Object const o, String const s, Value& out) {
	uint64_t i = o.find(s);
	if(o.attributes()[i].n != Strings::NA) out = o.attributes()[i].v;
	else _error("Subscript out of range");
}

inline void AtAssign(Object const o, String const s, Value const v, Object& out) {
	uint64_t i = o.find(s);
	uint64_t newlength = o.length;
	if(!v.isNil() && o.attributes()[i].n == Strings::NA) {
		// adding new key, might have to rehash...
		newlength++;
	}
	Object::Pair* attributes;
	if((newlength*2) > o.capacity()) {
		uint64_t newcapacity = std::max(o.capacity()*2,1);
		attributes = new Object::Pair[newcapacity];
		out = Object(o.base(), newlength, o.capacity()*2, attributes);
		
		// clear
		for(uint64_t j = 0; j < newcapacity; j++)
			attributes[j] = (Object::Pair) { Strings::NA, Value::Nil() };
		
		// rehash
		for(uint64_t j = 0; j < o.capacity(); j++)
			if(o.attributes()[j].n != Strings::NA)
				attributes[out.find(o.attributes()[j].n)] = o.attributes()[j].v;
	}
	else {
		// otherwise, just copy straight over
		attributes = new Object::Pair[o.capacity()];
		out = Object(o.base(), newlength, o.capacity(), attributes);
		
		for(uint64_t j = 0; j < o.capacity(); j++)
			attributes[j] = o.attributes()[j];
	}
	attributes[out.find(s)] = v;
}


inline Value CreateExpression(List const& list) {
        Value v;
        Object::Init(v, list, Value::Nil(), Character::c(Strings::Expression));
        return v;
}

inline Value CreateCall(List const& list, Value const& names = Value::Nil()) {
        Value v;
        Object::Init(v, list, names, Character::c(Strings::Call));
        return v;
}


#endif

