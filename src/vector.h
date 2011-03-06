
#ifndef _RIPOSTE_VECTOR_H
#define _RIPOSTE_VECTOR_H

#include "value.h"

// Vectors are defined in value. This defines common operations on vectors.

inline bool isVector(Value& v) {
	return 	v.type() == R_logical ||
			v.type() == R_integer ||
			v.type() == R_double ||
			v.type() == R_complex ||
			v.type() == R_character ||
			v.type() == R_raw ||
			v.type() == R_list ||
			v.type() == R_call ||
			v.type() == R_expression;
}

inline bool isBasicVector(Value& v) {
	return 	v.type() == R_logical ||
			v.type() == R_integer ||
			v.type() == R_double ||
			v.type() == R_complex ||
			v.type() == R_character ||
			v.type() == R_raw;
}

inline Value unpack(Value const& v) {
	if(v.packed == 0) {
		switch(v.type().internal()) {
			case R_logical: return Logical(0, false); break;
			case R_integer: return Integer(0, false); break;
			case R_double: return Double(0, false); break;
			case R_raw: return Raw(0, false); break;
			default: return v; break;
		}	
	} else if(v.packed == 1) {
		Value r;
		switch(v.type().internal()) {
			case R_logical: { Logical t(1, false); t[0] = Logical(v)[0]; t.toValue(r); } break;
			case R_integer: { Integer t(1, false); t[0] = Integer(v)[0]; t.toValue(r); } break;
			case R_double: { Double t(1, false); t[0] = Double(v)[0]; t.toValue(r); } break;
			case R_raw: { Raw t(1, false); t[0] = Raw(v)[0]; t.toValue(r); } break;
			default: return v; break;
		}	
	}
	else return v;
}

#endif
