
#include "value.h"
#include "type.h"
#include "bc.h"

#include <sstream>
#include <iomanip>
#include <math.h>

std::string pad(std::string s, uint64_t width)
{
	std::stringstream ss;
	ss << std::setw(width) << s;
	return ss.str();
}

inline std::string toString(State const& state, unsigned char a) {
	return (a ? "TRUE" : "FALSE");
}  

inline std::string toString(State const& state, int64_t a) {
	return intToStr(a);
}  

inline std::string toString(State const& state, double a) {
	return doubleToStr(a);
}  

inline std::string toString(State const& state, uint64_t a) {
	return std::string("\"") + state.outString(a) + "\"";
}  


template<class T>
std::string stringifyVector(State const& state, T const& v) {
	std::string result = "";
	uint64_t length = v.length();
	bool dots = false;
	if(length > 100) { dots = true; length = 100; }
	Value names = getNames(v.attributes);
	uint64_t maxlength = 1;
	for(uint64_t i = 0; i < length; i++) {
		if(names.type == Type::R_character) {
			Character c(names);
			maxlength = std::max((uint64_t)maxlength, (uint64_t)state.outString(c[i]).length());
		}
		maxlength = std::max((uint64_t)maxlength, (uint64_t)toString(state, v[i]).length());
	}
	uint64_t indexwidth = intToStr(length+1).length();
	uint64_t perline = std::max(floor(80.0/(maxlength+1) + indexwidth), 1.0);
	for(uint64_t i = 0; i < length; i+=perline) {
		if(names.type == Type::R_character) {
			Character c(names);
			result = result + pad("", indexwidth+2);
			for(uint64_t j = 0; j < perline && i+j < length; j++) {
				result = result + pad(state.outString(c[i+j]), maxlength+1);
			}
			result = result + "\n";
		}
		result = result + pad(std::string("[") + intToStr(i+1) + "]", indexwidth+2);
		for(uint64_t j = 0; j < perline && i+j < length; j++) {
			result = result + pad(toString(state, v[i+j]), maxlength+1);
		}
	
		if(i+perline < length)	
			result = result + "\n";
	}
	if(dots) result = result + " ...";
	return result;
}

std::string State::stringify(Value const& value) const {
	std::string result = "[1]";
	bool dots = false;

	switch(value.type.internal())
	{
		case Type::R_null:
			return "NULL";
		case Type::R_raw:
			return "raw";
		case Type::R_logical:
		{
			Logical v(value);
			return stringifyVector(*this, v);
		}
		case Type::R_integer:
		{
			Integer v(value);
			return stringifyVector(*this, v);
		}
		case Type::R_double:
		{
			Double v(value);
			return stringifyVector(*this, v);
		}
		case Type::R_complex:		
		{
			//for(uint64_t i = 0; i < length; i++) result = result + ((int64_t*)ptr)[i];
			return "complex";
		}
		case Type::R_character:
		{
			Character v(value);
			return stringifyVector(*this, v);
		}
		case Type::R_list:
		case Type::R_pairlist:
		{
			List v(value);
			Value names = getNames(v.attributes);

			uint64_t length = v.length();
			if(length > 100) { dots = true; length = 100; }
			result = "";
			if(names.type == Type::R_character) {
				Character n(names);
				for(uint64_t i = 0; i < length; i++) {
					result = result + "[[" + outString(n[i]) + "]]\n";
					result = result + stringify(v[i]) + "\n\n";
				}
			} else {
				for(uint64_t i = 0; i < length; i++) {
					result = result + "[[" + intToStr(i) + "]]\n";
					result = result + stringify(v[i]) + "\n\n";
				}
			}
			if(dots) result = result + " ...\n\n";
			return result;
		}
		case Type::R_symbol:
		{
			result = "`" + outString(Symbol(value).i) + "`";
			return result;
		}
		case Type::R_function:
		{
			//result = "function: " + intToHexStr((uint64_t)value.p) /*+ "\n" + Function(*this).body().toString()*/;
			result = outString(Function(value).str()[0]);
			return result;
		}
		case Type::I_bytecode:
		{
			Block b(value);
			std::string r = "block:\nconstants: " + intToStr(b.constants().size()) + "\n";
			for(uint64_t i = 0; i < b.constants().size(); i++)
				r = r + intToStr(i) + "=\t" + stringify(b.constants()[i]) + "\n";
		
			r = r + "code: " + intToStr(b.code().size()) + "\n";
			for(uint64_t i = 0; i < b.code().size(); i++)
				r = r + intToStr(i) + ":\t" + b.code()[i].toString() + "\n";
		
			return r;
		}
		default:
			return value.type.toString();
	};
}
