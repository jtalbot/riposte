
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
	return Logical::isNA(a) ? "NA" : (a ? "TRUE" : "FALSE");
}  

inline std::string toString(State const& state, int64_t a) {
	return Integer::isNA(a) ? "NA" : std::string("") + intToStr(a) + std::string("L");
}  

inline std::string toString(State const& state, double a) {
	return Double::isNA(a) ? "NA" : doubleToStr(a);
}  

inline std::string toString(State const& state, std::complex<double> a) {
	return Complex::isNA(a) ? "NA" : complexToStr(a);
}  

inline std::string toString(State const& state, Symbol const& a) {
	return Character::isNA(a) ? "NA" : std::string("\"") + a.toString(state) + "\"";
}  

template<class T>
std::string stringifyVector(State const& state, T const& v) {
	std::string result = "";
	uint64_t length = v.length;
	if(length == 0)
		return std::string(v.type.toString()) + "(0)";

	bool dots = false;
	if(length > 100) { dots = true; length = 100; }
	Value names = getNames(v.attributes);
	uint64_t maxlength = 1;
	for(uint64_t i = 0; i < length; i++) {
		if(names.type == Type::R_character) {
			Character c(names);
			maxlength = std::max((uint64_t)maxlength, (uint64_t)c[i].toString(state).length());
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
				result = result + pad(c[i+j].toString(state), maxlength+1);
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
	switch(value.type.Enum())
	{
		case Type::E_R_null:
			return "NULL";
		case Type::E_R_raw:
			return "raw";
		case Type::E_R_logical:
			return stringifyVector(*this, Logical(value));
		case Type::E_R_integer:
			return stringifyVector(*this, Integer(value));
		case Type::E_R_double:
			return stringifyVector(*this, Double(value));
		case Type::E_R_complex:		
			return stringifyVector(*this, Complex(value));
		case Type::E_R_character:
			return stringifyVector(*this, Character(value));
		
		case Type::E_R_list:
		case Type::E_R_pairlist:
		{
			List v(value);
			Value names = getNames(v.attributes);

			uint64_t length = v.length;
			if(length > 100) { dots = true; length = 100; }
			result = "";
			if(names.type == Type::R_character) {
				Character n(names);
				for(uint64_t i = 0; i < length; i++) {
					if(n[i].toString(*this)=="")
						result = result + "[[" + intToStr(i+1) + "]]\n";
					else
						result = result + "$" + n[i].toString(*this) + "\n";
					result = result + stringify(v[i]) + "\n";
					if(i < length-1) result = result + "\n";
				}
			} else {
				for(uint64_t i = 0; i < length; i++) {
					result = result + "[[" + intToStr(i+1) + "]]\n";
					result = result + stringify(v[i]) + "\n";
					if(i < length-1) result = result + "\n";
				}
			}
			if(dots) result = result + " ...\n\n";
			return result;
		}
		case Type::E_R_symbol:
		{
			result = "`" + Symbol(value).toString(*this) + "`";
			return result;
		}
		case Type::E_R_function:
		{
			//result = "function: " + intToHexStr((uint64_t)value.p) /*+ "\n" + Function(*this).body().toString()*/;
			result = (Function(value).str()[0]).toString(*this);
			return result;
		}
		case Type::E_R_environment:
		{
			return "environment";
		}
		case Type::E_I_closure:
		{
			Closure b(value);
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
