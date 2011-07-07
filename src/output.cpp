
#include "value.h"
#include "type.h"
#include "bc.h"
#include "internal.h"

#include <sstream>
#include <iomanip>
#include <math.h>

std::string pad(std::string s, int64_t width)
{
	std::stringstream ss;
	ss << std::setw(width) << s;
	return ss.str();
}

template<class T> std::string stringify(State const& state, typename T::Element a) {
	return "";
}

template<> std::string stringify<Logical>(State const& state, Logical::Element a) {
	return Logical::isNA(a) ? "NA" : (a ? "TRUE" : "FALSE");
}  

template<> std::string stringify<Integer>(State const& state, Integer::Element a) {
	return Integer::isNA(a) ? "NA" : std::string("") + intToStr(a) + std::string("L");
}  

template<> std::string stringify<Double>(State const& state, Double::Element a) {
	return Double::isNA(a) ? "NA" : doubleToStr(a);
}  

template<> std::string stringify<Complex>(State const& state, Complex::Element a) {
	return Complex::isNA(a) ? "NA" : complexToStr(a);
}  

template<> std::string stringify<Character>(State const& state, Character::Element a) {
	return Character::isNA(a) ? "NA" : std::string("\"") + a.toString(state) + "\"";
}  

template<> std::string stringify<List>(State const& state, List::Element a) {
	return state.stringify(a);
}  

template<> std::string stringify<Call>(State const& state, Call::Element a) {
	return state.stringify(a);
}  

template<> std::string stringify<Expression>(State const& state, Expression::Element a) {
	return state.stringify(a);
}  


template<class T>
std::string stringifyVector(State const& state, T const& v) {
	std::string result = "";
	int64_t length = v.length;
	if(length == 0)
		return std::string(v.type.toString()) + "(0)";

	bool dots = false;
	if(length > 100) { dots = true; length = 100; }
	int64_t maxlength = 1;
	for(int64_t i = 0; i < length; i++) {
		maxlength = std::max((int64_t)maxlength, (int64_t)stringify<T>(state, v[i]).length());
	}
	if(hasNames(v)) {
		Character c = getNames(v);
		for(int64_t i = 0; i < length; i++) {
			maxlength = std::max((int64_t)maxlength, (int64_t)c[i].toString(state).length());
		}
	}
	int64_t indexwidth = intToStr(length+1).length();
	int64_t perline = std::max(floor(80.0/(maxlength+1) + indexwidth), 1.0);
	if(hasNames(v)) {
		Character c = getNames(v);
		for(int64_t i = 0; i < length; i+=perline) {
			result = result + pad("", indexwidth+2);
			for(int64_t j = 0; j < perline && i+j < length; j++) {
				result = result + pad(c[i+j].toString(state), maxlength+1);
			}
			result = result + "\n";
			result = result + pad(std::string("[") + intToStr(i+1) + "]", indexwidth+2);
			for(int64_t j = 0; j < perline && i+j < length; j++) {
				result = result + pad(stringify<T>(state, v[i+j]), maxlength+1);
			}
	
			if(i+perline < length)	
				result = result + "\n";
		}
	}
	else {
		for(int64_t i = 0; i < length; i+=perline) {
			result = result + pad(std::string("[") + intToStr(i+1) + "]", indexwidth+2);
			for(int64_t j = 0; j < perline && i+j < length; j++) {
				result = result + pad(stringify<T>(state, v[i+j]), maxlength+1);
			}
	
			if(i+perline < length)	
				result = result + "\n";
		}
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

			int64_t length = v.length;
			if(length > 100) { dots = true; length = 100; }
			result = "";
			if(hasNames(v)) {
				Character n = getNames(v);
				for(int64_t i = 0; i < length; i++) {
					if(n[i].toString(*this)=="")
						result = result + "[[" + intToStr(i+1) + "]]\n";
					else
						result = result + "$" + n[i].toString(*this) + "\n";
					result = result + stringify(v[i]) + "\n";
					if(i < length-1) result = result + "\n";
				}
			} else {
				for(int64_t i = 0; i < length; i++) {
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
			//result = "function: " + intToHexStr((int64_t)value.p) /*+ "\n" + Function(*this).body().toString()*/;
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
			std::string r = "block:\nconstants: " + intToStr(b.code()->constants.size()) + "\n";
			for(int64_t i = 0; i < (int64_t)b.code()->constants.size(); i++)
				r = r + intToStr(i) + "=\t" + stringify(b.code()->constants[i]) + "\n";
		
			r = r + "code: " + intToStr(b.code()->bc.size()) + "\n";
			for(int64_t i = 0; i < (int64_t)b.code()->bc.size(); i++)
				r = r + intToStr(i) + ":\t" + b.code()->bc[i].toString() + "\n";
		
			return r;
		}
		default:
			return value.type.toString();
	};
}
std::string State::stringify(Trace const & t) const {
	std::string r = "trace:\nconstants: " + intToStr(t.constants.size()) + "\n";
	for(int64_t i = 0; i < (int64_t)t.constants.size(); i++)
		r = r + intToStr(i) + "=\t" + stringify(t.constants[i]) + "\n";

	r = r + "code: " + intToStr(t.recorded.size()) + "\n";
	for(int64_t i = 0; i < (int64_t)t.recorded.size(); i++)
		r = r + intToStr(i) + ":\t" + t.recorded[i].toString() + "\n";

	return r;
}

template<class T> std::string deparse(State const& state, typename T::Element a) {
	return "";
}

template<> std::string deparse<Logical>(State const& state, Logical::Element a) {
	return Logical::isNA(a) ? "NA" : (a ? "TRUE" : "FALSE");
}  

template<> std::string deparse<Integer>(State const& state, Integer::Element a) {
	return Integer::isNA(a) ? "NA_integer_" : std::string("") + intToStr(a) + std::string("L");
}  

template<> std::string deparse<Double>(State const& state, Double::Element a) {
	return Double::isNA(a) ? "NA_real_" : doubleToStr(a);
}  

template<> std::string deparse<Complex>(State const& state, Complex::Element a) {
	return Complex::isNA(a) ? "NA_complex_" : complexToStr(a);
}  

template<> std::string deparse<Character>(State const& state, Character::Element a) {
	return Character::isNA(a) ? "NA_character_" : std::string("\"") + a.toString(state) + "\"";
}  

template<> std::string deparse<List>(State const& state, List::Element a) {
	return state.deparse(a);
}  

template<> std::string deparse<Call>(State const& state, Call::Element a) {
	return state.deparse(a);
}  

template<> std::string deparse<Expression>(State const& state, Expression::Element a) {
	return state.deparse(a);
}  

template<class T>
std::string deparseVectorBody(State const& state, T const& v) {
	std::string result = "";
	if(hasNames(v)) {
		Character c = getNames(v);
		for(int64_t i = 0; i < v.length; i++) {
			result = result + c[i].toString(state) + " = " + deparse<T>(state, v[i]);
			if(i < v.length-1) result = result + ", ";
		}
	}
	else {
		for(int64_t i = 0; i < v.length; i++) {
			result = result + deparse<T>(state, v[i]);
			if(i < v.length-1) result = result + ", ";
		}
	}
	return result;
}



template<class T>
std::string deparseVector(State const& state, T const& v) {
	if(v.length == 0) return std::string(v.type.toString()) + "(0)";
	if(v.length == 1) return deparseVectorBody(state, v);
	else return "c(" + deparseVectorBody(state, v) + ")";
}

template<>
std::string deparseVector<Call>(State const& state, Call const& v) {
	return state.deparse(Call(v)[0]) + "(" + deparseVectorBody(state, Subset(v, 1, v.length-1)) + ")";
}

template<>
std::string deparseVector<Expression>(State const& state, Expression const& v) {
	return "expression(" + deparseVectorBody(state, v) + ")";
}

std::string State::deparse(Value const& value) const {
	switch(value.type.Enum())
	{
		case Type::E_R_null:
			return "NULL";
		case Type::E_R_raw:
			return "raw";
		case Type::E_R_logical:
			return deparseVector(*this, Logical(value));
		case Type::E_R_integer:
			return deparseVector(*this, Integer(value));
		case Type::E_R_double:
			return deparseVector(*this, Double(value));
		case Type::E_R_complex:		
			return deparseVector(*this, Complex(value));
		case Type::E_R_character:
			return deparseVector(*this, Character(value));
		case Type::E_R_list:
			return deparseVector(*this, List(value));
		case Type::E_R_pairlist:
			return deparseVector(*this, PairList(value));
		case Type::E_R_call:
			return deparseVector(*this, Call(value));
		case Type::E_R_expression:
			return deparseVector(*this, Expression(value));
		case Type::E_R_symbol:
			return Symbol(value).toString(*this); // NYI: need to check if this should be backticked.
		case Type::E_R_function:
			return (Function(value).str()[0]).toString(*this);
		case Type::E_R_environment:
			return "environment";
		default:
			return value.type.toString();
	};
}

