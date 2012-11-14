
#include "value.h"
#include "type.h"
#include "bc.h"
#include "interpreter.h"

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

template<> std::string stringify<Raw>(State const& state, Raw::Element a) {
	return rawToStr(a);
}  

template<> std::string stringify<Integer>(State const& state, Integer::Element a) {
	return Integer::isNA(a) ? "NA" : std::string("") + intToStr(a) + std::string("L");
}  

template<> std::string stringify<Double>(State const& state, Double::Element a) {
	return Double::isNA(a) ? "NA" : doubleToStr(a);
}  

template<> std::string stringify<Character>(State const& state, Character::Element a) {
	return Character::isNA(a) ? "NA" : std::string("\"") + state.externStr(a) + "\"";
}  

template<> std::string stringify<List>(State const& state, List::Element a) {
	return state.stringify(a);
}  

template<class T>
std::string stringifyVector(State const& state, T const& v) {
	std::string result = "";
	int64_t length = v.length;
	if(length == 0)
		return std::string(Type::toString(v.VectorType)) + "(0)";

	bool dots = false;
	if(length > 100) { dots = true; length = 100; }
	int64_t maxlength = 1;
	for(int64_t i = 0; i < length; i++) {
		maxlength = std::max((int64_t)maxlength, (int64_t)stringify<T>(state, v[i]).length());
	}
	int64_t indexwidth = intToStr(length+1).length();
	int64_t perline = std::max(floor(80.0/(maxlength+1) + indexwidth), 1.0);
	for(int64_t i = 0; i < length; i+=perline) {
		result = result + pad(std::string("[") + intToStr(i+1) + "]", indexwidth+2);
		for(int64_t j = 0; j < perline && i+j < length; j++) {
			result = result + pad(stringify<T>(state, v[i+j]), maxlength+1);
		}

		if(i+perline < length)	
			result = result + "\n";
	}
	if(dots) result = result + " ... (" + intToStr(v.length) + " elements)";
	return result;
}

std::string stringify(State const& state, Value const& value) {
	std::string result = "[1]";
	bool dots = false;
	switch(value.type)
	{
		case Type::Null:
			return "NULL";
		case Type::Raw:
			return stringifyVector(state, (Raw const&)value);
		case Type::Logical:
			return stringifyVector(state, (Logical const&)value);
		case Type::Integer:
			return stringifyVector(state, (Integer const&)value);
		case Type::Double:
			return stringifyVector(state, (Double const&)value);
		case Type::Character:
			return stringifyVector(state, (Character const&)value);
		
		case Type::List:
		{
			List const& v = (List const&)value;

			int64_t length = v.length;
			if(length == 0) return "list()";
			if(length > 100) { dots = true; length = 100; }
			result = "";
			for(int64_t i = 0; i < length; i++) {
				result = result + "[[" + intToStr(i+1) + "]]\n";
				if(!List::isNA(v[i])) result = result + state.stringify(v[i]);
				result = result + "\n";
				if(i < length-1) result = result + "\n";
			}
			if(dots) result = result + " ... (" + intToStr(v.length) + " elements)";
			return result;
		}
		case Type::Function:
		{
			result = state.externStr(((Function const&)value).prototype()->string);
			return result;
		}
		case Type::Environment:
		{
			Environment* env = (REnvironment(value)).ptr();
			if(env == state.global)
				result = std::string("environment <global>");
			else
				result = std::string("environment <") + intToHexStr((int64_t)env) + ">";
			result = result + "\nVariables:\n";
			Dictionary* d = REnvironment(value).ptr();
			for(Dictionary::const_iterator i = d->begin(); i != d->end(); ++i) {
				result = result + "\t" + state.externStr(i.string())
						+ ":\t" + state.stringify(i.value()) + "\n";
			}
			return result;
		}
		case Type::Object:
		{
			Object const& o = (Object const&)value;
			result = stringify(state, o.base());
			result = result + "\nAttributes:\n";
			Dictionary* d = o.dictionary();
			for(Dictionary::const_iterator i = d->begin(); i != d->end(); ++i) {
				result = result + "\t" + state.externStr(i.string())
						+ ":\t" + state.stringify(i.value()) + "\n";
			}
			return result;
		}
		case Type::Future:
			return std::string("future") + intToStr(value.i);
		default:
			return Type::toString(value.type);
	};
}


std::string State::stringify(Value const& value) const {
	return ::stringify(*this, value);
}

#ifdef ENABLE_EPEE
std::string State::stringify(Trace const & t) const {
	return "NYI";
}
#endif

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

template<> std::string deparse<Character>(State const& state, Character::Element a) {
	return Character::isNA(a) ? "NA_character_" : std::string("\"") + state.externStr(a) + "\"";
}  

template<> std::string deparse<List>(State const& state, List::Element a) {
	return state.deparse(a);
}  

template<class T>
std::string deparseVectorBody(State const& state, T const& v) {
	std::string result = "";
	for(int64_t i = 0; i < v.length; i++) {
		result = result + deparse<T>(state, v[i]);
		if(i < v.length-1) result = result + ", ";
	}
	return result;
}



template<class T>
std::string deparseVector(State const& state, T const& v) {
	if(v.length == 0) return std::string(Type::toString(v.VectorType)) + "(0)";
	if(v.length == 1) return deparseVectorBody(state, v);
	else return "c(" + deparseVectorBody(state, v) + ")";
}
/*
template<>
std::string deparseVector<Call>(State const& state, Call const& v, Value const& names) {
	return state.deparse(Call(v)[0]) + "(" + deparseVectorBody(state, Subset(v, 1, v.length-1), names) + ")";
}

template<>
std::string deparseVector<Expression>(State const& state, Expression const& v, Value const& names) {
	return "expression(" + deparseVectorBody(state, v, names) + ")";
}
*/
std::string deparse(State const& state, Value const& value) {
	switch(value.type)
	{
		case Type::Null:
			return "NULL";
		#define CASE(Name) case Type::Name: return deparseVector(state, (Name const&)value); break;
		VECTOR_TYPES_NOT_NULL(CASE)
		#undef CASE
		case Type::Function:
			return state.externStr(((Function const&)value).prototype()->string);
		case Type::Environment:
			return "environment";
		case Type::Object:
			return deparse(state, ((Object const&)value).base());
		default:
			return Type::toString(value.type);
	};
}

std::string State::deparse(Value const& value) const {
	return ::deparse(*this, value);
}
