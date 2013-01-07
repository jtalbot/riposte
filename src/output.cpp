
#include "value.h"
#include "type.h"
#include "bc.h"
#include "interpreter.h"
#include "parser.h"

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
	return Integer::isNA(a) ? "NA" : std::string("") + intToStr(a);// + std::string("L");
}  

template<> std::string stringify<Double>(State const& state, Double::Element a) {
	return Double::isNA(a) ? "NA" : doubleToStr(a);
}  

template<> std::string stringify<Character>(State const& state, Character::Element a) {
	return Character::isNA(a) ? "NA" : std::string("\"") + escape(state.externStr(a)) + "\"";
}  

template<> std::string stringify<List>(State const& state, List::Element a) {
	return state.stringify(a);
}  

template<class T>
std::string stringifyVector(State const& state, T const& v) {
	std::string result = "";
	int64_t length = v.length();
	if(length == 0)
		return std::string(Type::toString(v.ValueType)) + "(0)";

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
	if(dots) result = result + " ... (" + intToStr(v.length()) + " elements)";
	return result;
}

struct Format {
    bool scientific;
    uint64_t sdecimals;
    uint64_t fdecimals;
};

Format format(double d, int64_t maxsf) {
    Format f = (Format){ false, 0, 0 };
    if(std::isnan(d)) return f;
    else if(d == std::numeric_limits<double>::infinity()) return f;
    else if(d == -std::numeric_limits<double>::infinity()) return f;
    if(d == 0) return f;
   
    double e = fabs(d);

    // location of largest non-zero digit
    int64_t p = (int64_t)floor(log10(e));
   
    // count non-zero digits after p
    // remember last actual non-zero 
    int64_t q = p, qq = p;
    double q10 = pow(10, -q);

    while((e*q10 - floor(e*q10) != 0) && (p-q) < (maxsf-1)) {
        q -=1;
        q10 *= 10;
        if(fmod(floor(e*q10), 10.0) != 0)
            qq = q;
    }
    q = qq;

    f.sdecimals = p-q+1;
    f.fdecimals = std::max( (int64_t)0, -q );

    // now figure out how to format
    if(p >= 12 || p <= -4) {
        f.scientific = true;
    }
    else if(q >= 6) {
        f.scientific = true;
    }

    return f;
}

std::string stringify(State const& state, Double::Element a, Format f) {
	return Double::isNA(a) ? "NA" : doubleToStr(a, f.scientific ? f.sdecimals : f.fdecimals, !f.scientific);
}  

template<>
std::string stringifyVector<Double>(State const& state, Double const& v) {
	std::string result = "";
	int64_t length = v.length();
	if(length == 0)
		return std::string(Type::toString(v.ValueType)) + "(0)";

	bool dots = false;
	if(length > 100) { dots = true; length = 100; }

    Format f = { false, 0, 0 };
    for(uint64_t i = 0; i < length; i++) {
        Format tf = format(v[i], 7);
        f.scientific |= tf.scientific;
        f.sdecimals = std::max(f.sdecimals, tf.sdecimals);
        f.fdecimals = std::max(f.fdecimals, tf.fdecimals);
    }
	
    int64_t maxlength = 1;
	for(int64_t i = 0; i < length; i++) {
		maxlength = std::max((int64_t)maxlength, (int64_t)stringify(state, v[i], f).length());
	}
	int64_t indexwidth = intToStr(length+1).length();
	int64_t perline = std::max(floor(80.0/(maxlength+1) + indexwidth), 1.0);
	for(int64_t i = 0; i < length; i+=perline) {
		result = result + pad(std::string("[") + intToStr(i+1) + "]", indexwidth+2);
		for(int64_t j = 0; j < perline && i+j < length; j++) {
			result = result + pad(stringify(state, v[i+j], f), maxlength+1);
		}

		if(i+perline < length)	
			result = result + "\n";
	}
	if(dots) result = result + " ... (" + intToStr(v.length()) + " elements)";
	return result;
}

std::string stringify(State const& state, Value const& value, std::vector<int64_t> nest) {
	std::string result = "[1]";
	bool dots = false;
	switch(value.type())
	{
		case Type::Null:
			result = "NULL";
			break;
		case Type::Raw:
			result = stringifyVector(state, (Raw const&)value);
			break;
		case Type::Logical:
			result = stringifyVector(state, (Logical const&)value);
			break;
		case Type::Integer:
			result = stringifyVector(state, (Integer const&)value);
			break;
		case Type::Double:
			result = stringifyVector(state, (Double const&)value);
			break;
		case Type::Character:
			result = stringifyVector(state, (Character const&)value);
			break;
		
		case Type::List:
		{
			List const& v = (List const&)value;

			int64_t length = v.length();
			if(length == 0) return "list()";
			if(length > 100) { dots = true; length = 100; }
			result = "";

            std::string prefix = "";
            for(int64_t i = 0; i < nest.size(); ++i) {
                prefix = prefix + "[[" + intToStr(nest[i]) + "]]";
            }

            nest.push_back(0);
			for(int64_t i = 0; i < length; i++) {
                nest.back()++;
				result = result + prefix + "[[" + intToStr(i+1) + "]]\n";
				if(!List::isNA(v[i])) 
                    result = result + stringify(state, v[i], nest);
				result = result + "\n";
				if(i < length-1) result = result + "\n";
			}
			if(dots) result = result + " ... (" + intToStr(v.length()) + " elements)";
		} break;
		case Type::Function:
		{
			result = state.externStr(((Function const&)value).prototype()->string);
		} break;
		case Type::Environment:
		{
			REnvironment const& renv = (REnvironment const&)value;
			if(renv.environment() == state.global)
				result = std::string("environment <global>");
			else
				result = std::string("environment <") + intToHexStr((int64_t)renv.environment()) + ">";
			result = result + "\nVariables:\n";
			Dictionary* d = renv.environment();
			for(Dictionary::const_iterator i = d->begin(); i != d->end(); ++i) {
				result = result + "\t" + state.externStr(i.string())
						+ ":\t" + state.stringify(i.value()) + "\n";
			}
			return result;
		} break;
		case Type::Future:
			result = std::string("future") + intToStr(value.i);
			break;
		default:
			result = Type::toString(value.type());
			break;
	};
	if(value.isObject() && ((Object const&)value).hasAttributes()) {
		result = result + "\nAttributes:\n";
		Dictionary* d = ((Object const&)value).attributes();
		for(Dictionary::const_iterator i = d->begin(); i != d->end(); ++i) {
			result = result + "\t" + state.externStr(i.string())
				+ ":\t" + state.stringify(i.value()) + "\n";
		}
	}
	return result;
}


std::string State::stringify(Value const& value) const {
    std::vector<int64_t> emptyNest;
	return ::stringify(*this, value, emptyNest);
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

template<> std::string deparse<Character>(State const& state, Character::Element a) {
	return Character::isNA(a) ? "NA_character_" : std::string("\"") + state.externStr(a) + "\"";
}  

template<> std::string deparse<List>(State const& state, List::Element a) {
	return state.deparse(a);
}  

template<class T>
std::string deparseVectorBody(State const& state, T const& v) {
	std::string result = "";
	for(int64_t i = 0; i < v.length(); i++) {
		result = result + deparse<T>(state, v[i]);
		if(i < v.length()-1) result = result + ", ";
	}
	return result;
}



template<class T>
std::string deparseVector(State const& state, T const& v) {
	if(v.length() == 0) return std::string(Type::toString(v.ValueType)) + "(0)";
	if(v.length() == 1) return deparseVectorBody(state, v);
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
	switch(value.type())
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
		default:
			return Type::toString(value.type());
	};
}

std::string State::deparse(Value const& value) const {
	return ::deparse(*this, value);
}
