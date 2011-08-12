
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
	return Character::isNA(a) ? "NA" : std::string("\"") + state.SymToStr(a) + "\"";
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
		Character c = Character(getNames(v));
		for(int64_t i = 0; i < length; i++) {
			maxlength = std::max((int64_t)maxlength, (int64_t)state.SymToStr(c[i]).length());
		}
	}
	int64_t indexwidth = intToStr(length+1).length();
	int64_t perline = std::max(floor(80.0/(maxlength+1) + indexwidth), 1.0);
	if(hasNames(v)) {
		Character c = Character(getNames(v));
		for(int64_t i = 0; i < length; i+=perline) {
			result = result + pad("", indexwidth+2);
			for(int64_t j = 0; j < perline && i+j < length; j++) {
				result = result + pad(state.SymToStr(c[i+j]), maxlength+1);
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
				Character n = Character(getNames(v));
				for(int64_t i = 0; i < length; i++) {
					if(SymToStr(n[i])=="")
						result = result + "[[" + intToStr(i+1) + "]]\n";
					else
						result = result + "$" + SymToStr(n[i]) + "\n";
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
			result = "`" + SymToStr(Symbol(value)) + "`";
			return result;
		}
		case Type::E_R_function:
		{
			result = SymToStr(Symbol(Function(value).str()[0]));
			return result;
		}
		case Type::E_R_environment:
		{
			return std::string("environment <") + intToHexStr((uint64_t)REnvironment(value).ptr()) + "> (" + intToStr(REnvironment(value).ptr()->numVariables()) + " symbols defined)";
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


#define WRITE_OP(o,op) \
	do { \
		const IRNode & n = (op); \
		std::string result = (n.flags() & IRNode::REF_R) != 0 ?  ( "r" + intToStr(o)  + " =") : ""; \
		r = r + intToStr(i) + ":\t" + result + "\t" + n.toString() + "\n"; \
	} while(0)
#define WRITE_HEAD(name,size) \
	(r = r + name + ": " + intToStr(size) + "\n")

	std::string r = "trace:\n";

	WRITE_HEAD("constants",t.constants.size());

		for(size_t i = 0; i < t.constants.size(); i++)
			r = r + intToStr(i) + "=\t" + stringify(t.constants[i]) + "\n";

	WRITE_HEAD("recorded",t.recorded.size());

	for(size_t i = 0; i < t.recorded.size(); i++) {
		WRITE_OP(i,t.recorded[i]);
	}
	WRITE_HEAD("exits",t.exits.size());
	for(size_t i = 0; i < t.exits.size(); i++) {
		int32_t snapshot = t.exits[i].snapshot;
		r = r + intToStr(i) + "[" + intToStr(t.exits[i].offset)  + "]:\t";
		for(size_t j = 0; j < t.renaming_table.outputs.size(); j++) {
			const RenamingTable::Entry & e = t.renaming_table.outputs[j];
			if(e.location == RenamingTable::REG) {
				if(e.id >= t.exits[i].n_live_registers) //register is dead so it is not an output
					continue;
				r = r + "r";
			} else {
				r = r + "v";
			}
			r = r + intToStr(e.id) + " -> ";
			IRef node = -1;
			if(t.renaming_table.get(e.location,e.id,snapshot,false,&node,NULL,NULL)) {
				r = r + "r" + intToStr(node);
			} else {
				t.renaming_table.get(e.location,e.id,&node);
				r = r + "u[r" + intToStr(node) + "]";
			}
			r = r + "; ";
		}
		r = r + "\n";
	}

#define WRITE_CODE(arr) \
	do { \
	  WRITE_HEAD(#arr,t . arr . size()); \
	  for(size_t i = 0; i < t . arr . size(); i++) { \
		  WRITE_OP(t.arr[i], t.optimized[t.arr[i]]); \
	  } \
	} while(0)

	WRITE_CODE(loads);
	WRITE_CODE(phis);
	WRITE_CODE(loop_header);
	WRITE_CODE(loop_body);

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
	return Character::isNA(a) ? "NA_character_" : std::string("\"") + state.SymToStr(a) + "\"";
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
		Character c = Character(getNames(v));
		for(int64_t i = 0; i < v.length; i++) {
			result = result + state.SymToStr(c[i]) + " = " + deparse<T>(state, v[i]);
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
			return SymToStr(Symbol(value)); // NYI: need to check if this should be backticked.
		case Type::E_R_function:
			return SymToStr(Symbol(Function(value).str()[0]));
		case Type::E_R_environment:
			return "environment";
		default:
			return value.type.toString();
	};
}

