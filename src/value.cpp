
#include "value.h"
#include "internal.h"

_doublena doublena = {0x7fff000000001953};

const Null Null::singleton = Null(0);
const bool Character::CheckNA = true;
const Symbol Character::NAelement = Symbol::NA;
const bool Logical::CheckNA = true;
const unsigned char Logical::NAelement = 255;
const bool Double::CheckNA = false;
const double Double::NAelement = doublena.d;
const bool Complex::CheckNA = false;
const std::complex<double> Complex::NAelement = std::complex<double>(doublena.d, doublena.d);
const bool Integer::CheckNA = true;
const int64_t Integer::NAelement = std::numeric_limits<int64_t>::min();
const Value List::NAelement = Null::singleton;
const Value Value::Nil = {{0}, {0}, 0, Type::I_nil}; 

#define ENUM_CONST_CONSTRUCT(name, string, EnumType) const EnumType EnumType::name(EnumType::E_##name);
SYMBOLS_ENUM(ENUM_CONST_CONSTRUCT, Symbol)

	SymbolTable::SymbolTable() : next(0) {
		// insert predefined state into table at known positions (corresponding to their enum value)
#define ENUM_STRING_TABLE(name, string, EnumType) \
		symbolTable[string] = EnumType::E_##name; \
		reverseSymbolTable[EnumType::E_##name] = string;\
		assert(next==EnumType::E_##name);\
		next++;\

		SYMBOLS_ENUM(ENUM_STRING_TABLE,Symbol);
	}


	Function::Function(List const& parameters, Value const& body, Character const& str, Environment* s) 
		: inner(new Inner(parameters, body, str, s)), attributes(0) {
		inner->dots = parameters.length;
		if(parameters.length > 0) {
			Character names = getNames(parameters);
			int64_t i = 0;
			for(;i < names.length; i++) if(Symbol(names[i]) == Symbol::dots) inner->dots = i;
		}
	}

