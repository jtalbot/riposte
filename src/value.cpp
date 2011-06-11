
#include "value.h"
#include "internal.h"
#include "compiler.h"

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
const Value Value::NIL = {{0}, {0}, 0, Type::I_nil}; 

#define ENUM_CONST_CONSTRUCT(name, string, EnumType) const EnumType EnumType::name(EnumType::E_##name);
SYMBOLS_ENUM(ENUM_CONST_CONSTRUCT, Symbol)

	SymbolTable::SymbolTable() : next(0) {
		// insert predefined symbols into table at known positions (corresponding to their enum value)
#define ENUM_STRING_TABLE(name, string, EnumType) \
		symbolTable[string] = EnumType::E_##name; \
		reverseSymbolTable[EnumType::E_##name] = string;\
		assert(next==EnumType::E_##name);\
		next++; 

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

	CompiledCall::CompiledCall(Call const& call, State& state) {
		inner = new Inner();
		inner->call = call;
		inner->dots = call.length-1;
		List arguments(call.length-1);
		for(int64_t i = 1; i < call.length; i++) {
			if(call[i].type == Type::R_symbol && Symbol(call[i]) == Symbol::dots) {
				arguments[i-1] = call[i];
				inner->dots = i-1;
			} else if(call[i].type == Type::R_call ||
			   call[i].type == Type::R_symbol ||
			   call[i].type == Type::I_promise ||
			   call[i].type == Type::R_pairlist) {
				arguments[i-1] = Compiler::compile(state, call[i]);
				arguments[i-1].type = Type::I_promise;
			} else if(call[i].type == Type::I_closure) {
				arguments[i-1] = call[i];
				arguments[i-1].type = Type::I_promise;
			} else {
				arguments[i-1] = call[i];
			}
		}
		if(hasNames(call)) {
			setNames(arguments, Subset(Character(getNames(call)), 1, call.length-1));
		}
		inner->arguments = arguments;
	}

