
#include "value.h"
#include "internal.h"

union {
	uint64_t i;
	double d;
} narep = {0x7fff000000001953};

const Null Null::singleton = Null(0);
const bool Character::CheckNA = true;
const uint64_t Character::NAelement = 0;
const Character Character::NA = Character(1);
const bool Logical::CheckNA = true;
const unsigned char Logical::NAelement = 255;
const Logical Logical::NA = Logical::c(255);
const Logical Logical::False = Logical::c(false);
const Logical Logical::True = Logical::c(true);
const bool Double::CheckNA = false;
const double Double::NAelement = narep.d;
const Double Double::NA = Double::c(narep.d);
const Double Double::NaN = Double::c(0);
const Double Double::Inf = Double::c(1);
const bool Integer::CheckNA = true;
const int64_t Integer::NAelement = INT_MIN;
const Integer Integer::NA = Integer::c(INT_MIN);
const Value Value::NIL = {{0}, 0, Type::I_nil, 0}; 


	Function::Function(List const& parameters, Value const& body, Character const& str, Environment* s) 
		: inner(new Inner(parameters, body, str, s)), attributes(0) {
		Character names(getNames(parameters.attributes));
		uint64_t i = 0;
		for(;i < names.length(); i++) if(names[i] == DOTS_STRING) inner->dots = i+1;
	}

	CompiledCall::CompiledCall(Call const& call, State& state) {
		inner = new Inner();
		inner->call = call;
		inner->dots = 0;
		uint64_t j = 0;
		List parameters(call.length()-1);
		for(uint64_t i = 1; i < call.length(); i++) {
			if(call[i].type == Type::R_symbol && call[i].i == DOTS_STRING) {
				parameters[j] = call[i];
				inner->dots = i;
			} else if(call[i].type == Type::R_call ||
			   call[i].type == Type::R_symbol ||
			   call[i].type == Type::I_promise ||
			   call[i].type == Type::R_pairlist) {
				parameters[j] = compile(state, call[i]);
				parameters[j].type = Type::I_promise;
			} else if(call[i].type == Type::I_closure) {
				parameters[j] = call[i];
				parameters[j].type = Type::I_promise;
			} else {
				parameters[j] = call[i];
			}
			j++;
		}
		Vector n = getNames(call.attributes);
		if(n.type != Type::R_null) {
			setNames(parameters.attributes, Subset(n, 1, call.length()-1));
		}
		inner->parameters = parameters;
	}

