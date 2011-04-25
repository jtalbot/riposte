
#include "value.h"
#include "internal.h"

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

