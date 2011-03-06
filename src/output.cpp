
#include "value.h"
#include "type.h"
#include "bc.h"

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
			uint64_t length = v.length();
			if(length > 100) { dots = true; length = 100; }
			for(uint64_t i = 0; i < length; i++) result = result + (v[i] ? "  TRUE" : " FALSE");
			if(dots) result = result + " ...";
			return result;
		}
		case Type::R_integer:
		{
			Integer v(value);
			uint64_t length = v.length();
			if(length > 100) { dots = true; length = 100; }
			for(uint64_t i = 0; i < length; i++) result = result + " " + intToStr(v[i]);
			if(dots) result = result + " ...";
			return result;
		}
		case Type::R_double:
		{
			Double v(value);
			uint64_t length = v.length();
			if(length > 100) { dots = true; length = 100; }
			for(uint64_t i = 0; i < length; i++) result = result + " " + doubleToStr(v[i]);
			if(dots) result = result + " ...";
			return result;
		}
		case Type::R_complex:		
		{
			//for(uint64_t i = 0; i < length; i++) result = result + ((int64_t*)ptr)[i];
			return "complex";
		}
		case Type::R_character:
		{
			Character v(value);
			uint64_t length = v.length();
			if(length > 100) { dots = true; length = 100; }
			for(uint64_t i = 0; i < length; i++) result = result + " \"" + outString(v[i]) + "\"";
			if(dots) result = result + " ...";
			return result;
		}
		case Type::R_list:
		case Type::R_pairlist:
		{
			List v(value);
			Value names = Value::null;//v.names();

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
