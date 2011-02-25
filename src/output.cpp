
#include "value.h"
#include "type.h"
#include "bc.h"

std::string Value::toString() const {
	
	std::string result = "[1]";
	bool dots = false;

	switch(type().internal())
	{
		case Type::R_null:
			return "NULL";
		case Type::R_raw:
			return "raw";
		case Type::R_logical:
		{
			Logical v(*this);
			uint64_t length = v.length();
			if(length > 100) { dots = true; length = 100; }
			for(uint64_t i = 0; i < length; i++) result = result + (v[i] ? "  TRUE" : " FALSE");
			if(dots) result = result + " ...";
			return result;
		}
		case Type::R_integer:
		{
			Integer v(*this);
			uint64_t length = v.length();
			if(length > 100) { dots = true; length = 100; }
			for(uint64_t i = 0; i < length; i++) result = result + " " + intToStr(v[i]);
			if(dots) result = result + " ...";
			return result;
		}
		case Type::R_double:
		{
			Double v(*this);
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
			Character v(*this);
			uint64_t length = v.length();
			if(length > 100) { dots = true; length = 100; }
			for(uint64_t i = 0; i < length; i++) result = result + " \"" + v[i] + "\"";
			if(dots) result = result + " ...";
			return result;
		}
		case Type::R_list:
		case Type::R_pairlist:
		{
			List v(*this);
			Value names = v.names();

			uint64_t length = v.length();
			if(length > 100) { dots = true; length = 100; }
			result = "";
			if(names.type() == Type::R_character) {
				Character n(names);
				for(uint64_t i = 0; i < length; i++) {
					result = result + "[[" + n[i] + "]]\n";
					result = result + v[i].toString() + "\n\n";
				}
			} else {
				for(uint64_t i = 0; i < length; i++) {
					result = result + "[[" + intToStr(i) + "]]\n";
					result = result + v[i].toString() + "\n\n";
				}
			}
			if(dots) result = result + " ...\n\n";
			return result;
		}
		case Type::R_symbol:
		{
			result = "`" + Symbol(*this).toString() + "`";
			return result;
		}
		case Type::R_function:
		{
			result = "function: " + intToHexStr((uint64_t)ptr()) /*+ "\n" + Function(*this).body().toString()*/;
			return result;
		}
		case Type::I_bytecode:
		{
			return Block(*this).toString();
		}
		default:
			return type().toString();
	};
}
