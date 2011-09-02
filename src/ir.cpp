#include "ir.h"
#include "value.h"

DEFINE_ENUM_TO_STRING(IRScalarType, IR_TYPE)
DEFINE_ENUM_TO_STRING(IROpCode, IR_ENUM)

IRType::IRType(Value const & value) {
	isVector = value.length != 1;
	switch(value.type) {
	//case Type::Null: base_type = IRScalarType::T_null; isVector = 0; break;
	//case Type::Logical: base_type = IRScalarType::T_logical; break;
	//case Type::Integer: base_type = IRScalarType::T_integer; break;
	case Type::Double: base_type = IRScalarType::T_double; break;
	//case Type::Complex: base_type = IRScalarType::T_complex; break; (NYI in compiler)
	//case Type::Character: base_type = IRScalarType::T_character; break; (NYI in character)
	default: base_type = IRScalarType::T_unsupported; break;
	}

}