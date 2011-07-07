#include "ir.h"
#include "value.h"

DEFINE_ENUM(IRScalarType,IR_TYPE)
DEFINE_ENUM_TO_STRING(IRScalarType, IR_TYPE)

DEFINE_ENUM(IROpCode,IR_ENUM)
DEFINE_ENUM_TO_STRING(IROpCode, IR_ENUM)

IRType::IRType(Value const & value) {
	switch(value.type.Enum()) {
	case Type::E_R_null: base_type = IRScalarType::T_null; break;
	case Type::E_R_logical: base_type = IRScalarType::T_logical; break;
	case Type::E_R_integer: base_type = IRScalarType::T_integer; break;
	case Type::E_R_double: base_type = IRScalarType::T_double; break;
	case Type::E_R_complex: base_type = IRScalarType::T_complex; break;
	case Type::E_R_character: base_type = IRScalarType::T_character; break;
	default: base_type = IRScalarType::T_unsupported; break;
	}
	isVector = value.length != 1;
}
