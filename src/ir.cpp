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



#define IR_a____ 0
#define IR_b____ 0
#define IR_c____ 0
#define IR_a_ref (IRNode::REF_A)
#define IR_b_ref (IRNode::REF_B)
#define IR_c_ref (IRNode::REF_C)
#define MAKE_IR_FLAGS(op,str,p,a,b,c) (IR_a_##a | IR_b_##b | IR_c_##c),
static uint32_t ir_enum_flags[] = {IR_ENUM(MAKE_IR_FLAGS,0) 0};


uint32_t IRNode::flags() const {
	return ir_enum_flags[this->opcode.Enum()];
}
