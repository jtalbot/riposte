#ifndef _RIPOSTE_IR_H
#define _RIPOSTE_IR_H

#include "enum.h"
#include "common.h"

#define IR_TYPE(_,p) \
	_(T_null, 	"NULL", p)			\
	_(T_logical, 	"logical", p)		\
	_(T_integer, 	"integer", p)		\
	_(T_double, 	"double", p)		\
	_(T_complex, 	"complex", p)		\
	_(T_character, 	"character", p)		\

DECLARE_ENUM(IRScalarType,IR_TYPE)

#define IR_ENUM(_, p) 				\
	_(vload,"vload",p) \
	_(sload,"sload",p) \
	_(iassign,"iassign",p) \
	_(guard, "guard", p) \
	_(move,"move",p) \
	_(add, "add", p) \
	_(pos, "pos", p) \
	_(sub, "sub", p) \
	_(neg, "neg", p) \
	_(mul, "mul", p) \
	_(div, "div", p) \
	_(idiv, "idiv", p) \
	_(mod, "mod", p) \
	_(pow, "pow", p) \
	_(lt, "lt", p) \
	_(gt, "gt", p) \
	_(eq, "eq", p) \
	_(neq, "neq", p) \
	_(ge, "ge", p) \
	_(le, "le", p) \
	_(lnot, "lnot", p) \
	_(land, "land", p) \
	_(lor, "lor", p) \
	_(abs, "abs", p) \
	_(sign, "sign", p) \
	_(sqrt, "sqrt", p) \
	_(floor, "floor", p) \
	_(ceiling, "ceiling", p) \
	_(trunc, "trunc", p) \
	_(round, "round", p) \
	_(signif, "signif", p) \
	_(exp, "exp", p) \
	_(log, "log", p) \
	_(cos, "cos", p) \
	_(sin, "sin", p) \
	_(tan, "tan", p) \
	_(acos, "acos", p) \
	_(asin, "asin", p) \
	_(atan, "atan", p) \
	_(null, "null", p) \
	_(true1, "true", p) \
	_(false1, "false", p) \
	_(istrue, "istrue", p) \
	_(logical1, "logical", p) \
	_(integer1, "integer", p) \
	_(double1, "double", p) \
	_(complex1, "complex", p) \
	_(character1, "character", p) \
	_(seq, "seq", p) \
    _(kload,"kload",p)
/*	_(type, "type", p) currently always constant folded into a kload  \
 	_(NA, "NA", p) will be included when we handle NA types \
    _(if1, "if1", p) will be included when we do trace merging
*/

DECLARE_ENUM(IROpCode,IR_ENUM)

struct IRType {
	IRScalarType base_type;
	bool isVector;
};

struct IRNode {
	IROpCode opcode;
	IRType typ;
	int64_t a,b; //a and b are inputs that refer to the offset of the IRNode that defined them
};

#endif /* IR_H_ */
