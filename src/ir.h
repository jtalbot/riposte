#ifndef _RIPOSTE_IR_H
#define _RIPOSTE_IR_H

#include "enum.h"
#include "common.h"

//order matters since T_logical < T_integer < T_double < T_complex for subtyping rules
#define IR_TYPE(_) \
	_(T_null, 	"NULL", sizeof(int))			\
	_(T_logical, 	"logical", sizeof(bool))		\
	_(T_integer, 	"integer", sizeof(int))		\
	_(T_double, 	"double", sizeof(double))		\
	_(T_complex, 	"complex", sizeof(double)*2 )		\
	_(T_character, 	"character", sizeof(char))		\
	_(T_void,		"void",0)	\
	_(T_size,       "size",sizeof(size_t)) /*type that holds the size of vectors*/\
	_(T_unsupported, "unsupported", 0)

DECLARE_ENUM(IRScalarType,IR_TYPE)

#define IR_ENUM(_) 				\
	_(vload,      "vload",     ref, ___, ___) \
	_(rload,      "rload",     ref, ___, ___) \
	_(sload,      "sload",     ref, ___, ___) \
	_(cast,       "cast",      ref, ref, ___) \
	_(iassign,    "iassign",   ___, ___, ___) \
	_(guard,      "guard",     ___, ref, ___) \
	_(move,       "move",      ref, ref, ___) \
	_(add,        "add",       ref, ref, ref) \
	_(pos,        "pos",       ref, ref, ___) \
	_(sub,        "sub",       ref, ref, ref) \
	_(neg,        "neg",       ref, ref, ___) \
	_(mul,        "mul",       ref, ref, ref) \
	_(div,        "div",       ref, ref, ref) \
	_(idiv,       "idiv",      ref, ref, ref) \
	_(mod,        "mod",       ref, ref, ref) \
	_(pow,        "pow",       ref, ref, ref) \
	_(lt,         "lt",        ref, ref, ref) \
	_(gt,         "gt",        ref, ref, ref) \
	_(eq,         "eq",        ref, ref, ref) \
	_(neq,        "neq",       ref, ref, ref) \
	_(ge,         "ge",        ref, ref, ref) \
	_(le,         "le",        ref, ref, ref) \
	_(lnot,       "lnot",      ref, ref, ___) \
	_(land,       "land",      ref, ref, ref) \
	_(lor,        "lor",       ref, ref, ref) \
	_(sland,       "sland",      ref, ref, ref) \
	_(slor,        "slor",       ref, ref, ref) \
	_(abs,        "abs",       ref, ref, ___) \
	_(sign,       "sign",      ref, ref, ___) \
	_(sqrt,       "sqrt",      ref, ref, ___) \
	_(floor,      "floor",     ref, ref, ___) \
	_(ceiling,    "ceiling",   ref, ref, ___) \
	_(trunc,      "trunc",      ref, ref, ___) \
	_(round,      "round",     ref, ref, ___) \
	_(signif,     "signif",    ref, ref, ___) \
	_(exp,        "exp",       ref, ref, ___) \
	_(log,        "log",       ref, ref, ___) \
	_(cos,        "cos",       ref, ref, ___) \
	_(sin,        "sin",       ref, ref, ___) \
	_(tan,        "tan",       ref, ref, ___) \
	_(acos,       "acos",      ref, ref, ___) \
	_(asin,       "asin",      ref, ref, ___) \
	_(atan,       "atan",      ref, ref, ___) \
	_(istrue,     "istrue",    ref, ref, ___) \
	_(logical1,   "logical",   ref, ref, ___) \
	_(integer1,   "integer",   ref, ref, ___) \
	_(double1,    "double",    ref, ref, ___) \
	_(complex1,   "complex",   ref, ref, ___) \
	_(character1, "character", ref, ref, ___) \
	_(seq,        "seq",       ___, ___, ___) \
    _(kload,      "kload",     ref, ___, ___)
/*	_(type, "type", p) currently always constant folded into a kload  \
 	_(NA, "NA", p) will be included when we handle NA types \
    _(if1, "if1", p) will be included when we do trace merging
*/

DECLARE_ENUM(IROpCode,IR_ENUM)


class Value;

struct IRType {
	IRType() {}
	IRType(const Value & v);
	IRType(IRScalarType::Enum base_type, bool isVector) {
		this->base_type = base_type;
		this->isVector = isVector;
	}
	IRScalarType::Enum base_type;
	bool isVector;

	std::string toString() const {
		std::string bt = IRScalarType::toString(base_type);
		if(isVector) {
			return "[" + bt + "]";
		} else
			return bt;
	}
	IRType base() const {
		return IRType(base_type,false);
	}
	size_t width() const {
#define GET_WIDTH(x,y,w) w,
		static size_t widths[] = { IR_TYPE(GET_WIDTH) 0 };
#undef GET_WIDTH
		return widths[base_type];
	}
	bool operator==(IRType const& t) const { return base_type == t.base_type && isVector == t.isVector; } \
    bool operator!=(IRType const& t) const { return !(*this == t); } \
	//utility functions
	static IRType Void() { return IRType(IRScalarType::T_void,false); }
	static IRType Bool() { return IRType(IRScalarType::T_logical,false); }
	static IRType Int() { return IRType(IRScalarType::T_integer,false); }
	static IRType Size() { return IRType(IRScalarType::T_size,false); }
	static IRType Double() { return IRType(IRScalarType::T_double,false); }
};

struct IRNode {
	enum {REF_R = 1, REF_A = 2,  REF_B = 4};
	IRNode() {}
	IRNode(IROpCode::Enum opcode, IRType const & typ, int64_t a, int64_t b) {
		this->opcode = opcode;
		this->typ = typ;
		this->a = a;
		this->b = b;
	}
	IROpCode::Enum opcode;
	IRType typ;
	int64_t a, b; //3-op code, r is dest, r = a + b, where r is the position in the list of IRNodes where this op resides

	std::string toString() const {
		std::string aprefix = (flags() & REF_A) != 0 ? "r" : "";
		std::string bprefix = (flags() & REF_B) != 0 ? "r" : "";
		return std::string("") + IROpCode::toString(opcode) + "(" + typ.toString() +  ")\t" + aprefix + intToStr(a) + "\t" + bprefix + intToStr(b);
	}
	uint32_t flags() const;
};


typedef size_t IRef;

#endif /* IR_H_ */
