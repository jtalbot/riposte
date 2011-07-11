#ifndef _RIPOSTE_IR_H
#define _RIPOSTE_IR_H

#include "enum.h"
#include "common.h"

//order matters since T_logical < T_integer < T_double < T_complex for subtyping rules
#define IR_TYPE(_,p) \
	_(T_null, 	"NULL", p)			\
	_(T_logical, 	"logical", p)		\
	_(T_integer, 	"integer", p)		\
	_(T_double, 	"double", p)		\
	_(T_complex, 	"complex", p)		\
	_(T_character, 	"character", p)		\
	_(T_void,		"void",p)	\
	_(T_size,       "size",p) /*type that holds the size of vectors*/\
	_(T_unsupported, "unsupported",p)

DECLARE_ENUM(IRScalarType,IR_TYPE)

#define IR_ENUM(_, p) 				\
	_(vload,      "vload",     p, ref, ___, ___) \
	_(sload,      "sload",     p, ref, ___, ___) \
	_(cast,       "cast",      p, ref, ref, ___) \
	_(iassign,    "iassign",   p, ___, ___, ___) \
	_(guard,      "guard",     p, ___, ref, ___) \
	_(move,       "move",      p, ref, ref, ___) \
	_(add,        "add",       p, ref, ref, ref) \
	_(pos,        "pos",       p, ref, ref, ___) \
	_(sub,        "sub",       p, ref, ref, ref) \
	_(neg,        "neg",       p, ref, ref, ___) \
	_(mul,        "mul",       p, ref, ref, ref) \
	_(div,        "div",       p, ref, ref, ref) \
	_(idiv,       "idiv",      p, ref, ref, ref) \
	_(mod,        "mod",       p, ref, ref, ref) \
	_(pow,        "pow",       p, ref, ref, ref) \
	_(lt,         "lt",        p, ref, ref, ref) \
	_(gt,         "gt",        p, ref, ref, ref) \
	_(eq,         "eq",        p, ref, ref, ref) \
	_(neq,        "neq",       p, ref, ref, ref) \
	_(ge,         "ge",        p, ref, ref, ref) \
	_(le,         "le",        p, ref, ref, ref) \
	_(lnot,       "lnot",      p, ref, ref, ___) \
	_(land,       "land",      p, ref, ref, ref) \
	_(lor,        "lor",       p, ref, ref, ref) \
	_(abs,        "abs",       p, ref, ref, ___) \
	_(sign,       "sign",      p, ref, ref, ___) \
	_(sqrt,       "sqrt",      p, ref, ref, ___) \
	_(floor,      "floor",     p, ref, ref, ___) \
	_(ceiling,    "ceiling",   p, ref, ref, ___) \
	_(trunc, "    trunc",      p, ref, ref, ___) \
	_(round,      "round",     p, ref, ref, ___) \
	_(signif,     "signif",    p, ref, ref, ___) \
	_(exp,        "exp",       p, ref, ref, ___) \
	_(log,        "log",       p, ref, ref, ___) \
	_(cos,        "cos",       p, ref, ref, ___) \
	_(sin,        "sin",       p, ref, ref, ___) \
	_(tan,        "tan",       p, ref, ref, ___) \
	_(acos,       "acos",      p, ref, ref, ___) \
	_(asin,       "asin",      p, ref, ref, ___) \
	_(atan,       "atan",      p, ref, ref, ___) \
	_(istrue,     "istrue",    p, ref, ref, ___) \
	_(logical1,   "logical",   p, ref, ref, ___) \
	_(integer1,   "integer",   p, ref, ref, ___) \
	_(double1,    "double",    p, ref, ref, ___) \
	_(complex1,   "complex",   p, ref, ref, ___) \
	_(character1, "character", p, ref, ref, ___) \
	_(seq,        "seq",       p, ___, ___, ___) \
    _(kload,      "kload",     p, ref, ___, ___)
/*	_(type, "type", p) currently always constant folded into a kload  \
 	_(NA, "NA", p) will be included when we handle NA types \
    _(if1, "if1", p) will be included when we do trace merging
*/

DECLARE_ENUM(IROpCode,IR_ENUM)


class Value;

struct IRType {
	IRType() {}
	IRType(const Value & v);
	IRType(IRScalarType base_type, bool isVector) {
		this->base_type = base_type;
		this->isVector = isVector;
	}
	IRScalarType base_type;
	bool isVector;

	std::string toString() const {
		std::string bt = base_type.toString();
		if(isVector) {
			return "[" + bt + "]";
		} else
			return bt;
	}
	IRType base() const {
		return IRType(base_type,false);
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
	IRNode(IROpCode opcode, IRType const & typ, int64_t a, int64_t b) {
		this->opcode = opcode;
		this->typ = typ;
		this->a = a;
		this->b = b;
	}
	IROpCode opcode;
	IRType typ;
	int64_t a, b; //3-op code, r is dest, r = a + b, where r is the position in the list of IRNodes where this op resides

	std::string toString() const {
		std::string aprefix = (flags() & REF_A) != 0 ? "r" : "";
		std::string bprefix = (flags() & REF_B) != 0 ? "r" : "";
		return std::string("") + opcode.toString() + "(" + typ.toString() +  ")\t" + aprefix + intToStr(a) + "\t" + bprefix + intToStr(b);
	}
	uint32_t flags() const;
};


typedef size_t IRef;

#endif /* IR_H_ */
