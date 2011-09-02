#ifndef _RIPOSTE_IR_H
#define _RIPOSTE_IR_H

#include "enum.h"
#include "common.h"

//order matters since T_logical < T_integer < T_double < T_complex for subtyping rules
#define IR_TYPE(_) \
	/*_(T_logical, 	"logical", sizeof(bool))*/		\
	/*_(T_integer, 	"integer", sizeof(int))*/		\
	_(T_double, 	"double", sizeof(double))		\
	/*_(T_complex, 	"complex", sizeof(double)*2 )*/		\
	/*_(T_character, 	"character", sizeof(char))*/	\
	/*_(T_size,       "size",sizeof(size_t)) */ /*type that holds the size of vectors*/\
	_(T_unsupported, "unsupported", 0)

DECLARE_ENUM(IRScalarType,IR_TYPE)

			\
/*	_(vload,      "vload",     ref, ___, ___) \
	_(rload,      "rload",     ref, ___, ___) \
	_(sload,      "sload",     ref, ___, ___) \
	_(cast,       "cast",      ref, ref, ___) \
	_(iassign,    "iassign",   ___, ___, ___) \
	_(guard,      "guard",     ___, ref, ___) \
	_(move,       "move",      ref, ref, ___) \

	_(pos,        "pos",       ref, ref, ___) \
	_(idiv,       "idiv",      ref, ref, ref) \
	_(mod,        "mod",       ref, ref, ref) \

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
	_(sign,       "sign",      ref, ref, ___) \

	_(istrue,     "istrue",    ref, ref, ___)\
	_(logical1,   "logical",   ref, ref, ___) \
	_(integer1,   "integer",   ref, ref, ___) \
	_(double1,    "double",    ref, ref, ___) \
	_(complex1,   "complex",   ref, ref, ___) \
	_(character1, "character", ref, ref, ___) \
	_(seq,        "seq",       ___, ___, ___) \
	_(signif,     "signif",    ref, ref, ___) \
*/
	
#define IR_BINARY(_) \
	_(add,        "add",       a + b) \
	_(sub,        "sub",       a - b) \
	_(mul,        "mul",       a * b) \
	_(div,        "div",       a / b) \
	_(pow,        "pow",       pow(a,b) ) \
	
#define IR_UNARY(_) \
	_(neg,        "neg",       - a) \
	_(sqrt,       "sqrt",     sqrt(a) ) \
	_(floor,      "floor",     floor(a)) \
	_(ceiling,    "ceiling",   ceil(a)) \
	_(round,      "round",     rint(a)) \
	_(exp,        "exp",       exp(a)) \
	_(log,        "log",       log(a)) \
	_(cos,        "cos",       cos(a)) \
	_(sin,        "sin",       sin(a)) \
	_(tan,        "tan",       tan(a)) \
	_(acos,       "acos",      acos(a)) \
	_(asin,       "asin",      asin(a)) \
	_(atan,       "atan",      atan(a)) \
	_(abs,        "abs",       abs(a)) \
	
#define IR_SPECIAL(_) \
	_(broadcast,  "broadcast") \
    _(vload,	  "vload") \

#define IR_ENUM(_) 	\
	IR_BINARY(_) \
	IR_UNARY(_) \
	IR_SPECIAL(_) \

	
    

DECLARE_ENUM(IROpCode,IR_ENUM)


class Value;

struct IRType {
	IRType() {}
	IRType(const Value & v);
	IRType(char data) {
		this->data = data;
	}
	IRType(IRScalarType::Enum base_type, bool isVector) {
		this->data = 0;
		this->base_type = base_type;
		this->isVector = isVector;
	}

	union {
		struct {
			IRScalarType::Enum base_type : 4;
			unsigned isVector : 1;
		};
		char data;
	};

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
	bool operator==(IRType const& t) const { return data == t.data; } \
    bool operator!=(IRType const& t) const { return !(*this == t); } \
	//utility functions
	//static IRType Bool() { return IRType(IRScalarType::T_logical,false); }
	//static IRType Int() { return IRType(IRScalarType::T_integer,false); }
	//static IRType Size() { return IRType(IRScalarType::T_size,false); }
	static IRType Double() { return IRType(IRScalarType::T_double,false); }
};

struct IRNode {
	IRNode() {}
	IRNode(IROpCode::Enum opcode, int64_t a, int64_t b) {
		this->opcode = opcode;
		this->a = a;
		this->b = b;
	}
	IROpCode::Enum opcode;

	//3-op code, r is dest, r = a + b, where r is the position in the list of IRNodes where this op resides
	union {
		int64_t a;
		double * reg_a; //reg_a holds the register allocated value where a resides (currently we are not register allocating)
		double const_a;
	};
	union {
		int64_t b;
		//double * reg_b;
	};

	std::string toString() const {
		std::ostringstream out;
		out << IROpCode::toString(opcode) << "\t" << a <<  "(" << const_a << ")" << "\t" << b;
		return out.str();
	}
};


typedef size_t IRef;

#endif /* IR_H_ */
