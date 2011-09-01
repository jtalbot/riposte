
#ifndef _RIPOSTE_BC_H
#define _RIPOSTE_BC_H

#include "enum.h"
#include "common.h"

#define CONTROL_FLOW_BYTECODES(_) 	\
	_(call, "call") \
	_(forbegin, "forbegin") \
	_(forend, "forend") \
	_(iforbegin, "iforbegin") \
	_(iforend, "iforend") \
	_(whilebegin, "whilebegin") \
	_(whileend, "whileend") \
	_(repeatbegin, "repeatbegin") \
	_(repeatend, "repeatend") \
	_(next, "next") \
	_(break1, "break1") \
	_(if1, "if1") \
	_(if0, "if0") \
	_(jmp, "jmp") \
	_(UseMethod, "UseMethod") \
	_(invoketrace,"invoketrace") \
	_(ret, "ret") \

#define MEMORY_ACCESS_BYTECODES(_) \
	_(get, "get") \
	_(sget, "sget") \
	_(kget, "kget") \
	_(iget, "iget") \
	_(assign, "assign") \
	_(sassign, "sassign") \
	_(iassign, "iassign") \
	_(eassign, "eassign") \
	_(subset, "subset") \
	_(subset2, "subset2") \

#define MATH_BYTECODES(_) \
	_(add, "add") \
	_(pos, "pos") \
	_(sub, "sub") \
	_(neg, "neg") \
	_(mul, "mul") \
	_(div, "div") \
	_(idiv, "idiv") \
	_(mod, "mod") \
	_(pow, "pow") \
	_(lt, "lt") \
	_(gt, "gt") \
	_(eq, "eq") \
	_(neq, "neq") \
	_(ge, "ge") \
	_(le, "le") \
	_(lnot, "lnot") \
	_(land, "land") \
	_(lor, "lor") \
	_(sland, "sland") \
	_(slor, "slor") \
	_(abs, "abs") \
	_(sign, "sign") \
	_(sqrt, "sqrt") \
	_(floor, "floor") \
	_(ceiling, "ceiling") \
	_(trunc, "trunc") \
	_(round, "round") \
	_(signif, "signif") \
	_(exp, "exp") \
	_(log, "log") \
	_(cos, "cos") \
	_(sin, "sin") \
	_(tan, "tan") \
	_(acos, "acos") \
	_(asin, "asin") \
	_(atan, "atan") \

#define UTILITY_BYTECODES(_)\
	_(colon, "colon") \
	_(function, "function") \
	_(logical1, "logical") \
	_(integer1, "integer") \
	_(double1, "double") \
	_(complex1, "complex") \
	_(character1, "character") \
	_(raw1, "raw") \
	_(seq, "seq") \
	_(type, "type") \

#define SPECIAL_BYTECODES(_) 	\
	_(done, "done")       /* done must be the last instruction */

#define BYTECODES(_) \
	CONTROL_FLOW_BYTECODES(_) \
	MEMORY_ACCESS_BYTECODES(_) \
	MATH_BYTECODES(_) \
	UTILITY_BYTECODES(_) \
	SPECIAL_BYTECODES(_)	

DECLARE_ENUM(ByteCode, BYTECODES)

struct Instruction {
	int64_t a, b, c;
	union {
		ByteCode::Enum bc;
		void const* ibc;
	};

	Instruction(ByteCode::Enum bc, int64_t a=0, int64_t b=0, int64_t c=0) :
		a(a), b(b), c(c), bc(bc) {}
	
	Instruction(void const* ibc, int64_t a=0, int64_t b=0, int64_t c=0) :
		a(a), b(b), c(c), ibc(ibc) {}
	
	std::string toString() const {
		return std::string("") + ByteCode::toString(bc) + "\t" + intToStr(a) + "\t" + intToStr(b) + "\t" + intToStr(c);
	}
};

#endif
