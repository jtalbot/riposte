
#ifndef _RIPOSTE_BC_H
#define _RIPOSTE_BC_H

#include "enum.h"
#include "common.h"

#define BYTECODES(_) 	\
	_(call, "call") \
	_(get, "get") \
	_(sget, "sget") \
	_(kget, "kget") \
	_(iget, "iget") \
	_(assign, "assign") \
	_(sassign, "sassign") \
	_(iassign, "iassign") \
	_(eassign, "eassign") \
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
	_(colon, "colon") \
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
	_(jmp, "jmp") \
	_(function, "function") \
	_(logical1, "logical") \
	_(integer1, "integer") \
	_(double1, "double") \
	_(complex1, "complex") \
	_(character1, "character") \
	_(raw1, "raw") \
	_(UseMethod, "UseMethod") \
	_(seq, "seq") \
	_(type, "type") \
	_(invoketrace,"invoketrace") \
	_(ret, "ret") \
	_(done, "done")       /* done must be the last instruction */

DECLARE_ENUM(ByteCode, BYTECODES)


struct Instruction {
	union {
		ByteCode::Enum bc;
		void const* ibc;
	};
	int64_t a, b, c;

	Instruction(ByteCode::Enum bc, int64_t a=0, int64_t b=0, int64_t c=0) :
		bc(bc), a(a), b(b), c(c) {}
	
	Instruction(void const* ibc, int64_t a=0, int64_t b=0, int64_t c=0) :
		ibc(ibc), a(a), b(b), c(c) {}
	
	std::string toString() const {
		return std::string("") + ByteCode::toString(bc) + "\t" + intToStr(a) + "\t" + intToStr(b) + "\t" + intToStr(c);
	}
};

#endif
