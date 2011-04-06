
#ifndef _RIPOSTE_BC_H
#define _RIPOSTE_BC_H

#include "enum.h"
#include "common.h"

#define BC_ENUM(_) 				\
	_(call, "call") \
	_(inlinecall, "inlinecall") \
	_(get, "get") \
	_(kget, "kget") \
	_(assign, "assign") \
	_(classassign, "classassign") \
	_(namesassign, "namesassign") \
	_(dimassign, "dimassign") \
	_(iassign, "assign") \
	_(iclassassign, "classassign") \
	_(inamesassign, "namesassign") \
	_(idimassign, "dimassign") \
	_(pop, "pop") \
	_(forbegin, "forbegin") \
	_(forend, "forend") \
	_(whilebegin, "whilebegin") \
	_(whileend, "whileend") \
	_(repeatbegin, "repeatbegin") \
	_(repeatend, "repeatend") \
	_(if1, "if1") \
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
	_(lneg, "lneg") \
	_(land, "land") \
	_(sland, "sland") \
	_(lor, "lor") \
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
	_(null, "null") \
	_(ret, "ret")       /* ret must be the last instruction */
	/*_(frame, "frame") \
	_(function, "function") \
	_(rawfunction, "rawfunction") \
	_(quote, "quote") \
	_(force, "force") \
	_(forceall, "forceall") \
	_(code, "code") \
	_(slot, "slot") \
	_(zip2, "zip2") \*/

DECLARE_ENUM(ByteCode, BC_ENUM)


struct Instruction {
	ByteCode bc;
	void const* ibc;
	uint64_t a, b, c;

	Instruction(ByteCode bc, uint64_t a=0, uint64_t b=0, uint64_t c=0) :
		bc(bc), a(a), b(b), c(c) {}
	
	Instruction(void const* ibc, uint64_t a=0, uint64_t b=0, uint64_t c=0) :
		ibc(ibc), a(a), b(b), c(c) {}
	
	std::string toString() const {
		return std::string("") + bc.toString() + "\t" + intToStr(a);
	}
};


#endif
