
#ifndef _RIPOSTE_BC_H
#define _RIPOSTE_BC_H

#include "enum.h"
#include "common.h"

#define BC_ENUM(_, p) 				\
	_(call, "call", p) \
	/*_(inlinecall, "inlinecall")*/ \
	_(get, "get", p) \
	_(kget, "kget", p) \
	_(iget, "iget", p) \
	_(assign, "assign", p) \
	_(classassign, "classassign", p) \
	_(namesassign, "namesassign", p) \
	_(dimassign, "dimassign", p) \
	_(iassign, "assign", p) \
	_(iclassassign, "classassign", p) \
	_(inamesassign, "namesassign", p) \
	_(idimassign, "dimassign", p) \
	_(forbegin, "forbegin", p) \
	_(forend, "forend", p) \
	_(iforbegin, "iforbegin", p) \
	_(iforend, "iforend", p) \
	_(whilebegin, "whilebegin", p) \
	_(whileend, "whileend", p) \
	_(repeatbegin, "repeatbegin", p) \
	_(repeatend, "repeatend", p) \
	_(next, "next", p) \
	_(break1, "break1", p) \
	_(if1, "if1", p) \
	_(endif1, "endif1", p) \
	_(colon, "colon", p) \
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
	_(sland, "sland", p) \
	_(lor, "lor", p) \
	_(slor, "slor", p) \
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
	_(jmp, "jmp", p) \
	_(null, "null", p) \
	_(function, "function", p) \
	_(ret, "ret", p)       /* ret must be the last instruction */
	/*_(frame, "frame") \
	_(rawfunction, "rawfunction") \
	_(quote, "quote") \
	_(force, "force") \
	_(forceall, "forceall") \
	_(code, "code") \
	_(slot, "slot") \
	_(zip2, "zip2") \*/

DECLARE_ENUM(ByteCode, BC_ENUM)


struct Instruction {
	union {
		ByteCode bc;
		void const* ibc;
	};
	int64_t a, b, c;

	Instruction(ByteCode bc, int64_t a=0, int64_t b=0, int64_t c=0) :
		bc(bc), a(a), b(b), c(c) {}
	
	Instruction(void const* ibc, int64_t a=0, int64_t b=0, int64_t c=0) :
		ibc(ibc), a(a), b(b), c(c) {}
	
	std::string toString() const {
		return std::string("") + bc.toString() + "\t" + intToStr(a) + "\t" + intToStr(b) + "\t" + intToStr(c);
	}
};

#endif
