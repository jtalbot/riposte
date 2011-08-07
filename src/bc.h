
#ifndef _RIPOSTE_BC_H
#define _RIPOSTE_BC_H

#include "enum.h"
#include "common.h"

#define BC_ENUM(_, p) 				\
	_(call, "call", p) \
	_(get, "get", p) \
	_(sget, "sget", p) \
	_(kget, "kget", p) \
	_(iget, "iget", p) \
	_(assign, "assign", p) \
	_(sassign, "sassign", p) \
	_(iassign, "iassign", p) \
	_(eassign, "eassign", p) \
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
	_(if0, "if0", p) \
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
	_(lor, "lor", p) \
	_(sland, "sland", p) \
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
	_(function, "function", p) \
	_(logical1, "logical", p) \
	_(integer1, "integer", p) \
	_(double1, "double", p) \
	_(complex1, "complex", p) \
	_(character1, "character", p) \
	_(raw1, "raw", p) \
	_(UseMethod, "UseMethod", p) \
	_(seq, "seq", p) \
	_(type, "type", p) \
	_(invoketrace,"invoketrace",p) \
	_(ret, "ret", p) \
	_(done, "done", p)       /* done must be the last instruction */

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
