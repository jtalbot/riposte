
#ifndef _RIPOSTE_BC_H
#define _RIPOSTE_BC_H

#include "enum.h"
#include "common.h"

#define CONTROL_FLOW_BYTECODES(_) 	\
	_(jt, "jt") \
	_(jf, "jf") \
	_(jmp, "jmp") \
	_(call, "call") \
	_(icall, "icall") \
	_(UseMethod, "UseMethod") \
	_(ret, "ret") \
	_(forbegin, "forbegin") \
	_(forend, "forend") \

#define MEMORY_ACCESS_BYTECODES(_) \
	_(get, "get") \
	_(kget, "kget") \
	_(iget, "iget") \
	_(assign, "assign") \
	_(assign2, "assign2") \
	_(iassign, "iassign") \
	_(eassign, "eassign") \
	_(subset, "subset") \
	_(subset2, "subset2") \

#define UNARY_ARITH_MAP_BYTECODES(_) \
	_(pos, "pos", 	PosOp) \
	_(neg, "neg", 	NegOp) \
	_(abs, "abs", 	AbsOp) \
	_(sign, "sign",	SignOp) \
	_(sqrt, "sqrt",	SqrtOp) \
	_(floor, "floor",	FloorOp) \
	_(ceiling, "ceiling",	CeilingOp) \
	_(trunc, "trunc",	TruncOp) \
	_(round, "round",	RoundOp) \
	_(signif, "signif",	SignifOp) \
	_(exp, "exp",	ExpOp) \
	_(log, "log",	LogOp) \
	_(cos, "cos",	CosOp) \
	_(sin, "sin",	SinOp) \
	_(tan, "tan",	TanOp) \
	_(acos, "acos",	ACosOp) \
	_(asin, "asin",	ASinOp) \
	_(atan, "atan",	ATanOp) \

#define UNARY_LOGICAL_MAP_BYTECODES(_) \
	_(lnot, "lnot",	LNotOp) \

#define BINARY_ARITH_MAP_BYTECODES(_) \
	_(add, "add",	AddOp) \
	_(sub, "sub",	SubOp) \
	_(mul, "mul",	MulOp) \
	_(div, "div",	DivOp) \
	_(idiv, "idiv",	IDivOp) \
	_(mod, "mod",	ModOp) \
	_(pow, "pow",	PowOp) \

#define BINARY_LOGICAL_MAP_BYTECODES(_) \
	_(land, "land",	AndOp) \
	_(lor, "lor",	OrOp) \

#define BINARY_ORDINAL_MAP_BYTECODES(_) \
	_(eq, "eq",	EqOp) \
	_(neq, "neq",	NeqOp) \
	_(gt, "gt",	GTOp) \
	_(ge, "ge",	GEOp) \
	_(lt, "lt",	LTOp) \
	_(le, "le",	LEOp) \

#define ARITH_FOLD_BYTECODES(_) \
	_(sum, "sum",	SumOp) \
	_(prod, "prod",	ProdOp) \

#define ORDINAL_FOLD_BYTECODES(_) \
	_(min, "min",	MinOp) \
	_(max, "max",	MaxOp) \

#define LOGICAL_FOLD_BYTECODES(_) \
	_(any, "any",	AnyOp) \
	_(all, "all",	AllOp) \

#define ARITH_SCAN_BYTECODES(_) \
	_(cumsum, "cumsum",	SumOp) \
	_(cumprod, "cumprod",	ProdOp) \

#define ORDINAL_SCAN_BYTECODES(_) \
	_(cummin, "cummin",	MinOp) \
	_(cummax, "cummax",	MaxOp) \

#define LOGICAL_SCAN_BYTECODES(_) \
	_(cumany, "cumany",	AnyOp) \
	_(cumall, "cumall",	AllOp) \

#define UTILITY_BYTECODES(_)\
	_(sland, "sland") \
	_(slor, "slor") \
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
	_(list, "list") \
	_(length, "length")

#define SPECIAL_BYTECODES(_) 	\
	_(done, "done") 

#define MAP_BYTECODES(_) \
	UNARY_ARITH_MAP_BYTECODES(_) \
	UNARY_LOGICAL_MAP_BYTECODES(_) \
	BINARY_ARITH_MAP_BYTECODES(_) \
	BINARY_LOGICAL_MAP_BYTECODES(_) \
	BINARY_ORDINAL_MAP_BYTECODES(_) \

#define FOLD_BYTECODES(_) \
	ARITH_FOLD_BYTECODES(_) \
	ORDINAL_FOLD_BYTECODES(_) \
	LOGICAL_FOLD_BYTECODES(_) \

#define SCAN_BYTECODES(_) \
	ARITH_SCAN_BYTECODES(_) \
	ORDINAL_SCAN_BYTECODES(_) \
	LOGICAL_SCAN_BYTECODES(_) \

#define STANDARD_BYTECODES(_) \
	CONTROL_FLOW_BYTECODES(_) \
	MEMORY_ACCESS_BYTECODES(_) \
	MAP_BYTECODES(_) \
	FOLD_BYTECODES(_) \
	SCAN_BYTECODES(_) \
	UTILITY_BYTECODES(_) \

#define BYTECODES(_) \
	STANDARD_BYTECODES(_) \
	SPECIAL_BYTECODES(_)	

DECLARE_ENUM(ByteCode, BYTECODES)

struct Instruction {
	int64_t a, b, c;
	ByteCode::Enum bc;
	mutable void const* ibc;

	Instruction(ByteCode::Enum bc, int64_t a=0, int64_t b=0, int64_t c=0) :
		a(a), b(b), c(c), bc(bc), ibc(0) {}
	
	std::string toString() const {
		return std::string("") + ByteCode::toString(bc) + "\t" + intToStr(a) + "\t" + intToStr(b) + "\t" + intToStr(c);
	}
};

#endif
