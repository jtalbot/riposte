
#ifndef _RIPOSTE_BC_H
#define _RIPOSTE_BC_H

#include "enum.h"
#include "common.h"
#include "strings.h"

#define CONTROL_FLOW_BYTECODES(_) 	\
	_(jc, "jc") \
	_(jmp, "jmp") \
	_(branch, "branch") \
	_(call, "call") \
	_(ncall, "ncall") \
	_(ret, "ret") /* return from function */ \
	_(rets, "rets") /* return from top-level statement */ \
	_(retp, "retp") /* return from a promise or default */ \
	_(forbegin, "forbegin") \
	_(forend, "forend") \
	_(list, "list") \
	_(dotslist, "dotslist") \

#define MEMORY_ACCESS_BYTECODES(_) \
	_(mov, "mov") \
	_(fastmov, "fastmov") \
	_(dotdot, "dotdot") \
	_(assign, "assign") \
	_(assign2, "assign2") \
	_(iassign, "iassign") \
	_(eassign, "eassign") \
	_(subset, "subset") \
	_(subset2, "subset2") \

#define UTILITY_BYTECODES(_)\
	_(internal, "internal") \
	_(function, "function") \
	_(type, "type") \
	_(missing, "missing") \
	_(strip, "strip") \

#define GENERATOR_BYTECODES(_) \
	_(vector,	"vector",	Generator) \
	_(seq,		"seq", 		Generator) \
	_(index,	"index", 	Generator) \
	_(random,	"random", 	Generator) \

// ArithUnary1 ops perform Integer->Integer, ArithUnary2 ops perform Integer->Double
#define ARITH_UNARY_BYTECODES(_) \
	_(pos, "pos", 	ArithUnary1, 	PassNA(a, a)) \
	_(neg, "neg", 	ArithUnary1, 	PassNA(a, -a)) \
	_(abs, "abs", 	ArithUnary1, 	PassNA(a, Abs(a))) \
	_(sign, "sign",	ArithUnary2, 	((a>0)-(a<0))) \
	_(sqrt, "sqrt",	ArithUnary2,	sqrt(a)) \
	_(floor, "floor",	ArithUnary2,	floor(a)) \
	_(ceiling, "ceiling",	ArithUnary2,	ceil(a)) \
	_(trunc, "trunc",	ArithUnary2,	trunc(a)) \
	_(exp, "exp",	ArithUnary2,	exp(a)) \
	_(log, "log",	ArithUnary2,	log(a)) \
	_(cos, "cos",	ArithUnary2,	cos(a)) \
	_(sin, "sin",	ArithUnary2,	sin(a)) \
	_(tan, "tan",	ArithUnary2,	tan(a)) \
	_(acos, "acos",	ArithUnary2,	acos(a)) \
	_(asin, "asin",	ArithUnary2,	asin(a)) \
	_(atan, "atan",	ArithUnary2,	atan(a)) \

#define LOGICAL_UNARY_BYTECODES(_) \
	_(lnot, "lnot",	LogicalUnary, PassNA(a, ~a)) \

#define ORDINAL_UNARY_BYTECODES(_) \
	_(isna,	"isna",	OrdinalUnary,	MA::isNA(a)?-1:0) \
	_(isnan,	"isnan",	OrdinalUnary,	MA::isNaN(a)?-1:0) \
	_(isfinite,	"isfinite",	OrdinalUnary,	MA::isFinite(a)?-1:0) \
	_(isinfinite,	"isinfinite",	OrdinalUnary,	MA::isInfinite(a)?-1:0) \
/*
#define STRING_UNARY_BYTECODES(_) \
	_(nchar, "nchar", nchar, StringUnary, PassNA(a, strlen(a))) \
	_(nzchar, "nzchar", nzchar, StringUnary, PassNA(a, (*a>0))) \
*/
// ArithBinary1 ops perform Integer*Integer->Integer, ArithBinary2 ops perform Integer*Integer->Double
#define ARITH_BINARY_BYTECODES(_) \
	_(add, "add",	ArithBinary1,	PassNA(a,b,a+b)) \
	_(sub, "sub",	ArithBinary1,	PassNA(a,b,a-b)) \
	_(mul, "mul",	ArithBinary1,	PassNA(a,b,a*b)) \
	_(div, "div",	ArithBinary2,	a/b) \
	_(idiv, "idiv",	ArithBinary1,	PassNA(a,b,IDiv(a,b))) \
	_(mod, "mod",	ArithBinary1,	PassNA(a,b,Mod(a,b))) \
	_(pow, "pow",	ArithBinary2,	pow(a,b)) \
	_(atan2, "atan2",	ArithBinary2,	atan2(a,b)) \
	_(hypot, "hypot",	ArithBinary2,	hypot(a,b)) \

#define LOGICAL_BINARY_BYTECODES(_) \
	_(lor, "lor",	LogicalBinary,	a|b) \
	_(land, "land",	LogicalBinary,	a&b) \

#define UNIFY_BINARY_BYTECODES(_) \
	_(pmin, "pmin",	UnifyBinary,	PassNA(a,b,riposte_min(thread,a,b))) \
	_(pmax, "pmax",	UnifyBinary,	PassNA(a,b,riposte_max(thread,a,b))) \

#define ROUND_BINARY_BYTECODES(_) \
	_(round, "round", 	RoundBinary,	PassNA(a,b,riposte_round(thread,a,b))) \
	_(signif, "signif",	RoundBinary,	PassNA(a,b,riposte_signif(thread,a,b))) \

#define ORDINAL_BINARY_BYTECODES(_) \
	_(eq, "eq",	OrdinalBinary,	PassNA(a,b,a==b?-1:0)) \
	_(neq, "neq",	OrdinalBinary,	PassNA(a,b,a!=b?-1:0)) \
	_(gt, "gt",	OrdinalBinary,	PassNA(a,b,gt(thread,a,b)?-1:0)) \
	_(ge, "ge",	OrdinalBinary,	PassNA(a,b,ge(thread,a,b)?-1:0)) \
	_(lt, "lt",	OrdinalBinary,	PassNA(a,b,lt(thread,a,b)?-1:0)) \
	_(le, "le",	OrdinalBinary,	PassNA(a,b,le(thread,a,b)?-1:0)) \

#define SPECIAL_MAP_BYTECODES(_) \
	_(ifelse, "ifelse", IfElse) \
	_(split, "split", Split) \

#define ARITH_FOLD_BYTECODES(_) \
	_(sum, "sum",	ArithFold, 	add) \
	_(prod, "prod",	ArithFold, 	mul) \

#define LOGICAL_FOLD_BYTECODES(_) \
	_(any, "any",	LogicalFold, 	lor) \
	_(all, "all",	LogicalFold, 	land) \

#define UNIFY_FOLD_BYTECODES(_) \
	_(min, "min",	UnifyFold, 	pmin) \
	_(max, "max",	UnifyFold, 	pmax) \

#define SPECIAL_FOLD_BYTECODES(_) \
	_(length, "length", CountFold) \
	_(mean, "mean", MomentFold) \
	_(cm2, "cm2", Moment2Fold) \

#define ARITH_SCAN_BYTECODES(_) \
	_(cumsum, "cumsum",	ArithScan,	add) \
	_(cumprod, "cumprod",	ArithScan,	mul) \

#define UNIFY_SCAN_BYTECODES(_) \
	_(cummin, "cummin",	UnifyScan,	pmin) \
	_(cummax, "cummax",	UnifyScan,	pmax) \

#define SPECIAL_BYTECODES(_) 	\
	_(done, "done") 

#define UNARY_BYTECODES(_) \
	ARITH_UNARY_BYTECODES(_) \
	LOGICAL_UNARY_BYTECODES(_) \
	ORDINAL_UNARY_BYTECODES(_) \

#define BINARY_BYTECODES(_) \
	ARITH_BINARY_BYTECODES(_) \
	LOGICAL_BINARY_BYTECODES(_) \
	UNIFY_BINARY_BYTECODES(_) \
	ORDINAL_BINARY_BYTECODES(_) \
	ROUND_BINARY_BYTECODES(_) \

#define MAP_BYTECODES(_) \
	UNARY_BYTECODES(_) \
	BINARY_BYTECODES(_) \

#define FOLD_BYTECODES(_) \
	ARITH_FOLD_BYTECODES(_) \
	LOGICAL_FOLD_BYTECODES(_) \
	UNIFY_FOLD_BYTECODES(_) \

#define SCAN_BYTECODES(_) \
	ARITH_SCAN_BYTECODES(_) \
	UNIFY_SCAN_BYTECODES(_) \

#define STANDARD_BYTECODES(_) \
	CONTROL_FLOW_BYTECODES(_) \
	MEMORY_ACCESS_BYTECODES(_) \
	GENERATOR_BYTECODES(_) \
	MAP_BYTECODES(_) \
	SPECIAL_MAP_BYTECODES(_) \
	FOLD_BYTECODES(_) \
	SCAN_BYTECODES(_) \
	UTILITY_BYTECODES(_) \
	SPECIAL_FOLD_BYTECODES(_) \

#define BYTECODES(_) \
	STANDARD_BYTECODES(_) \
	SPECIAL_BYTECODES(_)	

#define ARITH_BYTECODES(_) \
	ARITH_FOLD_BYTECODES(_) \
	ARITH_SCAN_BYTECODES(_) \
	ARITH_UNARY_BYTECODES(_) \
	ARITH_BINARY_BYTECODES(_) \

#define ORDINAL_BYTECODES(_) \
	ORDINAL_BINARY_BYTECODES(_)

#define LOGICAL_BYTECODES(_) \
	LOGICAL_UNARY_BYTECODES(_) \

#define UNARY_FOLD_SCAN_BYTECODES(_) \
	UNARY_BYTECODES(_) \
	FOLD_BYTECODES(_) \
	SCAN_BYTECODES(_) \

DECLARE_ENUM(ByteCode, BYTECODES)

struct Instruction {
	int64_t a, b, c;
	ByteCode::Enum bc;
	mutable void const* ibc;

	Instruction(ByteCode::Enum bc, int64_t a=0, int64_t b=0, int64_t c=0) :
		a(a), b(b), c(c), bc(bc), ibc(0) {}
	
	std::string regToStr(int64_t a) const {
		//if(a <= 0) return intToStr(-a);
		//else return std::string((String)a);
		return intToStr(a);
	}

	std::string toString() const {
		return std::string("") + ByteCode::toString(bc) + "\t" + regToStr(a) + "\t" + regToStr(b) + "\t" + regToStr(c);
	}
};

#endif
