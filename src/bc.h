
#ifndef _RIPOSTE_BC_H
#define _RIPOSTE_BC_H

#include "enum.h"
#include "common.h"
#include "strings.h"

#define CONTROL_FLOW_BYTECODES(_) 	\
	_(call, "call") \
	_(ncall, "ncall") \
	_(ret, "ret") /* return from function */ \
	_(retp, "retp") /* return from a promise or default */ \
	_(rets, "rets") /* return from top-level statement */ \
    _(loop, "loop") \
	_(jc, "jc") \
	_(jmp, "jmp") \
	_(forbegin, "forbegin") \
	_(forend, "forend") \
	_(branch, "branch") \
	_(list, "list") \
	_(dotslist, "dotslist")

#define MEMORY_ACCESS_BYTECODES(_) \
	_(mov, "mov") \
	_(fastmov, "fastmov") \
	_(assign, "assign") \
	_(scatter1, "scatter1") \
	_(gather1, "gather1") \
	_(scatter, "scatter") \
	_(gather, "gather") \
	_(constant, "constant") \
	_(dotdot, "dotdot") \
	_(assign2, "assign2") \
    _(attrget, "attrget") \
    _(attrset, "attrset")

#define UTILITY_BYTECODES(_)\
	_(function, "function") \
	_(internal, "internal") \
	_(type, "type") \
	_(missing, "missing") \
	_(strip, "strip") \
	_(nargs, "nargs")

#define GENERATOR_BYTECODES(_) \
	_(seq,		"seq", 		Generator) \
	_(vector,	"vector",	Generator) \
	_(rep,		"rep", 		Generator) \
	_(random,	"random", 	Generator)

#define CAST_UNARY_BYTECODES(_) \
    _(aslogical, "as.logical", CastLogical, PassNA(a, a), 3) \
    _(asinteger, "as.integer", CastInteger, PassNA(a, a), 3) \
    _(asdouble, "as.double", CastDouble, PassNA(a, a), 3) \
    _(ascharacter, "as.character", CastCharacter, PassNA(a, a), 100) \
    _(aslist, "as.list", CastList, PassNA(a, a), 200)

// ArithUnary1 ops perform Integer->Integer, ArithUnary2 ops perform Integer->Double
#define ARITH_UNARY_BYTECODES(_) \
	_(pos, "pos", 	ArithUnary1, 	PassNA(a, a), 0) \
	_(neg, "neg", 	ArithUnary1, 	PassNA(a, -a), 1) \
	_(abs, "abs", 	ArithUnary1, 	PassNA(a, Abs(a)), 10) \
	_(sign, "sign",	ArithUnary2, 	((a>0)-(a<0)), 10) \
	_(floor, "floor",	ArithUnary2,	floor(a), 5) \
	_(ceiling, "ceiling",	ArithUnary2,	ceil(a), 5) \
	_(trunc, "trunc",	ArithUnary2,	trunc(a), 5) \
	_(sqrt, "sqrt",	ArithUnary2,	sqrt(a), 15) \
	_(exp, "exp",	ArithUnary2,	exp(a), 50) \
	_(log, "log",	ArithUnary2,	log(a), 50) \
	_(cos, "cos",	ArithUnary2,	cos(a), 50) \
	_(sin, "sin",	ArithUnary2,	sin(a), 50) \
	_(tan, "tan",	ArithUnary2,	tan(a), 50) \
	_(acos, "acos",	ArithUnary2,	acos(a), 100) \
	_(asin, "asin",	ArithUnary2,	asin(a), 100) \
	_(atan, "atan",	ArithUnary2,	atan(a), 100)

#define LOGICAL_UNARY_BYTECODES(_) \
	_(lnot, "lnot",	LogicalUnary, PassNA(a, ~a), 1)

#define ORDINAL_UNARY_BYTECODES(_) \
	_(isna,	"isna",	OrdinalUnary,	MA::isNA(a)?-1:0, 5) \
	_(isnan,	"isnan",	OrdinalUnary,	MA::isNaN(a)?-1:0, 5) \
	_(isfinite,	"isfinite",	OrdinalUnary,	MA::isFinite(a)?-1:0, 5) \
	_(isinfinite,	"isinfinite",	OrdinalUnary,	MA::isInfinite(a)?-1:0, 5)

/*
#define STRING_UNARY_BYTECODES(_) \
	_(nchar, "nchar", nchar, StringUnary, PassNA(a, strlen(a))) \
	_(nzchar, "nzchar", nzchar, StringUnary, PassNA(a, (*a>0))) \
*/
// ArithBinary1 ops perform Integer*Integer->Integer, ArithBinary2 ops perform Integer*Integer->Double
#define ARITH_BINARY_BYTECODES(_) \
	_(add, "add",	ArithBinary1,	PassNA(a,b,a+b), 1) \
	_(sub, "sub",	ArithBinary1,	PassNA(a,b,a-b), 1) \
	_(mul, "mul",	ArithBinary1,	PassNA(a,b,a*b), 3) \
	_(div, "div",	ArithBinary2,	a/b, 20) \
	_(idiv, "idiv",	ArithBinary1,	PassNA(a,b,IDiv(a,b)), 30) \
	_(mod, "mod",	ArithBinary1,	PassNA(a,b,Mod(a,b)), 30) \
	_(pow, "pow",	ArithBinary2,	pow(a,b), 50) \
	_(atan2, "atan2",	ArithBinary2,	atan2(a,b), 100) \
	_(hypot, "hypot",	ArithBinary2,	hypot(a,b), 50)

#define LOGICAL_BINARY_BYTECODES(_) \
	_(lor, "lor",	LogicalBinary,	a|b, 1) \
	_(land, "land",	LogicalBinary,	a&b, 1) \

#define UNIFY_BINARY_BYTECODES(_) \
	_(pmin, "pmin",	UnifyBinary,	PassNA(a,b,riposte_min(thread,a,b)), 3) \
	_(pmax, "pmax",	UnifyBinary,	PassNA(a,b,riposte_max(thread,a,b)), 3)

#define ROUND_BINARY_BYTECODES(_) \
	_(round, "round", 	RoundBinary,	PassNA(a,b,riposte_round(thread,a,b)), 5) \
	_(signif, "signif",	RoundBinary,	PassNA(a,b,riposte_signif(thread,a,b)), 5)

#define ORDINAL_BINARY_BYTECODES(_) \
	_(eq, "eq",	OrdinalBinary,	PassNA(a,b,a==b?-1:0), 1) \
	_(neq, "neq",	OrdinalBinary,	PassNA(a,b,a!=b?-1:0), 1) \
	_(gt, "gt",	OrdinalBinary,	PassNA(a,b,gt(thread,a,b)?-1:0), 1) \
	_(ge, "ge",	OrdinalBinary,	PassNA(a,b,ge(thread,a,b)?-1:0), 1) \
	_(lt, "lt",	OrdinalBinary,	PassNA(a,b,lt(thread,a,b)?-1:0), 1) \
	_(le, "le",	OrdinalBinary,	PassNA(a,b,le(thread,a,b)?-1:0), 1) \

#define TERNARY_BYTECODES(_) \
	_(ifelse, "ifelse", IfElse, , 3) \
	_(split, "split", Split, , 10)

#define ARITH_FOLD_BYTECODES(_) \
	_(sum, "sum",	ArithFold, 	add, 1) \
	_(prod, "prod",	ArithFold, 	mul, 3)

#define LOGICAL_FOLD_BYTECODES(_) \
	_(any, "any",	LogicalFold, 	lor, 1) \
	_(all, "all",	LogicalFold, 	land, 1)

#define UNIFY_FOLD_BYTECODES(_) \
	_(min, "min",	UnifyFold, 	pmin, 3) \
	_(max, "max",	UnifyFold, 	pmax, 3)

#define SPECIAL_FOLD_BYTECODES(_) \
	_(length, "length", CountFold, 0) \
	_(mean, "mean", MomentFold, 2) \
	_(cm2, "cm2", Moment2Fold, 20)

#define ARITH_SCAN_BYTECODES(_) \
	_(cumsum, "cumsum",	ArithScan,	add, 1) \
	_(cumprod, "cumprod",	ArithScan,	mul, 3)

#define UNIFY_SCAN_BYTECODES(_) \
	_(cummin, "cummin",	UnifyScan,	pmin, 3) \
	_(cummax, "cummax",	UnifyScan,	pmax, 3)

#define SPECIAL_BYTECODES(_) 	\
	_(done, "done") 

#define UNARY_BYTECODES(_) \
	ARITH_UNARY_BYTECODES(_) \
	LOGICAL_UNARY_BYTECODES(_) \
	ORDINAL_UNARY_BYTECODES(_) \
	CAST_UNARY_BYTECODES(_) \

#define BINARY_BYTECODES(_) \
	ARITH_BINARY_BYTECODES(_) \
	LOGICAL_BINARY_BYTECODES(_) \
	UNIFY_BINARY_BYTECODES(_) \
	ORDINAL_BINARY_BYTECODES(_) \
	ROUND_BINARY_BYTECODES(_) \

#define MAP_BYTECODES(_) \
	UNARY_BYTECODES(_) \
	BINARY_BYTECODES(_) \
	TERNARY_BYTECODES(_) \

#define FOLD_BYTECODES(_) \
	ARITH_FOLD_BYTECODES(_) \
	LOGICAL_FOLD_BYTECODES(_) \
	UNIFY_FOLD_BYTECODES(_) \

#define SCAN_BYTECODES(_) \
	ARITH_SCAN_BYTECODES(_) \
	UNIFY_SCAN_BYTECODES(_) \

#define STANDARD_BYTECODES(_) \
	MEMORY_ACCESS_BYTECODES(_) \
	GENERATOR_BYTECODES(_) \
	MAP_BYTECODES(_) \
	FOLD_BYTECODES(_) \
	SCAN_BYTECODES(_) \
	UTILITY_BYTECODES(_) \
	SPECIAL_FOLD_BYTECODES(_) \

#define BYTECODES(_) \
	STANDARD_BYTECODES(_) \
	CONTROL_FLOW_BYTECODES(_) \
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

	Instruction(ByteCode::Enum bc, int64_t a=0, int64_t b=0, int64_t c=0) :
		a(a), b(b), c(c), bc(bc) {}
	
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
