
#ifndef _RIPOSTE_BC_H
#define _RIPOSTE_BC_H

#include "enum.h"
#include "common.h"
#include "string.h"

#define CONTROL_FLOW_BYTECODES(_) 	\
	_(jc, "jc") \
	_(jmp, "jmp") \
	_(branch, "branch") \
	_(call, "call") \
	_(apply, "apply") \
	_(internal, "internal") \
	_(UseMethod, "UseMethod") \
	_(ret, "ret") \
	_(forbegin, "forbegin") \
	_(forend, "forend") \

#define MEMORY_ACCESS_BYTECODES(_) \
	_(get, "get") \
	_(kget, "kget") \
	_(assign, "assign") \
	_(assign2, "assign2") \
	_(iassign, "iassign") \
	_(eassign, "eassign") \
	_(subset, "subset") \
	_(subset2, "subset2") \

// ArithUnary1 ops perform Integer->Integer, ArithUnary2 ops perform Integer->Double
#define ARITH_UNARY_BYTECODES(_) \
	_(pos, "pos", 	add,	ArithUnary1, 	PassNA(a, a)) \
	_(neg, "neg", 	sub,	ArithUnary1, 	PassNA(a, -a)) \
	_(abs, "abs", 	abs,	ArithUnary1, 	PassNA(a, Abs(a))) \
	_(sign, "sign",	sign,	ArithUnary2, 	((a>0)-(a<0))) \
	_(sqrt, "sqrt",	sqrt,	ArithUnary2,	sqrt(a)) \
	_(floor, "floor",	floor,	ArithUnary2,	floor(a)) \
	_(ceiling, "ceiling",	ceiling,	ArithUnary2,	ceil(a)) \
	_(trunc, "trunc",	trunc,	ArithUnary2,	trunc(a)) \
	_(exp, "exp",	exp,	ArithUnary2,	exp(a)) \
	_(log, "log",	log,	ArithUnary2,	log(a)) \
	_(cos, "cos",	cos,	ArithUnary2,	cos(a)) \
	_(sin, "sin",	sin,	ArithUnary2,	sin(a)) \
	_(tan, "tan",	tan,	ArithUnary2,	tan(a)) \
	_(acos, "acos",	acos,	ArithUnary2,	acos(a)) \
	_(asin, "asin",	asin,	ArithUnary2,	asin(a)) \
	_(atan, "atan",	atan,	ArithUnary2,	atan(a)) \

#define LOGICAL_UNARY_BYTECODES(_) \
	_(lnot, "lnot",	lnot,	LogicalUnary, PassNA(a, ~a)) \

#define ORDINAL_UNARY_BYTECODES(_) \
	_(isna,	"isna",	isna,	OrdinalUnary,	M::isNA(a)?-1:0) \
	_(isnan,	"isnan",	isnan,	OrdinalUnary,	M::isNaN(a)?-1:0) \
	_(isfinite,	"isfinite",	isfinite,	OrdinalUnary,	M::isFinite(a)?-1:0) \
	_(isinfinite,	"isinfinite",	isinfinite,	OrdinalUnary,	M::isInfinite(a)?-1:0) \
/*
#define STRING_UNARY_BYTECODES(_) \
	_(nchar, "nchar", nchar, StringUnary, PassNA(a, strlen(a))) \
	_(nzchar, "nzchar", nzchar, StringUnary, PassNA(a, (*a>0))) \
*/
// ArithBinary1 ops perform Integer*Integer->Integer, ArithBinary2 ops perform Integer*Integer->Double
#define ARITH_BINARY_BYTECODES(_) \
	_(add, "add",	add,	ArithBinary1,	PassNA(a,b,a+b)) \
	_(sub, "sub",	sub,	ArithBinary1,	PassNA(a,b,a-b)) \
	_(mul, "mul",	mul,	ArithBinary1,	PassNA(a,b,a*b)) \
	_(div, "div",	div,	ArithBinary2,	a/b) \
	_(idiv, "idiv",	idiv,	ArithBinary1,	PassNA(a,b,IDiv(a,b))) \
	_(mod, "mod",	mod,	ArithBinary1,	PassNA(a,b,Mod(a,b))) \
	_(pow, "pow",	pow,	ArithBinary2,	pow(a,b)) \
	_(atan2, "atan2",	atan2,	ArithBinary2,	atan2(a,b)) \
	_(hypot, "hypot",	hypot,	ArithBinary2,	hypot(a,b)) \

#define LOGICAL_BINARY_BYTECODES(_) \
	_(lor, "lor",	lor,	LogicalBinary,	a|b) \
	_(land, "land",	land,	LogicalBinary,	a&b) \

#define UNIFY_BINARY_BYTECODES(_) \
	_(pmin, "pmin",	pmin,	UnifyBinary,	PassNA(a,b,riposte_min(thread,a,b))) \
	_(pmax, "pmax",	pmax,	UnifyBinary,	PassNA(a,b,riposte_max(thread,a,b))) \

/*
	_(round, "round",	round,	ArithBinary2,	round(a)) \
	_(signif, "signif",	signif,	ArithBinary2,	signif(a)) \
*/

#define ORDINAL_BINARY_BYTECODES(_) \
	_(eq, "eq",	eq,	OrdinalBinary,	PassNA(a,b,a==b?-1:0)) \
	_(neq, "neq",	neq,	OrdinalBinary,	PassNA(a,b,a!=b?-1:0)) \
	_(gt, "gt",	gt,	OrdinalBinary,	PassNA(a,b,gt(thread,a,b)?-1:0)) \
	_(ge, "ge",	ge,	OrdinalBinary,	PassNA(a,b,ge(thread,a,b)?-1:0)) \
	_(lt, "lt",	lt,	OrdinalBinary,	PassNA(a,b,lt(thread,a,b)?-1:0)) \
	_(le, "le",	le,	OrdinalBinary,	PassNA(a,b,le(thread,a,b)?-1:0)) \

#define SPECIAL_MAP_BYTECODES(_) \
	_(ifelse, "ifelse", IfElseOp, "ifelse") \

#define ARITH_FOLD_BYTECODES(_) \
	_(sum, "sum",	sum,	ArithFold, 	add) \
	_(prod, "prod",	prod,	ArithFold, 	mul) \

#define LOGICAL_FOLD_BYTECODES(_) \
	_(any, "any",	any,	LogicalFold, 	lor) \
	_(all, "all",	all,	LogicalFold, 	land) \

#define UNIFY_FOLD_BYTECODES(_) \
	_(min, "min",	min,	UnifyFold, 	pmin) \
	_(max, "max",	max,	UnifyFold, 	pmax) \

#define ARITH_SCAN_BYTECODES(_) \
	_(cumsum, "cumsum",	cumsum,	ArithScan,	add) \
	_(cumprod, "cumprod",	cumprod,	ArithScan,	mul) \

#define UNIFY_SCAN_BYTECODES(_) \
	_(cummin, "cummin",	cummin,	UnifyScan,	pmin) \
	_(cummax, "cummax",	cummax,	UnifyScan,	pmax) \

#define UTILITY_BYTECODES(_)\
	_(split, "split") \
	_(colon, "colon") \
	_(function, "function") \
	_(logical1, "logical") \
	_(integer1, "integer") \
	_(double1, "double") \
	_(character1, "character") \
	_(raw1, "raw") \
	_(seq, "seq") \
	_(type, "type") \
	_(list, "list") \
	_(length, "length") \
	_(missing, "missing") \
	_(mmul, "mmul") \
	_(strip, "strip") \

#define SPECIAL_BYTECODES(_) 	\
	_(done, "done") 

#define MAP_BYTECODES(_) \
	ARITH_UNARY_BYTECODES(_) \
	LOGICAL_UNARY_BYTECODES(_) \
	ORDINAL_UNARY_BYTECODES(_) \
	ARITH_BINARY_BYTECODES(_) \
	LOGICAL_BINARY_BYTECODES(_) \
	UNIFY_BINARY_BYTECODES(_) \
	ORDINAL_BINARY_BYTECODES(_) \

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
	MAP_BYTECODES(_) \
	SPECIAL_MAP_BYTECODES(_) \
	FOLD_BYTECODES(_) \
	SCAN_BYTECODES(_) \
	UTILITY_BYTECODES(_) \

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
	ARITH_UNARY_BYTECODES(_) \
	LOGICAL_UNARY_BYTECODES(_) \
	ORDINAL_UNARY_BYTECODES(_) \
	ARITH_FOLD_BYTECODES(_) \
	LOGICAL_FOLD_BYTECODES(_) \
	UNIFY_FOLD_BYTECODES(_) \
	ARITH_SCAN_BYTECODES(_) \
	UNIFY_SCAN_BYTECODES(_) \

#define BINARY_BYTECODES(_) \
	ARITH_BINARY_BYTECODES(_) \
	LOGICAL_BINARY_BYTECODES(_) \
	UNIFY_BINARY_BYTECODES(_) \
	ORDINAL_BINARY_BYTECODES(_) \

DECLARE_ENUM(ByteCode, BYTECODES)

struct Instruction {
	union {
		int64_t a;
		String s;
	};
	int64_t b, c;
	ByteCode::Enum bc;
	mutable void const* ibc;

	Instruction(ByteCode::Enum bc, int64_t a=0, int64_t b=0, int64_t c=0) :
		a(a), b(b), c(c), bc(bc), ibc(0) {}
	
	Instruction(ByteCode::Enum bc, String s, int64_t b=0, int64_t c=0) :
		s(s), b(b), c(c), bc(bc), ibc(0) {}
	
	std::string toString() const {
		return std::string("") + ByteCode::toString(bc) + "\t" + intToStr(a) + "\t" + intToStr(b) + "\t" + intToStr(c);
	}
};

#endif
