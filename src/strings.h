
#ifndef _RIPOSTE_SYMBOLS_H
#define _RIPOSTE_SYMBOLS_H

#include "common.h"

// predefined strings

#define STRINGS(_) 			\
	/* must match types */ 		\
	_(Null, 	"NULL")		\
	_(Raw, 		"raw")		\
	_(Logical, 	"logical")	\
	_(Integer, 	"integer")	\
	_(Double, 	"double")	\
	_(Complex, 	"complex")	\
	_(Character, 	"character")	\
	_(List,		"list")		\
	_(Function,	"function")	\
	_(BuiltIn,	"builtin")	\
	_(Environment,	"environment")	\
	_(Object,	"object")	\
	_(Promise, 	"promise") 	\
	_(Default, 	"default") 	\
	_(Dotdot, 	"dotdot") 	\
	_(Nil,		"nil")		\
	_(Future,	"future")		\
	_(Date, "date")		\
	/* Now all other strings */	\
	_(NArep, 	"<NA>") \
	_(empty, 	"") \
	_(dots, 	"...") \
	_(Name, 	"name") \
	_(Numeric, 	"numeric") \
	_(Pairlist,	"pairlist")	\
	_(Symbol,	"symbol")	\
	_(Call,		"call")		\
	_(Expression,	"expression")	\
	_(internal, 	".Internal") \
	_(assign, 	"<-") \
	_(assign2, 	"<<-") \
	_(eqassign, 	"=") \
	_(function, 	"function") \
	_(returnSym, 	"return") \
	_(forSym, 	"for") \
	_(whileSym, 	"while") \
	_(repeatSym, 	"repeat") \
	_(nextSym, 	"next") \
	_(breakSym,	"break") \
	_(ifSym, 	"if") \
	_(switchSym, 	"switch") \
	_(brace, 	"{") \
	_(paren, 	"(") \
	_(add, 		"+") \
	_(sub, 		"-") \
	_(mul, 		"*") \
	_(tilde, 	"~") \
	_(div, 		"/") \
	_(idiv, 	"%/%") \
	_(pow, 		"^") \
	_(mod, 		"%%") \
	_(atan2, 	"atan2") \
	_(hypot, 	"hypot") \
	_(lnot, 	"!") \
	_(land, 	"&") \
	_(lor, 		"|") \
	_(land2, 	"&&") \
	_(lor2, 	"||") \
	_(eq, 		"==") \
	_(neq, 		"!=") \
	_(lt, 		"<") \
	_(le, 		"<=") \
	_(gt, 		">") \
	_(ge, 		">=") \
	_(dollar, 	"$") \
	_(at, 		"@") \
	_(colon, 	":") \
	_(nsget, 	"::") \
	_(nsgetint, 	":::") \
	_(bracket, 	"[") \
	_(bb, 		"[[") \
	_(question, 	"?") \
	_(abs, 		"abs") \
	_(sign, 	"sign") \
	_(sqrt, 	"sqrt") \
	_(floor, 	"floor") \
	_(ceiling, 	"ceiling") \
	_(trunc, 	"trunc") \
	_(round, 	"round") \
	_(signif, 	"signif") \
	_(exp, 		"exp") \
	_(log, 		"log") \
	_(cos, 		"cos") \
	_(sin, 		"sin") \
	_(tan, 		"tan") \
	_(acos, 	"acos") \
	_(asin, 	"asin") \
	_(atan, 	"atan") \
	_(sum,		"sum") \
	_(prod,		"prod") \
	_(mean,		"mean") \
	_(cm2,		"cm2") \
	_(min,		"min") \
	_(max,		"max") \
	_(pmin,		"pmin") \
	_(pmax,		"pmax") \
	_(any,		"any") \
	_(all,		"all") \
	_(cumsum,	"cumsum") \
	_(cumprod,	"cumprod") \
	_(cummin,	"cummin") \
	_(cummax,	"cummax") \
	_(cumany,	"cumany") \
	_(cumall,	"cumall") \
	_(names, 	"names") \
	_(dim, 		"dim") \
	_(classSym, 	"class") \
	_(bracketAssign, "[<-") \
	_(bbAssign, 	"[[<-") \
	_(dollarAssign, "$<-") \
	_(UseMethod, 	"UseMethod") \
	_(seq, 		"seq") \
	_(rep, 		"rep") \
	_(random, 	"random") \
	_(True, 	"TRUE") \
	_(False, 	"FALSE") \
	_(type, 	"type") \
	_(length, 	"length") \
	_(value, 	"value") \
	_(dotGeneric, 	".Generic") \
	_(dotMethod, 	".Method") \
	_(dotClass, 	".Class") \
	_(docall,	"do.call") \
	_(list,		"list") \
	_(missing,	"missing") \
	_(quote,	"quote") \
	_(mmul,		"%*%") \
	_(apply,	"apply") \
	_(ifelse,	"ifelse") \
	_(split,	"split") \
	_(strip,	"strip") \
	_(isna,		"is.na") \
	_(isnan,	"is.nan") \
	_(isfinite,	"is.finite") \
	_(isinfinite,	"is.infinite") \
	_(vector,	"vector") \
	_(Maximal, "\255")

typedef const char* String;

namespace Strings {
	static const String NA = 0;
#define DECLARE(name, string, ...) extern String name;
	STRINGS(DECLARE)
#undef DECLARE
	extern String pos;
	extern String neg;
}

#endif
