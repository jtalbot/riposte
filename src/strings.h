
#ifndef _RIPOSTE_SYMBOLS_H
#define _RIPOSTE_SYMBOLS_H

#include "common.h"

// predefined strings

#define STRINGS(_) 			        \
	/* Type names */ 		        \
	_(Nil,		    "nil")		    \
	_(Promise, 	    "promise") 	    \
	_(Future,	    "future")		\
	_(Closure,	    "closure")	    \
	_(Environment,	"environment")	\
	_(Null, 	    "NULL")		    \
	_(Raw, 		    "raw")		    \
	_(Logical, 	    "logical")	    \
	_(Integer, 	    "integer")	    \
	_(Double, 	    "double")	    \
	_(Character, 	"character")	\
	_(List,		    "list")		    \
	/* Constant strings */	        \
	_(empty, 	    "")             \
	_(NArep, 	    "<NA>")         \
	_(dots, 	    "...")          \
    _(assignTmp,    "*tmp*")        \
	_(Maximal,      "\255")         \
    /* Primitive functions */ \
	_(Complex, 	    "complex")	\
	_(Function,	    "function")	\
	_(BuiltIn,	    "builtin")	\
	_(Object,	    "object")	\
	_(Default, 	    "default") 	\
	_(Dotdot, 	    "dotdot") 	\
	_(Date,         "date")		\
	_(Name, 	"name") \
	_(Numeric, 	"numeric") \
	_(Pairlist,	"pairlist")	\
	_(Symbol,	"symbol")	\
	_(Call,		"call")		\
	_(Expression,	"expression")	\
	_(internal, 	".Internal") \
	_(external, 	".External") \
	_(map_d, 	".Map.double") \
	_(map_i, 	".Map.integer") \
	_(map_l, 	".Map.logical") \
	_(map_c, 	".Map.character") \
	_(map_r, 	".Map.raw") \
	_(map_g, 	".Map.list") \
	_(fold_d, 	".Fold.double") \
	_(fold_i, 	".Fold.integer") \
	_(fold_l, 	".Fold.logical") \
	_(fold_c, 	".Fold.character") \
	_(fold_r, 	".Fold.row") \
	_(fold_g, 	".Fold.list") \
	_(scan_d, 	".Scan.double") \
	_(scan_i, 	".Scan.integer") \
	_(scan_l, 	".Scan.logical") \
	_(scan_c, 	".Scan.character") \
	_(scan_r, 	".Scan.row") \
	_(scan_g, 	".Scan.list") \
	_(assign, 	"<-") \
	_(assign2, 	"<<-") \
	_(eqassign, 	"=") \
    _(rm,           "rm") \
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
	_(round, 	"round") \
	_(signif, 	"signif") \
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
	_(index, 	"index") \
	_(random, 	"random") \
	_(True, 	"TRUE") \
	_(False, 	"FALSE") \
	_(type, 	"typeof") \
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
	_(vector,	"vector") \
	_(attrget,	"attr") \
	_(attrset,	"attr<-") \
	_(Re,	"Re") \
	_(Im,	"Im") \
    _(as, "as") \
    _(getNamespace, "getNamespace") \
    _(promise, "promise") \
    _(getenv, ".getenv") \
    _(setenv, ".setenv") \
    _(ls, "ls") \
    _(body, "body") \
    _(formals, "formals") \
    _(attributes, "attributes") \
    _(pr_expr, ".pr_expr") \
    _(pr_env, ".pr_env") \
    _(env_new, ".env_new") \
    _(env_exists, ".env_exists") \
    _(env_remove, ".env_remove") \
    _(fm_fn, ".fm_fn") \
    _(fm_call, ".fm_call") \
    _(fm_env, ".fm_env") \

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
