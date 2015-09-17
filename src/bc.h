
#ifndef _RIPOSTE_BC_H
#define _RIPOSTE_BC_H

#include "enum.h"
#include "common.h"

#define CONTROL_FLOW_BYTECODES(_)                                              \
    _(jc,           "jc")                                                      \
    _(jmp,          "jmp")                                                     \
    _(call,         "call")                                                    \
    _(ret,          "ret")          /* return from function */                 \
    _(forbegin,     "forbegin")                                                \
    _(forend,       "forend")                                                  \
    _(mov,          "mov")                                                     \
    _(external,     "external")                                                \
    _(invisible,    "invisible")                                               \
    _(visible,      "visible")                                                 \
    _(withVisible,  "withVisible")                                                 \
    _(force,        "force")                                                   \

#define LOAD_STORE_BYTECODES(_)                                                \
    _(load,         "load" )        /* = get(x, environment(), "any", TRUE) */ \
    _(loadfn,       "loadfn")       /* = get(x, environment(), "function", TRUE) */ \
    _(store,        "store")        /* = <-,  assign(x, environment(), FALSE) */\
    _(storeup,      "storeup")      /* = <<-, assign(x, parent.env(environment()), TRUE) */ \
    _(dotsv,        "dotsv")        /* = ..# */                                \
    _(dotsc,        "dotsc")        /* = length(...) */                        \
    _(dots,         "dots")         /* = list(...) */                          \

#define STACK_FRAME_BYTECODES(_)                                               \
    _(frame,        "frame")        /* get a stack frame by index */           \

#define PROMISE_BYTECODES(_)                                                   \
    _(pr_new,       "pr_new")       /* create a new promise (delayedAssign) */ \
    _(pr_expr,      "pr_expr")      /* the expression of a promise */          \
    _(pr_env,       "pr_env")       /* the environment of a promise */         \

#define OBJECT_BYTECODES(_)                                                    \
    _(isnil,        "isnil")                                                   \
    _(id,           "id")                                                      \
    _(nid,          "nid")                                                     \
    _(type,         "type")                                                    \
    _(length,       "length")                                                  \
    _(get,          "get")          /* $, [[   (works on all objects) */       \
    _(set,          "set")          /* $<-, [[<-  (works on all objects) */    \
    _(getsub,       "getsub")       /* [] */                                   \
    _(setsub,       "setsub")       /* [<- */                                  \
    _(getenv,       "getenv")       /* get the object's containing environment */ \
    _(setenv,       "setenv")       /* set the object's containing environment */ \
    _(getattr,      "getattr")                                                 \
    _(setattr,      "setattr")                                                 \
    _(attributes,   "attributes")   /* returns list with attributes & names */ \
    _(strip,        "strip")        /* return base object without any attributes */ \
    _(as,           "as")           /* coerce between object types */

#define ENVIRONMENT_BYTECODES(_)                                               \
    _(env_new,      "env_new")                                                 \
    _(env_has,      "env_has")      /* check a value in an environment */      \
    _(env_rm,       "env_rm" )      /* undefine values in an environment */    \
    _(env_missing,  "env_missing" ) /* see if an argument is missing */        \
    _(env_names,    "env_names")    /* get symbols defined in environment */   \
    _(env_global,   "env_global")   /* get the global environment */           \

#define FUNCTION_BYTECODES(_)                                                  \
    _(fn_new,       "fn_new")       /* closure constructor */                  \

#define GENERATOR_BYTECODES(_)                                                 \
	_(vector,	"vector",	Generator)                                         \
	_(seq,		"seq", 		Generator)                                         \
	_(index,	"index", 	Generator)                                         \
	_(random,	"random", 	Generator)                                         \

// ArithUnary1 ops perform Integer->Integer, ArithUnary2 ops perform Integer->Double
#define ARITH_UNARY_BYTECODES(_)                                               \
	_(pos, "pos", 	ArithUnary1, 	PassNA(a, a))                              \
	_(neg, "neg", 	ArithUnary1, 	PassNA(a, -a))                             \

#define LOGICAL_UNARY_BYTECODES(_)                                             \
	_(lnot, "lnot",	LogicalUnary, PassNA(a, ~a))                               \

#define CHARACTER_UNARY_BYTECODES(_)                                           \
    _(nchar, "nchar", CharacterUnary, PassNA(a, Length(a)))                    \

#define ORDINAL_UNARY_BYTECODES(_)                                             \
	_(isna,	"isna",	OrdinalUnary,	MA::isNA(a)?Logical::TrueElement:Logical::FalseElement) \

// ArithBinary1 ops perform Integer*Integer->Integer, ArithBinary2 ops perform Integer*Integer->Double
#define ARITH_BINARY_BYTECODES(_)                                              \
	_(add, "add",	ArithBinary1, PassNA(a,b,a+b))                             \
	_(sub, "sub",	ArithBinary1, PassNA(a,b,a-b))                             \
	_(mul, "mul",	ArithBinary1, PassNA(a,b,a*b))                             \
	_(div, "div",	ArithBinary2, PassNA(a,b,a/b))                             \
	_(idiv, "idiv",	ArithBinary1, PassNA(a,b,IDiv(a,b)))                       \
	_(mod, "mod",	ArithBinary1, PassNA(a,b,Mod(a,b)))                        \
	_(pow, "pow",	ArithBinary2, PassNA(a,b,pow(a,b)))                        \

#define LOGICAL_BINARY_BYTECODES(_)                                            \
	_(lor, "lor",	LogicalBinary, ((a<0||b<0)?-1:(a>0)?1:b))                  \
	_(land, "land",	LogicalBinary, ((!a||!b)  ?0:(a>0)?1:b))                   \

#define CHARACTER_BINARY_BYTECODES(_)                                          \
    _(pconcat, "pconcat", CharacterBinary, PassCheckedNA(a,b,Concat(a,b)))     \

#define UNIFY_BINARY_BYTECODES(_)                                              \
	_(pmin, "pmin",	UnifyBinary, PassCheckedNA(a,b,riposte_min(a,b)))   \
	_(pmax, "pmax",	UnifyBinary, PassCheckedNA(a,b,riposte_max(a,b)))   \

#define ORDINAL_BINARY_BYTECODES(_)                                            \
	_(eq, "eq", OrdinalBinary, PassCheckedNA(a,b,eq(a,b)?Logical::TrueElement:Logical::FalseElement)) \
	_(neq, "neq", OrdinalBinary, PassCheckedNA(a,b,neq(a,b)?Logical::TrueElement:Logical::FalseElement)) \
	_(gt, "gt",	OrdinalBinary, PassCheckedNA(a,b,gt(a,b)?Logical::TrueElement:Logical::FalseElement)) \
	_(ge, "ge",	OrdinalBinary, PassCheckedNA(a,b,ge(a,b)?Logical::TrueElement:Logical::FalseElement)) \
	_(lt, "lt",	OrdinalBinary, PassCheckedNA(a,b,lt(a,b)?Logical::TrueElement:Logical::FalseElement)) \
	_(le, "le",	OrdinalBinary, PassCheckedNA(a,b,le(a,b)?Logical::TrueElement:Logical::FalseElement)) \

#define SPECIAL_MAP_BYTECODES(_)                                               \
	_(ifelse, "ifelse", IfElse)                                                \
	_(split, "split", Split)

// ArithFold1 ops perform [Integer]->Integer, ArithFold2 ops perform [Integer]->Double
#define ARITH_FOLD_BYTECODES(_)                                                \
	_(sum, "sum",	ArithFold1, 	add)                                       \
	_(prod, "prod",	ArithFold1, 	mul) 

#define LOGICAL_FOLD_BYTECODES(_)                                              \
	_(any, "any",	LogicalFold, 	lor)                                       \
	_(all, "all",	LogicalFold, 	land) 

#define UNIFY_FOLD_BYTECODES(_)                                                \
	_(min, "min",	UnifyFold, 	pmin)                                          \
	_(max, "max",	UnifyFold, 	pmax) 

#define ARITH_SCAN_BYTECODES(_)                                                \
	_(cumsum, "cumsum",	ArithScan,	add)                                       \
	_(cumprod, "cumprod",	ArithScan,	mul) 

#define UNIFY_SCAN_BYTECODES(_)                                                \
	_(cummin, "cummin",	UnifyScan,	pmin)                                      \
	_(cummax, "cummax",	UnifyScan,	pmax) 

#define GENERIC_BYTECODES(_)                                                   \
    _(map, "map")                                                              \
    _(scan, "scan")                                                            \
    _(fold, "fold")

#define JOIN_BYTECODES(_)                                                      \
    _(semijoin, "semijoin")

#define SPECIAL_BYTECODES(_) 	                                               \
    _(stop,         "stop")     /* stop execution with an error */             \
	_(done,         "done")     /* finish execution of block and store result */

#define UNARY_BYTECODES(_) \
	ARITH_UNARY_BYTECODES(_) \
	LOGICAL_UNARY_BYTECODES(_) \
	CHARACTER_UNARY_BYTECODES(_) \
	ORDINAL_UNARY_BYTECODES(_) \

#define BINARY_BYTECODES(_) \
	ARITH_BINARY_BYTECODES(_) \
	LOGICAL_BINARY_BYTECODES(_) \
	CHARACTER_BINARY_BYTECODES(_) \
	UNIFY_BINARY_BYTECODES(_) \
	ORDINAL_BINARY_BYTECODES(_) \

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
	LOAD_STORE_BYTECODES(_) \
    STACK_FRAME_BYTECODES(_) \
    PROMISE_BYTECODES(_) \
    OBJECT_BYTECODES(_) \
    ENVIRONMENT_BYTECODES(_) \
    FUNCTION_BYTECODES(_) \
	GENERATOR_BYTECODES(_) \
	MAP_BYTECODES(_) \
	SPECIAL_MAP_BYTECODES(_) \
	FOLD_BYTECODES(_) \
	SCAN_BYTECODES(_) \
	GENERIC_BYTECODES(_) \
	JOIN_BYTECODES(_) \

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

#endif
