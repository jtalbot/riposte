
#ifndef _RIPOSTE_TYPES_H
#define _RIPOSTE_TYPES_H

#include "enum.h"

#define TYPE_ENUM(_, p) 				\
	/* First the R visible types */	\
	/*   In the "standard casting order */  \
	_(R_null, 	"NULL", p)			\
	_(R_raw, 	"raw", p)			\
	_(R_logical, 	"logical", p)		\
	_(R_integer, 	"integer", p)		\
	_(R_double, 	"double", p)		\
	_(R_complex, 	"complex", p)		\
	_(R_character, 	"character", p)		\
	_(R_list,	"list", p)			\
	_(R_expression,	"expression", p)		\
	_(R_symbol,	"symbol", p)		\
	_(R_pairlist,	"pairlist", p)		\
	_(R_function,	"function", p)		\
	_(R_environment,"environment", p)		\
	/* Note: no user-visible promise type in riposte */ \
	_(R_call,	"call", p)		\
	/* Note: specials are treated the same as normal functions right now */ \
	_(R_cfunction,	"builtin", p)		\
	_(R_any,	"any", p)	/* If there are no objects of this type, how can typeof return it? */	\
	_(R_externalptr,"externalptr", p)		\
	_(R_weakref,	"weakref", p)		\
	_(R_S4, 	"S4", p)			\
									\
	/* Now the internal types, not necessarily matching R */ \
	_(I_compiledcall,"compiled call", p)			\
	_(I_dots,			"...", p)			\
	_(I_closure, 	"closure", p) \
	_(I_nil, 		"NIL", p)			\

DECLARE_ENUM(Type, TYPE_ENUM)

#endif
