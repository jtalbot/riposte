
#ifndef _RIPOSTE_TYPES_H
#define _RIPOSTE_TYPES_H

#include "enum.h"

#define TYPE_ENUM(_) 				\
	/* First the R visible types */	\
	_(R_null, 		"NULL")			\
	_(R_symbol,		"symbol")		\
	_(R_pairlist,	"pairlist")		\
	_(R_function,	"closure")		\
	_(R_environment,"environment")	\
	/* Note: no user-visible promise type in riposte */ \
	_(R_call,		"language")		\
	/* Note: specials are treated the same as normal functions right now */ \
	_(R_cfunction,	"builtin")		\
	_(R_logical, 	"logical")		\
	_(R_integer, 	"integer")		\
	_(R_double, 	"double")		\
	_(R_complex, 	"complex")		\
	_(R_character, 	"character")	\
	_(R_any,		"any")	/* If there are no objects of this type, how can typeof return it? */	\
	_(R_expression,	"expression")	\
	_(R_list,		"list")			\
	_(R_externalptr,"externalptr")	\
	_(R_weakref,	"weakref")		\
	_(R_raw, 		"raw")			\
	_(R_S4, 		"S4")			\
									\
	/* Now the internal types, not necessarily matching R */ \
	_(I_dots,			"...")			\
	_(I_internalcall,"internalcall")			\
	_(I_promise,	"promise")		\
	_(I_sympromise,	"sympromise")		\
	_(I_default,	"default")		\
	_(I_bytecode, 	"bc") \
	_(I_nil, 		"NIL")			\

DECLARE_ENUM(Type, TYPE_ENUM)

#endif
