
#ifndef _RIPOSTE_TYPES_H
#define _RIPOSTE_TYPES_H

#include "enum.h"

#define TYPES(_) 				\
	/* First the R visible types */	\
	/*   In the "standard casting order */  \
	_(Null, 	"NULL")		\
	_(Raw, 		"raw")		\
	_(Logical, 	"logical")	\
	_(Integer, 	"integer")	\
	_(Double, 	"double")	\
	_(Complex, 	"complex")	\
	_(Character, 	"character")	\
	_(List,		"list")		\
	_(Expression,	"expression")	\
	_(Symbol,	"symbol")	\
	_(PairList,	"pairlist")	\
	_(Function,	"function")	\
	_(Environment,	"environment")	\
	/* Note: no user-visible promise type in riposte */ \
	_(Call,		"call")		\
	/* Note: specials are treated the same as normal functions right now */ \
	_(BuiltIn,	"builtin")	\
	_(Any,	"any")	/* If there are no objects of this type, how can typeof return it? */	\
	_(Externalptr,	"externalptr")	\
	_(Weakref,	"weakref")	\
	_(S4, 		"S4")		\
									\
	/* Now the internal types, not necessarily matching R. Order is important here. */ \
	_(CompiledCall,	"compiled call")\
	_(Closure, 	"closure") 	\
	_(Nil,		"NIL")		\

DECLARE_ENUM(Type, TYPES)

// just the vector types
#define VECTOR_TYPES(_)	\
	_(Null) 	\
	_(Raw) 		\
	_(Logical) 	\
	_(Integer) 	\
	_(Double) 	\
	_(Complex) 	\
	_(Character) 	\
	_(List) 	\
	_(PairList) 	\
	_(Call) 	\
	_(Expression) 	\

#define VECTOR_TYPES_NOT_NULL(_)	\
	_(Raw) 		\
	_(Logical) 	\
	_(Integer) 	\
	_(Double) 	\
	_(Complex) 	\
	_(Character) 	\
	_(List) 	\
	_(PairList) 	\
	_(Call) 	\
	_(Expression) 	\

#endif
