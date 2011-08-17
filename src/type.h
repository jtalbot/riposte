
#ifndef _RIPOSTE_TYPES_H
#define _RIPOSTE_TYPES_H

#include "enum.h"

#define TYPES(_) 			\
	/* First the R visible types */	\
	/*   In the "standard casting order" */  \
	_(Null, 	"NULL")		\
	_(Raw, 		"raw")		\
	_(Logical, 	"logical")	\
	_(Integer, 	"integer")	\
	_(Double, 	"double")	\
	_(Complex, 	"complex")	\
	_(Character, 	"character")	\
	_(List,		"list")		\
	_(PairList,	"pairlist")	\
	_(Call,		"call")		\
	_(Expression,	"expression")	\
	_(Function,	"function")	\
	_(BuiltIn,	"builtin")	\
	_(Symbol,	"symbol")	\
	_(Environment,	"environment")	\
					\
	/* Now the internal types, not necessarily matching R. Order is important here. */ \
	_(Promise, 	"promise") 	\
	_(Nil,		"nil")		\

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
