
#ifndef _RIPOSTE_TYPES_H
#define _RIPOSTE_TYPES_H

#include "enum.h"

#define TYPES(_) 			\
	/* First the internal types. Order is important here. The 'Nil' value is an invalid promise object with the header = 0*/ \
	_(Promise, 	"promise") 	\
	_(Default,	"default")	\
	_(Dotdot,	"dotdot")	\
	_(Future, 	"future")	\
	_(Object,	"object")	\
	/* The R visible types */	\
	/*   In the "standard casting order" */  \
	_(Null, 	"NULL")		\
	_(Raw, 		"raw")		\
	_(Logical, 	"logical")	\
	_(Integer, 	"integer")	\
	_(Double, 	"double")	\
	_(Character, 	"character")	\
	_(List,		"list")		\
	_(Function,	"function")	\
	_(Environment,	"environment")	\

DECLARE_ENUM(Type, TYPES)

// just the vector types
#define VECTOR_TYPES(_)	\
	_(Null) 	\
	_(Raw) 		\
	_(Logical) 	\
	_(Integer) 	\
	_(Double) 	\
	_(Character) 	\
	_(List) 	\

#define VECTOR_TYPES_NOT_NULL(_)	\
	_(Raw) 		\
	_(Logical) 	\
	_(Integer) 	\
	_(Double) 	\
	_(Character) 	\
	_(List) 	\

#define ATOMIC_VECTOR_TYPES(_) \
	_(Null)		\
	_(Raw)		\
	_(Logical)	\
	_(Integer)	\
	_(Double)	\
	_(Character)	\

#define LISTLIKE_VECTOR_TYPES(_) \
	_(List)		\

#define DEFAULT_TYPE_MEET(_) \
	_(Logical, Logical, Logical) \
	_(Integer, Logical, Integer) \
	_(Logical, Integer, Integer) \
	_(Integer, Integer, Integer) \
	_(Double,  Logical, Double) \
	_(Logical, Double,  Double) \
	_(Double,  Integer, Double) \
	_(Integer, Double,  Double) \
	_(Double,  Double,  Double) \
	_(Character, Logical, Character) \
	_(Logical, Character, Character) \
	_(Character, Integer, Character) \
	_(Integer, Character, Character) \
	_(Character, Double, Character) \
	_(Double, Character, Character) \
	_(Character, Character, Character) \

#endif
