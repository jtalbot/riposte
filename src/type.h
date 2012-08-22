
#ifndef _RIPOSTE_TYPES_H
#define _RIPOSTE_TYPES_H

#include "enum.h"

#define TYPES(_) 			\
	/* First the internal types. */ \
	_(Promise, 	"promise") 	\
	_(Default,	"default")	\
	_(Dotdot,	"dotdot")	\
	_(Future, 	"future")	\
	_(Object,	"object")	\
	/* The R visible types */	\
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

#define ORDINAL_TYPE_MEET(_) \
	_(Logical, Logical, Integer) \
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


#define UNARY_TYPES(_) \
    _(Double) \
    _(Integer) \
    _(Logical) 

#define BINARY_TYPES(_) \
    _(Double, Double) \
    _(Integer, Integer) \
    _(Logical, Logical) \
    _(Double, Integer) \
    _(Integer, Double) \
    _(Double, Logical) \
    _(Logical, Double) \
    _(Integer, Logical) \
    _(Logical, Integer)

#define TERNARY_TYPES(_) \
    _(Double, Double, Double) \
    _(Integer, Integer, Double) \
    _(Logical, Logical, Double) \
    _(Double, Integer, Double) \
    _(Integer, Double, Double) \
    _(Double, Logical, Double) \
    _(Logical, Double, Double) \
    _(Integer, Logical, Double) \
    _(Logical, Integer, Double) \
    _(Double, Double, Integer) \
    _(Integer, Integer, Integer) \
    _(Logical, Logical, Integer) \
    _(Double, Integer, Integer) \
    _(Integer, Double, Integer) \
    _(Double, Logical, Integer) \
    _(Logical, Double, Integer) \
    _(Integer, Logical, Integer) \
    _(Logical, Integer, Integer) \
    _(Double, Double, Logical) \
    _(Integer, Integer, Logical) \
    _(Logical, Logical, Logical) \
    _(Double, Integer, Logical) \
    _(Integer, Double, Logical) \
    _(Double, Logical, Logical) \
    _(Logical, Double, Logical) \
    _(Integer, Logical, Logical) \
    _(Logical, Integer, Logical)


#endif
