
#ifndef _RIPOSTE_TYPES_H
#define _RIPOSTE_TYPES_H

#include "enum.h"

#define TYPES(_) 			   \
	_(Promise, 	"promise",	0) \
	/* Objects ... */		   \
	_(Future, 	"future",       1) \
	_(Function,	"function",	2) \
	_(Environment,	"environment",	3) \
	_(Null, 	"NULL",		4) \
	_(Raw, 		"raw",		5) \
	_(Logical, 	"logical",	6) \
	_(Integer, 	"integer",	7) \
	_(Double, 	"double",	8) \
	_(Character, 	"character",	9) \
	_(List,		"list",	       10) \

DECLARE_ENUM_WITH_VALUES(Type, TYPES)

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

// Note that the Meet can be expresed as the max of the types
//  assuming all are vector types
#define DEFAULT_TYPE_MEET(_) \
	_(Null, Null, Null) \
	_(Logical, Null, Logical) \
	_(Null, Logical, Logical) \
	_(Logical, Logical, Logical) \
	_(Integer, Null, Integer) \
	_(Null, Integer, Integer) \
	_(Integer, Logical, Integer) \
	_(Logical, Integer, Integer) \
	_(Integer, Integer, Integer) \
	_(Double, Null, Double) \
	_(Null, Double, Double) \
	_(Double,  Logical, Double) \
	_(Logical, Double,  Double) \
	_(Double,  Integer, Double) \
	_(Integer, Double,  Double) \
	_(Double,  Double,  Double) \
	_(Character, Null, Character) \
	_(Null, Character, Character) \
	_(Character, Logical, Character) \
	_(Logical, Character, Character) \
	_(Character, Integer, Character) \
	_(Integer, Character, Character) \
	_(Character, Double, Character) \
	_(Double, Character, Character) \
	_(Character, Character, Character) \
	_(List, Null, List) \
	_(Null, List, List) \
	_(List, Logical, List) \
	_(Logical, List, List) \
	_(List, Integer, List) \
	_(Integer, List, List) \
	_(List, Double, List) \
	_(Double, List, List) \
	_(List, Character, List) \
	_(Character, List, List) \
	_(List, List, List) \

#endif
