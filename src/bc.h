
#ifndef _RIPOSTE_BC_H
#define _RIPOSTE_BC_H

#include "enum.h"
#include "common.h"

#define BC_ENUM(_) 				\
	_(call, "call") \
	_(get, "get") \
	_(kget, "kget") \
	_(delay, "delay") \
	_(symdelay, "symdelay") \
	_(assign, "assign") \
	_(pop, "pop") \
	_(forbegin, "forbegin") \
	_(forend, "forend") \
	_(fguard, "fguard") \
	_(add, "add") \
	_(ret, "ret")       /* ret must be the last instruction */
	/*_(frame, "frame") \
	_(function, "function") \
	_(rawfunction, "rawfunction") \
	_(quote, "quote") \
	_(force, "force") \
	_(forceall, "forceall") \
	_(code, "code") \
	_(slot, "slot") \
	_(zip2, "zip2") \*/

DECLARE_ENUM(ByteCode, BC_ENUM)


#endif
