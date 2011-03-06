
#ifndef _RIPOSTE_BC_H
#define _RIPOSTE_BC_H

#include "enum.h"
#include "common.h"

#define BC_ENUM(_) 				\
	_(call, "call") \
	_(inlinecall, "inlinecall") \
	_(get, "get") \
	_(kget, "kget") \
	_(assign, "assign") \
	_(pop, "pop") \
	_(forbegin, "forbegin") \
	_(forend, "forend") \
	_(whilebegin, "whilebegin") \
	_(whileend, "whileend") \
	_(add, "add") \
	_(pos, "pos") \
	_(sub, "sub") \
	_(neg, "neg") \
	_(mul, "mul") \
	_(div, "div") \
	_(jmp, "jmp") \
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


struct Instruction {
	ByteCode bc;
	void const* ibc;
	uint64_t a, b, c;

	Instruction(ByteCode bc, uint64_t a=0, uint64_t b=0, uint64_t c=0) :
		bc(bc), a(a), b(b), c(c) {}
	
	Instruction(void const* ibc, uint64_t a=0, uint64_t b=0, uint64_t c=0) :
		ibc(ibc), a(a), b(b), c(c) {}
	
	std::string toString() const {
		return std::string("") + bc.toString() + "\t" + intToStr(a);
	}
};


#endif
