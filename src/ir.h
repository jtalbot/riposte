#ifndef _RIPOSTE_IR_H
#define _RIPOSTE_IR_H

#include "enum.h"
#include "common.h"
#include "bc.h"
#include "type.h"

#define IR_ENUM(_) \
		BINARY_ARITH_MAP_BYTECODES(_) \
		_(seq,"seq", ___) \
		UNARY_ARITH_MAP_BYTECODES(_) \
		_(cast, "cast", ___) \
		_(loadc,"loadc", ___) \
		_(loadv,"loadv", ___) \
		_(storev,"storev", ___) \
		_(storec,"storec", ___) \
		/*ARITH_FOLD_BYTECODES(_) \
		ARITH_SCAN_BYTECODES(_) */\

DECLARE_ENUM(IROpCode,IR_ENUM)

struct Value;

typedef size_t IRef;
struct IRNode {
	IROpCode::Enum op;
	Type::Enum type;

	union {
		struct {
			int64_t a,b;
		} binary;
		struct {
			int64_t a;
		} unary;
		struct {
			union {
				int64_t i;
				double d;
			};
		} loadc;
		struct {
			void * p;
		} loadv;
		struct {
			Value * dst;
			int64_t a;
		} store;
	};
};

#endif /* IR_H_ */
