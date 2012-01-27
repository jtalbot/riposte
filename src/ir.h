#ifndef _RIPOSTE_IR_H
#define _RIPOSTE_IR_H

#include "enum.h"
#include "common.h"
#include "bc.h"
#include "type.h"
#include "value.h"

#define IR_ENUM(_) \
		MAP_BYTECODES(_) \
		_(cast, "cast", ___) \
		_(loadc,"loadc", ___) \
		_(loadv,"loadv", ___) \
		_(storev,"storev", ___) \
		_(storec,"storec", ___) \
		_(seq, "seq", ___) \
		_(gather, "gather", ___) \
		_(filter, "filter", ___) \
		ARITH_FOLD_BYTECODES(_) \
		ARITH_SCAN_BYTECODES(_) \
		_(nop, "nop", ___)

DECLARE_ENUM(IROpCode,IR_ENUM)

struct Value;

struct IRNode {


	enum Encoding {
		NOP,
		BINARY,
		SPECIAL,
		UNARY,
		FOLD,
		LOADC,
		LOADV,
		STORE,
	};

	Encoding enc;
	IROpCode::Enum op;
	Type::Enum type;
	int64_t length;
	bool used;

	bool isDouble() const { return type == Type::Double; }
	bool isInteger() const { return type == Type::Integer; }
	bool isLogical() const { return type == Type::Logical; }

	union {
		struct {
			int64_t a,b;
		} binary;
		struct {
			int64_t a,b;
		} special;
		struct {
			int64_t a,data;
		} unary;
		struct {
			int64_t a;
			union {
				int64_t i;
				double d;
			};
		} fold;
		struct {
			union {
				int64_t i;
				double d;
				uint8_t l;
			};
		} loadc;
		struct {
			Value src;
		} loadv;
		struct {
			int64_t a;
			Value dst;
		} store;
		
	};
};

#endif /* IR_H_ */
