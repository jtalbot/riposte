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
		_(constant,"constant", ___) \
		_(load,"load", ___) \
		_(seq, "seq", ___) \
		_(gather, "gather", ___) \
		_(filter, "filter", ___) \
		_(split, "split", ___) \
		ARITH_FOLD_BYTECODES(_) \
		ARITH_SCAN_BYTECODES(_) \
		_(nop, "nop", ___) \
		_(ifelse, "ifelse", ___) \

DECLARE_ENUM(IROpCode,IR_ENUM)

struct Value;

struct IRNode {

	enum Encoding {
		NOP,

		UNARY,
		BINARY,
		TRINARY,
		FOLD,

		SPECIAL,
		CONSTANT,
		LOAD
	};

	Encoding enc;
	IROpCode::Enum op;
	Type::Enum type;

	bool live;

	bool liveOut;
	Value out;
	
	struct Shape {
		int64_t length;
		IRef filter;
		int64_t levels;
		IRef split;

		bool operator==(Shape const& o) const {
			return length == o.length &&
				filter == o.filter &&
				levels == o.levels &&
				split == o.split;
		}

		bool operator!=(Shape const& o) const {
			return !(*this == o);
		}
	};

	Shape shape;

	bool isDouble() const { return type == Type::Double; }
	bool isInteger() const { return type == Type::Integer; }
	bool isLogical() const { return type == Type::Logical; }

	union {
		struct {
			IRef a;
			int64_t data;
		} unary;
		struct {
			IRef a,b;
			int64_t data;
		} binary;
		struct {
			IRef a, b, c;
		} trinary;

		struct {
			int64_t a,b;
		} special;
		struct {
			union {
				int64_t i;
				double d;
				char l;
			};
		} constant;

		struct {
			IRef a;
		} fold;
	};
};

#endif /* IR_H_ */
