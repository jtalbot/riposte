#ifndef _RIPOSTE_IR_H
#define _RIPOSTE_IR_H

#include "enum.h"
#include "common.h"
#include "bc.h"
#include "type.h"
#include "value.h"

#define IR_ENUM(_) \
		MAP_BYTECODES(_) \
		FOLD_BYTECODES(_) \
		SCAN_BYTECODES(_) \
		_(cast, "cast", ___) \
		_(constant,"constant", ___) \
		_(seq, "seq", ___) \
		_(load,"load", ___) \
		_(filter, "filter", ___) \
		_(split, "split", ___) \
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

		SEQUENCE,
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

	bool operator==(IRNode const& o) const {
		bool eq = (enc == o.enc && op == o.op && type == o.type && shape == o.shape);
		switch(enc) {
			case IRNode::TRINARY:
				return eq && trinary.a == o.trinary.a && trinary.b == o.trinary.b && o.trinary.c == o.trinary.c;
				break;
			case IRNode::BINARY:
				return eq && binary.a == o.binary.a && binary.b == o.binary.b;
				break;
			case IRNode::FOLD:
			case IRNode::UNARY:
				return eq && unary.a == o.unary.a;
				break;
			case IRNode::LOAD: /*fallthrough*/
				return eq && unary.a == o.unary.a && out == o.out;
				break;
			case IRNode::CONSTANT: /*fallthrough*/
				return eq && ((type == Type::Double && constant.d == o.constant.d) || 
						(type == Type::Integer && constant.i == o.constant.i) || 
						(type == Type::Logical && constant.l == o.constant.l));
				break;
			case IRNode::SEQUENCE: /*fallthrough*/
				return eq && ((type == Type::Double && sequence.da == o.sequence.da && sequence.db == o.sequence.db) || 
						(type == Type::Integer && sequence.ia == o.sequence.ia && sequence.ib == o.sequence.ia));
				break;
			case IRNode::NOP:
				return false;
				break;
		}
		return false;
	}

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
			union {
				struct { int64_t ia, ib; };
				struct { double da, db; };
			};
		} sequence;
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
