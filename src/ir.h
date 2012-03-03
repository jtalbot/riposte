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
		SPECIAL_FOLD_BYTECODES(_) \
		SCAN_BYTECODES(_) \
		_(cast, "cast", ___) \
		_(constant,"constant", ___) \
		_(seq, "seq", ___) \
		_(rep, "rep", ___) \
		_(random, "random", ___) \
		_(load,"load", ___) \
		_(filter, "filter", ___) \
		_(split, "split", ___) \
		_(nop, "nop", ___) \
		_(ifelse, "ifelse", ___) \

DECLARE_ENUM(IROpCode,IR_ENUM)

struct Value;

struct IRNode {

	enum Arity {
		NULLARY,
		UNARY,
		BINARY,
		TRINARY
	};

	enum Group {
		NOP,
		GENERATOR,
		MAP,
		FILTER,
		FOLD,
		SPLIT
	};

	struct Shape {
		int64_t length;
		IRef filter;
		int64_t levels;
		IRef split;
		bool blocking;

		bool operator==(Shape const& o) const {
			return length == o.length &&
				filter == o.filter &&
				levels == o.levels &&
				split == o.split &&
				blocking == o.blocking;
		}

		bool operator!=(Shape const& o) const {
			return !(*this == o);
		}
	};

	Arity arity;
	Group group;
	IROpCode::Enum op;

	Type::Enum type;
	Shape shape, outShape;

	bool live;
	bool liveOut;
	Value in;
	Value out;
	
	bool operator==(IRNode const& o) const {
		bool eq = (op == o.op && type == o.type && shape == o.shape);
		switch(group) {
			case IRNode::NOP:
				return false;
				break;
			case IRNode::GENERATOR: {
				if(op == IROpCode::load) {
					return eq && unary.a == o.unary.a && in == o.in;
				} else if(op == IROpCode::constant) {
					return eq && ((type == Type::Double && constant.d == o.constant.d) || 
							(type == Type::Integer && constant.i == o.constant.i) || 
							(type == Type::Logical && constant.l == o.constant.l));
				} else if(op == IROpCode::seq || op == IROpCode::rep) {
					return eq && ((type == Type::Double && sequence.da == o.sequence.da && sequence.db == o.sequence.db) || 
							(type == Type::Integer && sequence.ia == o.sequence.ia && sequence.ib == o.sequence.ib));
				} else if(op == IROpCode::random) {
					return false;	// two random number sources are never the same
				} 
				else {
					_error("Bad generator in node equality");
				}
			} break;
			default: {
				switch(arity) {
					case IRNode::TRINARY:
						return eq && trinary.a == o.trinary.a && trinary.b == o.trinary.b && o.trinary.c == o.trinary.c;
					break;
					case IRNode::BINARY:
						return eq && binary.a == o.binary.a && binary.b == o.binary.b;
					break;
					case IRNode::UNARY:
						return eq && unary.a == o.unary.a;
					break;
					case IRNode::NULLARY:
					default:
						return eq;
					break;
				}
			} break;
		}
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
