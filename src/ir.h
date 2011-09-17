#ifndef _RIPOSTE_IR_H
#define _RIPOSTE_IR_H

#include "enum.h"
#include "common.h"
#include "bc.h"
			\
/*	_(vload,      "vload",     ref, ___, ___) \
	_(rload,      "rload",     ref, ___, ___) \
	_(sload,      "sload",     ref, ___, ___) \
	_(cast,       "cast",      ref, ref, ___) \
	_(iassign,    "iassign",   ___, ___, ___) \
	_(guard,      "guard",     ___, ref, ___) \
	_(move,       "move",      ref, ref, ___) \

	_(pos,        "pos",       ref, ref, ___) \
	_(idiv,       "idiv",      ref, ref, ref) \
	_(mod,        "mod",       ref, ref, ref) \

	_(lt,         "lt",        ref, ref, ref) \
	_(gt,         "gt",        ref, ref, ref) \
	_(eq,         "eq",        ref, ref, ref) \
	_(neq,        "neq",       ref, ref, ref) \
	_(ge,         "ge",        ref, ref, ref) \
	_(le,         "le",        ref, ref, ref) \
	_(lnot,       "lnot",      ref, ref, ___) \
	_(land,       "land",      ref, ref, ref) \
	_(lor,        "lor",       ref, ref, ref) \
	_(sland,       "sland",      ref, ref, ref) \
	_(slor,        "slor",       ref, ref, ref) \
	_(sign,       "sign",      ref, ref, ___) \

	_(istrue,     "istrue",    ref, ref, ___)\
	_(logical1,   "logical",   ref, ref, ___) \
	_(integer1,   "integer",   ref, ref, ___) \
	_(double1,    "double",    ref, ref, ___) \
	_(complex1,   "complex",   ref, ref, ___) \
	_(character1, "character", ref, ref, ___) \
	_(seq,        "seq",       ___, ___, ___) \
	_(signif,     "signif",    ref, ref, ___) \
*/
	
#define IR_BINARY(_) \
	BINARY_ARITH_MAP_BYTECODES(_)
	
#define IR_UNARY(_) \
	UNARY_ARITH_MAP_BYTECODES(_)

#define IR_FOLD(_) \
	_(sum,        "sum",       SumOp) \
	_(prod,       "prod",      ProdOp) \

#define IR_SCAN(_) \
	_(cumsum,     "cumsum",    SumOp) \
	_(cumprod,    "cumprod",   ProdOp) \

#define IR_ENUM(_) 	\
	IR_BINARY(_) \
	IR_UNARY(_) \
	IR_FOLD(_) \
	IR_SCAN(_) \
	_(coerce, "coerce") \
	_(seq, "seq") 

    

DECLARE_ENUM(IROpCode,IR_ENUM)



inline static bool IROpCode_is_binary(IROpCode::Enum code) {
#define IR_COUNT(...) 1 +
	const int n_binary = IR_BINARY(IR_COUNT) 0;
#undef IR_COUNT
	return code < n_binary;
}

struct IROp {
	enum Encoding { E_SCALAR, E_VECTOR };
	enum Type { T_INT, T_DOUBLE };
	union {
		uint16_t op;
		struct {
			Encoding a_enc : 1;
			Encoding b_enc : 1;
			Type     a_typ : 1;
			Type	 b_typ : 1;
			IROpCode::Enum code : 13;
		};
	};
};

class Value;

struct IRNode {
	IRNode() {}

	IROp op;

	//for flags that are not part of the op encoding
	bool r_external;
	bool a_external;
	bool b_external;

	//3-op code, r is dest, r = a + b, where r is the position in the list of IRNodes where this op resides
	
	union InputReg {
		int64_t i;
		double * p;
		double d;
	};
	InputReg a;
	InputReg b;
	InputReg r;

	bool usesRegA() {
		return !a_external && op.a_enc == IROp::E_VECTOR;
	}
	bool usesRegB() {
		return IROpCode_is_binary(op.code) && !b_external && op.b_enc == IROp::E_VECTOR;
	}

	std::string toString() const {
		std::ostringstream out;
		out << IROpCode::toString(op.code);

		if(op.a_typ == IROp::T_INT)
			out << "_i";
		else
			out << "_d";

		if(op.b_typ == IROp::T_INT)
					out << "i";
				else
					out << "d";

		out << "(" << op.op << ")";
		out << "\t";

		if(a_external)
			out << "$" << a.p;
		else if(op.a_enc == IROp::E_SCALAR)
			out << a.d;
		else
			out << "n" << a.i;

		out << "\t";

		if(b_external)
			out << "$" << b.p;
		else if(op.b_enc == IROp::E_SCALAR)
			out << b.d;
		else if(IROpCode_is_binary(op.code))
			out << "n" << b.i;
		else
			out << "_";

		return out.str();
	}
};


typedef size_t IRef;

#endif /* IR_H_ */
