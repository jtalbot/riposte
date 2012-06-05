#ifndef RECORDING_H
#define RECORDING_H

#include "ops.h"

#define ENUM_RECORDING_STATUS(_) \
	_(NO_ERROR,"NO_ERROR") \
	_(FALLBACK, "trace falling back to normal interpreter but not exiting") \
	_(RECORD_LIMIT, "maximum record limit reached without executing any traces") \
	_(UNSUPPORTED_OP,"trace encountered unsupported op") \
	_(UNSUPPORTED_TYPE,"trace encountered an unsupported type") \
	_(NO_LIVE_TRACES, "all traces are empty")

DECLARE_ENUM(RecordingStatus,ENUM_RECORDING_STATUS)
DEFINE_ENUM_TO_STRING(RecordingStatus,ENUM_RECORDING_STATUS)

// Figure out the type of the operation given an input type

template<class X>
void unaryResultType(Type::Enum& mtype, Type::Enum& rtype) {
	mtype = X::MA::VectorType;
	rtype = X::R::VectorType;
}

void selectType(ByteCode::Enum bc, Type::Enum input, Type::Enum& mtype, Type::Enum& rtype) {
	switch(bc) {
#define UNARY_OP(Name, String, Op, Group, ...) \
		case ByteCode::Name: \
		switch(input) { \
			case Type::Logical: unaryResultType< Name##VOp<Logical> >(mtype, rtype); \
			case Type::Integer: unaryResultType< Name##VOp<Integer> >(mtype, rtype); \
			case Type::Double:  unaryResultType< Name##VOp<Double> >(mtype, rtype); \
			default: _error("Unsupported type in JIT"); \
		} break;
UNARY_FOLD_SCAN_BYTECODES(UNARY_OP)
#undef UNARY_OP
		default: _error("Called unaryResultType with non-unary bytecode");
	}
}

template< template<class X, class Y> class Group, class X, class Y>
void binaryResultType(Type::Enum& matype, Type::Enum& mbtype, Type::Enum& rtype) {
	matype = Group<X, Y>::MA::VectorType;
	mbtype = Group<X, Y>::MB::VectorType;
	rtype = Group<X, Y>::R::VectorType;
}


void selectType(ByteCode::Enum bc, Type::Enum inputa, Type::Enum inputb, Type::Enum& matype, Type::Enum& mbtype, Type::Enum& rtype) {
	switch(bc) {
#define BINARY_OP(Name, String, Op, Group, ...) \
		case ByteCode::Name: \
		if(inputa == Type::Logical) { \
			if(inputb == Type::Logical) \
				binaryResultType<Group, Logical, Logical>(matype, mbtype, rtype); \
			else if(inputb == Type::Integer) \
				binaryResultType<Group, Logical, Integer>(matype, mbtype, rtype); \
			else if(inputb == Type::Double) \
				binaryResultType<Group, Logical, Double>(matype, mbtype, rtype); \
		} else if(inputa == Type::Integer) { \
			if(inputb == Type::Logical) \
				binaryResultType<Group, Integer, Logical>(matype, mbtype, rtype); \
			else if(inputb == Type::Integer) \
				binaryResultType<Group, Integer, Integer>(matype, mbtype, rtype); \
			else if(inputb == Type::Double) \
				binaryResultType<Group, Integer, Double>(matype, mbtype, rtype); \
		} else if(inputa == Type::Double) { \
			if(inputb == Type::Logical) \
				binaryResultType<Group, Double, Logical>(matype, mbtype, rtype); \
			else if(inputb == Type::Integer) \
				binaryResultType<Group, Double, Integer>(matype, mbtype, rtype); \
			else if(inputb == Type::Double) \
				binaryResultType<Group, Double, Double>(matype, mbtype, rtype); \
		} else _error("Unsupported type in JIT"); 
BINARY_BYTECODES(BINARY_OP)
#undef BINARY_OP
		default: _error("Called binaryResultType with non-binary bytecode");
	}
}

#endif
