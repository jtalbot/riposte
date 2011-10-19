#include "recording.h"
#include "interpreter.h"
#include "ir.h"
#include "ops.h"

#define ENUM_RECORDING_STATUS(_) \
	_(NO_ERROR,"NO_ERROR") \
	_(FALLBACK, "trace falling back to normal interpreter but not exiting") \
	_(RESOURCE, "trace ran out of resources") \
	_(UNSUPPORTED_OP,"trace encountered unsupported op") \
	_(UNSUPPORTED_TYPE,"trace encountered an unsupported type") \
	_(INTERPRETER_DEPENDENCY, "non-traceable op requires the value of a future")

DECLARE_ENUM(RecordingStatus,ENUM_RECORDING_STATUS)
DEFINE_ENUM_TO_STRING(RecordingStatus,ENUM_RECORDING_STATUS)

//for brevity
#define TRACE (state.tracing.current_trace)
#define REG(state, i) (*(state.base+i))

#define OP_NOT_IMPLEMENTED(op,...) \
RecordingStatus::Enum op##_record(State & state, Instruction const & inst, Instruction const ** pc) { \
	return RecordingStatus::UNSUPPORTED_OP; \
} \


RecordingStatus::Enum get_record(State & state, Instruction const & inst, Instruction const ** pc) {
	if(!TRACE.Reserve(0,1))
		return RecordingStatus::RESOURCE;
	*pc = get_op(state,inst);
	Value & r = REG(state,inst.c);

	TRACE.UnionWithMaxLiveRegister(state.base,inst.c);

	if(r.isFuture()) {
		TRACE.EmitRegOutput(state.base,inst.c);
	}
	return RecordingStatus::NO_ERROR;
}

RecordingStatus::Enum kget_record(State & state, Instruction const & inst, Instruction const ** pc) {
	*pc = kget_op(state,inst);
	return RecordingStatus::NO_ERROR;
}


OP_NOT_IMPLEMENTED(iget)

RecordingStatus::Enum assign_record(State & state, Instruction const & inst, Instruction const ** pc) {
	if(!TRACE.Reserve(0,1))
		return RecordingStatus::RESOURCE;
	*pc = assign_op(state,inst);
	Value& r = REG(state, inst.c);
	if(r.isFuture()) {
		//Note: this call to makePointer is redundant:
		//if the variable is cached then we could construct the Pointer from the cache
		//otherwise the inline cache is updated, which involves creating a pointer

		//Inline this logic here would make the recorder more fragile, so for now we simply construct the pointer again:
		TRACE.EmitVarOutput(state,state.frame.environment->makePointer(String::Init(inst.a)));
	}
	TRACE.SetMaxLiveRegister(state.base,inst.c);
	return RecordingStatus::NO_ERROR;
}

OP_NOT_IMPLEMENTED(assign2)

#define CHECK_REG(r) (REG(state,r).isFuture())
//temporary defines to generate code for checked interpret
#define A CHECK_REG(inst.a) ||
#define B CHECK_REG(inst.b) ||
#define C CHECK_REG(inst.c) ||
#define B_1 CHECK_REG(inst.b - 1) ||
#define C_1 CHECK_REG(inst.c - 1) ||

//all operations that first verify their inputs are not futures, and then call into the scalar interpreter to fulfill them
#define CHECKED_INTERPRET(op, checks) \
		RecordingStatus::Enum op##_record(State & state, Instruction const & inst, Instruction const ** pc) { \
			if(checks false) \
				return RecordingStatus::INTERPRETER_DEPENDENCY; \
			*pc = op##_op(state,inst); \
			return RecordingStatus::NO_ERROR; \
		} \

CHECKED_INTERPRET(eassign, A B C)
CHECKED_INTERPRET(iassign, A B C)
CHECKED_INTERPRET(jt, B)
CHECKED_INTERPRET(jf, B)
CHECKED_INTERPRET(branch, A)
CHECKED_INTERPRET(subset, A B C)
CHECKED_INTERPRET(subset2, A B C)
CHECKED_INTERPRET(colon, A B C)
CHECKED_INTERPRET(forbegin, B_1)
CHECKED_INTERPRET(forend, B_1)
CHECKED_INTERPRET(UseMethod, A C)
CHECKED_INTERPRET(call, A)
#undef A
#undef B
#undef B_1
#undef C
#undef C_1
#undef CHECKED_INTERPRET

OP_NOT_IMPLEMENTED(icall)


struct LoadCache {
	IRef get(State & state, const Value& v) {
		uint64_t idx = (int64_t) v.p;
		idx += idx >> 32;
		idx += idx >> 16;
		idx += idx >> 8;
		idx &= 0xFF;
		IRef cached = cache[idx];
		if(cached < TRACE.n_pending &&
		   TRACE.nodes[cached].op == IROpCode::loadv &&
		   TRACE.nodes[cached].loadv.p == v.p) {
			return cached;
		} else {
			return (cache[idx] = TRACE.EmitLoadV(v.type,v.p));
		}
	}
	IRef cache[256];
};

bool get_input(State & state, Value & v, IRef * ref, bool * can_fallback, bool * should_record) {

	static LoadCache load_cache;

	//encode the type;
	switch(v.type) {
	case Type::Integer: /* fallthrough */
	case Type::Double:
		//constant or external reference?
		if(v.length == 1) {
			*ref = TRACE.EmitLoadC(v.type,v.i);
		} else if(v.length == TRACE.length) {
			*ref = load_cache.get(state,v);
			*should_record = true;
		} else {
			return false;
		}
		return true;
		break;
	case Type::Future:
		*can_fallback = false;
		*should_record = true;
		if(v.length != TRACE.length) {
			return false;
		} else {
			*ref = v.future.ref;
			return true;
		}
		break;
	default:
		return false;
		break;
	}
}

void coerce_scalar(Type::Enum to, IRNode & n) {
	switch(n.type) {
	case Type::Integer: switch(to) {
		case Type::Double: n.loadc.d = n.loadc.i; break;
		case Type::Logical: n.loadc.l = n.loadc.i; break;
		default: _error("unknown cast"); break;
	} break;
	case Type::Double: switch(to) {
		case Type::Integer: n.loadc.i = (int64_t) n.loadc.d; break;
		case Type::Logical: n.loadc.l = (char) n.loadc.d; break;
		default: _error("unknown cast"); break;
	} break;
	case Type::Logical: switch(to) {
		case Type::Double: n.loadc.d = n.loadc.l; break;
		case Type::Integer: n.loadc.i = n.loadc.l; break;
		default: _error("unknown cast"); break;
	} break;
	default: _error("unknown cast"); break;
	}
	n.type = to;
}

IRef coerce(State & state, Type::Enum dst_type, IRef v) {
	IRNode & n = TRACE.nodes[v];
	if(dst_type == n.type)
		return v;
	else if(n.op == IROpCode::loadc) {
		coerce_scalar(dst_type,n);
		return v;
	} else {
		return TRACE.EmitUnary(IROpCode::cast,dst_type,v);
	}
}

RecordingStatus::Enum binary_record(ByteCode::Enum bc, IROpCode::Enum op, State & state, Instruction const & inst) {
	Value & a = REG(state,inst.a);
	Value & b = REG(state,inst.b);
	if(!TRACE.Reserve(4,1)) //2 loads, 1 coerce, 1 op, 1 output
		return RecordingStatus::RESOURCE;
	bool can_fallback = true;
	bool should_record = false;
	IRef aref;
	IRef bref;
	bool ar = get_input(state,a,&aref,&can_fallback,&should_record);
	bool br = get_input(state,b,&bref,&can_fallback,&should_record);
	if(should_record && ar && br) {
		Type::Enum rtyp,atyp,btyp;
		selectType(bc,TRACE.nodes[aref].type,TRACE.nodes[bref].type,&atyp,&btyp,&rtyp);
		TRACE.EmitRegOutput(state.base,inst.c);
		TRACE.SetMaxLiveRegister(state.base,inst.c);
		Future::Init(REG(state,inst.c),
				     rtyp,
				     TRACE.length,
				     TRACE.EmitBinary(op,rtyp,coerce(state,atyp,aref),coerce(state,btyp,bref)));
		TRACE.Commit();
		return RecordingStatus::NO_ERROR;
	} else {
		TRACE.Rollback();
		return (can_fallback) ? RecordingStatus::FALLBACK : RecordingStatus::UNSUPPORTED_TYPE;
	}

}

RecordingStatus::Enum unary_record(ByteCode::Enum bc, IROpCode::Enum op, State & state, int64_t length, Instruction const & inst) {
	Value & a = REG(state,inst.a);
	if(!TRACE.Reserve(2,1))
		return RecordingStatus::RESOURCE;

	bool can_fallback = true;
	bool should_record = false;
    IRef aref;
	bool ar = get_input(state,a,&aref,&can_fallback,&should_record);

	if(should_record && ar) {
		Type::Enum rtyp,atyp;
		selectType(bc,TRACE.nodes[aref].type,&atyp,&rtyp);
		Future::Init(REG(state,inst.c),
				     rtyp,
				     length,
				     TRACE.EmitUnary(op,rtyp,coerce(state,atyp,aref)));
		TRACE.EmitRegOutput(state.base,inst.c);
		TRACE.SetMaxLiveRegister(state.base,inst.c);
		TRACE.Commit();
		return RecordingStatus::NO_ERROR;
	} else {
		TRACE.Rollback();
		return (can_fallback) ? RecordingStatus::FALLBACK : RecordingStatus::UNSUPPORTED_TYPE;
	}

}

//all arithmetic binary ops share the same recording implementation
#define BINARY_OP(op,...) RecordingStatus::Enum op##_record(State & state, Instruction const & inst, Instruction const ** pc) { \
	RecordingStatus::Enum status = binary_record(ByteCode :: op,IROpCode :: op, state, inst);\
	if(RecordingStatus::FALLBACK == status) { \
		*pc = op##_op(state,inst); \
		return RecordingStatus::NO_ERROR; \
	} \
	if(RecordingStatus::NO_ERROR == status) \
		(*pc)++; \
	return status; \
}
//all unary arithmetic ops share the same implementation as well
#define UNARY_OP(op,...) RecordingStatus::Enum op##_record(State & state, Instruction const & inst, Instruction const ** pc) { \
	RecordingStatus::Enum status = unary_record(ByteCode :: op , IROpCode :: op, state, TRACE.length, inst);\
	if(RecordingStatus::FALLBACK == status) { \
		*pc = op##_op(state,inst); \
		return RecordingStatus::NO_ERROR; \
	} \
	if(RecordingStatus::NO_ERROR == status) \
		(*pc)++; \
	return status; \
}
#define FOLD_OP(op,...) RecordingStatus::Enum op##_record(State & state, Instruction const & inst, Instruction const ** pc) { \
	RecordingStatus::Enum status = unary_record(ByteCode :: op, IROpCode :: op, state, 1, inst);\
	if(RecordingStatus::FALLBACK == status) { \
		*pc = op##_op(state,inst); \
		return RecordingStatus::NO_ERROR; \
	} \
	if(RecordingStatus::NO_ERROR == status) \
		(*pc)++; \
	return status; \
}


BINARY_ARITH_MAP_BYTECODES(BINARY_OP)
UNARY_ARITH_MAP_BYTECODES(UNARY_OP)
ARITH_FOLD_BYTECODES(FOLD_OP)

BINARY_ORDINAL_MAP_BYTECODES(BINARY_OP)
UNARY_LOGICAL_MAP_BYTECODES(UNARY_OP)
BINARY_LOGICAL_MAP_BYTECODES(BINARY_OP)

OP_NOT_IMPLEMENTED(sland)
OP_NOT_IMPLEMENTED(slor)

OP_NOT_IMPLEMENTED(list)
OP_NOT_IMPLEMENTED(logical1)
OP_NOT_IMPLEMENTED(integer1)
OP_NOT_IMPLEMENTED(double1)
OP_NOT_IMPLEMENTED(complex1)
OP_NOT_IMPLEMENTED(character1)
OP_NOT_IMPLEMENTED(raw1)

OP_NOT_IMPLEMENTED(min)
OP_NOT_IMPLEMENTED(max)
OP_NOT_IMPLEMENTED(any)
OP_NOT_IMPLEMENTED(all)

UNARY_OP(cumsum)
UNARY_OP(cumprod)

OP_NOT_IMPLEMENTED(cummin)
OP_NOT_IMPLEMENTED(cummax)
OP_NOT_IMPLEMENTED(cumany)
OP_NOT_IMPLEMENTED(cumall)


RecordingStatus::Enum jmp_record(State & state, Instruction const & inst, Instruction const ** pc) {
	//this is just a constant jump, nothing to record
	*pc = jmp_op(state,inst);
	return RecordingStatus::NO_ERROR;
}

RecordingStatus::Enum function_record(State & state, Instruction const & inst, Instruction const ** pc) {
	*pc = function_op(state,inst);
	return RecordingStatus::NO_ERROR;
}


//we can extract the type from the future so we can continue a trace beyond a type-check
RecordingStatus::Enum type_record(State & state, Instruction const & inst, Instruction const ** pc) {
	Character c(1);
	// Should have a direct mapping from type to symbol.
	Value & a = REG(state, inst.a);
	Type::Enum atyp = (a.isFuture()) ? a.future.typ : a.type;
	c[0] = state.internStr(Type::toString(atyp));
	REG(state, inst.c) = c;
	(*pc)++;
	return RecordingStatus::NO_ERROR;
}
RecordingStatus::Enum length_record(State & state, Instruction const & inst, Instruction const ** pc) {
	*pc = length_op(state, inst);
	return RecordingStatus::NO_ERROR;
}
RecordingStatus::Enum missing_record(State & state, Instruction const & inst, Instruction const ** pc) {
	*pc = length_op(state, inst);
	return RecordingStatus::NO_ERROR;
}
RecordingStatus::Enum ret_record(State & state, Instruction const & inst, Instruction const ** pc) {
	//ret writes a value into a register of the caller's frame. If this value is a future we need to
	//record it as a potential output location
	if(!TRACE.Reserve(0,1))
		return RecordingStatus::RESOURCE;
	Value * result = state.frame.result;
	int64_t offset = result - state.frame.returnbase;
	int64_t max_live = state.frame.returnbase - state.base;
	*pc = ret_op(state,inst); //warning: ret_op will change 'frame'

	if(result->isFuture()) {
		TRACE.EmitRegOutput(state.base,offset);
	}

	TRACE.SetMaxLiveRegister(state.base,max_live);

	return RecordingStatus::NO_ERROR;
}

//done forces the trace to flush to ensure that there are no futures when the interpreter exits
RecordingStatus::Enum done_record(State & state, Instruction const & inst, Instruction const ** pc) {
	return RecordingStatus::UNSUPPORTED_OP;
}

RecordingStatus::Enum seq_record(State & state, Instruction const & inst, Instruction const ** pc) {
	Value & a = REG(state,inst.a);
	Value & b = REG(state,inst.b);

	if(a.isFuture() || b.isFuture()) {
		// don't increment the pc
		return RecordingStatus::UNSUPPORTED_OP;
	}
	int64_t len = As<Integer>(state, REG(state, inst.a))[0];
	int64_t step = As<Integer>(state, REG(state, inst.b))[0];
	if(len != TRACE.length) {
		*pc = seq_op(state,inst); //this isn't ideal, as this will redo the As operators above
	} else {
		if(!TRACE.Reserve(1,1))
			return RecordingStatus::RESOURCE;

		Future::Init(REG(state,inst.c),
				     Type::Integer,
				     len,
				     TRACE.EmitSpecial(IROpCode::seq,Type::Integer,len,step));
		TRACE.SetMaxLiveRegister(state.base,inst.c);
		TRACE.Commit();
		(*pc)++;
	}
	return RecordingStatus::NO_ERROR;
}

//check trace exit condition
// -- do we need to abort due to conditions applying to all opcodes? if so, abort the trace, then exit the recorder
// -- is the trace complete? if so, install the trace, and exit the recorder
// -- otherwise, the recorder continues normally
//returns true if we should continue recording
static RecordingStatus::Enum recording_check_conditions(State& state, Instruction const * inst) {
	TRACE.n_recorded++;
	if(++TRACE.n_recorded > TRACE_MAX_RECORDED) {
		return RecordingStatus::RESOURCE;
	}
	return RecordingStatus::NO_ERROR;
}

Instruction const * recording_interpret(State& state, Instruction const* pc) {
	RecordingStatus::Enum status = RecordingStatus::NO_ERROR;
	while( true ) {
#define RUN_RECORD(name,str,...) case ByteCode::name: {  /*printf("rec " #name "\n");*/ status = name##_record(state, *pc,&pc); } break;
		switch(pc->bc) {
			BYTECODES(RUN_RECORD)
		}
#undef RUN_RECORD
		if(   RecordingStatus::NO_ERROR != status
		   || RecordingStatus::NO_ERROR != (status = recording_check_conditions(state,pc))) {
			if(state.tracing.verbose)
				printf("%s op ended trace: %s\n",ByteCode::toString(pc->bc),RecordingStatus::toString(status));
			state.tracing.end_tracing(state);
			return pc;
		}
		//printf(" .\n");
	}
	return pc;
}
