#include "interpreter.h"
#include "vector.h"
#include "ops.h"
#include "sse.h"


#define BINARY_VERSIONS(name,str,op,_,...) \
	_(name##dvv,#name  "dvv", Map2VV, op, TDouble, *inst.a.dpp, *inst.b.dpp ) \
	_(name##dvs,#name  "dvs", Map2VS, op, TDouble, *inst.a.dpp, inst.b.d ) \
	_(name##dsv,#name  "dsv", Map2SV, op, TDouble, inst.a.d, *inst.b.dpp ) \
	_(name##ivv,#name  "ivv", Map2VV, op, TInteger,*inst.a.ipp, *inst.b.ipp ) \
	_(name##ivs,#name  "ivs", Map2VS, op, TInteger,*inst.a.ipp,   inst.b.i ) \
	_(name##isv,#name  "isv", Map2SV, op, TInteger,  inst.a.i,   *inst.b.ipp ) \

#define UNARY_VERSIONS(name,str,op,_,...) \
	_(name##d, #name "d", op, TDouble, *inst.a.dpp) \
	_(name##i, #name "i", op, TInteger,*inst.a.ipp) \

#define FOLD_VERSIONS
#define SCAN_VERSIONS


#define BINARY_ARITH_MAP_TRACE_BC(_,p) \
	_(add, "add",	AddOp, p) \
	_(sub, "sub",	SubOp, p) \
	_(mul, "mul",	MulOp, p) \
	_(div, "div",	DivOp, p) \
	_(idiv, "idiv",	IDivOp, p) \
	_(mod, "mod",	ModOp, p) \
	_(pow, "pow",	PowOp, p) \

#define UNARY_ARITH_MAP_TRACE_BC(_,p) \
	_(pos, "pos", 	PosOp, p) \
	_(neg, "neg", 	NegOp, p) \
	_(abs, "abs", 	AbsOp, p) \
	_(sign, "sign",	SignOp, p) \
	_(sqrt, "sqrt",	SqrtOp, p) \
	_(floor, "floor",	FloorOp, p) \
	_(ceiling, "ceiling",	CeilingOp, p) \
	_(trunc, "trunc",	TruncOp, p) \
	_(round, "round",	RoundOp, p) \
	_(signif, "signif",	SignifOp, p) \
	_(exp, "exp",	ExpOp, p) \
	_(log, "log",	LogOp, p) \
	_(cos, "cos",	CosOp, p) \
	_(sin, "sin",	SinOp, p) \
	_(tan, "tan",	TanOp, p) \
	_(acos, "acos",	ACosOp, p) \
	_(asin, "asin",	ASinOp, p) \
	_(atan, "atan",	ATanOp, p) \

#define TRACE_BC(_) \
	BINARY_ARITH_MAP_TRACE_BC(BINARY_VERSIONS,_) \
	UNARY_ARITH_MAP_TRACE_BC(UNARY_VERSIONS,_) \
	_(casti2d,"casti2d")
		/*ARITH_FOLD_BYTECODES(FOLD_VERSIONS) \
		ARITH_SCAN_BYTECODES(SCAN_VERSIONS) \*/

DECLARE_ENUM(TraceBC,TRACE_BC)
DEFINE_ENUM_TO_STRING(TraceBC,TRACE_BC)



struct TraceInst {
	TraceBC::Enum bc;
	enum { REG_R = 1, REG_A = 2, REG_B = 4 };
	char flags; //which elements are registers? this simplifies the register allocation pass
	union {
		void * p;
		int64_t i;
		double d;
	} r;
	union Operand {
		void ** pp;
		double ** dpp;
		int64_t ** ipp;
		int64_t i;
		double d;
	};
	Operand a,b;
};



//bit-string based allocator for registers

struct Allocator {
	uint32_t a;
	Allocator() : a(~0) {}
	void print() {
		for(int i = 0; i < 32; i++)
			if( a & (1 << i))
				printf("-");
			else
				printf("a");
		printf("\n");
	}
	int allocate() {
		int reg = ffs(a) - 1;
		a &= ~(1 << reg);
		return reg;
	}
	void free(int reg) {
		a |= (1 << reg);
	}
};


struct TraceCode {
	TraceCode(Trace * t) { trace = t; n_insts = n_incrementing_pointers = 0; }
	Trace * trace;
	TraceInst insts[TRACE_MAX_NODES];
	size_t n_insts;
	double ** incrementing_pointers[TRACE_MAX_NODES];
	size_t n_incrementing_pointers;
	void ** reference_to_result[TRACE_MAX_NODES]; //mapping from IRef from IRNode to the result pointer in an instruction where the result of that node will be written
	double registers [TRACE_MAX_VECTOR_REGISTERS][TRACE_VECTOR_WIDTH] __attribute__ ((aligned (16))); //for sse alignment

	void compile() {

		//pass 1 instruction selection
		for(IRef i = 0; i < trace->n_nodes; i++) {
			IRNode & node = trace->nodes[i];
			switch(node.op) {
#define BINARY_OP(op,...) case IROpCode :: op : EmitBinary(TraceBC::op##isv,TraceBC::op##ivs,TraceBC::op##ivv,TraceBC::op##dsv,TraceBC::op##dvs,TraceBC::op##dvv,\
			                                               i); \
			                                          break;
#define UNARY_OP(op,...) case IROpCode :: op : EmitUnary(TraceBC::op##i,TraceBC::op##d,i); break;

			BINARY_ARITH_MAP_BYTECODES(BINARY_OP)
			UNARY_ARITH_MAP_BYTECODES(UNARY_OP)
#undef UNARY_OP
#undef BINARY_OP

			case IROpCode::cast: EmitUnary(TraceBC::casti2d,TraceBC::casti2d,i); break;
			case IROpCode::loadc:
				//nop, these will be inlined into arithmetic ops
				break;
			case IROpCode::loadv:
				//instructions referencing this load will look up its pointer field to read the value
				incrementing_pointers[n_incrementing_pointers++] = (double**)&node.loadv.p;
				break;
			case IROpCode::storev:
				*reference_to_result[node.store.a] = node.store.dst->p;
				incrementing_pointers[n_incrementing_pointers++] = (double**)reference_to_result[node.store.a];
				break;
			case IROpCode::storec:
				*reference_to_result[node.store.a] = &node.store.dst->p;
				break;
			default:
				_error("unsupported op");
			}
		}

		//pass 2 register allocation
		Allocator free_reg;
		for(int i = n_insts; i > 0; i--) {
			TraceInst & inst = insts[i - 1];
			if(inst.flags & TraceInst::REG_R) {
				if(inst.r.p == NULL) { //inst is dead but for now we just allocate a register for it anyway
					inst.r.p = registers[free_reg.allocate()];
				}
				int reg = ((double*)inst.r.p - &registers[0][0]) / TRACE_VECTOR_WIDTH;
				free_reg.free(reg);
			}
			if(inst.flags & TraceInst::REG_A) {
				if(*inst.a.pp == NULL) {
					*inst.a.pp = registers[free_reg.allocate()];
				}
			}
			if(inst.flags & TraceInst::REG_B) {
				if(*inst.b.pp == NULL) {
					*inst.b.pp = registers[free_reg.allocate()];
				}
			}
		}
	}

	void execute(State & state) {
		//interpret
		for(int64_t i = 0; i < trace->length; i += TRACE_VECTOR_WIDTH) {
			for(size_t j = 0; j < n_insts; j++) {
				TraceInst & inst = insts[j];
				switch(inst.bc) {
	#define BINARY_OP(name,str, map, op, typ, get_a, get_b) \
				case TraceBC :: name :  map< op < typ >, TRACE_VECTOR_WIDTH >::eval(state, get_a,get_b,(op < typ >::R*) inst.r.p); break;
	#define UNARY_OP(name,str, op, typ, get_a) \
				case TraceBC :: name :  Map1< op < typ >, TRACE_VECTOR_WIDTH >::eval(state, get_a,(op < typ >::R*) inst.r.p); break;
				BINARY_ARITH_MAP_TRACE_BC(BINARY_VERSIONS,BINARY_OP)
				UNARY_ARITH_MAP_TRACE_BC(UNARY_VERSIONS,UNARY_OP)
	#undef BINARY_OP
	#undef UNARY_OP
				case TraceBC :: casti2d: Map1< CastOp<Integer, Double> , TRACE_VECTOR_WIDTH>::eval(state, *inst.a.ipp , (double *)inst.r.p); break;
				}
			}
			for(size_t j = 0; j < n_incrementing_pointers; j++)
				(*incrementing_pointers[j]) += TRACE_VECTOR_WIDTH;
		}
	}
private:

	TraceInst::Operand GetOperand(IRef r, bool * isConstant, bool * isRegister) {
		TraceInst::Operand a;
		IRNode & node = trace->nodes[r];
		if(node.op == IROpCode::loadc) {
			*isConstant = true;
			*isRegister = false;
			a.i = node.loadc.i;
		} else if(node.op == IROpCode::loadv) {
			*isRegister = false;
			*isConstant = false;
			a.pp = &node.loadv.p;
		} else {
			*isRegister = true;
			*isConstant = false;
			a.pp = reference_to_result[r];
			assert(reference_to_result[r] != NULL);
		}
		return a;
	}

	void EmitBinary(TraceBC::Enum oisv,TraceBC::Enum oivs, TraceBC::Enum oivv,
                    TraceBC::Enum odsv,TraceBC::Enum odvs, TraceBC::Enum odvv,
                    IRef node_ref) {
		IRNode & node = trace->nodes[node_ref];
		bool a_is_reg; bool b_is_reg;
		bool a_is_const; bool b_is_const;
		TraceInst & inst = insts[n_insts++];
		inst.a = GetOperand(node.binary.a,&a_is_const,&a_is_reg);
		inst.b = GetOperand(node.binary.b,&b_is_const,&b_is_reg);
		inst.flags = TraceInst::REG_R;
		if(a_is_reg)
			inst.flags |= TraceInst::REG_A;
		if(b_is_reg)
			inst.flags |= TraceInst::REG_B;
		inst.r.p = NULL;
		reference_to_result[node_ref] = &inst.r.p;
		switch(node.type) {
		case Type::Integer:
			if(a_is_const) {
				inst.bc = oisv;
			} else if(b_is_const) {
				inst.bc = oivs;
			} else {
				inst.bc = oivv;
			}
			break;
		case Type::Double:
			if(a_is_const) {
				inst.bc = odsv;
			} else if(b_is_const) {
				inst.bc = odvs;
			} else {
				inst.bc = odvv;
			} break;
		default:
			_error("unsupported type");
			break;
		}
	}
	void EmitUnary(TraceBC::Enum oi, TraceBC::Enum od,
                    IRef node_ref) {
		IRNode & node = trace->nodes[node_ref];
		bool a_is_reg;
		bool a_is_const;
		TraceInst & inst = insts[n_insts++];
		inst.a = GetOperand(node.unary.a,&a_is_const,&a_is_reg);
		assert(!a_is_const);
		inst.flags = TraceInst::REG_R;
		if(a_is_reg)
			inst.flags |= TraceInst::REG_A;
		reference_to_result[node_ref] = &inst.r.p;
		inst.r.p = NULL;
		switch(node.type) {
		case Type::Integer:
			inst.bc = oi;
			break;
		case Type::Double:
			inst.bc = od;
			break;
		default:
			_error("unsupported type");
			break;
		}
	}

};


void Trace::Execute(State & state) {
	InitializeOutputs(state);
	if(state.tracing.verbose)
		printf("executing trace:\n%s\n",toString(state).c_str());

	TraceCode trace_code(this);

	trace_code.compile();
	trace_code.execute(state);

	WriteOutputs(state);
}
