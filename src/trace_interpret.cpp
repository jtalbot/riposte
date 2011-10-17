#include "interpreter.h"
#include "vector.h"
#include "ops.h"
#include "sse.h"


#define BINARY_VERSIONS(name,...) \
	name##dvv, name##dvs, name##dsv, name##ivv, name##ivs, name##isv,
#define UNARY_VERSIONS(name,...) \
	name ## d, name ## i,
#define BINARY_VERSIONS_MONOMORPHIC(name,...) \
	name##vv, name##vs, name##sv,

namespace TraceBC {
  enum Enum {
	  BINARY_ARITH_MAP_BYTECODES(BINARY_VERSIONS)
	  UNARY_ARITH_MAP_BYTECODES(UNARY_VERSIONS)
	  ARITH_FOLD_BYTECODES(UNARY_VERSIONS)
	  ARITH_SCAN_BYTECODES(UNARY_VERSIONS)
	  BINARY_ORDINAL_MAP_BYTECODES(BINARY_VERSIONS)
	  BINARY_LOGICAL_MAP_BYTECODES(BINARY_VERSIONS_MONOMORPHIC)
	  seq,
	  casti2d,
	  castl2i,
	  castl2d,
  };
}

static inline bool is_widening_cast(TraceBC::Enum e) {
	return e == TraceBC::castl2i || e == TraceBC::castl2d;
}


struct TraceInst {
	TraceBC::Enum bc;
	enum { REG_R = 1, REG_A = 2, REG_B = 4 };
	char flags; //which elements are registers? this simplifies the register allocation pass
	union {
		void * p;
		double * dp;
		uint8_t * ip;
		unsigned char * lp;
	} r;
	union Operand {
		void ** pp;
		double ** dpp;
		int64_t ** ipp;
		uint8_t ** lpp;
		int64_t i;
		double d;
		uint8_t l;
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

//which pointer incrementing list do we use?
static size_t size_for_type(Type::Enum t) {
	if(Type::Logical == t)
		return 1;
	else
		return 8;
}

struct TraceInterpret {
	TraceInterpret(Trace * t) { trace = t; n_insts = n_incrementing_pointers_1 = n_incrementing_pointers_8 = 0; }
	Trace * trace;
	TraceInst insts[TRACE_MAX_NODES];
	size_t n_insts;
	double ** incrementing_pointers_8[TRACE_MAX_NODES];
	size_t n_incrementing_pointers_8;
	uint8_t ** incrementing_pointers_1[TRACE_MAX_NODES];
	size_t n_incrementing_pointers_1;
	TraceInst * reference_to_instruction[TRACE_MAX_NODES]; //mapping from IRef from IRNode to the result pointer in an instruction where the result of that node will be written
	double registers [TRACE_MAX_VECTOR_REGISTERS][TRACE_VECTOR_WIDTH] __attribute__ ((aligned (16))); //for sse alignment

	void compile() {

		//pass 1 instruction selection
		for(IRef i = 0; i < trace->n_nodes; i++) {
			IRNode & node = trace->nodes[i];
			switch(node.op) {
#define BINARY_OP(op,...) case IROpCode :: op : EmitBinary(TraceBC::op##isv,TraceBC::op##ivs,TraceBC::op##ivv,TraceBC::op##dsv,TraceBC::op##dvs,TraceBC::op##dvv, i); break;
#define BINARYM_OP(op,...) case IROpCode :: op : EmitBinary(TraceBC::op##sv,TraceBC::op##vs,TraceBC::op##vv,TraceBC::op##sv,TraceBC::op##vs,TraceBC::op##vv, i); break;
#define UNARY_OP(op,...) case IROpCode :: op : EmitUnary(TraceBC::op##i,TraceBC::op##d,i); break;
#define FOLD_OP(op,name,OP,...) case IROpCode :: op : EmitFold(TraceBC::op##i,TraceBC::op##d, OP<TInteger>::base(),OP<TDouble>::base(), i); break;
			BINARY_ARITH_MAP_BYTECODES(BINARY_OP)
			BINARY_ORDINAL_MAP_BYTECODES(BINARY_OP)
			BINARY_LOGICAL_MAP_BYTECODES(BINARYM_OP)
			UNARY_ARITH_MAP_BYTECODES(UNARY_OP)
			ARITH_FOLD_BYTECODES(FOLD_OP)
			ARITH_SCAN_BYTECODES(FOLD_OP)
#undef UNARY_OP
#undef BINARY_OP
#undef BINARYM_OP
#undef FOLD_OP
#undef SCAN_OP
			case IROpCode::cast: {
				TraceBC::Enum bc;
				if(node.type == Type::Double) {
					if(trace->nodes[node.unary.a].type == Type::Integer) {
						bc = TraceBC::casti2d;
					} else {
						bc = TraceBC::castl2d;
					}
				} else {
					bc = TraceBC::castl2i;
				}
				EmitUnary(bc,bc,i); break;
			}
			case IROpCode::seq: EmitSpecial(TraceBC::seq,i); break;
			case IROpCode::loadc:
				//nop, these will be inlined into arithmetic ops
				break;
			case IROpCode::loadv:
				//instructions referencing this load will look up its pointer field to read the value
				if(size_for_type(node.type) == 1)
					incrementing_pointers_1[n_incrementing_pointers_1++] = (uint8_t**)&node.loadv.p;
				else
					incrementing_pointers_8[n_incrementing_pointers_8++] = (double**)&node.loadv.p;
				break;
			case IROpCode::storev: {
				TraceInst & rinst = *reference_to_instruction[node.store.a];
				rinst.r.p = node.store.dst->p;
				rinst.flags &= ~TraceInst::REG_R;
				if(size_for_type(node.type) == 1)
					incrementing_pointers_1[n_incrementing_pointers_1++] = (uint8_t**)&rinst.r.p;
				else
					incrementing_pointers_8[n_incrementing_pointers_8++] = (double**)&rinst.r.p;
			} break;
			case IROpCode::storec:
				reference_to_instruction[node.store.a]->r.p = &node.store.dst->p;
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
					int reg = free_reg.allocate();
					*inst.a.pp = registers[reg];
					//a cast from a smaller type to a wider one cannot alias to the same register,
					//otherwise the wider elements of the result will overwrite the operands before they are converted
					//if this is the case, we make sure the registers do not alias
					if(is_widening_cast(inst.bc) && inst.r.p == *inst.a.pp) {
						int reg2 = free_reg.allocate();
						*inst.a.pp = registers[reg2];
						free_reg.free(reg);
					}
				}
			}
			if(inst.flags & TraceInst::REG_B) {
				if(*inst.b.pp == NULL) {
					int reg = free_reg.allocate();
					*inst.b.pp = registers[reg];
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
#define BINARY_OP(name,str, op, ...) \
				case TraceBC :: name ##dvv :  Map2VV< op < TDouble >, TRACE_VECTOR_WIDTH >::eval(state, *inst.a.dpp,*inst.b.dpp,(op < TDouble >::R*) inst.r.p); break; \
				case TraceBC :: name ##dvs :  Map2VS< op < TDouble >, TRACE_VECTOR_WIDTH >::eval(state, *inst.a.dpp,inst.b.d,(op < TDouble >::R*) inst.r.p); break; \
				case TraceBC :: name ##dsv :  Map2SV< op < TDouble >, TRACE_VECTOR_WIDTH >::eval(state, inst.a.d,*inst.b.dpp,(op < TDouble >::R*) inst.r.p); break; \
				case TraceBC :: name ##ivv :  Map2VV< op < TInteger >,TRACE_VECTOR_WIDTH >::eval(state, *inst.a.ipp,*inst.b.ipp,(op < TInteger >::R*) inst.r.p); break; \
				case TraceBC :: name ##ivs :  Map2VS< op < TInteger >,TRACE_VECTOR_WIDTH >::eval(state, *inst.a.ipp,inst.b.i,(op < TInteger >::R*) inst.r.p); break; \
				case TraceBC :: name ##isv :  Map2SV< op < TInteger >,TRACE_VECTOR_WIDTH >::eval(state, inst.a.i,*inst.b.ipp, (op < TInteger >::R*) inst.r.p); break;

#define LOGICAL_OP(name,str, op, ...) \
				case TraceBC :: name ##vv :  Map2VV< op < TLogical >, TRACE_VECTOR_WIDTH >::eval(state, *inst.a.lpp,*inst.b.lpp,(op < TDouble >::R*) inst.r.p); break; \
				case TraceBC :: name ##vs :  Map2VS< op < TLogical >, TRACE_VECTOR_WIDTH >::eval(state, *inst.a.lpp,inst.b.l,(op < TDouble >::R*) inst.r.p); break; \
				case TraceBC :: name ##sv :  Map2SV< op < TLogical >, TRACE_VECTOR_WIDTH >::eval(state, inst.a.l,*inst.b.lpp,(op < TDouble >::R*) inst.r.p); break; \

#define UNARY_OP(name,str, op, ...) \
				case TraceBC :: name##d :  Map1< op < TDouble >, TRACE_VECTOR_WIDTH >::eval(state, *inst.a.dpp,(op < TDouble >::R*) inst.r.p); break; \
				case TraceBC :: name##i :  Map1< op < TInteger >, TRACE_VECTOR_WIDTH >::eval(state, *inst.a.ipp,(op < TInteger >::R*) inst.r.p); break;

#define FOLD_OP(name, str, op, ...) \
				case TraceBC :: name##d :  *inst.r.dp = inst.b.d = FoldLeftT< op < TDouble >, TRACE_VECTOR_WIDTH >::eval(state, *inst.a.dpp, inst.b.d); break; \
				case TraceBC :: name##i :  *inst.r.ip = inst.b.i = FoldLeftT< op < TInteger >, TRACE_VECTOR_WIDTH >::eval(state, *inst.a.ipp, inst.b.i); break;

#define SCAN_OP(name, str, op, ...) \
				case TraceBC :: name##d :  inst.b.d = ScanLeftT< op < TDouble >, TRACE_VECTOR_WIDTH >::eval(state, *inst.a.dpp, inst.b.d, (op < TDouble >::R*) inst.r.p); break; \
				case TraceBC :: name##i :  inst.b.i = ScanLeftT< op < TInteger >, TRACE_VECTOR_WIDTH >::eval(state, *inst.a.ipp, inst.b.i, (op < TInteger >::R*) inst.r.p); break;

				BINARY_ARITH_MAP_BYTECODES(BINARY_OP)
				BINARY_LOGICAL_MAP_BYTECODES(LOGICAL_OP)
				BINARY_ORDINAL_MAP_BYTECODES(BINARY_OP)
				UNARY_ARITH_MAP_BYTECODES(UNARY_OP)
				ARITH_FOLD_BYTECODES(FOLD_OP)
				ARITH_SCAN_BYTECODES(SCAN_OP)

#undef BINARY_OP
#undef LOGICAL_OP
#undef UNARY_OP
#undef FOLD_OP
#undef SCAN_OP

				case TraceBC :: casti2d:  Map1< CastOp<Integer, Double> , TRACE_VECTOR_WIDTH>::eval(state, *inst.a.ipp , (double *)inst.r.p); break;
				case TraceBC :: castl2d: Map1< CastOp<Logical, Double> , TRACE_VECTOR_WIDTH>::eval(state, *inst.a.lpp , (double *)inst.r.p); break;
				case TraceBC :: castl2i:  Map1< CastOp<Logical, Integer> , TRACE_VECTOR_WIDTH>::eval(state, *inst.a.lpp , (int64_t *)inst.r.p); break;
				case TraceBC :: seq:  Sequence<TRACE_VECTOR_WIDTH>(i*inst.b.i+1, inst.b.i, (int64_t*)inst.r.p);
				}
			}
			for(size_t j = 0; j < n_incrementing_pointers_1; j++)
				(*incrementing_pointers_1[j]) += TRACE_VECTOR_WIDTH;
			for(size_t j = 0; j < n_incrementing_pointers_8; j++)
				(*incrementing_pointers_8[j]) += TRACE_VECTOR_WIDTH;
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
			a.pp = &reference_to_instruction[r]->r.p;
			assert(reference_to_instruction[r] != NULL);
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
		reference_to_instruction[node_ref] = &inst;
		Type::Enum operand_type = trace->nodes[node.binary.a].type;
		switch(operand_type) {
		default: /*fallthrough*/
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
		reference_to_instruction[node_ref] = &inst;
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
	void EmitFold(TraceBC::Enum oi, TraceBC::Enum od,
			      int64_t basei,    double based, IRef node_ref) {
			IRNode & node = trace->nodes[node_ref];
			bool a_is_reg;
			bool a_is_const;
			TraceInst & inst = insts[n_insts++];
			inst.a = GetOperand(node.unary.a,&a_is_const,&a_is_reg);
			assert(!a_is_const);
			inst.flags = TraceInst::REG_R;
			if(a_is_reg)
				inst.flags |= TraceInst::REG_A;
			reference_to_instruction[node_ref] = &inst;
			inst.r.p = NULL;

			switch(node.type) {
			case Type::Integer:
				inst.bc = oi;
				inst.b.i = basei;
				break;
			case Type::Double:
				inst.bc = od;
				inst.b.d = based;
				break;
			default:
				_error("unsupported type");
				break;
			}
		}
	void EmitSpecial(TraceBC::Enum op, IRef node_ref) {
		IRNode & node = trace->nodes[node_ref];
		TraceInst & inst = insts[n_insts++];
		inst.bc = op;
		inst.a.i = node.special.a;
		inst.b.i = node.special.b;
		inst.flags = TraceInst::REG_R;
		reference_to_instruction[node_ref] = &inst;
		inst.r.p = NULL;
	}
};


void Trace::Interpret(State & state) {
	InitializeOutputs(state);
	if(state.tracing.verbose)
		printf("executing trace:\n%s\n",toString(state).c_str());

	TraceInterpret trace_code(this);

	trace_code.compile();
	trace_code.execute(state);

	WriteOutputs(state);
}
