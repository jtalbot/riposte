#include "interpreter.h"
#include "vector.h"
#include "ops.h"
#include <sys/mman.h>
#include "assembler-x64.h"
#include <math.h>

#include "register_set.h"

#ifdef USE_AMD_LIBM
#include <amdlibm.h>
#endif


using namespace v8::internal;

#define SIMD_WIDTH (2 * sizeof(double))
#define CODE_BUFFER_SIZE (16 * 1024)

struct Constant {
	Constant() {}
	Constant(int64_t i)
	: i0(i), i1(i) {}
	Constant(uint64_t i)
	: u0(i), u1(i) {}
	Constant(double d)
	: d0(d), d1(d) {}
	Constant(void * f)
	: f0(f),f1(f)  {}
	Constant(uint8_t l)
	: u0((l << 8) + l), u1(0) {}

	Constant(int64_t ii0, int64_t ii1)
	: i0(ii0), i1(ii1) {}
	union {
		uint64_t u0;
		int64_t i0;
		double d0;
		void * f0;
	};
	union {
		 uint64_t u1;
		int64_t i1;
		double d1;
		void * f1;
	};
};

enum ConstantTableEntry {
	C_ABS_MASK = 0x0,
	C_NEG_MASK = 0x1,
	C_NOT_MASK = 0x2,
	C_SEQ_VEC  = 0x3,
	C_PACK_LOGICAL = 0x4,
	C_LOGICAL_MASK = 0x5,
	C_LNOT_MASK = 0x6,
	C_DOUBLE_ZERO = 0x7,
	C_FUNCTION_SPILL_SPACE = 0x8,
	C_REGISTER_SPILL_SPACE = 0x9,
	C_FIRST_TRACE_CONST = 0x19
};

static int make_executable(char * data, size_t size) {
	int64_t page = (int64_t)data & ~0x7FFF;
	int64_t psize = (int64_t)data + size - page;
	return mprotect((void*)page,psize, PROT_READ | PROT_WRITE | PROT_EXEC);
}

//scratch space that is reused across traces
struct TraceCodeBuffer {
	Constant constant_table[TRACE_MAX_NODES] __attribute__((aligned(16)));
	char code[CODE_BUFFER_SIZE] __attribute__((aligned(16)));
	TraceCodeBuffer() {
		//make the code executable
		if(0 != make_executable(code,CODE_BUFFER_SIZE)) {
			_error("mprotect failed.");
		}
		//fill in the constant table
		constant_table[C_ABS_MASK] = Constant((uint64_t)0x7FFFFFFFFFFFFFFFULL);
		constant_table[C_NEG_MASK] = Constant((uint64_t)0x8000000000000000ULL);
		constant_table[C_NOT_MASK] = Constant((uint64_t)0xFFFFFFFFFFFFFFFFULL);
		constant_table[C_SEQ_VEC] = Constant(1LL,2LL);
		constant_table[C_LOGICAL_MASK] = Constant((uint64_t)0x1);
		constant_table[C_PACK_LOGICAL] = Constant(0x1010101010100800,0x1010101010101010);
		constant_table[C_LNOT_MASK] = Constant((uint8_t)0x1);
		constant_table[C_DOUBLE_ZERO] = Constant(0.0);
	}
};

#include <xmmintrin.h>

double debug_print(int64_t offset, __m128 a) {
	union {
		__m128 c;
		double d[2];
		int64_t i[2];
	};
	c = a;
	printf("%d %f %f %ld %ld\n", (int) offset, d[0],d[1],i[0],i[1]);

	return d[0];
}

//some helper functions
static double sign_fn(double a) {
	return a > 0 ? 1 : (a < 0 ? -1 : 0);
}
static double imod(double aa, double bb) { //these are actually integers passed in xmm0 and xmm1
	union {
		int64_t ia;
		double da;
	};
	union {
		int64_t ib;
		double db;
	};
	da = aa;
	db = bb;
	ia = ia % ib;
	return da; //also return the value in xmm0
}
static double imul(double aa, double bb) { //these are actually integers passed in xmm0 and xmm1
	union {
		int64_t ia;
		double da;
	};
	union {
		int64_t ib;
		double db;
	};
	da = aa;
	db = bb;
	ia = ia * ib;
	return da; //also return the value in xmm0
}
static double idiv(double aa, double bb) { //these are actually integers passed in xmm0 and xmm1
	union {
		int64_t ia;
		double da;
	};
	union {
		int64_t ib;
		double db;
	};
	da = aa;
	db = bb;
	ia = ia / ib;
	return da; //also return the value in xmm0
}
static double casti2d(double aa) { //these are actually integers passed in xmm0 and xmm1
	union {
		int64_t ia;
		double da;
	};
	da = aa;
	return (double) ia; //also return the value in xmm0
}
static double castd2i(double aa) { //these are actually doubles passed in xmm0 and xmm1
	union {
		int64_t ia;
		double da;
	};
	ia = (int64_t) aa;
	return da; //also return the value in xmm0
}
static double iabs(double aa) { //these are actually integers passed in xmm0 and xmm1
	union {
		int64_t ia;
		double da;
	};
	da = aa;
	ia = (ia < 0) ?  -ia : ia;
	return da; //also return the value in xmm0
}



#define FOLD_SCAN_FN(name, type, op) \
static __m128d name (__m128d input, type * last) { \
	union { \
		__m128d in; \
		type i[2]; \
	}; \
	in = input; \
	*last = i[0] = *last op i[0] op i[1]; \
	return in; \
} \
static __m128d cum##name(__m128d input, type * last) { \
	union { \
		__m128d in; \
		type i[2]; \
	}; \
	in = input; \
	i[0] = *last op i[0]; \
	*last = i[1] = i[0] op i[1]; \
	return in; \
}

FOLD_SCAN_FN(prodi, int64_t, *)
FOLD_SCAN_FN(sumi , int64_t, +)
FOLD_SCAN_FN(prodd, double , *)
FOLD_SCAN_FN(sumd , double , +)

struct TraceJIT {
	TraceJIT(Trace * t)
	:  trace(t), asm_(t->code_buffer->code,CODE_BUFFER_SIZE), alloc(XMMRegister::kNumAllocatableRegisters), next_constant_slot(C_FIRST_TRACE_CONST) {}

	Trace * trace;
	IRNode * store_inst[TRACE_MAX_NODES];
	RegisterSet live_registers[TRACE_MAX_NODES];
	int8_t allocated_register[TRACE_MAX_NODES];
	Assembler asm_;
	RegisterAllocator alloc;


	Register constant_base; //holds pointer to Trace object
	Register vector_index; //index into long vector where the short vector begins
	Register load_addr; //holds address of input vectors
	Register vector_length; //holds length of long vector
	uint32_t next_constant_slot;

	//some hardware ops like > do not exist, requiring the use of < instead.  For simplicity, this needs to be done before register allocation
	void SimplifyNode(IRNode & node) {
		switch(node.op) {
		case IROpCode::gt: node.op = IROpCode::lt; std::swap(node.binary.a,node.binary.b); break;
		case IROpCode::ge: node.op = IROpCode::le; std::swap(node.binary.a,node.binary.b); break;
		default: /*pass*/ break;
		}
	}

	void Compile() {
		bzero(store_inst,sizeof(IRNode *) * trace->n_nodes);
		memset(allocated_register,-1,sizeof(char) * trace->n_nodes);
		//pass 1 register allocation
		for(IRef i = trace->n_nodes; i > 0; i--) {
			IRef ref = i - 1;
			IRNode & node = trace->nodes[ref];
			SimplifyNode(node); //emulate ops like > and >= which have no hardware equivalent by modifying the operand order
			switch(node.enc) {
			case IRNode::BINARY: {
				AllocateBinary(ref,node.binary.a,node.binary.b);
			} break;
			case IRNode::FOLD: /*fallthrough*/
			case IRNode::UNARY: {
				AllocateUnary(ref,node.unary.a);
			} break;
			case IRNode::LOADC: /*fallthrough*/
			case IRNode::LOADV: /*fallthrough*/
			case IRNode::SPECIAL: {
				AllocateNullary(ref);
			} break;
			case IRNode::STORE: {
				store_inst[node.store.a] = &node;
			} break;
			default:
				_error("unsupported op in reg allocation");
			}
		}

		//pass 2 instruction selection

		//registers are callee saved so that we can make external function calls without saving the registers the tight loop
		//we need to explicitly save and restore these on entrace and exit to the function
		constant_base = r12;
		vector_index = r13;
		load_addr = r14;
		vector_length = r15;
		
		asm_.push(constant_base);
		asm_.push(vector_index);
		asm_.push(load_addr);
		asm_.push(vector_length);
		asm_.push(rbx);

		asm_.movq(constant_base, &trace->code_buffer->constant_table[0]);
		asm_.xor_(vector_index,vector_index);
		asm_.movq(vector_length, trace->length);

		Label begin;

		asm_.bind(&begin);

		for(IRef i = 0; i < trace->n_nodes; i++) {
			IRef ref = i;
			IRNode & node = trace->nodes[i];

			//emit encoding specific code
			switch(node.enc) {
			case IRNode::BINARY: {
				//register allocation should ensure r = a op r does not occur, only r = a + b, r = r + b, and r = r + r
				assert(!reg(ref).is(reg(node.binary.b)) || reg(ref).is(reg(node.binary.a)));
				//we perform r = a; r = r op b, the mov is not emitted if r == a
				EmitMove(reg(ref),reg(node.binary.a));
			} break;
			case IRNode::UNARY: break;
			case IRNode::LOADC: {
				Constant c;
				switch(node.type) {
				case Type::Integer: c = Constant(node.loadc.i); break;
				case Type::Logical: c = Constant(node.loadc.l); break;
				case Type::Double:  c = Constant(node.loadc.d); break;
				default: _error("unexpected type");
				}
				asm_.movdqa(reg(ref),PushConstant(c));
			} break;
			case IRNode::LOADV: {
				if(Type::Logical == node.type)
					EmitLogicalLoad(reg(ref),node.loadv.p);
				else
					EmitVectorLoad(reg(ref),node.loadv.p);
			} break;
			case IRNode::STORE: {
				//stores are generated right after the value that is stored
			} break;
			case IRNode::SPECIAL: break;
			case IRNode::FOLD: break;
			default:
				_error("unsupported op");
			}

			//right now reg(ref) holds a if binary
			if(node.type == Type::Double) {
				switch(node.op) {
				case IROpCode::add: asm_.addpd(reg(ref),reg(node.binary.b)); break;
				case IROpCode::sub: asm_.subpd(reg(ref),reg(node.binary.b)); break;
				case IROpCode::mul: asm_.mulpd(reg(ref),reg(node.binary.b)); break;
				case IROpCode::div: asm_.divpd(reg(ref),reg(node.binary.b)); break;
				case IROpCode::sqrt: asm_.sqrtpd(reg(ref),reg(node.unary.a)); break;
				case IROpCode::round: asm_.roundpd(reg(ref),reg(node.unary.a),Assembler::kRoundToNearest); break;
				case IROpCode::floor: asm_.roundpd(reg(ref),reg(node.unary.a),Assembler::kRoundDown); break;
				case IROpCode::ceiling: asm_.roundpd(reg(ref),reg(node.unary.a),Assembler::kRoundUp); break;
				case IROpCode::trunc: asm_.roundpd(reg(ref),reg(node.unary.a),Assembler::kRoundToZero); break;

				case IROpCode::abs: {
					EmitMove(reg(ref),reg(node.unary.a));
					asm_.andpd(reg(ref), ConstantTable(C_ABS_MASK));
				} break;
				case IROpCode::neg: {
					EmitMove(reg(ref),reg(node.unary.a));
					asm_.xorpd(reg(ref), ConstantTable(C_NEG_MASK));
				} break;
				case IROpCode::pos: EmitMove(reg(ref),reg(node.unary.a)); break;
				case IROpCode::sum:  {
					assert(store_inst[ref] != NULL);
					IRNode & str = *store_inst[ref];
					str.store.dst->d = 0.0;
					Operand op = EncodeOperand(&str.store.dst->p);
					asm_.haddpd(reg(ref),reg(node.unary.a));
					asm_.addsd(reg(ref),op);
					asm_.movsd(op,reg(ref));
					store_inst[ref] = NULL;
				} break;
				case IROpCode::prod: { //this could be more efficient if we had an additional register ...
					assert(store_inst[ref] != NULL);
					IRNode & str = *store_inst[ref];
					str.store.dst->d = 1.0;
					Operand op = EncodeOperand(&str.store.dst->p);
					EmitMove(reg(ref),reg(node.unary.a));
					asm_.mulsd(reg(ref),op); //r[0] *= sum;
					asm_.movsd(op,reg(ref)); //sum = r[0];
					asm_.unpckhpd(reg(ref),reg(ref)); //r[0] = r[1]
					asm_.mulsd(reg(ref),op); //r[0] *= sum;
					asm_.movsd(op,reg(ref)); //sum = r[0];
					store_inst[ref] = NULL;
				} break;

#ifdef USE_AMD_LIBM
				case IROpCode::exp: EmitVectorizedUnaryFunction(ref,amd_vrd2_exp); break;
				case IROpCode::log: EmitVectorizedUnaryFunction(ref,amd_vrd2_log); break;
				case IROpCode::cos: EmitVectorizedUnaryFunction(ref,amd_vrd2_cos); break;
				case IROpCode::sin: EmitVectorizedUnaryFunction(ref,amd_vrd2_sin); break;
				case IROpCode::tan: EmitVectorizedUnaryFunction(ref,amd_vrd2_tan); break;
				case IROpCode::pow: EmitVectorizedBinaryFunction(ref,amd_vrd2_pow); break;

				case IROpCode::acos: EmitUnaryFunction(ref,amd_acos); break;
				case IROpCode::asin: EmitUnaryFunction(ref,amd_asin); break;
				case IROpCode::atan: EmitUnaryFunction(ref,amd_atan); break;
				case IROpCode::atan2: EmitBinaryFunction(ref,amd_atan2); break;
				case IROpCode::hypot: EmitBinaryFunction(ref,amd_hypot); break;
#else
				case IROpCode::exp: EmitUnaryFunction(ref,exp); break;
				case IROpCode::log: EmitUnaryFunction(ref,log); break;
				case IROpCode::cos: EmitUnaryFunction(ref,cos); break;
				case IROpCode::sin: EmitUnaryFunction(ref,sin); break;
				case IROpCode::tan: EmitUnaryFunction(ref,tan); break;
				case IROpCode::acos: EmitUnaryFunction(ref,acos); break;
				case IROpCode::asin: EmitUnaryFunction(ref,asin); break;
				case IROpCode::atan: EmitUnaryFunction(ref,atan); break;
				case IROpCode::pow: EmitBinaryFunction(ref,pow); break;
				case IROpCode::atan2: EmitBinaryFunction(ref,atan2); break;
				case IROpCode::hypot: EmitBinaryFunction(ref,hypot); break;
#endif
				case IROpCode::sign: EmitUnaryFunction(ref,sign_fn); break;
				case IROpCode::mod: EmitBinaryFunction(ref,Mod); break;
				case IROpCode::cast: {
					if(trace->nodes[node.unary.a].type == Type::Integer)
						EmitUnaryFunction(ref,casti2d); //it should be possible to inline this, but I can't find the convert packed quadword to packed double instruction
					else
						_error("NYI - castl2d");
				} break;
				case IROpCode::gather: {
					Constant c(node.unary.data);
					Operand base = PushConstant(c);
					asm_.movq(r8, base);
					asm_.movq(r9, reg(node.unary.a));
					asm_.movhlps(reg(ref), reg(node.unary.a));
					asm_.movq(r10, reg(ref));
					asm_.movlpd(reg(ref),Operand(r8,r9,times_8,0));
					asm_.movhpd(reg(ref),Operand(r8,r10,times_8,0));
				} break;
				//placeholder for now
				case IROpCode::cumsum:  EmitFoldFunction(ref,(void*)cumsumd,Constant(0.0)); break;
				case IROpCode::cumprod:  EmitFoldFunction(ref,(void*)cumprodd,Constant(1.0)); break;
				case IROpCode::signif:
				default:
					if(node.enc == IRNode::BINARY || node.enc == IRNode::UNARY)
						_error("unimplemented op");
					break;
				}
			} else if(node.type == Type::Integer) {
				switch(node.op) {
				case IROpCode::add: asm_.paddq(reg(ref),reg(node.binary.b)); break;
				case IROpCode::sub: asm_.psubq(reg(ref),reg(node.binary.b)); break;
				case IROpCode::mul: EmitBinaryFunction(ref,imul); break;
				case IROpCode::div: EmitBinaryFunction(ref,idiv); break;
				case IROpCode::abs: EmitUnaryFunction(ref,iabs);break; //this can be inlined using bit-whacking, but we would need an additional register
				                                                        // if(r < 0) f = ~0 else 0; r = r xor f; r -= f;
				case IROpCode::neg: {
					EmitMove(reg(ref),reg(node.unary.a));
					asm_.xorpd(reg(ref),ConstantTable(C_NOT_MASK)); //r = ~r
					asm_.psubq(reg(ref),ConstantTable(C_NOT_MASK)); //r -= -1
				} break;
				case IROpCode::pos: EmitMove(reg(ref),reg(node.unary.a)); break;
				case IROpCode::mod: EmitBinaryFunction(ref,imod); break;
				case IROpCode::seq: {
					asm_.movq(reg(ref),vector_index);
					asm_.unpcklpd(reg(ref),reg(ref));
					asm_.paddq(reg(ref),ConstantTable(C_SEQ_VEC));
				} break;
				case IROpCode::gather: {
					Constant c(node.unary.data);
					Operand base = PushConstant(c);
					asm_.movq(r8, base);
					asm_.movq(r9, reg(node.unary.a));
					asm_.movhlps(reg(ref), reg(node.unary.a));
					asm_.movq(r10, reg(ref));
					asm_.movlpd(reg(ref),Operand(r8,r9,times_8,0));
					asm_.movhpd(reg(ref),Operand(r8,r10,times_8,0));
				} break;
				//placeholder for now
				case IROpCode::sum:  EmitFoldFunction(ref,(void*)sumi,Constant((int64_t)0LL)); break;
				case IROpCode::cumsum:  EmitFoldFunction(ref,(void*)cumsumi,Constant((int64_t)0LL)); break;
				case IROpCode::prod:  EmitFoldFunction(ref,(void*)prodi,Constant((int64_t)1LL)); break;
				case IROpCode::cumprod:  EmitFoldFunction(ref,(void*)cumprodi,Constant((int64_t)1LL)); break;
				case IROpCode::cast: {
					if(trace->nodes[node.unary.a].type == Type::Double)
						EmitUnaryFunction(ref,(void*)castd2i); 
					else
						_error("NYI - castl2i");
				} break;
				default:
					if(node.enc == IRNode::BINARY || node.enc == IRNode::UNARY)
						_error("unimplemented op");
					break;
				}
			} else if(node.type == Type::Logical) {
				switch(node.op) {
				case IROpCode::eq: EmitCompare(ref,Assembler::kEQ); break;
				case IROpCode::lt: EmitCompare(ref,Assembler::kLT); break;
				case IROpCode::le: EmitCompare(ref,Assembler::kLE); break;
				case IROpCode::neq: EmitCompare(ref,Assembler::kNEQ); break;
				case IROpCode::land: asm_.andpd(reg(ref),reg(node.binary.b)); break;
				case IROpCode::lor: asm_.orpd(reg(ref),reg(node.binary.b)); break;
				case IROpCode::lnot: {
					EmitMove(reg(ref),reg(node.unary.a));
					asm_.xorpd(reg(ref),ConstantTable(C_LNOT_MASK));
				} break;
				case IROpCode::cast: {
					Type::Enum operand_type = trace->nodes[node.unary.a].type;
					if(operand_type == Type::Double) {
						asm_.cmppd(reg(ref),ConstantTable(C_DOUBLE_ZERO),Assembler::kNEQ);
						EmitComparisonToLogical(reg(ref));
					} else {
						_error("NYI - casti2l");
					}
				} break;
				default:
					if(node.enc == IRNode::BINARY || node.enc == IRNode::UNARY)
						_error("unimplemented op");
					break;
				}
			} else {
				_error("type not supported");
			}

			/*
			switch(node.enc) {
			case IRNode::LOADV:
			case IRNode::LOADC:
			case IRNode::BINARY:
			case IRNode::UNARY:
				EmitDebugPrintResult(ref);
				break;
			case IRNode::STORE:
			case IRNode::SPECIAL:
			case IRNode::FOLD:
				break;
			}
			*/

			if(store_inst[ref] != NULL) {
				IRNode & str = *store_inst[ref];
				if(str.op == IROpCode::storev) {
					if(Type::Logical == str.type)
						EmitLogicalStore(str.store.dst->p,reg(str.store.a));
					else
						EmitVectorStore(str.store.dst->p,reg(str.store.a));
				} else {
					Operand op = EncodeOperand(&str.store.dst->p);
					asm_.movsd(op,reg(str.store.a));
				}
			}
		}

		asm_.addq(vector_index, Immediate(2));
		asm_.cmpq(vector_index,vector_length);
		asm_.j(less,&begin);

		asm_.pop(rbx);
		asm_.pop(vector_length);
		asm_.pop(load_addr);
		asm_.pop(vector_index);
		asm_.pop(constant_base);
		asm_.ret(0);
	}
	XMMRegister reg(IRef r) {
		assert(allocated_register[r] >= 0);
		return XMMRegister::FromAllocationIndex(allocated_register[r]);
	}
	void EmitMove(XMMRegister dst, XMMRegister src) {
		if(!dst.is(src)) {
			asm_.movapd(dst,src);
		}
	}
	Operand ConstantTable(int entry) {
		return Operand(constant_base,entry * sizeof(Constant));
	}

	Operand PushConstant(const Constant& data) {
		return ConstantTable(PushConstantOffset(data));
	}
	uint64_t PushConstantOffset(const Constant& data) {
		uint32_t offset = next_constant_slot;
		trace->code_buffer->constant_table[offset] = data;
		next_constant_slot++;
		return offset;
	}
	//slow path for Folds that don't have inline assembly
	void EmitFoldFunction(IRef ref, void * fn, const Constant& identity) {
		IRNode & node = trace->nodes[ref];
		uint64_t offset = PushConstantOffset(identity);
		SaveRegisters(live_registers[ref]);
		EmitMove(xmm0,reg(node.unary.a));
		asm_.movq(rdi,(void*)&trace->code_buffer->constant_table[offset]);
		EmitCall(fn);
		EmitMove(reg(ref),xmm0);
		RestoreRegisters(live_registers[ref]);
	}
	void EmitVectorLoad(XMMRegister dst, void * src) {
		asm_.movdqa(dst,EncodeOperand(src,vector_index,times_8));
	}
	void EmitVectorStore(void * dst,  XMMRegister src) {
		asm_.movdqa(EncodeOperand(dst,vector_index,times_8),src);
	}
	void EmitLogicalLoad(XMMRegister dst, void * src) {
		asm_.movss(dst,EncodeOperand(src,vector_index,times_1));
	}
	void EmitLogicalStore(void * dst,  XMMRegister src) {
		asm_.movq(rbx,src);
		asm_.movw(EncodeOperand(dst,vector_index,times_1),rbx);
	}

	Operand EncodeOperand(void * dst, Register idx, ScaleFactor scale) {
		int64_t diff = (int64_t)dst - (int64_t)trace->code_buffer->constant_table;
		if(is_int32(diff)) {
			return Operand(constant_base,idx,scale,diff);
		} else {
			asm_.movq(load_addr,dst);
			return Operand(load_addr,idx,scale,0);
		}
	}
	Operand EncodeOperand(void * src) {
		int64_t diff = (int64_t)src - (int64_t)trace->code_buffer->constant_table;
		if(is_int32(diff)) {
			return Operand(constant_base,diff);
		} else {
			asm_.movq(load_addr,src);
			return Operand(load_addr,0);
		}
	}
	IRef AllocateNullary(IRef ref, bool p = true) {
		if(allocated_register[ref] < 0) { //this instruction is dead, for now we just emit it anyway
			if(!alloc.allocate(&allocated_register[ref]))
				_error("exceeded available registers");
		}
		int r = allocated_register[ref];
		alloc.free(r);
		//we want to know the registers live out of this op, not including value that this op defines
		live_registers[ref] = alloc.live_registers();
		if(p) {
			//printf("%d = _, %d = _ \n",(int)allocated_register[ref],(int)ref);
		}
		return r;
	}
	IRef AllocateUnary(IRef ref, IRef a, bool p = true) {
		IRef r = AllocateNullary(ref,false);
		if(allocated_register[a] < 0) {
			//try to allocated register a and r to the same location,  this avoids moves in binary instructions
			if(!alloc.allocate(r,&allocated_register[a]))
				_error("exceeded available registers");
		}
		if(p) {
			//printf("%d = %d op, %d = %d op\n",(int)allocated_register[ref],(int)allocated_register[a],(int)ref,(int)a);
		}
		return r;
	}
	IRef AllocateBinary(IRef ref, IRef a, IRef b) {
		IRef r = AllocateUnary(ref,a,false);
		if(allocated_register[b] < 0) {
			RegisterSet mask = ~(1 << allocated_register[ref]);
			//we avoid allocating registers such that r = a op r
			//occurs
			if(!alloc.allocateWithMask(mask,&allocated_register[b])) {
				_error("exceeded available registers");
			}
			assert(allocated_register[ref] != allocated_register[b]);
		}


		//printf("%d = %d op %d, %d = %d op %d\n",(int)allocated_register[ref],(int)allocated_register[a],(int)allocated_register[b],(int)ref,(int)a,(int)b);

		//we need to avoid binary statements of the form r = a op r, a != r because these are hard to code gen with 2-op codes
		assert(!reg(ref).is(reg(b)) || reg(ref).is(reg(a)));
		//proof:
		// case: b is not live out of this stmt and b != a -> we allocated b s.t. r != b
		// case: b is not live out this stmt and b == a -> either this stmt is r = a + a or r = r + r, either of which is allowed in codegen
		// case: b is live out of this stmt -> b and r are live at the same time so b != r

		return r;
	}

	void EmitCall(void * fn) {
		int64_t diff = (int64_t)(trace->code_buffer + asm_.pc_offset() + 5 - (int64_t) fn);
		if(is_int32(diff)) {
			asm_.call((byte*)fn);
		} else {
			asm_.call(PushConstant(Constant(fn)));
		}
	}
	void EmitVectorizedUnaryFunction(IRef ref, __m128d (*fn)(__m128d)) {
		EmitVectorizedUnaryFunction(ref,(void*)fn);
	}
	void EmitVectorizedUnaryFunction(IRef ref, void * fn) {

		SaveRegisters(live_registers[ref]);

		EmitMove(xmm0,reg(trace->nodes[ref].unary.a));
		EmitCall(fn);
		EmitMove(reg(ref),xmm0);

		RestoreRegisters(live_registers[ref]);

	}

	void EmitVectorizedBinaryFunction(IRef ref, __m128d (*fn)(__m128d,__m128d)) {
		EmitBinaryFunction(ref,(void*)fn);
	}
	void EmitVectorizedBinaryFunction(IRef ref, void * fn) {

		SaveRegisters(live_registers[ref]);

		XMMRegister ar = reg(trace->nodes[ref].binary.a);
		XMMRegister br = reg(trace->nodes[ref].binary.b);

		if(ar.is(xmm1)) {
			if(br.is(xmm0)) { //swap
				EmitMove(xmm2,ar);
				EmitMove(xmm1,br);
				EmitMove(xmm0,xmm2);
			} else {
				EmitMove(xmm0,ar);
				EmitMove(xmm1,br);
			}
		} else {
			if(br.is(xmm0)) {
				EmitMove(xmm1,br);
				EmitMove(xmm0,ar);
			} else {
				EmitMove(xmm0,ar);
				EmitMove(xmm1,br);
			}
		}
		EmitCall(fn);
		EmitMove(reg(ref),xmm0);

		RestoreRegisters(live_registers[ref]);

	}
	void EmitUnaryFunction(IRef ref, double (*fn)(double)) {
		EmitUnaryFunction(ref,(void*)fn);
	}
	void EmitUnaryFunction(IRef ref, void * fn) {

		SaveRegisters(live_registers[ref]);

		EmitMove(xmm0,reg(trace->nodes[ref].unary.a));
		asm_.movapd(xmm1,xmm0);
		asm_.unpckhpd(xmm1,xmm1);
		asm_.movq(load_addr,xmm1);//we need the high value for the second call
		EmitCall(fn);
		asm_.movq(rbx,xmm0);
		asm_.movq(xmm0,load_addr);
		EmitCall(fn);
		asm_.movq(xmm1,rbx);
		asm_.unpcklpd(xmm1,xmm0);
		EmitMove(reg(ref),xmm1);

		RestoreRegisters(live_registers[ref]);
	}
	void EmitBinaryFunction(IRef ref, double (*fn)(double,double)) {
		EmitBinaryFunction(ref,(void*)fn);
	}
	void EmitBinaryFunction(IRef ref, void * fn) {
		//this isn't the most optimized way to do this but it works for now
		SaveRegisters(live_registers[ref]);
		IRNode & node = trace->nodes[ref];
		uint64_t spill_start = C_FUNCTION_SPILL_SPACE * sizeof(Constant);
		asm_.movdqa(Operand(constant_base,spill_start),reg(node.binary.a));
		asm_.movdqa(Operand(constant_base,spill_start + 0x10),reg(node.binary.b));
		asm_.movsd(xmm0,Operand(constant_base,spill_start));
		asm_.movsd(xmm1,Operand(constant_base,spill_start + 0x10));
		EmitCall(fn);
		asm_.movsd(Operand(constant_base,spill_start),xmm0);
		asm_.movsd(xmm0,Operand(constant_base,spill_start + 0x8));
		asm_.movsd(xmm1,Operand(constant_base,spill_start + 0x18));
		EmitCall(fn);
		asm_.movsd(Operand(constant_base,spill_start + 0x8),xmm0);
		asm_.movdqa(reg(ref),Operand(constant_base,spill_start));
		RestoreRegisters(live_registers[ref]);
	}

	void EmitDebugPrintResult(IRef i) {
		RegisterSet regs = live_registers[i];
		regs &= ~(1 << allocated_register[i]);

		SaveRegisters(regs);
		EmitMove(xmm0,reg(i));
		asm_.movq(rdi,vector_index);
		EmitCall((void*) debug_print);
		RestoreRegisters(regs);
	}
	void SaveRegisters(RegisterSet regs) {
		uint32_t offset = C_REGISTER_SPILL_SPACE * sizeof(Constant);
		for(RegisterIterator it(regs); !it.done(); it.next()) {
			asm_.movdqa(Operand(constant_base,offset),XMMRegister::FromAllocationIndex(it.value()));
			offset += SIMD_WIDTH;
		}
	}

	void RestoreRegisters(RegisterSet regs) {
		uint32_t offset = C_REGISTER_SPILL_SPACE * sizeof(Constant);
		for(RegisterIterator it(regs); !it.done(); it.next()) {
			asm_.movdqa(XMMRegister::FromAllocationIndex(it.value()),Operand(constant_base,offset));
			offset += SIMD_WIDTH;
		}
	}

	void EmitComparisonToLogical(XMMRegister r) {
		asm_.andpd(r,ConstantTable(C_LOGICAL_MASK));
		asm_.pshufb(r,ConstantTable(C_PACK_LOGICAL));
	}
	void EmitCompare(IRef ref, Assembler::ComparisonType typ) {
		IRNode & node = trace->nodes[ref];
		if(Type::Double == trace->nodes[node.binary.a].type) {
			asm_.cmppd(reg(ref),reg(node.binary.b),typ);
			EmitComparisonToLogical(reg(ref));
		} else {
			_error("NYI - integer compare");
		}
	}
	void Execute(State & state) {
		typedef void (*fn) (void);
		fn trace_code = (fn) trace->code_buffer->code;
		if(state.tracing.verbose) {
			timespec begin;
			get_time(begin);
			trace_code();
			double s = time_elapsed(begin) / trace->length * 1024.0 * 1024.0 * 1024.0;
			printf("trace elapsed %fns\n",s);
		} else {
			trace_code();
		}
	}
};

void Trace::JIT(State & state) {
	InitializeOutputs(state);
	if(state.tracing.verbose)
		printf("executing trace:\n%s\n",toString(state).c_str());

	if(code_buffer == NULL) { //since it is expensive to reallocate this, we reuse it across traces
		code_buffer = new TraceCodeBuffer();
	}

	TraceJIT trace_code(this);

	trace_code.Compile();
	trace_code.Execute(state);

	WriteOutputs(state);
}
