#include "interpreter.h"
#include "vector.h"
#include "ops.h"
#include <sys/mman.h>
#include "assembler-x64.h"
#include <math.h>

#include <pthread.h>

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
	Constant(char l)
	: i0(l), i1(l) {}

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
	C_PACK_LOGICAL = 0x3,
	C_SEQ_VEC  = 0x5,
	C_DOUBLE_ZERO = 0x6,
	C_DOUBLE_ONE = 0x7,
	C_FIRST_TRACE_CONST = 0x8
};

static int make_executable(char * data, size_t size) {
	int64_t page = (int64_t)data & ~0x7FFF;
	int64_t psize = (int64_t)data + size - page;
	return mprotect((void*)page,psize, PROT_READ | PROT_WRITE | PROT_EXEC);
}

//scratch space that is reused across traces
struct TraceCodeBuffer {
	Constant constant_table[128] __attribute__((aligned(16)));
	char code[CODE_BUFFER_SIZE] __attribute__((aligned(16)));
	uint64_t outer_loop;
	TraceCodeBuffer() {
		//make the code executable
		if(0 != make_executable(code,CODE_BUFFER_SIZE)) {
			_error("mprotect failed.");
		}
		//fill in the constant table
		constant_table[C_ABS_MASK] = Constant((uint64_t)0x7FFFFFFFFFFFFFFFULL);
		constant_table[C_NEG_MASK] = Constant((uint64_t)0x8000000000000000ULL);
		constant_table[C_NOT_MASK] = Constant((uint64_t)0xFFFFFFFFFFFFFFFFULL);
		constant_table[C_PACK_LOGICAL] = Constant(0x0706050403020800,0x0F0E0D0C0B0A0902);
		constant_table[C_SEQ_VEC] = Constant(1LL,2LL);
		constant_table[C_DOUBLE_ZERO] = Constant(0.0, 0.0);
		constant_table[C_DOUBLE_ONE] = Constant(1.0, 1.0);
		outer_loop = 0;
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
static double castl2d(double aa) { //these are actually logicals passed in xmm0 and xmm1
	union {
		double da;
		int64_t a;
	};
	da = aa;
	if(Logical::isTrue((char)a))
		return 1.0;
	else if(Logical::isFalse((char)a))
		return 0.0;
	else
		return Double::NAelement;
}
static double castl2i(double aa) { //these are actually logicals passed in xmm0 and xmm1
	union {
		double da;
		int64_t a;
	};
	da = aa;
	if(Logical::isTrue((char)a))
		a = 1;
	else if(Logical::isFalse((char)a))
		a = 0;
	else
		a = Integer::NAelement;
	return da;
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

static void store_conditional(__m128d input, __m128i mask, Value* out) {
	union { 
		__m128i ma; 
		int64_t m[2]; 
	}; 
	union { 
		__m128d in; 
		double i[2]; 
	};
	ma = mask; 
	in = input;
	if(m[0] == -1) 
		((double*)(out->p))[out->length++] = i[0];
	if(m[1] == -1) 
		((double*)(out->p))[out->length++] = i[1];
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
	:  trace(t), asm_(t->code_buffer->code,CODE_BUFFER_SIZE), alloc(XMMRegister::kNumAllocatableRegisters), next_constant_slot(C_FIRST_TRACE_CONST) {
		store_inst = new (PointerFreeGC) IRNode*[trace->nodes.size()];
		live_registers = new (PointerFreeGC) RegisterSet[trace->nodes.size()];
		allocated_register = new (PointerFreeGC) int8_t[trace->nodes.size()];
		// allocate xmm0 as a temporary register (needed for blend)
		int8_t temp;
		alloc.allocate(0,&temp);
	}

	Trace * trace;
	IRNode ** store_inst;
	RegisterSet* live_registers;
	int8_t* allocated_register;
	Assembler asm_;
	RegisterAllocator alloc;

	Register thread_index;
	Register constant_base; //holds pointer to Trace object
	Register vector_index; //index into long vector where the short vector begins
	Register load_addr; //holds address of input vectors
	Register vector_length; //holds length of long vector
	uint32_t next_constant_slot;

	void RegisterAllocate() {
		//pass 1 register allocation
		for(IRef i = trace->nodes.size(); i > 0; i--) {
			IRef ref = i - 1;
			IRNode & node = trace->nodes[ref];
			switch(node.enc) {
			case IRNode::IFELSE: {
				AllocateTrinary(ref, node.ifelse.no, node.ifelse.yes, node.ifelse.cond);
			} break;
			case IRNode::BINARY: {
				AllocateBinary(ref,node.binary.a,node.binary.b);
			} break;
			case IRNode::FOLD: {
				if(node.fold.mask < 0)
					AllocateBinary(ref, node.fold.a, -node.fold.mask);
				else
					AllocateUnary(ref, node.fold.a);
			} break;
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
			case IRNode::NOP: {
				/* nothing */
			} break;
			default:
				_error("unsupported op in reg allocation");
			}
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
		if(trace->nodes[ref].length < 0) {
			IRef mask = -trace->nodes[ref].length;
			if(allocated_register[mask] < 0) {
				if(!alloc.allocate(&allocated_register[mask]))
					_error("exceeded available registers");
			}
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

		//we need to avoid binary threadments of the form r = a op r, a != r because these are hard to code gen with 2-op codes
		assert(!reg(ref).is(reg(b)) || reg(ref).is(reg(a)));
		//proof:
		// case: b is not live out of this stmt and b != a -> we allocated b s.t. r != b
		// case: b is not live out this stmt and b == a -> either this stmt is r = a + a or r = r + r, either of which is allowed in codegen
		// case: b is live out of this stmt -> b and r are live at the same time so b != r

		return r;
	}
	IRef AllocateTrinary(IRef ref, IRef a, IRef b, IRef c) {
		IRef r = AllocateBinary(ref, a, b);
		if(allocated_register[c] < 0) {
			if(!alloc.allocate(&allocated_register[c]))
				_error("exceeded available registers");
		}
		return r;
	}

	void InstructionSelection() {
		//pass 2 instruction selection

		//registers are callee saved so that we can make external function calls without saving the registers the tight loop
		//we need to explicitly save and restore these on entrace and exit to the function
		thread_index = rbp;
		constant_base = r12;
		vector_index = r13;
		load_addr = r14;
		vector_length = r15;
		
		asm_.push(thread_index);
		asm_.push(constant_base);
		asm_.push(vector_index);
		asm_.push(load_addr);
		asm_.push(vector_length);
		asm_.push(rbx);
		asm_.subq(rsp, Immediate(0x8));

		asm_.movq(thread_index, rdi);
		asm_.movq(constant_base, &trace->code_buffer->constant_table[0]);
		asm_.movq(vector_index, rsi);
		asm_.movq(vector_length, rdx);

		Label begin;

		asm_.bind(&begin);

		for(IRef i = 0; i < trace->nodes.size(); i++) {
			IRef ref = i;
			IRNode & node = trace->nodes[i];
			if(node.length > 1) trace->code_buffer->outer_loop = node.length;

			switch(node.op) {

			case IROpCode::loadc: {
				Constant c;
				switch(node.type) {
					case Type::Integer: c = Constant(node.loadc.i); break;
					case Type::Logical: c = Constant(node.loadc.l); break;
					case Type::Double:  c = Constant(node.loadc.d); break;
					default: _error("unexpected type");
				}
				asm_.movdqa(reg(ref),PushConstant(c));
			} break;
			case IROpCode::loadv: {
				if(node.isLogical())
					asm_.pmovsxbq(reg(ref), EncodeOperand(node.loadv.src.p, vector_index, times_1));
				else
					asm_.movdqa(reg(ref),EncodeOperand(node.loadv.src.p,vector_index,times_8));
			} break;
			case IROpCode::storec:
			case IROpCode::storev: {
				// as an optimization, stores are generated right after the value that is stored
				// this moves the store as far forward as possible, reducing register pressure.
			} break;

			case IROpCode::add: {
				if(node.isDouble()) 	asm_.addpd(Move(ref, node.binary.a),reg(node.binary.b)); 
				else		 	asm_.paddq(Move(ref, node.binary.a),reg(node.binary.b));
			} break;
			case IROpCode::sub: {
				if(node.isDouble()) 	asm_.subpd(Move(ref, node.binary.a),reg(node.binary.b)); 
				else		 	asm_.psubq(Move(ref, node.binary.a),reg(node.binary.b));
			} break;
			case IROpCode::mul: {
				if(node.isDouble()) 	asm_.mulpd(Move(ref, node.binary.a),reg(node.binary.b)); 
				else			EmitBinaryFunction(ref,imul);
			} break;
			case IROpCode::div: 	asm_.divpd(Move(ref, node.binary.a),reg(node.binary.b)); break;
			case IROpCode::idiv: {
				if(node.isDouble()) {	
					asm_.divpd(Move(ref, node.binary.a),reg(node.binary.b)); 
					asm_.roundpd(Move(ref, node.binary.a), reg(ref), Assembler::kRoundDown);	
				} else	{ 
					EmitBinaryFunction(ref,idiv);
				}
			} break;
			case IROpCode::sqrt: 	asm_.sqrtpd(reg(ref),reg(node.unary.a)); break;
			case IROpCode::round:	asm_.roundpd(reg(ref),reg(node.unary.a), Assembler::kRoundToNearest); break;
			case IROpCode::floor: 	asm_.roundpd(reg(ref),reg(node.unary.a), Assembler::kRoundDown); break;
			case IROpCode::ceiling:	asm_.roundpd(reg(ref),reg(node.unary.a), Assembler::kRoundUp); break;
			case IROpCode::trunc: 	asm_.roundpd(reg(ref),reg(node.unary.a), Assembler::kRoundToZero); break;
			case IROpCode::abs: {
				if(node.isDouble()) {
					EmitMove(reg(ref),reg(node.unary.a));
					asm_.andpd(reg(ref), ConstantTable(C_ABS_MASK));
				} else {			
					//this could be inlined using bit-whacking, but we would need an additional register
					// if(r < 0) f = ~0 else 0; r = r xor f; r -= f;
					EmitUnaryFunction(ref,iabs); 
				}
			} break;
			case IROpCode::neg: {
				if(node.isDouble()) {	
					EmitMove(reg(ref),reg(node.unary.a));
					asm_.xorpd(reg(ref), ConstantTable(C_NEG_MASK));	
				} else {
					EmitMove(reg(ref),reg(node.unary.a));
					asm_.xorpd(reg(ref),ConstantTable(C_NOT_MASK)); //r = ~r
					asm_.psubq(reg(ref),ConstantTable(C_NOT_MASK)); //r -= -1	
				}
			} break;
			case IROpCode::pos: EmitMove(reg(ref), reg(node.unary.a)); break;
#ifdef USE_AMD_LIBM
			case IROpCode::exp: 	EmitVectorizedUnaryFunction(ref,amd_vrd2_exp); break;
			case IROpCode::log: 	EmitVectorizedUnaryFunction(ref,amd_vrd2_log); break;
			case IROpCode::cos: 	EmitVectorizedUnaryFunction(ref,amd_vrd2_cos); break;
			case IROpCode::sin: 	EmitVectorizedUnaryFunction(ref,amd_vrd2_sin); break;
			case IROpCode::tan: 	EmitVectorizedUnaryFunction(ref,amd_vrd2_tan); break;
			case IROpCode::pow: 	EmitVectorizedBinaryFunction(ref,amd_vrd2_pow); break;

			case IROpCode::acos: 	EmitUnaryFunction(ref,amd_acos); break;
			case IROpCode::asin: 	EmitUnaryFunction(ref,amd_asin); break;
			case IROpCode::atan: 	EmitUnaryFunction(ref,amd_atan); break;
			case IROpCode::atan2: 	EmitBinaryFunction(ref,amd_atan2); break;
			case IROpCode::hypot: 	EmitBinaryFunction(ref,amd_hypot); break;
#else
			case IROpCode::exp: 	EmitUnaryFunction(ref,exp); break;
			case IROpCode::log: 	EmitUnaryFunction(ref,log); break;
			case IROpCode::cos: 	EmitUnaryFunction(ref,cos); break;
			case IROpCode::sin: 	EmitUnaryFunction(ref,sin); break;
			case IROpCode::tan: 	EmitUnaryFunction(ref,tan); break;
			case IROpCode::acos: 	EmitUnaryFunction(ref,acos); break;
			case IROpCode::asin: 	EmitUnaryFunction(ref,asin); break;
			case IROpCode::atan: 	EmitUnaryFunction(ref,atan); break;
			case IROpCode::pow: 	EmitBinaryFunction(ref,pow); break;
			case IROpCode::atan2: 	EmitBinaryFunction(ref,atan2); break;
			case IROpCode::hypot: 	EmitBinaryFunction(ref,hypot); break;
#endif
			case IROpCode::sign: 	EmitUnaryFunction(ref,sign_fn); break;
			case IROpCode::mod: {
				if(node.isDouble()) {
					EmitBinaryFunction(ref, Mod); 
				} else {
					EmitBinaryFunction(ref, imod);
				}
			} break;
			
			case IROpCode::eq: EmitCompare(ref,Assembler::kEQ); break;
			case IROpCode::lt: EmitCompare(ref,Assembler::kLT); break;
			case IROpCode::le: EmitCompare(ref,Assembler::kLE); break;
			case IROpCode::neq: EmitCompare(ref,Assembler::kNEQ); break;

			case IROpCode::land: asm_.pand(reg(ref),reg(node.binary.b)); break;
			case IROpCode::lor: asm_.por(reg(ref),reg(node.binary.b)); break;
			case IROpCode::lnot: asm_.xorpd(Move(ref, node.unary.a), ConstantTable(C_NOT_MASK)); break;
			
			case IROpCode::cast: {
				if(node.type == trace->nodes[node.unary.a].type)
					EmitMove(reg(ref), reg(node.unary.a));
				else if(node.isDouble() && trace->nodes[node.unary.a].isInteger())
					EmitUnaryFunction(ref, casti2d);
				else if(node.isDouble() && trace->nodes[node.unary.a].isLogical())
					EmitUnaryFunction(ref, castl2d);
				else if(node.isInteger() && trace->nodes[node.unary.a].isDouble())
					EmitUnaryFunction(ref, castd2i);
				else if(node.isInteger() && trace->nodes[node.unary.a].isLogical())
					EmitUnaryFunction(ref, castl2i);
				else if(node.isLogical() && trace->nodes[node.unary.a].isDouble()) {
					asm_.cmppd(reg(ref),ConstantTable(C_DOUBLE_ZERO),Assembler::kNEQ);
					// need to propogate NAs here
				}
				else _error("Unimplemented cast");
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
			case IROpCode::seq: {
				if(node.isDouble()) {
					_error("No double seq yet in JIT");
				} else {
					asm_.movq(reg(ref),vector_index);
					asm_.unpcklpd(reg(ref),reg(ref));
					asm_.paddq(reg(ref),ConstantTable(C_SEQ_VEC));
				}
			} break;

			case IROpCode::sum:  {
				if(node.isDouble()) {
					assert(store_inst[ref] != NULL);
					IRNode & str = *store_inst[ref];
					for(uint64_t i = 0; i < 128; i++)
						((double*)str.store.dst.p)[i] = 0.0;
					Operand op = EncodeOperand(str.store.dst.p, thread_index, times_8);
					if(node.fold.mask >= 0) {
						asm_.addpd(Move(ref, node.unary.a), op);
					} else {
						IRef mask = -node.fold.mask;
						EmitMove(xmm0, reg(mask));
						asm_.xorpd(xmm0, ConstantTable(C_NOT_MASK));
						asm_.blendvpd(Move(ref, node.unary.a), ConstantTable(C_DOUBLE_ZERO));
						asm_.addpd(reg(ref), op);
					}
					asm_.movdqa(op,reg(ref));
				} 
				else {
					EmitFoldFunction(ref,(void*)sumi,Constant((int64_t)0LL)); break;
				}
			} break;

			case IROpCode::prod: { 
				if(node.isDouble()) {
					assert(store_inst[ref] != NULL);
					IRNode & str = *store_inst[ref];
					for(uint64_t i = 0; i < 128; i++)
						((double*)str.store.dst.p)[i] = 1.0;
					Operand op = EncodeOperand(str.store.dst.p, thread_index, times_8);
					asm_.mulpd(reg(ref),op);
					asm_.movdqa(op,reg(ref));
				}
				else {
					EmitFoldFunction(ref,(void*)prodi,Constant((int64_t)0LL)); break;
				}
			} break;
	
			//placeholder for now
			case IROpCode::cumsum: {
				if(node.isDouble()) 
					EmitFoldFunction(ref,(void*)cumsumd,Constant(0.0)); 
				else
					EmitFoldFunction(ref,(void*)cumsumi,Constant((int64_t)0LL));
			} break;
			case IROpCode::cumprod: {
				if(node.isDouble()) 
					EmitFoldFunction(ref,(void*)cumprodd,Constant(1.0)); 
				else
					EmitFoldFunction(ref,(void*)cumprodi,Constant((int64_t)1LL));
			} break;

			case IROpCode::nop:
			/* nothing */
			break;

			case IROpCode::filter:
				Move(ref, node.binary.a);
			/* nothing */
			break;

			case IROpCode::ifelse:
				EmitMove(xmm0, reg(node.ifelse.cond));
				asm_.blendvpd(Move(ref, node.ifelse.no), reg(node.ifelse.yes));
			break;

			case IROpCode::signif:
			default:	_error("unimplemented op"); break;
		
			}

			if(store_inst[ref] != NULL && store_inst[ref]->op == IROpCode::storev) {
				IRNode & str = *store_inst[ref];
				if(Type::Logical == str.type)
					EmitLogicalStore(ref, str.store.dst,reg(str.store.a),str.length);
				else
					EmitVectorStore(ref, str.store.dst,reg(str.store.a), str.length);
				/*} else {
					Operand op = EncodeOperand(&str.store.dst.p);
					asm_.movsd(op,reg(str.store.a));
				}*/
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
				break;
			}
		*/	

		}

		asm_.addq(vector_index, Immediate(2));
		asm_.cmpq(vector_index,vector_length);
		asm_.j(less,&begin);

		asm_.addq(rsp, Immediate(0x8));
		asm_.pop(rbx);
		asm_.pop(vector_length);
		asm_.pop(load_addr);
		asm_.pop(vector_index);
		asm_.pop(constant_base);
		asm_.pop(thread_index);
		asm_.ret(0);
	}
	XMMRegister reg(IRef r) {
		assert(allocated_register[r] >= 0);
		return XMMRegister::FromAllocationIndex(allocated_register[r]);
	}
	XMMRegister Move(IRef d, IRef s) {
		XMMRegister dst = reg(d);
		XMMRegister src = reg(s);
		if(!dst.is(src)) {
			asm_.movapd(dst,src);
		}
		return dst;
	}
	XMMRegister EmitMove(XMMRegister dst, XMMRegister src) {
		if(!dst.is(src)) {
			asm_.movapd(dst,src);
		}
		return dst;
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

	void EmitVectorStore(IRef ref, Value& dst, XMMRegister src, int64_t length) {
		if(length > 0)
			asm_.movdqa(EncodeOperand(dst.p,vector_index,times_8),src);
		else {
			XMMRegister filter = reg(-length);
			
			SaveRegisters(live_registers[ref]);
			EmitMove(xmm0,src);
			EmitMove(xmm1,filter);
			asm_.movq(rdi,&dst);
			EmitCall((void*)store_conditional);
			RestoreRegisters(live_registers[ref]);
		}
	}

	void EmitLogicalStore(IRef ref, Value& dst,  XMMRegister src, int64_t length) {
                asm_.pshufb(src,ConstantTable(C_PACK_LOGICAL));
		asm_.movq(rbx, src);
		asm_.movw(EncodeOperand(dst.p,vector_index,times_1),rbx);
                asm_.pshufb(src,ConstantTable(C_PACK_LOGICAL));
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
		asm_.subq(rsp, Immediate(0x20));
		asm_.movdqu(Operand(rsp,0),reg(node.binary.a));
		asm_.movdqu(Operand(rsp,0x10),reg(node.binary.b));
		asm_.movsd(xmm0,Operand(rsp,0));
		asm_.movsd(xmm1,Operand(rsp,0x10));
		EmitCall(fn);
		asm_.movsd(Operand(rsp, 0),xmm0);
		asm_.movsd(xmm0,Operand(rsp, 0x8));
		asm_.movsd(xmm1,Operand(rsp, 0x18));
		EmitCall(fn);
		asm_.movsd(Operand(rsp, 0x8),xmm0);
		asm_.movdqu(reg(ref),Operand(rsp, 0));
		asm_.addq(rsp, Immediate(0x20));
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
		asm_.subq(rsp, Immediate(256));
		uint64_t index = 0;
		for(RegisterIterator it(regs); !it.done(); it.next()) {
			asm_.movdqu(Operand(rsp, index), XMMRegister::FromAllocationIndex(it.value()));
			index += 16;
		}
	}

	void RestoreRegisters(RegisterSet regs) {
		uint64_t index = 0;
		for(RegisterIterator it(regs); !it.done(); it.next()) {
			asm_.movdqu(XMMRegister::FromAllocationIndex(it.value()), Operand(rsp, index));
			index += 16;
		}
		asm_.addq(rsp, Immediate(256));
	}

	void EmitCompare(IRef ref, Assembler::ComparisonType typ) {
		IRNode & node = trace->nodes[ref];
		if(Type::Double == trace->nodes[node.binary.a].type) {
			asm_.cmppd(reg(ref),reg(node.binary.b),typ);
		} else {
			_error("NYI - integer compare");
		}
	}

	typedef void (*fn) (uint64_t thread_index, uint64_t start, uint64_t end);
	
	static void executebody(void* args, void* h, uint64_t start, uint64_t end, Thread& thread) {
		//printf("%d: called with %d to %d\n", thread.index, start, end);
		fn code = (fn)args;
		code(thread.index*8, start, end);	
	}


	void Compile() {
		bzero(store_inst,sizeof(IRNode *) * trace->nodes.size());
		memset(allocated_register,-1,sizeof(char) * trace->nodes.size());

		RegisterAllocate();
		InstructionSelection();
	}

	void Execute(Thread & thread) {
		fn trace_code = (fn) trace->code_buffer->code;
		if(thread.state.verbose) {
			//timespec begin;
			//get_time(begin);
			thread.doall(NULL, executebody, (void*)trace_code, 0, trace->code_buffer->outer_loop, 4, 16*1024); 
			//trace_code(thread.index, 0, trace->length);
			//double s = trace->length / (time_elapsed(begin) * 10e9);
			//printf("elements computed / us: %f\n",s);
		} else {
			thread.doall(NULL, executebody, (void*)trace_code, 0, trace->code_buffer->outer_loop, 4, 16*1024); 
			//trace_code(thread.index, 0, trace->length);
		}
	}

	void GlobalReduce(Thread& thread) {
		for(IRef i = 0; i < trace->nodes.size(); i++) {
			IRef ref = i;
			IRNode & node = trace->nodes[i];
			switch(node.op) {
				case IROpCode::sum:  {
					IRNode & str = *store_inst[ref];
					double* d = (double*)str.store.dst.p;
					double sum = 0;
					for(int64_t j = 0; j < thread.state.nThreads; j++) {
						sum += d[j*8];
						sum += d[j*8+1];
					}
					str.store.dst = Double::c(sum);
				} break;
				case IROpCode::prod:  {
					IRNode & str = *store_inst[ref];
					double* d = (double*)str.store.dst.p;
					double sum = 1.0;
					for(int64_t j = 0; j < thread.state.nThreads; j++) {
						sum *= d[j*8];
						sum *= d[j*8+1];
					}
					str.store.dst = Double::c(sum);
				} break;
				default: break;
			}
		}
	}
};

void Trace::JIT(Thread & thread) {
	if(code_buffer == NULL) { //since it is expensive to reallocate this, we reuse it across traces
		code_buffer = new TraceCodeBuffer();
	}

	TraceJIT trace_code(this);

	trace_code.Compile();
	trace_code.Execute(thread);
	trace_code.GlobalReduce(thread);
}
