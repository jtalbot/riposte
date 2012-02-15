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

	Constant(double d0, double d1)
	: d0(d0), d1(d1) {}

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
		constant_table[C_PACK_LOGICAL] = Constant(0x0706050403020800LL,0x0F0E0D0C0B0A0902LL);
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

struct SSEValue {
	union {
		__m128d D;
		__m128i I;
		int64_t i[2];
		double  d[2];
	};
};

static __m128d sign_d(__m128d input) {
	SSEValue v;
	v.D = input;

	v.d[0] = v.d[0] > 0 ? 1 : (v.d[0] < 0 ? -1 : 0);
	v.d[1] = v.d[1] > 0 ? 1 : (v.d[1] < 0 ? -1 : 0);

	return v.D;
}

static __m128d mod_i(__m128d a, __m128d b) {
	SSEValue va, vb; 
	va.D = a;
	vb.D = b;
	
	va.i[0] = va.i[0] % vb.i[0];
	va.i[1] = va.i[1] % vb.i[1];

	return va.D;
}

static __m128d mul_i(__m128d a, __m128d b) {
	SSEValue va, vb; 
	va.D = a;
	vb.D = b;
	
	va.i[0] = va.i[0] * vb.i[0];
	va.i[1] = va.i[1] * vb.i[1];

	return va.D;
}

static __m128d idiv_i(__m128d a, __m128d b) {
	SSEValue va, vb; 
	va.D = a;
	vb.D = b;
	
	va.i[0] = va.i[0] / vb.i[0];
	va.i[1] = va.i[1] / vb.i[1];

	return va.D;
}

static __m128d abs_i(__m128d input) {
	SSEValue v;
	v.D = input;

	v.i[0] = (v.i[0] < 0) ? -v.i[0] : v.i[0];
	v.i[1] = (v.i[1] < 0) ? -v.i[1] : v.i[1];

	return v.D;
}

static __m128d casti2d(__m128d input) {
	SSEValue v; 
	v.D = input;
	
	v.d[0] = v.i[0];
	v.d[1] = v.i[1];

	return v.D;
}

static __m128d casti2l(__m128d input) {
	SSEValue v; 
	v.D = input;
	
	if(Integer::isNA(v.i[0])) v.i[0] = Logical::NAelement;
	else if(v.i[0] == 0) v.i[0] = Logical::FalseElement;
	else v.i[0] = Logical::TrueElement;

	if(Integer::isNA(v.d[1])) v.i[1] = Logical::NAelement;
	else if(v.i[1] == 0) v.i[1] = Logical::FalseElement;
	else v.i[1] = Logical::TrueElement;
	
	return v.D;
}

static __m128d castd2i(__m128d input) {
	SSEValue v; 
	v.D = input;
	
	v.i[0] = (int64_t)v.d[0];
	v.i[1] = (int64_t)v.d[1];

	return v.D;
}

static __m128d castd2l(__m128d input) {
	SSEValue v; 
	v.D = input;
	
	if(Double::isNA(v.d[0]) || Double::isNaN(v.d[0])) v.i[0] = Logical::NAelement;
	else if(v.d[0] == 0) v.i[0] = Logical::FalseElement;
	else v.i[0] = Logical::TrueElement;

	if(Double::isNA(v.d[1]) || Double::isNaN(v.d[1])) v.i[1] = Logical::NAelement;
	else if(v.d[1] == 0) v.i[1] = Logical::FalseElement;
	else v.i[1] = Logical::TrueElement;
	
	return v.D;
}

static __m128d castl2d(__m128d input) {
	SSEValue v; 
	v.D = input;
	
	if(Logical::isTrue((char)v.i[0])) v.d[0] = 1.0;
	else if(Logical::isFalse((char)v.i[0])) v.d[0] = 0.0;
	else v.d[0] = Double::NAelement;
	
	if(Logical::isTrue((char)v.i[1])) v.d[1] = 1.0;
	else if(Logical::isFalse((char)v.i[1])) v.d[1] = 0.0;
	else v.d[1] = Double::NAelement;

	return v.D;
}

static __m128d castl2i(__m128d input) {
	SSEValue v; 
	v.D = input;
	
	if(Logical::isTrue((char)v.i[0])) v.i[0] = 1;
	else if(Logical::isFalse((char)v.i[0])) v.i[0] = 0;
	else v.i[0] = Integer::NAelement;
	
	if(Logical::isTrue((char)v.i[1])) v.i[1] = 1;
	else if(Logical::isFalse((char)v.i[1])) v.i[1] = 0;
	else v.i[1] = Integer::NAelement;

	return v.D;
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

static void sum_by_d(__m128d add, __m128i split, Value* out, int64_t thread_index, int64_t step) {
	union { 
		__m128i ma; 
		int64_t m[2]; 
	}; 
	union { 
		__m128d in; 
		double i[2]; 
	};
	ma = split;
	in = add;
	int off = (int)thread_index*(int)step;
	((double*)(out->p))[off + m[0]*2] += i[0];
	((double*)(out->p))[off + m[1]*2+1] += i[1];
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
	TraceJIT(Trace * t, Thread& thread)
	:  trace(t), thread(thread), asm_(t->code_buffer->code,CODE_BUFFER_SIZE), alloc(XMMRegister::kNumAllocatableRegisters), next_constant_slot(C_FIRST_TRACE_CONST) {
		live_registers = new (PointerFreeGC) RegisterSet[trace->nodes.size()];
		allocated_register = new (PointerFreeGC) int8_t[trace->nodes.size()];

		// allocate xmm0 as a temporary register (needed for blend)
		int8_t temp;
		alloc.allocate(0,&temp);
	}

	Trace * trace;
	Thread const& thread;
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
			case IRNode::TRINARY: {
				AllocateTrinary(ref, node.trinary.a, node.trinary.b, node.trinary.c);
			} break;
			case IRNode::BINARY: {
				AllocateBinary(ref,node.binary.a,node.binary.b);
			} break;
			case IRNode::FOLD: {
				AllocateUnary(ref, node.fold.a);
			} break;
			case IRNode::UNARY: {
				AllocateUnary(ref,node.unary.a);
			} break;
			case IRNode::CONSTANT: /*fallthrough*/
			case IRNode::LOAD: /*fallthrough*/
			case IRNode::SEQUENCE: {
				AllocateNullary(ref);
			} break;
			case IRNode::NOP: {
				/* nothing */
			} break;
			default:
				_error("unsupported op in reg allocation");
			}
		}
	}

	int8_t AllocateNullary(IRef ref, bool p = true) {
		if(allocated_register[ref] < 0) {
			if(!alloc.allocate(&allocated_register[ref]))
				_error("exceeded available registers");
		}
		
		int8_t r = allocated_register[ref];
		alloc.free(r);
		
		live_registers[ref] = alloc.live_registers();
		
		return r;
	}

	int8_t AllocateUnary(IRef ref, IRef a, bool p = true) {
		int8_t r = AllocateNullary(ref,false);
		if(allocated_register[a] < 0) {
			//try to allocated register a and r to the same location,  this avoids moves in binary instructions
			if(!alloc.allocate(r,&allocated_register[a]))
				_error("exceeded available registers");
		}
		if(trace->nodes[ref].liveOut) {
			IRNode::Shape& s = trace->nodes[a].shape;
			if(s.filter > 0 && allocated_register[s.filter] < 0) {
				if(!alloc.allocate(&allocated_register[s.filter]))
					_error("exceeded available registers");
			}
			if(s.split > 0 && allocated_register[s.split] < 0) {
				if(!alloc.allocate(&allocated_register[s.split]))
					_error("exceeded available registers");
			}
		}
		return r;
	}

	int8_t AllocateBinary(IRef ref, IRef a, IRef b) {
		int8_t r = AllocateUnary(ref,a,false);
		if(allocated_register[b] < 0) {
			RegisterSet mask = ~(1 << allocated_register[ref]);
			//we avoid allocating registers such that r = a op r occurs
			if(!alloc.allocateWithMask(mask,&allocated_register[b])) {
				_error("exceeded available registers");
			}
			assert(allocated_register[ref] != allocated_register[b]);
		}

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
		
		asm_.push(thread_index);		// -0x08
		asm_.push(constant_base);		// -0x10
		asm_.push(vector_index);		// -0x18
		asm_.push(load_addr);			// -0x20
		asm_.push(vector_length);		// -0x28
		asm_.push(rbx);				// -0x30
		asm_.subq(rsp, Immediate(0x8));		// -0x38

		// reserve room for loop carried variables...
		// TODO: do this in register allocation
		//  so that loop carried variables can be placed in registers.
		//  Make this stack allocation simply part of spilling code.
		int64_t stackSpace = 0;
		for(IRef ref = 0; ref < trace->nodes.size(); ref++) {
			IRNode & node = trace->nodes[ref];
			if(node.op == IROpCode::seq)
				stackSpace += 0x10;
		}
		asm_.subq(rsp, Immediate(stackSpace));

		asm_.movq(thread_index, rdi);
		asm_.movq(constant_base, &trace->code_buffer->constant_table[0]);
		asm_.movq(vector_index, rsi);
		asm_.movq(vector_length, rdx);

		int64_t stackOffset = 0;
		for(IRef ref = 0; ref < trace->nodes.size(); ref++) {
			IRNode & node = trace->nodes[ref];
			if(node.op == IROpCode::seq) {
				if(node.isDouble()) {
					Constant initial(node.sequence.da-2*node.sequence.db, node.sequence.da-node.sequence.db);
					Operand o_initial = PushConstant(initial);
					asm_.movdqa(xmm0, o_initial);
					asm_.movdqa(Operand(rsp, stackOffset), xmm0);
				} else {
					Constant initial(node.sequence.ia-2*node.sequence.ib, node.sequence.ia-node.sequence.ib);
					Operand o_initial = PushConstant(initial);
					asm_.movdqa(xmm0, o_initial);
					asm_.movdqa(Operand(rsp, stackOffset), xmm0);
				}
				stackOffset += 0x10;
			}
		}

		Label begin;

		asm_.bind(&begin);

		stackOffset = 0;
		for(IRef ref = 0; ref < trace->nodes.size(); ref++) {
			IRNode & node = trace->nodes[ref];
			if(node.enc == IRNode::LOAD || node.enc == IRNode::SEQUENCE || (node.enc == IRNode::CONSTANT && node.shape.length > 1)) trace->code_buffer->outer_loop = node.shape.length;

			switch(node.op) {

			case IROpCode::constant: {
				Constant c;
				switch(node.type) {
					case Type::Integer: c = Constant(node.constant.i); break;
					case Type::Logical: c = Constant(node.constant.l); break;
					case Type::Double:  c = Constant(node.constant.d); break;
					default: _error("unexpected type");
				}
				asm_.movdqa(reg(ref),PushConstant(c));
			} break;
			case IROpCode::load: {
				if(node.isLogical())
					asm_.pmovsxbq(reg(ref), EncodeOperand(node.out.p, vector_index, times_1));
				else
					asm_.movdqa(reg(ref),EncodeOperand(node.out.p,vector_index,times_8));
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
				else			EmitVectorizedBinaryFunction(ref,mul_i);
			} break;
			case IROpCode::div: 	asm_.divpd(Move(ref, node.binary.a),reg(node.binary.b)); break;
			case IROpCode::idiv: {
				if(node.isDouble()) {	
					asm_.divpd(Move(ref, node.binary.a),reg(node.binary.b)); 
					asm_.roundpd(reg(ref), reg(ref), Assembler::kRoundDown);	
				} else	{ 
					EmitVectorizedBinaryFunction(ref,idiv_i);
				}
			} break;
			case IROpCode::sqrt: 	asm_.sqrtpd(reg(ref),reg(node.unary.a)); break;
			//case IROpCode::round:	asm_.roundpd(reg(ref),reg(node.unary.a), Assembler::kRoundToNearest); break;
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
					EmitVectorizedUnaryFunction(ref,abs_i); 
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
			case IROpCode::sign: 	EmitVectorizedUnaryFunction(ref,sign_d); break;
			case IROpCode::mod: {
				if(node.isDouble()) {
					EmitMove(xmm0, reg(node.binary.a));
					asm_.divpd(xmm0, reg(node.binary.b)); 
					asm_.roundpd(xmm0, xmm0, Assembler::kRoundDown);
					asm_.mulpd(xmm0, reg(node.binary.b));
					asm_.subpd(Move(ref, node.binary.a), xmm0);
				} else {
					EmitVectorizedBinaryFunction(ref, mod_i);
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
					EmitVectorizedUnaryFunction(ref, casti2d);
				else if(node.isDouble() && trace->nodes[node.unary.a].isLogical())
					EmitVectorizedUnaryFunction(ref, castl2d);
				else if(node.isInteger() && trace->nodes[node.unary.a].isDouble())
					EmitVectorizedUnaryFunction(ref, castd2i);
				else if(node.isInteger() && trace->nodes[node.unary.a].isLogical())
					EmitVectorizedUnaryFunction(ref, castl2i);
				else if(node.isLogical() && trace->nodes[node.unary.a].isDouble())
					EmitVectorizedUnaryFunction(ref, castd2l);
				else if(node.isLogical() && trace->nodes[node.unary.a].isInteger())
					EmitVectorizedUnaryFunction(ref, casti2l);
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
					Constant step(2*node.sequence.db, 2*node.sequence.db);
					Operand o_step = PushConstant(step);
					asm_.movdqa(reg(ref), Operand(rsp, stackOffset));
					asm_.addpd(reg(ref), o_step);
					asm_.movdqa(Operand(rsp, stackOffset), reg(ref));
				} else {
					Constant step(2*node.sequence.ib, 2*node.sequence.ib);
					Operand o_step = PushConstant(step);
					asm_.movdqa(reg(ref), Operand(rsp, stackOffset));
					asm_.paddq(reg(ref), o_step);
					asm_.movdqa(Operand(rsp, stackOffset), reg(ref));
				}
				stackOffset += 0x10;
			} break;

			case IROpCode::sum:  {
				if(node.isDouble()) {
					assert(node.live);
					for(int64_t i = 0; i < node.out.length; i++)
						((double*)node.out.p)[i] = 0.0;
					Move(ref, node.unary.a);
					if(trace->nodes[node.fold.a].shape.filter > 0) {
						IRef mask = trace->nodes[node.fold.a].shape.filter;
						EmitMove(xmm0, reg(mask));
						asm_.xorpd(xmm0, ConstantTable(C_NOT_MASK));
						asm_.blendvpd(reg(ref), ConstantTable(C_DOUBLE_ZERO));
					}
					if(trace->nodes[node.fold.a].shape.split > 0) {
						EmitSplitFold(ref, (void*)sum_by_d);
						/*// add to each independently?
						Constant c(node.dst.p);
						Operand base = PushConstant(c);
						asm_.movq(r8, base);
						asm_.movq(r9, reg(split));
						asm_.imul(r9, r9, Immediate(scale));
						asm_.addq(r9, r9, thread_index);
						asm_.movq(r10, reg(ref));
						asm_.addq(r10, Operand(r8, r9, times_8, 0));
						asm_.movq(Operand(r8, r9, times_8, 0), r10); 
						asm_.movhlps(xmm0, reg(split));
						asm_.movq(r9, xmm0);
						asm_.imul(r9, r9, Immediate(scale));
						asm_.addq(r9, r9, thread_index);
						asm_.movq(r10, reg(ref));
						asm_.addq(r10, Operand(r8, r9, times_8, 0));
						asm_.movq(Operand(r8, r9, times_8, 0), r10); */
					} else {
						Operand op = EncodeOperand(node.out.p, thread_index, times_8);
						asm_.addpd(reg(ref), op);
						asm_.movdqa(op,reg(ref));
					}
				} 
				else {
					EmitFoldFunction(ref,(void*)sumi,Constant((int64_t)0LL)); break;
				}
			} break;

			case IROpCode::prod: { 
				if(node.isDouble()) {
					/*assert(store_inst[ref] != NULL);
					IRNode & str = *store_inst[ref];
					for(uint64_t i = 0; i < 128; i++)
						((double*)str.store.dst.p)[i] = 1.0;
					Operand op = EncodeOperand(str.store.dst.p, thread_index, times_8);
					asm_.mulpd(reg(ref),op);
					asm_.movdqa(op,reg(ref));*/
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

			case IROpCode::split:
			case IROpCode::filter:
				Move(ref, node.binary.a);
			break;

			case IROpCode::ifelse:
				EmitMove(xmm0, reg(node.trinary.c));
				asm_.blendvpd(Move(ref, node.trinary.a), reg(node.trinary.b));
			break;

			//case IROpCode::signif:
			default:	_error("unimplemented op"); break;
		
			}

			if(node.liveOut) {
				switch(node.enc) {
					case IRNode::UNARY:
					case IRNode::BINARY:
					case IRNode::TRINARY:
					case IRNode::SEQUENCE:
					case IRNode::CONSTANT: {
						if(Type::Logical == node.type)
							EmitLogicalStore(ref, node.out, node.shape);
						else
							EmitVectorStore(ref, node.out, node.shape);
					} break;

					case IRNode::FOLD:
					case IRNode::LOAD:
					case IRNode::NOP:
						// do nothing...
					break;
				}
			}
		}

		asm_.addq(vector_index, Immediate(2));
		asm_.cmpq(vector_index,vector_length);
		asm_.j(less,&begin);

		asm_.addq(rsp, Immediate(stackSpace));
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
		return XMMRegister::FromAllocationIndex(allocated_register[r]);	// xmm0 is a temporary register
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
		SaveRegisters(ref);
		EmitMove(xmm0,reg(node.unary.a));
		asm_.movq(rdi,(void*)&trace->code_buffer->constant_table[offset]);
		EmitCall(fn);
		EmitMove(reg(ref),xmm0);
		RestoreRegisters(ref);
	}

	void EmitSplitFold(IRef ref, void* fn) {
		IRNode const& node = trace->nodes[ref];
		XMMRegister src = reg(ref);
		XMMRegister split = reg(trace->nodes[node.fold.a].shape.split);
		int64_t step = node.out.length/thread.state.nThreads;
		SaveRegisters(ref);
		EmitMove(xmm0,src);
		EmitMove(xmm1,split);
		asm_.movq(rdi,(int64_t)&node.out);
		asm_.movq(rsi,thread_index);
		asm_.movq(rdx,PushConstant(Constant(step)));
		EmitCall(fn);
		RestoreRegisters(ref);
	}

	void EmitVectorStore(IRef ref, Value& dst, IRNode::Shape const& shape) {
		XMMRegister src = reg(ref);
		if(shape.filter == 0)
			asm_.movdqa(EncodeOperand(dst.p,vector_index,times_8),src);
		else {
			dst.length = 0;
			XMMRegister filter = reg(shape.filter);
			
			SaveRegisters(ref);
			EmitMove(xmm0,src);
			EmitMove(xmm1,filter);
			asm_.movq(rdi,&dst);
			EmitCall((void*)store_conditional);
			RestoreRegisters(ref);
		}
	}

	void EmitLogicalStore(IRef ref, Value& dst, IRNode::Shape const& shape) {
		XMMRegister src = reg(ref);
                asm_.pshufb(src,ConstantTable(C_PACK_LOGICAL));
		asm_.movq(rbx, src);
		asm_.movw(EncodeOperand(dst.p,vector_index,times_1),rbx);
                asm_.pshufb(src,ConstantTable(C_PACK_LOGICAL));
	}

	void EmitVectorizedUnaryFunction(IRef ref, __m128d (*fn)(__m128d)) {
		EmitVectorizedUnaryFunction(ref,(void*)fn);
	}
	void EmitVectorizedUnaryFunction(IRef ref, void * fn) {

		SaveRegisters(ref);

		EmitMove(xmm0,reg(trace->nodes[ref].unary.a));
		EmitCall(fn);
		EmitMove(reg(ref),xmm0);

		RestoreRegisters(ref);

	}

	void EmitVectorizedBinaryFunction(IRef ref, __m128d (*fn)(__m128d,__m128d)) {
		EmitBinaryFunction(ref,(void*)fn);
	}
	void EmitVectorizedBinaryFunction(IRef ref, void * fn) {

		SaveRegisters(ref);

		EmitMove(xmm0, reg(trace->nodes[ref].binary.a));
		EmitMove(xmm1, reg(trace->nodes[ref].binary.b));
		EmitCall(fn);
		EmitMove(reg(ref),xmm0);

		RestoreRegisters(ref);

	}
	void EmitUnaryFunction(IRef ref, double (*fn)(double)) {
		EmitUnaryFunction(ref,(void*)fn);
	}
	void EmitUnaryFunction(IRef ref, void * fn) {

		SaveRegisters(ref);

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

		RestoreRegisters(ref);
	}
	void EmitBinaryFunction(IRef ref, double (*fn)(double,double)) {
		EmitBinaryFunction(ref,(void*)fn);
	}
	void EmitBinaryFunction(IRef ref, void * fn) {
		//this isn't the most optimized way to do this but it works for now
		SaveRegisters(ref);
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
		RestoreRegisters(ref);
	}

	void EmitDebugPrintResult(IRef i) {
		/*RegisterSet regs = live_registers[i];
		regs &= ~(1 << allocated_register[i]);

		SaveRegisters(i);
		EmitMove(xmm0,reg(i));
		asm_.movq(rdi,vector_index);
		EmitCall((void*) debug_print);
		RestoreRegisters(i);*/
	}

	void SaveRegisters(IRef i) {
		// We only need to save live out
		// minus the register allocated for this instruction
		RegisterSet regs = live_registers[i];
		regs |= (1 << allocated_register[i]);
		if(regs != ~0U) {
			asm_.subq(rsp, Immediate(256));
			uint64_t index = 0;
			for(RegisterIterator it(regs); !it.done(); it.next()) {
				if(it.value() > 0) {
					asm_.movdqu(Operand(rsp, index), XMMRegister::FromAllocationIndex(it.value()));
					index += 16;
				}
			}
		}
	}

	void RestoreRegisters(IRef i) {
		// RestoreRegisters must be the exact inverse of SaveRegisters
		RegisterSet regs = live_registers[i];
		regs |= (1 << allocated_register[i]);
		if(regs != ~0U) {
			uint64_t index = 0;
			for(RegisterIterator it(regs); !it.done(); it.next()) {
				if(it.value() > 0) {
					asm_.movdqu(XMMRegister::FromAllocationIndex(it.value()), Operand(rsp, index));
					index += 16;
				}
			}
			asm_.addq(rsp, Immediate(256));
		}
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
		for(IRef ref = 0; ref < trace->nodes.size(); ref++) {
			IRNode & node = trace->nodes[ref];
			if(node.liveOut) {
				switch(node.op) {
				case IROpCode::sum:  {
					Double r(node.shape.length);
					for(int64_t i = 0; i < r.length; i++)
						r[i] = 0.0;
					double const* d = (double const*)node.out.p;
					int64_t step = node.out.length/thread.state.nThreads;
					for(int64_t i = 0; i < node.shape.length; i++) {
						for(int64_t j = 0; j < thread.state.nThreads; j++) {
							r[i] += d[j*step+i*2];
							r[i] += d[j*step+i*2+1];
						}
					}
					node.out = r;
				} break;
				case IROpCode::prod:  {
					/*double* d = (double*)node.out.p;
					double sum = 1.0;
					for(int64_t j = 0; j < node.out.length; j++) {
						sum *= d[j*8];
						sum *= d[j*8+1];
					}
					node.out = Double::c(sum);*/
				} break;
				default: break;
				}
			}
		}
	}
};

void Trace::JIT(Thread & thread) {
	if(code_buffer == NULL) { //since it is expensive to reallocate this, we reuse it across traces
		code_buffer = new TraceCodeBuffer();
	}

	TraceJIT trace_code(this, thread);
	trace_code.Compile();
	trace_code.Execute(thread);
	trace_code.GlobalReduce(thread);
}
