#include <sys/mman.h>
#include <math.h>
#include <pthread.h>

#include "interpreter.h"
#include "vector.h"
#include "ops.h"
#include "assembler-x64.h"
#include "register_set.h"
#include "internal.h"

#ifdef USE_AMD_LIBM
#include <amdlibm.h>
#endif

using namespace v8::internal;

#define SIMD_WIDTH (2 * sizeof(double))
#define CODE_BUFFER_SIZE (256 * 2048)

#define BIG_CARDINALITY 1024 

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

	Constant(uint64_t ii0, uint64_t ii1)
	: u0(ii0), u1(ii1) {}

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
	C_INTEGER_ONE  = 0x5,
	C_INTEGER_TWO  = 0x6,
	C_DOUBLE_ZERO = 0x7,
	C_DOUBLE_ONE = 0x8,
	C_DOUBLE_NA = 0x9,
	C_INTEGER_MIN  = 0xa,
	C_INTEGER_MAX  = 0xb,
	C_DOUBLE_MIN  = 0xc,
	C_DOUBLE_MAX  = 0xd,
	C_FIRST_TRACE_CONST = 0xe
};

static int make_executable(char * data, size_t size) {
	int64_t page = (int64_t)data & ~0x7FFF;
	int64_t psize = (int64_t)data + size - page;
	return mprotect((void*)page,psize, PROT_READ | PROT_WRITE | PROT_EXEC);
}

//scratch space that is reused across traces
struct TraceCodeBuffer {
	Constant constant_table[8192] __attribute__((aligned(16)));
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
		constant_table[C_PACK_LOGICAL] = Constant((uint64_t)0x0706050403020800LL,(uint64_t)0x0F0E0D0C0B0A0901LL);
		constant_table[C_INTEGER_ONE] = Constant((int64_t)1LL,(int64_t)1LL);
		constant_table[C_INTEGER_TWO] = Constant((int64_t)2LL,(int64_t)2LL);
		constant_table[C_DOUBLE_ZERO] = Constant(0.0, 0.0);
		constant_table[C_DOUBLE_ONE] = Constant(1.0, 1.0);
		constant_table[C_DOUBLE_NA] = Constant(Double::NAelement, Double::NAelement);
		constant_table[C_INTEGER_MIN] = Constant((int64_t)std::numeric_limits<int64_t>::min(),(int64_t)std::numeric_limits<int64_t>::min());
		constant_table[C_INTEGER_MAX] = Constant((int64_t)std::numeric_limits<int64_t>::max(),(int64_t)std::numeric_limits<int64_t>::max());
		constant_table[C_DOUBLE_MIN] = Constant(-std::numeric_limits<double>::infinity(),-std::numeric_limits<double>::infinity());
		constant_table[C_DOUBLE_MAX] = Constant(std::numeric_limits<double>::infinity(),std::numeric_limits<double>::infinity());
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

static __m128d exp_d(__m128d input) {
       SSEValue v;
       v.D = input;
       for(int i = 0; i < 2; i++) v.d[i] = exp(v.d[i]);
       return v.D;
}

static __m128d log_d(__m128d input) {
       SSEValue v;
       v.D = input;
       for(int i = 0; i < 2; i++) v.d[i] = log(v.d[i]);
       return v.D;
}

static __m128d random_d(__m128d input) {
	SSEValue v; 
	v.D = input;

	uint64_t thread_index = v.i[0];
	
	Thread::RandomSeed& r = Thread::seed[thread_index];
	
	// advance three times to avoid taking powers of 2
	r.v[0] = r.v[0] * r.m[0] + r.a[0];
	r.v[1] = r.v[1] * r.m[1] + r.a[1];
	r.v[0] = r.v[0] * r.m[0] + r.a[0];
	r.v[1] = r.v[1] * r.m[1] + r.a[1];
	r.v[0] = r.v[0] * r.m[0] + r.a[0];
	r.v[1] = r.v[1] * r.m[1] + r.a[1];

	SSEValue o;

	o.d[0] = (double)r.v[0] / ((double)std::numeric_limits<uint64_t>::max() + 1);
	o.d[1] = (double)r.v[1] / ((double)std::numeric_limits<uint64_t>::max() + 1);
	return o.D;
}

static void double_push(Double* d, double v) {
	if(d->length >= 1) {
		if(d->length == (int64_t)nextPow2(d->length)) {
			// reallocate and copy
			Double n(nextPow2(d->length+1));
			memcpy(n.v(), d->v(), d->length*sizeof(double));
			d->p = n.p;
		}
		(*d)[d->length++] = v;
	} else {
		Double::InitScalar(*d, v);
	} 
}

static __m128d store_conditional(__m128d input, __m128i mask, Value* out) {
	SSEValue i, m; 
	i.D = input;
	m.I = mask;
	if(m.i[0] == -1)
		double_push((Double*)out, i.d[0]); 
		//((double*)(out->p))[out->length++] = i.d[0];
	if(m.i[1] == -1) 
		double_push((Double*)out, i.d[1]); 
		//((double*)(out->p))[out->length++] = i.d[1];
	return input;
}

static __m128d store_conditional_l(__m128d input, __m128i mask, Value* out) {
	SSEValue i, m; 
	i.D = input;
	m.I = mask;
	if(m.i[0] == -1) 
		((char*)(out->p))[out->length++] = (char)i.i[0];
	if(m.i[1] == -1) 
		((char*)(out->p))[out->length++] = (char)i.i[1];
	return input;
}

static __m128d sequenceStart_d(__m128d a, int64_t vector_index) {
	SSEValue s, v;
	s.D = a;
	v.d[0] = s.d[0] + (vector_index-2)*s.d[1];
	v.d[1] = s.d[0] + (vector_index-1)*s.d[1];
	return v.D;
}

static __m128d sequenceStart_i(__m128d a, int64_t vector_index) {
	SSEValue s, v;
	s.D = a;
	v.i[0] = s.i[0] + (vector_index-2)*s.i[1];
	v.i[1] = s.i[0] + (vector_index-1)*s.i[1];
	return v.D;
}

static __m128d repeatStart_i(__m128d a, int64_t vector_index) {
	SSEValue s, v;
	s.D = a;
	v.i[0] = ((vector_index-2) / s.i[1]) % s.i[0]; // j
	v.i[1] = ((vector_index-1) / s.i[1]) % s.i[0]; // j
	return v.D;
}

static __m128d repeatEach_i(__m128d a, int64_t vector_index) {
	SSEValue s, v;
	s.D = a;
	v.i[0] = (vector_index-2) % s.i[1]; // each
	v.i[1] = (vector_index-1) % s.i[1]; // each
	return v.D;
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
FOLD_SCAN_FN(prodd, double , *)
FOLD_SCAN_FN(cumsumi, int64_t, +)
FOLD_SCAN_FN(cumsumd, double , +)

struct TraceJIT {
	TraceJIT(Trace * t, Thread& thread)
	:  trace(t), thread(thread), asm_(t->code_buffer->code,CODE_BUFFER_SIZE), alloc(XMMRegister::kNumAllocatableRegisters-2), next_constant_slot(C_FIRST_TRACE_CONST) {
		// preserve the last register (xmm15) as a temporary exchange register
		// to make code gen easier for now 
		live_registers = new (PointerFreeGC) RegisterSet[trace->nodes.size()];
		allocated_register = new (PointerFreeGC) int8_t[trace->nodes.size()];

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
	uint64_t spills;

	struct RegisterAssignment {
		int8_t r;
		int8_t o;
	};

	struct OpAssignment {
		RegisterAssignment r, a, b, c, f, s;
	};

	std::vector<OpAssignment> assignment;
	IRef liveRegisters[14]; 

	bool usesNode(IRef ref, IRef use) {
		IRNode node = trace->nodes[ref];
		switch(node.arity) {
		case IRNode::TRINARY:
			if(node.trinary.c == use) return true;
		case IRNode::BINARY:
			if(node.binary.b == use) return true;
		case IRNode::UNARY:
			if(node.unary.a == use) return true;
		default:
			if(ref == use) return true;
			if(node.shape.filter == use) return true;
			if(node.shape.split == use) return true;
		}
		return false;
	}

	int8_t spillRegister(IRef currentOp) {
		spills++;
		// look for register with the farthest away use?
		// for now just do slow search backwards
		IRef minUse = 1000000;
		int8_t minReg = 0;
		for(int8_t i = 0; i < 14; i++) {
			IRef ref = currentOp;
			for(; ref >= 0; ref--) {
				if(usesNode(ref, liveRegisters[i])) {
					if(ref < minUse) {
						minUse = ref;
						minReg = i;
					}
					break;
				}
			}
		}
		int8_t r = minReg;
		//printf("spilled r%d (replaced n%d with n%d)\n", minReg, liveRegisters[r], currentOp);
		allocated_register[liveRegisters[r]] = spills++; // mark node as spilled
		liveRegisters[r] = -1;	// unassign spilled register
		return r;
	}

	void allocate(IRef currentOp, IRef node, RegisterAssignment& assignment, int8_t preferred) {
		int8_t r = allocated_register[node];
		// if b is not already assigned to a register
		assignment.o = r;
		if(r < 0 || r >= 14) {
			// Attempt to allocate
			if(!alloc.allocate(preferred, &r)) {
				//printf("spilling\n");
				r = spillRegister(currentOp);
			}
			//printf("Assigned n%d to r%d\n", node, r);
		}
		assignment.r = r;
		allocated_register[node] = r;
		liveRegisters[r] = node;
	}

	int8_t deallocate(IRef node) {
		int8_t r = allocated_register[node];
		if(r >= 0) {
			allocated_register[node] = -1;
			liveRegisters[r] = -1;
			alloc.free(r);
		}
		return r;
	}

	void RegisterAllocate() {
		spills = 16;
		assignment.resize(trace->nodes.size());
		for(size_t i = 0; i < trace->nodes.size(); i++) {
			allocated_register[i] = -1;
		}
		
		for(IRef ref = trace->nodes.size()-1; ref >= 0; ref--) {
			IRNode & node = trace->nodes[ref];
			
			if(node.group == IRNode::SCALAR)	
				continue;

			allocate(ref, ref, assignment[ref].r, -1);	
			
			switch(node.arity) {
			case IRNode::TRINARY:
				allocate(ref, node.trinary.c, assignment[ref].c, -1);
			case IRNode::BINARY:
				allocate(ref, node.binary.b, assignment[ref].b, -1);
			case IRNode::UNARY:
				allocate(ref, node.unary.a, assignment[ref].a, deallocate(ref));
			default:
				if(node.shape.filter >= 0)
					allocate(ref, node.shape.filter, assignment[ref].f, -1);
				if(node.shape.split >= 0)
					allocate(ref, node.shape.split,  assignment[ref].s, -1);
				deallocate(ref);
			}
			live_registers[ref] = alloc.live_registers();
		}
		
	}

	void unspill(RegisterAssignment const& a, int8_t& r) {
		if(a.r != r) {
			assert(a.r >= 0 && a.r < 14);
			asm_.movdqa(XMMRegister::FromAllocationIndex(a.r),
					Operand(rsp, r*0x10));
			r = a.r;
		}
	}

	void spill(RegisterAssignment const& a, int8_t& r) {
		if(a.o >= 14) {
			asm_.movdqa(Operand(rsp, a.o*0x10),
				XMMRegister::FromAllocationIndex(a.r));
		}
		r = a.o;
	}

	// a*1+(a*2+(a*3+(a*4+(a*5+(a*6+(a*7+(a*8+(a*9+(a*10+(a*11+(a*12+(a*13+(a*14+(a*15))))))))))))))

	XMMRegister RegR(IRef r) {
		return XMMRegister::FromAllocationIndex(assignment[r].r.r);
	}

	XMMRegister RegA(IRef r) {
		return XMMRegister::FromAllocationIndex(assignment[r].a.r);
	}

	XMMRegister RegB(IRef r) {
		return XMMRegister::FromAllocationIndex(assignment[r].b.r);
	}

	XMMRegister RegC(IRef r) {
		return XMMRegister::FromAllocationIndex(assignment[r].c.r);
	}

	XMMRegister RegF(IRef r) {
		return XMMRegister::FromAllocationIndex(assignment[r].f.r);
	}

	XMMRegister RegS(IRef r) {
		return XMMRegister::FromAllocationIndex(assignment[r].s.r);
	}

	XMMRegister MoveA2R(IRef r) {
		XMMRegister a = RegA(r);
		XMMRegister d = RegR(r);
		if(!a.is(d)) {
			asm_.movapd(d, a);
		}
		return d;
	}
	
	void InstructionSelection() {
		//pass 2 instruction selection

		int qq = 0;

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
		int64_t stackSpace = spills*0x10;
		for(IRef ref = 0; ref < (int64_t)trace->nodes.size(); ref++) {
			IRNode & node = trace->nodes[ref];
			
			if(node.op == IROpCode::seq)
				stackSpace += 0x10;
			else if(node.op == IROpCode::rep)
				stackSpace += 0x20;
			else if(node.op == IROpCode::random)
				stackSpace += 0x10;
			else if(node.group == IRNode::FOLD)
				stackSpace += 0x10;


			// TODO: allocate thread split filtered output (how to maintain ordering?)
			// allocate temporary space for folds (put in IRNode::in)
			if(node.group == IRNode::FOLD) {
				int64_t size = node.shape.levels <= BIG_CARDINALITY ? node.shape.levels*2 : node.shape.levels;
				if(node.type == Type::Double) {
					// 16 min fills possibly unaligned cache line
					node.in = Double((size+16LL)*thread.state.nThreads);
				} else if(node.type == Type::Integer) {
					node.in = Integer((size+16LL)*thread.state.nThreads);
				} else if(node.type == Type::Logical) {
					node.in = Logical((size+128LL)*thread.state.nThreads);
				} else {
					_error("Unknown type in initialize temporary space");
				}
			}

			// allocate outputs
			if(node.liveOut || (node.group == IRNode::FOLD && node.outShape.length <= BIG_CARDINALITY)) { 
				int64_t length = node.outShape.length;
				
				if(node.shape.levels != 1 && node.group != IRNode::FOLD)
					_error("Group by without aggregate not yet supported");
				if(node.shape.filter >= 0 && node.group != IRNode::FOLD)
					length = 0;
				
				if(node.type == Type::Double) {
					node.out = Double(length);
				} else if(node.type == Type::Integer) {
					node.out = Integer(length);
				} else if(node.type == Type::Logical) {
					node.out = Logical(length);
				} else if(node.type == Type::List) {
					node.out = List(length);
				} else {
					_error("Unknown type in initialize outputs");
				}
			}
		}
		asm_.subq(rsp, Immediate(stackSpace));

		asm_.movq(thread_index, rdi);
		asm_.movq(constant_base, &trace->code_buffer->constant_table[0]);
		asm_.movq(vector_index, rsi);
		asm_.movq(vector_length, rdx);

		int64_t stackOffset = spills*0x10;
		for(IRef ref = 0; ref < (int64_t)trace->nodes.size(); ref++) {
			IRNode & node = trace->nodes[ref];
			if(node.op == IROpCode::seq) {
				if(node.isDouble()) {
					Constant initial(node.sequence.da, node.sequence.db);
					Operand o_initial = PushConstant(initial);
					asm_.movdqa(xmm0, o_initial);
					asm_.movq(rdi,vector_index);
					EmitCall((void*)sequenceStart_d); 
					asm_.movdqa(Operand(rsp, stackOffset), xmm0);
				} else {
					Constant initial(node.sequence.ia, node.sequence.ib);
					Operand o_initial = PushConstant(initial);
					asm_.movdqa(xmm0, o_initial);
					asm_.movq(rdi,vector_index);
					EmitCall((void*)sequenceStart_i); 
					asm_.movdqa(Operand(rsp, stackOffset), xmm0);
				}
				stackOffset += 0x10;
			}
			else if(node.op == IROpCode::rep) {
				Constant initial(node.sequence.ia, node.sequence.ib);
				Operand o_initial = PushConstant(initial);
				asm_.movdqa(xmm0, o_initial);
				asm_.movq(rdi,vector_index);
				EmitCall((void*)repeatEach_i); 
				asm_.movdqa(Operand(rsp, stackOffset), xmm0);
				stackOffset += 0x10;
				asm_.movdqa(xmm0, o_initial);
				asm_.movq(rdi,vector_index);
				EmitCall((void*)repeatStart_i); 
				asm_.movdqa(Operand(rsp, stackOffset), xmm0);
				stackOffset += 0x10;
			}
			else if(node.op == IROpCode::random) {
				asm_.movq(Operand(rsp, stackOffset), thread_index);
				asm_.movq(Operand(rsp, stackOffset+0x8), thread_index);
				stackOffset += 0x10;
			}
			else if(node.group == IRNode::FOLD) { 
				int64_t step = node.in.length / thread.state.nThreads;
				asm_.movq(r11, Immediate(step));
				asm_.imulq(r11, thread_index);
				asm_.movq(Operand(rsp, stackOffset), r11);
				if(node.shape.levels <= BIG_CARDINALITY)
					asm_.addq(r11, Immediate(1));
				asm_.movq(Operand(rsp, stackOffset+0x8), r11);
				stackOffset += 0x10;
			}
		}

		// clear register assignments
		for(size_t i = 0; i < trace->nodes.size(); i++) {
			allocated_register[i] = -1;
		}
		
		Label begin;

		asm_.bind(&begin);

		stackOffset = spills*0x10;
		for(IRef ref = 0; ref < (int64_t)trace->nodes.size(); ref++) {
			IRNode & node = trace->nodes[ref];


			// unspill if necessary...
			if(node.group != IRNode::SCALAR) {
				switch(node.arity) {
					case IRNode::TRINARY: 
						unspill(assignment[ref].c, allocated_register[node.trinary.c]);
					case IRNode::BINARY:
						unspill(assignment[ref].b, allocated_register[node.binary.b]);
					case IRNode::UNARY:
						unspill(assignment[ref].a, allocated_register[node.unary.a]);
					default:
						if(node.shape.filter >= 0)
							unspill(assignment[ref].f, allocated_register[node.shape.filter]);
						if(node.shape.split >= 0)
							unspill(assignment[ref].s, allocated_register[node.shape.split]);
				}
				allocated_register[ref] = assignment[ref].r.r;
			}
			switch(node.op) {

			case IROpCode::constant: {
				Constant c;
				switch(node.type) {
					case Type::Integer: c = Constant(node.constant.i); break;
					case Type::Logical: c = Constant(node.constant.l); break;
					case Type::Double:  c = Constant(node.constant.d); break;
					default: _error("unexpected type");
				}
				asm_.movdqa(RegR(ref),PushConstant(c));
			} break;
			case IROpCode::load: {
				void* p;
				if(node.in.isLogical())
					p = ((Logical&)node.in).v();
				else if(node.in.isInteger())
					p = ((Integer&)node.in).v();
				else if(node.in.isDouble())
					p = ((Double&)node.in).v();
				//printf("load: %x\n", p);
				int64_t offset = node.constant.i;
				if(offset % 2 == 0) {
					if(node.isLogical())
						asm_.pmovsxbq(RegR(ref), EncodeOperand((char*)p+offset, vector_index, times_1));
					else
						asm_.movdqa(RegR(ref),EncodeOperand((char*)p+offset*8,vector_index,times_8));
				} else {
					if(node.isLogical())
						_error("NYI: unaligned load of logical");
					else
						asm_.movdqu(RegR(ref),EncodeOperand((char*)p+offset*8,vector_index,times_8));
				}
			} break;
			case IROpCode::gather: {
				void* p;
				if(node.in.isLogical()) {
					p = ((Logical&)node.in).v();
					_error("NYI: gather of logical");
				} else {
					if(node.in.isInteger())
						p = ((Integer&)node.in).v();
					else if(node.in.isDouble())
						p = ((Double&)node.in).v();
			
					asm_.movq(r8, RegA(ref));
					asm_.movhlps(RegR(ref), RegA(ref));
					asm_.movq(r9, RegR(ref));
					asm_.movlpd(RegR(ref),EncodeOperand(p,r8,times_8));
					asm_.movhpd(RegR(ref),EncodeOperand(p,r9,times_8));
				}
			} break;

			case IROpCode::add: {
				if(node.isDouble()) 	asm_.addpd(MoveA2R(ref),RegB(ref)); 
				else		 	asm_.paddq(MoveA2R(ref),RegB(ref));
			} break;
			case IROpCode::addc: {
				if(node.isDouble()) 	asm_.addpd(MoveA2R(ref),PushConstant(Constant(node.constant.d))); 
				else 			asm_.paddq(MoveA2R(ref),PushConstant(Constant(node.constant.i)));
			} break;
			case IROpCode::sub: {
				if(node.isDouble()) 	asm_.subpd(MoveA2R(ref),RegB(ref)); 
				else		 	asm_.psubq(MoveA2R(ref),RegB(ref));
			} break;
			case IROpCode::mul: {
				if(node.isDouble()) 	asm_.mulpd(MoveA2R(ref),RegB(ref)); 
				else			EmitVectorizedBinaryFunction(ref,mul_i);
			} break;
			case IROpCode::mulc: {
				if(node.isDouble()) 	asm_.mulpd(MoveA2R(ref),PushConstant(Constant(node.constant.d))); 
				else {
					EmitVectorizedUCFunction(ref,(void*)mul_i);
				}
			} break;
			case IROpCode::div: 	asm_.divpd(MoveA2R(ref),RegB(ref)); break;
			case IROpCode::idiv: {
				if(node.isDouble()) {	
					asm_.divpd(MoveA2R(ref),RegB(ref)); 
					asm_.roundpd(RegR(ref), RegR(ref), Assembler::kRoundDown);	
				} else	{ 
					EmitVectorizedBinaryFunction(ref,idiv_i);
				}
			} break;

			case IROpCode::pmin: {
				if(node.isDouble()) {
					asm_.minpd(MoveA2R(ref), RegB(ref));
				} else {
					_error("NYI: pmin on integers");
				}
			} break;

			case IROpCode::sqrt: 	asm_.sqrtpd(RegR(ref),RegA(ref)); break;
			//case IROpCode::round:	asm_.roundpd(RegR(ref),RegA(ref), Assembler::kRoundToNearest); break;
			case IROpCode::floor: 	asm_.roundpd(RegR(ref),RegA(ref), Assembler::kRoundDown); break;
			case IROpCode::ceiling:	asm_.roundpd(RegR(ref),RegA(ref), Assembler::kRoundUp); break;
			case IROpCode::trunc: 	asm_.roundpd(RegR(ref),RegA(ref), Assembler::kRoundToZero); break;
			case IROpCode::abs: {
				if(node.isDouble()) {
					asm_.andpd(MoveA2R(ref), ConstantTable(C_ABS_MASK));
				} else {			
					//this could be inlined using bit-whacking, but we would need an additional register
					// if(r < 0) f = ~0 else 0; r = r xor f; r -= f;
					EmitVectorizedUnaryFunction(ref,abs_i); 
				}
			} break;
			case IROpCode::neg: {
				if(node.isDouble()) {	
					asm_.xorpd(MoveA2R(ref), ConstantTable(C_NEG_MASK));	
				} else {
					asm_.pxor(MoveA2R(ref),ConstantTable(C_NOT_MASK)); //r = ~r
					asm_.psubq(RegR(ref),ConstantTable(C_NOT_MASK)); //r -= -1	
				}
			} break;
			case IROpCode::pos: MoveA2R(ref); break;
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
			case IROpCode::exp: 	EmitVectorizedUnaryFunction(ref,exp_d); break;
			case IROpCode::log: 	EmitVectorizedUnaryFunction(ref,log_d); break;
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
			case IROpCode::isna:
				asm_.pcmpeqq(MoveA2R(ref), ConstantTable(C_DOUBLE_NA)); break;
			case IROpCode::sign: 	EmitVectorizedUnaryFunction(ref,sign_d); break;
			case IROpCode::mod: {
				if(node.isDouble()) {
					EmitMove(xmm15, RegA(ref));
					asm_.divpd(xmm15, RegB(ref)); 
					asm_.roundpd(xmm15, xmm15, Assembler::kRoundDown);
					asm_.mulpd(xmm15, RegB(ref));
					asm_.subpd(MoveA2R(ref), xmm15);
				} else {
					EmitVectorizedBinaryFunction(ref, mod_i);
				}
			} break;
			
			case IROpCode::eq: EmitCompare(ref,Assembler::kEQ); break;
			case IROpCode::lt: EmitCompare(ref,Assembler::kLT); break;
			case IROpCode::le: EmitCompare(ref,Assembler::kLE); break;
			case IROpCode::neq: EmitCompare(ref,Assembler::kNEQ); break;

			case IROpCode::land: asm_.pand(RegR(ref),RegB(ref)); break;
			case IROpCode::lor: asm_.por(RegR(ref),RegB(ref)); break;
			case IROpCode::lnot: asm_.pxor(MoveA2R(ref), ConstantTable(C_NOT_MASK)); break;
			
			case IROpCode::cast: {
				if(node.type == trace->nodes[node.unary.a].type)
					MoveA2R(ref);
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
			case IROpCode::rep: {
				//TODO: Make these faster
				Operand maxEach = PushConstant(Constant(node.sequence.ib, node.sequence.ib));
				Operand maxN = PushConstant(Constant(node.sequence.ia, node.sequence.ia));
				if(node.sequence.ib == 1) {
					// if each=1, no need to update
					asm_.movdqa(RegR(ref), Operand(rsp, stackOffset+0x10));
					for(int i = 0; i < 2; i++) {
						asm_.paddq(RegR(ref), ConstantTable(C_INTEGER_ONE));
						asm_.movdqa(xmm15, maxN);
						asm_.pcmpeqq(xmm15, RegR(ref));
						asm_.pxor(xmm15, ConstantTable(C_NOT_MASK));
						asm_.pand(RegR(ref), xmm15);
					}
					asm_.movdqa(Operand(rsp, stackOffset+0x10), RegR(ref));
				}
				else if(node.sequence.ia*node.sequence.ib >= node.shape.length) {
					// if j will never wrap, no need to check
					asm_.movdqa(RegR(ref), Operand(rsp, stackOffset));
					for(int i = 0; i < 2; i++) {
						asm_.paddq(RegR(ref), ConstantTable(C_INTEGER_ONE));
						asm_.movdqa(xmm15, maxEach);
						asm_.pcmpeqq(xmm15, RegR(ref));
						asm_.pxor(xmm15, ConstantTable(C_NOT_MASK));
						asm_.pand(RegR(ref), xmm15);
						// load j and increment masked
						asm_.pxor(xmm15, ConstantTable(C_NOT_MASK));
						asm_.pand(xmm15, ConstantTable(C_INTEGER_ONE));
						asm_.paddq(xmm15, Operand(rsp, stackOffset+0x10));
						asm_.movdqa(Operand(rsp, stackOffset+0x10), xmm15);
					}
					asm_.movdqa(Operand(rsp, stackOffset), RegR(ref));
					asm_.movapd(RegR(ref), xmm15);
				}
				else {
					for(int i = 0; i < 2; i++) {
						// update each
						asm_.movdqa(RegR(ref), Operand(rsp, stackOffset));
						asm_.paddq(RegR(ref), ConstantTable(C_INTEGER_ONE));
						asm_.movdqa(xmm15, maxEach);
						asm_.pcmpeqq(xmm15, RegR(ref));
						asm_.pxor(xmm15, ConstantTable(C_NOT_MASK));
						asm_.pand(RegR(ref), xmm15);
						asm_.movdqa(Operand(rsp, stackOffset), RegR(ref));
						// load j and increment masked
						asm_.pxor(xmm15, ConstantTable(C_NOT_MASK));
						asm_.pand(xmm15, ConstantTable(C_INTEGER_ONE));
						asm_.paddq(xmm15, Operand(rsp, stackOffset+0x10));
						// compare j to n and conditionally set to 0
						asm_.movdqa(RegR(ref), maxN);
						asm_.pcmpeqq(RegR(ref), xmm15);
						asm_.pxor(RegR(ref), ConstantTable(C_NOT_MASK));
						asm_.pand(RegR(ref), xmm15);
						// store j
						asm_.movdqa(Operand(rsp, stackOffset+0x10), RegR(ref));
					}
				}
				stackOffset += 0x20;
			} break;
			case IROpCode::seq: {
				if(node.isDouble()) {
					Constant step(2*node.sequence.db, 2*node.sequence.db);
					Operand o_step = PushConstant(step);
					asm_.movdqa(RegR(ref), Operand(rsp, stackOffset));
					asm_.addpd(RegR(ref), o_step);
					asm_.movdqa(Operand(rsp, stackOffset), RegR(ref));
				} else {
					Constant step(2*node.sequence.ib, 2*node.sequence.ib);
					Operand o_step = PushConstant(step);
					asm_.movdqa(RegR(ref), Operand(rsp, stackOffset));
					asm_.paddq(RegR(ref), o_step);
					asm_.movdqa(Operand(rsp, stackOffset), RegR(ref));
				}
				stackOffset += 0x10;
			} break;
			case IROpCode::random: {
				asm_.movdqa(RegR(ref),Operand(rsp, stackOffset));
				SaveRegisters(ref);
				asm_.movapd(xmm0, RegR(ref));
				EmitCall((void*)random_d);
				EmitMove(RegR(ref),xmm0);
				RestoreRegisters(ref);
				stackOffset += 0x10;
			} break;
			case IROpCode::sum:  {
				// relying on doubles and integers to be the same length
				memset(node.in.p, 0, node.in.length*sizeof(double));
				//printf("sum intermediate: %x\n", node.in.p);	
				MoveA2R(ref);
				if(node.shape.filter >= 0) {
					asm_.pand(RegR(ref), RegF(ref));
				}

				Operand offset = Operand(rsp, stackOffset);
				if(node.shape.split >= 0 && node.shape.levels > 1) {
					if(qq == 0) {
					asm_.movapd(xmm15, RegS(ref));
					if(node.shape.levels <= BIG_CARDINALITY)
						asm_.paddq(xmm15, xmm15);
					asm_.paddq(xmm15, offset);
					asm_.movq(r8, xmm15);
					asm_.movhlps(xmm15, xmm15);
					asm_.movq(r9, xmm15);
					qq++;
					}
					Operand operand0 = EncodeOperand(node.in.p, r8, times_8);
					Operand operand1 = EncodeOperand(node.in.p, r9, times_8);
				
					if(node.shape.levels > BIG_CARDINALITY) {
						asm_.movhlps(xmm15, RegR(ref));
						if(node.isDouble())	asm_.addsd(RegR(ref), operand0);
						else {
							asm_.movq(xmm14, operand0);
							asm_.paddq(RegR(ref), xmm14);
						}
						asm_.movq(operand0, RegR(ref));
						if(node.isDouble())	asm_.addsd(xmm15, operand1);
						else {
							asm_.movq(xmm14, operand1);
							asm_.paddq(xmm15, xmm14);
						}
						asm_.movq(operand1, xmm15);
						asm_.movlhps(RegR(ref), xmm15);
					} else {
						asm_.movlpd(xmm15, operand0);
						asm_.movhpd(xmm15, operand1);
						if(node.isDouble())	asm_.addpd(RegR(ref), xmm15);
						else 			asm_.paddq(RegR(ref), xmm15);
						asm_.movlpd(operand0, RegR(ref));
						asm_.movhpd(operand1, RegR(ref));
					}
				} else {
					asm_.movq(r8, offset);
					Operand operand = EncodeOperand(node.in.p, r8, times_8);
					if(node.isDouble()) 	asm_.addpd(RegR(ref), operand);
					else			asm_.paddq(RegR(ref), operand);
					asm_.movdqa(operand, RegR(ref));
				}
				stackOffset += 0x10;
			} break;

			case IROpCode::length: {
				// relying on doubles and integers to be the same length
				memset(node.in.p, 0, node.in.length*sizeof(double));
				
				Operand offset = Operand(rsp, stackOffset);

				if(node.isInteger())
					asm_.movdqa(RegR(ref), ConstantTable(C_INTEGER_ONE));
				else 	
					asm_.movdqa(RegR(ref), ConstantTable(C_DOUBLE_ONE));

				if(node.shape.filter >= 0) {
					asm_.pand(RegR(ref), RegF(ref));
				}
				
				if(node.shape.split >= 0 && node.shape.levels > 1) {
					asm_.movapd(xmm15, RegS(ref));
					if(node.shape.levels <= BIG_CARDINALITY)
						asm_.paddq(xmm15, xmm15);
					asm_.paddq(xmm15, offset);
					asm_.movq(r8, xmm15);
					asm_.movhlps(xmm15, xmm15);
					asm_.movq(r9, xmm15);
					Operand operand0 = EncodeOperand(node.in.p, r8, times_8);
					Operand operand1 = EncodeOperand(node.in.p, r9, times_8);
				
					if(node.shape.levels <= BIG_CARDINALITY) {
						asm_.movhlps(xmm15, RegR(ref));
						if(node.isDouble())	asm_.addsd(RegR(ref), operand0);
						else {
							asm_.movq(xmm14, operand0);
							asm_.paddq(RegR(ref), xmm14);
						}
						asm_.movq(operand0, RegR(ref));
						if(node.isDouble())	asm_.addsd(xmm15, operand1);
						else {
							asm_.movq(xmm14, operand1);
							asm_.paddq(xmm15, xmm14);
						}
						asm_.movq(operand1, xmm15);
						asm_.movlhps(RegR(ref), xmm15);
					} else {
						asm_.movlpd(RegR(ref), operand0);
						asm_.movhpd(RegR(ref), operand1);
						if(node.isDouble())	asm_.addpd(RegR(ref), operand0);
						else 			asm_.paddq(RegR(ref), operand1);
						asm_.movlpd(operand0, RegR(ref));
						asm_.movhpd(operand1, RegR(ref));
					}
				} else {
					asm_.movq(r8, offset);
					Operand operand = EncodeOperand(node.in.p, r8, times_8);
					if(node.isDouble()) 	asm_.addpd(RegR(ref), operand);
					else			asm_.paddq(RegR(ref), operand);
					asm_.movdqa(operand, RegR(ref));
				}
				stackOffset += 0x10;
			} break;

			case IROpCode::mean: {
				// relying on doubles and integers to be the same length
				memset(node.in.p, 0, node.in.length*sizeof(double));
				
				// m' = m + 1/n * (x-m)
				// (x-m) must be in RegR at the end

				Operand offset = Operand(rsp, stackOffset);
				
				MoveA2R(ref);		// x
				
				if(node.shape.split >= 0 && node.shape.levels > 1) {
					asm_.movapd(xmm15, RegS(ref));
					if(node.shape.levels <= BIG_CARDINALITY)
						asm_.paddq(xmm15, xmm15);
					asm_.paddq(xmm15, offset);
					asm_.movq(r8, xmm15);
					asm_.movhlps(xmm15, xmm15);
					asm_.movq(r9, xmm15);
					Operand operand0 = EncodeOperand(node.in.p, r8, times_8);
					Operand operand1 = EncodeOperand(node.in.p, r9, times_8);
				
					if(node.shape.levels > BIG_CARDINALITY) {
						asm_.movhlps(xmm15, RegR(ref));

						asm_.subsd(RegR(ref), operand0);
						asm_.movapd(xmm14, RegR(ref));
						asm_.mulsd(xmm14, RegB(ref));
						if(node.shape.filter >= 0) {
							asm_.pand(xmm14, RegF(ref));
						}
						asm_.addsd(xmm14, operand0);
						asm_.movq(operand0, xmm14);

						asm_.movhlps(xmm14, RegB(ref));
						asm_.subsd(xmm15, operand1);
						asm_.movlhps(RegR(ref), xmm15);
						asm_.mulsd(xmm15, xmm14);
						if(node.shape.filter >= 0) {
							asm_.pand(xmm15, RegF(ref));
						}
						asm_.addsd(xmm15, operand1);
						asm_.movq(operand1, xmm15);
						asm_.movlhps(RegR(ref), xmm15);
					} else {
						asm_.movlpd(xmm14, operand0);
						asm_.movhpd(xmm14, operand1);

						asm_.subpd(RegR(ref), xmm14);
						asm_.movapd(xmm15, RegR(ref));
						asm_.mulpd(xmm15, RegB(ref));
						if(node.shape.filter >= 0) {
							asm_.pand(xmm15, RegF(ref));
						}
						asm_.addpd(xmm15, xmm14);
						asm_.movlpd(operand0, xmm15);
						asm_.movhpd(operand1, xmm15);
					}
				} else {
					asm_.movq(r8, offset);
					Operand operand = EncodeOperand(node.in.p, r8, times_8);
					asm_.subpd(RegR(ref), operand);
					asm_.movapd(xmm15, RegR(ref));
					asm_.mulpd(xmm15, RegB(ref));
					if(node.shape.filter >= 0) {
						asm_.pand(xmm15, RegF(ref));
					}
					asm_.addpd(xmm15, operand);
					asm_.movdqa(operand, xmm15);
				}
				stackOffset += 0x10;
			} break;

			case IROpCode::cm2: {
				// c' = c + (n-1)/n * (s-m1) * (t-m2)
				// (s-m1) is in a, (t-m2) is in b, 1/n is in c
				// compute as c' = c + (1-1/n)*(s-m1)*(t-m2) 
				memset(node.in.p, 0, node.in.length*sizeof(double));
				
				Operand offset = Operand(rsp, stackOffset);
				
				MoveA2R(ref);		// (s-m1)
				asm_.mulpd(RegR(ref), RegB(ref));
				asm_.movdqa(xmm15, ConstantTable(C_DOUBLE_ONE));
				asm_.subpd(xmm15, RegC(ref));
				asm_.mulpd(RegR(ref), xmm15);
				if(node.shape.filter >= 0) {
					asm_.pand(RegR(ref), RegF(ref));
				}

				if(node.shape.split >= 0 && node.shape.levels > 1) {
					asm_.movapd(xmm15, RegS(ref));
					if(node.shape.levels <= BIG_CARDINALITY)
						asm_.paddq(xmm15, xmm15);
					asm_.paddq(xmm15, offset);
					asm_.movq(r8, xmm15);
					asm_.movhlps(xmm15, xmm15);
					asm_.movq(r9, xmm15);
					Operand operand0 = EncodeOperand(node.in.p, r8, times_8);
					Operand operand1 = EncodeOperand(node.in.p, r9, times_8);
				
					if(node.shape.levels > BIG_CARDINALITY) {
						asm_.movhlps(xmm15, RegR(ref));
						asm_.addsd(RegR(ref), operand0);
						asm_.movq(operand0, RegR(ref));
						asm_.addsd(xmm15, operand1);
						asm_.movq(operand1, xmm15);
					} else {
						asm_.movlpd(xmm15, operand0);
						asm_.movhpd(xmm15, operand1);
						asm_.addpd(xmm15, RegR(ref));
						asm_.movlpd(operand0, xmm15);
						asm_.movhpd(operand1, xmm15);
					}
				} else {
					asm_.movq(r8, offset);
					Operand operand = EncodeOperand(node.in.p, r8, times_8);
					asm_.addpd(RegR(ref), operand);
					asm_.movdqa(operand, RegR(ref));
				}
				stackOffset += 0x10;
			} break;
			
			case IROpCode::min:  {
				for(int64_t i = 0; i < node.in.length; i++)
					((double*)node.in.p)[i] = std::numeric_limits<double>::infinity();
				
				Operand offset = Operand(rsp, stackOffset);
				XMMRegister index = no_xmm;
				
				MoveA2R(ref);
				if(node.shape.filter >= 0) {
					asm_.pand(RegR(ref), RegF(ref));
					asm_.pxor(EmitMove(xmm14, RegF(ref)), ConstantTable(C_NOT_MASK));
					asm_.pand(xmm14, ConstantTable(C_DOUBLE_MAX));
					asm_.por(RegR(ref), xmm14);
				}

				if(node.shape.split >= 0 && node.shape.levels > 1) {
					asm_.movapd(xmm15, RegS(ref));
					if(node.shape.levels <= BIG_CARDINALITY)
						asm_.paddq(xmm15, xmm15);
					asm_.paddq(xmm15, offset);
					asm_.movq(r8, xmm15);
					asm_.movhlps(xmm15, xmm15);
					asm_.movq(r9, xmm15);
					Operand operand0 = EncodeOperand(node.in.p, r8, times_8);
					Operand operand1 = EncodeOperand(node.in.p, r9, times_8);
				
					if(node.shape.levels <= BIG_CARDINALITY) {
						asm_.movhlps(xmm15, RegR(ref));
						if(node.isDouble())	asm_.minsd(RegR(ref), operand0);
						else			_error("NYI: min on integers");
						asm_.movq(operand0, RegR(ref));
						if(node.isDouble())	asm_.minsd(xmm15, operand1);
						else 			_error("NYI: min on integers");
						asm_.movq(operand1, xmm15);
						asm_.movlhps(RegR(ref), xmm15);
					} else {
						asm_.movlpd(RegR(ref), operand0);
						asm_.movhpd(RegR(ref), operand1);
						if(node.isDouble())	asm_.minpd(RegR(ref), operand0);
						else 			_error("NYI: min on integers");
						asm_.movlpd(operand0, RegR(ref));
						asm_.movhpd(operand1, RegR(ref));
					}
				} else {
					asm_.movq(r8, offset);
					Operand operand = EncodeOperand(node.in.p, r8, times_8);
					if(node.isDouble()) 	asm_.minpd(RegR(ref), operand);
					else			_error("NYI: min on integers");
					asm_.movdqa(operand, RegR(ref));
				}
				stackOffset += 0x10;
			} break;

			case IROpCode::max:  {
				// relying on doubles and integers to be the same length
				for(int64_t i = 0; i < node.in.length; i++)
					((double*)node.in.p)[i] = -std::numeric_limits<double>::infinity();
				
				Operand offset = Operand(rsp, stackOffset);
				XMMRegister index = no_xmm;
				
				MoveA2R(ref);
				if(node.shape.filter >= 0) {
					asm_.pand(RegR(ref), RegF(ref));
					asm_.pxor(EmitMove(xmm14, RegF(ref)), ConstantTable(C_NOT_MASK));
					asm_.pand(xmm14, ConstantTable(C_DOUBLE_MIN));
					asm_.por(RegR(ref), xmm14);
				}

				if(node.shape.split >= 0 && node.shape.levels > 1) {
					asm_.movapd(xmm15, RegS(ref));
					if(node.shape.levels <= BIG_CARDINALITY)
						asm_.paddq(xmm15, xmm15);
					asm_.paddq(xmm15, offset);
					asm_.movq(r8, xmm15);
					asm_.movhlps(xmm15, xmm15);
					asm_.movq(r9, xmm15);
					Operand operand0 = EncodeOperand(node.in.p, r8, times_8);
					Operand operand1 = EncodeOperand(node.in.p, r9, times_8);
				
					if(node.shape.levels <= BIG_CARDINALITY) {
						asm_.movhlps(xmm15, RegR(ref));
						if(node.isDouble())	asm_.maxsd(RegR(ref), operand0);
						else			_error("NYI: max on integers");
						asm_.movq(operand0, RegR(ref));
						if(node.isDouble())	asm_.maxsd(xmm15, operand1);
						else 			_error("NYI: max on integers");
						asm_.movq(operand1, xmm15);
						asm_.movlhps(RegR(ref), xmm15);
					} else {
						asm_.movlpd(RegR(ref), operand0);
						asm_.movhpd(RegR(ref), operand1);
						if(node.isDouble())	asm_.maxpd(RegR(ref), operand0);
						else 			_error("NYI: max on integers");
						asm_.movlpd(operand0, RegR(ref));
						asm_.movhpd(operand1, RegR(ref));
					}
				} else {
					asm_.movq(r8, offset);
					Operand operand = EncodeOperand(node.in.p, r8, times_8);
					if(node.isDouble()) 	asm_.maxpd(RegR(ref), operand);
					else			_error("NYI: max on integers");
					asm_.movdqa(operand, RegR(ref));
				}
				stackOffset += 0x10;
			} break;

			case IROpCode::prod: { 
				if(node.isDouble()) {
					/*assert(store_inst[ref] != NULL);
					IRNode & str = *store_inst[ref];
					for(uint64_t i = 0; i < 128; i++)
						((double*)str.store.dst.p)[i] = 1.0;
					Operand op = EncodeOperand(str.store.dst.p, thread_index, times_8);
					asm_.mulpd(RegR(ref),op);
					asm_.movdqa(op,RegR(ref));*/
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
				MoveA2R(ref);
				// subsplit if necessary...
				if(node.shape.split >= 0) {
					/*
					levels *= nodes[x].shape.levels;
					IRef e = EmitBinary(IROpCode::mul, Type::Integer, f, EmitConstant(Type::Integer, 1, levels), 0);
					f = EmitBinary(IROpCode::add, Type::Integer, e, f, 0);
					*/
					_error("NYI: subsplitting");
				}
				break;

			case IROpCode::filter:
				MoveA2R(ref);
				if(node.shape.filter >= 0)
					asm_.pand(RegR(ref),RegF(ref));
			break;

			case IROpCode::ifelse:
				if(RegA(ref).is(xmm0)) {
					EmitMove(xmm14, xmm0);
					EmitMove(xmm15, RegA(ref));
					EmitMove(xmm0, RegC(ref));
					asm_.blendvpd(xmm15, RegB(ref));
					EmitMove(RegR(ref), xmm15);
					if(!RegR(ref).is(xmm0))
						EmitMove(xmm0, xmm14);
				} else if(RegB(ref).is(xmm0)) {
					EmitMove(xmm14, xmm0);
					EmitMove(xmm15, RegA(ref));
					EmitMove(RegA(ref), RegB(ref));
					EmitMove(xmm0, RegC(ref));
					asm_.blendvpd(xmm15, RegA(ref));
					EmitMove(RegR(ref), xmm15);
					if(!RegR(ref).is(xmm0))
						EmitMove(xmm0, xmm14);
				} else if(RegC(ref).is(xmm0)) {
					EmitMove(RegR(ref), RegA(ref));
					asm_.blendvpd(RegR(ref), RegB(ref));
				} else {
					EmitMove(xmm14, xmm0);
					EmitMove(xmm0, RegC(ref));
					EmitMove(RegR(ref), RegA(ref));
					asm_.blendvpd(RegR(ref), RegB(ref));
					if(!RegR(ref).is(xmm0))
						EmitMove(xmm0, xmm14);
				}
			break;

			case IROpCode::sload:
			case IROpCode::sstore:
			break;

			//case IROpCode::signif:
			default:	_error("unimplemented op"); break;
		
			}

			if(node.liveOut) {
				switch(node.group) {
					case IRNode::MAP:
					case IRNode::GENERATOR: {
						if(Type::Logical == node.type)
							EmitLogicalStore(ref, node.out, node.shape);
						else
							EmitVectorStore(ref, node.out, node.shape);
					} break;
					default:
						// do nothing...
					break;
				}
			}


			// spill if necessary...
			if(node.group != IRNode::SCALAR) {
				switch(node.arity) {
					case IRNode::TRINARY: 
						spill(assignment[ref].c, allocated_register[node.trinary.c]);
					case IRNode::BINARY:
						spill(assignment[ref].b, allocated_register[node.binary.b]);
					case IRNode::UNARY:
						spill(assignment[ref].a, allocated_register[node.unary.a]);
					default:
						if(node.shape.filter >= 0)
							spill(assignment[ref].f, allocated_register[node.shape.filter]);
						if(node.shape.split >= 0)
							spill(assignment[ref].s, allocated_register[node.shape.split]);

						spill(assignment[ref].r, allocated_register[ref]);
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
		if(next_constant_slot > 8192) {
			printf("Used up all the constants!: %d\n", next_constant_slot);
		}
		trace->code_buffer->constant_table[offset] = data;
		next_constant_slot++;
		return offset;
	}
	//slow path for Folds that don't have inline assembly
	void EmitFoldFunction(IRef ref, void * fn, const Constant& identity) {
		uint64_t offset = PushConstantOffset(identity);
		SaveRegisters(ref);
		EmitMove(xmm0,RegA(ref));
		asm_.movq(rdi,(void*)&trace->code_buffer->constant_table[offset]);
		EmitCall(fn);
		EmitMove(RegR(ref),xmm0);
		RestoreRegisters(ref);
	}

	// put r0 in xmm0 and r1 in xmm1 being careful that they might overlap
	void Arguments2(XMMRegister r0, XMMRegister r1) {
		if(r1.is(xmm0)) EmitMove(xmm15, xmm0);
		EmitMove(xmm0, r0);
		if(r1.is(xmm0)) EmitMove(xmm1, xmm15);
		else 		EmitMove(xmm1, r1);
	}

	void EmitVectorStore(IRef ref, Value& dst, IRNode::Shape const& shape) {
		XMMRegister src = RegR(ref);
		if(shape.filter < 0)
			asm_.movdqa(EncodeOperand(dst.p,vector_index,times_8),src);
		else {
			dst.length = 0;
			XMMRegister filter = RegF(ref);
			
			SaveRegisters(ref);
			Arguments2(src, filter);
			asm_.movq(rdi,&dst);
			EmitCall((void*)store_conditional);
			EmitMove(RegR(ref),xmm0);
			RestoreRegisters(ref);
		}
	}

	void EmitLogicalStore(IRef ref, Value& dst, IRNode::Shape const& shape) {
		XMMRegister src = RegR(ref);
                if(shape.filter < 0) {
			asm_.pshufb(src,ConstantTable(C_PACK_LOGICAL));
			asm_.movq(rbx, src);
			asm_.movw(EncodeOperand(dst.p,vector_index,times_1),rbx);
       	         	asm_.pshufb(src,ConstantTable(C_PACK_LOGICAL));
		} else {
			dst.length = 0;
			XMMRegister filter = RegF(ref);
			
			SaveRegisters(ref);
			Arguments2(src, filter);
			asm_.movq(rdi,&dst);
			EmitCall((void*)store_conditional_l);
			EmitMove(RegR(ref),xmm0);
			RestoreRegisters(ref);
		}
	}

	void EmitVectorizedUnaryFunction(IRef ref, __m128d (*fn)(__m128d)) {
		EmitVectorizedUnaryFunction(ref,(void*)fn);
	}
	void EmitVectorizedUnaryFunction(IRef ref, void * fn) {

		SaveRegisters(ref);

		EmitMove(xmm0,RegA(ref));
		EmitCall(fn);
		EmitMove(RegR(ref),xmm0);

		RestoreRegisters(ref);

	}

	void EmitVectorizedBinaryFunction(IRef ref, __m128d (*fn)(__m128d,__m128d)) {
		EmitVectorizedBinaryFunction(ref,(void*)fn);
	}
	void EmitVectorizedBinaryFunction(IRef ref, void * fn) {

		SaveRegisters(ref);

		EmitMove(xmm0, RegA(ref));
		EmitMove(xmm1, RegB(ref));
		EmitCall(fn);
		EmitMove(RegR(ref),xmm0);

		RestoreRegisters(ref);

	}

	void EmitVectorizedUCFunction(IRef ref, void * fn) {
		SaveRegisters(ref);

		EmitMove(xmm0, RegA(ref));
		asm_.movdqa(xmm1, PushConstant(Constant(trace->nodes[ref].constant.i)));
		EmitCall(fn);
		EmitMove(RegR(ref),xmm0);

		RestoreRegisters(ref);

	}
	void EmitUnaryFunction(IRef ref, double (*fn)(double)) {
		EmitUnaryFunction(ref,(void*)fn);
	}
	void EmitUnaryFunction(IRef ref, void * fn) {

		SaveRegisters(ref);

		EmitMove(xmm0, RegA(ref));
		asm_.movapd(xmm1,xmm0);
		asm_.unpckhpd(xmm1,xmm1);
		asm_.movq(load_addr,xmm1);//we need the high value for the second call
		EmitCall(fn);
		asm_.movq(rbx,xmm0);
		asm_.movq(xmm0,load_addr);
		EmitCall(fn);
		asm_.movq(xmm1,rbx);
		asm_.unpcklpd(xmm1,xmm0);
		EmitMove(RegR(ref),xmm1);

		RestoreRegisters(ref);
	}
	void EmitBinaryFunction(IRef ref, double (*fn)(double,double)) {
		EmitBinaryFunction(ref,(void*)fn);
	}
	void EmitBinaryFunction(IRef ref, void * fn) {
		//this isn't the most optimized way to do this but it works for now
		SaveRegisters(ref);
		asm_.subq(rsp, Immediate(0x20));
		asm_.movdqu(Operand(rsp,0),RegA(ref));
		asm_.movdqu(Operand(rsp,0x10),RegB(ref));
		asm_.movq(xmm0,Operand(rsp,0));
		asm_.movq(xmm1,Operand(rsp,0x10));
		EmitCall(fn);
		asm_.movq(Operand(rsp, 0),xmm0);
		asm_.movq(xmm0,Operand(rsp, 0x8));
		asm_.movq(xmm1,Operand(rsp, 0x18));
		EmitCall(fn);
		asm_.movq(Operand(rsp, 0x8),xmm0);
		asm_.movdqu(RegR(ref),Operand(rsp, 0));
		asm_.addq(rsp, Immediate(0x20));
		RestoreRegisters(ref);
	}

	void EmitStartCall(void * fn) {
		// parameter is already in xmm0
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

	void EmitGather(XMMRegister dest, IRNode const& node, XMMRegister index, Operand offset) {
		// TODO: other fast cases here when index is a known seq
		if(!index.is(no_xmm)) {
			EmitMove(dest, index);
			asm_.paddq(dest, offset);
		
			asm_.movq(r8, dest);
			asm_.movhlps(dest, dest);
			asm_.movq(r9, dest);
			asm_.movlpd(dest, EncodeOperand(node.in.p, r8, times_8));
			asm_.movhpd(dest, EncodeOperand(node.in.p, r9, times_8));
		} else {
			asm_.movq(r8, offset);
			asm_.movdqa(dest, EncodeOperand(node.in.p, r8, times_8));
		}
	}

	void EmitScatter(IRNode const& node, XMMRegister src, XMMRegister index) {
		if(!index.is(no_xmm)) {
			asm_.movlpd(EncodeOperand(node.in.p, r8, times_8), src);
			asm_.movhpd(EncodeOperand(node.in.p, r9, times_8), src);
		}
		else {
			asm_.movdqa(EncodeOperand(node.in.p, r8, times_8), src);
		}
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
					asm_.movdqu(Operand(rsp, index), XMMRegister::FromAllocationIndex(it.value()));
					index += 16;
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
					asm_.movdqu(XMMRegister::FromAllocationIndex(it.value()), Operand(rsp, index));
					index += 16;
			}
			asm_.addq(rsp, Immediate(256));
		}
	}

	void EmitCompare(IRef ref, Assembler::ComparisonType typ) {
		IRNode & node = trace->nodes[ref];
		if(Type::Double == trace->nodes[node.binary.a].type) {
			asm_.cmppd(MoveA2R(ref),RegB(ref),typ);
		} else {
			_error("NYI - integer compare");
		}
	}

	typedef void (*fn) (uint64_t thread_index, uint64_t start, uint64_t end);
	
	static void executebody(void* args, void* h, uint64_t start, uint64_t end, Thread& thread) {
		//printf("%d: called with %d to %d\n", thread.index, start, end);
		fn code = (fn)args;
		code(thread.index, start, end);	
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
			thread.doall(NULL, executebody, (void*)trace_code, 0, trace->Size, 4, 1024); 
			//trace_code(thread.index, 0, trace->length);
			//double s = trace->length / (time_elapsed(begin) * 10e9);
			//printf("elements computed / us: %f\n",s);
		} else {
			thread.doall(NULL, executebody, (void*)trace_code, 0, trace->Size, 4, 1024); 
			//trace_code(thread.index, 0, trace->length);
		}
	}

	void mergeMin(IRNode& node, int64_t i, int64_t j) {
		if(node.isDouble())
			((double*)node.in.p)[i] = std::min(((double*)node.in.p)[i], ((double*)node.in.p)[j]);
		else
			((int64_t*)node.in.p)[i] = std::min(((int64_t*)node.in.p)[i], ((int64_t*)node.in.p)[j]);
	}
	
	void mergeMax(IRNode& node, int64_t i, int64_t j) {
		if(node.isDouble())
			((double*)node.in.p)[i] = std::max(((double*)node.in.p)[i], ((double*)node.in.p)[j]);
		else
			((int64_t*)node.in.p)[i] = std::max(((int64_t*)node.in.p)[i], ((int64_t*)node.in.p)[j]);
	}
	
	// merge value in j into i
	void mergeSum(IRNode& node, int64_t i, int64_t j) {
		//printf("summing (%d=>%d): %f %f\n", j, i, ((double*)node.in.p)[i], ((double*)node.in.p)[j]);
		if(node.isDouble())
			((double*)node.in.p)[i] += ((double*)node.in.p)[j];
		else
			((int64_t*)node.in.p)[i] += ((int64_t*)node.in.p)[j];
	}

	void mergeProd(IRNode& node, int64_t i, int64_t j) {
		if(node.isDouble())
			((double*)node.in.p)[i] *= ((double*)node.in.p)[j];
		else
			((int64_t*)node.in.p)[i] *= ((int64_t*)node.in.p)[j];
	}

	void mergeLength(IRNode& node, int64_t i, int64_t j) {
		if(node.isDouble())
			((double*)node.in.p)[i] += ((double*)node.in.p)[j];
		else
			((int64_t*)node.in.p)[i] += ((int64_t*)node.in.p)[j];
	}

	void mergeMean(IRNode& node, int64_t i, int64_t j) {
		IRNode& b = trace->nodes[trace->nodes[node.binary.b].binary.b];

		/*printf("Merging %d => %d: %f %f %f %f\n", i, j,
			((double*)node.in.p)[i],
			((double*)b.in.p)[i],
			((double*)node.in.p)[j],
			((double*)b.in.p)[j]);
		*/
		// u2 in a, n2 in b
		if(((double*)b.in.p)[i] > 0) {
			double m1 = ((double*)node.in.p)[i];

			((double*)node.in.p)[i] += 
				((double*)b.in.p)[j] / ((double*)b.in.p)[i]
				* (((double*)node.in.p)[j] - ((double*)node.in.p)[i] );

			// put m2-m1 in j for the cov to use
			((double*)node.in.p)[j] -= m1;
		}
	}

	void mergeCm2(IRNode& node, int64_t i, int64_t j) {
		IRNode& a = trace->nodes[node.binary.a];
		IRNode& b = trace->nodes[node.binary.b];
		IRNode& c = trace->nodes[trace->nodes[node.trinary.c].binary.b];

		double d1 = ((double*)a.in.p)[j];
		double d2 = ((double*)b.in.p)[j];
		double nn = (((double*)c.in.p)[i] - ((double*)c.in.p)[j])
				* ((double*)c.in.p)[j] / ((double*)c.in.p)[i];

		if(((double*)c.in.p)[i] > 0) {
			((double*)node.in.p)[i] +=
				((double*)node.in.p)[j] +
				nn * d1 * d2;
		}
	}

	void GlobalReduce(Thread& thread) {
		// merge across vector lanes
		for(IRef ref = 0; ref < (int64_t)trace->nodes.size(); ref++) {
			IRNode & node = trace->nodes[ref];

			if(node.group == IRNode::FOLD && node.outShape.length <= BIG_CARDINALITY) {
				int64_t step = node.in.length/thread.state.nThreads;

				for(int64_t j = 0; j < thread.state.nThreads; j++) {
					for(int64_t i = 0; i < node.outShape.length; i++) {
						int64_t a = j*step+i*2;
						int64_t b = j*step+i*2+1;

						switch(node.op) {
							case IROpCode::sum: 
								mergeSum(node, a, b);
								break;
							case IROpCode::prod:
								mergeProd(node, a, b);
								break;
							case IROpCode::length:
								mergeLength(node, a, b);
								break;
							case IROpCode::mean:
								mergeMean(node, a, b);
								break;
							case IROpCode::cm2:
								mergeCm2(node, a, b);
								break;
							case IROpCode::min:
								mergeMin(node, a, b);
								break;
							case IROpCode::max:
								mergeMax(node, a, b);
								break;
							default: // do nothing
								break;
						}
					}
				}
			}
		}

		// merge across threads
		for(IRef ref = 0; ref < (int64_t)trace->nodes.size(); ref++) {
			IRNode & node = trace->nodes[ref];
	
			if(node.group == IRNode::FOLD) {
				int64_t step = node.in.length/thread.state.nThreads;
				for(int64_t j = 1; j < thread.state.nThreads; j++) {
					for(int64_t i = 0; i < node.outShape.length; i++) {
						int64_t a = 0*step+i;
						int64_t b = j*step+i;
						switch(node.op) {
							case IROpCode::sum: 
								mergeSum(node, a, b);
								break;
							case IROpCode::prod:
								mergeProd(node, a, b);
								break;
							case IROpCode::length:
								mergeLength(node, a, b);
								break;
							case IROpCode::mean:
								mergeMean(node, a, b);
								break;
							case IROpCode::cm2:
								mergeCm2(node, a, b);
								break;
							case IROpCode::min:
								mergeMin(node, a, b);
								break;
							case IROpCode::max:
								mergeMax(node, a, b);
								break;
							default: /* do nothing */ break;
						}
					}
				}
			}
		}

		// TODO: merge filtered vectors!

		// copy to output vector
		for(IRef ref = 0; ref < (int64_t)trace->nodes.size(); ref++) {
			IRNode & node = trace->nodes[ref];

			if(node.op == IROpCode::sload) {
				node.out = node.in;
			} else if(node.op == IROpCode::sstore) {
				Integer index = Integer::c(node.binary.data);
				Subset2Assign(thread, 
					trace->nodes[node.binary.a].out, 
					true, 
					index, 
					trace->nodes[node.binary.b].out, 
					node.out);
			}
			else if(node.group == IRNode::FOLD) {
				if(node.shape.levels <= BIG_CARDINALITY) {
					if(node.isDouble()) {
						Double& d = (Double&)node.out;
						for(int64_t i = 0, j = 0; i < node.outShape.length; i++, j+=2) {
							d[i] = ((Double&)node.in)[j];
						}
					}
					else if(node.isInteger()) {
						Integer& d = (Integer&)node.out;
						for(int64_t i = 0, j = 0; i < node.outShape.length; i++, j+=2) {
							d[i] = ((Integer&)node.in)[j];
						}
					}
					else {
						_error("NYI");
					}
				}
				else {
					node.out = node.in;
					node.out.length = node.outShape.length;
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
