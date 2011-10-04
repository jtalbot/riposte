#include "interpreter.h"
#include "vector.h"
#include "ops.h"
#include <sys/mman.h>
#include "assembler-x64.h"
#include <math.h>

using namespace v8::internal;

#define SIMD_WIDTH (2 * sizeof(double))

//bit-string based allocator for registers

typedef uint32_t RegisterSet;
struct Allocator {
	uint32_t a;
	uint32_t n_allocated;
	static const uint32_t NUM_REGISTERS = 16;
	Allocator() : a(~0), n_allocated(0) {}
	void print() {
		for(int i = 0; i < 32; i++)
			if( a & (1 << i))
				printf("-");
			else
				printf("a");
		printf("\n");
	}
	//try to allocated preferred register
	int allocate(size_t preferred) {
		assert(preferred < NUM_REGISTERS);
		if(a & (1 << preferred)) {
			a &= ~(1 << preferred);
			return preferred;
		} else return allocate();
	}
	int allocateWithMask(RegisterSet valid_registers) {
		int reg = ffs(a & valid_registers) - 1;
		a &= ~(1 << reg);

		if(reg >= (int) NUM_REGISTERS)
			_error("ran out of registers");

		return reg;
	}
	int allocate() { return allocateWithMask(~0); }
	RegisterSet live_registers() {
		return a;
	}
	void free(int reg) {
		a |= (1 << reg);
	}
};

struct RegisterIterator {
	RegisterIterator(RegisterSet l) {
		live = ~l;
	}
	bool done() { return live == 0; }
	void next() {  live &= ~(1 << value()); }
	uint32_t value() { return ffs(live) - 1; }
private:
	bool forward;
	uint32_t live;
};

#define CODE_BUFFER_SIZE (4096)
static char * code_buffer = NULL;
static char constant_table[sizeof(double) * 2 * TRACE_MAX_NODES] __attribute__((aligned(16)));
enum ConstantTableEntry {
	C_ABS_MASK = 0,
	C_NEG_MASK = 0x10,
	C_NOT_MASK = 0x20,
	C_SEQ_VEC  = 0x30,
	C_REGISTER_SPILL_SPACE = 0x40,
	C_FUNCTION_SPILL_SPACE = 0x140,
	C_FIRST_TRACE_CONST = 0x160
};

static void set_constant_d(int offset, double v) {
	double * d = (double*) &constant_table[offset];
	d[0] = d[1] = v;
}
static void set_constant_i(int offset, uint64_t v) {
	uint64_t * i = (uint64_t*) &constant_table[offset];
	i[0] = i[1] = v;
}
static void set_constant_ii(int offset, uint64_t v0, uint64_t v1) {
	uint64_t * i = (uint64_t*) &constant_table[offset];
	i[0] = v0;
	i[1] = v1;
}
static void set_constant_dd(int offset, double v0, double v1) {
	double * i = (double*) &constant_table[offset];
	i[0] = v0;
	i[1] = v1;
}
static void set_constant_fn(int offset, void * fn) {
	void** i = (void**) &constant_table[offset];
	*i = fn;
}

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
static double iabs(double aa) { //these are actually integers passed in xmm0 and xmm1
	union {
		int64_t ia;
		double da;
	};
	da = aa;
	ia = (ia < 0) ?  -ia : ia;
	return da; //also return the value in xmm0
}


struct TraceJIT {
	TraceJIT(Trace * t)
	:  trace(t), asm_(code_buffer,CODE_BUFFER_SIZE), next_constant_slot(C_FIRST_TRACE_CONST) {}

	Trace * trace;
	IRNode * store_inst[TRACE_MAX_NODES];
	RegisterSet live_registers[TRACE_MAX_NODES];
	char allocated_register[TRACE_MAX_NODES];
	Assembler asm_;
	Allocator alloc;


	Register constant_base; //holds pointer to Trace object
	Register vector_offset; //holds byte offset into vector for current loop iteration
	Register load_addr; //holds address of input vectors
	Register end_offset; //holds address of input vectors
	uint32_t next_constant_slot;

	void compile() {
		bzero(store_inst,sizeof(IRNode *) * trace->n_nodes);
		memset(allocated_register,-1,sizeof(char) * trace->n_nodes);
		//pass 1 register allocation
		for(IRef i = trace->n_nodes; i > 0; i--) {
			IRef ref = i - 1;
			IRNode & node = trace->nodes[ref];
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
		vector_offset = r13;
		load_addr = r14;
		end_offset = r15;

		asm_.push(constant_base);
		asm_.push(vector_offset);
		asm_.push(load_addr);
		asm_.push(end_offset);
		asm_.push(rbx);

		asm_.movq(constant_base, &constant_table[0]);
		asm_.xor_(vector_offset,vector_offset);
		asm_.movq(end_offset, trace->length * sizeof(double));

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
				XMMRegister r = reg(ref);
				asm_.movdqa(r,Operand(constant_base,PushConstantD(node.loadc.d)));
			} break;
			case IRNode::LOADV: {
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
					asm_.andpd(reg(ref),Operand(constant_base,C_ABS_MASK));
				} break;
				case IROpCode::neg: {
					EmitMove(reg(ref),reg(node.unary.a));
					asm_.xorpd(reg(ref),Operand(constant_base,C_NEG_MASK));
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
				case IROpCode::sign: EmitUnaryFunction(ref,sign_fn); break;
				case IROpCode::mod: EmitBinaryFunction(ref,Mod); break;
				case IROpCode::cast: EmitUnaryFunction(ref,casti2d); break; //it should be possible to inline this, but I can't find the convert packed quadword to packed double instruction
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
					asm_.xorpd(reg(ref),Operand(constant_base,C_NOT_MASK)); //r = ~r
					asm_.psubq(reg(ref),Operand(constant_base,C_NOT_MASK)); //r -= -1
				} break;
				case IROpCode::pos: EmitMove(reg(ref),reg(node.unary.a)); break;
				case IROpCode::mod: EmitBinaryFunction(ref,imod); break;
				case IROpCode::seq: {
					asm_.movq(load_addr,vector_offset);
					asm_.shr(load_addr,Immediate(4));
					asm_.movq(reg(ref),load_addr);
					asm_.unpcklpd(reg(ref),reg(ref));
					asm_.paddq(reg(ref),Operand(constant_base,C_SEQ_VEC));
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
					EmitVectorStore(str.store.dst->p,reg(str.store.a));
				} else {
					//scalar stores only occur for reductions right now and are handled during the opcode
					_error("unsupported scalar store");
				}
			}
		}

		asm_.addq(vector_offset, Immediate(SIMD_WIDTH));
		asm_.cmpq(vector_offset,end_offset);
		asm_.j(less,&begin);

		asm_.pop(rbx);
		asm_.pop(end_offset);
		asm_.pop(load_addr);
		asm_.pop(vector_offset);
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
	void EmitVectorLoad(XMMRegister dst, void * src) {
		int64_t diff = (int64_t)src - (int64_t)constant_table;
		if(is_int32(diff)) {
			asm_.movdqa(dst,Operand(constant_base,vector_offset,times_1,diff));
		} else {
			asm_.movq(load_addr,src);
			asm_.movdqa(dst, Operand(load_addr,vector_offset,times_1,0));
		}
	}
	void EmitVectorStore(void * dst,  XMMRegister src) {
		int64_t diff = (int64_t)dst - (int64_t)constant_table;
		if(is_int32(diff)) {
			asm_.movdqa(Operand(constant_base,vector_offset,times_1,diff),src);
		} else {
			asm_.movq(load_addr,dst);
			asm_.movdqa(Operand(load_addr,vector_offset,times_1,0),src);
		}
	}
	Operand EncodeOperand(void * src) {
		int64_t diff = (int64_t)src - (int64_t)constant_table;
		if(is_int32(diff)) {
			return Operand(constant_base,diff);
		} else {
			asm_.movq(load_addr,src);
			return Operand(load_addr,0);
		}

	}
	IRef AllocateNullary(IRef ref, bool p = true) {
		if(allocated_register[ref] < 0) { //this instruction is dead, for now we just emit it anyway
			allocated_register[ref] = alloc.allocate();
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
		if(allocated_register[a] < 0)
			//try to allocated register a and r to the same location,  this avoids moves in binary instructions
			allocated_register[a] = alloc.allocate(r);
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
			allocated_register[b] = alloc.allocateWithMask(mask);
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

	uint32_t PushConstantD(double d) {
		uint32_t offset = next_constant_slot;
		set_constant_d(offset,d);
		next_constant_slot += SIMD_WIDTH;
		return offset;
	}
	uint32_t PushConstantFn(void * fn) {
		uint32_t offset = next_constant_slot;
		set_constant_fn(offset,fn);
		next_constant_slot += SIMD_WIDTH;
		return offset;
	}

	void EmitCall(void * fn) {
		int64_t diff = (int64_t)(code_buffer + asm_.pc_offset() + 5 - (int64_t) fn);
		if(is_int32(diff)) {
			asm_.call((byte*)fn);
		} else {
			asm_.call(Operand(constant_base,PushConstantFn(fn)));
		}
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
		asm_.movdqa(Operand(constant_base,C_FUNCTION_SPILL_SPACE),reg(node.binary.a));
		asm_.movdqa(Operand(constant_base,C_FUNCTION_SPILL_SPACE + 0x10),reg(node.binary.b));
		asm_.movsd(xmm0,Operand(constant_base,C_FUNCTION_SPILL_SPACE));
		asm_.movsd(xmm1,Operand(constant_base,C_FUNCTION_SPILL_SPACE + 0x10));
		EmitCall(fn);
		asm_.movsd(Operand(constant_base,C_FUNCTION_SPILL_SPACE),xmm0);
		asm_.movsd(xmm0,Operand(constant_base,C_FUNCTION_SPILL_SPACE + 0x8));
		asm_.movsd(xmm1,Operand(constant_base,C_FUNCTION_SPILL_SPACE + 0x18));
		EmitCall(fn);
		asm_.movsd(Operand(constant_base,C_FUNCTION_SPILL_SPACE + 0x8),xmm0);
		asm_.movdqa(reg(ref),Operand(constant_base,C_FUNCTION_SPILL_SPACE));
		RestoreRegisters(live_registers[ref]);
	}

	void EmitDebugPrintResult(IRef i) {
		RegisterSet regs = live_registers[i];
		regs &= ~(1 << allocated_register[i]);

		SaveRegisters(regs);
		EmitMove(xmm0,reg(i));
		asm_.movq(rdi,vector_offset);
		EmitCall((void*) debug_print);
		RestoreRegisters(regs);
	}

	void SaveRegisters(RegisterSet regs) {
		uint32_t offset = C_REGISTER_SPILL_SPACE;
		for(RegisterIterator it(regs); !it.done(); it.next()) {
			asm_.movdqa(Operand(constant_base,offset),XMMRegister::FromAllocationIndex(it.value()));
			offset += SIMD_WIDTH;
		}
	}

	void RestoreRegisters(RegisterSet regs) {
		uint32_t offset = C_REGISTER_SPILL_SPACE;
		for(RegisterIterator it(regs); !it.done(); it.next()) {
			asm_.movdqa(XMMRegister::FromAllocationIndex(it.value()),Operand(constant_base,offset));
			offset += SIMD_WIDTH;
		}
	}

	void execute(State & state) {
		typedef void (*fn) (void);
		fn trace_code = (fn) code_buffer;
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

	if(code_buffer == NULL) {
		//allocate the code buffer
		code_buffer = (char*) malloc(CODE_BUFFER_SIZE);//(char*) mmap(NULL,CODE_BUFFER_SIZE,PROT_READ | PROT_WRITE | PROT_EXEC, MAP_PRIVATE | MAP_ANON, -1, 0);
		int64_t page = (int64_t)code_buffer & ~0x7FFF;
		int64_t size = (int64_t)code_buffer + CODE_BUFFER_SIZE - page;
		int err = mprotect((void*)page,size, PROT_READ | PROT_WRITE | PROT_EXEC);
		if(err) {
			_error("mprotect failed.");
		}
		//also fill in the constant table
		set_constant_i(C_ABS_MASK,0x7FFFFFFFFFFFFFFFUL);
		set_constant_i(C_NEG_MASK,0x8000000000000000UL);
		set_constant_i(C_NOT_MASK,0xFFFFFFFFFFFFFFFFUL);
		set_constant_ii(C_SEQ_VEC,1UL,2UL);
	}

	TraceJIT trace_code(this);

	trace_code.compile();
	trace_code.execute(state);

	WriteOutputs(state);
}
