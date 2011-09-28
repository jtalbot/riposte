#include "interpreter.h"
#include "vector.h"
#include "ops.h"
#include <sys/mman.h>
#include "assembler-x64.h"
#include <math.h>

using namespace v8::internal;

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
			++n_allocated;
			return preferred;
		} else return allocate();
	}
	int allocate() {
		int reg = ffs(a) - 1;
		a &= ~(1 << reg);

		if(++n_allocated > NUM_REGISTERS)
			_error("ran out of registers...");
		return reg;
	}
	RegisterSet live_registers() {
		return a;
	}
	void free(int reg) {
		n_allocated--;
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
	C_EXP = 0x20,
	C_LOG = 0x28,
	C_COS = 0x30,
	C_SIN = 0x38,
	C_TAN = 0x40,
	C_ACOS = 0x48,
	C_ASIN = 0x50,
	C_ATAN = 0x58,
	C_FIRST_TRACE_CONST = 0x60
};
static void set_constant(int offset, double v) {
	double * d = (double*) &constant_table[offset];
	d[0] = d[1] = v;
}
static void set_constant(int offset, uint64_t v) {
	uint64_t * i = (uint64_t*) &constant_table[offset];
	i[0] = i[1] = v;
}
static void set_constant(int offset, double (*fn)(double)) {
	void** i = (void**) &constant_table[offset];
	*i = (void*)fn;
}

struct TraceCode {
	TraceCode(Trace * t)
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
			case IRNode::UNARY: {
				AllocateUnary(ref,node.unary.a);
			} break;
			case IRNode::LOADC: /*fallthrough*/
			case IRNode::LOADV: {
				AllocateNullary(ref);
			} break;
			case IRNode::STORE: {
				store_inst[node.store.a] = &node;
			} break;
			case IRNode::SPECIAL:
			case IRNode::FOLD:
			default:
				_error("unsupported op");
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
			if(node.type != Type::Double)
				_error("only support doubles for now");

			//emit encoding specific code
			switch(node.enc) {
			case IRNode::BINARY: {
				EmitMove(reg(ref),reg(node.binary.a));
			} break;
			case IRNode::UNARY: break;
			case IRNode::LOADC: {
				XMMRegister r = reg(ref);
				set_constant(next_constant_slot,node.loadc.d);
				asm_.movdqa(r,Operand(constant_base,next_constant_slot));
				next_constant_slot += sizeof(double)*2;
			} break;
			case IRNode::LOADV: {
				asm_.movq(load_addr,node.loadv.p);
				asm_.movdqa(reg(ref), Operand(load_addr,vector_offset,times_1,0));
			} break;
			case IRNode::STORE: {
				//stores are generated right after the value that is stored
			} break;
			case IRNode::SPECIAL:
			case IRNode::FOLD:
			default:
				_error("unsupported op");
			}

			//right now reg(ref) holds a if binary
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

			case IROpCode::exp: EmitUnaryFunction(ref,C_EXP); break;
			case IROpCode::log: EmitUnaryFunction(ref,C_LOG); break;
			case IROpCode::cos: EmitUnaryFunction(ref,C_COS); break;
			case IROpCode::sin: EmitUnaryFunction(ref,C_SIN); break;
			case IROpCode::tan: EmitUnaryFunction(ref,C_TAN); break;
			case IROpCode::acos: EmitUnaryFunction(ref,C_ACOS); break;
			case IROpCode::asin: EmitUnaryFunction(ref,C_ASIN); break;
			case IROpCode::atan: EmitUnaryFunction(ref,C_ATAN); break;

			case IROpCode::mod:
			case IROpCode::pow:
			case IROpCode::atan2:
			case IROpCode::hypot:
			case IROpCode::sign:

			case IROpCode::signif:
			default:
				if(node.enc == IRNode::BINARY || node.enc == IRNode::UNARY)
					_error("unimplemented op");
				break;
			}


			if(store_inst[ref] != NULL) {
				IRNode & str = *store_inst[ref];
				if(str.op == IROpCode::storev) {
					void * addr = str.store.dst->p;
					asm_.movq(load_addr,addr);
					asm_.movdqa(Operand(load_addr,vector_offset,times_1,0),reg(str.store.a));
				} else {
					_error("unsupported scalar store");
				}
			}
		}

		asm_.addq(vector_offset, Immediate(2 * sizeof(double)));
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
	IRef AllocateNullary(IRef ref) {
		if(allocated_register[ref] < 0) { //this instruction is dead, for now we just emit it anyway
			allocated_register[ref] = alloc.allocate();
		}
		int r = allocated_register[ref];
		alloc.free(r);
		//we want to know the registers live out of this op, not including value that this op defines
		live_registers[ref] = alloc.live_registers();
		return r;
	}
	IRef AllocateUnary(IRef ref, IRef a) {
		IRef r = AllocateNullary(ref);
		if(allocated_register[a] < 0)
			allocated_register[a] = alloc.allocate(r);
		return r;
	}
	IRef AllocateBinary(IRef ref, IRef a, IRef b) {
		IRef r = AllocateUnary(ref,a);
		if(allocated_register[b] < 0)
			allocated_register[b] = alloc.allocate();
		return r;
	}
	void EmitUnaryFunction(IRef ref, uint32_t fn_offset) {
		uint32_t spill_loc = next_constant_slot;
		for(RegisterIterator it(live_registers[ref]); !it.done(); it.next()) {
			asm_.movdqa(Operand(constant_base,spill_loc),XMMRegister::FromAllocationIndex(it.value()));
			spill_loc += 2 * sizeof(double);
		}

		EmitMove(xmm0,reg(trace->nodes[ref].unary.a));
		asm_.movapd(xmm1,xmm0);
		asm_.unpckhpd(xmm1,xmm1);
		asm_.movq(load_addr,xmm1);//we need the high value for the second call
		asm_.call(Operand(constant_base,fn_offset));
		asm_.movq(rbx,xmm0);
		asm_.movq(xmm0,load_addr);
		asm_.call(Operand(constant_base,fn_offset));
		asm_.movq(xmm1,rbx);
		asm_.unpcklpd(xmm1,xmm0);
		EmitMove(reg(ref),xmm1);

		spill_loc = next_constant_slot;
		for(RegisterIterator it(live_registers[ref]); !it.done(); it.next()) {
			asm_.movdqa(XMMRegister::FromAllocationIndex(it.value()),Operand(constant_base,spill_loc));
			spill_loc += 2 * sizeof(double);
		}
	}
	void execute(State & state) {
		typedef void (*fn) (void);
		fn trace_code = (fn) code_buffer;
		trace_code();
	}
};


void Trace::Execute(State & state) {
	InitializeOutputs(state);
	if(state.tracing.verbose)
		printf("executing trace:\n%s\n",toString(state).c_str());

	if(code_buffer == NULL) {
		//allocate the code buffer
		code_buffer = (char*) mmap(NULL,CODE_BUFFER_SIZE,PROT_READ | PROT_WRITE | PROT_EXEC, MAP_PRIVATE | MAP_ANON, -1, 0);
		//also fill in the constant table
		set_constant(C_ABS_MASK,0x7FFFFFFFFFFFFFFFUL);
		set_constant(C_NEG_MASK,0x8000000000000000UL);
		set_constant(C_EXP,exp);
		set_constant(C_LOG,log);
		set_constant(C_COS,cos);
		set_constant(C_SIN,sin);
		set_constant(C_TAN,tan);
		set_constant(C_ACOS,acos);
		set_constant(C_ASIN,asin);
		set_constant(C_ATAN,atan);
	}

	TraceCode trace_code(this);

	trace_code.compile();
	trace_code.execute(state);

	WriteOutputs(state);
}
