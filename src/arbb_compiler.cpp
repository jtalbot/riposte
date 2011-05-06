#include "arbb_compiler.h"
#include "arbb_vmapi.h"
#include "exceptions.h"
#include <vector>

//static int register_to_arbb[STATE_NUM_REGISTERS];

void error_break() {

}
#define ARBB_DO(fn) \
do { \
  /*fprintf(stderr,"%s:%d: %s\n",__FILE__,__LINE__,#fn);*/ \
  arbb_error_t err = (fn); \
  if(arbb_error_none != err) { \
	  std::ostringstream oss; \
	  oss << __FILE__ << ":" << __LINE__ << ": " << arbb_get_error_message(details); \
	  arbb_free_error_details(details); \
	  error_break(); \
	  throw CompileError(oss.str()); \
  } \
} while(0);

#define SCALAR_TYPE_MAP(_,p) \
	_(R_double,arbb_f64, p)

#define OP_MAP(_,p) \
	_(add,add,2,p) \
	_(pos,copy,1,p) \
	_(sub,sub,2,p) \
	_(neg,neg,1,p) \
	_(mul,mul,2,p) \
	_(div,div,2,p) \
	_(pow,pow,2,p) \
	_(abs,abs,1,p) \
	_(exp,exp,1,p) \
	_(log,log10,1,p) \
	_(cos, cos,1, p) \
	_(sin, sin,1, p) \
	_(tan, tan,1, p) \
	_(acos, acos,1, p) \
	_(asin, asin,1, p) \
	_(atan, atan,1, p) \
	_(lt,less,3,p) /*hack: for now n == 3 means a boolean operation that will require a cast to double */\
	_(gt,greater,3,p) \
	_(eq,equal,3,p) \
	_(neq,neq,3,p) \
	_(ge,geq,3,p) \
	_(le,leq,3,p)

bool bytecode_to_arbb_opcode(ByteCode bc, arbb_opcode_t * opcode, int * nary) {
#define OP_FN(rop,aop,n,_) case ByteCode::E_##rop: *opcode = arbb_op_##aop; *nary = n; return true;
	switch(bc.Enum()) {
	OP_MAP(OP_FN,0)
	default:
		return false;
	}
}

arbb_type_t get_base_type(arbb_context_t context, Type t) {
#define GET_TYPE(r_type,arbb_type,_) case Type::E_##r_type: st = arbb_type; break;
	arbb_scalar_type_t st;
	arbb_error_details_t details;
	switch(t.Enum()) {
		SCALAR_TYPE_MAP(GET_TYPE,0)
	default: throw CompileError("unsupported scalar type");
	}
#undef GET_TYPE
	arbb_type_t typ;
	ARBB_DO(arbb_get_scalar_type(context,&typ,st,&details));
	return typ;
}
arbb_type_t get_type(arbb_context_t context,Type t, uint64_t packed) {
	arbb_error_details_t details;
	arbb_type_t base = get_base_type(context,t);
	arbb_type_t arbb_type;
	if(packed < 2) {
		arbb_type = base;
	} else {
		ARBB_DO(arbb_get_dense_type(context,&arbb_type,base,1,&details));
	}
	return arbb_type;
}


struct ArType {
	ArType() {}
	ArType(arbb_context_t context, Type r_, uint64_t packed_) {
		r = r_;
		packed = packed_;
		a = get_type(context,r,packed);
	}
	Type r;
	unsigned packed : 2;
	bool is_vector() { return !(packed < 2); }
	arbb_type_t a;
};

//internal arbb variable
struct Variable {
	ArType type;
	arbb_variable_t var;
};

struct GlobalVariable {
	arbb_global_variable_t global_var;
	arbb_binding_t binding; //if this is an input-only vector variable, it will be bound directly
	int64_t r_name;

	Variable var;

	arbb_global_variable_t length;

	unsigned is_input : 1;
	unsigned is_output : 1;
	unsigned is_allocated : 1;
};



template<typename T>
int add_to_set(std::vector<T> & v, const T & item) {
	for(size_t i = 0; i < v.size(); i++) {
		if(v[i] == item) {
			return i;
		}

	}
	v.push_back(item);
	return v.size() - 1;
}
template<typename T>
int find_entry(std::vector<T> & v, T const & item) {
	for(size_t i = 0; i < v.size(); i++)
		if(v[i] == item)
			return (int)i;
	return -1;
}
template<typename T>
int contains(std::vector<T> & v, T const & item) {
	return find_entry(v,item) != -1;
}

class CompilationUnit {
private:
	arbb_context_t context;
	arbb_error_details_t details;
	State & state;
	Closure const& closure;
	bool is_instruction_supported(Instruction const & i) {
		switch(i.bc.Enum()) {
		case ByteCode::E_kget:
		case ByteCode::E_get:
		case ByteCode::E_ret:
		case ByteCode::E_add:
		case ByteCode::E_assign:
		case ByteCode::E_whilebegin:
		case ByteCode::E_whileend:
		case ByteCode::E_if1:
		case ByteCode::E_endif1:
		case ByteCode::E_jmp:
			return true;
		default: {
				arbb_opcode_t op;
				int n;
				return bytecode_to_arbb_opcode(i.bc,&op,&n);
			}
		}
	}
	bool can_compile() {
		for(size_t i = 0; i < closure.code().size(); i++)
			if(!is_instruction_supported(closure.code()[i]))
				return false;
		return true;
	}

	std::vector<GlobalVariable> global_variables;
	std::vector<GlobalVariable> global_constants;
#define RET_VALUE (-1)

	void allocate_input(GlobalVariable & v, Value const & val) {
		arbb_binding_t null_binding;

		arbb_set_binding_null(&null_binding);
		v.is_allocated = true;
		v.var.type = ArType(context,val.type,val.packed);


		if(v.var.type.is_vector()) {
			Vector vec_value(val);
			//create a binding to the vector
			arbb_binding_t binding;
			uint64_t sizes[1] = { vec_value.inner->length };
			uint64_t pitches[1] = { vec_value.inner->width };
			ARBB_DO(arbb_create_dense_binding(context,&binding,vec_value.inner->data,1,sizes,pitches,&details));
			arbb_global_variable_t input_data;
			ARBB_DO(arbb_create_global(context,&input_data,v.var.type.a,NULL,binding,NULL,&details));
			arbb_variable_t input_var;
			ARBB_DO(arbb_get_variable_from_global(context,&input_var,input_data,&details));
			if(!v.is_output) {
				printf("%ld: in\n",v.r_name);
				//we can just bind our variable directly!
				v.binding = binding;
				v.global_var = input_data;
				v.var.var = input_var;
			} else {
				printf("%ld: inout\n",v.r_name);
				//we might write to this and change its shape, so we need to copy in
				arbb_function_t null_fn;
				arbb_set_function_null(&null_fn);
				ARBB_DO(arbb_create_global(context,&v.global_var,v.var.type.a,NULL,null_binding,NULL,&details));
				ARBB_DO(arbb_get_variable_from_global(context,&v.var.var,v.global_var,&details));
				arbb_variable_t in[] = { input_var };
				arbb_variable_t out[] = { v.var.var };
				ARBB_DO(arbb_op(null_fn,arbb_op_copy,out,in,NULL,&details));
				//free the binding and the global
				arbb_refcountable_t rc = arbb_global_variable_to_refcountable(input_data);
				ARBB_DO(arbb_release_ref(rc,&details));
				ARBB_DO(arbb_free_binding(context,binding,&details));
			}
		} else {
			ARBB_DO(arbb_create_global(context,&v.global_var,v.var.type.a,NULL,null_binding,NULL,&details));
			ARBB_DO(arbb_get_variable_from_global(context,&v.var.var,v.global_var,&details));
			//still assuming all scalars are doubles...
			ARBB_DO(arbb_write_scalar(context,v.var.var,&val.d,&details));
		}
	}
	void allocate_output(GlobalVariable & v, ArType typ) {

		arbb_binding_t null_binding;

		arbb_set_binding_null(&null_binding);
		v.is_allocated = true;
		v.var.type = typ;
		printf("%ld: out\n",v.r_name);
		ARBB_DO(arbb_create_global(context,&v.global_var,v.var.type.a,NULL,null_binding,NULL,&details));
		ARBB_DO(arbb_get_variable_from_global(context,&v.var.var,v.global_var,&details));

		if(v.var.type.is_vector()) {
			//we need to store the length in another global variable...
			arbb_type_t usize;
			ARBB_DO(arbb_get_scalar_type(context,&usize,arbb_usize,&details));
			ARBB_DO(arbb_create_global(context,&v.length,usize,NULL,null_binding,NULL,&details));
		}
	}
	void allocate_globals() {
		std::vector<int64_t> in;
		std::vector<int64_t> out;
		std::vector<int64_t> all;

		for(size_t i = 0; i < closure.code().size(); i++) {
			Instruction const & inst = closure.code()[i];
			if(inst.bc == ByteCode::get) {
				add_to_set(in,inst.a);
				add_to_set(all,inst.a);
			} else if(inst.bc == ByteCode::assign) {
				add_to_set(out,inst.a);
				add_to_set(all,inst.a);
			}
		}
		out.push_back(RET_VALUE); //allocate a global to hold the return value as well
		all.push_back(RET_VALUE);

		for(size_t i = 0; i < all.size(); i++) {
			global_variables.push_back(GlobalVariable());
			GlobalVariable & v = global_variables.back();

			v.is_input = contains(in,all[i]);
			v.is_output = contains(out,all[i]);
			v.r_name = all[i];

			arbb_binding_t null_binding;
			arbb_set_binding_null(&null_binding);

			if(v.is_input) {
				//read the current value
				Value val;
				state.env->get(state, Symbol(v.r_name), val);
				allocate_input(v,val);
				if(v.is_output && v.var.type.is_vector()) {
					//we need to store the length in another global variable...
					arbb_type_t usize;
					ARBB_DO(arbb_get_scalar_type(context,&usize,arbb_usize,&details));
					ARBB_DO(arbb_create_global(context,&v.length,usize,NULL,null_binding,NULL,&details));
				}
			} else {
				//we don't know type information yet
				//so we will allocate this the first time we see it
				v.is_allocated = false;
			}
		}
	}

	GlobalVariable & global_by_name(int64_t name) {
		for(size_t i = 0; i < global_variables.size(); i++)
			if(global_variables[i].r_name == name)
				return global_variables[i];
		throw CompileError("variable not found");
	}
	void allocate_constants() {
		for(size_t i = 0; i < closure.constants().size(); i++) {
			Value const & val = closure.constants()[i];
			global_constants.push_back(GlobalVariable());
			GlobalVariable & v = global_constants.back();

			v.r_name = i;
			v.is_output = false;
			v.is_input = true;
			allocate_input(v,val);

		}
	}

	void abstract_interp() {
		//currently this fills in the types for global variables, later we can use it for more complicated type inference
		ArType registers[STATE_NUM_REGISTERS];
		std::vector<Instruction> const & code = closure.code();
		for(size_t i = 0; i < code.size(); i++) {
			Instruction const & inst = code[i];
			switch(inst.bc.Enum()) {
			case ByteCode::E_get: {
				registers[inst.c] = global_by_name(inst.a).var.type;
			} break;
			case ByteCode::E_kget: {
				registers[inst.c] = global_constants[inst.a].var.type;
			} break;
			case ByteCode::E_assign: {
				GlobalVariable & gv = global_by_name(inst.a);
				if(!gv.is_allocated) //output variables may not be allocated because we didn't know the type until now
					allocate_output(gv,registers[inst.c]);
			} break;
			case ByteCode::E_ret: {
				GlobalVariable & gv =  global_by_name(RET_VALUE);
				if(!gv.is_allocated)
					allocate_output(gv,registers[inst.a]);
			} break;
			case ByteCode::E_whilebegin: {
				//nop
			} break;
			case ByteCode::E_whileend: {
				//nop
			} break;
			case ByteCode::E_if1: {
				//nop
			} break;
			case ByteCode::E_jmp: {
				//nop
			} break;
			case ByteCode::E_endif1: {
				//nop
			} break;
			default:
				arbb_opcode_t op;
				int n;
				if(!bytecode_to_arbb_opcode(inst.bc,&op,&n)) {
					assert(!"unknown bytecode");
				}
				if(n != 3) {
					registers[inst.a] = infer_type(op,registers[inst.c],registers[inst.b]);
				} else {
					//boolean operators, result type will be cast to double
					registers[inst.a] = ArType(context,Type::R_double,1);
				}
			}
		}
	}

	std::vector<Variable> registers;
	int registers_to_var[STATE_NUM_REGISTERS];

	arbb_function_t fn;

	Variable & get_register(int64_t r) {
		int i = registers_to_var[r];

		assert(i >= 0 && (uint64_t) i < registers.size());
		return registers[i];
	}

	Variable & new_local(int64_t r, ArType typ) {
		registers.push_back(Variable());
		Variable & v = registers.back();
		v.type = typ;
		ARBB_DO(arbb_create_local(fn,&v.var,v.type.a,NULL,&details));
		registers_to_var[r] = registers.size() - 1;
		return v;
	}

	ArType infer_type(arbb_opcode_t op, ArType & t1, ArType & t2) {
		int n;
#define GET_SIZE(a,arbbop,num_args,_) case arbb_op_##arbbop: n = num_args; break;
		switch(op) {
			OP_MAP(GET_SIZE,0)
		default:
			n = 0; break;
		}
		if(n == 1 || t1.is_vector())
			return t1;
		else
			return t2;
	}
	void compile_fn() {
		std::fill(registers_to_var,registers_to_var + STATE_NUM_REGISTERS, -1);
		//some useful constants
		arbb_type_t bt,f64;
		ARBB_DO(arbb_get_scalar_type(context,&bt,arbb_boolean,&details));
		ARBB_DO(arbb_get_scalar_type(context,&f64,arbb_f64,&details));

		arbb_binding_t null_binding;
		arbb_set_binding_null(&null_binding);
		double z = 0.0;
		arbb_variable_t zero;
		arbb_global_variable_t zero_g;
		ARBB_DO(arbb_create_constant(context,&zero_g,f64,&z,NULL,&details));
		ARBB_DO(arbb_get_variable_from_global(context,&zero,zero_g,&details));
		arbb_type_t fn_type;
		ARBB_DO(arbb_get_function_type(context,&fn_type,0,NULL,0,NULL,&details));


		//code gen begins

		ARBB_DO(arbb_begin_function(context,&fn,fn_type,NULL,0,&details));


		std::vector<Instruction> const & code = closure.code();
		for(size_t i = 0; i < code.size(); i++) {
			Instruction const & inst = code[i];
			switch(inst.bc.Enum()) {
			case ByteCode::E_get: {
				Variable & v = global_by_name(inst.a).var;
				arbb_variable_t in[] = { v.var };
				arbb_variable_t out[] = { new_local(inst.c,v.type).var };
				ARBB_DO(arbb_op(fn,arbb_op_copy,out,in,NULL,&details));
			} break;
			case ByteCode::E_kget: {
				Variable & v = global_constants[inst.a].var;
				arbb_variable_t in[] = { v.var };
				arbb_variable_t out[] = { new_local(inst.c,v.type).var };
				ARBB_DO(arbb_op(fn,arbb_op_copy,out,in,NULL,&details));
			} break;
			case ByteCode::E_assign: {
				Variable & v = get_register(inst.c);
				GlobalVariable & gv = global_by_name(inst.a);

				arbb_variable_t in[] = { v.var };
				arbb_variable_t out[] = { gv.var.var };
				ARBB_DO(arbb_op(fn,arbb_op_copy,out,in,NULL,&details));
			} break;
			case ByteCode::E_ret: {
				Variable & input =  get_register(inst.a);
				arbb_variable_t in[] = { input.var };
				GlobalVariable & gv =  global_by_name(RET_VALUE);

				arbb_variable_t out[] = { gv.var.var };
				ARBB_DO(arbb_op(fn,arbb_op_copy,out,in,NULL,&details));
			} break;
			case ByteCode::E_whilebegin: {
				arbb_variable_t cond;
				ARBB_DO(arbb_create_local(fn,&cond,bt,NULL,&details));
				arbb_variable_t in[] = { get_register(inst.b).var, zero };
				arbb_variable_t out[] = { cond };
				ARBB_DO(arbb_op(fn,arbb_op_neq,out,in,NULL,&details));
				ARBB_DO(arbb_begin_loop(fn,arbb_loop_while,&details));
				ARBB_DO(arbb_begin_loop_block(fn,arbb_loop_block_cond,&details));
				ARBB_DO(arbb_loop_condition(fn,cond,&details));
				ARBB_DO(arbb_begin_loop_block(fn,arbb_loop_block_body,&details));
			} break;
			case ByteCode::E_whileend: {

				arbb_variable_t cond;
				ARBB_DO(arbb_create_local(fn,&cond,bt,NULL,&details));
				arbb_variable_t in[] = { get_register(inst.b).var, zero };
				arbb_variable_t out[] = { cond };
				ARBB_DO(arbb_op(fn,arbb_op_equal,out,in,NULL,&details));

				ARBB_DO(arbb_if(fn,cond,&details));
				ARBB_DO(arbb_break(fn,&details));
				ARBB_DO(arbb_end_if(fn,&details));

				ARBB_DO(arbb_end_loop(fn,&details));

			} break;
			case ByteCode::E_if1: {
				arbb_variable_t cond;
				ARBB_DO(arbb_create_local(fn,&cond,bt,NULL,&details));
				arbb_variable_t in[] = { get_register(inst.b).var, zero };
				arbb_variable_t out[] = { cond };
				ARBB_DO(arbb_op(fn,arbb_op_neq,out,in,NULL,&details));
				ARBB_DO(arbb_if(fn, cond, &details));
			} break;
			case ByteCode::E_jmp: {
				// right now the jmp instruction is only used to jump over an else block, so repurpose to emit ARBB else instruction
				ARBB_DO(arbb_else(fn, &details));
			} break;
			case ByteCode::E_endif1: {
				ARBB_DO(arbb_end_if(fn, &details));
			} break;
			default:
				arbb_opcode_t op;
				int n;
				if(!bytecode_to_arbb_opcode(inst.bc,&op,&n)) {
					assert(!"unknown bytecode");
				}
				if(n != 3) {
					//boolean operations, we need to first perform the op then cast to double
					Variable & v1 = get_register(inst.c);
					Variable & v2 = get_register(inst.b);
					arbb_variable_t in[] = { v1.var, v2.var };
					ArType tpe = infer_type(op,v1.type,v2.type);
					arbb_variable_t out[] = {  new_local(inst.a,tpe).var };
					ARBB_DO(arbb_op(fn,op,out,in,NULL,&details));
				} else {
					//boolean operations, we need to first perform the op then cast to double
					arbb_variable_t r;
					ARBB_DO(arbb_create_local(fn,&r,bt,NULL,&details));
					arbb_variable_t in[] = { get_register(inst.c).var, get_register(inst.b).var };
					arbb_variable_t out[] = { r };
					ARBB_DO(arbb_op(fn,op,out,in,NULL,&details));
					ArType dbl(context,Type::R_double,1);
					out[0] = new_local(inst.a,dbl).var;
					in[0] = r;
					ARBB_DO(arbb_op(fn,arbb_op_cast,out,in,NULL,&details));
				}
			}
		}


		//now we need to get all the lengths of output vectors

		for(size_t i = 0; i < global_variables.size(); i++) {
			GlobalVariable & v = global_variables[i];
			if(v.is_output && v.var.type.is_vector()) {
				arbb_variable_t var;
				ARBB_DO(arbb_get_variable_from_global(context,&var,v.length,&details));
				arbb_variable_t in[1] = { v.var.var };
				arbb_variable_t out[1] = { var };
				ARBB_DO(arbb_op(fn,arbb_op_length,out,in,NULL,&details));
			}
		}

		ARBB_DO(arbb_end_function(fn,&details));


		ARBB_DO(arbb_compile(fn,&details));

		arbb_string_t fn_str;
		ARBB_DO(arbb_serialize_function(fn,&fn_str,&details));
		printf("function is\n%s\n",arbb_get_c_string(fn_str));
		arbb_free_string(fn_str);
	}
	void write_outputs() {
		//read outputs into interpreter variables, conditional assigns don't work correctly
		for(size_t i = 0; i < global_variables.size(); i++) {
			GlobalVariable & v = global_variables[i];

			if(v.is_output) {
				Value out;
				if(v.var.type.is_vector()) {
					//first we read the length...
					uint64_t len;
					arbb_variable_t len_var;
					ARBB_DO(arbb_get_variable_from_global(context,&len_var,v.length,&details));
					ARBB_DO(arbb_read_scalar(context,len_var,&len,&details));
					//now get the data..
					void * data;
					uint64_t pitch;
					printf("length is %lld\n",len);
					ARBB_DO(arbb_map_to_host(context,v.var.var,&data,&pitch,arbb_read_only_range,&details));
					if(len == 1) {
						Double d = Double::c(*(double*)data);
						d.toValue(out);
					} else {
						Double d(len);
						memcpy(d.inner->data,data,sizeof(double) * len);
						d.toValue(out);
						std::cout << v.r_name << " " << state.stringify(d) << std::endl;
					}

				} else {
					double val;
					ARBB_DO(arbb_read_scalar(context,v.var.var,&val,&details));
					printf("reading %f from %ld\n",val,v.r_name);
					Double d = Double::c(val);
					d.toValue(out);
				}
				if(v.r_name == RET_VALUE)
					state.registers[0] = out;
				else
					state.env->assign(Symbol(v.r_name), out);
			}
		}
	}
public:
	CompilationUnit(State & state_, Closure const & closure_)
	: state(state_), closure(closure_) {
		ARBB_DO(arbb_get_default_context(&context,&details));
	}
	bool eval() {
		Environment * oldenv = state.env;
		if(closure.environment() != NULL)
			state.env = closure.environment();
		if(!can_compile()) {
			state.env = oldenv;
			return false;
		}

		std::cout << "code: " << state.stringify(closure) << std::endl;

		std::cout << "go for codegen!" << std::endl;

		this->allocate_globals();

		this->allocate_constants();

		this->abstract_interp(); //will create arbb objects for output globals

		this->compile_fn();

		ARBB_DO(arbb_execute(fn,NULL,NULL,&details));

		this->write_outputs();

		state.env = oldenv;

		return true;
	}
};

bool arbb_eval(State& state, Closure const& closure) {
	CompilationUnit unit(state,closure);
	return unit.eval();
}
