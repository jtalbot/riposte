#include "arbb_compiler.h"
#include "arbb_vmapi.h"
#include "exceptions.h"
#include <vector>

//static int register_to_arbb[STATE_NUM_REGISTERS];
static arbb_context_t context;
static arbb_error_details_t details;
static bool initialized = false;

#define ARBB_DO(fn) \
do { \
  arbb_error_t err = (fn); \
  if(arbb_error_none != err) { \
	  throw CompileError(arbb_get_error_message(details)); \
  } \
} while(0);

#define SCALAR_TYPE_MAP(_,p) \
	_(R_integer,arbb_u32, p)		\
	_(R_double,arbb_u32, p)

arbb_type_t get_base_type(Type t) {
#define GET_TYPE(r_type,arbb_type,_) case Type::E_##r_type: st = arbb_type; break;
	arbb_scalar_type_t st;
	switch(t.Enum()) {
		SCALAR_TYPE_MAP(GET_TYPE,0)
	default: throw CompileError("unsupported scalar type");
	}
#undef GET_TYPE
	arbb_type_t typ;
	ARBB_DO(arbb_get_scalar_type(context,&typ,st,&details));
	return typ;
}
arbb_type_t get_type(Type t, uint64_t packed) {
	arbb_type_t base = get_base_type(t);
	arbb_type_t arbb_type;
	if(packed < 2) {
		arbb_type = base;
	} else {
		ARBB_DO(arbb_get_dense_type(context,&arbb_type,base,1,&details));
	}
	return arbb_type;
}

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

//internal arbb local variable
struct Register {
	arbb_type_t arbb_type;
	arbb_variable_t arbb_var;
};

void arbb_init() {
	initialized = true;
	ARBB_DO(arbb_get_default_context(&context,&details));
}

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

bool is_supported(Instruction const & i) {
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
bool arbb_eval(State& state, Closure const& closure) {
	if(!initialized)
		arbb_init();
	Environment * oldenv = state.env;
	if(closure.environment() != NULL)
		state.env = closure.environment();

	std::vector<Instruction> const & code = closure.code();

	std::cout << "code: " << state.stringify(closure) << std::endl;
	//1. go through op codes and return false if there is one we don't handle
	for(size_t i = 0; i < code.size(); i++)
		if(!is_supported(code[i]))
			return false;
	std::cout << "go for codegen!" << std::endl;

	std::vector<int64_t> in;
	std::vector<int64_t> out;
	std::vector<int64_t> all;
	//2. go through gets and add to the list of "inputs" to function, which will be bound to input values
	for(size_t i = 0; i < code.size(); i++) {
		Instruction const & inst = code[i];
		if(inst.bc == ByteCode::get) {
			add_to_set(in,inst.a);
			add_to_set(all,inst.a);
		} else if(inst.bc == ByteCode::assign) {
			add_to_set(out,inst.a);
			add_to_set(all,inst.a);
		}
	}
	arbb_type_t f64;
	ARBB_DO(arbb_get_scalar_type(context,&f64,arbb_f64,&details));
	arbb_type_t bt;
	ARBB_DO(arbb_get_scalar_type(context,&bt,arbb_boolean,&details));
	//create arbb variables for all R variables, for now everything is f64

	std::vector<arbb_global_variable_t> arbb_globals;
	std::vector<arbb_variable_t> arbb_globals_as_vars;
	arbb_binding_t null_binding;
	arbb_set_binding_null(&null_binding);


	for(size_t i = 0; i < all.size(); i++) {
		int64_t v = all[i];
		arbb_global_variable_t var;
		ARBB_DO(arbb_create_global(context,&var,f64,NULL,null_binding,NULL,&details));
		arbb_variable_t gv;
		ARBB_DO(arbb_get_variable_from_global(context,&gv,var,&details));
		arbb_globals.push_back(var);
		arbb_globals_as_vars.push_back(gv);
	}

	arbb_global_variable_t ret_value_g;
	arbb_variable_t ret_value;
	ARBB_DO(arbb_create_global(context,&ret_value_g,f64,NULL,null_binding,NULL,&details));
	ARBB_DO(arbb_get_variable_from_global(context,&ret_value,ret_value_g,&details));


	std::vector<arbb_variable_t> constants;
	for(size_t i = 0; i < closure.constants().size(); i++) {
		Value const & v = closure.constants()[i];
		assert(v.type == Type::R_double);
		arbb_global_variable_t c;
		ARBB_DO(arbb_create_constant(context,&c,f64,(void*)&v.d,NULL,&details));
		arbb_variable_t vc;
		ARBB_DO(arbb_get_variable_from_global(context,&vc,c,&details));
		constants.push_back(vc);
	}

	double z = 0.0;
	arbb_variable_t zero;
	arbb_global_variable_t zero_g;
	ARBB_DO(arbb_create_constant(context,&zero_g,f64,&z,NULL,&details));
	ARBB_DO(arbb_get_variable_from_global(context,&zero,zero_g,&details));

#define GET_VAR(x) arbb_globals_as_vars[find_entry(all,(x))]

	arbb_function_t fn;
	arbb_type_t fn_type;
	ARBB_DO(arbb_get_function_type(context,&fn_type,0,NULL,0,NULL,&details));
	ARBB_DO(arbb_begin_function(context,&fn,fn_type,NULL,0,&details));

	arbb_variable_t registers[STATE_NUM_REGISTERS];
	size_t num_allocated_registers = 0;

#define ALLOCATE_UPTO(x) do { \
	for(size_t j = num_allocated_registers; j <= (size_t)(x); j++) \
		ARBB_DO(arbb_create_local(fn,&registers[j],f64,NULL,&details)); \
	num_allocated_registers = (x) + 1; \
} while(0)

	for(size_t i = 0; i < code.size(); i++) {
		Instruction const & inst = code[i];
		switch(inst.bc.Enum()) {
		case ByteCode::E_get: {
			ALLOCATE_UPTO(inst.c);
			arbb_variable_t in[] = { GET_VAR(inst.a) };
			arbb_variable_t out[] = { registers[inst.c] };
			ARBB_DO(arbb_op(fn,arbb_op_copy,out,in,NULL,&details));
		} break;
		case ByteCode::E_kget: {
			ALLOCATE_UPTO(inst.c);
			arbb_variable_t in[] = { constants[inst.a] };
			arbb_variable_t out[] = { registers[inst.c] };
			ARBB_DO(arbb_op(fn,arbb_op_copy,out,in,NULL,&details));
		} break;
		/*
		case ByteCode::E_add: {
			ALLOCATE_UPTO(inst.a);
			arbb_variable_t in[] = { registers[inst.b], registers[inst.c] };
			arbb_variable_t out[] = { registers[inst.a] };
			ARBB_DO(arbb_op(fn,arbb_op_add,out,in,NULL,&details));
		} break; */
		case ByteCode::E_assign: {
			arbb_variable_t in[] = { registers[inst.c] };
			arbb_variable_t out[] = { GET_VAR(inst.a) };
			ARBB_DO(arbb_op(fn,arbb_op_copy,out,in,NULL,&details));
		} break;
		case ByteCode::E_ret: {
			ALLOCATE_UPTO(inst.a);
			arbb_variable_t in[] = { registers[inst.a] };
			arbb_variable_t out[] = { ret_value };
			ARBB_DO(arbb_op(fn,arbb_op_copy,out,in,NULL,&details));
		} break;
		case ByteCode::E_whilebegin: {
			arbb_variable_t cond;
			ARBB_DO(arbb_create_local(fn,&cond,bt,NULL,&details));
			arbb_variable_t in[] = { registers[inst.b], zero };
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
			arbb_variable_t in[] = { registers[inst.b], zero };
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
			arbb_variable_t in[] = { registers[inst.b], zero };
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
				arbb_variable_t out[] = { registers[inst.c] };
				arbb_variable_t in[] = { registers[inst.c], registers[inst.b] };
				ARBB_DO(arbb_op(fn,op,out,in,NULL,&details));
			} else {
				//boolean operations, we need to first perform the op then cast to double
				arbb_variable_t r;
				ARBB_DO(arbb_create_local(fn,&r,bt,NULL,&details));
				arbb_variable_t in[] = { registers[inst.c], registers[inst.b] };
				arbb_variable_t out[] = { r };
				ARBB_DO(arbb_op(fn,op,out,in,NULL,&details));
				out[0] = registers[inst.c];
				in[0] = r;
				ARBB_DO(arbb_op(fn,arbb_op_cast,out,in,NULL,&details));
			}
		}
	}
	ARBB_DO(arbb_end_function(fn,&details));


	ARBB_DO(arbb_compile(fn,&details));

	arbb_string_t fn_str;
	ARBB_DO(arbb_serialize_function(fn,&fn_str,&details));
	printf("function is\n%s\n",arbb_get_c_string(fn_str));
	arbb_free_string(fn_str);

	//read inputs into arbb globals
	for(size_t i = 0; i < in.size(); i++) {
		Value val;
		state.env->get(state, Symbol(in[i]), val);
		assert(val.type == Type::R_double);
		printf("writing %f to %ld\n",val.d,in[i]);
		ARBB_DO(arbb_write_scalar(context,GET_VAR(in[i]),&val.d,&details));
	}

	//exec
	ARBB_DO(arbb_execute(fn,NULL,NULL,&details));

	//read outputs into interpreter variables, conditional assigns don't work correctly
	for(size_t i = 0; i < out.size(); i++) {
		double val;
		ARBB_DO(arbb_read_scalar(context,GET_VAR(out[i]),&val,&details));
		printf("reading %f from %ld\n",val,out[i]);
		Double d = Double::c(val);
		state.env->assign(Symbol(out[i]), d);
	}

	double val;
	ARBB_DO(arbb_read_scalar(context,ret_value,&val,&details));
	printf("return value is %f\n",val);
	Double d = Double::c(val);
	state.registers[0] = d;

	//some massive cleanup code should go here

	//6. generate code with assigns writing to arbb variables, and gets copying from arbb variables into arbb register objects.
	//7. for each output, if it is a vector, determine its size, and then allocate space for it in arbb, and assign it to the correct global
	//8. for each output copy its data back into the r environment.


	state.env = oldenv;
	//TODO: restore register[0] to return value correctly
	//state.registers[0] = state.registers[pc->a];
	return true;
}
