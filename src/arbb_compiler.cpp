#include "arbb_compiler.h"
#include "arbb_vmapi.h"
#include "exceptions.h"
#include <vector>

static int register_to_arbb[STATE_NUM_REGISTERS];
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

bool arbb_eval(State& state, Closure const& closure) {
	if(!initialized)
		arbb_init();
	std::cout << "arbb run on code: " << state.stringify(closure) << std::endl;
	Environment * oldenv = state.env;
	if(closure.environment() != NULL)
		state.env = closure.environment();

	std::vector<int64_t> in; std::vector<arbb_variable_t> input_globals;

	std::vector<int64_t> out;
	std::vector<arbb_variable_t> output_local_variables;
	std::vector<arbb_variable_t> output_global_variables;

	std::vector<int64_t> inout; //in intersect out

	//1. go through op codes and return false if there is one we don't handle
	//2. go through gets and add to the list of "inputs" to function, which will be bound to input values
	//3. go through assigns and add to list of "outputs" to function
	     //if an output is also an input, copy value into output variable at beginning of function
	//5. create a map from symbol identifiers to arbb_variable objects
	//6. generate code with assigns writing to arbb variables, and gets copying from arbb variables into arbb register objects.
	//7. for each output, if it is a vector, determine its size, and then allocate space for it in arbb, and assign it to the correct global
	//8. for each output copy its data back into the r environment.


	state.env = oldenv;
	//TODO: restore register[0] to return value correctly
	//state.registers[0] = state.registers[pc->a];
	return false;
}
