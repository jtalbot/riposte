#include "trace_compiler.h"
#include "trace.h"

#ifndef RIPOSTE_DISABLE_TRACING
#include "arbb_vmapi.h"
#include "stdlib.h"
#include "value.h"

//for arbb operations that should only fail if there is a compiler bug
#define ARBB_RUN(fn) \
	do { \
		arbb_error_t e = (fn); \
		if(arbb_error_none != e) \
			panic(__FILE__,__LINE__,#fn,e); \
	} while(0)
#define COMPILER_DO(fn) \
	do { \
		TCStatus s = (fn); \
		if(TCStatus::SUCCESS != s) \
			return s; \
	} while(0)

//mapping from ir opcode to arbb op code, if one exists
#define ARBB_MAPPING(_,p) \
	_(abs,abs) \
	_(acos,acos) \
	_(add,add) \
	_(asin,asin) \
	_(atan,atan) \
	_(ceiling,ceil) \
	_(character1,cast) \
	_(complex1,cast) \
	_(cos,cos) \
	_(div,div) \
	_(double1,cast) \
	_(eq,equal) \
	_(exp,exp) \
	_(floor,floor) \
	_(ge,geq) \
	_(gt,greater) \
	_(idiv,div) \
	_(integer1,cast) \
	_(land,log_and) \
	_(le,leq) \
	_(lnot,log_not) \
	_(log,log10) \
	_(logical1,cast) \
	_(lor,log_or) \
	_(lt,less) \
	_(mod,mod) \
	_(move,copy) \
	_(mul,mul) \
	_(neg,neg) \
	_(neq,neq) \
	_(pos,copy) \
	_(pow,pow) \
	_(round,round) \
	_(sin,sin) \
	_(sqrt,sqrt) \
	_(sub,sub) \
	_(tan,tan) \

//NYI: not included: iftrue, _load, seq, sign, signif, guard, trunc


struct TraceCompilerImpl : public TraceCompiler {

	struct Output {
		IRef definition;
		RenamingTable::Entry interpreter_location;
		arbb_type_t arbb_typ;
		arbb_global_variable_t var;
		arbb_global_variable_t length; //if typ is a vector this will hold the length of the vector
	};

	struct Input {
		IRType typ;
		RenamingTable::Entry interpreter_location;
		arbb_type_t arbb_typ;
	};

	struct TraceExit {
		int64_t offset;
		std::vector<Output> outputs;
	};

	arbb_context_t ctx;
	arbb_error_details_t details;
	arbb_function_t fn;


	std::vector<arbb_global_variable_t> constants; //mirrors trace->constants
	std::vector<arbb_binding_t> bindings; //bindings produced by vector constants

	std::vector<Input> inputs;

	arbb_global_variable_t exit_code;
	std::vector<TraceExit> exits;


	Trace * trace;

	arbb_type_t typeFor(IRType const & typ) {
		arbb_scalar_type_t scalar;
		switch(typ.base_type.Enum()) {
			case IRScalarType::E_T_null: scalar = arbb_i32; break;
			case IRScalarType::E_T_logical: scalar = arbb_boolean; break;
			case IRScalarType::E_T_integer: scalar = arbb_i64; break;
			case IRScalarType::E_T_double: scalar = arbb_f64; break;
			case IRScalarType::E_T_complex: panic("NYI: complex"); break;
			case IRScalarType::E_T_character: panic("NYI: character"); break;
			case IRScalarType::E_T_void: panic("void type is not an arbb type"); break;
			case IRScalarType::E_T_unsupported: panic("successful trace contains unsupported type."); break;
		}
		arbb_type_t scalar_arbb_type;
		ARBB_RUN(arbb_get_scalar_type(ctx,&scalar_arbb_type,scalar,&details));
		if(typ.isVector) {
			arbb_type_t typ;
			ARBB_RUN(arbb_get_dense_type(ctx,&typ,scalar_arbb_type,1,&details));
			return typ;
		} else {
			return scalar_arbb_type;
		}
	}
	arbb_variable_t varFor(arbb_global_variable_t var) {
		arbb_variable_t r;
		ARBB_RUN(arbb_get_variable_from_global(ctx,&r,var,&details));
		return r;
	}

	arbb_type_t sizeType() {
		arbb_type_t typ;
		ARBB_RUN(arbb_get_scalar_type(ctx,&typ,arbb_usize,&details));
		return typ;
	}

	TraceCompilerImpl(Trace * trace) {
		this->trace = trace;
	}

	void addExit(::TraceExit const & e, bool in_body) {
		exits.push_back(TraceExit());
		TraceExit & exit = exits.back();
		exit.offset = e.offset;
		for(size_t j = 0; j < trace->renaming_table.outputs.size(); j++) {
			const RenamingTable::Entry & entry = trace->renaming_table.outputs[j];
			IRef definition;
			if(  trace->renaming_table.get(entry.location,entry.id,e.snapshot,&definition,NULL,NULL)
			  || (in_body && trace->renaming_table.get(entry.location,entry.id,&definition))) {
				IRType typ = get(definition).typ;
				arbb_type_t artyp = typeFor(typ);
				arbb_global_variable_t var,length;
				arbb_binding_t null_binding; arbb_set_binding_null(&null_binding);
				ARBB_RUN(arbb_create_global(ctx,&var,artyp,NULL,null_binding,NULL,&details));
				if(typ.isVector)
					ARBB_RUN(arbb_create_global(ctx,&length,sizeType(),NULL,null_binding,NULL,&details));
				Output output = { definition, entry, artyp ,var,length};
				exit.outputs.push_back(output);
			}

		}
	}

	void addInput(IRNode const & node) {
		RenamingTable::Entry entry = { node.a, (node.opcode == IROpCode::sload) ? RenamingTable::SLOT : RenamingTable::VARIABLE };
		Input input = { node.typ, entry, typeFor(node.typ) };
		inputs.push_back(input);
	}

	IRNode const & get(IRef ref) {
		return trace->optimized[ref];
	}

	void defineVariable(IRef ref, std::vector<arbb_variable_t> & references) {
		IRNode const & node = get(ref);
		if( node.opcode == IROpCode::kload ) {
			references[ref] = varFor(constants[node.a]);
		} else if( (node.flags() & IRNode::REF_R) != 0) {
			ARBB_RUN(arbb_create_local(fn,&references[ref],typeFor(node.typ),NULL,&details));
		}
	}

	arbb_variable_t constantInt(int64_t i) {
		arbb_global_variable_t gv;
		ARBB_RUN(arbb_create_constant(ctx,&gv,typeFor(IRType::Int()),&i,NULL,&details));
		return varFor(gv);
	}
	arbb_variable_t constantBool(bool b) {
		arbb_global_variable_t gv;
		ARBB_RUN(arbb_create_constant(ctx,&gv,typeFor(IRType::Bool()),&b,NULL,&details));
		return varFor(gv);
	}

	enum Location { L_HEADER, L_FIRST, L_BODY };
	TCStatus emitir(IRef ref, Location loc, std::vector<arbb_variable_t> & references, size_t * if_depth) {
		IRNode const & node = get(ref);
		printf("emitting %s\n",node.toString().c_str());
		arbb_variable_t outputs[1];
		arbb_variable_t inputs[2];
		if(node.flags() & IRNode::REF_R)
			outputs[0] = references[ref];
		if(node.flags() & IRNode::REF_A)
			inputs[0] = references[node.a];
		if(node.flags() & IRNode::REF_B)
			inputs[1] = references[node.b];

		switch(node.opcode.Enum()) {
#define ELEMENT_WISE_OP(ir,arbb) \
		case IROpCode::E_##ir: \
			assert(!arbb_is_variable_null(outputs[0])); \
			assert(!arbb_is_variable_null(inputs[0])); \
			assert(!arbb_is_variable_null(inputs[1])); \
			ARBB_RUN(arbb_op(fn,arbb_op_##arbb,outputs,inputs,NULL,NULL,&details)); \
			break;
		ARBB_MAPPING(ELEMENT_WISE_OP,0)
		case IROpCode::E_kload:
			//pass -- we already handled this by aliasing the variable in the references array to the global variable holding the constant
			break;
		case IROpCode::E_guard: {
			int64_t exit = node.b;
			ARBB_RUN(arbb_if(fn,references[node.a],&details));
			switch(loc) {
			case L_HEADER:
				break; //haven't started the loop body, we need need to bail to the original instruction
				       //TODO: it may be possible to still recover some
				       //of the results calculated before this guard by generating an exit path that would calculate all loop-dependent variables, before exiting to the original exit point
			case L_BODY:
				exit += exits.size(); //shift to exit will all nodes defined
				/* fallthrough */
			case L_FIRST: {
				TraceExit & texit = exits[exit];
				for(size_t i = 0; i < texit.outputs.size(); i++) {
					Output & o = texit.outputs[i];
					outputs[0] = varFor(o.var);
					inputs[0] = references[o.definition];
					ARBB_RUN(arbb_op(fn,arbb_op_copy,outputs,inputs,NULL,NULL,&details));
				}
			} break;
			}

			//write the exit code
			outputs[0] = varFor(exit_code);
			inputs[0] = constantInt(exit);
			ARBB_RUN(arbb_op(fn,arbb_op_copy,outputs,inputs,NULL,NULL,&details));

			//arbb_return is unimplemented, so outside of the loop guards get compiled to nested if-statements
			//inside the loop, guards are compiled as if statements + breaks
			if(loc == L_BODY) {
				ARBB_RUN(arbb_break(fn,&details));
				ARBB_RUN(arbb_end_if(fn,&details));
			} else {
				ARBB_RUN(arbb_else(fn,&details));
				(*if_depth)++;
			}
		} break;
		default:
			printf("trace compiler: unsupported op %s\n",node.opcode.toString());
			return TCStatus::UNSUPPORTED_IR;
			break;
		}

		return TCStatus::SUCCESS;
	}

	//copy loop-carried definition into phi variables
	void emitPhis(std::vector<arbb_variable_t> & references) {
		for(size_t i = 0; i < trace->phis.size(); i++) {
			IRNode const & node = get(trace->phis[i]);
			arbb_variable_t input = references[node.b];
			arbb_variable_t output = references[trace->phis[i]];
			ARBB_RUN(arbb_op(fn,arbb_op_copy,&output,&input,NULL,NULL,&details));
		}
	}

	//returns false if typ doesn't match values typ
	bool loadValueGuarded(IRType typ, Value const & v, arbb_binding_t * pbinding, arbb_global_variable_t * result) {
		IRType vtyp(v);
		arbb_binding_t binding;
		if(typ == vtyp) {
			if(typ.isVector) {
				Vector vector(v);
				uint64_t length = vector.length;
				uint64_t width = vector.width;
				ARBB_RUN(arbb_create_dense_binding(ctx,&binding,vector._data,1,&length,&width,&details));
			} else {
				arbb_set_binding_null(&binding);
				ARBB_RUN(arbb_write_scalar(ctx,varFor(*result),&v.p,&details));
			}
			ARBB_RUN(arbb_create_global(ctx,result,typeFor(typ),NULL,binding,NULL,&details));
			if(pbinding)
				*pbinding = binding;
			return true;
		} else return false;
	}

	void loadConst(Value const & v, arbb_global_variable_t * result) {
		IRType typ(v);
		if(typ.isVector) {
			//no way to load vector constants, so we fallback to a global variable
			arbb_binding_t binding;
			loadValueGuarded(typ,v,&binding,result);
			bindings.push_back(binding);
		} else {
			ARBB_RUN(arbb_create_constant(ctx,result,typeFor(typ),const_cast<void**>(&v.p),NULL,&details));
		}
	}

	TCStatus compile() {
		ARBB_RUN(arbb_get_default_context(&ctx,&details));


		{	//create a variable to track which exit we took
			arbb_binding_t null_binding;
			arbb_set_binding_null(&null_binding);
			ARBB_RUN(arbb_create_global(ctx,&exit_code,typeFor(IRType::Int()),NULL,null_binding,NULL,&details));
		}
		//for each TraceExit in the trace, we create two exit paths, the first path is for the first iteration of the trace when
		//variables may not have been defined
		for(size_t i = 0; i < trace->exits.size(); i++)
			addExit(trace->exits[i],false);
		for(size_t i = 0; i < trace->exits.size(); i++)
			addExit(trace->exits[i],true);


		//create constants
		for(size_t i = 0; i < trace->constants.size(); i++) {
			constants.push_back(arbb_global_variable_t());
			loadConst(trace->constants[i],&constants.back());
		}

		//create input structs for each variable that needs to be loaded
		for(size_t i = 0; i < trace->loads.size(); i++)
			addInput(get(trace->loads[i]));
		for(size_t i = 0; i < trace->phis.size(); i++)
			addInput(get(trace->phis[i]));


		//construction function type from input types
		arbb_type_t input_ts[inputs.size()];
		for(size_t i = 0; i < inputs.size(); i++)
			input_ts[i] = inputs[i].arbb_typ;

		arbb_type_t fn_type;
		ARBB_RUN(arbb_get_function_type(ctx,&fn_type,0,NULL,inputs.size(),input_ts,&details));

		ARBB_RUN(arbb_begin_function(ctx,&fn,fn_type,NULL,true,&details));

		std::vector<arbb_variable_t> references(trace->optimized.size()); //table that maps IRef -> arbb variable that holds the value for that IRNode
		for(size_t i = 0; i < references.size(); i++)
			arbb_set_variable_null(&references[i]);
		size_t idx = 0;
		//initialize load and phi nodes variables to their input values
		for(size_t i = 0; i < trace->loads.size(); i++)
			ARBB_RUN(arbb_get_parameter(fn,&references[ trace->loads[i] ],0,idx++,&details));
		for(size_t i = 0; i < trace->phis.size(); i++)
			ARBB_RUN(arbb_get_parameter(fn,&references[ trace->phis[i] ],0,idx++,&details));

		//define variables for non-load nodes if the instruction defines a variable
		for(size_t i = 0; i < trace->loop_header.size(); i++)
			defineVariable( trace->loop_header[i], references);
		for(size_t i = 0; i < trace->loop_body.size(); i++)
			defineVariable( trace->loop_body[i], references);

		//emit code!
		size_t if_depth = 0;
		for(size_t i = 0; i < trace->loop_header.size(); i++)
			COMPILER_DO(emitir(trace->loop_header[i],L_HEADER,references,&if_depth));
		//unroll first loop iteration, so exits know what variables are defined
		for(size_t i = 0; i < trace->loop_body.size(); i++)
			COMPILER_DO(emitir(trace->loop_body[i],L_FIRST,references,&if_depth));
		//we now enter a loop
		/*ARBB_RUN(arbb_begin_loop(fn,arbb_loop_while,&details));
		ARBB_RUN(arbb_begin_loop_block(fn,arbb_loop_block_cond,&details));
		ARBB_RUN(arbb_loop_condition(fn,constantBool(true),&details));
		ARBB_RUN(arbb_begin_loop_block(fn,arbb_loop_block_body,&details));
		emitPhis(references);
		for(size_t i = 0; i < trace->loop_body.size(); i++)
			COMPILER_DO(emitir(trace->loop_body[i],L_BODY,references,&if_depth));

		ARBB_RUN(arbb_end_loop(fn,&details));
		*/
		while(if_depth-- > 0)
			ARBB_RUN(arbb_end_if(fn,&details)); //end all the else statements that the guards produced

		ARBB_RUN(arbb_end_function(fn,&details));
		ARBB_RUN(arbb_compile(fn,&details));
		{

			arbb_string_t fn_str;
			ARBB_RUN(arbb_serialize_function(fn,&fn_str,&details));
			printf("function is\n%s\n",arbb_get_c_string(fn_str));
			arbb_free_string(fn_str);
		}

		return TCStatus::SUCCESS;
	}
	TCStatus execute(State & s, int64_t * offset) {
		//NYI - invoke the trace
		*offset = 0;
		return TCStatus::SUCCESS;
	}

	//for internal compiler error problems that are the result of bugs in the compiler
	__attribute__ ((noreturn))
	void panic(const char * file, int line, const char * txt, arbb_error_t error) {
		printf("%s:%d: trace compiler: internal error %s (%s)\n",file,line,txt,arbb_get_error_message(details));
		exit(1);
	}
	__attribute__ ((noreturn))
	void panic(const char * what) {
		printf("trace compiler: internal error %s\n",what);
		exit(1);
	}
};


#else

struct TraceCompilerImpl : public TraceCompiler {
	TraceCompilerImpl(Trace * trace) {}
	TCStatus compile() {
		return TCStatus::DISABLED;
	}
	TCStatus execute(State & s, int64_t * offset) {
		*offset = 0;
		return TCStatus::DISABLED;
	}
};

#endif

DEFINE_ENUM(TCStatus,ENUM_TC_STATUS)
DEFINE_ENUM_TO_STRING(TCStatus,ENUM_TC_STATUS)
TraceCompiler * TraceCompiler::create(Trace * t) {
	return new TraceCompilerImpl(t);
}
