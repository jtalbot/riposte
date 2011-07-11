#include "trace_compiler.h"
#include "trace.h"

#ifndef RIPOSTE_DISABLE_TRACING
#include "arbb_vmapi.h"
#include "stdlib.h"
#include "value.h"

//for arbb operations that should only fail if there is a compiler bug
#define ARBB_RUN(fn) \
	do { \
		/*printf("%s:%d: %s\n",__FILE__,__LINE__,#fn);*/ \
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

	struct Input {
		IRType typ;
		arbb_type_t arbb_typ;
		RenamingTable::Entry interpreter_location; //where does this variable come from?
	};


	//a structure tracking the type of each output variable from the arbb function.
	//a particular TraceExit will have one Output struct for each variable it needs
	//to write the interpreter. Each Output has a reference to on OutputVar that will hold
	//the value to write to the interpreter for this exit.

	//A simpler approach would be to allocate one output var for
	//each output. However, arbb requires all outputs to be defined, making it undesirable to
	//have a huge number of output variables that need to be assigned dummy values.  Instead
	//we reuse OutputVars as much as possible. Multiple Output structs can refer to the same
	//OutputVar as long as they are not in the same TraceExit.

	struct OutputVar {
		arbb_type_t arbb_typ;
		IRType typ;
	};

	typedef size_t OVRef; //reference to an output variable

	struct Output {
		IRef definition; //the value that holds this variable for the current exit
		RenamingTable::Entry interpreter_location;

		OVRef var;
		OVRef length; //if typ is a vector this will hold the length of the vector
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
	std::vector<OutputVar> output_variables;

	OVRef exit_code;
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
			case IRScalarType::E_T_size: scalar = arbb_usize; break;
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

	arbb_variable_t varFor(OVRef ov) {
		arbb_variable_t var;
		ARBB_RUN(arbb_get_parameter(fn,&var,1,ov,&details));
		return var;
	}
	TraceCompilerImpl(Trace * trace) {
		this->trace = trace;
	}

	virtual ~TraceCompilerImpl() {}

	OVRef getOrCreateOutputVar(IRType const & t, std::vector<bool> & allocation) {
		for(size_t i = 0; i < output_variables.size(); i++) {
			if(!allocation[i] && t == output_variables[i].typ) {
				allocation[i] = true;
				return i;
			}
		}
		OutputVar v = { typeFor(t), t };
		allocation.push_back(true);
		output_variables.push_back(v);
		return output_variables.size() - 1;
	}

	void addExit(::TraceExit const & e, bool in_body) {
		exits.push_back(TraceExit());
		TraceExit & exit = exits.back();
		exit.offset = e.offset;
		std::vector<bool> allocation(output_variables.size(),false);


		for(size_t j = 0; j < trace->renaming_table.outputs.size(); j++) {
			const RenamingTable::Entry & entry = trace->renaming_table.outputs[j];
			IRef definition;
			if(  trace->renaming_table.get(entry.location,entry.id,e.snapshot,&definition,NULL,NULL)
			  || (in_body && trace->renaming_table.get(entry.location,entry.id,&definition))) {
				IRType typ = get(definition).typ;
				OVRef var = getOrCreateOutputVar(typ,allocation);
				OVRef length;
				if(typ.isVector)
					length = getOrCreateOutputVar(IRType::Size(),allocation);
				Output output = { definition, entry, var,length};
				exit.outputs.push_back(output);
			}
		}
	}

	void addInput(IRNode const & node) {
		RenamingTable::Entry entry = { node.a, (node.opcode == IROpCode::sload) ? RenamingTable::SLOT : RenamingTable::VARIABLE };
		Input input = { node.typ, typeFor(node.typ),entry };
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
	arbb_variable_t constantSize(size_t i) {
		arbb_global_variable_t gv;
		ARBB_RUN(arbb_create_constant(ctx,&gv,typeFor(IRType::Size()),&i,NULL,&details));
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
				exit += trace->exits.size(); //shift to exit will all nodes defined
				/* fallthrough */
			case L_FIRST: {
				TraceExit & texit = exits[exit];
				for(size_t i = 0; i < texit.outputs.size(); i++) {
					Output & o = texit.outputs[i];
					outputs[0] = references[trace->optimized.size() + o.var];
					inputs[0] = references[o.definition];
					ARBB_RUN(arbb_op(fn,arbb_op_copy,outputs,inputs,NULL,NULL,&details));
					if(get(o.definition).typ.isVector) {
						//write the length as well
						outputs[0] = varFor(o.length);
						ARBB_RUN(arbb_op(fn,arbb_op_length,outputs,inputs,NULL,NULL,&details));
					}
				}
			} break;
			}

			//write the exit code
			outputs[0] = references[trace->optimized.size() + exit_code];
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
				ARBB_RUN(arbb_create_global(ctx,result,typeFor(typ),NULL,binding,NULL,&details));
			} else {
				arbb_set_binding_null(&binding);
				ARBB_RUN(arbb_create_global(ctx,result,typeFor(typ),NULL,binding,NULL,&details));
				ARBB_RUN(arbb_write_scalar(ctx,varFor(*result),&v.p,&details));
			}
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

	//each output needs a bogus value because arbb needs all values defined at function exit
	void initializeOutput(OVRef o, std::vector<arbb_variable_t> & references) {
		OutputVar & ov = output_variables[o];
		arbb_global_variable_t gv;
		int64_t zero = 0;
		ARBB_RUN(arbb_create_constant(ctx,&gv,typeFor(ov.typ.base()),&zero,NULL,&details));
		arbb_variable_t local;
		ARBB_RUN(arbb_create_local(fn,&local,ov.arbb_typ,NULL,&details));
		arbb_variable_t output[] = { local };
		arbb_variable_t input[2];
		input[0] = varFor(gv);
		if(ov.typ.isVector) {
			input[1] = constantInt(0);
			ARBB_RUN(arbb_op(fn,arbb_op_const_vector,output,input,NULL,NULL,&details));
		} else {
			ARBB_RUN(arbb_op(fn,arbb_op_copy,output,input,NULL,NULL,&details));
		}
		references.push_back(local);
	}
	void writeOutput(OVRef o, std::vector<arbb_variable_t> & references) {
		arbb_variable_t input = references[trace->optimized.size() + o];
		arbb_variable_t output = varFor(o);
		ARBB_RUN(arbb_op(fn,arbb_op_copy,&output,&input,NULL,NULL,&details));
	}

	TCStatus compile() {
		ARBB_RUN(arbb_get_default_context(&ctx,&details));


		//for each TraceExit in the trace, we create two exit paths, the first path is for the first iteration of the trace when
		//variables may not have been defined
		for(size_t i = 0; i < trace->exits.size(); i++)
			addExit(trace->exits[i],false);
		for(size_t i = 0; i < trace->exits.size(); i++)
			addExit(trace->exits[i],true);

		{	//initialize exit variable
			exit_code = output_variables.size();
			OutputVar ov = { typeFor(IRType::Int()), IRType::Int()  };
			output_variables.push_back(ov);
		}

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


		//construction function type
		arbb_type_t input_ts[inputs.size()];
		for(size_t i = 0; i < inputs.size(); i++)
			input_ts[i] = inputs[i].arbb_typ;
		arbb_type_t output_ts[output_variables.size()];
		for(size_t i = 0; i < output_variables.size(); i++)
			output_ts[i] = output_variables[i].arbb_typ;

		arbb_type_t fn_type;
		ARBB_RUN(arbb_get_function_type(ctx,&fn_type,output_variables.size(),output_ts,inputs.size(),input_ts,&details));

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

		//initialize output variables (they must be defined along all paths or arbb will fail)
		for(size_t i = 0; i < output_variables.size(); i++) {
			initializeOutput(i,references);
		}

		//emit code!
		size_t if_depth = 0;
		for(size_t i = 0; i < trace->loop_header.size(); i++)
			COMPILER_DO(emitir(trace->loop_header[i],L_HEADER,references,&if_depth));
		//unroll first loop iteration, so exits know what variables are defined
		for(size_t i = 0; i < trace->loop_body.size(); i++)
			COMPILER_DO(emitir(trace->loop_body[i],L_FIRST,references,&if_depth));
		//we now enter a loop
		ARBB_RUN(arbb_begin_loop(fn,arbb_loop_while,&details));
		ARBB_RUN(arbb_begin_loop_block(fn,arbb_loop_block_cond,&details));
		{   //using a constant directly results in "CTE_OPERATOR_NOT_SUPPORTED OP_NOT_SUPPORT: The concrete operator is not supported yet not reach"
			arbb_variable_t input = constantBool(true);
			arbb_variable_t output;
			ARBB_RUN(arbb_create_local(fn,&output,typeFor(IRType::Bool()),NULL,&details));
			ARBB_RUN(arbb_op(fn,arbb_op_copy,&output,&input,NULL,NULL,&details));
			ARBB_RUN(arbb_loop_condition(fn,output,&details));
		}
		ARBB_RUN(arbb_begin_loop_block(fn,arbb_loop_block_body,&details));
		emitPhis(references);
		for(size_t i = 0; i < trace->loop_body.size(); i++)
			COMPILER_DO(emitir(trace->loop_body[i],L_BODY,references,&if_depth));

		ARBB_RUN(arbb_end_loop(fn,&details));

		while(if_depth-- > 0)
			ARBB_RUN(arbb_end_if(fn,&details)); //end all the else statements that the guards produced

		for(size_t i = 0; i < output_variables.size(); i++) {
			writeOutput(i,references);
		}
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

	void loadFromInterpreter(State & s, RenamingTable::Entry const & entry, Value * v) {
		if(entry.location == RenamingTable::SLOT)
			*v = s.registers[entry.id];
		else
			s.global->get(s,Symbol(entry.id),*v);
	}
	void constructValueFromData(IRType const & typ, int64_t length, void * data, Value * v) {
		assert(!typ.isVector && "NYI - loading vector from trace");
		switch(typ.base_type.Enum()) {
			case IRScalarType::E_T_null: *v = Null::singleton; break;
			case IRScalarType::E_T_logical: *v = Logical::c(*(bool*)data); break;
			case IRScalarType::E_T_integer: *v = Integer::c(*(int64_t*)data); break;
			case IRScalarType::E_T_double: *v = Double::c(*(double*)data); break;
			case IRScalarType::E_T_complex: panic("NYI: complex"); break;
			case IRScalarType::E_T_character: panic("NYI: character"); break;
			case IRScalarType::E_T_void: panic("void type is not an arbb type"); break;
			case IRScalarType::E_T_size: *v = Integer::c(*(size_t*)data); break;
			case IRScalarType::E_T_unsupported: panic("successful trace contains unsupported type."); break;
		}
	}

	void storeValue(State & s, Output const & o, arbb_global_variable_t * output_globals) {
		IRType const & typ = get(o.definition).typ;
		Value v;
		if(typ.isVector) {
			uint64_t length;
			ARBB_RUN(arbb_read_scalar(ctx,varFor(output_globals[o.length]),&length,&details));
			void * vdata;
			uint64_t pitch;
			ARBB_RUN(arbb_map_to_host(ctx,varFor(output_globals[o.var]),&vdata,&pitch,arbb_read_write_range,&details));
			//value will own the arbb object
			arbb_refcountable_t rc = arbb_global_variable_to_refcountable(output_globals[o.var]);
			ARBB_RUN(arbb_acquire_ref(rc,&details));
			constructValueFromData(typ,length,vdata,&v);
		} else {
			uint64_t sdata;
			ARBB_RUN(arbb_read_scalar(ctx,varFor(output_globals[o.var]),&sdata,&details));
			constructValueFromData(typ,1,&sdata,&v);
		}
		if(o.interpreter_location.location == RenamingTable::SLOT) {
			s.registers[o.interpreter_location.id] = v;
		} else {
			s.global->assign(Symbol(o.interpreter_location.id),v);
		}
	}

	TCStatus execute(State & s, int64_t * offset) {
		//check type specializations of inputs before creating a bunch of arbb state that we would have to destroy.
		for(size_t i = 0; i < inputs.size(); i++) {
			Input & input = inputs[i];
			Value v;
			loadFromInterpreter(s,input.interpreter_location,&v);
			if(input.typ != IRType(v)) {
				*offset = 0;
				return TCStatus::SUCCESS;
			}
		}

		//create arbb variables to hold inputs
		arbb_global_variable_t input_globals[inputs.size()];
		arbb_variable_t input_vars[inputs.size()];
		arbb_binding_t input_bindings[inputs.size()];
		for(size_t i = 0; i < inputs.size(); i++) {
			Input & input = inputs[i];
			Value v;
			loadFromInterpreter(s,input.interpreter_location,&v);
			if(!loadValueGuarded(input.typ,v,&input_bindings[i],&input_globals[i]))
				return TCStatus::RUNTIME_ERROR;
			input_vars[i] = varFor(input_globals[i]);
		}

		//create arbb variables to hold output values
		arbb_global_variable_t output_globals[output_variables.size()];
		arbb_variable_t output_vars[output_variables.size()];
		for(size_t i = 0; i < output_variables.size(); i++) {
			OutputVar & ov = output_variables[i];
			arbb_binding_t null_binding; arbb_set_binding_null(&null_binding);
			ARBB_RUN(arbb_create_global(ctx,&output_globals[i],ov.arbb_typ,NULL,null_binding,NULL,&details));
			output_vars[i] = varFor(output_globals[i]);
		}

		//execute the trace!
		ARBB_RUN(arbb_execute(fn,output_vars,input_vars,&details));

		//now we have to figure out what happened....
		//first read the exit code
		int64_t ec;
		ARBB_RUN(arbb_read_scalar(ctx,output_vars[exit_code],&ec,&details));
		if(ec == -1) {//we didn't make it out of the loop header, fall back to the interpter, and don't write any results out
			*offset = 0;
		} else { //success! write back the results to the interpreter
			TraceExit & exit = exits[ec];
			*offset = exit.offset;
			for(size_t i = 0; i < exit.outputs.size(); i++) {
				storeValue(s,exit.outputs[i],output_globals);
			}
		}
		//free the global variables and bindings.
		for(size_t i = 0; i < inputs.size(); i++) {
			arbb_refcountable_t gv = arbb_global_variable_to_refcountable(input_globals[i]);
			ARBB_RUN(arbb_release_ref(gv,&details));
			if(!arbb_is_binding_null(input_bindings[i]))
				ARBB_RUN(arbb_free_binding(ctx,input_bindings[i],&details));
		}
		for(size_t i = 0; i < output_variables.size(); i++) {
			//release outputs, if an exit variable is still using the value, it will have called acquire on the reference
			arbb_refcountable_t gv = arbb_global_variable_to_refcountable(output_globals[i]);
			ARBB_RUN(arbb_release_ref(gv,&details));
		}

		return TCStatus::SUCCESS;
	}

	//for internal compiler error problems that are the result of bugs in the compiler
	__attribute__ ((noreturn))
	void panic(const char * file, int line, const char * txt, arbb_error_t error) {
		fprintf(stderr,"%s:%d: trace compiler: %d\n",file,line,(int)error);
		fprintf(stderr,"details: %s\n",arbb_get_error_message(details));
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
