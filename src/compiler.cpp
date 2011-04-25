#include <string>
#include <sstream>
#include <stdexcept>
#include <string>

#include "value.h"
#include "type.h"
#include "bc.h"

// compilation routines
static void compile(State& state, Value const& expr, Closure& closure); 

static void compileConstant(State& state, Value const& expr, Closure& closure) {
	closure.constants().push_back(expr);
	closure.code().push_back(Instruction(ByteCode::kget, closure.constants().size()-1));
}

static void compileGetSymbol(State& state, Symbol const& symbol, Closure& closure) {
	closure.code().push_back(Instruction(ByteCode::get, symbol.i));
}

static void compileOp(State& state, Call const& call, Closure& closure) {
	Symbol func(call[0]);
	std::string funcStr = func.toString(state);
	if(funcStr == ".Internal") {
		assert(call[1].type == Type::R_symbol);
		closure.code().push_back(Instruction(ByteCode::iget, Symbol(call[1]).i));
	}
	else if(funcStr == "<-" || funcStr == "=" || funcStr == ".Assign") {
		ByteCode bc;
		Value v = call[1];
		
		// the source for the assignment
		compile(state, call[2], closure);
		
		// any indexing code
		bool indexed = false;
		if(v.type == Type::R_call && state.outString(Call(v)[0].i) == "[") {
			Call c(v);
			compile(state, c[2], closure);
			v = c[1];
			indexed = true;
		}
		
		if(v.type == Type::R_call) {
			Call c(v);
			if(state.outString(c[0].i) == "class")
				bc = indexed ? ByteCode::iclassassign : ByteCode::classassign;
			else if(state.outString(c[0].i) == "names")
				bc = indexed ? ByteCode::inamesassign : ByteCode::namesassign;
			else if(state.outString(c[0].i) == "dim")
				bc = indexed ? ByteCode::idimassign : ByteCode::dimassign;
			v = c[1];
		} else {
			bc = indexed ? ByteCode::iassign : ByteCode::assign;
		}
		closure.code().push_back(Instruction(bc, Symbol(v).i));
	}
	else if(funcStr == "function") {
		//compile the default parameters	
		List c = PairList(call[1]);
		List parameters(c.length());
		uint64_t j = 0;
		for(uint64_t i = 0; i < parameters.length(); i++) {
			parameters[j] = compile(state, c[i]);
			parameters[j].type = Type::I_default;
			j++;
		}
		Vector n = getNames(c.attributes);
		if(n.type != Type::R_null) {
			setNames(parameters.attributes, n);
		}
		closure.constants().push_back(parameters);

		//compile the source for the body
		Closure body = compile(state, call[2]);
		closure.constants().push_back(body);
	
		closure.code().push_back(Instruction(ByteCode::function, closure.constants().size()-2, closure.constants().size()-1));
	}
	else if(funcStr == "return") {
		if(call.length() == 1)
			closure.code().push_back(Instruction(ByteCode::null));
		else if(call.length() == 2)
			compile(state, call[1], closure);
		else
			printf("Too many parameters to return. Wouldn't multiple return values be nice?\n");
		closure.code().push_back(Instruction(ByteCode::ret));
	}
	else if(funcStr == "for" || funcStr == ".For") {
		compile(state, call[2], closure);
		closure.code().push_back(Instruction(ByteCode::forbegin, 0, Symbol(call[1]).i));
		uint64_t beginbody = closure.code().size();
		compile(state, call[3], closure);
		uint64_t endbody = closure.code().size();
		closure.code().push_back(Instruction(ByteCode::forend, endbody-beginbody, Symbol(call[1]).i));
		closure.code()[beginbody-1].a = endbody-beginbody+1;
	}
	else if(funcStr == "while" || funcStr == ".While") {
		compile(state, call[1], closure);
		closure.code().push_back(Instruction(ByteCode::whilebegin, 0));
		uint64_t beginbody = closure.code().size();
		compile(state, call[2], closure);
		compile(state, call[1], closure);
		uint64_t endbody = closure.code().size();
		closure.code().push_back(Instruction(ByteCode::whileend, endbody-beginbody));
		closure.code()[beginbody-1].a = endbody-beginbody+2;
	}
	else if(funcStr == "repeat" || funcStr == ".Repeat") {
		closure.code().push_back(Instruction(ByteCode::repeatbegin, 0));
		uint64_t beginbody = closure.code().size();
		compile(state, call[1], closure);
		uint64_t endbody = closure.code().size();
		closure.code().push_back(Instruction(ByteCode::repeatend, endbody-beginbody));
		closure.code()[beginbody-1].a = endbody-beginbody+2;
	}
	else if(funcStr == "if" || funcStr == ".If") {
		compile(state, call[1], closure);
		closure.code().push_back(Instruction(ByteCode::if1, 0));
		uint64_t begin1 = closure.code().size(), begin2 = 0;
		compile(state, call[2], closure);
		if(call.length() == 4) {
			closure.code().push_back(Instruction(ByteCode::jmp, 0));
			begin2 = closure.code().size();
			compile(state, call[3], closure);
		}
		else
			begin2 = closure.code().size();
		uint64_t end = closure.code().size();
		closure.code()[begin1-1].a = begin2-begin1+1;
		if(call.length() == 4)
			closure.code()[begin2-1].a = end-begin2+1;
	}
	else if(funcStr == ".Brace" || funcStr == "{") {
		uint64_t length = call.length();
		for(uint64_t i = 1; i < length; i++) {
			compile(state, call[i], closure);
			if(i < length-1)
				closure.code().push_back(Instruction(ByteCode::pop));
		}
		if(length == 0) {
			closure.code().push_back(Instruction(ByteCode::null));
		}
	}
	else if(funcStr == ".Paren" || funcStr == "(") {
		//uint64_t length = call.length();
		compile(state, call[1], closure);
	}
	else if(funcStr == ":") {
		compile(state, call[1], closure);
		compile(state, call[2], closure);
		closure.code().push_back(Instruction(ByteCode::colon, call.length()-1));
	}
	else if(funcStr == ".Add" || funcStr == "+") {
		if(call.length() == 3)
			compile(state, call[2], closure);
		compile(state, call[1], closure);
		if(call.length() == 3)
			closure.code().push_back(Instruction(ByteCode::add, call.length()-1));
		else
			closure.code().push_back(Instruction(ByteCode::pos, call.length()-1));
	}
	else if(funcStr == ".Sub" || funcStr == "-") {
		if(call.length() == 3)
			compile(state, call[2], closure);
		compile(state, call[1], closure);
		if(call.length() == 3)
			closure.code().push_back(Instruction(ByteCode::sub, call.length()-1));
		else
			closure.code().push_back(Instruction(ByteCode::neg, call.length()-1));
	}
	else if(funcStr == ".Mul" || funcStr == "*") {
		compile(state, call[2], closure);
		compile(state, call[1], closure);
		closure.code().push_back(Instruction(ByteCode::mul, call.length()-1));
	}
	else if(funcStr == ".Div" || funcStr == "/") {
		compile(state, call[2], closure);
		compile(state, call[1], closure);
		closure.code().push_back(Instruction(ByteCode::div, call.length()-1));
	}
	else if(funcStr == ".IDiv" || funcStr == "%/%") {
		compile(state, call[2], closure);
		compile(state, call[1], closure);
		closure.code().push_back(Instruction(ByteCode::idiv, call.length()-1));
	}
	else if(funcStr == ".Pow" || funcStr == "^" || funcStr == "**") {
		compile(state, call[2], closure);
		compile(state, call[1], closure);
		closure.code().push_back(Instruction(ByteCode::pow, call.length()-1));
	}
	else if(funcStr == ".Mod" || funcStr == "%%") {
		compile(state, call[2], closure);
		compile(state, call[1], closure);
		closure.code().push_back(Instruction(ByteCode::mod, call.length()-1));
	}
	else if(funcStr == ".Lneg" || funcStr == "!") {
		compile(state, call[2], closure);
		compile(state, call[1], closure);
		closure.code().push_back(Instruction(ByteCode::lneg, call.length()-1));
	}
	else if(funcStr == ".Land" || funcStr == "&") {
		compile(state, call[2], closure);
		compile(state, call[1], closure);
		closure.code().push_back(Instruction(ByteCode::land, call.length()-1));
	}
	else if(funcStr == ".Sland" || funcStr == "&&") {
		compile(state, call[2], closure);
		compile(state, call[1], closure);
		closure.code().push_back(Instruction(ByteCode::sland, call.length()-1));
	}
	else if(funcStr == ".Lor" || funcStr == "|") {
		compile(state, call[2], closure);
		compile(state, call[1], closure);
		closure.code().push_back(Instruction(ByteCode::lor, call.length()-1));
	}
	else if(funcStr == ".Slor" || funcStr == "||") {
		compile(state, call[2], closure);
		compile(state, call[1], closure);
		closure.code().push_back(Instruction(ByteCode::slor, call.length()-1));
	}
	else if(funcStr == ".Eq" || funcStr == "==") {
		compile(state, call[2], closure);
		compile(state, call[1], closure);
		closure.code().push_back(Instruction(ByteCode::eq, call.length()-1));
	}
	else if(funcStr == ".Neq" || funcStr == "!=") {
		compile(state, call[2], closure);
		compile(state, call[1], closure);
		closure.code().push_back(Instruction(ByteCode::neq, call.length()-1));
	}
	else if(funcStr == ".LT" || funcStr == "<") {
		compile(state, call[2], closure);
		compile(state, call[1], closure);
		closure.code().push_back(Instruction(ByteCode::lt, call.length()-1));
	}
	else if(funcStr == ".LE" || funcStr == "<=") {
		compile(state, call[2], closure);
		compile(state, call[1], closure);
		closure.code().push_back(Instruction(ByteCode::le, call.length()-1));
	}
	else if(funcStr == ".GT" || funcStr == ">") {
		compile(state, call[2], closure);
		compile(state, call[1], closure);
		closure.code().push_back(Instruction(ByteCode::gt, call.length()-1));
	}
	else if(funcStr == ".GE" || funcStr == ">=") {
		compile(state, call[2], closure);
		compile(state, call[1], closure);
		closure.code().push_back(Instruction(ByteCode::ge, call.length()-1));
	}
	else if(funcStr == ".Abs" || funcStr == "abs") {
		compile(state, call[1], closure);
		closure.code().push_back(Instruction(ByteCode::abs, call.length()-1));
	}
	else if(funcStr == ".Sign" || funcStr == "sign") {
		compile(state, call[1], closure);
		closure.code().push_back(Instruction(ByteCode::sign, call.length()-1));
	}
	else if(funcStr == ".Sqrt" || funcStr == "sqrt") {
		compile(state, call[1], closure);
		closure.code().push_back(Instruction(ByteCode::sqrt, call.length()-1));
	}
	else if(funcStr == ".Floor" || funcStr == "floor") {
		compile(state, call[1], closure);
		closure.code().push_back(Instruction(ByteCode::floor, call.length()-1));
	}
	else if(funcStr == ".Ceiling" || funcStr == "ceiling") {
		compile(state, call[1], closure);
		closure.code().push_back(Instruction(ByteCode::ceiling, call.length()-1));
	}
	else if(funcStr == ".Trunc" || funcStr == "trunc") {
		compile(state, call[1], closure);
		closure.code().push_back(Instruction(ByteCode::trunc, call.length()-1));
	}
	else if(funcStr == ".Round" || funcStr == "round") {
		compile(state, call[1], closure);
		closure.code().push_back(Instruction(ByteCode::round, call.length()-1));
	}
	else if(funcStr == ".Signif" || funcStr == "signif") {
		compile(state, call[1], closure);
		closure.code().push_back(Instruction(ByteCode::signif, call.length()-1));
	}
	else if(funcStr == ".Exp" || funcStr == "exp") {
		compile(state, call[1], closure);
		closure.code().push_back(Instruction(ByteCode::exp, call.length()-1));
	}
	else if(funcStr == ".Log" || funcStr == "log") {
		compile(state, call[1], closure);
		closure.code().push_back(Instruction(ByteCode::log, call.length()-1));
	}
	else if(funcStr == ".Cos" || funcStr == "cos") {
		compile(state, call[1], closure);
		closure.code().push_back(Instruction(ByteCode::cos, call.length()-1));
	}
	else if(funcStr == ".Sin" || funcStr == "sin") {
		compile(state, call[1], closure);
		closure.code().push_back(Instruction(ByteCode::sin, call.length()-1));
	}
	else if(funcStr == ".Tan" || funcStr == "tan") {
		compile(state, call[1], closure);
		closure.code().push_back(Instruction(ByteCode::tan, call.length()-1));
	}
	else if(funcStr == ".ACos" || funcStr == "acos") {
		compile(state, call[1], closure);
		closure.code().push_back(Instruction(ByteCode::acos, call.length()-1));
	}
	else if(funcStr == ".ASin" || funcStr == "asin") {
		compile(state, call[1], closure);
		closure.code().push_back(Instruction(ByteCode::asin, call.length()-1));
	}
	else if(funcStr == ".ATan" || funcStr == "atan") {
		compile(state, call[1], closure);
		closure.code().push_back(Instruction(ByteCode::atan, call.length()-1));
	}
}

static void compileCall(State& state, Call const& call, Closure& closure) {
	uint64_t length = call.length();
	if(length == 0) {
		printf("call without any stuff\n");
		return;
	}

	if(call[0].type == Type::R_symbol) {
		std::string funcStr = Symbol(call[0]).toString(state);
		if(	funcStr == ".Internal"
			|| funcStr == "function"
			|| funcStr == "return"
			|| funcStr == "<-" 
			|| funcStr == "=" 
			|| funcStr == "for"
			|| funcStr == "while"
			|| funcStr == "repeat"
			|| funcStr == "if"
			|| funcStr == "{"
			|| funcStr == "("
			|| funcStr == ":"
			|| funcStr == "+"
			|| funcStr == "-"
			|| funcStr == "*"
			|| funcStr == "/"
			|| funcStr == "^"
			|| funcStr == "**"
			|| funcStr == "%/%"
			|| funcStr == "%%"
			|| funcStr == "!"
			|| funcStr == "&"
			|| funcStr == "&&"
			|| funcStr == "|"
			|| funcStr == "||"
			|| funcStr == "=="
			|| funcStr == "!="
			|| funcStr == "<"
			|| funcStr == "<="
			|| funcStr == ">"
			|| funcStr == ">="
			|| funcStr == "abs"
			|| funcStr == "sign"
			|| funcStr == "sqrt"
			|| funcStr == "floor"
			|| funcStr == "ceiling"
			|| funcStr == "trunc"
			|| funcStr == "round"
			|| funcStr == "signif"
			|| funcStr == "exp"
			|| funcStr == "log"
			|| funcStr == "cos"
			|| funcStr == "sin"
			|| funcStr == "tan"
			|| funcStr == "acos"
			|| funcStr == "asin"
			|| funcStr == "atan"
			) {
			compileOp(state, call, closure);
			return; 
		}
	}
	// create a promise for each parameter...
	CompiledCall compiledCall(call, state);
	/*Call compiledCall(call.length());
	compiledCall.attributes = call.attributes;

	bool dots = false;
	for(uint64_t i = length-1; i >= 1; i--) {
		if(call[i].type == Type::R_symbol) {
			if(call[i].i == DOTS_STRING) {
				compiledCall[i] = call[i];
				dots = true;
			}
			else {
				Value v = call[i];
				v.type = Type::I_sympromise;
				compiledCall[i] = v;
			}
		}
		else if(isLanguage(call[i])) {
			Value v;
			compile(state, call[i]).toValue(v);
			v.type = Type::I_promise;
			compiledCall[i] = v;
		} else {
			compiledCall[i] = call[i];
		}
	}*/
	compile(state, call[0], closure);

	// insert call
	closure.constants().push_back(compiledCall);
	closure.code().push_back(Instruction(ByteCode::call, closure.constants().size()-1));
}

static void compileICCall(State& state, Call const& call, Closure& closure) {
	uint64_t length = call.length();
	if(length == 0) {
		printf("call without any stuff\n");
		return;
	}

	// we might be able to inline if the function is a known symbol
	//  and if no parameter is '...'
			
			/*compileCall(state, call, closure);
			
			Value spec_value;
			state.baseenv->get(state, Symbol(call[0]), spec_value);
			closure.constants().push_back(spec_value);
			uint64_t spec_value_index = closure.constants().size()-1;

			// check needs 1) function, 2) specialized value, 3) expensive call, and 4) skip amount
			Instruction& instr = closure.code().back();
			instr.bc = ByteCode::inlinecall;
			instr.b = spec_value_index;
			instr.c = 0;*/
			
			//uint64_t start = closure.code().size();
			//compileInternalCall(state, InternalCall(call), closure);
			//uint64_t end = closure.code().size();
			//instr.c = end-start+1;
			//return;
	//	}
	//}
	// generate a normal call
	compileCall(state, call, closure);

}

static void compileExpression(State& state, Expression const& values, Closure& closure) {
	uint64_t length = values.length();
	for(uint64_t i = 0; i < length; i++) {
		compile(state, values[i], closure);
		if(i < length-1)
			closure.code().push_back(Instruction(ByteCode::pop));
	}
}

void compile(State& state, Value const& expr, Closure& closure) {

	switch(expr.type.internal())
	{
		case Type::ER_symbol:
			compileGetSymbol(state, Symbol(expr), closure);
			break;
		case Type::ER_call:
			compileICCall(state, Call(expr), closure);
			break;
		case Type::ER_expression:
			compileExpression(state, Expression(expr), closure);
			break;
		default:
			compileConstant(state, expr, closure);
			break;
	};
}

Closure compile(State& state, Value const& expr) {
	Closure closure;
	compile(state, expr, closure);
	closure.expression() = expr;
	// insert return statement at end of closure
	closure.code().push_back(Instruction(ByteCode::ret));
	return closure;	
}


/*
void functionCall(Value const& func, Value const* values, uint64_t length, Environment* env, Value& result) {
	Function const& f = asFunction(func);
	// create a new environment for the function call 
	// (stack discipline is hard to prove in R, but can we do better?)
    // 1. If a function is returned, must assume that it contains upvalues to anything in either
	//    its static scope (true upvalues) or dynamic scope (promises)
    // 2. Upvalues can be contained in eval'ed code! consider, function(x) return(function() eval(parse(text='x')))
    // 3. Functions can be held in any non-basic datatype (though lists seem like the obvious possibility)
	// 4. So by (2) can't statically check, 
	//		by (3) we'd have to traverse entire returned data structure to check for a function.
	// 5. More conservatively, we could flag the creation of any function within a scope,
	//		if that scope returns a non-basic type, we'd have to move the environment off the stack.
    //    ---but, what about updating references to the environment. Ugly since we don't know
	//	  ---which function is the problem or which upvalues will be used.
	// Conclusion for now: heap allocate environments. Try to make that fast, maybe with a pooled allocator...
	Environment* call_env = new Environment(f.s, env);
	// populate with parameters
	Character names(f.args.names());
	for(uint64_t i = 0; i < length; i++) {
		call_env->assign(names[i], values[i]);
	}
	// call interpret
	eval(Closure(f.body), call_env, values, length, result);	
}

void functionCallInternal(Value const& func, Value const* values, uint64_t length, Environment* env, Value& result) {
	CFunction const& f = asCFunction(func);
	f.func(env, values, length, result);
}

void eval(Closure const& closure, Environment* env, Value const* slots, uint64_t slength, Value& result) {
	Value registers[16];
	Promise promises[16];
	uint64_t pindex = 0;
	const uint64_t length = closure.inner->code.size();
	for(uint64_t i = 0; i < length; i++) {
		Instruction const& inst = closure.inner->code[i];
		switch(inst.bc.internal()) {
			case ByteCode::call:
			case ByteCode::ccall:
			{
				Value func(registers[inst.a]);
				uint64_t start = inst.a+1;
				uint64_t length = inst.b;
			
				if(func.type() == Type::R_function) {
					functionCall(func, &registers[start], length, env, registers[inst.c]);
				} else if(func.type() == Type::R_cfunction) {
					functionCallInternal(func, &registers[start], length, env, registers[inst.c]);
				} else {
					printf("Non-function as first parameter to call\n");
				}
			} break;
			case ByteCode::slot:
				registers[inst.c] = slots[inst.a];
			break;
			case ByteCode::get:
				env->get(Symbol(inst.a), registers[inst.c]);
			break;
			case ByteCode::kget:
				registers[inst.c] = closure.inner->constants[inst.a];
			break;
			case ByteCode::delay:
				promises[inst.c].set(closure.inner->constants[inst.a], closure.inner->constants[inst.b], env);
				Value::set(registers[inst.c], Type::R_promise, &promises[inst.c]);
			break;
			case ByteCode::assign:
				env->assign(Symbol(registers[inst.a]), registers[inst.c]);
			break;
			case ByteCode::zip2:
				zip2(registers[inst.a], registers[inst.b], registers[inst.c], registers[inst.op]);
			break;
			case ByteCode::forbegin:
				env->assign(Symbol(inst.a), registers[inst.c]);
				if(asReal1(registers[inst.c]) > asReal1(registers[inst.b]))
					i = i + inst.op;
			break;
			case ByteCode::forend:
				Value::setDouble(registers[inst.c], asReal1(registers[inst.c])+1);
				if(asReal1(registers[inst.c]) <= asReal1(registers[inst.b])) {
					env->assign(Symbol(inst.a), registers[inst.c]);
					i = i - inst.op;
				} else {
					Value::set(registers[inst.c], Type::R_null, 0);
				}
			break;
			case ByteCode::function:
				Value::set(registers[inst.c], Type::R_function, new Function(List(registers[inst.a]), Closure(registers[inst.b]), env));
			break;
			case ByteCode::quote:
				if(registers[inst.a].type() == Type::R_promise)
					asPromise(registers[inst.a]).inner(registers[inst.c]);
				else
					registers[inst.c] = registers[inst.a];
				//env->getQuoted(Symbol(inst.a), registers[inst.c]);
			break;
			case ByteCode::force:
				if(registers[inst.a].type() == Type::R_promise)
					asPromise(registers[inst.a]).eval(registers[inst.c]);
				else
					registers[inst.c] = registers[inst.a];
			break;
			case ByteCode::forceall: {
				for(uint64_t i = 0; i < slength; i++) {
					if(slots[i].type() == Type::R_promise)
						asPromise(slots[i]).eval(registers[inst.c]);
					else
						registers[inst.c] = slots[i];
				}
			} break;
			case ByteCode::code:
				if(registers[inst.a].type() == Type::R_promise)
					asPromise(registers[inst.a]).code(registers[inst.c]);
				else
					registers[inst.c] = registers[inst.a];
				//env->getCode(Symbol(inst.a), registers[inst.c]);
			break;
		}
	}
	result = registers[closure.inner->code[length-1].c];
}

void eval(Closure const& closure, Environment* env, Value& result) {
	eval(closure, env, 0, 0, result);	
}

void compile(Value& expr, Environment* env, Closure& closure) {
	
	switch(expr.type().internal())
	{
		case Type::R_null:
		case Type::R_raw:
		case Type::R_logical:
		case Type::R_integer:
		case Type::R_double:
		case Type::R_scalardouble:
		case Type::R_complex:		
		case Type::R_character:
		case Type::R_list:
		case Type::R_pairlist:
		case Type::R_function:
		case Type::R_cfunction:
		case Type::R_promise:
		case Type::R_default:
		case Type::ByteCode:
			closure.inner->constants.push_back(expr);
			closure.inner->code.push_back(Instruction(ByteCode::kget, closure.inner->constants.size()-1,0,closure.inner->reg++));
			break;
		case Type::R_symbol:
			closure.inner->code.push_back(Instruction(ByteCode::get, Symbol(expr).index(), 0,closure.inner->reg++));
			break;
		case Type::R_call:
		{
			Call call(expr);
			uint64_t length = call.length();
			if(length == 0) printf("call without any stuff\n");
			uint64_t start = closure.inner->reg;
			compile(call[0], env, closure);

			// create a new closure for each parameter...
			// insert delay instruction to make promise
			for(uint64_t i = 1; i < length; i++) {
				if(isLanguage(call[i])) {
					Closure b;
					compile(call[i], env, b);
					Value v;
					b.toValue(v);
					closure.inner->constants.push_back(v);
					closure.inner->constants.push_back(call[i]);
					closure.inner->code.push_back(Instruction(ByteCode::delay, closure.inner->constants.size()-2,closure.inner->constants.size()-1,closure.inner->reg++));
				} else {
					compile(call[i], env, closure);
				}
			}
	
			// insert call
			closure.inner->code.push_back(Instruction(ByteCode::call, start, closure.inner->reg-1-start, start));
			closure.inner->reg = start+1;
		} break;
		case Type::R_internalcall: {
			InternalCall call(expr);
			Symbol func(call[0]);
			if(func.toString() == ".Assign") {
				uint64_t start = closure.inner->reg;
				compile(call[1], env, closure);
				uint64_t a = closure.inner->reg-1;
				compile(call[2], env, closure);
				uint64_t b = closure.inner->reg-1;
				compile(call[3], env, closure);
				uint64_t c = closure.inner->reg-1;
				closure.inner->code.push_back(Instruction(ByteCode::assign, 
					a,
					c,
					b));
				closure.inner->reg = start;
			}
			else if(func.toString() == ".Slot") {
				closure.inner->code.push_back(Instruction(ByteCode::slot, 
					asReal1(call[1]),
					0,
					closure.inner->reg++));
			}
			else if(func.toString() == ".Zip2") {
				uint64_t start = closure.inner->reg;
				compile(call[1], env, closure);
				uint64_t a = closure.inner->reg-1;
				compile(call[2], env, closure);
				uint64_t b = closure.inner->reg-1;
				compile(call[3], env, closure);
				uint64_t c = closure.inner->reg-1;
				closure.inner->code.push_back(Instruction(ByteCode::zip2, 
					a,
					b,
					start,
					c));
				closure.inner->reg = start+1;
			}
			else if(func.toString() == ".Brace") {
				InternalCall call(expr);
				uint64_t length = call.length();
				uint64_t start = closure.inner->reg;
				for(uint64_t i = 1; i < length; i++) {
					uint64_t istart = closure.inner->reg;
					compile(call[i], env, closure);
					closure.inner->reg = istart;
				}
				closure.inner->reg = start+1;
			}
			else if(func.toString() == ".Paren") {
				InternalCall call(expr);
				uint64_t length = call.length();
				if(length == 2) {
					uint64_t start = closure.inner->reg;
					compile(call[1], env, closure);
					closure.inner->reg = start+1;
				}
			}
			else if(func.toString() == ".For") {
				uint64_t start = closure.inner->reg;
				closure.inner->constants.push_back(call[1]);
				closure.inner->code.push_back(Instruction(ByteCode::kget, closure.inner->constants.size()-1,0,closure.inner->reg++));
				uint64_t lvar = closure.inner->reg-1;
	
				// FIXME: to special common case "i in x:y", need to check if ':' has been replaced, also only works if stepping forward...
				compile(Call(call[2])[1], env, closure);
				uint64_t lower = closure.inner->reg-1;
				compile(Call(call[2])[2], env, closure);
				uint64_t upper = closure.inner->reg-1;
				uint64_t begin = closure.inner->code.size();
				closure.inner->code.push_back(Instruction(ByteCode::forbegin, lvar, upper, lower));
				compile(call[3], env, closure);
				uint64_t endbody = closure.inner->code.size();
				closure.inner->code.push_back(Instruction(ByteCode::forend, lvar, upper, lower, endbody-upper-1));
				closure.inner->code[begin].op = endbody-begin;
				closure.inner->reg = start+1;
			}
			else if(func.toString() == ".Function") {
				// two parameters: argument as list and body
				uint64_t start = closure.inner->reg;
				compile(call[1], env, closure);
				uint64_t args = closure.inner->reg-1;
				//Closure b;
				//compile(call[2], env, b);
				//Value v;
				//b.toValue(v);
				//compile(v, env, closure);
				compile(call[2], env, closure);
				uint64_t body = closure.inner->reg-1;
				closure.inner->code.push_back(Instruction(ByteCode::function, args, body, start, 0));
				closure.inner->reg = start+1;
			}
			else if(func.toString() == ".RawFunction") {
				// two parameters: argument as list and body
				uint64_t start = closure.inner->reg;
				Closure b;
				compile(call[1], env, b);
				Value v;
				b.toValue(v);
				compile(v, env, closure);
				uint64_t body = closure.inner->reg-1;
				closure.inner->code.push_back(Instruction(ByteCode::rawfunction, 0, body, start, 0));
				closure.inner->reg = start+1;
			}
			else if(func.toString() == ".Quote") {
				uint64_t start = closure.inner->reg;
				compile(call[1], env, closure);
				uint64_t arg = closure.inner->reg-1;
				closure.inner->code.push_back(Instruction(ByteCode::quote, arg, 0, start, 0));
				closure.inner->reg = start+1;
				
				//closure.inner->code.push_back(Instruction(ByteCode::quote, Symbol(call[1]).index(), 0,closure.inner->reg++));
			}
			else if(func.toString() == ".Force") {
				uint64_t start = closure.inner->reg;
				compile(call[1], env, closure);
				uint64_t arg = closure.inner->reg-1;
				closure.inner->code.push_back(Instruction(ByteCode::force, arg, 0, start, 0));
				closure.inner->reg = start+1;
			}
			else if(func.toString() == ".ForceAll") {
				uint64_t start = closure.inner->reg;
				closure.inner->code.push_back(Instruction(ByteCode::forceall, 0, 0, start, 0));
				closure.inner->reg = start+1;
			}
			else if(func.toString() == ".Code") {
				uint64_t start = closure.inner->reg;
				compile(call[1], env, closure);
				uint64_t arg = closure.inner->reg-1;
				closure.inner->code.push_back(Instruction(ByteCode::code, arg, 0, start, 0));
				closure.inner->reg = start+1;
				//closure.inner->code.push_back(Instruction(ByteCode::code, Symbol(call[1]).index(), 0,closure.inner->reg++));
			}
			else if(func.toString() == ".Closure") {
				Closure b;
				compile(call[1], env, b);
				Value v;
				b.toValue(v);
				compile(v, env, closure);
			}
			else if(func.toString() == ".List") {
				Value v;
				call.subset(1, call.length()-1, v);
				v.t = Type::R_list;
				List l(v);
				if(call.names().type() == Type::R_character)
					Character(call.names()).subset(1, call.length()-1, l.inner->names);
				l.toValue(v);
				compile(v, env, closure);
			}
			else if(func.toString() == ".Const") {
				closure.inner->constants.push_back(call[1]);
				closure.inner->code.push_back(Instruction(ByteCode::kget, closure.inner->constants.size()-1,0,closure.inner->reg++));
			}
		} break;	
		case Type::R_expression:
		{
			Expression values(expr);
			uint64_t length = values.length();
			for(uint64_t i = 0; i < length; i++) {
				compile(values[i], env, closure);
			}
		} break;
	};
}
*/


/*void functionCall(Value const& func, Call const& values, Environment* env, Value& result) {
	// create a new environment for the function call
	Environment call_env;
	
	Function const& f = asFunction(func);
	call_env.initialize(f.s, env);
	// populate with parameters
	uint64_t length = values.length();
	for(uint64_t i = 0; i < length-1; i++) {
		if(isLanguage(values[i+1]))
			call_env.assign(f.formals[i].name, values[i+1], env);
		else
			call_env.assign(f.formals[i].name, values[i+1]);
	}
	// call interpret
	interpret(f.body, &call_env, result);	
}

void functionCallInternal(Value const& func, Call const& values, Environment* env, Value& result) {
	// static stack seems to be a bit faster than stack allocating slots
	static Value parameters[256];
	static Promise promises[256];
	static uint64_t index = 0;
	
	CFunction const& f = asCFunction(func);
	// populate with parameters
	uint64_t length = values.length();
	for(uint64_t i = 0; i < length-1; i++) {
		if(isLanguage(values[i+1])) {
			promises[i+index].set(values[i+1], values[i+1], env);
			Value::set(parameters[i+index], Type::R_promise, &promises[i+index]);
		} else
			parameters[i+index] = values[i+1];
	}
	// call internal
	index += length-1;
	f.func(env, &parameters[index-(length-1)], length-1, result);
	index -= length-1;
}

void vm(Closure const& closure, Environment* env, Value& result);

void interpret(Value const& expr, Environment* env, Value& result) {
	switch(expr.type().internal())
	{
		case Type::ByteCode:
		{
			Closure b(expr);
			vm(b, env, result);
		} break;
		case Type::R_null:
		case Type::R_raw:
		case Type::R_logical:
		case Type::R_integer:
		case Type::R_double:
		case Type::R_scalardouble:
		case Type::R_complex:		
		case Type::R_character:
		case Type::R_list:
		case Type::R_pairlist:
		case Type::R_function:
		case Type::R_cfunction:
			result = expr;
			break;			// don't have to do anything for primitive types
		case Type::R_symbol:
			env->get(Symbol(expr), result);
			break;
		case Type::R_call:
		{
			Call call(expr);
			uint64_t length = call.length();
			if(length == 0) printf("call without any stuff\n");
			Value func;
			interpret(call[0], env, func);
			
			if(func.type() == Type::R_function) {
				functionCall(func, call, env, result);
			} else if(func.type() == Type::R_cfunction) {
				functionCallInternal(func, call, env, result);
			} else {
				printf("Non-function as first parameter to call\n");
			}
		} 	break;
		case Type::R_expression:
		{
			Expression statements(expr);
			uint64_t length = statements.length();
			for(uint64_t i = 0; i < length; i++) {
				interpret(statements[i], env, result);
			}
		} 	break;
		case Type::R_promise:
		case Type::R_default:
			printf("promise or default value exposed at interpreter?\n");
			break;
	};
}*/
