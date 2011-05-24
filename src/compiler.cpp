
#include "compiler.h"

static ByteCode op(Symbol const& s) {
	switch(s.Enum()) {
		case Symbol::E_colon: return ByteCode::colon; break;
		case Symbol::E_mul: return ByteCode::mul; break;
		case Symbol::E_div: return ByteCode::div; break;
		case Symbol::E_idiv: return ByteCode::idiv; break;
		case Symbol::E_mod: return ByteCode::mod; break;
		case Symbol::E_pow: return ByteCode::pow; break;
		case Symbol::E_lt: return ByteCode::lt; break;
		case Symbol::E_gt: return ByteCode::gt; break;
		case Symbol::E_eq: return ByteCode::eq; break;
		case Symbol::E_neq: return ByteCode::neq; break;
		case Symbol::E_ge: return ByteCode::ge; break;
		case Symbol::E_le: return ByteCode::le; break;
		case Symbol::E_lnot: return ByteCode::lnot; break;
		case Symbol::E_land: return ByteCode::land; break;
		case Symbol::E_sland: return ByteCode::sland; break;
		case Symbol::E_lor: return ByteCode::slor; break;
		case Symbol::E_abs: return ByteCode::abs; break;
		case Symbol::E_sign: return ByteCode::sign; break;
		case Symbol::E_sqrt: return ByteCode::sqrt; break;
		case Symbol::E_floor: return ByteCode::floor; break;
		case Symbol::E_ceiling: return ByteCode::ceiling; break;
		case Symbol::E_trunc: return ByteCode::trunc; break;
		case Symbol::E_round: return ByteCode::round; break;
		case Symbol::E_signif: return ByteCode::signif; break;
		case Symbol::E_exp: return ByteCode::exp; break;
		case Symbol::E_log: return ByteCode::log; break;
		case Symbol::E_cos: return ByteCode::cos; break;
		case Symbol::E_sin: return ByteCode::sin; break;
		case Symbol::E_tan: return ByteCode::tan; break;
		case Symbol::E_acos: return ByteCode::acos; break;
		case Symbol::E_asin: return ByteCode::asin; break;
		case Symbol::E_atan: return ByteCode::atan; break;
		default: throw RuntimeError("unexpected symbol used as an operator"); break;
	}
}

static void resolveLoopReferences(Closure& closure, uint64_t start, uint64_t end, uint64_t depth, uint64_t nextTarget, uint64_t breakTarget) {
	for(uint64_t i = start; i < end; i++) {
		if(closure.code()[i].bc == ByteCode::next && closure.code()[i].a == (int64_t)depth) {
			closure.code()[i].a = nextTarget-i;
		} else if(closure.code()[i].bc == ByteCode::break1 && closure.code()[i].a == (int64_t)depth) {
			closure.code()[i].a = breakTarget-i;
		}
	}
}

uint64_t Compiler::compileConstant(Value const& expr, Closure& closure) {
	closure.constants().push_back(expr);
	closure.code().push_back(Instruction(ByteCode::kget, closure.constants().size()-1, 0, registerDepth));
	return registerDepth++;
}

uint64_t Compiler::compileSymbol(Symbol const& symbol, Closure& closure) {
	closure.code().push_back(Instruction(ByteCode::get, symbol.i, 0, registerDepth));
	return registerDepth++;
}

uint64_t Compiler::compileCall(Call const& call, Closure& closure) {
	uint64_t length = call.length();
	if(length == 0) {
		throw CompileError("invalid empty call");
	}

	uint64_t initialDepth = registerDepth;

	Symbol func = Symbol::E_NA;
	if(call[0].type == Type::R_symbol)
		func = Symbol(call[0]);
	
	switch(func.Enum()) {

	case Symbol::E_internal: 
	{
		if(call[1].type == Type::R_symbol) {
			// The riposte way... .Internal is a function on symbols, returning the internal function
			closure.code().push_back(Instruction(ByteCode::iget, Symbol(call[1]).i, initialDepth));
			registerDepth = initialDepth;
			return registerDepth++;
		} else if(call[1].type == Type::R_call) {
			// The R way... .Internal is a function on calls
			Call c = call[1];
			Call ic(2);
			ic[0] = Symbol(state, ".Internal");
			ic[1] = c[0];
			c[0] = ic;
			return compile(c, closure);
		} else {
			throw CompileError(".Internal has invalid arguments");
		}
	} break;
	case Symbol::E_assign:
	case Symbol::E_eqassign: 
	{
		ByteCode bc;
		Value v = call[1];
		
		// the source for the assignment
		uint64_t source = compile(call[2], closure);
		
		// any indexing code
		bool indexed = false;
		uint64_t index = 0;
		if(v.type == Type::R_call && state.outString(Call(v)[0].i) == "[") {
			Call c(v);
			index = compile(c[2], closure);
			v = c[1];
			indexed = true;
		}
		
		if(v.type == Type::R_call) {
			Call c(v);
			if(c[0] == Symbol::classSym)
				bc = indexed ? ByteCode::iclassassign : ByteCode::classassign;
			else if(c[0] == Symbol::names)
				bc = indexed ? ByteCode::inamesassign : ByteCode::namesassign;
			else if(c[0] == Symbol::dim)
				bc = indexed ? ByteCode::idimassign : ByteCode::dimassign;
			v = c[1];
		} else {
			bc = indexed ? ByteCode::iassign : ByteCode::assign;
		}
		closure.code().push_back(Instruction(bc, Symbol(v).i, index, source));
		if(source != initialDepth) throw CompileError("unbalanced registers in assign");
		registerDepth = initialDepth;
		return registerDepth++;
	} break;
	case Symbol::E_function: 
	{
		//compile the default parameters	
		List c = PairList(call[1]);
		List parameters(c.length());
		uint64_t j = 0;
		for(uint64_t i = 0; i < parameters.length(); i++) {
			parameters[j] = compile(c[i]);
			parameters[j].type = Type::I_default;
			j++;
		}
		Vector n = getNames(c.attributes);
		if(n.type != Type::R_null) {
			setNames(parameters.attributes, n);
		}
		closure.constants().push_back(parameters);

		/*CompileState s(state);
		s.inFunction = true;
		for(uint64_t i = 0; i < n.length(); i++) {
			s.slots.push_back(Character(n)[i]);
		}*/	

		//compile the source for the body
		Closure body = compile(call[2]);
		closure.constants().push_back(body);
	
		closure.code().push_back(Instruction(ByteCode::function, closure.constants().size()-2, closure.constants().size()-1, initialDepth));
		registerDepth = initialDepth;
		return registerDepth++;
	} break;
	case Symbol::E_returnSym: 
	{
		// return always writes to the 0 register
		uint64_t result;
		if(call.length() == 1) {
			closure.code().push_back(Instruction(ByteCode::null, 0, 0, 0));
			result = registerDepth;
		} else if(call.length() == 2)
			result = compile(call[1], closure);
		else
			throw CompileError("Too many parameters to return. Wouldn't multiple return values be nice?\n");
		closure.code().push_back(Instruction(ByteCode::ret, result, 0, 0));
		registerDepth = 0;
		return registerDepth++;
	} break;
	case Symbol::E_forSym: 
	{
		// special case common i in m:n case
		if(call[2].type == Type::R_call && state.outString(Symbol(Call(call[2])[0]).i) == ":") {
			uint64_t lim1 = compile(Call(call[2])[1], closure);
			uint64_t lim2 = compile(Call(call[2])[2], closure);
			registerDepth = initialDepth+1; // save space for NULL result
			if(lim2 != lim1+1) throw CompileError("limits aren't in adjacent registers");
			closure.code().push_back(Instruction(ByteCode::iforbegin, 0, Symbol(call[1]).i, lim1));
			loopDepth++;
			uint64_t beginbody = closure.code().size();
			compile(call[3], closure);
			uint64_t endbody = closure.code().size();
			resolveLoopReferences(closure, beginbody, endbody, loopDepth, endbody, endbody+1);
			loopDepth--;
			closure.code().push_back(Instruction(ByteCode::iforend, beginbody-endbody, Symbol(call[1]).i, lim1));
			closure.code()[beginbody-1].a = endbody-beginbody+1;
		}
		else {
			uint64_t lim = compile(call[2], closure);
			registerDepth = initialDepth+1; // save space for NULL result
			closure.code().push_back(Instruction(ByteCode::forbegin, 0, Symbol(call[1]).i, lim));
			loopDepth++;
			uint64_t beginbody = closure.code().size();
			compile(call[3], closure);
			uint64_t endbody = closure.code().size();
			resolveLoopReferences(closure, beginbody, endbody, loopDepth, endbody, endbody+1);
			loopDepth--;
			closure.code().push_back(Instruction(ByteCode::forend, beginbody-endbody, Symbol(call[1]).i, lim));
			closure.code()[beginbody-1].a = endbody-beginbody+1;
		}
		registerDepth = initialDepth;
		return registerDepth++;
	} break;
	case Symbol::E_whileSym: 
	{
		uint64_t lim = compile(call[1], closure);
		registerDepth = initialDepth+1; // save space for NULL result
		closure.code().push_back(Instruction(ByteCode::whilebegin, 0, lim, lim));
		loopDepth++;
		uint64_t beginbody = closure.code().size();
		compile(call[2], closure);
		registerDepth = initialDepth+1; // save space for NULL result
		uint64_t beforecond = closure.code().size();
		uint64_t lim2 = compile(call[1], closure);
		uint64_t endbody = closure.code().size();
		resolveLoopReferences(closure, beginbody, endbody, loopDepth, beforecond, endbody+1);
		loopDepth--;
		closure.code().push_back(Instruction(ByteCode::whileend, beginbody-endbody, lim2, lim));
		closure.code()[beginbody-1].a = endbody-beginbody+2;
		registerDepth = initialDepth;
		return registerDepth++;
	} break;
	case Symbol::E_repeatSym: 
	{
		registerDepth = initialDepth+1; // save space for NULL result
		closure.code().push_back(Instruction(ByteCode::repeatbegin, 0, 0, initialDepth));
		loopDepth++;
		uint64_t beginbody = closure.code().size();
		compile(call[1], closure);
		uint64_t endbody = closure.code().size();
		resolveLoopReferences(closure, beginbody, endbody, loopDepth, endbody, endbody+1);
		loopDepth--;
		closure.code().push_back(Instruction(ByteCode::repeatend, beginbody-endbody, 0, initialDepth));
		closure.code()[beginbody-1].a = endbody-beginbody+2;
		registerDepth = initialDepth;
		return registerDepth++;
	} break;
	case Symbol::E_nextSym:
	{
		if(loopDepth == 0) throw CompileError("next used outside of loop");
		closure.code().push_back(Instruction(ByteCode::next, loopDepth));
		return registerDepth;
	} break;
	case Symbol::E_breakSym:
	{
		if(loopDepth == 0) throw CompileError("break used outside of loop");
		closure.code().push_back(Instruction(ByteCode::break1, loopDepth));
		return registerDepth;
	} break;
	case Symbol::E_ifSym: 
	{
		uint64_t cond = compile(call[1], closure);
		closure.code().push_back(Instruction(ByteCode::if1, 0, cond));
		uint64_t begin1 = closure.code().size(), begin2 = 0;
		registerDepth = initialDepth;
		uint64_t result = compile(call[2], closure);
		if(call.length() == 4) {
			closure.code().push_back(Instruction(ByteCode::jmp, 0));
			registerDepth = initialDepth;
			begin2 = closure.code().size();
			uint64_t result2 = compile(call[3], closure);
			if(result != result2 || result != initialDepth) throw CompileError("then and else blocks don't put the result in the same register");
		}
		else
			begin2 = closure.code().size();
		uint64_t end = closure.code().size();
		closure.code()[begin1-1].a = begin2-begin1+1;
		if(call.length() == 4)
			closure.code()[begin2-1].a = end-begin2+1;
		registerDepth = initialDepth;
		return registerDepth++;
	} break;
	case Symbol::E_brace: 
	{
		uint64_t length = call.length();
		if(length == 0) {
			closure.code().push_back(Instruction(ByteCode::null, 0, 0, registerDepth));
			return registerDepth++;
		} else {
			uint64_t result = initialDepth;
			for(uint64_t i = 1; i < length; i++) {
				registerDepth = initialDepth;
				result = compile(call[i], closure);
				//if(i < length-1)
				//	closure.code().push_back(Instruction(ByteCode::pop));
			}
			return result;
		}
	} break;
	case Symbol::E_paren: 
	{
		compile(call[1], closure);
		return registerDepth;
	} break;
	case Symbol::E_add: 
	{
		if(call.length() == 2) {
			uint64_t a = compile(call[1], closure);
			closure.code().push_back(Instruction(ByteCode::pos, a, 0, initialDepth));
		} else if(call.length() == 3) {
			uint64_t a = compile(call[1], closure);
			uint64_t b = compile(call[2], closure);
			closure.code().push_back(Instruction(ByteCode::add, a, b, initialDepth));
		}
		registerDepth = initialDepth;
		return registerDepth++;
	} break;
	case Symbol::E_sub: 
	{
		if(call.length() == 2) {
			uint64_t a = compile(call[1], closure);
			closure.code().push_back(Instruction(ByteCode::neg, a, 0, initialDepth));
		} else if(call.length() == 3) {
			uint64_t a = compile(call[1], closure);
			uint64_t b = compile(call[2], closure);
			closure.code().push_back(Instruction(ByteCode::sub, a, b, initialDepth));
		}
		registerDepth = initialDepth;
		return registerDepth++;
	} break;
	// Binary operators
	case Symbol::E_colon:
	case Symbol::E_mul: 
	case Symbol::E_div: 
	case Symbol::E_idiv: 
	case Symbol::E_pow: 
	case Symbol::E_mod:
	case Symbol::E_land:
	case Symbol::E_lor:
	case Symbol::E_eq:
	case Symbol::E_neq:
	case Symbol::E_lt:
	case Symbol::E_gt:
	case Symbol::E_ge:
	case Symbol::E_le:
	case Symbol::E_sland:
	case Symbol::E_slor:
	{ 
		uint64_t a = compile(call[1], closure);
		uint64_t b = compile(call[2], closure);
		closure.code().push_back(Instruction(op(func), a, b, initialDepth));
		registerDepth = initialDepth;
		return registerDepth++;
	} break;
	// Unary operators
	case Symbol::E_lnot: 
	case Symbol::E_abs: 
	case Symbol::E_sign: 
	case Symbol::E_sqrt: 
	case Symbol::E_floor: 
	case Symbol::E_ceiling: 
	case Symbol::E_trunc: 
	case Symbol::E_round: 
	case Symbol::E_signif: 
	case Symbol::E_exp: 
	case Symbol::E_log: 
	case Symbol::E_cos: 
	case Symbol::E_sin: 
	case Symbol::E_tan: 
	case Symbol::E_acos: 
	case Symbol::E_asin: 
	case Symbol::E_atan:
	{ 
		uint64_t a = compile(call[1], closure);
		closure.code().push_back(Instruction(op(func), a, 0, initialDepth));
		registerDepth = initialDepth;
		return registerDepth++;
	} break;

	default:
	{
		// a standard call, not an op
		compile(call[0], closure);
		CompiledCall compiledCall(call, state);
		// insert call
		closure.constants().push_back(compiledCall);
		closure.code().push_back(Instruction(ByteCode::call, closure.constants().size()-1, 0, initialDepth));
		registerDepth = initialDepth;
		return registerDepth++;
	}
	};
}

uint64_t Compiler::compileExpression(Expression const& values, Closure& closure) {
	uint64_t initialDepth = registerDepth;
	uint64_t result;
	uint64_t length = values.length();
	for(uint64_t i = 0; i < length; i++) {
		registerDepth = initialDepth;
		result = compile(values[i], closure);
		//if(i < length-1)
		//	closure.code().push_back(Instruction(ByteCode::pop));
	}
	registerDepth = initialDepth;
	return registerDepth++;
}

uint64_t Compiler::compile(Value const& expr, Closure& closure) {

	switch(expr.type.Enum())
	{
		case Type::E_R_symbol:
			return compileSymbol(Symbol(expr), closure);
			break;
		case Type::E_R_call:
			return compileCall(Call(expr), closure);
			break;
		case Type::E_R_expression:
			return compileExpression(Expression(expr), closure);
			break;
		default:
			return compileConstant(expr, closure);
			break;
	};
}

Closure Compiler::compile(Value const& expr) {
	Closure closure;

	uint64_t oldLoopDepth = loopDepth;
	loopDepth = 0;
	uint64_t oldRegisterDepth = registerDepth;
	registerDepth = 0;
	compile(expr, closure);
	closure.expression() = expr;
	// insert return statement at end of closure
	closure.code().push_back(Instruction(ByteCode::ret));
	registerDepth = oldRegisterDepth;
	loopDepth = oldLoopDepth;

	return closure;	
}

/*
static void compileICCall(CompileState& state, Call const& call, Closure& closure) {
	uint64_t length = call.length();
	if(length == 0) {
		throw CompileError("invalid empty call");
		return;
	}

	// we might be able to inline if the function is a known symbol
	//  and if no parameter is '...'
			
			compileCall(state, call, closure);
			
			Value spec_value;
			state.baseenv->get(state, Symbol(call[0]), spec_value);
			closure.constants().push_back(spec_value);
			uint64_t spec_value_index = closure.constants().size()-1;

			// check needs 1) function, 2) specialized value, 3) expensive call, and 4) skip amount
			Instruction& instr = closure.code().back();
			instr.bc = ByteCode::inlinecall;
			instr.b = spec_value_index;
			instr.c = 0;
			
			//uint64_t start = closure.code().size();
			//compileInternalCall(state, InternalCall(call), closure);
			//uint64_t end = closure.code().size();
			//instr.c = end-start+1;
			//return;
	//	}
	//}
	// generate a normal call
	compileCall(state, call, closure);

}*/

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
