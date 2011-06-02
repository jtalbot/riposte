
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
		case Symbol::E_lor: return ByteCode::lor; break;
		case Symbol::E_slor: return ByteCode::slor; break;
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
	uint64_t length = call.length;
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
			if(Symbol(c[0]) == Symbol::classSym)
				bc = indexed ? ByteCode::iclassassign : ByteCode::classassign;
			else if(Symbol(c[0]) == Symbol::names)
				bc = indexed ? ByteCode::inamesassign : ByteCode::namesassign;
			else if(Symbol(c[0]) == Symbol::dim)
				bc = indexed ? ByteCode::idimassign : ByteCode::dimassign;
			else
				throw CompileError("invalid function on left side of assign");
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
		List parameters(c.length);
		uint64_t j = 0;
		for(uint64_t i = 0; i < parameters.length; i++) {
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
		for(uint64_t i = 0; i < n.length; i++) {
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
		if(call.length == 1) {
			closure.code().push_back(Instruction(ByteCode::null, 0, 0, 0));
			result = registerDepth;
		} else if(call.length == 2)
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
			registerDepth = initialDepth+3; // save space for NULL result and loop variables 
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
			registerDepth = initialDepth+3; // save space for NULL result and loop variables
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
		if(call.length == 4) {
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
		if(call.length == 4)
			closure.code()[begin2-1].a = end-begin2+1;
		registerDepth = initialDepth;
		return registerDepth++;
	} break;
	case Symbol::E_brace: 
	{
		uint64_t length = call.length;
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
		if(call.length == 2) {
			uint64_t a = compile(call[1], closure);
			closure.code().push_back(Instruction(ByteCode::pos, a, 0, initialDepth));
		} else if(call.length == 3) {
			uint64_t a = compile(call[1], closure);
			uint64_t b = compile(call[2], closure);
			closure.code().push_back(Instruction(ByteCode::add, a, b, initialDepth));
		}
		registerDepth = initialDepth;
		return registerDepth++;
	} break;
	case Symbol::E_sub: 
	{
		if(call.length == 2) {
			uint64_t a = compile(call[1], closure);
			closure.code().push_back(Instruction(ByteCode::neg, a, 0, initialDepth));
		} else if(call.length == 3) {
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
	uint64_t length = values.length;
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

