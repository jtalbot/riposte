
#include "compiler.h"
#include "internal.h"

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
		case Symbol::E_lor: return ByteCode::lor; break;
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
		case Symbol::E_Logical: return ByteCode::logical1; break;
		case Symbol::E_Integer: return ByteCode::integer1; break;
		case Symbol::E_Double: return ByteCode::double1; break;
		case Symbol::E_Complex: return ByteCode::complex1; break;
		case Symbol::E_Character: return ByteCode::character1; break;
		case Symbol::E_Raw: return ByteCode::raw1; break;
		case Symbol::E_type: return ByteCode::type; break;
		default: throw RuntimeError("unexpected symbol used as an operator"); break;
	}
}

static void resolveLoopReferences(Code* code, int64_t start, int64_t end, int64_t depth, int64_t nextTarget, int64_t breakTarget) {
	for(int64_t i = start; i < end; i++) {
		if(code->bc[i].bc == ByteCode::next && code->bc[i].a == (int64_t)depth) {
			code->bc[i].a = nextTarget-i;
		} else if(code->bc[i].bc == ByteCode::break1 && code->bc[i].a == (int64_t)depth) {
			code->bc[i].a = breakTarget-i;
		}
	}
}

int64_t Compiler::compileConstant(Value const& expr, Code* code) {
	if(expr.isNull()) code->bc.push_back(Instruction(ByteCode::null, 0, 0, registerDepth));
	else if(expr.isLogical() && expr.length == 1 && Logical::isTrue(Logical(expr)[0])) code->bc.push_back(Instruction(ByteCode::true1, 0, 0, registerDepth));
	else if(expr.isLogical() && expr.length == 1 && Logical::isFalse(Logical(expr)[0])) code->bc.push_back(Instruction(ByteCode::false1, 0, 0, registerDepth));
	else {
		code->constants.push_back(expr);
		code->bc.push_back(Instruction(ByteCode::kget, code->constants.size()-1, 0, registerDepth));
	}
	return registerDepth++;
}

int64_t Compiler::compileSymbol(Symbol const& symbol, Code* code) {
	// search for symbol in variables list
	if(scope.size() > 0) {
		for(uint64_t i = 0; i < scope.back().symbols.size(); i++) {
			if(scope.back().symbols[i] == symbol) {
				code->bc.push_back(Instruction(ByteCode::sget, i, 0, registerDepth));
				return registerDepth++;
			}
		}
	}
	code->bc.push_back(Instruction(ByteCode::get, symbol.i, 0, registerDepth));
	return registerDepth++;
}

int64_t Compiler::compileFunctionCall(Call const& call, Code* code) {
	int64_t initialDepth = registerDepth;
	// a standard call, not an op
	compile(call[0], code);
	CompiledCall compiledCall(call, state);
	// insert call
	code->constants.push_back(compiledCall);
	code->bc.push_back(Instruction(ByteCode::call, code->constants.size()-1, 0, initialDepth));
	registerDepth = initialDepth;
	return registerDepth++;
}

int64_t Compiler::compileCall(Call const& call, Code* code) {
	int64_t length = call.length;
	if(length == 0) {
		throw CompileError("invalid empty call");
	}

	int64_t initialDepth = registerDepth;

	Symbol func = Symbol::NA;
	if(call[0].type == Type::R_symbol)
		func = Symbol(call[0]);
	else if(call[0].type == Type::R_character && call[0].length > 0)
		func = Character(call[0])[0];
	
	switch(func.Enum()) {

	case Symbol::E_internal: 
	{
		// The riposte way... .Internal is a function on state, returning the internal function
		if(call[1].type == Type::R_symbol) {
			code->bc.push_back(Instruction(ByteCode::iget, Symbol(call[1]).i, initialDepth));
			registerDepth = initialDepth;
			return registerDepth++;
		 } else if(call[1].type == Type::R_character && call[1].length > 0) {
			code->bc.push_back(Instruction(ByteCode::iget, Character(call[1])[0].i, initialDepth));
			registerDepth = initialDepth;
			return registerDepth++;
		} else if(call[1].type == Type::R_call) {
			// The R way... .Internal is a function on calls
			Call c = call[1];
			Call ic(2);
			ic[0] = Symbol::internal;
			ic[1] = c[0];
			c[0] = ic;
			return compile(c, code);
		} else {
			throw CompileError(std::string(".Internal has invalid arguments (") + call[1].type.toString() + ")");
		}
	} break;
	case Symbol::E_assign:
	case Symbol::E_eqassign: 
	{
		Value dest = call[1];
		
		// recursively handle emitting assignment instructions...
		Value value = call[2];
		while(dest.isCall()) {
			Call c(dest);
			Call n(c.length+1);
			
			for(int64_t i = 0; i < c.length; i++) { n[i] = c[i]; }
			Character nnames(c.length+1);
			for(int64_t i = 0; i < c.length; i++) { nnames[i] = Symbol::empty; }
			if(hasNames(c)) {
				Insert(state, getNames(c), 0, nnames, 0, c.length);
			}
			n[0] = state.StrToSym(state.SymToStr(Symbol(c[0])) + "<-");
			n[c.length] = value;
			nnames[c.length] = Symbol::value;
			setNames(n, nnames);
			value = n; 
			dest = c[1];
		}
		
		// the source for the assignment
		int64_t source = compile(value, code);

		assert(dest.isSymbol());
		// check if destination is a reserved slot.
		int64_t dest_i = Symbol(dest).i;
		bool slot = false;
		if(scope.size() > 0) {
			for(uint64_t i = 0; i < scope.back().symbols.size(); i++) {
				if(scope.back().symbols[i] == Symbol(dest)) {
					dest_i = i;
					slot = true;
				}
			}
			if(!slot && scope.back().symbols.size() < 32) {
				scope.back().symbols.push_back(Symbol(dest));
				dest_i = scope.back().symbols.size()-1;
				slot = true;
			}
		}
	
		if(slot)	
			code->bc.push_back(Instruction(ByteCode::sassign, dest_i, 0, source));
		else
			code->bc.push_back(Instruction(ByteCode::assign, dest_i, 0, source));
		
		if(source != initialDepth) throw CompileError("unbalanced registers in assign");
		registerDepth = initialDepth;
		return registerDepth++;
	} break;
	case Symbol::E_bracketAssign: { 
		// if there's more than three parameters, we should call the library version...
		if(call.length > 4) return compileFunctionCall(call, code);
		int64_t dest = compile(call[1], code);
		assert(initialDepth == dest);
		int64_t index = compile(call[2], code);
		int64_t value = compile(call[3], code);
		code->bc.push_back(Instruction(ByteCode::iassign, value, index, dest));
		registerDepth = initialDepth;
		return registerDepth++;
	} break;
	case Symbol::E_bbAssign: {
		int64_t dest = compile(call[1], code);
		assert(initialDepth == dest);
		int64_t index = compile(call[2], code);
		int64_t value = compile(call[3], code);
		code->bc.push_back(Instruction(ByteCode::eassign, value, index, dest));
		registerDepth = initialDepth;
		return registerDepth++;
	} break;
	case Symbol::E_function: 
	{
		Scope scope;
		//compile the default parameters	
		List c = PairList(call[1]);
		List parameters(c.length);
		Character names = getNames(c);
		for(int64_t i = 0; i < parameters.length; i++) {
			if(!c[i].isNil()) {
				parameters[i] = Closure(compile(c[i]),NULL);
				parameters[i].type = Type::I_default;
			}
			else {
				parameters[i] = c[i];
			}
			scope.symbols.push_back(names[i]);
		}
		setNames(parameters, names);
		code->constants.push_back(parameters);

		// to support UseMethod we always have to pass on the list of arguments
		scope.symbols.push_back(Symbol::funargs);
		
		//compile the source for the body
		this->scope.push_back(scope);
		Code* functionCode = compile(call[2]);
		functionCode->slotSymbols = this->scope.back().symbols;
		functionCode->registers = 16;
		Closure body(functionCode, NULL);
		code->constants.push_back(body);
		this->scope.pop_back();

		//push back the source code.
		code->constants.push_back(call[3]);
	
		code->bc.push_back(Instruction(ByteCode::function, code->constants.size()-3, code->constants.size()-2, initialDepth));
		registerDepth = initialDepth;
		return registerDepth++;
	} break;
	case Symbol::E_returnSym: 
	{
		// return always writes to the 0 register
		int64_t result;
		if(call.length == 1) {
			code->bc.push_back(Instruction(ByteCode::null, 0, 0, 0));
			result = registerDepth;
		} else if(call.length == 2)
			result = compile(call[1], code);
		else
			throw CompileError("Too many parameters to return. Wouldn't multiple return values be nice?\n");
		code->bc.push_back(Instruction(ByteCode::ret, 0, 0, result));
		registerDepth = 0;
		return registerDepth++;
	} break;
	case Symbol::E_forSym: 
	{
		// special case common i in m:n case
		if(call[2].type == Type::R_call && Symbol(Call(call[2])[0]) == Symbol::colon) {
			int64_t lim1 = compile(Call(call[2])[1], code);
			int64_t lim2 = compile(Call(call[2])[2], code);
			registerDepth = initialDepth+3; // save space for NULL result and loop variables 
			if(lim2 != lim1+1) throw CompileError("limits aren't in adjacent registers");
			code->bc.push_back(Instruction(ByteCode::iforbegin, 0, Symbol(call[1]).i, lim1));
			loopDepth++;
			int64_t beginbody = code->bc.size();
			compile(call[3], code);
			int64_t endbody = code->bc.size();
			resolveLoopReferences(code, beginbody, endbody, loopDepth, endbody, endbody+1);
			loopDepth--;
			code->bc.push_back(Instruction(ByteCode::iforend, beginbody-endbody, Symbol(call[1]).i, lim1));
			code->bc[beginbody-1].a = endbody-beginbody+1;
		}
		else {
			int64_t lim = compile(call[2], code);
			registerDepth = initialDepth+3; // save space for NULL result and loop variables
			code->bc.push_back(Instruction(ByteCode::forbegin, 0, Symbol(call[1]).i, lim));
			loopDepth++;
			int64_t beginbody = code->bc.size();
			compile(call[3], code);
			int64_t endbody = code->bc.size();
			resolveLoopReferences(code, beginbody, endbody, loopDepth, endbody, endbody+1);
			loopDepth--;
			code->bc.push_back(Instruction(ByteCode::forend, beginbody-endbody, Symbol(call[1]).i, lim));
			code->bc[beginbody-1].a = endbody-beginbody+1;
		}
		registerDepth = initialDepth;
		return registerDepth++;
	} break;
	case Symbol::E_whileSym: 
	{
		int64_t lim = compile(call[1], code);
		registerDepth = initialDepth+1; // save space for NULL result
		code->bc.push_back(Instruction(ByteCode::whilebegin, 0, lim, lim));
		loopDepth++;
		int64_t beginbody = code->bc.size();
		compile(call[2], code);
		registerDepth = initialDepth+1; // save space for NULL result
		int64_t beforecond = code->bc.size();
		int64_t lim2 = compile(call[1], code);
		int64_t endbody = code->bc.size();
		resolveLoopReferences(code, beginbody, endbody, loopDepth, beforecond, endbody+1);
		loopDepth--;
		code->bc.push_back(Instruction(ByteCode::whileend, beginbody-endbody, lim2, lim));
		code->bc[beginbody-1].a = endbody-beginbody+2;
		registerDepth = initialDepth;
		return registerDepth++;
	} break;
	case Symbol::E_repeatSym: 
	{
		registerDepth = initialDepth+1; // save space for NULL result
		code->bc.push_back(Instruction(ByteCode::repeatbegin, 0, 0, initialDepth));
		loopDepth++;
		int64_t beginbody = code->bc.size();
		compile(call[1], code);
		int64_t endbody = code->bc.size();
		resolveLoopReferences(code, beginbody, endbody, loopDepth, endbody, endbody+1);
		loopDepth--;
		code->bc.push_back(Instruction(ByteCode::repeatend, beginbody-endbody, 0, initialDepth));
		code->bc[beginbody-1].a = endbody-beginbody+2;
		registerDepth = initialDepth;
		return registerDepth++;
	} break;
	case Symbol::E_nextSym:
	{
		if(loopDepth == 0) throw CompileError("next used outside of loop");
		code->bc.push_back(Instruction(ByteCode::next, loopDepth));
		return registerDepth;
	} break;
	case Symbol::E_breakSym:
	{
		if(loopDepth == 0) throw CompileError("break used outside of loop");
		code->bc.push_back(Instruction(ByteCode::break1, loopDepth));
		return registerDepth;
	} break;
	case Symbol::E_ifSym: 
	{
		code->bc.push_back(Instruction(ByteCode::null, 0, 0, registerDepth++));
		int64_t cond = compile(call[1], code);
		code->bc.push_back(Instruction(ByteCode::if1, 0, cond));
		int64_t begin1 = code->bc.size(), begin2 = 0;
		registerDepth = initialDepth;
		int64_t result = compile(call[2], code);
		if(call.length == 4) {
			code->bc.push_back(Instruction(ByteCode::jmp, 0));
			registerDepth = initialDepth;
			begin2 = code->bc.size();
			int64_t result2 = compile(call[3], code);
			if(result != result2 || result != initialDepth) throw CompileError("then and else blocks don't put the result in the same register");
		}
		else
			begin2 = code->bc.size();
		int64_t end = code->bc.size();
		code->bc[begin1-1].a = begin2-begin1+1;
		if(call.length == 4)
			code->bc[begin2-1].a = end-begin2+1;
		registerDepth = initialDepth;
		return registerDepth++;
	} break;
	case Symbol::E_brace: 
	{
		int64_t length = call.length;
		if(length == 0) {
			code->bc.push_back(Instruction(ByteCode::null, 0, 0, registerDepth));
			return registerDepth++;
		} else {
			int64_t result = initialDepth;
			for(int64_t i = 1; i < length; i++) {
				registerDepth = initialDepth;
				result = compile(call[i], code);
				//if(i < length-1)
				//	code->bc.push_back(Instruction(ByteCode::pop));
			}
			return result;
		}
	} break;
	case Symbol::E_paren: 
	{
		compile(call[1], code);
		return registerDepth;
	} break;
	case Symbol::E_add: 
	{
		if(call.length == 2) {
			int64_t a = compile(call[1], code);
			code->bc.push_back(Instruction(ByteCode::pos, a, 0, initialDepth));
		} else if(call.length == 3) {
			int64_t a = compile(call[1], code);
			int64_t b = compile(call[2], code);
			code->bc.push_back(Instruction(ByteCode::add, a, b, initialDepth));
		}
		registerDepth = initialDepth;
		return registerDepth++;
	} break;
	case Symbol::E_sub: 
	{
		if(call.length == 2) {
			int64_t a = compile(call[1], code);
			code->bc.push_back(Instruction(ByteCode::neg, a, 0, initialDepth));
		} else if(call.length == 3) {
			int64_t a = compile(call[1], code);
			int64_t b = compile(call[2], code);
			code->bc.push_back(Instruction(ByteCode::sub, a, b, initialDepth));
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
	{ 
		int64_t a = compile(call[1], code);
		int64_t b = compile(call[2], code);
		code->bc.push_back(Instruction(op(func), a, b, initialDepth));
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
	case Symbol::E_Logical:
	case Symbol::E_Integer:
	case Symbol::E_Double:
	case Symbol::E_Complex:
	case Symbol::E_Character:
	case Symbol::E_Raw:
	case Symbol::E_type:
	{ 
		int64_t a = compile(call[1], code);
		code->bc.push_back(Instruction(op(func), a, 0, initialDepth));
		registerDepth = initialDepth;
		return registerDepth++;
	} break;
	// Shortcut operators
	case Symbol::E_sland:
	{
		code->bc.push_back(Instruction(ByteCode::false1, 0, 0, registerDepth++));
		int64_t cond = compile(call[1], code);
		code->bc.push_back(Instruction(ByteCode::if1, 0, cond));
		int64_t begin1 = code->bc.size();
		registerDepth = initialDepth;
		int64_t cond2 = compile(call[2], code);
		code->bc.push_back(Instruction(ByteCode::istrue, cond2, 0, initialDepth));
		int64_t end = code->bc.size();
		code->bc[begin1-1].a = end-begin1+1;
		registerDepth = initialDepth;
		return registerDepth++;
	} break;
	case Symbol::E_slor:
	{
		code->bc.push_back(Instruction(ByteCode::true1, 0, 0, registerDepth++));
		int64_t cond = compile(call[1], code);
		code->bc.push_back(Instruction(ByteCode::if0, 0, cond));
		int64_t begin1 = code->bc.size();
		registerDepth = initialDepth;
		int64_t cond2 = compile(call[2], code);
		code->bc.push_back(Instruction(ByteCode::istrue, cond2, 0, initialDepth));
		int64_t end = code->bc.size();
		code->bc[begin1-1].a = end-begin1+1;
		registerDepth = initialDepth;
		return registerDepth++;
	} break;
	case Symbol::E_UseMethod:
	{
		int64_t generic = compile(call[1], code);
		if(call.length == 3) compile(call[2], code);
		code->bc.push_back(Instruction(ByteCode::UseMethod, generic, call.length==3 ? 1 : 0, initialDepth));
		registerDepth = initialDepth;
		return registerDepth++;
	} break;
	case Symbol::E_seq_len:
	{
		int64_t len = compile(call[1], code);
		code->bc.push_back(Instruction(ByteCode::seq, len, 0, initialDepth));
		registerDepth = initialDepth;
		return registerDepth++;
	} break;
	default:
	{
		return compileFunctionCall(call, code);
	}
	};
}

int64_t Compiler::compileExpression(Expression const& values, Code* code) {
	int64_t initialDepth = registerDepth;
	int64_t result;
	int64_t length = values.length;
	for(int64_t i = 0; i < length; i++) {
		registerDepth = initialDepth;
		result = compile(values[i], code);
		//if(i < length-1)
		//	code->bc.push_back(Instruction(ByteCode::pop));
	}
	registerDepth = initialDepth;
	return registerDepth++;
}

int64_t Compiler::compile(Value const& expr, Code* code) {
	switch(expr.type.Enum())
	{
		case Type::E_R_symbol:
			return compileSymbol(Symbol(expr), code);
			break;
		case Type::E_R_call:
			return compileCall(Call(expr), code);
			break;
		case Type::E_R_expression:
			return compileExpression(Expression(expr), code);
			break;
		default:
			return compileConstant(expr, code);
			break;
	};
}

Code* Compiler::compile(Value const& expr) {
	Code* code = new Code();

	int64_t oldLoopDepth = loopDepth;
	loopDepth = 0;
	int64_t oldRegisterDepth = registerDepth;
	registerDepth = 0;
	compile(expr, code);
	code->expression = expr;
	// insert return statement at end of code
	code->bc.push_back(Instruction(ByteCode::ret));
	registerDepth = oldRegisterDepth;
	loopDepth = oldLoopDepth;

	return code;	
}

