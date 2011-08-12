
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
		case Symbol::E_sland: return ByteCode::sland; break;
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

void Compiler::emit(Code* code, ByteCode bc, int64_t a, int64_t b, int64_t c) {
	code->bc.push_back(Instruction(bc, a, b, c));
}

static void resolveLoopReferences(Code* code, int64_t start, int64_t end, int64_t nextTarget, int64_t breakTarget) {
	for(int64_t i = start; i < end; i++) {
		if(code->bc[i].bc == ByteCode::next && code->bc[i].a == 0) {
			code->bc[i].a = nextTarget-i;
		} else if(code->bc[i].bc == ByteCode::break1 && code->bc[i].a == 0) {
			code->bc[i].a = breakTarget-i;
		}
	}
}

int64_t Compiler::compileConstant(Value const& expr, Code* code) {
	code->constants.push_back(expr);
	int64_t reg = scopes.back().allocRegister(Register::CONSTANT);
	emit(code, ByteCode::kget, code->constants.size()-1, 0, reg);
	return reg;
}

int64_t Compiler::compileSymbol(Symbol const& symbol, Code* code) {
	// search for symbol in variables list
	if(!scopes.back().topLevel) {
		for(uint64_t i = 0; i < scopes.back().symbols.size(); i++) {
			if(scopes.back().symbols[i] == symbol) {
				int64_t reg = scopes.back().allocRegister(Register::VARIABLE);
				emit(code, ByteCode::sget, i, 0, reg);
				return reg;
			}
		}
	}
	int64_t reg = scopes.back().allocRegister(Register::VARIABLE);
	emit(code, ByteCode::get, symbol.i, 0, reg);
	return reg;
}

CompiledCall Compiler::makeCall(Call const& call) {
	// compute compiled call...precompiles promise code and some necessary values
	int64_t dots = call.length-1;
	List arguments(call.length-1);
	for(int64_t i = 1; i < call.length; i++) {
		if(call[i].type == Type::R_symbol && Symbol(call[i]) == Symbol::dots) {
			arguments[i-1] = call[i];
			dots = i-1;
		} else if(call[i].type == Type::R_call ||
		   call[i].type == Type::R_symbol ||
		   call[i].type == Type::R_pairlist) {
			// promises should have access to the slots of the enclosing scope, but have their own register assignments
			arguments[i-1] = Closure(Compiler::compile(call[i]),NULL);
		} else {
			arguments[i-1] = call[i];
		}
	}
	if(hasNames(call)) {
		setNames(arguments, Subset(Character(getNames(call)), 1, call.length-1));
		// reserve room for cached name matching...
	}
	return CompiledCall(call, arguments, dots);
}

// a standard call, not an op
int64_t Compiler::compileFunctionCall(Call const& call, Code* code) {
	int64_t liveIn = scopes.back().live();
	int64_t function = compile(call[0], code);
	CompiledCall compiledCall(makeCall(call));
	code->constants.push_back(compiledCall);
	scopes.back().deadAfter(liveIn);
	int64_t result = scopes.back().allocRegister(Register::VARIABLE);
	emit(code, ByteCode::call, function, code->constants.size()-1, result);
	return result;
}

int64_t Compiler::compileCall(Call const& call, Code* code) {
	int64_t length = call.length;
	if(length == 0) {
		throw CompileError("invalid empty call");
	}

	int64_t liveIn = scopes.back().live();

	Symbol func = Symbol::NA;
	if(call[0].type == Type::R_symbol)
		func = Symbol(call[0]);
	else if(call[0].type == Type::R_character && call[0].length > 0)
		func = Character(call[0])[0];
	
	switch(func.Enum()) {

	case Symbol::E_internal: 
	{
		// The riposte way... .Internal is a function on symbols, returning the internal function
		if(call[1].type == Type::R_symbol) {
			int64_t reg = scopes.back().allocRegister(Register::VARIABLE);
			emit(code, ByteCode::iget, Symbol(call[1]).i, 0, reg);
			return reg;
		 } else if(call[1].type == Type::R_character && call[1].length > 0) {
			int64_t reg = scopes.back().allocRegister(Register::VARIABLE);
			emit(code, ByteCode::iget, Character(call[1])[0].i, 0, reg);
			return reg;
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
		if(!scopes.back().topLevel) {
			for(uint64_t i = 0; i < scopes.back().symbols.size(); i++) {
				if(scopes.back().symbols[i] == Symbol(dest)) {
					dest_i = i;
					slot = true;
				}
			}
			if(!slot && scopes.back().symbols.size() < 32) {
				scopes.back().symbols.push_back(Symbol(dest));
				dest_i = scopes.back().symbols.size()-1;
				slot = true;
			}
		}
	
		if(slot)
			emit(code, ByteCode::sassign, dest_i, 0, source);
		else
			emit(code, ByteCode::assign, dest_i, 0, source);
	
		scopes.back().deadAfter(source);	
		return source;
	} break;
	case Symbol::E_bracketAssign: { 
		// if there's more than three parameters, we should call the library version...
		if(call.length > 4) return compileFunctionCall(call, code);
		int64_t dest = compile(call[1], code);
		int64_t index = compile(call[2], code);
		int64_t value = compile(call[3], code);
		emit(code, ByteCode::iassign, value, index, dest);
		scopes.back().deadAfter(dest);	
		return dest;
	} break;
	case Symbol::E_bbAssign: {
		int64_t dest = compile(call[1], code);
		int64_t index = compile(call[2], code);
		int64_t value = compile(call[3], code);
		emit(code, ByteCode::eassign, value, index, dest);
		scopes.back().deadAfter(dest);	
		return dest;
	} break;
	case Symbol::E_function: 
	{
		Scope scope;
		scope.topLevel = false;
		//compile the default parameters	
		List c = List(PairList(call[1]));
		List parameters(c.length);
		Character names = getNames(c);
		scope.parameters = names;
		for(int64_t i = 0; i < parameters.length; i++) {
			if(!c[i].isNil()) {
				parameters[i] = Closure(compile(c[i]),NULL);
			}
			else {
				parameters[i] = c[i];
			}
			scope.symbols.push_back(names[i]);
		}
		setNames(parameters, names);
		code->constants.push_back(parameters);

		//compile the source for the body
		scopes.push_back(scope);
		Code* functionCode = compile(call[2]);
		functionCode->slotSymbols.swap(scopes.back().symbols);
		Closure body(functionCode, NULL);
		code->constants.push_back(body);
		scopes.pop_back();

		//push back the source code.
		code->constants.push_back(call[3]);

		scopes.back().deadAfter(liveIn);	
		int64_t reg = scopes.back().allocRegister(Register::CONSTANT);	
		emit(code, ByteCode::function, code->constants.size()-3, code->constants.size()-2, reg);
		return reg;
	} break;
	case Symbol::E_returnSym: 
	{
		int64_t result;
		if(call.length == 1) {
			result = compile(Null::Singleton(), code);
		} else if(call.length == 2)
			result = compile(call[1], code);
		else
			throw CompileError("Too many parameters to return. Wouldn't multiple return values be nice?\n");
		emit(code, ByteCode::ret, 0, 0, result);
		scopes.back().deadAfter(result);
		return result;
	} break;
	case Symbol::E_forSym: 
	{
		int64_t result = compile(Null::Singleton(), code);
		// special case common i in m:n case
		if(call[2].type == Type::R_call && Symbol(Call(call[2])[0]) == Symbol::colon) {
			int64_t lim1 = compile(Call(call[2])[1], code);
			int64_t lim2 = compile(Call(call[2])[2], code);
			if(lim1+1 != lim2) throw CompileError("limits aren't in adjacent registers");
			emit(code, ByteCode::iforbegin, 0, Symbol(call[1]).i, lim2);
			loopDepth++;
			int64_t beginbody = code->bc.size();
			compile(call[3], code);
			int64_t endbody = code->bc.size();
			resolveLoopReferences(code, beginbody, endbody, endbody, endbody+1);
			loopDepth--;
			emit(code, ByteCode::iforend, beginbody-endbody, Symbol(call[1]).i, lim2);
			code->bc[beginbody-1].a = endbody-beginbody+1;
		}
		else {
			int64_t loop_vector = compile(call[2], code);
			int64_t loop_var = scopes.back().allocRegister(Register::VARIABLE);	// save space for loop variable
			if(loop_var != loop_vector+1) throw CompileError("limits aren't in adjacent registers");
			emit(code, ByteCode::forbegin, 0, Symbol(call[1]).i, loop_var);
			loopDepth++;
			int64_t beginbody = code->bc.size();
			compile(call[3], code);
			int64_t endbody = code->bc.size();
			resolveLoopReferences(code, beginbody, endbody, endbody, endbody+1);
			loopDepth--;
			emit(code, ByteCode::forend, beginbody-endbody, Symbol(call[1]).i, loop_var);
			code->bc[beginbody-1].a = endbody-beginbody+1;
		}
		scopes.back().deadAfter(result);
		return result;
	} break;
	case Symbol::E_whileSym: 
	{
		int64_t result = compile(Null::Singleton(), code);
		int64_t lim = compile(call[1], code);
		emit(code, ByteCode::whilebegin, 0, lim, result);
		loopDepth++;
		int64_t beginbody = code->bc.size();
		compile(call[2], code);
		int64_t beforecond = code->bc.size();
		int64_t lim2 = compile(call[1], code);
		int64_t endbody = code->bc.size();
		resolveLoopReferences(code, beginbody, endbody, beforecond, endbody+1);
		loopDepth--;
		emit(code, ByteCode::whileend, beginbody-endbody, lim2, result);
		code->bc[beginbody-1].a = endbody-beginbody+2;
		scopes.back().deadAfter(result);
		return result;
	} break;
	case Symbol::E_repeatSym: 
	{
		int64_t result = compile(Null::Singleton(), code);
		emit(code, ByteCode::repeatbegin, 0, 0, result);
		loopDepth++;
		int64_t beginbody = code->bc.size();
		compile(call[1], code);
		int64_t endbody = code->bc.size();
		resolveLoopReferences(code, beginbody, endbody, endbody, endbody+1);
		loopDepth--;
		emit(code, ByteCode::repeatend, beginbody-endbody, 0, result);
		code->bc[beginbody-1].a = endbody-beginbody+2;
		scopes.back().deadAfter(result);
		return result;
	} break;
	case Symbol::E_nextSym:
	{
		if(loopDepth == 0) throw CompileError("next used outside of loop");
		int64_t result = scopes.back().allocRegister(Register::TEMP);
		emit(code, ByteCode::next, 0, 0, result);
		return result;
	} break;
	case Symbol::E_breakSym:
	{
		if(loopDepth == 0) throw CompileError("break used outside of loop");
		int64_t result = scopes.back().allocRegister(Register::TEMP);
		emit(code, ByteCode::break1, 0, 0, result);
		return result;
	} break;
	case Symbol::E_ifSym: 
	{
		int64_t resultT=0, resultF=0;
		if(call.length != 3 && call.length != 4)	
			throw CompileError("invalid if statement");
		if(call.length == 3)
			resultF = compile(Null::Singleton(), code);
		int64_t cond = compile(call[1], code);
		emit(code, ByteCode::if1, 0, cond, liveIn);
		int64_t begin1 = code->bc.size(), begin2 = 0;
		scopes.back().deadAfter(liveIn);
		resultT = compile(call[2], code);

		if(call.length == 4) {
			emit(code, ByteCode::jmp, 0, 0, 0);
			scopes.back().deadAfter(liveIn);
			begin2 = code->bc.size();
			resultF = compile(call[3], code);
		}
		else
			begin2 = code->bc.size();
		int64_t end = code->bc.size();
		code->bc[begin1-1].a = begin2-begin1+1;
		if(call.length == 4)
			code->bc[begin2-1].a = end-begin2+1;
	
		// TODO: if this can ever happen, should probably just insert a move into the lower numbered register	
		if(resultT != resultF) throw CompileError(std::string("then and else blocks don't put the result in the same register ") + intToStr(resultT) + " " + intToStr(resultF));
		scopes.back().deadAfter(resultT);
		return resultT;
	} break;
	case Symbol::E_brace: 
	{
		int64_t length = call.length;
		if(length <= 1) {
			return compile(Null::Singleton(), code);
		} else {
			int64_t result;
			for(int64_t i = 1; i < length; i++) {
				scopes.back().deadAfter(liveIn);
				result = compile(call[i], code);
			}
			scopes.back().deadAfter(result);
			return result;
		}
	} break;
	case Symbol::E_paren: 
	{
		return compile(call[1], code);
	} break;
	case Symbol::E_add: 
	{
		int64_t result = 0;
		if(call.length != 2 && call.length != 3)
			throw CompileError("invalid addition");
		if(call.length == 2) {
			int64_t a = compile(call[1], code);
			scopes.back().deadAfter(liveIn);
			result = scopes.back().allocRegister(Register::TEMP);
			emit(code, ByteCode::pos, a, 0, result);
		} else if(call.length == 3) {
			int64_t a = compile(call[1], code);
			int64_t b = compile(call[2], code);
			scopes.back().deadAfter(liveIn);
			result = scopes.back().allocRegister(Register::TEMP);
			emit(code, ByteCode::add, a, b, result);
		}
		return result;
	} break;
	case Symbol::E_sub: 
	{
		int64_t result = 0;
		if(call.length != 2 && call.length != 3)
			throw CompileError("invalid addition");
		if(call.length == 2) {
			int64_t a = compile(call[1], code);
			scopes.back().deadAfter(liveIn);
			result = scopes.back().allocRegister(Register::TEMP);
			emit(code, ByteCode::neg, a, 0, result);
		} else if(call.length == 3) {
			int64_t a = compile(call[1], code);
			int64_t b = compile(call[2], code);
			scopes.back().deadAfter(liveIn);
			result = scopes.back().allocRegister(Register::TEMP);
			emit(code, ByteCode::sub, a, b, result);
		}
		return result;
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
	case Symbol::E_slor:
	case Symbol::E_sland:
	case Symbol::E_eq:
	case Symbol::E_neq:
	case Symbol::E_lt:
	case Symbol::E_gt:
	case Symbol::E_ge:
	case Symbol::E_le:
	{ 
		int64_t a = compile(call[1], code);
		int64_t b = compile(call[2], code);
		scopes.back().deadAfter(liveIn);
		int64_t result = scopes.back().allocRegister(Register::TEMP);
		emit(code, op(func), a, b, result);
		return result;
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
		scopes.back().deadAfter(liveIn);
		int64_t result = scopes.back().allocRegister(Register::TEMP);
		emit(code, op(func), a, 0, result);
		return result; 
	} break;
	case Symbol::E_UseMethod:
	{
		if(scopes.back().topLevel)
			throw CompileError("Attempt to use UseMethod outside of function");
		
		// This doesn't match R's behavior. R always uses the original value of the first argument, not the most recent value. Blah.
		int64_t object = (call.length == 3) ? compile(call[2], code) : compile(scopes.back().symbols[0], code); 
		
		int64_t generic = compile(call[1], code);
		
		Character p(scopes.back().parameters);
		Call gcall(p.length+1);
		for(int64_t i = 0; i < p.length; i++) gcall[i+1] = p[i];
		CompiledCall compiledCall(makeCall(gcall));

		code->constants.push_back(compiledCall);
	
		int64_t arguments = code->constants.size()-1;	
		
		emit(code, ByteCode::UseMethod, generic, arguments, object);
		scopes.back().deadAfter(object);
		return object;
	} break;
	case Symbol::E_seq_len:
	{
		int64_t len = compile(call[1], code);
		scopes.back().deadAfter(liveIn);
		int64_t result = scopes.back().allocRegister(Register::TEMP);
		emit(code, ByteCode::seq, len, 0, result);
		return result;
	} break;
	default:
	{
		return compileFunctionCall(call, code);
	}
	};
}

int64_t Compiler::compileExpression(Expression const& values, Code* code) {
	int64_t liveIn = scopes.back().live();
	int64_t result = 0;
	if(values.length == 0) 
		throw CompileError("invalid empty expression");
	for(int64_t i = 0; i < values.length; i++) {
		scopes.back().deadAfter(liveIn);
		result = compile(values[i], code);
	}
	scopes.back().deadAfter(result);
	return result;
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
	
	std::vector<Register> oldRegisters;
	oldRegisters.swap(scopes.back().registers);
	int64_t oldMaxRegister = scopes.back().maxRegister;
	
	int64_t result = compile(expr, code);

	code->registers = scopes.back().maxRegister+1;
	code->expression = expr;
	// insert return statement at end of code
	emit(code, ByteCode::ret, 0, 0, result);
	
	oldRegisters.swap(scopes.back().registers);
	scopes.back().maxRegister = oldMaxRegister;	
	loopDepth = oldLoopDepth;

	return code;	
}

