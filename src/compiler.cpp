
#include "compiler.h"
#include "internal.h"

static ByteCode::Enum op(Symbol const& s) {
	switch(s.i) {
		case String::colon: return ByteCode::colon; break;
		case String::mul: return ByteCode::mul; break;
		case String::div: return ByteCode::div; break;
		case String::idiv: return ByteCode::idiv; break;
		case String::mod: return ByteCode::mod; break;
		case String::pow: return ByteCode::pow; break;
		case String::lt: return ByteCode::lt; break;
		case String::gt: return ByteCode::gt; break;
		case String::eq: return ByteCode::eq; break;
		case String::neq: return ByteCode::neq; break;
		case String::ge: return ByteCode::ge; break;
		case String::le: return ByteCode::le; break;
		case String::lnot: return ByteCode::lnot; break;
		case String::land: return ByteCode::land; break;
		case String::lor: return ByteCode::lor; break;
		case String::sland: return ByteCode::sland; break;
		case String::slor: return ByteCode::slor; break;
		case String::abs: return ByteCode::abs; break;
		case String::sign: return ByteCode::sign; break;
		case String::sqrt: return ByteCode::sqrt; break;
		case String::floor: return ByteCode::floor; break;
		case String::ceiling: return ByteCode::ceiling; break;
		case String::trunc: return ByteCode::trunc; break;
		case String::round: return ByteCode::round; break;
		case String::signif: return ByteCode::signif; break;
		case String::exp: return ByteCode::exp; break;
		case String::log: return ByteCode::log; break;
		case String::cos: return ByteCode::cos; break;
		case String::sin: return ByteCode::sin; break;
		case String::tan: return ByteCode::tan; break;
		case String::acos: return ByteCode::acos; break;
		case String::asin: return ByteCode::asin; break;
		case String::atan: return ByteCode::atan; break;
		case String::type: return ByteCode::type; break;
		case String::Logical: return ByteCode::logical1; break;	 	
		case String::Integer: return ByteCode::integer1; break;
		case String::Double: return ByteCode::double1; break;
		case String::Complex: return ByteCode::complex1; break;
		case String::Character: return ByteCode::character1; break;
		case String::Raw: return ByteCode::raw1; break;
		default: throw RuntimeError("unexpected symbol used as an operator"); break;
	}
}

void Compiler::emit(Prototype* code, ByteCode::Enum bc, int64_t a, int64_t b, int64_t c) {
	code->bc.push_back(Instruction(bc, a, b, c));
}

static void resolveLoopReferences(Prototype* code, int64_t start, int64_t end, int64_t nextTarget, int64_t breakTarget) {
	for(int64_t i = start; i < end; i++) {
		if(code->bc[i].bc == ByteCode::jmp && code->bc[i].a == 0 && code->bc[i].b == 1) {
			code->bc[i].a = nextTarget-i;
		} else if(code->bc[i].bc == ByteCode::jmp && code->bc[i].a == 0 && code->bc[i].b == 2) {
			code->bc[i].a = breakTarget-i;
		}
	}
}

int64_t Compiler::getSlot(Symbol s) {
	// check if destination is a reserved slot.
	int64_t slot = -1;
	if(!scopes.back().topLevel) {
		for(uint64_t i = 0; i < scopes.back().symbols.size(); i++) {
			if(scopes.back().symbols[i] == s) {
				slot = i;
			}
		}
		if(slot < 0 && scopes.back().symbols.size() < 32) {
			scopes.back().symbols.push_back(s);
			slot = scopes.back().symbols.size()-1;
		}
	}
	return slot;
}

int64_t Compiler::compileConstant(Value const& expr, Prototype* code) {
	code->constants.push_back(expr);
	int64_t reg = scopes.back().allocRegister(Register::CONSTANT);
	emit(code, ByteCode::kget, code->constants.size()-1, 0, reg);
	return reg;
}

int64_t Compiler::compileSymbol(Symbol const& symbol, Prototype* code) {
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

static bool isCall(Value const& v) {
	return v.isObject() && ((Object const&)v).hasClass() && ((Object const&)v).className() == Symbols::Call;
}

static bool isExpression(Value const& v) {
	return v.isObject() && ((Object const&)v).hasClass() && ((Object const&)v).className() == Symbols::Expression;
}

CompiledCall Compiler::makeCall(List const& call, Character const& names) {
	// compute compiled call...precompiles promise code and some necessary values
	int64_t dots = call.length-1;
	List arguments(call.length-1);
	for(int64_t i = 1; i < call.length; i++) {
		if(call[i].isSymbol() && Symbol(call[i]) == String::dots) {
			arguments[i-1] = call[i];
			dots = i-1;
		} else if(isCall(call[i]) || call[i].isSymbol()) {
			// promises should have access to the slots of the enclosing scope, but have their own register assignments
			arguments[i-1] = Function(Compiler::compile(call[i]),NULL).AsPromise();
		} else {
			arguments[i-1] = call[i];
		}
	}
	if(names.length > 0) {
		Character c(Subset(names, 1, call.length-1));
		return CompiledCall(call, arguments, c, dots);
		// reserve room for cached name matching...
	} else {
		return CompiledCall(call, arguments, names, dots);
	}
}

// a standard call, not an op
int64_t Compiler::compileFunctionCall(List const& call, Character const& names, Prototype* code) {
	int64_t liveIn = scopes.back().live();
	int64_t function = compile(call[0], code);
	code->calls.push_back(makeCall(call, names));
	scopes.back().deadAfter(liveIn);
	int64_t result = scopes.back().allocRegister(Register::VARIABLE);
	emit(code, ByteCode::call, function, code->calls.size()-1, result);
	return result;
}

int64_t Compiler::compileCall(List const& call, Character const& names, Prototype* code) {
	int64_t length = call.length;
	if(length == 0) {
		throw CompileError("invalid empty call");
	}

	int64_t liveIn = scopes.back().live();

	Symbol func = Symbols::NA;
	if(call[0].isSymbol())
		func = Symbol(call[0]);
	else if(call[0].isCharacter() && call[0].length > 0)
		func = Character(call[0])[0];
	
	switch(func.i) {

	case String::internal: 
	{
		// The riposte way... .Internal is a function on symbols, returning the internal function
		if(call[1].isSymbol()) {
			int64_t reg = scopes.back().allocRegister(Register::VARIABLE);
			emit(code, ByteCode::iget, Symbol(call[1]).i, 0, reg);
			return reg;
		} else if(call[1].isCharacter() && call[1].length > 0) {
			int64_t reg = scopes.back().allocRegister(Register::VARIABLE);
			emit(code, ByteCode::iget, Character(call[1])[0].i, 0, reg);
			return reg;
		} 
		// The R way... .Internal is a function on calls
		else if(isCall(call[1])) {
			List c = List(((Object const&)call[1]).base());
			Value names = ((Object const&)call[1]).getNames(); 
			List ic(2);
			ic[0] = Symbol(String::internal);
			ic[1] = c[0];
			c[0] = ic;
			return compile(CreateCall(c, names), code);
		} else {
			throw CompileError(std::string(".Internal has invalid arguments (") + Type::toString(call[1].type) + ")");
		}
	} break;
	case String::assign:
	case String::eqassign: 
	{
		Value dest = call[1];
		
		// recursively handle emitting assignment instructions...
		Value value = call[2];
		while(isCall(dest)) {
			List c = List(((Object const&)dest).base());
			Value names = ((Object const&)dest).getNames();
			List n(c.length+1);
			
			for(int64_t i = 0; i < c.length; i++) { n[i] = c[i]; }
			Character nnames(c.length+1);
			for(int64_t i = 0; i < c.length; i++) { nnames[i] = Symbols::empty; }
			if(names.isCharacter()) {
				for(int i = 0; i < c.length; i++) { nnames[i] = Character(names)[i]; }
			}

			n[0] = state.StrToSym(state.SymToStr(Symbol(c[0])) + "<-");
			n[c.length] = value;
			nnames[c.length] = Symbols::value;
			value = CreateCall(n, nnames);
			dest = c[1];
		}
		
		// the source for the assignment
		int64_t source = compile(value, code);

		assert(dest.isSymbol());
		int64_t slot = getSlot(Symbol(dest));
	
		if(slot >= 0)
			emit(code, ByteCode::sassign, slot, 0, source);
		else
			emit(code, ByteCode::assign, Symbol(dest).i, 0, source);
	
		scopes.back().deadAfter(source);	
		return source;
	} break;
	case String::bracket: {
		if(call.length != 3) return compileFunctionCall(call, names, code);
		int64_t value = compile(call[1], code);
		int64_t index = compile(call[2], code);
		scopes.back().deadAfter(liveIn);	
		int64_t reg = scopes.back().allocRegister(Register::CONSTANT);	
		emit(code, ByteCode::subset, value, index, reg);
		scopes.back().deadAfter(reg);	
		return reg;
	} break;
	case String::bb: 
	case String::dollar: {
		if(call.length != 3) return compileFunctionCall(call, names, code);
		int64_t value = compile(call[1], code);
		int64_t index = compile(call[2], code);
		scopes.back().deadAfter(liveIn);	
		int64_t reg = scopes.back().allocRegister(Register::CONSTANT);	
		emit(code, ByteCode::subset2, value, index, reg);
		scopes.back().deadAfter(reg);	
		return reg;
	} break;
	case String::bracketAssign: { 
		if(call.length != 4) return compileFunctionCall(call, names, code);
		int64_t dest = compile(call[1], code);
		int64_t index = compile(call[2], code);
		int64_t value = compile(call[3], code);
		emit(code, ByteCode::iassign, value, index, dest);
		scopes.back().deadAfter(dest);	
		return dest;
	} break;
	case String::bbAssign: {
		if(call.length != 4) return compileFunctionCall(call, names, code);
		int64_t dest = compile(call[1], code);
		int64_t index = compile(call[2], code);
		int64_t value = compile(call[3], code);
		emit(code, ByteCode::eassign, value, index, dest);
		scopes.back().deadAfter(dest);	
		return dest;
	} break;
	case String::function: 
	{
		Scope scope;
		scope.topLevel = false;
		//compile the default parameters
		assert(call[1].isObject());
		List c = List(((Object const&)call[1]).base());
		Character names = ((Object const&)call[1]).hasNames() ? 
			Character(((Object const&)call[1]).getNames()) :
			Character(0);
		
		List parameters(c.length);
		scope.parameters = names;
		for(int64_t i = 0; i < parameters.length; i++) {
			if(!c[i].isNil()) {
				parameters[i] = Function(compile(c[i]),NULL).AsPromise();
			}
			else {
				parameters[i] = c[i];
			}
			scope.symbols.push_back(names[i]);
		}

		//compile the source for the body
		scopes.push_back(scope);
		Prototype* functionCode = compile(call[2]);
		functionCode->slotSymbols.swap(scopes.back().symbols);
		scopes.pop_back();

		// Populate function info
		functionCode->parameters = parameters;
		functionCode->names = names;
		functionCode->string = Symbol(call[3]);

		functionCode->dots = parameters.length;
		if(parameters.length > 0) {
			for(int64_t i = 0;i < names.length; i++) 
				if(names[i] == Symbols::dots) functionCode->dots = i;
		}

		code->prototypes.push_back(functionCode);
		
		scopes.back().deadAfter(liveIn);	
		int64_t reg = scopes.back().allocRegister(Register::CONSTANT);	
		emit(code, ByteCode::function, code->prototypes.size()-1, 0, reg);
		return reg;
	} break;
	case String::returnSym: 
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
	case String::forSym: 
	{
		int64_t loop_vector = compile(call[2], code);
		int64_t loop_counter = scopes.back().allocRegister(Register::VARIABLE);	// save space for loop counter
		int64_t loop_variable = scopes.back().allocRegister(Register::VARIABLE);
		int64_t slot = getSlot(Symbol(call[1]));

		if(loop_counter != loop_vector+1) throw CompileError("limits aren't in adjacent registers");
		emit(code, ByteCode::forbegin, 0, loop_counter, loop_variable);
		loopDepth++;
		int64_t beginbody = code->bc.size();

		if(slot >= 0) emit(code, ByteCode::sassign, slot, 0, loop_variable);
		else emit(code, ByteCode::assign, Symbol(call[1]).i, 0, loop_variable); 	
		compile(call[3], code);

		int64_t endbody = code->bc.size();
		resolveLoopReferences(code, beginbody, endbody, endbody, endbody+1);
		loopDepth--;
		emit(code, ByteCode::forend, beginbody-endbody, loop_counter , loop_variable);
		code->bc[beginbody-1].a = endbody-beginbody+1;
		scopes.back().deadAfter(liveIn);
		int64_t result = compile(Null::Singleton(), code);
		return result;
	} break;
	case String::whileSym: 
	{
		int64_t head_condition = compile(call[1], code);
		emit(code, ByteCode::jf, 0, head_condition, liveIn);
		loopDepth++;
		
		int64_t beginbody = code->bc.size();
		compile(call[2], code);
		int64_t tail = code->bc.size();
		int64_t tail_condition = compile(call[1], code);
		int64_t endbody = code->bc.size();
		
		emit(code, ByteCode::jt, beginbody-endbody, tail_condition, liveIn);
		resolveLoopReferences(code, beginbody, endbody, tail, endbody+1);
		code->bc[beginbody-1].a = endbody-beginbody+2;
		
		loopDepth--;
		scopes.back().deadAfter(liveIn);
		int64_t result = compile(Null::Singleton(), code);
		return result;
	} break;
	case String::repeatSym: 
	{
		loopDepth++;

		int64_t beginbody = code->bc.size();
		compile(call[1], code);
		int64_t endbody = code->bc.size();
		resolveLoopReferences(code, beginbody, endbody, endbody, endbody+1);
		
		loopDepth--;
		emit(code, ByteCode::jmp, beginbody-endbody, 0, liveIn);
		scopes.back().deadAfter(liveIn);
		int64_t result = compile(Null::Singleton(), code);
		return result;
	} break;
	case String::nextSym:
	{
		if(loopDepth == 0) throw CompileError("next used outside of loop");
		int64_t result = scopes.back().allocRegister(Register::TEMP);	
		emit(code, ByteCode::jmp, 0, 1, result);
		return result;
	} break;
	case String::breakSym:
	{
		if(loopDepth == 0) throw CompileError("break used outside of loop");
		int64_t result = scopes.back().allocRegister(Register::TEMP);	
		emit(code, ByteCode::jmp, 0, 2, result);
		return result;
	} break;
	case String::ifSym: 
	{
		int64_t resultT=0, resultF=0;
		if(call.length != 3 && call.length != 4)	
			throw CompileError("invalid if statement");
		if(call.length == 3)
			resultF = compile(Null::Singleton(), code);
		int64_t cond = compile(call[1], code);
		emit(code, ByteCode::jf, 0, cond, liveIn);
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
	case String::brace: 
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
	case String::paren: 
	{
		return compile(call[1], code);
	} break;
	case String::add: 
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
	case String::sub: 
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
	case String::colon:
	case String::mul: 
	case String::div: 
	case String::idiv: 
	case String::pow: 
	case String::mod:
	case String::land:
	case String::lor:
	case String::slor:
	case String::sland:
	case String::eq:
	case String::neq:
	case String::lt:
	case String::gt:
	case String::ge:
	case String::le:
	{
		// if there aren't exactly two parameters, we should call the library version...
		if(call.length != 3) return compileFunctionCall(call, names, code);
		int64_t a = compile(call[1], code);
		int64_t b = compile(call[2], code);
		scopes.back().deadAfter(liveIn);
		int64_t result = scopes.back().allocRegister(Register::TEMP);
		emit(code, op(func), a, b, result);
		return result;
	} break;
	// Unary operators
	case String::lnot: 
	case String::abs: 
	case String::sign: 
	case String::sqrt: 
	case String::floor: 
	case String::ceiling: 
	case String::trunc: 
	case String::round: 
	case String::signif: 
	case String::exp: 
	case String::log: 
	case String::cos: 
	case String::sin: 
	case String::tan: 
	case String::acos: 
	case String::asin: 
	case String::atan:
	case String::type:
	case String::Logical:
	case String::Integer:
	case String::Double:
	case String::Complex:
	case String::Character:
	case String::Raw:
	{
		// if there isn't exactly one parameters, we should call the library version...
		if(call.length != 2) return compileFunctionCall(call, names, code);
		int64_t a = compile(call[1], code);
		scopes.back().deadAfter(liveIn);
		int64_t result = scopes.back().allocRegister(Register::TEMP);
		emit(code, op(func), a, 0, result);
		return result; 
	} break;
	case String::UseMethod:
	{
		if(scopes.back().topLevel)
			throw CompileError("Attempt to use UseMethod outside of function");
		
		// This doesn't match R's behavior. R always uses the original value of the first argument, not the most recent value. Blah.
		int64_t object = (call.length == 3) ? compile(call[2], code) : compile(scopes.back().symbols[0], code); 
		
		int64_t generic = compile(call[1], code);
		
		Character p(scopes.back().parameters);
		List gcall(p.length+1);
		for(int64_t i = 0; i < p.length; i++) gcall[i+1] = p[i];
		code->calls.push_back(makeCall(gcall, Character(0)));
	
		emit(code, ByteCode::UseMethod, generic, code->calls.size()-1, object);
		scopes.back().deadAfter(object);
		return object;
	} break;
	case String::seq_len:
	{
		int64_t len = compile(call[1], code);
		scopes.back().deadAfter(liveIn);
		int64_t result = scopes.back().allocRegister(Register::TEMP);
		emit(code, ByteCode::seq, len, 0, result);
		return result;
	} break;
	default:
	{
		return compileFunctionCall(call, names, code);
	}
	};
}

int64_t Compiler::compileExpression(List const& values, Prototype* code) {
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

int64_t Compiler::compile(Value const& expr, Prototype* code) {
	switch(expr.type)
	{
		case Type::Symbol:
			return compileSymbol(Symbol(expr), code);
			break;
		case Type::Object:
			{
				Object const& o = (Object const&) expr;
				if(o.className() == Symbols::Expression) {
					assert(o.base().isList());
					return compileExpression((List const&)o.base(), code);
				}
				else if(o.className() == Symbols::Call) {
					assert(o.base().isList());
					return compileCall((List const&)o.base(), 
						o.hasNames() ? Character(o.getNames()) : Character(0), code);
				}
				else {
					return compileConstant(expr, code);
				}
			}
			break;
		default:
			return compileConstant(expr, code);
			break;
	};
}

Prototype* Compiler::compile(Value const& expr) {
	Prototype* code = new Prototype();
	assert(((int64_t)code) % 16 == 0); // our type packing assumes that this is true

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

