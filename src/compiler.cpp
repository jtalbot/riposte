
#include "compiler.h"
#include "internal.h"

static ByteCode::Enum op(Symbol const& s) {
	switch(s.i) {
		case EStrings::colon: return ByteCode::colon; break;
		case EStrings::mul: return ByteCode::mul; break;
		case EStrings::div: return ByteCode::div; break;
		case EStrings::idiv: return ByteCode::idiv; break;
		case EStrings::mod: return ByteCode::mod; break;
		case EStrings::pow: return ByteCode::pow; break;
		case EStrings::atan2: return ByteCode::atan2; break;
		case EStrings::hypot: return ByteCode::hypot; break;
		case EStrings::lt: return ByteCode::lt; break;
		case EStrings::gt: return ByteCode::gt; break;
		case EStrings::eq: return ByteCode::eq; break;
		case EStrings::neq: return ByteCode::neq; break;
		case EStrings::ge: return ByteCode::ge; break;
		case EStrings::le: return ByteCode::le; break;
		case EStrings::lnot: return ByteCode::lnot; break;
		case EStrings::land: return ByteCode::land; break;
		case EStrings::lor: return ByteCode::lor; break;
		case EStrings::sland: return ByteCode::sland; break;
		case EStrings::slor: return ByteCode::slor; break;
		case EStrings::abs: return ByteCode::abs; break;
		case EStrings::sign: return ByteCode::sign; break;
		case EStrings::sqrt: return ByteCode::sqrt; break;
		case EStrings::floor: return ByteCode::floor; break;
		case EStrings::ceiling: return ByteCode::ceiling; break;
		case EStrings::trunc: return ByteCode::trunc; break;
		case EStrings::round: return ByteCode::round; break;
		case EStrings::signif: return ByteCode::signif; break;
		case EStrings::exp: return ByteCode::exp; break;
		case EStrings::log: return ByteCode::log; break;
		case EStrings::cos: return ByteCode::cos; break;
		case EStrings::sin: return ByteCode::sin; break;
		case EStrings::tan: return ByteCode::tan; break;
		case EStrings::acos: return ByteCode::acos; break;
		case EStrings::asin: return ByteCode::asin; break;
		case EStrings::atan: return ByteCode::atan; break;
		case EStrings::sum: return ByteCode::sum; break;
		case EStrings::prod: return ByteCode::prod; break;
		case EStrings::min: return ByteCode::min; break;
		case EStrings::max: return ByteCode::max; break;
		case EStrings::any: return ByteCode::any; break;
		case EStrings::all: return ByteCode::all; break;
		case EStrings::cumsum: return ByteCode::cumsum; break;
		case EStrings::cumprod: return ByteCode::cumprod; break;
		case EStrings::cummin: return ByteCode::cummin; break;
		case EStrings::cummax: return ByteCode::cummax; break;
		case EStrings::cumany: return ByteCode::cumany; break;
		case EStrings::cumall: return ByteCode::cumall; break;
		case EStrings::type: return ByteCode::type; break;
		case EStrings::Logical: return ByteCode::logical1; break;	 	
		case EStrings::Integer: return ByteCode::integer1; break;
		case EStrings::Double: return ByteCode::double1; break;
		case EStrings::Character: return ByteCode::character1; break;
		case EStrings::Raw: return ByteCode::raw1; break;
		case EStrings::length: return ByteCode::length; break;
		default: throw RuntimeError("unexpected symbol used as an operator"); break;
	}
}

int64_t Compiler::emit(Prototype* code, ByteCode::Enum bc, int64_t a, int64_t b, int64_t c) {
	code->bc.push_back(Instruction(bc, a, b, c));
	return code->bc.size()-1;
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

int64_t Compiler::compileConstant(Value const& expr, Prototype* code) {
	code->constants.push_back(expr);
	int64_t reg = scopes.back().allocRegister(Register::CONSTANT);
	emit(code, ByteCode::kget, code->constants.size()-1, 0, reg);
	return reg;
}

int64_t Compiler::compileSymbol(Symbol const& symbol, Prototype* code) {
	int64_t reg = scopes.back().allocRegister(Register::VARIABLE);
	emit(code, ByteCode::get, symbol.i, 0, reg);
	emit(code, ByteCode::assign, symbol.i, 0, reg);
	emit(code, ByteCode::get, 0, 0, 0);
	return reg;
}

static bool isCall(Value const& v) {
	return v.isObject() && ((Object const&)v).hasClass() && ((Object const&)v).className() == Strings::Call;
}

static bool isExpression(Value const& v) {
	return v.isObject() && ((Object const&)v).hasClass() && ((Object const&)v).className() == Strings::Expression;
}

CompiledCall Compiler::makeCall(List const& call, Character const& names) {
	// compute compiled call...precompiles promise code and some necessary values
	int64_t dots = call.length-1;
	List arguments(call.length-1);
	for(int64_t i = 1; i < call.length; i++) {
		if(call[i].isSymbol() && Symbol(call[i]) == Strings::dots) {
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
	code->calls.push_back(makeCall(call, names));
	int64_t function = compile(call[0], code);
	scopes.back().deadAfter(liveIn);
	int64_t result = scopes.back().allocRegister(Register::TEMP);
	emit(code, ByteCode::call, function, -code->calls.size(), result);
	return result;
}

int64_t Compiler::compileInternalFunctionCall(Object const& o, Prototype* code) {
	List const& call = (List const&)(o.base());
	int64_t liveIn = scopes.back().live();
	Symbol func(call[0]);
	std::map<String, int64_t>::const_iterator itr = state.internalFunctionIndex.find(func);
	if(itr == state.internalFunctionIndex.end()) {
		_error(std::string("Unimplemented internal function ") + state.externStr(func));
		//return compile(o, code);
	}
	int64_t function = itr->second;
	// check parameter count
	if(state.internalFunctions[function].params != call.length-1)
		_error(std::string("Incorrect number of arguments to internal function ") + state.externStr(func));
	// compile parameters directly...reserve registers for them.
	int64_t reg = liveIn;
	for(int64_t i = 1; i < call.length; i++) {
		int64_t r = compile(call[i], code);
		assert(r == reg+1);
		reg = r; 
	}
	scopes.back().deadAfter(liveIn);
	int64_t result = scopes.back().allocRegister(Register::TEMP);
	emit(code, ByteCode::icall, function, reg-(call.length-2), result);
	return result;
}

int64_t Compiler::compileCall(List const& call, Character const& names, Prototype* code) {
	int64_t length = call.length;
	if(length == 0) {
		throw CompileError("invalid empty call");
	}

	int64_t liveIn = scopes.back().live();

	if(!call[0].isSymbol() && !call[0].isCharacter1())
		return compileFunctionCall(call, names, code);

	Symbol func(call[0]);
	
	switch(func.i) {

	case EStrings::internal: 
	{
		if(!call[1].isObject())
			throw CompileError(std::string(".Internal has invalid arguments (") + Type::toString(call[1].type) + ")");
		Object const& o = (Object const&)call[1];
		assert(o.className() == Strings::Call && o.base().isList());
		return compileInternalFunctionCall(o, code);
	} break;
	case EStrings::assign:
	case EStrings::eqassign: 
	case EStrings::assign2: 
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
			for(int64_t i = 0; i < c.length; i++) { nnames[i] = Strings::empty; }
			if(names.isCharacter()) {
				for(int i = 0; i < c.length; i++) { nnames[i] = Character(names)[i]; }
			}

			n[0] = Symbol(state.internStr(state.externStr(Symbol(c[0])) + "<-"));
			n[c.length] = value;
			nnames[c.length] = Strings::value;
			value = CreateCall(n, nnames);
			dest = c[1];
		}
		
		// the source for the assignment
		int64_t source = compile(value, code);

		emit(code, func.i == EStrings::assign2 ? ByteCode::assign2 : ByteCode::assign, Symbol(dest).i, 0, source);
		emit(code, ByteCode::assign, 0, 0, 0);
	
		scopes.back().deadAfter(source);
		return source;
	} break;
	case EStrings::bracket: {
		if(call.length != 3) return compileFunctionCall(call, names, code);
		int64_t value = compile(call[1], code);
		int64_t index = compile(call[2], code);
		scopes.back().deadAfter(liveIn);	
		int64_t reg = scopes.back().allocRegister(Register::CONSTANT);	
		emit(code, ByteCode::subset, value, index, reg);
		scopes.back().deadAfter(reg);	
		return reg;
	} break;
	case EStrings::bb: 
	case EStrings::dollar: {
		if(call.length != 3) return compileFunctionCall(call, names, code);
		int64_t value = compile(call[1], code);
		int64_t index = compile(call[2], code);
		scopes.back().deadAfter(liveIn);	
		int64_t reg = scopes.back().allocRegister(Register::CONSTANT);	
		emit(code, ByteCode::subset2, value, index, reg);
		scopes.back().deadAfter(reg);	
		return reg;
	} break;
	case EStrings::bracketAssign: { 
		if(call.length != 4) return compileFunctionCall(call, names, code);
		int64_t dest = compile(call[1], code);
		int64_t index = compile(call[2], code);
		int64_t value = compile(call[3], code);
		emit(code, ByteCode::iassign, value, index, dest);
		scopes.back().deadAfter(dest);	
		return dest;
	} break;
	case EStrings::bbAssign: {
		if(call.length != 4) return compileFunctionCall(call, names, code);
		int64_t dest = compile(call[1], code);
		int64_t index = compile(call[2], code);
		int64_t value = compile(call[3], code);
		emit(code, ByteCode::eassign, value, index, dest);
		scopes.back().deadAfter(dest);	
		return dest;
	} break;
	case EStrings::function: 
	{
		Scope scope;
		scope.topLevel = false;
		//compile the default parameters
		assert(call[1].isObject());
		List c = List(((Object const&)call[1]).base());
		Character parameters = ((Object const&)call[1]).hasNames() ? 
			Character(((Object const&)call[1]).getNames()) :
			Character(0);
		
		List defaults(c.length);
		scope.parameters = parameters;
		for(int64_t i = 0; i < defaults.length; i++) {
			if(!c[i].isNil()) {
				defaults[i] = Function(compile(c[i]),NULL).AsPromise();
			}
			else {
				defaults[i] = c[i];
			}
		}

		//compile the source for the body
		scopes.push_back(scope);
		Prototype* functionCode = compile(call[2]);
		scopes.pop_back();

		// Populate function info
		functionCode->parameters = parameters;
		functionCode->defaults = defaults;
		functionCode->string = Symbol(call[3]);
		functionCode->dots = parameters.length;
		for(int64_t i = 0; i < parameters.length; i++) 
			if(parameters[i] == Strings::dots) functionCode->dots = i;

		code->prototypes.push_back(functionCode);
		
		scopes.back().deadAfter(liveIn);	
		int64_t reg = scopes.back().allocRegister(Register::CONSTANT);	
		emit(code, ByteCode::function, code->prototypes.size()-1, 0, reg);
		return reg;
	} break;
	case EStrings::returnSym: 
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
	case EStrings::forSym: 
	{
		// special case common i in m:n case
		/*if(call[2].isCall() && Symbol(Call(call[2])[0]) == EStrings::colon) {
			int64_t lim1 = compile(Call(call[2])[1], code);
			int64_t lim2 = compile(Call(call[2])[2], code);
			if(lim1+1 != lim2) throw CompileError("limits aren't in adjacent registers");
			//int64_t slot = getSlot(Symbol(call[1]));
			//if(slot < 0) throw CompileError("for loop variable not allocated to slot");
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
		else {*/
			int64_t loop_vector = compile(call[2], code);
			int64_t loop_counter = scopes.back().allocRegister(Register::VARIABLE);	// save space for loop counter
			int64_t loop_variable = scopes.back().allocRegister(Register::VARIABLE);
			
			if(loop_counter != loop_vector+1) throw CompileError("limits aren't in adjacent registers");
			emit(code, ByteCode::forbegin, 0, loop_counter, loop_variable);
			loopDepth++;
			int64_t beginbody = code->bc.size();

			emit(code, ByteCode::assign, Symbol(call[1]).i, 0, loop_variable);
			emit(code, ByteCode::assign, 0, 0, loop_variable);
			compile(call[3], code);

			int64_t endbody = code->bc.size();
			resolveLoopReferences(code, beginbody, endbody, endbody, endbody+1);
			loopDepth--;
			emit(code, ByteCode::forend, beginbody-endbody, loop_counter , loop_variable);
			code->bc[beginbody-1].a = endbody-beginbody+1;
		//}
		scopes.back().deadAfter(liveIn);
		int64_t result = compile(Null::Singleton(), code);
		return result;
	} break;
	case EStrings::whileSym: 
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
	case EStrings::repeatSym: 
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
	case EStrings::nextSym:
	{
		if(loopDepth == 0) throw CompileError("next used outside of loop");
		int64_t result = scopes.back().allocRegister(Register::TEMP);	
		emit(code, ByteCode::jmp, 0, 1, result);
		return result;
	} break;
	case EStrings::breakSym:
	{
		if(loopDepth == 0) throw CompileError("break used outside of loop");
		int64_t result = scopes.back().allocRegister(Register::TEMP);	
		emit(code, ByteCode::jmp, 0, 2, result);
		return result;
	} break;
	case EStrings::ifSym: 
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
	case EStrings::switchSym:
	{
		if(call.length == 0) _error("'EXPR' is missing");
		int64_t c = compile(call[1], code);
		int64_t n = call.length-2;
		int64_t branch = emit(code, ByteCode::branch, c, n, 0);
		for(int64_t i = 2; i < call.length; i++) {
			emit(code, ByteCode::branch, names.length > i ? names[i].i : Strings::empty.i, 0, 0);
		}
		scopes.back().deadAfter(liveIn);
		
		std::vector<int64_t> jmps;
		int64_t result = compile(Null::Singleton(), code);
		jmps.push_back(emit(code, ByteCode::jmp, 0, 0, 0));	
		
		for(int64_t i = 1; i <= n; i++) {
			code->bc[branch+i].c = code->bc.size()-branch;
			scopes.back().deadAfter(liveIn);
			if(!call[i+1].isNil()) {
				int64_t r = compile(call[i+1], code);
				if(r != result) throw CompileError(std::string("switch statement doesn't put all its results in the same register"));
				if(i < n)
					jmps.push_back(emit(code, ByteCode::jmp, 0, 0, 0));
			} else if(i == n) {
				compile(Null::Singleton(), code);
			}
		}
		for(int64_t i = 0; i < (int64_t)jmps.size(); i++) {
			code->bc[jmps[i]].a = code->bc.size()-jmps[i];
		}
		scopes.back().deadAfter(result);
		return result;
	} break;
	case EStrings::brace: 
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
	case EStrings::paren: 
	{
		return compile(call[1], code);
	} break;
	case EStrings::add: 
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
	case EStrings::sub: 
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
	case EStrings::colon:
	case EStrings::mul: 
	case EStrings::div: 
	case EStrings::idiv: 
	case EStrings::pow: 
	case EStrings::mod:
	case EStrings::atan2:
	case EStrings::hypot:
	case EStrings::land:
	case EStrings::lor:
	case EStrings::slor:
	case EStrings::sland:
	case EStrings::eq:
	case EStrings::neq:
	case EStrings::lt:
	case EStrings::gt:
	case EStrings::ge:
	case EStrings::le:
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
	case EStrings::lnot: 
	case EStrings::abs: 
	case EStrings::sign: 
	case EStrings::sqrt: 
	case EStrings::floor: 
	case EStrings::ceiling: 
	case EStrings::trunc: 
	case EStrings::round: 
	case EStrings::signif: 
	case EStrings::exp: 
	case EStrings::log: 
	case EStrings::cos: 
	case EStrings::sin: 
	case EStrings::tan: 
	case EStrings::acos: 
	case EStrings::asin: 
	case EStrings::atan:
	case EStrings::sum:
	case EStrings::prod:
	case EStrings::min:
	case EStrings::max:
	case EStrings::any:
	case EStrings::all:
	case EStrings::cumsum:
	case EStrings::cumprod:
	case EStrings::cummin:
	case EStrings::cummax:
	case EStrings::cumany:
	case EStrings::cumall:
	case EStrings::type:
	case EStrings::length:
	case EStrings::Logical:
	case EStrings::Integer:
	case EStrings::Double:
	case EStrings::Character:
	case EStrings::Raw:
	{
		// if there isn't exactly one parameter, we should call the library version...
		if(call.length != 2) return compileFunctionCall(call, names, code);
		int64_t a = compile(call[1], code);
		scopes.back().deadAfter(liveIn);
		int64_t result = scopes.back().allocRegister(Register::TEMP);
		emit(code, op(func), a, 0, result);
		return result; 
	} break;
	case EStrings::UseMethod:
	{
		if(scopes.back().topLevel)
			throw CompileError("Attempt to use UseMethod outside of function");
		
		// This doesn't match R's behavior. R always uses the original value of the first argument, not the most recent value. Blah.
		int64_t object = (call.length == 3) 
			? compile(call[2], code) : compile(Symbol(scopes.back().parameters[0]), code); 
		if(!call[1].isCharacter1())
			throw CompileError("First parameter to UseMethod must be a string");
		Symbol generic(call[1]);
		
		Character p(scopes.back().parameters);
		List gcall(p.length+1);
		for(int64_t i = 0; i < p.length; i++) gcall[i+1] = Symbol(p[i]);
		code->calls.push_back(makeCall(gcall, Character(0)));
	
		emit(code, ByteCode::UseMethod, generic.i, code->calls.size()-1, object);
		scopes.back().deadAfter(object);
		return object;
	} break;
	case EStrings::seq_len:
	{
		int64_t len = compile(call[1], code);
		int64_t step = compile(Integer::c(1), code);
		scopes.back().deadAfter(liveIn);
		int64_t result = scopes.back().allocRegister(Register::TEMP);
		emit(code, ByteCode::seq, len, step, result);
		return result;
	} break;
	case EStrings::docall:
	{
		int64_t what = compile(call[1], code);
		int64_t args = compile(call[2], code);
		scopes.back().deadAfter(liveIn);
		int64_t result = scopes.back().allocRegister(Register::TEMP);
		emit(code, ByteCode::call, what, args, result);
		return result;
	} break;
	case EStrings::list:
	{
		// we only handle the list(...) case through an op for now
		if(call.length != 2 || !call[1].isSymbol() || Symbol(call[1]) != Strings::dots)
			return compileFunctionCall(call, names, code);
		scopes.back().deadAfter(liveIn);
		int64_t result = scopes.back().allocRegister(Register::TEMP);
		int64_t counter = compileConstant(Integer::c(0), code);
		int64_t storage = scopes.back().allocRegister(Register::TEMP);
		emit(code, ByteCode::list, counter, storage, result); 
		scopes.back().deadAfter(result);
		return result;
	} break;
	case EStrings::missing:
	{
		if(call.length != 2) _error("missing requires one argument");
		if(!call[1].isSymbol() && !call[1].isCharacter1()) _error("wrong parameter to missing");
		Symbol s(call[1]);
		int64_t result = scopes.back().allocRegister(Register::TEMP);
		emit(code, ByteCode::missing, s.i, 0, result); 
		scopes.back().deadAfter(result);
		return result;
	} break;
	case EStrings::quote:
	{
		if(call.length != 2) _error("quote requires one argument");
		return compileConstant(call[1], code);
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
				if(o.className() == Strings::Expression) {
					assert(o.base().isList());
					return compileExpression((List const&)o.base(), code);
				}
				else if(o.className() == Strings::Call) {
					assert(o.base().isList());
					return compileCall((List const&)o.base(), 
						o.hasNames() ? Character(o.getNames()) : Character(0), code);
				}
				else {
					return compileConstant(expr, code);
				}
			}
			break;
		default: {
			int64_t i = compileConstant(expr, code);
			return i;
		}	break;
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

