
#include "compiler.h"
#include "internal.h"

static ByteCode::Enum op(String const& func) {
	if(func == Strings::colon) return ByteCode::colon;
	if(func == Strings::mul) return ByteCode::mul;
	if(func == Strings::div) return ByteCode::div; 
	if(func == Strings::idiv) return ByteCode::idiv; 
	if(func == Strings::mod) return ByteCode::mod; 
	if(func == Strings::pow) return ByteCode::pow; 
	if(func == Strings::atan2) return ByteCode::atan2; 
	if(func == Strings::hypot) return ByteCode::hypot; 
	if(func == Strings::lt) return ByteCode::lt; 
	if(func == Strings::gt) return ByteCode::gt; 
	if(func == Strings::eq) return ByteCode::eq; 
	if(func == Strings::neq) return ByteCode::neq; 
	if(func == Strings::ge) return ByteCode::ge; 
	if(func == Strings::le) return ByteCode::le; 
	if(func == Strings::lnot) return ByteCode::lnot; 
	if(func == Strings::land) return ByteCode::land; 
	if(func == Strings::lor) return ByteCode::lor; 
	if(func == Strings::sland) return ByteCode::sland; 
	if(func == Strings::slor) return ByteCode::slor; 
	if(func == Strings::abs) return ByteCode::abs; 
	if(func == Strings::sign) return ByteCode::sign; 
	if(func == Strings::sqrt) return ByteCode::sqrt; 
	if(func == Strings::floor) return ByteCode::floor; 
	if(func == Strings::ceiling) return ByteCode::ceiling; 
	if(func == Strings::trunc) return ByteCode::trunc; 
	if(func == Strings::round) return ByteCode::round; 
	if(func == Strings::signif) return ByteCode::signif; 
	if(func == Strings::exp) return ByteCode::exp; 
	if(func == Strings::log) return ByteCode::log; 
	if(func == Strings::cos) return ByteCode::cos; 
	if(func == Strings::sin) return ByteCode::sin; 
	if(func == Strings::tan) return ByteCode::tan; 
	if(func == Strings::acos) return ByteCode::acos; 
	if(func == Strings::asin) return ByteCode::asin; 
	if(func == Strings::atan) return ByteCode::atan; 
	if(func == Strings::sum) return ByteCode::sum; 
	if(func == Strings::prod) return ByteCode::prod; 
	if(func == Strings::min) return ByteCode::min; 
	if(func == Strings::max) return ByteCode::max; 
	if(func == Strings::any) return ByteCode::any; 
	if(func == Strings::all) return ByteCode::all; 
	if(func == Strings::cumsum) return ByteCode::cumsum; 
	if(func == Strings::cumprod) return ByteCode::cumprod; 
	if(func == Strings::cummin) return ByteCode::cummin; 
	if(func == Strings::cummax) return ByteCode::cummax; 
	if(func == Strings::cumany) return ByteCode::cumany; 
	if(func == Strings::cumall) return ByteCode::cumall; 
	if(func == Strings::type) return ByteCode::type; 
	if(func == Strings::Logical) return ByteCode::logical1; 	 	
	if(func == Strings::Integer) return ByteCode::integer1; 
	if(func == Strings::Double) return ByteCode::double1; 
	if(func == Strings::Character) return ByteCode::character1; 
	if(func == Strings::Raw) return ByteCode::raw1; 
	if(func == Strings::length) return ByteCode::length; 
	if(func == Strings::mmul) return ByteCode::mmul; 
	if(func == Strings::split) return ByteCode::split; 
	if(func == Strings::strip) return ByteCode::strip; 
	else throw RuntimeError("unexpected symbol used as an operator"); 
}

int64_t Compiler::emit(Prototype* code, ByteCode::Enum bc, int64_t a, int64_t b, int64_t c) {
	code->bc.push_back(Instruction(bc, a, b, c));
	return code->bc.size()-1;
}

int64_t Compiler::emit(Prototype* code, ByteCode::Enum bc, String s, int64_t b, int64_t c) {
	code->bc.push_back(Instruction(bc, s, b, c));
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

int64_t Compiler::compileSymbol(Value const& symbol, Prototype* code) {
	int64_t reg = scopes.back().allocRegister(Register::VARIABLE);
	String s = SymbolStr(symbol);
	emit(code, ByteCode::get, s, 0, reg);
	emit(code, ByteCode::assign, s, 0, reg);
	return reg;
}

CompiledCall Compiler::makeCall(List const& call, Character const& names) {
	// compute compiled call...precompiles promise code and some necessary values
	int64_t dots = call.length-1;
	List arguments(call.length-1);
	for(int64_t i = 1; i < call.length; i++) {
		if(isSymbol(call[i]) && SymbolStr(call[i]) == Strings::dots) {
			arguments[i-1] = call[i];
			dots = i-1;
		} else if(isCall(call[i]) || isSymbol(call[i])) {
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
	String func = SymbolStr(call[0]);
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
	emit(code, ByteCode::internal, function, reg-(call.length-2), result);
	return result;
}

int64_t Compiler::compileCall(List const& call, Character const& names, Prototype* code) {
	int64_t length = call.length;
	if(length == 0) {
		throw CompileError("invalid empty call");
	}

	int64_t liveIn = scopes.back().live();

	if(!isSymbol(call[0]) && !call[0].isCharacter1())
		return compileFunctionCall(call, names, code);

	String func = SymbolStr(call[0]);
	
	if(func == Strings::internal) 
	{
		if(!call[1].isObject())
			throw CompileError(std::string(".Internal has invalid arguments (") + Type::toString(call[1].type) + ")");
		Object const& o = (Object const&)call[1];
		assert(o.className() == Strings::Call && o.base().isList());
		return compileInternalFunctionCall(o, code);
	} 
	else if(func == Strings::assign ||
		func == Strings::eqassign || 
		func == Strings::assign2)
	{
		Value dest = call[1];
		
		// recursively handle emitting assignment instructions...
		Value value = call[2];
		while(isCall(dest)) {
			List c = List(((Object const&)dest).base());
			List n(c.length+1);

			for(int64_t i = 0; i < c.length; i++) { n[i] = c[i]; }
			Character nnames(c.length+1);
			if(((Object const&)dest).hasNames()) {
				Value names = ((Object const&)dest).getNames();
				for(int64_t i = 0; i < c.length; i++) { nnames[i] = Character(names)[i]; }
			} else {
				for(int64_t i = 0; i < c.length; i++) { nnames[i] = Strings::empty; }
			}

			n[0] = CreateSymbol(state.internStr(state.externStr(SymbolStr(c[0])) + "<-"));
			n[c.length] = value;
			nnames[c.length] = Strings::value;
			value = CreateCall(n, nnames);
			dest = c[1];
		}
		
		// the source for the assignment
		int64_t source = compile(value, code);

		emit(code, func == Strings::assign2 ? ByteCode::assign2 : ByteCode::assign, SymbolStr(dest), 0, source);
	
		scopes.back().deadAfter(source);
		return source;
	} 
	else if(func == Strings::bracket) {
		if(call.length != 3) return compileFunctionCall(call, names, code);
		int64_t value = compile(call[1], code);
		int64_t index = compile(call[2], code);
		scopes.back().deadAfter(liveIn);	
		int64_t reg = scopes.back().allocRegister(Register::CONSTANT);	
		emit(code, ByteCode::subset, value, index, reg);
		scopes.back().deadAfter(reg);	
		return reg;
	} 
	else if(func == Strings::bb ||
		func == Strings::dollar) {
		if(call.length != 3) return compileFunctionCall(call, names, code);
		int64_t value = compile(call[1], code);
		int64_t index = compile(call[2], code);
		scopes.back().deadAfter(liveIn);	
		int64_t reg = scopes.back().allocRegister(Register::CONSTANT);	
		emit(code, ByteCode::subset2, value, index, reg);
		scopes.back().deadAfter(reg);	
		return reg;
	} 
	else if(func == Strings::bracketAssign) { 
		if(call.length != 4) return compileFunctionCall(call, names, code);
		int64_t dest = compile(call[1], code);
		int64_t index = compile(call[2], code);
		int64_t value = compile(call[3], code);
		emit(code, ByteCode::iassign, value, index, dest);
		scopes.back().deadAfter(dest);	
		return dest;
	} 
	else if(func == Strings::bbAssign ||
		func == Strings::dollarAssign) {
		if(call.length != 4) return compileFunctionCall(call, names, code);
		int64_t dest = compile(call[1], code);
		int64_t index = compile(call[2], code);
		int64_t value = compile(call[3], code);
		emit(code, ByteCode::eassign, value, index, dest);
		scopes.back().deadAfter(dest);	
		return dest;
	} 
	else if(func == Strings::function) 
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
		functionCode->string = SymbolStr(call[3]);
		functionCode->dots = parameters.length;
		for(int64_t i = 0; i < parameters.length; i++) 
			if(parameters[i] == Strings::dots) functionCode->dots = i;

		code->prototypes.push_back(functionCode);
		
		scopes.back().deadAfter(liveIn);	
		int64_t reg = scopes.back().allocRegister(Register::CONSTANT);	
		emit(code, ByteCode::function, code->prototypes.size()-1, 0, reg);
		return reg;
	} 
	else if(func == Strings::returnSym)
	{
		int64_t result;
		if(call.length == 1) {
			result = compile(Null::Singleton(), code);
		} else if(call.length == 2)
			result = compile(call[1], code);
		else
			throw CompileError("Too many parameters to return. Wouldn't multiple return values be nice?\n");
		emit(code, ByteCode::ret, (int64_t)0, (int64_t)0, result);
		scopes.back().deadAfter(result);
		return result;
	} 
	else if(func == Strings::forSym) 
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
			emit(code, ByteCode::forbegin, (int64_t)0, loop_counter, loop_variable);
			loopDepth++;
			int64_t beginbody = code->bc.size();

			emit(code, ByteCode::assign, SymbolStr(call[1]), 0, loop_variable);
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
	} 
	else if(func == Strings::whileSym)
	{
		int64_t head_condition = compile(call[1], code);
		emit(code, ByteCode::jf, (int64_t)0, head_condition, liveIn);
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
	} 
	else if(func == Strings::repeatSym)
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
	}
	else if(func == Strings::nextSym)
	{
		if(loopDepth == 0) throw CompileError("next used outside of loop");
		int64_t result = scopes.back().allocRegister(Register::TEMP);	
		emit(code, ByteCode::jmp, (int64_t)0, (int64_t)1, result);
		return result;
	} 
	else if(func == Strings::breakSym)
	{
		if(loopDepth == 0) throw CompileError("break used outside of loop");
		int64_t result = scopes.back().allocRegister(Register::TEMP);	
		emit(code, ByteCode::jmp, (int64_t)0, (int64_t)2, result);
		return result;
	} 
	else if(func == Strings::ifSym) 
	{
		int64_t resultT=0, resultF=0;
		if(call.length != 3 && call.length != 4)	
			throw CompileError("invalid if statement");
		if(call.length == 3)
			resultF = compile(Null::Singleton(), code);
		int64_t cond = compile(call[1], code);
		emit(code, ByteCode::jf, (int64_t)0, cond, liveIn);
		int64_t begin1 = code->bc.size(), begin2 = 0;
		scopes.back().deadAfter(liveIn);
		resultT = compile(call[2], code);
		
		if(call.length == 4) {
			emit(code, ByteCode::jmp, (int64_t)0, (int64_t)0, (int64_t)0);
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
	} 
	else if(func == Strings::switchSym)
	{
		if(call.length == 0) _error("'EXPR' is missing");
		int64_t c = compile(call[1], code);
		int64_t n = call.length-2;
		int64_t branch = emit(code, ByteCode::branch, c, n, 0);
		for(int64_t i = 2; i < call.length; i++) {
			emit(code, ByteCode::branch, (names.length > i ? names[i] : Strings::empty), 0, 0);
		}
		scopes.back().deadAfter(liveIn);
		
		std::vector<int64_t> jmps;
		int64_t result = compile(Null::Singleton(), code);
		jmps.push_back(emit(code, ByteCode::jmp, (int64_t)0, (int64_t)0, (int64_t)0));	
		
		for(int64_t i = 1; i <= n; i++) {
			code->bc[branch+i].c = code->bc.size()-branch;
			scopes.back().deadAfter(liveIn);
			if(!call[i+1].isNil()) {
				int64_t r = compile(call[i+1], code);
				if(r != result) throw CompileError(std::string("switch statement doesn't put all its results in the same register"));
				if(i < n)
					jmps.push_back(emit(code, ByteCode::jmp, (int64_t)0, (int64_t)0, (int64_t)0));
			} else if(i == n) {
				compile(Null::Singleton(), code);
			}
		}
		for(int64_t i = 0; i < (int64_t)jmps.size(); i++) {
			code->bc[jmps[i]].a = code->bc.size()-jmps[i];
		}
		scopes.back().deadAfter(result);
		return result;
	} 
	else if(func == Strings::brace) 
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
	} 
	else if(func == Strings::paren) 
	{
		return compile(call[1], code);
	} 
	else if(func == Strings::add)
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
	} 
	else if(func == Strings::sub) 
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
	} 
	// Binary operators
	else if(func == Strings::colon ||
		func == Strings::mul ||
		func == Strings::div || 
		func == Strings::idiv || 
		func == Strings::pow || 
		func == Strings::mod ||
		func == Strings::atan2 ||
		func == Strings::hypot ||
		func == Strings::land || 
		func == Strings::lor ||
		func == Strings::slor ||
		func == Strings::sland ||
		func == Strings::eq ||
		func == Strings::neq ||
		func == Strings::lt ||
		func == Strings::gt ||
		func == Strings::ge ||
		func == Strings::le ||
		func == Strings::mmul)
	{
		// if there aren't exactly two parameters, we should call the library version...
		if(call.length != 3) return compileFunctionCall(call, names, code);
		int64_t a = compile(call[1], code);
		int64_t b = compile(call[2], code);
		scopes.back().deadAfter(liveIn);
		int64_t result = scopes.back().allocRegister(Register::TEMP);
		emit(code, op(func), a, b, result);
		return result;
	} 
	// Unary operators
	else if(func == Strings::lnot || 
		func == Strings::abs || 
		func == Strings::sign ||
		func == Strings::sqrt ||
		func == Strings::floor ||
		func == Strings::ceiling || 
		func == Strings::trunc ||
		func == Strings::round ||
		func == Strings::signif ||
		func == Strings::exp ||
		func == Strings::log ||
		func == Strings::cos ||
		func == Strings::sin ||
		func == Strings::tan ||
		func == Strings::acos ||
		func == Strings::asin ||
		func == Strings::atan ||
		func == Strings::sum ||
		func == Strings::prod ||
		func == Strings::min ||
		func == Strings::max ||
		func == Strings::any ||
		func == Strings::all ||
		func == Strings::cumsum ||
		func == Strings::cumprod ||
		func == Strings::cummin ||
		func == Strings::cummax ||
		func == Strings::cumany ||
		func == Strings::cumall ||
		func == Strings::type ||
		func == Strings::length ||
		func == Strings::Logical ||
		func == Strings::Integer ||
		func == Strings::Double ||
		func == Strings::Character ||
		func == Strings::Raw ||
		func == Strings::strip)
	{
		// if there isn't exactly one parameter, we should call the library version...
		if(call.length != 2) return compileFunctionCall(call, names, code);
		int64_t a = compile(call[1], code);
		scopes.back().deadAfter(liveIn);
		int64_t result = scopes.back().allocRegister(Register::TEMP);
		emit(code, op(func), a, 0, result);
		return result; 
	} 
	else if(func == Strings::UseMethod)
	{
		if(scopes.back().topLevel)
			throw CompileError("Attempt to use UseMethod outside of function");
		
		// This doesn't match R's behavior. R always uses the original value of the first argument, not the most recent value. Blah.
		int64_t object = (call.length == 3) 
			? compile(call[2], code) : compile(CreateSymbol(scopes.back().parameters[0]), code); 
		if(!call[1].isCharacter1())
			throw CompileError("First parameter to UseMethod must be a string");
		String generic = SymbolStr(call[1]);
		
		Character p(scopes.back().parameters);
		List gcall(p.length+1);
		for(int64_t i = 0; i < p.length; i++) gcall[i+1] = CreateSymbol(p[i]);
		code->calls.push_back(makeCall(gcall, Character(0)));
	
		emit(code, ByteCode::UseMethod, generic, code->calls.size()-1, object);
		scopes.back().deadAfter(object);
		return object;
	} 
	else if(func == Strings::seq_len)
	{
		int64_t len = compile(call[1], code);
		int64_t step = compile(Integer::c(1), code);
		scopes.back().deadAfter(liveIn);
		int64_t result = scopes.back().allocRegister(Register::TEMP);
		emit(code, ByteCode::seq, len, step, result);
		return result;
	}
	else if(func == Strings::ifelse)
	{
		if(call.length != 4)
			return compileFunctionCall(call, names, code);
		int64_t no = compile(call[3], code);
		int64_t yes = compile(call[2], code);
		int64_t cond = compile(call[1], code);
		scopes.back().deadAfter(liveIn);
		int64_t result = scopes.back().allocRegister(Register::TEMP);
		assert(no == result);
		emit(code, ByteCode::ifelse, cond, yes, no);
		return result;
	} 
	else if(func == Strings::split)
	{
		if(call.length != 4)
			return compileFunctionCall(call, names, code);
		int64_t levels = compile(call[3], code);
		int64_t factor = compile(call[2], code);
		int64_t x = compile(call[1], code);
		scopes.back().deadAfter(liveIn);
		int64_t result = scopes.back().allocRegister(Register::TEMP);
		assert(levels == result);
		emit(code, ByteCode::split, x, factor, levels);
		return result;
	} 
	else if(func == Strings::docall)
	{
		int64_t what = compile(call[1], code);
		int64_t args = compile(call[2], code);
		scopes.back().deadAfter(liveIn);
		int64_t result = scopes.back().allocRegister(Register::TEMP);
		emit(code, ByteCode::call, what, args, result);
		return result;
	} 
	else if(func == Strings::list)
	{
		// we only handle the list(...) case through an op for now
		if(call.length != 2 || !isSymbol(call[1]) || SymbolStr(call[1]) != Strings::dots)
			return compileFunctionCall(call, names, code);
		scopes.back().deadAfter(liveIn);
		int64_t result = scopes.back().allocRegister(Register::TEMP);
		int64_t counter = compileConstant(Integer::c(0), code);
		int64_t storage = scopes.back().allocRegister(Register::TEMP);
		emit(code, ByteCode::list, counter, storage, result); 
		scopes.back().deadAfter(result);
		return result;
	} 
	else if(func == Strings::missing)
	{
		if(call.length != 2) _error("missing requires one argument");
		if(!isSymbol(call[1]) && !call[1].isCharacter1()) _error("wrong parameter to missing");
		String s = SymbolStr(call[1]);
		int64_t result = scopes.back().allocRegister(Register::TEMP);
		emit(code, ByteCode::missing, s, 0, result); 
		scopes.back().deadAfter(result);
		return result;
	} 
	else if(func == Strings::quote)
	{
		if(call.length != 2) _error("quote requires one argument");
		return compileConstant(call[1], code);
	}
	else if(func == Strings::apply)
	{
		List c(Subset(call, 1, call.length-1));
		Character n(Subset(names, 1, call.length-1));
		code->calls.push_back(makeCall(c, n));
		int64_t function = compile(c[0], code);
		scopes.back().deadAfter(liveIn);
		int64_t result = scopes.back().allocRegister(Register::TEMP);
		emit(code, ByteCode::apply, function, -code->calls.size(), result);
		return result;
	}
	else
	{
		return compileFunctionCall(call, names, code);
	}
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
		//case Type::Symbol:
		//	return compileSymbol(Symbol(expr), code);
		//	break;			
		case Type::Object:
			{
				Object const& o = (Object const&) expr;
				if(o.className() == Strings::Symbol) {
					return compileSymbol(expr, code);
				}
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
			}	
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
	emit(code, ByteCode::ret, (int64_t)0, (int64_t)0, result);
	
	oldRegisters.swap(scopes.back().registers);
	scopes.back().maxRegister = oldMaxRegister;	
	loopDepth = oldLoopDepth;

	return code;	
}

