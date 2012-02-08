
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
	if(func == Strings::lor) return ByteCode::lor; 
	if(func == Strings::land) return ByteCode::land; 
	if(func == Strings::lnot) return ByteCode::lnot; 
	if(func == Strings::abs) return ByteCode::abs; 
	if(func == Strings::sign) return ByteCode::sign; 
	if(func == Strings::sqrt) return ByteCode::sqrt; 
	if(func == Strings::floor) return ByteCode::floor; 
	if(func == Strings::ceiling) return ByteCode::ceiling; 
	if(func == Strings::trunc) return ByteCode::trunc; 
	//if(func == Strings::round) return ByteCode::round; 
	//if(func == Strings::signif) return ByteCode::signif; 
	if(func == Strings::exp) return ByteCode::exp; 
	if(func == Strings::log) return ByteCode::log; 
	if(func == Strings::cos) return ByteCode::cos; 
	if(func == Strings::sin) return ByteCode::sin; 
	if(func == Strings::tan) return ByteCode::tan; 
	if(func == Strings::acos) return ByteCode::acos; 
	if(func == Strings::asin) return ByteCode::asin; 
	if(func == Strings::atan) return ByteCode::atan; 
	if(func == Strings::isna) return ByteCode::isna; 
	if(func == Strings::isnan) return ByteCode::isnan; 
	if(func == Strings::isfinite) return ByteCode::isfinite; 
	if(func == Strings::isinfinite) return ByteCode::isinfinite; 
	
	if(func == Strings::pmin) return ByteCode::pmin; 
	if(func == Strings::pmax) return ByteCode::pmax; 
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
	int64_t reg = allocRegister();
	constRegisters.push_back(reg);
	//emit(code, ByteCode::kget, -(code->constants.size()-1), 0, reg);
	return reg;
}

int64_t Compiler::compileSymbol(Value const& symbol, Prototype* code) {
	//int64_t reg = allocRegister();
	String s = SymbolStr(symbol);
	//emit(code, ByteCode::get, s, 0, reg);
	//emit(code, ByteCode::assign, s, 0, reg);
	//return reg;
	return (int64_t)s;
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
			arguments[i-1] = Function(Compiler::compilePromise(state, call[i]),NULL).AsPromise();
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
	code->calls.push_back(makeCall(call, names));
	int64_t function = compile(call[0], code);
	int64_t result = allocRegister();
	emit(code, ByteCode::call, function, -code->calls.size(), result);
	return result;
}

int64_t Compiler::compileInternalFunctionCall(Object const& o, Prototype* code) {
	List const& call = (List const&)(o.base());
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
	int64_t reg = this->n;
	for(int64_t i = 1; i < call.length; i++) {
		int64_t r = compile(call[i], code);
		//assert(r == reg+1);
		reg = r; 
	}
	int64_t result = allocRegister();
	emit(code, ByteCode::internal, function, reg-(call.length-2), result);
	return result;
}

int64_t Compiler::compileCall(List const& call, Character const& names, Prototype* code) {
	int64_t length = call.length;
	if(length == 0) {
		throw CompileError("invalid empty call");
	}

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
		// check if we can merge the assignment into the source statement...
		// otherwise, emit a new assignment statement
		if(func == Strings::assign2) { 
			emit(code, ByteCode::assign2, SymbolStr(dest), 0, source);
			return source;
		} else {
			if(code->bc.size() > 0 && code->bc.back().c < 0) {
				code->bc.back().c = (int64_t)SymbolStr(dest);
				return (int64_t)SymbolStr(dest);
			} else {
				emit(code, ByteCode::mov, source, 0, (int64_t)SymbolStr(dest));
				return source;
			}
		}
	} 
	else if(func == Strings::bracket) {
		if(call.length != 3) return compileFunctionCall(call, names, code);
		int64_t value = compile(call[1], code);
		int64_t index = compile(call[2], code);
		int64_t reg = allocRegister();	
		emit(code, ByteCode::subset, value, index, reg);
		return reg;
	} 
	else if(func == Strings::bb ||
		func == Strings::dollar) {
		if(call.length != 3) return compileFunctionCall(call, names, code);
		int64_t value = compile(call[1], code);
		int64_t index = compile(call[2], code);
		int64_t reg = allocRegister();	
		emit(code, ByteCode::subset2, value, index, reg);
		return reg;
	} 
	else if(func == Strings::bracketAssign) { 
		if(call.length != 4) return compileFunctionCall(call, names, code);
		int64_t dest = compile(call[1], code);
		int64_t index = compile(call[2], code);
		int64_t value = compile(call[3], code);
		emit(code, ByteCode::iassign, value, index, dest);
		return dest;
	} 
	else if(func == Strings::bbAssign ||
		func == Strings::dollarAssign) {
		if(call.length != 4) return compileFunctionCall(call, names, code);
		int64_t dest = compile(call[1], code);
		int64_t index = compile(call[2], code);
		int64_t value = compile(call[3], code);
		emit(code, ByteCode::eassign, value, index, dest);
		return dest;
	} 
	else if(func == Strings::function) 
	{
		//compile the default parameters
		assert(call[1].isObject());
		List c = List(((Object const&)call[1]).base());
		Character parameters = ((Object const&)call[1]).hasNames() ? 
			Character(((Object const&)call[1]).getNames()) :
			Character(0);
		
		List defaults(c.length);
		for(int64_t i = 0; i < defaults.length; i++) {
			if(!c[i].isNil()) {
				defaults[i] = Function(compilePromise(state, c[i]),NULL).AsDefault();
			}
			else {
				defaults[i] = c[i];
			}
		}

		//compile the source for the body
		Prototype* functionCode = Compiler::compileFunctionBody(state, call[2], parameters);

		// Populate function info
		functionCode->parameters = parameters;
		functionCode->defaults = defaults;
		functionCode->string = SymbolStr(call[3]);
		functionCode->dots = parameters.length;
		for(int64_t i = 0; i < parameters.length; i++) 
			if(parameters[i] == Strings::dots) functionCode->dots = i;

		code->prototypes.push_back(functionCode);
		
		int64_t reg = allocRegister();	
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
			int64_t loop_counter = allocRegister();	// save space for loop counter
			int64_t loop_variable = allocRegister();
			
			if(loop_counter != loop_vector+1) throw CompileError("limits aren't in adjacent registers");
			emit(code, ByteCode::forbegin, (int64_t)0, loop_counter, loop_variable);
			loopDepth++;
			int64_t beginbody = code->bc.size();

			emit(code, ByteCode::mov, loop_variable, 0, (int64_t)SymbolStr(call[1]));
			compile(call[3], code);

			int64_t endbody = code->bc.size();
			resolveLoopReferences(code, beginbody, endbody, endbody, endbody+1);
			loopDepth--;
			emit(code, ByteCode::forend, beginbody-endbody, loop_counter , loop_variable);
			code->bc[beginbody-1].a = endbody-beginbody+1;
		//}
		int64_t result = compile(Null::Singleton(), code);
		return result;
	} 
	else if(func == Strings::whileSym)
	{
		int64_t head_condition = compile(call[1], code);
		emit(code, ByteCode::jc, (int64_t)1, (int64_t)0, head_condition);
		loopDepth++;
		
		int64_t beginbody = code->bc.size();
		compile(call[2], code);
		int64_t tail = code->bc.size();
		int64_t tail_condition = compile(call[1], code);
		int64_t endbody = code->bc.size();
		
		emit(code, ByteCode::jc, beginbody-endbody, (int64_t)1, tail_condition);
		resolveLoopReferences(code, beginbody, endbody, tail, endbody+1);
		code->bc[beginbody-1].b = endbody-beginbody+2;
		
		loopDepth--;
		int64_t result = compile(Null::Singleton(), code);
		return result;
	} 
	else if(func == Strings::repeatSym)
	{
		loopDepth++;
		int64_t start = this->n;

		int64_t beginbody = code->bc.size();
		compile(call[1], code);
		int64_t endbody = code->bc.size();
		resolveLoopReferences(code, beginbody, endbody, endbody, endbody+1);
		
		loopDepth--;
		emit(code, ByteCode::jmp, beginbody-endbody, 0, start);
		int64_t result = compile(Null::Singleton(), code);
		return result;
	}
	else if(func == Strings::nextSym)
	{
		if(loopDepth == 0) throw CompileError("next used outside of loop");
		int64_t result = allocRegister();	
		emit(code, ByteCode::jmp, (int64_t)0, (int64_t)1, result);
		return result;
	} 
	else if(func == Strings::breakSym)
	{
		if(loopDepth == 0) throw CompileError("break used outside of loop");
		int64_t result = allocRegister();	
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
		emit(code, ByteCode::jc, (int64_t)1, (int64_t)0, cond);
		int64_t begin1 = code->bc.size(), begin2 = 0;
		resultT = compile(call[2], code);
		
		if(call.length == 4) {
			emit(code, ByteCode::jmp, (int64_t)0, (int64_t)0, (int64_t)0);
			begin2 = code->bc.size();
			resultF = compile(call[3], code);
		}
		else
			begin2 = code->bc.size();
		int64_t end = code->bc.size();
		code->bc[begin1-1].b = begin2-begin1+1;
		if(call.length == 4)
			code->bc[begin2-1].a = end-begin2+1;
	
		// TODO: if this can ever happen, should probably just insert a move into the lower numbered register	
		if(resultT != resultF) throw CompileError(std::string("then and else blocks don't put the result in the same register ") + intToStr(resultT) + " " + intToStr(resultF));
		return resultT;
	}
	else if(func == Strings::lor2 && call.length == 3)
	{
		int64_t start = compileConstant(Logical::True(), code);
		int64_t left = compile(call[1], code);
		emit(code, ByteCode::jc, (int64_t)0, (int64_t)1, left);
		int64_t j1 = code->bc.size()-1;
		int64_t right = compile(call[2], code);
		emit(code, ByteCode::jc, (int64_t)2, (int64_t)1, right);
		compileConstant(Logical::False(), code);
		code->bc[j1].a = code->bc.size()-j1;
		return start;
	}
	else if(func == Strings::land2 && call.length == 3)
	{
		int64_t start = compileConstant(Logical::False(), code);
		int64_t left = compile(call[1], code);
		emit(code, ByteCode::jc, (int64_t)1, (int64_t)0, left);
		int64_t j1 = code->bc.size()-1;
		int64_t right = compile(call[2], code);
		emit(code, ByteCode::jc, (int64_t)1, (int64_t)2, right);
		compileConstant(Logical::True(), code);
		code->bc[j1].b = code->bc.size()-j1;
		return start;
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
		
		std::vector<int64_t> jmps;
		int64_t result = compile(Null::Singleton(), code);
		jmps.push_back(emit(code, ByteCode::jmp, (int64_t)0, (int64_t)0, (int64_t)0));	
		
		for(int64_t i = 1; i <= n; i++) {
			code->bc[branch+i].c = code->bc.size()-branch;
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
				result = compile(call[i], code);
			}
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
			result = allocRegister();
			emit(code, ByteCode::pos, a, 0, result);
		} else if(call.length == 3) {
			int64_t a = compile(call[1], code);
			int64_t b = compile(call[2], code);
			result = allocRegister();
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
			result = allocRegister();
			emit(code, ByteCode::neg, a, 0, result);
		} else if(call.length == 3) {
			int64_t a = compile(call[1], code);
			int64_t b = compile(call[2], code);
			result = allocRegister();
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
		func == Strings::eq ||
		func == Strings::neq ||
		func == Strings::lt ||
		func == Strings::gt ||
		func == Strings::ge ||
		func == Strings::le ||
		func == Strings::pmin ||
		func == Strings::pmax ||
		func == Strings::lor ||
		func == Strings::land ||
		func == Strings::mmul)
	{
		// if there aren't exactly two parameters, we should call the library version...
		if(call.length != 3) return compileFunctionCall(call, names, code);
		int64_t a = compile(call[1], code);
		int64_t b = compile(call[2], code);
		int64_t result = allocRegister();
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
		func == Strings::isna ||
		func == Strings::isnan ||
		func == Strings::isfinite ||
		func == Strings::isinfinite ||
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
		int64_t result = allocRegister();
		emit(code, op(func), a, 0, result);
		return result; 
	} 
	else if(func == Strings::UseMethod)
	{
		if(scope != FUNCTION)
			throw CompileError("Attempt to use UseMethod outside of function");
		
		// This doesn't match R's behavior. R always uses the original value of the first argument, not the most recent value. Blah.
		int64_t object = (call.length == 3) 
			? compile(call[2], code) : compile(CreateSymbol(parameters[0]), code); 
		if(!call[1].isCharacter1())
			throw CompileError("First parameter to UseMethod must be a string");
		String generic = SymbolStr(call[1]);
		
		Character p(parameters);
		List gcall(p.length+1);
		for(int64_t i = 0; i < p.length; i++) gcall[i+1] = CreateSymbol(p[i]);
		code->calls.push_back(makeCall(gcall, Character(0)));
	
		emit(code, ByteCode::UseMethod, generic, code->calls.size()-1, object);
		return object;
	} 
	else if(func == Strings::seq_len)
	{
		int64_t len = compile(call[1], code);
		int64_t step = compile(Integer::c(1), code);
		int64_t result = allocRegister();
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
		int64_t result = allocRegister();
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
		int64_t result = allocRegister();
		assert(levels == result);
		emit(code, ByteCode::split, x, factor, levels);
		return result;
	} 
	else if(func == Strings::docall)
	{
		int64_t what = compile(call[1], code);
		int64_t args = compile(call[2], code);
		int64_t result = allocRegister();
		emit(code, ByteCode::call, what, args, result);
		return result;
	} 
	else if(func == Strings::list)
	{
		// we only handle the list(...) case through an op for now
		if(call.length != 2 || !isSymbol(call[1]) || SymbolStr(call[1]) != Strings::dots)
			return compileFunctionCall(call, names, code);
		int64_t result = allocRegister();
		int64_t counter = compileConstant(Integer::c(0), code);
		int64_t storage = allocRegister();
		emit(code, ByteCode::list, counter, storage, result); 
		return result;
	} 
	else if(func == Strings::missing)
	{
		if(call.length != 2) _error("missing requires one argument");
		if(!isSymbol(call[1]) && !call[1].isCharacter1()) _error("wrong parameter to missing");
		String s = SymbolStr(call[1]);
		int64_t result = allocRegister();
		emit(code, ByteCode::missing, s, 0, result); 
		return result;
	} 
	else if(func == Strings::quote)
	{
		if(call.length != 2) _error("quote requires one argument");
		return compileConstant(call[1], code);
	}
	else
	{
		return compileFunctionCall(call, names, code);
	}
}

int64_t Compiler::compileExpression(List const& values, Prototype* code) {
	int64_t result = 0;
	if(values.length == 0) 
		throw CompileError("invalid empty expression");
	for(int64_t i = 0; i < values.length; i++) {
		result = compile(values[i], code);
	}
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

	int64_t result = compile(expr, code);

	code->registers = n+1;
	code->expression = expr;
	// insert return statement at end of code
	emit(code, ByteCode::ret, (int64_t)0, (int64_t)0, result);
	// insert constants at the beginning
	for(int i = 0; i < code->constants.size(); i++)
		code->bc.insert(code->bc.begin(), Instruction(ByteCode::kget, -i, 0, constRegisters[i]));
	return code;	
}

