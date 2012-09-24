
#include "compiler.h"
#include "runtime.h"
#include "coerce.h"

static ByteCode::Enum op1(String const& func) {
	if(func == Strings::add) return ByteCode::pos; 
	if(func == Strings::sub) return ByteCode::neg; 
	if(func == Strings::abs) return ByteCode::abs; 
	if(func == Strings::sign) return ByteCode::sign; 
	
	if(func == Strings::floor) return ByteCode::floor; 
	if(func == Strings::ceiling) return ByteCode::ceiling; 
	if(func == Strings::trunc) return ByteCode::trunc; 
	if(func == Strings::sqrt) return ByteCode::sqrt; 
	if(func == Strings::exp) return ByteCode::exp; 
	if(func == Strings::log) return ByteCode::log; 
	if(func == Strings::cos) return ByteCode::cos; 
	if(func == Strings::sin) return ByteCode::sin; 
	if(func == Strings::tan) return ByteCode::tan; 
	if(func == Strings::acos) return ByteCode::acos; 
	if(func == Strings::asin) return ByteCode::asin; 
	if(func == Strings::atan) return ByteCode::atan; 
    
    if(func == Strings::lnot) return ByteCode::lnot; 
	
	if(func == Strings::isna) return ByteCode::isna; 
	if(func == Strings::isnan) return ByteCode::isnan; 
	if(func == Strings::isfinite) return ByteCode::isfinite; 
	if(func == Strings::isinfinite) return ByteCode::isinfinite; 
    
    if(func == Strings::sum) return ByteCode::sum; 
    if(func == Strings::length) return ByteCode::length; 
	if(func == Strings::prod) return ByteCode::prod; 
	if(func == Strings::mean) return ByteCode::mean; 
	if(func == Strings::min) return ByteCode::min; 
	if(func == Strings::max) return ByteCode::max; 
	if(func == Strings::any) return ByteCode::any; 
	if(func == Strings::all) return ByteCode::all; 
	if(func == Strings::cumsum) return ByteCode::cumsum; 
	if(func == Strings::cumprod) return ByteCode::cumprod; 
	if(func == Strings::cummin) return ByteCode::cummin; 
	if(func == Strings::cummax) return ByteCode::cummax; 
	
	if(func == Strings::type) return ByteCode::type; 
	if(func == Strings::strip) return ByteCode::strip; 
	
	if(func == Strings::random) return ByteCode::random; 

    if(func == Strings::aslogical) return ByteCode::aslogical;
    if(func == Strings::asinteger) return ByteCode::asinteger;
    if(func == Strings::asdouble) return ByteCode::asdouble;
    if(func == Strings::ascharacter) return ByteCode::ascharacter;
    if(func == Strings::aslist) return ByteCode::aslist;
    
    if(func == Strings::newenv) return ByteCode::newenv;
    if(func == Strings::parentframe) return ByteCode::parentframe;
	
    throw RuntimeError(std::string("unexpected symbol '") + func + "' used as a unary operator"); 
}

static ByteCode::Enum op2(String const& func) {
	if(func == Strings::add) return ByteCode::add; 
	if(func == Strings::sub) return ByteCode::sub; 
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

	if(func == Strings::pmin) return ByteCode::pmin; 
	if(func == Strings::pmax) return ByteCode::pmax; 
	if(func == Strings::cm2) return ByteCode::cm2; 
	
    if(func == Strings::bb) return ByteCode::gather1;
	if(func == Strings::bracket) return ByteCode::gather;

	if(func == Strings::vector) return ByteCode::vector;

	if(func == Strings::round) return ByteCode::round; 
	if(func == Strings::signif) return ByteCode::signif; 
	
    if(func == Strings::attrget) return ByteCode::attrget; 
	
    throw RuntimeError(std::string("unexpected symbol '") + func + "' used as a binary operator"); 
}

static ByteCode::Enum op3(String const& func) {
    if(func == Strings::seq) return ByteCode::seq;
	if(func == Strings::bbAssign) return ByteCode::scatter1;
	if(func == Strings::bracketAssign) return ByteCode::scatter;
	if(func == Strings::ifelse) return ByteCode::ifelse;
	if(func == Strings::split) return ByteCode::split;
	if(func == Strings::rep) return ByteCode::rep;
	
    if(func == Strings::attrset) return ByteCode::attrset; 

    throw RuntimeError(std::string("unexpected symbol '") + func + "' used as a trinary operator"); 
}

int64_t Compiler::emit(ByteCode::Enum bc, Operand a, Operand b, Operand c) {
	ir.push_back(IRNode(bc, a, b, c));
	return ir.size()-1;
}

void Compiler::resolveLoopExits(int64_t start, int64_t end, int64_t nextTarget, int64_t breakTarget) {
	for(int64_t i = start; i < end; i++) {
		if(ir[i].bc == ByteCode::jmp && ir[i].a.i == 0) {
			if(ir[i].b.i == 1) {
				ir[i].a.i = nextTarget-i;
			} else if(ir[i].b.i == 2) {
				ir[i].a.i = breakTarget-i;
			}
		}
	}
}

Compiler::Operand Compiler::compileConstant(Value const& expr, Prototype* code) {
	std::map<Value, int64_t>::const_iterator i = constants.find(expr);
	int64_t index = 0;
	if(i == constants.end()) {
		index = code->constants.size();
		code->constants.push_back(expr);
		constants[expr] = index;
	} else {
		index = i->second;
	}
    Operand t = allocRegister();
    emit(ByteCode::constant, index, 0, t);
	return t;
}

static int64_t isDotDot(String s) {
	if(s != 0 && s[0] == '.' && s[1] == '.') {
		int64_t v = 0;
		int64_t i = 2;
		// maximum 64-bit integer has 19 digits, but really who's going to pass
		// that many var args?
		while(i < (19+2) && s[i] >= '0' && s[i] <= '9') {
			v = v*10 + (s[i] - '0');
			i++;
		}
		if(i < (19+2) && s[i] == 0) return v;
	}
	return -1;	
}

Compiler::Operand Compiler::compileSymbol(Value const& symbol, Prototype* code) {
	String s = SymbolStr(symbol);
	int64_t dd = isDotDot(s);
	if(dd > 0) {
		Operand t = allocRegister();
		emit(ByteCode::dotdot, dd-1, 0, t);
		return t;
	}
	else {
		//return Operand(MEMORY, s);
        return forceInRegister(Operand(MEMORY,s));
	}
}

Compiler::Operand Compiler::placeInRegister(Operand r) {
	if(r.loc != REGISTER && r.loc != INVALID) {
		kill(r);
		Operand t = allocRegister();
		emit(ByteCode::fastmov, r, 0, t);
		return t;
	}
	return r;
}

Compiler::Operand Compiler::forceInRegister(Operand r) {
	if(r.loc != INVALID) {
		kill(r);
		Operand t = allocRegister();
		emit(ByteCode::mov, r, 0, t);
		return t;
	}
	return r;
}

CompiledCall Compiler::makeCall(List const& call, Character const& names) {
	// compute compiled call...precompiles promise code and some necessary values
	int64_t dotIndex = call.length-1;
	PairList arguments;
	for(int64_t i = 1; i < call.length; i++) {
		Pair p;
		if(names.length > 0) p.n = names[i]; else p.n = Strings::empty;

		if(isSymbol(call[i]) && SymbolStr(call[i]) == Strings::dots) {
			p.v = call[i];
			dotIndex = i-1;
		} else if(isCall(call[i]) || isSymbol(call[i])) {
			Promise::Init(p.v, Compiler::compilePromise(thread, call[i]),NULL);
		} else {
			p.v = call[i];
		}
		arguments.push_back(p);
	}
	return CompiledCall(call, arguments, dotIndex, names.length > 0);
}

// a standard call, not an op
Compiler::Operand Compiler::compileFunctionCall(List const& call, Character const& names, Prototype* code) {
	Operand function = compile(call[0], code);
	CompiledCall a = makeCall(call, names);
	code->calls.push_back(a);
	kill(function);
	Operand result = allocRegister();
	if(!a.named && a.dotIndex >= (int64_t)a.arguments.size())
		emit(ByteCode::call, function, code->calls.size()-1, result);
	else
		emit(ByteCode::ncall, function, code->calls.size()-1, result);
	return result;
}

Compiler::Operand Compiler::compileInternalFunctionCall(Object const& o, Prototype* code) {
	List const& call = (List const&)(o.base());
	String func = SymbolStr(call[0]);
	std::map<String, int64_t>::const_iterator itr = state.internalFunctionIndex.find(func);
	
	int64_t function = -1;
	
	if(itr == state.internalFunctionIndex.end()) {
		// this should eventually be an error, but so many are unimplemented right now that we'll just make it a warning
		_warning(thread, std::string("Unimplemented internal function ") + state.externStr(func));
	}
	else {
		function = itr->second;
		// check parameter count
		if(state.internalFunctions[function].params != call.length-1)
			_error(std::string("Incorrect number of arguments to internal function ") + state.externStr(func));
	}
	// compile parameters directly...reserve registers for them.
	Operand liveIn = top();
	int64_t reg = liveIn.i-1;
	for(int64_t i = 1; i < call.length; i++) {
		Operand r = placeInRegister(compile(call[i], code));
		assert(r.i == reg+1);
		reg = r.i; 
	}
	// kill all parameters
	kill(liveIn); 
	Operand result = allocRegister();
	emit(ByteCode::internal, function, result, result);
	return result;
}

Compiler::Operand Compiler::compileCall(List const& call, Character const& names, Prototype* code) {
	int64_t length = call.length;
	if(length == 0) {
		throw CompileError("invalid empty call");
	}

	if(!isSymbol(call[0]) && !call[0].isCharacter1())
		return compileFunctionCall(call, names, code);

	String func = SymbolStr(call[0]);
	// list is the only built in function that handles ... or named parameters
	// we only handle the list(...) case through an op for now
	if(func == Strings::list && call.length == 2 
		&& isSymbol(call[1]) && SymbolStr(call[1]) == Strings::dots)
	{
		Operand result = allocRegister();
		Operand counter = compileConstant(Integer::c(0), code);
		Operand storage = allocRegister();
		kill(storage); kill(counter);
		emit(ByteCode::dotslist, counter, storage, result); 
		return result;
	}
	// These functions can't be called directly if the arguments are named or if
	// there is a ... in the args list

	bool complicated = false;
	for(int64_t i = 0; i < length; i++) {
		if(isSymbol(call[i]) && SymbolStr(call[i]) == Strings::dots) 
			complicated = true;
	}

	if(complicated)
		return compileFunctionCall(call, names, code);

	// switch statement supports named args
	if(func == Strings::switchSym)
	{
		if(call.length == 0) _error("'EXPR' is missing");
		Operand c = compile(call[1], code);
		int64_t n = call.length-2;

		int64_t branch = emit(ByteCode::branch, kill(c), n, 0);
		for(int64_t i = 2; i < call.length; i++) {
			emit(ByteCode::branch, (int64_t)(names.length > i ? names[i] : Strings::empty), 0, 0);
		}
		
		std::vector<int64_t> jmps;
		Operand result = placeInRegister(compileConstant(Null::Singleton(), code));
		jmps.push_back(emit(ByteCode::jmp, (int64_t)0, (int64_t)0, (int64_t)0));
		
		for(int64_t i = 1; i <= n; i++) {
			ir[branch+i].c = (int64_t)ir.size()-branch;
			if(!call[i+1].isNil()) {
				kill(result);
				Operand r = placeInRegister(compile(call[i+1], code));
				if(r != result) 
					throw CompileError(std::string("switch statement doesn't put all its results in the same register"));
				if(i < n)
					jmps.push_back(emit(ByteCode::jmp, (int64_t)0, (int64_t)0, (int64_t)0));
				result = r;
			} else if(i == n) {
				kill(result);
				Operand r = placeInRegister(compileConstant(Null::Singleton(), code));
				if(r != result) 
					throw CompileError(std::string("switch statement doesn't put all its results in the same register"));
				result = r;
			}
		}
		for(int64_t i = 0; i < (int64_t)jmps.size(); i++) {
			ir[jmps[i]].a = (int64_t)ir.size()-jmps[i];
		}
		return result;
	}
	
    for(int64_t i = 0; i < length; i++) {
		if(names.length > i && names[i] != Strings::empty) 
			complicated = true;
	}
	
	if(complicated)
		return compileFunctionCall(call, names, code);


	if(func == Strings::internal) 
	{
		if(!call[1].isObject())
			throw CompileError(std::string(".Internal has invalid arguments (") + Type::toString(call[1].type) + ")");
		Object const& o = (Object const&)call[1];
		assert(isCall(o) && o.base().isList());
		return compileInternalFunctionCall(o, code);
	}
	else if(func == Strings::assign ||
		func == Strings::eqassign || 
		func == Strings::assign2)
	{
		Value dest = call[1];
		
		// handle complex LHS assignment instructions...
		// semantics here are kind of tricky. Consider compiling:
		//	 dim(a)[1] <- x
		// This is progressively turned `inside out`:
		//	1) dim(a) <- `[<-`(dim(a), 1, x)
		//	2) a <- `dim<-`(a, `[<-`(dim(a), 1, x))
		// TODO: One complication is that the result value of a complex assignment is the RHS
		// of the original expression, not the RHS of the inside out expression.
		Value value = call[2];
		while(isCall(dest)) {
			List const& c = (List const&)((Object const&)dest).base();
			List n(c.length+1);

			for(int64_t i = 0; i < c.length; i++) { n[i] = c[i]; }
			String as = state.internStr(state.externStr(SymbolStr(c[0])) + "<-");
			n[0] = CreateSymbol(as);
			n[c.length] = value;

			Character nnames(c.length+1);
	
			if(!hasNames(dest) && (as == Strings::bracketAssign || as == Strings::bbAssign) && c.length == 3) {
				for(int64_t i = 0; i < c.length+1; i++) { nnames[i] = Strings::empty; }
			}
			else {
				if(hasNames(dest)) {
					Value names = getNames((Object const&)dest);
					for(int64_t i = 0; i < c.length; i++) { nnames[i] = ((Character const&)names)[i]; }
				} else {
					for(int64_t i = 0; i < c.length; i++) { nnames[i] = Strings::empty; }
				}
				nnames[c.length] = Strings::value;
			}
			value = CreateCall(n, nnames);
			dest = c[1];
		}
		
		// the source for the assignment
		Operand source = compile(value, code);
		Operand result = Operand(MEMORY, SymbolStr(dest));

		emit(func == Strings::assign2 ? ByteCode::assign2 : ByteCode::assign, result, 0, source);
		return source;
	}
	else if(func == Strings::function) 
	{
		//compile the default parameters
		assert(call[1].isObject());
		List const& c = (List const&)((Object const&)call[1]).base();
		Character names = hasNames(call[1]) ? 
			(Character const&)getNames((Object&)call[1]) :
			Character(0);
		
		PairList parameters;
		for(int64_t i = 0; i < c.length; i++) {
			Pair p;
			if(names.length > 0) p.n = names[i]; else p.n = Strings::empty;
			if(!c[i].isNil()) {
				Default::Init(p.v, compilePromise(thread, c[i]),NULL);
			}
			else {
				p.v = c[i];
			}
			parameters.push_back(p);
		}

		//compile the source for the body
		Prototype* functionCode = Compiler::compileFunctionBody(thread, call[2]);

		// Populate function info
		functionCode->parameters = parameters;
		functionCode->string = SymbolStr(call[3]);
		functionCode->dotIndex = parameters.size();
		for(int64_t i = 0; i < (int64_t)parameters.size(); i++) 
			if(parameters[i].n == Strings::dots) functionCode->dotIndex = i;

		Value function;
		Function::Init(function, functionCode, 0);
		Operand funcOp = compileConstant(function, code);
	
		Operand reg = allocRegister();	
		emit(ByteCode::function, funcOp, 0, reg);
		return reg;
	} 
	else if(func == Strings::returnSym)
	{
		Operand result;
		if(call.length == 1) {
			result = compileConstant(Null::Singleton(), code);
		} else if(call.length == 2)
			result = placeInRegister(compile(call[1], code));
		else
			throw CompileError("Too many parameters to return. Wouldn't multiple return values be nice?\n");
		if(scope != FUNCTION)
			throw CompileError("Attempting to return from top-level expression or from non-function. Riposte doesn't support return inside promises currently, and may never do so");
		emit(ByteCode::ret, result, 0, 0);
		return result;
	} 
	else if(func == Strings::forSym) 
	{
		Operand loop_variable = Operand(MEMORY, SymbolStr(call[1]));
		Operand loop_vector = compile(call[2], code);
		Operand loop_counter = allocRegister();	// save space for loop counter
		Operand loop_length = allocRegister();	// save space for loop counter

		emit(ByteCode::forbegin, loop_variable, loop_vector, loop_counter);
		emit(ByteCode::jmp, 0, 0, 0);
		
		loopDepth++;
		int64_t beginbody = ir.size();
        emit(ByteCode::loop, 0, 0, 0);
		Operand body = compile(call[3], code);
		int64_t endbody = ir.size();
		resolveLoopExits(beginbody, endbody, endbody, endbody+2);
		loopDepth--;
		
		emit(ByteCode::forend, loop_variable, loop_vector, loop_counter);
		emit(ByteCode::jmp, beginbody-endbody, 0, 0);

		ir[beginbody-1].a.i = endbody-beginbody+4;
        ir[beginbody].a.i = endbody-beginbody+2;

		kill(body); kill(loop_variable); kill(loop_counter); kill(loop_vector);
		return compileConstant(Null::Singleton(), code);
	} 
	else if(func == Strings::whileSym)
	{
		Operand head_condition = compile(call[1], code);
		emit(ByteCode::jc, 1, 0, kill(head_condition));
		loopDepth++;
		
		int64_t beginbody = ir.size();
        emit(ByteCode::loop, 0, 0, 0);
		Operand body = compile(call[2], code);
		kill(body);
		int64_t tail = ir.size();
		Operand tail_condition = compile(call[1], code);
		int64_t endbody = ir.size();
		
		emit(ByteCode::jc, beginbody-endbody, 1, kill(tail_condition));
		resolveLoopExits(beginbody, endbody, tail, endbody+1);
		
        ir[beginbody-1].b = endbody-beginbody+2;
        ir[beginbody].a.i = endbody-beginbody+1;
		
        loopDepth--;
		
		return compileConstant(Null::Singleton(), code);
	} 
	else if(func == Strings::repeatSym)
	{
		loopDepth++;
		int64_t beginbody = ir.size();
        emit(ByteCode::loop, 0, 0, 0);
		Operand body = compile(call[1], code);
		int64_t endbody = ir.size();
		resolveLoopExits(beginbody, endbody, endbody, endbody+1);
		loopDepth--;
		emit(ByteCode::jmp, beginbody-endbody, 0, 0);
		
        ir[beginbody].a.i = endbody-beginbody+1;
		
        kill(body);
		return compileConstant(Null::Singleton(), code);
	}
	else if(func == Strings::nextSym)
	{
		if(loopDepth == 0) throw CompileError("next used outside of loop");
		emit(ByteCode::jmp, 0, 1, 0);
		return Operand(INVALID, (int64_t)0);
	} 
	else if(func == Strings::breakSym)
	{
		if(loopDepth == 0) throw CompileError("break used outside of loop");
		emit(ByteCode::jmp, 0, 2, 0);
		return Operand(INVALID, (int64_t)0);
	} 
	else if(func == Strings::ifSym) 
	{
		Operand resultT, resultF;
		if(call.length != 3 && call.length != 4)	
			throw CompileError("invalid if statement");
		
		// resultT and resultF need to be in the same register so if statements can be used as expressions.
		// if either one is not in a register move into register. They should be in the same register at
		// the end. They could also be in the same MEMORY location at the end, but that's harder to code
		// gen for...
		
		Operand cond = compile(call[1], code);
		emit(ByteCode::jc, 1, 0, kill(cond));
		int64_t begin1 = ir.size(), begin2 = 0;
		resultT = placeInRegister(compile(call[2], code));
		
		emit(ByteCode::jmp, (int64_t)0, (int64_t)0, (int64_t)0);
		begin2 = ir.size();
	
		kill(resultT);
		resultF = placeInRegister(
			call.length == 4 ? 	compile(call[3], code) :
						compileConstant(Null::Singleton(), code) );
		assert(resultT == resultF || resultT.loc == INVALID || resultF.loc == INVALID);
		int64_t end = ir.size();
		ir[begin2-1].a = end-begin2+1;
		
		ir[begin1-1].b = begin2-begin1+1;

		return resultT.loc == INVALID ? resultF : resultT;
	}
	else if(func == Strings::lor2 && call.length == 3)
	{
		Operand r0 = placeInRegister(compileConstant(Logical::True(), code));
		
		Operand left = compile(call[1], code);
		emit(ByteCode::jc, 0, 1, kill(left));
		int64_t j1 = ir.size()-1;
		Operand right = compile(call[2], code);
		emit(ByteCode::jc, 2, 1, kill(right));
	
		kill(r0);	
		Operand r1 = placeInRegister(compileConstant(Logical::False(), code));
		ir[j1].a = (int64_t)ir.size()-j1;
		assert(r0 == r1 || r0.loc == INVALID || r1.loc == INVALID);
		return r0;
	}
	else if(func == Strings::land2 && call.length == 3)
	{
		Operand r0 = placeInRegister(compileConstant(Logical::False(), code));

		Operand left = compile(call[1], code);
		emit(ByteCode::jc, 1, 0, kill(left));
		int64_t j1 = ir.size()-1;
		Operand right = compile(call[2], code);
		emit(ByteCode::jc, 1, 2, kill(right));

		kill(r0);
		Operand r1 = placeInRegister(compileConstant(Logical::True(), code));
		ir[j1].b = (int64_t)ir.size()-j1;
		assert(r0 == r1 || r0.loc == INVALID || r1.loc == INVALID);
		return r0;
	}
	else if(func == Strings::brace) 
	{
		int64_t length = call.length;
		if(length <= 1) {
			return compileConstant(Null::Singleton(), code);
		} else {
			Operand result;
			for(int64_t i = 1; i < length; i++) {
				// memory results need to be forced to handle things like:
				// 	function(x,y) { x; y }
				// if x is a promise, it must be forced
				if(result.loc == MEMORY) {
					result = placeInRegister(result);
				}
				kill(result);
				result = compile(call[i], code);
			}
			return result;
		}
	} 
	else if(func == Strings::paren) 
	{
		return compile(call[1], code);
	}
	else if(func == Strings::list) 
	{
		Operand f, r;
		for(int64_t i = 1; i < call.length; i++) {
			Operand t = forceInRegister(compile(call[i], code));
			if(f.loc == INVALID) { f = t; r = t; }
			else { assert(t.i == r.i+1); r = t; }
		}
		if(call.length > 1)
			kill(f);
		Operand result = allocRegister();
		emit(ByteCode::list, f, Operand((int64_t)call.length-1), result);
		return result;
	}
    else if(func == Strings::get) {
		Operand b = compile(call[2], code);
		Operand a = Operand(MEMORY, SymbolStr(call[1]));
		kill(a); kill(b);
		Operand result = allocRegister();
		emit(ByteCode::get, a, b, result);
		return result;
    }
    else if(func == Strings::gassign) {
		Operand c = placeInRegister(compile(call[2], code));
		Operand b = compile(call[3], code);
		Operand a = Operand(MEMORY, SymbolStr(call[1]));
		kill(a); kill(b); kill(c);
		Operand result = allocRegister();
		assert(c == result);
		emit(ByteCode::gassign, a, b, result);
		return result;
    } 
	// Trinary operators
	else if((func == Strings::bracketAssign ||
		func == Strings::bbAssign ||
		func == Strings::split ||
		func == Strings::ifelse ||
		func == Strings::seq ||
		func == Strings::rep ||
        func == Strings::attrset) &&
		call.length == 4) {
		Operand c = placeInRegister(compile(call[1], code));
		Operand b = compile(call[2], code);
		Operand a = compile(call[3], code);
		kill(a); kill(b); kill(c);
		Operand result = allocRegister();
		assert(c == result);
		emit(op3(func), a, b, result);
		return result;
	}
	// Binary operators
	else if((func == Strings::add ||
		func == Strings::sub ||
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
		func == Strings::cm2 ||
		func == Strings::lor ||
		func == Strings::land ||
		func == Strings::bracket ||
		func == Strings::bb ||
		func == Strings::vector ||
		func == Strings::round ||
		func == Strings::signif ||
        func == Strings::attrget) &&
		call.length == 3) 
	{
		Operand a = compile(call[1], code);
		Operand b = compile(call[2], code);
		kill(b); kill(a);
		Operand result = allocRegister();
		emit(op2(func), a, b, result);
		return result;
	} 
	// Unary operators
	else if((func == Strings::add ||
		func == Strings::sub ||
		func == Strings::lnot || 
		func == Strings::abs || 
		func == Strings::sign ||
		func == Strings::sqrt ||
		func == Strings::floor ||
		func == Strings::ceiling || 
		func == Strings::trunc ||
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
		func == Strings::mean ||
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
		func == Strings::strip ||
		func == Strings::aslogical ||
		func == Strings::asinteger ||
		func == Strings::asdouble ||
		func == Strings::ascharacter ||
		func == Strings::aslist ||
		func == Strings::random ||
        func == Strings::newenv ||
        func == Strings::parentframe) &&
		call.length == 2)
	{
		// if there isn't exactly one parameter, we should call the library version...
		Operand a = compile(call[1], code);
		kill(a);
		Operand result = allocRegister();
		emit(op1(func), a, 0, result);
		return result; 
	}
	else if(func == Strings::missing)
	{
		if(call.length != 2) _error("missing requires one argument");
		if(!isSymbol(call[1]) && !call[1].isCharacter1()) _error("wrong parameter to missing");
		Operand s = Operand(MEMORY, SymbolStr(call[1]));
		Operand result = allocRegister();
		emit(ByteCode::missing, s, 0, result); 
		return result;
	} 
	else if(func == Strings::quote)
	{
		if(call.length != 2) _error("quote requires one argument");
		return compileConstant(call[1], code);
	}
    else if(func == Strings::nargs)
    {
        Operand result = allocRegister();
        emit(ByteCode::nargs, 0, 0, result);
        return result;
    }
	/*else if(func == Strings::cc) 
	{
		Type::Enum type = Type::Null;
		for(size_t i = 1; i < call.length; i++) {
			type = (Type::Enum)std::max((int)type, (int)call[i].type);
		}
		if(type == Type::Logical) {
			Logical l(call.length-1);
			for(size_t i = 1; i < call.length; i++) 
				l[i-1] = As<Logical>(thread, call[i])[0];
			return compileConstant(l, code);
		}
		if(type == Type::Integer) {
			Integer l(call.length-1);
			for(size_t i = 1; i < call.length; i++) 
				l[i-1] = As<Integer>(thread, call[i])[0];
			return compileConstant(l, code);
		}
		if(type == Type::Double) {
			Double l(call.length-1);
			for(size_t i = 1; i < call.length; i++) 
				l[i-1] = As<Double>(thread, call[i])[0];
			return compileConstant(l, code);
		}
	}*/

	return compileFunctionCall(call, names, code);
}

Compiler::Operand Compiler::compileExpression(List const& values, Prototype* code) {
	Operand result;
	if(values.length == 0) result = compileConstant(Null::Singleton(), code);
	for(int64_t i = 0; i < values.length; i++) {
		// memory results need to be forced to handle things like:
		// 	function(x,y) { x; y }
		// if x is a promise, it must be forced
		if(result.loc == MEMORY) {
			result = placeInRegister(result);
		}
		kill(result);
		result = compile(values[i], code);
	}
	return result;
}

Compiler::Operand Compiler::compile(Value const& expr, Prototype* code) {
	switch(expr.type)
	{
		case Type::Object:
			{
				Object const& o = (Object const&) expr;
				if(isSymbol(o)) {
					return compileSymbol(expr, code);
				}
				if(isExpression(o)) {
					assert(o.base().isList());
					return compileExpression((List const&)o.base(), code);
				}
				else if(isCall(o)) {
					assert(o.base().isList());
					return compileCall((List const&)o.base(), 
						hasNames(o) ? (Character const&)getNames(o) : Character(0), code);
				}
				else {
					return compileConstant(expr, code);
				}
			} break;
		default: 
			{
			return compileConstant(expr, code);
			} break;
	};
}



void Compiler::dumpCode() const {
	for(size_t i = 0; i < ir.size(); i++) {
		std::cout << ByteCode::toString(ir[i].bc) << "\t" << ir[i].a.toString() << "\t" << ir[i].b.toString() << "\t" << ir[i].c.toString() << std::endl;
	}
}

// generate actual code from IR as follows...
// 	MEMORY and INTEGER operands unchanged
//	CONSTANT operands placed in lower N registers (starting at 0)
//	REGISTER operands placed in above those
//	all register ops encoded with negative integer.
//	INVALID operands just go to 0 since they will never be used
int64_t Compiler::encodeOperand(Operand op, int64_t n) const {
	if(op.loc == MEMORY || op.loc == INTEGER) return op.i;
	else if(op.loc == CONSTANT) _error("Constants are no more");
	else if(op.loc == REGISTER) return -(op.i);
	else return 0;
}

Prototype* Compiler::compile(Value const& expr) {
	Prototype* code = new Prototype();
	assert(((int64_t)code) % 16 == 0); // our type packing assumes that this is true

	Operand result = compile(expr, code);

	code->expression = expr;
	code->registers = max_n;
	
	// insert appropriate termination statement at end of code
	if(scope == FUNCTION)
		emit(ByteCode::ret, result, 0, 0);
	else if(scope == PROMISE)
		emit(ByteCode::retp, result, 0, 0);
	else { // TOPLEVEL
		emit(ByteCode::rets, result, 0, 0);
		emit(ByteCode::done, 0, 0, 0);
	}
	int64_t n = code->constants.size();
	for(size_t i = 0; i < ir.size(); i++) {
		code->bc.push_back(Instruction(ir[i].bc, encodeOperand(ir[i].a, n), encodeOperand(ir[i].b, n), encodeOperand(ir[i].c, n)));
	}

	return code;	
}

