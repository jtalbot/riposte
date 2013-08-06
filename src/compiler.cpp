
#include "compiler.h"
#include "runtime.h"

static ByteCode::Enum op0(String const& func) {

    if(func == Strings::dots) return ByteCode::dotsc;
    if(func == Strings::env_global) return ByteCode::env_global;
    if(func == Strings::stop) return ByteCode::stop;
    throw RuntimeError(std::string("unexpected symbol '") + func + "' used as a nullary operator"); 
}

static ByteCode::Enum op1(String const& func) {
	if(func == Strings::add) return ByteCode::pos; 
	if(func == Strings::sub) return ByteCode::neg; 
	
    if(func == Strings::isna) return ByteCode::isna; 
	
	if(func == Strings::lnot) return ByteCode::lnot; 
	
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
	
	if(func == Strings::seqlen) return ByteCode::seq;
	
    if(func == Strings::type) return ByteCode::type; 
	if(func == Strings::strip) return ByteCode::strip; 
    if(func == Strings::length) return ByteCode::length;
	
	if(func == Strings::random) return ByteCode::random; 
	
    if(func == Strings::dots) return ByteCode::dotsv;

    if(func == Strings::attributes) return ByteCode::attributes;
    if(func == Strings::getenv) return ByteCode::getenv;
    if(func == Strings::env_new) return ByteCode::env_new;
    if(func == Strings::env_names) return ByteCode::env_names;
    if(func == Strings::frame) return ByteCode::frame;
    
    if(func == Strings::invisible) return ByteCode::invisible;
    if(func == Strings::visible) return ByteCode::visible;
    
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

	if(func == Strings::bracket) return ByteCode::getsub;
	if(func == Strings::bb) return ByteCode::get;

	if(func == Strings::vector) return ByteCode::vector;

    if(func == Strings::attrget) return ByteCode::getattr;
    if(func == Strings::setenv) return ByteCode::setenv;
	
    if(func == Strings::env_exists) return ByteCode::env_exists;
	if(func == Strings::env_remove) return ByteCode::env_remove;

    if(func == Strings::semijoin) return ByteCode::semijoin;
	
    throw RuntimeError(std::string("unexpected symbol '") + func + "' used as a binary operator"); 
}

static ByteCode::Enum op3(String const& func) {
	if(func == Strings::bracketAssign) return ByteCode::setsub;
	if(func == Strings::bbAssign) return ByteCode::set;
	if(func == Strings::split) return ByteCode::split;
	if(func == Strings::ifelse) return ByteCode::ifelse;
	if(func == Strings::index) return ByteCode::index;
	if(func == Strings::attrset) return ByteCode::setattr;
	if(func == Strings::map) return ByteCode::map;
	if(func == Strings::scan) return ByteCode::scan;
	if(func == Strings::fold) return ByteCode::fold;
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

Compiler::Operand Compiler::compileConstant(Value const& expr, Code* code) {
	std::map<Value, int64_t>::const_iterator i = constants.find(expr);
	int64_t index = 0;
	if(i == constants.end()) {
		index = code->constants.size();
		code->constants.push_back(expr);
		constants[expr] = index;
	} else {
		index = i->second;
	}
	return Operand(CONSTANT, index);
}

static int64_t isDotDot(String s) {
	if(s != 0 && s[0] == '.' && s[1] == '.') {

        // catch ...
        if(s[2] == '.' && s[3] == 0)
            return 0;

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

Compiler::Operand Compiler::compileSymbol(Value const& symbol, Code* code, bool isClosure) {
	String s = SymbolStr(symbol);
	
	int64_t dd = isDotDot(s);
	if(dd > 0) {
        Operand idx = compileConstant(Integer::c(dd), code); 
		Operand t = allocRegister();
		emit(ByteCode::dotsv, idx, 0, t);
		return t;
	}
	else {
		Operand sym = compileConstant(Character::c(s), code);
		Operand t = allocRegister();
		emit(isClosure ? ByteCode::loadfn : ByteCode::load, sym, 0, t);
		return t;
	}
}

Compiler::Operand Compiler::placeInRegister(Operand r) {
	if(r.loc != REGISTER && r.loc != INVALID) {
		kill(r);
		Operand t = allocRegister();
		emit(ByteCode::mov, r, 0, t);
		return t;
	}
	return r;
}

CompiledCall Compiler::makeCall(List const& call, Character const& names) {
	// compute compiled call...precompiles promise code and some necessary values
    List rcall = call;
    int64_t dotIndex = call.length()-1;
	PairList arguments, extraArgs;
	for(int64_t i = 1; i < call.length(); i++) {
		Pair p;
		if(names.length() > 0) p.n = names[i]; else p.n = Strings::empty;

        if(p.n == Strings::__extraArgs__) {
            List const& l = (List const&)call[i];
            Character const& n = hasNames(l) ? 
				(Character const&)getNames((List const&)l) : 
				Character(0);
            
            for(size_t j = 0; j < l.length(); ++j) {
                if(n.length() > j && n[j] != Strings::empty) {
                    Pair q;
                    q.n = n[j];
                    q.v = l[j];
                    extraArgs.push_back(q);
                }
            }
            // also need to remove the extraArgs from the original call
            rcall = List(call.length()-1);
            for(size_t j=0, k=0; j < call.length(); ++j)
                if(j != i)
                    rcall[k++] = call[j];
            if(names.length() > 0 ) {
                Character rnames = Character(names.length()-1);
                for(size_t j=0, k=0; j < names.length(); ++j)
                    if(j != i)
                        rnames[k++] = names[j];
                rcall = CreateNamedList(rcall, rnames);
            }
        }
        else {
		    if(isSymbol(call[i]) && SymbolStr(call[i]) == Strings::dots) {
			    p.v = call[i];
    			dotIndex = i-1;
	    	} else if(isCall(call[i]) || isSymbol(call[i])) {
		    	Promise::Init(p.v, NULL, Compiler::compilePromise(thread, call[i]), false);
    		} else {
	    		p.v = call[i];
		    }
    		arguments.push_back(p);
	    }
    }
	return CompiledCall(rcall, arguments, dotIndex, names.length() > 0, extraArgs);
}

// a standard call, not an op
Compiler::Operand Compiler::compileFunctionCall(Operand function, List const& call, Character const& names, Code* code) {
	CompiledCall a = makeCall(call, names);
	code->calls.push_back(a);
	kill(function);
	Operand result = allocRegister();
	if(!a.named 
      && a.dotIndex >= (int64_t)a.arguments.size() 
      && a.extraArgs.size() == 0)
		emit(ByteCode::fastcall, function, code->calls.size()-1, result);
	else
		emit(ByteCode::call, function, code->calls.size()-1, result);
	return result;
}

Compiler::Operand Compiler::compileExternalFunctionCall(List const& call, Code* code) {
	String func = SymbolStr(call[0]);
	
	// compile parameters directly...reserve registers for them.
	Operand liveIn = top();
	int64_t reg = liveIn.i-1;
	for(int64_t i = 1; i < call.length(); i++) {
		Operand r = placeInRegister(compile(call[i], code));
		assert(r.i == reg+1);
		reg = r.i; 
	}
	// kill all parameters
	kill(reg); 	   // only necessary to silence unused warning
	kill(liveIn); 
	Operand result = allocRegister();
	emit(ByteCode::external, Operand(MEMORY, func), call.length()-1, result);
	return result;
}

Compiler::Operand Compiler::compileCall(List const& call, Character const& names, Code* code) {

	int64_t length = call.length();

	if(length == 0) {
		throw CompileError("invalid empty call");
	}

	if(!isSymbol(call[0]) && !call[0].isCharacter1())
		return compileFunctionCall(compile(call[0], code), call, names, code);

	String func = SymbolStr(call[0]);
	// list and missing are the only built in function that 
    // handles ... or named parameters
	// we only handle the list(...) case through an op for now
	if(func == Strings::list && call.length() == 2 
		&& isSymbol(call[1]) && SymbolStr(call[1]) == Strings::dots)
	{
		Operand result = allocRegister();
		Operand counter = placeInRegister(compile(Integer::c(0), code));
		Operand storage = allocRegister();
		kill(storage); kill(counter);
		emit(ByteCode::dots, counter, storage, result); 
		return result;
	}
	else if(func == Strings::missing)
	{
		if(call.length() != 2) _error("missing requires one argument");

        Operand s;
        // lots of special cases to handle
        if(call[1].isCharacter() && ((Character const&)call[1]).length() == 1) {
            int64_t dd = isDotDot(call[1].s);
            s = dd > 0
                ? compileConstant(Integer::c(dd), code)
                : dd == 0
                    ? compileConstant(Null::Singleton(), code)
                    : compileConstant(call[1], code);
        }
        else if(isCall(call[1]) 
            && ((List const&)call[1]).length() == 2
            && ((List const&)call[1])[0].s == Strings::dots) {
            s = compile(((List const&)call[1])[1], code);
        }
		else _error("wrong parameter to missing");

		Operand result = allocRegister();
		emit(ByteCode::missing, s, 0, result); 
		return result;
	}

	// These functions can't be called directly if the arguments are named or if
	// there is a ... in the args list

	bool complicated = false;
	for(int64_t i = 1; i < length; i++) {
		if(isSymbol(call[i]) && SymbolStr(call[i]) == Strings::dots) 
			complicated = true;
	}

	if(complicated)
		return compileFunctionCall(compile(call[0], code), call, names, code);

	// switch statement supports named args
	if(func == Strings::switchSym)
	{
		if(call.length() == 0) _error("'EXPR' is missing");
		Operand c = compile(call[1], code);
		int64_t n = call.length()-2;

		int64_t branch = emit(ByteCode::branch, kill(c), n, 0);
		for(int64_t i = 2; i < call.length(); i++) {
			Character ni = Character::c(names.length() > i ? names[i] : Strings::empty);
			emit(ByteCode::branch, compileConstant(ni, code), 0, 0);
		}
		
		std::vector<int64_t> jmps;
		Operand result = placeInRegister(compileConstant(Null::Singleton(), code));
		jmps.push_back(emit(ByteCode::jmp, (int64_t)0, (int64_t)0, (int64_t)0));
		
		for(int64_t i = 1; i <= n; i++) {
			ir[branch+i].c = (int64_t)ir.size()-branch;
			if(!call[i+1].isNil()) {
				kill(result);
				Operand r = placeInRegister(compile(call[i+1], code));
				if(r.loc != INVALID && r != result)
					throw CompileError(std::string("switch statement doesn't put all its results in the same register"));
                if(i < n)
					jmps.push_back(emit(ByteCode::jmp, (int64_t)0, (int64_t)0, (int64_t)0));
			} else if(i == n) {
				kill(result);
				Operand r = placeInRegister(compileConstant(Null::Singleton(), code));
				if(r.loc != INVALID && r != result) 
					throw CompileError(std::string("switch statement doesn't put all its results in the same register"));
			}
		}
		for(int64_t i = 0; i < (int64_t)jmps.size(); i++) {
			ir[jmps[i]].a = (int64_t)ir.size()-jmps[i];
			ir[jmps[i]].a = (int64_t)ir.size()-jmps[i];
		}
		return result;
	}

	for(int64_t i = 0; i < length; i++) {
		if(names.length() > i && names[i] != Strings::empty) 
			complicated = true;
	}
	
	//if(complicated)
	//	return compileFunctionCall(call, names, code);

    if(func == Strings::external)
    {
		if(!call[1].isList() || !isCall(call[1]))
			throw CompileError(std::string(".External has invalid arguments (") + Type::toString(call[1].type()) + ")");
		return compileExternalFunctionCall((List const&)call[1], code);
    }
	else if(func == Strings::assign ||
		func == Strings::eqassign || 
		func == Strings::assign2)
	{
		Value dest = call[1];
		
        Operand rhs = compile(call[2], code);
      
        // Handle simple assignment 
        if(!isCall(dest)) {
            Operand target = compileConstant(Character::c(SymbolStr(dest)), code);
		    emit(func == Strings::assign2 ? ByteCode::storeup : ByteCode::store, 
                target, 0, rhs);
        }
		
        // Handle complex LHS assignment instructions...
		// semantics here are kind of tricky. Consider compiling:
		//	 dim(a)[1] <- x
		// This is progressively turned `inside out`:
		//	1) dim(a) <- `[<-`(dim(a), 1, x)
		//	2) a <- `dim<-`(a, `[<-`(dim(a), 1, x))
        else {    
            Operand tmp = compileConstant(Character::c(Strings::assignTmp), code);
            emit( ByteCode::store, tmp, 0, rhs );

            Value value = CreateSymbol(Strings::assignTmp);
		    while(isCall(dest)) {
			    List const& c = (List const&)dest;
			    if(c.length() < 2L)
                    _error("invalid left side of assignment");

                List n(c.length()+1);

			    for(int64_t i = 0; i < c.length(); i++) { n[i] = c[i]; }
			    String as = state.internStr(state.externStr(SymbolStr(c[0])) + "<-");
			    n[0] = CreateSymbol(as);
			    n[c.length()] = value;

			    Character nnames(c.length()+1);
	
			    if(!hasNames(c) && (as == Strings::bracketAssign || as == Strings::bbAssign) && c.length() == 3) {
				    for(int64_t i = 0; i < c.length()+1; i++) { nnames[i] = Strings::empty; }
			    }
			    else {
				    if(hasNames(c)) {
					    Value names = getNames(c);
					    for(int64_t i = 0; i < c.length(); i++) { nnames[i] = ((Character const&)names)[i]; }
				    } else {
					    for(int64_t i = 0; i < c.length(); i++) { nnames[i] = Strings::empty; }
				    }
				    nnames[c.length()] = Strings::value;
			    }
			    value = CreateCall(n, nnames);
			    dest = c[1];
		    }

            Operand target = compileConstant(Character::c(SymbolStr(dest)), code);
		    Operand source = compile(value, code);
		    emit(func == Strings::assign2 ? ByteCode::storeup : ByteCode::store, 
                target, 0, source);
            kill( source );
		    
            Operand rm = allocRegister();
            emit( ByteCode::rm, tmp, 0, rm );
            kill( rm );
        }
		return rhs;
	}
    else if(func == Strings::rm 
        && call.length() == 2 
        && (isSymbol(call[1]) || call[1].isCharacter1()))
    {
        Operand symbol = compileConstant(Character::c(SymbolStr(call[1])), code);
		Operand rm = allocRegister();
        emit( ByteCode::rm, symbol, 0, rm );
        return rm;
    }
    else if(func == Strings::as && call.length() == 3 && call[2].isCharacter1())
    {
	    Operand src = compile(call[1], code);
        Operand type = compileConstant(call[2], code);
        kill( src );
		Operand as = allocRegister();
        emit( ByteCode::as, src, type, as );
        return as;
    }
	else if(func == Strings::function) 
	{
		//compile the default parameters
		assert(call[1].isList());
		List const& c = (List const&)call[1];
		Character names = hasNames(c) ? 
			(Character const&)getNames(c) : Character(0);
		
		PairList parameters;
		for(int64_t i = 0; i < c.length(); i++) {
			Pair p;
			if(names.length() > 0) p.n = names[i]; else p.n = Strings::empty;
			if(!c[i].isNil()) {
				Promise::Init(p.v, NULL, compilePromise(thread, c[i]), true);
			}
			else {
				p.v = c[i];
			}
			parameters.push_back(p);
		}

		//compile the source for the body
		Prototype* functionCode = Compiler::compileClosureBody(thread, call[2]);

		// Populate function info
        functionCode->formals = c;
		functionCode->parameters = parameters;
		functionCode->parametersSize = parameters.size();
		functionCode->string = SymbolStr(call[3]);
		functionCode->dotIndex = parameters.size();
		for(int64_t i = 0; i < (int64_t)parameters.size(); i++) 
			if(parameters[i].n == Strings::dots) functionCode->dotIndex = i;

		Value function;
		Closure::Init(function, functionCode, 0);
		Operand funcOp = compileConstant(function, code);
	
		Operand reg = allocRegister();	
		emit(ByteCode::fn_new, funcOp, 0, reg);
		return reg;
	} 
	else if(func == Strings::returnSym)
	{
		Operand result;
		if(call.length() == 1) {
			result = compileConstant(Null::Singleton(), code);
		} else if(call.length() == 2)
			result = placeInRegister(compile(call[1], code));
		else
			throw CompileError("Too many parameters to return. Wouldn't multiple return values be nice?\n");
		emit(ByteCode::ret, result, 0, 0);
		return result;
	} 
	else if(func == Strings::forSym) 
	{
		Operand loop_variable = Operand(MEMORY, SymbolStr(call[1]));
		Operand loop_vector = placeInRegister(compile(call[2], code));
		Operand loop_counter = allocRegister();	// save space for loop counter
		Operand loop_limit = allocRegister(); // save space for the loop limit

		emit(ByteCode::forbegin, loop_variable, loop_vector, loop_counter);
		emit(ByteCode::jmp, 0, 0, 0);
		
		loopDepth++;
		int64_t beginbody = ir.size();
		Operand body = compile(call[3], code);
		int64_t endbody = ir.size();
		resolveLoopExits(beginbody, endbody, endbody, endbody+2);
		loopDepth--;
		
		emit(ByteCode::forend, loop_variable, loop_vector, loop_counter);
		emit(ByteCode::jmp, beginbody-endbody, 0, 0);

		ir[beginbody-1].a.i = endbody-beginbody+4;

		kill(body); kill(loop_limit); kill(loop_variable); 
		kill(loop_counter); kill(loop_vector);
		return compileConstant(Null::Singleton(), code);
	} 
	else if(func == Strings::whileSym)
	{
		Operand head_condition = compile(call[1], code);
		emit(ByteCode::jc, 2, 0, kill(head_condition));
        emit(ByteCode::stop, (int64_t)0, (int64_t)0, (int64_t)0);
		loopDepth++;
		
		int64_t beginbody = ir.size();
		Operand body = compile(call[2], code);
		kill(body);
		int64_t tail = ir.size();
		Operand tail_condition = compile(call[1], code);
		int64_t endbody = ir.size();
		
		emit(ByteCode::jc, beginbody-endbody, 2, kill(tail_condition));
        emit(ByteCode::stop, (int64_t)0, (int64_t)0, (int64_t)0);
		resolveLoopExits(beginbody, endbody, tail, endbody+2);
		ir[beginbody-2].b = endbody-beginbody+4;
		loopDepth--;
		
		return compileConstant(Null::Singleton(), code);
	} 
	else if(func == Strings::repeatSym)
	{
		loopDepth++;
		int64_t beginbody = ir.size();
		Operand body = compile(call[1], code);
		int64_t endbody = ir.size();
		resolveLoopExits(beginbody, endbody, endbody, endbody+1);
		loopDepth--;
		emit(ByteCode::jmp, beginbody-endbody, 0, 0);
		
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
		Operand resultT, resultF, resultNA;
		if(call.length() < 3 || call.length() > 5)
			throw CompileError("invalid if statement");
		
		Operand cond = compile(call[1], code);
		emit(ByteCode::jc, 0, 0, kill(cond));
		
        int64_t begin1 = ir.size();
        if(call.length() == 5) {
            resultNA = placeInRegister(compile(call[4], code));
		    emit(ByteCode::jmp, (int64_t)0, (int64_t)0, (int64_t)0);
		}
        else {
            resultNA = Operand();
            emit(ByteCode::stop, (int64_t)0, (int64_t)0, (int64_t)0);
        }
        kill(resultNA);

		int64_t begin2 = ir.size();
        resultF = placeInRegister(
			call.length() >= 4 
                ? compile(call[3], code)
                : compileConstant(Null::Singleton(), code) );
        assert(resultNA == resultF || 
               resultNA.loc == INVALID || 
               resultF.loc == INVALID);
		emit(ByteCode::jmp, (int64_t)0, (int64_t)0, (int64_t)0);
		kill(resultF);
		
		int64_t begin3 = ir.size();
        resultT = placeInRegister(compile(call[2], code));
        assert(resultT == resultF || 
               resultT.loc == INVALID || 
               resultF.loc == INVALID);
		kill(resultT);

		int64_t end = ir.size();
		ir[begin3-1].a = end-begin3+1;
        ir[begin2-1].a = end-begin2+1;
		ir[begin1-1].a = begin3-begin1+1;
		ir[begin1-1].b = begin2-begin1+1;

		return resultT.loc != INVALID 
                    ? resultT
                    : ( resultF.loc != INVALID 
                        ? resultF 
                        : resultNA );
	}
	else if(func == Strings::brace) 
	{
		int64_t length = call.length();
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
		Operand result = compile(call[1], code);
        emit(ByteCode::visible, result, 0, result);
        return result;
	}
 
	// Trinary operators
	else if((func == Strings::bracketAssign ||
		func == Strings::bbAssign ||
		func == Strings::split ||
		func == Strings::ifelse ||
		func == Strings::index ||
        func == Strings::attrset ||
        func == Strings::map ||
        func == Strings::scan ||
        func == Strings::fold) &&
		call.length() == 4) {
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
		func == Strings::bracket ||
		func == Strings::bb ||
		func == Strings::vector ||
		func == Strings::attrget ||
        func == Strings::env_exists ||
        func == Strings::env_remove ||
        func == Strings::setenv ||
        func == Strings::semijoin) &&
		call.length() == 3) 
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
		func == Strings::isna ||
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
        func == Strings::seqlen ||
		func == Strings::type ||
		func == Strings::strip ||
		func == Strings::random ||
        func == Strings::dots ||
        func == Strings::attributes ||
        func == Strings::getenv ||
        func == Strings::env_new ||
        func == Strings::env_names ||
        func == Strings::length ||
        func == Strings::frame ||
        func == Strings::invisible ||
        func == Strings::visible)
		&& call.length() == 2)
	{
		// if there isn't exactly one parameter, we should call the library version...
		Operand a = compile(call[1], code);
		kill(a);
		Operand result = allocRegister();
		emit(op1(func), a, 0, result);
		return result; 
	} 
	// Nullary operators
	else if((func == Strings::dots ||
             func == Strings::env_global ||
             func == Strings::stop)
		&& call.length() == 1)
	{
		// if there isn't exactly zero parameters, we should call the library version...
		Operand result = allocRegister();
		emit(op0(func), 0, 0, result);
		return result; 
	}
    else if(func == Strings::promise)
    {
		Operand a = compile(call[1], code);
		Operand b = compile(call[2], code);
		Operand c = compile(call[3], code);
		Operand d = compile(call[4], code);
		kill(d); kill(c); kill(b); kill(a);
		Operand result = allocRegister();
		emit(ByteCode::pr_new, a, b, result);
		emit(ByteCode::pr_new, c, d, result);
		return result;
    }
	else if(func == Strings::pr_expr)
	{
		if(call.length() != 3) _error("pr_expr requires two arguments");
		Operand a = compile(call[1], code);
		Operand b = compile(call[2], code);
        kill(b); kill(a);
		Operand result = allocRegister();
		emit(ByteCode::pr_expr, a, b, result); 
		return result;
	}
	else if(func == Strings::pr_env)
	{
		if(call.length() != 3) _error("pr_env requires two arguments");
		Operand a = compile(call[1], code);
		Operand b = compile(call[2], code);
        kill(b); kill(a);
		Operand result = allocRegister();
		emit(ByteCode::pr_env, a, b, result); 
		return result;
	}

    // Otherwise, generate standard function call...
	return compileFunctionCall(compileSymbol(call[0], code, true), call, names, code);
}

Compiler::Operand Compiler::compileExpression(List const& values, Code* code) {
	Operand result;
	if(values.length() == 0) result = compileConstant(Null::Singleton(), code);
	for(int64_t i = 0; i < values.length(); i++) {
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

Compiler::Operand Compiler::compile(Value const& expr, Code* code) {
	if(isSymbol(expr)) {
		return compileSymbol(expr, code, false);
	}
	if(isExpression(expr)) {
		assert(expr.isList());
		return compileExpression((List const&)expr, code);
	}
	else if(isCall(expr)) {
		assert(expr.isList());
		return compileCall((List const&)expr, 
			hasNames((List const&)expr) ? 
				(Character const&)getNames((List const&)expr) : 
				Character(0), 
			code);
	}
	else {
		return compileConstant(expr, code);
	}
}



void Compiler::dumpCode() const {
	for(size_t i = 0; i < ir.size(); i++) {
		std::cout << ByteCode::toString(ir[i].bc) << "\t" << ir[i].a.toString() << "\t" << ir[i].b.toString() << "\t" << ir[i].c.toString() << std::endl;
	}
}

// generate actual code from IR as follows...
// 	MEMORY and INTEGER operands unchanged
//	CONSTANT operands encoded with negative integers
//	REGISTER operands encoded with non-negative integers
//	INVALID operands just go to 0 since they will never be used
int64_t Compiler::encodeOperand(Operand op) const {
	if(op.loc == MEMORY || op.loc == INTEGER) return op.i;
	else if(op.loc == CONSTANT) return -(op.i+1);
	else if(op.loc == REGISTER) return (op.i);
	else return 0;
}

Code* Compiler::compile(Value const& expr) {
	Code* code = new Code();
	assert(((int64_t)code) % 16 == 0); // our type packing assumes that this is true

    if(scope == PROMISE) {
        // promises use first two registers to pass environment info
        // for replacing promise with evaluated value
        allocRegister();
        allocRegister();
    }

    Operand result = compile(expr, code);

    // insert appropriate termination statement at end of code
    if(scope == CLOSURE)
        emit(ByteCode::ret, result, 0, 0);
    else if(scope == PROMISE)
        emit(ByteCode::retp, result, 0, 0);
    else { // TOPLEVEL
        result = placeInRegister(result);
        if(result.i != 0)
            _error("Top level expression did not place its result in register 0");
        emit(ByteCode::done, result, 0, 0);
    }
    
	code->expression = expr;
	code->registers = max_n;
	
	for(size_t i = 0; i < ir.size(); i++) {
		code->bc.push_back(Instruction(ir[i].bc, 
            encodeOperand(ir[i].a),
            encodeOperand(ir[i].b),
            encodeOperand(ir[i].c)));
	}

	return code;	
}

