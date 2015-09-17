
#include "compiler.h"
#include "runtime.h"

Compiler::EmitTable::EmitTable() {

    // R control flow
    add(Strings::function, -1, &Compiler::emitFunction); 
    add(Strings::returnSym, -1, &Compiler::emitReturn);
    add(Strings::ifSym, -1, &Compiler::emitIf); 
    
    add(Strings::forSym, -1, &Compiler::emitFor); 
    add(Strings::whileSym, -1, &Compiler::emitWhile); 
    add(Strings::repeatSym, -1, &Compiler::emitRepeat); 
    add(Strings::nextSym, -1, &Compiler::emitNext); 
    add(Strings::breakSym, -1, &Compiler::emitBreak); 
    
    add(Strings::brace, -1, &Compiler::emitBrace); 
    add(Strings::paren, -1, &Compiler::emitParen); 

    // R assignment operators
    add(Strings::assign,   -1, &Compiler::emitAssign,   ByteCode::store); 
    add(Strings::eqassign, -1, &Compiler::emitAssign,   ByteCode::store); 
    add(Strings::assign2,  -1, &Compiler::emitAssign,   ByteCode::storeup);

    // R arithmetic operators
    add(Strings::add, 1, &Compiler::emitUnary, ByteCode::pos);
    add(Strings::add, 2, &Compiler::emitBinary, ByteCode::add);
    add(Strings::sub, 1, &Compiler::emitUnary, ByteCode::neg);
    add(Strings::sub, 2, &Compiler::emitBinary, ByteCode::sub);
    add(Strings::mul, 2, &Compiler::emitBinary, ByteCode::mul);
    add(Strings::div, 2, &Compiler::emitBinary, ByteCode::div);
    add(Strings::idiv, 2, &Compiler::emitBinary, ByteCode::idiv);
    add(Strings::mod, 2, &Compiler::emitBinary, ByteCode::mod);
    add(Strings::pow, 2, &Compiler::emitBinary, ByteCode::pow);

    add(Strings::sum, 1, &Compiler::emitUnary, ByteCode::sum);
    add(Strings::prod, 1, &Compiler::emitUnary, ByteCode::prod);

    add(Strings::cumsum, 1, &Compiler::emitUnary, ByteCode::cumsum);
    add(Strings::cumprod, 1, &Compiler::emitUnary, ByteCode::cumprod);
    
    // R ordinal operators
    add(Strings::lt, 2, &Compiler::emitBinary, ByteCode::lt);
    add(Strings::gt, 2, &Compiler::emitBinary, ByteCode::gt);
    add(Strings::eq, 2, &Compiler::emitBinary, ByteCode::eq);
    add(Strings::neq, 2, &Compiler::emitBinary, ByteCode::neq);
    add(Strings::ge, 2, &Compiler::emitBinary, ByteCode::ge);
    add(Strings::le, 2, &Compiler::emitBinary, ByteCode::le);

    add(Strings::pmin, 2, &Compiler::emitBinary, ByteCode::pmin);
    add(Strings::pmax, 2, &Compiler::emitBinary, ByteCode::pmax);
    
    add(Strings::min, 1, &Compiler::emitUnary, ByteCode::min);
    add(Strings::max, 1, &Compiler::emitUnary, ByteCode::max);
    
    add(Strings::cummin, 1, &Compiler::emitUnary, ByteCode::cummin);
    add(Strings::cummax, 1, &Compiler::emitUnary, ByteCode::cummax);
    
    // R logical operators
    add(Strings::lor, 2, &Compiler::emitBinary, ByteCode::lor);
    add(Strings::land, 2, &Compiler::emitBinary, ByteCode::land);
    add(Strings::lnot, 1, &Compiler::emitUnary, ByteCode::lnot);
    add(Strings::isna, 1, &Compiler::emitUnary, ByteCode::isna);
    
    add(Strings::any, 1, &Compiler::emitUnary, ByteCode::any);
    add(Strings::all, 1, &Compiler::emitUnary, ByteCode::all);

    // R character operators
    add(Strings::nchar, 1, &Compiler::emitUnary, ByteCode::nchar);
    add(Strings::pconcat, 2, &Compiler::emitBinary, ByteCode::pconcat);

    // Object operators
    add(Strings::bracket, 2, &Compiler::emitBinary, ByteCode::getsub);
    add(Strings::bracketAssign, 3, &Compiler::emitTernary, ByteCode::setsub);
    
    add(Strings::bb, 2, &Compiler::emitBinary, ByteCode::get);
    add(Strings::bbAssign, 3, &Compiler::emitTernary, ByteCode::set);

    add(Strings::attrget, 2, &Compiler::emitBinary, ByteCode::getattr);
    add(Strings::attrset, 3, &Compiler::emitTernary, ByteCode::setattr);
    add(Strings::attributes, 1, &Compiler::emitUnary, ByteCode::attributes);
    
    add(Strings::getenv, 1, &Compiler::emitUnary, ByteCode::getenv);
    add(Strings::setenv, 2, &Compiler::emitBinary, ByteCode::setenv);
    
    add(Strings::as, 2, &Compiler::emitBinary, ByteCode::as);
    add(Strings::type, 1, &Compiler::emitUnary, ByteCode::type);
    add(Strings::strip, 1, &Compiler::emitUnary, ByteCode::strip);
    add(Strings::length, 1, &Compiler::emitUnary, ByteCode::length);
    
    add(Strings::invisible, 1, &Compiler::emitUnary, ByteCode::invisible);
    add(Strings::visible, 1, &Compiler::emitUnary, ByteCode::visible);
    add(Strings::withVisible, 1, &Compiler::emitUnary, ByteCode::withVisible);
    
    add(Strings::id, 2, &Compiler::emitBinary, ByteCode::id);
    add(Strings::nid, 2, &Compiler::emitBinary, ByteCode::nid);
    add(Strings::isnil, 1, &Compiler::emitUnary, ByteCode::isnil);
    
    // Riposte-specific operators
    add(Strings::external, -1, &Compiler::emitExternal);

    // Promise operators
    add(Strings::promise,  -1, &Compiler::emitPromise); 
    add(Strings::pr_expr, 2, &Compiler::emitBinary, ByteCode::pr_expr);
    add(Strings::pr_env, 2, &Compiler::emitBinary, ByteCode::pr_env);

    // Environment operators
    add(Strings::env_has, 2, &Compiler::emitBinary, ByteCode::env_has);
    add(Strings::env_rm, 2, &Compiler::emitBinary, ByteCode::env_rm);
    add(Strings::env_missing, 2, &Compiler::emitBinary, ByteCode::env_missing);
    add(Strings::env_new, 1, &Compiler::emitUnary, ByteCode::env_new);
    add(Strings::env_names, 1, &Compiler::emitUnary, ByteCode::env_names);
    add(Strings::env_global, 0, &Compiler::emitNullary, ByteCode::env_global);
    
    add(Strings::dots, 1, &Compiler::emitUnary, ByteCode::dotsv);
    add(Strings::dots, 0, &Compiler::emitNullary, ByteCode::dotsc);

    //   Vector ops 
    add(Strings::vector, 2, &Compiler::emitBinary, ByteCode::vector);
    add(Strings::seqlen, 1, &Compiler::emitUnary, ByteCode::seq);
    add(Strings::random, 1, &Compiler::emitUnary, ByteCode::random);
    
    add(Strings::ifelse, 3, &Compiler::emitTernary, ByteCode::ifelse);
    add(Strings::split, 3, &Compiler::emitTernary, ByteCode::split);
    add(Strings::index, 3, &Compiler::emitTernary, ByteCode::index);
   
    add(Strings::map, 2, &Compiler::emitBinaryMap, ByteCode::map);
    add(Strings::map, 3, &Compiler::emitTernary, ByteCode::map);
    add(Strings::scan, 3, &Compiler::emitTernary, ByteCode::scan);
    add(Strings::fold, 3, &Compiler::emitTernary, ByteCode::fold);

    add(Strings::semijoin, 2, &Compiler::emitBinary, ByteCode::semijoin);

    // Stack frame access
    add(Strings::frame, 1, &Compiler::emitUnary, ByteCode::frame);
    
    // Error handling
    add(Strings::stop, 0, &Compiler::emitNullary, ByteCode::stop);
}

// generate byte code arguments from operands as follows...
// 	INTEGER operands unchanged
//	CONSTANT operands encoded with negative integers
//	REGISTER operands encoded with non-negative integers
//	INVALID operands just go to 0 since they will never be used
int64_t Compiler::encodeOperand(Operand op) const {
	if(op.loc == INTEGER) return op.i;
	else if(op.loc == CONSTANT) return -(op.i+1);
	else if(op.loc == REGISTER) return (op.i);
	else return 0;
}

int64_t Compiler::emit(ByteCode::Enum bc, Operand a, Operand b, Operand c) {
	ir.push_back(Instruction(bc,
        encodeOperand(a), encodeOperand(b), encodeOperand(c)));
	return ir.size()-1;
}

Compiler::Operand Compiler::invisible(Operand op) {
    kill(op);
    Operand r = allocRegister();
    emit(ByteCode::invisible, op, 0, r);
    return r;
}

Compiler::Operand Compiler::visible(Operand op) {
    kill(op);
    Operand r = allocRegister();
    emit(ByteCode::visible, op, 0, r);
    return r;
}

void Compiler::resolveLoopExits(int64_t start, int64_t end, int64_t nextTarget, int64_t breakTarget) {
	for(int64_t i = start; i < end; i++) {
		if(ir[i].bc == ByteCode::jmp && ir[i].a == 0) {
			if(ir[i].b == 1) {
				ir[i].a = encodeOperand(nextTarget-i);
			} else if(ir[i].b == 2) {
				ir[i].a = encodeOperand(breakTarget-i);
			}
		}
	}
}

Compiler::Operand Compiler::compileConstant(Value expr) {

	std::map<Value, int64_t>::const_iterator i = constants.find(expr);
	int64_t index = 0;

	if(i == constants.end()) {
		index = constantList.size();
		constantList.push_back(expr);
		constants[expr] = index;
	} else {
		index = i->second;
	}
	return Operand(CONSTANT, index);
}

static int64_t isDotDot(String s) {
	if(s != 0 && s->s[0] == '.' && s->s[1] == '.') {

        // catch ...
        if(s->s[2] == '.' && s->s[3] == 0)
            return 0;

		int64_t v = 0;
		int64_t i = 2;
		// maximum 64-bit integer has 19 digits, but really who's going to pass
		// that many var args?
		while(i < (19+2) && s->s[i] >= '0' && s->s[i] <= '9') {
			v = v*10 + (s->s[i] - '0');
			i++;
		}
		if(i < (19+2) && s->s[i] == 0) return v;
	}
	return -1;	
}

Compiler::Operand Compiler::compileSymbol(Value const& symbol, bool isClosure) {
	String s = SymbolStr(symbol);
    
    // symbols have to be interned
    s = global.strings.intern(s);
	
	int64_t dd = isDotDot(s);
	if(dd > 0) {
        Operand idx = compileConstant(Integer::c(dd)); 
		Operand t = allocRegister();
		emit(ByteCode::dotsv, idx, 0, t);
		return t;
	}
	else {
		Operand sym = compileConstant(Character::c(s));
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

// a standard call, not an op
Compiler::Operand Compiler::compileFunctionCall(Operand function, List const& call, Character const& names, Code* code) {
	CompiledCall a = makeCall(state, call, names);
	compiledCalls.push_back(a);
	kill(function);
	Operand result = allocRegister();
    emit(ByteCode::call, function, compiledCalls.size()-1, result);
	return result;
}

Compiler::Operand Compiler::emitExternal(ByteCode::Enum bc, List const& call, Code* code) {
	if(call.length() < 2)
        _error(".Riposte needs at least one argument");
    Operand func = placeInRegister(compile(call[1], code));
	
	// compile parameters directly...reserve registers for them.
	int64_t reg = func.i;
	for(int64_t i = 2; i < call.length(); i++) {
		Operand r = placeInRegister(compile(call[i], code));
		assert(r.i == reg+1);
		reg = r.i; 
	}
	// kill all parameters
	kill(reg);	   // only necessary to silence unused warning
    kill(func); 
	Operand result = allocRegister();
	emit(ByteCode::external, func, call.length()-1, result);
	return result;
}

Compiler::Operand Compiler::emitAssign(ByteCode::Enum bc, List const& call, Code* code) {
    Value dest = call[1];
    
    Operand rhs = compile(call[2], code);

    // Handle simple assignment 
    if(!isCall(dest)) {
        Operand target = compileConstant(InternStrings(state, Character::c(SymbolStr(dest))));
        emit(bc, target, 0, rhs);
    }
    
    // Handle complex LHS assignment instructions...
    // semantics here are kind of tricky. Consider compiling:
    //	 dim(a)[1] <- x
    // This is progressively turned `inside out`:
    //	1) dim(a) <- `[<-`(dim(a), 1, x)
    //	2) a <- `dim<-`(a, `[<-`(dim(a), 1, x))
    else {
        Operand tmp = compileConstant(InternStrings(state, Character::c(Strings::assignTmp)));
        emit( ByteCode::store, tmp, 0, rhs );

        Value value = CreateSymbol(global, Strings::assignTmp);
        while(isCall(dest)) {
            List const& c = (List const&)dest;
            if(c.length() < 2L)
                _error("invalid left side of assignment");

            List n(c.length()+1);

            for(int64_t i = 0; i < c.length(); i++) { n[i] = c[i]; }
            String as = MakeString(global.externStr(SymbolStr(c[0])) + "<-");
            n[0] = CreateSymbol(global, as);
            n[c.length()] = value;

            Character nnames(c.length()+1);

            // Add 'value' to argument names. Necessary for correctly
            // calling user-defined replacement functions.
            if(hasNames(c)) {
                Value names = getNames(c);
                for(int64_t i = 0; i < c.length(); i++) { 
                    nnames[i] = ((Character const&)names)[i];
                }
            } else {
                for(int64_t i = 0; i < c.length(); i++) {
                    nnames[i] = Strings::empty;
                }
            }
            nnames[nnames.length()-1] = Strings::value;
            
            value = CreateCall(global, n, nnames);
            dest = c[1];
        }

        Operand target = compileConstant(InternStrings(state, Character::c(SymbolStr(dest))));
        Operand source = compile(value, code);
        emit(bc, target, 0, source);
        kill(source);
        kill(target);

        Operand rm = allocRegister();
        Operand env = compileConstant(Null());
        emit( ByteCode::env_rm, env, tmp, rm );
        kill( rm );
    }
    return rhs;
}

Compiler::Operand Compiler::emitPromise(ByteCode::Enum bc, List const& call, Code* code) {
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

Compiler::Operand Compiler::emitFunction(ByteCode::Enum bc, List const& call, Code* code) {
    //compile the default parameters
    assert(call[1].isList() || call[1].isNull());
    List formals = call[1].isList()
            ? (List const&)call[1]
            : List::c();
    Character parameters = hasNames(formals) 
            ? (Character const&)getNames(formals)
            : Character::c();

    if(formals.length() != parameters.length()) {
        std::stringstream ss;
        ss << "Function does not have the same number of parameter names as parameter defaults: (" << parameters.length() << " != " << formals.length() << ")";
        _error(ss.str());
    }
    
    List defaults(formals.length());
    int64_t dotIndex = formals.length();
    for(int64_t i = 0; i < formals.length(); i++) {
        // Make defaults. In order to detect defaults when using missing,
        // all defaults must be put into promises.
        if(Eq(parameters[i], Strings::dots)) {
            dotIndex = i;
            defaults[i] = Value::Nil();
        }
        else if(!formals[i].isNil()) {
            Promise::Init(defaults[i], NULL, 
                deferPromiseCompilation(state, formals[i]), true);
        }
        else {
            defaults[i] = formals[i];
        }
    }

    //compile the source for the body
    Prototype* functionCode = Compiler::compileClosureBody(state, call[2]);

    // Populate function info
    functionCode->formals = formals;
    functionCode->parameters = InternStrings(state, parameters);
    functionCode->defaults = defaults;
    functionCode->string = call[3].isCharacter()
        ? SymbolStr(call[3])
        : Strings::empty;
    functionCode->dotIndex = dotIndex;

    Value function;
    Closure::Init(function, functionCode, 0);
    Operand funcOp = compileConstant(function);

    Operand reg = allocRegister();	
    emit(ByteCode::fn_new, funcOp, 0, reg);
    return reg;
}

Compiler::Operand Compiler::emitReturn(ByteCode::Enum bc, List const& call, Code* code) {
		Operand result;
		if(call.length() == 1) {
			result = compileConstant(Null());
		} else if(call.length() == 2)
			result = compile(call[1], code);
		else
			throw CompileError("Too many parameters to return. Wouldn't multiple return values be nice?\n");
		emit(ByteCode::ret, result, 0, 0);
		return result;
}

Compiler::Operand Compiler::emitFor(ByteCode::Enum bc, List const& call, Code* code) {
    Operand loop_variable =
        compileConstant(InternStrings(state, Character::c(SymbolStr(call[1]))));
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

    ir[beginbody-1].a = encodeOperand(endbody-beginbody+4);

    kill(body); kill(loop_limit); kill(loop_variable); 
    kill(loop_counter); kill(loop_vector);
    //return invisible(compileConstant(Null(), code));
    
    Operand t = allocRegister();
    emit(ByteCode::mov, body, 0, t);
    return t;
}

Compiler::Operand Compiler::emitWhile(ByteCode::Enum bc, List const& call, Code* code) {
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
    
    return invisible(compileConstant(Null()));
}

Compiler::Operand Compiler::emitRepeat(ByteCode::Enum bc, List const& call, Code* code) {
    loopDepth++;
    int64_t beginbody = ir.size();
    Operand body = compile(call[1], code);
    int64_t endbody = ir.size();
    resolveLoopExits(beginbody, endbody, endbody, endbody+1);
    loopDepth--;
    emit(ByteCode::jmp, beginbody-endbody, 0, 0);
    
    kill(body);
    return invisible(compileConstant(Null()));
}

Compiler::Operand Compiler::emitNext(ByteCode::Enum bc, List const& call, Code* code) {
    if(loopDepth == 0)
        throw CompileError("next used outside of loop");
    emit(ByteCode::jmp, 0, 1, 0);
    return Operand(INVALID, (int64_t)0);
}

Compiler::Operand Compiler::emitBreak(ByteCode::Enum bc, List const& call, Code* code) {
    if(loopDepth == 0)
        throw CompileError("break used outside of loop");
    emit(ByteCode::jmp, 0, 2, 0);
    return Operand(INVALID, (int64_t)0);
}

Compiler::Operand Compiler::emitIf(ByteCode::Enum bc, List const& call, Code* code) {
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
            : invisible(compileConstant(Null())));
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

Compiler::Operand Compiler::emitBrace(ByteCode::Enum bc, List const& call, Code* code) {
		int64_t length = call.length();
		if(length <= 1) {
			return compileConstant(Null());
		} else {
			Operand result;
			for(int64_t i = 1; i < length; i++) {
				kill(result);
				result = compile(call[i], code);
			}
			return result;
		}
}

Compiler::Operand Compiler::emitParen(ByteCode::Enum bc, List const& call, Code* code) {
    Operand result = compile(call[1], code);
    emit(ByteCode::visible, result, 0, result);
    return result;
}

Compiler::Operand Compiler::emitTernary(ByteCode::Enum bc, List const& call, Code* code) {
    Operand c = placeInRegister(compile(call[1], code));
    Operand b = compile(call[2], code);
    Operand a = compile(call[3], code);
    kill(a); kill(b); kill(c);
    Operand result = allocRegister();
    assert(c == result);
    emit(bc, a, b, result);
    return result;
}

Compiler::Operand Compiler::emitBinaryMap(ByteCode::Enum bc, List const& call, Code* code) {
    Operand c = placeInRegister(compile(call[1], code));
    Operand b = compile(call[2], code);
    Operand a = compileConstant(Value::Nil());
    kill(a); kill(b); kill(c);
    Operand result = allocRegister();
    assert(c == result);
    emit(ByteCode::map, a, b, result);
    return result;
}

Compiler::Operand Compiler::emitBinary(ByteCode::Enum bc, List const& call, Code* code) {
    Operand a = compile(call[1], code);
    Operand b = compile(call[2], code);
    kill(b); kill(a);
    Operand result = allocRegister();
    emit(bc, a, b, result);
    return result;
}
		
Compiler::Operand Compiler::emitUnary(ByteCode::Enum bc, List const& call, Code* code) {
    Operand a = compile(call[1], code);
    kill(a);
    Operand result = allocRegister();
    emit(bc, a, 0, result);
    return result;
}
 
Compiler::Operand Compiler::emitNullary(ByteCode::Enum bc, List const& call, Code* code) {
    Operand result = allocRegister();
    emit(bc, 0, 0, result);
    return result;
}



// Compute compiled call...precompiles promise code and some necessary values
struct Pair { String n; Value v; };

CompiledCall Compiler::makeCall(State& state, List const& call, Character const& names) {
    List rcall = CreateCall(state.global, call, names.length() > 0 ? names : Value::Nil());
    int64_t dotIndex = call.length()-1;
	std::vector<Pair>  arguments;
    bool named = false;

    List extraArgs(0);
    Character extraNames(0);

	for(int64_t i = 1; i < call.length(); i++) {
		Pair p;
		if(names.length() > 0) p.n = names[i]; else p.n = Strings::empty;

        if(Eq(p.n, Strings::__extraArgs__)) {
            List const& l = (List const&)call[i];
            if(!hasNames(l) 
                || l.length() != ((Character const&)getNames(l)).length()) {
                _error("__extraArgs__ must be named");
            }
            extraArgs = l;
            extraArgs.attributes(0);
            extraNames = (Character const&)getNames(l);

            // remove the extraArgs from the original call
            rcall = List(call.length()-1);
            for(size_t j=0, k=0; j < call.length(); ++j)
                if(j != i)
                    rcall[k++] = call[j];
            if(names.length() > 0 ) {
                Character rnames = Character(names.length()-1);
                for(size_t j=0, k=0; j < names.length(); ++j)
                    if(j != i)
                        rnames[k++] = names[j];
                rcall = CreateCall(state.global, rcall, rnames);
            }
        }
        else {
		    if(isSymbol(call[i]) && Eq(SymbolStr(call[i]), Strings::dots)) {
			    p.v = call[i];
    			dotIndex = i-1;
                p.n = Strings::empty;
	    	} else if(isCall(call[i]) || isSymbol(call[i])) {
                if(!Eq(p.n, Strings::empty)) named = true;
                Promise::Init(p.v, NULL, deferPromiseCompilation(state, call[i]), false);
    		} else {
                if(!Eq(p.n, Strings::empty)) named = true;
	    		p.v = call[i];
		    }
    		arguments.push_back(p);
	    }
    }

    List args(arguments.size());
    for(size_t i = 0; i < arguments.size(); ++i) {
        args[i] = arguments[i].v;
    }
    Character argnames(named ? arguments.size() : 0);
    for(size_t i = 0; i < arguments.size() && named; ++i) {
        argnames[i] = arguments[i].n;
    }

	return CompiledCall(rcall, args, InternStrings(state, argnames),
                            dotIndex, extraArgs, InternStrings(state, extraNames));
}

Compiler::Operand Compiler::compileCall(List const& call, Character const& names, Code* code) {

	int64_t length = call.length();

	if(length == 0) {
		throw CompileError("invalid empty call");
	}

    // If the function is not a literal, compile a standard call.
	if(!isSymbol(call[0]) && !call[0].isCharacter1())
		return compileFunctionCall(compile(call[0], code), call, names, code);

    // Get the function literal
	String func = SymbolStr(call[0]);

    // Compile list(...), the only built-in function that handles ...
	if(Eq(func, Strings::list) && call.length() == 2 
		&& isSymbol(call[1]) && Eq(SymbolStr(call[1]), Strings::dots))
	{
		Operand result = allocRegister();
		Operand counter = placeInRegister(compile(Integer::c(-1), code));
		Operand storage = allocRegister();
		kill(storage); kill(counter);
		emit(ByteCode::dots, counter, storage, result); 
		return result;
	}

    // The rest of the built in functions don't support ...
    // or named arguments (except for 'value' in replacement functions),
    // so check and if there are any, compile a normal function call.
	bool hasDots = false;
	for(int64_t i = 1; i < length; i++) {
		if(isSymbol(call[i]) && Eq(SymbolStr(call[i]), Strings::dots)) 
			hasDots = true;
	}

    bool hasNames = names.length() > 0;

    // Check the three built-in replacement functions for the names
    // we do support. This is necessary since our support for compiling
    // complex assignments will introduce 'value' into the argument names.
    if( ( Eq(func, Strings::bracketAssign) ||
          Eq(func, Strings::bbAssign) ||
          Eq(func, Strings::attrset) ) && names.length() == 4 ) {

        hasNames = !(Eq(names[0], Strings::empty) &&
                     Eq(names[1], Strings::empty) &&
                     Eq(names[2], Strings::empty) &&
                     Eq(names[3], Strings::value));
    }

    return (hasDots || hasNames)
        ? compileFunctionCall(compileSymbol(call[0], true), call, names, code)
        : GetEmitTable()(*this, func, call, code);
}

Compiler::Operand Compiler::compileExpression(List const& values, Code* code) {
	Operand result;
	if(values.length() == 0) result = compileConstant(Null());
	for(int64_t i = 0; i < values.length(); i++) {
		kill(result);
		result = compile(values[i], code);
	}
	return result;
}

Compiler::Operand Compiler::compile(Value const& expr, Code* code) {
	if(isSymbol(expr)) {
		return compileSymbol(expr, false);
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
		return visible(compileConstant(expr));
	}
}



Code* Compiler::compileExpression(State& state, Value const& expr) {

    Compiler compiler(state);
	Code* code = new Code();
  
    // blocks use first two registers to potentially pass
    // return information.
    (void) compiler.allocRegister();
    (void) compiler.allocRegister();

    Operand result = isExpression(expr)
        ? compiler.compileExpression((List const&)expr, code)
        : compiler.compile(expr, code);
    compiler.emit(ByteCode::done, result, 0, 0);
    
	code->registers = compiler.max_n; 
	code->expression = expr;
    {
        Integer bc(compiler.ir.size());
        for(size_t i = 0; i < compiler.ir.size(); ++i)
            bc[i] = compiler.ir[i].i;
        code->bc = bc;
    }
    {
        List con(compiler.constantList.size());
        for(size_t i = 0; i < compiler.constantList.size(); ++i)
            con[i] = compiler.constantList[i];
        code->constants = con;
    }
    {
        List calls(compiler.compiledCalls.size());
        for(size_t i = 0; i < compiler.compiledCalls.size(); ++i)
            calls[i] = compiler.compiledCalls[i];
        code->calls = calls;
    }

    return code;
}

Prototype* Compiler::compileClosureBody(State& state, Value const& expr) {
    Compiler compiler(state);
	Code* code = new Code();
    Operand result = compiler.compile(expr, code);
   
    // A function that terminates without a return implicitly
    // returns the result value.
    compiler.emit(ByteCode::ret, result, 0, 0);
    
    // interpreter assumes at least 2 registers for each function
    // one for the return value and one for an onexit result 
	code->registers = std::max(compiler.max_n, 2LL); 
	code->expression = expr;
    {
        Integer bc(compiler.ir.size());
        for(size_t i = 0; i < compiler.ir.size(); ++i)
            bc[i] = compiler.ir[i].i;
        code->bc = bc;
    }
    {
        List con(compiler.constantList.size());
        for(size_t i = 0; i < compiler.constantList.size(); ++i)
            con[i] = compiler.constantList[i];
        code->constants = con;
    }
    {
        List calls(compiler.compiledCalls.size());
        for(size_t i = 0; i < compiler.compiledCalls.size(); ++i)
            calls[i] = compiler.compiledCalls[i];
        code->calls = calls;
    }

    Prototype* p = new Prototype();
    p->code = code;
    
    return p;
}

Code* Compiler::deferPromiseCompilation(State& state, Value const& expr) {
    Code* code = new Code();
    code->expression = expr;
    code->bc = Integer(0);
    return code;
}

void Compiler::compilePromise(State& state, Code* code)
{
    Compiler compiler(state);

    // promises use first two registers to pass environment info
    // for replacing promise with evaluated value
    (void) compiler.allocRegister();
    (void) compiler.allocRegister();

    Operand result = compiler.compile(code->expression, code);
    compiler.emit(ByteCode::done, result, 0, 0);

    code->registers = compiler.max_n;
    {
        Integer bc(compiler.ir.size());
        for(size_t i = 0; i < compiler.ir.size(); ++i)
            bc[i] = compiler.ir[i].i;
        code->bc = bc;
    }
    {
        List con(compiler.constantList.size());
        for(size_t i = 0; i < compiler.constantList.size(); ++i)
            con[i] = compiler.constantList[i];
        code->constants = con;
    }
    {
        List calls(compiler.compiledCalls.size());
        for(size_t i = 0; i < compiler.compiledCalls.size(); ++i)
            calls[i] = compiler.compiledCalls[i];
        code->calls = calls;
    }

    code->writeBarrier();
}

