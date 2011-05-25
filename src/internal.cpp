
#include "internal.h"
#include "compiler.h"
#include <math.h>

void checkNumArgs(List const& args, uint64_t nargs) {
	if(args.length() > nargs) _error("unused argument(s)");
	else if(args.length() < nargs) _error("too few arguments");
}

uint64_t function(State& state, Call const& call, List const& args) {
	Value parameters = force(state, args[0]);
	Value body = args[1];
	state.registers[0] = 	
		Function(parameters, body, Character::NA()/*force(state, args[2])*/, state.env);
	return 1;
}

uint64_t rm(State& state, Call const& call, List const& args) {
	for(uint64_t i = 0; i < args.length(); i++) 
		if(expression(args[i]).type != Type::R_symbol && expression(args[i]).type != Type::R_character) 
			_error("rm() arguments must be symbols or character vectors");
	for(uint64_t i = 0; i < args.length(); i++) {
		state.env->rm(expression(args[i]));
	}
	state.registers[0] = Null::singleton;
	return 1;
}

uint64_t sequence(State& state, Call const& call, List const& args) {
	checkNumArgs(args, 3);

	Value from = force(state, args[0]);
	Value by   = force(state, args[1]);
	Value len  = force(state, args[2]);

	double f = asReal1(from);
	double b = asReal1(by);
	double l = asReal1(len);

	state.registers[0] = Sequence(f, b, l);	
	return 1;
}

uint64_t repeat(State& state, Call const& call, List const& args) {
	checkNumArgs(args, 3);
	Value from = force(state, args[0]);
	assert(args.length() == 3);
	
	Value vec  = force(state, args[0]);
	Value each = force(state, args[1]);
	Value len  = force(state, args[2]);
	
	double v = asReal1(vec);
	//double e = asReal1(each);
	double l = asReal1(len);
	
	Double r(l);
	for(uint64_t i = 0; i < l; i++) {
		r[i] = v;
	}
	state.registers[0] = r;
	return 1;
}

uint64_t typeOf(State& state, Call const& call, List const& args) {
	checkNumArgs(args, 1);
	Character c(1);
	c[0] = state.inString(force(state, args[0]).type.toString());
	state.registers[0] = c;
	return 1;
}

uint64_t mode(State& state, Call const& call, List const& args) {
	checkNumArgs(args, 1);
	Character c(1);
	Value v = force(state, args[0]);
	if(v.type == Type::R_integer || v.type == Type::R_double)
		c[0] = state.inString("numeric");
	else if(v.type == Type::R_symbol)
		c[0] = state.inString("name");
	else
		c[0] = state.inString(v.type.toString());
	state.registers[0] = c;
	return 1;
}

uint64_t klass(State& state, Call const& call, List const& args)
{
	checkNumArgs(args, 1);
	Value v = force(state, args[0]);
	Vector r = getClass(v.attributes);	
	if(r.type == Type::R_null) {
		Character c(1);
		c[0] = state.inString((v).type.toString());
		state.registers[0] = c;
	}
	else {
		state.registers[0] = r;
	}
	return 1;
}

uint64_t assignKlass(State& state, Call const& call, List const& args)
{
	checkNumArgs(args, 2);
	Value v = force(state, args[0]);
	Value k = force(state, args[1]);
	setClass(v.attributes, k);
	state.registers[0] = v;
	return 1;
}

uint64_t names(State& state, Call const& call, List const& args)
{
	checkNumArgs(args, 1);
	Value v = force(state, args[0]);
	Value r = getNames(v.attributes);
	state.registers[0] = r;	
	return 1;
}

uint64_t assignNames(State& state, Call const& call, List const& args)
{
	checkNumArgs(args, 2);
	Value v = force(state, args[0]);
	Value k = force(state, args[2]);
	setNames(v.attributes, k);
	state.registers[0] = v;
	return 1;
}

uint64_t dim(State& state, Call const& call, List const& args)
{
	checkNumArgs(args, 1);
	Value v = force(state, args[0]);
	Value r = getDim(v.attributes);
	state.registers[0]= r;	
	return 1;
}

uint64_t assignDim(State& state, Call const& call, List const& args)
{
	checkNumArgs(args, 2);
	Value v = force(state, args[0]);
	Value k = force(state, args[1]);
	setDim(v.attributes, k);
	state.registers[0] = v;
	return 1;
}

Type cTypeCast(Value const& v, Type t)
{
	Type r;
	r.v = std::max(v.type.Enum(), t.Enum());
	return r;
}

uint64_t c(State& state, Call const& call, List const& Args) {
	uint64_t total = 0;
	Type type = Type::R_null;
	std::vector<Vector> args;
	for(uint64_t i = 0; i < Args.length(); i++) {
		args.push_back(Vector(force(state, Args[i])));
		total += args[i].length();
		type = cTypeCast(args[i], type);
	}
	Vector out(type, total);
	uint64_t j = 0;
	for(uint64_t i = 0; i < args.size(); i++) {
		Insert(args[i], 0, out, j, args[i].length());
		j += args[i].length();
	}
	
	Vector n = getNames(Args.attributes);
	if(n.type != Type::R_null)
	{
		Character names(n);
		Character outnames(total);
		uint64_t j = 0;
		for(uint64_t i = 0; i < args.size(); i++) {
			for(uint64_t m = 0; m < args[i].length(); m++, j++) {
				// NYI: R makes these names distinct
				outnames[j] = names[i];
			}
		}
		setNames(out.attributes, outnames);
	}
	state.registers[0] = out;
	return 1;
}

uint64_t list(State& state, Call const& call, List const& args) {
	List out(args.length());
	for(uint64_t i = 0; i < args.length(); i++) out[i] = force(state, args[i]);
	Vector n = getNames(args.attributes);
	if(n.type != Type::R_null)
		setNames(out.attributes, n);
	state.registers[0] = out;
	return 1;
}
/*
uint64_t UseMethod(State& state, uint64_t nargs)
{
	return 0;
}

uint64_t plusOp(State& state, uint64_t nargs) {
	if(nargs == 1)
		return unaryArith<Zip1, PosOp>(state, nargs);
	else
		return binaryArith<Zip2, AddOp>(state, nargs);
}

uint64_t minusOp(State& state, uint64_t nargs) {
	if(nargs == 1)
		return unaryArith<Zip1, NegOp>(state, nargs);
	else
		return binaryArith<Zip2, SubOp>(state, nargs);
}
*/
uint64_t subset(State& state, Call const& call, List const& args) {

	checkNumArgs(args, 2);

        Value a = force(state, args[0]);
        Value i = force(state, args[1]);

        Vector r;
        if(a.type == Type::R_double && i.type == Type::R_double) {
                r = SubsetIndex< Double, Double >::eval(a, i);
        }
        else if(a.type == Type::R_integer && i.type == Type::R_double) {
                r = SubsetIndex< Integer, Double >::eval(a, i);
        }
        else if(a.type == Type::R_double && i.type == Type::R_integer) {
                r = SubsetIndex< Double, Integer >::eval(a, i);
        }
        else if(a.type == Type::R_integer && i.type == Type::R_integer) {
                r = SubsetIndex< Integer, Integer >::eval(a, i);
        }
        else if(a.type == Type::R_logical && i.type == Type::R_double) {
                r = SubsetIndex< Logical, Double >::eval(a, i);
        }
        else if(a.type == Type::R_logical && i.type == Type::R_integer) {
                r = SubsetIndex< Logical, Integer >::eval(a, i);
        }
        else if(a.type == Type::R_character && i.type == Type::R_double) {
                r = SubsetIndex< Character, Double >::eval(a, i);
        }
        else if(a.type == Type::R_character && i.type == Type::R_integer) {
                r = SubsetIndex< Character, Integer >::eval(a, i);
        }
        else if(a.type == Type::R_list && i.type == Type::R_double) {
                r = SubsetIndex< List, Double >::eval(a, i);
        }
        else if(a.type == Type::R_list && i.type == Type::R_integer) {
                r = SubsetIndex< List, Integer >::eval(a, i);
        }
        else if(a.type == Type::R_double && i.type == Type::R_logical) {
                //r = SubsetIndex< Integer, Double >::eval(a, i);
        }
        else if(a.type == Type::R_integer && i.type == Type::R_logical) {
                //r = SubsetIndex< Integer, Double >::eval(a, i);
        }
        else if(a.type == Type::R_logical && i.type == Type::R_logical) {
                //r = SubsetIndex< Integer, Double >::eval(a, i);
        }
        else {
                _error("Invalid index\n");
        }
	state.registers[0] = r;
        return 1;
}

uint64_t subset2(State& state, Call const& call, List const& args) {
	checkNumArgs(args, 2);

        Value a = force(state, args[0]);
        Value b = force(state, args[1]);
	if(b.type == Type::R_character) {
		Symbol i = Character(b)[0];
		Value r = getNames(a.attributes);
		if(r.type != Type::R_null) {
			Character c(r);
			uint64_t j = 0;
			for(;j < c.length(); j++) {
				if(c[j] == i)
					break;
			}
			if(j < c.length()) {
				state.registers[0] = Element2(a, j);
				return 1;
			}
		}
	}
	else if(b.type == Type::R_integer) {
		state.registers[0] = Element2(a, Integer(b)[0]-1);
		return 1;
	}
	else if(b.type == Type::R_double) {
		state.registers[0] = Element2(a, (uint64_t)Double(b)[0]-1);
		return 1;
	}
	state.registers[0] = Null::singleton;
	return 1;
} 

uint64_t dollar(State& state, Call const& call, List const& args) {
	checkNumArgs(args, 2);

        Value a = force(state, args[0]);
        uint64_t i = Symbol(expression(args[1])).i;
	Value r = getNames(a.attributes);
	if(r.type != Type::R_null) {
		Character c(r);
		uint64_t j = 0;
		for(;j < c.length(); j++) {
			if(c[j] == i)
				break;
		}
		if(j < c.length()) {
			state.registers[0] = Element2(a, j);
			return 1;
		}
	}
	state.registers[0] = Null::singleton;
	return 1;
} 

uint64_t length(State& state, Call const& call, List const& args) {
	checkNumArgs(args, 1);
	Vector a = force(state, args[0]);
	Integer i(1);
	i[0] = a.length();
	state.registers[0] = i;
	return 1;
}

/*uint64_t stop(State& state, Call const& call, List const& args) {
	state.stopped = true;
	return 0;
}
*/
uint64_t quote(State& state, Call const& call, List const& args) {
	checkNumArgs(args, 1);
	state.registers[0] = expression(args[0]);
	return 1;
}

uint64_t eval_fn(State& state, Call const& call, List const& args) {
	checkNumArgs(args, 2);
	Value expr = force(state, args[0]);
	Value envir = force(state, args[1]);
	//Value enclos = force(state, call[3]);
	Closure closure = Compiler::compile(state, expr);
	closure.bind(REnvironment(envir).ptr());
	eval(state, closure);
	return 1;
}

uint64_t switch_fn(State& state, Call const& call, List const& args) {
	Value one = force(state, args[0]);
	if(one.type == Type::R_integer && Integer(one).length() == 1) {
		int64_t i = Integer(one)[0];
		if(i >= 1 && (uint64_t)i <= args.length()) {state.registers[0] = force(state, args[i]); return 1; }
	} else if(one.type == Type::R_double && Double(one).length() == 1) {
		int64_t i = (int64_t)Double(one)[0];
		if(i >= 1 && (uint64_t)i <= args.length()) {state.registers[0] = force(state, args[i]); return 1; }
	} else if(one.type == Type::R_character && Character(one).length() == 1 && 
			getNames(args.attributes).type != Type::R_null) {
		Character names(getNames(args.attributes));
		for(uint64_t i = 1; i < args.length(); i++) {
			if(names[i] == Character(one)[0]) {
				state.registers[0] = force(state, args[i]);
				return 1;
			}
		}
		for(uint64_t i = 1; args.length(); i++) {
			if(names[i] == Symbol::empty) {
				state.registers[0] = force(state, args[i]);
				return 1;
			}
		}
	}
	state.registers[0] = Null::singleton;
	return 1;
}

uint64_t environment(State& state, Call const& call, List const& args) {
	checkNumArgs(args, 1);
	Value e = force(state, args[0]);
	if(e.type == Type::R_null) {
		state.registers[0] = REnvironment(state.env);
		return 1;
	}
	else if(e.type == Type::R_function) {
		state.registers[0] = REnvironment(Function(e).s());
		return 1;
	}
	state.registers[0] = Null::singleton;
	return 1;
}

uint64_t parentframe(State& state, Call const& call, List const& args) {
	checkNumArgs(args, 1);
	uint64_t i = (uint64_t)asReal1(force(state, args[0]));
	Environment* e = state.env;
	for(uint64_t j = 0; j < i-1 && e != NULL; j++) {
		e = e->dynamicParent();
	}
	state.registers[0] = REnvironment(e);
	return 1;
}

uint64_t stop_fn(State& state, Call const& call, List const& args) {
	// this should stop whether or not the arguments are correct...
	std::string message = "user stop";
	if(args.length() > 0) {
		if(args[0].type == Type::R_character && Character(args[0]).length() > 0) {
			message = Character(args[0])[0].toString(state);
		}
	}
	_error(message);
	return 0;
}

uint64_t warning_fn(State& state, Call const& call, List const& args) {
	std::string message = "user warning";
	if(args.length() > 0) {
		if(args[0].type == Type::R_character && Character(args[0]).length() > 0) {
			message = Character(args[0])[0].toString(state);
		}
	}
	_warning(state, message);
	state.registers[0] = Character::c(state, message);
	return 1;
} 

void addMathOps(State& state)
{
	Value v;
	Environment* env = state.baseenv;

	// operators that are implemented as byte codes, thus, no actual implemention is necessary here.
	/*CFunction(forloop).toValue(v);
	env->assign(Symbol(state, "for"), v);
	CFunction(whileloop).toValue(v);
	env->assign(Symbol(state, "while"), v);
	CFunction(assign).toValue(v);
	env->assign(Symbol(state, "<-"), v);
	CFunction(curlyBrackets).toValue(v);
	env->assign(Symbol(state, "{"), v);
	CFunction(parentheses).toValue(v);
	env->assign(Symbol(state, "("), v);*/

	/*CFunction(plusOp).toValue(v);
	env->assign(Symbol(state, "+"), v);
	CFunction(minusOp).toValue(v);
	env->assign(Symbol(state, "-"), v);
	op = binaryArith<Zip2, MulOp>;
	CFunction(op).toValue(v);
	env->assign(Symbol(state, "*"), v);
	op = binaryDoubleArith<Zip2, DivOp>;
	CFunction(op).toValue(v);
	env->assign(Symbol(state, "/"), v);
	op = binaryArith<Zip2, IDivOp>;
	CFunction(op).toValue(v);
	env->assign(Symbol(state, "%/%"), v);
	op = binaryDoubleArith<Zip2, PowOp>;
	CFunction(op).toValue(v);
	env->assign(Symbol(state, "^"), v);
	op = binaryArith<Zip2, ModOp>;
	CFunction(op).toValue(v);
	env->assign(Symbol(state, "%%"), v);

	op = unaryLogical<Zip1, LNegOp>;
	CFunction(op).toValue(v);
	env->assign(Symbol(state, "!"), v);
	op = binaryLogical<Zip2, AndOp>;
	CFunction(op).toValue(v);
	env->assign(Symbol(state, "&"), v);
	op = binaryLogical<Zip2, OrOp>;
	CFunction(op).toValue(v);
	env->assign(Symbol(state, "|"), v);
	op = binaryOrdinal<Zip2, EqOp>;
	CFunction(op).toValue(v);
	env->assign(Symbol(state, "=="), v);
	op = binaryOrdinal<Zip2, NeqOp>;
	CFunction(op).toValue(v);
	env->assign(Symbol(state, "!="), v);
	op = binaryOrdinal<Zip2, LTOp>;
	CFunction(op).toValue(v);
	env->assign(Symbol(state, "<"), v);
	op = binaryOrdinal<Zip2, LEOp>;
	CFunction(op).toValue(v);
	env->assign(Symbol(state, "<="), v);
	op = binaryOrdinal<Zip2, GTOp>;
	CFunction(op).toValue(v);
	env->assign(Symbol(state, ">"), v);
	op = binaryOrdinal<Zip2, GEOp>;
	CFunction(op).toValue(v);
	env->assign(Symbol(state, ">="), v);*/
	
	CFunction(function).toValue(v);
	env->assign(Symbol(state, "function"), v);
	CFunction(rm).toValue(v);
	env->assign(Symbol(state, "rm"), v);
	CFunction(typeOf).toValue(v);
	env->assign(Symbol(state, "typeof"), v);
	env->assign(Symbol(state, "storage.mode"), v);
	CFunction(mode).toValue(v);
	env->assign(Symbol(state, "mode"), v);

	CFunction(sequence).toValue(v);
	env->assign(Symbol(state, "seq"), v);
	CFunction(repeat).toValue(v);
	env->assign(Symbol(state, "rep"), v);
	
	CFunction(klass).toValue(v);
	env->assign(Symbol(state, "class"), v);
	CFunction(assignKlass).toValue(v);
	env->assign(Symbol(state, "class<-"), v);
	CFunction(names).toValue(v);
	env->assign(Symbol(state, "names"), v);
	CFunction(assignNames).toValue(v);
	env->assign(Symbol(state, "names<-"), v);
	CFunction(dim).toValue(v);
	env->assign(Symbol(state, "dim"), v);
	CFunction(assignDim).toValue(v);
	env->assign(Symbol(state, "dim<-"), v);
	
	CFunction(c).toValue(v);
	env->assign(Symbol(state, "c"), v);
	CFunction(list).toValue(v);
	env->assign(Symbol(state, "list"), v);

	CFunction(length).toValue(v);
	env->assign(Symbol(state, "length"), v);
	
	CFunction(subset).toValue(v);
	env->assign(Symbol(state, "["), v);
	CFunction(subset2).toValue(v);
	env->assign(Symbol(state, "[["), v);
	CFunction(dollar).toValue(v);
	env->assign(Symbol(state, "$"), v);
/*
	CFunction(stop).toValue(v);
	env->assign(Symbol(state, "stop"), v);
*/
	
	CFunction(switch_fn).toValue(v);
	env->assign(Symbol(state, "switch"), v);
	
	CFunction(eval_fn).toValue(v);
	env->assign(Symbol(state, "eval"), v);
	CFunction(quote).toValue(v);
	env->assign(Symbol(state, "quote"), v);

	CFunction(environment).toValue(v);
	env->assign(Symbol(state, "environment"), v);
	CFunction(parentframe).toValue(v);
	env->assign(Symbol(state, "parent.frame"), v);
	
	CFunction(stop_fn).toValue(v);
	env->assign(Symbol(state, "stop"), v);
	CFunction(warning_fn).toValue(v);
	env->assign(Symbol(state, "warning"), v);
}

