
#include "internal.h"
#include <assert.h>
#include <math.h>

uint64_t function(State& state, Call const& call) {
	assert(call.length()-1 == 3);
	state.stack.push(
		Function(force(state, call[1]), 
			code(call[2]), 
			force(state, call[3]), state.env));
	return 1;
}

uint64_t rm(State& state, Call const& call) {
	assert(call.length()-1 == 1);
	state.env->rm(quoted(call[1]));
	state.stack.push(Null::singleton);
	return 1;
}

uint64_t sequence(State& state, Call const& call) {
	assert(call.length()-1 == 3);

	Value from = force(state, call[1]);
	Value by   = force(state, call[2]);
	Value len  = force(state, call[3]);
	
	double f = asReal1(from);
	double b = asReal1(by);
	double l = asReal1(len);

	Double r(l);
	double j = 0;
	for(uint64_t i = 0; i < l; i++) {
		r[i] = f+j;
		j = j + b;
	}
	state.stack.push(r);
	return 1;
}

uint64_t repeat(State& state, Call const& call) {
	assert(call.length()-1 == 3);
	
	Value vec  = force(state, call[1]);
	Value each = force(state, call[2]);
	Value len  = force(state, call[3]);
	
	double v = asReal1(vec);
	//double e = asReal1(each);
	double l = asReal1(len);
	
	Double r(l);
	for(uint64_t i = 0; i < l; i++) {
		r[i] = v;
	}
	state.stack.push(r);
	return 1;
}

uint64_t typeOf(State& state, Call const& call) {
	assert(call.length()-1 == 1);
	Character c(1);
	c[0] = state.inString(force(state, call[1]).type.toString());
	state.stack.push(c);
	return 1;
}

uint64_t mode(State& state, Call const& call) {
	assert(call.length()-1 == 1);
	Character c(1);
	Value v = force(state, call[1]);
	if(v.type == Type::R_integer || v.type == Type::R_double)
		c[0] = state.inString("numeric");
	else if(v.type == Type::R_symbol)
		c[0] = state.inString("name");
	else
		c[0] = state.inString(v.type.toString());
	state.stack.push(c);
	return 1;
}

uint64_t klass(State& state, Call const& call)
{
	assert(call.length()-1 == 1);
	Value v = force(state, call[1]);
	Vector r = getClass(v.attributes);	
	if(r.type == Type::R_null) {
		Character c(1);
		c[0] = state.inString((v).type.toString());
		state.stack.push(c);
	}
	else {
		state.stack.push(r);
	}
	return 1;
}

uint64_t assignKlass(State& state, Call const& call)
{
	assert(call.length()-1 == 2);
	Value v = force(state, call[1]);
	Value k = force(state, call[2]);
	setClass(v.attributes, k);
	state.stack.push(v);
	return 1;
}

uint64_t names(State& state, Call const& call)
{
	assert(call.length()-1 == 1);
	Value v = force(state, call[1]);
	Value r = getNames(v.attributes);
	state.stack.push(r);	
	return 1;
}

uint64_t assignNames(State& state, Call const& call)
{
	assert(call.length()-1 == 2);
	Value v = force(state, call[1]);
	Value k = force(state, call[2]);
	setNames(v.attributes, k);
	state.stack.push(v);
	return 1;
}

uint64_t dim(State& state, Call const& call)
{
	assert(call.length()-1 == 1);
	Value v = force(state, call[1]);
	Value r = getDim(v.attributes);
	state.stack.push(r);	
	return 1;
}

uint64_t assignDim(State& state, Call const& call)
{
	assert(call.length()-1 == 2);
	Value v = force(state, call[1]);
	Value k = force(state, call[2]);
	setDim(v.attributes, k);
	state.stack.push(v);
	return 1;
}

Type cTypeCast(Value const& v, Type t)
{
	return std::max(v.type.internal(), t.internal());
}

uint64_t c(State& state, Call const& call) {
	uint64_t total = 0;
	Type type = Type::R_null;
	std::vector<Vector> args;
	for(uint64_t i = 1; i < call.length(); i++) {
		args.push_back(Vector(force(state, call[i])));
		total += args[i-1].length();
		type = cTypeCast(args[i-1], type);
	}
	Vector out(type, total);
	uint64_t j = 0;
	for(uint64_t i = 0; i < args.size(); i++) {
		Insert(args[i], 0, out, j, args[i].length());
		j += args[i].length();
	}
	
	Vector n = getNames(call.attributes);
	if(n.type != Type::R_null)
	{
		Character names(n);
		Character outnames(total);
		uint64_t j = 0;
		for(uint64_t i = 0; i < args.size(); i++) {
			for(uint64_t m = 0; m < args[i].length(); m++, j++) {
				// NYI: R makes these names distinct
				outnames[j] = names[i+1];
			}
		}
		setNames(out.attributes, outnames);
	}
	state.stack.push(out);
	return 1;
}

uint64_t list(State& state, Call const& call) {
	List out(call.length()-1);
	for(uint64_t i = 1; i < call.length(); i++) out[i-1] = force(state, call[i]);
	Vector n = getNames(call.attributes);
	if(n.type != Type::R_null)
		setNames(out.attributes, Subset(n, 1, call.length()-1));
	state.stack.push(out);
	return 1;
}

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

uint64_t subset(State& state, Call const& call) {

        assert(call.length()-1 == 2);

        Value a = force(state, call[1]);
        Value i = force(state, call[2]);

        Vector r;
        if(a.type == Type::R_double && i.type == Type::R_double) {
                SubsetIndex< Double, Double >::eval(a, i).toVector(r);
        }
        else if(a.type == Type::R_integer && i.type == Type::R_double) {
                SubsetIndex< Integer, Double >::eval(a, i).toVector(r);
        }
        else if(a.type == Type::R_double && i.type == Type::R_integer) {
                SubsetIndex< Integer, Double >::eval(a, i).toVector(r);
        }
        else if(a.type == Type::R_integer && i.type == Type::R_integer) {
                SubsetIndex< Integer, Integer >::eval(a, i).toVector(r);
        }
        else if(a.type == Type::R_logical && i.type == Type::R_double) {
                SubsetIndex< Logical, Double >::eval(a, i).toVector(r);
        }
        else if(a.type == Type::R_logical && i.type == Type::R_integer) {
                SubsetIndex< Logical, Integer >::eval(a, i).toVector(r);
        }
        else if(a.type == Type::R_character && i.type == Type::R_double) {
                SubsetIndex< Character, Double >::eval(a, i).toVector(r);
        }
        else if(a.type == Type::R_character && i.type == Type::R_integer) {
                SubsetIndex< Character, Integer >::eval(a, i).toVector(r);
        }
        else if(a.type == Type::R_double && i.type == Type::R_logical) {
                //SubsetIndex< Integer, Double >::eval(a, i).toVector(r);
        }
        else if(a.type == Type::R_integer && i.type == Type::R_logical) {
                //SubsetIndex< Integer, Double >::eval(a, i).toVector(r);
        }
        else if(a.type == Type::R_logical && i.type == Type::R_logical) {
                //SubsetIndex< Integer, Double >::eval(a, i).toVector(r);
        }
        else {
                printf("Invalid index\n");
                assert(false);
        }
	state.stack.push(r);
        return 1;
}

uint64_t length(State& state, Call const& call) {
	Vector a = force(state, call[1]);
	Integer i(1);
	i[0] = a.length();
	state.stack.push(i);
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
}

