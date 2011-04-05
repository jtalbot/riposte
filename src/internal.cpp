
#include "internal.h"
#include <assert.h>
#include <math.h>

uint64_t assign(State& state, uint64_t nargs) {
	return 1;
}

uint64_t forloop(State& state, uint64_t nargs) {
	return 1;
}

uint64_t whileloop(State& state, uint64_t nargs) {
	return 1;
}

//uint64_t Switch(State& state, uint64_t nargs) {
//	Character c(stack.pop());
//	
//}

uint64_t function(State& state, uint64_t nargs)
{
	assert(nargs == 3);
	
	Stack& stack = state.stack;
	
	Value arg0 = force(state, stack.pop());
	Value arg1 = code(stack.pop());
	Value arg2 = force(state, stack.pop());

	Function func(PairList(arg0), arg1, Character(arg2), state.env);
	Value result;
	func.toValue(result);
	stack.push(result);

	return 1;
}

uint64_t rm(State& state, uint64_t nargs) {
	assert(nargs == 1);
	Stack& stack = state.stack;
	Value symbol = quoted(stack.pop());
	state.env->rm(Symbol(symbol));
	stack.push(Value::null);
	return 1;
}

uint64_t curlyBrackets(State& state, uint64_t nargs) {
	return 1;
}

uint64_t parentheses(State& state, uint64_t nargs) {
	return 1;
}

uint64_t sequence(State& state, uint64_t nargs)
{
	assert(nargs == 3);

	Stack& stack = state.stack;
	
	Value from = force(state, stack.pop());
	Value by   = force(state, stack.pop());
	Value len  = force(state, stack.pop());
	
	double f = asReal1(from);
	double b = asReal1(by);
	double l = asReal1(len);

	Double r(l);
	double j = 0;
	for(uint64_t i = 0; i < l; i++) {
		r[i] = f+j;
		j = j + b;
	}
	Value v;
	r.toValue(stack.reserve());
	
	return 1;
}

uint64_t repeat(State& state, uint64_t nargs)
{
	assert(nargs == 3);
	Stack& stack = state.stack;
	
	Value vec  = force(state, stack.pop());
	Value each = force(state, stack.pop());
	Value len  = force(state, stack.pop());
	
	double v = asReal1(vec);
	//double e = asReal1(each);
	double l = asReal1(len);
	
	Double r(l);
	for(uint64_t i = 0; i < l; i++) {
		r[i] = v;
	}
	r.toValue(stack.reserve());
	return 1;
}

uint64_t typeOf(State& state, uint64_t nargs)
{
	assert(nargs == 1);
	Stack& stack = state.stack;
	Character c(1);
	c[0] = state.inString(force(state, stack.pop()).type.toString());
	c.toValue(stack.reserve());
	return 1;
}

uint64_t mode(State& state, uint64_t nargs)
{
	assert(nargs == 1);
	Stack& stack = state.stack;
	Character c(1);
	Value v = force(state, stack.pop());
	if(v.type == Type::R_integer || v.type == Type::R_double)
		c[0] = state.inString("numeric");
	else if(v.type == Type::R_symbol)
		c[0] = state.inString("name");
	else
		c[0] = state.inString(v.type.toString());
	c.toValue(stack.reserve());
	return 1;
}

uint64_t klass(State& state, uint64_t nargs)
{
	assert(nargs == 1);
	Stack& stack = state.stack;
	Value v = force(state, stack.pop());
	Value r = getClass(v.attributes);	
	if(r == Value::null) {
		Character c(1);
		c[0] = state.inString((v).type.toString());
		c.toValue(stack.reserve());
	}
	else {
		stack.push(r);
	}
	return 1;
}

uint64_t assignKlass(State& state, uint64_t nargs)
{
	assert(nargs == 2);
	Stack& stack = state.stack;
	Value v = force(state, stack.pop());
	Value k = force(state, stack.pop());
	setClass(v.attributes, k);
	stack.push(v);
	return 1;
}

uint64_t names(State& state, uint64_t nargs)
{
	assert(nargs == 1);
	Stack& stack = state.stack;
	Value v = force(state, stack.pop());
	Value r = getNames(v.attributes);
	stack.push(r);	
	return 1;
}

uint64_t assignNames(State& state, uint64_t nargs)
{
	assert(nargs == 2);
	Stack& stack = state.stack;
	Value v = force(state, stack.pop());
	Value k = force(state, stack.pop());
	setNames(v.attributes, k);
	stack.push(v);
	return 1;
}

uint64_t dim(State& state, uint64_t nargs)
{
	assert(nargs == 1);
	Stack& stack = state.stack;
	Value v = force(state, stack.pop());
	Value r = getDim(v.attributes);
	stack.push(r);	
	return 1;
}

uint64_t assignDim(State& state, uint64_t nargs)
{
	assert(nargs == 2);
	Stack& stack = state.stack;
	Value v = force(state, stack.pop());
	Value k = force(state, stack.pop());
	setDim(v.attributes, k);
	stack.push(v);
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

void addMathOps(State& state)
{
	Value v;
	CFunction::Cffi op;

	Environment* env = state.baseenv;

	// operators that are implemented as byte codes, thus, no actual implemention is necessary here.
	CFunction(forloop).toValue(v);
	env->assign(Symbol(state, "for"), v);
	CFunction(whileloop).toValue(v);
	env->assign(Symbol(state, "while"), v);
	CFunction(assign).toValue(v);
	env->assign(Symbol(state, "<-"), v);
	CFunction(curlyBrackets).toValue(v);
	env->assign(Symbol(state, "{"), v);
	CFunction(parentheses).toValue(v);
	env->assign(Symbol(state, "("), v);

	CFunction(plusOp).toValue(v);
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
	env->assign(Symbol(state, ">="), v);
	
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
}

