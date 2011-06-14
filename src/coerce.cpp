
#include "internal.h"

int64_t asnull(State& state, Call const& call, List const& args) {
	Value from = force(state, args[0]);
	state.registers[0] = As<Null>(state, from);
	return 1;
}

int64_t aslogical(State& state, Call const& call, List const& args) {
	Value from = force(state, args[0]);
	state.registers[0] = As<Logical>(state, from);
	return 1;
}

int64_t asinteger(State& state, Call const& call, List const& args) {
	Value from = force(state, args[0]);
	state.registers[0] = As<Integer>(state, from);
	return 1;
}

int64_t asdouble(State& state, Call const& call, List const& args) {
	Value from = force(state, args[0]);
	state.registers[0] = As<Double>(state, from);
	return 1;
}

int64_t ascomplex(State& state, Call const& call, List const& args) {
	Value from = force(state, args[0]);
	state.registers[0] = As<Complex>(state, from);
	return 1;
}

int64_t ascharacter(State& state, Call const& call, List const& args) {
	Value from = force(state, args[0]);
	state.registers[0] = As<Character>(state, from);
	return 1;
}

int64_t aslist(State& state, Call const& call, List const& args) {
	Value from = force(state, args[0]);
	state.registers[0] = As<List>(state, from);
	return 1;
}

// Implement internally or as R library?
/*
int64_t isnull(State& state, Call const& call, List const& args) {
	Value from = force(state, args[0]);
	state.registers[0] = Logical::c(from.isNull());
	return 1;
}

int64_t islogical(State& state, Call const& call, List const& args) {
	Value from = force(state, args[0]);
	state.registers[0] = Logical::c(from.isLogical());
	return 1;
}

int64_t isinteger(State& state, Call const& call, List const& args) {
	Value from = force(state, args[0]);
	state.registers[0] = Logical::c(from.isInteger());
	return 1;
}

int64_t isdouble(State& state, Call const& call, List const& args) {
	Value from = force(state, args[0]);
	state.registers[0] = Logical::c(from.isDouble());
	return 1;
}

int64_t iscomplex(State& state, Call const& call, List const& args) {
	Value from = force(state, args[0]);
	state.registers[0] = Logical::c(from.isComplex());
	return 1;
}

int64_t ischaracter(State& state, Call const& call, List const& args) {
	Value from = force(state, args[0]);
	state.registers[0] = Logical::c(from.isCharacter());
	return 1;
}

int64_t islist(State& state, Call const& call, List const& args) {
	Value from = force(state, args[0]);
	state.registers[0] = Logical::c(from.isList());
	return 1;
}
*/

void importCoerceFunctions(State& state)
{
	Environment* env = state.path[0];

	env->assign(Symbol(state,"as.null"), CFunction(asnull));
	env->assign(Symbol(state,"as.logical"), CFunction(aslogical));
	env->assign(Symbol(state,"as.integer"), CFunction(asinteger));
	env->assign(Symbol(state,"as.double"), CFunction(asdouble));
	env->assign(Symbol(state,"as.numeric"), CFunction(asdouble));
	env->assign(Symbol(state,"as.complex"), CFunction(ascomplex));
	env->assign(Symbol(state,"as.character"), CFunction(ascharacter));
	env->assign(Symbol(state,"as.list"), CFunction(aslist));

/*
	env->assign(Symbol(state,"is.null"), CFunction(isnull));
	env->assign(Symbol(state,"is.logical"), CFunction(islogical));
	env->assign(Symbol(state,"is.integer"), CFunction(isinteger));
	env->assign(Symbol(state,"is.double"), CFunction(isdouble));
	env->assign(Symbol(state,"is.real"), CFunction(isdouble));
	env->assign(Symbol(state,"is.complex"), CFunction(iscomplex));
	env->assign(Symbol(state,"is.character"), CFunction(ischaracter));
	env->assign(Symbol(state,"is.list"), CFunction(islist));
*/
}
