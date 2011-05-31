
#include "internal.h"

uint64_t aslogical(State& state, Call const& call, List const& args) {
	Value from = force(state, args[0]);
	state.registers[0] = As<Logical>(state, from);
	return 1;
}

uint64_t asinteger(State& state, Call const& call, List const& args) {
	Value from = force(state, args[0]);
	state.registers[0] = As<Integer>(state, from);
	return 1;
}

uint64_t asdouble(State& state, Call const& call, List const& args) {
	Value from = force(state, args[0]);
	state.registers[0] = As<Double>(state, from);
	return 1;
}

uint64_t ascomplex(State& state, Call const& call, List const& args) {
	Value from = force(state, args[0]);
	state.registers[0] = As<Complex>(state, from);
	return 1;
}

uint64_t ascharacter(State& state, Call const& call, List const& args) {
	Value from = force(state, args[0]);
	state.registers[0] = As<Character>(state, from);
	return 1;
}

uint64_t aslist(State& state, Call const& call, List const& args) {
	Value from = force(state, args[0]);
	state.registers[0] = As<List>(state, from);
	return 1;
}


uint64_t isnull(State& state, Call const& call, List const& args) {
	Value from = force(state, args[0]);
	state.registers[0] = Logical::c(from.isNull());
	return 1;
}

uint64_t islogical(State& state, Call const& call, List const& args) {
	Value from = force(state, args[0]);
	state.registers[0] = Logical::c(from.isLogical());
	return 1;
}

uint64_t isinteger(State& state, Call const& call, List const& args) {
	Value from = force(state, args[0]);
	state.registers[0] = Logical::c(from.isInteger());
	return 1;
}

uint64_t isdouble(State& state, Call const& call, List const& args) {
	Value from = force(state, args[0]);
	state.registers[0] = Logical::c(from.isDouble());
	return 1;
}

uint64_t iscomplex(State& state, Call const& call, List const& args) {
	Value from = force(state, args[0]);
	state.registers[0] = Logical::c(from.isComplex());
	return 1;
}

uint64_t ischaracter(State& state, Call const& call, List const& args) {
	Value from = force(state, args[0]);
	state.registers[0] = Logical::c(from.isCharacter());
	return 1;
}

uint64_t islist(State& state, Call const& call, List const& args) {
	Value from = force(state, args[0]);
	state.registers[0] = Logical::c(from.isList());
	return 1;
}

void importCoerceFunctions(State& state)
{
	Value v;
	Environment* env = state.baseenv;

	CFunction(aslogical).toValue(v);
	env->assign(Symbol(state, "as.logical"), v);
	CFunction(asinteger).toValue(v);
	env->assign(Symbol(state, "as.integer"), v);
	CFunction(asdouble).toValue(v);
	env->assign(Symbol(state, "as.double"), v);
	env->assign(Symbol(state, "as.numeric"), v);
	CFunction(ascomplex).toValue(v);
	env->assign(Symbol(state, "as.complex"), v);
	CFunction(ascharacter).toValue(v);
	env->assign(Symbol(state, "as.character"), v);
	CFunction(aslist).toValue(v);
	env->assign(Symbol(state, "as.list"), v);

	CFunction(isnull).toValue(v);
	env->assign(Symbol(state, "is.null"), v);
	CFunction(islogical).toValue(v);
	env->assign(Symbol(state, "is.logical"), v);
	CFunction(isinteger).toValue(v);
	env->assign(Symbol(state, "is.integer"), v);
	CFunction(isdouble).toValue(v);
	env->assign(Symbol(state, "is.double"), v);
	env->assign(Symbol(state, "is.real"), v);
	CFunction(iscomplex).toValue(v);
	env->assign(Symbol(state, "is.complex"), v);
	CFunction(ischaracter).toValue(v);
	env->assign(Symbol(state, "is.character"), v);
	CFunction(islist).toValue(v);
	env->assign(Symbol(state, "is.list"), v);
}
