
#include "internal.h"

Value asnull(State& state, Call const& call, List const& args) {
	Value from = force(state, args[0]);
	return As<Null>(state, from);
}

Value aslogical(State& state, Call const& call, List const& args) {
	Value from = force(state, args[0]);
	return As<Logical>(state, from);
}

Value asinteger(State& state, Call const& call, List const& args) {
	Value from = force(state, args[0]);
	return As<Integer>(state, from);
}

Value asdouble(State& state, Call const& call, List const& args) {
	Value from = force(state, args[0]);
	return As<Double>(state, from);
}

Value ascomplex(State& state, Call const& call, List const& args) {
	Value from = force(state, args[0]);
	return As<Complex>(state, from);
}

Value ascharacter(State& state, Call const& call, List const& args) {
	Value from = force(state, args[0]);
	return As<Character>(state, from);
}

Value aslist(State& state, Call const& call, List const& args) {
	Value from = force(state, args[0]);
	return As<List>(state, from);
}

// Implement internally or as R library?
/*
Value isnull(State& state, Call const& call, List const& args) {
	Value from = force(state, args[0]);
	state.registers[0] = Logical::c(from.isNull());
	return 1;
}

Value islogical(State& state, Call const& call, List const& args) {
	Value from = force(state, args[0]);
	state.registers[0] = Logical::c(from.isLogical());
	return 1;
}

Value isinteger(State& state, Call const& call, List const& args) {
	Value from = force(state, args[0]);
	state.registers[0] = Logical::c(from.isInteger());
	return 1;
}

Value isdouble(State& state, Call const& call, List const& args) {
	Value from = force(state, args[0]);
	state.registers[0] = Logical::c(from.isDouble());
	return 1;
}

Value iscomplex(State& state, Call const& call, List const& args) {
	Value from = force(state, args[0]);
	state.registers[0] = Logical::c(from.isComplex());
	return 1;
}

Value ischaracter(State& state, Call const& call, List const& args) {
	Value from = force(state, args[0]);
	state.registers[0] = Logical::c(from.isCharacter());
	return 1;
}

Value islist(State& state, Call const& call, List const& args) {
	Value from = force(state, args[0]);
	state.registers[0] = Logical::c(from.isList());
	return 1;
}
*/

void importCoerceFunctions(State& state, Environment* env)
{
	env->assign(state.StrToSym("as.null"), CFunction(asnull));
	env->assign(state.StrToSym("as.logical"), CFunction(aslogical));
	env->assign(state.StrToSym("as.integer"), CFunction(asinteger));
	env->assign(state.StrToSym("as.double"), CFunction(asdouble));
	env->assign(state.StrToSym("as.numeric"), CFunction(asdouble));
	env->assign(state.StrToSym("as.complex"), CFunction(ascomplex));
	env->assign(state.StrToSym("as.character"), CFunction(ascharacter));
	env->assign(state.StrToSym("as.list"), CFunction(aslist));

/*
	env->assign(state.StrToSym("is.null"), CFunction(isnull));
	env->assign(state.StrToSym("is.logical"), CFunction(islogical));
	env->assign(state.StrToSym("is.integer"), CFunction(isinteger));
	env->assign(state.StrToSym("is.double"), CFunction(isdouble));
	env->assign(state.StrToSym("is.real"), CFunction(isdouble));
	env->assign(state.StrToSym("is.complex"), CFunction(iscomplex));
	env->assign(state.StrToSym("is.character"), CFunction(ischaracter));
	env->assign(state.StrToSym("is.list"), CFunction(islist));
*/
}
