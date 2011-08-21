
#include "internal.h"

Value asnull(State& state, List const& args, Character const& names) {
	Value from = force(state, args[0]);
	return As<Null>(state, from);
}

Value aslogical(State& state, List const& args, Character const& names) {
	Value from = force(state, args[0]);
	return As<Logical>(state, from);
}

Value asinteger(State& state, List const& args, Character const& names) {
	Value from = force(state, args[0]);
	return As<Integer>(state, from);
}

Value asdouble(State& state, List const& args, Character const& names) {
	Value from = force(state, args[0]);
	return As<Double>(state, from);
}

Value ascomplex(State& state, List const& args, Character const& names) {
	Value from = force(state, args[0]);
	return As<Complex>(state, from);
}

Value ascharacter(State& state, List const& args, Character const& names) {
	Value from = force(state, args[0]);
	return As<Character>(state, from);
}

Value aslist(State& state, List const& args, Character const& names) {
	Value from = force(state, args[0]);
	return As<List>(state, from);
}

// Implement internally or as R library?
/*
Value isnull(State& state, List const& args, Character const& names) {
	Value from = force(state, args[0]);
	state.registers[0] = Logical::c(from.isNull());
	return 1;
}

Value islogical(State& state, List const& args, Character const& names) {
	Value from = force(state, args[0]);
	state.registers[0] = Logical::c(from.isLogical());
	return 1;
}

Value isinteger(State& state, List const& args, Character const& names) {
	Value from = force(state, args[0]);
	state.registers[0] = Logical::c(from.isInteger());
	return 1;
}

Value isdouble(State& state, List const& args, Character const& names) {
	Value from = force(state, args[0]);
	state.registers[0] = Logical::c(from.isDouble());
	return 1;
}

Value iscomplex(State& state, List const& args, Character const& names) {
	Value from = force(state, args[0]);
	state.registers[0] = Logical::c(from.isComplex());
	return 1;
}

Value ischaracter(State& state, List const& args, Character const& names) {
	Value from = force(state, args[0]);
	state.registers[0] = Logical::c(from.isCharacter());
	return 1;
}

Value islist(State& state, List const& args, Character const& names) {
	Value from = force(state, args[0]);
	state.registers[0] = Logical::c(from.isList());
	return 1;
}
*/

void importCoerceFunctions(State& state, Environment* env)
{
	env->assign(state.StrToSym("as.null"), BuiltIn(asnull));
	env->assign(state.StrToSym("as.logical"), BuiltIn(aslogical));
	env->assign(state.StrToSym("as.integer"), BuiltIn(asinteger));
	env->assign(state.StrToSym("as.double"), BuiltIn(asdouble));
	env->assign(state.StrToSym("as.numeric"), BuiltIn(asdouble));
	env->assign(state.StrToSym("as.complex"), BuiltIn(ascomplex));
	env->assign(state.StrToSym("as.character"), BuiltIn(ascharacter));
	env->assign(state.StrToSym("as.list"), BuiltIn(aslist));

/*
	env->assign(state.StrToSym("is.null"), BuiltIn(isnull));
	env->assign(state.StrToSym("is.logical"), BuiltIn(islogical));
	env->assign(state.StrToSym("is.integer"), BuiltIn(isinteger));
	env->assign(state.StrToSym("is.double"), BuiltIn(isdouble));
	env->assign(state.StrToSym("is.real"), BuiltIn(isdouble));
	env->assign(state.StrToSym("is.complex"), BuiltIn(iscomplex));
	env->assign(state.StrToSym("is.character"), BuiltIn(ischaracter));
	env->assign(state.StrToSym("is.list"), BuiltIn(islist));
*/
}
