
#include "internal.h"

void asnull(State& state, Value const* args, Value& result) {
	result = As<Null>(state, args[0]);
}

void aslogical(State& state, Value const* args, Value& result) {
	result = As<Logical>(state, args[0]);
}

void asinteger(State& state, Value const* args, Value& result) {
	result = As<Integer>(state, args[0]);
}

void asdouble(State& state, Value const* args, Value& result) {
	result = As<Double>(state, args[0]);
}

void ascomplex(State& state, Value const* args, Value& result) {
	result = As<Complex>(state, args[0]);
}

void ascharacter(State& state, Value const* args, Value& result) {
	result = As<Character>(state, args[0]);
}

void aslist(State& state, Value const* args, Value& result) {
	result = As<List>(state, args[0]);
}

// Implement internally or as R library?
/*
void isnull(State& state, Value const* args, Value& result) {
	Value from = force(state, args[0]);
	state.registers[0] = Logical::c(from.isNull());
	return 1;
}

void islogical(State& state, Value const* args, Value& result) {
	Value from = force(state, args[0]);
	state.registers[0] = Logical::c(from.isLogical());
	return 1;
}

void isinteger(State& state, Value const* args, Value& result) {
	Value from = force(state, args[0]);
	state.registers[0] = Logical::c(from.isInteger());
	return 1;
}

void isdouble(State& state, Value const* args, Value& result) {
	Value from = force(state, args[0]);
	state.registers[0] = Logical::c(from.isDouble());
	return 1;
}

void iscomplex(State& state, Value const* args, Value& result) {
	Value from = force(state, args[0]);
	state.registers[0] = Logical::c(from.isComplex());
	return 1;
}

void ischaracter(State& state, Value const* args, Value& result) {
	Value from = force(state, args[0]);
	state.registers[0] = Logical::c(from.isCharacter());
	return 1;
}

void islist(State& state, Value const* args, Value& result) {
	Value from = force(state, args[0]);
	state.registers[0] = Logical::c(from.isList());
	return 1;
}
*/

void importCoerceFunctions(State& state, Environment* env)
{
	state.registerInternalFunction(state.internStr("as.null"), (asnull), 1);
	state.registerInternalFunction(state.internStr("as.logical"), (aslogical), 1);
	state.registerInternalFunction(state.internStr("as.integer"), (asinteger), 1);
	state.registerInternalFunction(state.internStr("as.double"), (asdouble), 1);
	state.registerInternalFunction(state.internStr("as.numeric"), (asdouble), 1);
	state.registerInternalFunction(state.internStr("as.complex"), (ascomplex), 1);
	state.registerInternalFunction(state.internStr("as.character"), (ascharacter), 1);
	state.registerInternalFunction(state.internStr("as.list"), (aslist), 1);

/*
	state.registerInternalFunction(state.internStr("is.null"), (isnull));
	state.registerInternalFunction(state.internStr("is.logical"), (islogical));
	state.registerInternalFunction(state.internStr("is.integer"), (isinteger));
	state.registerInternalFunction(state.internStr("is.double"), (isdouble));
	state.registerInternalFunction(state.internStr("is.real"), (isdouble));
	state.registerInternalFunction(state.internStr("is.complex"), (iscomplex));
	state.registerInternalFunction(state.internStr("is.character"), (ischaracter));
	state.registerInternalFunction(state.internStr("is.list"), (islist));
*/
}
