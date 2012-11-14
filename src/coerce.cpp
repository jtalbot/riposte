
#include "coerce.h"

void asnull(Thread& thread, Value const* args, Value& result) {
	result = As<Null>(thread, args[0]);
}

void aslogical(Thread& thread, Value const* args, Value& result) {
	result = As<Logical>(thread, args[0]);
}

void asinteger(Thread& thread, Value const* args, Value& result) {
	result = As<Integer>(thread, args[0]);
}

void asdouble(Thread& thread, Value const* args, Value& result) {
	result = As<Double>(thread, args[0]);
}

void ascharacter(Thread& thread, Value const* args, Value& result) {
	result = As<Character>(thread, args[0]);
}

void aslist(Thread& thread, Value const* args, Value& result) {
	result = As<List>(thread, args[0]);
}

// Implement internally or as R library?
/*
void isnull(Thread& thread, Value const* args, Value& result) {
	Value from = force(thread, args[0]);
	thread.registers[0] = Logical::c(from.isNull());
	return 1;
}

void islogical(Thread& thread, Value const* args, Value& result) {
	Value from = force(thread, args[0]);
	thread.registers[0] = Logical::c(from.isLogical());
	return 1;
}

void isinteger(Thread& thread, Value const* args, Value& result) {
	Value from = force(thread, args[0]);
	thread.registers[0] = Logical::c(from.isInteger());
	return 1;
}

void isdouble(Thread& thread, Value const* args, Value& result) {
	Value from = force(thread, args[0]);
	thread.registers[0] = Logical::c(from.isDouble());
	return 1;
}

void ischaracter(Thread& thread, Value const* args, Value& result) {
	Value from = force(thread, args[0]);
	thread.registers[0] = Logical::c(from.isCharacter());
	return 1;
}

void islist(Thread& thread, Value const* args, Value& result) {
	Value from = force(thread, args[0]);
	thread.registers[0] = Logical::c(from.isList());
	return 1;
}
*/

void registerCoerceFunctions(State& state)
{
	state.registerInternalFunction(state.internStr("as.null"), (asnull), 1);
	state.registerInternalFunction(state.internStr("as.logical"), (aslogical), 1);
	state.registerInternalFunction(state.internStr("as.integer"), (asinteger), 1);
	state.registerInternalFunction(state.internStr("as.double"), (asdouble), 1);
	state.registerInternalFunction(state.internStr("as.numeric"), (asdouble), 1);
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
