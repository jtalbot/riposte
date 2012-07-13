#include <string>
#include <sstream>
#include <stdexcept>
#include <string>

#include "value.h"
#include "type.h"
#include "bc.h"
#include "ops.h"
#include "internal.h"
#include "interpreter.h"
#include "sse.h"
#include "call.h"

void State::interpreter_init(Thread& thread) {
	// nothing for now
}

void Thread::beginEval(Environment* environment) {
	// Create a stack frame for interactive evaluation...
	push();
	frame.environment = environment;
	// Point env's slots to stack
	frame.slots = frame.reservedTo;
	environment->slots = frame.reservedTo;
}

Value Thread::continueEval(Code const* code) {
	// TODO: merge with buildStackFrame?
	// reserve stack space for registers after slots
	frame.registers = frame.slots + code->layout->m.size();
	// update reserved stack size
	frame.reservedTo = frame.registers + code->registers;
	frame.calls = &code->calls[0];
	
	// execute code in continuation's stack frame
	Value result = (code->ptr)(this);
	
	return result;
}

void Thread::endEval() {
	// pop stack frame
	pop();
}

Value Thread::eval(Code const* code, Environment* environment) {
	beginEval(environment);
	Value result = continueEval(code);
	endEval();
	return result;
}

const int64_t Random::primes[100] = 
{2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43,
47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131,
137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223,
227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311,
313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409,
419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503,
509, 521, 523, 541};

Thread::Thread(State& state, uint64_t index) : state(state), index(index), random(index),steals(1) {
	registers = new Value[DEFAULT_NUM_REGISTERS];
	frame.slots = frame.registers = frame.reservedTo = registers;
}

