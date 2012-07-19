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

Environment* Thread::beginEval(Environment* lexicalScope, Environment* dynamicScope) {
	
	// Create a stack frame for interactive evaluation...
	push();
	frame.registers = frame.reservedTo;
	frame.calls = 0;

	Environment* env = 
		new Environment(lexicalScope, dynamicScope, new Shape(), 0, Null::Singleton());
	frame.environment = env;
	return env;
}

Value Thread::continueEval(Code const* code) {

	frame.environment->mutateShape(code->shape);

	// update reserved stack size with room for registers
	frame.reservedTo = frame.registers + code->registers;
	frame.calls = &code->calls[0];
	frame.ics = (IC*)&code->ics[0];
	
	// execute code in continuation's stack frame
	size_t oldStackDepth = stack.size();
	try {
		return (code->ptr)(this);
	}
	catch(...) {
		while(stack.size() > oldStackDepth)
			pop();
		throw;
	}
}

Environment* Thread::endEval(bool liveOut) {
	Environment* result = frame.environment;
	
	pop();

	return result;
}

Value Thread::eval(Code const* code, Environment* lexicalScope, Environment* dynamicScope) {
	beginEval(lexicalScope, dynamicScope);
	Value result = continueEval(code);
	endEval(false);
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
	frame.registers = frame.reservedTo = registers;
}

