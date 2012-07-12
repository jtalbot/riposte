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
#include "compiler.h"
#include "sse.h"
#include "call.h"

typedef Value const* (*RunPtr)(Thread* thread);

void State::interpreter_init(Thread& thread) {
	// nothing for now
}

Value Thread::eval(Prototype const* prototype) {
	return eval(prototype, frame.environment);
}

Value Thread::eval(Prototype const* prototype, Environment* environment) {
	uint64_t stackSize = stack.size();

	// make room for the result
	buildStackFrame(*this, environment, prototype, 0, (Instruction const*)0);
	try {
		//interpret(*this, run);
		timespec a = get_time();
		Value result = *((RunPtr)(prototype->func))(this);
		printf("Time elapsed: %f\n", time_elapsed(a));
		assert(stackSize == stack.size());
		return result;
	} catch(...) {
		stack.resize(stackSize);
		throw;
	}
}

const int64_t Random::primes[100] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43,
47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131,
137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223,
227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311,
313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409,
419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503,
509, 521, 523, 541};

Thread::Thread(State& state, uint64_t index) : state(state), index(index), random(index),steals(1) {
	registers = new Value[DEFAULT_NUM_REGISTERS];
	frame.registers = registers;
}

void Prototype::printByteCode(Prototype const* prototype, State const& state) {
	std::cout << "Prototype: " << intToHexStr((int64_t)prototype) << std::endl;
	std::cout << "\tRegisters: " << prototype->registers << std::endl;
	if(prototype->constants.size() > 0) {
		std::cout << "\tConstants: " << std::endl;
		for(int64_t i = 0; i < (int64_t)prototype->constants.size(); i++)
			std::cout << "\t\t" << i << ":\t" << state.stringify(prototype->constants[i]) << std::endl;
	}
	if(prototype->bc.size() > 0) {
		std::cout << "\tCode: " << std::endl;
		for(int64_t i = 0; i < (int64_t)prototype->bc.size(); i++) {
			std::cout << std::hex << &prototype->bc[i] << std::dec << "\t" << i << ":\t" << prototype->bc[i].toString();
			if(prototype->bc[i].bc == ByteCode::call || prototype->bc[i].bc == ByteCode::fastcall) {
				std::cout << "\t\t(arguments: " << prototype->calls[prototype->bc[i].b].arguments.size() << ")";
			}
			std::cout << std::endl;
		}
	}
	std::cout << std::endl;
}


