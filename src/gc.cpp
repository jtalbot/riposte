
#include "gc.h"
#include "value.h"

Semispace::Semispace(State* state) : state(state) {
	head = new (GC) char[size*3];
	
	base = (char*)(((uint64_t)head+size) & (~(size-1)));
	newbase = base+size;

	bump = base;

	printf("Allocated semispace at %llx and %llx\n", base, newbase);
	assert(newbase + size <= head + size*3);
}

/*
HeapObject* Semispace::alloc(Type::Enum type, uint64_t bytes) {
	if(!inSpace((HeapObject*)(bump + bytes)))
		collect();
	HeapObject* result = (HeapObject*)bump;
	//result->type = type;
	//result->bytes = bytes;
	bump += bytes;
	return result;
}
*/

void Semispace::collect() {
	printf("Running the collector\n");
	// allocate new space, bigger if need be
	// iterate over roots copying stuff over
	// leave behind forwarding pointer
	// bump needs to point to end of copied data

	bump = newbase;

	// iterate over path, then stack, then trace locations, then registers
	//printf("--path--\n");
	for(uint64_t i = 0; i < state->path.size(); i++) {
		state->path[i].p = (Environment*)mark((HeapObject*)(state->path[i].p));
	}
	//printf("--global--\n");
	state->global.p = (Environment*)mark((HeapObject*)(state->global.p));

	//printf("--stack--\n");
	for(uint64_t i = 0; i < state->stack.size(); i++) {
		state->stack[i].environment.p = (Environment*)mark((HeapObject*)(state->stack[i].environment.p));
	}
	//printf("--frame--\n");
	state->frame.environment.p = (Environment*)mark((HeapObject*)(state->frame.environment.p));

	//printf("--trace--\n");
	Trace::Output* op = state->tracing.current_trace.outputs;
	for(uint64_t i = 0; i < state->tracing.current_trace.n_outputs; i++) {
		if(op->location.type == Trace::Location::VAR) {
			op->location.pointer.env.p = (Environment*)mark((HeapObject*)(op->location.pointer.env.p));
		}
		op++;
	}

	//printf("--registers--\n");
	for(Value* r = state->sp; r < state->registers+DEFAULT_NUM_REGISTERS; r++) {
		if(r->isEnvironment() || r->isFunction() || r->isPromise() || r->type == Type::HeapObject) {
			r->p = (void*)mark((HeapObject*)(r->p));
		}
	}
	
	//printf("--finger--\n");
	// now walk from newbase to bump walking each heap object
	char* finger = newbase;
	while(finger < bump) {
		HeapObject* hp = (HeapObject*)finger;
		hp->walk(this);
		finger += hp->bytes;
	}

	// clear old subspace so Boehm doesn't keep stuff around
	//memset(base, 0xF, size);
	
	char* t = base;
	base = newbase;
	newbase = t;
	printf("Finished running the collector\n");
}

HeapObject* Semispace::mark(HeapObject* o) {
	// only copy to new space if in old space to start
	if(inSpace((HeapObject*)o)) {
		if(o->forward) {
			return (HeapObject*)o->bytes;
		}

		// should check if room in new space I guess?
		assert(bump + o->bytes < newbase + size);
	
		memcpy(bump, o, o->bytes);
		HeapObject* result = (HeapObject*)bump;
		bump += o->bytes;
		
		o->forward = 1;
		o->bytes = (uint64_t)result;

		printf("copying from %llx to %llx\n", o, result);

		return result;
	}
	return o;
}


