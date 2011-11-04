
#include "gc.h"
#include "value.h"

Semispace::Semispace(State* state) : state(state) {
	head = new char[size*3];
	
	base = (char*)(((uint64_t)head+size) & (~(size-1)));
	newbase = base+size;

	bump = base;

	printf("Allocated semispace at %llx and %llx\n", base, newbase);
	assert(newbase + size <= head + size*3);
}

Semispace::~Semispace() {
	delete [] head;
	
	for(std::list<HeapObject*>::iterator i = lo.begin(); i != lo.end(); ++i) {
		munmap(*i, (*i)->bytes);
	}
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
	
	// TODO: track allocations into LOS. Trigger GC when that reaches some large value.
	// --Generational
	// --Non-moving
	// --Get std::vectors in Prototype into heap format
	// --Root stuff during parsing
	// --Compress representations
	// --Can we reduce the number of allocations necessary to parse simple input?
	// --

	printf("Running the collector\n");
	// allocate new space, bigger if need be
	// iterate over roots copying stuff over
	// leave behind forwarding pointer
	// bump needs to point to end of copied data

	// clear marks in the lo space
	for(std::list<HeapObject*>::iterator i = lo.begin(); i != lo.end(); ++i) {
		(*i)->forward = 0;
	}

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
		state->stack[i].prototype.p = (void*)mark((HeapObject*)(state->stack[i].prototype.p));
	}
	//printf("--frame--\n");
	state->frame.environment.p = (Environment*)mark((HeapObject*)(state->frame.environment.p));
	state->frame.prototype.p = (void*)mark((HeapObject*)(state->frame.prototype.p));

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
		walkValue(this, *r);
	}

	//printf("--handles--\n");
	for(Value* r = state->handleStack; r < state->hsp; r++) {
		walkValue(this, *r);
	}
	
	//printf("--finger--\n");
	// now walk from newbase to bump walking each heap object
	char* finger = newbase;
	while(finger < bump) {
		HeapObject* hp = (HeapObject*)finger;
		hp->walk(this);
		finger += hp->bytes;
	}

	// sweep the lo space
	for(std::list<HeapObject*>::iterator i = lo.begin(); i != lo.end();) {
		if(!(*i)->forward) { munmap(*i, (*i)->bytes); lo.erase(i++); }
		else i++;
	}

	// clear old subspace so Boehm doesn't keep stuff around
	memset(base, 0xff, size);
	
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
	} else if(o != 0) {
		//printf("non-nursery item: %llx\n", o);
		// mark in lo space
		o->forward = 1;
	}
	return o;
}
/*
MarkRegion::MarkRegion(State* state) : state(state) {
	existingRegions = 0;
	makeRegions(128);
}

void MarkRegion::makeRegions(uint64_t regions) {

	// allocate one extra so we can align the memory
	char* head = new (GC) char[(regions+1)*rSize];
	head = (char*)(((uint64_t)head+rSize-1) & (~(rSize-1)));
	for(uint64_t i = 0; i < regions; i++) {
		ar.push(Region(head));
		head += rSize;
	}
	existingRegions += regions;
}

HeapObject* alloc(uint64_t bytes) {
	if(bump+bytes < limit) {
		HeapObject* r = (HeapObject*)bump;
		r->forward = 0;
		r->bytes = bytes;
		bump += bytes;
		return bump;
	} else {
		return slowAlloc(uint64_t bytes);
	}
}

uint64_t MarkRegion::bitInRegion(Region& r, char* p) {
	return (p-r.ptr)>>5;
}

Region MarkRegion::getFreeRegion() {
	if(ar.size() == 0) makeRegions(existingRegions*0.5);
	Region result = ar.back();
	ar.pop_back();
	return result;
}

void MarkRegion::advanceBump() {
	// figure out which bit we're in...
	uint64_t bir = bitInRegion(*currentRegion, bump);
	uint64_t block = bir >> 6;
	uint64_t bit = bir & 15;
	// then find the next free line
	while(true) {
		for(uint64_t cblock = block; cblock < 16; cblock++) {
			// this check is redundant if full blocks are always pulled out of the list by the marking scheme
			if(currentRegion->mark[cblock] != 0xFFFFFFFFFFFFFFFFF) {
				for(uint64_t cbit = bit; cbit < 64; cbit++) {
					if(~currentRegion->mark[cblock] & (1 << cbit)) {
						bump = currentRegion->ptr + ((cblock << 6 + cbit)<<5);
						return;
					}
				}
			}
		}
		currentRegion++;
		if(currentRegion == br.end()) {
			br.push_back(getFreeRegion());
			currentRegion = br.back();
		}
		block = 0;
		bit = 0;
	}
}

void MarkRegion::advanceLimit() {
	// figure out which bit we're in...
	uint64_t bir = bitInRegion(*currentRegion, bump);
	uint64_t block = bir >> 6;
	uint64_t bit = bir & 15;
	// then find the next non-free line or the end of the current block
	for(uint64_t cblock = block; cblock < 16; cblock++) {
		// this check is redundant if full blocks are always pulled out of the list by the marking scheme
		if(currentRegion->mark[cblock] != 0x0) {
			for(uint64_t cbit = bit; cbit < 64; cbit++) {
				if(currentRegion->mark[cblock] & (1 << cbit)) {
					limit = currentRegion->ptr + ((cblock << 6 + cbit)<<5);
					return;
				}
			}
		}
	}
	limit = currentRegion->ptr + rSize;
}

HeapObject* MarkRegion::slowAlloc(uint64_t bytes) {

	HeapObject* r;

	// something here needs to trigger a collection

	// look for the next free line
	if(bytes <= lSize) {
		advanceBump();
		advanceLimit();
		r = (HeapObject*)bump;
		bump += bytes;
	}
	// we've already failed to place it in the next available space, so
	// if it's bigger than a line, place in free region.
	else {
		if(cbump+bytes >= climit) {
			cr.push_back(getFreeRegion());
			cbump = cr.back().ptr;
			climit = cr.back().ptr+rSize;
		}
		r = (HeapObject*)cbump;
		cbump += bytes;
	}
	r->forward = 0;
	r->bytes = bytes;
	return bump;
}

void MarkRegion::collect() {
	// clear all mark bits
	// call mark recursively
}

HeapObject* MarkRegion::mark(HeapObject* o) {
	// figure out which region we're in...
	o >> 15 << 15;
	// if not marked yet...
	// 	mark relevant bits given the size of our object
	//	recursively mark children
	// if marked already (implies that its already been visited AND it (and its children) are unchanged)...
	//	just return
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
*/
