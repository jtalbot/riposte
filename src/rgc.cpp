
#include "rgc.h"
#include "value.h"
#include "interpreter.h"

static void traverse(HeapObject const* ho) {
	if(ho != 0 && !ho->marked()) {
		ho->mark();
		ho->visit();
	}
}

static void traverse(Value const& v) {
	switch(v.type) {
		case Type::Object:
			traverse(((Object&)v).base());
			traverse(((Object&)v).dictionary());
		case Type::Environment:
			traverse(((REnvironment&)v).ptr());
			break;
		case Type::Function:
			traverse(((Function&)v).prototype());
			traverse(((Function&)v).environment());
			break;
		default:
			// do nothing
			break;
	}
}

void Dictionary::visit() const {
	for(uint64_t i = 0; i < size; i++) {
		if(d[i].n != Strings::NA)
			traverse(d[i].v);
	}
}

void Environment::visit() const {
	traverse(lexical);
	traverse(dynamic);
	traverse(call);
	for(int64_t i = 0; i < dots.size(); i++) {
		traverse(dots[i].v);
	}
	Dictionary::visit();
}

void Prototype::visit() const {
	/*traverse(expression);
	for(int64_t i = 0; i < dots.size(); i++) {
		traverse(parameters[i].v);
	}
	for(int64_t i = 0; i < constants.size(); i++) {
		traverse(constants[i]);
	}
	for(int64_t i = 0; i < prototypes.size(); i++) {
		traverse(prototypes[i]);
	}*/
}

void Heap::mark(State& state) {
	// traverse root set

	// iterate over path, then stack, then trace locations, then registers
	//printf("--path--\n");
	for(uint64_t i = 0; i < state.path.size(); i++) {
		traverse(state.path[i]);
	}
	//printf("--global--\n");
	traverse(state.global);

	for(uint64_t t = 0; t < state.threads.size(); t++) {
		Thread* thread = state.threads[t];

		//printf("--stack--\n");
		for(uint64_t i = 0; i < thread->stack.size(); i++) {
			traverse(thread->stack[i].environment);
			traverse(thread->stack[i].prototype);
		}
		//printf("--frame--\n");
		traverse(thread->frame.environment);
		traverse(thread->frame.prototype);

		//printf("--trace--\n");
		// traces only hold weak references...

		//printf("--registers--\n");
		for(Value* r = thread->base; r < thread->registers+DEFAULT_NUM_REGISTERS; ++r) {
			traverse(*r);
		}
	}
}

void Heap::sweep() {
	total = 0;
	GCObject** g = &root;
	while(*g != 0) {
		GCObject* h = *g;
		if(!h->marked()) {
			*g = h->next;
			delete h;	
		} else {
			total += h->size;
			h->unmark();
			g = &(h->next);
		}
	}
}

Heap Heap::Global;

