
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
			break;
		case Type::Environment:
			traverse(((REnvironment&)v).environment());
			break;
		case Type::Function:
			traverse(((Function&)v).prototype());
			traverse(((Function&)v).environment());
			break;
		case Type::Promise:
			traverse(((Promise&)v).prototype());
			traverse(((Promise&)v).environment());
			break;
		case Type::Default:
			traverse(((Default&)v).prototype());
			traverse(((Default&)v).environment());
			break;
		/*case Type::Double:
			traverse(((Double const&)v).inner());
			break;
		case Type::Integer:
			traverse(((Integer const&)v).inner());
			break;
		case Type::Logical:
			traverse(((Logical const&)v).inner());
			break;
		case Type::Character:
			traverse(((Character const&)v).inner());
			break;
		case Type::Raw:
			traverse(((Raw const&)v).inner());
			break;
		case Type::List:
			traverse(((List const&)v).inner());
			break;*/
		default:
			// do nothing
			break;
	}
}

void Dictionary::Inner::visit() const {
	/*for(uint64_t i = 0; i < size; i++) {
		if(d[i].n != Strings::NA)
			traverse(d[i].v);
	}*/
	// do nothing for now
}

void Dictionary::visit() const {
	traverse(d);
	for(uint64_t i = 0; i < size; i++) {
		if(d->d[i].n != Strings::NA)
			traverse(d->d[i].v);
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
	/*for(uint64_t i = 0; i < size; i++) {
		if(d[i].n != Strings::NA)
			traverse(d[i].v);
	}*/
}

void Prototype::visit() const {
	traverse(expression);
	for(int64_t i = 0; i < parameters.size(); i++) {
		traverse(parameters[i].v);
	}
	for(int64_t i = 0; i < constants.size(); i++) {
		traverse(constants[i]);
	}
	for(int64_t i = 0; i < calls.size(); i++) {
		for(int64_t j = 0; j < calls[i].arguments.size(); j++) {
			traverse(calls[i].arguments[j].v);
		}
	}
	//for(int64_t i = 0; i < prototypes.size(); i++) {
	//	traverse(prototypes[i]);
	//}
}

void CompiledCall::visit() const {
}
/*
template<>
void Vector<Type::List, Value, true>::Inner::visit() const {
	for(int64_t i = 0; i < length; i++) {
		traverse(data[i]);
	}
}
*/
void Heap::mark(State& state) {
	// traverse root set
	//printf("Marking\n");
	
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
	printf("Sweeping %d\n", total);
	total = 0;
	GCObject** g = &root;
	while(*g != 0) {
		GCObject* h = *g;
		if(!h->marked()) {
			*g = h->next;
			assert(memset(h, 0xff, h->size) == h);
			free(h);	
		} else {
			total += h->size;
			h->unmark();
			g = &(h->next);
		}
	}
	printf("Swept to %d\n", total);
}

Heap Heap::Global;

