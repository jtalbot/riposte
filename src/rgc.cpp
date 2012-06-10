
#include "rgc.h"
#include "value.h"
#include "interpreter.h"

#define VISIT(p) if((p) != 0 && !(p)->marked()) (p)->visit()

static void traverse(Value const& v) {
	switch(v.type) {
		case Type::Object:
			VISIT((Object::Inner const*)v.p);
			break;
		case Type::Environment:
			VISIT(((REnvironment&)v).environment());
			break;
		case Type::Function:
			VISIT(((Function&)v).prototype());
			VISIT(((Function&)v).environment());
			break;
		case Type::Promise:
			VISIT(((Promise&)v).prototype());
			VISIT(((Promise&)v).environment());
			break;
		case Type::Default:
			VISIT(((Default&)v).prototype());
			VISIT(((Default&)v).environment());
			break;
		case Type::Double:
			VISIT(((Double const&)v).inner());
			break;
		case Type::Integer:
			VISIT(((Integer const&)v).inner());
			break;
		case Type::Logical:
			VISIT(((Logical const&)v).inner());
			break;
		case Type::Character:
			VISIT(((Character const&)v).inner());
			break;
		case Type::Raw:
			VISIT(((Raw const&)v).inner());
			break;
		case Type::List:
			VISIT(((List const&)v).inner());
			{
				List const& l = (List const&)v;
				for(int64_t i = 0; i < l.length; i++)
					traverse(l[i]);
			}
			break;
		default:
			// do nothing
			break;
	}
}

/*void Dictionary::Inner::visit() const {
	for(uint64_t i = 0; i < size; i++) {
		if(d[i].n != Strings::NA)
			traverse(d[i].v);
	}
	// do nothing for now
}*/

void Dictionary::visit() const {
	HeapObject::visit();
	VISIT(d);
	for(uint64_t i = 0; i < size; i++) {
		if(d->d[i].n != Strings::NA)
			traverse(d->d[i].v);
	}
}

void Object::Inner::visit() const {
	HeapObject::visit();
	traverse(base);
	VISIT(d);
}

void Environment::visit() const {
	Dictionary::visit();
	VISIT(lexical);
	VISIT(dynamic);
	traverse(call);
	for(int64_t i = 0; i < dots.size(); i++) {
		traverse(dots[i].v);
	}
	/*for(uint64_t i = 0; i < size; i++) {
		if(d[i].n != Strings::NA)
			traverse(d[i].v);
	}*/
}

void Prototype::visit() const {
	HeapObject::visit();

	traverse(expression);
	for(int64_t i = 0; i < parameters.size(); i++) {
		traverse(parameters[i].v);
	}
	for(int64_t i = 0; i < constants.size(); i++) {
		traverse(constants[i]);
	}
	for(int64_t i = 0; i < calls.size(); i++) {
		traverse(calls[i].call);
		for(int64_t j = 0; j < calls[i].arguments.size(); j++) {
			traverse(calls[i].arguments[j].v);
		}
	}
	//for(int64_t i = 0; i < prototypes.size(); i++) {
	//	traverse(prototypes[i]);
	//}
}

void Heap::mark(State& state) {
	// traverse root set
	//printf("Marking\n");
	
	// iterate over path, then stack, then trace locations, then registers
	//printf("--path--\n");
	for(uint64_t i = 0; i < state.path.size(); i++) {
		VISIT(state.path[i]);
	}
	//printf("--global--\n");
	VISIT(state.global);
	traverse(state.arguments);

	for(uint64_t t = 0; t < state.threads.size(); t++) {
		Thread* thread = state.threads[t];

		//printf("--stack--\n");
		for(uint64_t i = 0; i < thread->stack.size(); i++) {
			VISIT(thread->stack[i].environment);
			VISIT(thread->stack[i].prototype);
		}
		//printf("--frame--\n");
		VISIT(thread->frame.environment);
		VISIT(thread->frame.prototype);

		//printf("--trace--\n");
		// traces only hold weak references...

		//printf("--registers--\n");
		for(Value const* r = thread->base-(thread->frame.prototype->registers); r < thread->registers+DEFAULT_NUM_REGISTERS; ++r) {
			traverse(*r);
		}

		for(uint64_t i = 0; i < thread->gcStack.size(); i++) {
			traverse(thread->gcStack[i]);
		}
	}
}

void Heap::sweep() {
	//printf("Sweeping %d\n", total);
	total = 0;
	GCObject** g = &root;
	while(*g != 0) {
		GCObject* h = *g;
		if(!h->marked()) {
		//	//printf("Deleting %llx\n", h);
			*g = h->next;
			assert(memset(h, 0xff, h->size) == h);
			free(h);	
		} else {
			total += h->size;
			h->unmark();
			g = &(h->next);
		}
	}
	//printf("Swept to %d\n", total);
}

Heap Heap::Global;

