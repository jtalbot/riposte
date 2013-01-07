
#include "rgc.h"
#include "value.h"
#include "interpreter.h"

bool HeapObject::marked() const {
	return (gcObject()->flags & slot()) != 0;
}

void HeapObject::visit() const {
	gcObject()->flags |= slot();
}

uint64_t HeapObject::slot() const {
	assert(((uint64_t)this & 63) == 0);
	uint64_t s = ((uint64_t)this & (PAGE_SIZE-1)) >> 6;
	assert(s >= 1 && s <= 63);
	return (((uint64_t)1) << s);
}

GCObject* HeapObject::gcObject() const {
	return (GCObject*)((uint64_t)this & ~(PAGE_SIZE-1));
	//return (GCObject*)((uint64_t)this - sizeof(GCObject));
}


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
	// mark the region that I'm currently allocating into
	((GCObject*)((uint64_t)bump & ~(PAGE_SIZE-1)))->flags |= 1;
	gcObject(bump)->flags |= 1;
	
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
	//uint64_t old_total = total;
	total = 0;
	void** g = &root;
	while(*g != 0) {
		void* t = *g;
		GCObject* h = gcObject(t);
		if(!h->marked()) {
		//	//printf("Deleting %llx\n", h);
			*g = h->next;
			if(h->size == 4096) {
				//printf("Freeing region %llx\n", t);
				//memset(t, 0xff, h->size);
				freeRegions.push_front(t);
			}
			else {
				//memset(t, 0xff, h->size);
				free(t);
			}
		} else {
			total += h->size;
			h->unmark();
			g = &(h->next);
		}
	}
	//printf("Swept: \t%d => \t %d\n", old_total, total);
}

void Heap::makeRegions(uint64_t regions) {
	char* head = (char*)malloc((regions+1)*regionSize);
	head = (char*)(((uint64_t)head+regionSize-1) & (~(regionSize-1)));
	for(uint64_t i = 0; i < regions; i++) {
		GCObject* r = (GCObject*)head;
		r->init(regionSize, 0);
		assert(((uint64_t)r & (regionSize-1)) == 0);
		freeRegions.push_back(r);
		head += regionSize;
	}
}

void Heap::popRegion() {
	//printf("Making new region: %d\n", freeRegions.size());
	if(freeRegions.empty())
		makeRegions(256);

	void* r = freeRegions.front();
	freeRegions.pop_front();
	GCObject* g = gcObject(r);
	//printf("Popping to %llx\n", g);
	g->init(regionSize, root);
	total += g->size;
	root = r;

	bump = (char*)(g->data);
	limit = ((char*)g) + regionSize;
}

Heap Heap::Global;

