
#include "gc.h"
#include "value.h"
#include "interpreter.h"
#include "api/api.h"

bool HeapObject::marked() const {
	return (gcObject()->flags & slot()) != 0;
}

void HeapObject::visit() const {
	gcObject()->flags |= slot();
}

uint64_t HeapObject::slot() const {
	assert(((uint64_t)this & 63) == 0);
	uint64_t s = ((uint64_t)this & (Heap::regionSize-1)) >> 6;
	assert(s >= 1 && s <= 63);
	return (((uint64_t)1) << s);
}

GCObject* HeapObject::gcObject() const {
	return (GCObject*)((uint64_t)this & ~(Heap::regionSize-1));
}


#define VISIT(p) if((p) != 0 && !(p)->marked()) (p)->visit()

static void traverse(Value const& v) {
	switch(v.type()) {
		case Type::Environment:
			VISIT(((REnvironment const&)v).attributes());
			VISIT(((REnvironment const&)v).environment());
			break;
		case Type::Closure:
			VISIT(((Closure const&)v).attributes());
			VISIT((Closure::Inner const*)v.p);
			VISIT(((Closure const&)v).prototype());
			VISIT(((Closure const&)v).environment());
			break;
        case Type::Externalptr:
            VISIT((Externalptr::Inner const*)v.p);
            traverse(((Externalptr const&)v).tag());
            traverse(((Externalptr const&)v).prot());
            break;
		case Type::Double:
			VISIT(((Double const&)v).attributes());
			VISIT(((Double const&)v).inner());
			break;
		case Type::Integer:
			VISIT(((Integer const&)v).attributes());
			VISIT(((Integer const&)v).inner());
			break;
		case Type::Logical:
			VISIT(((Logical const&)v).attributes());
			VISIT(((Logical const&)v).inner());
			break;
		case Type::Character:
			VISIT(((Character const&)v).attributes());
			VISIT(((Character const&)v).inner());
			break;
		case Type::Raw:
			VISIT(((Raw const&)v).attributes());
			VISIT(((Raw const&)v).inner());
			break;
		case Type::List:
			VISIT(((List const&)v).attributes());
			VISIT(((List const&)v).inner());
			{
				List const& l = (List const&)v;
				for(int64_t i = 0; i < l.length(); i++)
					traverse(l[i]);
			}
			break;
		case Type::Promise:
			VISIT(((Promise&)v).environment());
			if(((Promise&)v).isExpression())
				VISIT(((Promise&)v).code());
			break;
		case Type::Integer32:
			VISIT(((Integer32 const&)v).attributes());
			VISIT(((Integer32 const&)v).inner());
			break;
		case Type::Logical32:
			VISIT(((Logical32 const&)v).attributes());
			VISIT(((Logical32 const&)v).inner());
			break;
        case Type::Pairlist:
            VISIT((Pairlist::Inner const*)v.p);
            VISIT(((Pairlist const&)v).car());
            VISIT(((Pairlist const&)v).cdr());
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

void Environment::visit() const {
	Dictionary::visit();
    VISIT(enclosure);
    VISIT(attributes);
}

void Code::visit() const {
	HeapObject::visit();
	traverse(expression);
	
    for(uint64_t i = 0; i < constants.size(); i++) {
		traverse(constants[i]);
	}
	for(uint64_t i = 0; i < calls.size(); i++) {
		traverse(calls[i].call);
        traverse(calls[i].arguments);
        traverse(calls[i].names);
        traverse(calls[i].extraArgs);
        traverse(calls[i].extraNames);
	}
}

void Code::Finalize(HeapObject* o) {
    Code* code = (Code*)o;
    code->bc.clear();
    code->constants.clear();
    code->calls.clear();
}

void Prototype::visit() const {
	HeapObject::visit();
	VISIT(code);
	
    traverse(formals);
	traverse(parameters);
    traverse(defaults);
}

void SEXPREC::visit() const {
	HeapObject::visit();
    traverse(v);
}

void Heap::mark(Global& global) {
	// traverse root set
	// mark the region that I'm currently allocating into
    ((HeapObject*)bump)->visit();
	
	//printf("--global--\n");
	VISIT(global.global);
	traverse(global.arguments);
    VISIT(global.promiseCode);

    for(std::map<std::string, String>::const_iterator i = 
            global.strings.table().begin();
            i != global.strings.table().end();
            ++i) {
        VISIT(i->second);
    }

	for(std::list<State*>::iterator t = global.states.begin();
            t != global.states.end(); ++t) {
		State* state = *t;

		//printf("--stack--\n");
		for(uint64_t i = 0; i < state->stack.size(); i++) {
			VISIT(state->stack[i].code);
			VISIT(state->stack[i].environment);
		}
		//printf("--frame--\n");
		VISIT(state->frame.code);
		VISIT(state->frame.environment);

		//printf("--trace--\n");
		// traces only hold weak references...

		//printf("--registers--\n");
		for(Value const* r = state->registers; r < state->frame.registers+state->frame.code->registers; ++r) {
			traverse(*r);
		}

		//printf("--gc stack--\n");
		for(uint64_t i = 0; i < state->gcStack.size(); i++) {
			traverse(state->gcStack[i]);
		}
	}
	
    // R API support	
    for(std::list<SEXP>::iterator i = global.installedSEXPs.begin(); 
            i != global.installedSEXPs.end(); ++i) {
	    VISIT(*i);
	}

    if(global.apiStack) {
        for(int i = 0; i < *global.apiStack->size; ++i) {
            VISIT(global.apiStack->stack[i]);
        }
    }
}

void Heap::sweep() {
	//uint64_t old_total = total;
	total = 0;
	GCObject** g = &root;
	while(*g != 0) {
		GCObject* h = *g;
		if(!h->marked()) {
		//	//printf("Deleting %llx\n", h);
			*g = h->next;
            if(h->finalizer != 0) {
                h->finalizer((HeapObject*)h->data);
            }

			if(h->size == regionSize) {
				//printf("Freeing region %llx--%llx\n", h, h+h->size);
				//memset(h->data, 0xff, h->size);
				freeRegions.push_front(h);
			}
			else {
				//memset(h->data, 0xff, h->size);
				free(h->head);
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
	for(uint64_t i = 0; i < regions; i++) {
	    GCObject* r = ((HeapObject*)(head+regionSize-1))->gcObject();
		r->Init(head, regionSize);
		assert(((uint64_t)r & (regionSize-1)) == 0);
		freeRegions.push_back(r);
		head += regionSize;
	}
}

void Heap::popRegion() {
	//printf("Making new region: %d\n", freeRegions.size());
	if(freeRegions.empty())
		makeRegions(256);

	GCObject* g = freeRegions.front();
	freeRegions.pop_front();
	//printf("Popping %llx--%llx\n", g, g+g->size);
	total += g->size;
	root = g->Activate(root, 0);

	bump = (char*)(g->data);
	limit = ((char*)g) + regionSize;
}

Heap Heap::GlobalHeap;


