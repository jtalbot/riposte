
#ifndef RIPOSTE_GC_H
#define RIPOSTE_GC_H

#include "common.h"

#define PAGE_SIZE 4096

struct GCObject {
	GCObject* next;
	uint64_t size;
	uint64_t flags;
	uint64_t padding;

	GCObject* init(uint64_t s, GCObject* n) {
		next = n;
		size = s;
		flags = 0;
		return this;
	}

	bool marked() const {
		return flags != 0;
	}

	void unmark() {
		flags = 0;
	}
};

struct HeapObject {
	
	bool marked() const { 
		return (gcObject()->flags & slot()) != 0;
	}

	void mark() const {
		gcObject()->flags |= slot();
	}

	uint64_t slot() const {
		return (1 << (((uint64_t)this & (PAGE_SIZE-1)) >> 6));
	}

	GCObject* gcObject() const {
		//return (GCObject*)((uint64_t)this & ~(PAGE_SIZE-1));
		return (GCObject*)((uint64_t)this - sizeof(GCObject));
	}

	virtual void visit() const = 0;

	void* operator new(unsigned long bytes);
	void* operator new(unsigned long bytes, unsigned long extra);
};

class State;

class Heap {
private:
	GCObject* root;
	uint64_t heapSize;
	uint64_t total;

	void mark(State& state);
	void sweep();
public:
	Heap() : root(0), heapSize(1<<20), total(0) {}

	HeapObject* alloc(uint64_t bytes);
	void collect(State& state);

	static Heap Global;
};

inline HeapObject* Heap::alloc(uint64_t bytes) {
	bytes += sizeof(GCObject);
	total += bytes;
	GCObject* g = (GCObject*)malloc(bytes);
	root = g->init(bytes, root);
	return (HeapObject*)((char*)g+sizeof(GCObject));
}

inline void Heap::collect(State& state) {
	if(total > heapSize) {
		mark(state);
		sweep();
		if(total > heapSize && heapSize < (1<<30))
			heapSize *= 2;
	}
}


inline void* HeapObject::operator new(unsigned long bytes) {
	return Heap::Global.alloc(bytes);
}

inline void* HeapObject::operator new(unsigned long bytes, unsigned long extra) {
	return Heap::Global.alloc(bytes+extra);
}

#endif

