
#ifndef RIPOSTE_GC_H
#define RIPOSTE_GC_H

#include <list>
#include "common.h"
#include <assert.h>

#define PAGE_SIZE 4096

struct GCObject {
	void* next;
	uint64_t size;
	uint64_t flags;
	uint64_t padding[5];
	char data[];

	void* init(uint64_t s, void* n) {
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

	bool marked() const;
	void visit() const;
	uint64_t slot() const;
	GCObject* gcObject() const;

	void* operator new(unsigned long bytes);
	void* operator new(unsigned long bytes, unsigned long extra);
};

class State;

class Heap {
private:
	static const uint64_t regionSize = (1<<12);

	void* root;
	uint64_t heapSize;
	uint64_t total;

	void mark(State& state);
	void sweep();
	
	void makeRegions(uint64_t regions);
	void popRegion();	

	std::list<void*> freeRegions;
	char* bump, *limit;

	GCObject* gcObject(void* v) const {
		return (GCObject*)(((uint64_t)v+(regionSize-1)) & (~(regionSize-1)));
	}

public:
	Heap() : root(0), heapSize(1<<20), total(0) {
		popRegion();
	}

	HeapObject* smallalloc(uint64_t bytes);
	HeapObject* alloc(uint64_t bytes);
	void collect(State& state);

	static Heap Global;
};

inline HeapObject* Heap::smallalloc(uint64_t bytes) {
	bytes = (bytes + 63) & (~63);
	if(bump+bytes >= limit)
		popRegion();
	
	//printf("Region: allocating %d at %llx\n", bytes, (uint64_t)bump);
	HeapObject* o = (HeapObject*)bump;
	assert(((uint64_t) o & 63) == 0);
	//memset(o, 0xba, bytes);
	bump += bytes;
	return o;
}

inline HeapObject* Heap::alloc(uint64_t bytes) {
	bytes += sizeof(GCObject);
	bytes = (bytes + 63) & (~63);
	
	total += bytes+regionSize;
	void* head = (void*)malloc(bytes+regionSize);
	//memset(head, 0xab, bytes+regionSize);
	GCObject* g = gcObject(head);
	assert(((uint64_t) g & 63) == 0);
	g->init(bytes+regionSize, root);
	root = head;

	return (HeapObject*)(g->data);
}

inline void Heap::collect(State& state) {
	if(total > heapSize) {
		mark(state);
		sweep();
		if(total > heapSize*0.6 && heapSize < (1<<30))
			heapSize *= 2;
	}
}


inline void* HeapObject::operator new(unsigned long bytes) {
	assert(bytes <= 2048);
	return Heap::Global.smallalloc(bytes);
}

inline void* HeapObject::operator new(unsigned long bytes, unsigned long extra) {
	unsigned long total = bytes + extra;
	return total <= 2048 ? 
		Heap::Global.smallalloc(total) : 
		Heap::Global.alloc(total);
}

#endif

