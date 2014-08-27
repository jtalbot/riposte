
#ifndef RIPOSTE_GC_H
#define RIPOSTE_GC_H

#include <deque>
#include "common.h"
#include <assert.h>

struct HeapObject;
typedef void (*GCFinalizer)(HeapObject*);

struct GCObject {
    void* head;
	uint64_t size;
	
    GCObject* next;
    GCFinalizer finalizer;
	
    uint64_t flags;
	uint64_t padding[3];
	char data[];

    void Init(void* h, uint64_t s) {
        head = h;
        size = s;
        next = 0;
        finalizer = 0;
        flags = 0;
    }
    
	GCObject* Activate(GCObject* n, GCFinalizer f) {
		next = n;
        finalizer = f;
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
	void* operator new(unsigned long bytes, GCFinalizer finalizer);
};

class Global;

class Heap {
private:

	GCObject* root;
	uint64_t heapSize;
	uint64_t total;

	void mark(Global& global);
	void sweep();
	
	void makeRegions(uint64_t regions);
	void popRegion();	

	std::deque<GCObject*> freeRegions;
	char* bump, *limit;

public:
	static const uint64_t regionSize = (1<<12);
	
    Heap() : root(0), heapSize(1<<20), total(0) {
		popRegion();
	}

	HeapObject* smallalloc(uint64_t bytes);
	HeapObject* alloc(uint64_t bytes, GCFinalizer finalizer = 0);
	
    void collect(Global& global);

	static Heap GlobalHeap;
};

inline HeapObject* Heap::smallalloc(uint64_t bytes) {
	// so the slot marking scheme works, objects have to take up
    // at least page size/slot = 64 bytes
    // this is actually only true for recursive objects so we could
    // be a bit more aggressive about this
    bytes = (bytes + 63) & (~63);
	if(bump+bytes >= limit)
		popRegion();
	
	//printf("Region: allocating %d at %llx\n", bytes, (uint64_t)bump);
	HeapObject* o = (HeapObject*)bump;
	assert(((uint64_t) o & 63) == 0);
	memset(o, 0xba, bytes);
	bump += bytes;
	return o;
}

inline HeapObject* Heap::alloc(uint64_t bytes, GCFinalizer finalizer) {
	bytes += sizeof(GCObject);
	
	total += bytes+regionSize;
	char* head = (char*)malloc(bytes+regionSize);
	memset(head, 0xab, bytes+regionSize);
	GCObject* g = ((HeapObject*)(head+regionSize-1))->gcObject();
	g->Init(head, bytes+regionSize);
    root = g->Activate(root, finalizer);

	return (HeapObject*)(g->data);
}

inline void Heap::collect(Global& global) {
    if(total > heapSize) {
        mark(global);
        sweep();
        if(total > heapSize*0.6 && heapSize < (1<<30))
            heapSize *= 2;
    }
}


inline void* HeapObject::operator new(unsigned long bytes) {
    assert(bytes <= 2048);
    return Heap::GlobalHeap.smallalloc(bytes);
}

inline void* HeapObject::operator new(unsigned long bytes, unsigned long extra) {
    unsigned long total = bytes + extra;
    return total <= 2048 ? 
        Heap::GlobalHeap.smallalloc(total) : 
        Heap::GlobalHeap.alloc(total);
}

inline void* HeapObject::operator new(unsigned long bytes, GCFinalizer finalizer) {
    return Heap::GlobalHeap.alloc(bytes, finalizer);
}
#endif

