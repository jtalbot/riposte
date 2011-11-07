
#ifndef GC_H
#define GC_H

#include "type.h"
#include "common.h"
#include "exceptions.h"
#include <assert.h>
#include <sys/mman.h>

#include <list>
#include <vector>

/*

	GC strategy...
		Copying semispace collector for small objects
		Mark and sweep collector for large objects
		This won't work for very large lists, but we'll delay that for now

*/
// roots...
	// registers
	// stack frame's environments and prototypes
	// environments in the path
	// parser stack...will have to figure out how to walk this.

/*
	Simplified approach:
		GC only environments allocated on function calls for now. Everything else will be allocated through Boehm
		Value -> Environment -> Pair[]
		My GC can allocate pairs in block allocated from Boehm, will have to wipe memory afterwards to avoid holding old pointers
	
		Copying collector with two spaces
		Bump pointer allocation
		When we reach the maximum size and can't allocate,
			run through roots and copy over environments
			then run through new space finding references back to the old space and copy environments over...
				have to ignore pointers to new space
				have to leave forwarding pointers behind
			within an environment have to copy
				lexical and dynamic envs iff in old space
				any REnv entry in map
				any env of a function
		Should do a tail call instruction eventually...
*/

class State;
class Heap;
class Semispace;

struct HeapObject {
        uint64_t forward:1;
	uint64_t bytes;

	virtual void walk(Heap*) = 0;
	void* operator new(size_t size, State& state);
	void* operator new(size_t size, State& state, size_t extra);
};

class Semispace {

	static const uint64_t size = (1 << 20);
	char *head;
	char *base, *newbase;		// must be aligned by size
	char *bump;

	std::list<HeapObject*> lo;	// large object space

	State* state;

public:

	Semispace(State* state);
	~Semispace();

	bool inSpace(HeapObject* p) const {
		return ((uint64_t)p & (~(size-1))) == (uint64_t)base;
	}

	// should only be used with small allocations
	HeapObject* alloc(Heap* heap, uint64_t bytes) {
		bytes = (bytes + 15) & (~15);
		if(!inSpace((HeapObject*)(bump + bytes)))
			collect(heap);
		HeapObject* result = (HeapObject*)bump;
		result->forward = 0;
		result->bytes = bytes;
		bump += bytes;
		printf("allocated %lld at %llx\n", bytes, (uint64_t)result);
		return result;
	}
	
	// used with things that can potentially be large allocations
	HeapObject* varalloc(Heap* heap, uint64_t bytes) {
		bytes = (bytes + 15) & (~15);
		if(bytes < (1<<11)) {
			if(!inSpace((HeapObject*)(bump + bytes)))
				collect(heap);
			if(!inSpace((HeapObject*)(bump + bytes)))
				_error("Out of memory in the nursery");
			HeapObject* result = (HeapObject*)bump;
			result->forward = 0;
			result->bytes = bytes;
			bump += bytes;
			printf("varallocated %lld at %llx\n", bytes, (uint64_t)result);
			return result;
		} else {
			HeapObject* result = (HeapObject*)mmap(0, bytes, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANON, -1, 0);	
			result->forward = 0;
			result->bytes = bytes;
			lo.push_back(result);
			printf("lo allocated %lld at %llx\n", bytes, (uint64_t)result);
			return result;
		}
	}


	void collect(Heap* heap);
	HeapObject* mark(Heap* heap, HeapObject* o);

};

// A brain dead MarkSweep collector for debugging purposes.
// In debug mode, fills dead memory with 0xFF to help detect dangling pointers.
class MarkSweep {
private:
	Heap& heap;
	std::list<HeapObject*> data;
	uint64_t total;
	uint64_t heapSize;
public:
	MarkSweep(Heap& heap) : heap(heap), total(0), heapSize(1<<21) {}
	~MarkSweep() {}

	HeapObject* alloc(uint64_t bytes);
	HeapObject* mark(HeapObject* o);
	void clear();
	void sweep();
};

class MarkRegion {
private:
	static const uint64_t regionSize = 1 << 16;
	Heap& heap;
	struct Region {
		uint64_t mark;
		uint64_t count;
		char data[];
	};
	char *bump, *limit;
	Region* current;
	std::list<Region*> empty, used;
public:
	MarkRegion(Heap& heap) : heap(heap) {
		makeRegions(32);
		popRegion();
	}
	~MarkRegion() {/* should delete the regions*/}
	
	HeapObject* alloc(uint64_t bytes);
	HeapObject* mark(HeapObject* o);
	void clear();
	void sweep();

	void makeRegions(uint64_t regions);
	void popRegion();
};

class Heap {
private:
	State* state;
	MarkSweep space;
	MarkRegion region;
	std::vector<HeapObject*> stack;
public:
	Heap(State* state) : state(state), space(*this), region(*this) {}

	// should only be used with small allocations
	HeapObject* alloc(uint64_t bytes) { assert(bytes < 4096); return region.alloc(bytes); }
	
	// used with things that can potentially be large allocations
	HeapObject* varalloc(uint64_t bytes) { if(bytes < 4096) return region.alloc(bytes); else return space.alloc(bytes); }

	HeapObject* mark(HeapObject* o) { if(o == 0) return 0; if(o->bytes < 4096) return region.mark(o); else return space.mark(o); }
	void push(HeapObject* o) { stack.push_back(o); }
	void collect();
};


inline HeapObject* MarkSweep::alloc(uint64_t bytes) {
	total += bytes;
	if(total > heapSize) {
		heap.collect();
		if(total > heapSize/2) {
			heapSize *= 2;
		} 
	}
	HeapObject* result = (HeapObject*)malloc(bytes);
	result->forward = 0;
	result->bytes = bytes;
	data.push_back(result);
	return result;
}

inline HeapObject* MarkSweep::mark(HeapObject* o) {
	if(o->forward == 0) {
		heap.push(o);
        }
        o->forward = 1;
       	return o;
}

inline HeapObject* MarkRegion::alloc(uint64_t bytes) {
	bytes = (bytes + 15) & (~15);
	if(bump+bytes >= limit)
		popRegion();
	//printf("Region: allocating %d at %llx\n", bytes, (uint64_t)bump);
	HeapObject* o = (HeapObject*)bump;
	o->forward = 0;
	o->bytes = bytes;
	bump += bytes;
	current->count++;
	return o;
}

inline HeapObject* MarkRegion::mark(HeapObject* o) {
	if(o->forward == 0) {
		heap.push(o);
	}
	o->forward = 1;

	// round to region header and mark region
	Region* r = (Region*)(((uint64_t)o) & (~(regionSize-1)));
	r->mark = 1;
	return o;
}

inline void MarkRegion::popRegion() {
	if(empty.size() == 0) {
		heap.collect();
		if(empty.size() < 8)
			makeRegions(32);
	}
	Region* r = empty.front();
	empty.pop_front();
	used.push_back(r);
	bump = r->data;
	limit = ((char*)r) + regionSize;
	r->mark = 0;
	r->count = 0;
	current = r;
}
/*
class MarkRegion : public Heap {

	State* state;

	// 16*64 32B lines in a region
	static const uint64_t rSize = 2 << 15;
	static const uint64_t lSize = 2 << 5;

	struct Region {
		// marks
		uint64_t mark[16];
		// start
		char* ptr;

		Region(char* ptr) : ptr(ptr) {
			for(uint64_t i = 0; i < 16; i++) mark = 0;
		}
	};
	
	// lists of regions
	std::list<Region> ar, br, cr, dr;

	MarkRegion(State* state);

	uint64_t existingRegions;
	void makeRegions(uint64_t regions);

	std::list<Region>::iterator currentRegion;
	char *bump, *limit;
	char *cbump, *climit;

	HeapObject* alloc(uint64_t bytes);
	HeapObject* slowAlloc(uint64_t bytes);
	virtual HeapObject* mark(HeapObject* o);

	Region getFreeRegion();
	uint64_t bitInRegion(Region& r, char* p);

	void advanceBump();
	void advanceLimit();
};
*/
#endif

