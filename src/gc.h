
#ifndef GC_H
#define GC_H

#include "type.h"
#include "common.h"
#include <assert.h>

#include <gc/gc_cpp.h>
#include <gc/gc_allocator.h>

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
	void* operator new(size_t size, State& state, size_t extra=0);
};

class Heap : public gc {
public:
        virtual HeapObject* mark(HeapObject*) = 0;
};

class Semispace : public Heap {

	static const uint64_t size = (1 << 20);
	char *head;
	char *base, *newbase;		// must be aligned by size
	char *bump;

	State* state;

public:

	Semispace(State* state);

	bool inSpace(HeapObject* p) const {
		return ((uint64_t)p & (~(size-1))) == (uint64_t)base;
	}

	HeapObject* alloc(uint64_t bytes) { 
		if(!inSpace((HeapObject*)(bump + bytes)))
			collect();
		HeapObject* result = (HeapObject*)bump;
		result->forward = 0;
		result->bytes = bytes;
		bump += bytes;
		printf("allocated %d at %llx\n", bytes, result);
		return result;
	}

	void collect();
	virtual HeapObject* mark(HeapObject* o);

};

#endif

