
#ifndef RIPOSTE_GC_H
#define RIPOSTE_GC_H

#include <deque>
#include "common.h"
#include <assert.h>

const uint64_t CELL_SIZE = 32;
const uint64_t REGION_SIZE = 4096;

struct HeapObject;
typedef void (*GCFinalizer)(HeapObject*);

struct GCObject
{
    static const uint64_t WORDS = REGION_SIZE/CELL_SIZE/64;

    uint64_t mark[WORDS];
    uint64_t block[WORDS];
    char data[];

    void Init() {
        memset(&mark, 0, WORDS*8);
        memset(&block, 0, WORDS*8);
    }
    
    bool marked() const {
        for(uint64_t i = 0; i < WORDS; ++i)
            if(block[i]) return true;
        return false;
    }

    void sweep();
};

class Heap;

struct HeapObject
{
    bool marked() const;
    void visit() const;
    void block() const;
    uint64_t word() const;
    GCObject* gcObject() const;

    void* operator new(unsigned long bytes, Heap& heap);
    void* operator new(unsigned long bytes, unsigned long extra, Heap& heap);
    void* operator new(unsigned long bytes, GCFinalizer finalizer, Heap& heap);

    void* operator new(unsigned long bytes);
    void* operator new(unsigned long bytes, unsigned long extra);
    void* operator new(unsigned long bytes, GCFinalizer finalizer);
};

class Global;

class Heap
{
private:

    uint64_t heapSize;
    uint64_t total;

    void mark(Global& global);
    void sweep(Global& global);
    
    void makeArenas(uint64_t regions);
    void popRegion(uint64_t bytes);    

    std::deque<GCObject*> arenas;
    std::deque<GCObject*> blocks;
    std::deque< std::pair<HeapObject*, GCFinalizer> > finalizers;

    char* bump, *limit;
    uint64_t arenaIndex;

public:
    Heap();

    HeapObject* smallalloc(uint64_t bytes);
    HeapObject* alloc(uint64_t bytes);

    HeapObject* addFinalizer(HeapObject*, GCFinalizer);
 
    void collect(Global& global);

    static Heap GlobalHeap;
    static Heap ConstHeap;
};

inline HeapObject* Heap::smallalloc(uint64_t bytes)
{
    // Round up to a multiple of the cell size
    bytes = (bytes + (CELL_SIZE-1)) & (~(CELL_SIZE-1));

    if(bump+bytes > limit)
        popRegion(bytes);
    
    assert(((uint64_t) bump & (CELL_SIZE-1)) == 0);
    
    HeapObject* o = (HeapObject*)bump;
    o->block();
    memset(o, 0xba, bytes);

    bump += bytes;
    return o;
}

ALWAYS_INLINE
void Heap::collect(Global& global)
{
    if(total > heapSize)
    {
        mark(global);
        sweep(global);
        if(total > heapSize*0.6 && heapSize < (1<<30))
            heapSize *= 2;
    }
}


ALWAYS_INLINE
void* HeapObject::operator new(unsigned long bytes, Heap& heap)
{
    assert(bytes <= (REGION_SIZE/2));
    return heap.smallalloc(bytes);
}

ALWAYS_INLINE
void* HeapObject::operator new(unsigned long bytes, unsigned long extra, Heap& heap)
{
    unsigned long total = bytes + extra;
    return total <= (REGION_SIZE/2) ? 
        heap.smallalloc(total) : 
        heap.alloc(total);
}

ALWAYS_INLINE
void* HeapObject::operator new(unsigned long bytes, GCFinalizer finalizer, Heap& heap)
{
    assert(bytes <= (REGION_SIZE/2));
    return heap.addFinalizer( heap.smallalloc(bytes), finalizer );
}


inline void* HeapObject::operator new(unsigned long bytes)
{
    return HeapObject::operator new(bytes, Heap::GlobalHeap);
}

inline void* HeapObject::operator new(unsigned long bytes, unsigned long extra) {
    return HeapObject::operator new(bytes, extra, Heap::GlobalHeap);
}

inline void* HeapObject::operator new(unsigned long bytes, GCFinalizer finalizer)
{
    return HeapObject::operator new(bytes, finalizer, Heap::GlobalHeap);
}

#endif

