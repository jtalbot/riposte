
#ifndef RIPOSTE_GC_H
#define RIPOSTE_GC_H

#include <deque>
#include <map>
#include <unordered_map>

#include "common.h"
#include <assert.h>

const uint64_t CELL_SIZE = 32;
const uint64_t REGION_SIZE = 65536;
const uint64_t CELL_COUNT = REGION_SIZE/CELL_SIZE;

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
    void sweepMinor();
};

class Heap;

struct HeapObject
{
    bool marked() const;
    uint64_t word() const;
    
    GCObject* gcObject() const {
        return (GCObject*)((uint64_t)this & ~(REGION_SIZE-1));
    }


    void block() const
    {
        uint64_t i = ((uint64_t)this & (REGION_SIZE-1)) / CELL_SIZE;
        uint64_t slot = i % 64;
        uint64_t word = i / 64;
    
        gcObject()->mark[word] &= ~(((uint64_t)1) << slot);
        gcObject()->block[word] |= (((uint64_t)1) << slot);
    }

    bool visit() const;

    void* operator new(unsigned long bytes, Heap& heap);
    void* operator new(unsigned long bytes, unsigned long extra, Heap& heap);
    void* operator new(unsigned long bytes, GCFinalizer finalizer, Heap& heap);

    void* operator new(unsigned long bytes);
    void* operator new(unsigned long bytes, unsigned long extra);
    void* operator new(unsigned long bytes, GCFinalizer finalizer);
};


struct GrayHeapObject : HeapObject
{
public:
    uint8_t type;
    mutable bool gray;
    GrayHeapObject(uint8_t type) : type(type), gray(true) {}

    void writeBarrier() const;
    bool visit() const;
};

struct Arena
{
    Arena(GCObject* ptr) : ptr(ptr) {}

    GCObject* ptr;
};

struct LargeArena : Arena
{
    LargeArena(GCObject* ptr, uint64_t bytes)
        : Arena(ptr), bytes(bytes) {}

    uint64_t bytes;
};

class Global;

class Heap
{
private:

    void makeArenas(uint64_t regions);
    void popRegion(uint64_t bytes);    

    std::deque<Arena>      arenas;
    std::deque<LargeArena> larges;
    std::deque< std::pair<HeapObject*, GCFinalizer> > finalizers;

    std::multimap<uint64_t, char*> freeBlocks;

    char* bump, *limit;
    uint64_t arenaIndex, arenaOffset;

public:
    Heap();

    uint64_t sweep();
    
    HeapObject* smallalloc(uint64_t bytes);
    HeapObject* alloc(uint64_t bytes);

    HeapObject* addFinalizer(HeapObject*, GCFinalizer);

    bool contains(HeapObject const* o) const
    {
        GCObject const* g = o->gcObject();
        
        if(arenas.size() == 1 && larges.size() == 0)
            return g == arenas.back().ptr;
        else
            return containsSlow(g);
    }

    bool containsSlow(GCObject const* g) const;
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
    //memset(o, 0xba, bytes);

    bump += bytes;
    return o;
}

class Memory
{
public:
    uint64_t heapSize;
    uint64_t total;

private:
    uint64_t sweeps;
    std::unordered_map<GCObject*, std::deque<GrayHeapObject const*>> grays;

    void mark(Global& global);
    uint64_t sweep();
    
public:
    Memory();
    ~Memory();
 
    ALWAYS_INLINE
    void collect(Global& global)
    {
        if(total > heapSize)
        {
            mark(global);
            total = sweep();
            if(total > heapSize*0.75 && heapSize < (1<<30))
                heapSize *= 2;
        }
    }
 
    void pushGray(GrayHeapObject const*);

    Heap GlobalHeap;
    Heap ConstHeap;
   
    static Memory All;
};

ALWAYS_INLINE
void GrayHeapObject::writeBarrier() const
{
    if(!gray) {
        Memory::All.pushGray(this);
        gray = true;
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
    return HeapObject::operator new(bytes, Memory::All.GlobalHeap);
}

inline void* HeapObject::operator new(unsigned long bytes, unsigned long extra) {
    return HeapObject::operator new(bytes, extra, Memory::All.GlobalHeap);
}

inline void* HeapObject::operator new(unsigned long bytes, GCFinalizer finalizer)
{
    return HeapObject::operator new(bytes, finalizer, Memory::All.GlobalHeap);
}

#endif

