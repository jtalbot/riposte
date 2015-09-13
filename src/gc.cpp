
#include <bitset>
#include <iostream>
#include <strings.h>
#include <stdlib.h>

#include "gc.h"
#include "value.h"
#include "interpreter.h"
#include "api/api.h"

void GCObject::sweep()
{
    for(uint64_t i = 0; i < WORDS; ++i)
    {
        uint64_t b = block[i] & mark[i];
        uint64_t m = block[i] ^ mark[i];

        block[i] = b;
        mark[i] = m;
    }
}

void GCObject::sweepMinor()
{
    for(uint64_t i = 0; i < WORDS; ++i)
    {
        uint64_t b = block[i] & mark[i];
        uint64_t m = block[i] | mark[i];

        block[i] = b;
        mark[i] = m;
    }
}

bool HeapObject::marked() const {
    uint64_t i = ((uint64_t)this & (REGION_SIZE-1)) / CELL_SIZE;
    uint64_t slot = i % 64;
    uint64_t word = i / 64;
    
    return (gcObject()->mark[word] & (((uint64_t)1) << slot)) != 0;
}

bool HeapObject::visit() const
{
    if(this)
    {
        uint64_t i = ((uint64_t)this & (REGION_SIZE-1)) / CELL_SIZE;
        uint64_t slot = i % 64;
        uint64_t word = i / 64;
    
        bool marked = (gcObject()->mark[word] & (((uint64_t)1) << slot)) != 0;
        gcObject()->mark[word] |= (((uint64_t)1) << slot);

        return !marked;
    }
    return false;
}

bool GrayHeapObject::visit() const
{
    if(this)
    {
        bool result = HeapObject::visit() || gray;
        gray = false;
        return result;
    }
    return false;
}

uint64_t NumberOfSetBits(uint64_t i)
{
    i = i - ((i >> 1) & 0x5555555555555555UL);
    i = (i & 0x3333333333333333UL) + ((i >> 2) & 0x3333333333333333UL);
    return (((i + (i >> 4)) & 0xF0F0F0F0F0F0F0FUL) * 0x101010101010101UL) >> 56;
}

uint64_t bytes;

uint64_t type_count[17];

static void traverse(Value const& v) {
    //type_count[v.type()]++;
    switch(v.type()) {
        case Type::Nil:
            // do nothing
            break;
        case Type::Environment:
            ((REnvironment const&)v).attributes()->visit();
            ((REnvironment const&)v).environment()->visit();
            break;
        case Type::Closure:
            ((Closure const&)v).attributes()->visit();
            ((Closure::Inner const*)v.p)->visit();
            ((Closure const&)v).prototype()->visit();
            ((Closure const&)v).environment()->visit();
            //bytes += 16;
            break;
        case Type::Externalptr:
            ((Externalptr::Inner const*)v.p)->visit();
            if(v.p) {
                traverse(((Externalptr const&)v).tag());
                traverse(((Externalptr const&)v).prot());
            }
            //bytes += 48;
            break;
        case Type::Null:
            // do nothing
            break;
        case Type::Double:
            assert(((Double const&)v).length() <= ((Double const&)v).capacity());
            ((Double const&)v).attributes()->visit();
            ((Double const&)v).inner()->visit();
            //bytes += 8*((Double const&)v).length();
            break;
        case Type::Integer:
            assert(((Integer const&)v).length() <= ((Integer const&)v).capacity());
            ((Integer const&)v).attributes()->visit();
            ((Integer const&)v).inner()->visit();
            //bytes += 8*((Integer const&)v).length();
            break;
        case Type::Logical:
            assert(((Logical const&)v).length() <= ((Logical const&)v).capacity());
            ((Logical const&)v).attributes()->visit();
            ((Logical const&)v).inner()->visit();
            //bytes += 1*((Logical const&)v).length();
            break;
        case Type::Character:
            assert(((Character const&)v).length() <= ((Character const&)v).capacity());
            ((Character const&)v).attributes()->visit();
            ((Character const&)v).inner()->visit();
            for(size_t i = 0; i < ((Character const&)v).length(); ++i)
                ((Character const&)v)[i]->visit();
                //bytes += 8*((Character const&)v).length();
            break;
        case Type::Raw:
            assert(((Raw const&)v).length() <= ((Raw const&)v).capacity());
            ((Raw const&)v).attributes()->visit();
            ((Raw const&)v).inner()->visit();
            //bytes += 1*((Raw const&)v).length();
            break;
        case Type::List:
            assert(((List const&)v).length() <= ((List const&)v).capacity());
            ((List const&)v).attributes()->visit();
            ((List const&)v).inner()->visit();
            {
                List const& l = (List const&)v;
                for(int64_t i = 0; i < l.length(); i++)
                    traverse(l[i]);
            }
            //bytes += 16*((List const&)v).length();
            break;
        case Type::Promise:
            ((Promise&)v).environment()->visit();
            if(((Promise&)v).isExpression())
                ((Promise&)v).code()->visit();
            break;
        case Type::Integer32:
            ((Integer32 const&)v).attributes()->visit();
            ((Integer32 const&)v).inner()->visit();
            break;
        case Type::Logical32:
            ((Logical32 const&)v).attributes()->visit();
            ((Logical32 const&)v).inner()->visit();
            break;
        case Type::ScalarString:
            ((ScalarString const&)v).s->visit();
            //bytes += strlen(((ScalarString const&)v).s->s);
            break;
        case Type::Pairlist:
            ((Pairlist::Inner const*)v.p)->visit();
            ((Pairlist const&)v).car()->visit();
            ((Pairlist const&)v).cdr()->visit();
            ((Pairlist const&)v).tag()->visit();
            //bytes += 24;
            break;
        default:
            printf("Unimplemented type: %d\n", v.type()); 
            // do nothing
            break;
    }
}

uint64_t dictionary_count;

void Dictionary::visit() const
{
    if(HeapObject::visit())
    {
        //dictionary_count++;
        for(uint64_t i = 0; i < size; i++)
        {
            d[i].n->visit();
            if(d[i].n != Strings::NA)
                traverse(d[i].v);
        }
    }
    //bytes += 24*size+48;
}

void HashMap::visit() const
{
    HeapObject::visit();
    for(uint64_t i = 0; i < capacity; i++)
    {
        d[i].n->visit();
        if(d[i].n != Strings::NA)
            traverse(d[i].v);
    }
}

uint64_t environment_count;

void Environment::visit() const
{
    if(GrayHeapObject::visit())
    {
        //environment_count++;
        enclosure->visit();
        attributes->visit();
        map->visit(); 
            
        // also have to traverse the dots for now
        Value const* dots = get(Strings::__dots__);
        if(dots && dots->isList())
        {
            ((List const&)*dots).attributes()->visit();
            ((List const&)*dots).inner()->visit();
            {
                List const& l = (List const&)*dots;
                for(int64_t i = 0; i < l.length(); i++)
                    traverse(l[i]);
            //bytes += 16*((List const&)v).length();
            }
        }
    }
    //bytes += 16;
}

uint64_t code_count;

void Code::visit() const
{
    if(GrayHeapObject::visit())
    {
        //code_count++;
        traverse(expression);
        traverse(bc);
        traverse(constants);
        traverse(calls);
    }
    //bytes += 8*bc.size() + 16*constants.size() + 88*calls.size() + 3;
}

uint64_t prototype_count;

void Prototype::visit() const
{
    //prototype_count++;
    if(HeapObject::visit())
    {
        code->visit();
        string->visit();

        traverse(formals);
        traverse(parameters);
        traverse(defaults);
    }
    //bytes += 72;
}

void SEXPREC::visit() const
{
    if(HeapObject::visit())
    {
        traverse(v);
    }
    //bytes += 16;
}

void Pairlist::Inner::visit() const
{
    if(GrayHeapObject::visit())
    {
        car->visit();
        cdr->visit();
        tag->visit();
    }
}


Heap::Heap()
{
    makeArenas((1<<20)/REGION_SIZE);
    bump = arenas[0].ptr->data;
    limit = arenas[0].ptr->data;
    arenaIndex = 0;
    arenaOffset = (arenas[0].ptr->data-(char*)arenas[0].ptr)/CELL_SIZE;
}

HeapObject* Heap::addFinalizer(HeapObject* h, GCFinalizer f)
{
    if(h && f)
        finalizers.push_back(std::make_pair(h, f));
    return h;
}

bool Heap::containsSlow(GCObject const* g) const
{ 
    for(auto& a : arenas)
    {
        if(a.ptr == g)
            return true;
    }
    for(auto& a : larges)
    {
        if(a.ptr == g)
            return true;
    }

    return false;
}

uint64_t Heap::sweep()
{
    static uint64_t t = 0;

    // sweep heap
    uint64_t total = 0;

    // Should really be kept per arena for cache locality
    // TODO: this should really be run after the sweep below
    for(auto i = finalizers.begin(); i != finalizers.end();)
    {
        if(!i->first->marked())
        {
            i->second(i->first);
            i = finalizers.erase(i);
        }
        else
        {
            ++i;
        }
    }

    for(auto& b : arenas)
    {
        if(t%16 == 0)
            b.ptr->sweep();
        else
            b.ptr->sweepMinor();

        if(b.ptr->marked())
            total += REGION_SIZE/32;
    }

    uint64_t bigtotal = 0;
    for(auto i = larges.begin(); i != larges.end();)
    {
        auto b = *i;

        if(t%16 == 0)
            b.ptr->sweep();
        else
            b.ptr->sweepMinor();
        if(!b.ptr->marked())
        {
            free(b.ptr);
            i = larges.erase(i);
        }
        else
        {
            bigtotal += b.bytes;
            ++i;
        }
    }

    bump = arenas[0].ptr->data;
    limit = arenas[0].ptr->data;
    arenaIndex = 0;
    arenaOffset = (arenas[0].ptr->data-(char*)arenas[0].ptr)/CELL_SIZE;
    
    // sweep may have changed free block sizes, so delete saved blocks
    //freeBlocks.clear();
    t++;

    return total += bigtotal;
}

HeapObject* Heap::alloc(uint64_t bytes)
{
    Memory::All.total += bytes;
	bytes += sizeof(GCObject);

    char* head;
    posix_memalign((void**)&head, REGION_SIZE, bytes);
	memset(head, 0xab, bytes);
	GCObject* g = (GCObject*)head;
	g->Init();
    larges.push_back(LargeArena(g, bytes));

	HeapObject* o = (HeapObject*)(g->data);
    o->block();
    return o;
}

void Heap::makeArenas(uint64_t regions)
{
    char* head;
    posix_memalign((void**)&head, REGION_SIZE, regions*REGION_SIZE);
    for(uint64_t i = 0; i < regions; i++) {
        GCObject* r = (GCObject*)head;
        r->Init();
        r->mark[0] |= ((uint64_t)1) << ((r->data-(char*)r)/CELL_SIZE);
        assert(((uint64_t)r & (REGION_SIZE-1)) == 0);
        arenas.push_back(Arena(r));
        head += REGION_SIZE;
    }
}

bool next(GCObject* g, uint64_t& start, uint64_t& end)
{
    // Advance start until we're at a free block.
    uint64_t s = start;

    if(s >= CELL_COUNT)
        return false;

    // Precondition: start should be on a block boundary.
    assert(((g->mark[s/64] >> (s%64)) & 1) |
           ((g->block[s/64] >> (s%64)) & 1));
    
    while(s < CELL_COUNT && ((g->block[s/64] >> (s%64)) & 1))
    {
        if((g->mark[s/64] >> (s%64)) & ~1ull)
        {
            s += ffsll((g->mark[s/64] >> (s%64)) & ~1ull)-1;
        }
        else
        {
            s = (s+64)&~63;
            while(s < CELL_COUNT && !g->mark[s/64])
                s += 64;
            if(s < CELL_COUNT)
                s += ffsll(g->mark[s/64])-1;
        }
    }

    // If we're at the end of the arena, there is no next free block.
    if(s >= CELL_COUNT)
        return false;

    // Post-condition: s should be at a free block
    assert((g->mark[s/64] >> (s%64)) & 1);

    // Coalesce free blocks (clearing mark flags) up to the next non-free block.
    uint64_t e = s;
    
    if(g->block[e/64] >> (e%64))
    {
        e += ffsll((g->block[e/64] >> (e%64)) & ~1ull)-1;
        // TODO: can this be any simpler while avoiding undef shift behavior?
        g->mark[e/64] &=  ((1ull << s%64)-1) |
                          ((1ull << s%64)  ) |
                         ~((1ull << e%64)-1);
    }
    else
    {
        // TODO: can this be any simpler while avoiding undef shift behavior?
        g->mark[s/64] &= ((1ull << s%64)-1) |
                         ((1ull << s%64)  );
        e = (e+64)&~63;
        while(e < CELL_COUNT && !g->block[e/64])
        {
            g->mark[e/64] = 0;
            e += 64;
        }
        if(e < CELL_COUNT)
        {
            e += ffsll(g->block[e/64])-1;
            g->mark[e/64] &= ~((1ull << e%64)-1); 
        }
    }

    // Post-condition: e should be at a non-free block or past the end
    assert(e == CELL_COUNT || ((g->block[e/64] >> (e%64)) & 1));

    start = s;
    end = e;
    return true;
}

void Heap::popRegion(uint64_t bytes)
{
    // mark the remaining space free
    if(bump < limit)
    {
        GCObject* g = ((HeapObject*)bump)->gcObject();
        uint64_t i = (bump-(char*)g)/CELL_SIZE;
        g->mark[i/64] |= 1ull << (i%64);
        //freeBlocks.insert(std::make_pair((limit-bump)/CELL_SIZE, bump));
    }

    // find the next range at least bytes large
    uint64_t cells = (bytes+(CELL_SIZE-1))/CELL_SIZE;
    GCObject* g = arenas[arenaIndex].ptr;
    uint64_t start = arenaOffset, end = 0;

    // try to get a best fit
    /*if(cells <= 16)
    {
        if(!freeM[cells-1].empty())
        {
            //printf("Best fit %d into %d\n", cells, cells);
            bump = freeM[cells-1].back();
            freeM[cells-1].pop_back();
            limit = bump+cells*CELL_SIZE;
            return;
        }
    }*/

    // constrained scan for best fit 
    
/*    while(next(g, start, end) && count++ < 64)
    {
        uint64_t size = end-start;
        //printf("Adding %d\n", size);
        freeM[size-1].push_back((char*)g + start*CELL_SIZE); 

        start  = end;
        arenaOffset = end;
    }*/

    // first fit from lists
    /*if(cells <= 2048)
    {
        uint64_t c = cells;
        while(c < 2048 && freeM[c-1].empty())
            c++;

        if(!freeM[c-1].empty())
        {
            freeCount--;
            //printf("First fit %d into %d\n", cells, c);
            bump = freeM[c-1].back();
            freeM[c-1].pop_back();
            limit = bump+c*CELL_SIZE;
            return;
        }
    }*/

    //auto freeBlock = freeBlocks.lower_bound(cells); 
    //if(freeBlock != freeBlocks.end()) {
    //    uint64_t size = freeBlock->first;
    //    bump = freeBlock->second;
    //    freeBlocks.erase(freeBlock);
        
        //printf("First fit %d into %d\n", cells, size);
        //if(freeBlocks.size() < 250 || size == cells)
        //{
   //         limit = bump + size*CELL_SIZE;
        //}
        //else
        //{
        //    limit = bump + cells*CELL_SIZE;
        //    ((HeapObject*)bump)->gcObject()->mark[arenaOffset/64] |= (1ull << arenaOffset%64);
        //    freeBlocks.insert(std::make_pair(size-cells, limit));
        //}
    //    return;
   // }

    // open scan for first fit 
    while(next(g, start, end))
    {
        uint64_t size = end-start;
        if(size >= cells)
        {
            bump = (char*)(g) + start*CELL_SIZE;
            limit = (char*)(g) + end*CELL_SIZE;
            arenaOffset = end;
            return;
        }
        //freeBlocks.insert(std::make_pair(size, (char*)g + start*CELL_SIZE)); 
        start  = end;
    }

    //printf("Next region for %d\n", cells);

    while(true)
    {
        arenaIndex++;
        
        if(arenaIndex >= arenas.size()) {
            makeArenas((1<<20)/REGION_SIZE);
        }

        GCObject* g = arenas[arenaIndex].ptr;
        
        if(!g->marked())
        {
            memset(&g->mark, 0, GCObject::WORDS*8);
            memset(&g->block, 0, GCObject::WORDS*8);
            g->mark[0] |= ((uint64_t)1) << ((g->data-(char*)g)/CELL_SIZE);
            bump = (char*)(g->data);
            limit = ((char*)g) + REGION_SIZE;
            arenaOffset = CELL_COUNT;
            Memory::All.total += REGION_SIZE;
            return;
        }

        arenaOffset = (g->data-(char*)g)/CELL_SIZE;
    uint64_t start = arenaOffset, end = 0;
   
    // try to get a best fit
    /*if(cells <= 16)
    {
        if(!freeM[cells-1].empty())
        {
            //printf("Best fit %d into %d\n", cells, cells);
            bump = freeM[cells-1].back();
            freeM[cells-1].pop_back();
            limit = bump+cells*CELL_SIZE;
            return;
        }
    }*/

    // constrained scan for best fit 
    /*while(next(g, start, end) && count++ < 64)
    {
        uint64_t size = end-start;
        //printf("Adding %d\n", size);
        freeM[size-1].push_back((char*)g + start*CELL_SIZE); 

        start  = end;
        arenaOffset = end;
    }*/

    // first fit from lists
    /*if(cells <= 2048)
    {
        uint64_t c = cells;
        while(c < 2048 && freeM[c-1].empty())
            c++;

        if(!freeM[c-1].empty())
        {
            freeCount--;
            //printf("First fit %d into %d\n", cells, c);
            bump = freeM[c-1].back();
            freeM[c-1].pop_back();
            limit = bump+c*CELL_SIZE;
            return;
        }
    }*/

    /*auto freeBlock = freeBlocks.lower_bound(cells); 
    if(freeBlock != freeBlocks.end()) {
        bump = freeBlock->second;
        limit = freeBlock->second + freeBlock->first*CELL_SIZE;
        freeBlocks.erase(freeBlock);
        return;
    }*/

    // open scan for first fit 
    start = arenaOffset;
    while(next(g, start, end))
    {
        uint64_t size = end-start;
        if(size >= cells)
        {
            bump = (char*)(g) + start*CELL_SIZE;
            limit = (char*)(g) + end*CELL_SIZE;
            arenaOffset = end;
            return;
        }
        
        //freeBlocks.insert(std::make_pair(size, (char*)g + start*CELL_SIZE)); 
        start  = end;
    }

    }
}

Memory::Memory() : heapSize(1<<20), total(0), sweeps(0)
{
}

Memory::~Memory() {
    //printf("Sweeps: %d\n", sweeps);
}

void Memory::mark(Global& global)
{
    /*for(int i = 0; i < 17; ++i)
        type_count[i] = 0;
    environment_count = 0;
    dictionary_count = 0;
    code_count = 0;
    prototype_count = 0;*/
    //bytes = 0;

    // traverse root set
    // mark the region that I'm currently allocating into
    
    //printf("--global--\n");
    global.empty->visit();
    global.global->visit();
    global.promiseCode->visit();
    
    traverse(global.arguments);

    global.symbolDict->visit();
    global.callDict->visit();
    global.exprDict->visit();

    for(std::list<State*>::iterator t = global.states.begin();
            t != global.states.end(); ++t) {
        State* state = *t;

        //printf("--stack--\n");
        for(uint64_t i = 0; i < state->stack.size(); i++) {
            state->stack[i].code->visit();
            state->stack[i].environment->visit();
        }
        //printf("--frame--\n");
        state->frame.code->visit();
        state->frame.environment->visit();

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
    for(auto& i : global.installedSEXPs)
    {
        i->visit();
    }

    if(global.apiStack) {
        for(int i = 0; i < *global.apiStack->size; ++i) {
            global.apiStack->stack[i]->visit();
        }
    }

    // gray sets
    for(auto& a : grays)
    {
        for(auto& g : a.second)
        {
            if(g->type == 0)
                static_cast<Environment const*>(g)->visit();
            else if(g->type == 1)
                static_cast<Pairlist::Inner const*>(g)->visit();
            else if(g->type == 2)
                static_cast<Code const*>(g)->visit();
            else {
                printf("Unknown gray type %d\n", g->type);
            }
        }
    }
    grays.clear();

    /*for(int i = 0; i < 17; ++i)
        printf("%d: %d\n", i, type_count[i]);
    printf("env: %d\n", environment_count);
    printf("dict: %d\n", dictionary_count);
    printf("code: %d\n", code_count);
    printf("proto: %d\n", prototype_count);*/
    //printf("bytes: %d\n", bytes);
}

uint64_t Memory::sweep()
{
    sweeps++;
    uint64_t v = GlobalHeap.sweep();
    //printf("Sweeping => %d\n", v);
    return v;
}

void Memory::pushGray(GrayHeapObject const* h) {
    //printf("Marking gray\n");
    if(h->marked()) {
        GCObject* g = h->gcObject();
        grays[g].push_back(h);
    }
}

Memory Memory::All;


