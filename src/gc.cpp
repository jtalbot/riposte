
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

bool HeapObject::marked() const {
    uint64_t i = ((uint64_t)this & (REGION_SIZE-1)) / CELL_SIZE;
    uint64_t slot = i % 64;
    uint64_t word = i / 64;
    
    return (gcObject()->mark[word] & (((uint64_t)1) << slot)) != 0;
}

void HeapObject::visit() const {
    uint64_t i = ((uint64_t)this & (REGION_SIZE-1)) / CELL_SIZE;
    uint64_t slot = i % 64;
    uint64_t word = i / 64;
    
    gcObject()->mark[word] |= (((uint64_t)1) << slot);
}

void HeapObject::block() const {
    uint64_t i = ((uint64_t)this & (REGION_SIZE-1)) / CELL_SIZE;
    uint64_t slot = i % 64;
    uint64_t word = i / 64;
    
    gcObject()->mark[word] &= ~(((uint64_t)1) << slot);
    gcObject()->block[word] |= (((uint64_t)1) << slot);
}

GCObject* HeapObject::gcObject() const {
    return (GCObject*)((uint64_t)this & ~(REGION_SIZE-1));
}

uint64_t NumberOfSetBits(uint64_t i)
{
    i = i - ((i >> 1) & 0x5555555555555555UL);
    i = (i & 0x3333333333333333UL) + ((i >> 2) & 0x3333333333333333UL);
    return (((i + (i >> 4)) & 0xF0F0F0F0F0F0F0FUL) * 0x101010101010101UL) >> 56;
}

uint64_t bytes;

uint64_t type_count[17];

#define VISIT(p) if((p) != 0 && !(p)->marked()) (p)->visit()

static void traverse(Value const& v) {
    //type_count[v.type()]++;
    switch(v.type()) {
        case Type::Nil:
            // do nothing
            break;
        case Type::Environment:
            VISIT(((REnvironment const&)v).attributes());
            VISIT(((REnvironment const&)v).environment());
            break;
        case Type::Closure:
            VISIT(((Closure const&)v).attributes());
            VISIT((Closure::Inner const*)v.p);
            VISIT(((Closure const&)v).prototype());
            VISIT(((Closure const&)v).environment());
            //bytes += 16;
            break;
        case Type::Externalptr:
            VISIT((Externalptr::Inner const*)v.p);
            traverse(((Externalptr const&)v).tag());
            traverse(((Externalptr const&)v).prot());
            //bytes += 48;
            break;
        case Type::Null:
            // do nothing
            break;
        case Type::Double:
            VISIT(((Double const&)v).attributes());
            VISIT(((Double const&)v).inner());
            //bytes += 8*((Double const&)v).length();
            break;
        case Type::Integer:
            VISIT(((Integer const&)v).attributes());
            VISIT(((Integer const&)v).inner());
            //bytes += 8*((Integer const&)v).length();
            break;
        case Type::Logical:
            VISIT(((Logical const&)v).attributes());
            VISIT(((Logical const&)v).inner());
            //bytes += 1*((Logical const&)v).length();
            break;
        case Type::Character:
            VISIT(((Character const&)v).attributes());
            VISIT(((Character const&)v).inner());
            {
                if(v.packed() > 1) {
                    auto p = (VectorImpl<Type::Character, String, false>::Inner const*)v.p;
                    for(size_t i = 0; i < p->length; ++i) {
                        VISIT(p->data[i]);
                        //bytes += p->data[i] ? strlen(p->data[i]->s)+1 : 0;
                    }
                }
                else if(v.packed() == 1) {
                    VISIT(v.s);
                }
                //bytes += 8*((Character const&)v).length();
            }
            break;
        case Type::Raw:
            VISIT(((Raw const&)v).attributes());
            VISIT(((Raw const&)v).inner());
            //bytes += 1*((Raw const&)v).length();
            break;
        case Type::List:
            VISIT(((List const&)v).attributes());
            VISIT(((List const&)v).inner());
            {
                List const& l = (List const&)v;
                for(int64_t i = 0; i < l.length(); i++)
                    traverse(l[i]);
            //bytes += 16*((List const&)v).length();
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
        case Type::ScalarString:
            VISIT(((ScalarString const&)v).s);
            //bytes += strlen(((ScalarString const&)v).s->s);
            break;
        case Type::Pairlist:
            VISIT((Pairlist::Inner const*)v.p);
            VISIT(((Pairlist const&)v).car());
            VISIT(((Pairlist const&)v).cdr());
            VISIT(((Pairlist const&)v).tag());
            //bytes += 24;
            break;
        default:
            printf("Unimplemented type: %d\n", v.type()); 
            // do nothing
            break;
    }
}

uint64_t dictionary_count;

void Dictionary::visit() const {
    //dictionary_count++;
    HeapObject::visit();
    VISIT(d);
    for(uint64_t i = 0; i < size; i++) {
        VISIT(d->d[i].n);
        if(d->d[i].n != Strings::NA)
            traverse(d->d[i].v);
    }
    //bytes += 24*size+48;
}

void Environment::visit() const {
    Dictionary::visit();
    VISIT(enclosure);
    VISIT(attributes);
    //bytes += 16;
}

uint64_t code_count;

void Code::visit() const {
    //code_count++;
    HeapObject::visit();
    traverse(expression);
    traverse(bc);
    traverse(constants);
    traverse(calls);
    //bytes += 8*bc.size() + 16*constants.size() + 88*calls.size() + 3;
}

uint64_t prototype_count;

void Prototype::visit() const {
    //prototype_count++;
    HeapObject::visit();
    VISIT(code);
    VISIT(string);

    traverse(formals);
    traverse(parameters);
    traverse(defaults);

    //bytes += 72;
}

void SEXPREC::visit() const {
    HeapObject::visit();
    traverse(v);

    //bytes += 16;
}


Heap::Heap() : heapSize(1<<20), total(0)
{
    makeArenas((1<<20)/REGION_SIZE);
    bump = arenas[0]->data;
    limit = arenas[0]->data;
}

HeapObject* Heap::addFinalizer(HeapObject* h, GCFinalizer f)
{
    if(h && f)
        finalizers.push_back(std::make_pair(h, f));
    return h;
}

void Heap::mark(Global& global) {

    /*for(int i = 0; i < 17; ++i)
        type_count[i] = 0;
    dictionary_count = 0;
    code_count = 0;
    prototype_count = 0;*/
    //bytes = 0;

    // traverse root set
    // mark the region that I'm currently allocating into
    ((HeapObject*)bump)->visit();
    
    //printf("--global--\n");
    VISIT(global.empty);
    VISIT(global.global);
    VISIT(global.promiseCode);
    
    traverse(global.arguments);

    VISIT(global.symbolDict);
    VISIT(global.callDict);
    VISIT(global.exprDict);

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

    /*for(int i = 0; i < 17; ++i)
        printf("%d: %d\n", i, type_count[i]);
    printf("dict: %d\n", dictionary_count);
    printf("code: %d\n", code_count);
    printf("proto: %d\n", prototype_count);*/
    //printf("bytes: %d\n", bytes);
}


void Heap::sweep(Global& global)
{

    // sweep heap
    uint64_t old_total = total, bytes = 0;
    total = 0;

    // Should really be kept per arena for cache locality
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

    for(auto b : arenas)
    {
        b->sweep();
        if(b->marked())
            total += REGION_SIZE;
    }

    uint64_t bigtotal = 0;
    for(auto i = blocks.begin(); i != blocks.end();)
    {
        auto b = *i;

        b->sweep();
        if(!b->marked())
        {
            free(b);

            i = blocks.erase(i);
        }
        else
        {
            bigtotal += REGION_SIZE;
            ++i;
        }
    }

    bump = arenas[0]->data;
    limit = arenas[0]->data;
    arenaIndex = 0;

    total += bigtotal;

    printf("Swept: %d   =>    %d + %d\n", old_total, total-bigtotal, bigtotal);
}

HeapObject* Heap::alloc(uint64_t bytes)
{
	bytes += sizeof(GCObject);
	total += bytes;

    char* head;
    posix_memalign((void**)&head, REGION_SIZE, bytes);
	memset(head, 0xab, bytes);
	GCObject* g = (GCObject*)head;
	g->Init();
    blocks.push_back(g);

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
        arenas.push_back(r);
        head += REGION_SIZE;
    }
}

bool next(GCObject* g, uint64_t& start, uint64_t& end)
{
    // Advance start until we're at a free block.
    uint64_t s = start;

    if(s >= CELL_COUNT)
        return false;

    if(g->mark[s/64] >> (s%64))
    {
        s += ffsll(g->mark[s/64] >> (s%64))-1;
    }
    else
    {
        s = (s+64)&~63;
        while(s < CELL_COUNT && !g->mark[s/64])
            s += 64;
        if(s < CELL_COUNT)
            s += ffsll(g->mark[s/64])-1;
    }

    // Post-condition: s should be at a free block or past the end
    assert(s == CELL_COUNT || ((g->mark[s/64] >> (s%64)) & 1));

    // Coalesce free blocks (clearing mark flags) up to the next non-free block.
    uint64_t e = s;
    
    if(e >= CELL_COUNT)
        return false;

    if(g->block[e/64] >> (e%64))
    {
        e += ffsll(g->block[e/64] >> (e%64))-1;
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

bool find(GCObject* g, uint64_t cells, uint64_t& start, uint64_t& size)
{
    if(start/64 >= GCObject::WORDS)
        return false;

    // on entry, start should be pointing to a block boundary
    assert( (g->block[start/64] & (1ull << (start%64))) ^
            (g->mark[start/64] & (1ull << (start%64))) );

    bool free = !(g->block[start/64] & (1ull << (start%64)));
    // i always points after the start of the next block
    uint64_t i = start+1;

    for(uint64_t word = i/64; word < GCObject::WORDS; ++word)
    {
        // While still in this word, advance the search.
        while(i/64 == word)
        {
            // If free, we're looking for the next block flag.
            if(free)
            {
                uint64_t b = g->block[word] >> (i%64);
                uint64_t p = ffsll(b);

                if(p) {
                    // clear mark flags to coalesce free blocks
    //                g->mark[word] &= ~(((1ull << ((i%64)+(p-1)))-1) ^ ((1ull << (i%64))-1));
                    
                    i += p;
                    free = false;
                    if(i-start-1 >= cells) {
                        size = i-start-1;
                        return true;
                    }
                }
                else {
                    // clear mark flags to coalesce free blocks
      //              g->mark[word] &= ~((~0) ^ ((1ull << i)-1));
                    i = (i+64)&~63;
                }
            }
            // If used, we're looking for the next mark flag.
            else
            {
                uint64_t b = g->mark[word] >> (i%64);
                uint64_t p = ffsll(b);

                if(p) {
                    i += p;
                    start = (i-1);
                    free = true;
                }
                else {
                    i = (i+64)&~63;
                }
            }
        }
    }

    size = i-start;
    return free && size >= cells;
}

bool find2(GCObject* g, uint64_t cells, uint64_t& start, uint64_t& size)
{
uint64_t i = start;
bool free = true;
        for(uint64_t word = i/64; word < GCObject::WORDS; ++word)
        {
            for(uint64_t slot = i%64; slot < 64; ++slot, ++i)
            {
                uint64_t b = g->block[word] & (((uint64_t)1) << slot);
                uint64_t m = g->mark[word] & (((uint64_t)1) << slot);
            
                if(b && !m) {
                    if(free && size >= cells) return true;
                    free = false;
                }
                else if(!b && m && !free) { free = true; start = i; size = 1; }
                else if(!b && m && free) {
                    //g->mark[word] &= ~(((uint64_t)1) << slot);
                    size++;
                }
                else {
                    size++;
                }
            }
        }

        return free && size >= cells;
}

bool find3(GCObject* g, uint64_t cells, uint64_t& start, uint64_t& size)
{
    uint64_t end = 0;
    while(next(g, start, end))
    {
        size = end-start;
        if(size >= cells)
            return true;
        start  = end;
    }

    return false;
}

void Heap::popRegion(uint64_t bytes)
{
    // find the next range at least bytes large
    uint64_t cells = (bytes+(CELL_SIZE-1))/CELL_SIZE;

    GCObject* g = arenas[arenaIndex];
    
    // mark the remaining space free
    if(bump < limit)
    {
        uint64_t i = (bump-(char*)g)/CELL_SIZE;
        g->mark[i/64] |= 1ull << (i%64);
    }

    // try to do something useful in the current arenaIndex
    uint64_t start = (limit-(char*)g)/CELL_SIZE, size = 0;
    //uint64_t start2 = (limit-(char*)g)/CELL_SIZE, size2 = 0;

    //uint64_t b0 = g->block[0], b1 = g->block[1];
    //uint64_t m0 = g->mark[0], m1 = g->mark[1];

    //bool a = find(g, cells, start, size);
    //bool b = find3(g, cells, start2, size2);
    bool a = find3(g, cells, start, size);

    /*if(a!=b || (a && b && (start != start2 || size != size2)))
    {
        printf("%d: looking for %d @ %d >> \n", arenaIndex, cells, (limit-(char*)g)/CELL_SIZE);
        std::cout << std::bitset<64>(b0) << " " << std::bitset<64>(b1) << std::endl;
        std::cout << std::bitset<64>(m0) << " " << std::bitset<64>(m1) << std::endl;
        printf("(%d %d %d)   (%d %d %d)\n", a, start, size, b, start2, size2);
        std::cout << std::bitset<64>(g->block[0]) << " " << std::bitset<64>(g->block[1]) << std::endl;
        std::cout << std::bitset<64>(g->mark[0]) << " " << std::bitset<64>(g->mark[1]) << std::endl;
        std::cout << std::endl;
    }*/

    if(a)
    {
        bump = (char*)(g) + start*CELL_SIZE;
        limit = (char*)(g) + (start+size)*CELL_SIZE;
        return;
    }

    while(true)
    {
        arenaIndex++;
        
        if(arenaIndex >= arenas.size()) {
            makeArenas((1<<20)/REGION_SIZE);
        }

        GCObject* g = arenas[arenaIndex];
        
        if(!g->marked())
        {
            memset(&g->mark, 0, GCObject::WORDS*8);
            memset(&g->block, 0, GCObject::WORDS*8);
            g->mark[0] |= ((uint64_t)1) << ((g->data-(char*)g)/CELL_SIZE);
            bump = (char*)(g->data);
            limit = ((char*)g) + REGION_SIZE;
            total += REGION_SIZE;
            return;
        }

        uint64_t start = (g->data-(char*)g)/CELL_SIZE, size = 0;
        //uint64_t start2 = (g->data-(char*)g)/CELL_SIZE, size2 = 0;

    //uint64_t b0 = g->block[0], b1 = g->block[1];
    //uint64_t m0 = g->mark[0], m1 = g->mark[1];

    //bool a = find(g, cells, start, size);
    //bool b = find3(g, cells, start2, size2);
    bool a = find3(g, cells, start, size);

    /*if(a!=b || (a && b && (start != start2 || size != size2)))
    {
        printf("%d: looking for %d @ %d >> \n", arenaIndex, cells, (g->data-(char*)g)/CELL_SIZE);
        std::cout << std::bitset<64>(b0) << " " << std::bitset<64>(b1) << std::endl;
        std::cout << std::bitset<64>(m0) << " " << std::bitset<64>(m1) << std::endl;
        printf("(%d %d %d)   (%d %d %d)\n", a, start, size, b, start2, size2);
        std::cout << std::bitset<64>(g->block[0]) << " " << std::bitset<64>(g->block[1]) << std::endl;
        std::cout << std::bitset<64>(g->mark[0]) << " " << std::bitset<64>(g->mark[1]) << std::endl;
        std::cout << std::endl;
    }*/

        if(a)
        {
            bump = (char*)(g) + start*CELL_SIZE;
            limit = (char*)(g) + (start+size)*CELL_SIZE;
            return;
        }
    }
}

Heap Heap::GlobalHeap;
Heap Heap::ConstHeap;


