#ifndef _RIPOSTE_INTERPRETER_H
#define _RIPOSTE_INTERPRETER_H

#include <map>
#include <unordered_map>
#include <unordered_set>
#include <list>
#include <iostream>

#include "riposte.h"
#include "value.h"
#include "thread.h"
#include "random.h"
#include "bc.h"

#ifdef EPEE
#include "epee/ir.h"
#include "epee/trace.h"
#endif

class Global;

////////////////////////////////////////////////////////////////////
// VM data structures
///////////////////////////////////////////////////////////////////

struct Instruction {
    union {
        int64_t i;
        struct {
            int16_t a, b, c;
            ByteCode::Enum bc:16;
        };
    };

    explicit Instruction(int64_t i) :
        i(i) {}

    Instruction(ByteCode::Enum bc, int16_t a=0, int16_t b=0, int16_t c=0) :
        a(a), b(b), c(c), bc(bc) {}
    
    std::string toString() const {
        return std::string("") 
            + ByteCode::toString(bc) 
            + "\t" + intToStr(a) 
            + "\t" + intToStr(b) 
            + "\t" + intToStr(c);
    }
};

Character InternStrings(State& state, Character const& c);

class StringTable
{
    struct StringHash {
        size_t operator()(String s) const {
            return Hash(s);
        }
    };

    struct StringEq {
        // We know that they won't be NA nor equal pointers here
        // so this can be simpler than the generic Eq implementation.
        bool operator()(String s, String t) const {
            return s->length == t->length &&
                   strncmp(s->s, t->s, s->length) == 0;
        }
    };

    std::unordered_set<String, StringHash, StringEq> stringTable;

public:

    ALWAYS_INLINE
    String get(String s) const
    {
        if(Memory::All.ConstHeap.contains(s) || s == Strings::NA)
            return s;

        auto i = stringTable.find(s);

        return (i != stringTable.end())
            ? *i 
            : Strings::NA;
    }

    String intern(String s)
    {
        if(Memory::All.ConstHeap.contains(s) || s == Strings::NA)
            return s;

        auto i = stringTable.find(s);

        String result;
        if(i != stringTable.end())
        {
            result = *i;
        }
        else
        {
            result = new (s->length+1, Memory::All.ConstHeap) StringImpl(s->length);
            memcpy((void*)result->s, s->s, s->length+1);
            stringTable.insert(result);
        }
        
        return result;
    }
};

struct CompiledCall : public List
{
    explicit CompiledCall(
        List const& call, 
        List const& arguments, 
        Character const& names,
        int64_t dotIndex,
        List const& extraArgs,
        Character const& extraNames) 
        : List(6)
    {
        (*this)[0] = call;
        (*this)[1] = arguments;
        (*this)[2] = names;
        (*this)[3] = Integer::c(dotIndex);
        (*this)[4] = extraArgs;
        (*this)[5] = extraNames;
    }

    List const&      call() const {
        return static_cast<List const&>((*this)[0]);
    }

    List const&      arguments() const {
        return static_cast<List const&>((*this)[1]);
    }

    // Need a non-const version because Map in runtime.cpp modifies this.
    List&            arguments() {
        return static_cast<List&>((*this)[1]);
    }

    Character const& names() const {
        return static_cast<Character const&>((*this)[2]);
    }

    int64_t          dotIndex() const {
        return static_cast<Integer const&>((*this)[3]).i;
    }

    List const&      extraArgs() const {
        return static_cast<List const&>((*this)[4]);
    }

    Character const& extraNames() const {
        return static_cast<Character const&>((*this)[5]);
    }
};

// Code can be lazily compiled, so needs to be a gray object
struct Code : public GrayHeapObject {

    Code() : GrayHeapObject(2) {}

    Value expression;
    Integer bc;
    List constants;
    List calls;
    uint64_t registers;
    bool isPromise;

    void printByteCode(Global const& global) const;
    void visit() const;
};

struct Prototype : public HeapObject {
    Code const* code;
    String string;
    Value formals;

    Character parameters;
    List defaults;
    int dotIndex;

    void visit() const;
};

// A simple immutable named list implementation for attributes
// Eventually, we may want to store entries sorted
// so we can do a binary search instead of a scan.
class Dictionary : public HeapObject
{
protected:
    struct Pair { String n; Value v; };
    
    uint64_t size;
    Pair d[];

    Dictionary(uint64_t size)
        : size(size)
    {}

    static Dictionary* Make(uint64_t size)
    {
        return new (sizeof(Pair)*size) Dictionary(size); 
    }

public:
    static Dictionary const* Make(String name, Value const& v)
    {
        Dictionary* result = Make(1);
        result->d[0] = Pair { name, v };
        return result;
    }

    static Dictionary const* Make(
        String name0, Value const& v0,
        String name1, Value const& v1)
    {
        Dictionary* result = Make(2);
        result->d[0] = Pair { name0, v0 };
        result->d[1] = Pair { name1, v1 };
        return result;
    }

    uint64_t Size() const {
        if(!this) return 0;
        return size;
    }

    // Returns a pointer to the entry for `name`
    // or a nullptr if it doesn't exist.
    Value const* get(String name) const
    {
        if(!this) return nullptr;
        
        for(uint64_t i = 0; i < size; ++i)
            if(Eq(d[i].n, name)) return &d[i].v;

        return nullptr;
    }

    // clone without an entry
    Dictionary const* cloneWithout(String name) const
    {
        if(!this || size == 0) return nullptr;
        
        auto v = get(name);
        if(!v) return this;
        if(size == 1) return nullptr;

        // copy over elements
        Dictionary* clone = Make(size-1);
        for(uint64_t i = 0, j = 0; i < size; i++)
        {
            if(!Eq(d[i].n, name))
                clone->d[j++] = d[i];
        }
        return clone;
    }

    // clone with a new entry
    Dictionary const* cloneWith(String name, Value const& v) const
    {
        if(!this || size == 0)
        {
            Dictionary* clone = Make(1);
            clone->d[0] = Pair { name, v };
            return clone;
        }

        auto p = get(name);

        // copy over elements
        Dictionary* clone = Make(size+(p?0:1));
        uint64_t j = 0;
        for(uint64_t i = 0; i < size; i++)
        {
            if(!Eq(d[i].n, name))
                clone->d[j++] = d[i];
        }
        clone->d[j] = Pair { name, v };

        return clone;
    }

    List list() const
    {
        Character n(size);
        List v(size);
        for(size_t i = 0, j = 0; i < size; ++i)
        {
            n[j  ] = d[i].n;
            v[j++] = d[i].v;
        }

        v.attributes(Make(Strings::names, n));
        return v;
    }

    void visit() const;
};

class HashMap : public HeapObject 
{
    HashMap(uint64_t size, uint64_t capacity)
        : size(size), capacity(capacity)
    {
        memset(d, 0, sizeof(Pair)*capacity); 
    }

public:
    struct Pair { String n; Value v; };
   
    uint64_t size, capacity;
    Pair d[]; 
   
    // assumes capacity is a power of 2 
    static HashMap* Make(uint64_t size, uint64_t capacity)
    {
        return new (sizeof(Pair)*capacity) HashMap(size, capacity);
    }

    // Returns whether or not 'name' exists.
    // Fast path if the field does not exist.
    bool has(String name) const
    {
        uint64_t i = ((uint64_t)name >> 5) & (capacity-1);

        if(__builtin_expect(d[i].n == name, false))
            return true;

        while(d[i].n != Strings::NA)
        {
            i = (i+1) & (capacity-1);
            if(__builtin_expect(d[i].n == name, false))
                return true;
        }

        return false;
    }

    // Returns a pointer to the entry for `name`
    // or a nullptr if it doesn't exist. Fast path if the field exists.
    Value const* get(String name) const
    {
        uint64_t i = ((uint64_t)name >> 5) & (capacity-1);

        if(__builtin_expect(d[i].n == name, true))
            return &d[i].v;

        while(d[i].n != Strings::NA)
        {
            i = (i+1) & (capacity-1);
            if(__builtin_expect(d[i].n == name, true))
                return &d[i].v;
        }

        return nullptr;
    }

    // Returns a pointer to the entry pair for `name`
    // or to the slot where it should be added.
    Pair* find(String name)
    {
        uint64_t i = ((uint64_t)name >> 5) & (capacity-1);

        if(__builtin_expect(d[i].n == name, true))
            return &d[i];

        while(d[i].n != Strings::NA)
        {
            i = (i+1) & (capacity-1);
            if(__builtin_expect(d[i].n == name, true))
                return &d[i];
        }

        return &d[i];
    }

    // Returns a pointer where variable `name` should be inserted.
    // Assumes that `name` doesn't exist in the hash table yet.
    // Used for rehash and insert where this is known to be true.
    Pair* slot(String name)
    {
        uint64_t i = ((uint64_t)name >> 5) & (capacity-1);
        if(__builtin_expect(d[i].n == Strings::NA, true))
            return &d[i];
        
        while(d[i].n != Strings::NA)
            i = (i+1) & (capacity-1);
        
        return &d[i];
    }

    void remove(String name)
    {
        // find the entry
        uint64_t i = ((uint64_t)name >> 5) & (capacity-1);
        while(d[i].n != name && d[i].n != Strings::NA)
            i = (i+1) & (capacity-1);

        if(d[i].n != Strings::NA)
            size--;

        while(d[i].n != Strings::NA)
        {
            uint64_t j = i, k;
            do {
                j = (j+1) % (capacity-1);
                k = ((uint64_t)d[j].n >> 5) & (capacity-1);
            } while(d[j].n != Strings::NA &&
                  ((i<=j) ? ((i<k)&&(k<=j)) : ((i<k)||(k<=j))));
            d[i] = d[j];
            i = j;
        }
    }

    HashMap* rehash(uint64_t new_capacity) const;

    void visit() const;
};

class Environment : public GrayHeapObject
{
    Environment* enclosure;
    Dictionary const* attributes;
    HashMap* map;

public:
    explicit Environment(uint64_t size, Environment* enclosure)
        : GrayHeapObject(0)
        , enclosure(enclosure)
        , attributes(nullptr)
        , map(HashMap::Make(0, nextPow2(std::max(size+1, size*2))))
        {}

    Environment* getEnclosure() const { return enclosure; }
    void setEnclosure(Environment* env) {
        enclosure = env;
        writeBarrier();
    }

    Dictionary const* getAttributes() const { return attributes; }
    void setAttributes(Dictionary const* d) {
        attributes = d;
        writeBarrier();
    }
    bool hasAttributes() const { return attributes != 0; }

    // Caller asserts that name doesn't yet exist in the environment
    void init(String name, Value v)
    {
        HashMap::Pair* p = map->slot(name);
        *p = { name, v };
        map->size++;
    }

    bool has(String name) const
    {
        assert(Memory::All.ConstHeap.contains(name));
        return map->has(name);
    }

    // Returns a pointer to the entry for `name`
    // or a nullptr if it doesn't exist.
    Value const* get(String name) const
    {
        assert(Memory::All.ConstHeap.contains(name));
        return map->get(name);
    }

    ALWAYS_INLINE
    Value& insert(String name)
    {
        assert(Memory::All.ConstHeap.contains(name));

        // TODO: is this write barrier always necessary?
        writeBarrier();
        
        HashMap::Pair* p = map->find(name);
        if(p->n == name)
        {
            return p->v;
        }
        else
        {
            if(((map->size+1) * 4) > (map->capacity*3))
            {
                map = map->rehash(map->capacity*2);
                p = map->slot(name);
            }
            
            map->size++;
            p->n = name;
            return p->v;
        }
    }

    void remove(String name)
    {
        assert(Memory::All.ConstHeap.contains(name));
        map->remove(name);
    }

    // Look up variable through enclosure chain 
    Value* getRecursive(String name, Environment*& env)
    {
        assert(Memory::All.ConstHeap.contains(name));

        env = this;
        do {
            HashMap::Pair* p = env->map->find(name);
            if(p->n == name) return &p->v;
        } while((env = env->getEnclosure()));

        return nullptr;
    }

    Character names() const
    {
        Character result(map->size);
        for(uint64_t i = 0, j = 0; i < map->capacity; ++i)
        {
            if(map->d[i].n != Strings::NA)
                result[j++] = map->d[i].n;
        }
        return result;
    }

    void visit() const;
};

struct StackFrame
{
    Value* registers;
    Environment* environment;
    Code const* code;
    Instruction const* returnpc;
};

// For R API support
struct SEXPREC : public HeapObject
{
    Value v;
    SEXPREC(Value const& v) : v(v) {}
    void visit() const;
};

typedef SEXPREC* SEXP;

struct SEXPStack
{
    int* size;
    SEXP* stack;
};

////////////////////////////////////////////////////////////////////
// Global state (shared across all threads) 
///////////////////////////////////////////////////////////////////

class State;

class Global
{
public:
    StringTable strings;

    Environment* empty;
    Environment* global;
    Code* promiseCode;

    Dictionary const* symbolDict;
    Dictionary const* callDict;
    Dictionary const* exprDict;
    Dictionary const* pairlistDict;
    Dictionary const* complexDict;

    // For R API support
    Lock apiLock;
    SEXPStack* apiStack;
    // SEXPs that the API needs to have live between calls.
    std::list<SEXP> installedSEXPs;
    SEXP installSEXP(SEXP s) {
        installedSEXPs.push_back(s);
        return s;
    }
    SEXP installSEXP(Value const& v) {
        return installSEXP(new SEXPREC(v));
    }
    void uninstallSEXP(SEXP s) {
        // go back to front, assuming we're uninstalling something
        // we recently installed.
        for(std::list<SEXP>::reverse_iterator i = installedSEXPs.rbegin();
            i != installedSEXPs.rend(); ++i) {
            if(*i == s) {
                installedSEXPs.erase((++i).base());
                break;
            }
        }
    }

    std::list<State*> states;
    State* getState();
    void deleteState(State* s);

    bool profile;
    bool verbose;
    bool epeeEnabled;

    Riposte::Format format;
    
    Character arguments;

    TaskQueues queues;

    Global(uint64_t threads, int64_t argc, char** argv);

    void dumpProfile(std::string filename);
    std::string stringify(Value const& v) const;
    std::string deparse(Value const& v) const;

    std::string externStr(const String& s) const {
        return std::string(s->s, s->length);
    }

    std::unordered_map<std::string, void*> dl_handles;
    std::unordered_map<std::string, void*> dl_symbols;

    void* get_dl_symbol(std::string const& s);
};

// Global pointer, used by the R API, which
// doesn't get passed a per-thread State.
extern Global* global;

////////////////////////////////////////////////////////////////////
// Per-thread State 
///////////////////////////////////////////////////////////////////

#define DEFAULT_NUM_REGISTERS 10000 

class State
{
public:
    // Shared global state
    Global& global;

    // Interpreter execution data structures
    Value* registers;
    std::vector<StackFrame> stack;
    StackFrame frame;
    bool visible;
    int64_t assignment[256], set[256]; // temporary space for matching arguments
    
#ifdef EPEE
    Traces traces;
#endif

    Random random;
    std::vector<Value> gcStack;
    TaskQueue* queue;

    State(Global& global, TaskQueue* queue);

    StackFrame& push() {
        stack.push_back(frame);
        return frame;
    }

    void pop() {
        frame = stack.back();
        stack.pop_back();
    }

    std::string stringify(Value const& v) const { return global.stringify(v); }
    std::string deparse(Value const& v) const { return global.deparse(v); }
    std::string externStr(String s) const { return global.externStr(s); }

    Value evalTopLevel(Code const* code, Environment* environment, int64_t resultSlot = 0); 
    Value eval(Code const* code, Environment* environment, int64_t resultSlot = 0); 
    Value eval(Code const* code);
    Value eval(Promise const& p, int64_t resultSlot = 0);
};

#endif

