#include <string>
#include <sstream>
#include <stdexcept>
#include <string>

#include "value.h"
#include "type.h"
#include "bc.h"
#include "interpreter.h"
#include "call.h"

extern "C"
void MARKER(char* a) {
    marker(a);
}

extern "C"
void DUMP(char* a, int64_t i) {
    printf("%s : %d\n", a, i);
}

extern "C"
int64_t SLENGTH(Thread& thread, int64_t i) {
    Value const& v = (thread.registers+DEFAULT_NUM_REGISTERS)[i];
    return v.isVector() ? v.length : 1;
}

extern "C"
int64_t ELENGTH(Thread& thread, Environment* env, int64_t i) {
    Value const& v = env->getRecursive((String)i);
    return v.isVector() ? v.length : 1;
}

extern "C"
Value SLOAD(Thread& thread, int64_t i) {
    return (thread.registers+DEFAULT_NUM_REGISTERS)[i];
}

extern "C"
Value ELOAD(Thread& thread, Environment* env, int64_t i) {
    return env->getRecursive((String)i);
}

extern "C"
void SSTORE(Thread& thread, int64_t i, Value v) {
    (thread.registers+DEFAULT_NUM_REGISTERS)[i] = v;
}

extern "C"
void ESTORE(Thread& thread, Environment* env, int64_t i, Value v) {
    env->insertRecursive((String)i) = v;
}

// Inline this!
extern "C"
bool GTYPE(Thread& thread, Value value, int64_t type) {
    return (value.type == type);
}

extern "C"
double const* UNBOX_double(Thread& thread, Value& a) {
    return ((Double const&)a).v();
}

extern "C"
int64_t const* UNBOX_integer(Thread& thread, Value& a) {
    return ((Integer const&)a).v();
}

extern "C"
int8_t const* UNBOX_logical(Thread& thread, Value& a) {
    return (int8_t const*)((Logical const&)a).v();
}

extern "C"
int8_t const** UNBOX_character(Thread& thread, Value& a) {
    return (int8_t const**)((Character const&)a).v();
}

extern "C"
Value BOX_double(Thread& thread, double* d, int64_t len) {
    if(len <= 16) {
    	Double a(len);
    	memcpy(a.v(), d, len*sizeof(double));
        return a;
    }
    else {
        Value a;
        Value::Init(a, Type::Double, len);
        a.p = d;
        return a;
    }
}

extern "C"
Value BOX_integer(Thread& thread, int64_t* d, int64_t len) {
    if(len <= 16) {
        Integer a(len);
        memcpy(a.v(), d, len*sizeof(int64_t));
        return a;
    }
    else {
        Value a;
        Value::Init(a, Type::Integer, len);
        a.p = d;
        return a;
    }
}

extern "C"
Value BOX_logical(Thread& thread, int8_t* d, int64_t len) {
	if(len <= 16) {
        Logical a(len);
	    memcpy(a.v(), d, len*sizeof(int8_t));
        return a;
    }
    else {
        Value a;
        Value::Init(a, Type::Logical, len);
        a.p = d;
        return a;
    }
}

extern "C"
Value BOX_character(Thread& thread, int8_t** d, int64_t len) {
    if(len <= 16) {
        Character a(len);
        memcpy(a.v(), d, len*sizeof(int8_t*));
        return a;
    }
    else {
        Value a;
        Value::Init(a, Type::Character, len);
        a.p = d;
        return a;
    }
}

extern "C"
Environment* LOAD_environment(Thread& thread, int64_t i) {
    return (Environment*)i;
}

extern "C"
Environment* NEW_environment(Thread& thread, Environment* l, Environment* d, Value v) {
    Environment* env = new Environment();
    env->init(l, d, v);
    return env;
}

extern "C"
void NEW_frame(Thread& thread, Environment* environment, int64_t prototype, int64_t returnpc, int64_t returnbase, int64_t dest, Environment* env) {
    StackFrame& frame = thread.push();
    frame.environment = environment;
    frame.prototype = (Prototype const*) prototype;
    frame.returnpc = (Instruction const*) returnpc;
    frame.returnbase = (Value*) returnbase;
    frame.dest = dest;
    frame.env = env;
}

extern "C"
Prototype const* GET_prototype(Thread& thread, Value v) {
    return ((Function const&)v).prototype();
}

extern "C"
Value GET_attr(Thread& thread, Value a, int8_t** name) {
    // Should inline this check so we can avoid a guard when we know the result type.
    if(a.isObject()) {
        //printf("Getting attribute: %s: ", (String)name[0]);
        //std::cout << thread.state.stringify(((Object const&)a).get((String)name[0]));
        return ((Object const&)a).get((String)name[0]);
    }
    else
        return Null::Singleton();
}

extern "C"
int64_t ALENGTH(Thread& thread, Value a, int8_t** name) {
    // Should inline this check so we can avoid a guard when we know the result type.
    if(a.isObject()) {
        Value const& v = ((Object const&)a).get((String)name[0]);
        return v.isVector() ? v.length : 1;
    }
    else
        return 0;
}

extern "C"
Value GET_strip(Thread& thread, Value a) {
    return ((Object const&)a).base();
}

extern "C"
int64_t OLENGTH(Thread& thread, Value a) {
    Value const& v = ((Object const&)a).base();
    return v.isVector() ? v.length : 1;
}

void* MALLOC(int64_t length, size_t elementsize) {
    int64_t l = length;
    // round l up to nearest even number so SSE can work on tail region
    l += (int64_t)((uint64_t)l & 1);
    int64_t length_aligned = (l < 128) ? (l + 1) : l;
    void* p = GC_malloc_atomic(elementsize*length_aligned);
    assert(l < 128 || (0xF & (int64_t)p) == 0);
    if( (0xF & (int64_t)p) != 0)
        p =  (char*)p + 0x8;
    return p;
}

extern "C"
double* MALLOC_double(Thread& thread, int64_t length) {
    return (double*)MALLOC(length, sizeof(Double::Element));
}

extern "C"
int64_t* MALLOC_integer(Thread& thread, int64_t length) {
    return (int64_t*)MALLOC(length, sizeof(Integer::Element));
}

extern "C"
int8_t* MALLOC_logical(Thread& thread, int64_t length) {
    return (int8_t*)MALLOC(length, sizeof(Logical::Element));
}

extern "C"
int8_t** MALLOC_character(Thread& thread, int64_t length) {
    return (int8_t**)MALLOC(length, sizeof(Character::Element));
}


