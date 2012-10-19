#include <string>
#include <sstream>
#include <stdexcept>
#include <string>
#include <algorithm>
#include <math.h>

#include "value.h"
#include "type.h"
#include "bc.h"
#include "interpreter.h"
#include "call.h"

extern void DebugMarker(char* a);

extern "C"
void MARKER(char* a) {
    DebugMarker(a);
}

extern "C"
void DUMP(char* a, int64_t i) {
    printf("%s : %d\n", a, i);
}

extern "C"
void DUMPD(char* a, double d) {
    printf("%s : %f\n", a, d);
}

extern "C"
void DUMPL(char* a, char d) {
    printf("%s : %d\n", a, d);
}

extern "C"
Value curenv(Thread& thread) {
    Value v;
    return REnvironment::Init(v, thread.frame.environment);
}

extern "C"
Value has(Thread& thread, REnvironment e, int8_t** name) {
    return Logical::c(e.environment()->has((String)*name) 
            ? Logical::TrueElement : Logical::FalseElement);
}

extern "C"
int64_t LENGTH(Thread& thread, Value v) {
    return v.isVector() ? v.length : 1;
}

extern "C"
Value SLOAD(Thread& thread, int64_t i) {
    return (thread.registers+DEFAULT_NUM_REGISTERS)[i];
}

extern "C"
Value ELOAD(Thread& thread, REnvironment env, int8_t** i) {
    return env.environment()->get((String)*i);
}

extern "C"
void SSTORE(Thread& thread, int64_t i, Value v) {
    (thread.registers+DEFAULT_NUM_REGISTERS)[i] = v;
}

extern "C"
void ESTORE(Thread& thread, REnvironment env, int8_t** i, Value v) {
    env.environment()->insert((String)*i) = v;
}

// Inline this!
extern "C"
bool GTYPE(Thread& thread, Value value, int64_t type) {
    return (value.type == type);
}

extern "C"
double const* UNBOX_double(Thread& thread, Value& a, int64_t length) {
    if(!a.isDouble() || a.length != length)
        return 0;
    else {
        return ((Double const&)a).v();
    }
}

extern "C"
int64_t const* UNBOX_integer(Thread& thread, Value& a, int64_t length) {
    if(!a.isInteger() || a.length != length)
        return 0;
    else {
        return ((Integer const&)a).v();
    }
}

extern "C"
int8_t const* UNBOX_logical(Thread& thread, Value& a, int64_t length) {
    if(!a.isLogical() || a.length != length)
        return 0;
    else
        return (int8_t const*)((Logical const&)a).v();
}

extern "C"
int8_t const** UNBOX_character(Thread& thread, Value& a, int64_t length) {
    if(!a.isCharacter() || a.length != length)
        return 0;
    else
        return (int8_t const**)((Character const&)a).v();
}

extern "C"
Value const* UNBOX_list(Thread& thread, Value& a, int64_t length) {
    if(!a.isList() || a.length != length)
        return 0;
    else
        return (Value const*)((List const&)a).v();
}

extern "C"
Value BOX_double(Thread& thread, double* d, int64_t len, bool takeOwnership) {
    if(len == 1) {
        Double a(1);
        a[0] = *d;
        return a;
    }

    if(false/*takeOwnership*/) {
        Double a;
        Value::Init(a, Type::Double, len);
        a.p = d;
        return a;
    }
    else {
        Double a(len);
        memcpy(a.v(), d, len*sizeof(double));
        return a;
    }
}

extern "C"
Value BOX_integer(Thread& thread, int64_t* d, int64_t len, bool takeOwnership) {
    Integer a(len);
    memcpy(a.v(), d, len*sizeof(int64_t));
    return a;
}

extern "C"
Value BOX_logical(Thread& thread, int8_t* d, int64_t len, bool takeOwnership) {
    Logical a(len);
    memcpy(a.v(), d, len*sizeof(int8_t));
    return a;
}

extern "C"
Value BOX_character(Thread& thread, int8_t** d, int64_t len, bool takeOwnership) {
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
Value BOX_list(Thread& thread, Value* d, int64_t len, bool takeOwnership) {
    List a(len);
    memcpy(a.v(), d, len*sizeof(List::Element));
    return a;
}

extern "C"
Value NEW_environment(Thread& thread) {
    Environment* env = new Environment();
    Value v;
    return REnvironment::Init(v, env);
}

extern "C"
void PUSH(Thread& thread, REnvironment environment, int64_t prototype, int64_t returnpc, int64_t returnbase, int64_t env1, int64_t env2, int64_t dest) {
    StackFrame& frame = thread.push();
    frame.environment = environment.environment();
    frame.prototype = (Prototype const*) prototype;
    frame.returnpc = (Instruction const*) returnpc;
    frame.returnbase = (Value*) returnbase;
    frame.dest = dest;
    REnvironment env;
    env.header = env1;
    env.i = env2;
    frame.env = env.environment();
}

extern "C"
void POP(Thread& thread) {
    thread.pop();
}

extern "C"
Prototype const* GET_prototype(Thread& thread, Function f) {
    return f.prototype();
}

extern "C"
Value GET_environment(Thread& thread, Function f) {
    Value v;
    return REnvironment::Init(v, f.environment());
}

extern "C"
Value GET_attr(Thread& thread, Value a, int8_t** name) {
    // Should inline this check so we can avoid a guard when we know the result type.
    //printf("Getting attribute: %s: ", (String)name[0]);
    //std::cout << thread.state.stringify(((Object const&)a).get((String)name[0]));
    return ((Object const&)a).get((String)name[0]);
}

extern "C"
void SET_attr(Thread& thread, Value a, int8_t** name, Value v) {
    ((Object&)a).insertMutable((String)name[0], v);
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
Value SET_strip(Thread& thread, Value a) {
    _error("Can't set object base yet");
}

extern "C"
int64_t OLENGTH(Thread& thread, Value a) {
    Value const& v = ((Object const&)a).base();
    return v.isVector() ? v.length : 1;
}

void* MALLOC(int64_t length, size_t elementsize) {
    int64_t l = length;
    // round l up to nearest even number so SSE can work on tail region
    int64_t length_aligned = (l < 128) ? (l + 1) : l;
    length_aligned += (int64_t)((uint64_t)length_aligned & 1);
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

extern "C"
Value* MALLOC_list(Thread& thread, int64_t length) {
    return (Value*)MALLOC(length, sizeof(List::Element));
}

extern "C"
double* REALLOC_double(Thread& thread, double* v, int64_t& alloclen, int64_t length) {
    if(length > alloclen) {
        if(alloclen == 0) {
            alloclen = length;
            return (double*)MALLOC(length, sizeof(Double::Element) );
        }
        else {
            length = nextPow2(length);
            double* w = (double*)MALLOC(length, sizeof(Double::Element));
            memcpy(w, v, alloclen*sizeof(Double::Element));
            // fill remainder with NAs
            for(size_t i = alloclen; i < length; i++) w[i] = Double::NAelement;
            //printf("REALLOC_double: %d->%d\n", alloclen, length);
            alloclen = length;
            return w;
        }
    }
    else {
        return v;
    }
}

extern "C"
int64_t* REALLOC_integer(Thread& thread, int64_t* v, int64_t& alloclen, int64_t length) {
    if(length > alloclen) {
        if(alloclen == 0) {
            alloclen = length;
            return (int64_t*)MALLOC(length, sizeof(Integer::Element));
        }
        else {
            length = nextPow2(length);
            int64_t* w = (int64_t*)MALLOC(length, sizeof(Integer::Element));
            memcpy(w, v, alloclen*sizeof(Integer::Element));
            // fill remainder with NAs
            for(size_t i = alloclen; i < length; i++) w[i] = Integer::NAelement;
            //printf("REALLOC_integer: %d->%d   (%li)\n", alloclen, length, w);
            alloclen = length;
            return w;
        }
    }
    else {
        return v;
    }
}

extern "C"
int8_t* REALLOC_logical(Thread& thread, int8_t* v, int64_t& alloclen, int64_t length) {
    if(length > alloclen) {
        if(alloclen == 0) {
            alloclen = length;
            return (int8_t*)MALLOC(length, sizeof(Logical::Element));
        }
        else {
            length = nextPow2(length);
            int8_t* w = (int8_t*)MALLOC(length, sizeof(Logical::Element));
            memcpy(w, v, alloclen*sizeof(Logical::Element));
            // fill remainder with NAs
            for(size_t i = alloclen; i < length; i++) w[i] = Logical::NAelement;
            //printf("REALLOC_logical: %d->%d   (%li)\n", alloclen, length, w);
            alloclen = length;
            return w;
        }
    }
    else {
        return v;
    }
}

extern "C"
int8_t** REALLOC_character(Thread& thread, int8_t** v, int64_t& alloclen, int64_t length) {
    if(length > alloclen) {
        if(alloclen == 0) {
            alloclen = length;
            return (int8_t**)MALLOC(length, sizeof(Character::Element));
        }
        else {
            length = nextPow2(length);
            int8_t** w = (int8_t**)MALLOC(length, sizeof(Character::Element));
            memcpy(w, v, alloclen*sizeof(Character::Element));
            // fill remainder with NAs
            for(size_t i = alloclen; i < length; i++) w[i] = (int8_t*)Character::NAelement;
            alloclen = length;
            return w;
        }
    }
    else {
        return v;
    }
}

struct ValueStub {
    int64_t a, b;
};

extern "C"
Value* REALLOC_list(Thread& thread, Value* v, int64_t& alloclen, int64_t length) {
    if(length > alloclen) {
        if(alloclen == 0) {
            alloclen = length;
            return (Value*)MALLOC(length, sizeof(List::Element));
        }
        else {
            length = nextPow2(length);
            Value* w = (Value*)MALLOC(length, sizeof(List::Element));
            memcpy(w, v, alloclen*sizeof(List::Element));
            // fill remainder with NAs
            for(size_t i = alloclen; i < length; i++) w[i] = Null::Singleton();
            alloclen = length;
            return w;
        }
    }
    else {
        return v;
    }
}

extern "C"
Value GET_lenv(Thread& thread, REnvironment env) {
    Value v;
    return REnvironment::Init(v, env.environment()->lexical);
}

extern "C"
Value GET_denv(Thread& thread, REnvironment env) {
    Value v;
    return REnvironment::Init(v, env.environment()->dynamic);
}

extern "C"
Value GET_call(Thread& thread, REnvironment env) {
    return env.environment()->call;
}

extern "C"
void SET_lenv(Thread& thread, REnvironment env, REnvironment lenv) {
    env.environment()->lexical = lenv.environment();
}

extern "C"
void SET_denv(Thread& thread, REnvironment env, REnvironment denv) {
    env.environment()->dynamic = denv.environment();
}

extern "C"
void SET_call(Thread& thread, REnvironment env, Value call) {
    env.environment()->call = call;
}

extern "C"
double Riposte_random(Thread& thread) {
    return rand() / (double)RAND_MAX;
}

extern "C"
double sin(double d) {
    return ::sin(d);
}

extern "C"
double pow(double d, double e) {
    return ::pow(d, e);
}

extern "C"
double log(double d) {
    return ::log(d);
}

extern "C"
double exp(double d) {
    return ::exp(d);
}
