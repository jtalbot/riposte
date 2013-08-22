
#include <sys/types.h>
#include <sys/stat.h>

#include "../../../src/runtime.h"
#include "../../../src/coerce.h"

extern "C"
Value charToRaw(Thread& thread, Value const* args)
{
    Character const& c = (Character const&)args[0];

    String s = c[0];
    int64_t len = strlen(s->s);

    // eventually we should be able to get rid of this copy since
    // String and Raw will be the same thing.
    Raw r(len);
    memcpy(r.v(), s->s, len);

    return r;
}

extern "C"
Value rawToChar(Thread& thread, Value const* args)
{
    Raw const& c = (Raw const&)args[0];

    std::string s((const char*)c.v(), c.length());

    return Character::c(thread.internStr(s));
}

extern "C"
Value rawToCharacters(Thread& thread, Value const* args)
{
    Raw const& c = (Raw const&)args[0];

    Character r(c.length());
    for(int64_t i = 0; i < c.length(); ++i) {
        r[i] = thread.internStr(std::string(1, c[i]));
    }
    return r;
}

extern "C"
Value rawToDouble(Thread& thread, Value const* args)
{
    Raw const& c = (Raw const&)args[0];
    int64_t size = ((Integer const&)args[1])[0];

    int64_t n = c.length() / size;
    Double r(n);

    if(size == 4) {
        union {
            float f;
            char b[4];
        } t;
        for(int64_t i = 0; i < n; ++i) {
            memcpy(t.b, c.v()+4*i, 4);
            r[i] = t.f;
        }
    }
    else if(size == 8) {
        union {
            double d;
            char b[8];
        } t;
        for(int64_t i = 0; i < n; ++i) {
            memcpy(t.b, c.v()+8*i, 8);
            r[i] = t.d;
        }
    }
    else
        _error("Unknown size");
    
    return r;
}

extern "C"
Value rawToInteger(Thread& thread, Value const* args)
{
    Raw const& c = (Raw const&)args[0];
    int64_t size = ((Integer const&)args[1])[0];
    bool sign = Logical::isTrue(((Logical const&)args[2])[0]);

    int64_t n = c.length() / size;
    Integer r(n);

    if(size == 1) {
        union {
            int8_t s;
            uint8_t u;
            char b[1];
        } t;
        for(int64_t i = 0; i < n; ++i) {
            memcpy(t.b, c.v()+i, 1);
            r[i] = sign ? (int64_t)t.s : (int64_t)t.u;
        }
    }
    else if(size == 2) {
        union {
            int16_t s;
            uint16_t u;
            char b[2];
        } t;
        for(int64_t i = 0; i < n; ++i) {
            memcpy(t.b, c.v()+2*i, 2);
            r[i] = sign ? (int64_t)t.s : (int64_t)t.u;
        }
    }
    else if(size == 4) {
        union {
            int32_t s;
            uint32_t u;
            char b[4];
        } t;
        for(int64_t i = 0; i < n; ++i) {
            memcpy(t.b, c.v()+4*i, 4);
            r[i] = sign ? (int64_t)t.s : (int64_t)t.u;
        }
    }
    else if(size == 8) {
        union {
            int64_t s;
            char b[8];
        } t;
        for(int64_t i = 0; i < n; ++i) {
            memcpy(t.b, c.v()+8*i, 8);
            r[i] = t.s;
        }
    }
    else
        _error("Unknown size");
    
    return r;
}

extern "C"
Value rawToLogical(Thread& thread, Value const* args)
{
    Raw const& c = (Raw const&)args[0];
    int64_t size = ((Integer const&)args[1])[0];

    int64_t n = c.length() / size;
    Logical r(n);

    if(size == 1) {
        union {
            int8_t s;
            char b[1];
        } t;
        for(int64_t i = 0; i < n; ++i) {
            memcpy(t.b, c.v()+i, 1);
            r[i] = t.s ? Logical::TrueElement : Logical::FalseElement;
        }
    }
    else if(size == 2) {
        union {
            int16_t s;
            char b[2];
        } t;
        for(int64_t i = 0; i < n; ++i) {
            memcpy(t.b, c.v()+2*i, 2);
            r[i] = t.s ? Logical::TrueElement : Logical::FalseElement;
        }
    }
    else if(size == 4) {
        union {
            int32_t s;
            char b[4];
        } t;
        for(int64_t i = 0; i < n; ++i) {
            memcpy(t.b, c.v()+4*i, 4);
            r[i] = t.s ? Logical::TrueElement : Logical::FalseElement;
        }
    }
    else if(size == 8) {
        union {
            int64_t s;
            char b[8];
        } t;
        for(int64_t i = 0; i < n; ++i) {
            memcpy(t.b, c.v()+8*i, 8);
            r[i] = t.s ? Logical::TrueElement : Logical::FalseElement;
        }
    }
    else
        _error("Unknown size");
    
    return r;
}

