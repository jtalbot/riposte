
#include <sys/types.h>
#include <sys/stat.h>

#include "../../../src/runtime.h"
#include "../../../src/coerce.h"

extern "C"
Value charToRaw(State& state, Value const* args)
{
    auto c = static_cast<Character const&>(args[0]);

    String s = c[0];
    int64_t len = strlen(s->s);

    // eventually we should be able to get rid of this copy since
    // String and Raw will be the same thing.
    Raw r(len);
    memcpy(r.v(), s->s, len);

    return r;
}

extern "C"
Value rawToChar(State& state, Value const* args)
{
    auto c = static_cast<Raw const&>(args[0]);

    std::string s((const char*)c.v(), c.length());

    return Character::c(MakeString(s));
}

extern "C"
Value rawToCharacters(State& state, Value const* args)
{
    auto c = static_cast<Raw const&>(args[0]);

    Character r(c.length());
    for(int64_t i = 0; i < c.length(); ++i) {
        r[i] = MakeString(std::string(1, c[i]));
    }
    return r;
}

extern "C"
Value rawToDouble(State& state, Value const* args)
{
    auto c = static_cast<Raw const&>(args[0]);
    int64_t size = static_cast<Integer const&>(args[1])[0];

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
Value rawToInteger(State& state, Value const* args)
{
    auto c = static_cast<Raw const&>(args[0]);
    int64_t size = static_cast<Integer const&>(args[1])[0];
    bool sign = Logical::isTrue(static_cast<Logical const&>(args[2])[0]);

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
            // Handle NA for 4-byte integers since that's what gnu R
            // will give us. The R documentation says that, in general,
            // size changes won't preserve NAs.
            r[i] =
                t.s == std::numeric_limits<int32_t>::min()
                    ? Integer::NAelement
                    : sign ? (int64_t)t.s : (int64_t)t.u;
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
Value rawToLogical(State& state, Value const* args)
{
    auto c = static_cast<Raw const&>(args[0]);
    int64_t size = static_cast<Integer const&>(args[1])[0];

    int64_t n = c.length() / size;
    Logical r(n);

    if(size == 1) {
        union {
            int8_t s;
            char b[1];
        } t;
        for(int64_t i = 0; i < n; ++i) {
            memcpy(t.b, c.v()+i, 1);
            r[i] = t.s < 0
                        ? Logical::NAelement
                        : t.s ? Logical::TrueElement : Logical::FalseElement;
        }
    }
    else if(size == 2) {
        union {
            int16_t s;
            char b[2];
        } t;
        for(int64_t i = 0; i < n; ++i) {
            memcpy(t.b, c.v()+2*i, 2);
            r[i] = t.s < 0
                        ? Logical::NAelement
                        : t.s ? Logical::TrueElement : Logical::FalseElement;
        }
    }
    else if(size == 4) {
        union {
            int32_t s;
            char b[4];
        } t;
        for(int64_t i = 0; i < n; ++i) {
            memcpy(t.b, c.v()+4*i, 4);
            r[i] = t.s < 0 
                        ? Logical::NAelement
                        : t.s ? Logical::TrueElement : Logical::FalseElement;
        }
    }
    else if(size == 8) {
        union {
            int64_t s;
            char b[8];
        } t;
        for(int64_t i = 0; i < n; ++i) {
            memcpy(t.b, c.v()+8*i, 8);
            r[i] = t.s < 0
                        ? Logical::NAelement
                        : t.s ? Logical::TrueElement : Logical::FalseElement;
        }
    }
    else
        _error("Unknown size");
    
    return r;
}

