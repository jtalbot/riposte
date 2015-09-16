#include "strings.h"

// Strings are actually initialized in State::State

#define DEFINE(name, string, ...) String Strings::name = 0;
STRINGS(DEFINE)
#undef DEFINE

extern uint64_t
siphash( const uint8_t *in, uint64_t inlen, const uint64_t k0, const uint64_t k1 );

size_t HashSlow(String s)
{
    size_t h = (size_t)siphash((uint8_t*)s->s, s->length,
        0x3d9c62a3403c404e,
        0xe6ab9a6ad910c7c2);

    s->hash = h;

    return h;
}
