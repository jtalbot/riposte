#include "strings.h"

// Strings are actually initialized in State::State

#define DEFINE(name, string, ...) String Strings::name = 0;
STRINGS(DEFINE)
#undef DEFINE

bool Eq(String s, String t)
{
    return s == t ||
            (s && t && strcmp(s->s, t->s) == 0);
}

bool Neq(String s, String t)
{
    return s != t &&
        (!s || !t || strcmp(s->s, t->s) != 0);
}

