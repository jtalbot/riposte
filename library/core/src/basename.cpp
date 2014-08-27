
#include <libgen.h>

#include "../../../src/runtime.h"
#include "../../../src/compiler.h"
#include "../../../src/parser.h"
#include "../../../src/library.h"
#include "../../../src/coerce.h"

extern "C"
void basename_map(State& state,
    Character::Element& r, Character::Element f)
{
    // basename may mutate the string, so copy...
    char* copy = (char*)malloc(1+strlen(f->s));
    strcpy(copy, f->s);
    char* p = basename(copy);
    r = state.internStr(p);
    free(copy);
}

extern "C"
void dirname_map(State& state,
    Character::Element& r, Character::Element f)
{
    // dirname may mutate the string, so copy...
    char* copy = (char*)malloc(1+strlen(f->s));
    strcpy(copy, f->s);
    char* p = dirname(copy);
    r = state.internStr(p);
    free(copy);
}

