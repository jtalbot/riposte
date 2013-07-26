
#include <libgen.h>

#include "../../../src/runtime.h"
#include "../../../src/compiler.h"
#include "../../../src/parser.h"
#include "../../../src/library.h"
#include "../../../src/coerce.h"

extern "C"
void basename_map(Thread& thread,
    Character::Element& r, Character::Element f)
{
    // basename may mutate the string, so copy...
    char* copy = (char*)malloc(1+strlen(f));
    strcpy(copy, f);
    char* p = basename(copy);
    r = thread.internStr(p);
    free(copy);
}

extern "C"
void dirname_map(Thread& thread,
    Character::Element& r, Character::Element f)
{
    // dirname may mutate the string, so copy...
    char* copy = (char*)malloc(1+strlen(f));
    strcpy(copy, f);
    char* p = dirname(copy);
    r = thread.internStr(p);
    free(copy);
}

