

#include <locale.h>

#include "../../../src/runtime.h"
#include "../../../src/compiler.h"
#include "../../../src/parser.h"
#include "../../../src/library.h"
#include "../../../src/coerce.h"

extern "C"
void getlocale_map(Thread& thread,
    Character::Element& r, Integer::Element category)
{
    char const* v = setlocale(category, NULL);
    if(v)
        r = thread.internStr(v);
    else
        r = Strings::empty;
}

extern "C"
void setlocale_map(Thread& thread,
    Character::Element& r, Integer::Element category, Character::Element locale)
{
    char const* v = setlocale(category, locale->s);
    if(v)
        r = thread.internStr(v);
    else
        r = Strings::empty;
}

