
#include <stdlib.h>
#include <unistd.h>

#include "../../../src/runtime.h"
#include "../../../src/compiler.h"
#include "../../../src/parser.h"
#include "../../../src/library.h"
#include "../../../src/coerce.h"

extern "C"
void syssetenv_map(Thread& thread,
    Logical::Element& r,
    Character::Element name, Character::Element value)
{
    r = setenv(name, value, 1) == 0
            ? Logical::TrueElement
            : Logical::FalseElement;
} 

extern "C"
void sysunsetenv_map(Thread& thread,
    Logical::Element& r,
    Character::Element name)
{
    r = unsetenv(name) == 0
            ? Logical::TrueElement
            : Logical::FalseElement;
} 

