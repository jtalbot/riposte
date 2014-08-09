
#include <sys/types.h>
#include <sys/stat.h>

#include "../../../src/runtime.h"
#include "../../../src/compiler.h"
#include "../../../src/parser.h"
#include "../../../src/library.h"
#include "../../../src/coerce.h"

extern "C"
void fileexists_map(Thread& thread,
    Logical::Element& r, Character::Element f)
{
    struct stat t;
    r = lstat(f->s, &t) == 0
            ? Logical::TrueElement
            : Logical::FalseElement;
}

extern "C"
void direxists_map(Thread& thread,
    Logical::Element& r, Character::Element f)
{
    struct stat t;
    r = (lstat(f->s, &t) == 0 && S_ISDIR(t.st_mode))
            ? Logical::TrueElement
            : Logical::FalseElement;
}

