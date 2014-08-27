
#include <stdlib.h>
#include <unistd.h>

#include "../../../src/runtime.h"
#include "../../../src/compiler.h"
#include "../../../src/parser.h"
#include "../../../src/library.h"
#include "../../../src/coerce.h"

extern "C"
void sysgetenv_map(State& state,
    Character::Element& r,
    Character::Element name, Character::Element unset)
{
    char const* var = getenv(name->s);
    r = (var == NULL) 
        ? unset 
        : state.internStr(var);
} 

extern "C"
char **environ;

extern "C"
Value sysgetenv(State& state, Value const* args)
{
    // TODO: implement. Apparently you can only get the
    // environment variables from the executable, not a library?
    return Null::Singleton();
}


