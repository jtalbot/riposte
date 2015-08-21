
#include <stdlib.h>

#include "../../../src/riposte.h"
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
        : MakeString(var);
} 

extern "C"
Value sysgetenv(State& state, Value const* args)
{
    size_t cnt = 0;
    for(char** env = Riposte::getEnv(); *env; ++env) {
        cnt++;
    }

    Character result(cnt);

    size_t i = 0;
    for(char** env = Riposte::getEnv(); *env; ++env) {
        result[i] = MakeString(*env);
        ++i;
    }
    
    return result;
}


