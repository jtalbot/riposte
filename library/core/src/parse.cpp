
#include <sys/types.h>
#include <sys/stat.h>

#include "../../../src/runtime.h"
#include "../../../src/compiler.h"
#include "../../../src/parser.h"
#include "../../../src/library.h"
#include "../../../src/coerce.h"

extern "C"
Value parse(State& state, Value const* args)
{
    auto c = static_cast<Character const&>(args[0]);
    //auto n = static_cast<Integer const&>(args[1]);
    auto name = static_cast<Character const&>(args[2]);

    Value result;
    parse(state.global, name[0]->s, c[0]->s, strlen(c[0]->s), true, result);
    return result;
}

