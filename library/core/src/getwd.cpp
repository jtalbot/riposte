
#include <unistd.h>

#include "../../../src/runtime.h"
#include "../../../src/coerce.h"

extern "C"
Value getwd_impl(State& state, Value const* args) {
    char buf[2048];
    char* wd = getcwd(buf, 2048);
    if(wd != 0) {
        Character r(1);
        r[0] = MakeString(wd);
        return r;
    }
    else {
        return Null();
    }
}

extern "C"
Value setwd_impl(State& state, Value const* args) {
    Value r = getwd_impl(state, 0);
    auto wd = static_cast<Character const&>(args[0]);
    if(wd.length() == 1)
        chdir(wd[0]->s);
    return r;
}

