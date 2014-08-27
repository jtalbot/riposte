
#include <string>

#include "../../../src/runtime.h"
#include "../../../src/coerce.h"

extern "C"
Value tabulate(State& state, Value const* args) {
    Integer const& b = (Integer const&)args[0];
    Integer const& n = (Integer const&)args[1];

    Integer r(n[0]);
   
    for(int64_t i = 0; i < r.length(); ++i)
        r[i] = 0;
     
    for(int64_t i = 0; i < b.length(); ++i)
        r[b[i]-1]++;

    return r;
}

