
#include <unistd.h>

#include "../../../src/runtime.h"

extern "C"
Value sysunlink(State& state, Value const* args)
{
    auto c = static_cast<Character const&>(args[0]);
    //auto recursive = static_cast<Logical const&>(args[1]);
    //auto force = static_cast<Logical const&>(args[2]);

    bool ok = true;

    for(int64_t j = 0; j < c.length(); ++j) {
        if(c[j] == NULL)
            continue;
        
        int result = unlink(c[j]->s);
        ok = ok || (result == 0);
    }
    
    return Integer::c(ok ? 0 : 1);
}
