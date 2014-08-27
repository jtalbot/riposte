
#include <stdlib.h>
#include <time.h>

#include "../../../src/value.h"
class State;

extern "C"
Value systime(State& state, Value const* args) {
    return Integer::c(time(NULL));
}

