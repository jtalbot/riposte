
#include <stdlib.h>
#include <time.h>

#include "../../../src/runtime.h"

extern "C"
Value systime(Thread& thread, Value const* args) {
    time_t t = time(NULL);
    Integer r = Integer::c(t);

    return r;
}

