
#include <stdio.h>
#include <stdarg.h>

#include "../../../src/runtime.h"
#include "../../../src/coerce.h"

extern "C"
void sprintf_map(Thread& thread, String& r, String fmt, ...) {
    static const size_t maxlength = 8192; // from R documentation
    char out[maxlength];

    va_list arglist;
    va_start( arglist, fmt );
    vsnprintf( out, maxlength, fmt, arglist );
    va_end( arglist );

    r = thread.internStr(out);
}

