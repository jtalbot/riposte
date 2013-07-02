
#include <time.h>
#include <xlocale.h>

#include "../../../src/runtime.h"
#include "../../../src/compiler.h"
#include "../../../src/parser.h"
#include "../../../src/library.h"
#include "../../../src/coerce.h"

extern "C"
List strptime(Thread& thread, String x, String format, String tz) {
    tm t;
    strptime_l(x, format, &t, loc);
}
