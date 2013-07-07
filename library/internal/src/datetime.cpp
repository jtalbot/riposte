
#include <time.h>
#include <xlocale.h>

#include "../../../src/runtime.h"
#include "../../../src/compiler.h"
#include "../../../src/parser.h"
#include "../../../src/library.h"
#include "../../../src/coerce.h"

extern "C"
void strptime_map(Thread& thread,
    double& sec, int64_t& min, int64_t& hour, int64_t& mday, int64_t& mon, 
    int64_t& year, int64_t& wday, int64_t& yday, int64_t& isdst,
    String x, String format, String tz) {

    tm t;
    memset(&t, 0, sizeof(tm));
    if( strptime(x, format, &t) != NULL ) {
        // sets the wday and yday elements 
        mktime(&t);

        sec = t.tm_sec;
        min = t.tm_min;
        hour = t.tm_hour;
        mday = t.tm_mday;
        mon = t.tm_mon;
        year = t.tm_year;
        wday = t.tm_wday;
        yday = t.tm_yday;
        isdst = t.tm_isdst;
    }
    else {
        sec = Double::NAelement;
        min = hour = mday = mon = year = wday = yday = isdst 
            = Integer::NAelement;
    }
}

extern "C"
void mktime_map(Thread& thread,
    double& out,
    double sec, int64_t min, int64_t hour, int64_t mday, int64_t mon, 
    int64_t year, int64_t wday, int64_t yday, int64_t isdst) {
    tm t;
    t.tm_sec = sec;
    t.tm_min = min;
    t.tm_hour = hour;
    t.tm_mday = mday;
    t.tm_mon = mon;
    t.tm_year = year;
    t.tm_wday = wday;
    t.tm_yday = yday;
    t.tm_isdst = isdst;

    out = (double)mktime(&t);
}
