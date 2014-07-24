
#include "../value.h"
#include <limits>

Double::_doublena doublena = {0x7fff000000001953};

double R_NaN    = std::numeric_limits<double>::quiet_NaN();  /* IEEE NaN */
double R_PosInf = std::numeric_limits<double>::infinity();   /* IEEE Inf */
double R_NegInf = -std::numeric_limits<double>::infinity();  /* IEEE -Inf */
double R_NaReal = doublena.d;                                /* NA_REAL: IEEE */
int    R_NaInt  = std::numeric_limits<int>::min();           /* NA_INTEGER:= INT_MIN currently */

int R_IsNA(double d) {         /* True for R's NA only */
    return Double::isNA(d);
}

int R_IsNaN(double d) {        /* True for special NaN, *not* for NA */
    return Double::isNaN(d);
}

int R_finite(double d) {       /* True if none of NA, NaN, +/-Inf */
    return (d==d) && d < R_PosInf && d > R_NegInf;
}

int R_isnancpp(double d) {     /* in arithmetic.c */
    return d!=d;
}

