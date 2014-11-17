
#include "api.h"
#include <R_ext/Utils.h>

/* ../../main/sort.c : */
void    R_isort(int* x, int n) {
    std::sort(x, x+n);
}

void    R_rsort(double* x, int n) {
    std::sort(x, x+n);
}

void    rsort_with_index(double *, int *, int) {
    _NYI("rsort_with_index");
}

void    rPsort(double*, int, int) {
    _NYI("rPsort");
}

/* ../../main/util.c  and others : */
const char *R_ExpandFileName(const char *) {
    _NYI("R_ExpandFileName");
}

Rboolean Rf_isBlankString(const char *) {
    _NYI("Rf_isBlankString");
}

char *R_tmpnam(const char *prefix, const char *tempdir) {
    _NYI("R_tmpnam");
}

void R_CheckUserInterrupt(void) {
    // TODO: actually check for errors.
    // Figure out how to jump out of the user code.
}

void R_CheckStack(void) {
    _NYI("R_CheckStack");
}

void R_CheckStack2(size_t) {
    _NYI("R_CheckStack2");
}

extern "C" {
int interv_(double *xt, int *n, double *x,
            Rboolean *rightmost_closed, Rboolean *all_inside,
            int *ilo, int *mflag) {
    _NYI("interv");
}
}
