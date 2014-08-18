
#include "api.h"
#include <R_ext/Utils.h>

/* ../../main/sort.c : */
void    R_isort(int*, int) {
    _NYI("R_isort");
}

void    R_rsort(double*, int) {
    _NYI("R_rsort");
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
    _NYI("R_CheckUserInterrupt");
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
