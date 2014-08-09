
#include <R_ext/Utils.h>

/* ../../main/sort.c : */
void    R_isort(int*, int) {
    throw "NYI: R_isort";
}

void    R_rsort(double*, int) {
    throw "NYI: R_rsort";
}

void    rsort_with_index(double *, int *, int) {
    throw "NYI: rsort_with_index";
}

void    rPsort(double*, int, int) {
    throw "NYI: rPsort";
}

/* ../../main/util.c  and others : */
const char *R_ExpandFileName(const char *) {
    throw "NYI: R_ExpandFileName";
}

Rboolean Rf_isBlankString(const char *) {
    throw "NYI: Rf_isBlankString";
}

char *R_tmpnam(const char *prefix, const char *tempdir) {
    throw "NYI: R_tmpnam";
}

void R_CheckUserInterrupt(void) {
    throw "NYI: R_CheckUserInterrupt";
}

void R_CheckStack(void) {
    throw "NYI: R_CheckStack";
}

void R_CheckStack2(size_t) {
    throw "NYI: R_CheckStack2";
}

extern "C" {
int interv_(double *xt, int *n, double *x,
            Rboolean *rightmost_closed, Rboolean *all_inside,
            int *ilo, int *mflag) {
    throw "NYI: interv";
}
}
