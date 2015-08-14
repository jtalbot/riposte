
#include "api.h"
#include <R_ext/PrtUtil.h>

void formatLogical(int *, R_xlen_t, int *) {
    _NYI("formatLogical");
}

void formatInteger(int *, R_xlen_t, int *) {
    _NYI("formatInteger");
}

void formatReal(double *, R_xlen_t, int *, int *, int *, int) {
    _NYI("formatReal");
}

void formatComplex(Rcomplex *, R_xlen_t, int *, int *, int *, int *, int *, int *, int) {
    _NYI("formatComplex");
}

/* Formating of values */
const char *EncodeLogical(int, int) {
    _NYI("EncodeLogical");
}

const char *EncodeInteger(int, int) {
    _NYI("EncodeInteger");
}

const char *EncodeReal0(double, int, int, int, const char*) {
    _NYI("EncodeReal0");
}

const char *EncodeReal(double, int, int, int, char) {
    _NYI("EncodeReal");
}

const char *EncodeComplex(Rcomplex, int, int, int, int, int, int, const char*) {
    _NYI("EncodeComplex");
}

extern "C"
{

const char *Rf_EncodeChar(SEXP x)
{
    _NYI("EncodeChar");
}

}

