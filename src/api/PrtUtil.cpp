
#include <R_ext/PrtUtil.h>

void Rf_formatLogical(int *, R_xlen_t, int *) {
    throw "NYI: Rf_formatLogical";
}

void Rf_formatInteger(int *, R_xlen_t, int *) {
    throw "NYI: Rf_formatInteger";
}

void Rf_formatReal(double *, R_xlen_t, int *, int *, int *, int) {
    throw "NYI: Rf_formatReal";
}

void Rf_formatComplex(Rcomplex *, R_xlen_t, int *, int *, int *, int *, int *, int *, int) {
    throw "NYI: Rf_formatComplex";
}

/* Formating of values */
const char *Rf_EncodeLogical(int, int) {
    throw "NYI: Rf_EncodeLogical";
}

const char *Rf_EncodeInteger(int, int) {
    throw "NYI: Rf_EncodeInteger";
}

const char *Rf_EncodeReal(double, int, int, int, char) {
    throw "NYI: Rf_EncodeReal";
}

const char *Rf_EncodeComplex(Rcomplex, int, int, int, int, int, int, char) {
    throw "NYI: Rf_EncodeComplex";
}


