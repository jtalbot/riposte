
#include "api.h"
#include <R_ext/Print.h>


void Rprintf(const char *, ...) {
    _NYI("Rprintf");
}

void REprintf(const char *, ...) {
    _NYI("REprintf");
}

// The following is from Print.h

typedef void* SEXP;

typedef struct {
    int width;
    int na_width;
    int na_width_noquote;
    int digits;
    int scipen;
    int gap;
    int quote;
    int right;
    int max;
    SEXP na_string;
    SEXP na_string_noquote;
    int useSource;
    int cutoff; // for deparsed language objects
} R_print_par_t;

R_print_par_t R_print;

extern "C" {
const char *Rf_EncodeElement0(SEXP, int, int, const char *) {
    _NYI("Rf_EncodeElement0");
}

const char Rf_EncodeElement(SEXP, int, int, char) {
    _NYI("Rf_EncodeElement");
}

}

extern "C" {
    // Needed by libRblas
    void xerbla_(const char *srname, int *info) {
        _NYI("xerbla");
    }
}

