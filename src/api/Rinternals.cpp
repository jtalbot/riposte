
#include <stddef.h>

#define HAVE_POPEN

#include <Rinternals.h>
#include <libintl.h>

extern "C" {
R_len_t R_BadLongVector(SEXP, const char *, int) {
    throw "NYI: R_BadLongVector";
}
}

const char *(R_CHAR)(SEXP x) {
    throw "NYI: R_CHAR";
}

/* Accessor functions.  Many are declared using () to avoid the macro
   definitions in the USE_RINTERNALS section.
   The function STRING_ELT is used as an argument to arrayAssign even
   if the macro version is in use.
*/

/* General Cons Cell Attributes */
int  (TYPEOF)(SEXP x) {
    throw "NYI: TYPEOF";
}

void (SET_TYPEOF)(SEXP x, int v) {
    throw "NYI: SET_TYPEOF";
}

/* Vector Access Functions */
int  (LENGTH)(SEXP x) {
    throw "NYI: LENGTH";
}

R_xlen_t  (XLENGTH)(SEXP x) {
    throw "NYI: XLENGTH";
}

int  *(INTEGER)(SEXP x) {
    throw "NYI: INTEGER";
}

double *(REAL)(SEXP x) {
    throw "NYI: REAL";
}

SEXP (STRING_ELT)(SEXP x, R_xlen_t i) {
    throw "NYI: STRING_ELT";
}

SEXP (VECTOR_ELT)(SEXP x, R_xlen_t i) {
    throw "NYI: VECTOR_ELT";
}

void SET_STRING_ELT(SEXP x, R_xlen_t i, SEXP v) {
    throw "NYI: SET_STRING_ELT";
}

SEXP SET_VECTOR_ELT(SEXP x, R_xlen_t i, SEXP v) {
    throw "NYI: SET_VECTOR_ELT";
}

/* List Access Functions */
/* These also work for ... objects */
SEXP (CDR)(SEXP e) {
    throw "NYI: CDR";
}

/* Various tests with macro versions below */
Rboolean (Rf_isNull)(SEXP s) {
    throw "NYI: Rf_isNull";
}

Rboolean (Rf_isString)(SEXP s) {
    throw "NYI: Rf_isString";
}

Rboolean (Rf_isObject)(SEXP s) {
    throw "NYI: Rf_isObject";
}

/* Type Coercions of all kinds */
SEXP Rf_asChar(SEXP) {
    throw "NYI: Rf_asChar";
}

SEXP Rf_coerceVector(SEXP, SEXPTYPE) {
    throw "NYI: Rf_coerceVector";
}

int Rf_asLogical(SEXP x) {
    throw "NYI: Rf_asLogical";
}
int Rf_asInteger(SEXP x) {
    throw "NYI: Rf_asInteger";
}
double Rf_asReal(SEXP x) {
    throw "NYI: Rf_asReal";
}

/* Other Internally Used Functions, excluding those which are inline-able*/

char * Rf_acopy_string(const char *) {
    throw "NYI: Rf_acopy_string";
}

SEXP Rf_allocMatrix(SEXPTYPE, int, int) {
    throw "NYI: Rf_allocMatrix";
}

SEXP Rf_allocVector3(SEXPTYPE, R_xlen_t, R_allocator_t*) {
    throw "NYI: Rf_allocVector3";
}

void Rf_copyVector(SEXP, SEXP) {
    throw "NYI: Rf_copyVector";
}

SEXP Rf_duplicate(SEXP) {
    throw "NYI: Rf_duplicate";
}

SEXP Rf_duplicated(SEXP, Rboolean) {
    throw "NYI: Rf_duplicated";
}

SEXP Rf_eval(SEXP, SEXP) {
    throw "NYI: Rf_eval";
}

SEXP Rf_findVar(SEXP, SEXP) {
    throw "NYI: Rf_findVar";
}

SEXP Rf_getAttrib(SEXP, SEXP) {
    throw "NYI: Rf_getAttrib";
}

SEXP Rf_GetOption1(SEXP) {
    throw "NYI: Rf_GetOptions1";
}

void Rf_gsetVar(SEXP, SEXP, SEXP) {
    throw "NYI: Rf_gsetVar";
}

SEXP Rf_install(const char *) {
    throw "NYI: Rf_install";
}

SEXP Rf_matchE(SEXP, SEXP, int, SEXP) {
    throw "NYI: Rf_matchE";
}

SEXP Rf_mkChar(const char *) {
    throw "NYI: Rf_mkChar";
}

int Rf_nrows(SEXP) {
    throw "NYI: Rf_nrows";
}

SEXP Rf_protect(SEXP) {
    throw "NYI: Rf_protect";
}

SEXP Rf_setAttrib(SEXP, SEXP, SEXP) {
    throw "NYI: Rf_setAttrib";
}

const char * Rf_translateChar(SEXP) {
    throw "NYI: Rf_translateChar";
}

void Rf_unprotect(int) {
    throw "NYI: Rf_unprotect";
}

void R_signal_protect_error(void) {
    throw "NYI: R_signal_protect_error";
}

void R_signal_reprotect_error(PROTECT_INDEX i) {
    throw "NYI: R_signal_reprotect_error";
}

/*
   These are the inlinable functions that are provided in Rinlinedfuns.h
   It is *essential* that these do not appear in any other header file,
   with or without the Rf_ prefix.
*/

SEXP     Rf_allocVector(SEXPTYPE, R_xlen_t) {
    throw "NYI: Rf_allocVector";
}

Rboolean Rf_inherits(SEXP, const char *) {
    throw "NYI: Rf_inherits";
}

Rboolean Rf_isArray(SEXP) {
    throw "NYI: Rf_isArray";
}

Rboolean Rf_isVectorAtomic(SEXP) {
    throw "NYI: Rf_isVectorAtomic";
}

Rboolean Rf_isVectorList(SEXP) {
    throw "NYI: Rf_isVectorList";
}

R_len_t  Rf_length(SEXP) {
    throw "NYI: Rf_length";
}

SEXP     Rf_mkNamed(SEXPTYPE, const char **) {
    throw "NYI: Rf_mkNamed";
}

SEXP     Rf_mkString(const char *) {
    throw "NYI: Rf_mkString";
}

SEXP     Rf_ScalarInteger(int) {
    throw "NYI: Rf_ScalarInteger";
}

SEXP     Rf_ScalarLogical(int) {
    throw "NYI: Rf_ScalarLogical";
}

/* Environment and Binding Features */
SEXP R_FindNamespace(SEXP info) {
    throw "NYI: R_FindNamespace";
}

/* preserve objects across GCs */
void R_PreserveObject(SEXP) {
    throw "NYI: R_PreserveObject";
}

void R_ReleaseObject(SEXP) {
    throw "NYI: R_ReleaseObject";
}

/* Replacements for popen and system */
FILE *R_popen(const char *, const char *) {
    throw "NYI: R_popen";
}

int R_system(const char *) {
    throw "NYI: R_system";
}

extern "C" {

typedef struct {
 char *data;
 size_t bufsize;
 size_t defaultSize;
} R_StringBuffer;

// From RBufferUtils.h, which isn't in the external
// API, but which is used by the utils package
void *R_AllocStringBuffer(size_t blen, R_StringBuffer *buf) {
    throw "NYI: R_AllocStringBuffer";
}

void R_FreeStringBuffer(R_StringBuffer *buf) {
    throw "NYI: R_FreeStringBuffer";
}

// From main/unique.c, used by the utils package
SEXP Rf_csduplicated(SEXP x) {
    throw "NYI: Rf_csduplicated";
}

// For some reason the utils package also wants this without the Rf_.
SEXP csduplicated(SEXP x) {
    return Rf_csduplicated(x); 
}


// From main/internet.c, used by the utils package
SEXP Rsockclose(SEXP ssock) {
    throw "NYI: Rsockclose";
}

SEXP Rsockconnect(SEXP sport, SEXP shost) {
    throw "NYI: Rsockconnect";
}

SEXP Rsocklisten(SEXP ssock) {
    throw "NYI: Rsocklisten";
}

SEXP Rsockopen(SEXP sport) {
    throw "NYI: Rsockopen";
}

SEXP Rsockread(SEXP ssock, SEXP smaxlen) {
    throw "NYI: Rsockread";
}

SEXP Rsockwrite(SEXP ssock, SEXP sstring) {
    throw "NYI: Rsockwrite";
}

// From main/dounzip.c, used by the utils package
SEXP Runzip(SEXP args) {
    throw "NYI: Runzip";
}

// From main/eval.c, used by the utils package
SEXP do_Rprof(SEXP args) {
    throw "NYI: do_Rprof";
}

// From main/memory.c, used by the utils package
SEXP do_Rprofmem(SEXP args) {
    throw "NYI: do_Rprofmem";
}

// From main/edit.c, used by the utils package
SEXP do_edit(SEXP call, SEXP op, SEXP args, SEXP rho) {
    throw "NYI: do_edit";
}

// From main/rlocale.c, used by the grDevices package
int Ri18n_wcwidth(wchar_t c) {
    throw "NYI: Ri18n_wcwidth";
}

}

