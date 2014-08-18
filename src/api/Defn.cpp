
#include "api.h"
#include <cwchar>
#include <stdint.h>

#define R_NO_REMAP
#include <Rinternals.h>

extern "C" {

int ENC_KNOWN(SEXP x) {
    _NYI("ENC_KNOWN");
}

int IS_CACHED(SEXP x) {
    _NYI("IS_CACHED");
}

/*--- Global Variables ---------------------------------------------------- */

SEXP  R_SrcfileSymbol;    /* "srcfile" */
SEXP  R_SrcrefSymbol;     /* "srcref" */


Rboolean R_interrupts_suspended;
int R_interrupts_pending;

/* Pointer  type and utilities for dispatch in the methods package */
typedef SEXP (*R_stdGen_ptr_t)(SEXP, SEXP, SEXP); /* typedef */
//R_stdGen_ptr_t R_get_standardGeneric_ptr(void); /* get method */
R_stdGen_ptr_t R_set_standardGeneric_ptr(R_stdGen_ptr_t, SEXP) { /* set method */
    printf("R_set_standardGeneric_ptr is not implemented, S4 dispatch probably won't work\n"); 
    return NULL;
}

SEXP R_MethodsNamespace;

SEXP R_deferred_default_method(void) {
    _NYI("R_deferred_default_method");
}

SEXP R_set_prim_method(SEXP fname, SEXP op, SEXP code_vec, SEXP fundef,
               SEXP mlist) {
    _NYI("R_set_prim_method");
}

SEXP do_set_prim_method(SEXP op, const char *code_string, SEXP fundef,
            SEXP mlist) {
    _NYI("do_set_prim_method");
}

void R_set_quick_method_check(R_stdGen_ptr_t) {
    printf("R_set_quick_method_check is not implemented, S4 dispatch probably won't work\n"); 
}

SEXP R_primitive_methods(SEXP op) {
    _NYI("R_primitive_methods");
}

SEXP R_primitive_generic(SEXP op) {
    _NYI("R_primitive_generic");
}

/* R Home Directory */
char *R_Home;           /* Root of the R tree */

unsigned int max_contour_segments = 25000;

/* used in package utils */
Rboolean known_to_be_latin1 = FALSE;
Rboolean known_to_be_utf8 = FALSE;

Rboolean mbcslocale = FALSE;

char const* OutDec = ".";

Rboolean R_isForkedChild = FALSE;

Rboolean R_Visible;

uintptr_t R_CStackLimit = -1;  /* C stack limit */
uintptr_t R_CStackStart = -1;  /* Initial stack address */
int  R_CStackDir = 1;  /* C stack direction */

char *R_TempDir;

/* Objects Used In Parsing  */
int   R_ParseError; /* Line where parse error occurred */
int R_ParseErrorCol;    /* Column of start of token where parse error occurred */
SEXP    R_ParseErrorFile;   /* Source file where parse error was seen.  Either a
                       STRSXP or (when keeping srcrefs) a SrcFile ENVSXP */
#define PARSE_ERROR_SIZE 256        /* Parse error messages saved here */
char  R_ParseErrorMsg[PARSE_ERROR_SIZE];
#define PARSE_CONTEXT_SIZE 256      /* Recent parse context kept in a circular buffer */
char  R_ParseContext[PARSE_CONTEXT_SIZE];
int   R_ParseContextLast; /* last character in context buffer */
int   R_ParseContextLine; /* Line in file of the above */

// Not in a header but used by grDevices
Rboolean useaqua;

/* ../../main/printutils.c : */
const char *EncodeChar(SEXP) {
    _NYI("EncodeChar");
}

/* main/subassign.c */
SEXP R_subassign3_dflt(SEXP, SEXP, SEXP, SEXP) {
    _NYI("R_subassign3_dflt");
}

/* main/util.c */
void UNIMPLEMENTED_TYPE(const char *s, SEXP x) {
    _NYI("UNIMPLEMENTED_TYPE");
}

Rboolean Rf_strIsASCII(const char *str) {
    _NYI("Rf_strIsASCII");
}

typedef unsigned short ucs2_t;
size_t mbcsToUcs2(const char *in, ucs2_t *out, int nout, int enc) {
    _NYI("mbcsToUcs2");
}

size_t Rf_utf8towcs(wchar_t *wc, const char *s, size_t n) {
    _NYI("Rf_utf8towcs");
}

SEXP Rf_installTrChar(SEXP) {
    _NYI("Rf_installTrChar");
}

size_t Rf_mbrtowc(wchar_t *wc, const char *s, size_t n, mbstate_t *ps) {
    _NYI("Rf_mbrtowc");
}

char *Rf_strchr(const char *s, int c) {
    _NYI("strchr");
}

int R_OutputCon; /* from connections.c */
int R_InitReadItemDepth, R_ReadItemDepth; /* from serialize.c */

FILE *RC_fopen(const SEXP fn, const char *mode, const Rboolean expand) {
    _NYI("RC_fopen");
}

struct RCNTXT;
void Rf_begincontext(RCNTXT*, int, SEXP, SEXP, SEXP, SEXP, SEXP) {
    _NYI("Rf_begincontext");
}

void Rf_endcontext(RCNTXT*) {
    _NYI("Rf_endcontext");
}

int R_ReadConsole(const char *, unsigned char *, int, int) {
    _NYI("R_ReadConsole");
}
void R_WriteConsole(const char *, int); /* equivalent to R_WriteConsoleEx(a, b, 0) */
void R_WriteConsoleEx(const char *, int, int);
void R_ResetConsole(void);

void R_ClearerrConsole(void) {
    _NYI("R_ClearerrConsole");
}
void R_Busy(int);
int R_ShowFiles(int, const char **, const char **, const char *,
            Rboolean, const char *);
int R_EditFiles(int, const char **, const char **, const char *) {
    _NYI("R_EditFiles");
}
int R_ChooseFile(int, char *, int);
char *R_HomeDir(void);
Rboolean R_FileExists(const char *) {
    _NYI("R_FileExists");
}

Rboolean R_HiddenFile(const char *);
double R_FileMtime(const char *);

/* Coercion functions */

/* Other Internally Used Functions */
void Rf_copyMostAttribNoTs(SEXP, SEXP) {
    _NYI("Rf_copyMostAttribNoTs");
}

SEXP Rf_deparse1(SEXP,Rboolean,int) {
    _NYI("Rf_deparse1");
}

SEXP Rf_deparse1line(SEXP,Rboolean) {
    _NYI("Rf_deparse1line");
}

SEXP R_data_class(SEXP , Rboolean) {
    _NYI("R_data_class");
}

/* environment cell access */
typedef struct R_varloc_st *R_varloc_t;
R_varloc_t R_findVarLocInFrame(SEXP, SEXP) {
    _NYI("R_findVarLocInFrame");
}

Rboolean R_GetVarLocMISSING(R_varloc_t) {
    _NYI("R_GetVarLocMISSING");
}

double R_strtod5(const char *str, char **endptr, char dec,
         Rboolean NA, int exact) {
    _NYI("R_strtod5");
}

SEXP Rf_mkFalse(void) {
    _NYI("Rf_mkFalse");
}

SEXP Rf_NewEnvironment(SEXP, SEXP, SEXP) {
    _NYI("Rf_NewEnvironment");
}

void Rf_PrintDefaults(void) {
    _NYI("Rf_PrintDefaults");
}

int Rf_envlength(SEXP) {
    _NYI("Rf_envlength");
}

void Rf_sortVector(SEXP, Rboolean) {
    _NYI("Rf_sortVector");
}

SEXP R_NewHashedEnv(SEXP, SEXP) {
    _NYI("R_NewHashedEnv");
}

double R_atof(const char *str) {
    _NYI("R_atof");
}

/* From localecharset.c */
const char *locale2charset(const char *) {
    _NYI("locale2charset");
}

/* From Parse.h */
void parseError(SEXP call, int linenum) {
    _NYI("parseError");
}

}


