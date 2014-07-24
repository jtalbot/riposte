
#include <Rinternals.h>

Rboolean R_interrupts_suspended;
int R_interrupts_pending;

char *R_Home;

Rboolean known_to_be_latin1;
Rboolean known_to_be_utf8;

Rboolean mbcslocale;

Rboolean R_isForkedChild;

SEXP R_TrueValue;
SEXP R_FalseValue;
SEXP R_LogicalNAValue;


int   R_PPStackSize;
int   R_PPStackTop; 
SEXP* R_PPStack;


Rboolean R_Visible;


char *R_TempDir;

// Not in a header but used by grDevices
Rboolean useaqua;

extern "C" {

/* main/util.c */
void UNIMPLEMENTED_TYPE(const char *s, SEXP x) {
    throw "NYI: UNIMPLEMENTED_TYPE";
}

typedef unsigned short ucs2_t;
size_t mbcsToUcs2(const char *in, ucs2_t *out, int nout, int enc) {
    throw "NYI: mbcsToUcs2";
}

Rboolean Rf_strIsASCII(const char *str) {
    throw "NYI: Rf_strIsASCII";
}

size_t Rf_utf8towcs(wchar_t *wc, const char *s, size_t n) {
    throw "NYI: Rf_utf8towcs";
}

struct RCNTXT;
void Rf_begincontext(RCNTXT*, int, SEXP, SEXP, SEXP, SEXP, SEXP) {
    throw "NYI: Rf_begincontext";
}

void Rf_endcontext(RCNTXT*) {
    throw "NYI: Rf_endcontext";
}

int R_ReadConsole(const char *, unsigned char *, int, int) {
    throw "NYI: R_ReadConsole";
}
void R_WriteConsole(const char *, int); /* equivalent to R_WriteConsoleEx(a, b, 0) */
void R_WriteConsoleEx(const char *, int, int);
void R_ResetConsole(void);
void R_ClearerrConsole(void) {
    throw "NYI: R_ClearerrConsole";
}
void R_Busy(int);
int R_ShowFiles(int, const char **, const char **, const char *,
            Rboolean, const char *);
int R_EditFiles(int, const char **, const char **, const char *) {
    throw "NYI: R_EditFiles";
}
int R_ChooseFile(int, char *, int);
char *R_HomeDir(void);
Rboolean R_FileExists(const char *);
Rboolean R_HiddenFile(const char *);
double R_FileMtime(const char *);

double R_strtod5(const char *str, char **endptr, char dec,
         Rboolean NA, int exact) {
    throw "NYI: R_strtod5";
}

void Rf_PrintDefaults(void) {
    throw "NYI: Rf_PrintDefaults";
}

int Rf_envlength(SEXP) {
    throw "NYI: Rf_envlength";
}

void Rf_sortVector(SEXP, Rboolean) {
    throw "NYI: Rf_sortVector";
}

/* From localecharset.c */
const char *locale2charset(const char *) {
    throw "NYI: locale2charset";
}

}


