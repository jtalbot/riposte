
#include <stddef.h>

#include "../frontend.h"

#define HAVE_POPEN
#define R_NO_REMAP

#include <Rinternals.h>
#include <libintl.h>

extern int R_PPStackSize;
extern int R_PPStackTop;
extern SEXP* R_PPStack;

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

/* Various tests with macro versions below */
Rboolean (Rf_isNull)(SEXP s) {
    return s->v.isNull() ? TRUE : FALSE;
}

Rboolean (Rf_isSymbol)(SEXP s) {
    throw "NYI: Rf_isSymbol";
}

Rboolean (Rf_isLogical)(SEXP s) {
    throw "NYI: Rf_isLogical";
}

Rboolean (Rf_isReal)(SEXP s) {
    throw "NYI: Rf_isReal";
}

Rboolean (Rf_isComplex)(SEXP s) {
    throw "NYI: Rf_isComplex";
}

Rboolean (Rf_isExpression)(SEXP s) {
    throw "NYI: Rf_isExpression";
}

Rboolean (Rf_isEnvironment)(SEXP s) {
    throw "NYI: Rf_isEnvironment";
}

Rboolean (Rf_isString)(SEXP s) {
    throw "NYI: Rf_isString";
}

Rboolean (Rf_isObject)(SEXP s) {
    throw "NYI: Rf_isObject";
}


/* General Cons Cell Attributes */
SEXP (ATTRIB)(SEXP x) {
    throw "NYI: ATTRIB";
}

int  (TYPEOF)(SEXP x) {

    int type = x->v.type();

    switch(type) {
        case Type::Nil:         return NILSXP;
        case Type::Promise:     return PROMSXP;
        case Type::Future:      throw "NYI: TYPEOF for future";
        case Type::Closure:     return CLOSXP;
        case Type::Environment: return ENVSXP;
        case Type::Externalptr: return EXTPTRSXP;
        case Type::Null:        return NILSXP;
        case Type::Raw:         return RAWSXP;
        case Type::Logical:     return LGLSXP;
        case Type::Integer:     return INTSXP;
        case Type::Double:      return REALSXP;
        case Type::Character:
            if(isSymbol(x->v))  return SYMSXP;
            else                return STRSXP;
        case Type::List:
            if(isCall(x->v))    return LANGSXP;
            else if(isExpression(x->v)) return EXPRSXP;
            else if(isPairlist(x->v)) return LISTSXP;
            else                return VECSXP;
        default:
            throw "Unknown type in TYPEOF";
    }
}

int  (NAMED)(SEXP x) {
    throw "NYI: NAMED";
}

void (SET_OBJECT)(SEXP x, int v) {
    throw "NYI: SET_OBJECT";
}

void (SET_TYPEOF)(SEXP x, int v) {
    throw "NYI: SET_TYPEOF";
}

void (SET_NAMED)(SEXP x, int v) {
    throw "NYI: SET_NAMED";
}

void SET_ATTRIB(SEXP x, SEXP v) {
    throw "NYI: SET_ATTRIB";
}

void DUPLICATE_ATTRIB(SEXP to, SEXP from) {
    throw "NYI: DUPLICATE_ATTRIB";
}

/* Vector Access Functions */
int  (LENGTH)(SEXP x) {
    throw "NYI: LENGTH";
}

R_xlen_t  (XLENGTH)(SEXP x) {
    throw "NYI: XLENGTH";
}

int  (IS_LONG_VEC)(SEXP x) {
    throw "NYI: IS_LONG_VEC";
}

int  *(LOGICAL)(SEXP x) {
    throw "NYI: LOGICAL";
}

int  *(INTEGER)(SEXP x) {
    throw "NYI: INTEGER";
}

Rbyte *(RAW)(SEXP x) {
    throw "NYI: RAW";
}

double *(REAL)(SEXP x) {
    throw "NYI: REAL";
}

Rcomplex *(COMPLEX)(SEXP x) {
    throw "NYI: COMPLEX";
}

SEXP (STRING_ELT)(SEXP x, R_xlen_t i) {
    throw "NYI: STRING_ELT";
}

SEXP (VECTOR_ELT)(SEXP x, R_xlen_t i) {
    throw "NYI: VECTOR_ELT";
}

void SET_STRING_ELT(SEXP x, R_xlen_t i, SEXP a) {
    if(!x->v.isCharacter()) {
        printf("Argument is not a Character vector in SET_STRING_ELT");
        throw;
    }
    Character& c = (Character&)x->v;
    if(i >= c.length()) {
        printf("Assigning past the end in SET_STRING_ELT");
        throw;
    }
    c[i] = (String)a;
}

SEXP SET_VECTOR_ELT(SEXP x, R_xlen_t i, SEXP v) {
    throw "NYI: SET_VECTOR_ELT";
}

/* List Access Functions */
/* These also work for ... objects */

SEXP (TAG)(SEXP e) {
    throw "NYI: TAG";
}

SEXP (CAR)(SEXP e) {
    throw "NYI: CAR";
}

SEXP (CDR)(SEXP e) {
    throw "NYI: CDR";
}

SEXP (CADR)(SEXP e) {
    throw "NYI: CADR";
}

SEXP (CDDR)(SEXP e) {
    throw "NYI: CDDR";
}

SEXP (CADDR)(SEXP e) {
    throw "NYI: CADDR";
}

SEXP (CADDDR)(SEXP e) {
    throw "NYI: CADDDR";
}

SEXP (CAD4R)(SEXP e) {
    throw "NYI: CAD4R";
}

void SET_TAG(SEXP x, SEXP y) {
    throw "NYI: SET_TAG";
}

SEXP SETCAR(SEXP x, SEXP y) {
    throw "NYI: SETCAR";
}

SEXP SETCDR(SEXP x, SEXP y) {
    throw "NYI: SETCDR";
}

SEXP SETCADR(SEXP x, SEXP y) {
    throw "NYI: SETCADR";
}

SEXP SETCADDR(SEXP x, SEXP y) {
    throw "NYI: SETCADDR";
}

SEXP SETCADDDR(SEXP x, SEXP y) {
    throw "NYI: SETCADDDR";
}

SEXP SETCAD4R(SEXP e, SEXP y) {
    throw "NYI: SETCAD4R";
}

/* Closure Access Functions */
SEXP (FORMALS)(SEXP x) {
    throw "NYI: FORMALS";
}

SEXP (CLOENV)(SEXP x) {
    throw "NYI: CLOENV";
}

void SET_FORMALS(SEXP x, SEXP v) {
    throw "NYI: SET_FORMALS";
}

void SET_BODY(SEXP x, SEXP v) {
    throw "NYI: SET_BODY";
}

void SET_CLOENV(SEXP x, SEXP v) {
    throw "NYI: SET_CLOENV";
}

/* Symbol Access Functions */
SEXP (PRINTNAME)(SEXP x) {
    throw "NYI: PRINTNAME";
}

SEXP (SYMVALUE)(SEXP x) {
    throw "NYI: SYMVALUE";
}

int  (DDVAL)(SEXP x) {
    throw "NYI: DDVAL";
}

/* Environment Access Functions */
SEXP (ENCLOS)(SEXP x) {
    throw "NYI: ENCLOS";
}

/* Promise Access Functions */
SEXP (PRCODE)(SEXP x) {
    throw "NYI: PRCODE";
}

SEXP (PRENV)(SEXP x) {
    throw "NYI: PRENV";
}

SEXP (PRVALUE)(SEXP x) {
    throw "NYI: PRVALUE";
}

void SET_PRVALUE(SEXP x, SEXP v) {
    throw "NYI: SET_PRVALUE";
}

/* Type Coercions of all kinds */
SEXP Rf_asChar(SEXP) {
    throw "NYI: Rf_asChar";
}

SEXP Rf_coerceVector(SEXP, SEXPTYPE) {
    throw "NYI: Rf_coerceVector";
}

SEXP Rf_PairToVectorList(SEXP x) {
    throw "NYI: Rf_PairToVectorList";
}

SEXP Rf_VectorToPairList(SEXP x) {
    throw "NYI: Rf_VectorTpPairList";
}

int Rf_asLogical(SEXP x) {
    throw "NYI: Rf_asLogical";
}
int Rf_asInteger(SEXP x) {
    Value const& a = x->v;
    if(a.isInteger1())
        return (int)a.i;
    else if(a.isDouble1())
        return (int)a.d;
    else 
        throw "NYI: Rf_asInteger";
}
double Rf_asReal(SEXP x) {
    throw "NYI: Rf_asReal";
}

/* Other Internally Used Functions, excluding those which are inline-able*/

char * Rf_acopy_string(const char *) {
    throw "NYI: Rf_acopy_string";
}

SEXP Rf_alloc3DArray(SEXPTYPE, int, int, int) {
    throw "NYI: Rf_alloc3DArray";
}

SEXP Rf_allocMatrix(SEXPTYPE, int, int) {
    throw "NYI: Rf_allocMatrix";
}

SEXP Rf_allocList(int) {
    throw "NYI: Rf_allocList";
}

SEXP Rf_allocS4Object(void) {
    throw "NYI: Rf_allocS4Object";
}

SEXP Rf_allocSExp(SEXPTYPE) {
    throw "NYI: Rf_allocSExp";
}

SEXP Rf_allocVector3(SEXPTYPE type, R_xlen_t len, R_allocator_t* alloc) {
    if(alloc != NULL)
        printf("Rf_allocVector3 does not support custom allocator");
    
    Value v;
    switch(type) {
        case STRSXP: v = Character(len); break;
        default: printf("Unsupported type in Rf_allocVector3"); throw; break;
    }

    return new SEXPREC(v);
}

SEXP Rf_cons(SEXP, SEXP) {
    throw "NYI: Rf_cons";
}

void Rf_copyVector(SEXP, SEXP) {
    throw "NYI: Rf_copyVector";
}

void Rf_defineVar(SEXP, SEXP, SEXP) {
    throw "NYI: Rf_defineVar";
}

SEXP Rf_duplicate(SEXP) {
    throw "NYI: Rf_duplicate";
}

SEXP Rf_duplicated(SEXP, Rboolean) {
    throw "NYI: Rf_duplicated";
}

SEXP Rf_eval(SEXP v, SEXP env) {
    if(!v->v.isPromise())
        return v;

    Promise const& p = (Promise const&)v->v;
    Code* code = p.isExpression() ? p.code() : thread.promiseCode;
    Compiler::doPromiseCompilation(thread, code);

    // Going to have to build a temporary thread to execute in...
}

SEXP Rf_findFun(SEXP, SEXP) {
    throw "NYI: Rf_findFun";
}

SEXP Rf_findVar(SEXP symbol, SEXP env) {
    if(!env->v.isEnvironment()) {
        printf("argument to findVar is not an environment");
        throw;
    }
    if(!symbol->v.isCharacter() ||
       ((Character const&)symbol->v).length() != 1) {
        printf("argument to findVar is not a one element Character");
        throw;
    }
    Environment* foundEnv;
    Value v = ((REnvironment&)env->v).environment()->getRecursive(symbol->v.s, foundEnv);
    if(v.isNil())
        return R_UnboundValue;
    else
        return new SEXPREC(v);
}

SEXP Rf_findVarInFrame(SEXP, SEXP) {
    throw "NYI: Rf_findVarInFrame";
}

SEXP Rf_findVarInFrame3(SEXP, SEXP, Rboolean) {
    throw "NYI: Rf_findVarInFrame3";
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

SEXP Rf_install(const char * s) {
    return globalState->installSEXP(CreateSymbol(globalState->internStr(s)));
}

SEXP Rf_lengthgets(SEXP, R_len_t) {
    throw "NYI: Rf_lengthgets";
}

SEXP Rf_xlengthgets(SEXP, R_xlen_t) {
    throw "Rf_xlengthgets";
}

SEXP Rf_matchE(SEXP, SEXP, int, SEXP) {
    throw "NYI: Rf_matchE";
}

SEXP Rf_mkChar(const char * str) {
    return (SEXP)globalState->internStr(str);
}

int Rf_ncols(SEXP) {
    throw "NYI: Rf_ncols";
}

int Rf_nrows(SEXP) {
    throw "NYI: Rf_nrows";
}

SEXP Rf_nthcdr(SEXP, int) {
    throw "NYI: Rf_nthcdr";
}

SEXP Rf_protect(SEXP s) {
    if (R_PPStackTop < R_PPStackSize)
        R_PPStack[R_PPStackTop++] = s;
    else R_signal_protect_error();
    return s;
}

SEXP Rf_setAttrib(SEXP in, SEXP attr, SEXP value) {
    Value a = attr->v;
    if(!in->v.isObject()) {
        printf("in argument is not an object in Rf_setAttrib");
    }
    if(!a.isCharacter() || ((Character const&)a).length() != 1) {
        printf("attr argument is not a 1-element Character in Rf_setAttrib");
    }

    // TODO: This needs to erase attributes too.
    // I should probably just be calling a shared SetAttr API function
    // which is also used by the interpreter.
    Object o = (Object&)in->v;

    Dictionary* d = o.hasAttributes()
                    ? o.attributes()->clone(1)
                    : new Dictionary(1);
    d->insert(((Character const&)a)[0]) = value->v;
    o.attributes(d);
    return new SEXPREC(o);
}

SEXP Rf_substitute(SEXP,SEXP) {
    throw "NYI: Rf_substitute";
}

const char * Rf_translateChar(SEXP) {
    throw "NYI: Rf_translateChar";
}

const char * Rf_translateCharUTF8(SEXP) {
    throw "NYI: Rf_translateCharUTF8";
}

const char * Rf_type2char(SEXPTYPE) {
    throw "NYI: Rf_type2char";
}

void Rf_unprotect(int l) {
    if (R_PPStackTop >=  l)
        R_PPStackTop -= l;
    else R_signal_unprotect_error();
}

void Rf_unprotect_ptr(SEXP) {
    throw "Rf_unprotect_ptr";
}

void R_signal_protect_error(void) {
    throw "NYI: R_signal_protect_error";
}

void R_signal_unprotect_error(void) {
    throw "NYI: R_signal_unprotect_error";
}

void R_signal_reprotect_error(PROTECT_INDEX i) {
    throw "NYI: R_signal_reprotect_error";
}

void R_ProtectWithIndex(SEXP, PROTECT_INDEX *) {
    throw "NYI: R_ProtectWithIndex";
}

void R_Reprotect(SEXP, PROTECT_INDEX) {
    throw "NYI: R_Reprotect";
}

SEXP R_tryEvalSilent(SEXP, SEXP, int *) {
    throw "NYI: R_tryEvalSilent";
}

const char *R_curErrorBuf() {
    throw "NYI: R_curErrorBuf";
}

cetype_t Rf_getCharCE(SEXP) {
    throw "NYI: Rf_getCharCE";
}

SEXP Rf_mkCharCE(const char *, cetype_t) {
    throw "NYI: Rf_mkCharCE";
}

SEXP Rf_mkCharLenCE(const char *, int, cetype_t) {
    throw "NYI: Rf_mkCharLenCE";
}


/* External pointer interface */
SEXP R_MakeExternalPtr(void *p, SEXP tag, SEXP prot) {
    throw "NYI: R_MakeExternalPtr";
}

void *R_ExternalPtrAddr(SEXP s) {
    throw "NYI: R_ExternalPtrAddr";
}

SEXP R_ExternalPtrTag(SEXP s) {
    throw "NYI: R_ExternalPtrTag";
}

/* Environment and Binding Features */
SEXP R_FindNamespace(SEXP info) {
    throw "NYI: R_FindNamespace";
}

/* needed for R_load/savehistory handling in front ends */
void Rf_warningcall(SEXP, const char *, ...) {
    throw "NYI: Rf_warningcall";
}

/* slot management (in attrib.c) */
SEXP R_do_slot(SEXP obj, SEXP name) {
    throw "NYI: R_do_slot";
}

SEXP R_do_slot_assign(SEXP obj, SEXP name, SEXP value) {
    throw "NYI: R_do_slot_assign";
}

int R_has_slot(SEXP obj, SEXP name) {
    throw "NYI: R_has_slot";
}

/* class definition, new objects (objects.c) */
SEXP R_do_MAKE_CLASS(const char *what) {
    throw "NYI: R_do_MAKE_CLASS";
}

SEXP R_do_new_object(SEXP class_def) {
    throw "NYI: R_do_new_object";
}

/* preserve objects across GCs */
void R_PreserveObject(SEXP object) {
    globalState->installSEXP(object);
}

void R_ReleaseObject(SEXP object) {
    globalState->uninstallSEXP(object);
}

/* Replacements for popen and system */
FILE *R_popen(const char *, const char *) {
    throw "NYI: R_popen";
}

int R_system(const char *) {
    throw "NYI: R_system";
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

Rboolean Rf_isFrame(SEXP) {
    throw "NYI: Rf_isFrame";
}

Rboolean Rf_isFunction(SEXP) {
    throw "NYI: Rf_isFunction";
}

Rboolean Rf_isInteger(SEXP) {
    throw "NYI: Rf_isInteger";
}

Rboolean Rf_isMatrix(SEXP) {
    throw "NYI: Rf_isMatrix";
}

Rboolean Rf_isNewList(SEXP) {
    throw "NYI: Rf_isNewList";
}

Rboolean Rf_isNumeric(SEXP) {
    throw "NYI: Rf_isNumeric";
}

Rboolean Rf_isVectorAtomic(SEXP) {
    throw "NYI: Rf_isVectorAtomic";
}

Rboolean Rf_isVectorList(SEXP) {
    throw "NYI: Rf_isVectorList";
}

SEXP     Rf_lang1(SEXP) {
    throw "NYI: Rf_lang1";
}

SEXP     Rf_lang2(SEXP, SEXP) {
    throw "NYI: Rf_lang2";
}

SEXP     Rf_lang3(SEXP, SEXP, SEXP) {
    throw "NYI: Rf_lang3";
}

SEXP     Rf_lang4(SEXP, SEXP, SEXP, SEXP) {
    throw "NYI: Rf_lang4";
}

SEXP     Rf_lang5(SEXP, SEXP, SEXP, SEXP, SEXP) {
    throw "NYI: Rf_lang5";
}

R_len_t  Rf_length(SEXP) {
    throw "NYI: Rf_length";
}

SEXP     Rf_list4(SEXP, SEXP, SEXP, SEXP) {
    throw "NYI: Rf_list4";
}

SEXP     Rf_mkNamed(SEXPTYPE, const char **) {
    throw "NYI: Rf_mkNamed";
}

SEXP     Rf_mkString(const char *) {
    throw "NYI: Rf_mkString";
}

int  Rf_nlevels(SEXP) {
    throw "NYI: Rf_nlevels";
}

SEXP     Rf_ScalarInteger(int) {
    throw "NYI: Rf_ScalarInteger";
}

SEXP     Rf_ScalarLogical(int f) {
    return new SEXPREC(Logical::c(f ? Logical::TrueElement : Logical::FalseElement));
}

SEXP     Rf_ScalarReal(double) {
    throw "NYI: Rf_ScalarReal";
}

extern "C" {
void R_SignalCStackOverflow(intptr_t) {
    throw "NYI: R_SignalCStackOverflow";
}
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

// From main/internet.c, used by the tools package
int extR_HTTPDCreate(const char *ip, int port) {
    throw "NYI: extR_HTTPDCreate";
}

void extR_HTTPDStop(void) {
    throw "NYI: extR_HTTPDStop";
}

// From main/dounzip.c, used by the utils package
SEXP Runzip(SEXP args) {
    throw "NYI: Runzip";
}

// From main/eval.c, used by the utils package
SEXP do_Rprof(SEXP args) {
    throw "NYI: do_Rprof";
}

// From main/eval.c, used by the stats package
SEXP R_execMethod(SEXP op, SEXP rho) {
    throw "NYI: R_execMethod";
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

// From main/names.c, used by the methods package
const char *getPRIMNAME(SEXP object) {
    throw "NYI: getPRIMNAME";
}

// From main/xxxpr.f, used by the stats package
void intpr_() {
    throw "NYI: intpr";
}

void dblepr_() {
    throw "NYI: dblepr";
}

void rexit_() {
    throw "NYI: rexit";
}

void rwarn_() {
    throw "NYI: rwarn";
}

// From main/util.c, used by stats
void rchkusr_(void) {
    throw "NYI: rchkusr";
}

}

