
#include "api.h"
#include <stddef.h>

#include "../frontend.h"
#include "../compiler.h"

#define HAVE_POPEN
#define R_NO_REMAP

#include <Rinternals.h>
#include <libintl.h>

extern int R_PPStackSize;
extern int R_PPStackTop;
extern SEXP* R_PPStack;

extern "C" {
R_len_t R_BadLongVector(SEXP, const char *, int) {
    _NYI("R_BadLongVector");
}
}

const char *(R_CHAR)(SEXP x) {
    if(x->getType() != SEXPREC::STRING) {
        printf("R_CHAR is not a string\n");
        return NULL;
    }

    return x->getString()->s;
}

/* Accessor functions.  Many are declared using () to avoid the macro
   definitions in the USE_RINTERNALS section.
   The function STRING_ELT is used as an argument to arrayAssign even
   if the macro version is in use.
*/

/* Various tests with macro versions below */
Rboolean (Rf_isNull)(SEXP s) {
    Value v = s->getValue();
    return v.isNull() ? TRUE : FALSE;
}

Rboolean (Rf_isSymbol)(SEXP s) {
    Value v = s->getValue();
    return isSymbol(v) ? TRUE : FALSE;
}

Rboolean (Rf_isLogical)(SEXP s) {
    Value v = s->getValue();
    return v.isLogical() ? TRUE : FALSE;
}

Rboolean (Rf_isReal)(SEXP s) {
    Value v = s->getValue();
    return v.isDouble() ? TRUE : FALSE;
}

Rboolean (Rf_isComplex)(SEXP s) {
    _NYI("Rf_isComplex");
}

Rboolean (Rf_isExpression)(SEXP s) {
    Value v = s->getValue();
    return isExpression(v) ? TRUE : FALSE;
}

Rboolean (Rf_isEnvironment)(SEXP s) {
    Value v = s->getValue();
    return v.isEnvironment() ? TRUE : FALSE;
}

Rboolean (Rf_isString)(SEXP s) {
    Value v = s->getValue();
    return v.isCharacter() ? TRUE : FALSE;
}

Rboolean (Rf_isObject)(SEXP s) {
    _NYI("Rf_isObject");
}


/* General Cons Cell Attributes */
SEXP (ATTRIB)(SEXP x) {
    _NYI("ATTRIB");
}

int  (TYPEOF)(SEXP x) {

    if(x->getType() == SEXPREC::INT32)
        return INTSXP;
    else if(x->getType() == SEXPREC::STRING)
        return CHARSXP;

    Value v = x->getValue();
    int type = v.type();

    switch(type) {
        case Type::Nil:         throw "Nil type cannot be extracted in TYPEOF";
        case Type::Promise:     return PROMSXP;
        case Type::Future:      _NYI("TYPEOF for future");
        case Type::Closure:     return CLOSXP;
        case Type::Environment: return ENVSXP;
        case Type::Externalptr: return EXTPTRSXP;
        case Type::Null:        return NILSXP;
        case Type::Raw:         return RAWSXP;
        case Type::Logical:     return LGLSXP;
        case Type::Integer:     return INTSXP;
        case Type::Double:      return REALSXP;
        case Type::Character:
            if(isSymbol(v))     return SYMSXP;
            else                return STRSXP;
        case Type::List:
            if(isCall(v))       return LANGSXP;
            else if(isExpression(v)) return EXPRSXP;
            else if(isPairlist(v)) return LISTSXP;
            else                return VECSXP;
        default:
            throw "Unknown type in TYPEOF";
    }
}

int  (NAMED)(SEXP x) {
    _NYI("NAMED");
}

void (SET_OBJECT)(SEXP x, int v) {
    _NYI("SET_OBJECT");
}

void (SET_TYPEOF)(SEXP x, int v) {
    _NYI("SET_TYPEOF");
}

void (SET_NAMED)(SEXP x, int v) {
    _NYI("SET_NAMED");
}

void SET_ATTRIB(SEXP x, SEXP v) {
    _NYI("SET_ATTRIB");
}

void DUPLICATE_ATTRIB(SEXP to, SEXP from) {
    _NYI("DUPLICATE_ATTRIB");
}

/* S4 object testing */
int (IS_S4_OBJECT)(SEXP x) {
    _NYI("IS_S4_OBJECT");
}

void (SET_S4_OBJECT)(SEXP x) {
    _NYI("SET_S4_OBJECT");
}

void (UNSET_S4_OBJECT)(SEXP x) {
    _NYI("UNSET_S4_OBJECT");
}

/* Vector Access Functions */
int  (LENGTH)(SEXP x) {
    Value v = x->getValue();
    if(!v.isVector()) {
        printf("LENGTH called on non-vector\n");
        return 0;
    }

    // TODO: should check to make sure the length doesn't overflow.
    return (int)((Vector const&)v).length();    
}

R_xlen_t  (XLENGTH)(SEXP x) {
    Value v = x->getValue();
    if(!v.isVector()) {
        printf("LENGTH called on non-vector\n");
        return 0;
    }

    return (R_xlen_t)((Vector const&)v).length();    
}

void (SETLENGTH)(SEXP x, int v) {
    _NYI("SETLENGTH");
}

int  (IS_LONG_VEC)(SEXP x) {
    _NYI("IS_LONG_VEC");
}

int  *(LOGICAL)(SEXP x) {
    _NYI("LOGICAL");
}

int  *(INTEGER)(SEXP x) {
    if(x->getType() == SEXPREC::INT32)
        return x->getInt32()->data;
    else {
        Value v = x->getValue();
        if(!v.isInteger()) {
            printf("Called INTEGER on something that is not an integer\n");
            return NULL;
        }

        Integer const& i = (Integer const&)v;

        SEXPREC::Int32* i32 = new (sizeof(int32_t) * i.length()) SEXPREC::Int32();
        i32->length = i.length();
        // TODO: warn when we truncate down to 32-bits
        for(int64_t j = 0; j < i.length(); ++j)
            i32->data[j] = (int32_t)i[j];

        // TODO: this is going to lose all attributes...
        x = new SEXPREC(i32);
        return x->getInt32()->data;
    }
    _NYI("INTEGER");
}

Rbyte *(RAW)(SEXP x) {
    _NYI("RAW");
}

double *(REAL)(SEXP x) {
    _NYI("REAL");
}

Rcomplex *(COMPLEX)(SEXP x) {
    _NYI("COMPLEX");
}

SEXP (STRING_ELT)(SEXP x, R_xlen_t i) {
    Value v = x->getValue();
    if(!v.isCharacter()) {
        printf("Argument to STRING_ELT is not a list");
        throw;
    }
    Character const& l = (Character const&)v;
    if(i >= l.length()) {
        printf("Accessing past the end of the vector in STRING_ELT");
        throw;
    }
    String s = l[i];
    return new SEXPREC(s);
}

SEXP (VECTOR_ELT)(SEXP x, R_xlen_t i) {
    Value v = x->getValue();
    if(!v.isList()) {
        printf("Argument to VECTOR_ELT is not a list");
        throw;
    }
    List const& l = (List const&)v;
    if(i >= l.length()) {
        printf("Accessing past the end of the list in VECTOR_ELT");
        throw;
    }
    Value r = l[i];
    return new SEXPREC(r);
}

void SET_STRING_ELT(SEXP x, R_xlen_t i, SEXP a) {
    Value v = x->getValue();
    if(!v.isCharacter()) {
        printf("Argument is not a Character vector in SET_STRING_ELT");
        throw;
    }
    Character& c = (Character&)v;
    if(i >= c.length()) {
        printf("Assigning past the end in SET_STRING_ELT");
        throw;
    }
    c[i] = (String)a;
}

SEXP SET_VECTOR_ELT(SEXP x, R_xlen_t i, SEXP v) {
    _NYI("SET_VECTOR_ELT");
}

/* List Access Functions */
/* These also work for ... objects */

SEXP (TAG)(SEXP e) {
    _NYI("TAG");
}

SEXP (CAR)(SEXP e) {
    _NYI("CAR");
}

SEXP (CDR)(SEXP e) {
    _NYI("CDR");
}

SEXP (CADR)(SEXP e) {
    _NYI("CADR");
}

SEXP (CDDR)(SEXP e) {
    _NYI("CDDR");
}

SEXP (CADDR)(SEXP e) {
    _NYI("CADDR");
}

SEXP (CADDDR)(SEXP e) {
    _NYI("CADDDR");
}

SEXP (CAD4R)(SEXP e) {
    _NYI("CAD4R");
}

void SET_TAG(SEXP x, SEXP y) {
    _NYI("SET_TAG");
}

SEXP SETCAR(SEXP x, SEXP y) {
    _NYI("SETCAR");
}

SEXP SETCDR(SEXP x, SEXP y) {
    _NYI("SETCDR");
}

SEXP SETCADR(SEXP x, SEXP y) {
    _NYI("SETCADR");
}

SEXP SETCADDR(SEXP x, SEXP y) {
    _NYI("SETCADDR");
}

SEXP SETCADDDR(SEXP x, SEXP y) {
    _NYI("SETCADDDR");
}

SEXP SETCAD4R(SEXP e, SEXP y) {
    _NYI("SETCAD4R");
}

/* Closure Access Functions */
SEXP (FORMALS)(SEXP x) {
    _NYI("FORMALS");
}

SEXP (CLOENV)(SEXP x) {
    _NYI("CLOENV");
}

void SET_FORMALS(SEXP x, SEXP v) {
    _NYI("SET_FORMALS");
}

void SET_BODY(SEXP x, SEXP v) {
    _NYI("SET_BODY");
}

void SET_CLOENV(SEXP x, SEXP v) {
    _NYI("SET_CLOENV");
}

/* Symbol Access Functions */
SEXP (PRINTNAME)(SEXP x) {
    _NYI("PRINTNAME");
}

SEXP (SYMVALUE)(SEXP x) {
    _NYI("SYMVALUE");
}

int  (DDVAL)(SEXP x) {
    _NYI("DDVAL");
}

/* Environment Access Functions */
SEXP (ENCLOS)(SEXP x) {
    _NYI("ENCLOS");
}

/* Promise Access Functions */
SEXP (PRCODE)(SEXP x) {
    _NYI("PRCODE");
}

SEXP (PRENV)(SEXP x) {
    _NYI("PRENV");
}

SEXP (PRVALUE)(SEXP x) {
    _NYI("PRVALUE");
}

void SET_PRVALUE(SEXP x, SEXP v) {
    _NYI("SET_PRVALUE");
}

/* Type Coercions of all kinds */
SEXP Rf_asChar(SEXP) {
    _NYI("Rf_asChar");
}

SEXP Rf_coerceVector(SEXP, SEXPTYPE) {
    _NYI("Rf_coerceVector");
}

SEXP Rf_PairToVectorList(SEXP x) {
    _NYI("Rf_PairToVectorList");
}

SEXP Rf_VectorToPairList(SEXP x) {
    _NYI("Rf_VectorTpPairList");
}

int Rf_asLogical(SEXP x) {
    Value a = x->getValue();
    if(a.isLogical1()) {
        if(Logical::isTrue(a.c)) return 1;
        else if(Logical::isFalse(a.c)) return 0;
        else return R_NaInt;
    }
    printf("_NYI type in Rf_asLogical");
    _NYI("Rf_asLogical");
}
int Rf_asInteger(SEXP x) {
    Value a = x->getValue();
    if(a.isInteger1())
        return (int)a.i;
    else if(a.isDouble1())
        return (int)a.d;
    else 
        _NYI("Rf_asInteger");
}
double Rf_asReal(SEXP x) {
    _NYI("Rf_asReal");
}

/* Other Internally Used Functions, excluding those which are inline-able*/

char * Rf_acopy_string(const char *) {
    _NYI("Rf_acopy_string");
}

SEXP Rf_alloc3DArray(SEXPTYPE, int, int, int) {
    _NYI("Rf_alloc3DArray");
}

SEXP Rf_allocMatrix(SEXPTYPE, int, int) {
    _NYI("Rf_allocMatrix");
}

SEXP Rf_allocList(int) {
    _NYI("Rf_allocList");
}

SEXP Rf_allocS4Object(void) {
    _NYI("Rf_allocS4Object");
}

SEXP Rf_allocSExp(SEXPTYPE) {
    _NYI("Rf_allocSExp");
}

SEXP Rf_allocVector3(SEXPTYPE type, R_xlen_t len, R_allocator_t* alloc) {
    if(alloc != NULL)
        printf("Rf_allocVector3 does not support custom allocator");
    
    Value v;
    switch(type) {
        case LGLSXP: v = Logical(len); break;
        case INTSXP: v = Integer(len); break;
        case REALSXP: v = Double(len); break;
        case STRSXP: v = Character(len); break;
        default: printf("Unsupported type in Rf_allocVector3: %d\n", type); throw; break;
    }

    return new SEXPREC(v);
}

SEXP Rf_classgets(SEXP, SEXP) {
    _NYI("Rf_classgets");
}

SEXP Rf_cons(SEXP, SEXP) {
    _NYI("Rf_cons");
}

void Rf_copyMatrix(SEXP, SEXP, Rboolean) {
    _NYI("Rf_copyMatrix");
}

void Rf_copyVector(SEXP, SEXP) {
    _NYI("Rf_copyVector");
}

void Rf_defineVar(SEXP symbol, SEXP val, SEXP env) {
    Value s = symbol->getValue();
    if(!isSymbol(s)) {
        printf("Rf_defineVar called without a symbol\n");
        throw;
    }

    Value e = env->getValue();
    if(!e.isEnvironment()) {
        printf("Rf_defineVar called without an environment\n");
        throw;
    }

    Value v = val->getValue();

    ((REnvironment&)e).environment()->insert(SymbolStr(s)) = v;
}

SEXP Rf_dimnamesgets(SEXP, SEXP) {
    _NYI("Rf_dimnamesgets");
}

SEXP Rf_duplicate(SEXP) {
    _NYI("Rf_duplicate");
}

SEXP Rf_duplicated(SEXP, Rboolean) {
    _NYI("Rf_duplicated");
}

SEXP Rf_eval(SEXP x, SEXP env) {
    Value v = x->getValue();
    Value e = env->getValue();

    if(!v.isPromise())
        printf("v in Rf_eval is not a promise");

    if(!e.isEnvironment())
        printf("env in Rf_eval is not an environment");

    //Environment* evalenv = ((REnvironment&)env->v).environment();

    Promise const& p = (Promise const&)v;

    Thread* thread = globalState->getThread();
    Value r = thread->eval(p, 0);
    globalState->deleteThread(thread);
    
    return new SEXPREC(r);
}

SEXP Rf_findFun(SEXP, SEXP) {
    _NYI("Rf_findFun");
}

SEXP Rf_findVar(SEXP symbol, SEXP env) {
    Value v = symbol->getValue();
    Value e = env->getValue();

    if(!e.isEnvironment()) {
        printf("argument to findVar is not an environment");
        throw;
    }
    if(!v.isCharacter() ||
       ((Character const&)v).length() != 1) {
        printf("argument to findVar is not a one element Character");
        throw;
    }
    Environment* foundEnv;
    Value r = ((REnvironment&)e).environment()->getRecursive(v.s, foundEnv);
    if(r.isNil())
        return R_UnboundValue;
    else
        return new SEXPREC(r);
}

SEXP Rf_findVarInFrame(SEXP, SEXP) {
    _NYI("Rf_findVarInFrame");
}

SEXP Rf_findVarInFrame3(SEXP, SEXP, Rboolean) {
    _NYI("Rf_findVarInFrame3");
}

SEXP Rf_getAttrib(SEXP, SEXP) {
    _NYI("Rf_getAttrib");
}

SEXP Rf_GetOption1(SEXP) {
    _NYI("Rf_GetOptions1");
}

void Rf_gsetVar(SEXP, SEXP, SEXP) {
    _NYI("Rf_gsetVar");
}

SEXP Rf_install(const char * s) {
    return globalState->installSEXP(CreateSymbol(globalState->internStr(s)));
}

SEXP Rf_lengthgets(SEXP, R_len_t) {
    _NYI("Rf_lengthgets");
}

SEXP Rf_xlengthgets(SEXP, R_xlen_t) {
    throw "Rf_xlengthgets";
}

SEXP Rf_matchE(SEXP, SEXP, int, SEXP) {
    _NYI("Rf_matchE");
}

SEXP Rf_namesgets(SEXP, SEXP) {
    _NYI("Rf_namesgets");
}

SEXP Rf_mkChar(const char * str) {
    return new SEXPREC(globalState->internStr(str));
}

int Rf_ncols(SEXP) {
    _NYI("Rf_ncols");
}

int Rf_nrows(SEXP) {
    _NYI("Rf_nrows");
}

SEXP Rf_nthcdr(SEXP, int) {
    _NYI("Rf_nthcdr");
}

SEXP Rf_protect(SEXP s) {
    if (R_PPStackTop < R_PPStackSize)
        R_PPStack[R_PPStackTop++] = s;
    else R_signal_protect_error();
    return s;
}

SEXP Rf_setAttrib(SEXP in, SEXP attr, SEXP value) {
    Value i = in->getValue();
    Value a = attr->getValue();
    if(!i.isObject()) {
        printf("in argument is not an object in Rf_setAttrib");
    }
    if(!a.isCharacter() || ((Character const&)a).length() != 1) {
        printf("attr argument is not a 1-element Character in Rf_setAttrib");
    }

    // TODO: This needs to erase attributes too.
    // I should probably just be calling a shared SetAttr API function
    // which is also used by the interpreter.
    Object o = (Object&)i;

    Dictionary* d = o.hasAttributes()
                    ? o.attributes()->clone(1)
                    : new Dictionary(1);
    d->insert(((Character const&)a)[0]) = value->getValue();
    o.attributes(d);
    return new SEXPREC(o);
}

void Rf_setVar(SEXP, SEXP, SEXP) {
    _NYI("Rf_setVar");
}

SEXP Rf_substitute(SEXP,SEXP) {
    _NYI("Rf_substitute");
}

const char * Rf_translateChar(SEXP) {
    _NYI("Rf_translateChar");
}

const char * Rf_translateCharUTF8(SEXP) {
    _NYI("Rf_translateCharUTF8");
}

const char * Rf_type2char(SEXPTYPE) {
    _NYI("Rf_type2char");
}

void Rf_unprotect(int l) {
    if (R_PPStackTop >=  l)
        R_PPStackTop -= l;
    else R_signal_unprotect_error();
}

void Rf_unprotect_ptr(SEXP) {
    _NYI("Rf_unprotect_ptr");
}

void R_signal_protect_error(void) {
    _NYI("R_signal_protect_error");
}

void R_signal_unprotect_error(void) {
    _NYI("R_signal_unprotect_error");
}

void R_signal_reprotect_error(PROTECT_INDEX i) {
    _NYI("R_signal_reprotect_error");
}

void R_ProtectWithIndex(SEXP, PROTECT_INDEX *) {
    _NYI("R_ProtectWithIndex");
}

void R_Reprotect(SEXP, PROTECT_INDEX) {
    _NYI("R_Reprotect");
}

SEXP R_tryEvalSilent(SEXP, SEXP, int *) {
    _NYI("R_tryEvalSilent");
}

const char *R_curErrorBuf() {
    _NYI("R_curErrorBuf");
}

cetype_t Rf_getCharCE(SEXP) {
    _NYI("Rf_getCharCE");
}

SEXP Rf_mkCharCE(const char *, cetype_t) {
    _NYI("Rf_mkCharCE");
}

SEXP Rf_mkCharLenCE(const char *, int, cetype_t) {
    _NYI("Rf_mkCharLenCE");
}


/* External pointer interface */
SEXP R_MakeExternalPtr(void *p, SEXP tag, SEXP prot) {
    _NYI("R_MakeExternalPtr");
}

void *R_ExternalPtrAddr(SEXP s) {
    _NYI("R_ExternalPtrAddr");
}

SEXP R_ExternalPtrTag(SEXP s) {
    _NYI("R_ExternalPtrTag");
}

/* Environment and Binding Features */
SEXP R_FindNamespace(SEXP info) {
    Value i = info->getValue();
    List call(2);
    
    Character fn(1);
    fn[0] = globalState->internStr("getNamespace");
    
    call[0] = fn;
    call[1] = i;

    call = CreateCall(call);
   
    Thread* thread = globalState->getThread();
    Code* code = Compiler::compileTopLevel(*thread, call);
    Value r = thread->eval(code, globalState->global);
    globalState->deleteThread(thread);

    return new SEXPREC(r);
}

/* needed for R_load/savehistory handling in front ends */
void Rf_warningcall(SEXP, const char *, ...) {
    _NYI("Rf_warningcall");
}

/* slot management (in attrib.c) */
SEXP R_do_slot(SEXP obj, SEXP name) {
    _NYI("R_do_slot");
}

SEXP R_do_slot_assign(SEXP obj, SEXP name, SEXP value) {
    _NYI("R_do_slot_assign");
}

int R_has_slot(SEXP obj, SEXP name) {
    _NYI("R_has_slot");
}

/* class definition, new objects (objects.c) */
SEXP R_do_MAKE_CLASS(const char *what) {
    _NYI("R_do_MAKE_CLASS");
}

SEXP R_do_new_object(SEXP class_def) {
    _NYI("R_do_new_object");
}

/* supporting  a C-level version of  is(., .) : */
int R_check_class_and_super(SEXP x, const char **valid, SEXP rho) {
    _NYI("R_check_class_and_super");
}

int R_check_class_etc      (SEXP x, const char **valid) {
    _NYI("R_check_class_etc");
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
    _NYI("R_popen");
}

int R_system(const char *) {
    _NYI("R_system");
}

/*
   These are the inlinable functions that are provided in Rinlinedfuns.h
   It is *essential* that these do not appear in any other header file,
   with or without the Rf_ prefix.
*/

SEXP     Rf_allocVector(SEXPTYPE type, R_xlen_t length) {
    return Rf_allocVector3(type, length, NULL);
}

Rboolean Rf_inherits(SEXP, const char *) {
    _NYI("Rf_inherits");
}

Rboolean Rf_isArray(SEXP) {
    _NYI("Rf_isArray");
}

Rboolean Rf_isFrame(SEXP) {
    _NYI("Rf_isFrame");
}

Rboolean Rf_isFunction(SEXP) {
    _NYI("Rf_isFunction");
}

Rboolean Rf_isInteger(SEXP) {
    _NYI("Rf_isInteger");
}

Rboolean Rf_isLanguage(SEXP) {
    _NYI("Rf_isLanguage");
}

Rboolean Rf_isMatrix(SEXP) {
    _NYI("Rf_isMatrix");
}

Rboolean Rf_isNewList(SEXP) {
    _NYI("Rf_isNewList");
}

Rboolean Rf_isNumeric(SEXP) {
    _NYI("Rf_isNumeric");
}

Rboolean Rf_isValidString(SEXP) {
    _NYI("Rf_isValidString");
}

Rboolean Rf_isVector(SEXP) {
    _NYI("Rf_isVector");
}

Rboolean Rf_isVectorAtomic(SEXP) {
    _NYI("Rf_isVectorAtomic");
}

Rboolean Rf_isVectorList(SEXP) {
    _NYI("Rf_isVectorList");
}

SEXP     Rf_lang1(SEXP) {
    _NYI("Rf_lang1");
}

SEXP     Rf_lang2(SEXP, SEXP) {
    _NYI("Rf_lang2");
}

SEXP     Rf_lang3(SEXP, SEXP, SEXP) {
    _NYI("Rf_lang3");
}

SEXP     Rf_lang4(SEXP, SEXP, SEXP, SEXP) {
    _NYI("Rf_lang4");
}

SEXP     Rf_lang5(SEXP, SEXP, SEXP, SEXP, SEXP) {
    _NYI("Rf_lang5");
}

R_len_t  Rf_length(SEXP) {
    _NYI("Rf_length");
}

SEXP     Rf_list4(SEXP, SEXP, SEXP, SEXP) {
    _NYI("Rf_list4");
}

SEXP     Rf_mkNamed(SEXPTYPE, const char **) {
    _NYI("Rf_mkNamed");
}

SEXP     Rf_mkString(const char * s) {
    Character v(1);
    v[0] = globalState->internStr(s);
    return new SEXPREC(v);
}

int  Rf_nlevels(SEXP) {
    _NYI("Rf_nlevels");
}

SEXP     Rf_ScalarInteger(int i) {
    return new SEXPREC(Integer::c(i));
}

SEXP     Rf_ScalarLogical(int f) {
    return new SEXPREC(Logical::c(f ? Logical::TrueElement : Logical::FalseElement));
}

SEXP     Rf_ScalarReal(double d) {
    return new SEXPREC(Double::c(d));
}

SEXP     Rf_ScalarString(SEXP s) {
    if(s->getType() != SEXPREC::STRING) {
        printf("Rf_ScalarString called without a scalar string\n");
        throw;
    }

    Character r(1);
    r[0] = s->getString();
    return new SEXPREC(r);
}

extern "C" {
void R_SignalCStackOverflow(intptr_t) {
    _NYI("R_SignalCStackOverflow");
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
    _NYI("R_AllocStringBuffer");
}

void R_FreeStringBuffer(R_StringBuffer *buf) {
    _NYI("R_FreeStringBuffer");
}

// From main/unique.c, used by the utils package
SEXP Rf_csduplicated(SEXP x) {
    _NYI("Rf_csduplicated");
}

// For some reason the utils package also wants this without the Rf_.
SEXP csduplicated(SEXP x) {
    return Rf_csduplicated(x); 
}


// From main/internet.c, used by the utils package
SEXP Rsockclose(SEXP ssock) {
    _NYI("Rsockclose");
}

SEXP Rsockconnect(SEXP sport, SEXP shost) {
    _NYI("Rsockconnect");
}

SEXP Rsocklisten(SEXP ssock) {
    _NYI("Rsocklisten");
}

SEXP Rsockopen(SEXP sport) {
    _NYI("Rsockopen");
}

SEXP Rsockread(SEXP ssock, SEXP smaxlen) {
    _NYI("Rsockread");
}

SEXP Rsockwrite(SEXP ssock, SEXP sstring) {
    _NYI("Rsockwrite");
}

// From main/internet.c, used by the tools package
int extR_HTTPDCreate(const char *ip, int port) {
    _NYI("extR_HTTPDCreate");
}

void extR_HTTPDStop(void) {
    _NYI("extR_HTTPDStop");
}

// From main/dounzip.c, used by the utils package
SEXP Runzip(SEXP args) {
    _NYI("Runzip");
}

// From main/eval.c, used by the utils package
SEXP do_Rprof(SEXP args) {
    _NYI("do_Rprof");
}

// From main/eval.c, used by the stats package
SEXP R_execMethod(SEXP op, SEXP rho) {
    _NYI("R_execMethod");
}

// From main/memory.c, used by the utils package
SEXP do_Rprofmem(SEXP args) {
    _NYI("do_Rprofmem");
}

// From main/edit.c, used by the utils package
SEXP do_edit(SEXP call, SEXP op, SEXP args, SEXP rho) {
    _NYI("do_edit");
}

// From main/rlocale.c, used by the grDevices package
int Ri18n_wcwidth(wchar_t c) {
    _NYI("Ri18n_wcwidth");
}

// From main/names.c, used by the methods package
const char *getPRIMNAME(SEXP object) {
    _NYI("getPRIMNAME");
}

// From main/xxxpr.f, used by the stats package
void intpr_() {
    _NYI("intpr");
}

void dblepr_() {
    _NYI("dblepr");
}

void rexit_() {
    _NYI("rexit");
}

void rwarn_() {
    _NYI("rwarn");
}

// From main/util.c, used by stats
void rchkusr_(void) {
    _NYI("rchkusr");
}

}

