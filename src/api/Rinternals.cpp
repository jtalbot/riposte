
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
    if(x->v.type() != Type::ScalarString) {
        printf("R_CHAR is not a string\n");
        return NULL;
    }

    return ((ScalarString const&)x->v).string()->s;
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
    return isSymbol(s->v) ? TRUE : FALSE;
}

Rboolean (Rf_isLogical)(SEXP s) {
    return s->v.isLogical() ? TRUE : FALSE;
}

Rboolean (Rf_isReal)(SEXP s) {
    return s->v.isDouble() ? TRUE : FALSE;
}

Rboolean (Rf_isComplex)(SEXP s) {
    _NYI("Rf_isComplex");
}

Rboolean (Rf_isExpression)(SEXP s) {
    return isExpression(s->v) ? TRUE : FALSE;
}

Rboolean (Rf_isEnvironment)(SEXP s) {
    return s->v.isEnvironment() ? TRUE : FALSE;
}

Rboolean (Rf_isString)(SEXP s) {
    return s->v.isCharacter() ? TRUE : FALSE;
}

Rboolean (Rf_isObject)(SEXP s) {
    _NYI("Rf_isObject");
}


/* General Cons Cell Attributes */
SEXP (ATTRIB)(SEXP x) {
    _NYI("ATTRIB");
}

int  (TYPEOF)(SEXP x) {

    int type = x->v.type();

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
            if(isSymbol(x->v))  return SYMSXP;
            else                return STRSXP;
        case Type::List:
            if(isCall(x->v))    return LANGSXP;
            else if(isExpression(x->v)) return EXPRSXP;
            else if(isPairlist(x->v)) return LISTSXP;
            else                return VECSXP;
        case Type::Integer32:   return INTSXP;
        case Type::Logical32:   return LGLSXP;
        case Type::ScalarString: return CHARSXP;
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
    if(!x->v.isVector()) {
        printf("LENGTH called on non-vector\n");
        return 0;
    }

    // TODO: should check to make sure the length doesn't overflow.
    return (int)((Vector const&)x->v).length();    
}

R_xlen_t  (XLENGTH)(SEXP x) {
    if(!x->v.isVector()) {
        printf("LENGTH called on non-vector\n");
        return 0;
    }

    return (R_xlen_t)((Vector const&)x->v).length();    
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

    if(x->v.isInteger()) {
        x->v = Integer32::fromInteger((Integer const&)x->v);
    }
    
    if(x->v.type() == Type::Integer32) {
        return ((Integer32&)x->v).v();
    }
    else {
        printf("Called INTEGER on something that is not an integer\n");
        return NULL;
    }
}

Rbyte *(RAW)(SEXP x) {
    _NYI("RAW");
}

double *(REAL)(SEXP x) {
    if(x->v.isDouble()) {
        return ((Double&)x->v).v();
    }
    else {
        printf("Called REAL on something that is not a double\n");
        return NULL;
    }

}

Rcomplex *(COMPLEX)(SEXP x) {
    _NYI("COMPLEX");
}

SEXP (STRING_ELT)(SEXP x, R_xlen_t i) {
    if(!x->v.isCharacter()) {
        printf("Argument to STRING_ELT is not a list");
        throw;
    }
    Character const& l = (Character const&)x->v;
    if(i >= l.length()) {
        printf("Accessing past the end of the vector in STRING_ELT");
        throw;
    }
    Value r;
    ScalarString::Init(r, l[i]);
    return new SEXPREC(r);
}

SEXP (VECTOR_ELT)(SEXP x, R_xlen_t i) {
    if(!x->v.isList()) {
        printf("Argument to VECTOR_ELT is not a list");
        throw;
    }
    List const& l = (List const&)x->v;
    if(i >= l.length()) {
        printf("Accessing past the end of the list in VECTOR_ELT");
        throw;
    }
    return new SEXPREC(l[i]);
}

void SET_STRING_ELT(SEXP x, R_xlen_t i, SEXP a) {
    if(!x->v.isCharacter()) {
        printf("Argument is not a Character vector in SET_STRING_ELT");
        throw;
    }
    if(a->v.type() != Type::ScalarString) {
        printf("Argument is not a scalar string in SET_STRING_ELT");
        throw;
    }
    
    Character& c = (Character&)x->v;
    if(i >= c.length()) {
        printf("Assigning past the end in SET_STRING_ELT");
        throw;
    }
    
    c[i] = ((ScalarString const&)a).string();
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
    if(!e->v.isList()) {
        printf("Argument is not a list in CAR: %d\n", e->v.type());
        throw;
    }
    // TODO: check length
    return new SEXPREC(((List const&)e->v)[0]);
}

SEXP (CDR)(SEXP e) {
    if(!e->v.isList()) {
        printf("Argument is not a list in CDR\n");
        throw;
    }
    // TODO: this is going to lose names.
    // TODO: check length
    List const& l = (List const&)e->v;
    List r(l.length()-1);
    for(int64_t i = 1; i < l.length(); ++i)
        r[i-1] = l[i];
    
    return new SEXPREC(r);
}

SEXP (CADR)(SEXP e) {
    if(!e->v.isList()) {
        printf("Argument is not a list in CADR\n");
        throw;
    }
    // TODO: check length
    return new SEXPREC(((List const&)e->v)[1]);
}

SEXP (CDDR)(SEXP e) {
    if(!e->v.isList()) {
        printf("Argument is not a list in CDDR\n");
        throw;
    }
    // TODO: this is going to lose names.
    // TODO: check length
    List const& l = (List const&)e->v;
    List r(l.length()-2);
    for(int64_t i = 2; i < l.length(); ++i)
        r[i-2] = l[i];
    
    return new SEXPREC(r);
}

SEXP (CADDR)(SEXP e) {
    if(!e->v.isList()) {
        printf("Argument is not a list in CADDR\n");
        throw;
    }
    // TODO: check length
    return new SEXPREC(((List const&)e->v)[2]);
}

SEXP (CADDDR)(SEXP e) {
    if(!e->v.isList()) {
        printf("Argument is not a list in CADDR\n");
        throw;
    }
    // TODO: check length
    return new SEXPREC(((List const&)e->v)[3]);
}

SEXP (CAD4R)(SEXP e) {
    if(!e->v.isList()) {
        printf("Argument is not a list in CAD4R\n");
        throw;
    }
    // TODO: check length
    return new SEXPREC(((List const&)e->v)[4]);
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

SEXP Rf_coerceVector(SEXP v, SEXPTYPE type) {
    if(TYPEOF(v) == type)
        return v;

    _NYI("Rf_coerceVector");
}

SEXP Rf_PairToVectorList(SEXP x) {
    _NYI("Rf_PairToVectorList");
}

SEXP Rf_VectorToPairList(SEXP x) {
    _NYI("Rf_VectorToPairList");
}

int Rf_asLogical(SEXP x) {
    if(x->v.isLogical1()) {
        if(Logical::isTrue(x->v.c)) return 1;
        else if(Logical::isFalse(x->v.c)) return 0;
        else return R_NaInt;
    }
    printf("_NYI type in Rf_asLogical");
    _NYI("Rf_asLogical");
}
int Rf_asInteger(SEXP x) {
    if(x->v.isInteger1())
        return (int)x->v.i;
    else if(x->v.isDouble1())
        return (int)x->v.d;
    else 
        _NYI("Rf_asInteger");
}
double Rf_asReal(SEXP x) {
    if(x->v.isInteger1())
        return (double)x->v.i;
    else if(x->v.isDouble1())
        return (double)x->v.d;
    else 
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
    if(!isSymbol(symbol->v)) {
        printf("Rf_defineVar called without a symbol\n");
        throw;
    }

    if(!env->v.isEnvironment()) {
        printf("Rf_defineVar called without an environment\n");
        throw;
    }

    Value v = ToRiposteValue(val->v);

    ((REnvironment&)env->v).environment()->insert(SymbolStr(symbol->v)) = v;
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
    if(!x->v.isPromise())
        printf("v in Rf_eval is not a promise");

    if(!env->v.isEnvironment())
        printf("env in Rf_eval is not an environment");

    //Environment* evalenv = ((REnvironment&)env->v).environment();

    Promise const& p = (Promise const&)x->v;

    Thread* thread = globalState->getThread();
    Value r = thread->eval(p, 0);
    globalState->deleteThread(thread);
    
    return new SEXPREC(r);
}

SEXP Rf_findFun(SEXP, SEXP) {
    _NYI("Rf_findFun");
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
    Value r = ((REnvironment&)env->v).environment()->getRecursive(symbol->v.s, foundEnv);
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

SEXP Rf_getAttrib(SEXP obj, SEXP symbol) {
    if(!obj->v.isObject()) {
        printf("argument to getAttrib is not an object");
        throw;
    }
    if(!symbol->v.isCharacter() ||
       ((Character const&)symbol->v).length() != 1) {
        printf("argument to getAttrib is not a one element Character");
        throw;
    }

    if( ((Object const&)obj->v).hasAttributes() &&
        ((Object const&)obj->v).attributes()->has(SymbolStr(symbol->v)) ) {
        return new SEXPREC(
            ((Object const&)obj->v).attributes()->get(SymbolStr(symbol->v)) );
    }

    return R_NilValue;
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
    String s = globalState->internStr(str);
    Value r;
    ScalarString::Init(r, s);
    return new SEXPREC(r);
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
    if(!in->v.isObject()) {
        printf("in argument is not an object in Rf_setAttrib");
    }
    if(!attr->v.isCharacter() || ((Character const&)attr->v).length() != 1) {
        printf("attr argument is not a 1-element Character in Rf_setAttrib");
    }

    // TODO: This needs to erase attributes too.
    // I should probably just be calling a shared SetAttr API function
    // which is also used by the interpreter.
    Object o = (Object&)in->v;

    Dictionary* d = o.hasAttributes()
                    ? o.attributes()->clone(1)
                    : new Dictionary(1);
    d->insert(((Character const&)attr->v)[0]) = ToRiposteValue(value->v);
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
    return CE_NATIVE;
}

SEXP Rf_mkCharCE(const char * str, cetype_t type) {
    String s = globalState->internStr(str);
    Value r;
    ScalarString::Init(r, s);
    return new SEXPREC(r);
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
    List call(2);
    
    Character fn(1);
    fn[0] = globalState->internStr("getNamespace");
    
    call[0] = fn;
    call[1] = ToRiposteValue(info->v);

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

Rboolean Rf_isNewList(SEXP s) {
    return (TYPEOF(s) == NILSXP || TYPEOF(s) == VECSXP) ? TRUE : FALSE;
}

Rboolean Rf_isNumeric(SEXP s) {
    switch(TYPEOF(s)) {
    case INTSXP:
    case REALSXP:
        return TRUE;
    default:
        return FALSE;
    }
}

Rboolean Rf_isValidString(SEXP s) {
    _NYI("Rf_isValidString");
}

Rboolean Rf_isVector(SEXP s) {
    switch(TYPEOF(s)) {
    case LGLSXP:
    case INTSXP:
    case REALSXP:
    case CPLXSXP:
    case STRSXP:
    case RAWSXP:

    case VECSXP:
    case EXPRSXP:
        return TRUE;
    default:
        return FALSE;
    }
}

Rboolean Rf_isVectorAtomic(SEXP s) {
    switch (TYPEOF(s)) {
    case LGLSXP:
    case INTSXP:
    case REALSXP:
    case CPLXSXP:
    case STRSXP:
    case RAWSXP:
        return TRUE;
    default: /* including NULL */
        return FALSE;
    }
}

Rboolean Rf_isVectorList(SEXP s) {
    switch (TYPEOF(s)) {
    case VECSXP:
    case EXPRSXP:
        return TRUE;
    default:
        return FALSE;
    }
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

R_len_t  Rf_length(SEXP x) {
    if(!x->v.isVector()) {
        printf("Rf_length called on non-vector\n");
        return 0;
    }

    // TODO: should check to make sure the length doesn't overflow.
    return (R_len_t)((Vector const&)x->v).length();    
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
    if(s->v.type() != Type::ScalarString) {
        printf("Rf_ScalarString called without a scalar string\n");
        throw;
    }

    Character r(1);
    r[0] = ((ScalarString const&)s->v).string();
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

