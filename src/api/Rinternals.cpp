
#include "api.h"
#include <stddef.h>

#include "../riposte.h"
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

int  (OBJECT)(SEXP x) {
    _NYI("OBJECT");
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
        case Type::Pairlist:
            if(isCall(x->v))    return LANGSXP;
            else if(isExpression(x->v)) return EXPRSXP;
            else                return LISTSXP;
        default:
            throw "Unknown type in TYPEOF";
    }
}

int  (NAMED)(SEXP x) {
    return (x->v.isObject() && hasNames((Object const&)x->v))
        ? 1 : 0;
}

void (SET_OBJECT)(SEXP x, int v) {
    // do nothing, we don't have a flag
}

void (SET_TYPEOF)(SEXP x, int v) {
    switch(v) {
        case LANGSXP: {
            if(!(x->v).isList())
                printf("SET_TYPEOF called with LANGSXP on non-list\n");
            x->v = CreateCall(*global, (List const&)x->v);
        }
        break;
        default:
            printf("SET_TYPEOF called with unknown type: %d\n", v);
    }
}

void (SET_NAMED)(SEXP x, int v) {
    // do nothing, we don't have a flag
}

void SET_ATTRIB(SEXP x, SEXP v) {
    printf("SET:ATTRIB %d %d\n", x->v.type(), v->v.type());
    printf("NYI: SET_ATTRIB. What does it do?\n");
}

void DUPLICATE_ATTRIB(SEXP to, SEXP from) {
    if(!to->v.isObject() || !from->v.isObject()) {
        printf("DUPLICATE_ATTRIB arguments are not objects");
        throw;
    }
        
    Object& t = (Object&)to->v;
    Object const& f = (Object const&)from->v;
    t.attributes(f.attributes());
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
        printf("XLENGTH called on non-vector\n");
        return 0;
    }

    return (R_xlen_t)((Vector const&)x->v).length();    
}

void (SETLENGTH)(SEXP x, int v) {
    _NYI("SETLENGTH");
}

int  (IS_LONG_VEC)(SEXP x) {
    if(!x->v.isVector()) {
        printf("IS_LONG_VEC called on non-vector\n");
        return 0;
    }

    int64_t len = ((Vector const&)x->v).length();

    return len > std::numeric_limits<int>::max() ? 1 : 0;
}

int  *(LOGICAL)(SEXP x) {

    if(x->v.isLogical()) {
        x->v = Logical32::fromLogical((Logical const&)x->v);
    }
    
    if(x->v.type() == Type::Logical32) {
        return ((Logical32&)x->v).v();
    }
    else {
        printf("Called LOGICAL on something that is not a logical\n");
        return NULL;
    }
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
    return ToSEXP(r);
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
    return ToSEXP(l[i]);
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
    c[i] = ((ScalarString const&)a->v).string();
}

SEXP SET_VECTOR_ELT(SEXP x, R_xlen_t i, SEXP v) {
    if(!x->v.isList()) {
        printf("Argument is not a list in SET_VECTOR_ELT");
        throw;
    }
    
    List& l = (List&)x->v;
    if(i >= l.length()) {
        printf("Assigning past the end in SET_VECTOR_ELT");
        throw;
    }
    l[i] = v->v;

    return v;
}

/* List Access Functions */
/* These also work for ... objects */

SEXP (TAG)(SEXP e) {
    if(e->v.isList())
        e->v = Pairlist::fromList((List const&)e->v);

    if(e->v.type() == Type::Pairlist) {
        return ToSEXP(((Pairlist const&)e->v).tag()->s);
    } else {
        printf("Argument to TAG is not a pairlist: %d\n", e->v.type());
        throw;
    }
}

SEXP (CAR)(SEXP e) {
    if(e->v.isList())
        e->v = Pairlist::fromList((List const&)e->v);

    if(e->v.type() == Type::Pairlist) {
        return ((Pairlist const&)e->v).car();
    } else {
        printf("Argument to CAR is not a pairlist: %d\n", e->v.type());
        throw;
    }
}

SEXP (CDR)(SEXP e) {
    if(e->v.isList())
        e->v = Pairlist::fromList((List const&)e->v);

    if(e->v.type() == Type::Pairlist) {
        return ((Pairlist const&)e->v).cdr();
    } else {
        printf("Argument to CDR is not a pairlist: %d\n", e->v.type());
        throw;
    }
}

SEXP (CADR)(SEXP e) {
    return CAR(CDR(e));
}

SEXP (CDDR)(SEXP e) {
    return CDR(CDR(e));
}

SEXP (CADDR)(SEXP e) {
    return CAR(CDR(CDR(e)));
}

SEXP (CADDDR)(SEXP e) {
    return CAR(CDR(CDR(CDR(e))));
}

SEXP (CAD4R)(SEXP e) {
    return CAR(CDR(CDR(CDR(CDR(e)))));
}

void SET_TAG(SEXP e, SEXP y) {
    if(e->v.isList())
        e->v = Pairlist::fromList((List const&)e->v);

    if(e->v.type() == Type::Pairlist) {
        if(isSymbol(y->v)) {
            return ((Pairlist const&)e->v).tag(SymbolStr(y->v));
        }
        else {
            printf("Tag argument to SET_TAG is not a symbol: %d\n", y->v.type());
            throw;
        }
    } else {
        printf("Argument to SET_TAG is not a pairlist: %d\n", e->v.type());
        throw;
    }
}

SEXP SETCAR(SEXP e, SEXP y) {
    if(e->v.isList())
        e->v = Pairlist::fromList((List const&)e->v);

    if(e->v.type() == Type::Pairlist) {
        ((Pairlist const&)e->v).car(y);
    } else {
        printf("Argument to CDR is not a pairlist: %d\n", e->v.type());
        throw;
    }

    return y;
}

SEXP SETCDR(SEXP e, SEXP y) {
    if(e->v.isList())
        e->v = Pairlist::fromList((List const&)e->v);

    if(e->v.type() == Type::Pairlist) {
        ((Pairlist const&)e->v).cdr(y);
    } else {
        printf("Argument to CDR is not a pairlist: %d\n", e->v.type());
        throw;
    }

    return y;
}

SEXP SETCADR(SEXP e, SEXP y) {
    return SETCAR(CDR(e), y);
}

SEXP SETCADDR(SEXP e, SEXP y) {
    return SETCAR(CDR(CDR(e)), y);
}

SEXP SETCADDDR(SEXP e, SEXP y) {
    return SETCAR(CDR(CDR(CDR(e))), y);
}

SEXP SETCAD4R(SEXP e, SEXP y) {
    return SETCAR(CDR(CDR(CDR(CDR(e)))), y);
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
    else if(x->v.isLogical1())
        return Logical::isTrue(x->v.c) ? 1 : 0;
    
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

SEXP Rf_allocMatrix(SEXPTYPE type, int rows, int cols) {
    SEXP r = Rf_allocVector3(type, ((R_xlen_t)rows)*cols, NULL);

    Dictionary* d = new Dictionary(2);
    
    d->insert(MakeString("class")) =
        Character::c(MakeString("matrix"));
    d->insert(MakeString("dim")) =
        Integer::c(rows, cols);

    ((Object&)r->v).attributes(d);

    return r;
}

SEXP Rf_allocList(int n) {
    SEXP r = R_NilValue;
    for(int i = 0; i < n; ++i) {
        Pairlist v;
        Pairlist::Init(v, R_NilValue, r, Strings::empty);
        r = ToSEXP(v);
    }
    return r;
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
        case LGLSXP: {
            v = Logical(len); 
		    for(int64_t i = 0; i < len; i++) ((Logical&)v)[i] = Logical::FalseElement;
        } break;
        case INTSXP: {
            v = Integer(len); 
		    for(int64_t i = 0; i < len; i++) ((Integer&)v)[i] = 0;
        } break;
        case REALSXP: {
            v = Double(len);
		    for(int64_t i = 0; i < len; i++) ((Double&)v)[i] = 0;
        } break;
        case STRSXP: {
            v = Character(len);
		    for(int64_t i = 0; i < len; i++) ((Character&)v)[i] = Strings::empty;
        } break;
        case VECSXP: {
            v = List(len);
		    for(int64_t i = 0; i < len; i++) ((List&)v)[i] = Null::Singleton();
        } break;
        default: printf("Unsupported type in Rf_allocVector3: %d\n", type); throw; break;
    }

    return ToSEXP(v);
}

SEXP Rf_classgets(SEXP, SEXP) {
    _NYI("Rf_classgets");
}

SEXP Rf_cons(SEXP x, SEXP y) {
    if(!(y->v).isList() && !(y->v).isNull()) {
        printf("Rf_cons called without lists\n");
    }

    if((y->v).isNull()) {
        List r(1);
        r[0] = x->v;
        return ToSEXP(CreatePairlist(r));
    }
    else {
        List ly = (List const&)y->v;
        List r(1+ly.length());
        r[0] = x->v;
        for(int64_t i = 0; i < ly.length(); ++i)
            r[i+1] = ly[i];
        
        // TODO: copy names over too
        return ToSEXP(CreatePairlist(r));
    }

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

SEXP Rf_duplicate(SEXP x) {
    
    int type = x->v.type();

    Value r;
    switch(type) {
        case Type::Nil:
        case Type::Promise:
        case Type::Future:
        case Type::Closure:
        case Type::Environment:
        case Type::Externalptr:
        case Type::Null:
        case Type::ScalarString:
            return x;
        case Type::Raw: {
            Raw a(((Raw const&)x->v).length());
            for(int64_t i = 0; i < a.length(); ++i)
                a[i] = ((Raw const&)x->v)[i];
            r = a;
        } break;
        case Type::Logical: {
            Logical a(((Logical const&)x->v).length());
            for(int64_t i = 0; i < a.length(); ++i)
                a[i] = ((Logical const&)x->v)[i];
            r = a;
        } break;
        case Type::Integer: {
            Integer a(((Integer const&)x->v).length());
            for(int64_t i = 0; i < a.length(); ++i)
                a[i] = ((Integer const&)x->v)[i];
            r = a;
        } break;
        case Type::Double: {
            Double a(((Double const&)x->v).length());
            for(int64_t i = 0; i < a.length(); ++i)
                a[i] = ((Double const&)x->v)[i];
            r = a;
        } break;
        case Type::Character: {
            Character a(((Character const&)x->v).length());
            for(int64_t i = 0; i < a.length(); ++i)
                a[i] = ((Character const&)x->v)[i];
            r = a;
        } break;
        case Type::List: {
            List a(((List const&)x->v).length());
            for(int64_t i = 0; i < a.length(); ++i)
                a[i] = ((List const&)x->v)[i];
            r = a;
        } break;
        case Type::Integer32: {
            Integer32 a(((Integer32 const&)x->v).length());
            for(int64_t i = 0; i < a.length(); ++i)
                a[i] = ((Integer32 const&)x->v)[i];
            r = a;
        } break;
        case Type::Logical32: {
            Logical32 a(((Logical32 const&)x->v).length());
            for(int64_t i = 0; i < a.length(); ++i)
                a[i] = ((Logical32 const&)x->v)[i];
            r = a;
        } break;
        case Type::Pairlist: {
            Pairlist a = (Pairlist const&)x->v;
            Pairlist::Init(a, a.car(), a.cdr(), a.tag() );
            r = a;
            while(a.cdr()->v.type() == Type::Pairlist) {
                a = (Pairlist const&)a.cdr()->v;
                Pairlist n;
                Pairlist::Init(n, a.car(), a.cdr(), a.tag() );
                a.cdr(ToSEXP(n));
                a = n;
            }
        } break;
        default:
            throw "Unknown type in Rf_duplicate";
    }
    
    if((x->v).isObject()) {
        ((Object&)r).attributes(((Object const&)x->v).attributes());
    }

    return ToSEXP(r);
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

    Riposte::State& state = Riposte::newState();
    Value r = ((::State*)&state)->eval(p, 0);
    Riposte::deleteState(state);
    
    return ToSEXP(r);
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

    String in = global->strings.intern(symbol->v.s->s);
    Environment* foundEnv;
    Value const* v = ((REnvironment&)env->v).environment()->getRecursive(in, foundEnv);

    return (v && !v->isNil()) ? ToSEXP(*v) : R_UnboundValue;
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

    Value const* v = ((Object const&)obj->v).attributes()->get(SymbolStr(symbol->v));

    return v ? ToSEXP(*v) : R_NilValue;
}

SEXP Rf_GetOption1(SEXP) {
    _NYI("Rf_GetOptions1");
}

void Rf_gsetVar(SEXP, SEXP, SEXP) {
    _NYI("Rf_gsetVar");
}

SEXP Rf_install(const char * s) {
    return ToSEXP(s);
}

SEXP Rf_installChar(SEXP s) {
    return ToSEXP(s->v);
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
    Value r;
    ScalarString::Init(r, MakeString(str));
    return ToSEXP(r);
}

SEXP Rf_mkCharLen(const char * str, int len) {
    Value r;
    ScalarString::Init(r, MakeString(std::string(str, len)));
    return ToSEXP(r);
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

    // TODO:
    // I should probably just be calling a shared SetAttr API function
    // which is also used by the interpreter.
    Object o = (Object&)in->v;

    String s = ((Character const&)attr->v)[0];
    Value v = ToRiposteValue(value->v);

    o.attributes(v.isNull()
        ? o.attributes()->cloneWithout(s)
        : o.attributes()->cloneWith(s, v));
    
    return ToSEXP(o);
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

void R_ProtectWithIndex(SEXP s, PROTECT_INDEX * i) {
    *i = R_PPStackTop;
    Rf_protect(s);
}

void R_Reprotect(SEXP s, PROTECT_INDEX i) {
    if(i < R_PPStackTop && i >= 0)
        R_PPStack[i] = s;
    else
        R_signal_reprotect_error(i);
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
    Value r;
    ScalarString::Init(r, MakeString(str));
    SEXP v = ToSEXP(r);
    return v;
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
    fn[0] = MakeString("getNamespace");
    
    call[0] = fn;
    call[1] = ToRiposteValue(info->v);

    call = CreateCall(*global, call);
  
    Riposte::State& state = Riposte::newState(); 
    Code* code = Compiler::compileExpression((::State&)state, call);
    Value r = ((::State*)&state)->eval(code, global->global);
    Riposte::deleteState(state);

    return ToSEXP(r);
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
    global->installSEXP(object);
}

void R_ReleaseObject(SEXP object) {
    global->uninstallSEXP(object);
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

Rboolean Rf_isLanguage(SEXP x) {
    return isCall(x->v) ? TRUE : FALSE;
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

    if(x->v.type() == Type::Pairlist) {
        R_len_t size = 1;
        while(((Pairlist const&)x->v).cdr()->v.type() == Type::Pairlist) {
            x = ((Pairlist const&)x->v).cdr();
            size++;
        }
        return size;
    }

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
    v[0] = MakeString(s);
    return ToSEXP(v);
}

int  Rf_nlevels(SEXP) {
    _NYI("Rf_nlevels");
}

SEXP     Rf_ScalarInteger(int i) {
    return ToSEXP(Integer::c(i));
}

SEXP     Rf_ScalarLogical(int f) {
    return ToSEXP(Logical::c(f ? Logical::TrueElement : Logical::FalseElement));
}

SEXP     Rf_ScalarReal(double d) {
    return ToSEXP(Double::c(d));
}

SEXP     Rf_ScalarString(SEXP s) {
    if(s->v.type() != Type::ScalarString) {
        printf("Rf_ScalarString called without a scalar string\n");
        throw;
    }

    Character r(1);
    r[0] = ((ScalarString const&)s->v).string();
    return ToSEXP(r);
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

