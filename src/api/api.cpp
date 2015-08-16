// Some of this code is taken from R header files under the LGPL.

#include "api.h"

#include "../frontend.h"

#define R_NO_REMAP
#include <Rinternals.h>
#include <lzma.h>

const int32_t Integer32::NAelement = std::numeric_limits<int32_t>::min();
const int32_t Logical32::NAelement = std::numeric_limits<int32_t>::min();

Integer Integer32::toInteger(Integer32 const& i) {
    Integer result(i.length());
    result.attributes(i.attributes());
    for(int64_t j = 0; j < i.length(); ++j)
        result[j] = i[j];
    return result;
}

Integer32 Integer32::fromInteger(Integer const& i) {
    Integer32 result(i.length());
    result.attributes(i.attributes());
    // TODO: warn on truncation
    for(int64_t j = 0; j < i.length(); ++j)
        result[j] = (int32_t)i[j];
    return result;
}

Logical Logical32::toLogical(Logical32 const& i) {
    Logical result(i.length());
    result.attributes(i.attributes());
    for(int64_t j = 0; j < i.length(); ++j)
        result[j] = Logical32::isNA(i[j])
            ? Logical::NAelement
            : i[j] ? Logical::TrueElement : Logical::FalseElement;
    return result;
}

Logical32 Logical32::fromLogical(Logical const& i) {
    Logical32 result(i.length());
    result.attributes(i.attributes());
    for(int64_t j = 0; j < i.length(); ++j)
        result[j] = Logical::isNA(i[j])
            ? Logical32::NAelement
            : i[j] == Logical::TrueElement ? 1 : 0;
    return result;
}

List Pairlist::toList(Pairlist const& pairlist) {
    std::vector<Value> values;
    std::vector<String> names;
    bool anyNames = false;

    Pairlist const* p = &pairlist;

    do {
        values.push_back(p->car()->v);
        names.push_back(p->tag());
        if(names.back() != Strings::empty)
            anyNames = true;
        if(p->cdr()->v.type() != Type::Pairlist)
            break;
        p = (Pairlist*)&p->cdr()->v;
    } while(true);

    List result(values.size());
    for(size_t i = 0; i < values.size(); ++i)
        result[i] = values[i];

    Value rnames = Value::Nil();

    if(anyNames) {
        Character tnames(names.size());
        for(size_t i = 0; i < names.size(); ++i)
            tnames[i] = names[i];
        rnames = tnames;
    }

    return CreatePairlist(result, rnames);
}

Value Pairlist::fromList(List const& list) {
    if(list.length() == 0)
        return Value::Nil();

    SEXP tail = R_NilValue;
    for(size_t i = list.length()-1; i > 0; --i) {
        Value v;
        String tag = (hasNames(list)) ? ((Character const&)getNames(list))[i] : Strings::empty;
        Pairlist::Init(v, ToSEXP(list[i]), tail, tag);
        tail = ToSEXP(v);
    }
    Pairlist v;
    String tag = (hasNames(list)) ? ((Character const&)getNames(list))[0] : Strings::empty;
    Pairlist::Init(v, ToSEXP(list[0]), tail, tag);
    
    v.attributes(list.attributes());

    return v;
}

Value ToRiposteValue(Value const& v) {
    if(v.type() == Type::Integer32)
        return Integer32::toInteger((Integer32 const&)v);
    else if(v.type() == Type::Logical32)
        return Logical32::toLogical((Logical32 const&)v);
    else if(v.type() == Type::ScalarString)
        return Character::c(((ScalarString const&)v).string());
    else if(v.type() == Type::Pairlist)
        return Pairlist::toList((Pairlist const&)v);
    else
        return v;
}

std::map<String, SEXP> symbols;
SEXP ToSEXP(Value const& v) {
    if(isSymbol(v)) {
        std::map<String, SEXP>::const_iterator i =
           symbols.find(SymbolStr(v));
        if(i != symbols.end())
            return i->second;

        SEXP sexp = global->installSEXP(CreateSymbol(*global, SymbolStr(v)));
        symbols[SymbolStr(v)] = sexp;
        return sexp;
    }
    else return new SEXPREC(v);
}

SEXP ToSEXP(char const* s) {
    String str = global->internStr(s);
    return ToSEXP(CreateSymbol(*global, str));
}

// Rinternals.h

/* Evaluation Environment */
SEXP  R_GlobalEnv;        /* The "global" environment */

SEXP  R_EmptyEnv;     /* An empty environment at the root of the
                        environment tree */
SEXP  R_BaseEnv;      /* The base environment; formerly R_NilValue */
SEXP  R_BaseNamespace;    /* The (fake) namespace for base */
SEXP  R_NamespaceRegistry;/* Registry for registered namespaces */

SEXP  R_Srcref;           /* Current srcref, for debuggers */

/* Special Values */
SEXP  R_NilValue;     /* The nil object */
SEXP  R_UnboundValue;     /* Unbound marker */
SEXP  R_MissingArg;       /* Missing argument marker */

SEXP    R_RestartToken;     /* Marker for restarted function calls */

/* Symbol Table Shortcuts */
SEXP  R_Bracket2Symbol;   /* "[[" */
SEXP  R_BracketSymbol;    /* "[" */
SEXP  R_BraceSymbol;      /* "{" */
SEXP  R_ClassSymbol;      /* "class" */
SEXP  R_DeviceSymbol;     /* ".Device" */
SEXP  R_DimNamesSymbol;   /* "dimnames" */
SEXP  R_DimSymbol;        /* "dim" */
SEXP  R_DollarSymbol;     /* "$" */
SEXP  R_DotsSymbol;       /* "..." */
SEXP  R_baseSymbol;       /* "base" */
SEXP  R_DropSymbol;       /* "drop" */
SEXP  R_LastvalueSymbol;  /* ".Last.value" */
SEXP  R_LevelsSymbol;     /* "levels" */
SEXP  R_ModeSymbol;       /* "mode" */
SEXP  R_NameSymbol;       /* "name" */
SEXP  R_NamesSymbol;      /* "names" */
SEXP  R_NaRmSymbol;       /* "na.rm" */
SEXP  R_PackageSymbol;    /* "package" */
SEXP  R_QuoteSymbol;      /* "quote" */
SEXP  R_RowNamesSymbol;   /* "row.names" */
SEXP  R_SeedsSymbol;      /* ".Random.seed" */
SEXP  R_SourceSymbol;     /* "source" */
SEXP  R_TspSymbol;        /* "tsp" */

SEXP  R_dot_defined;      /* ".defined" */
SEXP  R_dot_Method;       /* ".Method" */
SEXP  R_dot_target;       /* ".target" */

/* Missing Values - others from Arith.h */
SEXP  R_NaString;     /* NA_STRING as a CHARSXP */
SEXP  R_BlankString;      /* "" as a CHARSXP */

// Defn.h

SEXP R_TrueValue;
SEXP R_FalseValue;
SEXP R_LogicalNAValue;

int R_PPStackSize;
int R_PPStackTop;
SEXP* R_PPStack;

Rboolean R_Interactive;

extern "C" {

struct DLLInfo;

void R_init_libR(DLLInfo *) {

    R_PPStackSize = 50000;
    R_PPStackTop = 0;
    R_PPStack = new SEXP[R_PPStackSize];

    SEXPStack* stack = new SEXPStack;
    stack->size = &R_PPStackTop;
    stack->stack = R_PPStack;
    global->apiStack = stack;

    Value globalEnv;
    REnvironment::Init(globalEnv, global->global);
    R_GlobalEnv = global->installSEXP(globalEnv);
    
    Value empty;
    REnvironment::Init(empty, global->empty);
    R_EmptyEnv = global->installSEXP(empty);

    R_NilValue = global->installSEXP(Null::Singleton());
    R_UnboundValue = R_MissingArg =
        global->installSEXP(Value::Nil());

    R_TrueValue = global->installSEXP(Logical::c(Logical::TrueElement));
    R_FalseValue = global->installSEXP(Logical::c(Logical::FalseElement));
    R_LogicalNAValue = global->installSEXP(Logical::c(Logical::NAelement));

    R_PackageSymbol =
        ToSEXP("package");
    R_Bracket2Symbol =   /* "[[" */
        ToSEXP("[[");
    R_BracketSymbol =    /* "[" */
        ToSEXP("[");
    R_BraceSymbol =      /* "{" */
        ToSEXP("{");
    R_ClassSymbol =      /* "class" */
        ToSEXP("class");
    R_DeviceSymbol =     /* ".Device" */
        ToSEXP(".Device");
    R_DimNamesSymbol =   /* "dimnames" */
        ToSEXP("dimnames");
    R_DimSymbol =        /* "dim" */
        ToSEXP("dim");
    R_DollarSymbol =     /* "$" */
        ToSEXP("$");
    R_DotsSymbol =       /* "..." */
        ToSEXP("...");
    R_DropSymbol =       /* "drop" */
        ToSEXP("drop");
    R_LastvalueSymbol =  /* ".Last.value" */
        ToSEXP(".Last.value");
    R_LevelsSymbol =     /* "levels" */
        ToSEXP("levels");
    R_ModeSymbol =       /* "mode" */
        ToSEXP("mode");
    R_NameSymbol =       /* "name" */
        ToSEXP("name");
    R_NamesSymbol =      /* "names" */
        ToSEXP("names");
    R_NaRmSymbol =       /* "na.rm" */
        ToSEXP("na.rm");
    R_PackageSymbol =    /* "package" */
        ToSEXP("package");
    R_QuoteSymbol =      /* "quote" */
        ToSEXP("quote");
    R_RowNamesSymbol =   /* "row.names" */
        ToSEXP("row.names");
    R_SeedsSymbol =      /* ".Random.seed" */
        ToSEXP(".Random.seed");
    R_SourceSymbol =     /* "source" */
        ToSEXP("source");
    R_TspSymbol =        /* "tsp" */
        ToSEXP("tsp");

    R_dot_defined =      /* ".defined" */
        ToSEXP(".defined");
    R_dot_Method =       /* ".Method" */
        ToSEXP(".Method");
    R_dot_target =       /* ".target" */
        ToSEXP(".target");

    Value NaString;
    ScalarString::Init(NaString, Strings::NA);
    R_NaString = ToSEXP(NaString); /* NA_STRING as a CHARSXP */
    Value BlankString;
    ScalarString::Init(BlankString, Strings::empty);
    R_BlankString = ToSEXP(BlankString);     /* "" as a CHARSXP */

    R_Interactive = TRUE;

}

}

