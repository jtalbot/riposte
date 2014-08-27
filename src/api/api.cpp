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

Value ToRiposteValue(Value const& v) {
    if(v.type() == Type::Integer32)
        return Integer32::toInteger((Integer32 const&)v);
    else if(v.type() == Type::Logical32)
        return Logical32::toLogical((Logical32 const&)v);
    else if(v.type() == Type::ScalarString)
        return Character::c(((ScalarString const&)v).string());
    else
        return v;
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
        global->installSEXP(CreateSymbol(global->internStr("package")));
    R_Bracket2Symbol =   /* "[[" */
        global->installSEXP(CreateSymbol(global->internStr("[[")));
    R_BracketSymbol =    /* "[" */
        global->installSEXP(CreateSymbol(global->internStr("[")));
    R_BraceSymbol =      /* "{" */
        global->installSEXP(CreateSymbol(global->internStr("{")));
    R_ClassSymbol =      /* "class" */
        global->installSEXP(CreateSymbol(global->internStr("class")));
    R_DeviceSymbol =     /* ".Device" */
        global->installSEXP(CreateSymbol(global->internStr(".Device")));
    R_DimNamesSymbol =   /* "dimnames" */
        global->installSEXP(CreateSymbol(global->internStr("dimnames")));
    R_DimSymbol =        /* "dim" */
        global->installSEXP(CreateSymbol(global->internStr("dim")));
    R_DollarSymbol =     /* "$" */
        global->installSEXP(CreateSymbol(global->internStr("$")));
    R_DotsSymbol =       /* "..." */
        global->installSEXP(CreateSymbol(global->internStr("...")));
    R_DropSymbol =       /* "drop" */
        global->installSEXP(CreateSymbol(global->internStr("drop")));
    R_LastvalueSymbol =  /* ".Last.value" */
        global->installSEXP(CreateSymbol(global->internStr(".Last.value")));
    R_LevelsSymbol =     /* "levels" */
        global->installSEXP(CreateSymbol(global->internStr("levels")));
    R_ModeSymbol =       /* "mode" */
        global->installSEXP(CreateSymbol(global->internStr("mode")));
    R_NameSymbol =       /* "name" */
        global->installSEXP(CreateSymbol(global->internStr("name")));
    R_NamesSymbol =      /* "names" */
        global->installSEXP(CreateSymbol(global->internStr("names")));
    R_NaRmSymbol =       /* "na.rm" */
        global->installSEXP(CreateSymbol(global->internStr("na.rm")));
    R_PackageSymbol =    /* "package" */
        global->installSEXP(CreateSymbol(global->internStr("package")));
    R_QuoteSymbol =      /* "quote" */
        global->installSEXP(CreateSymbol(global->internStr("quote")));
    R_RowNamesSymbol =   /* "row.names" */
        global->installSEXP(CreateSymbol(global->internStr("row.names")));
    R_SeedsSymbol =      /* ".Random.seed" */
        global->installSEXP(CreateSymbol(global->internStr(".Random.seed")));
    R_SourceSymbol =     /* "source" */
        global->installSEXP(CreateSymbol(global->internStr("source")));
    R_TspSymbol =        /* "tsp" */
        global->installSEXP(CreateSymbol(global->internStr("tsp")));

    R_dot_defined =      /* ".defined" */
        global->installSEXP(CreateSymbol(global->internStr(".defined")));
    R_dot_Method =       /* ".Method" */
        global->installSEXP(CreateSymbol(global->internStr(".Method")));
    R_dot_target =       /* ".target" */
        global->installSEXP(CreateSymbol(global->internStr(".target")));

    R_NaString = (SEXP)Strings::NA;        /* NA_STRING as a CHARSXP */
    R_BlankString = (SEXP)Strings::empty;     /* "" as a CHARSXP */

    R_Interactive = TRUE;

}

}

