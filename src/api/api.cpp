// Some of this code is taken from R header files under the LGPL.

#include "api.h"

#include "../frontend.h"

#define R_NO_REMAP
#include <Rinternals.h>


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
    globalState->apiStack = stack;

    Value global;
    REnvironment::Init(global, globalState->global);
    R_GlobalEnv = globalState->installSEXP(global);
    
    Value empty;
    REnvironment::Init(empty, globalState->empty);
    R_EmptyEnv = globalState->installSEXP(empty);

    R_NilValue = globalState->installSEXP(Null::Singleton());
    R_UnboundValue = R_MissingArg =
        globalState->installSEXP(Value::Nil());

    R_TrueValue = globalState->installSEXP(Logical::c(Logical::TrueElement));
    R_FalseValue = globalState->installSEXP(Logical::c(Logical::FalseElement));
    R_LogicalNAValue = globalState->installSEXP(Logical::c(Logical::NAelement));

    R_PackageSymbol =
        globalState->installSEXP(CreateSymbol(globalState->internStr("package")));
    R_Bracket2Symbol =   /* "[[" */
        globalState->installSEXP(CreateSymbol(globalState->internStr("[[")));
    R_BracketSymbol =    /* "[" */
        globalState->installSEXP(CreateSymbol(globalState->internStr("[")));
    R_BraceSymbol =      /* "{" */
        globalState->installSEXP(CreateSymbol(globalState->internStr("{")));
    R_ClassSymbol =      /* "class" */
        globalState->installSEXP(CreateSymbol(globalState->internStr("class")));
    R_DeviceSymbol =     /* ".Device" */
        globalState->installSEXP(CreateSymbol(globalState->internStr(".Device")));
    R_DimNamesSymbol =   /* "dimnames" */
        globalState->installSEXP(CreateSymbol(globalState->internStr("dimnames")));
    R_DimSymbol =        /* "dim" */
        globalState->installSEXP(CreateSymbol(globalState->internStr("dim")));
    R_DollarSymbol =     /* "$" */
        globalState->installSEXP(CreateSymbol(globalState->internStr("$")));
    R_DotsSymbol =       /* "..." */
        globalState->installSEXP(CreateSymbol(globalState->internStr("...")));
    R_DropSymbol =       /* "drop" */
        globalState->installSEXP(CreateSymbol(globalState->internStr("drop")));
    R_LastvalueSymbol =  /* ".Last.value" */
        globalState->installSEXP(CreateSymbol(globalState->internStr(".Last.value")));
    R_LevelsSymbol =     /* "levels" */
        globalState->installSEXP(CreateSymbol(globalState->internStr("levels")));
    R_ModeSymbol =       /* "mode" */
        globalState->installSEXP(CreateSymbol(globalState->internStr("mode")));
    R_NameSymbol =       /* "name" */
        globalState->installSEXP(CreateSymbol(globalState->internStr("name")));
    R_NamesSymbol =      /* "names" */
        globalState->installSEXP(CreateSymbol(globalState->internStr("names")));
    R_NaRmSymbol =       /* "na.rm" */
        globalState->installSEXP(CreateSymbol(globalState->internStr("na.rm")));
    R_PackageSymbol =    /* "package" */
        globalState->installSEXP(CreateSymbol(globalState->internStr("package")));
    R_QuoteSymbol =      /* "quote" */
        globalState->installSEXP(CreateSymbol(globalState->internStr("quote")));
    R_RowNamesSymbol =   /* "row.names" */
        globalState->installSEXP(CreateSymbol(globalState->internStr("row.names")));
    R_SeedsSymbol =      /* ".Random.seed" */
        globalState->installSEXP(CreateSymbol(globalState->internStr(".Random.seed")));
    R_SourceSymbol =     /* "source" */
        globalState->installSEXP(CreateSymbol(globalState->internStr("source")));
    R_TspSymbol =        /* "tsp" */
        globalState->installSEXP(CreateSymbol(globalState->internStr("tsp")));

    R_dot_defined =      /* ".defined" */
        globalState->installSEXP(CreateSymbol(globalState->internStr(".defined")));
    R_dot_Method =       /* ".Method" */
        globalState->installSEXP(CreateSymbol(globalState->internStr(".Method")));
    R_dot_target =       /* ".target" */
        globalState->installSEXP(CreateSymbol(globalState->internStr(".target")));

    R_NaString = (SEXP)Strings::NA;        /* NA_STRING as a CHARSXP */
    R_BlankString = (SEXP)Strings::empty;     /* "" as a CHARSXP */

    R_Interactive = TRUE;

}

}

