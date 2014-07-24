// Some of this code is taken from R header files under the LGPL.


typedef void* SEXP;


// Rinternals.h

SEXP  R_GlobalEnv;        /* The "global" environment */

SEXP  R_EmptyEnv;         /* An empty environment at the root of the
                                       environment tree */
SEXP  R_BaseEnv;          /* The base environment; formerly R_NilValue */
SEXP  R_BaseNamespace;    /* The (fake) namespace for base */
SEXP  R_NamespaceRegistry;/* Registry for registered namespaces */

SEXP  R_Srcref;           /* Current srcref, for debuggers */

/* Special Values */
SEXP  R_NilValue;         /* The nil object */
SEXP  R_UnboundValue;     /* Unbound marker */
SEXP  R_MissingArg;       /* Missing argument marker */

SEXP    R_RestartToken;             /* Marker for restarted function calls */

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

SEXP  R_NaString;         /* NA_STRING as a CHARSXP */
SEXP  R_BlankString;      /* "" as a CHARSXP */

void test() {
}
