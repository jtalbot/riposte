/* Driver template for the LEMON parser generator.
** The author disclaims copyright to this source code.
*/
/* First off, code is included that follows the "include" declaration
** in the input grammar file. */
#include <stdio.h>
#line 48 "parser.y"

	#include <iostream>
	#include "internal.h"
#line 12 "parser.c"
/* Next is all token values, in a form suitable for use by makeheaders.
** This section will be null unless lemon is run with the -m switch.
*/
/* 
** These constants (all generated automatically by the parser generator)
** specify the various kinds of tokens (terminals) that the parser
** understands. 
**
** Each symbol here is a terminal symbol in the grammar.
*/
/* Make sure the INTERFACE macro is defined.
*/
#ifndef INTERFACE
# define INTERFACE 1
#endif
/* The next thing included is series of defines which control
** various aspects of the generated parser.
**    YYCODETYPE         is the data type used for storing terminal
**                       and nonterminal numbers.  "unsigned char" is
**                       used if there are fewer than 250 terminals
**                       and nonterminals.  "int" is used otherwise.
**    YYNOCODE           is a number of type YYCODETYPE which corresponds
**                       to no legal terminal or nonterminal number.  This
**                       number is used to fill in empty slots of the hash 
**                       table.
**    YYFALLBACK         If defined, this indicates that one or more tokens
**                       have fall-back values which should be used if the
**                       original value of the token will not parse.
**    YYACTIONTYPE       is the data type used for storing terminal
**                       and nonterminal numbers.  "unsigned char" is
**                       used if there are fewer than 250 rules and
**                       states combined.  "int" is used otherwise.
**    ParseTOKENTYPE     is the data type used for minor tokens given 
**                       directly to the parser from the tokenizer.
**    YYMINORTYPE        is the data type used for all minor tokens.
**                       This is typically a union of many types, one of
**                       which is ParseTOKENTYPE.  The entry in the union
**                       for base tokens is called "yy0".
**    YYSTACKDEPTH       is the maximum depth of the parser's stack.  If
**                       zero the stack is dynamically sized using realloc()
**    ParseARG_SDECL     A static variable declaration for the %extra_argument
**    ParseARG_PDECL     A parameter declaration for the %extra_argument
**    ParseARG_STORE     Code to store %extra_argument into yypParser
**    ParseARG_FETCH     Code to extract %extra_argument from yypParser
**    YYNSTATE           the combined number of states.
**    YYNRULE            the number of rules in the grammar
**    YYERRORSYMBOL      is the code number of the error symbol.  If not
**                       defined, then do no error processing.
*/
#define YYCODETYPE unsigned char
#define YYNOCODE 65
#define YYACTIONTYPE unsigned short int
#define ParseTOKENTYPE Value
typedef union {
  int yyinit;
  ParseTOKENTYPE yy0;
  Pairs* yy82;
  int yy129;
} YYMINORTYPE;
#ifndef YYSTACKDEPTH
#define YYSTACKDEPTH 1000
#endif
#define ParseARG_SDECL Parser* parser;
#define ParseARG_PDECL ,Parser* parser
#define ParseARG_FETCH Parser* parser = yypParser->parser
#define ParseARG_STORE yypParser->parser = parser
#define YYNSTATE 218
#define YYNRULE 77
#define YYERRORSYMBOL 54
#define YYERRSYMDT yy129
#define YY_NO_ACTION      (YYNSTATE+YYNRULE+2)
#define YY_ACCEPT_ACTION  (YYNSTATE+YYNRULE+1)
#define YY_ERROR_ACTION   (YYNSTATE+YYNRULE)

/* The yyzerominor constant is used to initialize instances of
** YYMINORTYPE objects to zero. */
static const YYMINORTYPE yyzerominor = { 0 };

/* Define the yytestcase() macro to be a no-op if is not already defined
** otherwise.
**
** Applications can choose to define yytestcase() in the %include section
** to a macro that can assist in verifying code coverage.  For production
** code the yytestcase() macro should be turned off.  But it is useful
** for testing.
*/
#ifndef yytestcase
# define yytestcase(X)
#endif


/* Next are the tables used to determine what action to take based on the
** current state and lookahead token.  These tables are used to implement
** functions that take a state number and lookahead value and return an
** action integer.  
**
** Suppose the action integer is N.  Then the action is determined as
** follows
**
**   0 <= N < YYNSTATE                  Shift N.  That is, push the lookahead
**                                      token onto the stack and goto state N.
**
**   YYNSTATE <= N < YYNSTATE+YYNRULE   Reduce by rule N-YYNSTATE.
**
**   N == YYNSTATE+YYNRULE              A syntax error has occurred.
**
**   N == YYNSTATE+YYNRULE+1            The parser accepts its input.
**
**   N == YYNSTATE+YYNRULE+2            No such action.  Denotes unused
**                                      slots in the yy_action[] table.
**
** The action table is constructed as a single large table named yy_action[].
** Given state S and lookahead X, the action is computed as
**
**      yy_action[ yy_shift_ofst[S] + X ]
**
** If the index value yy_shift_ofst[S]+X is out of range or if the value
** yy_lookahead[yy_shift_ofst[S]+X] is not equal to X or if yy_shift_ofst[S]
** is equal to YY_SHIFT_USE_DFLT, it means that the action is not in the table
** and that yy_default[S] should be used instead.  
**
** The formula above is for computing the action when the lookahead is
** a terminal symbol.  If the lookahead is a non-terminal (as occurs after
** a reduce action) then the yy_reduce_ofst[] array is used in place of
** the yy_shift_ofst[] array and YY_REDUCE_USE_DFLT is used in place of
** YY_SHIFT_USE_DFLT.
**
** The following are the tables generated in this section:
**
**  yy_action[]        A single table containing all actions.
**  yy_lookahead[]     A table containing the lookahead for each entry in
**                     yy_action.  Used to detect hash collisions.
**  yy_shift_ofst[]    For each state, the offset into yy_action for
**                     shifting terminals.
**  yy_reduce_ofst[]   For each state, the offset into yy_action for
**                     shifting non-terminals after a reduce.
**  yy_default[]       Default action for each state.
*/
static const YYACTIONTYPE yy_action[] = {
 /*     0 */    88,  217,  130,  274,   61,   69,  190,   99,  193,  100,
 /*    10 */    87,   96,   98,   95,   97,   63,   94,   93,   89,   90,
 /*    20 */    91,   92,   78,   80,   82,   84,   86,   76,   62,   64,
 /*    30 */    85,  104,  105,  200,   66,  101,  103,  102,   86,   76,
 /*    40 */   206,  130,   85,  104,  105,  200,  107,  101,  103,  102,
 /*    50 */   200,  181,   88,  209,   62,   64,  296,    1,  127,   99,
 /*    60 */    71,  100,   87,   96,   98,   95,   97,  108,   94,   93,
 /*    70 */    89,   90,   91,   92,   78,   80,   82,   84,   86,   76,
 /*    80 */   126,  202,   85,  104,  105,   88,  136,  101,  103,  102,
 /*    90 */   201,  130,   99,   67,  100,   87,   96,   98,   95,   97,
 /*   100 */   167,   94,   93,   89,   90,   91,   92,   78,   80,   82,
 /*   110 */    84,   86,   76,  129,  202,   85,  104,  105,  203,  136,
 /*   120 */   101,  103,  102,  136,   99,    8,  100,   87,   96,   98,
 /*   130 */    95,   97,  171,   94,   93,   89,   90,   91,   92,   78,
 /*   140 */    80,   82,   84,   86,   76,  128,  202,   85,  104,  105,
 /*   150 */    38,  136,  101,  103,  102,   87,   96,   98,   95,   97,
 /*   160 */    10,   94,   93,   89,   90,   91,   92,   78,   80,   82,
 /*   170 */    84,   86,   76,  214,  130,   85,  104,  105,    7,  109,
 /*   180 */   101,  103,  102,   96,   98,   95,   97,  123,   94,   93,
 /*   190 */    89,   90,   91,   92,   78,   80,   82,   84,   86,   76,
 /*   200 */   175,  130,   85,  104,  105,   13,   12,  101,  103,  102,
 /*   210 */    95,   97,  177,   94,   93,   89,   90,   91,   92,   78,
 /*   220 */    80,   82,   84,   86,   76,   74,  130,   85,  104,  105,
 /*   230 */   210,  130,  101,  103,  102,   94,   93,   89,   90,   91,
 /*   240 */    92,   78,   80,   82,   84,   86,   76,  204,  130,   85,
 /*   250 */   104,  105,  195,  130,  101,  103,  102,   83,   70,  119,
 /*   260 */   114,  122,  110,   82,   84,   86,   76,   81,  178,   85,
 /*   270 */   104,  105,   79,  179,  101,  103,  102,   23,   76,   77,
 /*   280 */    75,   85,  104,  105,   65,  192,  101,  103,  102,  161,
 /*   290 */   206,  130,   73,  127,  274,   72,  197,  164,  196,  166,
 /*   300 */   211,  130,  216,  130,    6,  215,  213,   83,   70,  119,
 /*   310 */   114,  122,  110,   85,  104,  105,   19,   81,  101,  103,
 /*   320 */   102,  182,   79,   21,   14,  287,   15,  287,  287,   77,
 /*   330 */    75,  200,  283,  287,  283,  283,  200,  169,  200,   41,
 /*   340 */   283,  274,   73,   37,  198,   72,  197,   57,   60,   58,
 /*   350 */   127,  165,   11,  163,  168,  215,  213,  284,  173,  284,
 /*   360 */   284,  180,  183,  200,   25,  284,   61,   69,  189,  139,
 /*   370 */    18,  150,  131,  151,  297,  152,  153,  145,  140,  146,
 /*   380 */   138,  147,  143,  200,  144,  148,  133,   17,  160,  159,
 /*   390 */   132,  154,  149,  134,  297,  155,  142,  157,  137,  156,
 /*   400 */   162,  135,  158,  141,   51,   44,  186,  188,  185,  187,
 /*   410 */   113,  218,  219,  199,  207,  205,  176,  174,  184,  125,
 /*   420 */   297,  170,  191,  194,   16,    4,    9,  172,   39,   20,
 /*   430 */    40,   42,   43,   45,   46,  297,   47,   48,   49,   50,
 /*   440 */    52,   53,   22,   24,   26,   27,   28,   29,   31,   33,
 /*   450 */    35,   36,   34,   32,   30,  106,    3,    5,    2,   68,
 /*   460 */    59,  111,  115,  212,  120,  124,  116,  112,  208,   54,
 /*   470 */    55,   56,  297,  117,  118,  121,
};
static const YYCODETYPE yy_lookahead[] = {
 /*     0 */     1,   61,   62,    9,   34,   35,   41,    8,   43,   10,
 /*    10 */    11,   12,   13,   14,   15,   60,   17,   18,   19,   20,
 /*    20 */    21,   22,   23,   24,   25,   26,   27,   28,   34,   35,
 /*    30 */    31,   32,   33,   51,   55,   36,   37,   38,   27,   28,
 /*    40 */    61,   62,   31,   32,   33,   51,   60,   36,   37,   38,
 /*    50 */    51,   54,    1,   48,   34,   35,   59,   60,   53,    8,
 /*    60 */     9,   10,   11,   12,   13,   14,   15,   58,   17,   18,
 /*    70 */    19,   20,   21,   22,   23,   24,   25,   26,   27,   28,
 /*    80 */    56,   57,   31,   32,   33,    1,   62,   36,   37,   38,
 /*    90 */    61,   62,    8,   43,   10,   11,   12,   13,   14,   15,
 /*   100 */    60,   17,   18,   19,   20,   21,   22,   23,   24,   25,
 /*   110 */    26,   27,   28,   56,   57,   31,   32,   33,   57,   62,
 /*   120 */    36,   37,   38,   62,    8,   60,   10,   11,   12,   13,
 /*   130 */    14,   15,   60,   17,   18,   19,   20,   21,   22,   23,
 /*   140 */    24,   25,   26,   27,   28,   56,   57,   31,   32,   33,
 /*   150 */    60,   62,   36,   37,   38,   11,   12,   13,   14,   15,
 /*   160 */    60,   17,   18,   19,   20,   21,   22,   23,   24,   25,
 /*   170 */    26,   27,   28,   61,   62,   31,   32,   33,   60,   45,
 /*   180 */    36,   37,   38,   12,   13,   14,   15,   53,   17,   18,
 /*   190 */    19,   20,   21,   22,   23,   24,   25,   26,   27,   28,
 /*   200 */    61,   62,   31,   32,   33,   51,   52,   36,   37,   38,
 /*   210 */    14,   15,   60,   17,   18,   19,   20,   21,   22,   23,
 /*   220 */    24,   25,   26,   27,   28,   61,   62,   31,   32,   33,
 /*   230 */    61,   62,   36,   37,   38,   17,   18,   19,   20,   21,
 /*   240 */    22,   23,   24,   25,   26,   27,   28,   61,   62,   31,
 /*   250 */    32,   33,   61,   62,   36,   37,   38,    1,    2,    3,
 /*   260 */     4,    5,    6,   25,   26,   27,   28,   11,   60,   31,
 /*   270 */    32,   33,   16,   60,   36,   37,   38,   60,   28,   23,
 /*   280 */    24,   31,   32,   33,   55,   45,   36,   37,   38,   62,
 /*   290 */    61,   62,   36,   53,    9,   39,   40,   41,   42,   43,
 /*   300 */    61,   62,   61,   62,   60,   49,   50,    1,    2,    3,
 /*   310 */     4,    5,    6,   31,   32,   33,   62,   11,   36,   37,
 /*   320 */    38,   60,   16,   60,   60,   45,   60,   47,   48,   23,
 /*   330 */    24,   51,   45,   53,   47,   48,   51,   60,   51,   60,
 /*   340 */    53,    9,   36,   60,   47,   39,   40,   41,   42,   43,
 /*   350 */    53,   60,   60,   60,   60,   49,   50,   45,   60,   47,
 /*   360 */    48,   60,   60,   51,   60,   53,   34,   35,   63,   62,
 /*   370 */    62,   62,   62,   62,   64,   62,   62,   62,   62,   62,
 /*   380 */    62,   62,   62,   51,   62,   62,   62,   62,   62,   62,
 /*   390 */    62,   62,   62,   62,   64,   62,   62,   62,   62,   62,
 /*   400 */    62,   62,   62,   62,   60,   60,   60,   60,   60,   60,
 /*   410 */     7,    0,    0,   63,   63,   63,   60,   60,   60,    9,
 /*   420 */    64,   60,   63,   63,   60,   60,   60,   60,   60,   60,
 /*   430 */    60,   60,   60,   60,   60,   64,   60,   60,   60,   60,
 /*   440 */    60,   60,   60,   60,   60,   60,   60,   60,   60,   60,
 /*   450 */    60,   60,   60,   60,   60,   36,   60,   60,   60,   60,
 /*   460 */    43,   36,   36,   45,   36,    9,   43,   45,   44,    9,
 /*   470 */     9,    9,   64,   46,   45,   45,
};
#define YY_SHIFT_USE_DFLT (-36)
#define YY_SHIFT_MAX 188
static const short yy_shift_ofst[] = {
 /*     0 */   -18,  256,  306,  306,  256,  306,  256,  256,  256,  256,
 /*    10 */   256,  306,  256,  256,  256,  256,  256,   -1,   -1,   -1,
 /*    20 */   256,  256,  256,  256,  256,  256,  256,  256,  256,  256,
 /*    30 */   256,  256,  256,  256,  256,  256,  256,  256,  256,  256,
 /*    40 */   256,  256,  256,  256,  256,  256,  256,  256,  256,  256,
 /*    50 */   256,  256,  256,  256,  312,  280,  287,  332,   -6,  285,
 /*    60 */   285,  -35,  -35,  -35,  -35,  154,  154,  285,  -35,  -35,
 /*    70 */   -18,  -18,  -18,  -18,  -18,  -18,  -18,  -18,  -18,  -18,
 /*    80 */   -18,  -18,  -18,  -18,  -18,  -18,  -18,  -18,  -18,  -18,
 /*    90 */   -18,  -18,  -18,  -18,  -18,  -18,  -18,  -18,  -18,  -18,
 /*   100 */   -18,  -18,  -18,  -18,  -18,  -18,  -18,   50,  -18,  -18,
 /*   110 */   -18,  -18,  -18,  -18,  -18,  -18,  -18,  -18,  -18,  -18,
 /*   120 */   -18,  -18,  -18,  -18,  -18,  -18,  -18,  -18,  -18,  -18,
 /*   130 */    51,   84,   84,   84,   84,   84,   84,  116,  116,  116,
 /*   140 */   144,  171,  171,  196,  196,  218,  218,  218,  218,  218,
 /*   150 */   218,  218,  218,  218,  238,  238,   11,   11,  250,  282,
 /*   160 */   282,  282,  282,  297,  -30,    5,   20,  134,  240,  417,
 /*   170 */   419,  425,  418,  422,  424,  403,  411,  426,  423,  427,
 /*   180 */   429,  412,  428,  430,  410,  456,  460,  461,  462,
};
#define YY_REDUCE_USE_DFLT (-61)
#define YY_REDUCE_MAX 129
static const short yy_reduce_ofst[] = {
 /*     0 */    -3,  229,   24,   57,  -21,   89,  191,  186,  169,  164,
 /*    10 */   139,   61,  112,   29,  -60,  241,  239,  298,  301,  302,
 /*    20 */   227,  254,  307,  308,  309,  310,  311,  313,  314,  315,
 /*    30 */   316,  317,  318,  319,  320,  322,  323,  324,  325,  326,
 /*    40 */   327,  328,  329,  330,  331,  333,  334,  335,  336,  337,
 /*    50 */   338,  339,  340,  341,  304,  344,  345,  346,  347,  348,
 /*    60 */   349,  305,  350,  351,  352,  356,  357,  358,  359,  360,
 /*    70 */   361,  364,  365,  366,  367,  368,  369,  370,  371,  372,
 /*    80 */   373,  374,  376,  377,  378,  379,  380,  381,  382,  383,
 /*    90 */   384,  385,  386,  387,  388,  389,  390,  391,  392,  393,
 /*   100 */   394,  396,  397,  398,  399,  -45,  -14,    9,   40,   65,
 /*   110 */    72,   90,  100,  118,  152,  208,  213,  217,  244,  261,
 /*   120 */   263,  264,  266,  277,  279,  283,  291,  292,  293,  294,
};
static const YYACTIONTYPE yy_default[] = {
 /*     0 */   274,  279,  282,  282,  279,  282,  295,  295,  295,  295,
 /*    10 */   295,  282,  276,  273,  295,  295,  295,  274,  274,  274,
 /*    20 */   295,  295,  295,  295,  295,  295,  295,  295,  295,  295,
 /*    30 */   295,  295,  295,  295,  295,  295,  295,  295,  295,  295,
 /*    40 */   295,  295,  295,  295,  295,  295,  295,  295,  295,  295,
 /*    50 */   295,  295,  295,  295,  274,  274,  274,  223,  225,  293,
 /*    60 */   224,  295,  295,  295,  295,  274,  274,  291,  295,  295,
 /*    70 */   274,  274,  274,  274,  274,  274,  274,  274,  274,  274,
 /*    80 */   274,  274,  274,  274,  274,  274,  274,  274,  274,  274,
 /*    90 */   274,  274,  274,  274,  274,  274,  274,  274,  274,  274,
 /*   100 */   274,  274,  274,  274,  274,  274,  274,  290,  274,  274,
 /*   110 */   274,  274,  274,  274,  274,  274,  274,  274,  274,  274,
 /*   120 */   274,  274,  274,  274,  274,  274,  274,  274,  274,  274,
 /*   130 */   221,  286,  294,  292,  285,  288,  289,  232,  252,  241,
 /*   140 */   253,  240,  231,  251,  249,  246,  247,  248,  250,  230,
 /*   150 */   242,  243,  244,  245,  234,  235,  237,  236,  239,  229,
 /*   160 */   228,  233,  238,  295,  223,  295,  225,  295,  295,  295,
 /*   170 */   295,  295,  295,  295,  295,  256,  295,  295,  295,  295,
 /*   180 */   295,  295,  295,  295,  295,  295,  295,  295,  295,  264,
 /*   190 */   271,  267,  255,  272,  266,  258,  224,  222,  261,  263,
 /*   200 */   273,  277,  280,  281,  257,  265,  278,  268,  226,  262,
 /*   210 */   254,  220,  227,  270,  275,  269,  260,  259,
};
#define YY_SZ_ACTTAB (int)(sizeof(yy_action)/sizeof(yy_action[0]))

/* The next table maps tokens into fallback tokens.  If a construct
** like the following:
** 
**      %fallback ID X Y Z.
**
** appears in the grammar, then ID becomes a fallback token for X, Y,
** and Z.  Whenever one of the tokens X, Y, or Z is input to the parser
** but it does not parse, the type of the token is changed to ID and
** the parse is retried before an error is thrown.
*/
#ifdef YYFALLBACK
static const YYCODETYPE yyFallback[] = {
};
#endif /* YYFALLBACK */

/* The following structure represents a single element of the
** parser's stack.  Information stored includes:
**
**   +  The state number for the parser at this level of the stack.
**
**   +  The value of the token stored at this level of the stack.
**      (In other words, the "major" token.)
**
**   +  The semantic value stored at this level of the stack.  This is
**      the information used by the action routines in the grammar.
**      It is sometimes called the "minor" token.
*/
struct yyStackEntry {
  YYACTIONTYPE stateno;  /* The state-number */
  YYCODETYPE major;      /* The major token value.  This is the code
                         ** number for the token at this stack level */
  YYMINORTYPE minor;     /* The user-supplied minor token value.  This
                         ** is the value of the token  */
};
typedef struct yyStackEntry yyStackEntry;

/* The state of the parser is completely contained in an instance of
** the following structure */
struct yyParser {
  int yyidx;                    /* Index of top element in stack */
#ifdef YYTRACKMAXSTACKDEPTH
  int yyidxMax;                 /* Maximum value of yyidx */
#endif
  int yyerrcnt;                 /* Shifts left before out of the error */
  ParseARG_SDECL                /* A place to hold %extra_argument */
#if YYSTACKDEPTH<=0
  int yystksz;                  /* Current side of the stack */
  yyStackEntry *yystack;        /* The parser's stack */
#else
  yyStackEntry yystack[YYSTACKDEPTH];  /* The parser's stack */
#endif
};
typedef struct yyParser yyParser;

#ifndef NDEBUG
#include <stdio.h>
static FILE *yyTraceFILE = 0;
static char *yyTracePrompt = 0;
#endif /* NDEBUG */

#ifndef NDEBUG
/* 
** Turn parser tracing on by giving a stream to which to write the trace
** and a prompt to preface each trace message.  Tracing is turned off
** by making either argument NULL 
**
** Inputs:
** <ul>
** <li> A FILE* to which trace output should be written.
**      If NULL, then tracing is turned off.
** <li> A prefix string written at the beginning of every
**      line of trace output.  If NULL, then tracing is
**      turned off.
** </ul>
**
** Outputs:
** None.
*/
void ParseTrace(FILE *TraceFILE, char *zTracePrompt){
  yyTraceFILE = TraceFILE;
  yyTracePrompt = zTracePrompt;
  if( yyTraceFILE==0 ) yyTracePrompt = 0;
  else if( yyTracePrompt==0 ) yyTraceFILE = 0;
}
#endif /* NDEBUG */

#ifndef NDEBUG
/* For tracing shifts, the names of all terminals and nonterminals
** are required.  The following table supplies these names */
static const char *const yyTokenName[] = { 
  "$",             "QUESTION",      "FUNCTION",      "WHILE",       
  "FOR",           "REPEAT",        "IF",            "ELSE",        
  "LEFT_ASSIGN",   "EQ_ASSIGN",     "RIGHT_ASSIGN",  "TILDE",       
  "OR",            "OR2",           "AND",           "AND2",        
  "NOT",           "GT",            "GE",            "LT",          
  "LE",            "EQ",            "NE",            "PLUS",        
  "MINUS",         "TIMES",         "DIVIDE",        "SPECIALOP",   
  "COLON",         "UMINUS",        "UPLUS",         "POW",         
  "DOLLAR",        "AT",            "NS_GET",        "NS_GET_INT",  
  "LPAREN",        "LBRACKET",      "LBB",           "LBRACE",      
  "NUM_CONST",     "STR_CONST",     "NULL_CONST",    "SYMBOL",      
  "RBRACE",        "RPAREN",        "IN",            "RBB",         
  "RBRACKET",      "NEXT",          "BREAK",         "NEWLINE",     
  "SEMICOLON",     "COMMA",         "error",         "statementlist",
  "sublist",       "sub",           "formallist",    "prog",        
  "optnl",         "statement",     "expr",          "symbolstr",   
};
#endif /* NDEBUG */

#ifndef NDEBUG
/* For tracing reduce actions, the names of all rules are required.
*/
static const char *const yyRuleName[] = {
 /*   0 */ "prog ::= optnl statementlist optnl",
 /*   1 */ "prog ::= error",
 /*   2 */ "statement ::= expr EQ_ASSIGN optnl statement",
 /*   3 */ "statement ::= expr",
 /*   4 */ "expr ::= NUM_CONST",
 /*   5 */ "expr ::= STR_CONST",
 /*   6 */ "expr ::= NULL_CONST",
 /*   7 */ "expr ::= SYMBOL",
 /*   8 */ "expr ::= LBRACE optnl statementlist optnl RBRACE",
 /*   9 */ "expr ::= LPAREN optnl statement optnl RPAREN",
 /*  10 */ "expr ::= MINUS optnl expr",
 /*  11 */ "expr ::= PLUS optnl expr",
 /*  12 */ "expr ::= NOT optnl expr",
 /*  13 */ "expr ::= TILDE optnl expr",
 /*  14 */ "expr ::= QUESTION optnl expr",
 /*  15 */ "expr ::= expr COLON optnl expr",
 /*  16 */ "expr ::= expr PLUS optnl expr",
 /*  17 */ "expr ::= expr MINUS optnl expr",
 /*  18 */ "expr ::= expr TIMES optnl expr",
 /*  19 */ "expr ::= expr DIVIDE optnl expr",
 /*  20 */ "expr ::= expr POW optnl expr",
 /*  21 */ "expr ::= expr SPECIALOP optnl expr",
 /*  22 */ "expr ::= expr TILDE optnl expr",
 /*  23 */ "expr ::= expr QUESTION optnl expr",
 /*  24 */ "expr ::= expr LT optnl expr",
 /*  25 */ "expr ::= expr LE optnl expr",
 /*  26 */ "expr ::= expr EQ optnl expr",
 /*  27 */ "expr ::= expr NE optnl expr",
 /*  28 */ "expr ::= expr GE optnl expr",
 /*  29 */ "expr ::= expr GT optnl expr",
 /*  30 */ "expr ::= expr AND optnl expr",
 /*  31 */ "expr ::= expr OR optnl expr",
 /*  32 */ "expr ::= expr AND2 optnl expr",
 /*  33 */ "expr ::= expr OR2 optnl expr",
 /*  34 */ "expr ::= expr LEFT_ASSIGN optnl expr",
 /*  35 */ "expr ::= expr RIGHT_ASSIGN optnl expr",
 /*  36 */ "expr ::= FUNCTION optnl LPAREN optnl formallist optnl RPAREN optnl statement",
 /*  37 */ "expr ::= expr LPAREN optnl sublist optnl RPAREN",
 /*  38 */ "expr ::= IF optnl LPAREN optnl expr optnl RPAREN optnl statement",
 /*  39 */ "expr ::= IF optnl LPAREN optnl expr optnl RPAREN optnl statement ELSE optnl statement",
 /*  40 */ "expr ::= FOR optnl LPAREN optnl SYMBOL optnl IN optnl expr optnl RPAREN optnl statement",
 /*  41 */ "expr ::= WHILE optnl LPAREN optnl expr optnl RPAREN optnl statement",
 /*  42 */ "expr ::= REPEAT optnl statement",
 /*  43 */ "expr ::= expr LBB optnl sublist optnl RBB",
 /*  44 */ "expr ::= expr LBRACKET optnl sublist optnl RBRACKET",
 /*  45 */ "expr ::= SYMBOL NS_GET symbolstr",
 /*  46 */ "expr ::= STR_CONST NS_GET symbolstr",
 /*  47 */ "expr ::= SYMBOL NS_GET_INT symbolstr",
 /*  48 */ "expr ::= STR_CONST NS_GET_INT symbolstr",
 /*  49 */ "expr ::= expr DOLLAR optnl symbolstr",
 /*  50 */ "expr ::= expr AT optnl symbolstr",
 /*  51 */ "expr ::= NEXT",
 /*  52 */ "expr ::= BREAK",
 /*  53 */ "symbolstr ::= STR_CONST",
 /*  54 */ "symbolstr ::= SYMBOL",
 /*  55 */ "optnl ::= NEWLINE",
 /*  56 */ "optnl ::=",
 /*  57 */ "statementlist ::= statementlist SEMICOLON statement",
 /*  58 */ "statementlist ::= statementlist SEMICOLON",
 /*  59 */ "statementlist ::= statementlist NEWLINE statement",
 /*  60 */ "statementlist ::= statement",
 /*  61 */ "statementlist ::=",
 /*  62 */ "sublist ::= sub",
 /*  63 */ "sublist ::= sublist optnl COMMA optnl sub",
 /*  64 */ "sub ::=",
 /*  65 */ "sub ::= SYMBOL optnl EQ_ASSIGN",
 /*  66 */ "sub ::= STR_CONST optnl EQ_ASSIGN",
 /*  67 */ "sub ::= SYMBOL optnl EQ_ASSIGN optnl expr",
 /*  68 */ "sub ::= STR_CONST optnl EQ_ASSIGN optnl expr",
 /*  69 */ "sub ::= NULL_CONST optnl EQ_ASSIGN",
 /*  70 */ "sub ::= NULL_CONST optnl EQ_ASSIGN optnl expr",
 /*  71 */ "sub ::= expr",
 /*  72 */ "formallist ::=",
 /*  73 */ "formallist ::= SYMBOL",
 /*  74 */ "formallist ::= SYMBOL optnl EQ_ASSIGN optnl expr",
 /*  75 */ "formallist ::= formallist optnl COMMA optnl SYMBOL",
 /*  76 */ "formallist ::= formallist optnl COMMA optnl SYMBOL optnl EQ_ASSIGN optnl expr",
};
#endif /* NDEBUG */


#if YYSTACKDEPTH<=0
/*
** Try to increase the size of the parser stack.
*/
static void yyGrowStack(yyParser *p){
  int newSize;
  yyStackEntry *pNew;

  newSize = p->yystksz*2 + 100;
  pNew = realloc(p->yystack, newSize*sizeof(pNew[0]));
  if( pNew ){
    p->yystack = pNew;
    p->yystksz = newSize;
#ifndef NDEBUG
    if( yyTraceFILE ){
      fprintf(yyTraceFILE,"%sStack grows to %d entries!\n",
              yyTracePrompt, p->yystksz);
    }
#endif
  }
}
#endif

/* 
** This function allocates a new parser.
** The only argument is a pointer to a function which works like
** malloc.
**
** Inputs:
** A pointer to the function used to allocate memory.
**
** Outputs:
** A pointer to a parser.  This pointer is used in subsequent calls
** to Parse and ParseFree.
*/
void *ParseAlloc(void *(*mallocProc)(size_t)){
  yyParser *pParser;
  pParser = (yyParser*)(*mallocProc)( (size_t)sizeof(yyParser) );
  if( pParser ){
    pParser->yyidx = -1;
#ifdef YYTRACKMAXSTACKDEPTH
    pParser->yyidxMax = 0;
#endif
#if YYSTACKDEPTH<=0
    pParser->yystack = NULL;
    pParser->yystksz = 0;
    yyGrowStack(pParser);
#endif
  }
  return pParser;
}

/* The following function deletes the value associated with a
** symbol.  The symbol can be either a terminal or nonterminal.
** "yymajor" is the symbol code, and "yypminor" is a pointer to
** the value.
*/
static void yy_destructor(
  yyParser *yypParser,    /* The parser */
  YYCODETYPE yymajor,     /* Type code for object to destroy */
  YYMINORTYPE *yypminor   /* The object to be destroyed */
){
  ParseARG_FETCH;
  switch( yymajor ){
    /* Here is inserted the actions which take place when a
    ** terminal or non-terminal is destroyed.  This can happen
    ** when the symbol is popped from the stack during a
    ** reduce or during error processing or when a parser is 
    ** being destroyed before it is finished parsing.
    **
    ** Note: during a reduce, the only symbols destroyed are those
    ** which appear on the RHS of the rule, but which are not used
    ** inside the C code.
    */
      /* TERMINAL Destructor */
    case 1: /* QUESTION */
    case 2: /* FUNCTION */
    case 3: /* WHILE */
    case 4: /* FOR */
    case 5: /* REPEAT */
    case 6: /* IF */
    case 7: /* ELSE */
    case 8: /* LEFT_ASSIGN */
    case 9: /* EQ_ASSIGN */
    case 10: /* RIGHT_ASSIGN */
    case 11: /* TILDE */
    case 12: /* OR */
    case 13: /* OR2 */
    case 14: /* AND */
    case 15: /* AND2 */
    case 16: /* NOT */
    case 17: /* GT */
    case 18: /* GE */
    case 19: /* LT */
    case 20: /* LE */
    case 21: /* EQ */
    case 22: /* NE */
    case 23: /* PLUS */
    case 24: /* MINUS */
    case 25: /* TIMES */
    case 26: /* DIVIDE */
    case 27: /* SPECIALOP */
    case 28: /* COLON */
    case 29: /* UMINUS */
    case 30: /* UPLUS */
    case 31: /* POW */
    case 32: /* DOLLAR */
    case 33: /* AT */
    case 34: /* NS_GET */
    case 35: /* NS_GET_INT */
    case 36: /* LPAREN */
    case 37: /* LBRACKET */
    case 38: /* LBB */
    case 39: /* LBRACE */
    case 40: /* NUM_CONST */
    case 41: /* STR_CONST */
    case 42: /* NULL_CONST */
    case 43: /* SYMBOL */
    case 44: /* RBRACE */
    case 45: /* RPAREN */
    case 46: /* IN */
    case 47: /* RBB */
    case 48: /* RBRACKET */
    case 49: /* NEXT */
    case 50: /* BREAK */
    case 51: /* NEWLINE */
    case 52: /* SEMICOLON */
    case 53: /* COMMA */
{
#line 16 "parser.y"
(void)parser;
#line 642 "parser.c"
}
      break;
    case 55: /* statementlist */
    case 56: /* sublist */
    case 57: /* sub */
    case 58: /* formallist */
{
#line 19 "parser.y"
delete (yypminor->yy82);
#line 652 "parser.c"
}
      break;
    default:  break;   /* If no destructor action specified: do nothing */
  }
}

/*
** Pop the parser's stack once.
**
** If there is a destructor routine associated with the token which
** is popped from the stack, then call it.
**
** Return the major token number for the symbol popped.
*/
static int yy_pop_parser_stack(yyParser *pParser){
  YYCODETYPE yymajor;
  yyStackEntry *yytos = &pParser->yystack[pParser->yyidx];

  if( pParser->yyidx<0 ) return 0;
#ifndef NDEBUG
  if( yyTraceFILE && pParser->yyidx>=0 ){
    fprintf(yyTraceFILE,"%sPopping %s\n",
      yyTracePrompt,
      yyTokenName[yytos->major]);
  }
#endif
  yymajor = yytos->major;
  yy_destructor(pParser, yymajor, &yytos->minor);
  pParser->yyidx--;
  return yymajor;
}

/* 
** Deallocate and destroy a parser.  Destructors are all called for
** all stack elements before shutting the parser down.
**
** Inputs:
** <ul>
** <li>  A pointer to the parser.  This should be a pointer
**       obtained from ParseAlloc.
** <li>  A pointer to a function used to reclaim memory obtained
**       from malloc.
** </ul>
*/
void ParseFree(
  void *p,                    /* The parser to be deleted */
  void (*freeProc)(void*)     /* Function used to reclaim memory */
){
  yyParser *pParser = (yyParser*)p;
  if( pParser==0 ) return;
  while( pParser->yyidx>=0 ) yy_pop_parser_stack(pParser);
#if YYSTACKDEPTH<=0
  free(pParser->yystack);
#endif
  (*freeProc)((void*)pParser);
}

/*
** Return the peak depth of the stack for a parser.
*/
#ifdef YYTRACKMAXSTACKDEPTH
int ParseStackPeak(void *p){
  yyParser *pParser = (yyParser*)p;
  return pParser->yyidxMax;
}
#endif

/*
** Find the appropriate action for a parser given the terminal
** look-ahead token iLookAhead.
**
** If the look-ahead token is YYNOCODE, then check to see if the action is
** independent of the look-ahead.  If it is, return the action, otherwise
** return YY_NO_ACTION.
*/
static int yy_find_shift_action(
  yyParser *pParser,        /* The parser */
  YYCODETYPE iLookAhead     /* The look-ahead token */
){
  int i;
  int stateno = pParser->yystack[pParser->yyidx].stateno;
 
  if( stateno>YY_SHIFT_MAX || (i = yy_shift_ofst[stateno])==YY_SHIFT_USE_DFLT ){
    return yy_default[stateno];
  }
  assert( iLookAhead!=YYNOCODE );
  i += iLookAhead;
  if( i<0 || i>=YY_SZ_ACTTAB || yy_lookahead[i]!=iLookAhead ){
    if( iLookAhead>0 ){
#ifdef YYFALLBACK
      YYCODETYPE iFallback;            /* Fallback token */
      if( iLookAhead<sizeof(yyFallback)/sizeof(yyFallback[0])
             && (iFallback = yyFallback[iLookAhead])!=0 ){
#ifndef NDEBUG
        if( yyTraceFILE ){
          fprintf(yyTraceFILE, "%sFALLBACK %s => %s\n",
             yyTracePrompt, yyTokenName[iLookAhead], yyTokenName[iFallback]);
        }
#endif
        return yy_find_shift_action(pParser, iFallback);
      }
#endif
#ifdef YYWILDCARD
      {
        int j = i - iLookAhead + YYWILDCARD;
        if( j>=0 && j<YY_SZ_ACTTAB && yy_lookahead[j]==YYWILDCARD ){
#ifndef NDEBUG
          if( yyTraceFILE ){
            fprintf(yyTraceFILE, "%sWILDCARD %s => %s\n",
               yyTracePrompt, yyTokenName[iLookAhead], yyTokenName[YYWILDCARD]);
          }
#endif /* NDEBUG */
          return yy_action[j];
        }
      }
#endif /* YYWILDCARD */
    }
    return yy_default[stateno];
  }else{
    return yy_action[i];
  }
}

/*
** Find the appropriate action for a parser given the non-terminal
** look-ahead token iLookAhead.
**
** If the look-ahead token is YYNOCODE, then check to see if the action is
** independent of the look-ahead.  If it is, return the action, otherwise
** return YY_NO_ACTION.
*/
static int yy_find_reduce_action(
  int stateno,              /* Current state number */
  YYCODETYPE iLookAhead     /* The look-ahead token */
){
  int i;
#ifdef YYERRORSYMBOL
  if( stateno>YY_REDUCE_MAX ){
    return yy_default[stateno];
  }
#else
  assert( stateno<=YY_REDUCE_MAX );
#endif
  i = yy_reduce_ofst[stateno];
  assert( i!=YY_REDUCE_USE_DFLT );
  assert( iLookAhead!=YYNOCODE );
  i += iLookAhead;
#ifdef YYERRORSYMBOL
  if( i<0 || i>=YY_SZ_ACTTAB || yy_lookahead[i]!=iLookAhead ){
    return yy_default[stateno];
  }
#else
  assert( i>=0 && i<YY_SZ_ACTTAB );
  assert( yy_lookahead[i]==iLookAhead );
#endif
  return yy_action[i];
}

/*
** The following routine is called if the stack overflows.
*/
static void yyStackOverflow(yyParser *yypParser, YYMINORTYPE *yypMinor){
   ParseARG_FETCH;
   yypParser->yyidx--;
#ifndef NDEBUG
   if( yyTraceFILE ){
     fprintf(yyTraceFILE,"%sStack Overflow!\n",yyTracePrompt);
   }
#endif
   while( yypParser->yyidx>=0 ) yy_pop_parser_stack(yypParser);
   /* Here code is inserted which will execute if the parser
   ** stack every overflows */
#line 61 "parser.y"

	parser->errors++;
	fprintf(stderr,"Giving up.  Parser stack overflow\n");
#line 829 "parser.c"
   ParseARG_STORE; /* Suppress warning about unused %extra_argument var */
}

/*
** Perform a shift action.
*/
static void yy_shift(
  yyParser *yypParser,          /* The parser to be shifted */
  int yyNewState,               /* The new state to shift in */
  int yyMajor,                  /* The major token to shift in */
  YYMINORTYPE *yypMinor         /* Pointer to the minor token to shift in */
){
  yyStackEntry *yytos;
  yypParser->yyidx++;
#ifdef YYTRACKMAXSTACKDEPTH
  if( yypParser->yyidx>yypParser->yyidxMax ){
    yypParser->yyidxMax = yypParser->yyidx;
  }
#endif
#if YYSTACKDEPTH>0 
  if( yypParser->yyidx>=YYSTACKDEPTH ){
    yyStackOverflow(yypParser, yypMinor);
    return;
  }
#else
  if( yypParser->yyidx>=yypParser->yystksz ){
    yyGrowStack(yypParser);
    if( yypParser->yyidx>=yypParser->yystksz ){
      yyStackOverflow(yypParser, yypMinor);
      return;
    }
  }
#endif
  yytos = &yypParser->yystack[yypParser->yyidx];
  yytos->stateno = (YYACTIONTYPE)yyNewState;
  yytos->major = (YYCODETYPE)yyMajor;
  yytos->minor = *yypMinor;
#ifndef NDEBUG
  if( yyTraceFILE && yypParser->yyidx>0 ){
    int i;
    fprintf(yyTraceFILE,"%sShift %d\n",yyTracePrompt,yyNewState);
    fprintf(yyTraceFILE,"%sStack:",yyTracePrompt);
    for(i=1; i<=yypParser->yyidx; i++)
      fprintf(yyTraceFILE," %s",yyTokenName[yypParser->yystack[i].major]);
    fprintf(yyTraceFILE,"\n");
  }
#endif
}

/* The following table contains information about every rule that
** is used during the reduce.
*/
static const struct {
  YYCODETYPE lhs;         /* Symbol on the left-hand side of the rule */
  unsigned char nrhs;     /* Number of right-hand side symbols in the rule */
} yyRuleInfo[] = {
  { 59, 3 },
  { 59, 1 },
  { 61, 4 },
  { 61, 1 },
  { 62, 1 },
  { 62, 1 },
  { 62, 1 },
  { 62, 1 },
  { 62, 5 },
  { 62, 5 },
  { 62, 3 },
  { 62, 3 },
  { 62, 3 },
  { 62, 3 },
  { 62, 3 },
  { 62, 4 },
  { 62, 4 },
  { 62, 4 },
  { 62, 4 },
  { 62, 4 },
  { 62, 4 },
  { 62, 4 },
  { 62, 4 },
  { 62, 4 },
  { 62, 4 },
  { 62, 4 },
  { 62, 4 },
  { 62, 4 },
  { 62, 4 },
  { 62, 4 },
  { 62, 4 },
  { 62, 4 },
  { 62, 4 },
  { 62, 4 },
  { 62, 4 },
  { 62, 4 },
  { 62, 9 },
  { 62, 6 },
  { 62, 9 },
  { 62, 12 },
  { 62, 13 },
  { 62, 9 },
  { 62, 3 },
  { 62, 6 },
  { 62, 6 },
  { 62, 3 },
  { 62, 3 },
  { 62, 3 },
  { 62, 3 },
  { 62, 4 },
  { 62, 4 },
  { 62, 1 },
  { 62, 1 },
  { 63, 1 },
  { 63, 1 },
  { 60, 1 },
  { 60, 0 },
  { 55, 3 },
  { 55, 2 },
  { 55, 3 },
  { 55, 1 },
  { 55, 0 },
  { 56, 1 },
  { 56, 5 },
  { 57, 0 },
  { 57, 3 },
  { 57, 3 },
  { 57, 5 },
  { 57, 5 },
  { 57, 3 },
  { 57, 5 },
  { 57, 1 },
  { 58, 0 },
  { 58, 1 },
  { 58, 5 },
  { 58, 5 },
  { 58, 9 },
};

static void yy_accept(yyParser*);  /* Forward Declaration */

/*
** Perform a reduce action and the shift that must immediately
** follow the reduce.
*/
static void yy_reduce(
  yyParser *yypParser,         /* The parser */
  int yyruleno                 /* Number of the rule by which to reduce */
){
  int yygoto;                     /* The next state */
  int yyact;                      /* The next action */
  YYMINORTYPE yygotominor;        /* The LHS of the rule reduced */
  yyStackEntry *yymsp;            /* The top of the parser's stack */
  int yysize;                     /* Amount to pop the stack */
  ParseARG_FETCH;
  yymsp = &yypParser->yystack[yypParser->yyidx];
#ifndef NDEBUG
  if( yyTraceFILE && yyruleno>=0 
        && yyruleno<(int)(sizeof(yyRuleName)/sizeof(yyRuleName[0])) ){
    fprintf(yyTraceFILE, "%sReduce [%s].\n", yyTracePrompt,
      yyRuleName[yyruleno]);
  }
#endif /* NDEBUG */

  /* Silence complaints from purify about yygotominor being uninitialized
  ** in some cases when it is copied into the stack after the following
  ** switch.  yygotominor is uninitialized when a rule reduces that does
  ** not set the value of its left-hand side nonterminal.  Leaving the
  ** value of the nonterminal uninitialized is utterly harmless as long
  ** as the value is never used.  So really the only thing this code
  ** accomplishes is to quieten purify.  
  **
  ** 2007-01-16:  The wireshark project (www.wireshark.org) reports that
  ** without this code, their parser segfaults.  I'm not sure what there
  ** parser is doing to make this happen.  This is the second bug report
  ** from wireshark this week.  Clearly they are stressing Lemon in ways
  ** that it has not been previously stressed...  (SQLite ticket #2172)
  */
  /*memset(&yygotominor, 0, sizeof(yygotominor));*/
  yygotominor = yyzerominor;


  switch( yyruleno ){
  /* Beginning here are the reduction cases.  A typical example
  ** follows:
  **   case 0:
  **  #line <lineno> <grammarfile>
  **     { ... }           // User supplied code
  **  #line <lineno> <thisfile>
  **     break;
  */
      case 0: /* prog ::= optnl statementlist optnl */
#line 71 "parser.y"
{ parser->result = Expression(List(*yymsp[-1].minor.yy82)); }
#line 1020 "parser.c"
        break;
      case 1: /* prog ::= error */
#line 72 "parser.y"
{ parser->result = Expression(0); }
#line 1025 "parser.c"
        break;
      case 2: /* statement ::= expr EQ_ASSIGN optnl statement */
      case 15: /* expr ::= expr COLON optnl expr */ yytestcase(yyruleno==15);
      case 16: /* expr ::= expr PLUS optnl expr */ yytestcase(yyruleno==16);
      case 17: /* expr ::= expr MINUS optnl expr */ yytestcase(yyruleno==17);
      case 18: /* expr ::= expr TIMES optnl expr */ yytestcase(yyruleno==18);
      case 19: /* expr ::= expr DIVIDE optnl expr */ yytestcase(yyruleno==19);
      case 20: /* expr ::= expr POW optnl expr */ yytestcase(yyruleno==20);
      case 21: /* expr ::= expr SPECIALOP optnl expr */ yytestcase(yyruleno==21);
      case 22: /* expr ::= expr TILDE optnl expr */ yytestcase(yyruleno==22);
      case 23: /* expr ::= expr QUESTION optnl expr */ yytestcase(yyruleno==23);
      case 24: /* expr ::= expr LT optnl expr */ yytestcase(yyruleno==24);
      case 25: /* expr ::= expr LE optnl expr */ yytestcase(yyruleno==25);
      case 26: /* expr ::= expr EQ optnl expr */ yytestcase(yyruleno==26);
      case 27: /* expr ::= expr NE optnl expr */ yytestcase(yyruleno==27);
      case 28: /* expr ::= expr GE optnl expr */ yytestcase(yyruleno==28);
      case 29: /* expr ::= expr GT optnl expr */ yytestcase(yyruleno==29);
      case 30: /* expr ::= expr AND optnl expr */ yytestcase(yyruleno==30);
      case 31: /* expr ::= expr OR optnl expr */ yytestcase(yyruleno==31);
      case 32: /* expr ::= expr AND2 optnl expr */ yytestcase(yyruleno==32);
      case 33: /* expr ::= expr OR2 optnl expr */ yytestcase(yyruleno==33);
      case 34: /* expr ::= expr LEFT_ASSIGN optnl expr */ yytestcase(yyruleno==34);
      case 49: /* expr ::= expr DOLLAR optnl symbolstr */ yytestcase(yyruleno==49);
      case 50: /* expr ::= expr AT optnl symbolstr */ yytestcase(yyruleno==50);
#line 74 "parser.y"
{ yygotominor.yy0 = Call::c(yymsp[-2].minor.yy0, yymsp[-3].minor.yy0, yymsp[0].minor.yy0); }
#line 1052 "parser.c"
        break;
      case 3: /* statement ::= expr */
      case 4: /* expr ::= NUM_CONST */ yytestcase(yyruleno==4);
      case 5: /* expr ::= STR_CONST */ yytestcase(yyruleno==5);
      case 6: /* expr ::= NULL_CONST */ yytestcase(yyruleno==6);
      case 7: /* expr ::= SYMBOL */ yytestcase(yyruleno==7);
      case 53: /* symbolstr ::= STR_CONST */ yytestcase(yyruleno==53);
      case 54: /* symbolstr ::= SYMBOL */ yytestcase(yyruleno==54);
#line 75 "parser.y"
{ yygotominor.yy0 = yymsp[0].minor.yy0; }
#line 1063 "parser.c"
        break;
      case 8: /* expr ::= LBRACE optnl statementlist optnl RBRACE */
#line 82 "parser.y"
{ yymsp[-2].minor.yy82->push_front(Symbol::empty, yymsp[-4].minor.yy0); yygotominor.yy0 = Call(List(*yymsp[-2].minor.yy82));   yy_destructor(yypParser,44,&yymsp[0].minor);
}
#line 1069 "parser.c"
        break;
      case 9: /* expr ::= LPAREN optnl statement optnl RPAREN */
#line 83 "parser.y"
{ yygotominor.yy0 = yymsp[-2].minor.yy0;   yy_destructor(yypParser,36,&yymsp[-4].minor);
  yy_destructor(yypParser,45,&yymsp[0].minor);
}
#line 1076 "parser.c"
        break;
      case 10: /* expr ::= MINUS optnl expr */
      case 11: /* expr ::= PLUS optnl expr */ yytestcase(yyruleno==11);
      case 12: /* expr ::= NOT optnl expr */ yytestcase(yyruleno==12);
      case 13: /* expr ::= TILDE optnl expr */ yytestcase(yyruleno==13);
      case 14: /* expr ::= QUESTION optnl expr */ yytestcase(yyruleno==14);
      case 42: /* expr ::= REPEAT optnl statement */ yytestcase(yyruleno==42);
#line 85 "parser.y"
{ yygotominor.yy0 = Call::c(yymsp[-2].minor.yy0, yymsp[0].minor.yy0); }
#line 1086 "parser.c"
        break;
      case 35: /* expr ::= expr RIGHT_ASSIGN optnl expr */
#line 112 "parser.y"
{ yygotominor.yy0 = Call::c(yymsp[-2].minor.yy0, yymsp[0].minor.yy0, yymsp[-3].minor.yy0); }
#line 1091 "parser.c"
        break;
      case 36: /* expr ::= FUNCTION optnl LPAREN optnl formallist optnl RPAREN optnl statement */
#line 113 "parser.y"
{ yygotominor.yy0 = Call::c(yymsp[-8].minor.yy0, PairList(List(*yymsp[-4].minor.yy82)), yymsp[0].minor.yy0);   yy_destructor(yypParser,36,&yymsp[-6].minor);
  yy_destructor(yypParser,45,&yymsp[-2].minor);
}
#line 1098 "parser.c"
        break;
      case 37: /* expr ::= expr LPAREN optnl sublist optnl RPAREN */
#line 114 "parser.y"
{ yymsp[-2].minor.yy82->push_front(Symbol::empty, yymsp[-5].minor.yy0); yygotominor.yy0 = Call(List(*yymsp[-2].minor.yy82));   yy_destructor(yypParser,36,&yymsp[-4].minor);
  yy_destructor(yypParser,45,&yymsp[0].minor);
}
#line 1105 "parser.c"
        break;
      case 38: /* expr ::= IF optnl LPAREN optnl expr optnl RPAREN optnl statement */
      case 41: /* expr ::= WHILE optnl LPAREN optnl expr optnl RPAREN optnl statement */ yytestcase(yyruleno==41);
#line 115 "parser.y"
{ yygotominor.yy0 = Call::c(yymsp[-8].minor.yy0, yymsp[-4].minor.yy0, yymsp[0].minor.yy0);   yy_destructor(yypParser,36,&yymsp[-6].minor);
  yy_destructor(yypParser,45,&yymsp[-2].minor);
}
#line 1113 "parser.c"
        break;
      case 39: /* expr ::= IF optnl LPAREN optnl expr optnl RPAREN optnl statement ELSE optnl statement */
#line 116 "parser.y"
{ yygotominor.yy0 = Call::c(yymsp[-11].minor.yy0, yymsp[-7].minor.yy0, yymsp[-3].minor.yy0, yymsp[0].minor.yy0);   yy_destructor(yypParser,36,&yymsp[-9].minor);
  yy_destructor(yypParser,45,&yymsp[-5].minor);
  yy_destructor(yypParser,7,&yymsp[-2].minor);
}
#line 1121 "parser.c"
        break;
      case 40: /* expr ::= FOR optnl LPAREN optnl SYMBOL optnl IN optnl expr optnl RPAREN optnl statement */
#line 117 "parser.y"
{ yygotominor.yy0 = Call::c(yymsp[-12].minor.yy0, yymsp[-8].minor.yy0, yymsp[-4].minor.yy0, yymsp[0].minor.yy0);   yy_destructor(yypParser,36,&yymsp[-10].minor);
  yy_destructor(yypParser,46,&yymsp[-6].minor);
  yy_destructor(yypParser,45,&yymsp[-2].minor);
}
#line 1129 "parser.c"
        break;
      case 43: /* expr ::= expr LBB optnl sublist optnl RBB */
#line 120 "parser.y"
{ yymsp[-2].minor.yy82->push_front(Symbol::empty, yymsp[-5].minor.yy0); yymsp[-2].minor.yy82->push_front(Symbol::empty, yymsp[-4].minor.yy0); yygotominor.yy0 = Call(List(*yymsp[-2].minor.yy82));   yy_destructor(yypParser,47,&yymsp[0].minor);
}
#line 1135 "parser.c"
        break;
      case 44: /* expr ::= expr LBRACKET optnl sublist optnl RBRACKET */
#line 121 "parser.y"
{ yymsp[-2].minor.yy82->push_front(Symbol::empty, yymsp[-5].minor.yy0); yymsp[-2].minor.yy82->push_front(Symbol::empty, yymsp[-4].minor.yy0); yygotominor.yy0 = Call(List(*yymsp[-2].minor.yy82));   yy_destructor(yypParser,48,&yymsp[0].minor);
}
#line 1141 "parser.c"
        break;
      case 45: /* expr ::= SYMBOL NS_GET symbolstr */
      case 46: /* expr ::= STR_CONST NS_GET symbolstr */ yytestcase(yyruleno==46);
      case 47: /* expr ::= SYMBOL NS_GET_INT symbolstr */ yytestcase(yyruleno==47);
      case 48: /* expr ::= STR_CONST NS_GET_INT symbolstr */ yytestcase(yyruleno==48);
#line 122 "parser.y"
{ yygotominor.yy0 = Call::c(yymsp[-1].minor.yy0, yymsp[-2].minor.yy0, yymsp[0].minor.yy0); }
#line 1149 "parser.c"
        break;
      case 51: /* expr ::= NEXT */
      case 52: /* expr ::= BREAK */ yytestcase(yyruleno==52);
#line 128 "parser.y"
{ yygotominor.yy0 = Call::c(yymsp[0].minor.yy0); }
#line 1155 "parser.c"
        break;
      case 55: /* optnl ::= NEWLINE */
#line 134 "parser.y"
{
  yy_destructor(yypParser,51,&yymsp[0].minor);
}
#line 1162 "parser.c"
        break;
      case 57: /* statementlist ::= statementlist SEMICOLON statement */
#line 137 "parser.y"
{ yygotominor.yy82 = yymsp[-2].minor.yy82; yygotominor.yy82->push_back(Symbol::empty, yymsp[0].minor.yy0);   yy_destructor(yypParser,52,&yymsp[-1].minor);
}
#line 1168 "parser.c"
        break;
      case 58: /* statementlist ::= statementlist SEMICOLON */
#line 138 "parser.y"
{ yygotominor.yy82 = yymsp[-1].minor.yy82;   yy_destructor(yypParser,52,&yymsp[0].minor);
}
#line 1174 "parser.c"
        break;
      case 59: /* statementlist ::= statementlist NEWLINE statement */
#line 139 "parser.y"
{ yygotominor.yy82 = yymsp[-2].minor.yy82; yygotominor.yy82->push_back(Symbol::empty, yymsp[0].minor.yy0);   yy_destructor(yypParser,51,&yymsp[-1].minor);
}
#line 1180 "parser.c"
        break;
      case 60: /* statementlist ::= statement */
      case 71: /* sub ::= expr */ yytestcase(yyruleno==71);
#line 140 "parser.y"
{ yygotominor.yy82 = new Pairs(); yygotominor.yy82->push_back(Symbol::empty, yymsp[0].minor.yy0); }
#line 1186 "parser.c"
        break;
      case 61: /* statementlist ::= */
      case 64: /* sub ::= */ yytestcase(yyruleno==64);
      case 72: /* formallist ::= */ yytestcase(yyruleno==72);
#line 141 "parser.y"
{ yygotominor.yy82 = new Pairs(); }
#line 1193 "parser.c"
        break;
      case 62: /* sublist ::= sub */
#line 143 "parser.y"
{ yygotominor.yy82 = yymsp[0].minor.yy82; }
#line 1198 "parser.c"
        break;
      case 63: /* sublist ::= sublist optnl COMMA optnl sub */
#line 144 "parser.y"
{ yygotominor.yy82 = yymsp[-4].minor.yy82; if(yymsp[0].minor.yy82->length() == 1) yygotominor.yy82->push_back(yymsp[0].minor.yy82->name(0), yymsp[0].minor.yy82->value(0));   yy_destructor(yypParser,53,&yymsp[-2].minor);
}
#line 1204 "parser.c"
        break;
      case 65: /* sub ::= SYMBOL optnl EQ_ASSIGN */
      case 66: /* sub ::= STR_CONST optnl EQ_ASSIGN */ yytestcase(yyruleno==66);
#line 147 "parser.y"
{ yygotominor.yy82 = new Pairs(); yygotominor.yy82->push_back(yymsp[-2].minor.yy0, Value::NIL);   yy_destructor(yypParser,9,&yymsp[0].minor);
}
#line 1211 "parser.c"
        break;
      case 67: /* sub ::= SYMBOL optnl EQ_ASSIGN optnl expr */
      case 68: /* sub ::= STR_CONST optnl EQ_ASSIGN optnl expr */ yytestcase(yyruleno==68);
      case 74: /* formallist ::= SYMBOL optnl EQ_ASSIGN optnl expr */ yytestcase(yyruleno==74);
#line 149 "parser.y"
{ yygotominor.yy82 = new Pairs(); yygotominor.yy82->push_back(Symbol(yymsp[-4].minor.yy0), yymsp[0].minor.yy0);   yy_destructor(yypParser,9,&yymsp[-2].minor);
}
#line 1219 "parser.c"
        break;
      case 69: /* sub ::= NULL_CONST optnl EQ_ASSIGN */
#line 151 "parser.y"
{ yygotominor.yy82 = new Pairs(); yygotominor.yy82->push_back(Symbol::empty, Value::NIL);   yy_destructor(yypParser,42,&yymsp[-2].minor);
  yy_destructor(yypParser,9,&yymsp[0].minor);
}
#line 1226 "parser.c"
        break;
      case 70: /* sub ::= NULL_CONST optnl EQ_ASSIGN optnl expr */
#line 152 "parser.y"
{ yygotominor.yy82 = new Pairs(); yygotominor.yy82->push_back(Symbol::empty, yymsp[0].minor.yy0);   yy_destructor(yypParser,42,&yymsp[-4].minor);
  yy_destructor(yypParser,9,&yymsp[-2].minor);
}
#line 1233 "parser.c"
        break;
      case 73: /* formallist ::= SYMBOL */
#line 156 "parser.y"
{ yygotominor.yy82 = new Pairs(); yygotominor.yy82->push_back(Symbol(yymsp[0].minor.yy0), Value::NIL); }
#line 1238 "parser.c"
        break;
      case 75: /* formallist ::= formallist optnl COMMA optnl SYMBOL */
#line 158 "parser.y"
{ yygotominor.yy82 = yymsp[-4].minor.yy82; yygotominor.yy82->push_back(Symbol(yymsp[0].minor.yy0), Value::NIL);   yy_destructor(yypParser,53,&yymsp[-2].minor);
}
#line 1244 "parser.c"
        break;
      case 76: /* formallist ::= formallist optnl COMMA optnl SYMBOL optnl EQ_ASSIGN optnl expr */
#line 159 "parser.y"
{ yygotominor.yy82 = yymsp[-8].minor.yy82; yygotominor.yy82->push_back(Symbol(yymsp[-4].minor.yy0), yymsp[0].minor.yy0);   yy_destructor(yypParser,53,&yymsp[-6].minor);
  yy_destructor(yypParser,9,&yymsp[-2].minor);
}
#line 1251 "parser.c"
        break;
      default:
      /* (56) optnl ::= */ yytestcase(yyruleno==56);
        break;
  };
  yygoto = yyRuleInfo[yyruleno].lhs;
  yysize = yyRuleInfo[yyruleno].nrhs;
  yypParser->yyidx -= yysize;
  yyact = yy_find_reduce_action(yymsp[-yysize].stateno,(YYCODETYPE)yygoto);
  if( yyact < YYNSTATE ){
#ifdef NDEBUG
    /* If we are not debugging and the reduce action popped at least
    ** one element off the stack, then we can push the new element back
    ** onto the stack here, and skip the stack overflow test in yy_shift().
    ** That gives a significant speed improvement. */
    if( yysize ){
      yypParser->yyidx++;
      yymsp -= yysize-1;
      yymsp->stateno = (YYACTIONTYPE)yyact;
      yymsp->major = (YYCODETYPE)yygoto;
      yymsp->minor = yygotominor;
    }else
#endif
    {
      yy_shift(yypParser,yyact,yygoto,&yygotominor);
    }
  }else{
    assert( yyact == YYNSTATE + YYNRULE + 1 );
    yy_accept(yypParser);
  }
}

/*
** The following code executes when the parse fails
*/
#ifndef YYNOERRORRECOVERY
static void yy_parse_failed(
  yyParser *yypParser           /* The parser */
){
  ParseARG_FETCH;
#ifndef NDEBUG
  if( yyTraceFILE ){
    fprintf(yyTraceFILE,"%sFail!\n",yyTracePrompt);
  }
#endif
  while( yypParser->yyidx>=0 ) yy_pop_parser_stack(yypParser);
  /* Here code is inserted which will be executed whenever the
  ** parser fails */
  ParseARG_STORE; /* Suppress warning about unused %extra_argument variable */
}
#endif /* YYNOERRORRECOVERY */

/*
** The following code executes when a syntax error first occurs.
*/
static void yy_syntax_error(
  yyParser *yypParser,           /* The parser */
  int yymajor,                   /* The major type of the error token */
  YYMINORTYPE yyminor            /* The minor type of the error token */
){
  ParseARG_FETCH;
#define TOKEN (yyminor.yy0)
#line 53 "parser.y"

        parser->errors++;
#line 1317 "parser.c"
  ParseARG_STORE; /* Suppress warning about unused %extra_argument variable */
}

/*
** The following is executed when the parser accepts
*/
static void yy_accept(
  yyParser *yypParser           /* The parser */
){
  ParseARG_FETCH;
#ifndef NDEBUG
  if( yyTraceFILE ){
    fprintf(yyTraceFILE,"%sAccept!\n",yyTracePrompt);
  }
#endif
  while( yypParser->yyidx>=0 ) yy_pop_parser_stack(yypParser);
  /* Here code is inserted which will be executed whenever the
  ** parser accepts */
#line 57 "parser.y"

	parser->complete = true;
#line 1339 "parser.c"
  ParseARG_STORE; /* Suppress warning about unused %extra_argument variable */
}

/* The main parser program.
** The first argument is a pointer to a structure obtained from
** "ParseAlloc" which describes the current state of the parser.
** The second argument is the major token number.  The third is
** the minor token.  The fourth optional argument is whatever the
** user wants (and specified in the grammar) and is available for
** use by the action routines.
**
** Inputs:
** <ul>
** <li> A pointer to the parser (an opaque structure.)
** <li> The major token number.
** <li> The minor token number.
** <li> An option argument of a grammar-specified type.
** </ul>
**
** Outputs:
** None.
*/
void Parse(
  void *yyp,                   /* The parser */
  int yymajor,                 /* The major token code number */
  ParseTOKENTYPE yyminor       /* The value for the token */
  ParseARG_PDECL               /* Optional %extra_argument parameter */
){
  YYMINORTYPE yyminorunion;
  int yyact;            /* The parser action. */
  int yyendofinput;     /* True if we are at the end of input */
#ifdef YYERRORSYMBOL
  int yyerrorhit = 0;   /* True if yymajor has invoked an error */
#endif
  yyParser *yypParser;  /* The parser */

  /* (re)initialize the parser, if necessary */
  yypParser = (yyParser*)yyp;
  if( yypParser->yyidx<0 ){
#if YYSTACKDEPTH<=0
    if( yypParser->yystksz <=0 ){
      /*memset(&yyminorunion, 0, sizeof(yyminorunion));*/
      yyminorunion = yyzerominor;
      yyStackOverflow(yypParser, &yyminorunion);
      return;
    }
#endif
    yypParser->yyidx = 0;
    yypParser->yyerrcnt = -1;
    yypParser->yystack[0].stateno = 0;
    yypParser->yystack[0].major = 0;
  }
  yyminorunion.yy0 = yyminor;
  yyendofinput = (yymajor==0);
  ParseARG_STORE;

#ifndef NDEBUG
  if( yyTraceFILE ){
    fprintf(yyTraceFILE,"%sInput %s\n",yyTracePrompt,yyTokenName[yymajor]);
  }
#endif

  do{
    yyact = yy_find_shift_action(yypParser,(YYCODETYPE)yymajor);
    if( yyact<YYNSTATE ){
      assert( !yyendofinput );  /* Impossible to shift the $ token */
      yy_shift(yypParser,yyact,yymajor,&yyminorunion);
      yypParser->yyerrcnt--;
      yymajor = YYNOCODE;
    }else if( yyact < YYNSTATE + YYNRULE ){
      yy_reduce(yypParser,yyact-YYNSTATE);
    }else{
      assert( yyact == YY_ERROR_ACTION );
#ifdef YYERRORSYMBOL
      int yymx;
#endif
#ifndef NDEBUG
      if( yyTraceFILE ){
        fprintf(yyTraceFILE,"%sSyntax Error!\n",yyTracePrompt);
      }
#endif
#ifdef YYERRORSYMBOL
      /* A syntax error has occurred.
      ** The response to an error depends upon whether or not the
      ** grammar defines an error token "ERROR".  
      **
      ** This is what we do if the grammar does define ERROR:
      **
      **  * Call the %syntax_error function.
      **
      **  * Begin popping the stack until we enter a state where
      **    it is legal to shift the error symbol, then shift
      **    the error symbol.
      **
      **  * Set the error count to three.
      **
      **  * Begin accepting and shifting new tokens.  No new error
      **    processing will occur until three tokens have been
      **    shifted successfully.
      **
      */
      if( yypParser->yyerrcnt<0 ){
        yy_syntax_error(yypParser,yymajor,yyminorunion);
      }
      yymx = yypParser->yystack[yypParser->yyidx].major;
      if( yymx==YYERRORSYMBOL || yyerrorhit ){
#ifndef NDEBUG
        if( yyTraceFILE ){
          fprintf(yyTraceFILE,"%sDiscard input token %s\n",
             yyTracePrompt,yyTokenName[yymajor]);
        }
#endif
        yy_destructor(yypParser, (YYCODETYPE)yymajor,&yyminorunion);
        yymajor = YYNOCODE;
      }else{
         while(
          yypParser->yyidx >= 0 &&
          yymx != YYERRORSYMBOL &&
          (yyact = yy_find_reduce_action(
                        yypParser->yystack[yypParser->yyidx].stateno,
                        YYERRORSYMBOL)) >= YYNSTATE
        ){
          yy_pop_parser_stack(yypParser);
        }
        if( yypParser->yyidx < 0 || yymajor==0 ){
          yy_destructor(yypParser,(YYCODETYPE)yymajor,&yyminorunion);
          yy_parse_failed(yypParser);
          yymajor = YYNOCODE;
        }else if( yymx!=YYERRORSYMBOL ){
          YYMINORTYPE u2;
          u2.YYERRSYMDT = 0;
          yy_shift(yypParser,yyact,YYERRORSYMBOL,&u2);
        }
      }
      yypParser->yyerrcnt = 3;
      yyerrorhit = 1;
#elif defined(YYNOERRORRECOVERY)
      /* If the YYNOERRORRECOVERY macro is defined, then do not attempt to
      ** do any kind of error recovery.  Instead, simply invoke the syntax
      ** error routine and continue going as if nothing had happened.
      **
      ** Applications can set this macro (for example inside %include) if
      ** they intend to abandon the parse upon the first syntax error seen.
      */
      yy_syntax_error(yypParser,yymajor,yyminorunion);
      yy_destructor(yypParser,(YYCODETYPE)yymajor,&yyminorunion);
      yymajor = YYNOCODE;
      
#else  /* YYERRORSYMBOL is not defined */
      /* This is what we do if the grammar does not define ERROR:
      **
      **  * Report an error message, and throw away the input token.
      **
      **  * If the input token is $, then fail the parse.
      **
      ** As before, subsequent error messages are suppressed until
      ** three input tokens have been successfully shifted.
      */
      if( yypParser->yyerrcnt<=0 ){
        yy_syntax_error(yypParser,yymajor,yyminorunion);
      }
      yypParser->yyerrcnt = 3;
      yy_destructor(yypParser,(YYCODETYPE)yymajor,&yyminorunion);
      if( yyendofinput ){
        yy_parse_failed(yypParser);
      }
      yymajor = YYNOCODE;
#endif
    }
  }while( yymajor!=YYNOCODE && yypParser->yyidx>=0 );
  return;
}
