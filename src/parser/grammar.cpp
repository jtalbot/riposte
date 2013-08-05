/* Driver template for the LEMON parser generator.
** The author disclaims copyright to this source code.
*/
/* First off, code is included that follows the "include" declaration
** in the input grammar file. */
#include <stdio.h>
#line 48 "grammar.y"

	#include <iostream>
	#include "../runtime.h"
#line 12 "grammar.c"
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
#define YYNOCODE 64
#define YYACTIONTYPE unsigned short int
#define ParseTOKENTYPE Value
typedef union {
  int yyinit;
  ParseTOKENTYPE yy0;
  Pairs* yy122;
  int yy127;
} YYMINORTYPE;
#ifndef YYSTACKDEPTH
#define YYSTACKDEPTH 1000
#endif
#define ParseARG_SDECL Parser* parser;
#define ParseARG_PDECL ,Parser* parser
#define ParseARG_FETCH Parser* parser = yypParser->parser
#define ParseARG_STORE yypParser->parser = parser
#define YYNSTATE 219
#define YYNRULE 78
#define YYERRORSYMBOL 53
#define YYERRSYMDT yy127
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
 /*     0 */    74,  289,   47,  289,  218,  130,  193,  107,  289,  109,
 /*    10 */    76,   99,  104,   97,  102,  193,   96,   84,   72,   75,
 /*    20 */    78,   81,  113,  108,  100,   85,   73,  119,  206,  130,
 /*    30 */    79,   86,   89,  185,   74,  111,  124,  118,  298,    6,
 /*    40 */    48,  107,  103,  109,   76,   99,  104,   97,  102,  193,
 /*    50 */    96,   84,   72,   75,   78,   81,  113,  108,  100,   85,
 /*    60 */    73,  119,   77,  203,   79,   86,   89,   74,  134,  111,
 /*    70 */   124,  118,  190,  130,  107,   46,  109,   76,   99,  104,
 /*    80 */    97,  102,  219,   96,   84,   72,   75,   78,   81,  113,
 /*    90 */   108,  100,   85,   73,  119,   80,  203,   79,   86,   89,
 /*   100 */    93,  134,  111,  124,  118,  184,  107,  110,  109,   76,
 /*   110 */    99,  104,   97,  102,  156,   96,   84,   72,   75,   78,
 /*   120 */    81,  113,  108,  100,   85,   73,  119,   83,  203,   79,
 /*   130 */    86,   89,  193,  134,  111,  124,  118,   76,   99,  104,
 /*   140 */    97,  102,  276,   96,   84,   72,   75,   78,   81,  113,
 /*   150 */   108,  100,   85,   73,  119,  202,  199,   79,   86,   89,
 /*   160 */   134,   82,  111,  124,  118,   99,  104,   97,  102,    4,
 /*   170 */    96,   84,   72,   75,   78,   81,  113,  108,  100,   85,
 /*   180 */    73,  119,   45,  193,   79,   86,   89,   16,   11,  111,
 /*   190 */   124,  118,   97,  102,   44,   96,   84,   72,   75,   78,
 /*   200 */    81,  113,  108,  100,   85,   73,  119,  192,  130,   79,
 /*   210 */    86,   89,  174,  130,  111,  124,  118,   98,   92,  120,
 /*   220 */   127,  114,   91,  100,   85,   73,  119,  105,  276,   79,
 /*   230 */    86,   89,  112,   10,  111,  124,  118,   73,  119,  117,
 /*   240 */   123,   79,   86,   89,   60,   13,  111,  124,  118,  145,
 /*   250 */   197,  130,  128,   61,   62,   88,  191,  168,  205,  165,
 /*   260 */    28,   79,   86,   89,  217,  213,  111,  124,  118,  193,
 /*   270 */    96,   84,   72,   75,   78,   81,  113,  108,  100,   85,
 /*   280 */    73,  119,  209,  130,   79,   86,   89,  276,   43,  111,
 /*   290 */   124,  118,   98,   92,  120,  127,  114,   91,    2,  196,
 /*   300 */   119,  200,  105,   79,   86,   89,  189,  112,  111,  124,
 /*   310 */   118,   82,   63,   64,  117,  123,  286,  160,  286,   66,
 /*   320 */    41,  193,  181,  286,  195,  197,  130,  128,  193,   65,
 /*   330 */    88,  191,   57,   67,   55,  197,  130,  210,   12,  217,
 /*   340 */   213,  285,   29,  285,   82,   35,  193,    1,  285,  204,
 /*   350 */   194,  130,  178,   61,   62,  198,  130,  177,   63,   64,
 /*   360 */   214,  130,  176,    5,   14,    7,  182,  173,  179,   58,
 /*   370 */   208,  153,  146,  158,  139,  141,   17,   19,  135,   18,
 /*   380 */    94,  136,  132,  131,  161,  299,  299,  147,  137,  150,
 /*   390 */   149,  151,  159,  133,  162,  154,  148,  140,  138,  155,
 /*   400 */   143,  142,  152,  144,  157,   34,  175,   42,  183,  171,
 /*   410 */    32,  187,  188,  129,  125,  169,  212,  215,  180,  216,
 /*   420 */   201,  186,   59,   95,   24,  211,   25,   26,   23,  299,
 /*   430 */   299,   27,  167,   21,   22,  164,   40,  166,   54,   39,
 /*   440 */   221,   20,   68,   15,    3,   90,  126,   69,   31,  172,
 /*   450 */   170,    9,   71,  163,   38,   36,   37,   52,   53,   30,
 /*   460 */    51,    8,   49,   50,   33,   87,   56,  122,  121,  116,
 /*   470 */   115,   70,  106,  101,  220,  299,  299,  299,  299,  207,
};
static const YYCODETYPE yy_lookahead[] = {
 /*     0 */     1,   45,   59,   47,   60,   61,   50,    8,   52,   10,
 /*    10 */    11,   12,   13,   14,   15,   50,   17,   18,   19,   20,
 /*    20 */    21,   22,   23,   24,   25,   26,   27,   28,   60,   61,
 /*    30 */    31,   32,   33,   53,    1,   36,   37,   38,   58,   59,
 /*    40 */    59,    8,    9,   10,   11,   12,   13,   14,   15,   50,
 /*    50 */    17,   18,   19,   20,   21,   22,   23,   24,   25,   26,
 /*    60 */    27,   28,   55,   56,   31,   32,   33,    1,   61,   36,
 /*    70 */    37,   38,   60,   61,    8,   59,   10,   11,   12,   13,
 /*    80 */    14,   15,    0,   17,   18,   19,   20,   21,   22,   23,
 /*    90 */    24,   25,   26,   27,   28,   55,   56,   31,   32,   33,
 /*   100 */    45,   61,   36,   37,   38,   59,    8,   52,   10,   11,
 /*   110 */    12,   13,   14,   15,   61,   17,   18,   19,   20,   21,
 /*   120 */    22,   23,   24,   25,   26,   27,   28,   55,   56,   31,
 /*   130 */    32,   33,   50,   61,   36,   37,   38,   11,   12,   13,
 /*   140 */    14,   15,    9,   17,   18,   19,   20,   21,   22,   23,
 /*   150 */    24,   25,   26,   27,   28,   56,   47,   31,   32,   33,
 /*   160 */    61,   52,   36,   37,   38,   12,   13,   14,   15,   59,
 /*   170 */    17,   18,   19,   20,   21,   22,   23,   24,   25,   26,
 /*   180 */    27,   28,   59,   50,   31,   32,   33,   50,   51,   36,
 /*   190 */    37,   38,   14,   15,   59,   17,   18,   19,   20,   21,
 /*   200 */    22,   23,   24,   25,   26,   27,   28,   60,   61,   31,
 /*   210 */    32,   33,   60,   61,   36,   37,   38,    1,    2,    3,
 /*   220 */     4,    5,    6,   25,   26,   27,   28,   11,    9,   31,
 /*   230 */    32,   33,   16,   59,   36,   37,   38,   27,   28,   23,
 /*   240 */    24,   31,   32,   33,   54,   59,   36,   37,   38,   61,
 /*   250 */    60,   61,   36,   34,   35,   39,   40,   41,   42,   43,
 /*   260 */    59,   31,   32,   33,   48,   49,   36,   37,   38,   50,
 /*   270 */    17,   18,   19,   20,   21,   22,   23,   24,   25,   26,
 /*   280 */    27,   28,   60,   61,   31,   32,   33,    9,   59,   36,
 /*   290 */    37,   38,    1,    2,    3,    4,    5,    6,   59,   41,
 /*   300 */    28,   43,   11,   31,   32,   33,   47,   16,   36,   37,
 /*   310 */    38,   52,   34,   35,   23,   24,   45,   61,   47,   54,
 /*   320 */    59,   50,   59,   52,   62,   60,   61,   36,   50,   54,
 /*   330 */    39,   40,   41,   42,   43,   60,   61,   45,   59,   48,
 /*   340 */    49,   45,   59,   47,   52,   59,   50,   59,   52,   62,
 /*   350 */    60,   61,   59,   34,   35,   60,   61,   59,   34,   35,
 /*   360 */    60,   61,   59,   59,   59,   59,   59,   59,   59,    9,
 /*   370 */    45,   61,   61,   61,   61,   61,   61,   61,   61,   61,
 /*   380 */    36,   61,   61,   61,   61,   63,   63,   61,   61,   61,
 /*   390 */    61,   61,   61,   61,   61,   61,   61,   61,   61,   61,
 /*   400 */    61,   61,   61,   61,   61,   59,   59,   59,   59,   59,
 /*   410 */    59,   59,   59,    7,   43,   59,   62,   62,   59,   62,
 /*   420 */    62,   59,   43,   57,   59,   44,   59,   59,   59,   63,
 /*   430 */    63,   59,   59,   59,   59,   59,   59,   59,    9,   59,
 /*   440 */     0,   59,   59,   59,   59,   36,   36,   59,   59,   59,
 /*   450 */    59,   59,   59,   59,   59,   59,   59,   59,   59,   59,
 /*   460 */    59,   59,   59,   59,   59,   45,    9,   46,   45,   36,
 /*   470 */    45,   43,    9,    9,    0,   63,   63,   63,   63,   47,
};
#define YY_SHIFT_USE_DFLT (-45)
#define YY_SHIFT_MAX 189
static const short yy_shift_ofst[] = {
 /*     0 */    82,  291,  291,  216,  291,  216,  216,  291,  216,  216,
 /*    10 */   216,  216,  216,  216,  216,  216,  216,   -1,   -1,   -1,
 /*    20 */   216,  216,  216,  216,  216,  216,  216,  216,  216,  216,
 /*    30 */   216,  216,  216,  216,  216,  216,  216,  216,  216,  216,
 /*    40 */   216,  216,  216,  216,  216,  216,  216,  216,  216,  216,
 /*    50 */   216,  216,  216,  216,  -44,  278,  271,  219,  296,  133,
 /*    60 */   137,  258,  258,  258,  258,  137,  137,  133,  258,  258,
 /*    70 */   133,  379,  -35,  -35,  -35,  -35,  -35,  -35,  -35,  -35,
 /*    80 */   -35,  -35,  -35,  -35,  -35,  -35,  -35,  -35,  -35,  -35,
 /*    90 */   -35,  -35,  -35,  -35,  -35,  -35,  -35,  -35,  -35,  -35,
 /*   100 */   -35,  -35,  -35,  -35,  -35,  -35,  -35,  -35,  -35,  -35,
 /*   110 */   -35,  -35,  -35,  -35,  -35,  -35,  -35,  -35,  -35,  -35,
 /*   120 */   -35,  -35,  -35,  -35,  -35,  -35,  -35,  -35,  -35,  -35,
 /*   130 */    33,   66,   66,   66,   66,   66,   66,   98,   98,   98,
 /*   140 */   126,  153,  153,  178,  178,  253,  253,  253,  253,  253,
 /*   150 */   253,  253,  253,  253,  198,  198,  210,  210,  272,  230,
 /*   160 */   230,  230,  230,   55,  259,  324,  109,  292,  319,  325,
 /*   170 */   344,  381,  409,  420,  406,  360,  410,  371,  421,  423,
 /*   180 */   429,  433,  425,  457,  428,  440,  463,  464,  474,  432,
};
#define YY_REDUCE_USE_DFLT (-58)
#define YY_REDUCE_MAX 129
static const short yy_reduce_ofst[] = {
 /*     0 */   -20,   72,   40,  275,    7,  265,  190,   99,   12,  -32,
 /*    10 */   -56,  300,  295,  290,  222,  152,  147,  307,  308,  309,
 /*    20 */    53,  188,  256,  310,  311,  312,  313,  314,  315,  316,
 /*    30 */   317,  318,  320,  321,  322,  323,  326,  327,  328,  329,
 /*    40 */   330,  331,  332,  333,  334,  335,  336,  337,  338,  339,
 /*    50 */   340,  341,  342,  343,  346,  347,  348,  349,  351,  352,
 /*    60 */   353,  262,  287,  354,  355,  350,  356,  359,  357,  358,
 /*    70 */   362,  366,  365,  367,  368,  369,  372,  373,  374,  375,
 /*    80 */   376,  377,  306,  378,  380,  382,  383,  384,  385,  388,
 /*    90 */   389,  390,  391,  392,  393,  394,  395,  396,  397,  398,
 /*   100 */   399,  400,  401,  402,  403,  404,  405,  -57,  -19,   16,
 /*   110 */    46,  110,  123,  135,  174,  186,  201,  229,  239,  261,
 /*   120 */   263,  279,  283,  286,  288,  293,  298,  303,  304,  305,
};
static const YYACTIONTYPE yy_default[] = {
 /*     0 */   276,  284,  284,  281,  284,  281,  281,  284,  297,  297,
 /*    10 */   297,  278,  297,  297,  297,  297,  275,  276,  276,  276,
 /*    20 */   297,  297,  297,  297,  297,  297,  297,  297,  297,  297,
 /*    30 */   297,  297,  297,  297,  297,  297,  297,  297,  297,  297,
 /*    40 */   297,  297,  297,  297,  297,  297,  297,  297,  297,  297,
 /*    50 */   297,  297,  297,  297,  276,  227,  276,  225,  276,  293,
 /*    60 */   276,  297,  297,  297,  297,  276,  276,  226,  297,  297,
 /*    70 */   295,  292,  276,  276,  276,  276,  276,  276,  276,  276,
 /*    80 */   276,  276,  276,  276,  276,  276,  276,  276,  276,  276,
 /*    90 */   276,  276,  276,  276,  276,  276,  276,  276,  276,  276,
 /*   100 */   276,  276,  276,  276,  276,  276,  276,  276,  276,  276,
 /*   110 */   276,  276,  276,  276,  276,  276,  276,  276,  276,  276,
 /*   120 */   276,  276,  276,  276,  276,  276,  276,  276,  276,  276,
 /*   130 */   223,  290,  296,  288,  291,  294,  287,  234,  254,  243,
 /*   140 */   255,  242,  233,  253,  251,  246,  244,  250,  232,  248,
 /*   150 */   249,  247,  252,  245,  236,  237,  239,  238,  241,  235,
 /*   160 */   240,  230,  231,  297,  297,  227,  297,  297,  225,  297,
 /*   170 */   297,  297,  297,  297,  258,  297,  297,  297,  297,  297,
 /*   180 */   297,  297,  297,  297,  297,  297,  297,  297,  297,  297,
 /*   190 */   222,  224,  279,  275,  261,  266,  273,  280,  260,  264,
 /*   200 */   274,  270,  283,  282,  268,  226,  256,  263,  229,  259,
 /*   210 */   257,  228,  265,  272,  277,  267,  269,  271,  262,
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
  "RBRACE",        "RPAREN",        "IN",            "RBRACKET",    
  "NEXT",          "BREAK",         "NEWLINE",       "SEMICOLON",   
  "COMMA",         "error",         "statementlist",  "sublist",     
  "sub",           "formallist",    "prog",          "optnl",       
  "statement",     "expr",          "symbolstr",   
};
#endif /* NDEBUG */

#ifndef NDEBUG
/* For tracing reduce actions, the names of all rules are required.
*/
static const char *const yyRuleName[] = {
 /*   0 */ "prog ::=",
 /*   1 */ "prog ::= optnl statementlist optnl",
 /*   2 */ "prog ::= error",
 /*   3 */ "statement ::= expr EQ_ASSIGN optnl statement",
 /*   4 */ "statement ::= expr",
 /*   5 */ "expr ::= NUM_CONST",
 /*   6 */ "expr ::= STR_CONST",
 /*   7 */ "expr ::= NULL_CONST",
 /*   8 */ "expr ::= SYMBOL",
 /*   9 */ "expr ::= LBRACE optnl statementlist optnl RBRACE",
 /*  10 */ "expr ::= LPAREN optnl statementlist optnl RPAREN",
 /*  11 */ "expr ::= MINUS optnl expr",
 /*  12 */ "expr ::= PLUS optnl expr",
 /*  13 */ "expr ::= NOT optnl expr",
 /*  14 */ "expr ::= TILDE optnl expr",
 /*  15 */ "expr ::= QUESTION optnl expr",
 /*  16 */ "expr ::= expr COLON optnl expr",
 /*  17 */ "expr ::= expr PLUS optnl expr",
 /*  18 */ "expr ::= expr MINUS optnl expr",
 /*  19 */ "expr ::= expr TIMES optnl expr",
 /*  20 */ "expr ::= expr DIVIDE optnl expr",
 /*  21 */ "expr ::= expr POW optnl expr",
 /*  22 */ "expr ::= expr SPECIALOP optnl expr",
 /*  23 */ "expr ::= expr TILDE optnl expr",
 /*  24 */ "expr ::= expr QUESTION optnl expr",
 /*  25 */ "expr ::= expr LT optnl expr",
 /*  26 */ "expr ::= expr LE optnl expr",
 /*  27 */ "expr ::= expr EQ optnl expr",
 /*  28 */ "expr ::= expr NE optnl expr",
 /*  29 */ "expr ::= expr GE optnl expr",
 /*  30 */ "expr ::= expr GT optnl expr",
 /*  31 */ "expr ::= expr AND optnl expr",
 /*  32 */ "expr ::= expr OR optnl expr",
 /*  33 */ "expr ::= expr AND2 optnl expr",
 /*  34 */ "expr ::= expr OR2 optnl expr",
 /*  35 */ "expr ::= expr LEFT_ASSIGN optnl expr",
 /*  36 */ "expr ::= expr RIGHT_ASSIGN optnl expr",
 /*  37 */ "expr ::= FUNCTION optnl LPAREN optnl formallist optnl RPAREN optnl statement",
 /*  38 */ "expr ::= expr LPAREN optnl sublist optnl RPAREN",
 /*  39 */ "expr ::= IF optnl LPAREN optnl expr optnl RPAREN optnl statement",
 /*  40 */ "expr ::= IF optnl LPAREN optnl expr optnl RPAREN optnl statement ELSE optnl statement",
 /*  41 */ "expr ::= FOR optnl LPAREN optnl SYMBOL optnl IN optnl expr optnl RPAREN optnl statement",
 /*  42 */ "expr ::= WHILE optnl LPAREN optnl expr optnl RPAREN optnl statement",
 /*  43 */ "expr ::= REPEAT optnl statement",
 /*  44 */ "expr ::= expr LBB optnl sublist optnl RBRACKET RBRACKET",
 /*  45 */ "expr ::= expr LBRACKET optnl sublist optnl RBRACKET",
 /*  46 */ "expr ::= SYMBOL NS_GET symbolstr",
 /*  47 */ "expr ::= STR_CONST NS_GET symbolstr",
 /*  48 */ "expr ::= SYMBOL NS_GET_INT symbolstr",
 /*  49 */ "expr ::= STR_CONST NS_GET_INT symbolstr",
 /*  50 */ "expr ::= expr DOLLAR optnl symbolstr",
 /*  51 */ "expr ::= expr AT optnl symbolstr",
 /*  52 */ "expr ::= NEXT",
 /*  53 */ "expr ::= BREAK",
 /*  54 */ "symbolstr ::= STR_CONST",
 /*  55 */ "symbolstr ::= SYMBOL",
 /*  56 */ "optnl ::= NEWLINE",
 /*  57 */ "optnl ::=",
 /*  58 */ "statementlist ::= statementlist SEMICOLON statement",
 /*  59 */ "statementlist ::= statementlist SEMICOLON",
 /*  60 */ "statementlist ::= statementlist NEWLINE statement",
 /*  61 */ "statementlist ::= statement",
 /*  62 */ "statementlist ::=",
 /*  63 */ "sublist ::= sub",
 /*  64 */ "sublist ::= sublist optnl COMMA optnl sub",
 /*  65 */ "sub ::=",
 /*  66 */ "sub ::= SYMBOL optnl EQ_ASSIGN",
 /*  67 */ "sub ::= STR_CONST optnl EQ_ASSIGN",
 /*  68 */ "sub ::= SYMBOL optnl EQ_ASSIGN optnl expr",
 /*  69 */ "sub ::= STR_CONST optnl EQ_ASSIGN optnl expr",
 /*  70 */ "sub ::= NULL_CONST optnl EQ_ASSIGN",
 /*  71 */ "sub ::= NULL_CONST optnl EQ_ASSIGN optnl expr",
 /*  72 */ "sub ::= expr",
 /*  73 */ "formallist ::=",
 /*  74 */ "formallist ::= SYMBOL",
 /*  75 */ "formallist ::= SYMBOL optnl EQ_ASSIGN optnl expr",
 /*  76 */ "formallist ::= formallist optnl COMMA optnl SYMBOL",
 /*  77 */ "formallist ::= formallist optnl COMMA optnl SYMBOL optnl EQ_ASSIGN optnl expr",
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
    case 47: /* RBRACKET */
    case 48: /* NEXT */
    case 49: /* BREAK */
    case 50: /* NEWLINE */
    case 51: /* SEMICOLON */
    case 52: /* COMMA */
{
#line 16 "grammar.y"
(void)parser;
#line 642 "grammar.c"
}
      break;
    case 54: /* statementlist */
    case 55: /* sublist */
    case 56: /* sub */
    case 57: /* formallist */
{
#line 19 "grammar.y"
delete (yypminor->yy122);
#line 652 "grammar.c"
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
#line 61 "grammar.y"

	parser->errors++;
	fprintf(stderr,"Giving up.  Parser stack overflow\n");
#line 829 "grammar.c"
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
  { 58, 0 },
  { 58, 3 },
  { 58, 1 },
  { 60, 4 },
  { 60, 1 },
  { 61, 1 },
  { 61, 1 },
  { 61, 1 },
  { 61, 1 },
  { 61, 5 },
  { 61, 5 },
  { 61, 3 },
  { 61, 3 },
  { 61, 3 },
  { 61, 3 },
  { 61, 3 },
  { 61, 4 },
  { 61, 4 },
  { 61, 4 },
  { 61, 4 },
  { 61, 4 },
  { 61, 4 },
  { 61, 4 },
  { 61, 4 },
  { 61, 4 },
  { 61, 4 },
  { 61, 4 },
  { 61, 4 },
  { 61, 4 },
  { 61, 4 },
  { 61, 4 },
  { 61, 4 },
  { 61, 4 },
  { 61, 4 },
  { 61, 4 },
  { 61, 4 },
  { 61, 4 },
  { 61, 9 },
  { 61, 6 },
  { 61, 9 },
  { 61, 12 },
  { 61, 13 },
  { 61, 9 },
  { 61, 3 },
  { 61, 7 },
  { 61, 6 },
  { 61, 3 },
  { 61, 3 },
  { 61, 3 },
  { 61, 3 },
  { 61, 4 },
  { 61, 4 },
  { 61, 1 },
  { 61, 1 },
  { 62, 1 },
  { 62, 1 },
  { 59, 1 },
  { 59, 0 },
  { 54, 3 },
  { 54, 2 },
  { 54, 3 },
  { 54, 1 },
  { 54, 0 },
  { 55, 1 },
  { 55, 5 },
  { 56, 0 },
  { 56, 3 },
  { 56, 3 },
  { 56, 5 },
  { 56, 5 },
  { 56, 3 },
  { 56, 5 },
  { 56, 1 },
  { 57, 0 },
  { 57, 1 },
  { 57, 5 },
  { 57, 5 },
  { 57, 9 },
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
      case 0: /* prog ::= */
      case 2: /* prog ::= error */ yytestcase(yyruleno==2);
#line 71 "grammar.y"
{ parser->result = Value::Nil(); }
#line 1022 "grammar.c"
        break;
      case 1: /* prog ::= optnl statementlist optnl */
#line 72 "grammar.y"
{ parser->result = CreateExpression(yymsp[-1].minor.yy122->values()); }
#line 1027 "grammar.c"
        break;
      case 3: /* statement ::= expr EQ_ASSIGN optnl statement */
      case 16: /* expr ::= expr COLON optnl expr */ yytestcase(yyruleno==16);
      case 17: /* expr ::= expr PLUS optnl expr */ yytestcase(yyruleno==17);
      case 18: /* expr ::= expr MINUS optnl expr */ yytestcase(yyruleno==18);
      case 19: /* expr ::= expr TIMES optnl expr */ yytestcase(yyruleno==19);
      case 20: /* expr ::= expr DIVIDE optnl expr */ yytestcase(yyruleno==20);
      case 21: /* expr ::= expr POW optnl expr */ yytestcase(yyruleno==21);
      case 22: /* expr ::= expr SPECIALOP optnl expr */ yytestcase(yyruleno==22);
      case 23: /* expr ::= expr TILDE optnl expr */ yytestcase(yyruleno==23);
      case 24: /* expr ::= expr QUESTION optnl expr */ yytestcase(yyruleno==24);
      case 25: /* expr ::= expr LT optnl expr */ yytestcase(yyruleno==25);
      case 26: /* expr ::= expr LE optnl expr */ yytestcase(yyruleno==26);
      case 27: /* expr ::= expr EQ optnl expr */ yytestcase(yyruleno==27);
      case 28: /* expr ::= expr NE optnl expr */ yytestcase(yyruleno==28);
      case 29: /* expr ::= expr GE optnl expr */ yytestcase(yyruleno==29);
      case 30: /* expr ::= expr GT optnl expr */ yytestcase(yyruleno==30);
      case 31: /* expr ::= expr AND optnl expr */ yytestcase(yyruleno==31);
      case 32: /* expr ::= expr OR optnl expr */ yytestcase(yyruleno==32);
      case 33: /* expr ::= expr AND2 optnl expr */ yytestcase(yyruleno==33);
      case 34: /* expr ::= expr OR2 optnl expr */ yytestcase(yyruleno==34);
      case 35: /* expr ::= expr LEFT_ASSIGN optnl expr */ yytestcase(yyruleno==35);
      case 51: /* expr ::= expr AT optnl symbolstr */ yytestcase(yyruleno==51);
#line 75 "grammar.y"
{ yygotominor.yy0 = CreateCall(List::c(yymsp[-2].minor.yy0, yymsp[-3].minor.yy0, yymsp[0].minor.yy0)); }
#line 1053 "grammar.c"
        break;
      case 4: /* statement ::= expr */
      case 5: /* expr ::= NUM_CONST */ yytestcase(yyruleno==5);
      case 6: /* expr ::= STR_CONST */ yytestcase(yyruleno==6);
      case 7: /* expr ::= NULL_CONST */ yytestcase(yyruleno==7);
      case 8: /* expr ::= SYMBOL */ yytestcase(yyruleno==8);
      case 54: /* symbolstr ::= STR_CONST */ yytestcase(yyruleno==54);
      case 55: /* symbolstr ::= SYMBOL */ yytestcase(yyruleno==55);
#line 76 "grammar.y"
{ yygotominor.yy0 = yymsp[0].minor.yy0; }
#line 1064 "grammar.c"
        break;
      case 9: /* expr ::= LBRACE optnl statementlist optnl RBRACE */
#line 83 "grammar.y"
{ yymsp[-2].minor.yy122->push_front(Strings::empty, yymsp[-4].minor.yy0); yygotominor.yy0 = CreateCall(yymsp[-2].minor.yy122->values(), yymsp[-2].minor.yy122->names(false));   yy_destructor(yypParser,44,&yymsp[0].minor);
}
#line 1070 "grammar.c"
        break;
      case 10: /* expr ::= LPAREN optnl statementlist optnl RPAREN */
#line 84 "grammar.y"
{ yymsp[-2].minor.yy122->push_front(Strings::empty, yymsp[-4].minor.yy0); yygotominor.yy0 = CreateCall(yymsp[-2].minor.yy122->values(), yymsp[-2].minor.yy122->names(false));   yy_destructor(yypParser,45,&yymsp[0].minor);
}
#line 1076 "grammar.c"
        break;
      case 11: /* expr ::= MINUS optnl expr */
      case 12: /* expr ::= PLUS optnl expr */ yytestcase(yyruleno==12);
      case 13: /* expr ::= NOT optnl expr */ yytestcase(yyruleno==13);
      case 14: /* expr ::= TILDE optnl expr */ yytestcase(yyruleno==14);
      case 15: /* expr ::= QUESTION optnl expr */ yytestcase(yyruleno==15);
      case 43: /* expr ::= REPEAT optnl statement */ yytestcase(yyruleno==43);
#line 86 "grammar.y"
{ yygotominor.yy0 = CreateCall(List::c(yymsp[-2].minor.yy0, yymsp[0].minor.yy0)); }
#line 1086 "grammar.c"
        break;
      case 36: /* expr ::= expr RIGHT_ASSIGN optnl expr */
#line 113 "grammar.y"
{ yygotominor.yy0 = CreateCall(List::c(yymsp[-2].minor.yy0, yymsp[0].minor.yy0, yymsp[-3].minor.yy0)); }
#line 1091 "grammar.c"
        break;
      case 37: /* expr ::= FUNCTION optnl LPAREN optnl formallist optnl RPAREN optnl statement */
#line 114 "grammar.y"
{ yygotominor.yy0 = CreateCall(List::c(yymsp[-8].minor.yy0, CreatePairlist(yymsp[-4].minor.yy122->values(), yymsp[-4].minor.yy122->names(true)), yymsp[0].minor.yy0, Character::c(parser->popSource())));   yy_destructor(yypParser,36,&yymsp[-6].minor);
  yy_destructor(yypParser,45,&yymsp[-2].minor);
}
#line 1098 "grammar.c"
        break;
      case 38: /* expr ::= expr LPAREN optnl sublist optnl RPAREN */
#line 115 "grammar.y"
{ yymsp[-2].minor.yy122->push_front(Strings::empty, yymsp[-5].minor.yy0); yygotominor.yy0 = CreateCall(yymsp[-2].minor.yy122->values(), yymsp[-2].minor.yy122->names(false));   yy_destructor(yypParser,36,&yymsp[-4].minor);
  yy_destructor(yypParser,45,&yymsp[0].minor);
}
#line 1105 "grammar.c"
        break;
      case 39: /* expr ::= IF optnl LPAREN optnl expr optnl RPAREN optnl statement */
      case 42: /* expr ::= WHILE optnl LPAREN optnl expr optnl RPAREN optnl statement */ yytestcase(yyruleno==42);
#line 116 "grammar.y"
{ yygotominor.yy0 = CreateCall(List::c(yymsp[-8].minor.yy0, yymsp[-4].minor.yy0, yymsp[0].minor.yy0));   yy_destructor(yypParser,36,&yymsp[-6].minor);
  yy_destructor(yypParser,45,&yymsp[-2].minor);
}
#line 1113 "grammar.c"
        break;
      case 40: /* expr ::= IF optnl LPAREN optnl expr optnl RPAREN optnl statement ELSE optnl statement */
#line 117 "grammar.y"
{ yygotominor.yy0 = CreateCall(List::c(yymsp[-11].minor.yy0, yymsp[-7].minor.yy0, yymsp[-3].minor.yy0, yymsp[0].minor.yy0));   yy_destructor(yypParser,36,&yymsp[-9].minor);
  yy_destructor(yypParser,45,&yymsp[-5].minor);
  yy_destructor(yypParser,7,&yymsp[-2].minor);
}
#line 1121 "grammar.c"
        break;
      case 41: /* expr ::= FOR optnl LPAREN optnl SYMBOL optnl IN optnl expr optnl RPAREN optnl statement */
#line 118 "grammar.y"
{ yygotominor.yy0 = CreateCall(List::c(yymsp[-12].minor.yy0, yymsp[-8].minor.yy0, yymsp[-4].minor.yy0, yymsp[0].minor.yy0));   yy_destructor(yypParser,36,&yymsp[-10].minor);
  yy_destructor(yypParser,46,&yymsp[-6].minor);
  yy_destructor(yypParser,45,&yymsp[-2].minor);
}
#line 1129 "grammar.c"
        break;
      case 44: /* expr ::= expr LBB optnl sublist optnl RBRACKET RBRACKET */
#line 121 "grammar.y"
{ yymsp[-3].minor.yy122->push_front(Strings::empty, yymsp[-6].minor.yy0); yymsp[-3].minor.yy122->push_front(Strings::empty, yymsp[-5].minor.yy0); yygotominor.yy0 = CreateCall(yymsp[-3].minor.yy122->values(), yymsp[-3].minor.yy122->names(false));   yy_destructor(yypParser,47,&yymsp[-1].minor);
  yy_destructor(yypParser,47,&yymsp[0].minor);
}
#line 1136 "grammar.c"
        break;
      case 45: /* expr ::= expr LBRACKET optnl sublist optnl RBRACKET */
#line 122 "grammar.y"
{ yymsp[-2].minor.yy122->push_front(Strings::empty, yymsp[-5].minor.yy0); yymsp[-2].minor.yy122->push_front(Strings::empty, yymsp[-4].minor.yy0); yygotominor.yy0 = CreateCall(yymsp[-2].minor.yy122->values(), yymsp[-2].minor.yy122->names(false));   yy_destructor(yypParser,47,&yymsp[0].minor);
}
#line 1142 "grammar.c"
        break;
      case 46: /* expr ::= SYMBOL NS_GET symbolstr */
      case 47: /* expr ::= STR_CONST NS_GET symbolstr */ yytestcase(yyruleno==47);
      case 48: /* expr ::= SYMBOL NS_GET_INT symbolstr */ yytestcase(yyruleno==48);
      case 49: /* expr ::= STR_CONST NS_GET_INT symbolstr */ yytestcase(yyruleno==49);
#line 123 "grammar.y"
{ yygotominor.yy0 = CreateCall(List::c(yymsp[-1].minor.yy0, yymsp[-2].minor.yy0, yymsp[0].minor.yy0)); }
#line 1150 "grammar.c"
        break;
      case 50: /* expr ::= expr DOLLAR optnl symbolstr */
#line 127 "grammar.y"
{ if(isSymbol(yymsp[0].minor.yy0)) yymsp[0].minor.yy0 = Character::c(SymbolStr(yymsp[0].minor.yy0)); yygotominor.yy0 = CreateCall(List::c(yymsp[-2].minor.yy0, yymsp[-3].minor.yy0, yymsp[0].minor.yy0)); }
#line 1155 "grammar.c"
        break;
      case 52: /* expr ::= NEXT */
      case 53: /* expr ::= BREAK */ yytestcase(yyruleno==53);
#line 129 "grammar.y"
{ yygotominor.yy0 = CreateCall(List::c(yymsp[0].minor.yy0)); }
#line 1161 "grammar.c"
        break;
      case 56: /* optnl ::= NEWLINE */
#line 135 "grammar.y"
{
  yy_destructor(yypParser,50,&yymsp[0].minor);
}
#line 1168 "grammar.c"
        break;
      case 58: /* statementlist ::= statementlist SEMICOLON statement */
#line 138 "grammar.y"
{ yygotominor.yy122 = yymsp[-2].minor.yy122; yygotominor.yy122->push_back(Strings::empty, yymsp[0].minor.yy0);   yy_destructor(yypParser,51,&yymsp[-1].minor);
}
#line 1174 "grammar.c"
        break;
      case 59: /* statementlist ::= statementlist SEMICOLON */
#line 139 "grammar.y"
{ yygotominor.yy122 = yymsp[-1].minor.yy122;   yy_destructor(yypParser,51,&yymsp[0].minor);
}
#line 1180 "grammar.c"
        break;
      case 60: /* statementlist ::= statementlist NEWLINE statement */
#line 140 "grammar.y"
{ yygotominor.yy122 = yymsp[-2].minor.yy122; yygotominor.yy122->push_back(Strings::empty, yymsp[0].minor.yy0);   yy_destructor(yypParser,50,&yymsp[-1].minor);
}
#line 1186 "grammar.c"
        break;
      case 61: /* statementlist ::= statement */
      case 72: /* sub ::= expr */ yytestcase(yyruleno==72);
#line 141 "grammar.y"
{ yygotominor.yy122 = new Pairs(); yygotominor.yy122->push_back(Strings::empty, yymsp[0].minor.yy0); }
#line 1192 "grammar.c"
        break;
      case 62: /* statementlist ::= */
      case 65: /* sub ::= */ yytestcase(yyruleno==65);
      case 73: /* formallist ::= */ yytestcase(yyruleno==73);
#line 142 "grammar.y"
{ yygotominor.yy122 = new Pairs(); }
#line 1199 "grammar.c"
        break;
      case 63: /* sublist ::= sub */
#line 144 "grammar.y"
{ yygotominor.yy122 = yymsp[0].minor.yy122; }
#line 1204 "grammar.c"
        break;
      case 64: /* sublist ::= sublist optnl COMMA optnl sub */
#line 145 "grammar.y"
{ yygotominor.yy122 = yymsp[-4].minor.yy122; if(yygotominor.yy122->length() == 0) yygotominor.yy122->push_back(Strings::empty, Value::Nil()); if(yymsp[0].minor.yy122->length() == 1) yygotominor.yy122->push_back(yymsp[0].minor.yy122->name(0), yymsp[0].minor.yy122->value(0)); else if(yymsp[0].minor.yy122->length() == 0) yygotominor.yy122->push_back(Strings::empty, Value::Nil());   yy_destructor(yypParser,52,&yymsp[-2].minor);
}
#line 1210 "grammar.c"
        break;
      case 66: /* sub ::= SYMBOL optnl EQ_ASSIGN */
      case 67: /* sub ::= STR_CONST optnl EQ_ASSIGN */ yytestcase(yyruleno==67);
#line 148 "grammar.y"
{ yygotominor.yy122 = new Pairs(); yygotominor.yy122->push_back(SymbolStr(yymsp[-2].minor.yy0), Value::Nil());   yy_destructor(yypParser,9,&yymsp[0].minor);
}
#line 1217 "grammar.c"
        break;
      case 68: /* sub ::= SYMBOL optnl EQ_ASSIGN optnl expr */
      case 69: /* sub ::= STR_CONST optnl EQ_ASSIGN optnl expr */ yytestcase(yyruleno==69);
      case 75: /* formallist ::= SYMBOL optnl EQ_ASSIGN optnl expr */ yytestcase(yyruleno==75);
#line 150 "grammar.y"
{ yygotominor.yy122 = new Pairs(); yygotominor.yy122->push_back(SymbolStr(yymsp[-4].minor.yy0), yymsp[0].minor.yy0);   yy_destructor(yypParser,9,&yymsp[-2].minor);
}
#line 1225 "grammar.c"
        break;
      case 70: /* sub ::= NULL_CONST optnl EQ_ASSIGN */
#line 152 "grammar.y"
{ yygotominor.yy122 = new Pairs(); yygotominor.yy122->push_back(Strings::Null, Value::Nil());   yy_destructor(yypParser,42,&yymsp[-2].minor);
  yy_destructor(yypParser,9,&yymsp[0].minor);
}
#line 1232 "grammar.c"
        break;
      case 71: /* sub ::= NULL_CONST optnl EQ_ASSIGN optnl expr */
#line 153 "grammar.y"
{ yygotominor.yy122 = new Pairs(); yygotominor.yy122->push_back(Strings::Null, yymsp[0].minor.yy0);   yy_destructor(yypParser,42,&yymsp[-4].minor);
  yy_destructor(yypParser,9,&yymsp[-2].minor);
}
#line 1239 "grammar.c"
        break;
      case 74: /* formallist ::= SYMBOL */
#line 157 "grammar.y"
{ yygotominor.yy122 = new Pairs(); yygotominor.yy122->push_back(SymbolStr(yymsp[0].minor.yy0), Value::Nil()); }
#line 1244 "grammar.c"
        break;
      case 76: /* formallist ::= formallist optnl COMMA optnl SYMBOL */
#line 159 "grammar.y"
{ yygotominor.yy122 = yymsp[-4].minor.yy122; yygotominor.yy122->push_back(SymbolStr(yymsp[0].minor.yy0), Value::Nil());   yy_destructor(yypParser,52,&yymsp[-2].minor);
}
#line 1250 "grammar.c"
        break;
      case 77: /* formallist ::= formallist optnl COMMA optnl SYMBOL optnl EQ_ASSIGN optnl expr */
#line 160 "grammar.y"
{ yygotominor.yy122 = yymsp[-8].minor.yy122; yygotominor.yy122->push_back(SymbolStr(yymsp[-4].minor.yy0), yymsp[0].minor.yy0);   yy_destructor(yypParser,52,&yymsp[-6].minor);
  yy_destructor(yypParser,9,&yymsp[-2].minor);
}
#line 1257 "grammar.c"
        break;
      default:
      /* (57) optnl ::= */ yytestcase(yyruleno==57);
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
#line 53 "grammar.y"

        parser->errors++;
#line 1323 "grammar.c"
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
#line 57 "grammar.y"

	parser->complete = true;
#line 1345 "grammar.c"
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
