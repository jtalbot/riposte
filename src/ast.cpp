/* Driver template for the LEMON parser generator.
** The author disclaims copyright to this source code.
*/
/* First off, code is included that follows the "include" declaration
** in the input grammar file. */
#include <stdio.h>
#line 41 "parser.y"

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
#define YYNOCODE 64
#define YYACTIONTYPE unsigned short int
#define ParseTOKENTYPE Value
typedef union {
  int yyinit;
  ParseTOKENTYPE yy0;
  Pairs yy72;
  int yy127;
} YYMINORTYPE;
#ifndef YYSTACKDEPTH
#define YYSTACKDEPTH 100
#endif
#define ParseARG_SDECL Parser::Result* result;
#define ParseARG_PDECL ,Parser::Result* result
#define ParseARG_FETCH Parser::Result* result = yypParser->result
#define ParseARG_STORE yypParser->result = result
#define YYNSTATE 216
#define YYNRULE 75
#define YYERRORSYMBOL 54
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
 /*     0 */    92,  110,  200,  213,   11,  212,  137,  103,  104,  105,
 /*    10 */    91,  100,  102,   99,  101,  202,   98,   97,   93,   94,
 /*    20 */    95,   96,   81,   83,   85,   87,   90,   79,   68,  200,
 /*    30 */    89,  109,  111,  137,   13,    1,  108,  106,   90,   79,
 /*    40 */    77,  200,   89,  109,  111,  137,  116,    1,  108,  106,
 /*    50 */   202,  176,   92,   16,   73,   18,  292,    5,   45,  103,
 /*    60 */   104,  105,   91,  100,  102,   99,  101,   47,   98,   97,
 /*    70 */    93,   94,   95,   96,   81,   83,   85,   87,   90,   79,
 /*    80 */   271,  199,   89,  109,  111,  137,    3,    1,  108,  106,
 /*    90 */   120,  103,  104,  105,   91,  100,  102,   99,  101,  181,
 /*   100 */    98,   97,   93,   94,   95,   96,   81,   83,   85,   87,
 /*   110 */    90,   79,  205,    2,   89,  109,  111,   70,  202,    1,
 /*   120 */   108,  106,  202,  103,  104,  105,   91,  100,  102,   99,
 /*   130 */   101,   66,   98,   97,   93,   94,   95,   96,   81,   83,
 /*   140 */    85,   87,   90,   79,   24,   19,   89,  109,  111,  174,
 /*   150 */    62,    1,  108,  106,   91,  100,  102,   99,  101,   15,
 /*   160 */    98,   97,   93,   94,   95,   96,   81,   83,   85,   87,
 /*   170 */    90,   79,   63,   60,   89,  109,  111,  114,  115,    1,
 /*   180 */   108,  106,  100,  102,   99,  101,  171,   98,   97,   93,
 /*   190 */    94,   95,   96,   81,   83,   85,   87,   90,   79,  203,
 /*   200 */    37,   89,  109,  111,  184,   72,    1,  108,  106,   99,
 /*   210 */   101,   35,   98,   97,   93,   94,   95,   96,   81,   83,
 /*   220 */    85,   87,   90,   79,   64,   53,   89,  109,  111,  152,
 /*   230 */   133,    1,  108,  106,   88,   71,  126,  121,  112,  117,
 /*   240 */    85,   87,   90,   79,   86,  215,   89,  109,  111,   84,
 /*   250 */    72,    1,  108,  106,   50,   79,   82,   80,   89,  109,
 /*   260 */   111,   67,  186,    1,  108,  106,  170,  133,  187,   78,
 /*   270 */    61,   69,   75,  206,  172,  210,  175,  188,   41,   32,
 /*   280 */   190,   28,  197,  196,   88,   71,  126,  121,  112,  117,
 /*   290 */    89,  109,  111,  155,   86,    1,  108,  106,   54,   84,
 /*   300 */   189,  185,  285,  191,  285,  285,   82,   80,  202,  282,
 /*   310 */   285,  282,  282,  179,   14,  202,  130,  282,   26,   78,
 /*   320 */   207,  141,   75,  206,   58,  107,   59,  151,  168,  139,
 /*   330 */   129,  153,  197,  196,   10,  159,  158,   98,   97,   93,
 /*   340 */    94,   95,   96,   81,   83,   85,   87,   90,   79,  134,
 /*   350 */   157,   89,  109,  111,  135,  156,    1,  108,  106,   63,
 /*   360 */    60,  281,    9,  281,  281,  165,  132,  202,  150,  281,
 /*   370 */   145,  164,   61,   69,  144,    8,  202,  208,  167,  143,
 /*   380 */   293,  214,  293,  166,  149,  146,  293,    7,  161,  202,
 /*   390 */   131,  160,  140,  136,  148,  163,  169,  142,  162,  217,
 /*   400 */   154,  138,  147,   43,  180,  182,  216,   55,  198,  209,
 /*   410 */   178,  193,  194,  177,  195,  293,   56,  211,  201,  183,
 /*   420 */     4,  192,   46,    6,   30,  173,   21,   12,   17,   48,
 /*   430 */    49,   51,   52,   29,   31,   33,   34,   36,   38,   39,
 /*   440 */    40,  204,   42,   57,   44,   20,   22,  113,   23,   25,
 /*   450 */    27,  118,  123,  119,  122,  127,  124,  125,  128,   74,
 /*   460 */    65,   76,  293,  293,  293,  293,  293,   72,
};
static const YYCODETYPE yy_lookahead[] = {
 /*     0 */     1,   56,   57,   41,   60,   43,   61,    8,    9,   10,
 /*    10 */    11,   12,   13,   14,   15,   51,   17,   18,   19,   20,
 /*    20 */    21,   22,   23,   24,   25,   26,   27,   28,   56,   57,
 /*    30 */    31,   32,   33,   61,   60,   36,   37,   38,   27,   28,
 /*    40 */    56,   57,   31,   32,   33,   61,   45,   36,   37,   38,
 /*    50 */    51,   54,    1,   60,   53,   60,   59,   60,   60,    8,
 /*    60 */     9,   10,   11,   12,   13,   14,   15,   60,   17,   18,
 /*    70 */    19,   20,   21,   22,   23,   24,   25,   26,   27,   28,
 /*    80 */     9,   57,   31,   32,   33,   61,   60,   36,   37,   38,
 /*    90 */     7,    8,    9,   10,   11,   12,   13,   14,   15,   60,
 /*   100 */    17,   18,   19,   20,   21,   22,   23,   24,   25,   26,
 /*   110 */    27,   28,   45,   60,   31,   32,   33,   60,   51,   36,
 /*   120 */    37,   38,   51,    8,    9,   10,   11,   12,   13,   14,
 /*   130 */    15,   43,   17,   18,   19,   20,   21,   22,   23,   24,
 /*   140 */    25,   26,   27,   28,   51,   52,   31,   32,   33,   60,
 /*   150 */    60,   36,   37,   38,   11,   12,   13,   14,   15,   60,
 /*   160 */    17,   18,   19,   20,   21,   22,   23,   24,   25,   26,
 /*   170 */    27,   28,   34,   35,   31,   32,   33,   60,   58,   36,
 /*   180 */    37,   38,   12,   13,   14,   15,   60,   17,   18,   19,
 /*   190 */    20,   21,   22,   23,   24,   25,   26,   27,   28,   47,
 /*   200 */    60,   31,   32,   33,   60,   53,   36,   37,   38,   14,
 /*   210 */    15,   60,   17,   18,   19,   20,   21,   22,   23,   24,
 /*   220 */    25,   26,   27,   28,   55,   60,   31,   32,   33,   61,
 /*   230 */    61,   36,   37,   38,    1,    2,    3,    4,    5,    6,
 /*   240 */    25,   26,   27,   28,   11,   48,   31,   32,   33,   16,
 /*   250 */    53,   36,   37,   38,   60,   28,   23,   24,   31,   32,
 /*   260 */    33,   55,   60,   36,   37,   38,   61,   61,   60,   36,
 /*   270 */    34,   35,   39,   40,   41,   42,   43,   60,   60,   60,
 /*   280 */    60,   60,   49,   50,    1,    2,    3,    4,    5,    6,
 /*   290 */    31,   32,   33,   61,   11,   36,   37,   38,   60,   16,
 /*   300 */    60,   60,   45,   60,   47,   48,   23,   24,   51,   45,
 /*   310 */    53,   47,   48,   60,   60,   51,   61,   53,   60,   36,
 /*   320 */    44,   61,   39,   40,   41,   42,   43,   61,   61,   61,
 /*   330 */    61,   61,   49,   50,   61,   61,   61,   17,   18,   19,
 /*   340 */    20,   21,   22,   23,   24,   25,   26,   27,   28,   61,
 /*   350 */    61,   31,   32,   33,   61,   61,   36,   37,   38,   34,
 /*   360 */    35,   45,   61,   47,   48,   61,   61,   51,   61,   53,
 /*   370 */    61,   61,   34,   35,   61,   61,   51,   62,   61,   61,
 /*   380 */    63,   62,   63,   61,   61,   61,   63,   61,   61,   51,
 /*   390 */    61,   61,   61,   61,   61,   61,   61,   61,   61,    0,
 /*   400 */    61,   61,   61,   60,   60,   60,    0,    9,   62,   62,
 /*   410 */    60,   60,   60,   60,   60,   63,    9,   62,   62,   60,
 /*   420 */    60,   60,   60,   60,   60,   60,   60,   60,   60,   60,
 /*   430 */    60,   60,   60,   60,   60,   60,   60,   60,   60,   60,
 /*   440 */    60,   45,   60,    9,   60,   60,   60,   36,   60,   60,
 /*   450 */    60,   36,   43,   45,   36,   36,   46,   45,   45,    9,
 /*   460 */    43,    9,   63,   63,   63,   63,   63,   53,
};
#define YY_SHIFT_USE_DFLT (-39)
#define YY_SHIFT_MAX 195
static const short yy_shift_ofst[] = {
 /*     0 */   -36,  283,  283,  283,  283,  233,  233,   -1,   -1,   -1,
 /*    10 */    -1,  233,  233,  233,  233,  233,  233,  233,  233,  233,
 /*    20 */   233,  233,  233,  233,  233,  233,  233,  233,  233,  233,
 /*    30 */   233,  233,  233,  233,  233,  233,  233,  233,  233,  233,
 /*    40 */   233,  233,  233,  233,  233,  233,  233,  233,  233,  233,
 /*    50 */   233,  233,  233,  233,  233,  264,  257,  316,  338,  325,
 /*    60 */   -38,  -38,  -38,  -38,   93,   71,   71,   93,   67,  -38,
 /*    70 */   -38,  -36,  -36,  -36,  -36,  -36,  -36,  -36,  -36,  -36,
 /*    80 */   -36,  -36,  -36,  -36,  -36,  -36,  -36,  -36,  -36,  -36,
 /*    90 */   -36,  -36,  -36,  -36,  -36,  -36,  -36,  -36,  -36,  -36,
 /*   100 */   -36,  -36,  -36,  -36,  -36,  -36,  -36,  -36,  -36,  -36,
 /*   110 */   -36,  -36,  -36,  -36,   88,  -36,  -36,  -36,  -36,  -36,
 /*   120 */   -36,  -36,  -36,  -36,  -36,  -36,  -36,  -36,  -36,   51,
 /*   130 */    51,   51,   51,   51,   51,   51,   51,   51,   83,  115,
 /*   140 */   115,  115,  115,  115,  115,  115,  115,  115,  143,  170,
 /*   150 */   170,  195,  195,  320,  320,  320,  320,  320,  320,  320,
 /*   160 */   320,  320,  215,  215,   11,   11,  227,  259,  259,  259,
 /*   170 */   259,    1,  236,  197,  152,  138,  399,  406,  276,  396,
 /*   180 */   398,  407,  434,  411,  415,  408,  418,  409,  410,  412,
 /*   190 */   419,  413,  417,  450,  452,  414,
};
#define YY_REDUCE_USE_DFLT (-57)
#define YY_REDUCE_MAX 128
static const short yy_reduce_ofst[] = {
 /*     0 */    -3,  -28,  -16,  -55,   24,  206,  169,  240,  241,  243,
 /*    10 */   253,  168,  205,  232,  255,  260,  266,  267,  268,  269,
 /*    20 */   270,  273,  274,  275,  288,  289,  293,  294,  301,  304,
 /*    30 */   305,  307,  309,  310,  313,  314,  317,  318,  322,  323,
 /*    40 */   324,  326,  327,  329,  330,  331,  332,  333,  334,  335,
 /*    50 */   336,  337,  339,  340,  341,  254,  258,  343,  344,  345,
 /*    60 */   315,  319,  346,  347,  350,  351,  352,  353,  354,  355,
 /*    70 */   356,  359,  360,  361,  362,  363,  364,  365,  366,  367,
 /*    80 */   368,  369,  370,  371,  372,  373,  374,  375,  376,  377,
 /*    90 */   378,  379,  380,  382,  384,  385,  386,  388,  389,  390,
 /*   100 */   -56,  -26,   -7,   -5,   -2,    7,   26,   39,   53,   57,
 /*   110 */    89,   90,   99,  117,  120,  126,  140,  144,  151,  165,
 /*   120 */   194,  202,  208,  217,  218,  219,  220,  221,  238,
};
static const YYACTIONTYPE yy_default[] = {
 /*     0 */   271,  279,  279,  279,  279,  276,  276,  271,  271,  271,
 /*    10 */   271,  291,  291,  291,  291,  291,  291,  291,  291,  273,
 /*    20 */   291,  291,  291,  291,  270,  291,  291,  291,  291,  291,
 /*    30 */   291,  291,  291,  291,  291,  291,  291,  291,  291,  291,
 /*    40 */   291,  291,  291,  291,  291,  291,  291,  291,  291,  291,
 /*    50 */   291,  291,  291,  291,  291,  271,  271,  271,  219,  221,
 /*    60 */   291,  291,  291,  291,  271,  289,  287,  271,  271,  291,
 /*    70 */   291,  271,  271,  271,  271,  271,  271,  271,  271,  271,
 /*    80 */   271,  271,  271,  271,  271,  271,  271,  271,  271,  271,
 /*    90 */   271,  271,  271,  271,  271,  271,  271,  271,  271,  271,
 /*   100 */   271,  271,  271,  271,  271,  271,  271,  220,  271,  271,
 /*   110 */   271,  271,  271,  271,  291,  271,  271,  271,  271,  271,
 /*   120 */   271,  271,  271,  271,  271,  271,  271,  271,  271,  272,
 /*   130 */   284,  283,  288,  275,  274,  286,  290,  280,  253,  248,
 /*   140 */   249,  257,  254,  251,  228,  255,  237,  256,  250,  236,
 /*   150 */   227,  247,  245,  240,  226,  246,  244,  243,  242,  241,
 /*   160 */   239,  238,  231,  230,  233,  232,  235,  234,  224,  225,
 /*   170 */   229,  291,  219,  291,  291,  221,  291,  291,  291,  291,
 /*   180 */   291,  291,  291,  291,  291,  291,  291,  291,  291,  291,
 /*   190 */   291,  291,  291,  291,  291,  291,  267,  266,  265,  278,
 /*   200 */   277,  264,  270,  258,  223,  252,  218,  222,  262,  260,
 /*   210 */   220,  263,  269,  268,  261,  259,
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
  "SEMICOLON",     "COMMA",         "error",         "exprlist",    
  "sublist",       "sub",           "formallist",    "prog",        
  "optnl",         "expr",          "symbolstr",   
};
#endif /* NDEBUG */

#ifndef NDEBUG
/* For tracing reduce actions, the names of all rules are required.
*/
static const char *const yyRuleName[] = {
 /*   0 */ "prog ::= optnl exprlist optnl",
 /*   1 */ "prog ::= error",
 /*   2 */ "expr ::= NUM_CONST",
 /*   3 */ "expr ::= STR_CONST",
 /*   4 */ "expr ::= NULL_CONST",
 /*   5 */ "expr ::= SYMBOL",
 /*   6 */ "expr ::= LBRACE optnl exprlist optnl RBRACE",
 /*   7 */ "expr ::= LPAREN optnl expr optnl RPAREN",
 /*   8 */ "expr ::= MINUS optnl expr",
 /*   9 */ "expr ::= PLUS optnl expr",
 /*  10 */ "expr ::= NOT optnl expr",
 /*  11 */ "expr ::= TILDE optnl expr",
 /*  12 */ "expr ::= QUESTION optnl expr",
 /*  13 */ "expr ::= expr COLON optnl expr",
 /*  14 */ "expr ::= expr PLUS optnl expr",
 /*  15 */ "expr ::= expr MINUS optnl expr",
 /*  16 */ "expr ::= expr TIMES optnl expr",
 /*  17 */ "expr ::= expr DIVIDE optnl expr",
 /*  18 */ "expr ::= expr POW optnl expr",
 /*  19 */ "expr ::= expr SPECIALOP optnl expr",
 /*  20 */ "expr ::= expr TILDE optnl expr",
 /*  21 */ "expr ::= expr QUESTION optnl expr",
 /*  22 */ "expr ::= expr LT optnl expr",
 /*  23 */ "expr ::= expr LE optnl expr",
 /*  24 */ "expr ::= expr EQ optnl expr",
 /*  25 */ "expr ::= expr NE optnl expr",
 /*  26 */ "expr ::= expr GE optnl expr",
 /*  27 */ "expr ::= expr GT optnl expr",
 /*  28 */ "expr ::= expr AND optnl expr",
 /*  29 */ "expr ::= expr OR optnl expr",
 /*  30 */ "expr ::= expr AND2 optnl expr",
 /*  31 */ "expr ::= expr OR2 optnl expr",
 /*  32 */ "expr ::= expr LEFT_ASSIGN optnl expr",
 /*  33 */ "expr ::= expr EQ_ASSIGN optnl expr",
 /*  34 */ "expr ::= expr RIGHT_ASSIGN optnl expr",
 /*  35 */ "expr ::= FUNCTION optnl LPAREN optnl formallist optnl RPAREN optnl expr",
 /*  36 */ "expr ::= expr LPAREN sublist RPAREN",
 /*  37 */ "expr ::= IF optnl LPAREN optnl expr optnl RPAREN optnl expr",
 /*  38 */ "expr ::= IF optnl LPAREN optnl expr optnl RPAREN optnl expr ELSE optnl expr",
 /*  39 */ "expr ::= FOR optnl LPAREN optnl SYMBOL optnl IN optnl expr optnl RPAREN optnl expr",
 /*  40 */ "expr ::= WHILE optnl LPAREN optnl expr optnl RPAREN optnl expr",
 /*  41 */ "expr ::= REPEAT optnl expr",
 /*  42 */ "expr ::= expr LBB optnl sublist optnl RBB",
 /*  43 */ "expr ::= expr LBRACKET optnl sublist optnl RBRACKET",
 /*  44 */ "expr ::= SYMBOL NS_GET symbolstr",
 /*  45 */ "expr ::= STR_CONST NS_GET symbolstr",
 /*  46 */ "expr ::= SYMBOL NS_GET_INT symbolstr",
 /*  47 */ "expr ::= STR_CONST NS_GET_INT symbolstr",
 /*  48 */ "expr ::= expr DOLLAR optnl symbolstr",
 /*  49 */ "expr ::= expr AT optnl symbolstr",
 /*  50 */ "expr ::= NEXT",
 /*  51 */ "expr ::= BREAK",
 /*  52 */ "symbolstr ::= STR_CONST",
 /*  53 */ "symbolstr ::= SYMBOL",
 /*  54 */ "optnl ::= NEWLINE",
 /*  55 */ "optnl ::=",
 /*  56 */ "exprlist ::= exprlist SEMICOLON expr",
 /*  57 */ "exprlist ::= exprlist SEMICOLON",
 /*  58 */ "exprlist ::= exprlist NEWLINE expr",
 /*  59 */ "exprlist ::= expr",
 /*  60 */ "exprlist ::=",
 /*  61 */ "sublist ::= sub",
 /*  62 */ "sublist ::= sublist optnl COMMA optnl sub",
 /*  63 */ "sub ::=",
 /*  64 */ "sub ::= expr",
 /*  65 */ "sub ::= SYMBOL optnl EQ_ASSIGN",
 /*  66 */ "sub ::= STR_CONST optnl EQ_ASSIGN",
 /*  67 */ "sub ::= SYMBOL optnl EQ_ASSIGN optnl expr",
 /*  68 */ "sub ::= STR_CONST optnl EQ_ASSIGN optnl expr",
 /*  69 */ "sub ::= NULL_CONST optnl EQ_ASSIGN",
 /*  70 */ "sub ::= NULL_CONST optnl EQ_ASSIGN optnl expr",
 /*  71 */ "formallist ::= SYMBOL",
 /*  72 */ "formallist ::= SYMBOL optnl EQ_ASSIGN optnl expr",
 /*  73 */ "formallist ::= formallist optnl COMMA optnl SYMBOL",
 /*  74 */ "formallist ::= formallist optnl COMMA optnl SYMBOL optnl EQ_ASSIGN optnl expr",
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
  { 61, 4 },
  { 61, 9 },
  { 61, 4 },
  { 61, 9 },
  { 61, 12 },
  { 61, 13 },
  { 61, 9 },
  { 61, 3 },
  { 61, 6 },
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
  { 57, 1 },
  { 57, 3 },
  { 57, 3 },
  { 57, 5 },
  { 57, 5 },
  { 57, 3 },
  { 57, 5 },
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
      case 0: /* prog ::= optnl exprlist optnl */
#line 60 "parser.y"
{ result->value = yygotominor.yy0 = Expression(List(yymsp[-1].minor.yy72)); }
#line 940 "parser.c"
        break;
      case 1: /* prog ::= error */
#line 61 "parser.y"
{ result->value = yygotominor.yy0 = Expression(0); }
#line 945 "parser.c"
        break;
      case 2: /* expr ::= NUM_CONST */
      case 3: /* expr ::= STR_CONST */ yytestcase(yyruleno==3);
      case 4: /* expr ::= NULL_CONST */ yytestcase(yyruleno==4);
      case 5: /* expr ::= SYMBOL */ yytestcase(yyruleno==5);
      case 52: /* symbolstr ::= STR_CONST */ yytestcase(yyruleno==52);
      case 53: /* symbolstr ::= SYMBOL */ yytestcase(yyruleno==53);
#line 63 "parser.y"
{ yygotominor.yy0 = yymsp[0].minor.yy0; }
#line 955 "parser.c"
        break;
      case 6: /* expr ::= LBRACE optnl exprlist optnl RBRACE */
#line 68 "parser.y"
{ yymsp[-2].minor.yy72.push_front(Symbol(0), yymsp[-4].minor.yy0); yygotominor.yy0 = Expression(List(yymsp[-2].minor.yy72)); }
#line 960 "parser.c"
        break;
      case 7: /* expr ::= LPAREN optnl expr optnl RPAREN */
#line 69 "parser.y"
{ yygotominor.yy0 = yymsp[-2].minor.yy0; }
#line 965 "parser.c"
        break;
      case 8: /* expr ::= MINUS optnl expr */
      case 9: /* expr ::= PLUS optnl expr */ yytestcase(yyruleno==9);
      case 10: /* expr ::= NOT optnl expr */ yytestcase(yyruleno==10);
      case 11: /* expr ::= TILDE optnl expr */ yytestcase(yyruleno==11);
      case 12: /* expr ::= QUESTION optnl expr */ yytestcase(yyruleno==12);
      case 41: /* expr ::= REPEAT optnl expr */ yytestcase(yyruleno==41);
#line 71 "parser.y"
{ yygotominor.yy0 = Call::c(yymsp[-2].minor.yy0, yymsp[0].minor.yy0); }
#line 975 "parser.c"
        break;
      case 13: /* expr ::= expr COLON optnl expr */
      case 14: /* expr ::= expr PLUS optnl expr */ yytestcase(yyruleno==14);
      case 15: /* expr ::= expr MINUS optnl expr */ yytestcase(yyruleno==15);
      case 16: /* expr ::= expr TIMES optnl expr */ yytestcase(yyruleno==16);
      case 17: /* expr ::= expr DIVIDE optnl expr */ yytestcase(yyruleno==17);
      case 18: /* expr ::= expr POW optnl expr */ yytestcase(yyruleno==18);
      case 19: /* expr ::= expr SPECIALOP optnl expr */ yytestcase(yyruleno==19);
      case 20: /* expr ::= expr TILDE optnl expr */ yytestcase(yyruleno==20);
      case 21: /* expr ::= expr QUESTION optnl expr */ yytestcase(yyruleno==21);
      case 22: /* expr ::= expr LT optnl expr */ yytestcase(yyruleno==22);
      case 23: /* expr ::= expr LE optnl expr */ yytestcase(yyruleno==23);
      case 24: /* expr ::= expr EQ optnl expr */ yytestcase(yyruleno==24);
      case 25: /* expr ::= expr NE optnl expr */ yytestcase(yyruleno==25);
      case 26: /* expr ::= expr GE optnl expr */ yytestcase(yyruleno==26);
      case 27: /* expr ::= expr GT optnl expr */ yytestcase(yyruleno==27);
      case 28: /* expr ::= expr AND optnl expr */ yytestcase(yyruleno==28);
      case 29: /* expr ::= expr OR optnl expr */ yytestcase(yyruleno==29);
      case 30: /* expr ::= expr AND2 optnl expr */ yytestcase(yyruleno==30);
      case 31: /* expr ::= expr OR2 optnl expr */ yytestcase(yyruleno==31);
      case 32: /* expr ::= expr LEFT_ASSIGN optnl expr */ yytestcase(yyruleno==32);
      case 33: /* expr ::= expr EQ_ASSIGN optnl expr */ yytestcase(yyruleno==33);
      case 48: /* expr ::= expr DOLLAR optnl symbolstr */ yytestcase(yyruleno==48);
      case 49: /* expr ::= expr AT optnl symbolstr */ yytestcase(yyruleno==49);
#line 77 "parser.y"
{ yygotominor.yy0 = Call::c(yymsp[-2].minor.yy0, yymsp[-3].minor.yy0, yymsp[0].minor.yy0); }
#line 1002 "parser.c"
        break;
      case 34: /* expr ::= expr RIGHT_ASSIGN optnl expr */
#line 99 "parser.y"
{ yygotominor.yy0 = Call::c(yymsp[-2].minor.yy0, yymsp[0].minor.yy0, yymsp[-3].minor.yy0); }
#line 1007 "parser.c"
        break;
      case 35: /* expr ::= FUNCTION optnl LPAREN optnl formallist optnl RPAREN optnl expr */
#line 100 "parser.y"
{ yygotominor.yy0 = Call::c(yymsp[-8].minor.yy0, PairList(List(yymsp[-4].minor.yy72)), yymsp[0].minor.yy0); }
#line 1012 "parser.c"
        break;
      case 36: /* expr ::= expr LPAREN sublist RPAREN */
#line 101 "parser.y"
{ yymsp[-1].minor.yy72.push_front(Symbol(0), yymsp[-3].minor.yy0); yygotominor.yy0 = Call(List(yymsp[-1].minor.yy72)); }
#line 1017 "parser.c"
        break;
      case 37: /* expr ::= IF optnl LPAREN optnl expr optnl RPAREN optnl expr */
      case 40: /* expr ::= WHILE optnl LPAREN optnl expr optnl RPAREN optnl expr */ yytestcase(yyruleno==40);
#line 102 "parser.y"
{ yygotominor.yy0 = Call::c(yymsp[-8].minor.yy0, yymsp[-4].minor.yy0, yymsp[0].minor.yy0); }
#line 1023 "parser.c"
        break;
      case 38: /* expr ::= IF optnl LPAREN optnl expr optnl RPAREN optnl expr ELSE optnl expr */
#line 103 "parser.y"
{ yygotominor.yy0 = Call::c(yymsp[-11].minor.yy0, yymsp[-7].minor.yy0, yymsp[-3].minor.yy0, yymsp[0].minor.yy0); }
#line 1028 "parser.c"
        break;
      case 39: /* expr ::= FOR optnl LPAREN optnl SYMBOL optnl IN optnl expr optnl RPAREN optnl expr */
#line 104 "parser.y"
{ yygotominor.yy0 = Call::c(yymsp[-12].minor.yy0, yymsp[-8].minor.yy0, yymsp[-4].minor.yy0, yymsp[0].minor.yy0); }
#line 1033 "parser.c"
        break;
      case 42: /* expr ::= expr LBB optnl sublist optnl RBB */
      case 43: /* expr ::= expr LBRACKET optnl sublist optnl RBRACKET */ yytestcase(yyruleno==43);
#line 107 "parser.y"
{ yymsp[-2].minor.yy72.push_front(Symbol(0), yymsp[-5].minor.yy0); yymsp[-2].minor.yy72.push_front(Symbol(0), yymsp[-4].minor.yy0); yygotominor.yy0 = Call(List(yymsp[-2].minor.yy72)); }
#line 1039 "parser.c"
        break;
      case 44: /* expr ::= SYMBOL NS_GET symbolstr */
      case 45: /* expr ::= STR_CONST NS_GET symbolstr */ yytestcase(yyruleno==45);
      case 46: /* expr ::= SYMBOL NS_GET_INT symbolstr */ yytestcase(yyruleno==46);
      case 47: /* expr ::= STR_CONST NS_GET_INT symbolstr */ yytestcase(yyruleno==47);
#line 109 "parser.y"
{ yygotominor.yy0 = Call::c(yymsp[-1].minor.yy0, yymsp[-2].minor.yy0, yymsp[0].minor.yy0); }
#line 1047 "parser.c"
        break;
      case 50: /* expr ::= NEXT */
      case 51: /* expr ::= BREAK */ yytestcase(yyruleno==51);
#line 115 "parser.y"
{ yygotominor.yy0 = Call::c(yymsp[0].minor.yy0); }
#line 1053 "parser.c"
        break;
      case 56: /* exprlist ::= exprlist SEMICOLON expr */
      case 58: /* exprlist ::= exprlist NEWLINE expr */ yytestcase(yyruleno==58);
#line 124 "parser.y"
{ yygotominor.yy72 = yymsp[-2].minor.yy72; yygotominor.yy72.push_back(Symbol(0), yymsp[0].minor.yy0); }
#line 1059 "parser.c"
        break;
      case 57: /* exprlist ::= exprlist SEMICOLON */
#line 125 "parser.y"
{ yygotominor.yy72 = yymsp[-1].minor.yy72; }
#line 1064 "parser.c"
        break;
      case 59: /* exprlist ::= expr */
      case 64: /* sub ::= expr */ yytestcase(yyruleno==64);
      case 70: /* sub ::= NULL_CONST optnl EQ_ASSIGN optnl expr */ yytestcase(yyruleno==70);
#line 127 "parser.y"
{ yygotominor.yy72 = Pairs::Make(); yygotominor.yy72.push_back(Symbol(0), yymsp[0].minor.yy0); }
#line 1071 "parser.c"
        break;
      case 60: /* exprlist ::= */
#line 128 "parser.y"
{ yygotominor.yy72 = Pairs::Make(); }
#line 1076 "parser.c"
        break;
      case 61: /* sublist ::= sub */
#line 130 "parser.y"
{ yygotominor.yy72 = yymsp[0].minor.yy72; }
#line 1081 "parser.c"
        break;
      case 62: /* sublist ::= sublist optnl COMMA optnl sub */
#line 131 "parser.y"
{ yygotominor.yy72 = yymsp[-4].minor.yy72; yygotominor.yy72.push_back(yymsp[0].minor.yy72.name(0), yymsp[0].minor.yy72.value(0)); }
#line 1086 "parser.c"
        break;
      case 63: /* sub ::= */
#line 133 "parser.y"
{ yygotominor.yy72 = Pairs::Make(); yygotominor.yy72.push_back(Symbol(0), Symbol(0)); }
#line 1091 "parser.c"
        break;
      case 65: /* sub ::= SYMBOL optnl EQ_ASSIGN */
      case 66: /* sub ::= STR_CONST optnl EQ_ASSIGN */ yytestcase(yyruleno==66);
#line 135 "parser.y"
{ yygotominor.yy72 = Pairs::Make(); yygotominor.yy72.push_back(yymsp[-2].minor.yy0, Value::NIL); }
#line 1097 "parser.c"
        break;
      case 67: /* sub ::= SYMBOL optnl EQ_ASSIGN optnl expr */
      case 68: /* sub ::= STR_CONST optnl EQ_ASSIGN optnl expr */ yytestcase(yyruleno==68);
      case 72: /* formallist ::= SYMBOL optnl EQ_ASSIGN optnl expr */ yytestcase(yyruleno==72);
#line 137 "parser.y"
{ yygotominor.yy72 = Pairs::Make(); yygotominor.yy72.push_back(Symbol(yymsp[-4].minor.yy0), yymsp[0].minor.yy0); }
#line 1104 "parser.c"
        break;
      case 69: /* sub ::= NULL_CONST optnl EQ_ASSIGN */
#line 139 "parser.y"
{ yygotominor.yy72 = Pairs::Make(); yygotominor.yy72.push_back(Symbol(0), Value::NIL); }
#line 1109 "parser.c"
        break;
      case 71: /* formallist ::= SYMBOL */
#line 142 "parser.y"
{ yygotominor.yy72 = Pairs::Make(); yygotominor.yy72.push_back(Symbol(yymsp[0].minor.yy0), Value::NIL); }
#line 1114 "parser.c"
        break;
      case 73: /* formallist ::= formallist optnl COMMA optnl SYMBOL */
#line 144 "parser.y"
{ yygotominor.yy72 = yymsp[-4].minor.yy72; yygotominor.yy72.push_back(Symbol(yymsp[0].minor.yy0), Value::NIL); }
#line 1119 "parser.c"
        break;
      case 74: /* formallist ::= formallist optnl COMMA optnl SYMBOL optnl EQ_ASSIGN optnl expr */
#line 145 "parser.y"
{ yygotominor.yy72 = yymsp[-8].minor.yy72; yygotominor.yy72.push_back(Symbol(yymsp[-4].minor.yy0), yymsp[0].minor.yy0); }
#line 1124 "parser.c"
        break;
      default:
      /* (54) optnl ::= NEWLINE */ yytestcase(yyruleno==54);
      /* (55) optnl ::= */ yytestcase(yyruleno==55);
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
#line 55 "parser.y"

     result->state = -1;
     printf("Giving up.  Parser is hopelessly lost...\n");
#line 1178 "parser.c"
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
#line 46 "parser.y"

        result->state = -1;
	std::cout << "Syntax error!" << std::endl;
#line 1197 "parser.c"
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
#line 51 "parser.y"

     result->state = 1;
#line 1219 "parser.c"
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
