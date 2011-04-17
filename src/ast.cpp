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
#define YYNOCODE 72
#define YYACTIONTYPE unsigned char
#define ParseTOKENTYPE Value
typedef union {
  int yyinit;
  ParseTOKENTYPE yy0;
  Pairs yy74;
  int yy143;
} YYMINORTYPE;
#ifndef YYSTACKDEPTH
#define YYSTACKDEPTH 100
#endif
#define ParseARG_SDECL Value* result;
#define ParseARG_PDECL ,Value* result
#define ParseARG_FETCH Value* result = yypParser->result
#define ParseARG_STORE yypParser->result = result
#define YYNSTATE 155
#define YYNRULE 83
#define YYERRORSYMBOL 60
#define YYERRSYMDT yy143
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
 /*     0 */    34,  130,   54,  150,  153,   54,  150,   45,  115,   16,
 /*    10 */    33,   42,   44,   41,   43,   85,   40,   39,   35,   36,
 /*    20 */    37,   38,   23,   25,   27,   29,   31,   21,  107,   91,
 /*    30 */    30,   97,   95,  239,  131,   54,  150,   92,    6,  146,
 /*    40 */   103,  142,  131,   54,  150,   63,   11,  126,   32,   34,
 /*    50 */    15,   10,   46,    7,  114,  125,   45,  127,   16,   33,
 /*    60 */    42,   44,   41,   43,   52,   40,   39,   35,   36,   37,
 /*    70 */    38,   23,   25,   27,   29,   31,   21,   98,  100,   30,
 /*    80 */    97,   95,  140,  144,  143,   14,  109,   54,  150,   15,
 /*    90 */    66,  132,  120,   54,  150,   11,    8,   32,   34,  133,
 /*   100 */    10,  135,    7,    9,   47,   45,   12,   16,   33,   42,
 /*   110 */    44,   41,   43,  128,   40,   39,   35,   36,   37,   38,
 /*   120 */    23,   25,   27,   29,   31,   21,   84,   48,   30,   97,
 /*   130 */    95,  156,  134,   54,  150,  151,   54,  150,  121,   54,
 /*   140 */   150,  139,  101,  142,   11,   15,   32,   63,   34,   10,
 /*   150 */   158,    7,  105,  104,  145,   45,  147,   16,   33,   42,
 /*   160 */    44,   41,   43,   69,   40,   39,   35,   36,   37,   38,
 /*   170 */    23,   25,   27,   29,   31,   21,   98,  100,   30,   97,
 /*   180 */    95,  149,   54,  150,   83,  152,   54,  150,  112,   54,
 /*   190 */   150,  157,    4,   74,   11,  129,   32,   34,  136,   10,
 /*   200 */   137,    7,   82,  116,   45,   88,   16,   33,   42,   44,
 /*   210 */    41,   43,   90,   40,   39,   35,   36,   37,   38,   23,
 /*   220 */    25,   27,   29,   31,   21,  102,  142,   30,   97,   95,
 /*   230 */    63,   14,  154,  141,  124,  105,  104,   63,   87,   64,
 /*   240 */    58,   55,   72,   11,   68,   32,   81,   67,   10,    9,
 /*   250 */     7,   71,   45,   60,   16,   33,   42,   44,   41,   43,
 /*   260 */    78,   40,   39,   35,   36,   37,   38,   23,   25,   27,
 /*   270 */    29,   31,   21,   59,   77,   30,   97,   95,   62,   76,
 /*   280 */    57,   79,   56,   49,  118,    2,   96,    3,   50,  155,
 /*   290 */    80,   11,   73,   32,   75,   65,   10,   70,    7,   61,
 /*   300 */    18,   86,   19,   33,   42,   44,   41,   43,  117,   40,
 /*   310 */    39,   35,   36,   37,   38,   23,   25,   27,   29,   31,
 /*   320 */    21,   17,   89,   30,   97,   95,  240,  240,  240,  240,
 /*   330 */   240,  240,  240,  240,  240,  240,  240,  240,  240,   11,
 /*   340 */   240,   32,  240,  240,   10,  240,    7,  240,  240,  240,
 /*   350 */   240,  240,   42,   44,   41,   43,  240,   40,   39,   35,
 /*   360 */    36,   37,   38,   23,   25,   27,   29,   31,   21,  240,
 /*   370 */   240,   30,   97,   95,  240,  240,  240,  240,  240,  240,
 /*   380 */   240,  240,  240,  240,  240,  240,  240,   11,  240,   32,
 /*   390 */   240,  240,   10,  240,    7,  240,  240,  240,  240,  240,
 /*   400 */   240,  240,   41,   43,  240,   40,   39,   35,   36,   37,
 /*   410 */    38,   23,   25,   27,   29,   31,   21,  240,  240,   30,
 /*   420 */    97,   95,  240,   28,  240,   53,  113,    5,   51,  240,
 /*   430 */   240,  240,  240,   26,  240,   11,  240,   32,   24,  240,
 /*   440 */    10,  240,    7,  240,  240,   22,   20,  240,   31,   21,
 /*   450 */   240,  240,   30,   97,   95,  240,  240,  240,  240,  240,
 /*   460 */   240,  119,  108,  148,   99,  138,  106,    1,   11,   13,
 /*   470 */    32,  240,  111,   10,  240,    7,  240,  240,  122,  123,
 /*   480 */    40,   39,   35,   36,   37,   38,   23,   25,   27,   29,
 /*   490 */    31,   21,  240,  240,   30,   97,   95,  240,   28,  240,
 /*   500 */    53,  113,    5,   51,  240,  240,  240,  240,   26,  240,
 /*   510 */    11,  240,   32,   24,  240,   10,  240,    7,  240,  240,
 /*   520 */    22,   20,  240,  240,  240,  240,  240,  240,  240,  240,
 /*   530 */   240,  240,  240,  240,  240,  240,  240,  240,  148,   99,
 /*   540 */   138,  106,    1,  240,   13,  240,   28,  111,   53,  113,
 /*   550 */     5,   51,  240,  122,  123,  240,   26,  240,  240,  240,
 /*   560 */   240,   24,  240,  240,  240,  240,  240,  240,   22,   20,
 /*   570 */   240,  240,  240,  240,  240,  240,   27,   29,   31,   21,
 /*   580 */   240,  240,   30,   97,   95,  240,  148,   94,  110,   93,
 /*   590 */     1,  240,   13,  240,  240,  111,  240,  240,   11,  240,
 /*   600 */    32,  122,  123,   10,  240,    7,   21,  240,  240,   30,
 /*   610 */    97,   95,  240,  240,  240,   30,   97,   95,  240,  240,
 /*   620 */   240,  240,  240,  240,  240,   11,  240,   32,  240,  240,
 /*   630 */    10,   11,    7,   32,  240,  240,   10,  240,    7,
};
static const YYCODETYPE yy_lookahead[] = {
 /*     0 */     1,   66,   67,   68,   66,   67,   68,    8,   44,   10,
 /*    10 */    11,   12,   13,   14,   15,   67,   17,   18,   19,   20,
 /*    20 */    21,   22,   23,   24,   25,   26,   27,   28,   60,   61,
 /*    30 */    31,   32,   33,   65,   66,   67,   68,   61,    7,   48,
 /*    40 */    62,   63,   66,   67,   68,   67,   47,   48,   49,    1,
 /*    50 */    59,   52,    9,   54,   47,   42,    8,   44,   10,   11,
 /*    60 */    12,   13,   14,   15,   47,   17,   18,   19,   20,   21,
 /*    70 */    22,   23,   24,   25,   26,   27,   28,   34,   35,   31,
 /*    80 */    32,   33,   42,   53,   44,   40,   66,   67,   68,   59,
 /*    90 */    67,   46,   66,   67,   68,   47,   48,   49,    1,   42,
 /*   100 */    52,   44,   54,   58,    9,    8,    9,   10,   11,   12,
 /*   110 */    13,   14,   15,   48,   17,   18,   19,   20,   21,   22,
 /*   120 */    23,   24,   25,   26,   27,   28,   67,    9,   31,   32,
 /*   130 */    33,    0,   66,   67,   68,   66,   67,   68,   66,   67,
 /*   140 */    68,   55,   62,   63,   47,   59,   49,   67,    1,   52,
 /*   150 */     0,   54,   34,   35,   42,    8,   44,   10,   11,   12,
 /*   160 */    13,   14,   15,   67,   17,   18,   19,   20,   21,   22,
 /*   170 */    23,   24,   25,   26,   27,   28,   34,   35,   31,   32,
 /*   180 */    33,   66,   67,   68,   67,   66,   67,   68,   66,   67,
 /*   190 */    68,    0,   48,   67,   47,   48,   49,    1,   42,   52,
 /*   200 */    44,   54,   67,   59,    8,   67,   10,   11,   12,   13,
 /*   210 */    14,   15,   67,   17,   18,   19,   20,   21,   22,   23,
 /*   220 */    24,   25,   26,   27,   28,   62,   63,   31,   32,   33,
 /*   230 */    67,   40,   42,   63,   44,   34,   35,   67,   67,   67,
 /*   240 */    67,   67,   67,   47,   67,   49,   67,   67,   52,   58,
 /*   250 */    54,   67,    8,   67,   10,   11,   12,   13,   14,   15,
 /*   260 */    67,   17,   18,   19,   20,   21,   22,   23,   24,   25,
 /*   270 */    26,   27,   28,   67,   67,   31,   32,   33,   67,   67,
 /*   280 */    67,   67,   67,   47,   44,   69,   64,   70,   47,    0,
 /*   290 */    67,   47,   67,   49,   67,   67,   52,   67,   54,   67,
 /*   300 */     9,   67,    9,   11,   12,   13,   14,   15,   44,   17,
 /*   310 */    18,   19,   20,   21,   22,   23,   24,   25,   26,   27,
 /*   320 */    28,   51,   67,   31,   32,   33,   71,   71,   71,   71,
 /*   330 */    71,   71,   71,   71,   71,   71,   71,   71,   71,   47,
 /*   340 */    71,   49,   71,   71,   52,   71,   54,   71,   71,   71,
 /*   350 */    71,   71,   12,   13,   14,   15,   71,   17,   18,   19,
 /*   360 */    20,   21,   22,   23,   24,   25,   26,   27,   28,   71,
 /*   370 */    71,   31,   32,   33,   71,   71,   71,   71,   71,   71,
 /*   380 */    71,   71,   71,   71,   71,   71,   71,   47,   71,   49,
 /*   390 */    71,   71,   52,   71,   54,   71,   71,   71,   71,   71,
 /*   400 */    71,   71,   14,   15,   71,   17,   18,   19,   20,   21,
 /*   410 */    22,   23,   24,   25,   26,   27,   28,   71,   71,   31,
 /*   420 */    32,   33,   71,    1,   71,    3,    4,    5,    6,   71,
 /*   430 */    71,   71,   71,   11,   71,   47,   71,   49,   16,   71,
 /*   440 */    52,   71,   54,   71,   71,   23,   24,   71,   27,   28,
 /*   450 */    71,   71,   31,   32,   33,   71,   71,   71,   71,   71,
 /*   460 */    71,   39,   40,   41,   42,   43,   44,   45,   47,   47,
 /*   470 */    49,   71,   50,   52,   71,   54,   71,   71,   56,   57,
 /*   480 */    17,   18,   19,   20,   21,   22,   23,   24,   25,   26,
 /*   490 */    27,   28,   71,   71,   31,   32,   33,   71,    1,   71,
 /*   500 */     3,    4,    5,    6,   71,   71,   71,   71,   11,   71,
 /*   510 */    47,   71,   49,   16,   71,   52,   71,   54,   71,   71,
 /*   520 */    23,   24,   71,   71,   71,   71,   71,   71,   71,   71,
 /*   530 */    71,   71,   71,   71,   71,   71,   71,   71,   41,   42,
 /*   540 */    43,   44,   45,   71,   47,   71,    1,   50,    3,    4,
 /*   550 */     5,    6,   71,   56,   57,   71,   11,   71,   71,   71,
 /*   560 */    71,   16,   71,   71,   71,   71,   71,   71,   23,   24,
 /*   570 */    71,   71,   71,   71,   71,   71,   25,   26,   27,   28,
 /*   580 */    71,   71,   31,   32,   33,   71,   41,   42,   43,   44,
 /*   590 */    45,   71,   47,   71,   71,   50,   71,   71,   47,   71,
 /*   600 */    49,   56,   57,   52,   71,   54,   28,   71,   71,   31,
 /*   610 */    32,   33,   71,   71,   71,   31,   32,   33,   71,   71,
 /*   620 */    71,   71,   71,   71,   71,   47,   71,   49,   71,   71,
 /*   630 */    52,   47,   54,   49,   71,   71,   52,   71,   54,
};
#define YY_SHIFT_USE_DFLT (-37)
#define YY_SHIFT_MAX 119
static const short yy_shift_ofst[] = {
 /*     0 */   422,  497,  497,  497,  497,  497,  497,  545,  497,  497,
 /*    10 */   545,  545,  497,  497,  497,  545,  497,  497,  497,  497,
 /*    20 */   497,  497,  497,  497,  497,  497,  497,  497,  497,  497,
 /*    30 */   497,  497,  497,  497,  497,  497,  497,  497,  497,  497,
 /*    40 */   497,  497,  497,  497,  497,  497,  497,  497,  497,  497,
 /*    50 */   497,  236,  240,  241,   97,   48,   -1,  147,  196,  196,
 /*    60 */   196,  196,  196,  196,  196,  244,  244,  244,  292,  340,
 /*    70 */   340,  388,  388,  463,  463,  463,  463,  463,  463,  463,
 /*    80 */   463,  463,  551,  551,  421,  421,  578,  584,  584,  584,
 /*    90 */   584,  191,   45,  118,   43,  190,  144,   13,  112,  142,
 /*   100 */    40,   30,   86,   -9,   57,  156,  201,  150,  131,   65,
 /*   110 */    95,   17,   31,    7,  -36,  270,  264,  293,  291,  289,
};
#define YY_REDUCE_USE_DFLT (-66)
#define YY_REDUCE_MAX 53
static const short yy_reduce_ofst[] = {
 /*     0 */   -32,  -24,  122,   26,  -62,   72,  119,  163,   69,   66,
 /*    10 */    80,  -22,  115,   20,  -65,  170,  177,  174,  173,  172,
 /*    20 */   171,  145,  138,  135,  126,  117,   96,   59,   23,  -52,
 /*    30 */   255,  234,  232,  230,  228,  227,  225,  223,  214,  212,
 /*    40 */   207,  193,  184,  179,  175,  180,  186,  206,  211,  213,
 /*    50 */   215,  216,  222,  217,
};
static const YYACTIONTYPE yy_default[] = {
 /*     0 */   238,  238,  238,  238,  238,  238,  238,  238,  238,  222,
 /*    10 */   238,  238,  238,  238,  224,  238,  238,  238,  238,  238,
 /*    20 */   238,  238,  238,  238,  238,  238,  238,  238,  238,  238,
 /*    30 */   238,  238,  238,  238,  238,  238,  238,  238,  238,  238,
 /*    40 */   238,  238,  238,  238,  238,  238,  230,  232,  228,  238,
 /*    50 */   238,  238,  238,  238,  159,  238,  238,  238,  235,  233,
 /*    60 */   231,  180,  229,  227,  237,  182,  172,  193,  194,  171,
 /*    70 */   181,  190,  192,  184,  170,  183,  187,  188,  189,  186,
 /*    80 */   185,  191,  174,  175,  176,  177,  179,  168,  169,  178,
 /*    90 */   173,  238,  238,  165,  163,  238,  238,  238,  238,  163,
 /*   100 */   238,  238,  238,  238,  238,  238,  165,  238,  238,  238,
 /*   110 */   164,  238,  197,  238,  238,  238,  238,  236,  234,  238,
 /*   120 */   200,  201,  216,  217,  214,  213,  218,  212,  167,  219,
 /*   130 */   223,  220,  166,  209,  221,  208,  205,  204,  164,  203,
 /*   140 */   211,  226,  225,  210,  202,  207,  196,  206,  162,  161,
 /*   150 */   160,  199,  198,  195,  215,
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
  "$",             "QUESTION",      "LOW",           "WHILE",       
  "FOR",           "REPEAT",        "IF",            "ELSE",        
  "LEFT_ASSIGN",   "EQ_ASSIGN",     "RIGHT_ASSIGN",  "TILDE",       
  "OR",            "OR2",           "AND",           "AND2",        
  "NOT",           "GT",            "GE",            "LT",          
  "LE",            "EQ",            "NE",            "PLUS",        
  "MINUS",         "TIMES",         "DIVIDE",        "SPECIALOP",   
  "COLON",         "UMINUS",        "UPLUS",         "POW",         
  "DOLLAR",        "AT",            "NS_GET",        "NS_GET_INT",  
  "PAREN",         "BRACKET",       "BB",            "END_OF_INPUT",
  "NEWLINE",       "NUM_CONST",     "STR_CONST",     "NULL_CONST",  
  "SYMBOL",        "LBRACE",        "RBRACE",        "LPAREN",      
  "RPAREN",        "PERCENT",       "FUNCTION",      "IN",          
  "LBB",           "RBB",           "LBRACKET",      "RBRACKET",    
  "NEXT",          "BREAK",         "SEMICOLON",     "COMMA",       
  "error",         "exprlist",      "sublist",       "sub",         
  "formlist",      "prog",          "expr_or_assign",  "expr",        
  "equal_assign",  "ifcond",        "cond",        
};
#endif /* NDEBUG */

#ifndef NDEBUG
/* For tracing reduce actions, the names of all rules are required.
*/
static const char *const yyRuleName[] = {
 /*   0 */ "prog ::= END_OF_INPUT",
 /*   1 */ "prog ::= NEWLINE",
 /*   2 */ "prog ::= exprlist",
 /*   3 */ "prog ::= error",
 /*   4 */ "expr_or_assign ::= expr",
 /*   5 */ "expr_or_assign ::= equal_assign",
 /*   6 */ "equal_assign ::= expr EQ_ASSIGN expr_or_assign",
 /*   7 */ "expr ::= NUM_CONST",
 /*   8 */ "expr ::= STR_CONST",
 /*   9 */ "expr ::= NULL_CONST",
 /*  10 */ "expr ::= SYMBOL",
 /*  11 */ "expr ::= LBRACE exprlist RBRACE",
 /*  12 */ "expr ::= LPAREN expr_or_assign RPAREN",
 /*  13 */ "expr ::= MINUS expr",
 /*  14 */ "expr ::= PLUS expr",
 /*  15 */ "expr ::= NOT expr",
 /*  16 */ "expr ::= TILDE expr",
 /*  17 */ "expr ::= QUESTION expr",
 /*  18 */ "expr ::= expr COLON expr",
 /*  19 */ "expr ::= expr PLUS expr",
 /*  20 */ "expr ::= expr MINUS expr",
 /*  21 */ "expr ::= expr TIMES expr",
 /*  22 */ "expr ::= expr DIVIDE expr",
 /*  23 */ "expr ::= expr POW expr",
 /*  24 */ "expr ::= expr SPECIALOP expr",
 /*  25 */ "expr ::= expr PERCENT expr",
 /*  26 */ "expr ::= expr TILDE expr",
 /*  27 */ "expr ::= expr QUESTION expr",
 /*  28 */ "expr ::= expr LT expr",
 /*  29 */ "expr ::= expr LE expr",
 /*  30 */ "expr ::= expr EQ expr",
 /*  31 */ "expr ::= expr NE expr",
 /*  32 */ "expr ::= expr GE expr",
 /*  33 */ "expr ::= expr GT expr",
 /*  34 */ "expr ::= expr AND expr",
 /*  35 */ "expr ::= expr OR expr",
 /*  36 */ "expr ::= expr AND2 expr",
 /*  37 */ "expr ::= expr OR2 expr",
 /*  38 */ "expr ::= expr LEFT_ASSIGN expr",
 /*  39 */ "expr ::= expr RIGHT_ASSIGN expr",
 /*  40 */ "expr ::= FUNCTION LPAREN formlist RPAREN expr_or_assign",
 /*  41 */ "expr ::= expr LPAREN sublist RPAREN",
 /*  42 */ "expr ::= IF ifcond expr_or_assign",
 /*  43 */ "expr ::= IF ifcond expr_or_assign ELSE expr_or_assign",
 /*  44 */ "expr ::= FOR LPAREN SYMBOL IN expr RPAREN expr_or_assign",
 /*  45 */ "expr ::= WHILE cond expr_or_assign",
 /*  46 */ "expr ::= REPEAT expr_or_assign",
 /*  47 */ "expr ::= expr LBB sublist RBB",
 /*  48 */ "expr ::= expr LBRACKET sublist RBRACKET",
 /*  49 */ "expr ::= SYMBOL NS_GET SYMBOL",
 /*  50 */ "expr ::= SYMBOL NS_GET STR_CONST",
 /*  51 */ "expr ::= STR_CONST NS_GET SYMBOL",
 /*  52 */ "expr ::= STR_CONST NS_GET STR_CONST",
 /*  53 */ "expr ::= SYMBOL NS_GET_INT SYMBOL",
 /*  54 */ "expr ::= SYMBOL NS_GET_INT STR_CONST",
 /*  55 */ "expr ::= STR_CONST NS_GET_INT SYMBOL",
 /*  56 */ "expr ::= STR_CONST NS_GET_INT STR_CONST",
 /*  57 */ "expr ::= expr DOLLAR SYMBOL",
 /*  58 */ "expr ::= expr DOLLAR STR_CONST",
 /*  59 */ "expr ::= expr AT SYMBOL",
 /*  60 */ "expr ::= expr AT STR_CONST",
 /*  61 */ "expr ::= NEXT",
 /*  62 */ "expr ::= BREAK",
 /*  63 */ "cond ::= LPAREN expr RPAREN",
 /*  64 */ "ifcond ::= LPAREN expr RPAREN",
 /*  65 */ "exprlist ::= expr_or_assign",
 /*  66 */ "exprlist ::= exprlist SEMICOLON expr_or_assign",
 /*  67 */ "exprlist ::= exprlist SEMICOLON",
 /*  68 */ "exprlist ::= exprlist NEWLINE expr_or_assign",
 /*  69 */ "exprlist ::= exprlist NEWLINE",
 /*  70 */ "sublist ::= sub",
 /*  71 */ "sublist ::= sublist COMMA sub",
 /*  72 */ "sub ::= expr",
 /*  73 */ "sub ::= SYMBOL EQ_ASSIGN",
 /*  74 */ "sub ::= SYMBOL EQ_ASSIGN expr",
 /*  75 */ "sub ::= STR_CONST EQ_ASSIGN",
 /*  76 */ "sub ::= STR_CONST EQ_ASSIGN expr",
 /*  77 */ "sub ::= NULL_CONST EQ_ASSIGN",
 /*  78 */ "sub ::= NULL_CONST EQ_ASSIGN expr",
 /*  79 */ "formlist ::= SYMBOL",
 /*  80 */ "formlist ::= SYMBOL EQ_ASSIGN expr",
 /*  81 */ "formlist ::= formlist COMMA SYMBOL",
 /*  82 */ "formlist ::= formlist COMMA SYMBOL EQ_ASSIGN expr",
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
  { 65, 1 },
  { 65, 1 },
  { 65, 1 },
  { 65, 1 },
  { 66, 1 },
  { 66, 1 },
  { 68, 3 },
  { 67, 1 },
  { 67, 1 },
  { 67, 1 },
  { 67, 1 },
  { 67, 3 },
  { 67, 3 },
  { 67, 2 },
  { 67, 2 },
  { 67, 2 },
  { 67, 2 },
  { 67, 2 },
  { 67, 3 },
  { 67, 3 },
  { 67, 3 },
  { 67, 3 },
  { 67, 3 },
  { 67, 3 },
  { 67, 3 },
  { 67, 3 },
  { 67, 3 },
  { 67, 3 },
  { 67, 3 },
  { 67, 3 },
  { 67, 3 },
  { 67, 3 },
  { 67, 3 },
  { 67, 3 },
  { 67, 3 },
  { 67, 3 },
  { 67, 3 },
  { 67, 3 },
  { 67, 3 },
  { 67, 3 },
  { 67, 5 },
  { 67, 4 },
  { 67, 3 },
  { 67, 5 },
  { 67, 7 },
  { 67, 3 },
  { 67, 2 },
  { 67, 4 },
  { 67, 4 },
  { 67, 3 },
  { 67, 3 },
  { 67, 3 },
  { 67, 3 },
  { 67, 3 },
  { 67, 3 },
  { 67, 3 },
  { 67, 3 },
  { 67, 3 },
  { 67, 3 },
  { 67, 3 },
  { 67, 3 },
  { 67, 1 },
  { 67, 1 },
  { 70, 3 },
  { 69, 3 },
  { 61, 1 },
  { 61, 3 },
  { 61, 2 },
  { 61, 3 },
  { 61, 2 },
  { 62, 1 },
  { 62, 3 },
  { 63, 1 },
  { 63, 2 },
  { 63, 3 },
  { 63, 2 },
  { 63, 3 },
  { 63, 2 },
  { 63, 3 },
  { 64, 1 },
  { 64, 3 },
  { 64, 3 },
  { 64, 5 },
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
      case 0: /* prog ::= END_OF_INPUT */
      case 1: /* prog ::= NEWLINE */ yytestcase(yyruleno==1);
      case 3: /* prog ::= error */ yytestcase(yyruleno==3);
#line 58 "parser.y"
{ *result = yygotominor.yy0 = Expression(0); }
#line 973 "parser.c"
        break;
      case 2: /* prog ::= exprlist */
#line 62 "parser.y"
{ *result = yygotominor.yy0 = Expression(List(yymsp[0].minor.yy74)); }
#line 978 "parser.c"
        break;
      case 4: /* expr_or_assign ::= expr */
      case 5: /* expr_or_assign ::= equal_assign */ yytestcase(yyruleno==5);
      case 7: /* expr ::= NUM_CONST */ yytestcase(yyruleno==7);
      case 8: /* expr ::= STR_CONST */ yytestcase(yyruleno==8);
      case 9: /* expr ::= NULL_CONST */ yytestcase(yyruleno==9);
      case 10: /* expr ::= SYMBOL */ yytestcase(yyruleno==10);
#line 65 "parser.y"
{ yygotominor.yy0 = yymsp[0].minor.yy0; }
#line 988 "parser.c"
        break;
      case 6: /* equal_assign ::= expr EQ_ASSIGN expr_or_assign */
      case 18: /* expr ::= expr COLON expr */ yytestcase(yyruleno==18);
      case 19: /* expr ::= expr PLUS expr */ yytestcase(yyruleno==19);
      case 20: /* expr ::= expr MINUS expr */ yytestcase(yyruleno==20);
      case 21: /* expr ::= expr TIMES expr */ yytestcase(yyruleno==21);
      case 22: /* expr ::= expr DIVIDE expr */ yytestcase(yyruleno==22);
      case 23: /* expr ::= expr POW expr */ yytestcase(yyruleno==23);
      case 24: /* expr ::= expr SPECIALOP expr */ yytestcase(yyruleno==24);
      case 25: /* expr ::= expr PERCENT expr */ yytestcase(yyruleno==25);
      case 26: /* expr ::= expr TILDE expr */ yytestcase(yyruleno==26);
      case 27: /* expr ::= expr QUESTION expr */ yytestcase(yyruleno==27);
      case 28: /* expr ::= expr LT expr */ yytestcase(yyruleno==28);
      case 29: /* expr ::= expr LE expr */ yytestcase(yyruleno==29);
      case 30: /* expr ::= expr EQ expr */ yytestcase(yyruleno==30);
      case 31: /* expr ::= expr NE expr */ yytestcase(yyruleno==31);
      case 32: /* expr ::= expr GE expr */ yytestcase(yyruleno==32);
      case 33: /* expr ::= expr GT expr */ yytestcase(yyruleno==33);
      case 34: /* expr ::= expr AND expr */ yytestcase(yyruleno==34);
      case 35: /* expr ::= expr OR expr */ yytestcase(yyruleno==35);
      case 36: /* expr ::= expr AND2 expr */ yytestcase(yyruleno==36);
      case 37: /* expr ::= expr OR2 expr */ yytestcase(yyruleno==37);
      case 38: /* expr ::= expr LEFT_ASSIGN expr */ yytestcase(yyruleno==38);
      case 49: /* expr ::= SYMBOL NS_GET SYMBOL */ yytestcase(yyruleno==49);
      case 50: /* expr ::= SYMBOL NS_GET STR_CONST */ yytestcase(yyruleno==50);
      case 51: /* expr ::= STR_CONST NS_GET SYMBOL */ yytestcase(yyruleno==51);
      case 52: /* expr ::= STR_CONST NS_GET STR_CONST */ yytestcase(yyruleno==52);
      case 53: /* expr ::= SYMBOL NS_GET_INT SYMBOL */ yytestcase(yyruleno==53);
      case 54: /* expr ::= SYMBOL NS_GET_INT STR_CONST */ yytestcase(yyruleno==54);
      case 55: /* expr ::= STR_CONST NS_GET_INT SYMBOL */ yytestcase(yyruleno==55);
      case 56: /* expr ::= STR_CONST NS_GET_INT STR_CONST */ yytestcase(yyruleno==56);
      case 57: /* expr ::= expr DOLLAR SYMBOL */ yytestcase(yyruleno==57);
      case 58: /* expr ::= expr DOLLAR STR_CONST */ yytestcase(yyruleno==58);
      case 59: /* expr ::= expr AT SYMBOL */ yytestcase(yyruleno==59);
      case 60: /* expr ::= expr AT STR_CONST */ yytestcase(yyruleno==60);
#line 68 "parser.y"
{ yygotominor.yy0 = Call::c(yymsp[-1].minor.yy0, yymsp[-2].minor.yy0, yymsp[0].minor.yy0); }
#line 1026 "parser.c"
        break;
      case 11: /* expr ::= LBRACE exprlist RBRACE */
#line 75 "parser.y"
{ yymsp[-1].minor.yy74.push_front(Symbol(0), yymsp[-2].minor.yy0); yygotominor.yy0 = Expression(List(yymsp[-1].minor.yy74)); }
#line 1031 "parser.c"
        break;
      case 12: /* expr ::= LPAREN expr_or_assign RPAREN */
      case 63: /* cond ::= LPAREN expr RPAREN */ yytestcase(yyruleno==63);
      case 64: /* ifcond ::= LPAREN expr RPAREN */ yytestcase(yyruleno==64);
#line 76 "parser.y"
{ yygotominor.yy0 = yymsp[-1].minor.yy0; }
#line 1038 "parser.c"
        break;
      case 13: /* expr ::= MINUS expr */
      case 14: /* expr ::= PLUS expr */ yytestcase(yyruleno==14);
      case 15: /* expr ::= NOT expr */ yytestcase(yyruleno==15);
      case 16: /* expr ::= TILDE expr */ yytestcase(yyruleno==16);
      case 17: /* expr ::= QUESTION expr */ yytestcase(yyruleno==17);
      case 46: /* expr ::= REPEAT expr_or_assign */ yytestcase(yyruleno==46);
#line 78 "parser.y"
{ yygotominor.yy0 = Call::c(yymsp[-1].minor.yy0, yymsp[0].minor.yy0); }
#line 1048 "parser.c"
        break;
      case 39: /* expr ::= expr RIGHT_ASSIGN expr */
#line 106 "parser.y"
{ yygotominor.yy0 = Call::c(yymsp[-1].minor.yy0, yymsp[0].minor.yy0, yymsp[-2].minor.yy0); }
#line 1053 "parser.c"
        break;
      case 40: /* expr ::= FUNCTION LPAREN formlist RPAREN expr_or_assign */
#line 107 "parser.y"
{ yygotominor.yy0 = Call::c(yymsp[-4].minor.yy0, PairList(List(yymsp[-2].minor.yy74)), yymsp[0].minor.yy0); }
#line 1058 "parser.c"
        break;
      case 41: /* expr ::= expr LPAREN sublist RPAREN */
#line 108 "parser.y"
{ yymsp[-1].minor.yy74.push_front(Symbol(0), yymsp[-3].minor.yy0); yygotominor.yy0 = Call(List(yymsp[-1].minor.yy74)); }
#line 1063 "parser.c"
        break;
      case 42: /* expr ::= IF ifcond expr_or_assign */
      case 45: /* expr ::= WHILE cond expr_or_assign */ yytestcase(yyruleno==45);
#line 109 "parser.y"
{ yygotominor.yy0 = Call::c(yymsp[-2].minor.yy0, yymsp[-1].minor.yy0, yymsp[0].minor.yy0); }
#line 1069 "parser.c"
        break;
      case 43: /* expr ::= IF ifcond expr_or_assign ELSE expr_or_assign */
#line 110 "parser.y"
{ yygotominor.yy0 = Call::c(yymsp[-4].minor.yy0, yymsp[-3].minor.yy0, yymsp[-2].minor.yy0, yymsp[0].minor.yy0); }
#line 1074 "parser.c"
        break;
      case 44: /* expr ::= FOR LPAREN SYMBOL IN expr RPAREN expr_or_assign */
#line 111 "parser.y"
{ yygotominor.yy0 = Call::c(yymsp[-6].minor.yy0, yymsp[-4].minor.yy0, yymsp[-2].minor.yy0, yymsp[0].minor.yy0); }
#line 1079 "parser.c"
        break;
      case 47: /* expr ::= expr LBB sublist RBB */
      case 48: /* expr ::= expr LBRACKET sublist RBRACKET */ yytestcase(yyruleno==48);
#line 114 "parser.y"
{ yymsp[-1].minor.yy74.push_front(Symbol(0), yymsp[-3].minor.yy0); yymsp[-1].minor.yy74.push_front(Symbol(0), yymsp[-2].minor.yy0); yygotominor.yy0 = Call(List(yymsp[-1].minor.yy74)); }
#line 1085 "parser.c"
        break;
      case 61: /* expr ::= NEXT */
      case 62: /* expr ::= BREAK */ yytestcase(yyruleno==62);
#line 128 "parser.y"
{ yygotominor.yy0 = Call::c(yymsp[0].minor.yy0); }
#line 1091 "parser.c"
        break;
      case 65: /* exprlist ::= expr_or_assign */
      case 72: /* sub ::= expr */ yytestcase(yyruleno==72);
      case 78: /* sub ::= NULL_CONST EQ_ASSIGN expr */ yytestcase(yyruleno==78);
#line 134 "parser.y"
{ yygotominor.yy74 = Pairs::Make(); yygotominor.yy74.push_back(Symbol(0), yymsp[0].minor.yy0); }
#line 1098 "parser.c"
        break;
      case 66: /* exprlist ::= exprlist SEMICOLON expr_or_assign */
      case 68: /* exprlist ::= exprlist NEWLINE expr_or_assign */ yytestcase(yyruleno==68);
#line 135 "parser.y"
{ yygotominor.yy74 = yymsp[-2].minor.yy74; yygotominor.yy74.push_back(Symbol(0), yymsp[0].minor.yy0); }
#line 1104 "parser.c"
        break;
      case 67: /* exprlist ::= exprlist SEMICOLON */
      case 69: /* exprlist ::= exprlist NEWLINE */ yytestcase(yyruleno==69);
#line 136 "parser.y"
{ yygotominor.yy74 = yymsp[-1].minor.yy74; }
#line 1110 "parser.c"
        break;
      case 70: /* sublist ::= sub */
#line 140 "parser.y"
{ yygotominor.yy74 = yymsp[0].minor.yy74; }
#line 1115 "parser.c"
        break;
      case 71: /* sublist ::= sublist COMMA sub */
#line 141 "parser.y"
{ yygotominor.yy74 = yymsp[-2].minor.yy74; yygotominor.yy74.push_back(yymsp[0].minor.yy74.name(0), yymsp[0].minor.yy74.value(0)); }
#line 1120 "parser.c"
        break;
      case 73: /* sub ::= SYMBOL EQ_ASSIGN */
      case 75: /* sub ::= STR_CONST EQ_ASSIGN */ yytestcase(yyruleno==75);
#line 144 "parser.y"
{ yygotominor.yy74 = Pairs::Make(); yygotominor.yy74.push_back(yymsp[-1].minor.yy0, Value::NIL); }
#line 1126 "parser.c"
        break;
      case 74: /* sub ::= SYMBOL EQ_ASSIGN expr */
      case 76: /* sub ::= STR_CONST EQ_ASSIGN expr */ yytestcase(yyruleno==76);
      case 80: /* formlist ::= SYMBOL EQ_ASSIGN expr */ yytestcase(yyruleno==80);
#line 145 "parser.y"
{ yygotominor.yy74 = Pairs::Make(); yygotominor.yy74.push_back(Symbol(yymsp[-2].minor.yy0), yymsp[0].minor.yy0); }
#line 1133 "parser.c"
        break;
      case 77: /* sub ::= NULL_CONST EQ_ASSIGN */
#line 148 "parser.y"
{ yygotominor.yy74 = Pairs::Make(); yygotominor.yy74.push_back(Symbol(0), Value::NIL); }
#line 1138 "parser.c"
        break;
      case 79: /* formlist ::= SYMBOL */
#line 151 "parser.y"
{ yygotominor.yy74 = Pairs::Make(); yygotominor.yy74.push_back(Symbol(yymsp[0].minor.yy0), Value::NIL); }
#line 1143 "parser.c"
        break;
      case 81: /* formlist ::= formlist COMMA SYMBOL */
#line 153 "parser.y"
{ yygotominor.yy74 = yymsp[-2].minor.yy74; yygotominor.yy74.push_back(Symbol(yymsp[0].minor.yy0), Value::NIL); }
#line 1148 "parser.c"
        break;
      case 82: /* formlist ::= formlist COMMA SYMBOL EQ_ASSIGN expr */
#line 154 "parser.y"
{ yygotominor.yy74 = yymsp[-4].minor.yy74; yygotominor.yy74.push_back(Symbol(yymsp[-2].minor.yy0), yymsp[0].minor.yy0); }
#line 1153 "parser.c"
        break;
      default:
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
#line 54 "parser.y"

     printf("Giving up.  Parser is hopelessly lost...\n");
#line 1204 "parser.c"
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

	std::cout << "Syntax error!" << std::endl;
#line 1222 "parser.c"
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
#line 50 "parser.y"

      printf("parsing complete!\n");
#line 1244 "parser.c"
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
