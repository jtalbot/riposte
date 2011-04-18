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
#define YYNOCODE 73
#define YYACTIONTYPE unsigned char
#define ParseTOKENTYPE Value
typedef union {
  int yyinit;
  ParseTOKENTYPE yy0;
  Pairs yy108;
  int yy145;
} YYMINORTYPE;
#ifndef YYSTACKDEPTH
#define YYSTACKDEPTH 100
#endif
#define ParseARG_SDECL Parser::Result* result;
#define ParseARG_PDECL ,Parser::Result* result
#define ParseARG_FETCH Parser::Result* result = yypParser->result
#define ParseARG_STORE yypParser->result = result
#define YYNSTATE 160
#define YYNRULE 85
#define YYERRORSYMBOL 60
#define YYERRSYMDT yy145
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
 /*     0 */    34,  157,   59,  154,  158,   59,  154,   45,    3,   46,
 /*    10 */    33,   42,   44,   41,   43,  121,   40,   39,   35,   36,
 /*    20 */    37,   38,   54,   17,   27,   29,   31,   22,  110,   97,
 /*    30 */    30,  105,  104,  246,   52,  133,   59,  154,  103,  101,
 /*    40 */   133,   59,  154,  112,   59,  154,   11,  145,   32,  147,
 /*    50 */    34,   10,   79,    8,  156,   59,  154,   45,   81,   46,
 /*    60 */    33,   42,   44,   41,   43,   76,   40,   39,   35,   36,
 /*    70 */    37,   38,   54,   17,   27,   29,   31,   22,  143,   49,
 /*    80 */    30,  105,  104,   68,  153,   59,  154,  108,  144,  152,
 /*    90 */    59,  154,   82,   68,   98,  144,   11,  128,   32,   34,
 /*   100 */    68,   10,   75,    8,  103,  101,   45,   71,   46,   33,
 /*   110 */    42,   44,   41,   43,    2,   40,   39,   35,   36,   37,
 /*   120 */    38,   54,   17,   27,   29,   31,   22,   72,   66,   30,
 /*   130 */   105,  104,    9,   30,  105,  104,  115,   59,  154,  136,
 /*   140 */    59,  154,  124,   59,  154,   11,    7,   32,   34,   11,
 /*   150 */    10,   32,    8,   62,   10,   45,    8,   46,   33,   42,
 /*   160 */    44,   41,   43,   64,   40,   39,   35,   36,   37,   38,
 /*   170 */    54,   17,   27,   29,   31,   22,  107,  144,   30,  105,
 /*   180 */   104,   60,   68,  155,   59,  154,  162,  146,  129,  148,
 /*   190 */   130,   14,   61,   16,   11,  131,   32,   34,  122,   10,
 /*   200 */    16,    8,  119,  137,   45,  138,   46,   33,   42,   44,
 /*   210 */    41,   43,   51,   40,   39,   35,   36,   37,   38,   54,
 /*   220 */    17,   27,   29,   31,   22,    5,   15,   30,  105,  104,
 /*   230 */   139,  126,  140,  127,  149,  141,  150,  102,  100,   16,
 /*   240 */   135,   24,   69,   11,    9,   32,  109,    1,   10,   50,
 /*   250 */     8,   48,   45,    4,   46,   33,   42,   44,   41,   43,
 /*   260 */    47,   40,   39,   35,   36,   37,   38,   54,   17,   27,
 /*   270 */    29,   31,   22,   88,   73,   30,  105,  104,   77,   87,
 /*   280 */    94,   92,   93,   67,   63,  102,  100,   58,   86,  163,
 /*   290 */   161,   11,   55,   32,  134,   12,   10,  117,    8,  118,
 /*   300 */    18,  120,   20,   33,   42,   44,   41,   43,   19,   40,
 /*   310 */    39,   35,   36,   37,   38,   54,   17,   27,   29,   31,
 /*   320 */    22,  132,  160,   30,  105,  104,   89,   91,   90,   65,
 /*   330 */    74,   70,   80,   78,   85,   84,   83,  247,  247,   11,
 /*   340 */   247,   32,  247,  247,   10,  247,    8,  247,  247,  247,
 /*   350 */   247,  247,   42,   44,   41,   43,  247,   40,   39,   35,
 /*   360 */    36,   37,   38,   54,   17,   27,   29,   31,   22,  247,
 /*   370 */   247,   30,  105,  104,  247,  247,  247,  247,  247,  247,
 /*   380 */   247,  247,  247,  247,  247,  247,  247,   11,  247,   32,
 /*   390 */   247,  247,   10,  247,    8,  247,  247,  247,  247,  247,
 /*   400 */   247,  247,   41,   43,  247,   40,   39,   35,   36,   37,
 /*   410 */    38,   54,   17,   27,   29,   31,   22,  247,  247,   30,
 /*   420 */   105,  104,  247,   28,  247,   53,  116,    6,   57,  247,
 /*   430 */   247,  247,  247,   26,  247,   11,  247,   32,   25,  247,
 /*   440 */    10,  247,    8,  247,  247,   23,   21,  247,   31,   22,
 /*   450 */   247,  247,   30,  105,  104,  247,  247,  247,  247,  247,
 /*   460 */   247,  123,  111,  151,  106,  142,   99,   56,   11,   13,
 /*   470 */    32,  247,  114,   10,  247,    8,  247,  247,  159,  125,
 /*   480 */    40,   39,   35,   36,   37,   38,   54,   17,   27,   29,
 /*   490 */    31,   22,  247,  247,   30,  105,  104,  247,   28,  247,
 /*   500 */    53,  116,    6,   57,  247,  247,  247,  247,   26,  247,
 /*   510 */    11,  247,   32,   25,  247,   10,  247,    8,  247,  247,
 /*   520 */    23,   21,  247,  247,  247,  247,  247,  247,  247,  247,
 /*   530 */   247,  247,  247,  247,  247,  247,  247,  247,  151,  106,
 /*   540 */   142,   99,   56,  164,   13,  247,   28,  114,   53,  116,
 /*   550 */     6,   57,  247,  159,  125,  247,   26,  247,  247,  247,
 /*   560 */   247,   25,  247,  247,  247,  247,  247,  247,   23,   21,
 /*   570 */   247,  247,  247,  247,  247,  247,  247,  247,  247,  247,
 /*   580 */   247,  247,  247,  247,  247,  247,  151,  106,  142,   99,
 /*   590 */    56,  247,   13,  247,   28,  114,   53,  116,    6,   57,
 /*   600 */   247,  159,  125,  247,   26,  247,  247,  247,  247,   25,
 /*   610 */   247,  247,  247,  247,  247,  247,   23,   21,  247,  247,
 /*   620 */   247,  247,  247,  247,   27,   29,   31,   22,  247,  247,
 /*   630 */    30,  105,  104,  247,  151,   96,  113,   95,   56,  247,
 /*   640 */    13,  247,  247,  114,  247,  247,   11,  247,   32,  159,
 /*   650 */   125,   10,  247,    8,   22,  247,  247,   30,  105,  104,
 /*   660 */   247,  247,  247,  247,  247,  247,  247,  247,  247,  247,
 /*   670 */   247,  247,  247,   11,  247,   32,  247,  247,   10,  247,
 /*   680 */     8,
};
static const YYCODETYPE yy_lookahead[] = {
 /*     0 */     1,   67,   68,   69,   67,   68,   69,    8,    9,   10,
 /*    10 */    11,   12,   13,   14,   15,   44,   17,   18,   19,   20,
 /*    20 */    21,   22,   23,   24,   25,   26,   27,   28,   60,   61,
 /*    30 */    31,   32,   33,   65,   61,   67,   68,   69,   34,   35,
 /*    40 */    67,   68,   69,   67,   68,   69,   47,   42,   49,   44,
 /*    50 */     1,   52,   68,   54,   67,   68,   69,    8,   68,   10,
 /*    60 */    11,   12,   13,   14,   15,   68,   17,   18,   19,   20,
 /*    70 */    21,   22,   23,   24,   25,   26,   27,   28,   63,    9,
 /*    80 */    31,   32,   33,   68,   67,   68,   69,   62,   63,   67,
 /*    90 */    68,   69,   68,   68,   62,   63,   47,   48,   49,    1,
 /*   100 */    68,   52,   68,   54,   34,   35,    8,   68,   10,   11,
 /*   110 */    12,   13,   14,   15,   40,   17,   18,   19,   20,   21,
 /*   120 */    22,   23,   24,   25,   26,   27,   28,   68,   68,   31,
 /*   130 */    32,   33,   58,   31,   32,   33,   67,   68,   69,   67,
 /*   140 */    68,   69,   67,   68,   69,   47,   48,   49,    1,   47,
 /*   150 */    52,   49,   54,   68,   52,    8,   54,   10,   11,   12,
 /*   160 */    13,   14,   15,   68,   17,   18,   19,   20,   21,   22,
 /*   170 */    23,   24,   25,   26,   27,   28,   62,   63,   31,   32,
 /*   180 */    33,   68,   68,   67,   68,   69,    0,   53,   42,   48,
 /*   190 */    44,   48,   68,   59,   47,   48,   49,    1,   66,   52,
 /*   200 */    59,   54,   59,   42,    8,   44,   10,   11,   12,   13,
 /*   210 */    14,   15,   47,   17,   18,   19,   20,   21,   22,   23,
 /*   220 */    24,   25,   26,   27,   28,   71,   40,   31,   32,   33,
 /*   230 */    42,   42,   44,   44,   42,   55,   44,   34,   35,   59,
 /*   240 */    40,   66,   68,   47,   58,   49,   64,   66,   52,   47,
 /*   250 */    54,    9,    8,   70,   10,   11,   12,   13,   14,   15,
 /*   260 */     9,   17,   18,   19,   20,   21,   22,   23,   24,   25,
 /*   270 */    26,   27,   28,   68,   68,   31,   32,   33,   68,   68,
 /*   280 */    68,   68,   68,   68,   68,   34,   35,   68,   68,    0,
 /*   290 */     0,   47,   47,   49,   48,    7,   52,   47,   54,   44,
 /*   300 */    51,   44,    9,   11,   12,   13,   14,   15,    9,   17,
 /*   310 */    18,   19,   20,   21,   22,   23,   24,   25,   26,   27,
 /*   320 */    28,   46,    0,   31,   32,   33,   68,   68,   68,   68,
 /*   330 */    68,   68,   68,   68,   68,   68,   68,   72,   72,   47,
 /*   340 */    72,   49,   72,   72,   52,   72,   54,   72,   72,   72,
 /*   350 */    72,   72,   12,   13,   14,   15,   72,   17,   18,   19,
 /*   360 */    20,   21,   22,   23,   24,   25,   26,   27,   28,   72,
 /*   370 */    72,   31,   32,   33,   72,   72,   72,   72,   72,   72,
 /*   380 */    72,   72,   72,   72,   72,   72,   72,   47,   72,   49,
 /*   390 */    72,   72,   52,   72,   54,   72,   72,   72,   72,   72,
 /*   400 */    72,   72,   14,   15,   72,   17,   18,   19,   20,   21,
 /*   410 */    22,   23,   24,   25,   26,   27,   28,   72,   72,   31,
 /*   420 */    32,   33,   72,    1,   72,    3,    4,    5,    6,   72,
 /*   430 */    72,   72,   72,   11,   72,   47,   72,   49,   16,   72,
 /*   440 */    52,   72,   54,   72,   72,   23,   24,   72,   27,   28,
 /*   450 */    72,   72,   31,   32,   33,   72,   72,   72,   72,   72,
 /*   460 */    72,   39,   40,   41,   42,   43,   44,   45,   47,   47,
 /*   470 */    49,   72,   50,   52,   72,   54,   72,   72,   56,   57,
 /*   480 */    17,   18,   19,   20,   21,   22,   23,   24,   25,   26,
 /*   490 */    27,   28,   72,   72,   31,   32,   33,   72,    1,   72,
 /*   500 */     3,    4,    5,    6,   72,   72,   72,   72,   11,   72,
 /*   510 */    47,   72,   49,   16,   72,   52,   72,   54,   72,   72,
 /*   520 */    23,   24,   72,   72,   72,   72,   72,   72,   72,   72,
 /*   530 */    72,   72,   72,   72,   72,   72,   72,   72,   41,   42,
 /*   540 */    43,   44,   45,   46,   47,   72,    1,   50,    3,    4,
 /*   550 */     5,    6,   72,   56,   57,   72,   11,   72,   72,   72,
 /*   560 */    72,   16,   72,   72,   72,   72,   72,   72,   23,   24,
 /*   570 */    72,   72,   72,   72,   72,   72,   72,   72,   72,   72,
 /*   580 */    72,   72,   72,   72,   72,   72,   41,   42,   43,   44,
 /*   590 */    45,   72,   47,   72,    1,   50,    3,    4,    5,    6,
 /*   600 */    72,   56,   57,   72,   11,   72,   72,   72,   72,   16,
 /*   610 */    72,   72,   72,   72,   72,   72,   23,   24,   72,   72,
 /*   620 */    72,   72,   72,   72,   25,   26,   27,   28,   72,   72,
 /*   630 */    31,   32,   33,   72,   41,   42,   43,   44,   45,   72,
 /*   640 */    47,   72,   72,   50,   72,   72,   47,   72,   49,   56,
 /*   650 */    57,   52,   72,   54,   28,   72,   72,   31,   32,   33,
 /*   660 */    72,   72,   72,   72,   72,   72,   72,   72,   72,   72,
 /*   670 */    72,   72,   72,   47,   72,   49,   72,   72,   52,   72,
 /*   680 */    54,
};
#define YY_SHIFT_USE_DFLT (-30)
#define YY_SHIFT_MAX 123
static const short yy_shift_ofst[] = {
 /*     0 */   422,  545,  497,  545,  545,  545,  545,  545,  593,  545,
 /*    10 */   593,  593,  545,  545,  545,  545,  593,  545,  545,  545,
 /*    20 */   545,  545,  545,  545,  545,  545,  545,  545,  545,  545,
 /*    30 */   545,  545,  545,  545,  545,  545,  545,  545,  545,  545,
 /*    40 */   545,  545,  545,  545,  545,  545,  545,  545,  545,  545,
 /*    50 */   545,  545,   74,  165,  200,  -29,  200,  202,   98,   -1,
 /*    60 */   147,   49,  196,  196,  196,  196,  196,  196,  196,  244,
 /*    70 */   244,  244,  292,  340,  340,  388,  388,  463,  463,  463,
 /*    80 */   463,  463,  463,  463,  463,  463,  599,  599,  421,  421,
 /*    90 */   626,  102,  102,  102,  102,   70,  251,  186,  134,    4,
 /*   100 */     5,  161,  192,  188,  189,  146,  203,  180,  141,  143,
 /*   110 */   289,  290,  246,  242,  245,  288,  250,  255,  249,  257,
 /*   120 */   293,  299,  275,  322,
};
#define YY_REDUCE_USE_DFLT (-67)
#define YY_REDUCE_MAX 57
static const short yy_reduce_ofst[] = {
 /*     0 */   -32,  -27,   17,   22,   69,  -66,   75,  116,  114,   72,
 /*    10 */    32,   25,  -13,  -24,  -63,   17,   15,  220,  219,  216,
 /*    20 */   215,  214,  213,  212,  211,  210,  206,  205,  174,  258,
 /*    30 */   259,  260,  261,  262,  263,  264,  265,  266,  267,  268,
 /*    40 */   -16,  -10,   -3,   24,   34,   39,   59,   60,   85,   95,
 /*    50 */   113,  124,  132,  154,  175,  182,  181,  183,
};
static const YYACTIONTYPE yy_default[] = {
 /*     0 */   245,  245,  231,  245,  245,  245,  245,  245,  245,  229,
 /*    10 */   245,  245,  245,  245,  245,  231,  245,  245,  245,  245,
 /*    20 */   245,  245,  245,  245,  245,  245,  245,  245,  245,  245,
 /*    30 */   245,  245,  245,  245,  245,  245,  245,  245,  245,  245,
 /*    40 */   245,  245,  245,  245,  245,  245,  245,  237,  239,  235,
 /*    50 */   245,  245,  165,  245,  165,  245,  165,  245,  245,  166,
 /*    60 */   245,  245,  240,  242,  236,  187,  238,  244,  234,  179,
 /*    70 */   189,  200,  201,  178,  188,  199,  197,  177,  191,  195,
 /*    80 */   190,  196,  198,  194,  193,  192,  182,  181,  183,  184,
 /*    90 */   186,  185,  180,  175,  176,  172,  170,  245,  245,  172,
 /*   100 */   245,  245,  245,  245,  245,  245,  170,  245,  245,  245,
 /*   110 */   245,  245,  245,  171,  245,  204,  245,  245,  245,  245,
 /*   120 */   243,  241,  245,  245,  208,  224,  222,  221,  225,  220,
 /*   130 */   219,  226,  173,  227,  174,  164,  228,  216,  215,  212,
 /*   140 */   211,  210,  171,  233,  232,  218,  209,  217,  203,  214,
 /*   150 */   213,  169,  168,  230,  167,  206,  205,  207,  202,  223,
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
  "formlist",      "prog",          "optnl",         "expr_or_assign",
  "expr",          "equal_assign",  "ifcond",        "cond",        
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
 /*   4 */ "optnl ::= NEWLINE",
 /*   5 */ "optnl ::=",
 /*   6 */ "expr_or_assign ::= expr",
 /*   7 */ "expr_or_assign ::= equal_assign",
 /*   8 */ "equal_assign ::= expr EQ_ASSIGN expr_or_assign",
 /*   9 */ "expr ::= NUM_CONST",
 /*  10 */ "expr ::= STR_CONST",
 /*  11 */ "expr ::= NULL_CONST",
 /*  12 */ "expr ::= SYMBOL",
 /*  13 */ "expr ::= LBRACE optnl exprlist optnl RBRACE",
 /*  14 */ "expr ::= LPAREN expr_or_assign RPAREN",
 /*  15 */ "expr ::= MINUS expr",
 /*  16 */ "expr ::= PLUS expr",
 /*  17 */ "expr ::= NOT expr",
 /*  18 */ "expr ::= TILDE expr",
 /*  19 */ "expr ::= QUESTION expr",
 /*  20 */ "expr ::= expr COLON expr",
 /*  21 */ "expr ::= expr PLUS optnl expr",
 /*  22 */ "expr ::= expr MINUS expr",
 /*  23 */ "expr ::= expr TIMES expr",
 /*  24 */ "expr ::= expr DIVIDE expr",
 /*  25 */ "expr ::= expr POW expr",
 /*  26 */ "expr ::= expr SPECIALOP expr",
 /*  27 */ "expr ::= expr PERCENT expr",
 /*  28 */ "expr ::= expr TILDE expr",
 /*  29 */ "expr ::= expr QUESTION expr",
 /*  30 */ "expr ::= expr LT expr",
 /*  31 */ "expr ::= expr LE expr",
 /*  32 */ "expr ::= expr EQ expr",
 /*  33 */ "expr ::= expr NE expr",
 /*  34 */ "expr ::= expr GE expr",
 /*  35 */ "expr ::= expr GT expr",
 /*  36 */ "expr ::= expr AND expr",
 /*  37 */ "expr ::= expr OR expr",
 /*  38 */ "expr ::= expr AND2 expr",
 /*  39 */ "expr ::= expr OR2 expr",
 /*  40 */ "expr ::= expr LEFT_ASSIGN expr",
 /*  41 */ "expr ::= expr RIGHT_ASSIGN expr",
 /*  42 */ "expr ::= FUNCTION LPAREN formlist RPAREN expr_or_assign",
 /*  43 */ "expr ::= expr LPAREN sublist RPAREN",
 /*  44 */ "expr ::= IF ifcond expr_or_assign",
 /*  45 */ "expr ::= IF ifcond expr_or_assign ELSE expr_or_assign",
 /*  46 */ "expr ::= FOR LPAREN SYMBOL IN expr RPAREN expr_or_assign",
 /*  47 */ "expr ::= WHILE cond expr_or_assign",
 /*  48 */ "expr ::= REPEAT expr_or_assign",
 /*  49 */ "expr ::= expr LBB sublist RBB",
 /*  50 */ "expr ::= expr LBRACKET sublist RBRACKET",
 /*  51 */ "expr ::= SYMBOL NS_GET SYMBOL",
 /*  52 */ "expr ::= SYMBOL NS_GET STR_CONST",
 /*  53 */ "expr ::= STR_CONST NS_GET SYMBOL",
 /*  54 */ "expr ::= STR_CONST NS_GET STR_CONST",
 /*  55 */ "expr ::= SYMBOL NS_GET_INT SYMBOL",
 /*  56 */ "expr ::= SYMBOL NS_GET_INT STR_CONST",
 /*  57 */ "expr ::= STR_CONST NS_GET_INT SYMBOL",
 /*  58 */ "expr ::= STR_CONST NS_GET_INT STR_CONST",
 /*  59 */ "expr ::= expr DOLLAR SYMBOL",
 /*  60 */ "expr ::= expr DOLLAR STR_CONST",
 /*  61 */ "expr ::= expr AT SYMBOL",
 /*  62 */ "expr ::= expr AT STR_CONST",
 /*  63 */ "expr ::= NEXT",
 /*  64 */ "expr ::= BREAK",
 /*  65 */ "cond ::= LPAREN expr RPAREN",
 /*  66 */ "ifcond ::= LPAREN expr RPAREN",
 /*  67 */ "exprlist ::= expr_or_assign",
 /*  68 */ "exprlist ::= exprlist SEMICOLON expr_or_assign",
 /*  69 */ "exprlist ::= exprlist SEMICOLON",
 /*  70 */ "exprlist ::= exprlist NEWLINE expr_or_assign",
 /*  71 */ "exprlist ::= exprlist NEWLINE",
 /*  72 */ "sublist ::= sub",
 /*  73 */ "sublist ::= sublist COMMA sub",
 /*  74 */ "sub ::= expr",
 /*  75 */ "sub ::= SYMBOL EQ_ASSIGN",
 /*  76 */ "sub ::= SYMBOL EQ_ASSIGN expr",
 /*  77 */ "sub ::= STR_CONST EQ_ASSIGN",
 /*  78 */ "sub ::= STR_CONST EQ_ASSIGN expr",
 /*  79 */ "sub ::= NULL_CONST EQ_ASSIGN",
 /*  80 */ "sub ::= NULL_CONST EQ_ASSIGN expr",
 /*  81 */ "formlist ::= SYMBOL",
 /*  82 */ "formlist ::= SYMBOL EQ_ASSIGN expr",
 /*  83 */ "formlist ::= formlist COMMA SYMBOL",
 /*  84 */ "formlist ::= formlist COMMA SYMBOL EQ_ASSIGN expr",
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
  { 66, 0 },
  { 67, 1 },
  { 67, 1 },
  { 69, 3 },
  { 68, 1 },
  { 68, 1 },
  { 68, 1 },
  { 68, 1 },
  { 68, 5 },
  { 68, 3 },
  { 68, 2 },
  { 68, 2 },
  { 68, 2 },
  { 68, 2 },
  { 68, 2 },
  { 68, 3 },
  { 68, 4 },
  { 68, 3 },
  { 68, 3 },
  { 68, 3 },
  { 68, 3 },
  { 68, 3 },
  { 68, 3 },
  { 68, 3 },
  { 68, 3 },
  { 68, 3 },
  { 68, 3 },
  { 68, 3 },
  { 68, 3 },
  { 68, 3 },
  { 68, 3 },
  { 68, 3 },
  { 68, 3 },
  { 68, 3 },
  { 68, 3 },
  { 68, 3 },
  { 68, 3 },
  { 68, 5 },
  { 68, 4 },
  { 68, 3 },
  { 68, 5 },
  { 68, 7 },
  { 68, 3 },
  { 68, 2 },
  { 68, 4 },
  { 68, 4 },
  { 68, 3 },
  { 68, 3 },
  { 68, 3 },
  { 68, 3 },
  { 68, 3 },
  { 68, 3 },
  { 68, 3 },
  { 68, 3 },
  { 68, 3 },
  { 68, 3 },
  { 68, 3 },
  { 68, 3 },
  { 68, 1 },
  { 68, 1 },
  { 71, 3 },
  { 70, 3 },
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
#line 60 "parser.y"
{ result->value = yygotominor.yy0 = Expression(0); }
#line 988 "parser.c"
        break;
      case 2: /* prog ::= exprlist */
#line 64 "parser.y"
{ result->value = yygotominor.yy0 = Expression(List(yymsp[0].minor.yy108)); }
#line 993 "parser.c"
        break;
      case 6: /* expr_or_assign ::= expr */
      case 7: /* expr_or_assign ::= equal_assign */ yytestcase(yyruleno==7);
      case 9: /* expr ::= NUM_CONST */ yytestcase(yyruleno==9);
      case 10: /* expr ::= STR_CONST */ yytestcase(yyruleno==10);
      case 11: /* expr ::= NULL_CONST */ yytestcase(yyruleno==11);
      case 12: /* expr ::= SYMBOL */ yytestcase(yyruleno==12);
#line 70 "parser.y"
{ yygotominor.yy0 = yymsp[0].minor.yy0; }
#line 1003 "parser.c"
        break;
      case 8: /* equal_assign ::= expr EQ_ASSIGN expr_or_assign */
      case 20: /* expr ::= expr COLON expr */ yytestcase(yyruleno==20);
      case 22: /* expr ::= expr MINUS expr */ yytestcase(yyruleno==22);
      case 23: /* expr ::= expr TIMES expr */ yytestcase(yyruleno==23);
      case 24: /* expr ::= expr DIVIDE expr */ yytestcase(yyruleno==24);
      case 25: /* expr ::= expr POW expr */ yytestcase(yyruleno==25);
      case 26: /* expr ::= expr SPECIALOP expr */ yytestcase(yyruleno==26);
      case 27: /* expr ::= expr PERCENT expr */ yytestcase(yyruleno==27);
      case 28: /* expr ::= expr TILDE expr */ yytestcase(yyruleno==28);
      case 29: /* expr ::= expr QUESTION expr */ yytestcase(yyruleno==29);
      case 30: /* expr ::= expr LT expr */ yytestcase(yyruleno==30);
      case 31: /* expr ::= expr LE expr */ yytestcase(yyruleno==31);
      case 32: /* expr ::= expr EQ expr */ yytestcase(yyruleno==32);
      case 33: /* expr ::= expr NE expr */ yytestcase(yyruleno==33);
      case 34: /* expr ::= expr GE expr */ yytestcase(yyruleno==34);
      case 35: /* expr ::= expr GT expr */ yytestcase(yyruleno==35);
      case 36: /* expr ::= expr AND expr */ yytestcase(yyruleno==36);
      case 37: /* expr ::= expr OR expr */ yytestcase(yyruleno==37);
      case 38: /* expr ::= expr AND2 expr */ yytestcase(yyruleno==38);
      case 39: /* expr ::= expr OR2 expr */ yytestcase(yyruleno==39);
      case 40: /* expr ::= expr LEFT_ASSIGN expr */ yytestcase(yyruleno==40);
      case 51: /* expr ::= SYMBOL NS_GET SYMBOL */ yytestcase(yyruleno==51);
      case 52: /* expr ::= SYMBOL NS_GET STR_CONST */ yytestcase(yyruleno==52);
      case 53: /* expr ::= STR_CONST NS_GET SYMBOL */ yytestcase(yyruleno==53);
      case 54: /* expr ::= STR_CONST NS_GET STR_CONST */ yytestcase(yyruleno==54);
      case 55: /* expr ::= SYMBOL NS_GET_INT SYMBOL */ yytestcase(yyruleno==55);
      case 56: /* expr ::= SYMBOL NS_GET_INT STR_CONST */ yytestcase(yyruleno==56);
      case 57: /* expr ::= STR_CONST NS_GET_INT SYMBOL */ yytestcase(yyruleno==57);
      case 58: /* expr ::= STR_CONST NS_GET_INT STR_CONST */ yytestcase(yyruleno==58);
      case 59: /* expr ::= expr DOLLAR SYMBOL */ yytestcase(yyruleno==59);
      case 60: /* expr ::= expr DOLLAR STR_CONST */ yytestcase(yyruleno==60);
      case 61: /* expr ::= expr AT SYMBOL */ yytestcase(yyruleno==61);
      case 62: /* expr ::= expr AT STR_CONST */ yytestcase(yyruleno==62);
#line 73 "parser.y"
{ yygotominor.yy0 = Call::c(yymsp[-1].minor.yy0, yymsp[-2].minor.yy0, yymsp[0].minor.yy0); }
#line 1040 "parser.c"
        break;
      case 13: /* expr ::= LBRACE optnl exprlist optnl RBRACE */
#line 80 "parser.y"
{ yymsp[-2].minor.yy108.push_front(Symbol(0), yymsp[-4].minor.yy0); yygotominor.yy0 = Expression(List(yymsp[-2].minor.yy108)); }
#line 1045 "parser.c"
        break;
      case 14: /* expr ::= LPAREN expr_or_assign RPAREN */
      case 65: /* cond ::= LPAREN expr RPAREN */ yytestcase(yyruleno==65);
      case 66: /* ifcond ::= LPAREN expr RPAREN */ yytestcase(yyruleno==66);
#line 81 "parser.y"
{ yygotominor.yy0 = yymsp[-1].minor.yy0; }
#line 1052 "parser.c"
        break;
      case 15: /* expr ::= MINUS expr */
      case 16: /* expr ::= PLUS expr */ yytestcase(yyruleno==16);
      case 17: /* expr ::= NOT expr */ yytestcase(yyruleno==17);
      case 18: /* expr ::= TILDE expr */ yytestcase(yyruleno==18);
      case 19: /* expr ::= QUESTION expr */ yytestcase(yyruleno==19);
      case 48: /* expr ::= REPEAT expr_or_assign */ yytestcase(yyruleno==48);
#line 83 "parser.y"
{ yygotominor.yy0 = Call::c(yymsp[-1].minor.yy0, yymsp[0].minor.yy0); }
#line 1062 "parser.c"
        break;
      case 21: /* expr ::= expr PLUS optnl expr */
#line 90 "parser.y"
{ yygotominor.yy0 = Call::c(yymsp[-2].minor.yy0, yymsp[-3].minor.yy0, yymsp[0].minor.yy0); }
#line 1067 "parser.c"
        break;
      case 41: /* expr ::= expr RIGHT_ASSIGN expr */
#line 111 "parser.y"
{ yygotominor.yy0 = Call::c(yymsp[-1].minor.yy0, yymsp[0].minor.yy0, yymsp[-2].minor.yy0); }
#line 1072 "parser.c"
        break;
      case 42: /* expr ::= FUNCTION LPAREN formlist RPAREN expr_or_assign */
#line 112 "parser.y"
{ yygotominor.yy0 = Call::c(yymsp[-4].minor.yy0, PairList(List(yymsp[-2].minor.yy108)), yymsp[0].minor.yy0); }
#line 1077 "parser.c"
        break;
      case 43: /* expr ::= expr LPAREN sublist RPAREN */
#line 113 "parser.y"
{ yymsp[-1].minor.yy108.push_front(Symbol(0), yymsp[-3].minor.yy0); yygotominor.yy0 = Call(List(yymsp[-1].minor.yy108)); }
#line 1082 "parser.c"
        break;
      case 44: /* expr ::= IF ifcond expr_or_assign */
      case 47: /* expr ::= WHILE cond expr_or_assign */ yytestcase(yyruleno==47);
#line 114 "parser.y"
{ yygotominor.yy0 = Call::c(yymsp[-2].minor.yy0, yymsp[-1].minor.yy0, yymsp[0].minor.yy0); }
#line 1088 "parser.c"
        break;
      case 45: /* expr ::= IF ifcond expr_or_assign ELSE expr_or_assign */
#line 115 "parser.y"
{ yygotominor.yy0 = Call::c(yymsp[-4].minor.yy0, yymsp[-3].minor.yy0, yymsp[-2].minor.yy0, yymsp[0].minor.yy0); }
#line 1093 "parser.c"
        break;
      case 46: /* expr ::= FOR LPAREN SYMBOL IN expr RPAREN expr_or_assign */
#line 116 "parser.y"
{ yygotominor.yy0 = Call::c(yymsp[-6].minor.yy0, yymsp[-4].minor.yy0, yymsp[-2].minor.yy0, yymsp[0].minor.yy0); }
#line 1098 "parser.c"
        break;
      case 49: /* expr ::= expr LBB sublist RBB */
      case 50: /* expr ::= expr LBRACKET sublist RBRACKET */ yytestcase(yyruleno==50);
#line 119 "parser.y"
{ yymsp[-1].minor.yy108.push_front(Symbol(0), yymsp[-3].minor.yy0); yymsp[-1].minor.yy108.push_front(Symbol(0), yymsp[-2].minor.yy0); yygotominor.yy0 = Call(List(yymsp[-1].minor.yy108)); }
#line 1104 "parser.c"
        break;
      case 63: /* expr ::= NEXT */
      case 64: /* expr ::= BREAK */ yytestcase(yyruleno==64);
#line 133 "parser.y"
{ yygotominor.yy0 = Call::c(yymsp[0].minor.yy0); }
#line 1110 "parser.c"
        break;
      case 67: /* exprlist ::= expr_or_assign */
      case 74: /* sub ::= expr */ yytestcase(yyruleno==74);
      case 80: /* sub ::= NULL_CONST EQ_ASSIGN expr */ yytestcase(yyruleno==80);
#line 139 "parser.y"
{ yygotominor.yy108 = Pairs::Make(); yygotominor.yy108.push_back(Symbol(0), yymsp[0].minor.yy0); }
#line 1117 "parser.c"
        break;
      case 68: /* exprlist ::= exprlist SEMICOLON expr_or_assign */
      case 70: /* exprlist ::= exprlist NEWLINE expr_or_assign */ yytestcase(yyruleno==70);
#line 140 "parser.y"
{ yygotominor.yy108 = yymsp[-2].minor.yy108; yygotominor.yy108.push_back(Symbol(0), yymsp[0].minor.yy0); }
#line 1123 "parser.c"
        break;
      case 69: /* exprlist ::= exprlist SEMICOLON */
      case 71: /* exprlist ::= exprlist NEWLINE */ yytestcase(yyruleno==71);
#line 141 "parser.y"
{ yygotominor.yy108 = yymsp[-1].minor.yy108; }
#line 1129 "parser.c"
        break;
      case 72: /* sublist ::= sub */
#line 145 "parser.y"
{ yygotominor.yy108 = yymsp[0].minor.yy108; }
#line 1134 "parser.c"
        break;
      case 73: /* sublist ::= sublist COMMA sub */
#line 146 "parser.y"
{ yygotominor.yy108 = yymsp[-2].minor.yy108; yygotominor.yy108.push_back(yymsp[0].minor.yy108.name(0), yymsp[0].minor.yy108.value(0)); }
#line 1139 "parser.c"
        break;
      case 75: /* sub ::= SYMBOL EQ_ASSIGN */
      case 77: /* sub ::= STR_CONST EQ_ASSIGN */ yytestcase(yyruleno==77);
#line 149 "parser.y"
{ yygotominor.yy108 = Pairs::Make(); yygotominor.yy108.push_back(yymsp[-1].minor.yy0, Value::NIL); }
#line 1145 "parser.c"
        break;
      case 76: /* sub ::= SYMBOL EQ_ASSIGN expr */
      case 78: /* sub ::= STR_CONST EQ_ASSIGN expr */ yytestcase(yyruleno==78);
      case 82: /* formlist ::= SYMBOL EQ_ASSIGN expr */ yytestcase(yyruleno==82);
#line 150 "parser.y"
{ yygotominor.yy108 = Pairs::Make(); yygotominor.yy108.push_back(Symbol(yymsp[-2].minor.yy0), yymsp[0].minor.yy0); }
#line 1152 "parser.c"
        break;
      case 79: /* sub ::= NULL_CONST EQ_ASSIGN */
#line 153 "parser.y"
{ yygotominor.yy108 = Pairs::Make(); yygotominor.yy108.push_back(Symbol(0), Value::NIL); }
#line 1157 "parser.c"
        break;
      case 81: /* formlist ::= SYMBOL */
#line 156 "parser.y"
{ yygotominor.yy108 = Pairs::Make(); yygotominor.yy108.push_back(Symbol(yymsp[0].minor.yy0), Value::NIL); }
#line 1162 "parser.c"
        break;
      case 83: /* formlist ::= formlist COMMA SYMBOL */
#line 158 "parser.y"
{ yygotominor.yy108 = yymsp[-2].minor.yy108; yygotominor.yy108.push_back(Symbol(yymsp[0].minor.yy0), Value::NIL); }
#line 1167 "parser.c"
        break;
      case 84: /* formlist ::= formlist COMMA SYMBOL EQ_ASSIGN expr */
#line 159 "parser.y"
{ yygotominor.yy108 = yymsp[-4].minor.yy108; yygotominor.yy108.push_back(Symbol(yymsp[-2].minor.yy0), yymsp[0].minor.yy0); }
#line 1172 "parser.c"
        break;
      default:
      /* (4) optnl ::= NEWLINE */ yytestcase(yyruleno==4);
      /* (5) optnl ::= */ yytestcase(yyruleno==5);
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
#line 1226 "parser.c"
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
#line 1245 "parser.c"
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
#line 1267 "parser.c"
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
