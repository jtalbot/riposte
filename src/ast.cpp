/* Driver template for the LEMON parser generator.
** The author disclaims copyright to this source code.
*/
/* First off, code is included that follows the "include" declaration
** in the input grammar file. */
#include <stdio.h>
#line 42 "parser.y"

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
#define YYNOCODE 68
#define YYACTIONTYPE unsigned char
#define ParseTOKENTYPE Value
typedef union {
  int yyinit;
  ParseTOKENTYPE yy0;
  Pairs yy18;
  int yy135;
} YYMINORTYPE;
#ifndef YYSTACKDEPTH
#define YYSTACKDEPTH 100
#endif
#define ParseARG_SDECL Parser::Result* result;
#define ParseARG_PDECL ,Parser::Result* result
#define ParseARG_FETCH Parser::Result* result = yypParser->result
#define ParseARG_STORE yypParser->result = result
#define YYNSTATE 149
#define YYNRULE 78
#define YYERRORSYMBOL 55
#define YYERRSYMDT yy135
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
 /*     0 */    37,  103,  143,   18,   53,   52,   63,   49,    9,   16,
 /*    10 */    36,   46,   48,   44,   47,   22,   43,   42,   38,   39,
 /*    20 */    40,   41,   27,   29,   31,   33,   35,   25,   53,   52,
 /*    30 */    34,   55,   54,   37,  142,    7,   11,   13,   63,  135,
 /*    40 */    49,  133,   16,   36,   46,   48,   44,   47,   72,   43,
 /*    50 */    42,   38,   39,   40,   41,   27,   29,   31,   33,   35,
 /*    60 */    25,  100,  143,   34,   55,   54,   63,   98,    7,   11,
 /*    70 */    13,   37,  147,   62,  140,  121,   62,  140,   49,  134,
 /*    80 */    16,   36,   46,   48,   44,   47,   65,   43,   42,   38,
 /*    90 */    39,   40,   41,   27,   29,   31,   33,   35,   25,    5,
 /*   100 */   144,   34,   55,   54,  124,   15,    7,   11,   13,   37,
 /*   110 */    67,    8,   17,  148,   62,  140,   49,    6,   16,   36,
 /*   120 */    46,   48,   44,   47,   66,   43,   42,   38,   39,   40,
 /*   130 */    41,   27,   29,   31,   33,   35,   25,   51,   50,   34,
 /*   140 */    55,   54,   64,   60,    7,   11,   13,   37,  141,  130,
 /*   150 */    62,  140,   15,   59,   49,  137,   16,   36,   46,   48,
 /*   160 */    44,   47,   61,   43,   42,   38,   39,   40,   41,   27,
 /*   170 */    29,   31,   33,   35,   25,  104,  143,   34,   55,   54,
 /*   180 */    63,   94,    7,   11,   13,   92,   49,   12,   16,   36,
 /*   190 */    46,   48,   44,   47,  110,   43,   42,   38,   39,   40,
 /*   200 */    41,   27,   29,   31,   33,   35,   25,   51,   50,   34,
 /*   210 */    55,   54,   93,   87,    7,   11,   13,   36,   46,   48,
 /*   220 */    44,   47,   78,   43,   42,   38,   39,   40,   41,   27,
 /*   230 */    29,   31,   33,   35,   25,  125,   86,   34,   55,   54,
 /*   240 */     3,   74,    7,   11,   13,   46,   48,   44,   47,   89,
 /*   250 */    43,   42,   38,   39,   40,   41,   27,   29,   31,   33,
 /*   260 */    35,   25,   70,   88,   34,   55,   54,  145,  127,    7,
 /*   270 */    11,   13,   44,   47,   15,   43,   42,   38,   39,   40,
 /*   280 */    41,   27,   29,   31,   33,   35,   25,   91,   90,   34,
 /*   290 */    55,   54,  151,   73,    7,   11,   13,   32,  115,   56,
 /*   300 */   108,    2,   58,   31,   33,   35,   25,   30,   69,   34,
 /*   310 */    55,   54,   28,   79,    7,   11,   13,   35,   25,   26,
 /*   320 */    24,   34,   55,   54,   77,   80,    7,   11,   13,   81,
 /*   330 */    82,  132,    4,    5,   83,    1,  116,  114,  138,  101,
 /*   340 */   129,  102,  139,   62,  140,    8,   84,  131,  118,   68,
 /*   350 */    43,   42,   38,   39,   40,   41,   27,   29,   31,   33,
 /*   360 */    35,   25,   75,   85,   34,   55,   54,   76,   71,    7,
 /*   370 */    11,   13,   32,  115,   56,  108,    2,   58,  136,  120,
 /*   380 */    25,  119,   30,   34,   55,   54,   99,   28,    7,   11,
 /*   390 */    13,  113,   21,   14,   26,   24,   32,  115,   56,  108,
 /*   400 */     2,   58,  107,   62,  140,   10,   30,    4,  122,   19,
 /*   410 */     1,   28,  109,  138,  101,  129,  102,  112,   26,   24,
 /*   420 */   117,   96,  131,  118,  111,  228,  147,   62,  140,   20,
 /*   430 */    23,    4,   45,  150,    1,  149,  152,  138,   95,  105,
 /*   440 */    97,  229,   57,   34,   55,   54,  131,  118,    7,   11,
 /*   450 */    13,  106,   62,  140,  229,  229,  229,  123,   62,  140,
 /*   460 */   146,   62,  140,  128,   62,  140,  126,   62,  140,
};
static const YYCODETYPE yy_lookahead[] = {
 /*     0 */     1,   57,   58,    9,   34,   35,   62,    8,    9,   10,
 /*    10 */    11,   12,   13,   14,   15,   36,   17,   18,   19,   20,
 /*    20 */    21,   22,   23,   24,   25,   26,   27,   28,   34,   35,
 /*    30 */    31,   32,   33,    1,   58,   36,   37,   38,   62,   43,
 /*    40 */     8,   45,   10,   11,   12,   13,   14,   15,   62,   17,
 /*    50 */    18,   19,   20,   21,   22,   23,   24,   25,   26,   27,
 /*    60 */    28,   57,   58,   31,   32,   33,   62,   56,   36,   37,
 /*    70 */    38,    1,   61,   62,   63,   61,   62,   63,    8,   47,
 /*    80 */    10,   11,   12,   13,   14,   15,   62,   17,   18,   19,
 /*    90 */    20,   21,   22,   23,   24,   25,   26,   27,   28,   41,
 /*   100 */    49,   31,   32,   33,   46,   54,   36,   37,   38,    1,
 /*   110 */    62,   53,    9,   61,   62,   63,    8,   47,   10,   11,
 /*   120 */    12,   13,   14,   15,   62,   17,   18,   19,   20,   21,
 /*   130 */    22,   23,   24,   25,   26,   27,   28,   34,   35,   31,
 /*   140 */    32,   33,   62,   62,   36,   37,   38,    1,   50,   61,
 /*   150 */    62,   63,   54,   62,    8,   47,   10,   11,   12,   13,
 /*   160 */    14,   15,   62,   17,   18,   19,   20,   21,   22,   23,
 /*   170 */    24,   25,   26,   27,   28,   57,   58,   31,   32,   33,
 /*   180 */    62,   62,   36,   37,   38,   62,    8,   47,   10,   11,
 /*   190 */    12,   13,   14,   15,   54,   17,   18,   19,   20,   21,
 /*   200 */    22,   23,   24,   25,   26,   27,   28,   34,   35,   31,
 /*   210 */    32,   33,   62,   62,   36,   37,   38,   11,   12,   13,
 /*   220 */    14,   15,   62,   17,   18,   19,   20,   21,   22,   23,
 /*   230 */    24,   25,   26,   27,   28,   66,   62,   31,   32,   33,
 /*   240 */    65,   62,   36,   37,   38,   12,   13,   14,   15,   62,
 /*   250 */    17,   18,   19,   20,   21,   22,   23,   24,   25,   26,
 /*   260 */    27,   28,   62,   62,   31,   32,   33,   47,   66,   36,
 /*   270 */    37,   38,   14,   15,   54,   17,   18,   19,   20,   21,
 /*   280 */    22,   23,   24,   25,   26,   27,   28,   62,   62,   31,
 /*   290 */    32,   33,    0,   62,   36,   37,   38,    1,    2,    3,
 /*   300 */     4,    5,    6,   25,   26,   27,   28,   11,   62,   31,
 /*   310 */    32,   33,   16,   62,   36,   37,   38,   27,   28,   23,
 /*   320 */    24,   31,   32,   33,   62,   62,   36,   37,   38,   62,
 /*   330 */    62,   66,   36,   41,   62,   39,   40,   41,   42,   43,
 /*   340 */    44,   45,   61,   62,   63,   53,   62,   51,   52,   62,
 /*   350 */    17,   18,   19,   20,   21,   22,   23,   24,   25,   26,
 /*   360 */    27,   28,   62,   62,   31,   32,   33,   62,   62,   36,
 /*   370 */    37,   38,    1,    2,    3,    4,    5,    6,   66,   66,
 /*   380 */    28,   66,   11,   31,   32,   33,   59,   16,   36,   37,
 /*   390 */    38,   45,   36,   64,   23,   24,    1,    2,    3,    4,
 /*   400 */     5,    6,   61,   62,   63,    7,   11,   36,   47,    9,
 /*   410 */    39,   16,   36,   42,   43,   44,   45,   45,   23,   24,
 /*   420 */    55,   56,   51,   52,   45,   60,   61,   62,   63,    9,
 /*   430 */    48,   36,    9,    0,   39,    0,    0,   42,   43,   44,
 /*   440 */    45,   67,   36,   31,   32,   33,   51,   52,   36,   37,
 /*   450 */    38,   61,   62,   63,   67,   67,   67,   61,   62,   63,
 /*   460 */    61,   62,   63,   61,   62,   63,   61,   62,   63,
};
#define YY_SHIFT_USE_DFLT (-31)
#define YY_SHIFT_MAX 117
static const short yy_shift_ofst[] = {
 /*     0 */   296,  371,  371,  371,  371,  371,  371,  395,  371,  371,
 /*    10 */   371,  395,  371,  395,  371,  395,  371,  371,  371,  371,
 /*    20 */   371,  371,  371,  371,  371,  371,  371,  371,  371,  371,
 /*    30 */   371,  371,  371,  371,  371,  371,  371,  371,  371,  371,
 /*    40 */   371,  371,  371,  371,  371,  371,  371,  371,  371,  371,
 /*    50 */    -4,   -4,   -4,   -4,   -4,   -4,  -21,  346,  356,   32,
 /*    60 */   108,   70,   -1,  146,  146,  146,  146,  146,  146,  178,
 /*    70 */   178,  178,  206,  233,  233,  258,  258,  333,  333,  333,
 /*    80 */   333,  333,  333,  333,  333,  333,  278,  278,  290,  290,
 /*    90 */   352,  412,  412,  412,  412,   -6,  292,  103,   58,  140,
 /*   100 */    51,  -30,  173,  220,   98,  400,  398,  361,  376,  372,
 /*   110 */   379,  420,  382,  423,  433,  406,  435,  436,
};
#define YY_REDUCE_USE_DFLT (-57)
#define YY_REDUCE_MAX 58
static const short yy_reduce_ofst[] = {
 /*     0 */   365,   11,   88,  402,  341,  399,  405,  -56,   52,  281,
 /*    10 */   396,  118,   14,    4,  390,  -24,  -14,   24,   48,   62,
 /*    20 */    80,   81,   91,  100,  119,  123,  150,  151,  160,  174,
 /*    30 */   179,  187,  200,  201,  225,  226,  231,  246,  251,  262,
 /*    40 */   263,  267,  268,  272,  284,  287,  300,  301,  305,  306,
 /*    50 */   169,  202,  265,  312,  313,  315,  175,  327,  329,
};
static const YYACTIONTYPE yy_default[] = {
 /*     0 */   227,  227,  227,  227,  227,  213,  227,  227,  211,  227,
 /*    10 */   227,  227,  227,  227,  227,  227,  227,  217,  218,  221,
 /*    20 */   227,  227,  227,  227,  227,  227,  227,  227,  227,  227,
 /*    30 */   227,  227,  227,  227,  227,  227,  227,  227,  227,  227,
 /*    40 */   227,  227,  227,  227,  227,  227,  227,  227,  227,  227,
 /*    50 */   227,  227,  227,  227,  227,  227,  227,  227,  227,  227,
 /*    60 */   227,  227,  153,  216,  226,  219,  222,  220,  224,  175,
 /*    70 */   166,  186,  187,  174,  165,  183,  185,  177,  164,  176,
 /*    80 */   178,  179,  180,  181,  182,  184,  169,  168,  171,  170,
 /*    90 */   173,  172,  167,  163,  162,  157,  227,  159,  227,  227,
 /*   100 */   227,  157,  159,  227,  227,  158,  190,  227,  227,  227,
 /*   110 */   227,  225,  227,  223,  227,  227,  227,  227,  204,  201,
 /*   120 */   202,  188,  161,  191,  160,  199,  192,  197,  193,  158,
 /*   130 */   194,  203,  200,  208,  205,  207,  198,  206,  156,  155,
 /*   140 */   154,  196,  215,  214,  195,  189,  212,  209,  210,
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
  "END_OF_INPUT",  "NEWLINE",       "NUM_CONST",     "STR_CONST",   
  "NULL_CONST",    "SYMBOL",        "RBRACE",        "RPAREN",      
  "IN",            "RBB",           "RBRACKET",      "NEXT",        
  "BREAK",         "SEMICOLON",     "COMMA",         "error",       
  "exprlist",      "sublist",       "sub",           "formlist",    
  "prog",          "expr_or_assign",  "expr",          "equal_assign",
  "ifcond",        "cond",          "symbolstr",   
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
 /*  25 */ "expr ::= expr TILDE expr",
 /*  26 */ "expr ::= expr QUESTION expr",
 /*  27 */ "expr ::= expr LT expr",
 /*  28 */ "expr ::= expr LE expr",
 /*  29 */ "expr ::= expr EQ expr",
 /*  30 */ "expr ::= expr NE expr",
 /*  31 */ "expr ::= expr GE expr",
 /*  32 */ "expr ::= expr GT expr",
 /*  33 */ "expr ::= expr AND expr",
 /*  34 */ "expr ::= expr OR expr",
 /*  35 */ "expr ::= expr AND2 expr",
 /*  36 */ "expr ::= expr OR2 expr",
 /*  37 */ "expr ::= expr LEFT_ASSIGN expr",
 /*  38 */ "expr ::= expr RIGHT_ASSIGN expr",
 /*  39 */ "expr ::= FUNCTION LPAREN formlist RPAREN expr_or_assign",
 /*  40 */ "expr ::= expr LPAREN sublist RPAREN",
 /*  41 */ "expr ::= IF ifcond expr_or_assign",
 /*  42 */ "expr ::= IF ifcond expr_or_assign ELSE expr_or_assign",
 /*  43 */ "expr ::= FOR LPAREN SYMBOL IN expr RPAREN expr_or_assign",
 /*  44 */ "expr ::= WHILE cond expr_or_assign",
 /*  45 */ "expr ::= REPEAT expr_or_assign",
 /*  46 */ "expr ::= expr LBB sublist RBB",
 /*  47 */ "expr ::= expr LBRACKET sublist RBRACKET",
 /*  48 */ "expr ::= SYMBOL NS_GET symbolstr",
 /*  49 */ "expr ::= STR_CONST NS_GET symbolstr",
 /*  50 */ "expr ::= SYMBOL NS_GET_INT symbolstr",
 /*  51 */ "expr ::= STR_CONST NS_GET_INT symbolstr",
 /*  52 */ "expr ::= expr DOLLAR symbolstr",
 /*  53 */ "expr ::= expr AT symbolstr",
 /*  54 */ "expr ::= NEXT",
 /*  55 */ "expr ::= BREAK",
 /*  56 */ "cond ::= LPAREN expr RPAREN",
 /*  57 */ "ifcond ::= LPAREN expr RPAREN",
 /*  58 */ "symbolstr ::= STR_CONST",
 /*  59 */ "symbolstr ::= SYMBOL",
 /*  60 */ "exprlist ::= expr_or_assign",
 /*  61 */ "exprlist ::= exprlist SEMICOLON expr_or_assign",
 /*  62 */ "exprlist ::= exprlist SEMICOLON",
 /*  63 */ "exprlist ::= exprlist NEWLINE expr_or_assign",
 /*  64 */ "exprlist ::= exprlist NEWLINE",
 /*  65 */ "sublist ::= sub",
 /*  66 */ "sublist ::= sublist COMMA sub",
 /*  67 */ "sub ::= expr",
 /*  68 */ "sub ::= SYMBOL EQ_ASSIGN",
 /*  69 */ "sub ::= STR_CONST EQ_ASSIGN",
 /*  70 */ "sub ::= SYMBOL EQ_ASSIGN expr",
 /*  71 */ "sub ::= STR_CONST EQ_ASSIGN expr",
 /*  72 */ "sub ::= NULL_CONST EQ_ASSIGN",
 /*  73 */ "sub ::= NULL_CONST EQ_ASSIGN expr",
 /*  74 */ "formlist ::= SYMBOL",
 /*  75 */ "formlist ::= SYMBOL EQ_ASSIGN expr",
 /*  76 */ "formlist ::= formlist COMMA SYMBOL",
 /*  77 */ "formlist ::= formlist COMMA SYMBOL EQ_ASSIGN expr",
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
  { 60, 1 },
  { 60, 1 },
  { 60, 1 },
  { 60, 1 },
  { 61, 1 },
  { 61, 1 },
  { 63, 3 },
  { 62, 1 },
  { 62, 1 },
  { 62, 1 },
  { 62, 1 },
  { 62, 3 },
  { 62, 3 },
  { 62, 2 },
  { 62, 2 },
  { 62, 2 },
  { 62, 2 },
  { 62, 2 },
  { 62, 3 },
  { 62, 3 },
  { 62, 3 },
  { 62, 3 },
  { 62, 3 },
  { 62, 3 },
  { 62, 3 },
  { 62, 3 },
  { 62, 3 },
  { 62, 3 },
  { 62, 3 },
  { 62, 3 },
  { 62, 3 },
  { 62, 3 },
  { 62, 3 },
  { 62, 3 },
  { 62, 3 },
  { 62, 3 },
  { 62, 3 },
  { 62, 3 },
  { 62, 3 },
  { 62, 5 },
  { 62, 4 },
  { 62, 3 },
  { 62, 5 },
  { 62, 7 },
  { 62, 3 },
  { 62, 2 },
  { 62, 4 },
  { 62, 4 },
  { 62, 3 },
  { 62, 3 },
  { 62, 3 },
  { 62, 3 },
  { 62, 3 },
  { 62, 3 },
  { 62, 1 },
  { 62, 1 },
  { 65, 3 },
  { 64, 3 },
  { 66, 1 },
  { 66, 1 },
  { 56, 1 },
  { 56, 3 },
  { 56, 2 },
  { 56, 3 },
  { 56, 2 },
  { 57, 1 },
  { 57, 3 },
  { 58, 1 },
  { 58, 2 },
  { 58, 2 },
  { 58, 3 },
  { 58, 3 },
  { 58, 2 },
  { 58, 3 },
  { 59, 1 },
  { 59, 3 },
  { 59, 3 },
  { 59, 5 },
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
#line 61 "parser.y"
{ result->value = yygotominor.yy0 = Expression(0); }
#line 927 "parser.c"
        break;
      case 2: /* prog ::= exprlist */
#line 63 "parser.y"
{ result->value = yygotominor.yy0 = Expression(List(yymsp[0].minor.yy18)); }
#line 932 "parser.c"
        break;
      case 4: /* expr_or_assign ::= expr */
      case 5: /* expr_or_assign ::= equal_assign */ yytestcase(yyruleno==5);
      case 7: /* expr ::= NUM_CONST */ yytestcase(yyruleno==7);
      case 8: /* expr ::= STR_CONST */ yytestcase(yyruleno==8);
      case 9: /* expr ::= NULL_CONST */ yytestcase(yyruleno==9);
      case 10: /* expr ::= SYMBOL */ yytestcase(yyruleno==10);
      case 58: /* symbolstr ::= STR_CONST */ yytestcase(yyruleno==58);
      case 59: /* symbolstr ::= SYMBOL */ yytestcase(yyruleno==59);
#line 66 "parser.y"
{ yygotominor.yy0 = yymsp[0].minor.yy0; }
#line 944 "parser.c"
        break;
      case 6: /* equal_assign ::= expr EQ_ASSIGN expr_or_assign */
      case 18: /* expr ::= expr COLON expr */ yytestcase(yyruleno==18);
      case 19: /* expr ::= expr PLUS expr */ yytestcase(yyruleno==19);
      case 20: /* expr ::= expr MINUS expr */ yytestcase(yyruleno==20);
      case 21: /* expr ::= expr TIMES expr */ yytestcase(yyruleno==21);
      case 22: /* expr ::= expr DIVIDE expr */ yytestcase(yyruleno==22);
      case 23: /* expr ::= expr POW expr */ yytestcase(yyruleno==23);
      case 24: /* expr ::= expr SPECIALOP expr */ yytestcase(yyruleno==24);
      case 25: /* expr ::= expr TILDE expr */ yytestcase(yyruleno==25);
      case 26: /* expr ::= expr QUESTION expr */ yytestcase(yyruleno==26);
      case 27: /* expr ::= expr LT expr */ yytestcase(yyruleno==27);
      case 28: /* expr ::= expr LE expr */ yytestcase(yyruleno==28);
      case 29: /* expr ::= expr EQ expr */ yytestcase(yyruleno==29);
      case 30: /* expr ::= expr NE expr */ yytestcase(yyruleno==30);
      case 31: /* expr ::= expr GE expr */ yytestcase(yyruleno==31);
      case 32: /* expr ::= expr GT expr */ yytestcase(yyruleno==32);
      case 33: /* expr ::= expr AND expr */ yytestcase(yyruleno==33);
      case 34: /* expr ::= expr OR expr */ yytestcase(yyruleno==34);
      case 35: /* expr ::= expr AND2 expr */ yytestcase(yyruleno==35);
      case 36: /* expr ::= expr OR2 expr */ yytestcase(yyruleno==36);
      case 37: /* expr ::= expr LEFT_ASSIGN expr */ yytestcase(yyruleno==37);
      case 48: /* expr ::= SYMBOL NS_GET symbolstr */ yytestcase(yyruleno==48);
      case 49: /* expr ::= STR_CONST NS_GET symbolstr */ yytestcase(yyruleno==49);
      case 50: /* expr ::= SYMBOL NS_GET_INT symbolstr */ yytestcase(yyruleno==50);
      case 51: /* expr ::= STR_CONST NS_GET_INT symbolstr */ yytestcase(yyruleno==51);
      case 52: /* expr ::= expr DOLLAR symbolstr */ yytestcase(yyruleno==52);
      case 53: /* expr ::= expr AT symbolstr */ yytestcase(yyruleno==53);
#line 69 "parser.y"
{ yygotominor.yy0 = Call::c(yymsp[-1].minor.yy0, yymsp[-2].minor.yy0, yymsp[0].minor.yy0); }
#line 975 "parser.c"
        break;
      case 11: /* expr ::= LBRACE exprlist RBRACE */
#line 76 "parser.y"
{ yymsp[-1].minor.yy18.push_front(Symbol(0), yymsp[-2].minor.yy0); yygotominor.yy0 = Expression(List(yymsp[-1].minor.yy18)); }
#line 980 "parser.c"
        break;
      case 12: /* expr ::= LPAREN expr_or_assign RPAREN */
      case 56: /* cond ::= LPAREN expr RPAREN */ yytestcase(yyruleno==56);
      case 57: /* ifcond ::= LPAREN expr RPAREN */ yytestcase(yyruleno==57);
#line 77 "parser.y"
{ yygotominor.yy0 = yymsp[-1].minor.yy0; }
#line 987 "parser.c"
        break;
      case 13: /* expr ::= MINUS expr */
      case 14: /* expr ::= PLUS expr */ yytestcase(yyruleno==14);
      case 15: /* expr ::= NOT expr */ yytestcase(yyruleno==15);
      case 16: /* expr ::= TILDE expr */ yytestcase(yyruleno==16);
      case 17: /* expr ::= QUESTION expr */ yytestcase(yyruleno==17);
      case 45: /* expr ::= REPEAT expr_or_assign */ yytestcase(yyruleno==45);
#line 79 "parser.y"
{ yygotominor.yy0 = Call::c(yymsp[-1].minor.yy0, yymsp[0].minor.yy0); }
#line 997 "parser.c"
        break;
      case 38: /* expr ::= expr RIGHT_ASSIGN expr */
#line 106 "parser.y"
{ yygotominor.yy0 = Call::c(yymsp[-1].minor.yy0, yymsp[0].minor.yy0, yymsp[-2].minor.yy0); }
#line 1002 "parser.c"
        break;
      case 39: /* expr ::= FUNCTION LPAREN formlist RPAREN expr_or_assign */
#line 107 "parser.y"
{ yygotominor.yy0 = Call::c(yymsp[-4].minor.yy0, PairList(List(yymsp[-2].minor.yy18)), yymsp[0].minor.yy0); }
#line 1007 "parser.c"
        break;
      case 40: /* expr ::= expr LPAREN sublist RPAREN */
#line 108 "parser.y"
{ yymsp[-1].minor.yy18.push_front(Symbol(0), yymsp[-3].minor.yy0); yygotominor.yy0 = Call(List(yymsp[-1].minor.yy18)); }
#line 1012 "parser.c"
        break;
      case 41: /* expr ::= IF ifcond expr_or_assign */
      case 44: /* expr ::= WHILE cond expr_or_assign */ yytestcase(yyruleno==44);
#line 109 "parser.y"
{ yygotominor.yy0 = Call::c(yymsp[-2].minor.yy0, yymsp[-1].minor.yy0, yymsp[0].minor.yy0); }
#line 1018 "parser.c"
        break;
      case 42: /* expr ::= IF ifcond expr_or_assign ELSE expr_or_assign */
#line 110 "parser.y"
{ yygotominor.yy0 = Call::c(yymsp[-4].minor.yy0, yymsp[-3].minor.yy0, yymsp[-2].minor.yy0, yymsp[0].minor.yy0); }
#line 1023 "parser.c"
        break;
      case 43: /* expr ::= FOR LPAREN SYMBOL IN expr RPAREN expr_or_assign */
#line 111 "parser.y"
{ yygotominor.yy0 = Call::c(yymsp[-6].minor.yy0, yymsp[-4].minor.yy0, yymsp[-2].minor.yy0, yymsp[0].minor.yy0); }
#line 1028 "parser.c"
        break;
      case 46: /* expr ::= expr LBB sublist RBB */
      case 47: /* expr ::= expr LBRACKET sublist RBRACKET */ yytestcase(yyruleno==47);
#line 114 "parser.y"
{ yymsp[-1].minor.yy18.push_front(Symbol(0), yymsp[-3].minor.yy0); yymsp[-1].minor.yy18.push_front(Symbol(0), yymsp[-2].minor.yy0); yygotominor.yy0 = Call(List(yymsp[-1].minor.yy18)); }
#line 1034 "parser.c"
        break;
      case 54: /* expr ::= NEXT */
      case 55: /* expr ::= BREAK */ yytestcase(yyruleno==55);
#line 122 "parser.y"
{ yygotominor.yy0 = Call::c(yymsp[0].minor.yy0); }
#line 1040 "parser.c"
        break;
      case 60: /* exprlist ::= expr_or_assign */
      case 67: /* sub ::= expr */ yytestcase(yyruleno==67);
      case 73: /* sub ::= NULL_CONST EQ_ASSIGN expr */ yytestcase(yyruleno==73);
#line 131 "parser.y"
{ yygotominor.yy18 = Pairs::Make(); yygotominor.yy18.push_back(Symbol(0), yymsp[0].minor.yy0); }
#line 1047 "parser.c"
        break;
      case 61: /* exprlist ::= exprlist SEMICOLON expr_or_assign */
      case 63: /* exprlist ::= exprlist NEWLINE expr_or_assign */ yytestcase(yyruleno==63);
#line 132 "parser.y"
{ yygotominor.yy18 = yymsp[-2].minor.yy18; yygotominor.yy18.push_back(Symbol(0), yymsp[0].minor.yy0); }
#line 1053 "parser.c"
        break;
      case 62: /* exprlist ::= exprlist SEMICOLON */
      case 64: /* exprlist ::= exprlist NEWLINE */ yytestcase(yyruleno==64);
#line 133 "parser.y"
{ yygotominor.yy18 = yymsp[-1].minor.yy18; }
#line 1059 "parser.c"
        break;
      case 65: /* sublist ::= sub */
#line 137 "parser.y"
{ yygotominor.yy18 = yymsp[0].minor.yy18; }
#line 1064 "parser.c"
        break;
      case 66: /* sublist ::= sublist COMMA sub */
#line 138 "parser.y"
{ yygotominor.yy18 = yymsp[-2].minor.yy18; yygotominor.yy18.push_back(yymsp[0].minor.yy18.name(0), yymsp[0].minor.yy18.value(0)); }
#line 1069 "parser.c"
        break;
      case 68: /* sub ::= SYMBOL EQ_ASSIGN */
      case 69: /* sub ::= STR_CONST EQ_ASSIGN */ yytestcase(yyruleno==69);
#line 141 "parser.y"
{ yygotominor.yy18 = Pairs::Make(); yygotominor.yy18.push_back(yymsp[-1].minor.yy0, Value::NIL); }
#line 1075 "parser.c"
        break;
      case 70: /* sub ::= SYMBOL EQ_ASSIGN expr */
      case 71: /* sub ::= STR_CONST EQ_ASSIGN expr */ yytestcase(yyruleno==71);
      case 75: /* formlist ::= SYMBOL EQ_ASSIGN expr */ yytestcase(yyruleno==75);
#line 143 "parser.y"
{ yygotominor.yy18 = Pairs::Make(); yygotominor.yy18.push_back(Symbol(yymsp[-2].minor.yy0), yymsp[0].minor.yy0); }
#line 1082 "parser.c"
        break;
      case 72: /* sub ::= NULL_CONST EQ_ASSIGN */
#line 145 "parser.y"
{ yygotominor.yy18 = Pairs::Make(); yygotominor.yy18.push_back(Symbol(0), Value::NIL); }
#line 1087 "parser.c"
        break;
      case 74: /* formlist ::= SYMBOL */
#line 148 "parser.y"
{ yygotominor.yy18 = Pairs::Make(); yygotominor.yy18.push_back(Symbol(yymsp[0].minor.yy0), Value::NIL); }
#line 1092 "parser.c"
        break;
      case 76: /* formlist ::= formlist COMMA SYMBOL */
#line 150 "parser.y"
{ yygotominor.yy18 = yymsp[-2].minor.yy18; yygotominor.yy18.push_back(Symbol(yymsp[0].minor.yy0), Value::NIL); }
#line 1097 "parser.c"
        break;
      case 77: /* formlist ::= formlist COMMA SYMBOL EQ_ASSIGN expr */
#line 151 "parser.y"
{ yygotominor.yy18 = yymsp[-4].minor.yy18; yygotominor.yy18.push_back(Symbol(yymsp[-2].minor.yy0), yymsp[0].minor.yy0); }
#line 1102 "parser.c"
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
#line 56 "parser.y"

     result->state = -1;
     printf("Giving up.  Parser is hopelessly lost...\n");
#line 1154 "parser.c"
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
#line 47 "parser.y"

        result->state = -1;
	std::cout << "Syntax error!" << std::endl;
#line 1173 "parser.c"
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
#line 52 "parser.y"

     result->state = 1;
#line 1195 "parser.c"
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
