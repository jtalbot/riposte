/*	A Lemon parser for R.
	Almost exactly the same as the R parser definition.
	But, forcond folded into for rule.
	TODO: support for interactive input
		combine SYMBOL and STR_CONST into a single rule, rather than have 4 variations for sub, NS_GET, and NS_GET_INT 
		similarly, combine NEWLINE and SEMICOLON into single rule, rather than have 2 variations.
		can I factor out the actions?
 */

%token_prefix	TOKEN_
%token_type {Value}
%extra_argument {Parser::Result* result}

%type exprlist {Pairs}
%type sublist  {Pairs}
%type sub  {Pairs}
%type formlist {Pairs}

%left           QUESTION.
%left           LOW WHILE FOR REPEAT.
%right          IF.
%left           ELSE.
%right          LEFT_ASSIGN.
%right          EQ_ASSIGN.
%left           RIGHT_ASSIGN.
%left           TILDE.
%left           OR OR2.
%left           AND AND2.
%left           NOT.
%nonassoc       GT GE LT LE EQ NE.
%left           PLUS MINUS.
%left           TIMES DIVIDE.
%left           SPECIALOP.
%left           COLON.
%left           UMINUS UPLUS.
%right          POW.
%left           DOLLAR AT.
%left           NS_GET NS_GET_INT.
%nonassoc       PAREN BRACKET BB.

%include {
	#include <iostream>
	#include "internal.h"
}

%syntax_error {
        result->state = -1;
	std::cout << "Syntax error!" << std::endl;
}

%parse_accept {
     result->state = 1;
}

%parse_failure {
     result->state = -1;
     printf("Giving up.  Parser is hopelessly lost...\n");
}

prog(A) ::= END_OF_INPUT. { result->value = A = Expression(0); }
prog(A) ::= NEWLINE. { result->value = A = Expression(0); }
//prog ::= expr_or_assign NEWLINE.
//prog ::= expr_or_assign SEMICOLON.
prog(A) ::= exprlist(B). { result->value = A = Expression(List(B)); }
prog(A) ::= error. { result->value = A = Expression(0); }

optnl ::= NEWLINE.
optnl ::= .

expr_or_assign(A) ::= expr(B). { A = B; }
expr_or_assign(A) ::= equal_assign(B). { A = B; }

equal_assign(A) ::= expr(B) EQ_ASSIGN(C) expr_or_assign(D). { A = Call::c(C, B, D); }

expr(A) ::= NUM_CONST(B). { A = B; }
expr(A) ::= STR_CONST(B). { A = B; }
expr(A) ::= NULL_CONST(B). { A = B; }
expr(A) ::= SYMBOL(B). { A = B; }

expr(A) ::= LBRACE(B) exprlist(C) RBRACE. { C.push_front(Symbol(0), B); A = Expression(List(C)); }
expr(A) ::= LPAREN expr_or_assign(B) RPAREN. [PAREN] { A = B; }

expr(A) ::= MINUS(B) expr(C). [UMINUS] { A = Call::c(B, C); }
expr(A) ::= PLUS(B) expr(C).  [UPLUS]  { A = Call::c(B, C); }
expr(A) ::= NOT(B) expr(C).            { A = Call::c(B, C); }
expr(A) ::= TILDE(B) expr(C). { A = Call::c(B, C); }
expr(A) ::= QUESTION(B) expr(C). { A = Call::c(B, C); }

expr(A) ::= expr(B) COLON(C) expr(D). { A = Call::c(C, B, D); }
expr(A) ::= expr(B) PLUS(C) optnl expr(D). { A = Call::c(C, B, D); }
expr(A) ::= expr(B) MINUS(C) expr(D). { A = Call::c(C, B, D); }
expr(A) ::= expr(B) TIMES(C) expr(D). { A = Call::c(C, B, D); }
expr(A) ::= expr(B) DIVIDE(C) expr(D). { A = Call::c(C, B, D); }
expr(A) ::= expr(B) POW(C) expr(D). { A = Call::c(C, B, D); }
expr(A) ::= expr(B) SPECIALOP(C) expr(D). { A = Call::c(C, B, D); }
expr(A) ::= expr(B) PERCENT(C) expr(D). { A = Call::c(C, B, D); }
expr(A) ::= expr(B) TILDE(C) expr(D). { A = Call::c(C, B, D); }
expr(A) ::= expr(B) QUESTION(C) expr(D). { A = Call::c(C, B, D); }
expr(A) ::= expr(B) LT(C) expr(D). { A = Call::c(C, B, D); }
expr(A) ::= expr(B) LE(C) expr(D). { A = Call::c(C, B, D); }
expr(A) ::= expr(B) EQ(C) expr(D). { A = Call::c(C, B, D); }
expr(A) ::= expr(B) NE(C) expr(D). { A = Call::c(C, B, D); }
expr(A) ::= expr(B) GE(C) expr(D). { A = Call::c(C, B, D); }
expr(A) ::= expr(B) GT(C) expr(D). { A = Call::c(C, B, D); }
expr(A) ::= expr(B) AND(C) expr(D). { A = Call::c(C, B, D); }
expr(A) ::= expr(B) OR(C) expr(D). { A = Call::c(C, B, D); }
expr(A) ::= expr(B) AND2(C) expr(D). { A = Call::c(C, B, D); }
expr(A) ::= expr(B) OR2(C) expr(D). { A = Call::c(C, B, D); }

expr(A) ::= expr(B) LEFT_ASSIGN(C) expr(D). { A = Call::c(C, B, D); }
expr(A) ::= expr(B) RIGHT_ASSIGN(C) expr(D). { A = Call::c(C, D, B); }
expr(A) ::= FUNCTION(B) LPAREN formlist(C) RPAREN expr_or_assign(D).  [LOW] { A = Call::c(B, PairList(List(C)), D); }
expr(A) ::= expr(B) LPAREN sublist(C) RPAREN. { C.push_front(Symbol(0), B); A = Call(List(C)); }  /* Function call */
expr(A) ::= IF(B) ifcond(C) expr_or_assign(D). { A = Call::c(B, C, D); }
expr(A) ::= IF(B) ifcond(C) expr_or_assign(D) ELSE expr_or_assign(E). { A = Call::c(B, C, D, E); }
expr(A) ::= FOR(B) LPAREN SYMBOL(C) IN expr(D) RPAREN expr_or_assign(E). [FOR] { A = Call::c(B, C, D, E); }
expr(A) ::= WHILE(B) cond(C) expr_or_assign(D). { A = Call::c(B, C, D); }
expr(A) ::= REPEAT(B) expr_or_assign(C). { A = Call::c(B, C); }
expr(A) ::= expr(B) LBB(C) sublist(D) RBB. [BB] { D.push_front(Symbol(0), B); D.push_front(Symbol(0), C); A = Call(List(D)); }
expr(A) ::= expr(B) LBRACKET(C) sublist(D) RBRACKET. [BRACKET] { D.push_front(Symbol(0), B); D.push_front(Symbol(0), C); A = Call(List(D)); }
expr(A) ::= SYMBOL(B) NS_GET(C) SYMBOL(D). { A = Call::c(C, B, D); }
expr(A) ::= SYMBOL(B) NS_GET(C) STR_CONST(D). { A = Call::c(C, B, D); }
expr(A) ::= STR_CONST(B) NS_GET(C) SYMBOL(D). { A = Call::c(C, B, D); }
expr(A) ::= STR_CONST(B) NS_GET(C) STR_CONST(D). { A = Call::c(C, B, D); }
expr(A) ::= SYMBOL(B) NS_GET_INT(C) SYMBOL(D). { A = Call::c(C, B, D); }
expr(A) ::= SYMBOL(B) NS_GET_INT(C) STR_CONST(D). { A = Call::c(C, B, D); }
expr(A) ::= STR_CONST(B) NS_GET_INT(C) SYMBOL(D). { A = Call::c(C, B, D); }
expr(A) ::= STR_CONST(B) NS_GET_INT(C) STR_CONST(D). { A = Call::c(C, B, D); }
expr(A) ::= expr(B) DOLLAR(C) SYMBOL(D). { A = Call::c(C, B, D); }
expr(A) ::= expr(B) DOLLAR(C) STR_CONST(D). { A = Call::c(C, B, D); }
expr(A) ::= expr(B) AT(C) SYMBOL(D). { A = Call::c(C, B, D); }
expr(A) ::= expr(B) AT(C) STR_CONST(D). { A = Call::c(C, B, D); }
expr(A) ::= NEXT(B). { A = Call::c(B); }
expr(A) ::= BREAK(B). { A = Call::c(B); }

cond(A) ::= LPAREN expr(B) RPAREN. { A = B; }
ifcond(A) ::= LPAREN expr(B) RPAREN. { A = B; }

exprlist(A) ::= expr_or_assign(B). { A = Pairs::Make(); A.push_back(Symbol(0), B); }
exprlist(A) ::= exprlist(B) SEMICOLON expr_or_assign(C). { A = B; A.push_back(Symbol(0), C); }
exprlist(A) ::= exprlist(B) SEMICOLON. { A = B; }
exprlist(A) ::= exprlist(B) NEWLINE expr_or_assign(C). { A = B; A.push_back(Symbol(0), C); }
exprlist(A) ::= exprlist(B) NEWLINE. { A = B; }

sublist(A) ::= sub(B). { A = B; }
sublist(A) ::= sublist(B) COMMA sub(C). { A = B; A.push_back(C.name(0), C.value(0)); }

sub(A) ::= expr(B). { A = Pairs::Make(); A.push_back(Symbol(0), B); }
sub(A) ::= SYMBOL(B) EQ_ASSIGN. { A = Pairs::Make(); A.push_back(B, Value::NIL); }
sub(A) ::= SYMBOL(B) EQ_ASSIGN expr(C). { A = Pairs::Make(); A.push_back(Symbol(B), C); }
sub(A) ::= STR_CONST(B) EQ_ASSIGN. { A = Pairs::Make(); A.push_back(B, Value::NIL); }
sub(A) ::= STR_CONST(B) EQ_ASSIGN expr(C). { A = Pairs::Make(); A.push_back(Symbol(B), C); }
sub(A) ::= NULL_CONST EQ_ASSIGN. { A = Pairs::Make(); A.push_back(Symbol(0), Value::NIL); }
sub(A) ::= NULL_CONST EQ_ASSIGN expr(C). { A = Pairs::Make(); A.push_back(Symbol(0), C); }

formlist(A) ::= SYMBOL(B). { A = Pairs::Make(); A.push_back(Symbol(B), Value::NIL); }
formlist(A) ::= SYMBOL(B) EQ_ASSIGN expr(C). { A = Pairs::Make(); A.push_back(Symbol(B), C); }
formlist(A) ::= formlist(B) COMMA SYMBOL(C). { A = B; A.push_back(Symbol(C), Value::NIL); }
formlist(A) ::= formlist(B) COMMA SYMBOL(C) EQ_ASSIGN expr(D). { A = B; A.push_back(Symbol(C), D); }

