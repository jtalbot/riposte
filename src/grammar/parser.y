/*	A Lemon parser for R.
	Almost exactly the same as the R parser definition.
	But, forcond folded into for rule.
	TODO: support for interactive input
		combine SYMBOL and STR_CONST into a single rule, rather than have 4 variations for sub, NS_GET, and NS_GET_INT 
		similarly, combine NEWLINE and SEMICOLON into single rule, rather than have 2 variations.
		R parser supports a rule: expr % expr, but this doesn't appear to work with the lexer which treats anything that starts with % as a special op. 
		can I factor out the actions?
 */

%stack_size 1000
%token_prefix	TOKEN_
%token_type {Value}
%extra_argument {Parser* parser}

%token_destructor {(void)parser;}

%type statementlist {Pairs*}
%destructor statementlist {delete $$;}
%type sublist  {Pairs*}
%destructor sublist {delete $$;}
%type sub  {Pairs*}
%destructor sub {delete $$;}
%type formallist {Pairs*}
%destructor formallist {delete $$;}

%left           QUESTION.
%left           FUNCTION WHILE FOR REPEAT.
%right          IF.
%left           ELSE.
%right          LEFT_ASSIGN EQ_ASSIGN.
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
%nonassoc       LPAREN LBRACKET LBB LBRACE.

%include {
	#include <iostream>
	#include "internal.h"
}

%syntax_error {
        parser->errors++;
}

%parse_accept {
	parser->complete = true;
}

%stack_overflow {
	parser->errors++;
	fprintf(stderr,"Giving up.  Parser stack overflow\n");
}

/*%parse_failure {
     result->state = -1;
     printf("Giving up.  Parser is hopelessly lost...\n");
}*/

prog ::= optnl statementlist(B) optnl. { parser->result = Expression(B->toList(false)); }
prog ::= error. { parser->result = Expression(0); }

statement(A) ::= expr(B) EQ_ASSIGN(C) optnl statement(D). { A = Call::c(C, B, D); }
statement(A) ::= expr(B). { A = B; }

expr(A) ::= NUM_CONST(B). { A = B; }
expr(A) ::= STR_CONST(B). { A = B; }
expr(A) ::= NULL_CONST(B). { A = B; }
expr(A) ::= SYMBOL(B). { A = B; }

expr(A) ::= LBRACE(B) optnl statementlist(C) optnl RBRACE. { C->push_front(Symbol::empty, B); A = Call(C->toList(false)); }
expr(A) ::= LPAREN optnl statement(B) optnl RPAREN. { A = B; }

expr(A) ::= MINUS(B) optnl expr(C). [UMINUS] { A = Call::c(B, C); }
expr(A) ::= PLUS(B) optnl expr(C).  [UPLUS]  { A = Call::c(B, C); }
expr(A) ::= NOT(B) optnl expr(C).            { A = Call::c(B, C); }
expr(A) ::= TILDE(B) optnl expr(C). { A = Call::c(B, C); }
expr(A) ::= QUESTION(B) optnl expr(C). { A = Call::c(B, C); }

expr(A) ::= expr(B) COLON(C) optnl expr(D). { A = Call::c(C, B, D); }
expr(A) ::= expr(B) PLUS(C) optnl expr(D). { A = Call::c(C, B, D); }
expr(A) ::= expr(B) MINUS(C) optnl expr(D). { A = Call::c(C, B, D); }
expr(A) ::= expr(B) TIMES(C) optnl expr(D). { A = Call::c(C, B, D); }
expr(A) ::= expr(B) DIVIDE(C) optnl expr(D). { A = Call::c(C, B, D); }
expr(A) ::= expr(B) POW(C) optnl expr(D). { A = Call::c(C, B, D); }
expr(A) ::= expr(B) SPECIALOP(C) optnl expr(D). { A = Call::c(C, B, D); }
expr(A) ::= expr(B) TILDE(C) optnl expr(D). { A = Call::c(C, B, D); }
expr(A) ::= expr(B) QUESTION(C) optnl expr(D). { A = Call::c(C, B, D); }
expr(A) ::= expr(B) LT(C) optnl expr(D). { A = Call::c(C, B, D); }
expr(A) ::= expr(B) LE(C) optnl expr(D). { A = Call::c(C, B, D); }
expr(A) ::= expr(B) EQ(C) optnl expr(D). { A = Call::c(C, B, D); }
expr(A) ::= expr(B) NE(C) optnl expr(D). { A = Call::c(C, B, D); }
expr(A) ::= expr(B) GE(C) optnl expr(D). { A = Call::c(C, B, D); }
expr(A) ::= expr(B) GT(C) optnl expr(D). { A = Call::c(C, B, D); }
expr(A) ::= expr(B) AND(C) optnl expr(D). { A = Call::c(C, B, D); }
expr(A) ::= expr(B) OR(C) optnl expr(D). { A = Call::c(C, B, D); }
expr(A) ::= expr(B) AND2(C) optnl expr(D). { A = Call::c(C, B, D); }
expr(A) ::= expr(B) OR2(C) optnl expr(D). { A = Call::c(C, B, D); }

expr(A) ::= expr(B) LEFT_ASSIGN(C) optnl expr(D). { A = Call::c(C, B, D); }
expr(A) ::= expr(B) RIGHT_ASSIGN(C) optnl expr(D). { A = Call::c(C, D, B); }
expr(A) ::= FUNCTION(B) optnl LPAREN optnl formallist(C) optnl RPAREN optnl statement(D).  { A = Call::c(B, PairList(C->toList(true)), D, parser->popSource()); }
expr(A) ::= expr(B) LPAREN optnl sublist(C) optnl RPAREN. { C->push_front(Symbol::empty, B); A = Call(C->toList(false)); } 
expr(A) ::= IF(B) optnl LPAREN optnl expr(C) optnl RPAREN optnl statement(D). { A = Call::c(B, C, D); }
expr(A) ::= IF(B) optnl LPAREN optnl expr(C) optnl RPAREN optnl statement(D) ELSE optnl statement(E). { A = Call::c(B, C, D, E); }
expr(A) ::= FOR(B) optnl LPAREN optnl SYMBOL(C) optnl IN optnl expr(D) optnl RPAREN optnl statement(E). { A = Call::c(B, C, D, E); }
expr(A) ::= WHILE(B) optnl LPAREN optnl expr(C) optnl RPAREN optnl statement(D). { A = Call::c(B, C, D); }
expr(A) ::= REPEAT(B) optnl statement(C). { A = Call::c(B, C); }
expr(A) ::= expr(B) LBB(C) optnl sublist(D) optnl RBRACKET RBRACKET. { D->push_front(Symbol::empty, B); D->push_front(Symbol::empty, C); A = Call(D->toList(false)); }
expr(A) ::= expr(B) LBRACKET(C) optnl sublist(D) optnl RBRACKET. { D->push_front(Symbol::empty, B); D->push_front(Symbol::empty, C); A = Call(D->toList(false)); }
expr(A) ::= SYMBOL(B) NS_GET(C) symbolstr(D). { A = Call::c(C, B, D); }
expr(A) ::= STR_CONST(B) NS_GET(C) symbolstr(D). { A = Call::c(C, B, D); }
expr(A) ::= SYMBOL(B) NS_GET_INT(C) symbolstr(D). { A = Call::c(C, B, D); }
expr(A) ::= STR_CONST(B) NS_GET_INT(C) symbolstr(D). { A = Call::c(C, B, D); }
expr(A) ::= expr(B) DOLLAR(C) optnl symbolstr(D). { A = Call::c(C, B, D); }
expr(A) ::= expr(B) AT(C) optnl symbolstr(D). { A = Call::c(C, B, D); }
expr(A) ::= NEXT(B). { A = Call::c(B); }
expr(A) ::= BREAK(B). { A = Call::c(B); }

symbolstr(A) ::= STR_CONST(B). { A = B; }
symbolstr(A) ::= SYMBOL(B). { A = B; }

optnl ::= NEWLINE.
optnl ::= .

statementlist(A) ::= statementlist(B) SEMICOLON statement(C). { A = B; A->push_back(Symbol::empty, C); }
statementlist(A) ::= statementlist(B) SEMICOLON. { A = B; }
statementlist(A) ::= statementlist(B) NEWLINE statement(C). { A = B; A->push_back(Symbol::empty, C); }
statementlist(A) ::= statement(B). { A = new Pairs(); A->push_back(Symbol::empty, B); }
statementlist(A) ::= . { A = new Pairs(); }

sublist(A) ::= sub(B). { A = B; }
sublist(A) ::= sublist(B) optnl COMMA optnl sub(C). { A = B; if(C->length() == 1) A->push_back(C->name(0), C->value(0)); }

sub(A) ::= . { A = new Pairs(); }
sub(A) ::= SYMBOL(B) optnl EQ_ASSIGN. { A = new Pairs(); A->push_back(Symbol(B), Nil); }
sub(A) ::= STR_CONST(B) optnl EQ_ASSIGN. { A = new Pairs(); A->push_back(Symbol(B), Nil); }
sub(A) ::= SYMBOL(B) optnl EQ_ASSIGN optnl expr(C). { A = new Pairs(); A->push_back(Symbol(B), C); }
sub(A) ::= STR_CONST(B) optnl EQ_ASSIGN optnl expr(C). { A = new Pairs(); A->push_back(Symbol(B), C); }
sub(A) ::= NULL_CONST optnl EQ_ASSIGN. { A = new Pairs(); A->push_back(Symbol::empty, Nil); }
sub(A) ::= NULL_CONST optnl EQ_ASSIGN optnl expr(C). { A = new Pairs(); A->push_back(Symbol::empty, C); }
sub(A) ::= expr(B). { A = new Pairs(); A->push_back(Symbol::empty, B); }

formallist(A) ::= . { A = new Pairs(); }
formallist(A) ::= SYMBOL(B). { A = new Pairs(); A->push_back(Symbol(B), Nil); }
formallist(A) ::= SYMBOL(B) optnl EQ_ASSIGN optnl expr(C). { A = new Pairs(); A->push_back(Symbol(B), C); }
formallist(A) ::= formallist(B) optnl COMMA optnl SYMBOL(C). { A = B; A->push_back(Symbol(C), Nil); }
formallist(A) ::= formallist(B) optnl COMMA optnl SYMBOL(C) optnl EQ_ASSIGN optnl expr(D). { A = B; A->push_back(Symbol(C), D); }

